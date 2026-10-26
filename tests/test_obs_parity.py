"""numpy <-> torch observation parity (CONTRACT §6, binding).

`envs/asv_env.py::raw_sample_torch` + the env's normalize/buffer must reproduce
`runtime/obs_np.py` (`raw_sample`, `normalize`, `ObsBuilder`) on the same inputs.
Tolerance: max |obs_torch - obs_numpy| <= 1e-5 on all 150 float32 components.
"""

import copy

import numpy as np
import torch

from envs.asv_env import AsvEnv, load_config, raw_sample_torch
from runtime.obs_np import K_HISTORY, RAW_DIM, ObsBuilder, normalize, raw_sample

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = 8
TOL = 1e-5            # CONTRACT §6
STEPS = 25


def _cfg():
    cfg = copy.deepcopy(load_config())
    cfg["episode"]["geofence_m"] = 1e6      # no auto-reset during the parity run
    cfg["episode_s"] = 1e4
    return cfg


def _rand_inputs(rng, n=N):
    est = np.stack([
        rng.uniform(-40, 40, n), rng.uniform(-40, 40, n), rng.uniform(-np.pi, np.pi, n),
        rng.uniform(-1, 1, n), rng.uniform(-0.5, 0.5, n), rng.uniform(-1, 1, n),
    ], axis=1)
    goal = np.stack([
        rng.uniform(-60, 60, n), rng.uniform(-60, 60, n),
        rng.uniform(-np.pi, np.pi, n), rng.integers(0, 2, n).astype(float),
    ], axis=1)
    a = rng.uniform(-1, 1, (n, 3))
    wp1 = rng.uniform(-60, 60, (n, 2))
    wp2 = rng.uniform(-60, 60, (n, 2))
    return est, goal, a, wp1, wp2


def test_happy_constants_match_contract():
    cfg = _cfg()
    assert RAW_DIM == 15 and K_HISTORY == 10
    assert len(cfg["obs"]["scale"]) == RAW_DIM and len(cfg["obs"]["offset"]) == RAW_DIM
    env = AsvEnv(cfg=cfg, num_envs=2, device=DEV)
    assert env.obs_dim == RAW_DIM * K_HISTORY == 150
    assert np.allclose(env.obs_scale.cpu().numpy(), cfg["obs"]["scale"])
    assert np.allclose(env.obs_offset.cpu().numpy(), cfg["obs"]["offset"])


def test_raw_sample_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(5):
        est, goal, a, wp1, wp2 = _rand_inputs(rng)
        ref = raw_sample(est, goal, a, wp1, wp2).astype(np.float32)
        got = raw_sample_torch(
            torch.as_tensor(est, dtype=torch.float32, device=DEV),
            torch.as_tensor(goal, dtype=torch.float32, device=DEV),
            torch.as_tensor(a, dtype=torch.float32, device=DEV),
            torch.as_tensor(wp1, dtype=torch.float32, device=DEV),
            torch.as_tensor(wp2, dtype=torch.float32, device=DEV),
        ).cpu().numpy()
        assert got.shape == ref.shape == (N, RAW_DIM)
        assert np.abs(got - ref).max() <= TOL


def test_raw_sample_none_waypoints_and_action():
    rng = np.random.default_rng(1)
    est, goal, _, _, _ = _rand_inputs(rng)
    ref = raw_sample(est, goal)                       # prev_action/wp default to zeros
    got = raw_sample_torch(torch.as_tensor(est, dtype=torch.float32, device=DEV),
                           torch.as_tensor(goal, dtype=torch.float32, device=DEV))
    assert np.abs(got.cpu().numpy() - ref.astype(np.float32)).max() <= TOL
    assert np.allclose(ref[:, 7:14], 0.0)


def test_normalize_and_history_match_obs_builder():
    """Full pipeline: normalize per step, K=10 ring buffer, oldest first."""
    cfg = _cfg()
    rng = np.random.default_rng(2)
    ref = ObsBuilder(offset=cfg["obs"]["offset"], scale=cfg["obs"]["scale"],
                     clip_norm=cfg["obs"]["clip_norm"], k_history=K_HISTORY)
    env = AsvEnv(cfg=cfg, num_envs=N, device=DEV, seed=0)
    env.reset()
    offset = torch.as_tensor(cfg["obs"]["offset"], dtype=torch.float32, device=DEV)
    scale = torch.as_tensor(cfg["obs"]["scale"], dtype=torch.float32, device=DEV)
    clip = float(cfg["obs"]["clip_norm"])
    torch_buf = None

    for step in range(STEPS):
        est, goal, a, wp1, wp2 = _rand_inputs(rng)
        ref_out = ref.build(est, goal, a, wp1, wp2)
        raw = raw_sample_torch(torch.as_tensor(est, dtype=torch.float32, device=DEV),
                              torch.as_tensor(goal, dtype=torch.float32, device=DEV),
                              torch.as_tensor(a, dtype=torch.float32, device=DEV),
                              torch.as_tensor(wp1, dtype=torch.float32, device=DEV),
                              torch.as_tensor(wp2, dtype=torch.float32, device=DEV))
        sample = ((raw - offset) / scale).clamp(-clip, clip)
        if torch_buf is None:
            torch_buf = sample.repeat(1, K_HISTORY)
        else:
            torch_buf = torch.cat([torch_buf[:, RAW_DIM:], sample], dim=1)
        diff = np.abs(torch_buf.cpu().numpy() - ref_out).max()
        assert diff <= TOL, f"step {step}: max diff {diff:.3e}"
    # normalization: the numpy reference is a divisor, clipped to +-clip_norm
    big = np.full((1, RAW_DIM), 1e4)
    assert np.abs(normalize(big)).max() == clip


def test_env_obs_matches_numpy_builder():
    """The env's own obs == numpy ObsBuilder fed with the env's own estimator output."""
    cfg = _cfg()
    env = AsvEnv(cfg=cfg, num_envs=N, device=DEV, seed=5)
    ref = ObsBuilder(offset=cfg["obs"]["offset"], scale=cfg["obs"]["scale"],
                     clip_norm=cfg["obs"]["clip_norm"], k_history=K_HISTORY)
    env.reset()
    rng = np.random.default_rng(3)
    env._sync_level(force=True)
    env.set_curriculum(0)                     # no disturbance: pure geometry comparison

    def _ref_build(act):
        wp1, wp2 = env._next_wps()
        return ref.build(
            env.meas["est"].cpu().double().numpy(),
            np.stack([env.goal_xy[:, 0].cpu().numpy(), env.goal_xy[:, 1].cpu().numpy(),
                      env.psi_goal.cpu().numpy(), env.mode.cpu().numpy()], axis=1),
            act.cpu().double().numpy(),
            None if wp1 is None else wp1.cpu().double().numpy(),
            None if wp2 is None else wp2.cpu().double().numpy(),
        )

    _ref_build(torch.zeros(N, 3, device=DEV))   # reset sample fills all K slots (both sides)
    for step in range(12):
        a = torch.as_tensor(rng.uniform(-1, 1, (N, 3)), dtype=torch.float32, device=DEV)
        act_prev = env.prev_action.clone()     # a_{t-1}: what the obs of this step carries
        obs, _, done, _ = env.step(a)
        assert done.max() == 0.0              # auto-reset would desync the builder
        ref_out = _ref_build(act_prev)
        diff = np.abs(obs.cpu().numpy() - ref_out).max()
        assert diff <= TOL, f"step {step}: max diff {diff:.3e}"
