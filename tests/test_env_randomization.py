"""R2: per-env randomisation in `envs/asv_env.py` (CONTRACT §3.4).

Covers what reports/R1.md §3 handed to the env owner: every reset draws mass, inertia,
drag, thruster gain/lag/deadband/asymmetry and sensor noise/bias/latency **per env**,
through `dynamics.set_params` / `thrusters.set_gains` / `sensors.set_noise` — no more
`setattr` on the sensor model. reports/R2.md documents the factor table and the change
in behaviour against the old env writer.
"""

import copy
import hashlib
import inspect
import math
import re

import torch

from envs import asv_env
from envs.asv_env import AsvEnv, load_config

N = 8
# `sensor_bias_deg` dropped in R3: env.yaml's key is no longer read, the magnetometer
# bias bound comes from sensors.yaml `mag.bias_deg` (reports/R3.md §3)
ZERO_KEYS = ("mass", "inertia", "drag", "thruster_gain", "thruster_lag",
             "thruster_deadband", "fwd_rev_asymmetry", "weak_thruster_prob",
             "sensor_noise", "sensor_latency_s")

# sha256 of an 8-env, 15-step rollout with every randomisation range at 0.
# Re-recorded in R3: the pre-R2 value (0af78a8e…) pinned a rollout whose episodes started
# with the estimator at the ENU origin. R3 seeds the estimator + GPS ring with the spawn
# pose (reports/R3.md §1), so the reset obs legitimately changed. Verified that ignoring
# `pos`/`heading` in `SensorModel.reset` reproduces 0af78a8e… bit for bit, i.e. seeding is
# the only difference; the value below still fails on any other drift.
PRE_CHANGE_HASH = "e115cfc54cfdc7dcaf33c515cac4da0ff4abe414ed17761fb337f5e43d307ece"


def _cfg(zero: bool = False, **over) -> dict:
    cfg = copy.deepcopy(load_config())
    if zero:
        for key in ZERO_KEYS:
            cfg["randomization"][key] = 0.0
    cfg["episode"]["geofence_m"] = 1e6          # no auto-reset inside a rollout
    cfg["episode_s"] = 1e4
    for path, val in over.items():
        node, keys = cfg, path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = val
    return cfg


def test_every_randomised_quantity_differs_across_envs():
    """One randomised reset leaves two envs of the batch with different everything."""
    env = AsvEnv(cfg=_cfg(), num_envs=N, device="cpu", seed=11)
    env.reset()
    dyn, thr, sen = env.dynamics, env.thrusters, env.sensors

    assert dyn.M_env[0].tolist() != dyn.M_env[1].tolist()
    assert dyn.D_lin_env[0].tolist() != dyn.D_lin_env[1].tolist()
    assert dyn.D_quad_env[0].tolist() != dyn.D_quad_env[1].tolist()
    assert not torch.allclose(thr.scale_fwd[0], thr.scale_fwd[1])
    assert thr.lag_tau[0] != thr.lag_tau[1]
    assert thr.deadband[0] != thr.deadband[1]
    # sensor side is per env now, not one batch-wide factor
    for name in ("pos_sigma_env", "vel_sigma_env", "cog_sigma_env", "gyro_sigma_env",
                 "mag_sigma_env", "gyro_bias_max_env", "mag_bias_max_env", "latency_env"):
        t = getattr(sen, name)
        assert t[0] != t[1], name
    assert sen.gyro_bias[0] != sen.gyro_bias[1]
    assert sen.mag_bias[0] != sen.mag_bias[1]
    # and the draws stay inside the yaml ranges
    assert (thr.lag_tau / env.lag0 - 1.0).abs().max() <= env.r_lag + 1e-5
    assert (sen.pos_sigma_env / sen.gps_pos_sigma - 1.0).abs().max() <= env.r_noise + 1e-5
    assert (sen.latency_env - env._latency0).abs().max() <= env.r_latency + 1e-5


def test_randomisation_off_rollout_matches_pre_change_snapshot():
    """All ranges 0 -> nominal params -> the pre-R2 rollout, bit for bit."""
    torch.manual_seed(1234)                      # sensor noise draws from the global RNG
    env = AsvEnv(cfg=_cfg(zero=True), num_envs=N, device="cpu", seed=0)
    h = hashlib.sha256()

    def feed(t):
        h.update(t.detach().to(torch.float32).cpu().numpy().tobytes())

    feed(env.obs_buf)
    for step in range(15):
        a = torch.tensor([[math.sin(0.3 * step + i), math.cos(0.2 * step - i),
                           0.1 * i - 0.05 * step] for i in range(N)], dtype=torch.float32)
        obs, rew, done, _ = env.step(a)
        assert float(done.max()) == 0.0
        feed(obs), feed(rew), feed(env.state)
        feed(env.dynamics.M_env), feed(env.dynamics.D_lin_env), feed(env.dynamics.D_quad_env)
    assert h.hexdigest() == PRE_CHANGE_HASH


def test_randomisation_only_touches_reset_envs():
    env = AsvEnv(cfg=_cfg(), num_envs=N, device="cpu", seed=3)
    env.reset()
    before = env.dynamics.D_lin_env[0].clone()
    env.reset(torch.tensor([1, 2]))
    assert torch.equal(env.dynamics.D_lin_env[0], before)


def test_effective_factors_on_mass_and_drag_are_the_documented_ones():
    """M/D_lin/D_quad per axis for (mass_scale, inertia_scale, drag_scale) — R3.md §4."""
    ms, ins, ds = 1.3, 0.8, 1.2
    env = AsvEnv(cfg=_cfg(zero=True), num_envs=N, device="cpu", seed=0)
    dyn = env.dynamics
    dyn.set_params(None, mass_scale=ms, inertia_scale=ins, drag_scale=ds)
    want_M = torch.tensor([ms, ms, ins], dtype=torch.float32)
    want_D = torch.tensor([ds, ds, ds], dtype=torch.float32)   # drag only: mass is decoupled
    for i in range(N):
        assert torch.allclose(dyn.M_env[i] / dyn.M, want_M, atol=1e-6)
        assert torch.allclose(dyn.D_lin_env[i] / dyn.D_lin, want_D, atol=1e-6)
        assert torch.allclose(dyn.D_quad_env[i] / dyn.D_quad, want_D, atol=1e-6)

    # the env's own reset path applies exactly that: drag factor == drag_scale alone
    env.reset()
    ratio_M = env.dynamics.M_env / env.dynamics.M
    ratio_L = env.dynamics.D_lin_env / env.dynamics.D_lin
    ratio_Q = env.dynamics.D_quad_env / env.dynamics.D_quad
    assert (ratio_M[:, :2] - 1.0).abs().max() <= env.r_mass + 1e-5
    assert (ratio_M[:, 2] - 1.0).abs().max() <= env.r_inertia + 1e-5
    assert (ratio_L - 1.0).abs().max() <= env.r_drag + 1e-5
    assert torch.allclose(ratio_L, ratio_L[:, 2:3].expand(N, 3), atol=1e-6)   # same ds, 3 axes
    assert torch.allclose(ratio_Q, ratio_L, atol=1e-6)                        # no quad range


def test_old_setattr_sensor_writer_is_gone():
    """No `setattr` on the sensor model and no writes to its nominal float attributes."""
    src = inspect.getsource(asv_env)
    assert not re.search(r"setattr\(\s*self\.sensors", src)
    for attr in ("gps_latency", "mag_bias_max", "gps_pos_sigma", "gyro_sigma", "mag_sigma"):
        assert not re.search(rf"self\.sensors\.{attr}\s*(=[^=]|\*=)", src), attr
    # the per-env entry points are what the env calls instead
    assert "set_noise(" in src and "set_params(" in src


def test_episode_starts_with_the_estimator_at_the_spawn_pose():
    """R3 §1: reset seeds `pos_hat`/`gps_enu`/the GPS ring with the spawn pose, so the
    first estimate is where the vessel is — not the ENU origin (the old phantom error)."""
    env = AsvEnv(cfg=_cfg(), num_envs=N, device="cpu", seed=5)
    env.reset()
    truth = env.state[:, :2]
    err = (env.meas["est"][:, :2] - truth).norm(dim=1)
    far = truth.norm(dim=1) > 5.0                    # envs whose old error was ≥ 5 m
    assert bool(far.any())
    assert float(err[far].max()) < 2.0, float(err[far].max())   # pos_noise_m = 0.4
