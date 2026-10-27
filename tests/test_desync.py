"""R5-A: `AsvEnv.desync_episodes` staggers training episode phase.

Checks, against the real env on CPU:
1. after desync, `ep_step` spans most of [0, max_steps);
2. state / goals / obs are NOT touched by desync, and the env's physics RNG stream
   (`env._gen`) is not consumed by it;
3. a roll-out of 1.5*max_steps produces timeouts at >= 5 distinct step indices
   (instead of all envs at one);
4. `reset(ids)` still gives `ep_step == 0`, so evaluate.py keeps full-length episodes;
5. the trainer calls it once, right after its initial reset, with its own generator.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from envs.asv_env import AsvEnv, load_config

N = 64
RUN_CFG = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"

def _env(seed: int = 0, num_envs: int = N) -> AsvEnv:
    cfg = copy.deepcopy(load_config())
    cfg["num_envs"] = num_envs
    cfg["randomization"] = {k: (v if k == "weak_factor" else 0.0)
                            for k, v in cfg["randomization"].items()}
    return AsvEnv(cfg=cfg, num_envs=num_envs, device="cpu", seed=seed)


def test_step_count_spans_range():
    env = _env()
    gen = torch.Generator().manual_seed(1)
    env.desync_episodes(gen)
    m = env.max_steps
    s = env.ep_step
    assert s.dtype == torch.long and s.shape == (N,)
    assert int(s.min()) >= 0 and int(s.max()) < m
    assert int(s.min()) <= m // 10, f"min ep_step {int(s.min())} not near 0"
    assert int(s.max()) >= 9 * m // 10, f"max ep_step {int(s.max())} not near max_steps"
    # uniform-ish: every decile of [0, max_steps) hit, counts within loose bounds
    per = torch.bincount((s * 10 // m).clamp(max=9), minlength=10)
    assert per.numel() == 10 and int((per > 0).sum()) == 10, per.tolist()
    exp = N / 10
    assert bool((per > 0.25 * exp).all()), per.tolist()


def test_desync_leaves_state_goals_obs_and_physics_rng_alone():
    env = _env()
    st0 = env.state.clone()
    goal0 = env.goal_xy.clone()
    wps0 = env.wps.clone()
    obs0 = env.obs_buf.clone()
    meas0 = env.meas["est"].clone()
    rng0 = env._gen.get_state().clone()
    env.desync_episodes(torch.Generator().manual_seed(7))
    assert torch.equal(env.state, st0)
    assert torch.equal(env.goal_xy, goal0)
    assert torch.equal(env.wps, wps0)
    assert torch.equal(env.obs_buf, obs0)
    assert torch.equal(env.meas["est"], meas0)
    assert torch.equal(env._gen.get_state(), rng0), "desync consumed the physics RNG"
    # reproducible from the trainer generator
    env.desync_episodes(torch.Generator().manual_seed(7))
    a = env.ep_step.clone()
    env.desync_episodes(torch.Generator().manual_seed(7))
    assert torch.equal(env.ep_step, a)


def test_timeouts_land_on_many_step_indices():
    env = _env()
    env.desync_episodes(torch.Generator().manual_seed(3))
    m = env.max_steps
    seen: set[int] = set()
    zero = torch.zeros(N, 3)
    start = env.ep_step.clone()          # info["ep_step"] is post-reset -> reconstruct
    for _ in range(int(1.5 * m)):
        _obs, _rew, done, _info = env.step(zero)
        d = done.bool()
        if bool(d.any()):
            seen.update((start[d] + 1).tolist())
            start[d] = env.ep_step[d]    # fresh episodes, counted from their own start
    assert len(seen) >= 5, f"timeouts at only {sorted(seen)}"


def test_reset_ids_zeroes_step_count():
    env = _env()
    env.desync_episodes(torch.Generator().manual_seed(5))
    assert int(env.ep_step.max()) > 0
    ids = torch.arange(0, 8)
    env.reset(ids)
    assert bool((env.ep_step[ids] == 0).all())
    assert int(env.ep_step[8:].min()) >= 0        # others untouched by reset(ids)
    env.reset()
    assert int(env.ep_step.abs().max()) == 0


def test_trainer_calls_desync_once_after_reset():
    pytest.importorskip("train")
    from train import PPOTrainer, load_config as load_train_cfg

    cfg = copy.deepcopy(load_train_cfg(RUN_CFG))
    cfg.update({"num_envs": 32, "device": "cpu", "rollout_steps": 4})
    env = _env(num_envs=32)
    tr = PPOTrainer(cfg, env=env, writer=None)
    assert int(env.ep_step.max()) > 0, "trainer did not stagger episode phase"
    assert int(env.ep_step.min()) >= 0 and int(env.ep_step.max()) < env.max_steps
    tr.collect()                                  # still runs with a desynced env
