"""Disturbance tests: level 0 exact zeros, shape/dtype, level monotonicity,
determinism, per-env reset isolation, set_level latency, and statelessness in t.

Contract: CONTRACT.md §3.2 (config), §4.3 (interface: `level = 0` => exact zeros,
stateless w.r.t. `t` except wave phases fixed at `reset()`).
"""

import copy
import math

import pytest
import torch

from sim.disturbance import Disturbance, load_config

CFG = load_config()
DEVICE = "cpu"
N = 4
DT = 0.05


def cfg_with(**overrides) -> dict:
    c = copy.deepcopy(CFG)
    for dotted, value in overrides.items():
        node = c
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return c


def dist(n: int = N, *, seed: int | None = None, dtype=torch.float32,
         cfg: dict | None = None) -> Disturbance:
    c = copy.deepcopy(CFG) if cfg is None else copy.deepcopy(cfg)
    if seed is not None:
        c["seed_offset"] = seed
    return Disturbance(c, n, device=DEVICE, dtype=dtype)


def state(n: int = N, *, u: float = 0.0, v: float = 0.0, r: float = 0.0,
          dtype=torch.float32) -> torch.Tensor:
    s = torch.zeros(n, 6, device=DEVICE, dtype=dtype)
    s[:, 3] = u
    s[:, 4] = v
    s[:, 5] = r
    return s


def test_level_zero_is_exact_zero_everywhere():
    for dtype in (torch.float32, torch.float64):
        d = dist(dtype=dtype)
        d.reset(level=0.0)
        s = state(u=0.7, v=-0.3, r=0.2, dtype=dtype)
        assert torch.equal(d.level, torch.zeros(N, device=DEVICE, dtype=dtype))
        for k in range(200):
            out = d.sample(k * DT, DT, s)
            assert torch.isfinite(out).all()
            assert torch.equal(out, torch.zeros(N, 3, device=DEVICE, dtype=dtype))


def test_shape_dtype_device():
    for dtype in (torch.float32, torch.float64):
        d = dist(dtype=dtype)
        d.set_level(1.0)
        out = d.sample(0.4, DT, state(u=0.5, v=0.1, r=0.05, dtype=dtype))
        assert tuple(out.shape) == (N, 3)
        assert out.dtype == dtype
        assert out.device.type == DEVICE


def test_level_monotonicity_rms():
    d = dist(n=2)
    s = state(n=2, u=0.6, v=0.2, r=0.1)
    rms = {}
    for level in (0.0, 0.5, 1.0):
        d.set_level(level)
        acc = []
        for k in range(200):
            acc.append(d.sample(k * DT, DT, s))
        rms[level] = torch.stack(acc).pow(2).mean().sqrt().item()
    assert rms[0.0] == 0.0
    assert rms[0.5] > rms[0.0]
    assert rms[1.0] > rms[0.5]


def test_determinism_same_seed_and_seed_sensitivity():
    s = state(u=0.4, v=0.15, r=0.1)
    a, b = dist(seed=7), dist(seed=7)
    a.set_level(1.0)
    b.set_level(1.0)
    for k in range(50):
        out_a = a.sample(k * DT, DT, s)
        out_b = b.sample(k * DT, DT, s)
        assert torch.equal(out_a, out_b)

    c = dist(seed=8)
    c.set_level(1.0)
    assert not torch.equal(a.sample(0.3, DT, s), c.sample(0.3, DT, s))


def test_reset_only_affects_given_envs():
    s = state(u=0.3, v=0.1, r=0.05)
    d = dist(seed=3)
    ref = dist(seed=3)          # same seed => identical per-env randomness
    for k in range(10):
        assert torch.equal(d.sample(k * DT, DT, s), ref.sample(k * DT, DT, s))

    phase_before = d.wave_phase.clone()
    wind_before = d.wind_speed.clone()
    d.reset(torch.tensor([0], dtype=torch.long), level=1.0)

    # untouched env 1 is bit-identical in every per-episode field
    for name in ("wind_speed", "wind_dir", "wave_dir", "wave_phase", "wave_phase_yaw"):
        assert torch.equal(getattr(d, name)[1:], getattr(ref, name)[1:]), name
    # ... and its sample matches the no-reset run, while env 0 changed
    assert torch.equal(d.sample(0.7, DT, s)[1:], ref.sample(0.7, DT, s)[1:])
    assert not torch.equal(d.wave_phase[0], phase_before[0])
    assert not torch.equal(d.wind_speed[0], wind_before[0])
    assert d.level.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert not torch.equal(d.sample(0.7, DT, s)[0], ref.sample(0.7, DT, s)[0])


def test_set_level_takes_effect_on_next_sample():
    d = dist(seed=11)
    d.set_level(1.0)
    s = state(u=0.5, v=0.2, r=0.1)
    t = 1.234
    out_on = d.sample(t, DT, s)
    assert out_on.abs().sum() > 0

    d.set_level(0.0)
    out_off = d.sample(t, DT, s)          # same t, same state
    assert torch.equal(out_off, torch.zeros(N, 3, device=DEVICE, dtype=torch.float32))

    d.set_level(1.0)
    assert torch.equal(d.sample(t, DT, s), out_on)


def test_wind_and_current_are_stateless_in_t():
    cfg = cfg_with(**{"waves.enabled": False})
    d = dist(seed=5, cfg=cfg)
    d.set_level(1.0)
    s = state(u=0.5, v=0.2, r=0.1)
    first = d.sample(0.0, DT, s)
    assert torch.equal(d.sample(1234.5, DT, s), first)   # t unused when waves off
    assert torch.equal(d.sample(0.0, DT * 10, s), first)  # dt unused


def test_wave_phase_depends_on_t_and_call_history_is_irrelevant():
    cfg = cfg_with(**{"wind.enabled": False, "current.enabled": False})
    d = dist(seed=9, cfg=cfg)
    d.set_level(1.0)
    s = state()
    ph = d.wave_phase.clone()
    t = 2.0
    out = d.sample(t, DT, s)
    d.sample(t + 10.0, DT, s)             # a later call must not disturb phase
    assert torch.equal(d.wave_phase, ph)
    assert torch.equal(d.sample(t, DT, s), out)
    # waves actually advance with t
    assert not torch.equal(d.sample(t + 0.5, DT, s), out)
