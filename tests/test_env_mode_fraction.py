"""Regression: `station_keeping_fraction` must draw the fraction it was asked for.

`AsvEnv.reset` drew the mode with `self._unif(k) < self.sk_frac`, but `_unif` returns
U(-1, 1), not U(0, 1).  The realised station-keeping fraction was therefore
`(sk_frac + 1) / 2`: a config of 0.0 still produced ~50 % SK episodes and the shipped
0.3 produced ~65 %, so every R4-R7 training run saw about twice the station keeping it
was configured for.

`evaluate.py` was unaffected (it overwrites `env.mode` from the scenario), which is why
this stayed invisible: the eval numbers were fine and only the training mix was wrong.

These tests are statistical (20 k envs, seed 0) but the tolerance is far inside the
0.5 offset the bug produced, so they cannot pass with the old behaviour.
"""

from __future__ import annotations

import pytest


from envs.asv_env import AsvEnv


def _env(sk_frac: float, n: int = 20_000) -> AsvEnv:
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(Path("configs/env.yaml").read_text(encoding="utf-8"))
    cfg["episode"]["station_keeping_fraction"] = sk_frac
    cfg["num_envs"] = n
    return AsvEnv(cfg, seed=0, device="cpu")


@pytest.mark.parametrize("sk_frac", [0.0, 0.1, 0.3, 0.5, 0.9, 1.0])
def test_realised_station_keeping_fraction_matches_config(sk_frac):
    env = _env(sk_frac)
    env.reset()
    realised = float(env.mode.mean())
    assert realised == pytest.approx(sk_frac, abs=0.02), (
        f"station_keeping_fraction={sk_frac} realised {realised:.4f}; "
        f"the buggy draw gave {(sk_frac + 1) / 2:.4f}")


def test_zero_fraction_means_no_station_keeping_episodes():
    """The sharpest form of the bug: 0.0 used to give ~50 %."""
    env = _env(0.0)
    env.reset()
    assert float(env.mode.mean()) == 0.0
    assert int(env.n_wp.min()) >= 1, "a waypoint episode must have at least one waypoint"


def test_full_fraction_means_only_station_keeping_episodes():
    env = _env(1.0)
    env.reset()
    assert float(env.mode.mean()) == 1.0
    assert int(env.n_wp.max()) == 0, "a station-keeping episode has no waypoint list"


def test_mode_draw_is_a_pure_function_of_the_configured_fraction():
    """A `sk_frac` of 0 or 1 must be exact, not merely improbable.

    With the old draw, 0.0 still gave ~50 % SK envs because the comparison was against a
    signed uniform. Exactness at the endpoints is the property that makes the bug
    impossible to reintroduce quietly.
    """
    for sk_frac, expected in ((0.0, 0.0), (1.0, 1.0)):
        env = _env(sk_frac, n=4096)
        for _ in range(3):                      # several resets, not one lucky draw
            env.reset()
            assert float(env.mode.mean()) == expected, (
                f"sk_frac={sk_frac} gave {float(env.mode.mean())} after a reset")
