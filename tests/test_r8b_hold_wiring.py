"""Phase 1 (R8b-1): a REAL sk_wind rollout must produce a finite hold_rms_m everywhere.

Before the fix, `evaluate.py` never consumed the scenario's `mode` field, so `sk_wind` ran
at the training mix (`station_keeping_fraction: 0.3`) and roughly 70 % of its episodes were
waypoint episodes with no final hold.  `final_hold_window` then correctly returned (-1, -1)
for those and `hold_rms_m` was nan -- the metric was right and the scenario was wrong.

These are real rollouts on the real checkpoint (8 envs, short episodes), not fakes: the bug
was a wiring one between scenario definition and env construction, which no synthetic trace
can catch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import evaluate as E
from models import build_policy

CKPT = E.ROOT / "runs" / "r8_A" / "ckpt" / "ckpt_last.pt"

pytestmark = pytest.mark.skipif(
    not CKPT.exists(), reason=f"checkpoint absent ({CKPT})")


def _policy():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    pol = build_policy(ck["cfg"])
    pol.load_state_dict(ck["model"])
    pol.eval()
    return pol


def _rollout(scen_name: str, episode_s: float = 40.0, n_envs: int = 8, seed: int = 0):
    scen = dict(E.SCENARIOS[scen_name])
    scen["episode_s"] = episode_s
    rec = E.run_rollout("ppo", _policy(), torch.device("cpu"), scen, seed=seed,
                        episodes=1, num_envs=n_envs, rough_scale=2.0,
                        square_side=E.SQUARE_SIDE_M, hold_s=10.0, dwell_stay=1.0,
                        env_index=1)
    meta = rec.pop("_meta")
    for k in ("_path", "_waypoints", "_arrival", "_conditions", "_log_std"):
        rec.pop(k, None)
    ep = E.episode_arrays(rec, meta["max_steps"], 1)
    t = np.arange(ep["xy"].shape[2], dtype=np.float64) * float(meta["policy_dt"])
    return ep, t


def test_sk_wind_forces_station_keeping_on_every_env():
    """The scenario says mode=1; every env must actually run station keeping."""
    ep, _ = _rollout("sk_wind")
    mode = np.asarray(ep["mode"])[0]                 # [N, T]
    frac = (mode > 0.5).mean(axis=1)
    assert np.all(frac == 1.0), (
        f"sk_wind envs with station-keeping fraction {frac.tolist()}; "
        "the scenario's mode is not reaching the env's station_keeping_fraction")


def test_hold_rms_is_finite_on_every_complete_sk_wind_episode():
    from evalkit import metrics as M

    ep, t = _rollout("sk_wind")
    complete = np.asarray(ep["complete"])[0]         # [N]
    xy = np.asarray(ep["xy"])[0]
    goal = np.asarray(ep["goal_xy"])[0]
    mode = np.asarray(ep["mode"])[0]

    checked = 0
    for i in range(xy.shape[0]):
        if not complete[i]:
            continue
        rms, p95 = M.hold_rms(t, xy[i], goal[i], mode[i])
        assert np.isfinite(rms), f"env {i}: hold_rms_m is nan on a complete sk_wind episode"
        assert np.isfinite(p95), f"env {i}: hold_p95_m is nan on a complete sk_wind episode"
        assert rms >= 0.0 and p95 >= rms
        checked += 1
    assert checked > 0, "no complete episode to check -- the fixture is wrong, not the code"


def test_wp_square_still_reports_nan_hold_because_it_has_no_final_hold():
    """The fix must not invent a hold where the scenario has none."""
    from evalkit import metrics as M

    ep, t = _rollout("wp_square", episode_s=40.0)
    xy = np.asarray(ep["xy"])[0]
    goal = np.asarray(ep["goal_xy"])[0]
    mode = np.asarray(ep["mode"])[0]
    finite = 0
    for i in range(xy.shape[0]):
        rms, _ = M.hold_rms(t, xy[i], goal[i], mode[i])
        if np.isfinite(rms):
            finite += 1
    # a 40 s slice of a square mission ends mid-leg for most envs; a hold is the exception
    assert finite < xy.shape[0], (
        f"{finite}/{xy.shape[0]} wp_square envs reported a hold; the episode mode is leaking")


def test_scenario_episode_cfg_maps_mode_to_station_keeping_fraction():
    assert E.scenario_episode_cfg({"mode": 1})["episode"]["station_keeping_fraction"] == 1.0
    assert E.scenario_episode_cfg({"mode": 0})["episode"]["station_keeping_fraction"] == 0.0
    # scenarios without a mode keep the config value untouched
    base = E.env_cfg()["episode"]["station_keeping_fraction"]
    assert E.scenario_episode_cfg({})["episode"]["station_keeping_fraction"] == base
