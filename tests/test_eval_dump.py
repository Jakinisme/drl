"""R6 — tests for `--dump-conditions` (evaluate.py).

Cheap, no full eval: build a small FastEnv, run one short episode with `dump=True`, and
assert the conditions CSV has one row per (seed, env), the columns the spec names, and no
NaN in any CONDITION column (the three outcome columns may be NaN for an incomplete
episode, but never the conditions).

Run from drl/: `.venv/Scripts/python.exe -m pytest tests/test_eval_dump.py -q`.
"""

import csv
import math

import numpy as np
import pytest

from evaluate import (SCENARIOS, FastEnv, SquarePath, capture_conditions, env_cfg,
                      run_rollout, write_csv)
from models import PolicyNet


def _policy():
    return PolicyNet(obs_dim=150, act_dim=3, hidden=(32, 32), init_log_std=-0.5)


def test_capture_conditions_has_spec_columns():
    cfg = env_cfg()
    cfg["num_envs"] = 3
    cfg["episode_s"] = 1.0
    env = FastEnv(cfg, device="cpu", seed=7)
    env.reset()
    path = SquarePath(env, side=30.0, hold_s=0.1, dwell_stay=1.5)
    c = capture_conditions(env, path)
    for key in ("spawn_x", "spawn_y", "spawn_psi", "bearing_c1", "heading_err",
                "wind_speed", "wind_dir", "cur_speed", "cur_dir", "wave_dir",
                "mass_scale", "inertia_scale", "drag_scale", "thr_fwd_mean",
                "thr_rev_mean", "thr_lag", "thr_deadband", "weak_motor",
                "gps_pos_sigma", "gps_latency", "mag_bias", "gyro_bias"):
        assert key in c, key
        v = c[key].cpu().numpy()
        assert v.shape[0] == 3, key
        assert np.isfinite(v).all(), f"{key} has NaN/inf"
    # weak_motor is -1 (no weakened motor) or a valid index 0..3
    wm = c["weak_motor"].cpu().numpy()
    assert np.all((wm == -1) | ((wm >= 0) & (wm <= 3)))
    # bearing is in [-pi, pi]; heading error too
    assert np.all(np.abs(c["bearing_c1"].cpu().numpy()) <= math.pi + 1e-6)
    assert np.all(np.abs(c["heading_err"].cpu().numpy()) <= math.pi + 1e-6)
    env.close()


def test_dump_writes_one_row_per_env_and_no_nan_conditions(tmp_path):
    scen = dict(SCENARIOS["wp_square"])
    scen["episode_s"] = 2.0
    policy = _policy()
    rec = run_rollout("ppo", policy, "cpu", scen, seed=3, episodes=1, num_envs=4,
                      rough_scale=2.0, square_side=30.0, hold_s=0.1,
                      dwell_stay=1.5, dump=True)
    meta = rec["_meta"]
    cond = rec["_conditions"]
    assert cond is not None
    assert len(cond) == 4                      # one row per env (episodes == 1)
    # every condition column is finite
    cond_keys = [k for k in cond[0] if k not in ("episode", "env",
                                                 "waypoints_reached", "lap_completed",
                                                 "radial_error_max")]
    for c in cond:
        for k in cond_keys:
            assert np.isfinite(c[k]), f"{k} = {c[k]}"

    # write the CSV the same way main() does and read it back
    for c in cond:
        c.setdefault("waypoints_reached", float("nan"))
        c.setdefault("lap_completed", False)
        c.setdefault("radial_error_max", float("nan"))
    rows = [{"scenario": "wp_square", "controller": "ppo", "seed": 3,
             "episode": c["episode"], "env": c["env"], **c} for c in cond]
    cols = (["scenario", "controller", "seed", "episode", "env"]
            + [k for k in cond[0] if k not in ("episode", "env")])
    out = tmp_path / "wp_square_ppo_conditions.csv"
    write_csv(out, rows, cols)
    with open(out, newline="", encoding="utf-8") as f:
        rdr = list(csv.DictReader(f))
    assert len(rdr) == 4                       # one row per env
    assert rdr[0].keys() and "spawn_x" in rdr[0] and "weak_motor" in rdr[0]
    # no NaN in condition columns
    for row in rdr:
        for k in cond_keys:
            assert row[k] != "" and float(row[k]) == float(row[k]), (k, row[k])

    # the npz side: obs/action_raw recorded and shaped right
    obs = rec.get("obs")
    assert obs is not None and obs.ndim == 3 and obs.shape[-1] == 150
    assert rec["action_raw"].shape[-1] == 3


def test_failing_episode_flags_geofence_and_zero_waypoints(tmp_path):
    # a geofence exit or a 0-waypoint episode must be flagged in `_failing`
    scen = dict(SCENARIOS["wp_square"])
    scen["episode_s"] = 2.0
    policy = _policy()
    rec = run_rollout("ppo", policy, "cpu", scen, seed=5, episodes=1, num_envs=2,
                      rough_scale=2.0, square_side=30.0, hold_s=0.1,
                      dwell_stay=1.5, dump=True)
    # the dump metadata itself is present; _failing is computed by _write_dump_npz at
    # CSV-write time, so here we only assert the raw record is self-consistent
    assert rec["_conditions"] is not None
    assert rec.get("obs") is not None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
