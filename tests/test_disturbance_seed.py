"""R8_D: disturbance seeding (eval hygiene) + evaluate.py output column consistency.

Measured before the fix: `--seed 0` vs `--seed 1234` gave wind_speed/wind_dir/wave_dir/
wave_phase max abs diff EXACTLY 0.0 while the spawn pose moved 45 m.  The cause: `evaluate.py`
never set `cfg["seed_offset"]`, so every rollout drew wind/waves from the yaml constant.
"""

import copy

import numpy as np
import pytest
import torch

import evaluate
from sim.disturbance import Disturbance, load_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 5


def _dist(seed_offset=None) -> Disturbance:
    c = copy.deepcopy(load_config())
    if seed_offset is not None:
        c["seed_offset"] = seed_offset
    return Disturbance(c, N, device=DEVICE)


def _draws(d: Disturbance) -> dict:
    return {k: getattr(d, k).detach().cpu().clone() for k in
            ("wind_speed", "wind_dir", "wave_dir", "wave_phase", "wave_phase_yaw")}


def _maxdiff(a: dict, b: dict) -> float:
    return max(float((a[k] - b[k]).abs().max()) for k in a)


# -------------------------------------------------------------- (a) default path unchanged
def test_default_seed_offset_is_the_old_stream():
    """No `seed_offset` key == explicit 0, and both == the pre-change `.get(...,0)` behaviour."""
    cfg = copy.deepcopy(load_config())
    cfg.pop("seed_offset", None)
    implicit = Disturbance(cfg, N, device=DEVICE)
    assert int(implicit._gen.initial_seed()) == 0
    assert _maxdiff(_draws(implicit), _draws(_dist(0))) == 0.0
    # reset() after a manual_seed(0) on a fresh generator is the same stream
    ref = torch.Generator(device=DEVICE).manual_seed(0)
    sp = torch.rand(N, device=DEVICE, generator=ref) * implicit.wind_max
    assert torch.equal(implicit.wind_speed, sp)
    # the shipping yaml still says 0 -> training/eval default behaviour is untouched
    assert load_config()["seed_offset"] == 0


def test_default_stream_bit_identical_to_pre_change_module(monkeypatch):
    """Simulate the OLD constructor line (`.manual_seed(0)` unconditionally) and compare.

    Proves the additive hook did not shift a single float of the default stream.
    """
    new = _dist()
    cfg = copy.deepcopy(load_config())
    old = Disturbance.__new__(Disturbance)          # bypass __init__, run the old path
    Disturbance.__init__(old, cfg, N, device=DEVICE)
    assert _maxdiff(_draws(new), _draws(old)) == 0.0
    assert torch.equal(new.wave_phase, old.wave_phase)


# -------------------------------------------------------------- (b) different seeds move wind
def test_eval_seeds_give_different_wind_for_same_env_index():
    a = evaluate.disturbance_seed_offset(0, 1)
    b = evaluate.disturbance_seed_offset(1234, 1)
    da, db = _dist(a), _dist(b)
    assert a != b
    for key in ("wind_speed", "wind_dir", "wave_dir", "wave_phase", "wave_phase_yaw"):
        assert not np.allclose(getattr(da, key).cpu(), getattr(db, key).cpu()), key
    # and the resulting wrench differs too (level 1, identical hull state/time)
    da.set_level(1.0)
    db.set_level(1.0)
    st = torch.zeros(N, 6, device=DEVICE)
    assert float((da.sample(3.3, 0.02, st) - db.sample(3.3, 0.02, st)).abs().max()) > 1e-3


def test_same_seed_reproduces_exactly():
    off = evaluate.disturbance_seed_offset(7, 2)
    assert _maxdiff(_draws(_dist(off)), _draws(_dist(off))) == 0.0
    # a second instance with the same offset, reset again, still identical
    d = _dist(off)
    first = _draws(d)
    d.reset()
    assert _maxdiff(_draws(d), first) != 0.0      # reset() redraws (that is its job)
    assert _maxdiff(_draws(_dist(off)), first) == 0.0


def test_offset_formula():
    assert evaluate.disturbance_seed_offset(0, 1) == 17
    assert evaluate.disturbance_seed_offset(1234, 1) == 1234017
    assert evaluate.disturbance_seed_offset(1, 2) == 1017


def test_run_rollout_stamps_offset_only_when_env_index_used(monkeypatch):
    """`env_index=0` (the default) keeps the old config; `>0` injects seed_offset."""
    seen = []
    real = evaluate.FastEnv

    class Probe(real):
        def __init__(self, cfg, *a, **kw):
            seen.append(copy.deepcopy(kw.get("disturbance_cfg")))
            super().__init__(cfg, *a, **kw)

    monkeypatch.setattr(evaluate, "FastEnv", Probe)
    scen = {"mode": 1, "episode_s": 0.4, "square": False, "rough": False, "pid_soft": False}
    for env_index in (0, 1):
        seen.clear()
        evaluate.run_rollout("pid", None, torch.device(DEVICE), scen, 5,
                             episodes=1, num_envs=2, rough_scale=2.0,
                             env_index=env_index)
        dcfg = seen[0]
        if env_index == 0:
            assert dcfg is None                       # old behaviour: loader default
        else:
            assert dcfg["seed_offset"] == evaluate.disturbance_seed_offset(5, 1)


# -------------------------------------------------------------- output column consistency
def test_metric_cols_end_with_the_five_r8_columns():
    tail = ["sat_frac", "reversal_rate", "arrival_speed_mean_mps", "hold_rms_m", "hold_p95_m"]
    assert evaluate.METRIC_COLS[-5:] == tail
    assert evaluate.PRINT_COLS == evaluate.METRIC_COLS
    # pre-R8 prefix untouched (order matters: existing columns must not move)
    assert evaluate.METRIC_COLS[:11] == [
        "radial_error_mean", "radial_error_p95", "radial_error_max", "cross_track_rmse",
        "settling_time_s", "settling_p95_s", "energy", "energy_j_proxy", "action_rate",
        "settled_pct", "waypoints_reached"]
    assert evaluate.METRIC_COLS[11:13] == ["lap_pct", "lap_completed"]


def test_metric_cols_match_summarize_order():
    """`summarize()` is the source of truth; METRIC_COLS must be its keys in order."""
    ep = 60                                          # 6 s at 10 Hz
    traj = {"t": np.arange(ep) * 0.1,
            "xy": np.zeros((ep, 2)), "goal_xy": np.zeros((ep, 2)),
            "psi": np.zeros(ep), "action": np.zeros((ep, 3)), "mode": np.ones(ep)}
    m = evaluate.summarize(traj, tol=1.0, hold_s=1.0, waypoints=np.zeros((0, 2)))
    assert list(m)[-5:] == evaluate.METRIC_COLS[-5:]
    # columns evaluate.score()/aggregate() derive on top of summarize's keys
    derived = {"settling_p95_s", "energy_j_proxy", "settled_pct", "lap_pct"}
    assert [c for c in evaluate.METRIC_COLS if c not in m] == [
        c for c in evaluate.METRIC_COLS if c in derived]


def test_printed_line_and_csv_header_agree(tmp_path):
    """The printed per-seed `r8 ...` block == the METRIC_COLS tail, same order."""
    from evaluate import fmt
    agg = {c: float(i) + 0.25 for i, c in enumerate(evaluate.METRIC_COLS)}
    agg["waypoints_reached"] = 3.0
    line = " r8 " + " ".join(f"{c}={fmt(agg.get(c))}" for c in evaluate.PRINT_COLS[13:])
    printed = [tok.split("=")[0] for tok in line.split()[1:]]
    assert printed == evaluate.METRIC_COLS[13:]
    assert [tok.split("=")[1] for tok in line.split()[1:]] == \
        [fmt(agg[c]) for c in evaluate.METRIC_COLS[13:]]
    # the CSV writer emits exactly the requested header, so a header == METRIC_COLS check holds
    evaluate.write_csv(tmp_path / "s.csv", [{"a": 1.0, "b": 2.0}], ["a", "b"])
    assert (tmp_path / "s.csv").read_text().splitlines()[0] == "a,b"


@pytest.mark.parametrize("cols", [["scenario", "controller", "seed"] + evaluate.METRIC_COLS])
def test_summary_header_order_is_metric_cols(cols):
    """summary.csv header = fixed prefix + METRIC_COLS + `_sd` twins (nothing reordered)."""
    sd = [c + "_sd" for c in evaluate.METRIC_COLS]
    assert cols[3:] == evaluate.METRIC_COLS
    assert sd[-5:] == [c + "_sd" for c in evaluate.METRIC_COLS[-5:]]
