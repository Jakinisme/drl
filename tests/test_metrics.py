"""Tests for evalkit: every metric checked against a value computable by hand.

Synthetic trajectories only (straight lines, circles, constant error, constant thrust).
Nothing here is a statement about control performance of the vehicle -- the physical
model is placeholder-parameterised, so these numbers test arithmetic, not the vessel.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from evalkit.metrics import (action_rate, cross_track_rmse, energy, mean_p95,
                             radial_error, settling_time, summarize)

DRL = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- radial error
def test_radial_error_straight_line_3_4_5():
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    goal = np.zeros((3, 2))
    np.testing.assert_allclose(radial_error(xy, goal), [0.0, 3.0, 4.0])


def test_radial_error_broadcasts_single_goal_and_offsets():
    xy = np.array([[1.0, 1.0], [4.0, 5.0]])
    # goal (1,1) -> distances 0 and hypot(3,4) = 5
    np.testing.assert_allclose(radial_error(xy, np.array([1.0, 1.0])), [0.0, 5.0])


def test_radial_error_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    xy = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    goal = torch.zeros(2, 2)
    out = radial_error(xy, goal)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [5.0, 0.0])


# ---------------------------------------------------------------- mean / p95
def test_mean_p95_linear_interpolation_on_known_ramp():
    err = np.arange(100, dtype=float)          # 0..99
    mean, p95 = mean_p95(err)
    assert mean == pytest.approx(49.5)
    # linear interpolation: rank = 0.95 * 99 = 94.05 -> 94 + 0.05
    assert p95 == pytest.approx(94.05)


def test_mean_p95_constant_and_single_sample():
    assert mean_p95(np.full(7, 2.0)) == pytest.approx((2.0, 2.0))
    assert mean_p95(np.array([1.25])) == pytest.approx((1.25, 1.25))
    # single sample is its own p95 -- the boundary case where p95 == max
    err = np.array([0.0, 10.0])
    assert mean_p95(err) == pytest.approx((5.0, 9.5))


def test_mean_p95_ignores_nan_and_handles_empty():
    assert mean_p95(np.array([1.0, np.nan, 3.0]))[0] == pytest.approx(2.0)
    m, p = mean_p95(np.array([]))
    assert np.isnan(m) and np.isnan(p)


# ---------------------------------------------------------------- cross-track
def test_cross_track_rmse_perpendicular_offset():
    path = np.array([[0.0, 0.0], [10.0, 0.0]])
    xy = np.array([[1.0, 1.0], [2.0, -1.0]])
    assert cross_track_rmse(xy, path) == pytest.approx(1.0)
    # constant 0.5 m offset -> RMSE 0.5 (point-to-segment, not to the waypoint)
    xy = np.array([[0.0, 0.5], [5.0, 0.5], [10.0, 0.5]])
    assert cross_track_rmse(xy, path) == pytest.approx(0.5)


def test_cross_track_rmse_clamps_beyond_path_ends():
    # only one path point -> distance to that point: hypot(3,4) = 5
    assert cross_track_rmse(np.array([[-3.0, 4.0]]), np.array([[0.0, 0.0]])) == pytest.approx(5.0)
    # past the end of a segment: projection clamps to the last vertex
    path = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert cross_track_rmse(np.array([[13.0, 4.0]]), path) == pytest.approx(5.0)


def test_cross_track_rmse_on_path_is_zero():
    path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    xy = np.array([[5.0, 0.0], [10.0, 5.0]])
    assert cross_track_rmse(xy, path) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- settling time
def _settle_trace():
    t = np.arange(0.0, 10.0, 0.5)              # 20 samples, 0 .. 9.5
    err = np.full(t.size, 0.4)
    err[:4] = [5.0, 4.0, 3.0, 2.0]             # out of band for the first 4 samples
    return t, err


def test_settling_time_clear_signal():
    t, err = _settle_trace()
    # last out-of-band sample is t = 1.5, so the band starts at t = 2.0
    assert settling_time(t, err, tol=1.0, hold_s=5.0) == pytest.approx(2.0)


def test_settling_time_requires_the_band_to_be_held():
    t, err = _settle_trace()
    # in-band tail is 9.5 - 2.0 = 7.5 s, so a 10 s hold requirement fails
    assert settling_time(t, err, tol=1.0, hold_s=10.0) == float("inf")


def test_settling_time_never_settles_and_nan_counts_as_bad():
    t = np.arange(0.0, 10.0, 0.5)
    err = np.full(t.size, 0.1)
    err[-1] = 5.0                              # leaves the band again -> not settled
    assert settling_time(t, err, tol=1.0) == float("inf")
    err = np.full(t.size, 0.1)
    err[-1] = np.nan
    assert settling_time(t, err, tol=1.0) == float("inf")


def test_settling_time_already_settled_at_start_is_zero():
    t = np.arange(0.0, 10.0, 0.5)
    assert settling_time(t, np.zeros(t.size), tol=1.0, hold_s=5.0) == pytest.approx(0.0)


# ---------------------------------------------------------------- energy / rate
def test_energy_constant_command():
    t = np.arange(0.0, 10.0 + 1e-9, 0.1)        # 101 samples over 10 s
    action = np.full((t.size, 3), 0.5)
    # sum_i a_i^2 = 3 * 0.25 = 0.75, times 10 s
    assert energy(t, action) == pytest.approx(7.5)


def test_energy_zero_and_degenerate():
    t = np.arange(0.0, 1.0, 0.1)
    assert energy(t, np.zeros((t.size, 3))) == pytest.approx(0.0)
    assert energy(np.array([0.0]), np.zeros((1, 3))) == 0.0


def test_energy_uses_all_three_axes():
    t = np.array([0.0, 2.0])
    action = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])   # 1 -> 2 s, trapezoid exact
    assert energy(t, action) == pytest.approx(2.0)


def test_action_rate_step_and_constant():
    t = np.array([0.0, 1.0, 2.0])
    step = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # total variation ||1|| = 1 over 2 s
    assert action_rate(t, step) == pytest.approx(0.5)
    assert action_rate(t, np.zeros((3, 3))) == pytest.approx(0.0)


def test_action_rate_uses_l2_norm_of_the_delta():
    t = np.array([0.0, 1.0])
    action = np.array([[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]])   # ||delta|| = 1
    assert action_rate(t, action) == pytest.approx(1.0)


# ---------------------------------------------------------------- summarize
def test_summarize_circle_around_goal():
    t = np.arange(0.0, 20.0, 0.5)
    radius = 2.0
    ang = np.linspace(0.0, 2.0 * np.pi, t.size)
    traj = {
        "t": t,
        "xy": radius * np.stack([np.cos(ang), np.sin(ang)], axis=1),
        "psi": np.zeros(t.size),
        "action": np.zeros((t.size, 3)),
        "goal_xy": np.zeros((t.size, 2)),      # station keeping at the origin
        "mode": np.ones(t.size),
    }
    m = summarize(traj, tol=3.0)
    assert m["n_samples"] == t.size
    assert m["duration_s"] == pytest.approx(19.5)
    for key in ("radial_error_mean", "radial_error_p95", "radial_error_max"):
        assert m[key] == pytest.approx(radius)          # error is exactly R everywhere
    assert m["cross_track_rmse"] == pytest.approx(radius)  # single hold point
    assert m["energy"] == pytest.approx(0.0)
    assert m["action_rate"] == pytest.approx(0.0)
    assert m["settling_time_s"] == pytest.approx(0.0)   # tol 3 > R=2, held all along
    assert summarize(traj, tol=1.0)["settling_time_s"] == float("inf")


def test_summarize_waypoint_uses_goal_sequence_as_path():
    t = np.arange(0.0, 11.0, 1.0)
    xy = np.column_stack([np.zeros(t.size), np.full(t.size, 0.5)])  # 0.5 m north of y=0
    goal = np.array([[0.0, 0.0]] * 6 + [[10.0, 0.0]] * 5)           # unreached 2nd waypoint
    traj = {"t": t, "xy": xy, "psi": np.zeros(t.size),
            "action": np.zeros((t.size, 3)), "goal_xy": goal, "mode": np.zeros(t.size)}
    m = summarize(traj)
    assert m["cross_track_rmse"] == pytest.approx(0.5, abs=1e-12)
    # explicit path wins over the derived one
    m2 = summarize({**traj, "path": np.array([[0.0, 10.0], [10.0, 10.0]])})
    assert m2["cross_track_rmse"] == pytest.approx(9.5)


# ---------------------------------------------------------------- import hygiene
def test_metrics_module_does_not_import_matplotlib():
    code = textwrap.dedent(
        "import sys; import evalkit.metrics; "
        "assert 'matplotlib' not in sys.modules, 'metrics must not pull in matplotlib'"
    )
    subprocess.run([sys.executable, "-c", code], cwd=DRL, check=True)


def test_plots_use_agg_backend_and_write_pngs(tmp_path):
    import matplotlib
    from evalkit import plots

    assert matplotlib.get_backend().lower() == "agg"
    t = np.arange(0.0, 10.0, 0.5)
    traj = {
        "t": t,
        "xy": np.column_stack([t, np.full(t.size, 0.5)]),
        "psi": np.zeros(t.size),
        "action": np.full((t.size, 3), 0.2),
        "goal_xy": np.column_stack([t, np.zeros(t.size)]),
        "mode": np.ones(t.size),
        "path": np.array([[0.0, 0.0], [9.5, 0.0]]),
    }
    outs = [tmp_path / "err.png", tmp_path / "traj.png", tmp_path / "table.png"]
    figs = [plots.plot_error(traj, out=outs[0]),
            plots.plot_trajectory(traj, out=outs[1]),
            plots.summary_table({"pid": summarize(traj), "ppo": summarize(traj)},
                                out=outs[2])]
    for p in outs:
        assert p.exists() and p.stat().st_size > 1000
    for f in figs:
        matplotlib.pyplot.close(f)
