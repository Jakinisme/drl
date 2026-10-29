"""R8 metric tests: every number below is computable by hand from the literal arrays.

Synthetic traces only (constant errors, hand-built 1 m sticks, rectangular actions).
Nothing here says anything about the vehicle -- the physics is placeholder (SYSID.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from evalkit.metrics import (arrival_speed_mean_mps, final_hold_window, hold_rms,
                             reversal_rate, sat_frac, summarize)

R8_COLS = ["sat_frac", "reversal_rate", "arrival_speed_mean_mps", "hold_rms_m", "hold_p95_m"]
# every summarize() key that existed BEFORE R8, in order
PRE_R8_COLS = ["n_samples", "duration_s", "radial_error_mean", "radial_error_p95",
               "radial_error_max", "cross_track_rmse", "settling_time_s", "energy",
               "action_rate", "waypoints_reached", "lap_completed", "time_to_waypoint"]


def _surge(vals):
    a = np.zeros((len(vals), 3))
    a[:, 0] = vals
    return a


# ------------------------------------------------------------------ sat_frac
def test_sat_frac_hand_count_strict_threshold():
    a = np.zeros((6, 3))
    a[:, 0] = [0.5, 0.5, -0.9, 0.2, 0.3, 0.0]          # nothing is > 0.9
    assert sat_frac(a) == 0.0
    a = np.zeros((6, 3))
    a[2, 1] = 0.95                                     # sway, step 2 only
    assert sat_frac(a) == pytest.approx(1.0 / 6.0)
    a = np.zeros((6, 3))
    a[:, 0] = [1.0, 0.5, 0.6, 0.2, 0.3, 0.0]           # only sample 0 is > 0.9
    assert sat_frac(a) == pytest.approx(1.0 / 6.0)
    a = np.zeros((6, 3))
    a[:, 0] = [1.0, 0.95, -0.9, 0.2, 0.3, 0.0]         # 1.0 and 0.95 are > 0.9
    assert sat_frac(a) == pytest.approx(2.0 / 6.0)
    a = np.zeros((6, 3))
    a[:, 0] = [-1.0, 0.95, 0.5, 0.2, 0.3, 0.0]         # -1.0 and 0.95 are > 0.9
    assert sat_frac(a) == pytest.approx(2.0 / 6.0)
    a = np.zeros((6, 3))
    a[:, 0] = [0.0, 0.9, -0.9, 0.0, 0.0, 0.0]          # threshold is strict
    assert sat_frac(a) == 0.0


def test_sat_frac_any_axis_counts_once():
    a = np.array([[0.0, 0.95, 0.0],     # sway saturated -> counts
                  [0.0, 0.0, -0.99],    # yaw saturated -> counts
                  [-0.91, 0.91, 0.0]])  # two axes, still ONE step
    assert sat_frac(a) == pytest.approx(1.0)


def test_sat_frac_all_none_empty():
    assert sat_frac(np.ones((4, 3))) == 1.0
    assert sat_frac(np.full((4, 3), 0.5)) == 0.0
    assert sat_frac(np.zeros((0, 3))) == 0.0


# ------------------------------------------------------------------ reversal_rate
def test_reversal_rate_hand_count():
    # surges: -0.8 -0.8 0.8 | 0.2 | 0.8 -0.8 ; t = 0..5 s (span 5 s)
    # flips: idx2 (-0.8 -> 0.8) and idx5 (0.8 -> -0.8), both sides gated -> 2
    #        idx3 / idx4 each have a 0.2 side -> not counted
    a = _surge([-0.8, -0.8, 0.8, 0.2, 0.8, -0.8])
    assert reversal_rate(np.arange(6.0), a) == pytest.approx(24.0)


def test_reversal_rate_flip_through_the_deadband_is_not_a_reversal():
    a = _surge([0.8, 0.0, -0.8])                     # each pair has one side under 0.5
    assert reversal_rate(np.array([0.0, 1.0, 2.0]), a) == 0.0


def test_reversal_rate_exact_threshold_is_not_counted():
    a = _surge([0.5, -0.8])                          # 0.5 is not > 0.5
    assert reversal_rate(np.array([0.0, 10.0]), a) == 0.0


def test_reversal_rate_uses_surge_only():
    a = np.zeros((2, 3))
    a[:, 1] = [0.9, -0.9]                            # sway flip ignored
    assert reversal_rate(np.array([0.0, 10.0]), a) == 0.0


def test_reversal_rate_scales_with_span_and_degenerate_input():
    a = _surge([1.0, -1.0, 1.0])                     # 2 flips
    assert reversal_rate(np.array([0.0, 10.0, 20.0]), a) == pytest.approx(6.0)
    assert reversal_rate(np.array([0.0, 5.0, 20.0]), a) == pytest.approx(120.0 / 20.0)
    assert reversal_rate(np.array([3.0, 3.0]), _surge([1.0, -1.0])) == 0.0   # zero span
    assert reversal_rate(np.array([0.0]), _surge([1.0])) == 0.0


# ------------------------------------------------------------------ arrival speed
# 1 m sticks every 0.5 s -> 2 m/s.  Same strict 2.0 m radius as waypoint_progress.
def test_arrival_speed_constant_speed_two_corners():
    xy = np.column_stack([np.arange(21) * 1.0, np.zeros(21)])
    t = np.arange(0.0, 10.5, 0.5)
    c = np.array([[4.0, 0.0], [12.0, 0.0]])
    # corner 1: |x-4| < 2 first at x = 3 (x = 2 is exactly 2 m -> excluded)
    # corner 2: |x-12| < 2 first at x = 11 (x = 10 excluded)
    # both entries 1 m in 0.5 s -> mean 2.0
    assert arrival_speed_mean_mps(t, xy, c) == pytest.approx(2.0)


def test_arrival_speed_uses_the_step_before_entry():
    # x = 0,2,4 (4 m/s) then 4.5, 5 (1 m/s); corner (4,0)
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.5, 0.0], [5.0, 0.0]])
    # x = 2 is exactly 2 m -> not inside; first inside is x = 4 (sample 2)
    # step before entry: 2 m in 0.5 s = 4 m/s (already slow afterwards: not measured)
    assert arrival_speed_mean_mps(t, xy, np.array([[4.0, 0.0]])) == pytest.approx(4.0)


def test_arrival_speed_entry_off_axis_averaged_over_two_corners():
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.5])
    xy = np.array([[0.0, 5.0], [1.0, 5.0], [5.0, 0.0], [6.0, 0.0], [7.0, 0.0]])
    c = np.array([[6.0, 0.0], [20.0, 0.0]])
    # dist to (6,0): sample 0 -> 7.81, 1 -> 7.07, 2 -> 1.0 -> first inside is sample 2
    # step before: |(5,0)-(1,5)| = hypot(4,5) in 0.5 s ; (20,0) never entered -> dropped
    assert arrival_speed_mean_mps(t, xy, c) == pytest.approx(np.hypot(4.0, 5.0) / 0.5)


def test_arrival_speed_both_corners_in_one_sample():
    # both corners entered on the same sample (sample 2): two identical speeds -> mean 2.0
    t = np.arange(7.0)
    xy = np.column_stack([np.array([0.0, 2.0, 4.0, 20.0, 20.0, 15.0, 16.0]), np.zeros(7)])
    c = np.array([[5.0, 0.0], [4.0, 0.0]])
    assert arrival_speed_mean_mps(t, xy, c) == pytest.approx(2.0)


def test_arrival_speed_unreached_missing_and_degenerate_are_nan():
    t = np.arange(4.0)
    xy = np.zeros((4, 2))
    assert np.isnan(arrival_speed_mean_mps(t, xy, np.array([[50.0, 50.0]])))
    assert np.isnan(arrival_speed_mean_mps(t, xy, np.zeros((0, 2))))
    assert np.isnan(arrival_speed_mean_mps(np.array([0.0]), np.zeros((1, 2)),
                                           np.array([[0.0, 0.0]])))
    # corner entered only at the final sample -> speed undefined -> nan
    t3 = np.array([0.0, 1.0, 2.0])
    xy3 = np.array([[100.0, 0.0], [50.0, 0.0], [0.0, 0.0]])
    assert np.isnan(arrival_speed_mean_mps(t3, xy3, np.array([[0.0, 0.0]])))


def test_arrival_speed_start_of_trace_entry_is_nan():
    # spawn already inside the radius: no previous sample -> nan, and the nan entry
    # makes the mean nan rather than silently dropping that waypoint
    t = np.array([0.0, 0.5, 1.0])
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert np.isnan(arrival_speed_mean_mps(t, xy, np.array([[0.0, 0.0]])))


def test_arrival_speed_entry_on_last_sample_is_nan():
    t = np.array([0.0, 1.0, 2.0])
    xy = np.array([[100.0, 0.0], [50.0, 0.0], [0.0, 0.0]])   # 50 / 0 m
    assert np.isnan(arrival_speed_mean_mps(t, xy, np.array([[0.0, 0.0]])))


# ------------------------------------------------------------------ final hold
def test_final_hold_window_splits_on_last_non_sk_sample():
    t = np.arange(10.0)
    assert final_hold_window(t, np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1], float)) == (5, 10)
    assert final_hold_window(t, np.ones(10)) == (0, 10)          # SK the whole episode
    assert final_hold_window(t, np.zeros(10)) == (-1, -1)        # waypoint episode
    assert final_hold_window(t, np.r_[np.ones(5), np.zeros(5)]) == (-1, -1)  # ends on a leg
    assert final_hold_window(np.zeros(0), np.zeros(0)) == (-1, -1)


def test_hold_constant_offset():
    # err = 0.5 m everywhere -> (0.5, 0.5); a constant window makes the count moot
    t = np.arange(100.0)
    xy = np.tile([0.5, 0.0], (100, 1))
    assert hold_rms(t, xy, np.zeros((100, 2)), np.ones(100)) == pytest.approx((0.5, 0.5))


def test_hold_window_is_the_last_30_seconds():
    # window = samples 69..99 (31 samples).  err = 0.5 there -> rms = 0.5; a single 1.0
    # INSIDE the window (t = 69) would give sqrt((30*0.25 + 1)/31) - different, so this
    # pins the window edge; err = 1.0 only OUTSIDE it (t = 68) must not matter.
    t = np.arange(100.0)
    e = np.where(t >= 69.0, 0.5, 2.0)
    xy = np.column_stack([e, np.zeros(100)])
    rms, p95 = hold_rms(t, xy, np.zeros((100, 2)), np.ones(100))
    assert rms == pytest.approx(0.5)
    assert p95 == pytest.approx(0.5)
    # non-constant window: err alternates 0 / 1 across the window -> count matters
    e2 = np.where(t >= 69.0, (t.astype(int) % 2).astype(float), 0.0)
    nz = int(e2[69:].sum())                      # 0.95*... window has 31 samples
    rms2, _ = hold_rms(t, np.column_stack([e2, np.zeros(100)]),
                       np.zeros((100, 2)), np.ones(100))
    assert nz == 16 and rms2 == pytest.approx(np.sqrt(16.0 / 31.0))
    # a 1 m spike exactly at t = 68 (one sample outside the window) is ignored
    e3 = np.where(t == 68.0, 1.0, 0.0)
    assert hold_rms(t, np.column_stack([e3, np.zeros(100)]),
                    np.zeros((100, 2)), np.ones(100)) == (0.0, 0.0)


def test_hold_alternating_error_by_hand():
    # 10 s episode, shorter than 30 s -> the whole trace is the window.
    # err = 0,1,0,1,... -> 5 zeros, 5 ones.  rms = sqrt(5/10); p95 rank = 0.95*9 = 8.55
    # -> sorted[8] + 0.55 * (sorted[9] - sorted[8]) = 1.0
    t = np.arange(10.0)
    e = (np.arange(10) % 2).astype(float)
    rms, p95 = hold_rms(t, np.column_stack([e, np.zeros(10)]),
                        np.zeros((10, 2)), np.ones(10))
    assert rms == pytest.approx(np.sqrt(0.5))
    assert p95 == pytest.approx(1.0)


def test_hold_window_clipped_to_the_final_sk_run():
    # SK only for the last 5 samples; err = 2 there, 0 before.  The 30 s window would
    # include samples 69..94 (err 0) but mode 0 excludes them -> exactly (2, 2).
    t = np.arange(100.0)
    e = np.where(t >= 95.0, 2.0, 0.0)
    mode = np.r_[np.zeros(95), np.ones(5)]
    assert hold_rms(t, np.column_stack([e, np.zeros(100)]),
                    np.zeros((100, 2)), mode) == pytest.approx((2.0, 2.0))


def test_hold_is_nan_without_a_final_hold():
    t = np.arange(10.0)
    z = np.zeros((10, 2))
    for mode in (np.zeros(10), np.r_[np.ones(5), np.zeros(5)]):
        rms, p95 = hold_rms(t, z, z, mode)
        assert np.isnan(rms) and np.isnan(p95)
    rms, p95 = hold_rms(np.zeros(0), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
    assert np.isnan(rms) and np.isnan(p95)


def test_hold_is_nan_on_length_mismatch_and_nan_goal():
    t = np.arange(5.0)
    # t has 5 samples but xy only 4 -> the trace's last t carries no position
    rms, p95 = hold_rms(t, np.zeros((4, 2)), np.zeros((4, 2)), np.ones(5))
    assert np.isnan(rms) and np.isnan(p95)
    g = np.zeros((5, 2))
    g[-1, 0] = np.nan                              # broken goal at the hold point
    rms, _ = hold_rms(t, np.zeros((5, 2)), g, np.ones(5))
    assert np.isnan(rms)


# ------------------------------------------------------------------ summarize wiring
def _hold_traj():
    t = np.arange(40.0)
    xy = np.column_stack([0.5 * np.ones(40), np.zeros(40)])
    return {"t": t, "xy": xy, "psi": np.zeros(40), "action": np.zeros((40, 3)),
            "goal_xy": np.zeros((40, 2)), "mode": np.ones(40)}


def _wp_traj():
    t = np.arange(0.0, 11.0, 1.0)                  # 11 samples, x = 0..10
    return {"t": t, "xy": np.column_stack([t, np.zeros(t.size)]), "psi": np.zeros(t.size),
            "action": _surge([1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "goal_xy": np.array([[5.0, 0.0]] * 6 + [[10.0, 0.0]] * 5),
            "mode": np.zeros(t.size), "path": np.array([[0.0, 0.0], [10.0, 0.0]]),
            "waypoints": np.array([[5.0, 0.0], [10.0, 0.0]])}


def test_summarize_appends_r8_columns_last_in_order():
    m = summarize(_hold_traj())
    keys = list(m)
    assert keys[:len(PRE_R8_COLS)] == PRE_R8_COLS
    assert keys[len(PRE_R8_COLS):] == R8_COLS
    assert m["hold_rms_m"] == pytest.approx(0.5) and m["hold_p95_m"] == pytest.approx(0.5)
    assert m["sat_frac"] == 0.0 and m["reversal_rate"] == 0.0
    assert np.isnan(m["arrival_speed_mean_mps"])   # no planned waypoints in this traj


def test_summarize_preexisting_values_unchanged_hold_episode():
    """Golden values computed by hand from the R1-R6 definitions."""
    m = summarize(_hold_traj())
    assert list(m)[:len(PRE_R8_COLS)] == PRE_R8_COLS
    assert m["n_samples"] == 40 and m["duration_s"] == 39.0
    assert m["radial_error_mean"] == pytest.approx(0.5)      # err = 0.5 for all 40
    assert m["radial_error_p95"] == pytest.approx(0.5)
    assert m["radial_error_max"] == pytest.approx(0.5)
    assert m["cross_track_rmse"] == pytest.approx(0.5)       # one hold point
    # err = 0.5 <= tol 1.0 from sample 0, band held 39 s >= 5 -> settles at t[0]
    assert m["settling_time_s"] == 0.0
    assert m["energy"] == 0.0 and m["action_rate"] == 0.0
    assert np.isnan(m["waypoints_reached"]) and m["lap_completed"] is False
    assert m["time_to_waypoint"] == []


def test_summarize_preexisting_values_unchanged_waypoint_episode():
    m = summarize(_wp_traj())
    assert list(m) == PRE_R8_COLS + R8_COLS
    assert m["n_samples"] == 11 and m["duration_s"] == 10.0
    err = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    assert m["radial_error_mean"] == pytest.approx(float(err.mean()))
    assert m["radial_error_p95"] == pytest.approx(float(np.percentile(err, 95)))
    assert m["radial_error_max"] == 5.0
    assert m["cross_track_rmse"] == pytest.approx(0.0)       # on the explicit path y = 0
    # settling: err <= 1.0 only from sample 9 -> 1 s of band left < hold 5 -> inf
    assert m["settling_time_s"] == float("inf")
    # sum a^2 = 1,1,1,0,...; trapezoid over 1 s steps: 0.5*(1+1) + 0.5*(1+1) + 0.5*(1+0)
    assert m["energy"] == pytest.approx(2.5)
    # total variation 2 + 2 + 1 = 5 over 10 s
    assert m["action_rate"] == pytest.approx(0.5)
    assert m["waypoints_reached"] == 2 and m["lap_completed"] is True
    assert m["time_to_waypoint"] == [4.0, 9.0]     # 2 m radius: x = 9 is 1 m from corner 2
    # R8 columns on the same trace
    assert m["sat_frac"] == pytest.approx(3.0 / 11.0)        # |1| > 0.9 on 3 samples
    assert m["reversal_rate"] == pytest.approx(12.0)         # 2 flips / 10 s * 60
    # corner (5,0): first inside 2 m at x = 4 (sample 4, step from sample 3: 1 m/s)
    # corner (10,0): first inside at x = 9 (sample 9, step from sample 8: 1 m/s)
    assert m["arrival_speed_mean_mps"] == pytest.approx(1.0)
    assert np.isnan(m["hold_rms_m"]) and np.isnan(m["hold_p95_m"])   # waypoint episode


def test_summarize_sk_scenario_gets_a_hold():
    """sk_wind-like: 120 s of station keeping 1 m off the hold point -> (1, 1)."""
    t = np.arange(120.0)
    xy = np.column_stack([np.ones(120), np.zeros(120)])
    m = summarize({"t": t, "xy": xy, "psi": np.zeros(120), "action": np.zeros((120, 3)),
                   "goal_xy": np.zeros((120, 2)), "mode": np.ones(120)})
    assert m["hold_rms_m"] == pytest.approx(1.0) and m["hold_p95_m"] == pytest.approx(1.0)
    assert list(m)[-5:] == R8_COLS
