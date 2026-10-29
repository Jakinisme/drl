"""Evaluation metrics for the DRL ASV controller (CONTRACT.md §4.9).

Every function takes numpy arrays or torch tensors (duck-typed on ``detach``), so the
training side can pass tensors straight in.  No torch import (not needed) and no
matplotlib import (evalkit/plots.py owns that) -- checked in tests/test_metrics.py.

Units: SI (m, s) for geometry/time.  ``energy`` and ``action_rate`` are computed on the
normalized body command ``[surge, sway, yaw]`` in [-1, 1], so they are dimensionless
(1/s and 1/s respectively).  Anything reported here is a simulator number: the physical
model is placeholder-parameterised (see SYSID.md), so no metric in this module says
anything about the real vessel.

Metric definitions (also in reports/B3.md):

* radial error   ``e_t = ||xy_t - goal_t||_2``
* mean / p95     arithmetic mean and linear-interpolated 95th percentile of ``e``
* cross-track    ``sqrt(mean_t d_t^2)``, ``d_t`` = distance from ``xy_t`` to the closest
                 point of the polyline ``path`` (clamped to segment ends)
* settling time  first time at which ``e`` stays inside ``tol`` for the rest of the trace,
                 required to hold for at least ``hold_s``; seconds measured from ``t[0]``;
                 ``inf`` when it never settles
* energy         ``∫ Σ_i a_i(t)^2 dt`` (trapezoid), dimensionless·s
* action rate    ``Σ_t ||a_t - a_{t-1}||_2 / (t[-1] - t[0])``, 1/s
* waypoints      ``waypoint_progress`` walks the trace in time order and credits planned
                 waypoint ``c`` (1-based, in order) the first time ``||xy - c|| < radius``
                 while ``c`` is the ACTIVE goal, i.e. ``goal_xy`` is near ``c``.  Radial
                 error cannot see the R4 failure mode (a policy that parks 3.4 m short of
                 a corner keeps a small mean error), so arrival gets its own counters.

R8 additions (reports/R8_V3_metrics.md; appended to ``summarize`` so no earlier column moves):

* sat_frac       fraction of policy steps with ``max_axis |a| > 0.9``
* reversal_rate  surge sign changes per minute, both sides ``|surge| > 0.5``
* arrival_speed  mean speed at the FIRST entry (``< 2.0 m``) into each reached waypoint,
                 one-sided difference ``||xy_i - xy_{i-1}|| / dt``
* hold_rms_m     RMS radial error to the final goal over the last 30 s of a trace that
  hold_p95_m     ends in station keeping (``mode > 0.5``); its p95.  Shorter hold -> the
                 whole hold.  No final hold -> nan.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "radial_error",
    "mean_p95",
    "cross_track_rmse",
    "settling_time",
    "energy",
    "action_rate",
    "waypoint_progress",
    "sat_frac",
    "reversal_rate",
    "arrival_speed_mean_mps",
    "hold_rms",
    "summarize",
]

# R8 thresholds (reports/R8_V3_metrics.md).
SAT_A = 0.9             # |a| above this on any axis = one saturated policy step
REV_A = 0.5             # surge sign changes counted only where both sides exceed this
HOLD_WINDOW_S = 30.0    # final-hold window
ARRIVAL_RADIUS_M = 2.0  # same default as `waypoint_progress`

# numpy 2 renamed trapz -> trapezoid; keep working on both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _np(x) -> np.ndarray:
    """numpy or torch -> float64 ndarray. No torch import."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def radial_error(xy, goal_xy) -> np.ndarray:
    """Per-sample Euclidean distance to the goal. ``goal_xy`` may be ``[2]`` (broadcast)."""
    d = _np(xy) - _np(goal_xy)
    return np.sqrt(np.sum(d * d, axis=-1))


def mean_p95(err) -> tuple[float, float]:
    """Mean and 95th percentile of ``err``. NaNs ignored (all-NaN/empty -> ``(nan, nan)``)."""
    e = _np(err).ravel()
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(e)), float(np.percentile(e, 95))


def _distance_to_polyline(p: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Distance from each row of ``p`` [T,2] to the nearest point of ``path`` [P,2]."""
    if path.shape[0] == 1:
        return np.sqrt(np.sum((p - path[0]) ** 2, axis=-1))
    a, b = path[:-1], path[1:]
    ab = b - a                                                  # [S,2]
    denom = np.maximum(np.sum(ab * ab, axis=-1), 1e-12)
    ap = p[:, None, :] - a[None, :, :]                           # [T,S,2]
    s = np.clip(np.sum(ap * ab[None], axis=-1) / denom, 0.0, 1.0)
    proj = a[None] + s[..., None] * ab[None]                     # [T,S,2]
    return np.sqrt(np.sum((p[:, None, :] - proj) ** 2, axis=-1)).min(axis=1)


def cross_track_rmse(xy, path) -> float:
    """RMSE of the distance to the polyline ``path`` (waypoint list, [P,2]).

    Point-to-*segment*, not point-to-vertex, so a straight path gives the true
    perpendicular offset.  ``P == 1`` degenerates to distance-to-point.
    """
    p, path = _np(xy), _np(path)
    if p.size == 0 or path.size == 0:
        return float("nan")
    d = _distance_to_polyline(p.reshape(-1, 2), path.reshape(-1, 2))
    return float(np.sqrt(np.mean(d * d)))


def settling_time(t, err, tol: float, hold_s: float = 5.0) -> float:
    """Seconds from ``t[0]`` until ``err`` stays <= ``tol`` for the rest of the trace.

    ``hold_s`` guards short traces: the in-band tail must last at least ``hold_s``,
    otherwise the trace is reported as not settled (``inf``).  NaN counts as out of
    band, so a broken estimator can never look settled.
    """
    t, e = _np(t).ravel(), _np(err).ravel()
    if t.size == 0 or t.size != e.size:
        return float("inf")
    bad = ~(e <= tol)                       # NaN -> True -> not settled
    idx = 0 if not bad.any() else int(np.flatnonzero(bad)[-1]) + 1
    if idx >= t.size or (t[-1] - t[idx]) < hold_s:
        return float("inf")
    return float(t[idx] - t[0])


def energy(t, action) -> float:
    """``∫ Σ_i a_i^2 dt`` of the normalized command (trapezoid; 0 for <2 samples)."""
    t, a = _np(t).ravel(), _np(action)
    if t.size < 2 or a.size == 0:
        return 0.0
    per_sample = np.sum(a.reshape(t.size, -1) ** 2, axis=-1)
    return float(_trapz(per_sample, t))


def action_rate(t, action) -> float:
    """Total variation of the command per second, ``1/s``.  Constant action -> 0."""
    t, a = _np(t).ravel(), _np(action)
    if t.size < 2 or a.size == 0:
        return 0.0
    a = a.reshape(t.size, -1)
    tv = float(np.sum(np.sqrt(np.sum(np.diff(a, axis=0) ** 2, axis=-1))))
    span = float(t[-1] - t[0])
    return tv / span if span > 0 else 0.0


def waypoint_progress(t, xy, goal_xy, corners, radius: float = 2.0,
                      reach_tol: float = 5.0) -> dict:
    """Ordered waypoint bookkeeping along one trajectory (CONTRACT §4.9).

    ``corners`` is the PLANNED waypoint list ``[P, 2]`` in visit order; ``radius`` the
    arrival radius.  A waypoint counts when it is reached in order AND ``goal_xy`` at that
    sample is within ``reach_tol`` of it (so "active goal" = what the env was tracking, not
    a guess).  Returns ``waypoints_reached`` (int), ``lap_completed`` (bool, all P in
    order), ``time_to_waypoint`` (list of seconds from ``t[0]`` per waypoint, ``nan`` when
    never reached).
    """
    t = _np(t).ravel()
    p, g, c = _np(xy).reshape(-1, 2), _np(goal_xy).reshape(-1, 2), _np(corners).reshape(-1, 2)
    n = min(t.size, p.shape[0], g.shape[0])
    times: list[float] = [float("nan")] * c.shape[0]
    k = 0                                        # next waypoint expected, in order
    for i in range(n):
        if k >= c.shape[0]:
            break
        if (np.hypot(*(p[i] - c[k])) < radius
                and np.hypot(*(g[i] - c[k])) <= reach_tol):
            times[k] = float(t[i] - t[0]) if t.size else float("nan")
            k += 1
    return {"waypoints_reached": int(k),
            "lap_completed": bool(k == c.shape[0] and c.shape[0] > 0),
            "time_to_waypoint": times}


def _unique_rows(a: np.ndarray) -> np.ndarray:
    """Unique rows, first-occurrence order preserved."""
    _, idx = np.unique(a, axis=0, return_index=True)
    return a[np.sort(idx)]


# ------------------------------------------------------------------- R8: settling quality
# Five columns appended by `summarize` (reports/R8_V3_metrics.md).  All are pure functions
# of one episode's arrays; missing data gives nan (or 0.0 where the count is well defined),
# never an exception.
def sat_frac(action, tol: float = SAT_A) -> float:
    """Fraction of policy steps with ``max_axis |a| > tol`` (strict).  No samples -> 0.0."""
    a = _np(action)
    if a.size == 0:
        return 0.0
    a = a.reshape(a.shape[0], -1)
    return float(np.mean((np.abs(a) > tol).any(axis=1)))


def reversal_rate(t, action, tol: float = REV_A) -> float:
    """Surge sign changes per minute, counted only between pairs that BOTH exceed ``tol``.

    Consecutive samples whose surge is ``<= tol`` are allowed to differ, so a sign flip
    separated by a stretch of |surge| ~ 0 is NOT a reversal.  Rate denominator is the trace
    span ``t[-1] - t[0]``; span <= 0 -> 0.0.
    """
    t = _np(t).ravel()
    a = _np(action)
    if t.size < 2 or a.size == 0:
        return 0.0
    s = a.reshape(t.size, -1)[:, 0]                      # surge = axis 0 (CONTRACT §4.5)
    ok = np.abs(s) > tol
    n = int(np.count_nonzero(ok[:-1] & ok[1:] & (np.sign(s[:-1]) != np.sign(s[1:]))))
    span = float(t[-1] - t[0])
    return 60.0 * n / span if span > 0 else 0.0


def _entry_speeds(t: np.ndarray, xy: np.ndarray, corners: np.ndarray,
                  radius: float = ARRIVAL_RADIUS_M) -> list[float]:
    """Speed (m/s, one-sided first difference) at the FIRST sample with
    ``||xy - c|| < radius`` for every corner, in planned order; nan when never entered.

    Standalone per corner: unlike `waypoint_progress` there is no in-order / active-goal
    bookkeeping, so an ``goal_xy`` trace is not needed.  Speed at sample 0 (no previous
    sample) is nan, which makes the mean nan rather than a wrong number.  A corner whose
    entry sample is the last sample of the trace is nan for the same reason.
    """
    p = xy.reshape(-1, 2)
    n = t.size                                   # entry index must have a previous sample
    d = np.sqrt(np.sum((p[:n, None, :] - corners[None, :, :]) ** 2, axis=-1))  # [n, P]
    out: list[float] = []
    for k in range(corners.shape[0]):
        hit = np.flatnonzero(d[:, k] < radius)
        i = int(hit[0]) if hit.size else -1
        if i < 1 or i >= n - 1:
            out.append(float("nan"))
            continue
        dt_ = float(t[i] - t[i - 1])
        v = float(np.hypot(*(p[i] - p[i - 1])) / dt_) if dt_ > 0 else float("nan")
        out.append(v)
    return out


def arrival_speed_mean_mps(t, xy, corners, radius: float = ARRIVAL_RADIUS_M) -> float:
    """Mean speed at the FIRST entry into each planned waypoint.

    Convention: entry = first sample with ``||xy_i - c_k|| < radius`` (same 2.0 m default and
    the same strict ``<`` as `waypoint_progress`, but evaluated per corner independently, in
    planned order, with no in-order bookkeeping); speed is the one-sided difference
    ``||xy_i - xy_{i-1}|| / (t_i - t_{i-1})`` -- the speed the vessel HAD on entering.
    Average over REACHED waypoints only (per-waypoint nans dropped); nan only when no
    waypoint was reached with a defined speed.  ``xy`` shorter than ``t`` -> nan.
    """
    t = _np(t).ravel()
    xy = _np(xy).reshape(-1, 2)
    c = _np(corners).reshape(-1, 2)
    if t.size < 2 or c.size == 0 or xy.shape[0] < t.size:
        return float("nan")
    v = np.array(_entry_speeds(t, xy, c, radius), dtype=np.float64)
    # Drop individual undefined entries (a corner entered on the last sample has no
    # previous sample to difference against). Requiring ALL of them finite would let one
    # such corner turn the whole metric into nan, which is not "per-waypoint nans dropped"
    # and is what made this column read NOT AVAILABLE on the R7 baseline.
    good = v[np.isfinite(v)]
    if good.size == 0:
        return float("nan")
    return float(np.mean(good))


def final_hold_window(t, mode) -> tuple[int, int]:
    """Sample index range ``[i0, n)`` of the FINAL station-keeping hold, or ``(-1, -1)``.

    station keeping is ``mode > 0.5`` (as in `envs/asv_env.py`).  The hold must end at the
    last sample of the trace; otherwise (waypoint episode, or the trace ends mid-leg) there
    is no final hold and the R8 hold columns are nan.
    """
    t, m = _np(t).ravel(), _np(mode).ravel()
    n = min(t.size, m.size)
    if n == 0 or not (m[n - 1] > 0.5):
        return -1, -1
    sk = np.flatnonzero(~(m[:n] > 0.5))                  # last non-SK sample
    return (int(sk[-1]) + 1 if sk.size else 0), n


def _hold_errors(t: np.ndarray, xy: np.ndarray, goal: np.ndarray,
                 i0: int, i1: int, window_s: float = HOLD_WINDOW_S) -> np.ndarray:
    """Radial errors to ``goal[i1 - 1]`` over samples ``[j0, i1)`` with
    ``t >= t[i1 - 1] - window_s``; empty if there is no hold."""
    if i0 < 0 or i1 - i0 < 1:
        return np.zeros(0)
    j0 = int(np.searchsorted(t[i0:i1], t[i1 - 1] - window_s, side="left")) + i0
    return radial_error(xy[j0:i1], goal[i1 - 1])


def hold_rms(t, xy, goal_xy, mode, window_s: float = HOLD_WINDOW_S) -> tuple[float, float]:
    """``(rms, p95)`` of the radial error to the final goal over the last ``window_s``.

    Window: samples with ``t >= t[-1] - window_s`` that belong to the final station-keeping
    run (see `final_hold_window`); goal = ``goal_xy[-1]`` (the final commanded hold point).
    An episode shorter than ``window_s`` uses its whole hold -- the window is clipped, not
    padded, so a short hold is reported on what exists.  No final hold, empty window, or a
    non-finite error -> ``(nan, nan)``.
    """
    t = _np(t).ravel()
    xy = _np(xy).reshape(-1, 2)
    g = _np(goal_xy).reshape(-1, 2)
    m = _np(mode).ravel()
    # hold requires at least one complete (position, goal) sample at the trace's last t
    if t.size == 0 or xy.shape[0] < t.size or g.shape[0] < t.size:
        return float("nan"), float("nan")
    i0, i1 = final_hold_window(t, m)
    i1 = min(i1, t.size)
    e = _hold_errors(t, xy, g, i0, i1, window_s)
    if e.size == 0 or not np.all(np.isfinite(e)):
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(e * e))), float(np.percentile(e, 95))



def summarize(traj: dict, tol: float | None = None, hold_s: float = 5.0,
              waypoints=None, arrival_radius: float = 2.0) -> dict:
    """All metrics for one trajectory.

    ``traj`` keys (CONTRACT.md §4.9): ``t``, ``xy``, ``psi``, ``action``, ``goal_xy``,
    ``mode``.  Optional ``path`` ([P,2]); when absent the ordered unique ``goal_xy``
    points are used as the path, which is the waypoint sequence for waypoint episodes
    and a single hold point for station keeping.  ``tol`` defaults to ``traj["tol_m"]``,
    then 1.0 m.  ``psi``/``mode`` are not used by these metrics (no heading metric is
    specified in §4.9) -- the whole trace is scored regardless of mode.

    ``waypoints`` (planned list [P,2], or ``traj["waypoints"]``) adds
    ``waypoints_reached`` / ``lap_completed`` / ``time_to_waypoint`` via
    `waypoint_progress`; when absent, ``waypoints_reached`` is ``nan`` and
    ``lap_completed`` is ``False`` (no plan to check against).

    R8 columns are APPENDED after the R1-R6 keys, in this order and nothing else changes:
    ``sat_frac``, ``reversal_rate``, ``arrival_speed_mean_mps``, ``hold_rms_m``,
    ``hold_p95_m``.  ``arrival_speed_mean_mps`` needs the planned ``waypoints`` (same
    source as above; nan without them).  The hold pair needs ``traj["mode"]`` and a trace
    that ENDS in station keeping; without ``mode`` or without a final hold they are nan.
    """
    t = _np(traj["t"]).ravel()
    xy = _np(traj["xy"]).reshape(-1, 2)
    goal = _np(traj["goal_xy"]).reshape(-1, 2)
    act = _np(traj["action"])
    err = radial_error(xy, goal)

    path = traj.get("path")
    path = _np(path).reshape(-1, 2) if path is not None else _unique_rows(goal)
    if tol is None:
        tol = float(traj.get("tol_m", 1.0))

    m, p95 = mean_p95(err)
    wps = waypoints if waypoints is not None else traj.get("waypoints")
    wp = (waypoint_progress(t, xy, goal, wps, arrival_radius) if wps is not None and
          np.asarray(wps).size else
          {"waypoints_reached": float("nan"), "lap_completed": False, "time_to_waypoint": []})
    arrival_v = (arrival_speed_mean_mps(t, xy, wps, arrival_radius)
                 if wps is not None and np.asarray(wps).size else float("nan"))
    mode = traj.get("mode")
    h_rms, h_p95 = (hold_rms(t, xy, goal, mode) if mode is not None else
                    (float("nan"), float("nan")))
    return {
        "n_samples": int(t.size),
        "duration_s": float(t[-1] - t[0]) if t.size else 0.0,
        "radial_error_mean": m,
        "radial_error_p95": p95,
        "radial_error_max": float(np.nanmax(err)) if err.size else float("nan"),
        "cross_track_rmse": cross_track_rmse(xy, path),
        "settling_time_s": settling_time(t, err, tol, hold_s),
        "energy": energy(t, act),
        "action_rate": action_rate(t, act),
        **wp,
        # ---- R8 (appended: existing keys above keep their values and position)
        "sat_frac": sat_frac(act),
        "reversal_rate": reversal_rate(t, act),
        "arrival_speed_mean_mps": arrival_v,
        "hold_rms_m": h_rms,
        "hold_p95_m": h_p95,
    }
