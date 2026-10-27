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
    "summarize",
]

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
    }
