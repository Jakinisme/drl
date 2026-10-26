"""Plot helpers for evalkit.  The ONLY module in the repo that imports matplotlib.

Backend is forced to Agg *before* pyplot is imported, so nothing tries to open a
window (the evaluation runs headless on Windows / over ssh).  Inputs are the arrays
and dicts produced by ``evalkit.metrics``: a ``traj`` dict (CONTRACT.md §4.9) and/or a
``summarize()`` result.  Every function returns the ``Figure`` and writes it to ``out``
when a path is given.
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")                       # must precede pyplot
import matplotlib.pyplot as plt             # noqa: E402

from evalkit.metrics import radial_error    # noqa: E402

__all__ = ["plot_error", "plot_trajectory", "summary_table"]


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def plot_error(traj: dict, tol: float | None = None, ax=None, out=None):
    """Radial error vs time, with the tolerance band marked."""
    t = _np(traj["t"]).ravel()
    err = radial_error(_np(traj["xy"]).reshape(-1, 2), _np(traj["goal_xy"]).reshape(-1, 2))
    if tol is None:
        tol = float(traj.get("tol_m", 1.0))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    else:
        fig = ax.figure
    ax.plot(t, err, lw=1.2, label="radial error")
    ax.axhline(tol, ls="--", lw=0.8, color="k", label=f"tol = {tol:g} m")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("radial error [m]")
    ax.set_title("Position error")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    return fig


def plot_trajectory(traj: dict, ax=None, out=None):
    """XY track in the local ENU frame: east right, north up, equal aspect."""
    xy = _np(traj["xy"]).reshape(-1, 2)
    goals = _np(traj["goal_xy"]).reshape(-1, 2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    path = traj.get("path")
    if path is not None:
        p = _np(path).reshape(-1, 2)
        ax.plot(p[:, 0], p[:, 1], "k--", lw=0.8, label="path")
    ax.plot(xy[:, 0], xy[:, 1], lw=1.2, label="track")
    ax.plot(goals[:, 0], goals[:, 1], "x", ms=5, label="goal")
    ax.plot(xy[0, 0], xy[0, 1], "o", ms=4, label="start")
    ax.plot(xy[-1, 0], xy[-1, 1], "s", ms=4, label="end")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title("Track (local ENU)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    return fig


def summary_table(summaries: dict[str, dict], out=None, title: str = "Summary"):
    """Compact table: one row per run, one column per scalar metric.

    ``summaries`` = ``{run_name: summarize() result}``.  Column order follows the
    first run's dict; non-scalar values are skipped.
    """
    rows = {k: v for k, v in summaries.items() if isinstance(v, dict)}
    if not rows:
        raise ValueError("summary_table needs at least one summarize() dict")
    cols = [k for k, v in next(iter(rows.values())).items() if np.isscalar(v)]
    cell = [[f"{rows[r][c]:.3g}" if isinstance(rows[r][c], float) else str(rows[r][c])
             for c in cols] for r in rows]

    fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 2.0, 0.5 * len(cell) + 1.4))
    ax.axis("off")
    ax.set_title(title)
    tab = ax.table(cellText=cell, rowLabels=list(rows), colLabels=cols, loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.3)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    return fig
