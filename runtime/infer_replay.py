"""Offline replay of a logged sensor CSV through the ONNX policy — CONTRACT §4.8.

No hardware, no torch: numpy + onnxruntime only. Deployment rehearsal — the same
`RLController` the Pi runs, fed from a CSV instead of the serial link.

CSV (header row required), columns per CONTRACT §4.4 / §4.8 units:

    required:  t, lat, lon, heading_deg, speed_mps
    optional:  yaw_rate_deg_s, course_deg, gps_valid, imu_valid, mag_valid, est_age_s
    optional goal:  goal_lat, goal_lon, goal_heading_deg, mode,
                    next_lat, next_lon, next2_lat, next2_lon

`t` [s] is both the replay clock and the estimator timestamp. `est_age_s` (optional)
pretends the sample is that old, which is what trips the §4.6 watchdog — with it
absent the watchdog only sees gaps between rows. Booleans default to true,
`course_deg` defaults to `heading_deg`, and a missing goal holds the current point.
`mode` accepts `waypoint` (default) or `station_keeping` (CONTRACT §3.4/§4.6).

    python -m runtime.infer_replay --csv log.csv --onnx policy.onnx \
        [--meta policy_meta.json] [--limit N]

Writes a summary to stdout: step count, per-channel action min/max, and how often
each safety source (policy / fallback / geofence / zero) and fault (gps / sensor /
watchdog / nan) was active. Exit code 0 on success, 2 on bad input.

TODO: `--fallback` is zeros because `baseline_pid.py` exposes the batched `act()`
interface (CONTRACT §4.6), not the `(est, goal) -> (surge, sway, yaw)` hook
`RLController` expects; adapt it when the Pi glue picks a fallback for real.
"""

import argparse
import csv
import sys

import numpy as np

from runtime.policy import RLController

__all__ = ["main", "load_rows"]

_TRUE = {"1", "true", "t", "yes", "y", "on"}


class _CsvClock:
    """Replay clock: the caller sets it to the row timestamp before each step."""

    def __init__(self):
        self.t = 0.0

    def set(self, t):
        self.t = float(t)

    def __call__(self):
        return self.t


def _num(row, key, default=None):
    v = row.get(key)
    if v is None or str(v).strip() == "":
        if default is None:
            raise ValueError(f"missing or empty column {key!r}")
        return default
    return float(v)


def _opt(row, key):
    v = row.get(key)
    return None if v is None or str(v).strip() == "" else float(v)


def _bool(row, key):
    v = row.get(key)
    return True if v is None or str(v).strip() == "" else str(v).strip().lower() in _TRUE


def _state(row):
    """CSV row -> (`estimator_state`, `goal`) dicts of CONTRACT §4.8."""
    lat, lon = _num(row, "lat"), _num(row, "lon")
    heading = _num(row, "heading_deg", 0.0)
    t = _num(row, "t", 0.0)
    est = {
        "lat": lat, "lon": lon, "heading_deg": heading,
        "yaw_rate_deg_s": _num(row, "yaw_rate_deg_s", 0.0),
        "speed_mps": _num(row, "speed_mps", 0.0),
        "course_deg": _num(row, "course_deg", heading),
        "gps_valid": _bool(row, "gps_valid"),
        "imu_valid": _bool(row, "imu_valid"),
        "mag_valid": _bool(row, "mag_valid"),
        "t": t - _num(row, "est_age_s", 0.0),
    }
    gl, go = _opt(row, "goal_lat"), _opt(row, "goal_lon")
    mode = (str(row.get("mode") or "waypoint")).strip()
    if mode not in ("waypoint", "station_keeping"):
        try:                        # a logger may store the flag itself (0/1)
            mode = float(mode)
        except ValueError:
            pass
    goal = {
        "lat": lat if gl is None else gl,
        "lon": lon if go is None else go,
        "heading_deg": _opt(row, "goal_heading_deg"),
        "mode": mode,
    }
    for k in ("next_lat", "next_lon", "next2_lat", "next2_lon"):
        v = _opt(row, k)
        if v is not None:
            goal[k] = v
    return est, goal


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="infer_replay", description="Replay a sensor CSV through the ONNX policy.")
    ap.add_argument("--csv", required=True, help="logged telemetry, see module docstring")
    ap.add_argument("--onnx", required=True, help="exported policy (CONTRACT §4.8)")
    ap.add_argument("--meta", default=None, help="policy_meta.json next to the .onnx")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (0 = all)")
    args = ap.parse_args(argv)

    try:
        rows = load_rows(args.csv)
        if args.limit > 0:
            rows = rows[:args.limit]
        clock = _CsvClock()
        ctl = RLController(onnx_path=args.onnx, meta_path=args.meta, clock=clock)

        actions, sources, faults = [], {}, {}
        for row in rows:
            est, goal = _state(row)
            clock.set(_num(row, "t", 0.0))
            actions.append(ctl.step(est, goal))
            src, flt = ctl.status["source"], ctl.status["fault"]
            sources[src] = sources.get(src, 0) + 1
            if flt:
                faults[flt] = faults.get(flt, 0) + 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"infer_replay: {exc}", file=sys.stderr)
        return 2

    a = np.asarray(actions, dtype=np.float64).reshape(-1, 3)
    print(f"steps: {a.shape[0]}  ({args.csv} -> {args.onnx})")
    for i, name in enumerate(("surge", "sway", "yaw")):
        print(f"  {name:5s} min {a[:, i].min():+.3f}  max {a[:, i].max():+.3f}")
    print("sources: " + "  ".join(f"{k}={v}" for k, v in sorted(sources.items())))
    print("faults:  " + ("  ".join(f"{k}={v}" for k, v in sorted(faults.items()))
                         or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
