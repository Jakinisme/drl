"""R8/V4 — pre-export BEHAVIOUR gate for a PPO checkpoint.

`tests/test_onnx_parity.py` / `tests/test_runtime.py` only prove shapes, parity and |a|<=1
against a dummy ONNX model; none of them ever loads a real checkpoint, so a policy that
passes straight through its waypoints could be exported and called "R7 works".  This tool
closes that hole: it runs the same two scenarios `evaluate.py` runs, on the same strict
settings, and fails loudly when the policy does not behave.

Usage (from drl/ with the venv python; `python` on PATH is 3.14 without torch):

    .venv/Scripts/python.exe tools/gate_policy.py --ckpt runs/r7_300m/ckpt/ckpt_last.pt
    .venv/Scripts/python.exe tools/gate_policy.py --ckpt runs/r7_300m/ckpt/ckpt_last.pt \
        --seeds 2 --envs 4 --json /tmp/gate.json          # fast smoke of the gate itself

Exit codes: 0 = every row PASS, 1 = at least one FAIL or NOT AVAILABLE, 2 = usage / missing
checkpoint.  NOT a pytest test: it needs a real checkpoint and takes minutes.

Scenarios and settings are pinned in `configs/gates.yaml` (`settings:`), NOT here, so the
human can see what was gated: sk_wind and wp_square at episode-s 300, dwell-stay-factor 1.0,
seeds 4.  Those flags exist in `evaluate.py`'s argparse (--episode-s, --dwell-stay-factor,
--seeds, --envs, --tol, --hold-s) and are passed positionally to `run_rollout` exactly as
evaluate.py does it; `--episode-s` is honoured for wp_square only, which is the case the gate
sets.

Metrics (evalkit/metrics.py §R8, reports/R8_V3_metrics.md) are read DEFENSIVELY out of the
summary dict: `summarize` may not add them yet, and an `aggregate()` over per-episode dicts
drops unknown keys the same way — a key that is absent, NaN, or inf is reported as NOT
AVAILABLE and fails the gate instead of silently passing.  `settled_pct` is not a
`summarize` key at all: it is derived the way `evaluate.score` derives it
(`100 if settling_time_s < inf else 0`), so settling_time_s NaN/inf -> 0 %.

Numbers from this tool are simulator numbers on placeholder physics (SYSID.md); nothing here
says anything about the real vessel.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate as ev                                            # noqa: E402
from models import build_policy                                  # noqa: E402

GATES = ROOT / "configs" / "gates.yaml"
DEFAULT_CKPT = ROOT / "runs" / "r7_300m" / "ckpt" / "ckpt_last.pt"
NOT_AVAILABLE = "NOT AVAILABLE"


# --------------------------------------------------------------------------- config
def load_gates(path: Path = GATES) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _num(v) -> float:
    """Metric -> float, or NaN when absent / non-numeric / not finite."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


# --------------------------------------------------------------------------- episodes
def gate_episodes(name: str, scen: dict, policy, device, args) -> list[dict]:
    """Per-episode metric dicts for every COMPLETE episode of every seed of `scen`.

    Mirrors evaluate.main's seed/rollout/scoring loop (seed = base + 1000*scenario index +
    k, K=1 episode per env, only clean episodes scored) but returns the rows instead of
    writing CSVs, and does not run the PID baseline or plot anything.
    """
    si = list(ev.SCENARIOS).index(name)
    rows: list[dict] = []
    for k in range(int(args.seeds)):
        seed = int(args.seed) + 1000 * si + k
        rec = ev.run_rollout("ppo", policy, device, scen, seed, 1, args.envs,
                             args.rough_scale, ev.SQUARE_SIDE_M, args.square_hold,
                             args.dwell_stay)
        meta = rec.pop("_meta")
        path = rec.pop("_path", None)
        wps = rec.pop("_waypoints", None)
        arrival = rec.pop("_arrival", 2.0)
        rec.pop("_conditions", None)
        rec.pop("_log_std", None)
        ep = ev.episode_arrays(rec, meta["max_steps"], 1)
        for e in range(ep["xy"].shape[0]):
            for i in range(ep["xy"].shape[1]):
                tr = ev.clean_episode(ep, e, i, meta["policy_dt"])
                if tr is None:
                    continue                       # geofence exit / cut block: not scored
                rows.append(ev.score(
                    tr, args.tol, args.hold_s, args.max_thrust,
                    path=None if path is None else path[i],
                    waypoints=None if wps is None else wps[i], arrival=arrival))
    return rows


def pooled(per_ep: list[dict]) -> dict:
    """`evaluate.aggregate` over the per-episode dicts (+ the % settled derivation)."""
    agg = ev.aggregate(per_ep)
    st = _num(agg.get("settling_time_s"))
    agg["settled_pct"] = 100.0 if agg.get("settling_time_s") is not None and st < float("inf") \
        else 0.0
    agg["geofence_dropped_episodes"] = float("nan")     # never dropped: counted below
    return agg


# --------------------------------------------------------------------------- gate
def check(gate: dict, value: float) -> tuple[bool | None, str]:
    """(pass, text) for one threshold row; pass is None -> NOT AVAILABLE."""
    kind = gate["dir"].lower()
    lim = float(gate["value"])
    if not math.isfinite(value):
        return None, NOT_AVAILABLE
    ok = value >= lim if kind == "min" else value <= lim
    return bool(ok), f"{value:.3g}"


def gate_table(summary: dict, gates: dict, n_envs_x_seeds: int,
               dropped: dict[str, int] | None = None) -> list[dict]:
    """One row per threshold of every scenario in `gates`."""
    dropped = dropped or {}
    out: list[dict] = []
    for name, sc in gates["scenarios"].items():
        agg = summary.get(name)
        for metric, spec in sc.items():
            if not isinstance(spec, dict) or "value" not in spec:
                continue
            value = float("nan") if agg is None else _num(agg.get(metric))
            if metric == "geofence_dropped_episodes":
                value = float(dropped.get(name, 0))
            ok, text = check(spec, value)
            out.append({"scenario": name, "metric": metric, "dir": spec["dir"],
                        "limit": float(spec["value"]), "value": value,
                        "status": "PASS" if ok else ("FAIL" if ok is False else "NOT AVAILABLE"),
                        "shown": text,
                        "note": str(spec.get("note", ""))})
    return out


def render(rows: list[dict], n_ep: dict[str, int], real_episodes: int) -> str:
    w = [max(len(str(r[k])) for r in rows + [{"scenario": "scenario", "metric": "metric",
                                             "dir": "dir", "shown": "value",
                                             "status": "status"}]) for k in
         ("scenario", "metric", "dir", "shown", "status")]
    head = ("{:<%ds}  {:<%ds}  {:<%ds}  {:>%ds}  {:<%ds}" %
            (w[0], w[1], w[2], w[3], w[4])).format("scenario", "metric", "dir", "value", "status")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(("{:<%ds}  {:<%ds}  {:<%ds}  {:>%ds}  {:<%ds}" %
                      (w[0], w[1], w[2], w[3], w[4])).format(
            r["scenario"], r["metric"], r["dir"], r["shown"], r["status"]))
    lines.append("")
    lines.append("episodes scored (complete only): " + ", ".join(
        f"{k} {v}" for k, v in n_ep.items())
        + f" | score('complete') cannot see auto-reset repeats, so >= {real_episodes} "
          f"env-episodes were simulated per scenario")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pre-export behaviour gate (R8/V4)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--gates", default=str(GATES), help="thresholds + pinned settings")
    ap.add_argument("--seeds", type=int, default=None, help="override settings.seeds")
    ap.add_argument("--envs", type=int, default=None, help="override settings.envs")
    ap.add_argument("--scenarios", default=None, help="override settings.scenarios (csv)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write the gate JSON here (optional)")
    args = ap.parse_args(argv)

    gates = load_gates(Path(args.gates))
    st = gates.get("settings", {})
    args.episode_s = st.get("episode_s")
    args.dwell_stay = float(st.get("dwell_stay_factor", 1.0))
    args.square_hold = float(st.get("square_hold_s", 10.0))
    args.tol = float(st.get("tol_m", 1.0))
    args.hold_s = float(st.get("hold_s", 5.0))
    args.rough_scale = 2.0                   # unused: no rough scenario in the gate
    args.envs = int(args.envs if args.envs is not None else st.get("envs", 8))
    args.seeds = int(args.seeds if args.seeds is not None else st.get("seeds", 4))
    args.seed = int(st.get("seed", 0))
    names = ([s.strip() for s in args.scenarios.split(",") if s.strip()]
             if args.scenarios else list(gates["scenarios"]))

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"checkpoint not found: {ckpt_path}")
        return 2
    torch.set_num_threads(1)                 # same as evaluate.py: the env is launch bound
    device = torch.device(args.device)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy = build_policy(ck["cfg"])
    policy.load_state_dict(ck["model"])      # restores obs_mean/obs_std buffers too
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    max_thrust = float(ev._load_yaml(ROOT / "configs" / "vehicle.yaml")["thrusters"]
                       ["max_thrust_fwd_n"])
    args.max_thrust = max_thrust
    print(f"ckpt {ckpt_path} steps={ck.get('steps')} iteration={ck.get('iteration')}")
    print(f"gate {args.gates} | device {device.type} | scenario "
          f"{','.join(names)} | seeds {args.seeds} x envs {args.envs} | "
          f"episode-s {args.episode_s} dwell-stay {args.dwell_stay} "
          f"tol {args.tol} m hold {args.hold_s} s "
          f"(sim, placeholder physics — says nothing about the real vessel)")

    summary: dict[str, dict] = {}
    n_ep: dict[str, int] = {}
    real_ep: dict[str, int] = {}
    for name in names:
        if name not in ev.SCENARIOS:
            print(f"unknown scenario {name}; choose from {sorted(ev.SCENARIOS)}")
            return 2
        scen = dict(ev.SCENARIOS[name])
        if name == "wp_square" and args.episode_s is not None:
            scen["episode_s"] = float(args.episode_s)
        per_ep = gate_episodes(name, scen, policy, device, args)
        summary[name] = pooled(per_ep)
        n_ep[name] = len(per_ep)
        real_ep[name] = args.seeds * args.envs
        st_ = _num(summary[name].get("settling_time_s"))
        print(f"  {name:10s} scored {n_ep[name]} episodes, settled "
              f"{summary[name]['settled_pct']:.1f}%, radial "
              f"{_num(summary[name].get('radial_error_mean')):.2f} m, "
              f"settling {('inf' if not math.isfinite(st_) else f'{st_:.1f} s')}")

    rows = gate_table(summary, gates, args.seeds * args.envs)
    print("\n" + render(rows, n_ep, args.seeds * args.envs))
    bad = [r for r in rows if r["status"] != "PASS"]
    exit_code = 0 if not bad else 1
    print("\nGATE " + ("PASS: every threshold met" if exit_code == 0 else "FAIL: "
                      + ", ".join(f"{r['scenario']}.{r['metric']}={r['status']}"
                                  for r in bad)))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"ckpt": str(ckpt_path), "steps": ck.get("steps"),
             "settings": {"seeds": args.seeds, "envs": args.envs, "episode_s": args.episode_s,
                          "dwell_stay_factor": args.dwell_stay, "tol_m": args.tol,
                          "hold_s": args.hold_s, "scenarios": names},
             "per_scenario": summary, "n_episodes": n_ep, "gate": rows,
             "exit_code": exit_code}, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
