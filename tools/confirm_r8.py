"""Apply frozen rule D1's CONFIRMATION step to the fresh-seed runs.

Rule (reports/AUTO_RULES.md, D1): "Confirmation: on fresh seeds 100-103 the same variant must
again satisfy primary (same direction, same thresholds halved) and all guards; otherwise it
is not a winner."

Same thresholds halved means:
  settled_pct(V) - settled_pct(control) >= 5 percentage points   (was 10)
  hold_rms_m(V) <= 0.8**2 * hold_rms_m(control) = 0.64x           (was 0.8x)
and the guards keep their original form (sat_frac <= control, waypoints >= 0.85*control,
geofence not worse, settled_pct(S) >= control-5).

Reads only summary.json from results_a1_conf_{A,control}_{S,C}.
Exit 0 = confirmed, 4 = not confirmed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANT = "A"


def pooled(arm: str, cond: str) -> dict:
    p = ROOT / f"results_a1_conf_{arm}_{cond}" / "summary.json"
    if not p.is_file():
        raise SystemExit(f"missing {p}")
    s = json.loads(p.read_text(encoding="utf-8"))
    return {(str(r["controller"]), str(r["scenario"])): r for r in s.get("pooled", []) or []}


def seeds_of(arm: str, cond: str) -> list[int]:
    p = ROOT / f"results_a1_conf_{arm}_{cond}" / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    return sorted({int(r["seed"]) for r in s.get("per_seed", []) or []})


def g(row: dict, key: str) -> float:
    v = row.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def geo(arm: str, cond: str, scen: str) -> float:
    p = ROOT / f"results_a1_conf_{arm}_{cond}" / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    tot = 0.0
    for r in s.get("per_seed", []) or []:
        if str(r.get("scenario")) == scen and str(r.get("controller")) == "ppo":
            try:
                f = float(r.get("geofence_dropped"))
            except (TypeError, ValueError):
                continue
            if f == f:
                tot += f
    return tot


def main() -> int:
    d = {(a, c): pooled(a, c) for a in (VARIANT, "control") for c in ("S", "C")}

    print("Confirmation on fresh seeds — D1 halves both primary thresholds")
    for a in (VARIANT, "control"):
        print(f"  {a:8} seeds: {seeds_of(a, 'C')}")

    ctl_C = d[("control", "C")][("ppo", "sk_wind")]
    use_S = g(ctl_C, "settled_pct") >= 90.0
    pcond = "S" if use_S else "C"
    print(f"\ncontrol settled_pct in C = {g(ctl_C, 'settled_pct'):.1f} -> primary in {pcond}")

    v = d[(VARIANT, pcond)][("ppo", "sk_wind")]
    c = d[("control", pcond)][("ppo", "sk_wind")]

    d_settled = g(v, "settled_pct") - g(c, "settled_pct")
    v_hold, c_hold = g(v, "hold_rms_m"), g(c, "hold_rms_m")
    ok_pts = d_settled >= 5.0
    ok_hold = (v_hold == v_hold) and (c_hold == c_hold) and c_hold > 0 and v_hold <= 0.64 * c_hold
    print(f"\nPRIMARY ({pcond}, sk_wind):")
    print(f"  settled_pct {g(v,'settled_pct'):.2f} - {g(c,'settled_pct'):.2f} = {d_settled:+.2f} "
          f"pts, need >= 5.0  -> {'PASS' if ok_pts else 'FAIL'}")
    print(f"  hold_rms {v_hold:.4f} <= 0.64*{c_hold:.4f}={0.64*c_hold:.4f}"
          f"  -> {'PASS' if ok_hold else 'FAIL'}")
    primary = ok_pts or ok_hold

    print("\nGUARDS (S, unchanged form):")
    ok_all = True
    for scen in ("sk_wind", "wp_square"):
        sv = g(d[(VARIANT, "S")].get(("ppo", scen), {}), "sat_frac")
        sc = g(d[("control", "S")].get(("ppo", scen), {}), "sat_frac")
        good = (sv == sv) and (sc == sc) and sv <= sc + 1e-12
        ok_all &= good
        print(f"  sat_frac[{scen}] {sv:.4f} <= {sc:.4f} -> {'PASS' if good else 'FAIL'}")
    wv = g(d[(VARIANT, "S")][("ppo", "wp_square")], "waypoints_reached")
    wc = g(d[("control", "S")][("ppo", "wp_square")], "waypoints_reached")
    good = (wv == wv) and (wc == wc) and wv >= 0.85 * wc
    ok_all &= good
    print(f"  waypoints {wv:.4f} >= 0.85*{wc:.4f}={0.85*wc:.4f} -> {'PASS' if good else 'FAIL'}")
    sv = g(d[(VARIANT, "S")][("ppo", "sk_wind")], "settled_pct")
    sc = g(d[("control", "S")][("ppo", "sk_wind")], "settled_pct")
    good = (sv == sv) and (sc == sc) and sv >= sc - 5.0
    ok_all &= good
    print(f"  settled_pct(S) {sv:.2f} >= {sc:.2f}-5={sc-5:.2f} -> {'PASS' if good else 'FAIL'}")
    gv = geo(VARIANT, "S", "sk_wind") + geo(VARIANT, "S", "wp_square")
    gc = geo("control", "S", "sk_wind") + geo("control", "S", "wp_square")
    good = gv <= gc
    ok_all &= good
    print(f"  geofence dropped {gv:.0f} <= {gc:.0f} -> {'PASS' if good else 'FAIL'}")

    print("\n" + "=" * 70)
    if primary and ok_all:
        print(f"OUTCOME=WINNER({VARIANT}) — primary and all guards hold on fresh seeds")
        return 0
    print(f"OUTCOME=NOT CONFIRMED (primary {'ok' if primary else 'failed'}, "
          f"guards {'ok' if ok_all else 'failed'}) -> D1: not a winner")
    print("Per D1 the single permitted extra round (C1/C2) is the next step, not a re-run "
          "of the same seeds.")
    return 4


if __name__ == "__main__":
    sys.exit(main())
