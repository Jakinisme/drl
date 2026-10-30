"""Apply frozen rule D1 to the phase-2 results.

Reads only `summary.json` from each results_a1_r8_<arm>_<S|C> dir, prints every intermediate
number the rule depends on, and states the outcome. No thresholds are read from anywhere but
AUTO_RULES.md's D1, transcribed here once with the rule text quoted.

Exit codes: 0 = winner found, 3 = PROVISIONAL_A, 4 = NONE / not decidable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["control", "A", "B"]


def pooled(arm: str, cond: str) -> dict:
    """{(controller, scenario): pooled row} for one arm/condition."""
    p = ROOT / f"results_a1_r8_{arm}_{cond}" / "summary.json"
    if not p.is_file():
        raise SystemExit(f"missing {p}")
    s = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for r in s.get("pooled", []) or []:
        out[(str(r["controller"]), str(r["scenario"]))] = r
    return out


def g(row: dict, key: str) -> float:
    v = row.get(key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def show(label: str, val: float) -> str:
    if val != val:
        return f"{label}=nan"
    if val in (float("inf"), float("-inf")):
        return f"{label}=inf"
    return f"{label}={val:.4g}"


def geo_dropped(arm: str, cond: str, scen: str) -> float:
    """geofence-dropped episode count for a scenario, from per_seed rows.

    `geofence_dropped` is not a pooled column; sum it over seeds from per_seed.
    """
    p = ROOT / f"results_a1_r8_{arm}_{cond}" / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    tot = 0.0
    for r in s.get("per_seed", []) or []:
        if str(r.get("scenario")) == scen and str(r.get("controller")) == "ppo":
            v = r.get("geofence_dropped")
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f:
                tot += f
    return tot


def main() -> int:
    print("Rule D1 (frozen, reports/AUTO_RULES.md):")
    print("  primary (C, sk_wind): settled_pct(V)-settled_pct(ctl) >= 10 pts")
    print("                     OR hold_rms_m(V) <= 0.8 * hold_rms_m(ctl)")
    print("     -> if settled_pct(ctl,C) >= 90 the primary moves to S")
    print("     -> if all settled_pct==0 AND hold_rms differ <5% in C: NOT DECIDABLE")
    print("  guards (S): sat_frac(V)<=sat_frac(ctl) in sk_wind AND wp_square;")
    print("              waypoints_reached(V) >= 0.85*ctl; geo_drop(V)<=geo_drop(ctl);")
    print("              settled_pct(S,V) >= settled_pct(S,ctl)-5")
    print()

    data = {(a, c): pooled(a, c) for a in ARMS for c in ("S", "C")}

    # ---- the numbers every decision depends on
    print("=" * 78)
    for cond in ("C", "S"):
        print(f"\n--- condition {cond} ---")
        for a in ARMS:
            d = data[(a, cond)]
            sk = d.get(("ppo", "sk_wind"), {})
            wp = d.get(("ppo", "wp_square"), {})
            print(f"{a:8} sk: " + " ".join([
                show("settled%", g(sk, "settled_pct")),
                show("hold_rms", g(sk, "hold_rms_m")),
                show("sat", g(sk, "sat_frac")),
            ]) + "  wp: " + " ".join([
                show("wps", g(wp, "waypoints_reached")),
                show("sat", g(wp, "sat_frac")),
                show("settled%", g(wp, "settled_pct")),
            ]) + f"  geo_sk={geo_dropped(a, cond, 'sk_wind'):.0f}"
                 f" geo_wp={geo_dropped(a, cond, 'wp_square'):.0f}")

    ctl_C_sk = data[("control", "C")][("ppo", "sk_wind")]
    ctl_C_hold = g(ctl_C_sk, "hold_rms_m")
    ctl_C_settled = g(ctl_C_sk, "settled_pct")

    # ---- which condition carries the primary
    use_S = bool(ctl_C_settled >= 90.0)
    print(f"\ncontrol settled_pct in C = {ctl_C_settled:.1f} -> primary evaluated in "
          f"{'S (ceiling hit)' if use_S else 'C'}")

    pcond = "S" if use_S else "C"
    sc = ("S", "C") if use_S else ("C", "S")

    def primary_beats(v: dict, ctl: dict, halve: bool = False) -> tuple[bool, str]:
        d_settled = g(v, "settled_pct") - g(ctl, "settled_pct")
        v_hold, c_hold = g(v, "hold_rms_m"), g(ctl, "hold_rms_m")
        thr_pts = 10.0 / (2 if halve else 1)
        ratio = 0.8 ** (2 if halve else 1)
        ok_pts = d_settled >= thr_pts
        ok_hold = (v_hold == v_hold) and (c_hold == c_hold) and c_hold > 0 and v_hold <= ratio * c_hold
        why = (f"d_settled={d_settled:+.2f} pts (need >= {thr_pts:.1f}) "
               f"{'OK' if ok_pts else 'no'}; hold {v_hold:.3f} vs {ratio:.3f}*{c_hold:.3f}="
               f"{ratio*c_hold:.3f} {'OK' if ok_hold else 'no'}")
        return (ok_pts or ok_hold), why

    def guards_ok(v_arms: dict, ctl_arms: dict) -> tuple[bool, list[str]]:
        msgs, ok = [], True
        for scen in ("sk_wind", "wp_square"):
            sv = g(v_arms.get(("ppo", scen), {}), "sat_frac")
            sc_ = g(ctl_arms.get(("ppo", scen), {}), "sat_frac")
            good = (sv == sv) and (sc_ == sc_) and sv <= sc_ + 1e-12
            ok &= good
            msgs.append(f"sat[{scen}] {sv:.4f} <= {sc_:.4f} {'OK' if good else 'FAIL'}")
        wv = g(v_arms.get(("ppo", "wp_square"), {}), "waypoints_reached")
        wc = g(ctl_arms.get(("ppo", "wp_square"), {}), "waypoints_reached")
        good = (wv == wv) and (wc == wc) and wv >= 0.85 * wc
        ok &= good
        msgs.append(f"wps {wv:.3f} >= 0.85*{wc:.3f}={0.85*wc:.3f} {'OK' if good else 'FAIL'}")
        sv = g(v_arms.get(("ppo", "sk_wind"), {}), "settled_pct")
        scc = g(ctl_arms.get(("ppo", "sk_wind"), {}), "settled_pct")
        good = (sv == sv) and (scc == scc) and sv >= scc - 5.0
        ok &= good
        msgs.append(f"settled_pct(S) {sv:.1f} >= {scc:.1f}-5 {'OK' if good else 'FAIL'}")
        return ok, msgs

    # ---- evaluate each variant in S (guards are defined in S)
    results = {}
    for a in ("A", "B"):
        ok_p, why = primary_beats(data[(a, pcond)][("ppo", "sk_wind")],
                                  data[("control", pcond)][("ppo", "sk_wind")])
        ok_g, gmsgs = guards_ok(data[(a, "S")], data[("control", "S")])
        # geofence guard, summed over seeds
        gd_v = geo_dropped(a, "S", "sk_wind") + geo_dropped(a, "S", "wp_square")
        gd_c = geo_dropped("control", "S", "sk_wind") + geo_dropped("control", "S", "wp_square")
        good_geo = gd_v <= gd_c
        ok_g &= good_geo
        gmsgs.append(f"geo_drop {gd_v:.0f} <= {gd_c:.0f} {'OK' if good_geo else 'FAIL'}")
        results[a] = (ok_p, why, ok_g, gmsgs)
        print(f"\nvariant {a}: primary {'PASS' if ok_p else 'FAIL'} ({why})")
        for m in gmsgs:
            print(f"   {m}")
        print(f"   -> {'QUALIFIES' if (ok_p and ok_g) else 'does not qualify'}")

    # ---- NOT DECIDABLE check (sk_wind, condition C)
    settles = [g(data[(a, "C")][("ppo", "sk_wind")], "settled_pct") for a in ARMS]
    holds = [g(data[(a, "C")][("ppo", "sk_wind")], "hold_rms_m") for a in ARMS]
    all_zero = all((s == s) and abs(s) < 1e-9 for s in settles)
    holds_f = [h for h in holds if h == h]
    spread = ((max(holds_f) - min(holds_f)) / max(min(holds_f), 1e-12)) if len(holds_f) > 1 else 0.0
    if all_zero and spread < 0.05:
        print(f"\nALL arms settled_pct==0 and hold_rms spread {spread:.2%} < 5% in C")
        print("-> primary NOT DECIDABLE, counts as 'does not beat control'")

    winners = [a for a, (op, _w, og, _gm) in results.items() if op and og]
    print("\n" + "=" * 78)
    if winners:
        print(f"WINNER CANDIDATE(S): {winners} -> confirmation on seeds 100-103 REQUIRED")
        print("OUTCOME=WINNER_PENDING_CONFIRMATION")
        return 0

    # ---- PROVISIONAL_A check
    A_S = data[("A", "S")]
    ctl_S = data[("control", "S")]
    a_sat_sk = g(A_S[("ppo", "sk_wind")], "sat_frac")
    c_sat_sk = g(ctl_S[("ppo", "sk_wind")], "sat_frac")
    a_sat_wp = g(A_S[("ppo", "wp_square")], "sat_frac")
    c_sat_wp = g(ctl_S[("ppo", "wp_square")], "sat_frac")
    a_wp = g(A_S[("ppo", "wp_square")], "waypoints_reached")
    c_wp = g(ctl_S[("ppo", "wp_square")], "waypoints_reached")
    prov = (a_sat_sk <= c_sat_sk and a_sat_wp <= c_sat_wp and a_wp >= 0.85 * c_wp)
    print(f"PROVISIONAL_A test: sat sk {a_sat_sk:.4f}<={c_sat_sk:.4f} "
          f"({a_sat_sk <= c_sat_sk}), sat wp {a_sat_wp:.4f}<={c_sat_wp:.4f} "
          f"({a_sat_wp <= c_sat_wp}), wps {a_wp:.3f} >= 0.85*{c_wp:.3f}={0.85*c_wp:.3f} "
          f"({a_wp >= 0.85*c_wp})")
    if prov:
        print("OUTCOME=PROVISIONAL_A (A is a fixed reward for R9 ONLY, not a validated "
              "improvement; every R9 table must be labelled 'reward provisional')")
        return 3
    print("OUTCOME=NONE -> skip phases 4-7, write final report")
    return 4


if __name__ == "__main__":
    sys.exit(main())
