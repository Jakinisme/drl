# A1 — autonomous experiment run (final report)

Branch **`auto/a1`**. All numbers below come from the files cited beside them; the pooled and
per-seed tables are generated mechanically by `tools/collect_results.py` into
**`reports/AUTO_TABLES.md`**.

**Everything here is simulator on placeholder physics (bollard pull unmeasured, H6 unresolved,
SYSID open). Nothing transfers to the real vessel.** No policy below is described as working,
ready or validated. Nothing in this task touched hardware, the Pi, or `pi-controller`.

---

## 1. Outcome in one line

| | result |
|---|---|
| R8 decision | **WINNER(A)** — confirmed on fresh seeds 100-103 |
| R9 | **NOT DECIDED** — implementation only, never compared or trained (phases 6-7 skipped) |
| Phases reached | 0, 1, 2, 3, 4 complete; 5 partly (implementation, unwired to a decision); 6-7 skipped |
| Budget | **OVER — 21.6 h used of 12 h.** Phase 4 finished, then the run was concluded |

---

## 2. What ran

| phase | what | files |
|---|---|---|
| 0 | preflight, branch `auto/a1`, `AUTO_RULES.md` written and hashed before any experiment | `reports/AUTO_RULES.md` |
| 1 | R8b: instrument fixes, 3 new CLI flags, 4 new tests | `reports/R8b.md` |
| 2 | 3 arms x 2 conditions (S, C), seeds 0-3, 8 envs | `results_a1_r8_{control,A,B}_{S,C}/summary.json` |
| 3 | rule D1 + fresh-seed confirmation | `results_a1_conf_A_{S,C}/`, `results_a1_conf2_control_{S,C}/` |
| 4 | 100M long run of the winning reward | `runs/a1_long_A/ckpt/ckpt_last.pt`, `results_a1_long_A_{S,C}/` |
| 5 | R9 structured waypoint generator | `pathgen.py`, `tests/test_pathgen.py`, `reports/R9_impl.md` |
| 6-7 | **skipped** (budget) | — |

Frozen files: `configs/gates.yaml` received only the one permitted edit (see §3);
`evalkit/metrics.py` metric definitions were **not** changed; `AUTO_RULES.md` sha256 verified
unchanged since it was hashed:
`536ac3c1c3a9b4f311963f34a3c58d7320f8c389f083135164de283bfc3199c6`.

---

## 3. Phase 1 — the instrument bug, and why earlier sk_wind numbers are not comparable

`hold_rms_m` was NaN on every `sk_wind` episode. Root cause was **not** the metrics wiring: on
envs that did end in station keeping the metric already returned `2.5175`. `SCENARIOS` declares
`mode=1` (station keeping) for `sk_wind`, but **nothing consumed it**, so the scenario ran at
the training mix `station_keeping_fraction: 0.3` and ~70 % of its episodes were waypoint
episodes with no final hold. Measured final mode per env was `[0 0 1 0 0 0 0 1]`. Fixed with
`scenario_episode_cfg()`; a real rollout now gives final mode all-1 and `hold_rms_m 2.5175`.
Test: `tests/test_r8b_hold_wiring.py` (4 cases, real rollout, no fakes).

**Consequence the human must act on:** every `sk_wind` number measured before commit `cd7d0c5`
is not comparable with numbers after it — including the R7 reference in
`results_r7_reseeded` (`sk_wind settled_pct 15.6`). Those were measured on a scenario that was
~30 % station keeping, not 100 %. They were flagged, not silently adjusted, and **no R7 sk_wind
number is used in any decision below.**

Also fixed: `fmt()` printed NaN as `"inf"`, so a missing metric looked like a failed one
(`arrival_speed_mean_mps` showed `inf` on sk_wind while `summary.json` held `nan`).

New flags: `--skip-pid`, `--sk-episode-s` (station-keeping scenarios only; `--episode-s` stays
`wp_square`), `--level` (**curriculum scale, not index** — `set_curriculum` is index-based and
`scale_by_level[1] == 0.1`, so a raw index would have been silently wrong).

Report-only measurement: **~70 % of a 120 s `sk_wind` episode is available for settling**
(median first entry inside 1 m at 34.4 s). Episode length is not the constraint; the arms fail
to settle because they never stop.

---

## 4. Phase 2-3 — the R8 decision

Condition **C** (calm, `--level 0`) is where D1's primary applies, because control's
`settled_pct` there (28.1 %) is far below the 90 % ceiling.

**Primary, condition C, `sk_wind`, seeds 0-3** (`results_a1_r8_*_C/summary.json`):

| arm | settled% | hold_rms_m | vs control |
|---|---|---|---|
| control | 28.12 | 1.167 | — |
| **A** | **87.50** | **0.687** | +59.38 pts, 0.59x |
| B | 71.88 | 0.692 | +43.75 pts, 0.59x |

Both beat control on both criteria (need +10 pts OR <= 0.80x). Tie-break per D1 ("larger
primary improvement") selects **A**.

**Guards, condition S, seeds 0-3** — all pass for A:

| guard | control | A | verdict |
|---|---|---|---|
| sat_frac sk_wind | 0.0795 | 0.0318 | pass |
| sat_frac wp_square | 0.2132 | 0.0464 | pass |
| wp_square waypoints | 3.5625 | 3.4688 | pass (>= 0.85x = 3.028) |
| settled_pct(S) sk | 18.75 | 21.88 | pass |
| geofence dropped | 0 | 0 | pass |

**Confirmation, fresh seeds 100-103, thresholds halved**
(`results_a1_conf_A_{S,C}/`, `results_a1_conf2_control_{S,C}/`):

| | A | control |
|---|---|---|
| sk_wind C settled% | **81.25** | 25.00 |
| sk_wind C hold_rms | **0.6563** | 1.1664 |
| sk_wind S sat_frac | **0.0281** | 0.0613 |
| wp_square S wps | 3.6875 | 3.6563 |
| wp_square S sat_frac | **0.0389** | 0.2446 |

Primary: **+56.25 pts** (need >= 5) and hold **0.656 <= 0.64 x 1.166 = 0.746**. All guards hold.

> **OUTCOME = WINNER(A).** R8's variant-A reward is the R8 outcome.

Honest asymmetry: in **S** (full disturbance) A still reports `settling_time_s = inf` — it
settles in **C**, the calm condition. The reward's hold bonus is speed-gated, and wind keeps the
vessel moving. That is why D1 evaluates the primary in C.

---

## 5. Phase 4 — the 100M long run, and the trade it reveals

`runs/a1_long_A/ckpt/ckpt_last.pt` — 99,942,400 steps, 1525 iterations, level 5,
29,113 steps/s, ~57 min. Condition S, seeds 0-3 (`results_a1_long_A_S/summary.json`):

| | ctl 10M | A 10M | **A 100M** |
|---|---|---|---|
| sk_wind settled% | 18.75 | 21.88 | **53.12** |
| sk_wind hold_rms_m | 2.341 | 1.940 | **0.958** |
| wp_square waypoints | 3.562 | 3.469 | **3.938** |
| wp_square lap% | 89.1 | 86.7 | **98.4** |
| wp_square settled% | 56.2 | 56.2 | **75.0** |
| **sk_wind sat_frac** | 0.0795 | **0.0318** | **0.1467** |
| **wp_square sat_frac** | 0.2132 | **0.0464** | **0.2667** |

**Longer training buys settling and tracking and gives back the saturation improvement.**
At 100M the policy settles better than any 10M arm (53.1 % vs 18.8-21.9 %) and tracks better
than control (98.4 % lap vs 89.1 %), but its commands are back on the rails: `sat_frac` rises
above control on both scenarios, i.e. it would again ride the actuators hard on the real boat.

Condition **C** (calm) tells the same story with one nuance
(`results_a1_long_A_C/summary.json`):

| condition C | ctl 10M | A 10M | **A 100M** |
|---|---|---|---|
| sk_wind settled% | 28.12 | 87.50 | **87.50** |
| sk_wind hold_rms_m | 1.1673 | 0.6865 | **0.4364** |
| sk_wind sat_frac | 0.0792 | **0.0454** | 0.0633 |
| wp_square settled% | 78.12 | **100.00** | 90.62 |
| wp_square waypoints | 4.000 | 4.000 | 3.812 |
| wp_square sat_frac | 0.2689 | **0.1156** | **0.3541** |

The settling gain **survives** 100M steps in both conditions (`hold_rms` improves further,
87.5 % settled in C). The saturation gain **does not**: in C, `wp_square sat_frac` at 100M
(0.354) is worse than control (0.269), and `waypoints_reached` slips from 4.000 to 3.812 —
the first arm in this run to lose a waypoint against control.

This is a **trade, not a win and not a failure**, and it is the single most decision-relevant
number in this report: the R8 reward's headline benefit (a calm command) does not survive a
longer run of the same reward. Reporting only the settling improvement would be misleading.

Phase 4 applies no decision rule, per the brief. Condition C for this checkpoint and the gate
rows are in `results_a1_long_A_C/summary.json` and `reports/AUTO_TABLES.md`.

---

## 6. R9 — implementation exists, decision does not

`pathgen.py` implements `StructuredPath`, a drop-in for `evaluate.SquarePath`: patterns
`legacy_square` / `straight` / `turn` / `zigzag` / `random`, an orbit-angle penalty excluding
the hold zone, and per-pattern bookkeeping. `tests/test_pathgen.py` → **12 passed**, legacy
mode **bit-identical** to `SquarePath` (max abs diff 0.0 over 200 steps x 8 envs).
`evaluate.py` now carries `wp_line` / `wp_zigzag` / `wp_hairpin` wired to it, `wp_square` left
on `legacy_square` so existing results keep their meaning.

**Phases 6 and 7 were skipped on budget.** So R9 has **no comparison and no outcome**: P0/P1/P2
were never trained, D2 was never applied. Do not read the existence of `pathgen.py` as an R9
result. Two caveats from the implementer that bound the next attempt: `zigzag` cannot exceed 4
waypoints at the default `chain_radius_m=60`/`min_leg_m=15` (needs 90 for 6), and the
`episode_s` values for the three new scenarios (120/160/200) are **guesses**, not measured.

---

## 7. Anomalies the human must review first

1. **GPU evaluation hangs silently.** Three runs printed the checkpoint line and then produced
   neither a row nor an error; one timed out with `EXIT=124`. The same command on `--device cpu`
   completes. This cost significant time, killed a confirmation job overnight, and is **not**
   diagnosed. Any GPU evaluation you launch will likely do this. The affected measurements (arm
   B) were re-run on CPU and are sound; `device` is recorded in every `summary.json`.
2. **`evaluate.py --seeds` is a count, not a list.** `--seeds 100,101,102,103` is rejected. To
   reproduce "seeds 100-103": `--seed 100 --seeds 4` (the code is
   `seed = --seed + 1000*scenario_index + k`, so scenario 2 gets 1100-1103).
3. **`zigzag` cannot reach 6 waypoints** at the default chain radius (see §6).
4. My own duplicate `B_S` launch put two writers in one directory; killed, ~12 min lost, no
   data affected. Logged in `AUTO_LOG.md`.

---

## 8. What remains placeholder — do not build on it

- **Physics.** Bollard pull unmeasured, H6 unresolved, `SYSID.md` open. Every number here is
  simulation.
- **Gate calibration.** `configs/gates.yaml` thresholds are uncalibrated; the gate table is
  informational and no decision used it. The gate FAILs on `sk_wind` for all arms.
- **Thruster sign table (H6)** and **bollard pull** are still unknown; nothing here resolves
  either.
- **R9 scenario lengths** are guesses.
- **The R8 reward's saturation benefit did not survive 100M steps** (§5). Treat "A is a better
  reward" as true at 10M and **unsettled** at 100M.

---

## 9. Reproduce

```bash
cd E:/buoy-project/drl        # git checkout auto/a1
PY=./.venv/Scripts/python.exe

# phase 1 instruments
$PY -m pytest tests/test_r8b_hold_wiring.py -q

# phase 2: one arm, one condition (S; add --level 0 for C)
$PY evaluate.py --ckpt runs/r8_A/ckpt/ckpt_last.pt \
  --scenarios sk_wind,wp_square --dwell-stay-factor 1.0 --episode-s 300 \
  --sk-episode-s 300 --seeds 4 --envs 8 --skip-pid --no-plots --out results_a1_r8_A_S

# phase 2 decision
$PY tools/decide_r8.py

# phase 3 confirmation (fresh seeds 100-103) — note --seed + --seeds, see §7
$PY evaluate.py --ckpt runs/r8_A/ckpt/ckpt_last.pt --scenarios sk_wind,wp_square \
  --dwell-stay-factor 1.0 --episode-s 300 --sk-episode-s 300 \
  --seed 100 --seeds 4 --envs 8 --skip-pid --no-plots --out results_a1_conf_A_S

# phase 4 long run
$PY train.py --config configs/a1/train_long_A.yaml

# tables
$PY tools/collect_results.py --out reports/AUTO_TABLES.md
```

Add `--device cpu` to any `evaluate.py` command if it hangs at the checkpoint line (§7.1).

---

## 10. Files

- Rules: `reports/AUTO_RULES.md` (frozen, hashed) · Run log: `reports/AUTO_LOG.md`
- Instruments: `reports/R8b.md` · R8 context: `reports/R8.md`
- R9: `reports/R9_impl.md`, `pathgen.py`, `tests/test_pathgen.py`
- Tables: `reports/AUTO_TABLES.md` · Decision tools: `tools/decide_r8.py`, `tools/confirm_r8.py`,
  `tools/collect_results.py`
- Results: `results_a1_r8_*/`, `results_a1_conf_A_*/`, `results_a1_conf2_control_*/`,
  `results_a1_long_A_*/` (each `summary.json` is authoritative)
- Long run: `runs/a1_long_A/ckpt/ckpt_last.pt` (100M)
