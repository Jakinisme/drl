# R8_D — evaluate.py plumbing: R8 metric columns + disturbance seeding

Owner: eval plumbing only. Files touched: `evaluate.py`, `sim/disturbance.py`,
`tests/test_disturbance_seed.py`, this report. `envs/asv_env.py`, `configs/env.yaml`,
`train.py`, `evalkit/metrics.py` NOT touched. Nothing committed.

**All numbers below are simulator numbers on placeholder physics (SYSID.md). Nothing here
transfers to the real vessel.**

## 0. Headline

| item | state |
|---|---|
| 5 R8 columns in per-episode CSV, summary.csv, per_seed.csv, printed per-seed line | done, appended at the END, no existing column moved |
| `--seed` now moves wind/waves (was exactly 0.0 diff) | fixed, `seed_offset = seed*1000 + 17` (eval path only) |
| default disturbance stream (train.py, all old tests) | unchanged — `.get("seed_offset", 0)` is the only read, default still 0 |
| R7 reference run, reseeded | done: `results_r7_reseeded`, 4 seeds, 1674 s |
| **before/after seeding NOT comparable** | results produced before this change and `results_r7_reseeded` are different scenarios; only the reseeded run is the reference R8 must beat |

## 1. evaluate.py — where the column list was duplicated

Two lists, and only `METRIC_COLS` fed all three outputs:

| # | location (pre-change) | fed |
|---|---|---|
| 1 | `METRIC_COLS` (module level) | per_seed.csv + summary.csv header, the `_sd` twins, the markdown/`summary.png` table |
| 2 | `write_csv(out_dir / f"{name}_{kind}_{seed}_per_episode.csv", eps_rows, ["scenario","controller","seed","episode","env"] + list(agg) + TW_COLS)` | per-episode CSV header — **derived from `agg` key order, not from a literal list** |
| 3 | the `print(f"  {name} ...")` per-seed line | printed, **hardcoded format string**, no R8 fields existed |

Consequence: (1) and (2) would have picked the R8 keys up automatically (they are in `agg`
via `summarize`), but the printed line would silently NOT have them — the divergence the task
warns about. Fixed by:

* `METRIC_COLS` — appended, in this order, nothing above touched:
  `sat_frac, reversal_rate, arrival_speed_mean_mps, hold_rms_m, hold_p95_m`
  (pre-R8 prefix still `radial_error_mean … settled_pct, waypoints_reached, lap_pct,
  lap_completed`).
* new `PRINT_COLS = list(METRIC_COLS)` and the printed line now emits
  `... (time s) r8 sat_frac=… reversal_rate=… arrival_speed_mean_mps=… hold_rms_m=… hold_p95_m=… tw [...]`
  built **from `PRINT_COLS`**, so print and summary share one list; `tw` stays last as before
  (it is per-episode only, not aggregated).
* the per-episode CSV needed no literal change (it derives from `agg`); verified the R8 block
  now sits at the end of the metric span, before the R5 `time_to_waypoint_1..4` tail:
  `…,settled_pct,energy_j_proxy,lap_pct,n_episodes,time_to_waypoint_1,…,time_to_waypoint_4`
  → `…,lap_pct,n_episodes,sat_frac,reversal_rate,arrival_speed_mean_mps,hold_rms_m,hold_p95_m,time_to_waypoint_1,…`
  (the new names are new, so no existing column's name or position changed; `waypoints_reached`
  and `lap_completed` keep their summarize positions).

Also fixed, pre-existing, would have crashed ANY run (not caused by R8): `plots()` did
`int(wp["waypoints_reached"])` on `nan` for scenarios with no waypoint plan (`sk_wind`,
`rough_sea`) → `ValueError: cannot convert float NaN to integer`, killing the whole eval at
the first plot. Now `-1` for "no plan". Found because this run crashed on it.

`summary.json` gained `disturbance_seeding: {scheme, applies_to, per_seed:[{scenario, seed,
seed_offset}]}`; each per_seed row also carries `disturbance_seed`.

## 2. Disturbance seeding (`sim/disturbance.py` + `evaluate.py`)

Change in `sim/disturbance.py` is 3 comment lines only: line 93 already reads
`self._gen.manual_seed(int(cfg.get("seed_offset", 0)))`, and that IS the additive hook —
no key → stream unchanged. `configs/disturbance.yaml` still ships `seed_offset: 0`, so
`train.py` (which never sets it) is bit-identical, and `tests/test_disturbance.py` (which
already passes explicit offsets) is untouched.

`evaluate.py`:

```python
def disturbance_seed_offset(seed, env_index=0):     # seed*1000 + 17
run_rollout(..., env_index=0)                       # >0 -> stamps seed_offset into the
                                                    #   disturbance cfg (evaluate CLI passes si+1)
```

`env_index` defaults to 0 = old behaviour, so `tests/test_evaluate_dump.py`, `test_eval_dump.py`
and any caller keep the shared-stream rollout. The offset is per ROLLOUT, not per env: `AsvEnv`
constructs one `Disturbance` for the whole batch, so the wind/wave stream is shared by the envs
of a batch (unchanged structure; noted, not fixed — it is an env-side change).

Measured, same env index, cold `Disturbance`:

| case | wind_speed | wind_dir | wave_dir | wave_phase |
|---|---|---|---|---|
| default vs default (the bug, `--seed 0` vs `--seed 1234`) | 0.0 | 0.0 | 0.0 | 0.0 |
| `seed_offset 17` vs `1234017` (fixed) | 6.08 | 2.96 | 2.67 | 4.58 |

## 3. Tests — `tests/test_disturbance_seed.py` (10 passed)

```
$ .venv/Scripts/python.exe -m pytest tests/test_disturbance_seed.py -q
..........                                                               [100%]
10 passed in 2.78s
$ .venv/Scripts/python.exe -m pytest tests/test_disturbance_seed.py tests/test_disturbance.py tests/test_evaluate.py -q
........................                                                 [100%]
24 passed in 3.46s
```

Covered: (a) no-key == explicit 0 == a freshly `manual_seed(0)` generator and == an instance
built the old way (`_maxdiff == 0.0`, `torch.equal` on wave_phase); (b) seeds 0 vs 1234 give
different wind/wave draws AND a different level-1 wrench at the same hull state/time; (c) same
seed reproduces bit-exactly while `reset()` redraws; `env_index=0` passes `disturbance_cfg=None`
(old path) and `env_index=1` stamps 1017; `METRIC_COLS` ends with the 5 R8 names and its
pre-R8 prefix is unchanged; `PRINT_COLS == METRIC_COLS`; `summarize()`'s last 5 keys equal the
5 names; the printed `r8` token order == `METRIC_COLS[13:]` == summary.csv tail order;
`write_csv` header == requested columns; `_sd` twins follow the same order.

Known limitation of the column-consistency test: the printed line is rendered inside `main()`,
so the test asserts on the SAME generator expression (`f"{c}={fmt(agg.get(c))}" for c in
PRINT_COLS[13:]`) rather than capturing stdout. A future edit that rewrites the print statement
with a literal format string would not be caught by it — keep the line generated from
`PRINT_COLS`.

## 4. R7 reference run (reseeded) — `results_r7_reseeded`

```
.venv/Scripts/python.exe evaluate.py --ckpt runs/r7_300m/ckpt/ckpt_last.pt \
    --scenarios sk_wind,wp_square --episode-s 300 --dwell-stay-factor 1.0 \
    --seeds 4 --out results_r7_reseeded
```
4 seeds x 8 envs x 1 episode, 1674 s total, cuda. Note `--episode-s` only affects
`wp_square` (sk_wind stays 120 s, see `--episode-s` help). `n_episodes` < seeds*8 where an
env left the geofence (incomplete episodes are dropped, unchanged accounting).

````
| scenario | controller | seeds | n_episodes | radial_mean | radial_p95 | radial_max | xtrack_rmse | energy | energy_j_proxy | action_rate | settled_pct | wps_reached | lap_pct | lap_done | sat_frac | reversal_rate | arrival_speed_mean_mps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sk_wind   | ppo | 4 | 32 | 18.96 | 45.67 | 48.45 | 10.99 | 205.2 | 39417  | 0.388 | 15.6 | nan  | nan  | 0     | 0.840 | 0 | nan |
| sk_wind   | pid | 4 | 31 | 35.64 | 45.42 | 47.09 | 35.06 | 176.6 | 33912  | 1.014 |  0.0 | nan  | nan  | 0     | 0.761 | 0 | nan |
| wp_square | ppo | 4 | 31 |  8.97 | 29.72 | 44.21 |  1.28 | 530.0 | 101797 | 0.561 | 73.7 | 3.97 | 99.2 | 0.969 | 0.753 | 0 | 1.031 |
| wp_square | pid | 4 | 26 | 20.17 | 32.46 | 41.82 |  8.19 | 365.9 |  70275 | 1.184 |  0.0 | 1.01 | 25.3 | 0     | 0.688 | 0 | 0.488 |
````

Per-seed sd (from summary.csv): sk_wind ppo radial 2.03 / p95 5.46; sk_wind pid radial 4.12;
wp_square ppo radial 0.27, lap 0.054, arrival_speed sd 0.062; wp_square pid radial 5.57.

`s settling_time_s` is `inf` (→ printed `nan` in the CSV means never settled) for every row: the
`--tol 1.0 m` settled test fails on these traces. `hold_rms_m` / `hold_p95_m` are `nan`/`inf`
for every row: sk_wind never closes to the hold point (radial 19-36 m), and `wp_square` ends
mid-leg / inside a corner hold only on some envs, so the aggregate hides the few defined
values. `reversal_rate` is 0 everywhere — surge |a| > 0.5 flips are absent with the rate-limited
action. These columns are wired and populated; they are simply not informative for R7.

Two gates for R8 that fall out of this table:
* `wp_square`: PPO already reaches ~4/4 waypoints (lap 99.2 %, xtrack 1.28 m) but
  `settled_pct=73.7` and `sat_frac=0.75` — a high-`sat_frac`, poorly-settling lap.
* `sk_wind`: PPO is *not* station keeping (19 m mean, 15.6 % settled vs PID 35.6 m / 0 %);
  it wins on sk_wind only relative to PID, not in absolute terms.

**Comparison rule for R8:** results from before this change (`results_r7`, `results_r6_*`,
anything in reports/C2..R6 for sk_wind/wp_square) used a FIXED disturbance stream
(`seed_offset 0`) for every seed, so their cross-seed spread measured spawn-pose noise only.
`results_r7_reseeded` is the only R7 number a reseeded or new R8 variant may be compared against.
