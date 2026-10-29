# R8 / V3 — new evaluation columns (settling quality + command harshness)

Owner: metrics change only (`evalkit/metrics.py`, `tests/test_metrics_r8.py`). Nothing here
touches `evaluate.py`, `envs/`, `configs/` or any pre-existing test.

## What changed

`evalkit/metrics.py` gains five metrics and five new keys appended to the END of
`summarize()`'s per-episode dict. `git diff --stat`: `1 file changed, 167 insertions(+)`,
**0 deletions** — every pre-existing key keeps its value and its position.

New keys, in this order, after `time_to_waypoint` (the last R1–R6 key):

| # | key | type |
|---|-----|------|
| 1 | `sat_frac` | float in [0, 1] |
| 2 | `reversal_rate` | float, 1/min |
| 3 | `arrival_speed_mean_mps` | float, m/s, or nan |
| 4 | `hold_rms_m` | float, m, or nan |
| 5 | `hold_p95_m` | float, m, or nan |

Public functions added and exported in `__all__`: `sat_frac`, `reversal_rate`,
`arrival_speed_mean_mps`, `hold_rms` (returns the `(rms, p95)` pair so the two columns are
computed on the same sample set), plus `final_hold_window` (the split helper, tested
directly). Module constants `SAT_A = 0.9`, `REV_A = 0.5`, `HOLD_WINDOW_S = 30.0`,
`ARRIVAL_RADIUS_M = 2.0`.

All five are pure functions of one episode's arrays. They default to nan / 0.0 when the
data they need is absent, never raise.

## Formulas as implemented

`t` [T], `xy` [T,2], `goal_xy` [T,2], `action` [T,3] (`[surge, sway, yaw]`, normalized to
[-1, 1] — CONTRACT §4.5), `mode` [T] (1 = station keeping, 0 = waypoints — CONTRACT §4.6).

### `sat_frac` — fraction of saturated policy steps

```
sat_frac = mean_t( any_axis( |a_t| > 0.9 ) )
```

Any one axis over the threshold in absolute value marks that step; a step with two
saturated axes still counts once. Strict `>`. Exactly `|a| = 0.9` does not count. No
samples -> `0.0`.

### `reversal_rate` — surge sign changes per minute, gated

```
ok_t       = |a_surge,t| > 0.5
n_reversal = #{ t : ok_t  and  ok_{t+1}  and  sign(a_surge,t) != sign(a_surge,t+1) }
reversal_rate = 60 * n_reversal / (t[-1] - t[0])      [1/min]
```

Two consecutive samples must BOTH exceed 0.5 with opposite signs. A flip that passes
through the ±0.5 band (`+0.8, 0.0, -0.8`) is **not** a reversal — that is deliberate: it
measures effort reversal, not every zero crossing. Exactly 0.5 is not > 0.5. Sway/yaw
ignored. `t` shorter than 2 samples or zero span -> `0.0`.

### `arrival_speed_mean_mps` — how fast the vessel was when it first got there

```
entry_k = min{ i : ||xy_i - c_k|| < 2.0 }              (strict <, 2.0 m default)
v_k     = ||xy_{entry_k} - xy_{entry_k - 1}|| / (t_{entry_k} - t_{entry_k - 1})
arrival_speed_mean_mps = mean over REACHED corners of v_k     (nan if none)
```

**Convention used, stated explicitly:** the same 2.0 m radius and the same strict `<` test
as `waypoint_progress` (`radius=2.0` default there, `ARRIVAL_RADIUS_M` here), but evaluated
**per corner independently in planned order**, with no in-order / active-goal bookkeeping.
`waypoint_progress` deliberately credits a corner only when `goal_xy` is near it and only in
order; a stationary speed metric should not inherit that (it would silently drop a corner
the policy passed early). A near-miss sequence (`... c1 c2 c1 ...`) therefore reports the
FIRST entry of each corner, which is what "arrival" means for a speed.

Consequences, all deliberate:
- speed is the **one-sided difference** — the speed the vessel HAD on entering, not after;
- entry at sample 0 (spawned inside the radius) has no previous sample -> that corner's
  speed is nan, and the mean over reached corners is nan rather than dropping it;
- entry at the LAST sample of the trace is nan for the same reason;
- unreached corners are dropped; nothing reached, no `waypoints`, `xy` shorter than `t`, or
  `t` shorter than 2 -> nan.

### `hold_rms_m` / `hold_p95_m` — how well the last hold is held

```
hold run  = maximal suffix of the trace with mode > 0.5        (final SK run)
window    = samples in the hold run with t >= t[-1] - 30.0
e_i       = ||xy_i - goal_xy[-1]||                            (final hold point)
hold_rms_m = sqrt(mean(e^2))     hold_p95_m = percentile(e, 95)  (linear interpolation)
```

- **A final hold means the trace ENDS in station keeping.** A waypoint episode, or an SK
  episode whose trace ends mid-leg (`mode` 0 at the last sample), has no final hold ->
  `(nan, nan)`. This is the R8 "sk_wind" column: `sk_wind` and `rough_sea` are `mode=1`
  for the whole 120 s episode, so their hold is the last 30 s of the trace.
- **Episode shorter than 30 s:** the window is *clipped*, not padded — the whole hold is
  used. No synthetic samples, no extrapolation. Say it in the report text, not by a 0.
- The 30 s window is intersected with the hold run: a trace that went into SK only 5 s
  before the end is scored on those 5 s, not diluted with the 25 s of approach that precede
  them inside the same 30 s.
- Goal = `goal_xy[-1]`, the final commanded hold point (equals the constant hold point for
  an SK episode; use of the last sample rather than a mean keeps it a pure function of the
  trace).
- No final hold, empty window, `xy`/`goal_xy` shorter than `t`, or any non-finite error in
  the window -> `(nan, nan)`. A scenario without a final hold yields NaN, never 0.0.

## Where the `traj` dict does NOT carry what a metric needs

Stated explicitly, not worked around:

1. **No per-sample "was this a policy step?" flag.** `sat_frac` and `reversal_rate` treat
   every row of `traj["action"]` as one policy step. `evaluate.py::clean_episode` builds
   `t = arange(T) * dt` and `action` = the applied command per policy step (`env.prev_action`,
   §4.5), so this holds today; if a future dump ever decimates the action stream, these two
   columns must be recomputed at the policy rate.
2. **No per-step "saturated before or after the env's rate clamp".** `action` is the
   **applied** command (post-clamp). `action_raw` (pre-clamp tanh mean) exists in the
   evaluate rollout but is not part of the `traj` contract, so it is not used. `sat_frac`
   therefore measures commanded saturation as the vehicle saw it.
3. **`goal_xy` says nothing about a "final goal" distinct from the last sample.** The hold
   metrics use `goal_xy[-1]`. For an SK episode the goal is constant by construction
   (`envs/asv_env.py`), so this is exact; for a hypothetical scenario with a moving goal the
   column would be measuring against the last commanded point only.
4. **No "final hold" label.** Derivable only from `mode`; `traj` has no `scenario` /
   `is_sk` field, which is why the rule is "the trace ends in SK". Nothing in the metrics
   module knows a scenario name — evaluating `sk_wind` is what makes the hold columns
   populated.
5. **Waypoints may be absent.** `arrival_speed_mean_mps` follows `summarize`'s existing
   `waypoints` / `traj["waypoints"]` source and is nan when neither is present — the same
   contract as `waypoints_reached`. An infeasible fallback (ordered unique `goal_xy`, as the
   path column uses) is NOT used: at an SK hold point the ordered-unique list is a single
   point at the goal, which would report a meaningless "arrival speed".
6. **No arrival-radius per waypoint.** One `arrival_radius` (default 2.0) for all corners,
   forwarded from `summarize(arrival_radius=...)` — same as `waypoint_progress`.

## Hand-computed test cases (`tests/test_metrics_r8.py`)

26 tests, all compare against numbers computed with pencil and paper from the literal
arrays in the test. Run: `26 passed in 0.18s` (plus `tests/test_metrics.py` unchanged:
`53 passed` combined; `tests/test_evaluate.py tests/test_eval_dump.py`: `9 passed`).

### sat_frac
- `[0.5, 0.5, -0.9, 0.2, 0.3, 0.0]` -> 0 saturated -> `0.0`.
- only `a[2,1] = 0.95` (sway) -> 1/6.
- `[1.0, 0.5, 0.6, 0.2, 0.3, 0.0]` -> only sample 0 -> 1/6.
- `[1.0, 0.95, -0.9, 0.2, 0.3, 0.0]` -> 1.0 and 0.95 -> 2/6.
- `[-1.0, 0.95, 0.5, 0.2, 0.3, 0.0]` -> 2/6.
- `[0.0, 0.9, -0.9, 0.0, 0.0, 0.0]` -> threshold strict -> `0.0`.
- `[[0,0.95,0], [0,0,-0.99], [-0.91,0.91,0]]` -> 3 steps -> `1.0` (third step: two axes, once).
- `ones((4,3))` -> 1.0; `full((4,3), 0.5)` -> 0.0; `zeros((0,3))` -> 0.0.

Note: `numpy` 2.2 prints `-0.96 > 0.9` as `False` for the scalar but `abs(-0.96) > 0.9` is
`True`; the implementation uses `np.abs`, and an earlier draft of this test that asserted
on raw signed comparison was wrong and was corrected.

### reversal_rate
- surges `-0.8 -0.8 0.8 0.2 0.8 -0.8`, `t = 0..5 s`: flips at idx 2 and idx 5 (idx 3/4 have
  a 0.2 side) = 2 flips -> `60*2/5 = 24.0` per minute.
- `+0.8, 0.0, -0.8` -> 0 (deadband crossing).
- `+0.5, -0.8` -> 0 (0.5 is not > 0.5).
- sway `+0.9, -0.9` -> 0 (surge only).
- `+1, -1, +1` over `t = 0, 10, 20`: 2 flips over a 20 s span -> `60*2/20 = 6.0` per minute
  (asserted as `120.0/20.0`); over `t = 0, 5, 20` -> `60*2/20 = 6.0` too, because the
  denominator is the full span, not the flip interval. Zero span and a single sample -> 0.0.

### arrival_speed_mean_mps
- 1 m sticks at 0.5 s on the x axis, corners `(4,0)` and `(12,0)`: first inside 2.0 m is at
  `x = 3` and `x = 11` (a distance of exactly 2.0 m is excluded), each entered at
  1 m / 0.5 s -> mean `2.0`.
- `x = 0, 2, 4, 4.5, 5` at 0.5 s, corner `(4,0)`: first inside is `x = 4` (2.0 m excluded),
  the step BEFORE it is `2 m / 0.5 s = 4 m/s` while the vessel is already slow afterwards ->
  `4.0`. Pins "speed on entering, one-sided".
- off-axis: `(0,5), (1,5), (5,0), …`, corner `(6,0)`: distances `7.81, 7.07, 1.00` ->
  entry at sample 2; `||(5,0)-(1,5)||/0.5 = hypot(4,5)/0.5`; corner `(20,0)` unreached ->
  dropped.
- both corners `(5,0)` and `(4,0)` inside 2 m at the same sample -> two 2 m/s entries -> `2.0`.
- unreached only -> nan; empty corner list -> nan; single-sample trace -> nan; entry only at
  the final sample -> nan; spawn already inside the radius -> nan.

### hold
- `final_hold_window`: `mode = 0 0 1 0 0 1 1 1 1 1` -> `(5, 10)`; all-SK -> `(0, 10)`;
  all-waypoint -> `(-1, -1)`; `1 1 1 1 1 0 0 0 0 0` (ends on a leg) -> `(-1, -1)`;
  empty -> `(-1, -1)`.
- constant 0.5 m offset over 100 s -> `(0.5, 0.5)`.
- window edge: err `0.5` for `t >= 69`, `2.0` before, 100 s -> `(0.5, 0.5)`; err alternating
  0/1 over the window (`t >= 69`, 31 samples, 16 ones) -> `sqrt(16/31)`; a 1 m spike at
  `t = 68` (one sample outside) -> exactly `(0.0, 0.0)`.
- 10 s episode (shorter than 30 s -> whole trace), err `0,1,0,1,…`: 5 ones of 10 ->
  `sqrt(0.5)`; p95 rank `0.95*9 = 8.55` -> `1.0`.
- SK only for the last 5 samples, err 2 there and 0 before -> exactly `(2.0, 2.0)` (the
  30 s window is clipped to the SK run, so the earlier zero-error approach samples do not
  dilute it).
- no final hold (waypoint episode, or ends mid-leg) -> `(nan, nan)`; empty trace ->
  `(nan, nan)`; `xy` one row short of `t` -> `(nan, nan)`; nan final goal -> nan.

### Regression: pre-existing columns unchanged
- `test_summarize_appends_r8_columns_last_in_order`: `list(summarize(traj))[:12]` equals
  the pre-R8 key list, and `list(...)[12:]` equals the five new keys in order.
- `test_summarize_preexisting_values_unchanged_hold_episode`: hand-computed goldens for a
  40 s constant-0.5 m SK trace — `n_samples=40`, `duration_s=39.0`, all three radial
  columns `0.5`, `cross_track_rmse=0.5`, `settling_time_s=0.0`, `energy=0.0`,
  `action_rate=0.0`, `waypoints_reached=nan`, `lap_completed=False`,
  `time_to_waypoint=[]`.
- `test_summarize_preexisting_values_unchanged_waypoint_episode`: 11-sample square-leg
  trace (surge `+1, -1, +1`, goal `(5,0)` x6 then `(10,0)` x5) — `radial_error_mean`,
  `radial_error_p95`, `radial_error_max=5.0`, `cross_track_rmse=0.0`,
  `settling_time_s=inf`, `energy=2.5` (trapezoid `0.5*(1+1)+0.5*(1+1)+0.5*(1+0)`),
  `action_rate=0.5` (TV `2+2+1` over 10 s), `waypoints_reached=2`, `lap_completed=True`,
  `time_to_waypoint=[4.0, 9.0]` (the 2 m radius catches corner 2 at x = 9) — plus the R8
  columns on the same trace: `sat_frac=3/11`, `reversal_rate=12.0`, `arrival=1.0`, hold
  `nan`.
- Existing `tests/test_metrics.py` (27 tests) passes untouched, including its own
  `summarize` assertions.

`git diff evalkit/metrics.py` reports `0` removed lines: the change is purely additive.

## Real test output

```
$ .venv/Scripts/python.exe -m pytest tests/test_metrics_r8.py -q
26 passed in 0.18s

$ .venv/Scripts/python.exe -m pytest tests/test_metrics_r8.py tests/test_metrics.py -q
53 passed, 14 warnings in 2.91s      (warnings: pyparsing/matplotlib deprecations, pre-existing)

$ .venv/Scripts/python.exe -m pytest tests/test_evaluate.py tests/test_eval_dump.py -q
9 passed in 2.89s
```

## Wiring notes for whoever touches `evaluate.py`

- The five keys are already in every `summarize()` dict, so `evaluate.py::score` carries
  them for free and `aggregate` will mean them (nan-mean is nan for a scenario with no hold —
  that is the intended "not applicable" signal, not a bug). Whether a CSV column is emitted
  is `evaluate.py`'s `METRIC_COLS` / per-episode `cols` decision — not changed here.
- `arrival_speed_mean_mps` needs `waypoints=` to be passed to `summarize` (or
  `traj["waypoints"]`); `evaluate.py::score` already forwards `waypoints`.
- `hold_*` need `traj["mode"]`, which `clean_episode` already records.
