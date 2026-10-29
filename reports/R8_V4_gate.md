# R8 / V4 — pre-export behaviour gate

**Scope: simulator only, placeholder physics (SYSID.md). Nothing in this report says anything
about the real vessel.** Do not claim sim results transfer to the real vessel.

## Why this exists

`drl/tests/` had no behaviour gate. `tests/test_onnx_parity.py` and `tests/test_runtime.py`
check observation/action shapes, ONNX-vs-torch parity and `|a| <= 1` — all against a **dummy
random MLP built inside the test**, never against `runs/r7_policy.onnx` or any real
checkpoint. A policy that reaches every waypoint and never settles passes every one of those
tests, which is how "R7 works" got shipped while `settling_time_s` was NaN/inf and
`settled_pct` between 0 and 53 %.

`tools/gate_policy.py` + `configs/gates.yaml` close that hole for **future** checkpoints: a
human runs it before exporting, and a non-zero exit blocks the export.

## Files

| file | role |
|---|---|
| `tools/gate_policy.py` | the gate: loads a ckpt, runs sk_wind + wp_square, prints PASS/FAIL, exits non-zero on any FAIL |
| `configs/gates.yaml` | thresholds **and** the pinned run settings, so a gate run is reproducible from the file alone |

## CLI contract

```
cd E:/buoy-project/drl
.venv/Scripts/python.exe tools/gate_policy.py --ckpt runs/r7_300m/ckpt/ckpt_last.pt
```

| flag | default | meaning |
|---|---|---|
| `--ckpt PATH` | `runs/r7_300m/ckpt/ckpt_last.pt` | checkpoint dict `{"model","optimizer","steps","iteration","cfg"}`; loaded exactly as `evaluate.load_policy` does |
| `--gates PATH` | `configs/gates.yaml` | thresholds + `settings:` block |
| `--seeds N` | `settings.seeds` = 4 | override seeds per scenario |
| `--envs N` | `settings.envs` = 8 | override envs per seed |
| `--scenarios CSV` | all keys of `scenarios:` | override scenario list |
| `--device` | cuda if available | torch device |
| `--out PATH` | none | optional gate JSON (per-scenario summary + every gate row + exit code) |

**Exit codes:** `0` every row PASS · `1` at least one FAIL **or** NOT AVAILABLE · `2` bad
usage / missing checkpoint / unknown scenario.

### Pinned standard-strict settings (from `evaluate.py`'s real argparse)

All of these flags exist in `evaluate.py` argparse; nothing was invented. They live in
`gates.yaml` under `settings:` and are passed to `evaluate.run_rollout` positionally the way
`evaluate.main` passes them.

| setting | value | `evaluate.py` flag |
|---|---|---|
| scenarios | `sk_wind`, `wp_square` | `--scenarios` |
| episode length | 300 s | `--episode-s` (**wp_square only**; sk_wind keeps its own 120 s) |
| dwell hysteresis | 1.0 | `--dwell-stay-factor` (1.0 = no hysteresis, pre-R5 behaviour) |
| seeds | 4 | `--seeds` (seed = `base + 1000*scenario_idx + k`) |
| envs | 8 | `--envs` |
| corner dwell | 10 s | `--square-hold` |
| settled tol | 1.0 m | `--tol` |
| settling hold | 5 s | `--hold-s` |

The gate runs `ppo` only (the PID baseline is irrelevant to export) and writes no CSVs or
plots. It does **not** run under pytest (needs a real checkpoint, takes ~11 min).

### Metric reading is defensive

The five R8 columns (`sat_frac`, `reversal_rate`, `arrival_speed_mean_mps`, `hold_rms_m`,
`hold_p95_m`) are read out of the summary dict, not recomputed. A key that is absent, NaN or
inf is printed as **NOT AVAILABLE** and **fails** the gate — an absent metric can never pass
silently. `settled_pct` is not a `summarize` key at all; it is derived the way
`evaluate.score` derives it (`100 if settling_time_s < inf else 0`).
`geofence_dropped_episodes` is counted by the gate itself (simulated env-episodes minus
episodes `evaluate.clean_episode` kept).

The R8 metrics landed in `evalkit/metrics.py` while this tool was being written; the gate
needed no change and picked them up on load (`sat_frac`, `reversal_rate`, `hold_rms_m`,
`hold_p95_m` are live; `arrival_speed_mean_mps` is live but still returns NaN, see below).

## The gate table

```
scenario   metric                     dir          value  status
-----------------------------------------------------------------------
sk_wind    settled_pct                min              0  FAIL
sk_wind    hold_rms_m                 max  NOT AVAILABLE  NOT AVAILABLE
sk_wind    sat_frac                   max          0.713  FAIL
sk_wind    reversal_rate              max              0  PASS
sk_wind    arrival_speed_mean_mps     max  NOT AVAILABLE  NOT AVAILABLE
sk_wind    geofence_dropped_episodes  max              0  PASS
wp_square  waypoints_reached          min           3.84  PASS
wp_square  lap_completed              min          0.906  PASS
wp_square  sat_frac                   max          0.755  FAIL
```

## Real R7 baseline run

**Checkpoint:** `runs/r7_300m/ckpt/ckpt_last.pt`, `steps=299958272`, `iteration=4577`.
**Settings:** the full pinned set above — seeds 4 × envs 8 per scenario, episode-s 300,
dwell-stay 1.0. No seeds were dropped; runtime ≈ 11 min total on cuda. 32 complete episodes
scored per scenario (of 32 simulated — no geofence exits).

```
.venv/Scripts/python.exe tools/gate_policy.py --ckpt runs/r7_300m/ckpt/ckpt_last.pt
EXIT=1
```

| scenario | metric | value | gate | verdict |
|---|---|---|---|---|
| sk_wind | settled_pct | **0.0 %** | min 80 | **FAIL** |
| sk_wind | hold_rms_m | NaN | max 1.5 | **NOT AVAILABLE → FAIL** |
| sk_wind | sat_frac | **0.713** | max 0.30 | **FAIL** |
| sk_wind | reversal_rate | 0.0 /min | max 30 | PASS |
| sk_wind | arrival_speed_mean_mps | NaN (n/a for sk: no waypoint plan) | max 0.6 | NOT AVAILABLE → FAIL |
| sk_wind | geofence_dropped_episodes | 0 | max 0 | PASS |
| wp_square | waypoints_reached | 3.84 / 4 | min 3.5 | PASS |
| wp_square | lap_completed | 0.906 | min 0.75 | PASS |
| wp_square | sat_frac | **0.755** | max 0.30 | **FAIL** |

Supporting pooled numbers from the same run: sk_wind `radial_error_mean` 15.53 m,
`radial_error_p95` 41.40 m, `settling_time_s` inf, `action_rate` 0.433 1/s. wp_square
`radial_error_mean` 10.07 m, `radial_error_p95` 29.74 m, `settling_time_s` inf, `action_rate`
0.539 1/s.

### How R7 fails, in plain words

1. **It never settles at all.** sk_wind `settled_pct` is 0.0 % over 32 episodes and
   `settling_time_s` is `inf` in both scenarios. This is the exact failure R7 was shipped
   with; the gate's `settled_pct` row is what would have caught it.
2. **It sits hard against the actuator limit.** 71 % of sk_wind steps and 76 % of wp_square
   steps have `max|a| > 0.9` on some axis. That is not a controller holding station — it is a
   policy pinned at saturation, which is consistent with (1): a saturated surge command cannot
   make the fine corrections a 1 m hold needs.
3. **It does pass the waypoint rows.** 3.84/4 corners and a 0.906 lap-completion fraction.
   This is precisely why "it reaches waypoints" was mistaken for "R7 works" — the pass-through
   behaviour passes the arrival metrics while failing every settling metric.

### Two NOT AVAILABLE rows are honest metrics artifacts, not gate bugs

Both were traced to source and confirmed with direct runs against the same checkpoint:

- **`sk_wind.hold_rms_m` = NaN.** `evalkit.metrics.final_hold_window` requires the trace to
  **end** in station keeping (`mode[-1] > 0.5`). On seeds 1000 and 3000 the R7 policy had
  `mode` uniformly 0 for the whole 300 s episode — it never entered the hold at all — so
  there is no final hold window and the metric is correctly NaN. Verified directly:
  `final_hold_window` returned `(-1, -1)` and `np.unique(mode) == [0.]` on those seeds.
  On seeds where it does hold, `hold_rms` returns real numbers (e.g. 6.21 m / p95 7.61 m).
  **NaN here is itself the finding: R7 frequently never reaches station-keeping mode.**
- **`sk_wind.arrival_speed_mean_mps` = NaN.** sk_wind has no waypoint plan, so `summarize`
  is called with `waypoints=None` and returns NaN by design. On wp_square it is also NaN, for
  a different reason: `metrics._entry_speeds` returns `nan` for any corner that is never
  entered, and `arrival_speed_mean_mps` does `np.mean` over all corners — so one missed
  corner poisons the mean despite the docstring saying "average over REACHED waypoints only".
  Measured per-corner entry speeds on wp_square seeds 1001-1004 were
  `[0.588, 0.468, 1.202, nan]`, `[0.964, 0.733, 1.43, nan]`, `[0.769, 0.805, 1.278, nan]`,
  `[0.871, 0.774, 1.422, nan]` — corner 4 never entered, honest NaN.

Per the ownership rules `evalkit/metrics.py` was **not** edited. Reporting this as a bug for
its owner: `arrival_speed_mean_mps` should use `np.nanmean` (or filter non-finite) to match
its documented "reached waypoints only" convention. Until then the gate will keep failing
this row on any partially-completed lap, which is arguably the conservative direction.

## Which thresholds are guesses

**Every threshold in `configs/gates.yaml` is a plain guess and every one is annotated
`# UNTUNED placeholder — no calibration data exists`.** None is derived from data. There is
no per-seed calibration sweep, no PID baseline envelope and no real-vessel data behind any
number:

| scenario | metric | guess | basis for the guess |
|---|---|---|---|
| sk_wind | `settled_pct` min | 80 % | "most episodes should hold"; R7 measures 0-53 % |
| sk_wind | `hold_rms_m` max | 1.5 m | 1.5× the 1.0 m GPS noise floor used as `--tol` |
| sk_wind | `sat_frac` max | 0.30 | "under a third of steps saturated" |
| sk_wind | `reversal_rate` max | 30 /min | "no thrash"; R7 measures exactly 0.0, so this row is not currently discriminating |
| sk_wind | `arrival_speed_mean_mps` max | 0.6 m/s | close to `max_surge_mps` 1.0; no arrival-speed sweep exists |
| sk_wind | `geofence_dropped_episodes` max | 0 | any exit is a failure — the only non-arbitrary value here |
| wp_square | `waypoints_reached` min | 3.5 / 4 | "3 of 4 corners on average"; R7 measures 3.84 |
| wp_square | `lap_completed` min | 0.75 | "three quarters of episodes close the lap"; R7 measures 0.906 |
| wp_square | `sat_frac` max | 0.30 | same as sk_wind |

Consequence to keep in mind: **a PASS from this gate means "not obviously broken", never
"good".** In particular the wp_square rows are already met by R7, a policy that never
settles — do not read a green wp_square block as evidence of a usable controller. The rows
that actually discriminate R7 today are `sk_wind.settled_pct` and both `sat_frac` rows.

Before this gate is trusted to block a real export, the thresholds need to come from a seed
sweep (per-seed best/worst band across several checkpoints), not from these guesses. Replacing
them is a `configs/gates.yaml` edit only — `tools/gate_policy.py` has no numbers in it.

## Reproducing

```bash
cd E:/buoy-project/drl
# full gate (the baseline above, ~11 min on cuda)
.venv/Scripts/python.exe tools/gate_policy.py --ckpt runs/r7_300m/ckpt/ckpt_last.pt --out gate.json

# fast smoke of the gate itself (~1 min)
.venv/Scripts/python.exe tools/gate_policy.py --ckpt runs/r7_300m/ckpt/ckpt_last.pt \
    --seeds 1 --envs 2
```
