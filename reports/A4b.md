# A4b — runtime tests + offline replay (Wave 1, closing piece)

Date: 2026-10-26. Owner: A4b.
Files: `drl/tests/test_runtime.py` (new, 20 tests), `drl/runtime/infer_replay.py` (new),
`drl/reports/A4b.md` (this file). `runtime/policy.py`, `runtime/obs_np.py` and every
other module untouched — no test exposed a defect there.

`cd drl && .venv/Scripts/python.exe -m pytest tests -q` -> **79 passed** (59 pre-existing
+ 20 new). Nothing here claims behaviour on the water: there is still no trained model,
so every test builds a dummy ONNX MLP and asserts plumbing only.

## `tests/test_runtime.py`

No torch anywhere in the file: the dummy model is an `onnx.helper` graph
(150 -> 8 tanh -> 3 tanh, opset 17, input `obs` `[B,150]` f32, output `action`
`[B,3]` f32, seeded random weights — never trained, never measured for accuracy).
Model injection paths covered: `onnx_bytes`, `session=`, `onnx_path` + `meta_path`.

| test | what it pins down |
|---|---|
| `test_raw_sample_matches_contract_layout` | §5 order, hand-computed values: bow north + goal 10 m east -> `(goal_dx, goal_dy) = (0, 10)`; `psi_err=90°` -> `sin=1, cos=0`; `u,v,r`, `a_prev`, `wp1/wp2` (ENU -> body), `mode`; missing wp/`a_prev` -> zeros |
| `test_obs_history_oldest_first_and_deterministic` | 12 distinct builds -> slot `k` holds sample `t-9+k`, newest last, `float32[1,150]`; slots not duplicated; `reset()` re-tiles; same input sequence -> bit-identical output |
| `test_controller_uses_model_and_builds_contract_obs` | recording session wrapper proves `step()` really feeds the ONNX model: input `[1,150]` f32, last 15 = hand-computed §5 sample (incl. `(u,v)` from SOG 1.2 / COG 60° / heading 30° per §4.10), first 9 slots tiled; returned command == model output; step 2 shifts the ring by one and `a_prev` (idx 7:10) == previous applied command |
| `test_controller_reads_policy_meta` | `policy_meta.json`: input/output names, `safety.rate_limit_per_s`, `policy_dt_s` honoured |
| `test_goal_mode_flag_and_missing_heading` | `mode="station_keeping"` -> obs idx 14 == 1.0; `goal.heading_deg=None` -> `sin/cos = 0/1`; `next_lat/next_lon` -> wp1 non-zero |
| `test_controller_clips_to_unit_range` | fake session returning `[5, -9, 0.25]` -> `[1, -1, 0.25]` |
| `test_slew_rate_limit` | 8 steps at 0.1 s: every `|Δa| <= rate_limit_per_s · dt = 0.2`, and the limit is actually active (max Δ > 0.05) — not a vacuous check |
| `test_geofence_zeroes_the_command` | 79 m from the latched origin -> `policy`; 81 m > `geofence_m` 80 -> `source="geofence"`, command zeros |
| `test_watchdog_on_stale_sample` | `est["t"]` 1.0 s old (> `watchdog_s` 0.5) -> `fallback`/`watchdog` |
| `test_watchdog_on_gap_between_calls` | caller stalls 1.0 s -> `fallback`/`watchdog` |
| `test_gps_invalid_uses_fallback_hook` | `gps_valid=False` -> hook called once, its output returned, `fault="gps"`; no hook -> zeros; `gps_valid=None` and `lat=None` also fault |
| `test_fallback_output_is_slewed_too` | hook returning 1.0 is rate-limited on handover (no step) |
| `test_sensor_validity_semantics` | explicit `sensor_ok=False` -> `fault="sensor"`; key absent -> derived from `imu_valid & mag_valid` (§4.4) |
| `test_legacy_has_fix_is_not_used` | `has_fix=False` + `gps_valid=True` -> `source="policy"` (INSPECTION H9: not the legacy flag) |
| `test_non_finite_policy_output_is_zeroed` | NaN, inf, and a 2-wide output -> zeros, `source="zero"`, `fault="nan"` |
| `test_history_is_reset_after_a_fault` | the obs after a fallback is re-tiled — stale K=10 samples never replay into the first post-fault observation |
| `test_no_model_raises` | `RLController()` -> `ValueError` |
| `test_infer_replay_runs_on_a_logged_csv` | subprocess `python -m runtime.infer_replay --csv tmp.csv --onnx tmp.onnx` -> exit 0, stdout `steps: 20`, per-channel min/max, `fallback=2 gps=1 watchdog=1` (row 3 no fix, row 7 `est_age_s=0.9`) |
| `test_infer_replay_rejects_a_missing_csv` | `main()` -> exit code 2 |
| `test_runtime_imports_without_torch` | subprocess imports `runtime.obs_np/policy/infer_replay` and asserts `torch not in sys.modules` (must be a subprocess: other test modules import torch) |

## `runtime/infer_replay.py`

`python -m runtime.infer_replay --csv log.csv --onnx policy.onnx [--meta policy_meta.json]
[--limit N]`. numpy + onnxruntime + stdlib only. Builds the same `RLController` the Pi
runs, drives it row by row from the CSV and prints:

```
steps: 20  (log.csv -> policy.onnx)
  surge min -0.115  max +0.114
  sway  min +0.200  max +0.680
  yaw   min +0.168  max +0.501
sources: fallback=2  policy=18
faults:  gps=1  watchdog=1
```

CSV schema (header row, units per CONTRACT §4.4/§4.8; `t` is both replay clock and
estimator timestamp, so the watchdog sees real step gaps):

- required: `t, lat, lon, heading_deg, speed_mps`
- optional state: `yaw_rate_deg_s, course_deg, gps_valid, imu_valid, mag_valid, est_age_s`
- optional goal: `goal_lat, goal_lon, goal_heading_deg, mode, next_lat, next_lon,
  next2_lat, next2_lon`

Defaults: missing booleans = true, `course_deg` = `heading_deg`, missing goal = hold the
current point, `mode` = `waypoint` (numeric 0/1 also accepted). `est_age_s` pretends the
sample is that old, which is the only way an offline log can trip the §4.6 watchdog on
data age rather than on row spacing. Bad file / bad column -> exit code 2.

## Bugs

**None found in `runtime/policy.py` or `runtime/obs_np.py`.** Both behaved per
CONTRACT §4.6/§4.8/§5 on first contact: history order and tiling, `(u,v)` from
`common/estimator.py`, clip/slew on both the policy and the fallback path, `gps_valid` /
`sensor_ok` semantics, geofence, watchdog, NaN guard, history reset after a fault. The
only failures while writing the tests were in the tests themselves (a slew assertion that
assumed the first command starts at zero, and a stdout label check).

Changes made in A4b's own file, `runtime/infer_replay.py`:

1. A numeric `mode` column (a logger writing the 0/1 flag) fell through the
   `_MODES` dict lookup and was silently treated as `waypoint` — now parsed to `float`.
2. Removed a duplicated `_opt(row, "goal_lat")` call in `_state()` (no behaviour change).

Findings left in place, not "fixed" (each needs a decision, not a patch):

- **Slew magnitude wording.** `A4a.md` says "slew 0.1/step". The code is
  `rate_limit_per_s (2.0/s) · measured dt`, i.e. **0.2 per step** at the policy rate of
  0.1 s, and 0.1 per step only if the caller runs the real 20 Hz loop. Both are
  consistent with CONTRACT §3.4 (`|Δa| <= rate · policy_dt`); the tests assert the
  measured-dt form.
- **`gps_timeout_s` (2.0 s) in `DEFAULT_SAFETY` and in `policy_meta.json` is never read.**
  `watchdog_s` (0.5 s) is the only staleness gate. Harmless today, but two keys suggesting
  two different timeouts is a trap for whoever tunes this; deciding which wins is a
  contract edit, so it is reported rather than changed.

## TODO

1. **No trained ONNX model** — every number above comes from a random dummy. Accuracy and
   behaviour are unmeasured; sim-to-real validity is not claimed anywhere.
2. **`est["t"]` semantics must be confirmed in the Pi glue.** The watchdog compares
   `now - est["t"]` against 0.5 s. If `pi-control` passes the time of the last GPS fix
   (1 Hz) instead of the fusion update time, the controller sits in PID fallback
   permanently. Must be settled before the first on-water run.
3. **Replay fallback is zeros.** `baseline_pid.py` exposes the batched `act()` interface
   (CONTRACT §4.6), not the `(est, goal) -> (surge, sway, yaw)` hook `RLController` wants;
   an adapter belongs with the Pi glue, not in `infer_replay.py`.
4. **The CSV schema above is a proposal** — no logger writes it yet. When `pi-control`
   starts logging, either match these column names or update `_state()` and this report.
5. **No `est_age_s` in real logs**, so an offline replay of a real run can only detect
   watchdog gaps, not stale sensor data. Add the column to the logger.
6. **Geofence origin = first valid fix = launch point**, not a surveyed site (same open
   question as A4a TODO 4 / CONTRACT T6).
7. **Not covered here, by design:** torch-vs-ONNX parity < 1e-4 is C3
   (`tests/test_onnx_parity.py`); numpy-vs-torch observation parity is B1
   (`tests/test_obs_parity.py`). `obs_offset/obs_scale` from `configs/env.yaml` do not
   exist yet, so the controller uses the CONTRACT §3.4 literals.
