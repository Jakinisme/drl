# A4a — runtime policy + numpy observation builder (Wave 1)

Date: 2026-10-26. Owner: A4a. Files: `runtime/obs_np.py`, `runtime/policy.py`.
numpy + onnxruntime + `common/` only — importing `runtime.policy` leaves `torch` out of
`sys.modules` (asserted in the smoke check). No real model exists yet, so nothing here
claims accuracy on the vessel or in sim.

## `runtime/obs_np.py` — CONTRACT §5, parity §6

| name | meaning |
|---|---|
| `raw_sample(est, goal, prev_action=None, wp1=None, wp2=None)` | float64 `[B,15]`, order exactly §5: `goal_dx, goal_dy, sin(psi_err), cos(psi_err), u, v, r, a_prev(3), wp1_dx, wp1_dy, wp2_dx, wp2_dy, mode` |
| `normalize(raw, offset, scale, clip_norm)` | `clip((raw-offset)/scale, ±clip_norm)`; `scale` divides |
| `ObsBuilder(offset, scale, clip_norm, k_history=10)` | `.reset()`, `.build(...) -> float32 [B,150]` |
| `DEFAULT_OFFSET/SCALE/CLIP_NORM` | CONTRACT §3.4 values, so the module needs no YAML |

- Frame math is only `common.frames.goal_in_body` / `heading_error_rad` — same helpers B1
  must mirror in torch; no `atan2` shortcut, no second formula.
- `goal[:,3]` is the mode flag (0 waypoint, 1 station keeping). `wp1`/`wp2` are absolute
  ENU positions; `None` -> `0.0` (last waypoint / station keeping, per §5).
- Normalization per step **before** buffering; buffer is oldest-first, newest-last;
  first `build()` after `reset()` tiles that sample into all K slots.
- float64 internally, single cast to float32 at return (§6).
- Batch size change re-initialises the buffer (no mixed-batch history).

## `runtime/policy.py` — CONTRACT §4.6 / §4.8

```python
RLController(onnx_path=None, meta_path=None, fallback=None, *, onnx_bytes=None,
             session=None, obs_cfg=None, safety=None, min_speed_mps=0.3,
             decay_tau_s=1.0, clock=time.monotonic)
    .reset()                                 # history, origin, (u,v), last action
    .step(estimator_state: dict, goal: dict) -> (surge, sway, yaw)   # floats in [-1,1]
    .status                                  # {"source", "fault", "action"}
```

Model injection for testing without a trained network: `onnx_path` + `meta_path`,
`onnx_bytes`, or a ready `session` (any object with `get_inputs/get_outputs/run`, so the
dummy model does not even have to be ORT). Input/output names come from
`policy_meta.json`, else from the session itself.

`estimator_state` is read per §4.8 (`lat, lon, heading_deg, yaw_rate_deg_s, speed_mps,
course_deg, gps_valid, imu_valid, mag_valid, t`); `u, v` come from
`common/estimator.estimate_body_velocity` with the measured `dt` (clipped 1 ms…0.5 s),
never duplicated here. `heading_deg: None` in `goal` means no heading hold ->
`psi_goal = psi` (obs `sin/cos` = 0/1). Optional `next_lat/next_lon`,
`next2_lat/next2_lon` fill `wp1/wp2`.

### Safety decisions

Order per step: **validity -> watchdog -> geofence -> policy -> clip -> slew -> fallback**.

| condition | source | output |
|---|---|---|
| `gps_valid == False` or `lat/lon is None` | `fallback` | PID hook, else zeros |
| `sensor_ok == False` (derived from `imu_valid & mag_valid` when the key is absent) | `fallback` | same |
| `now - est["t"] > watchdog_s` or gap between `step()` calls `> watchdog_s` | `fallback` | same |
| distance from ENU origin `> geofence_m` | `geofence` | zeros (PID would steer further out) |
| non-finite policy output | `zero` | zeros |
| otherwise | `policy` | model action |

- CONTRACT §4.4 `gps_valid` / `sensor_ok` semantics are used; the legacy `has_fix` of
  `pi-control/src/sensors/gps.py` is not read anywhere (INSPECTION H9: `has_fix` can be
  false while RMC is valid, and GGA/RMC double-update in H8).
- Clip `±safety.clip` (1.0) then slew `|Δa| <= rate_limit_per_s · dt` (measured `dt`,
  falling back to `policy_dt_s` when `dt == 0`). The fallback path is slewed too, so a
  handover is not a step; geofence/fault also `reset()` the K=10 history so a stale
  buffer is never replayed after recovery.
- Origin `lat0/lon0` latches on the first valid fix (CONTRACT §1) and is cleared by
  `reset()`. Defaults: clip 1.0, rate 2.0/s, geofence 80 m, gps timeout 2.0 s, watchdog
  0.5 s, overridable by `policy_meta.json.safety` and the `safety` kwarg.

## Verification (one-shot smoke script, not in `tests/`)

Dummy ONNX MLP (150->8 tanh->3 tanh, opset 17) written with `onnx.helper`, injected path +
meta, fake clock. Asserted, all passing:

- `raw_sample` matches hand-computed body frame for a known pose (goal 10 m east at
  `psi=0` -> `dy=+10`), `psi_err=90°` -> `sin=1, cos=0`; wp1/wp2 equal
  `frames.goal_in_body`; normalization divisor + ±5 clip.
- History: 10 identical slots after first build, one-slot shift on the next, `reset()`
  reproduces the tiled first sample, random `B=7` batch gives float32 `[7,150]` whose last
  15 values equal the float64 formula.
- Controller: valid step -> `source="policy"`, `|a|<=1`; sliding goal changes the command;
  consecutive commands respect the 0.1/step slew; `gps_valid=False` -> `fault="gps"`,
  fallback called, no jump; explicit `sensor_ok=False` -> `fault="sensor"`; stale `t` ->
  `fault="watchdog"`; 200 m > 80 m geofence -> `source="geofence"`; NaN model output ->
  `fault="nan"`; `RLController()` without a model raises `ValueError`; station keeping +
  `heading_deg=None` -> `sin/cos = 0/1`, `mode=1`, next-waypoint components non-zero.
- `torch not in sys.modules` after importing both runtime modules.

`pytest tests -q`: **58 passed, 1 failed** — the failure is
`tests/test_pid_parity.py::test_station_keeping_multistep_parity`, owned by A3 (running in
parallel), not touched here. No test file was created by this agent.

## TODO

1. **No trained ONNX model exists**: accuracy/behaviour on the water is untested.
   The dummy-model path only proves the plumbing. Parity torch-vs-ORT < 1e-4 is C3's job
   (`tests/test_onnx_parity.py`), obs parity numpy-vs-torch is B1's (`test_obs_parity.py`).
2. **Navigator wiring**: `next_lat/next_lon`, `next2_lat/next2_lon` are not produced by
   `pi-control/src/navigation.py` today. Until they are, the policy sees `wp1/wp2 = 0`
   whenever it walks a multi-waypoint route (train/deploy mismatch). `infer_replay.py` and
   the Pi glue must pass them.
3. **`(u, v)` quality** (CONTRACT T4, INSPECTION H7): SOG/COG at 0.3 m/s threshold is
   noisy; the 1 s decay constant is a placeholder. Needs a logged run to tune
   `min_speed_mps` / `decay_tau_s`; exposed as constructor kwargs for that reason.
4. **Geofence origin**: ENU origin = first valid fix, so the fence is relative to the
   launch point, not a surveyed site. Same question as CONTRACT T6 (fixed vs site origin).
5. **Geofence action**: zeros, not PID fallback — deliberate, but if the human prefers
   "return to origin" behaviour the policy/PID needs a homing goal instead.
6. **Loop rate**: real loop is 20 Hz (INSPECTION §3), policy `policy_dt_s = 0.1`; the
   caller must decimate. `step()` uses the measured `dt` for the estimator and slew, so a
   caller that does not decimate silently changes the effective rate limit.
7. **`configs/env.yaml` does not exist yet**, so `DEFAULT_OFFSET/SCALE` are the CONTRACT
   §3.4 literals; once B1 writes the file, pass the trained values via
   `policy_meta.json` (`obs_offset`, `obs_scale`, `obs_clip_norm`).
8. **`sensor_ok` key** is not in the §4.8 table (only `imu_valid` / `mag_valid`); the
   controller accepts an explicit `sensor_ok` and otherwise derives it. Suggest adding the
   key to CONTRACT §4.8 at the next contract edit.
