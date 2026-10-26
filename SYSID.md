# SYSID.md — identification plan for every `TODO SYSID` placeholder

No physical quantity of the ASV has been measured. Every number currently in
`configs/vehicle.yaml`, `configs/disturbance.yaml` and `configs/sensors.yaml` is a
placeholder, except the values cited as measured facts (mixing matrix, `MOTOR_INVERT`,
`MOTOR_DEADZONE`, Pi loop rates, fusion weights). Simulator output is never identification
evidence.

**Status: nothing measured.** No experiment below has been executed.

## 0. Status register

| yaml key | placeholder | experiment | status |
|---|---|---|---|
| `mass_kg` | 12.0 | B1 | NOT MEASURED |
| `inertia_izz_kgm2` | 1.2 | B2 | NOT MEASURED |
| `added_mass.x` | 3.0 | B3 | NOT MEASURED |
| `added_mass.y` | 8.0 | B3 | NOT MEASURED |
| `added_mass.n` | 0.6 | B3 | NOT MEASURED |
| `drag_linear.x` | 6.0 | B4 | NOT MEASURED |
| `drag_linear.y` | 18.0 | B4 | NOT MEASURED |
| `drag_linear.n` | 1.5 | B4 | NOT MEASURED |
| `drag_quadratic.x` | 4.0 | B4 | NOT MEASURED |
| `drag_quadratic.y` | 12.0 | B4 | NOT MEASURED |
| `drag_quadratic.n` | 0.8 | B4 | NOT MEASURED |
| `thrusters.positions_m.*` | ±0.45 / ±0.30 | B7 (geometry) | NOT MEASURED |
| `thrusters.thrust_dir.*` | ±0.707 diagonals | B7 (H6) | NOT MEASURED |
| `max_thrust_fwd_n` | 9.8 | B5 | NOT MEASURED |
| `max_thrust_rev_n` | 4.9 | B5 | NOT MEASURED |
| `lag_tau_s` | 0.12 | B6 | NOT MEASURED |
| `signs.sway_sign`, `signs.yaw_sign` | 1.0 | B7 | NOT MEASURED |
| `signs.motor_sign.*` | 1.0 | B5 / B8 | NOT MEASURED |
| `signs.thrust_scale_fwd/rev.*` | 1.0 / 0.5 | B5 | NOT MEASURED |
| `max_surge_mps` | 1.0 | B4 acceptance only | POLICY VALUE, not physical |
| `wind.cd_area_m2`, `wind.max_speed_mps` | 0.35 / 8.0 | B12 | NOT MEASURED |
| `current.speed_mps`, `current.direction_deg`, `current.drag_linear` | 0.25 / 0 / 6.0 | B11 | NOT MEASURED |
| `waves.freq_hz`, `waves.amp_n`, `waves.yaw_amp_nm` | 4 components | B13 | NOT MEASURED |
| `gps.pos_noise_m`, `vel_noise_mps`, `cog_noise_deg`, `latency_s`, `rate_hz` | 0.4 / 0.05 / 3.0 / 0.2 / 5.0 | B9 | NOT MEASURED |
| `imu.gyro_noise_dps`, `imu.gyro_bias_dps`, `imu.accel_noise_mps2` | 0.3 / 0.2 / 0.05 | B10 | NOT MEASURED |
| `mag.noise_deg`, `mag.bias_deg` | 2.0 / 5.0 | B10 | NOT MEASURED |
| `gps.origin_lat/lon` | null | B9 | NOT MEASURED (site) |

## 1. Safety envelope

1. **Props off, battery disconnected, for anything that moves the hull by hand** (B1–B4,
   B7, B11, B12). The firmware watchdog is 2000 ms (`config.h`), which is longer than a
   person takes to reach into a prop guard.
2. **Two-person rule** at the tank: one on the transmitter/kill switch, one at the hull.
3. **Never run a motor dry** for more than a few seconds (B5, B6, B8): bearings and ESC
   heat. Submerge the prop or remove it and use the stand.
4. Tether the hull (float line, slack ≥ 2× hull length) for every powered on-water run.
5. Record battery voltage before/after each run: thrust scales with voltage.

## 2. Experiments

Each experiment: raw file name, operator, date and the reduction formula. A value becomes
"measured" only when all four exist (§4).

### B1 — mass and CG
Rig: two scales (or one scale + a fulcrum), plumb line, tape measure.
1. Weigh the hull complete (battery, Pi, ESP32, sensors). → `mass_kg`
2. Find the CG: rest the hull on a knife edge along x, mark balance line; repeat along y.
3. Record CG offset from the geometric centre; `thrusters.positions_m` is measured **from
   the CG**, not from the hull centre.
4. Repeat with 10 % ballast to bound the range used by `randomization.mass`.
Reduction: none, direct. Acceptance: two weighings within 1 %.
Order of magnitude GUESS only: 8–20 kg.

### B2 — yaw inertia `Izz`
Rig: suspend the hull from the ceiling by two parallel lines (bifilar), length `L`,
separation `d`; measure the yaw period `T` of small oscillations, ~10 cycles.
Formula: `Izz = m * g * d^2 * T^2 / (16 * pi^2 * L)`.
Check: repeat with `d/2`; `Izz` must scale as `d^2`.
Order of magnitude GUESS only: 0.5–3 kg·m².

### B3 — added mass (`added_mass.x/y/n`)
Rig: the B2 bifilar for yaw; a linear rail or a long still-water tank for surge/sway.
1. Displace the hull, release, log free-decay oscillation period `T_n` per axis.
2. `M_eff = k / (2*pi/T_n)^2` with `k` the known spring (bifilar) or the B4 linear drag
   slope for the tank test.
3. `added_mass = M_eff - mass_kg` (or `Izz` for the n axis).
Cheap fallback: strip the appendages, repeat B4 drag runs, and fit the added mass from
the acceleration transient. Label the result `estimate` and keep the TODO comment until a
second method agrees within 30 %.

### B4 — linear and quadratic drag
Rig: constant-speed tow behind a rowing boat or a winch, load cell in line, GPS for speed;
or free-decay runs (motors off) with the IMU/GPS at 20 Hz.
1. Tow at ≥ 8 speeds from 0.1 to 1.2 m/s forward, repeat in reverse, and in pure sway
   (crab tow) if the tank allows.
2. Free decay: push to 0.8 m/s, release, log `u(t)`; fit `du/dt = -(c1*u + c2*u^2)/M_eff`.
3. Least squares on `F(v) = c1*v + c2*v^2` (tows) → `drag_linear.*`, `drag_quadratic.*`.
4. Repeat with 10 % mass ballast to check the fit is physical, not an artefact.
Acceptance: R² > 0.95 on the tow fit, and the decay fit predicts the tow fit within 20 %.
Order of magnitude GUESS only: `c1` 2–20 N/(m/s), `c2` 1–20 N/(m/s)².

### B5 — bollard pull per motor, forward and reverse
Rig: hull tied to a fixed point, spring scale or load cell on the bow line; one motor at a
time on a thrust stand if available. Props free and submerged.
1. Sweep PWM command 1400 → 1900 µs in steps of 50 µs, forward and reverse, 5 s per step,
   log the steady force.
2. `max_thrust_fwd_n` = force at full forward, `max_thrust_rev_n` = |force| at full reverse.
3. `signs.thrust_scale_fwd/rev[motor]` = `F_motor / F_max` at the same command, so the four
   motors' relative strengths are captured even before the absolute value is trusted.
4. Record the per-motor deadband: the first command step that produces motion, converted to
   the `[-1, 1]` scale. Compare with `MOTOR_DEADZONE = 0.03` (`config.h:110`) — that firmware
   value is measured, this experiment only confirms it is not masked by stiction.
Order of magnitude GUESS only: 5–15 N forward, 2–8 N reverse.

### B6 — motor lag `lag_tau_s`
Rig: B5 stand, load cell sampled at ≥ 50 Hz.
1. Step the command 0 → 80 % and hold; log the force.
2. Fit a first-order rise; `lag_tau_s` = time to reach 63 % of the final value.
3. Repeat forward and reverse; use the slower (conservative) value. Repeat at two
   battery voltages to see whether the lag is electrical (ESC) or hydrodynamic.
Order of magnitude GUESS only: 0.05–0.3 s.

### B7 — per-axis geometry (INSPECTION H6) and `thrust_dir`
Props removed for the first pass (hull motion only, no thrust asymmetry), then props on,
submerged, hull tethered in a still tank.
1. Command pure surge `[1,0,0]`: record which way the hull translates (should be bow-first).
   If it is not, `signs.motor_sign` or a `thrust_dir` x sign is wrong.
2. Command pure sway `[0,1,0]`: record starboard/port translation and any yaw.
3. Command pure yaw `[0,0,1]`: record the rotation direction (CW positive per CONTRACT §1).
   This is H6 — the documentation claims `+yaw` is CW and the layout comment suggests the
   port pair, which is the opposite.
4. Only after the three observations: set `signs.sway_sign`, `signs.yaw_sign` (a single
   `-1` where the sim is mirrored) and, if a motor pair clearly drives the wrong way,
   `signs.motor_sign`. **Never** edit `mix_matrix` or `motor_invert`: they are firmware
   values (`config.h:85-105`).
5. Record `positions_m` from the B1 CG, along with the measured prop shaft angles for
   `thrust_dir` (protractor on the shaft axis, hull level).
Constraint: `thrust_dir` must keep the pairing forced by the mixing matrix — sway row is
`+1` on NW,SE and `-1` on NE,SW, so NW must pair with SE and NE with SW in y sign.
Breaking that pairing makes a pure sway command sum to exactly zero force.

### B8 — does `MOTOR_INVERT[NE]` double-count? (open bench question)
`config.h:100-105` inverts NE *after* mixing. The sim applies the same inversion. If the
physical NE motor is already mounted with reversed spin (so "invert" is what makes it push
the same way as the others), the sim is right. If the inversion exists to counteract a
reversed prop, the sim double-counts and NE's force direction is wrong.
1. On the B5 stand, command pure surge `[1,0,0]`.
2. Measure NE's force direction on the stand.
3. If NE pushes **forward**, `MOTOR_INVERT` compensates correctly and `signs.motor_sign.NE`
   stays `1.0`. If it pushes **aft**, set `signs.motor_sign.NE = -1.0` and record it here.
Do not change this without the measurement: it silently mirrors one corner of the vehicle.

### B9 — GPS noise, rate, latency, site origin
Rig: receiver on the stationary hull, 15 min static log; then a straight constant-speed
tow (or a slow run) for latency.
1. Static log → `pos_noise_m` = 1σ of the ENU residuals about the mean; `vel_noise_mps` and
   `cog_noise_deg` = 1σ at rest (SOG/COG are meaningless at rest, note it).
2. Rate: count NMEA epochs per second as configured on the receiver → `rate_hz`.
3. Latency: compare the GPS speed step with the IMU/load-cell step in the same run →
   `latency_s`.
4. `origin_lat/lon` = the site's mean fix (or keep null in sim, which falls back to 0.0).
Also confirm the heading convention of COG (0 = north, CW) with one known course.

### B10 — IMU and magnetometer noise/bias
Rig: hull stationary, level, away from motors and the Pi's switching supply; 10 min log at
the vessel's real rates (IMU 50 Hz, mag 15 Hz).
1. `gyro_noise_dps` = 1σ of the high-passed rate; `gyro_bias_dps` = max |mean| over several
   power cycles (bias changes with temperature) → also the episode draw range.
2. `mag.noise_deg` = 1σ of `atan2` heading samples; `mag.bias_deg` = deviation from the
   surveyed north after hard/soft-iron calibration, i.e. the residual.
3. Log the same run with the motors at 50 % to bound the magnetic disturbance of the ESCs.
Note: these are the values the sim randomises; the real values behind H1/H3 (`155°` mount
offset, `-1.5°` declination) must be re-derived here, not copied.

### B11 — current (`current.speed_mps`, `direction_deg`, `current.drag_linear`)
Rig: hull in the water, motors off, GPS logging; a drifting drogue as a cross-check.
1. Release the hull, log the GPS track for 5 min; the drift velocity vector is the current.
   `speed_mps` = |v|, `direction_deg` = the compass direction it moves **toward**.
2. `drag_linear` here is the hull drag used to convert current to force — reuse the B4
   value; if the site differs from the tow site, say so.
3. Repeat at two states of tide if the site is tidal; the curriculum should sample the
   larger value.

### B12 — wind drag area (`wind.cd_area_m2`, `wind.max_speed_mps`)
1. `cd_area_m2 = Cd * A` with `A` from hull + superstructure silhouette photos (front and
   side), `Cd ≈ 1.0–1.2` for a blunt box — state the assumed `Cd` explicitly.
2. Cross-check by towing the hull in a crosswind with an anemometer: fit the force residual
   not explained by B4.
3. `max_speed_mps` = the strongest wind the vessel will ever be deployed in (operational
   decision, not physics).
Force model in the sim is `level^2 * 0.5 * rho * cd_area * |V_rel| * V_rel` — the level
factor is curriculum, not a physical value.

### B13 — wave force amplitudes
Rig: hull moored, load cell on the mooring line, IMU; 20 min record in the target sea state.
1. Welch PSD of the surge/sway force and yaw moment → peak frequencies → `waves.freq_hz`.
2. Integrate the PSD in bands around each peak → force RMS per band → `waves.amp_n`,
   `waves.yaw_amp_nm`.
3. Record the sea state (Beaufort / wave height) next to the numbers: the values are only
   valid for it.
Cap: the sim sums sinusoids; `n_components = 4` is a modelling choice, keep it unless the
PSD shows more than 4 resolvable peaks.

## 3. Order of work

1. B1, B2 (dry, no power) — unblocks the whole dynamics model.
2. B5, B6, B8 (bench, props on a stand) — unblocks thrusters and the NE question.
3. B7 (tank) — unblocks the signs, H6.
4. B4, B3 (tank/tow) — unblocks drag and added mass.
5. B9, B10 (static logs) — unblocks the sensor model; cheapest of all, can run in parallel.
6. B11, B12, B13 (site) — last; only needed before curriculum levels > 0.

## 4. Acceptance rules

- A yaml key may lose its `TODO SYSID` comment only when `reports/SYSID/<experiment>.md`
  holds: date, operator, raw log path, the reduction script or the arithmetic, and the
  resulting value with its uncertainty.
- Two independent methods within 30 %: keep the conservative (larger drag / smaller thrust)
  value and record both.
- Sim results never count. A tuned parameter that makes the sim "look right" is not a
  measurement.
- Every flip of `signs.*` needs its B7/B8 observation written down next to the value.

## 5. Results template (copy into the configs when measured)

```yaml
mass_kg: TODO            # B1
inertia_izz_kgm2: TODO   # B2
added_mass: {x: TODO, y: TODO, n: TODO}                 # B3
drag_linear: {x: TODO, y: TODO, n: TODO}                # B4
drag_quadratic: {x: TODO, y: TODO, n: TODO}             # B4
thrusters:
  max_thrust_fwd_n: TODO  # B5
  max_thrust_rev_n: TODO  # B5
  lag_tau_s: TODO         # B6
  deadband: 0.03          # config.h:110, measured; B5 only confirms
signs:
  sway_sign: TODO         # B7
  yaw_sign: TODO          # B7
  motor_sign: {NW: TODO, NE: TODO, SW: TODO, SE: TODO}  # B5 + B8
  thrust_scale_fwd: {NW: TODO, NE: TODO, SW: TODO, SE: TODO}   # B5
  thrust_scale_rev: {NW: TODO, NE: TODO, SW: TODO, SE: TODO}   # B5
```
