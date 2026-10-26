# INSPECTION.md — existing vessel stack, measured

Wave 0 reconnaissance for `drl/`. Every claim below cites `file:line` in the existing
repo (read-only: nothing outside `drl/` was modified). Anything not derivable from
the code is marked **TODO**.

Repo roots inspected: `E:\buoy-project\pi-control` (Python, Pi 3B) and
`E:\buoy-project\firmware-esp` (C++/PlatformIO, ESP32). Runtime log sampled:
`E:\buoy-project\asv.log`.

---

## 1. Heading convention

| Fact | Evidence |
|---|---|
| Fused heading is a `float` documented as **degrees [0, 360)** | `pi-control/src/sensors/fusion.py:39` |
| Primary heading source is the magnetometer, computed as `atan2(y, x)` in degrees | `pi-control/src/sensors/magnetometer.py:251-259` |
| Then `+= MAGNETIC_DECLINATION_DEG`, `+= HEADING_MOUNT_OFFSET_DEG`, `% 360` | `pi-control/src/sensors/magnetometer.py:255-259` |
| `MAGNETIC_DECLINATION_DEG = -1.5`, `HEADING_MOUNT_OFFSET_DEG = 155.0` ("Sensor mounted 180° from bow — N↔S swap fix") | `pi-control/src/config.py:30-31` |
| Gyro yaw rate is **deg/s**, name says "yaw rate (important for heading PID)" | `pi-control/src/sensors/imu.py:48-52`, `fusion.py:43` |
| Compass-style use elsewhere: `bearing()` returns degrees **0 = north, clockwise positive** | `pi-control/src/utils/geo.py:33-50` (formula `atan2(sin Δλ cos φ2, …)`, `+360 % 360`) |
| `heading_error()` = `(desired - current + 180) % 360 - 180`, doc: "Positive = turn right" | `pi-control/src/utils/geo.py:98-109` |
| Body rotation assumes compass heading: `surge = N cosψ + E sinψ`, `sway = -N sinψ + E cosψ` | `pi-control/src/control/thrust.py:164-167` |
| `yaw` documented as "Clockwise (+) / counter-clockwise (-)" | `pi-control/src/control/thrust.py:45`, `firmware-esp/src/lib/motor_mixer.h:28` |
| Gyro Z is added to heading **without sign flip** | `pi-control/src/sensors/fusion.py:116-117`, `:140-141` |

**Conclusion.** The *consumers* (navigation, PID, body-frame rotation, GPS course,
`geo.bearing`) treat heading as a **compass bearing: degrees, 0 = north, clockwise
positive** (i.e. a port/starboard yaw angle). The *producer* (magnetometer) emits
`atan2(y, x)` — a **counter-clockwise** mathematical angle from the sensor's +X axis —
and then fudges it into the compass frame with two hand-tuned constants
(declination −1.5°, mount offset 155.0°). So:

- The **contract convention used by drl/ is: heading ψ in radians, 0 = north, positive
  clockwise, `ψ = deg2rad(compass_heading_deg)`** (see `CONTRACT.md`, `common/frames.py`).
- The 155° mount offset is **not** a documented sensor orientation; the comment says
  180°. It is a bench fudge factor. **TODO**: measure the actual mag board axes and
  replace both constants with a real rotation.
- Whether the fused angle really is "0 = north" is unverified without a compass swing
  test. Treat north-referenced heading as **TODO / to be confirmed on the water**.

### How fusion.py fuses mag + IMU (+ GPS course)

`fusion.py:99-146`, at `FUSION_UPDATE_RATE_HZ = 20` (`config.py:36`):

1. On the first valid mag sample, heading snaps to the mag value, and the internal
   `_gyro_heading` integrator is seeded from it (`fusion.py:108-112`).
2. If mag and IMU both valid: `_gyro_heading += gyro_z * dt` (`:116-117`), then a
   shortest-path blend toward the mag heading with `FUSION_MAG_WEIGHT = 0.02`
   (`:120-122`, weights `0.98/0.02` at `config.py:38-39`), then `_gyro_heading` is
   **resynchronised to the fused value** (`:131`).
3. If GPS is valid and `speed_over_ground > 0.3 m/s` (`GPS_HEADING_MIN_SPEED`,
   `fusion.py:54`), the fused heading is blended 5 % toward `gps.data.true_course`
   (`:125-127`).
4. Mag-only: heading = mag value (`:133-136`). Gyro-only: pure integration, logged as
   "will drift", rate-limited warning (`:138-146`). **Both invalid: heading is silently
   frozen** — no branch, no flag beyond `imu_valid`/`mag_valid`.
5. `_blend_angle()` (`fusion.py:84-90`) does implement shortest-path wrapping
   (`((target-base+180)%360)-180`) — this one is correct.

## 2. Units

SI in the navigation/control layers, degrees for all angles, knots only at the NMEA
boundary:

| Quantity | Unit | Evidence |
|---|---|---|
| latitude / longitude | degrees (WGS84) | `gps.py:33-34`, `fusion.py:37-38` |
| speed over ground | m/s, converted `knots * 0.514444` | `gps.py:168` |
| true course / heading / roll / pitch | degrees | `gps.py:169`, `fusion.py:39-42` |
| yaw rate `gyro_z` | **deg/s** (LSB ÷ 131.0 for ±250 °/s) | `imu.py:34`, `imu.py:48-50` |
| accelerometer | m/s² (`g * 9.81`) | `imu.py:211-213` |
| position error | metres (flat-earth, R = 6 371 000 m) | `navigation.py:211-212`, `geo.py:10` |
| thruster commands | normalized `[-1, 1]`, dimensionless | `thrust.py:43-45`, `motor_mixer.h:26-28` |
| PWM | microseconds, 1000 / 1500 / 2000 µs (rev / neutral / fwd) | `config.h:66-68` |
| horizon / dt | seconds | `pid.py:78-81` |

`drl/` uses SI (m, s, rad, N, kg, N·m) with angles in radians internally; degrees only
at the Pi/ONNX boundary.

## 3. Control loop rates (code + log)

| Loop | Rate | Evidence |
|---|---|---|
| Main control loop (nav → PID → UART) | **20 Hz** (`FUSION_UPDATE_RATE_HZ`) | `main.py:93`, `main.py:154-161`; log `asv.log:52` "Control loop started at 20Hz" |
| Sensor fusion update | 20 Hz | `config.py:36`, `fusion.py:183`; `asv.log:51` |
| IMU (MPU6050) sample/read | **50 Hz** | `config.py:24`, `imu.py:254`; `asv.log:48` "IMU read_loop started at 50Hz" |
| Magnetometer (QMC5883P) read | **15 Hz** | `config.py:29`, `magnetometer.py:289`; `asv.log:1652` |
| GPS | NMEA parsed whenever bytes arrive; read loop polls the pigpio bit-bang buffer every **50 ms** (`gps.py:136`). Actual fix rate is the module's NMEA output rate — **TODO** verify (NEO-M8N default 1 Hz; the code never sends UBX config). Staleness timeout 2.0 s (`config.py:17`, `gps.py:194-195`). |
| Pi → ESP32 UART frames | 20 Hz while active (`main.py:184`), else zero-thrust heartbeat every 0.5 s (`config.py:91`, `esp32_link.py:175-177`) |
| ESP32 PWM update | **50 Hz** (`PWM_UPDATE_MS = 20`) | `config.h:146`, `main.cpp:279-282` |
| ESP32 status → Pi | **10 Hz** (`STATUS_REPORT_MS = 100`) | `config.h:145`, `main.cpp:285-288` |
| ESP32 failsafe | **2000 ms** without a valid command | `config.h:134`, `main.cpp:269-277` |
| PID internal dt | wall clock, default 0.05 s, clamped `[0.001, 0.5]` | `pid.py:76-81` |

## 4. GPS lat/lon → local frame

**There is no persistent local frame and no entry point that converts lat/lon to
(x, y) state.** Two unrelated ad-hoc conversions exist:

1. `navigation._update_station_keeping()` (`navigation.py:209-212`) — flat earth,
   relative to the *hold point*: `north_m = dlat·π/180·6371000`,
   `east_m = dlon·π/180·6371000·cos(hold_lat)`. Never stored as a state; recomputed
   every cycle and immediately fed to the PID.
2. `geo.lat_lon_offset()` (`geo.py:82-95`) — the same equirectangular projection, plus
   `haversine_distance()` / `bearing()` (spherical, `geo.py:13-50`) for waypoint ranges
   and bearings.

Waypoint navigation therefore works directly in lat/lon + bearing and never needs an
ENU frame. There is no origin, no x/y in `FusedState` (`fusion.py:34-47`), and no
velocity in the local frame.

**Consequence for drl/**: the ENU frame is a new concept. `drl/common/frames.py`
defines it with the *same* projection constants as the repo (`R = 6 371 000`,
`cos(lat0)`) so numbers agree with the Pi to ~1e-9 relative over a few km, but the
origin (`lat0, lon0` = first valid fix) and the (x, y) state must be added in
`runtime/` — **TODO** (owned by A4).

## 5. ESP32 UART message format (both directions)

Physical link: ESP32 **UART2, GPIO16 = RX2 / GPIO17 = TX2**, `SERIAL_BAUD = 115200 8N1`
(`serial_parser.cpp:42-48`, `config.h:129`); UART0 stays free for USB debug
(`serial_parser.h:5-7`). Pi side: `/dev/serial0` @ 115200 (`config.py:89-90`).

**Pi → ESP32** — one JSON object per line, `\n` terminated, ASCII, keys rounded to 3 dp
(`esp32_link.py:96-105`):

```
{"s":0.4,"sw":-0.2,"y":0.1,"ts":123456.789}\n
```

| key | meaning | range |
|---|---|---|
| `s` | surge, forward + | [-1, 1] |
| `sw` | sway, starboard + | [-1, 1] |
| `y` | yaw, clockwise + | [-1, 1] |
| `ts` | Pi unix timestamp | optional, echoed but unused |

ESP32 parser: `strstr()` per key, `strtof()`, requires `s`/`sw`/`y`, clamps to
`[-1, 1]`, keeps last valid frame on a bad line, 256-byte line buffer, overflow drops
the line (`serial_parser.cpp:50-110`).

**ESP32 → Pi** — `{"ok":1,"fs":0,"m":[1500,1500,1500,1500]}\n` at 10 Hz, `m` = **PWM
pulse widths in µs** per motor in order **NW, NE, SW, SE** (`serial_parser.cpp:112-124`,
`config.h:42`). Pi stores the last parsed dict in `esp32_status` and otherwise ignores
it (`esp32_link.py:129-144`).

## 6. Thruster mixing matrix (authority: `config.h`)

Layout comment (`config.h:8-24`): NW = front-left, NE = front-right, SW = rear-left,
SE = rear-right; rows = motor, columns = `[surge, sway, yaw]`.

Verbatim, `firmware-esp/src/header/config.h:85-90`:

```c
static constexpr float MIX_MATRIX[NUM_MOTORS][3] = {
    /* NW */ { +1.0f, +1.0f, +1.0f },
    /* NE */ { +1.0f, -1.0f, -1.0f },
    /* SW */ { +1.0f, -1.0f, +1.0f },
    /* SE */ { +1.0f, +1.0f, -1.0f },
};
```

Expanded (`config.h:80-83`): `T_NW = +surge +sway +yaw`, `T_NE = +surge -sway -yaw`,
`T_SW = +surge -sway +yaw`, `T_SE = +surge +sway -yaw`. Motor order
`enum MotorID { M_NW = 0, M_NE, M_SW, M_SE }` (`config.h:42`).

**Per-motor direction inversion**, `config.h:100-105` — applied *after* mixing, in
`thrustToPulse3D()` (`motor_mixer.cpp:301-303`):

```c
static constexpr bool MOTOR_INVERT[NUM_MOTORS] = {
    /* NW */ false,
    /* NE */ true,    // kanan atas (kanan depan) — putaran terbalik
    /* SW */ false,
    /* SE */ false,
};
```

so the **effective** motor thrust is `flip_i * (MIX_MATRIX[i] · [surge, sway, yaw])`
with `flip_NE = -1`. Mixing output is clamped to `[-1, 1]` per motor before conversion
(`motor_mixer.cpp:186-193`); `|thrust| <= MOTOR_DEADZONE = 0.03` maps to 1500 µs
neutral (`config.h:110`, `motor_mixer.cpp:292-294`); otherwise linear
`1500 ± thrust·500` µs (`motor_mixer.cpp:309-313`), with slew limiting 40 µs/cycle
(12 µs/cycle when leaving neutral) at 50 Hz (`config.h:114-119`).

`thrust.py:9-13` repeats the matrix as the Pi-side source of truth (comment only, no
code path).

## 7. Bugs and inconsistencies found (reported, not fixed)

**H1 — heading rotation sense is self-contradictory (high impact).**
`magnetometer.py:251` produces a CCW mathematical angle `atan2(y, x)`, while
`fusion.py:116-117` adds `gyro_z·dt` with the MPU6050 sign convention (Z up → CCW
positive for a right-hand sensor) and `geo.bearing()` / `thrust.py:164-167` /
`pid.py` heading mode consume a CW-from-north compass angle. Both cannot be true unless
the IMU is physically mounted upside-down (which would also invert `roll`,
`imu.py:221`). Worst case the complementary filter integrates yaw with the wrong sign
and the 2 % mag blend only drags it back slowly. **TODO**: verify with a hand-rotation
test before any sign is frozen in `sim/`.

**H2 — GPS course is blended into a heading of possibly opposite convention**
(`fusion.py:125-127`): `true_course` is CW from true north by NMEA definition
(`gps.py:169`); if the fused heading is CCW, the 5 % correction pulls the wrong way.

**H3 — heading offset constants are a fudge, and the calibration tool disagrees.**
`config.py:31` `HEADING_MOUNT_OFFSET_DEG = 155.0` with a comment claiming 180°.
`calibrate_mag.py:82-86` computes `atan2(y,x) + declination` and **omits**
`HEADING_MOUNT_OFFSET_DEG`, so the calibration UI displays a heading 155° away from the
value the controller uses. Declination sign (`-1.5`, `config.py:30`) is also unverified.

**H4 — deadband mismatch Pi vs firmware.** Pi zeroes each body axis below
`THRUST_DEADZONE = 0.05` (`config.py:84`, `thrust.py:113-118`); the ESP32 declares the
*whole* command idle when all three axes are `<= MOTOR_DEADZONE = 0.03`
(`motor_mixer.cpp:218-223`) and additionally snaps each individual *mixed* motor below
0.03 to neutral (`motor_mixer.cpp:292-294`). Consequence: small differential commands
(e.g. a policy output of 0.02 on three axes, or a fine station-keeping correction) are
either discarded on the Pi or silently become neutral on the ESP32, and per-motor
deadband eats the small mixed values after mixing.

**H5 — failsafe timing documented three ways.** Firmware `FAILSAFE_TIMEOUT_MS = 2000`
(`config.h:134`); Pi `ESP32_TIMEOUT_S = 1.0` with comment "ESP32 failsafe: no msg for
1s -> shutdown motor" (`config.py:92`); `esp32_link.py:7-8` docstring "RPi sends
commands at ~20Hz … nothing for 1s". The Pi constant is unused by the protocol; the
real deadline is 2 s. A Pi stall of >2 s kills the motors.

**H6 — sway/yaw geometry cannot be reconciled with the layout comment.** With the
documented layout, `+yaw` drives the *port* pair (NW, SW) forward, which produces a
bow-to-port (CCW, negative-compass) torque, the opposite of the `yaw CW (+)` doc in
`motor_mixer.h:28` and `thrust.py:45`; `+sway` drives a *diagonal* pair (NW, SE). Either
the physical thrust directions are not "forward = +x_body" per hull, or the sway/yaw
signs in the matrix are inverted relative to the docs. Unresolvable from code.
**TODO**: bench test each axis in isolation (surge only, sway only, yaw only, prop
removed) and record the resulting motion; this fixes the sign of the sim's thruster
geometry map. Note also that `MOTOR_INVERT[NE] = true` means the effective matrix row
for NE is negated — a sim that uses `MIX_MATRIX` alone will not match the vessel.

**H7 — no body velocity (u, v) exists on the real vessel.** `FusedState.speed` is GPS
SOG only (`fusion.py:159`); there is no `u`, no `v`, no course-vs-heading decomposition.
The RL observation requires `(u, v, r)`, so the Pi runtime needs a new estimator.
**TODO** (A4): `u = SOG·cos(COG − ψ)`, `v = SOG·sin(COG − ψ)` gated on
`SOG > 0.3 m/s`, else decay to 0. No such code exists today.

**H8 — GPS double-update.** Both GGA (`gps.py:160`) and RMC (`gps.py:170`) set
`data.timestamp`; `fusion.py:149` treats a changed timestamp as a *new sample* and
applies the position LPF (`alpha = 0.7`) twice per 1 Hz epoch.

**H9 — `has_fix` can be false with a valid RMC.** `GPSReader.has_fix` requires
`fix_quality > 0` (`gps.py:183-184`) but only GGA sets `fix_quality` (`gps.py:153`);
RMC alone sets `valid = True` with a stale/zero quality.

**H10 — first-fix sentinel is `latitude == 0.0`** (`fusion.py:152`): correct in
Indonesia, broken on the equator; also `state.latitude/longitude` are never invalidated
when GPS is lost (only `gps_valid` is), so `navigation.update()` gates on
`gps_valid` (`navigation.py:134-142`) — that part is handled.

**H11 — no angle-wrap bug found** in `_blend_angle` (`fusion.py:84-90`) or
`heading_error` (`geo.py:108`); both are correct. The station-keeping PID sign is
also correct: `navigation.py:211` passes *actual − hold* as the measurement to
`pid.compute(0.0, …)` (`thrust.py:161-162`), which yields the negated error and thus a
correct restoring command. `pid.py:100-108` applies a filtered derivative on the error
with no setpoint weighting, so a mode switch produces a derivative kick
(`_prev_error` is not reset by `set_heading_mode`).

**H12 — Windows incompatibility in existing code (informational).**
`main.py:345-346` uses `loop.add_signal_handler`, which raises on Windows; the Pi code
cannot be run on the dev box as-is (relevant only if `baseline_pid.py` tries to import
it).

---

## 8. ENVIRONMENT (measured on this host, 2026-10-26)

Commands run from bash. `python3` does not exist; `python` is 3.14; all checks use the
`py -3.13` launcher.

```
$ py -3.13 -V
Python 3.13.3
$ py -3.13 -c "import sys; print(sys.executable)"
C:\Users\AORUS\AppData\Local\Programs\Python\Python313\python.exe
$ py -3.13 -m pip --version
pip 25.0.1 from ...\Python313\Lib\site-packages\pip (python 3.13)
```

| Package | Status | Notes |
|---|---|---|
| numpy | **2.2.6** | `importlib.metadata` |
| PyYAML | **6.0.3** | |
| pytest | **9.1.1** | |
| torch | **BROKEN / not registered** | `pip show torch` → `Package(s) not found: torch`; `import torch` → `ImportError: DLL load failed while importing _C: The specified module could not be found.` |
| onnx | absent | |
| onnxruntime | absent | |
| rsl_rl | absent | API must be checked at install time (Wave B2) |
| gymnasium | absent | |
| tensorboard | absent | |

**torch detail (measured, not inferred):**

- `torch/version.py` says `__version__ = '2.6.0+cu124'`, `cuda = '12.4'`, dated Nov 22 2025.
- No `torch-*.dist-info` exists for the real tree; leftovers present:
  `~orch`, `~orchgen`, `~unctorch`, `~orch-2.9.1.dist-info` (pip warns
  `Ignoring invalid distribution ~orch`).
- `torch/lib` has **30 entries** and contains `c10.dll`, `c10_cuda.dll`,
  `cublas64_12.dll`, `cudnn64_9.dll`, … but **`torch_cpu.dll` and `torch_cuda.dll` are
  missing**; `torch/_C.cp313-win_amd64.pyd` is present, so the pyd loads but its
  dependencies cannot be resolved.
- `torchvision-0.24.1.dist-info` is also present (0.24 pairs with torch 2.9, not 2.6),
  and `import torchvision` fails with the same `_C` error.
- Total `site-packages/torch` size: 2.5 GB.

**What repair would take** (NOT executed — installation is the orchestrator's/human's
decision):

1. Delete the leftover invalid dirs `~orch`, `~orchgen`, `~unctorch`,
   `~orch-2.9.1.dist-info` (and `torch/` + `torchvision*` for a clean slate).
2. `py -3.13 -m pip install --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
   (or a single consistent pair, e.g. `torch==2.9.1 torchvision==0.24.1` from the
   matching cu12x index). Mixing 2.6 files with a 2.9 metadata/vision tree is what
   produced the current state.
3. Verify: `py -3.13 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"`
   must print `2.6.0+cu124 True 12.4` (or the 2.9 equivalent).
4. Windows note: the cu124/cu126 wheels bundle their own CUDA runtime; only the NVIDIA
   driver is required (present: 596.49).

**GPU**

```
$ nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv
name, memory.total [MiB], memory.used [MiB], memory.free [MiB], driver_version
NVIDIA GeForce RTX 3050, 6144 MiB, 934 MiB, 5069 MiB, 596.49
```

Budget for training: **~5.0 GB free of 6.0 GB**; that constrains `train.py`
(thousands of envs on a 3-DOF state of ~20 floats/env is fine; the MLP is tiny — the
memory risk is PPO rollout buffers, so keep `num_envs × rollout_steps × obs_dim` in
float32 under ~1 GB).

---

## 9. TODO summary (unknown / unverifiable from the repo)

1. Whether the fused heading is truly north-referenced, and the real mag/IMU mounting
   rotation (H1–H3). Needs a compass-swing + hand-rotation test.
2. Sign of yaw and sway as felt by the hull (H6). Needs a per-axis bench test with
   props removed, plus one in-water low-throttle test.
3. GPS NMEA output rate actually configured on the NEO-M8N (code assumes "as they
   arrive").
4. All physical parameters (mass, Izz, added mass, drag, max thrust fwd/rev, thruster
   positions/orientations, motor lag) — no measurement exists anywhere in the repo
   (`config.py` only has control gains and `WAYPOINT_MAX_SPEED = 0.55 m/s`, which the
   comment attributes to a logged peak of 0.69 m/s at surge 1.0,
   `config.py:64-67`).
5. `u, v` estimator on the Pi (H7) — does not exist.
6. Declination sign/value for the actual operating site.
