# PROGRESS.md — DRL ASV controller

Orchestrator log. One section per wave. Tests are the real command output, never estimates.

Test command used for every gate (run from `E:\buoy-project\drl`, bash):

```
py -3.13 -m pytest tests -q
```

(`python -m pytest` form puts `drl/` on `sys.path`, so `import common.frames` works.
`python3` does not exist on this host; `python` is 3.14, the project uses the `py -3.13`
launcher.)

---

## Wave 0 — Inspector/Architect (A0) — DONE 2026-10-26

Agent: A0. Files created (all under `drl/`):

| File | Lines | Content |
|---|---|---|
| `INSPECTION.md` | 356 | measured facts about the existing vessel stack, every claim cited `file:line`; bugs H1–H12; environment §8; TODO register §9 |
| `CONTRACT.md` | 558 | frames, state/action layout, every yaml path + key, module signatures, observation layout (K=10), policy net spec, numpy↔torch parity requirement |
| `requirements.txt` | 43 | pinned training-side deps, marked present/broken/absent as measured |
| `common/frames.py` | 134 | lat/lon ↔ ENU + heading conversion, numpy only |
| `tests/test_frames.py` | 106 | round-trip tests |
| `{sim,envs,runtime,configs,evalkit,common,tests}/__init__.py` | — | package markers |

Test result: `py -3.13 -m pytest tests -q` → **17 passed in 0.13s**.

### Headline findings

- Heading: consumers use **compass degrees, 0 = north, clockwise positive**; producer
  (`magnetometer.py:251`) emits CCW `atan2(y,x)` then fudges with
  `MAGNETIC_DECLINATION_DEG = -1.5` and `HEADING_MOUNT_OFFSET_DEG = 155.0`
  (`config.py:30-31`). `drl/` convention = radians, 0 = north, CW positive.
- Loop rates: main control loop / fusion **20 Hz**, IMU 50 Hz, mag 15 Hz, GPS as-arrives
  (NMEA rate unconfigured), Pi→ESP32 20 Hz, ESP32 PWM 50 Hz, status 10 Hz, ESP32
  failsafe 2000 ms.
- UART: `{"s":..,"sw":..,"y":..,"ts":..}\n` at 115200 8N1 on UART2 (GPIO16/17); ESP32
  replies `{"ok":1,"fs":0,"m":[1500,1500,1500,1500]}` with PWM µs in order NW, NE, SW, SE.
- Mixing matrix authority `config.h:85-90`: NW `+,+,+` / NE `+,-,-` / SW `+,-,+` /
  SE `+,+,-` (rows = motor, cols = surge, sway, yaw), **plus** `MOTOR_INVERT[NE] = true`
  applied after mixing (`config.h:100-105`) — a sim using the matrix alone will not match
  the vessel.
- GPS→local: no persistent local frame exists on the Pi; only ad-hoc equirectangular
  offsets (`navigation.py:209-212`, `geo.py:82-95`, R = 6 371 000). ENU origin is new
  work for `runtime/` (A4).

### Bugs reported (not fixed — existing repo is read-only)

- **H1** heading rotation sense self-contradictory (mag CCW vs gyro sign vs compass
  consumers) — highest impact, unresolved without a hand-rotation test.
- **H6** sway/yaw geometry cannot be reconciled with the layout comment; `+yaw` drives
  the port pair → CCW torque, opposite of the documented `yaw CW (+)`. Needs per-axis
  bench test with props removed. Blocks the sign of the sim's thruster geometry map.
- H2 GPS course blended into possibly-opposite-convention heading.
- H3 `HEADING_MOUNT_OFFSET_DEG = 155.0` is a fudge (comment says 180°), and
  `calibrate_mag.py` omits it entirely.
- H4 deadband mismatch: Pi `THRUST_DEADZONE = 0.05` per axis vs ESP32 0.03 per mixed motor.
- H5 failsafe documented 1 s on the Pi, actually 2000 ms in firmware.
- H7 no `(u, v)` body velocity exists on the real vessel — RL observation needs a new
  estimator (A4 owns).
- H8 GPS double-update (GGA + RMC both bump the timestamp → LPF applied twice per epoch).
- H9 `has_fix` can be false while RMC is valid (only GGA sets `fix_quality`).
- H10 first-fix sentinel is `latitude == 0.0` (broken on the equator).
- H11 no angle-wrap bug in `_blend_angle` / `heading_error`; station-keeping PID sign is
  correct; `set_heading_mode` does not reset `_prev_error` → derivative kick.
- H12 `main.py:345` `loop.add_signal_handler` raises on Windows (matters if
  `baseline_pid.py` imports it).

### Environment (measured)

| Package | State |
|---|---|
| numpy 2.2.6, PyYAML 6.0.3, pytest 9.1.1 | OK for `py -3.13` (3.13.3) |
| torch | **BROKEN** — `ImportError: DLL load failed while importing _C`; partial `2.6.0+cu124` tree, no dist-info, `torch_cpu.dll`/`torch_cuda.dll` missing, leftovers `~orch`, `~orchgen`, `~unctorch`, `~orch-2.9.1.dist-info` |
| onnx, onnxruntime, tensorboard, matplotlib | absent |
| rsl_rl | absent → `train.py` checks the installed API, else compact PPO (CONTRACT §3.5) |
| gymnasium | absent and not needed (plain vectorized torch env) |
| GPU | RTX 3050, 6144 MiB total, **5069 MiB free**, driver 596.49 |

### Open items / TODO

1. **Wave 0 GATE: human reviews `INSPECTION.md` and `CONTRACT.md` before Wave 1.**
2. **torch must be repaired before Wave 1 A1** (`sim/dynamics.py` is torch/CUDA).
   Proposed, not executed: venv inside the project (`drl/.venv`) +
   `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
   (+ `onnx`, `onnxruntime`, `tensorboard`, `matplotlib`). ~2.5–3 GB download. Awaiting
   the human's decision — nothing was installed by Wave 0.
3. Physical parameters (mass, `Izz`, added mass, drag, max thrust fwd/rev, thruster
   positions/orientations, motor lag): no measurement exists anywhere in the repo — all
   TODO placeholders in `configs/vehicle.yaml`, experiments in `SYSID.md` (A1).
4. Unverified conventions that must not be frozen as truth: north-referenced heading +
   real mag/IMU mounting rotation (H1–H3), hull sense of yaw/sway (H6), NEO-M8N NMEA
   output rate, declination for the operating site.

**Status: Wave 0 complete, stopped at the gate.** Nothing claimed about real-vessel
behaviour; all numbers above are from the code and from commands run on the dev box.
