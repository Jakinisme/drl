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

---

## Environment setup — drl/.venv — DONE 2026-10-26

Decision: project-local venv (nothing installed globally, the broken global torch tree
is left untouched and still unused). Every python/pytest run from here on uses
`drl\.venv\Scripts\python.exe`. Full pip output: `drl/install.log` (gitignored, with
`.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`).

```
py -3.13 -m venv .venv                                  # Python 3.13.3, pip 26.2.2
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install numpy==2.2.6 PyYAML==6.0.3 pytest==9.1.1 \
    onnxruntime==1.20.1 tensorboard==2.18.0 matplotlib==3.9.2
.venv\Scripts\python.exe -m pip install --only-binary=:all: onnx
```

Verified:

```
torch 2.6.0+cu124  cuda True  cuda_ver 12.4  gpu NVIDIA GeForce RTX 3050
numpy 2.2.6  yaml 6.0.3  pytest 9.1.1
onnx 1.23.2  ort 1.20.1  tb 2.18.0  mpl 3.9.2
cuda tensor op: (torch.zeros(4, device='cuda')+1).sum() -> 4.0
.venv\Scripts\python.exe -m pytest tests -q -> 17 passed
```

Two deviations from the literal instruction, both forced:

1. **`onnx==1.17.0` does not exist for cp313** — no wheel, pip fell back to a source
   build that died at `CMake Error: CMAKE_C_COMPILER not set`. Installed the latest
   binary wheel instead: **onnx 1.23.2**.
2. **rsl-rl-lib NOT installed.** `pip install --dry-run rsl-rl-lib` resolves to
   `torch-2.14.1` plus `tensordict`, `onnxscript`, `onnx-ir`, `sympy 1.14` — it would
   replace torch 2.6.0+cu124, which the rule forbids. `train.py` (B2) therefore
   implements the compact PPO already specified in CONTRACT.md §3.5.

`requirements.txt` updated to the measured pins (it still said `onnx==1.17.0` and
described torch as broken).

---

---

## Wave 1 — batch 1 (A1 Physics, A2 Disturbance + Sensors) — DONE 2026-10-26

Test command (all runs from `E:\buoy-project\drl`, venv only):

```
.venv/Scripts/python.exe -m pytest tests -q     ->  52 passed in 25.77s
```

| File | Lines | Owner | Content |
|---|---|---|---|
| `sim/thrusters.py` | 136 | A1 | mixing (firmware matrix verbatim), `MOTOR_INVERT`, ESP32 deadband 0.03 per motor, first-order lag, fwd/rev asymmetry, body wrench, per-env gains |
| `sim/dynamics.py` | 135 | A1 | 3-DOF Fossen, diagonal `M_eff`, linear+quadratic drag, Coriolis, semi-implicit Euler with implicit linear drag |
| `configs/vehicle.yaml` | 50 | A1 | all CONTRACT §3.1 keys + `signs` block, every physical value `TODO SYSID` |
| `SYSID.md` | 238 | A1 | identification experiments B1–B13 for every placeholder |
| `sim/disturbance.py` | 195 | A2 | wind (quadratic, relative), current (relative-flow drag), multi-sinusoid waves, `level`-scaled, seeded |
| `sim/sensors.py` | 264 | A2 | GPS rate/latency/hold/staleness, IMU gyro + bias, mag + bias, complementary heading fusion, position LPF |
| `common/estimator.py` | 103 | A2 | `estimate_body_velocity`, arithmetic-only, numpy **and** torch (series cos/sin) |
| `configs/disturbance.yaml`, `configs/sensors.yaml` | 31 + 35 | A2 | CONTRACT §3.2/§3.3 keys |
| `tests/test_{thrusters,dynamics,disturbance,sensors,estimator,frames}.py` | 8+8+8+8+3+17 | A1/A2 | 52 tests total |
| `reports/A1.md`, `reports/A2.md` | 107 + 121 | A1/A2 | per-agent reports, open issues |

### Behaviour verified by the suite

- `mix_matrix`/`motor_invert` equal `config.h` (parsed from the C file when present); NE
  inverted; deadband 0.03 on the **mixed motor** command (Pi's 0.05 not modelled);
  per-motor saturation; lag reaches target; fwd/rev ratio = `max_fwd/max_rev × scale`;
  command → wrench map is rank 3.
- Free decay monotone, steady state equals the drag balance, `psi` wrapped, surge at
  `psi=0` moves north, sway moves east, `step()` differentiable in `tau`, stable at large
  `D_lin·dt`, batch shapes.
- Disturbance: `level=0` bit-exact zero (both dtypes, 200 steps), RMS monotone in level,
  same seed reproducible, `reset(env_ids)` touches only those rows, stateless in `t`/`dt`
  except wave phase.
- Sensors: keys/shapes/dtypes per CONTRACT §4.4, noise std matches config, zero-noise
  reproduces truth, 5 Hz epoch + hold, heading stays in `[-180,180)`, stale fix →
  `gps_valid=False` and decayed (not snapped) `u`, no NaN over 500 steps.
- Estimator: numpy vs torch agree to 1e-6, module AST has no imports and no `isinstance`.

### Corrections the orchestrator made on top of the agents

1. **`thrust_dir` regression reverted.** A1 re-paired the four props (NW with SE, NE with
   SW) to stop pure sway summing to zero. That made `cmd → wrench` **rank 2**: pure surge
   and pure sway produced the *identical* wrench (measured: both `[19.05, -8.66, ±1.30]`),
   i.e. a sim the policy cannot steer sideways. `configs/vehicle.yaml` is back to the
   CONTRACT §3.1 geometry (all props `x > 0`, NW/SW port, NE/SE starboard) and the test is
   now `test_body_axes_are_independent`: rank 3, yaw command dominates its own moment,
   surge and sway push oppositely sideways. A1's 8 tests otherwise kept unchanged.
2. `requirements.txt` (onnx pin) and `.gitignore` were already fixed during setup.

### Open issues carried forward

1. **A1-1: does `MOTOR_INVERT[NE]` double-count?** If the flag already cancels a reversed
   prop, applying it again to the force direction mirrors one corner. Bench test
   `SYSID.md B8` (thrust stand, pure surge, measure NE's force direction).
2. **A1-2: every axis couples.** MIX sway column `[+,-,-,+]` + NE invert = three motors
   forward, one reverse → surge ≈ 15 N against sway ≈ 7 N for a full sway command. This is
   what the firmware describes; the env/controller must live with it, not "fix" it.
3. **H6 unresolved**: physical sense of `+sway`/`+yaw` on the hull; `signs.*` are bench-only,
   `SYSID.md B7`.
4. **`sim/sensors.py` calls `enu_to_latlon` on the host every step** (`cpu().double().numpy()`),
   one sync per step. Fine at small `num_envs`, must be revisited before 2048-env training.
5. No physical constant, noise level or disturbance magnitude is measured — everything is
   `TODO SYSID`. No sim number here says anything about the real vessel.

### Process note (provider)

The sub-agent provider failed repeatedly (`HTTP 502` / `model_disabled`, "Out of stock"):
5 child runs died mid-task, twice after writing nothing. Files that did land were written
before the crash, so the wave was recovered from disk plus inline work (thrusters,
dynamics and their tests were written by the orchestrator when A1 had produced nothing).
Sub-agent reports kept as `reports/A1.md`, `reports/A2.md`.

**Status: batch 1 complete, 52 passed. Next: A3 PID baseline + A4 runtime.**

---

## Wave 1 — batch 2 (A3 PID baseline, A4 runtime) — DONE 2026-10-26

Test command, real output:

```
cd /e/buoy-project/drl && .venv/Scripts/python.exe -m pytest tests -q
79 passed in 24.56s
```

| File | Lines | Owner | Content |
|---|---|---|---|
| `baseline_pid.py` | 191 | A3 | `PIDBaseline`, batched torch, CONTRACT §4.6 interface (`act`/`reset`), rad + ENU (not the Pi's degrees) |
| `tests/test_pid_parity.py` | 343 | A3 | 7 tests: 4 parity vs the real `thrust.py`, closed-loop toy plant |
| `runtime/obs_np.py` | 146 | A4a | numpy observation builder, CONTRACT §5 layout, K=10 history, no torch |
| `runtime/policy.py` | 237 | A4a | `RLController.step()`, onnxruntime only, safety wrapper (clip → slew → geofence/watchdog/fault fallback) |
| `runtime/infer_replay.py` | 159 | A4b | offline CSV replay with argparse, prints step count + action range + fault counters |
| `tests/test_runtime.py` | 439 | A4b | 20 tests, dummy ONNX built with `onnx.helper` (no torch) |
| `reports/A3.md`, `reports/A4a.md`, `reports/A4b.md` | 176 + 120 + 117 | | per-agent reports |

### What the 20 new tests actually show

- Observation builder: CONTRACT §5 layout, history K=10 oldest-first, deterministic.
- Controller: model is really invoked, action in `[-1,1]`, slew limit `2.0/s · dt` = **0.2 per step at dt = 0.1 s** (A4a's report said 0.1 — that is the 20 Hz figure; **code is right, report was wrong**).
- Faults: `gps_valid=False`, `sensor_ok=False`, watchdog (stale sample and inter-call gap), NaN action → fallback path (slew-limited, not a step); the legacy `has_fix` flag is **ignored**, per the human's decision.
- Geofence 80 m → zero command. History resets on episode change.
- `infer_replay.py` runs as a subprocess on a temp CSV, exit 0, reports `fallback=2 gps=1 watchdog=1`.
- `import runtime.policy / obs_np / infer_replay` leaves `torch` out of `sys.modules` (checked directly).

### PID parity (A3, measured)

Parity against the real `pi-control/src/control/thrust.py` with its hardware imports stubbed
and the compute path untouched: **max deviation 3.0e-7** at atol 1e-4, including an 8-step
integrator/derivative trace with a frozen clock. Closed-loop toy plant: station keeping
settles at 0.688 m (= 0.5·√2, the DP deadband corner), waypoint 23.6 m reached inside 2 m at
33.3 s.

### Orchestrator fix on top of the agents

`runtime/policy.py:DEFAULT_SAFETY` carried a `gps_timeout_s` key that no code path read
(A4b reported it, did not change it). Removed, with a comment stating why: GPS loss is
communicated by the caller's `gps_valid` / `sensor_ok` flags (CONTRACT §4.4), and a second
stale-fix timer inside the wrapper would fight the sensor model. Suite re-run after the edit:
79 passed.

### Open issues carried into Wave 2/4

1. **`est["t"]` semantics must be pinned in the Pi glue.** If it is the GPS fix time (1 Hz by
   default) and not the control-loop time, the 0.5 s watchdog will trip permanently and the
   policy will run on the fallback path forever. Wave 4 must define it.
2. **PID baseline's `output_deadband` (0.05) is the Pi convenience, not the actuator.**
   `sim/thrusters.py` owns the real 0.03 per mixed motor. Wave 3/4 must state which is used
   for scoring the baseline, or the comparison is biased against the PID.
3. **A3 suspects inherited from `thrust.py`, unfixed by design:** per-axis 0.5 m DP deadband
   caps station keeping at 0.707 m; derivative kick (~6·err) on the first step after a reset;
   `speed_norm` is a no-op; H1 (heading sign) inherited.
4. **`infer_replay` fallback emits zeros**, while `baseline_pid` offers a batched `act()`.
   Wiring the real PID as the runtime fallback is Wave 4 work.
5. **CSV schema in `infer_replay` is a proposal** — no logger writes it yet.
6. No trained model exists: every runtime number above is plumbing on a dummy ONNX graph.
   Nothing here says anything about control quality, and nothing transfers to the vessel.

**Status: Wave 1 complete — 79 passed. Gate: stopping here for review before Wave 2.**

---

## Wave 2 — Environment, Trainer, Metrics (B1, B2, B3) — DONE 2026-10-26

Test command, real output:

```
cd /e/buoy-project/drl && .venv/Scripts/python.exe -m pytest tests -q
129 passed, 14 warnings in 33.77s
```

| File | Lines | Owner | Content |
|---|---|---|---|
| `envs/asv_env.py` | 466 | B1 | vectorised env: dt 0.02, policy 10 Hz, obs + K=10 ring buffer, action-rate limit, reward terms, episodes (waypoints + station keeping), geofence/timeout, domain randomisation, curriculum, `est["t"]` clock |
| `configs/env.yaml` | 59 | B1 | every reward weight + randomisation range (no magic numbers in code) |
| `models.py` | 145 | B2 | `PolicyNet` 88,583 params, `forward → (mean, value, std)`, normalisation baked in, 1.08 MB checkpoint |
| `train.py` | 326 | B2 | compact PPO: GAE, clip, minibatches, entropy, lr anneal, TensorBoard, checkpoint + `--resume`, CLI overrides |
| `configs/train.yaml` | 43 | B2 | all hyperparameters |
| `evalkit/metrics.py` | 167 | B3 | mean/p95 radial error, cross-track RMSE, settling time, energy, action rate (CONTRACT §4.9 signatures, numpy **or** torch input) |
| `evalkit/plots.py` | 104 | B3 | the only matplotlib import (`Agg`), writes PNG via `out=` |
| `tests/test_env.py` / `test_obs_parity.py` | 235 / 143 | B1 | 17 + 5 tests |
| `tests/test_train_smoke.py` | 175 | B2 | 6 tests, dummy env per CONTRACT §4.5, CPU |
| `tests/test_metrics.py` | 232 | B3 | hand-computable synthetic trajectories |
| `reports/B1.md`, `B2.md`, `B3.md` | 150 + 181 + 98 | | per-agent reports |

### Verified independently on disk (not taken from the reports)

- Suite green at **129 passed**; the 79 Wave-1 tests are untouched.
- `tests/test_obs_parity.py` really imports `runtime.obs_np` (`raw_sample`, `normalize`,
  `ObsBuilder`, `K_HISTORY`, `RAW_DIM`) and compares it with `raw_sample_torch` — max |Δ| ≤ 1e-5
  over all 150 observation components. `runtime/obs_np.py` was not modified.
- `evalkit.metrics` imports neither matplotlib nor torch (`sys.modules` check run directly);
  the matplotlib mention in that file is a comment.
- Deadband for scoring is the ESP32 one: `0.03` base from `sim/thrusters.py:deadband0`,
  randomised per env, **no 0.05 anywhere** in the env.
- `est["t"]` = `env.t_loop`, a control-loop monotonic clock, not reset per episode
  (`envs/asv_env.py:213,403,426`), with a test asserting `now − est["t"] ≤ watchdog_s` — the
  runtime watchdog of `runtime/policy.py` cannot trip spuriously.

### Measured numbers

- B1 performance: 2048 envs → **53 ms/step** (≈19.4 policy steps/s), VRAM 12 MB. The sensor
  model is ~40 % of the step time because `sim/sensors.py` converts lat/lon on the CPU.
- B2: **24,281 env-steps/s** at 2048 envs × 32 steps, peak **200 MiB reserved / 161 MiB
  allocated** of the 6 GB card (~3 %). Dummy-env reward rose over the smoke iterations;
  checkpoint round-trips bit-exact.
- B3: p95 by linear interpolation; settling = last entry into the band + `hold_s` guard;
  action rate = total variation per second.

### rsl_rl decision (B2, verified again here)

`rsl_rl` is absent (`find_spec → None`); `pip install --dry-run rsl-rl-lib` resolves to
`rsl-rl-lib 5.5.1` **plus torch-2.14.1**, which would replace the working
torch 2.6.0+cu124. Nothing was installed. The compact PPO of CONTRACT §3.5 is what runs.

### Orchestrator fix

`.gitignore` gained `runs/` (B2's default `log_dir` / `ckpt_dir`, flagged by the agent but
outside its Owns list).

### Open issues carried forward

1. **`sim/dynamics.py:set_params` takes `float()` only** — per-env mass/inertia/drag
   randomisation is currently the same scalar for the whole batch. B1 worked around it
   (batch-wide); a real per-env randomisation needs a small edit to `set_params`.
2. **`sim/sensors.py` has no per-env noise/latency hooks** — those knobs are batch-wide, so
   sensor domain randomisation is coarser than the thruster one.
3. **Sensor model dominates step time** (~40 %) due to the per-step CPU lat/lon conversion.
   Fix before the full curriculum (torch-side ENU→lat/lon, or compute on GPS epochs only).
4. **No trained checkpoint exists** — `evaluate.py`/`export_onnx.py` (Wave 4) are unproven
   against a real policy; the smoke test only shows the loop runs and reward does not NaN.
5. Tuning still placeholder: geofence radius, acceptance radius, all reward weights (T7/T8),
   metric tolerance 1.0 m, energy reported in normalised units until per-motor `max_thrust`
   scaling is decided (Wave 4).
6. Curriculum is driven inside the env (`set_curriculum`); `train.py` has no curriculum hook —
   fine for now, must be settled before the long run.

**Status: Wave 2 complete — 129 passed. Gate: stopping here for review before Wave 3.**

---

## Fix unit R1 — per-env randomisation (dynamics + sensors) — DONE 2026-10-26

```
cd /e/buoy-project/drl && .venv/Scripts/python.exe -m pytest tests -q
135 passed, 14 warnings in 35.10s
```

- `sim/dynamics.py`: `set_params` now takes per-env `[N]` tensors (scalar still accepted,
  `None` = nominal) and gained `quad_drag_scale`; two envs in one batch can differ in
  mass/inertia/linear+quadratic drag.
- `sim/sensors.py`: new `set_noise(env_ids, pos_noise_m=, vel_noise_mps=, cog_noise_deg=,
  gyro_noise_dps=, gyro_bias_dps=, mag_noise_deg=, mag_bias_deg=, gps_latency_s=)` stored as
  per-env tensors. Per-env latency uses **one** `[N, L+1, 2]` ring with `L = max_i
  round(latency_i/dt)` (capped 1024 steps) read through a single `torch.gather` — no Python
  loop, ~0.5 MB at 2048 envs.
- `tests/test_randomization.py` (6 tests): differing mass/drag proven through the numeric
  response of `step()`, differing noise/latency through statistics and delay, and the
  non-randomised path proven **bit-identical** against `float.hex()` snapshots taken before
  the change (`torch.equal`).
- `envs/asv_env.py` was not touched (mtime unchanged). Six required call-site edits are
  tabulated in `reports/R1.md` §3, with current call → replacement text.
- Known limits stated by the agent: latency stays quantised to whole steps with a 1-step
  floor (`latency_s = 0` ≠ zero delay, pre-existing), and GPS *rate* is still not randomised
  — only noise, bias bounds and latency.

**Status: R1 done — 135 passed. Still stopped at the Wave 2 gate; the env edits above are
B1's to make before any training run.**

---

## Fix unit R2 — env samples per env (Wave 2 follow-up) — DONE 2026-10-26

```
cd /e/buoy-project/drl && .venv/Scripts/python.exe -m pytest tests -q
140 passed, 14 warnings in 35.21s
```

- `envs/asv_env.py` now calls `dynamics.set_params(idx, …)` and `sensors.set_noise(idx, …)`
  per env on **every** reset, with the randomisation moved **before** `sensors.reset(idx)` so
  an episode's bias draw sees its own bounds; the old `if full:` guard is gone. New
  `randomize` flag turns it all off.
- `configs/env.yaml` unchanged — every range already existed, no key added, no old value
  touched (mtime unchanged, confirmed).
- `sim/*.py` untouched (still R1's 16:29 version), `tests/test_obs_parity.py` untouched and
  passing, so the torch observation builder still matches `runtime/obs_np.py`.
- New `tests/test_env_randomization.py` (5 tests): two envs differ after one randomised reset
  (mass, drag, sensor noise/bias/latency); with randomisation off the rollout is identical to
  the pre-change snapshot for the same seed; and a source-level assertion that the old
  `setattr(self.sensors, "gps_latency"/…)` style is gone.

### Effective factor on the physics parameters (verified independently)

`set_params` applies, per axis, with `ms = mass_scale`, `ins = inertia_scale`, `ds = drag_scale`:

| tensor | x (surge) | y (sway) | n (yaw) |
|---|---|---|---|
| `M_env` | `· ms` | `· ms` | `· ins` |
| `D_lin_env` | `· ms · ds` | `· ms · ds` | `· ds` |
| `D_quad_env` | `· ms · ds` | `· ms · ds` | `· ds` |

Spot-checked directly: `ms = 1.3, ds = 1.2` → `M_x` 15.0 → **19.5**, `D_lin_x` 6.0 → **9.36**,
`D_quad_x` 4.0 → **6.24**, `D_lin_n` 1.5 → **1.8**. The **old** env writer applied `ds` only,
so surge/sway drag ranges widen (`0.49–1.69` vs `0.70–1.30`; e.g. `D_lin_x` 9.36 vs 7.20 for
the same draw). **Open question, not fixed:** coupling `mass_scale` into drag is a modelling
choice (a heavier hull does not necessarily have more drag); if it is unwanted, the fix belongs
in `sim/dynamics.py:set_params`, not in the env.

### Reported, not fixed

1. `SensorModel.reset` accepts no initial position and the env passes none, so every episode
   starts the estimator at the ENU origin while the true pose is elsewhere.
2. Consequence: the GPS latency ring is zero-filled, so for the first `lat` steps the fix
   reports ENU `(0, 0)` (0.2 s → 10 steps; measured `est (0.18, 0.07)` vs truth `(1.15,
   18.61)`). The policy therefore sees a large phantom error at episode start.
3. `env.yaml:sensor_bias_deg` and `sensors.yaml:mag.bias_deg` are two sources for the same
   quantity.
4. A partial reset that raises the batch's maximum latency rebuilds the GPS ring for **all**
   envs, wiping their fix history.

**Status: R2 done — 140 passed. Still stopped at the Wave 2 gate; items 1-4 above are
candidates for Wave 3 (C1).**

---

## Fix unit R3 — sensor reset/ring/bias + mass/drag decoupling — DONE 2026-10-26

```
cd /e/buoy-project/drl && .venv/Scripts/python.exe -m pytest tests -q
143 passed, 14 warnings in 34.76s
```

- `SensorModel.reset(env_ids, pos=None, heading=None)` seeds `pos_hat`, `gps_enu`, `heading`
  **and every GPS-ring slot** with the episode's true start pose; the no-argument call is
  bit-identical to before. Phantom start error: **38.1 m max → 1.53 m** (0.52 m mean = the GPS
  noise floor), same on CUDA.
- The GPS ring is allocated **once**, `cap = clamp(round(max_latency_s/dt), 1, 1024)` (env
  passes `0.2 + 0.1 s` → 15 steps, 256 KB at 2048 envs, still one `gather`). A partial reset
  writes only `_buf[idx]`; no reallocation, no other env's history touched.
- Bias: single source `configs/sensors.yaml:mag.bias_deg` (5.0). `env.yaml:sensor_bias_deg`
  was read by nobody after the change, so the dead key is **removed** (orchestrator edit, no
  numeric value changed; `sensor_noise` / `sensor_latency_s` untouched and now marked per-env).
- `sim/dynamics.py:set_params` decoupled — verified directly with `ms=1.3, ins=1.1, ds=1.2`:
  `M` = **[19.5, 26.0, 1.98]** (= 15·ms, 20·ms, 1.8·ins), `D_lin` = **[7.2, 21.6, 1.8]**
  (= 6·ds, 18·ds, 1.5·ds), `D_quad` = **[4.8, 14.4, 0.96]**. Mass no longer leaks into drag;
  the factor table in the R2 section above is obsolete.
- Env edits are six lines, listed in `reports/R3.md` §5. `tests/test_obs_parity.py` and
  `configs/*` are untouched, so the torch observation builder still matches `runtime/obs_np.py`.
- One rollout hash was re-recorded (`0af78a8e…` → `e115cfc5…`); the agent proved the only cause
  is pose seeding by patching `reset` to ignore `pos`/`heading` and reproducing the old hash
  bit-exactly.

**Status: R3 done — 143 passed. Still stopped at the Wave 2 gate.**

---

## Wave 3 — Integration smoke test (C1) — DONE 2026-10-26

```
cd /e/buoy-project/drl && .venv/Scripts/python.exe -m pytest tests -q
148 passed, 14 warnings in 63.19s
```

Files: `tests/test_integration.py` (11,262 B, 5 tests), `INTEGRATION.md`, `reports/C1.md`,
`reports/c1_train.log`, `reports/c1_pid.log`.

### PPO smoke run — measured, not estimated

Conditions: level 0 (no disturbance, no curriculum), seed 0, 1024 envs, 32-step rollout,
compact PPO, cuda.

| quantity | value |
|---|---|
| reward, iteration 0 | **+0.2079** |
| reward, iteration 39 (last) | **+0.4811** (max +0.6531 at it 36; noisy, non-monotone) |
| first-5-iter mean → last-5-iter mean | +0.2022 → +0.5327 |
| return | 2.06 → 53.86 |
| total steps | **1,310,720** |
| wall time | **69.7 s** (18,817 steps/s) |
| peak VRAM | **89.7 MiB allocated / 110.0 MiB reserved** of 6 GB |

Reward rose; the curve is noisy, so this says the loop learns *something* in 1.3 M steps,
nothing about control quality.

### PID baseline inside the env — it does not converge (real finding)

16 envs × 1200 policy steps = 120 s, level 0, physics randomisation off, seed 0:

| scenario | radial mean | p95 | cross-track RMSE | settled | energy | action rate | error trend |
|---|---|---|---|---|---|---|---|
| station keeping | **45.54 m** | 93.96 m | 47.37 m | **0 %** | 272.4 | 1.416 /s | 40.03 → 48.39 m |
| waypoint | **39.24 m** | 84.36 m | 25.12 m | **0 %** | 223.4 | 1.436 /s | 38.79 → 40.90 m |

Cause found: **every body command produces the same surge force.** Measured independently
here with nominal gains: `[1,0,0] → (19.05, −8.66, −1.30)`, `[0,1,0] → (19.05, +8.66, −6.50)`,
`[0,0,1] → (19.05, −8.66, +6.50)` N / N·m. (C1's report quotes 17.32 N / ±10.39 / ±7.80 — a
randomised-gain run; the qualitative result is identical.) Surge is coupled into sway and
yaw on every command, the heading loop saturates and limit-cycles, and the PID — tuned for the
real vessel — cannot close the loop in this sim. Root cause is the unmeasured
`thrusters.thrust_dir` / `positions_m` (TODO SYSID, INSPECTION H6), not a controller bug.

**Consequence for Wave 4:** a PID-vs-PPO table on this sim is not yet a fair comparison until
the thruster geometry is either bench-identified or the coupling is acknowledged in the report.

### Pins that C1 added (regression guards)

1. Estimator error right after reset and after step 1: max **1.95 m** against a 23.0 m mean
   spawn distance (asserted < 5 m) — the R3 pose seeding holds.
2. Effective drag factor over 2000 sampled resets: min **0.7010**, max **1.2997** inside the
   env.yaml drag range `[0.70, 1.30]`; mass-drag decoupling asserted too.

### Fixes to earlier files (complete list, all from C1)

- `train.py`: per-iteration stdout line, `--curriculum-level` flag, peak-VRAM print.
- `tests/test_env.py::test_domain_randomization_ranges`: drag bound tightened to `r_drag`,
  `D_quad` added (was loose after R3).
- Nothing else was touched. Reward weights, geofence (80 m) and arrival radius (2.0 m) are
  **unchanged and still open decisions** (T7/T8).

### Open decisions the human must make before/with the full run

1. **Reward weights** (`configs/env.yaml`, all seven) — placeholder, unvalidated.
2. **Geofence 80 m and arrival radius 2.0 m** — placeholder (T8).
3. **Observation clip ±5** hides position errors beyond ~50 m (10 m scale × 5), so a badly
   lost episode looks the same as a mildly lost one to the policy.
4. **Metrics mix episodes** because the env auto-resets inside a rollout; `evalkit` numbers
   from a training rollout are not per-episode.
5. **Thruster geometry / coupling** as described above (SYSID B7/B8).

**Status: Wave 3 complete — 148 passed. The full curriculum training run is the human's to
launch; nothing was started by the orchestrator.**

---

## Wave 4 — Evaluate (C2) + Export (C3) — 2026-10-27

Training input (human-launched, verified from the checkpoint and its event file — not from a
report): `runs/checkpoints/ckpt_last.pt`, **steps 19,988,480**, iteration **305**,
`cfg.total_steps` 20,000,000, num_envs 2048, hidden `[256,128,128]`. TensorBoard
`train/reward` mean per iteration **0.4271 → 0.9166** (first-20 vs last-20), max 1.1222,
`train/return` 2.05 → 91.95, entropy flat **2.759 → 2.763** (converged, not still learning),
~30.5k env-steps/s.

### C3 Export — DONE, verified twice independently

Artifacts `runs/policy.onnx` (356,448 B) + `runs/policy_meta.json`; code `export_onnx.py`,
`tests/test_onnx_parity.py`; report `reports/C3.md`.

- §4.8 satisfied: opset 17, `obs` `[B,150]` → `action` `[B,3]`, **dynamic batch**, `Sub`/`Div`
  with the checkpoint's `obs_mean`/`obs_std` baked as graph constants, `Tanh` as the last node,
  no sampling ops.
- **Parity measured twice: 1.341e-06 and 1.330e-06** max abs diff on 256 obs in `[-3,3]`
  (limit 1e-4). B=1 → 5.364e-07, B=17 → 1.207e-06.
- Artifact is genuinely weight-bearing: on-disk `.onnx` is byte-identical to a fresh re-export,
  and a random-init export differs by max |Δ| **1.059**.
- `runtime/infer_replay.py` exit 0 on `results/replay_sample.csv`: 50 steps,
  `sources: fallback=1 policy=49`, `faults: gps=1`. **The CSV is synthetic, not a boat log.**
- `pytest tests/test_onnx_parity.py` → **6 passed**.

### C2 Evaluate — script delivered, results measured, agent did NOT finish

`evaluate.py` (CLI documented in its docstring), `tests/test_evaluate.py` (6 tests),
per-seed CSVs and plots under `results/`. The C2 agent **hit its iteration cap and reported
itself incomplete**; it never wrote `results/summary.*` or `reports/C2.md`. The orchestrator
finished those.

Measured, 8 envs per seed, real checkpoint, disturbances at full scale (`scale_by_level`
top row = 1.0). PPO vs PID on identical seeds:

| scenario | seeds | PPO radial mean | PID radial mean | PPO settled | PID settled |
|---|---|---|---|---|---|
| `sk_wind` (SK 120 s, level 1.0) | 0–3 | **9.09 – 17.34 m** | 25.37 – 45.26 m | 12.5 – 50.0 % | **0.0 % every seed** |
| `wp_square` (4-leg 30 m, 200 s) | 1000–1003 | **4.06 – 5.66 m** | 25.66 – 28.97 m | 0.0 – 37.5 % | 0.0 – 37.5 % |
| `rough_sea` (T7 rough-scale 2.0) | 2000 | **6.84 m** | 32.21 m | 16.7 % | **0.0 %** |

PPO wins on radial error in every scenario and every seed. **This is not yet a fair
comparison:** the PID baseline cannot close the loop in this sim (Wave 3 finding — every body
command produces the same surge force, so heading saturates and limit-cycles). The gap above
measures that modelling defect as much as it measures the policy. Energy is reported in
**normalised command units** (CONTRACT §4.9 left the joule-equivalent open; decided here).

### Bugs found this wave

1. **`set_curriculum` takes a level INDEX, not a scale.** `envs/asv_env.py:234` does
   `int(round(level))` and indexes `scale_by_level[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]`, so
   `set_curriculum(1.0)` selects **scale 0.1**. The auto-advance path used by the training run
   is index-based throughout and was **not** affected; the training run was correct.
   - Fixed the two live call sites: `evaluate.py:check_fast_env` (was passing 1.0 → 0.1) and
     `train.py --curriculum-level` help text, which previously described the argument as "the
     level" and would silently train at 10 % disturbance.
2. **`tests/test_evaluate.py` had a wrong expectation, not a code bug.** `episode_arrays`
   right-aligns blocks per env, so `measured=1` selects each env's **last** block; the fixture
   assumed episode 0. Corrected the fixture to assert the real right-aligned semantics
   (`[[True,False],[False,True]]`, and `[[False,False]]` with a geofence exit on the last
   block). `evaluate.py`'s logic was correct and is unchanged apart from item 1.
3. **`evalkit/plots.py:58` leaks matplotlib figures** — never calls `close()`, so a 3-scenario
   × 4-seed run opens >20 and warns. Cosmetic today, a memory risk on longer sweeps. Not fixed
   (own by B3); recorded.
4. **Contract/meta drift:** CONTRACT §4.8 lists `"gps_timeout_s": 2.0` inside `safety`, but the
   exported meta omits it — `gps_timeout_s` now exists only in `CONTRACT.md:449` and a test
   literal (`tests/test_runtime.py:242`); no runtime code reads it (A4 removed the dead key).
   `RLController` works. Either the contract text or the meta must be reconciled.

### Caveats carried forward (unchanged)

- Reward weights, geofence 80 m, arrival radius 2.0 m are placeholders (T7/T8).
- `obs.clip_norm` ±5 at a 10 m scale hides position errors beyond ~50 m.
- Thruster geometry/coupling unmeasured (SYSID B7/B8) — the reason the PID table is unfair.
- Episode counts below 8/8 mean geofence truncation, not completion; `complete` flags this.
- Env is CPU-synchronous (~80–180 ms/step at any batch size; `.item()`/`.cpu()` per step in
  `sim/disturbance.py` and `sim/sensors.py`). `FastEnv` in `evaluate.py` skips the unread
  `gps_lat`/`gps_lon` round trip; disturbance wrench parity vs `AsvEnv` measured **0.0**.

**Status: Wave 4 complete — `160 passed, 14 warnings in 245.16s`.** C3 verified twice
independently. C2's agent hit its iteration cap and self-reported incomplete; the orchestrator
finished `results/summary.*` and this section, and corrected its two bugs above.

---


**Nothing here was validated against the real vessel.** All numbers above are from
commands run on the dev box; the physical model is placeholder-parameterised throughout.
