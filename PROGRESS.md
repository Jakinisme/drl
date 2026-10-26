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

**Nothing here was validated against the real vessel.** All numbers above are from
commands run on the dev box; the physical model is placeholder-parameterised throughout.
