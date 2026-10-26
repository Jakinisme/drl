# TASK: Build a DRL (PPO) controller to replace the PID in my ASV, trained in a custom vectorized PyTorch simulator (NOT Isaac Sim).

## Context
- Vehicle: catamaran ASV. 4x A2212 brushless thrusters in a 45° X configuration, fully actuated in 3-DOF (surge, sway, yaw).
- Existing stack: Raspberry Pi 3B (Python, `pi-controller`) + ESP32 (C++/PlatformIO, `esp-firmware`) over UART. The ESP32 already mixes body commands to motors using this matrix in `config.h` (columns = surge, sway, yaw):
  FL: +,+,+ | FR: +,-,- | RL: +,-,+ | RR: +,+,-
- Sensors: GPS NEO-M8N, IMU MPU6500, magnetometer QMC5883P (heading is already fused in `fusion.py`).
- Current controller: PID in `thrust.py` (waypoint following + station keeping). It must be kept as baseline and as fallback.
- Dev machine: Windows 11, NVIDIA RTX 3050 with 6GB VRAM. Isaac Sim/Isaac Lab will NOT run, so do not use them. Keep everything light (PyTorch CUDA, numpy, onnx, onnxruntime, rsl_rl or own PPO).

## Goal
A single goal-conditioned policy that does both waypoint following and station keeping (station keeping = fixed goal, target velocity 0), trained in simulation, exported to ONNX, and runnable on the Pi via onnxruntime. Policy outputs normalized body commands [surge, sway, yaw] in [-1, 1]; the existing ESP32 mixing stays unchanged.

## First, inspect before coding
Read `fusion.py`, `thrust.py` and config files in the repo. Report: heading convention (compass? degrees? 0=north?), units, control loop rate, how GPS lat/lon is converted, what the ESP32 UART message format is. Match these conventions exactly in sim and in the Pi runtime. Do NOT modify firmware. Do not rewrite existing code; only add new files.

## Deliverables (new directory `drl/`)
1. `sim/dynamics.py`: vectorized 3-DOF Fossen-style model on GPU, state per env: pose (x, y, psi) in a local ENU frame and body velocity nu = (u, v, r).
   M_eff * nu_dot = tau_thrusters + tau_env - D_lin*nu - D_quad*|nu|*nu - coriolis(nu).
   All parameters (mass, added mass, inertia Iz, drag coefficients, thruster positions, max thrust per motor fwd/rev) live in `configs/vehicle.yaml` with clearly marked TODO placeholders. Never invent final values; use plausible placeholders and flag them.
2. `sim/thrusters.py`: per-motor model: normalized command -> thrust with deadband, asymmetric forward/reverse max, saturation, first-order lag. Use the same mixing matrix above (command -> 4 motors -> body wrench via geometry).
3. `sim/disturbance.py`: wind force (proportional to relative wind speed squared, random direction), current (relative-velocity effect), waves (sum of several random sinusoidal horizontal force/torque components). Magnitudes scheduled by a curriculum level.
4. `sim/sensors.py`: simulate what the real Pi sees: GPS position at configurable rate (default 1-5 Hz) with noise, latency and hold; heading noise and bias; velocity from a simple complementary filter/EKF run inside the sim loop. The policy must only observe estimator outputs, never ground truth.
5. `envs/asv_env.py`: vectorized env (N envs, thousands on GPU). Sim dt 0.02 s, policy rate 10 Hz (configurable).
   - Observation: goal in body frame (dx, dy), sin/cos of heading error, body velocity (u, v, r), previous action, next waypoint and the one after (body frame), mode flag; history of K=10 steps stacked (option for GRU). History/memory is required so the policy can infer constant wind/current (replaces the PID integral term).
   - Action: 3 values in [-1,1] with a configurable action-rate limit.
   - Reward (all weights in config): exponential position-error and heading-error tracking, progress toward waypoint, cross-track penalty, penalty near-goal speed, action-rate penalty, thrust-squared energy penalty.
   - Episodes: random start pose, random waypoint sequences (acceptance radius configurable) and station-keeping episodes. Terminate on timeout or leaving a large geofence.
   - Domain randomization per episode (ranges in config): mass/inertia/drag ±30%, thruster gain, lag, deadband, fwd/rev asymmetry, one weakened thruster, sensor noise/bias/latency, wind/current/wave magnitudes.
   - Curriculum: start without disturbances, increase gradually.
6. `train.py`: PPO training. Prefer `rsl_rl` (check the INSTALLED version's API before writing the wrapper; if it does not fit cleanly, implement a compact PPO). Log to TensorBoard, save checkpoints. Must fit comfortably in 6GB VRAM.
7. `baseline_pid.py`: port my PID logic (read `thrust.py`) into the sim as a controller with the same env interface, for fair comparison.
8. `evaluate.py`: run identical scenarios for PID vs PPO and output a table + plots: station keeping (120 s with wind/current/waves), square waypoint path, rough-sea scenario. Metrics: mean and 95th-percentile radial error, cross-track RMSE, settling time, energy (integral of thrust squared), action rate. Fixed seeds.
9. `export_onnx.py`: export the deterministic policy (mean action), with observation normalization baked into the graph (and hidden state I/O if recurrent). Include a parity test: torch vs onnxruntime max abs diff < 1e-4 on random inputs.
10. `runtime/policy.py` (for the Pi 3B, only numpy + onnxruntime): class `RLController` with `step(estimator_state, goal) -> (surge, sway, yaw)`; includes safety wrapper: output clipping, slew-rate limit, GPS-fix/sensor-validity check, geofence, watchdog, and a hook to fall back to the existing PID. Also `infer_replay.py` that replays a logged sensor CSV through the ONNX policy offline (deployment rehearsal) without touching hardware.

## Working rules
- Work incrementally: dynamics -> unit tests -> env -> baseline PID in sim -> training -> evaluation -> export -> runtime. Stop and summarize after each stage; run tests yourself.
- Unit tests: free-decay of velocity, thrust saturation, mixing-matrix consistency with `config.h`, frame/heading conversion round-trip, ONNX parity.
- SI units everywhere, document frame conventions in `drl/README.md`.
- Every unknown physical parameter must be a TODO in `configs/vehicle.yaml`, plus a short `SYSID.md` listing the experiments needed to identify it on the real vessel (bollard pull per thruster, step/decay tests, zigzag).
- Windows 11 compatible; no Linux-only dependencies in the training side.
- Do not claim sim results transfer to the real vessel; report only what you measured.