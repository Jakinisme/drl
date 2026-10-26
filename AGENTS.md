# AGENTS.md — DRL ASV controller: orchestration plan
Full spec: drl/TASK.md. Shared contract: drl/CONTRACT.md (created in Wave 0).
You are the ORCHESTRATOR. Use delegate_task. Run ONE wave at a time.

## Rules to include in EVERY sub-agent goal
1. Read drl/TASK.md and (after Wave 0) drl/CONTRACT.md first. You know nothing else.
2. Create/edit ONLY files in your "Owns" list. Existing repo code (fusion.py, thrust.py, config.h, firmware) is read-only.
3. You cannot ask the user. Unknown conventions or physical values -> TODO in the file and in your report. Never invent final values.
4. If you find a bug in existing code (e.g. PID sign, angle averaging near 0/360 deg), report it; do not fix silently.
5. Run your own tests with pytest and fix failures. Windows 11 compatible, SI units.
6. Final report max 15 lines: files created, test results, assumptions/TODOs, open problems. Never claim sim results transfer to the real vessel.

## Wave 0 (1 agent)
### A0 Inspector/Architect
Owns: drl/INSPECTION.md, drl/CONTRACT.md, drl/requirements.txt, drl/common/frames.py, drl/tests/test_frames.py, drl/{sim,envs,runtime,configs,tests,common}/__init__.py
- Read fusion.py, thrust.py, config.h, config files, UART code.
- INSPECTION.md: heading convention (compass? degrees? 0=north?), units, control loop rate, GPS lat/lon -> local conversion, ESP32 UART message format.
- Check environment: torch + CUDA available, installed rsl_rl version and API (or absent), onnx, onnxruntime.
- CONTRACT.md: frames; state layout (x,y,psi,u,v,r); tensor shapes + function signatures for dynamics, thrusters, disturbance, sensors, env (reset/step), controller interface (batched estimator_state + goal -> action); observation layout with K=10 history and normalization; policy network spec shared by train and export; config file and key names; numpy-vs-torch observation builder parity requirement.
- common/frames.py: lat/lon <-> ENU and heading conversion, with round-trip test.
GATE: human reviews INSPECTION.md and CONTRACT.md before Wave 1.

## Wave 1 (4 agents in parallel, depend only on CONTRACT.md)
### A1 Physics
Owns: sim/dynamics.py, sim/thrusters.py, configs/vehicle.yaml, SYSID.md, tests/test_dynamics.py, tests/test_thrusters.py
- Vectorized 3-DOF Fossen-style model on GPU; per-motor model (deadband, asymmetric fwd/rev, saturation, first-order lag); mixing matrix identical to config.h.
- All unknown parameters = TODO placeholders in vehicle.yaml. SYSID.md lists identification experiments (bollard pull, step/decay, zigzag).
- Tests: free-decay, thrust saturation, mixing-matrix consistency with config.h.
### A2 Disturbance + Sensors
Owns: sim/disturbance.py, sim/sensors.py, configs/disturbance.yaml, configs/sensors.yaml, tests/test_disturbance.py, tests/test_sensors.py
- Wind (quadratic in relative speed), current, multi-sinusoid waves, scaled by curriculum level.
- GPS rate/noise/latency/hold, heading noise+bias, complementary filter/EKF inside the sim; policy sees estimator output only.
- Tests: quadratic wind scaling, latency/hold behavior, noise statistics, level 0 = zero disturbance.
### A3 PID baseline
Owns: baseline_pid.py, tests/test_pid_parity.py
- Port PID logic from thrust.py (waypoint following + station keeping) as a batched controller with the same interface as the RL policy (see CONTRACT.md).
- Parity test against the original thrust.py on fixed inputs (import it read-only). Port faithfully; report suspected bugs, do not fix.
### A4 Runtime
Owns: runtime/policy.py, runtime/obs_np.py, runtime/infer_replay.py, tests/test_runtime.py
- numpy+onnxruntime only. RLController.step(estimator_state, goal) -> (surge, sway, yaw). numpy observation builder per CONTRACT.md.
- Safety wrapper: clipping, slew-rate limit, GPS-fix/sensor-validity check, geofence, watchdog, PID-fallback hook.
- infer_replay.py replays a sensor CSV offline. Tests use a dummy ONNX model (random MLP built inside the test).
GATE: orchestrator runs `pytest drl/tests` and writes drl/PROGRESS.md, then STOPS.

## Wave 2 (3 agents in parallel)
### B1 Environment
Owns: envs/asv_env.py, configs/env.yaml, tests/test_env.py, tests/test_obs_parity.py
- Vectorized env on GPU (dt 0.02, policy 10 Hz), observation + K=10 history, action-rate limit, reward terms with all weights in env.yaml, episode types (waypoints + station keeping), geofence/timeout termination, domain randomization ranges, curriculum.
- Torch observation builder must match runtime/obs_np.py (parity test).
### B2 Trainer
Owns: train.py, models.py, configs/train.yaml, tests/test_train_smoke.py
- Check the INSTALLED rsl_rl API first; if it does not fit, implement compact PPO. TensorBoard + checkpoints. Must fit 6GB VRAM.
- models.py holds the policy network used by both training and export. Test against a dummy env that follows CONTRACT.md.
### B3 Metrics
Owns: evalkit/metrics.py, evalkit/plots.py, tests/test_metrics.py
- Mean and p95 radial error, cross-track RMSE, settling time, energy (integral of thrust^2), action rate; plotting helpers. Test on synthetic trajectories.
GATE: orchestrator runs tests, updates PROGRESS.md, STOPS.

## Wave 3 (1 agent)
### C1 Integration smoke test
Owns: tests/test_integration.py, drl/INTEGRATION.md (may fix small interface bugs in Wave 1-2 files; every fix must be listed in the report)
- Run PID baseline inside the env (sanity check). Run a short PPO (few minutes, no disturbances) and confirm reward rises. Report the numbers actually measured.
GATE: STOP. Human launches the full curriculum training in a separate terminal (not via sub-agent).

## Wave 4 (2 agents in parallel, need a trained checkpoint)
### C2 Evaluate
Owns: evaluate.py, drl/results/
- Same seeds for PID vs PPO: station keeping 120 s with wind/current/waves, square waypoint path, rough-sea scenario. Output table + plots.
### C3 Export
Owns: export_onnx.py, tests/test_onnx_parity.py
- Deterministic mean action, observation normalization baked in, hidden state I/O if recurrent. Parity test torch vs onnxruntime max abs diff < 1e-4. Run infer_replay.py on a sample CSV.

## Wave 5 (1 agent)
### D1 Docs
Owns: README.md
- Frame conventions, how to train/evaluate/export/run on the Pi, full `pytest drl/tests` result, measured results only.