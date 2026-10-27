# INTEGRATION.md — how to run the Wave 3 smoke test

Everything here is run from `E:\buoy-project\drl` with the venv interpreter
`.venv/Scripts/python.exe` (torch 2.6.0+cu124, CUDA, RTX 3050). Numbers and findings live
in `reports/C1.md`; this file is the how-to.

**No result in this directory transfers to the real vessel.** `configs/vehicle.yaml` is
placeholder parameters (TODO SYSID) and `configs/env.yaml` reward/geofence/episode values
are un-tuned.

## 1. Full suite

```
.venv/Scripts/python.exe -m pytest tests -q
```

148 tests, ~63 s on the dev box. `tests/test_integration.py` adds 5 of them:

| test | what breaks it |
|---|---|
| `test_estimator_error_small_at_first_step` | estimator not seeded with the episode pose (the pre-R3 38 m phantom error) — asserts < 5 m and < 25 % of the mean spawn distance after `reset()` and after one `step()` |
| `test_drag_factor_of_1000_sampled_episodes_inside_config_range` | drag randomisation outside `configs/env.yaml` `randomization.drag` 0.30 → [0.70, 1.30], or mass leaking back into drag |
| `test_pid_station_keeping_runs_through_the_env` | PID/env interface (shapes, clip, metric plumbing) — not control quality |
| `test_pid_waypoint_runs_through_the_env` | same for waypoint mode, plus NaN in the physics |
| `test_ppo_loop_runs_on_the_real_env` | `train.PPOTrainer` against `AsvEnv` (2 iterations, 32 envs): finite losses, correct step count |

## 2. PID baseline measurement (120 s episodes)

```
.venv/Scripts/python.exe tests/test_integration.py
```

Prints mean/p95 radial error, cross-track RMSE, settled fraction and settling time, energy
and action rate for station keeping and waypoint, each over 16 envs × 1200 policy steps
(120 s), seed 0, no disturbance, physics randomisation off. Takes ~3 min on the box.

Result, short version: the PID **does not converge** in this simulator — 0 % of envs settle,
and the radial error grows over the episode (station keeping 40.0 → 48.4 m batch mean;
waypoint 38.8 → 40.9 m). Cause is the placeholder actuator geometry, not the PID port or
the env: every body command produces ~17 N of forward thrust (measured tau for `[1,0,0]`,
`[0,1,0]`, `[0,0,1]` is `(17.32, -10.39, -1.56)`, `(17.32, +10.39, -7.80)`,
`(17.32, -10.39, +7.80)`), so the heading loop saturates every step. Details and the open
decision: `reports/C1.md` §3 and §6.

## 3. PPO smoke run (background, hard cap, no disturbance, no curriculum)

```
.venv/Scripts/python.exe train.py --config configs/train.yaml \
  --num-envs 1024 --iterations 40 --total-steps 1310720 --curriculum-level 0 --seed 0 \
  --device cuda --log-dir runs/c1_smoke --ckpt-dir runs/c1_smoke/ckpt \
  > reports/c1_train.log 2>&1 &
```

- `--iterations` is the hard cap: `iterations × rollout_steps × num_envs` steps, nothing runs
  after it. `train.py` also saves `ckpt_last.pt` at the end.
- `--curriculum-level 0` pins disturbance level 0 **and disables the env's auto-advance**.
  Without it the env raises the level every `0.6 × 1200 = 720` policy steps
  (~72 s at 1024 envs), i.e. any longer "no disturbance" run would silently get disturbance.
- Each iteration prints one line (`it`, `steps`, `reward`, `return`, `pi_loss`, `lr`,
  `steps/s`), so the log can be polled while the run proceeds; the last two lines are the
  final stats and `peak_vram_mib`.
- Measured for the command above: reward +0.21 → +0.48 (5-iteration window means +0.20 →
  +0.53), 1 310 720 steps, 69.7 s, peak VRAM 89.7 MiB allocated / 110.0 reserved.

## 4. Reading the numbers honestly

- Reward is an episode-mixed average over random spawn poses; a rising average is not a
  working controller, and no PID-vs-PPO comparison has been run yet (Wave 4).
- `AsvEnv.step` auto-resets finished envs (CONTRACT §4.5), so long traces from
  `test_integration.py` contain post-reset segments — read terminal states from `info` when
  a real evaluation is built.
- `obs.scale` clips at ±5, i.e. ±50 m for the goal terms: beyond 50 m of error the policy
  cannot see how far away the goal is. The geofence is 80 m.
