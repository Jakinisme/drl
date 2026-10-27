# R5-A — Desynchronise episode phase (training only)

## Change

`envs/asv_env.py` — new method, no other edit:

```python
def desync_episodes(self, generator: torch.Generator) -> None:
    """Training only (R5-A): stagger episode phase ..."""
    self.ep_step = torch.randint(0, self.max_steps, (self.num_envs,),
                                 device=self.device, generator=generator)
```

`train.py`, `PPOTrainer.__init__`, immediately after `self.obs = self._to(self.env.reset())`:

```python
desync = getattr(self.env, "desync_episodes", None)
if desync is not None:
    desync(self.gen)          # self.gen = trainer's own seeded generator
```

Notes:

- Only `ep_step` is written. `state`, `goal_xy`, `wps`, `hold_xy`, `psi_goal`,
  `prev_action`, `obs_buf`, `meas` and the per-env randomisation are untouched, so the
  first observation, the goal and the episode's domain-randomisation draws stay the ones
  that `reset()` produced. Episode phase is the only thing that moves.
- The draw comes from the trainer's generator (`PPOTrainer.gen`, seeded with `cfg["seed"]`
  at line 113), never from `env._gen`, which every physics/randomisation draw uses. Runs
  stay reproducible: same seed -> same phase offsets; and the env RNG stream is identical
  to a non-desynced run (asserted in the test).
- `reset(env_ids)` was not touched, so `Ep_step == 0` after any reset; `evaluate.py` builds
  its own env and never calls `desync_episodes`, so its episodes are still full length.
- `getattr` guard: the dummy `ScriptEnv`/`DummyEnv` used by `tests/test_train_smoke.py`
  and `tests/test_train_loop.py` has no such method and keeps working unchanged.
- `PPOTrainer.load()` does its own `env.reset()` without desync — intentional: a resumed
  run's first rollout is phase-aligned for at most one episode, then spread by the
  auto-reset. Adding a second desync there would also desync the fresh-`reset()` test
  expectations for no measurable gain.

## Verification (real runs, `.venv` Python, CPU, `tests/test_desync.py`)

```
$ ./.venv/Scripts/python.exe -m pytest tests/test_desync.py -q
5 passed in 31.97s
```

Five tests, each proving one claim:

1. `test_step_count_spans_range` — distribution over 64 envs: `0 <= ep_step < max_steps`,
   min in the first decile, max in the last decile, all 10 deciles of
   `[0, max_steps)` populated with counts within 25 % of `N/10`.
2. `test_desync_leaves_state_goals_obs_and_physics_rng_alone` — state / goal / waypoints /
   `obs_buf` / `meas["est"]` bit-identical before and after, `env._gen.get_state()` bit
   identical (physics stream untouched), and repeated calls with the same seed give the
   same `ep_step` (reproducibility).
3. `test_timeouts_land_on_many_step_indices` — 64 envs, constant zero action, `1.5 *
   max_steps` policy steps: the set of step indices at which an env timed out has
   `>= 5` distinct members (before this change it would be exactly `{max_steps}`).
4. `test_reset_ids_zeroes_step_count` — after desync, `reset(arange(8))` leaves
   `ep_step[:8] == 0`; a later full `reset()` leaves every `ep_step == 0`.
5. `test_trainer_calls_desync_once_after_reset` — a `PPOTrainer` with the real env on CPU
   has `0 < ep_step.max() < max_steps` straight after construction, and `collect()` still
   runs on the desynced env.

Measured at production scale (2048 envs, `max_steps = 1200`, seed 0):

```
max_steps 1200 min 0 max 1199 mean 614.9
deciles [178, 190, 198, 218, 211, 199, 221, 193, 215, 225]
reset(ids) all-zero: True
```

Deciles are flat within ~±12 % around the expected 204.8 — uniform, as intended.

## Regression runs

```
$ ./.venv/Scripts/python.exe -m pytest tests/test_env.py -q
17 passed in 6.14s

$ ./.venv/Scripts/python.exe -m pytest tests/test_desync.py tests/test_env.py tests/test_train_loop.py -q
FAILED tests/test_train_loop.py::test_episode_stats_nan_and_finite - assert 2...
FAILED tests/test_train_loop.py::test_curriculum_override_wins_over_checkpoint
FAILED tests/test_train_loop.py::test_train_logs_new_metrics - assert (False)
3 failed, 28 passed, 1 warning in 29.66s
```

The three `test_train_loop.py` failures are **not caused by this change**: I copied
`train.py` to a scratch module with the desync hook stripped out and ran that file against
it — identical result, `3 failed, 6 passed`. Those tests belong to the R5-B/C train-loop
work (episode-stat logging, truncation bootstrap, curriculum override) and another agent
owns that file; I did not touch it.

Hazard found while checking: an earlier run of the same file showed `6 failed, 3 passed`
because the sibling agent was mid-write on `train.py` at that moment. Re-running after it
settled gave the stable `3 failed`. If the orchestrator's final suite shows more failures
in that file than these three, re-run before attributing them.

## Not done / open

- Reward (R4) untouched, as instructed.
- A partial desync is not offered: every env starts `ep_step` uniform, incl. 0, which is
  what breaks the common-mode timeout. First-iteration batches are therefore made of
  envs that are mid-episode but whose state is a fresh spawn — physically consistent for
  this env (spawn pose is not a function of `ep_step`), and the only cost is that the
  first rollout's `ep_len_mean` is noisy.
- No hardware claim: this is simulator bookkeeping only.
