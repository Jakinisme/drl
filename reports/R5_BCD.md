# R5 items B, C, D — training loop (terminated/truncated, timeout bootstrap, resume state, episode metrics)

Owner: train.py, envs/asv_env.py (info keys only), tests/test_train_loop.py (new).
Reward untouched (frozen by R4). PPO hyperparameters and defaults untouched.

## 1. How `env.step()` produces `done` (envs/asv_env.py:466-468, 490-493)

```
timeout  = self.ep_step >= self.max_steps          # time limit, int compare -> bool [N]
outside  = self.state[:, :2].norm(dim=1) > self.geofence   # geofence exit -> bool [N]
done     = timeout | outside
```
* `max_steps` = `cfg["max_steps"] or round(episode_s * policy_hz)` = 1200 at the shipped config
  (env.yaml:9-10). `ep_step` is per-env (reset to 0 in `reset()`), so episodes end staggered.
* Geofence -> **true termination** (the hull is outside the fence: the episode is over, there is
  no future to bootstrap).
* Time limit -> **truncation** (the MDP is cut by a clock, not by the agent's state; the value
  function is still meaningful after it, so GAE must bootstrap from V(s_T+1)).
* Nothing else terminates: no arrival/collision/goal-complete termination exists in the env.
* `timeout` and `geofence` can both be 1 on the same env in the same step (left the fence on the
  step the clock expired). Precedence chosen: **terminated wins, truncated = timeout & ~outside**.

## 2. Is the pre-reset observation recoverable? Where does the reset happen?

`step()` builds `sample`/`obs` from the *terminal* state (lines 470-473) and only then, if
`done.any()`, overwrites the finished rows with `self.reset(ids)` post-reset observations
(line 492). So the pre-reset (terminal) obs **is** the value in the returned tensor before that
line, and after the call it is lost — `self.obs_buf` holds the new episodes' history.

Decision: **no change to the reset location**. Auto-reset stays inside `step()` (CONTRACT §4.5).
For item B the terminal observation is captured at one place: `env.step()` computes
`info["terminal_obs"] = obs.clone()` right before the reset (the rows of finished envs are the
terminal ones, all other rows are the normal post-step obs). That is 1 line + 1 info key in the
env; the trainer then evaluates the critic on it under `torch.no_grad()` and stores
`gamma * V(next)` as the reward of that step. No rollout-side `infos` bookkeeping, no change to
`collect()`'s data flow other than the bootstrap column.
Memory cost: one extra [N,150] clone per step (~1.2 MB at 2048 envs) that is immediately dropped.

## 3. `Trainer.save()` / `load()` and the env state that is NOT saved

Saved today (`train.py`): `model.state_dict()`, `opt.state_dict()`, `steps` (trainer's own
counter, `+= T*N` per iteration), `iteration`, `cfg` dict. `load()` restores model/opt/steps/it
and then hard-resets the env (`self.env.reset()`).

NOT saved, all of it lives in the env:
* `env.total_steps` — number of `step()` calls; drives the **curriculum auto-advance**
  (`level_index = start_level + total_steps // level_up_steps`, `_sync_level`).
* `env.level_index` (i.e. the curriculum level actually applied to `disturbance`),
  `env._level_override` (set by `--curriculum-level`).
* `env.t_loop` (control-loop clock, also drives the periodic disturbance sampling phase),
* per-env episode state: `ep_step`, `wp_idx`, `final_latch`, randomisation draws, sensor rings.

Consequence before this change: `--resume` restarted the curriculum at `start_level` and
`total_steps=0` (level regression), while `self.steps` (lr schedule) continued. Fixed in C:
`total_steps` + level are stored and restored; the per-env episode state is intentionally NOT
saved (resume starts fresh episodes — documented, not fixed here).

## 4. What was changed

### envs/asv_env.py
* `info["terminated"]`, `info["truncated"]` as bool [N] with the precedence above; `done`,
  `info["timeout"]`, `info["geofence"]` and every other key unchanged.
* `info["terminal_obs"]` — [N,150] clone taken before the auto-reset (§2).

### train.py
* `collect()`: reads `info.get("terminated")/("truncated")`; for `truncated & ~terminated`
  computes `value_only(terminal_obs)` under `no_grad` and adds `gamma * V` to that step's reward;
  `b_done` stays 1 for all `done` envs (no GAE bleed). Envs without the keys (dummy envs, old
  envs) fall back to the previous behaviour — plain reward, `b_done = done`.
* `_gae` unchanged (still `live = 1 - b_done`), `update()` unchanged.
* `save()`: extra keys `env_total_steps`, `env_level_index`, `env_level_override`;
  `load()`: restores them when present, otherwise prints one `[warn]` and keeps env defaults.
  `_apply_env_state()` sets `env.total_steps` + `_level_override` then `_sync_level(force=True)`.
* New per-iteration episode stats (`collect()`), over episodes whose `done` fired **this
  iteration**: `ep_return_mean`, `ep_len_mean` (NaN when no episode finished), `arrivals_per_ep`
  (`sum(info["arrive_evt"]) / n_finished` — the env's own arrival counter, no `info`
  reshape bookkeeping), `action_std_mean` (`exp(log_std).mean()`), plus the raw counters
  `episodes_finished`, `arrivals`. Logged with the existing `train/...` tag style and added to
  the one-line console log.
* `arrivals_per_ep` is NaN when no episode finished in that iteration (arrival events do happen
  on non-final steps, but with no completed episode there is nothing to divide by).
* `--curriculum-level` still applied after `load()`, so it still overrides the checkpoint.

## 5. Tests (tests/test_train_loop.py)

| test | what it pins |
|---|---|
| `test_info_keys_real_env` | terminated/truncated exist, bool [N], `done <= terminated+truncated`, timeout==1 ⊆ truncated, geofence==1 ⊆ terminated |
| `test_gae_bootstrap_truncated_vs_terminated` | 1 env, constant r, critic frozen to `value_bias=V`, `lam=0` (per-step TD, so the assertion is exact and not recursive): truncated last step has `b_rew == r + g*V` and `b_done == 1`, `_ret[last] == r + g*V`; terminated last step has `b_rew == r`, `_ret[last] == r`; identical model/env otherwise |
| `test_truncation_does_not_bleed_across_episodes` | `b_done` stays 1 on truncated steps (GAE `live` mask is 0) |
| `test_episode_stats_and_nan` | ep_return/ep_len/arrivals finite when episodes finish, NaN when none finish |
| `test_resume_restores_total_steps_and_level` | save -> new trainer -> load: `env.total_steps`, `_level_override`, `level_index` restored, plus `steps`/`it` |
| `test_old_checkpoint_without_env_state_warns` | checkpoint with the keys stripped loads, warns once, no crash, keeps env defaults |
| `test_curriculum_override_wins_over_checkpoint` | `set_curriculum(0)` after `load()` wins |
| `test_train_logs_new_metrics` | `train()` history holds the new keys with correct values and the console line carries them |

## 6. Results (measured)

```
$ .venv/Scripts/python.exe -m pytest tests/test_train_loop.py -q
9 passed, 1 warning in 3.40s        # warning = torch adv.std() dof at T=N=1 in the bootstrap test
$ .venv/Scripts/python.exe -m pytest tests/test_train_loop.py tests/test_env.py -q
26 passed, 1 warning in 7.07s
```

Smoke (`--device cuda`, 256 envs, 20 iterations, 163840 steps, `--curriculum-level 0`, seed 0):
the shipped `episode_s: 120` (= 1200 policy steps) cannot finish one episode inside 20 iterations,
so a short-episode config was derived **without editing `configs/`**:
`runs/r5_smoke_bcd/env_short.yaml` = `configs/env.yaml` with `max_steps: 24` (`episode_s: 2.4`,
geofence 500 m so episodes end on the time limit, not the fence) and
`runs/r5_smoke_bcd/train_short.yaml` = `configs/train.yaml` with `env_config` pointing at it.
No hyperparameter changed.

```
it    0 steps     8192 ... ep_ret +3.351 ep_len 24.0 arrivals/ep 0.000 std 0.607
it   19 steps   163840 reward +1.1294 return +21.881 ... ep_ret +5.482 ep_len 24.0 arrivals/ep 0.003 std 0.601
last: {'ep_return_mean': 5.48225, 'ep_len_mean': 24.0, 'arrivals_per_ep': 0.003125,
       'action_std_mean': 0.601264, 'episodes': 320.0, 'arrivals': 1.0, 'v_loss': 4.234274, ...}
peak_vram_mib 36.9 reserved 50.0      (~3.16k steps/s)
```
No NaN: every iteration has `ep_ret`/`ep_len`/`arrivals/ep` finite because the time limit fires
every 24 steps. `ep_len_mean` is exactly 24.0 as expected (all episodes truncate, none hit the
fence). Low `arrivals_per_ep` is expected with 24-step episodes at level 0, not a bug.

Resume check on the same checkpoint:
```
ckpt: steps 163840 iteration 20 env_total_steps 640 env_level_index 0 env_level_override 0.0
--resume ... -> it 20 steps 172032 ... (no [warn], iteration counter continues, override kept)
```

## 7. Open problems / not done

* Resume still starts fresh episodes: `ep_step`, waypoint pointer, `final_latch`, `t_loop` and the
  randomisation draws are not checkpointed. Level + `total_steps` are the only curriculum state.
* The bootstrap uses the critic on the *terminal estimator state*; the env's terminal obs is
  produced from `meas` of the last substep, which is the correct `s_T` per CONTRACT§4.5.
* No real-vessel claim: everything above is simulation + unit tests.
