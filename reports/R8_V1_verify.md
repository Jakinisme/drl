# R8 V1 — measurement prerequisites for the reward redesign

Scope: answers three questions the R8 reward redesign depends on. **No reward, env, config or
metric file was edited by this task.** All numbers are SIMULATOR numbers from placeholder
physics (`SYSID.md`); nothing here transfers to the real vessel.

Sources: `envs/asv_env.py` (read), `evalkit/metrics.py` (read), `evaluate.py` (read),
`sim/disturbance.py` (read), `configs/env.yaml`, `configs/disturbance.yaml`, and executed code:
`runs/r7_300m/ckpt/ckpt_last.pt` (steps 299958272, iteration 4577) replayed on CPU with 8 envs.
Scratch scripts (not in the repo): `…/hermes/cache/scratch/r8_v1_speed_at_arrival.py`,
`…/r8_v1_dist_seed.py`; raw numbers in `…/r8_v1_speeds.json`.

---

## Q1 — Arrival bonus, final latch, and the real speed at arrival

### Q1.1 Exact current formula

`envs/asv_env.py:441-460` (`_reward`):

```python
450  r = (w["w_position"] * torch.exp(-d / w["pos_sigma_m"]) * final
451       + w["w_position_fine"] * torch.exp(-d / w["pos_sigma_fine_m"]) * final
452       + w["w_heading"] * torch.exp(-perr / w["head_sigma_rad"])
453       + w["w_progress"] * (dd / (self.max_surge * self.policy_dt)).clamp(-1.0, 1.0)
454       - w["w_cross_track"] * (self._xtrack() / w["xtrack_sigma_m"]).pow(2).clamp(max=1.0)
455       - w["w_near_goal_speed"] * (speed / self.max_surge).pow(2) * near * final
456       - w["w_action_rate"] * (a - a_prev).pow(2).mean(dim=1)
457       - w["w_energy"] * a.pow(2).mean(dim=1))
458  if arrive_evt is not None:
459      r = r + w["w_arrival"] * arrive_evt
```

The bonus line is **`:458-459`**: `r = r + w["w_arrival"] * arrive_evt`, with
`w_arrival = 10.0` (`configs/env.yaml:44`, `# TODO T8 placeholder, untuned`). It is a pure
one-shot constant: **no function of `d`, the speed, the heading, or the aim point.**

`speed` in the near-goal penalty is `self.state[:, 3:5].norm(dim=1)` — the true body speed
`hypot(u, v)` (ground truth, not the estimator). `near = (d < slowdown)`, `slowdown = 5.0 m`
(`configs/env.yaml:24`), and `final` multiplies it, so the near-goal speed penalty is live only
at the LAST waypoint or in station keeping.

### Q1.2 Where `final_latch` is set

`envs/asv_env.py:432-439`, `_arrive_evt`:

```python
435  d = self._goal_dist()
436  final = self._final_mask() & (self.mode < 0.5)          # final WP in waypoint mode
437  enter = final & (d < self.arrival) & ~self.final_latch
438  self.final_latch |= enter
439  return (adv | enter).to(self.dtype)
```

* `enter` is true on the **first policy step whose post-step `d` is below
  `arrival_radius_m = 2.0`** (`configs/env.yaml:23`, quoted as measured from
  `config.py:63`) with the final waypoint active; `final_latch` is a bool `[N]`
  (`:232`) cleared for the reset envs in `reset()` (`:285`). Leave-and-re-enter pays nothing.
* `adv` comes from `_update_goals()` (`:402-421`): `(self.mode < 0.5) & (self.wp_idx + 1 < self.n_wp) & (d < self.arrival)`
  — the intermediate-waypoint advance mask, evaluated on the state **after** the 5 sim sub-steps
  of the policy step (`step()`: `:473-484`, then `:486` `_arrive_evt(adv)`).
* Both masks are evaluated **on the discrete 10 Hz policy grid** (`policy_dt = 0.1 s`), not on
  the 50 Hz sim grid, and on ground-truth `state`, not the estimator.

### Q1.3 Measured speed at arrival (R7 checkpoint, CPU, 8 envs, seeds 0-3, 1 episode, level 1.0)

`speed = ||(u, v)||` at the exact step `arrive_evt == 1`, from the true state.
`wp_square` = `evalkit`/`evaluate.py` square, side 30 m, `--square-hold 10 s`;
`sk_wind` = 120 s station keeping with full disturbance.

| group | n | mean | p50 | p90 | p95 | max |
|---|---|---|---|---|---|---|
| (a) **intermediate** waypoint advance (`adv`), wp_square | **1** | 0.548 | 0.548 | 0.548 | 0.548 | 0.548 |
| (a) intermediate, sk_wind (stray event, see note) | 1 | 0.695 | 0.695 | 0.695 | 0.695 | 0.695 |
| (b) **first entry to final goal** (`enter`), wp_square | **38** | **1.002** | **0.999** | 1.274 | **1.335** | **1.468** |
| (b) first entry to final goal, sk_wind | 14 | 0.997 | 0.997 | 1.182 | 1.196 | 1.206 |
| reference: ALL policy steps, wp_square | 64000 | 0.707 | 0.723 | 1.224 | 1.293 | 1.569 |

Event counts per seed — `wp_square`: inter 0,0,1,0 and final 11,8,9,10; `sk_wind`: inter 0,0,1,0
and final 4,0,5,5.

Reading:

* **The final-goal entry is at full speed.** Median 0.999 m/s against the reward's own
  `max_surge_mps = 1.0` (`configs/vehicle.yaml:12`, "soft cap, reward/obs scaling only"),
  mean 1.002, p95 1.335 — i.e. the vehicle arrives **above** the scale the near-goal penalty
  uses, and 38/38 final arrivals are ≥ 0.50 m/s. It overshoots the 2 m radius rather than
  decelerating into it.
* **The intermediate group is not measurable from R7.** Only **one** `adv` event fired in 8
  episodes (8×2000 = 16 000 policy steps of wp_square), so its "distribution" is a single
  sample and must be read as such, not as a mean. The cause is structural, not a bug in the
  measurement: `SquarePath.__call__` (`evaluate.py:450-482`) writes `env.mode = holding` and
  keeps `env.goal_xy` on the **active corner**; when the vessel arrives, `SquarePath` switches
  the env to station keeping (`mode 1`) and `_update_goals()` unconditionally short-circuits on
  `self.mode < 0.5`. `arrive_evt` then returns 0; the bonus is never paid for corners 1-3.
  So under the current eval harness the arrival bonus is effectively **only ever paid at the
  final corner**, at ~1 m/s.
* The stray `adv` events in `sk_wind` (1 of 4 seeds) come from station-keeping episodes that
  spawn in waypoint mode (`station_keeping_fraction: 0.3` → some envs are `mode 0` for the whole
  station-keeping scenario at level 1.0). They are not part of either requested group and are
  reported only so the counts add up.

### Q1.4 One sentence: why flying through at speed pays the same as stopping

Because `w_arrival * arrive_evt` is a constant 10.0 paid on the single policy step on which
`d` first drops below 2.0 m, and nothing about that payment depends on the speed, the heading,
or how centred the crossing is — while the only terms that do care about stopping near the goal
(`w_near_goal_speed * (speed/max_surge)^2`, weight 0.1, and the two kernels once `final` is
true) are worth at most 0.1·(1.0)² = 0.1 per step for the few steps the vessel is inside the
5 m slowdown radius, i.e. far less than the 10.0 the constant pays for the crossing itself.

---

## Q2 — What `settled_pct` and `settling_time_s` actually are

### Q2.1 Definitions, verbatim from `evalkit/metrics.py`

`settling_time(t, err, tol, hold_s=5.0)` (`:98-112`):

```python
105  t, e = _np(t).ravel(), _np(err).ravel()
106  if t.size == 0 or t.size != e.size:
107      return float("inf")
108  bad = ~(e <= tol)                       # NaN -> True -> not settled
109  idx = 0 if not bad.any() else int(np.flatnonzero(bad)[-1]) + 1
110  if idx >= t.size or (t[-1] - t[idx]) < hold_s:
111      return float("inf")
112  return float(t[idx] - t[0])
```

* `tol` is in **metres of radial error to the CURRENT `goal_xy`** — `err = radial_error(xy, goal_xy)`
  (`summarize`, `:189`), i.e. `||xy_t − goal_xy_t||₂`, and `goal_xy` is the *active* goal of that
  sample. It is **not** distance to the planned path and **not** a speed.
* `hold_s` (default **5.0 s**, `evaluate.py:120` `HOLD_S = 5.0`, CLI `--hold-s`) is the minimum
  duration the in-band tail must last. Implementation: take the LAST out-of-band sample, step
  one past it; if the remaining tail is shorter than `hold_s` → `inf`. Equivalent to "stays inside
  `tol` for the rest of the trace, which must be ≥ `hold_s` long".
* `<=` not `<`: exactly `tol` counts as in band. NaN counts as out of band (`~(`NaN` <= tol)` is True).
* Result is seconds from `t[0]`, or `inf`; `evaluate.py` never prints `inf`, it prints `settling_time_s`
  as `inf` and `settling_p95_s` from `np.percentile` (`evaluate.py:687-688`), which on any `inf`
  entry yields `inf`.
* `summarize` (`:169-212`) picks `tol` in this order: explicit argument → `traj["tol_m"]` →
  **1.0** (`:194`). It computes `settling_time_s = settling_time(t, err, tol, hold_s)` (`:208`).

`settled_pct` (`evaluate.py:662`, inside `score()`):

```python
662  m["settled_pct"] = 100.0 if m["settling_time_s"] < np.inf else 0.0
```

So per episode it is strictly **binary — 100.0 or 0.0** — and the printed/aggregated number is
the mean over episodes (`aggregate`, `:678-690`), i.e. **the percentage of episodes whose
`settling_time_s` is finite**. There is no partial credit and no notion of "mostly settled".
`tol` reaches it only through `--tol` (default `TOL_M = 1.0`, `evaluate.py:119`, documented as
the GPS noise floor from `configs/sensors.yaml` `pos_noise_m: 0.4` + hold/latency, **not** the
2.0 m env arrival radius).

### Q2.2 Which episodes count in `wp_square`, given there is no hold at the end of the lap

Every complete episode counts — `summarize` is called once per (episode, env) in `score()`
and `aggregate` averages all of them; `clean_episode` (`evaluate.py:693-701`) drops only
incomplete (geofence-exit) episodes. In `wp_square`:

* `t` is the per-episode policy clock `np.arange(n)*policy_dt` (0.1 s), full episode
  (`--episode-s`, default 200 s → 2000 samples).
* The lap is 4 legs with a `--square-hold 10 s` station-keeping dwell at **each** corner
  (`SquarePath`, `:409-482`) — including the last one, where it stays mode 1 on `c[4]`
  (`self.done`), so there **is** a terminal hold of `hold_s = 10 s` in `wp_square`.
  10 s ≥ 5 s `hold_s`, so the metric's own hold requirement is satisfiable in principle.
* But that dwell is the *only* place the trace can be in band, and the trace ends in it, so
  `settling_time_s` for a completed lap is `t[idx] − t[0]` measured from the **final** hold —
  it is effectively "time to the last corner", not "time to settle at each corner". A policy
  that is out of band once anywhere after the last dwell began gets `inf`.
* Note the inconsistency with CONTRACT §3.4's env semantics: in the env a `mode 1` transition
  makes `arrive_evt` 0, and `w_near_goal_speed` becomes live; so the dwell is trained-for in
  training only via `station_keeping_fraction`, while in eval it is injected mid-episode
  (already flagged as a train/eval mismatch in `reports/R4.md` §7(b)).

### Q2.3 Can a pass-through ever register as settled?

**Only if the pass-through is followed by ≥ `hold_s` seconds inside `tol` at the end of the
trace.** The function is a tail condition (`:108-111`), not a dwell time: it never asks how
long the vessel stayed inside the radius on entry, and it never looks at speed. Two
concrete consequences:

1. A pure pass-through with **no** terminal hold (any `wp_square` episode that ends mid-leg, or
   any episode whose final dwell is shorter than 5 s) can never be settled: the last in-band
   window is far from `t[-1]` → last bad sample late → `idx >= t.size` or the tail is short →
   `inf` → `settled_pct = 0`.
2. A pass-through that flies into the 2 m radius at 1 m/s and then holds still for the
   remaining ≥ 5 s **does** register as settled — the entry speed is invisible to the metric.
   Conversely, a controller that *slows down* nicely but drifts 1.2 m off in the last 5 s
   scores 0.

This is exactly the "`settled_pct` 0-53 %, `settling_time_s = inf`" signature: the metric
measures *final* in-band residence, and R7 is not producing it (median arrival speed
0.999 m/s, Q1.3). Q1's arrival measurement and Q2's settling measurement therefore look at
two different properties; neither alone tells you "R7 brakes".

One more precision point for whoever edits `tol`: because `err` is distance to the **active**
`goal_xy`, an intermediate corner only contributes in-band samples while it is the active goal;
once `SquarePath` switches the goal away, the same position error is measured against a
corner 30 m away. `evalkit.metrics` currently has **uncommitted R8 edits from another agent**
(`git status`: `M evalkit/metrics.py`, adding `sat_frac`/`reversal_rate`/`arrival_speed_mean_mps`/
`hold_rms`). The definitions quoted above are the `settling_time`/`summarize` lines, which that
diff does not touch.

---

## Q3 — Is the disturbance seeding constant w.r.t. `--seed`?

**Yes. Confirmed, and measured as exactly 0.0 difference.**

### Code path

`sim/disturbance.py:92-93`:

```python
92  self._gen = torch.Generator(device=self.device)
93  self._gen.manual_seed(int(cfg.get("seed_offset", 0)))
```

`_rand()` (`:98-99`) is the **only** source of the per-episode draws, and all of them
(`wind_speed`, `wind_dir`, `wave_dir`, `wave_phase`, `wave_phase_yaw`, `:117-121`) come from
`self._gen` in a fixed order on every `reset()`.

`cfg` is the disturbance YAML dict and nothing else:

* `evaluate.py:577`: `kw = {"disturbance_cfg": rough_disturbance(rough_scale)} if scen["rough"] else {}`
  → `rough_disturbance()` (`:166-173`) = `configs/disturbance.yaml` with amplitudes × `scale`;
  **it never touches `seed_offset`**, and for the non-rough scenarios no `disturbance_cfg` is
  passed at all, so `envs/asv_env.py:176` does `Disturbance(disturbance_cfg or load_disturbance_cfg(), …)`
  = the raw YAML.
* `envs/asv_env.py:113` takes `seed` and uses it **only** for the env's own generator:
  `:216-217  self._gen = torch.Generator(device=self.device); self._gen.manual_seed(int(seed if seed is not None else cfg.get("seed", 0)))`.
  That generator drives spawn pose, `mode`, waypoints, randomisation and sensor draws — **not**
  the disturbance, which already exists (constructed at `:176`, before `self._gen` at `:216`).
* `evaluate.py:810`: `seed = int(args.seed) + 1000 * si + k` — reaches only `AsvEnv(seed=...)`.
* `configs/disturbance.yaml:31`: `seed_offset: 0`.

`train.py` is the same: `train.py:100` constructs the env with `seed=int(cfg["seed"])`, and
`train.py:108/120` seed the global torch RNG and the trainer's own generator; there is no
`seed_offset` override and no `Disturbance` construction anywhere in `train.py`. So in
**both** training and evaluation the wind/wave RNG is `manual_seed(0)` and advances only by
`reset()` calls, in call order.

### Measured

`AsvEnv(num_envs=8, device="cpu", seed=S)`, `set_curriculum(n_levels-1)`, comparing the live
`env.disturbance` tensors after construction:

| quantity | `S=0` vs `S=1234` (same env index) |
|---|---|
| `wind_speed` | max abs diff **0.0** |
| `wind_dir` | **0.0** |
| `wave_dir` | **0.0** |
| `wave_phase` | **0.0** |
| spawn `x,y,psi` (contrast) | **45.15** — spawn DOES change with seed |

`wind_speed` per env at every seed (identical rows) and `wind_dir` at seed 0:

```
wind_speed  [5.5382, 1.6308, 5.4664, 6.0228, 6.8635, 5.4956, 0.0411, 1.4052]  m/s
wind_dir    [269.88, 217.67,  39.58,  76.35, 349.33, 301.29, 101.52, 134.70]  deg
```

The four seeds `evaluate.py` actually uses for a scenario (`--seed 0 --seeds 4` → 1000, 1001,
1002, 1003) give a **within-env spread across seeds of exactly 0.0** for `wind_speed`, confirming
R6 §3's zero-variance signature from the other direction (there: reading the dumped CSV; here:
reading the live tensors of freshly constructed envs).

The knob that *does* change the realisation is `seed_offset`: with
`disturbance_cfg["seed_offset"] = 7` and `seed=0`, `wind_speed` becomes
`[4.2224, 7.4848, 1.1758, 1.6148, 5.8073, 3.7850, 0.3548, 1.2314]` m/s.

### Statement

**Changing `--seed` in `evaluate.py` does not change the wind/wave realisation for a given env
index.** It changes the spawn pose, waypoint list, domain randomisation and sensor noise, but env
`i` always sees the same wind speed/direction and wave phases in every seed of every scenario,
and the same ones it saw during training (which used the same `manual_seed(0)` for the same
reason). The only lever is `seed_offset` in the disturbance config, which is never overridden
by `evaluate.py` or `train.py`.

### Caveat on "for a given env index"

The draw order is `reset(k)` → `k` new samples appended to the generator stream, so the value an
env index gets depends on **which reset calls happened before it and how many envs each one
covered**. With `FastEnv`/`AsvEnv` auto-reset inside `step()` (`:521-523`,
`self.reset(ids)` for the finished rows only), the stream advances by the number of finished
envs each step, so after the first mid-episode auto-reset the per-index map is a function of the
*episode history* as well as the index — still seed-invariant, but no longer a fixed lookup
table from env index alone. The seed-invariance claim above holds either way.

---

## What is NOT measured here

* No distribution of speed at **intermediate** waypoint advance: only 1 event exists in 8 R7
  episodes, because `SquarePath` turns intermediate arrivals into station keeping and
  `_update_goals` is gated on `mode < 0.5`. Reported as n=1, not as a distribution.
* No PID comparison, no training run, no reward modification — out of this task's scope.
* `settled_pct` was not re-run through `evaluate.py` here; it is quoted from the code path
  (`evaluate.py:662` + `evalkit/metrics.py:98-112`, `:169-212`) and cross-checked against the
  recorded `results_r7/summary.json` values (0.0 / 43.75 / 53.125 per scenario).
* The working tree has an **uncommitted** `evalkit/metrics.py` from another agent (R8 metrics
  work); it does not change `settling_time`, `summarize`'s `tol` resolution or the keys read here.
