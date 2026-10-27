# R5 item E — wp_square evaluation harness fix

Owner: evaluate.py, tests/test_square_path.py, this report. Nothing else touched.
All numbers are SIMULATOR numbers with placeholder physics (SYSID.md). Nothing here says
anything about the real vessel.

## 1. Finding 4 — CONFIRMED (code read + data read)

R4 `SquarePath.__call__` did this, unconditionally, for every env:

```python
self.done = self.done | leave                      # done set on EVERY leave
holding   = self.done | (self.hold > 0)
goal_xy   = torch.where(holding.unsqueeze(1), c[:, 1], ...)   # pinned to corner 1
```

So after the FIRST dwell completes (`leave` on leg 0):

* `self.done` is set and **never reset** anywhere in the class (only the constructor writes it),
* `holding` is therefore permanently True,
* `goal_xy` is pinned to `c[:, 1]` (corner 1, 30 m east of each spawn) for every env for the
  rest of the episode, and `env.mode` is frozen at 1 (station keeping).

Consistent evidence in `results_r4_5m`: 64 complete episodes (32 ppo + 32 pid across
wp_square seeds 0..3, 8 envs) and **none** exceeds `waypoints_reached == 1`
(`awk -F, '{print $16}'` over the 8 per-episode CSVs: ppo 27x1, 5x0; pid 14x1, 18x0).
`waypoint_progress` credits a corner only while it is the ACTIVE goal (`reach_tol` 5 m), so a
policy that visits corners 2..4 while the goal is pinned to corner 1 scores 0 for them. That is
the mechanism, not a metric artefact.

## 2. What changed in evaluate.py

1. `done` = "lap finished" only: set when the dwell of the FINAL corner (`nxt == 4`) completes.
   `idx` is not advanced past 3, `mode` stays 1, goal stays on `c[4]`.
2. Station-keeping goal is the ACTIVE corner: `c[idx+1]` while that corner's dwell runs,
   `c[idx]` after `leave`, travelling goal `c[idx+1]`. `c[:, 1]` is gone.
3. After `leave` on a non-final corner: counter restarts at 0 and `holding` is False, i.e.
   `mode` 0 toward the next corner; the next dwell starts at that corner.
4. `_path` = `path.c.cpu().numpy()` — **5 points, corner 0 (spawn) included**.
   `SquarePath.waypoints()` still returns corners 1..4, so `waypoints_reached` semantics are
   unchanged.
5. Dwell hysteresis: counter starts below `arrival`, keeps counting while below
   `dwell_stay_factor * arrival`, resets beyond that. New flag
   `--dwell-stay-factor FLOAT` (default 1.5); `1.0` reproduces the pre-R5 radius test exactly
   (both thresholds collapse to `arrival`).
6. New flag `--episode-s FLOAT`, applied to `wp_square` ONLY (default = SCENARIOS 200.0).
   `sk_wind` / `rough_sea` / `sk_pid_soft` untouched.
7. `score()` still pops `time_to_waypoint` out of the aggregate path; the per-episode CSV gets
   `time_to_waypoint_1..4` (s, `nan` when never reached) appended AT THE END via `tw_columns()`.
   Existing columns, their order, `summary.csv` and `summary.png` are unchanged.
8. `--hold-s` untouched (still the evalkit SETTLING hold, default 5.0). The corner dwell is
   `--square-hold`. `summary.json` now also carries a `square` block with the effective
   `episode_s`, `dwell_stay_factor`, `square_hold_s`, `square_side_m`.

## 3. IMPORTANT — earlier `cross_track_rmse` values are NOT comparable

Until now `_path` was `path.c[:, 1:, :]`, i.e. a POLYLINE WITHOUT LEG 1 (spawn -> corner 1).
Every `cross_track_rmse` published earlier, notably **PPO 4.75 m and PID 10.8 m in
`results_r4_5m`**, was measured against that truncated 4-point polyline, so it under-counted
the distance during leg 1 (the whole first 30 m leg was outside the polyline, clamped to its
first vertex). New values below use the true 5-point closed lap and are lower for the same
behaviour; do not compare old and new cross-track numbers.

## 4. Tests — `tests/test_square_path.py` (NEW)

```
.venv/Scripts/python.exe -m pytest tests/test_square_path.py tests/test_obs_parity.py -q
..........                                                               [100%]
10 passed in 22.07s
```
(also re-ran `tests/test_evaluate.py` unchanged: 6 passed.)

* (a) scripted teleport through all 4 corners: `waypoints_reached == 4`, `lap_completed True`,
  `lap_pct 100`, finite `time_to_waypoint_1..4` — for `--square-hold` 0 s AND 10 s.
* (b) parked 2.5 m off corner 1: `--dwell-stay-factor 1.0` never advances (2.5 > arrival 2.0);
  1.5 advances after one touch < 2.0 m then staying at 2.5 m (< 3.0 m), and resets beyond
  3.0 m. Same trace at 1.0 never leaves.
* (c) after the first `leave`, `done` is False, `mode` is 0 and the goal is corner 2 (not the
  pinned corner 1); on the next dwell `goal` is `c[2]`.
* (d) `_path` shape (N, 5, 2), first point == the spawn sample, closes on the spawn, 30 m legs,
  `_waypoints` still 4 points starting at corner 1, `tw_columns()` keys/NaN behaviour.
* `tests/test_obs_parity.py` passes unchanged.

## 5. Real re-evaluations — `runs/r4_5m/ckpt/ckpt_last.pt`, wp_square, `--seeds 1 --envs 4`

Same checkpoint, same seeds, same 200 s, only the harness differs. 4 envs per run.

| run | controller | waypoints_reached | lap_pct | cross_track_rmse | radial mean |
|---|---|---|---|---|---|
| results_r5_hyst10 (`--dwell-stay-factor 1.0`) | ppo | 1.50 | 37.5 % | 0.94 m | 7.29 m |
| results_r5_hyst10 | pid | 0.25 | 6.2 % | 9.79 m | 32.4 m |
| results_r5_hyst15 (default 1.5) | ppo | 2.00 | 50.0 % | 1.48 m | 9.81 m |
| results_r5_hyst15 | pid | 0.00 | 0.0 % | 10.3 m | 32.8 m |

Per-env `time_to_waypoint_1..4` (s, `nan` = never reached), PPO:

* hyst1.0: `56.5,-,-,-` / `-,-,-,-` / `70.5,-,-,-` / `40.2, 86.8, 136.0, 177.7` (full lap)
* hyst1.5: `77.6,-,-,-` / `-,-,-,-` / `71.4, 126.6, 163.1, -` / `40.3, 87.4, 129.1, 171.0`
  (full lap)
* PID: all `nan` at hyst15; one corner at hyst1.0 (197.8 s).

Reading: with the pinned-goal bug gone the PPO's lap counter now moves past 1 (1.5 and 2.0 of
4 corners on average, one env completing the lap in both runs), and the hysteresis flag changes
which corners are credited. PID still laps nothing: it parks far outside the 2 m arrival
radius (radial mean ~32 m), which the counter correctly refuses to credit.

## 6. Open problems

* 4 envs x 1 seed is a smoke sample, not a measurement. Nothing here is a performance claim.
* One PPO env (env 1) scores 0 waypoints in both runs with a radial mean of 4.5 m: it sits
  ~4.5 m from its current goal and never enters the 2 m arrival radius. Not a harness bug —
  the counter correctly refuses to credit a corner never touched (`time_to_waypoint_*` all
  `nan`).
* `--square-hold 10` s at 4 corners = 40 s of the 200 s episode in station keeping; with
  30 m legs the lap needs roughly 4 x (30/0.55 + 10) ~ 260 s at the PID's own waypoint speed,
  so a 200 s episode cannot complete it at PID speeds. Use `--episode-s` for a longer scenario.
* `cross_track_rmse` discontinuity vs R4 (see §3): old and new values are different quantities.
* The env artefact documented in the module docstring (last obs of an episode carries the NEW
  goal) is unchanged; it is env-side (`envs/asv_env.py`), outside this task's ownership.
