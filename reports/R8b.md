# R8b — instruments (task A1 phase 1)

Phase 1 of task A1. Instruments only: no training, no decision rule applied here.

---

## 1. `hold_rms_m` / `hold_p95_m` NaN on every sk_wind episode — root cause

The brief listed candidates: `info["mode"]`, `goal_xy` at the last sample, the `good` mask,
the auto-reset sample, window clipping. **None of them was wrong.**

Measured on a real 8-env `sk_wind` rollout of `runs/r8_A` at 300 s, the per-env final mode
was:

```
final mode per env: [0. 0. 1. 0. 0. 0. 0. 1.]
```

Six of eight envs ended **not** in station keeping. `final_hold_window` therefore returned
`(-1, -1)` for them, correctly, and `hold_rms_m` was nan. On the two envs that did end in
station keeping the metric already worked:

```
env2 (SK):  hold_rms -> (2.5175, 3.1917)
```

### The actual defect

`SCENARIOS` pins `mode` (1 = station keeping, 0 = waypoints) in `evaluate.py`, but **no code
consumed it**. `grep -n "station_keeping_fraction\|sk_frac"` over `evaluate.py`,
`evalkit/*.py` and `tools/gate_policy.py` returned nothing. So `sk_wind` ran at the training
mix from `configs/env.yaml`:

```yaml
episode:
  station_keeping_fraction: 0.3
```

About 30 % of `sk_wind` episodes were station keeping and 70 % were waypoint episodes with no
final hold. The scenario never forced the mode it advertised. The metric was right; the
scenario was wrong.

### Fix

New `scenario_episode_cfg(scen)` in `evaluate.py` maps `mode` onto
`episode.station_keeping_fraction` (0.0 or 1.0). `run_rollout` now builds the env from it.
Scenarios that do not declare a `mode` keep the config value, so nothing else changes.

After the fix, a real `sk_wind` rollout gives:

```
final mode per env: [1. 1. 1. 1. 1. 1. 1. 1.]
hold_rms_m 2.5175   hold_p95_m 3.1917
```

### Test

`tests/test_r8b_hold_wiring.py`, 4 cases, real rollout on `runs/r8_A` — no fakes, because the
bug lived in the seam between scenario definition and env construction:

1. `sk_wind` forces station keeping on every env.
2. `hold_rms_m`/`hold_p95_m` finite on **every complete** sk_wind episode.
3. `wp_square` still reports nan where there is genuinely no final hold (the fix must not
   invent a hold).
4. `scenario_episode_cfg` maps mode -> fraction and leaves modelless scenarios untouched.

`4 passed in 22.97s`.

### Consequence the human must know

**Every sk_wind number measured before this fix is not comparable with numbers after it.**
That includes the R7 reseeded reference in `results_r7_reseeded` (`sk_wind` PPO
`settled_pct 15.6`, `radial_error_mean 18.96`) — it was measured on a scenario that was ~30 %
station keeping. Those numbers were not silently adjusted; they are flagged here and will be
flagged in `AUTO_FINAL.md`. Phase 2 measurement therefore happens entirely after the fix, and
every arm (control included) is re-measured on the same footing.

---

## 2. `gates.yaml` — `arrival_speed_mean_mps` removed from the sk_wind block

The one permitted edit to `gates.yaml`. `sk_wind` has no planned waypoints, so the metric is
undefined there and always read NOT AVAILABLE, failing the gate for a reason that says nothing
about the policy. It stays enforced on `wp_square`, where the waypoints exist.

---

## 3. New flags

### `evaluate.py`

| flag | meaning |
|---|---|
| `--skip-pid` | run only PPO; halves wall time; PID is checkpoint-independent so PPO rows are unchanged |
| `--sk-episode-s FLOAT` | episode length for **station-keeping scenarios only** (`sk_wind`, `rough_sea`, `sk_pid_soft`); default keeps their built-in 120 s. `--episode-s` still applies to `wp_square` only |
| `--level FLOAT` | pin the disturbance curriculum **scale** (0..1) |

`--level` takes a **scale, not an index**. `env.set_curriculum` takes an index and
`scale_by_level == [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]`, so passing a raw index would have been a
silent error where `--level 1` meant scale 0.1. The flag snaps to the nearest entry of
`scale_by_level` and prints when it had to snap.

All three are recorded in `summary.json` under `episode_lengths`, `controller_selection` and
`level_flag`, with an explicit note of which scenario each episode length applies to.

### `tools/gate_policy.py`

Same three flags. `--skip-pid` is a documented **no-op** there (the gate never runs PID) and is
recorded in the settings block so a log cannot imply PID was measured. The gate header now
states which scenario each length applies to:

```
episode-s 300.0 (wp_square only) sk-episode-s 300.0 (sk_wind/rough_sea only) level ... 
```

`configs/gates.yaml` gained `settings.sk_episode_s: 300.0` so a gate run stays reproducible
from the file alone.

### A formatting bug fixed on the way

`fmt()` printed NaN as `"inf"`, so `arrival_speed_mean_mps` on `sk_wind` displayed `inf` while
`summary.json` correctly held `nan`. NaN ("not applicable") and inf ("would never settle") are
different answers; they now print differently.

---

## 4. Report-only: how much of a 120 s sk_wind episode is left to settle?

Measured, `runs/r8_A`, `sk_wind`, 8 envs, 120 s, `sk_frac` forced to 1.0.

**Initial distance to the hold point** (512 envs):

| p10 | p50 | p90 | p99 | mean | max |
|---|---|---|---|---|---|
| 12.9 m | 34.6 m | 55.2 m | 69.7 m | 34.5 m | 79.6 m |

31.2 % of envs start in the 30-45 m band the brief asks about.

**Observed first entry into `tol = 1.0 m`** (8 envs): median **34.4 s**
(values 20.5, 23.9, 29.7, 39.0, 57.9, 63.2 s; two envs never entered).

**Budget:**

| case | transit | left to settle | share of episode |
|---|---|---|---|
| 30 m straight at `max_surge` 1.0 m/s | 30 s | 90 s | 75 % |
| 45 m straight at `max_surge` 1.0 m/s | 45 s | 75 s | 62 % |
| **measured median** | **34.4 s** | **85.6 s** | **71 %** |

So roughly **70 % of a 120 s sk_wind episode is available for settling** after the approach,
and `evalkit.settling_time_s` additionally demands a 5 s in-band tail.

**This is not the constraint.** A policy that settles would have ~86 s to do it. R7 and the R8
10M arms report `settling_time_s = inf` because they never stop, not because the episode is too
short. Anyone tempted to lengthen the episode to "give settling a chance" would be treating a
symptom.

---

## 5. Status

Full suite after all phase-1 changes: see `AUTO_LOG.md` for the pasted summary line and
`runs/AUTO_STATUS.json` for the live phase.
