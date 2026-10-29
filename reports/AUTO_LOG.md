# AUTO_LOG.md — task A1 autonomous run

Append-only. One entry per action: UTC time, command, outcome, next step.
Status mirror: `runs/AUTO_STATUS.json`.

---

## Phase 0 — preflight

**2026-10-29 09:19 UTC**

- `date -u` -> Thu, 29 Oct 2026 09:19:03 +0000
- Branch before: `master`. Created `auto/a1` from `master`. Remote: `origin`
  https://github.com/Jakinisme/drl.git
- Working tree before start: 1 modified file (`evaluate.py`, the `--skip-pid` flag added but
  not yet committed). No other training/eval process running (2 idle `python.exe`, 32-416 MB,
  not torch workers).
- GPU: `NVIDIA GeForce RTX 3050`, 6144 MiB total, 959 MiB used.
- Free disk on E:: 277 GB available (limit is 10 GB).
- R8 checkpoints present and complete — all three at `steps 9961472`, `iter 152`,
  `env_level_index 5`:
  - `runs/r8_control/ckpt/ckpt_last.pt`
  - `runs/r8_A/ckpt/ckpt_last.pt`
  - `runs/r8_B/ckpt/ckpt_last.pt`

### Frozen rules

`reports/AUTO_RULES.md` written **before** any experiment ran.

```
sha256(reports/AUTO_RULES.md) = 536ac3c1c3a9b4f311963f34a3c58d7320f8c389f083135164de283bfc3199c6
```

This hash is recorded here per section 0 of the brief. Every later check recomputes it; a
mismatch means the rules were edited and the run stops.

### Pre-existing condition noted, not fixed

`master` contains commit `bec3274 "Refactor code structure for improved readability and
maintainability"`, which committed 171 files including evaluation PNG artefacts and
`results_r7_reseeded/` (141 files) — these match the `results_*/` pattern I added to
`.gitignore` earlier. It is history on `master`; per the brief I do not rewrite history and do
not delete results. Flagged for the human in `AUTO_FINAL.md`.

Next: verify the R8 test suite is green on this branch, then Phase 1.

---

## Phase 1 — R8b instruments

**2026-10-29 09:28 UTC**

### 1. hold_rms_m NaN — root cause was NOT the metrics wiring

The brief suggested looking at `info["mode"]`, `goal_xy`, the `good` mask or window
clipping. None of those were wrong. Measured on a real 8-env sk_wind rollout
(`runs/r8_A`, 300 s), the per-env final `mode` was:

```
[0. 0. 1. 0. 0. 0. 0. 1.]
```

Only 2 of 8 envs ended in station keeping, so `final_hold_window` correctly returned
`(-1, -1)` for the other 6 and `hold_rms_m` was nan. On the envs that DID end in SK, the
metric already worked (`hold_rms 2.5175`).

Cause: `SCENARIOS` pins `mode` (1 = station keeping) but **nothing in the codebase consumed
it** (`grep station_keeping_fraction` over evaluate.py/evalkit/tools returned nothing). So
`sk_wind` ran at the training mix `station_keeping_fraction: 0.3` and ~70 % of its episodes
were waypoint episodes with no final hold. The scenario never forced the mode it advertised.

Fix: new `scenario_episode_cfg(scen)` in `evaluate.py` maps `mode` -> 
`station_keeping_fraction` (0.0 / 1.0). Scenarios without a `mode` key keep the config value.

After the fix, a real sk_wind rollout gives `final mode [1. 1. 1. 1. 1. 1. 1. 1.]` and
`hold_rms_m 2.5175`, `hold_p95_m 3.1917` — finite.

Test: `tests/test_r8b_hold_wiring.py`, 4 cases, real rollout on `runs/r8_A`, no fakes.
`4 passed in 22.97s`.

**Consequence to carry forward:** every earlier sk_wind number (including the R7 reseeded
reference `results_r7_reseeded`: sk settled_pct 15.6, radial 18.96) was measured on a
scenario that was ~30 % station keeping, not 100 %. Those numbers are not comparable with
sk_wind numbers measured after this fix. Recorded, not silently adjusted.

### 2. gates.yaml — arrival_speed_mean_mps removed from the sk_wind block

The one permitted edit. sk_wind has no planned waypoints so the metric is undefined there
and always read NOT AVAILABLE, failing the gate for a reason that says nothing about the
policy. It stays enforced on wp_square.

### 3. New flags

`evaluate.py`: `--skip-pid`, `--sk-episode-s`, `--level`.
`tools/gate_policy.py`: same three; `--skip-pid` is a documented no-op there (the gate never
runs PID) and is recorded in the settings block. `configs/gates.yaml` gained
`settings.sk_episode_s: 300.0` and its header now states which scenario each length applies
to.

`--level` takes the disturbance curriculum **SCALE** (0..1), not an index — `set_curriculum`
takes an index and `scale_by_level[1] == 0.1`, so passing a raw index would have been an easy
silent mistake. It snaps to the nearest entry of `scale_by_level` and says so. All three
flags are recorded in `summary.json` (`episode_lengths`, `controller_selection`,
`level_flag`) and in the gate header.

Next: full suite, then the phase-1 report item (settling time left in a 120 s sk_wind
episode) and commit.

### 1b. Formatting bug found while validating the flags

`fmt()` in `evaluate.py` printed NaN as `"inf"`, so the per-seed line showed
`arrival_speed_mean_mps=inf` on sk_wind while `summary.json` correctly held `nan`. NaN means
"not applicable", inf means "would never settle" — they now print differently. Validated by a
real `--level 0` run whose summary.json held `nan` where the old line said `inf`.

### 4. Settling budget in a 120 s sk_wind episode (report only)

Measured, not estimated. Initial distance to the hold point over 512 envs: p50 34.6 m,
p90 55.2 m, mean 34.5 m; 31.2 % of envs start in the 30-45 m band.

`runs/r8_A` on sk_wind (120 s, 8 envs): median first entry inside tol=1.0 m at **34.4 s**,
leaving **85.6 s = 71 %** of the episode to settle, plus evalkit's 5 s in-band tail.

Conclusion recorded in `reports/R8b.md` §4: episode length is NOT the constraint. The arms
report `settling_time_s = inf` because they never stop, not because there is no time. Lengthening
the episode would treat a symptom.

### Suite

`243 passed, 15 warnings in 213.72s (0:03:33)` on the run that included the 4 new
`tests/test_r8b_hold_wiring.py` cases. Re-running after the `fmt` change before committing.

Next: commit phase 1, then phase 2 (R8 comparison on the new instruments).
