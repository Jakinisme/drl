# A2_RULES.md — frozen decision rules for task A2

These rules are **frozen**. They were written before the first experiment of A2 ran, committed,
and their sha256 recorded in `reports/A2_LOG.md`. If any rule needs to change, the run **STOPS**
(section 6 of the brief); the rule is not edited and the run does not carry on. They are copied
verbatim from the run brief, section 3.

Nothing here has been tuned against a result. No threshold, metric, seed list or scenario may be
changed to make a policy pass.

Condition labels used throughout:

- **S (standard):** `--dwell-stay-factor 1.0 --episode-s 300 --sk-episode-s 300`
- **C (calm):** S plus `--level 0`

All evaluations use `--device cpu --skip-pid --no-plots --envs 8 --episodes 1`, PPO rows only.
`--seeds N` is a COUNT (`--seed 100 --seeds 4` = fresh seeds 100-103).

---

## D3 — A vs control, both 100M

Both arms are 100M, training seed 0, evaluated by this run **on CPU**, seeds 0-3.

### Primary (condition S, sk_wind, PPO)

A beats control if:

- `settled_pct(A) - settled_pct(control) >= 10` points **OR**
- `hold_rms_m(A) <= 0.8 * hold_rms_m(control)` (both finite).

If control already has `settled_pct >= 90 %` in S (ceiling), evaluate the primary in **C**
instead.

### Guards (S and C)

- `waypoints_reached(wp_square, A) >= 0.85 * control`
- `geofence-dropped(A) <= control`
- `settled_pct(wp_square, A) >= settled_pct(wp_square, control) - 10` points in S.

### Outcome

`A_BETTER_100M` = primary + all guards; else `A_NOT_BETTER_100M`.

Confirmation on seeds **100-103** (same inequalities, primary thresholds **halved**) is required
for `A_BETTER_100M`; if it fails the outcome is `A_NOT_BETTER_100M`.

### Saturation label (reported, never decisive)

Over the four cells {sk_wind, wp_square} x {S, C}:

- `SAT_BENEFIT_HOLDS` if `sat_frac(A) <= 0.8 * sat_frac(control)` in all four,
- `SAT_BENEFIT_PARTIAL` if in 2 or 3,
- `SAT_BENEFIT_LOST` otherwise.

### Caveat that must appear in the report

One training seed per arm.

---

## R9-REWARD — which reward the R9 variants use

The reward used from Phase 3 on is **A's reward** if D3 = `A_BETTER_100M`, or if D3 =
`A_NOT_BETTER_100M` but control does not beat A on the D3 primary by the mirrored criterion;
otherwise it is **control's reward**.

---

## D2 — R9 variants vs P0

Condition **S**, seeds 0-3.

### Primary

Mean `waypoints_reached` over `wp_square`, `wp_zigzag`, `wp_hairpin`.

### P1/P2 win if

- primary `>= P0 + 0.15`

**and** guards:

- `sk_wind settled_pct >= P0 - 5` points
- `sat_frac <= P0 + 0.05` on **every** scenario
- `orbit_frac <= orbit_frac(P0)` — **P2 must also be strictly lower**
- `geofence-dropped <= P0`

Winner = best primary among those that qualify; confirmed on seeds **100-103** with the same
inequalities, primary threshold **halved**. No winner → **P0** is the R9 outcome.

---

## Budget and stop conditions (frozen, brief §1 §6)

- Wall-clock cap 10 h from Phase 0. Per-phase caps as given; a phase that exceeds 2x its cap is
  abandoned (`SKIPPED_BUDGET`) and the run moves on.
- Training runs total <= 7, each <= 100M steps. **No 300M run.**
- STOP if: free disk < 10 GB; NaN/Inf in a training log; CUDA errors repeat twice; the same test
  fails twice after two fix attempts; or the only way forward bends a frozen rule.

A clean stop with `STOPPED.md` is a success. Silently pushing on is not.
