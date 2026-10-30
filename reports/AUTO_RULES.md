# AUTO_RULES.md — frozen decision rules for task A1

These rules are **frozen**. They were written before the first experiment ran, committed,
and their sha256 recorded in `reports/AUTO_LOG.md`. If any rule needs to change, the run
STOPS; the rule is not edited and the run does not carry on.

Nothing here has been tuned against a result. No threshold, metric, seed list or scenario may
be changed to make a policy pass. The gate PASS/FAIL table from `tools/gate_policy.py` is
informational; decisions use the relative rules below.

Condition labels used throughout:

- **S (standard):** `--dwell-stay-factor 1.0 --episode-s 300 --sk-episode-s 300`
- **C (calm):** S plus `--level 0`

All runs use `--skip-pid`, `--envs 8`, `--episodes 1`, PPO rows only.

---

## D1 — R8 variants vs control

For each variant V in {A, B} (plus C1, C2 only if the single permitted extra round is run):

### Primary (calm condition C, sk_wind, PPO, seeds 0-3)

V beats control if **either**:

- `settled_pct(V) - settled_pct(control) >= 10` percentage points, **or**
- `hold_rms_m(V) <= 0.8 * hold_rms_m(control)` (both finite)

Adjustments, applied in this order:

1. If control already has `settled_pct >= 90 %` in C (ceiling), the primary is evaluated in
   **S** instead of C.
2. If every arm has `settled_pct == 0` **and** `hold_rms_m` differs by `< 5 %` in C, the
   primary is **NOT DECIDABLE**, which counts as "does not beat control".

### Guards in S (all must hold)

- `sat_frac(V) <= sat_frac(control)` in **both** sk_wind and wp_square
- `waypoints_reached(wp_square, V) >= 0.85 * control`
- `geofence-dropped(V) <= geofence-dropped(control)`
- `settled_pct(S, V) >= settled_pct(S, control) - 5` percentage points

### Winner selection

Winner = the variant satisfying primary **and** every guard. If several qualify, the larger
primary improvement wins; a tie breaks to the lower `sat_frac`.

### Confirmation

On **fresh seeds 100-103** the same variant must again satisfy the primary (same direction,
same thresholds **halved**) and all guards. Otherwise it is **not** a winner.

### Outcome enumeration

Exactly one of:

- `WINNER(w)` — D1 satisfied and confirmed on fresh seeds.
- `PROVISIONAL_A` — no variant satisfies D1, but A has `sat_frac <= control` **and**
  `waypoints_reached >= 0.85 * control` in S (strictly better than control on saturation
  without harming tracking). A is then only a **fixed reward for R9**, NOT a validated
  improvement; every R9 table carries the label **"reward provisional"**.
- `NONE` — neither; skip phases 4-7 and write the final report.

### The single permitted extra round

If, and only if, no variant beats control: two new 10M variants

- **C1** = A with `station_keeping_fraction 0.5`
- **C2** = A with `gamma 0.995`

Same instruments, same rule D1 against control. Then decide again. **No further rounds.**

---

## D2 — R9 variants vs P0

Primary = mean `waypoints_reached` over `wp_square`, `wp_zigzag`, `wp_hairpin` in **S**.

P1/P2 win if:

- primary `>= P0 + 0.15`

**and** guards hold:

- `sk_wind settled_pct(S) >= P0 - 5` percentage points
- `sat_frac <= P0 + 0.05` on **every** scenario
- `orbit_frac(V) <= orbit_frac(P0)` — **P2 must also be strictly lower**
- `geofence-dropped <= P0`

Winner = the best primary among those that qualify. Confirmed on fresh seeds 100-103 with the
same inequalities, primary threshold **halved**.

If neither wins, P0 is the R9 outcome.

---

## Budget and stop conditions (frozen)

- Wall clock total <= 12 h. If exceeded: finish the current run, skip the rest, write the
  final report.
- Max 8 training runs total, max 100M steps each. **No 300M run.**
- STOP if: free disk < 10 GB; any training shows NaN/Inf; CUDA errors repeat twice; the same
  test fails twice after two fix attempts; or the only way forward is to bend a frozen rule.

A clean stop with a clear `STOPPED.md` is a success. Silently pushing on is not.
