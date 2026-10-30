# R9_impl — `pathgen.StructuredPath` (waypoint-pattern generator + orbit penalty)

Files created: `pathgen.py`, `tests/test_pathgen.py`, this report. Nothing else touched, nothing
committed. **All numbers are simulator numbers on placeholder physics (SYSID.md).**

Test summary line (`pytest tests/test_pathgen.py -q`, CPU):

```
12 passed in 40.54s
```

## 1. Interface — drop-in for `evaluate.SquarePath`

```python
StructuredPath(env, side, hold_s, dwell_stay=1.5, *, pattern="legacy_square", seed=0,
               min_leg_m=15.0, max_leg_m=45.0, chain_radius_m=60.0,
               min_turn_deg=60.0, max_turn_deg=120.0, max_wind_deg=330.0,
               arrival_radius_m=2.0, orient_deg=None)
```

Same surface as `SquarePath`: `.c` [N, P+1, 2], `.idx`, `.hold`, `.done`, `.holding`,
`.goal_xy`, `.arrival`, `.hold_steps`, `.stay`, `waypoints(i)` → [P, 2], `__call__(env)` → [N, 4].
Extra: `.P`, `.fallback` [N] bool, `.orbit` [N] deg, `.spawn`, `pattern_names()`,
`pattern_index()` → [N] long, `orbit_frac()` → float, `turn_angles(chain)` (static).

The state machine is `evaluate.SquarePath` **inherited**, not re-typed: `_StateMachine(SquarePath)`
overrides only `__init__` (skips the square, installs `self.c`) and `__call__` (SquarePath's body
with the hardcoded `4` → `c.shape[1] - 1`). `legacy_square` builds corners with
`evaluate.square_corners` verbatim.

**Legacy compat, measured:** max abs diff **0.0** over 200 steps × 8 envs vs `SquarePath`
(returned goal tensors, `goal_xy`, `psi_goal`, `env.goal_xy`, plus `idx`/`hold`/`done`/`holding`
`torch.equal` every step). Bit-identical. `capture_conditions()` works unchanged (reads `c[:, 1]`).

## 2. Patterns (`NPTS`: waypoint count)

| pattern | waypoints | construction |
|---|---|---|
| `legacy_square` | 4 | `square_corners(env, side)`; ignores every other param |
| `straight` | 2–3 | collinear; legs drawn then clamped so the last wp lands on the ring |
| `turn` | 3 | exactly one 90° turn (turn 0); all other turns 0 |
| `zigzag` | 4–6 | alternating-sign turns, drawn U[min_turn_deg, max_turn_deg] |
| `random` | 4 | legs U[min_leg_m, max_leg_m], turns U[min_turn_deg, max_turn_deg] |

Chains start along the **spawn heading** (`orient_deg=None`), corner 0 = spawn. Measured over
50 seeds × 8 envs (and 200 seeds in the tests):

| pattern | P | leg range (m) | interior turn range (deg) | fallback | max swept |
|---|---|---|---|---|---|
| straight | 2–3 | 15.01–30.00 | 0.0 | 0.00 % | 0.0° |
| turn | 3 | 15.00–44.79 | 90.0 | 0.00 % | 53.2° |
| zigzag | 4 | 15.04–44.99 | 60.2–120.0 | 0.00 % | 315.5° |
| random | 4 | 15.01–44.98 | 60.2–119.9 | 0.00 % | 320.2° |

`zigzag`'s P caps at 4, not 6: `chain_radius_m / min_leg_m = 60/15 = 4` legs is the geometric
maximum, so `hi = min(hi, floor(chain_radius_m // min_leg_m))`. With the defaults this makes
`zigzag` and `random` the same length; raise `chain_radius_m` to 90 to get 6-leg zigzags.
Fallback rate over 200 seeds × 64 envs (zigzag, random): **0.0 % / 0 seeds**.

Bounded regeneration: `TRIES = 32` draws per env; a draw is rejected if any waypoint exceeds
`chain_radius_m` of spawn or the swept angle exceeds `max_wind_deg`. Exhausted → that env
gets a `straight` chain and is flagged in `.fallback`.

## 3. Orbit penalty

```python
swept_angle(chain, hold, radius)   # degrees
```
`v_k = p_k - hold`; `dtheta_k = |atan2(cross(v_k, v_k-1), dot(v_k, v_k-1))|`; folds `>180°` to
`2π - dtheta` (atan2 branch cut alone would invent a full turn); exact zeros stay zero. A segment
is **skipped** when *either* endpoint is within `radius` of `hold` (`radius = arrival_radius_m`).
`orbit = sum(dtheta over kept segments)`; an env trips when `orbit > max_wind_deg` → regenerate.
`orbit_frac()` returns the fraction currently above the cap (0.0 whenever regeneration was
possible; >0 only for a hand-fed `.c`).

Verified: 480° spiral trips, 0° for a line (even through the hold), 45° for one 90° corner
(90° between leg *directions* = 45° between hold→wp vectors), 60° with an outside-radius arc.

Test assesses: `pytest tests/test_pathgen.py -q` — 12 passed: legacy bit-identity (200 steps),
legacy dwell machine in lockstep, per-pattern 200-seed legality, orbit firing/no-firing, hold
exclusion, determinism (same seed same plan; different seed; different spawn; different pattern).

## 4. `SCENARIOS` entries for the parent (one line each + the `run_rollout` wiring)

```python
    "wp_line":    dict(mode=0, episode_s=120.0, square=False, rough=False, pid_soft=False,
                       pattern="straight"),
    "wp_zigzag":  dict(mode=0, episode_s=200.0, square=False, rough=False, pid_soft=False,
                       pattern="zigzag"),
    "wp_hairpin": dict(mode=0, episode_s=160.0, square=False, rough=False, pid_soft=False,
                       pattern="turn"),
```

and in `run_rollout`, replace the two lines

```python
        path = (SquarePath(env, square_side, hold_s, dwell_stay)
                if scen["square"] else None)
```
with
```python
        path = (StructuredPath(env, square_side, hold_s, dwell_stay,
                               pattern=scen.get("pattern", "legacy_square"), seed=seed)
                if scen["square"] or scen.get("pattern") else None)
```
(`wp_square` keeps `square=True, pattern` absent → `legacy_square`, bit-identical, so existing
`wp_square` results stay comparable.)

`episode_s` rationale: `turn` is ~2 legs (≈30–90 m), `straight` ~2–3 legs, `zigzag` 4 legs
(≈120 m); at the sim's surge the square's 4×30 m needed 200 s. `wp_line` 120 s and `wp_hairpin`
160 s are **guesses**, not measured — check `lap_pct` on the first run and adjust.

## 5. Not reproduced / caveats

- **`zigzag` 4–6 waypoints is not achievable at the default `chain_radius_m=60`, `min_leg_m=15`**
  (max 4 legs inside the ring). Spec bounds met for 4; 5–6 needs `chain_radius_m >= 90`.
- `turn_angles` returns `P-1` values; the last one is the (never-drawn) turn after the final leg
  and reads 0. Use `turn_angles(c)[:, 0]` for the real bends.
- `straight` cannot regenerate into a different plan (collinear), so an over-long straight is
  solved by clamping legs, not by re-rolling — by design, documented in `_sample`.
- `orient_deg` defaults to the spawn heading, so legs are not world-frame uniform. Set
  `orient_deg=<bearing>` if a scenario needs a fixed frame.
- Vessel dynamics did **not** reach a corner within 200 steps under a naive bearing controller,
  so the legacy test teleports both envs onto corner 1 at step 120 to exercise the dwell/leave
  branch; the 200-step equality itself is unaffected (envs verified state-identical each step).
