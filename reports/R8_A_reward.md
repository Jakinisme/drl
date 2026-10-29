# R8-A — reward redesign so the policy SETTLES

Owner: R8-A (T8). Files touched: `envs/asv_env.py` (`_reward`, `_arrive_evt`, new
`_speed`/`_gate`, `speed_arrival` bookkeeping, `R8_OFF` defaults), `configs/env.yaml`
(`reward:` keys only), `CONTRACT.md` (§3.4 reward keys, additive), plus the new control
arm config `configs/r8/env_control_A.yaml`. Physics, observation, action space, rate
limiter and network are UNTOUCHED. Nothing here is committed.

Every number below is from the placeholder sim (`configs/vehicle.yaml` + `disturbance.yaml`
are un-tuned, SYSID pending). **Nothing here transfers to the real vessel.**

---

## 1. The bug, restated as arithmetic

`_reward` paid `w_arrival * arrive_evt` = a flat **10.0** on the arrival step, with no
speed dependence. The only speed-aware term was `w_near_goal_speed = 0.1` over a 5 m
radius — 100× smaller. Measured on the R7 checkpoint: **mean speed at first entry to the
final goal 1.002 m/s, p95 1.335, max 1.468**, i.e. the hull is still at `max_surge`
(1.0 m/s) or above when it crosses the 2 m radius. Flying through and stopping paid the
same 10.0.

Reproduced here offline from `runs/r7_300m/ckpt/ckpt_last.pt` (16 envs, 1200 steps,
policy mean action, no training):

| quantity | value |
|---|---|
| arrival events | 34 |
| speed at arrival: mean / p95 / max | **1.117 / 1.393 / 1.505 m/s** |
| envs inside 2 m at the end | 0.6875 |

which matches the R7 eval (settling_time `inf`, sat_frac 0.71–0.76).

## 2. The new formula

`speed = ||v_body|| = ||(u, v)||` (`self.state[:, 3:5].norm(dim=1)`, unchanged expression,
factored into `_speed()`); `gate(speed, ref) = exp(-(speed / ref)^2)`; `final` and
`arrive_evt` exactly as R4.

```
final       = (mode > 0.5) | (wp_idx + 1 >= n_wp)          # unchanged (R4)
arrive_evt  = adv | enter                                  # unchanged (R4 latch)
              + NEW side effect: speed_arrival[i] = speed[i] on every step evt[i] == 1

r =  w_position      * exp(-d / pos_sigma_m)      * final
   + w_position_fine * exp(-d / pos_sigma_fine_m) * final
   + w_heading       * exp(-|psi_err| / head_sigma_rad)
   + w_progress      * clamp(dd / (max_surge * policy_dt), -1, 1)
   - w_cross_track   * clamp((xtrack / xtrack_sigma_m)^2, max=1)
   - w_near_goal_speed * (speed / max_surge)^2 * (d < slowdown_radius_m) * final     # retuned
   - w_action_rate   * mean_axes((a - a_prev)^2)                                     # retuned
   - w_energy        * mean_axes(a^2)                                                # retuned

   # R8, speed-gated arrival (replaces the flat `+ w_arrival * arrive_evt`)
   + w_arrival * arrive_evt * (arrive_gate_floor
                               + (1 - arrive_gate_floor) * gate(speed_arrival, settle_speed_ref_mps))

   # R8, dense hold bonus — FINAL goal only (station keeping or last waypoint)
   + w_hold * [d < hold_radius_m] * final * gate(speed, settle_speed_ref_mps)

   # R8, saturation penalty — every step
   - w_sat  * mean_axes( clamp(|a| - 0.8, min=0) / 0.2 )^2
```

`arrive_evt` stays 0 in station keeping, so station keeping pays **no** arrival bonus but
**does** pay the dense hold bonus (it is `final`).

## 3. Keys and defaults (`configs/env.yaml`, all `# TODO T8 placeholder, untuned`)

| key | default | pre-R8 / OFF value | role |
|---|---|---|---|
| `w_hold` | 0.5 | `0.0` | dense hold, final goal + slow |
| `hold_radius_m` | 1.5 | — (inert while `w_hold = 0`) | inside `arrival_radius_m = 2.0` on purpose |
| `w_sat` | 0.1 | `0.0` | penalty on `|a| > 0.8` |
| `arrive_gate_floor` | 0.3 | `1.0` | what a FAST intermediate crossing still pays, ×`w_arrival` |
| `settle_speed_ref_mps` | 0.4 | `1e20` (see §3.1) | gate scale |
| `w_near_goal_speed` | 0.3 | `0.1` (retuned) | unchanged gating (`near * final`) |
| `w_action_rate` | 0.15 | `0.05` (retuned) | |
| `w_energy` | 0.02 | `0.01` (retuned) | |

Untouched: `w_position`, `w_position_fine`, `w_heading`, `w_progress`, `w_cross_track`,
`w_arrival`, `pos_sigma_m`, `pos_sigma_fine_m`, `head_sigma_rad`, `xtrack_sigma_m`.

### 3.1 Bit-compatibility — and one place where the brief's number is WRONG

**`settle_speed_ref_mps = 1e6` does NOT give `gate == 1.0`.** In float32:

| speed | `exp(-(v/1e6)^2)` |
|---|---|
| 1.0 m/s | 1.0 |
| 1000 m/s | **0.99999899** |
| 10000 m/s | **0.99990000** |
| 100000 m/s | **0.99004984** |

`-(1000/1e6)^2 = -1e-6` is representable, so `exp` returns `1 − 1e-6`. Measured, not
argued. Any reachable speed is < 10 m/s, where 1e6 *is* exactly 1.0 — but a control arm
built on a 1e-4 seam is not a control arm. **Used instead: `1e20`**, which is exactly 1.0
for every float32 `|v| < 1e18` (`v=1e15 → 1.0`; `v=1e19 → 0.99`; unreachable). Verified by
sweep over exponent/mantissa grid, not a single point.

Two further traps found the hard way, both now closed:

1. **PyYAML reads `1e20` AND `1.0e20` as strings** (YAML 1.1 wants a dot plus a signed
   exponent). A string ref silently left torch's scalar path and the gate returned 1.0 for
   every speed — the new terms were *inert* and the smoke looked "fine". The YAML now
   writes `100000000000000000000.0`, and `_gate` coerces its ref with
   `torch.as_tensor(float(ref), dtype=speed.dtype)` so a future string/int ref cannot
   re-open this.
2. **`cfg["reward"]["settle_speed_ref_mps"]` raised `KeyError`** on any pre-R8 config
   (`configs/r8/env_*.yaml` written by other agents, `git show HEAD:configs/env.yaml`).
   The ctor now `setdefault`s all five R8 keys from `R8_OFF` (= the OFF values in the table
   above, ref `3.4028234663852886e38`), so an old config runs and runs the R4 reward.

**Bit-exactness verified**, not assumed. Control arm = `configs/r8/env_control_A.yaml`
(R8 terms off AND the three retunes undone). Pre-R8 side = the **actual HEAD code**
(`git show HEAD:envs/asv_env.py`, imported as a separate module) with
`git show HEAD:configs/env.yaml`. Same trajectory (R7 policy, 16 envs, 1200 steps):

```
obs    bitwise equal = True
reward bitwise equal = True   max|diff| = 0.0
```

Also verified: `obs(first) HEAD == obs(new default config)` → **True**. The observation
stream is independent of the reward config (checked at 200 and 1200 steps, same hash),
and `tests/test_obs_parity.py` passes unchanged (**5 passed**).

Caveat for whoever writes the control test: the sensor model draws from the **global**
torch RNG (`sim/sensors.py`, `torch.randn` without a generator), so two envs built in the
same process need `torch.manual_seed(...)` between them. That is pre-existing (HEAD is
non-deterministic against itself otherwise) and not mine to change.

The switch is exactly four config values — that is what "reproducible by a config" means
here:

```yaml
reward:
  w_hold: 0.0
  w_sat: 0.0
  arrive_gate_floor: 1.0
  w_near_goal_speed: 0.1
  w_action_rate: 0.05
  w_energy: 0.01
  # settle_speed_ref_mps: irrelevant once arrive_gate_floor = 1.0 and w_hold = 0,
  # but use 100000000000000000000.0 if you want it stated.
```

## 4. Speed in `gate()`: `|v_body|`, NOT including yaw rate

**Choice: `||(u, v)||`**, the existing expression, unchanged.

- `settling` is a *translation* metric. The runtime's `speed_mps` (`gps_sog`), which the
  metrics and the real operator see, is surge/sway only; a reward keyed to a different
  quantity than the one being optimised for is a silent mismatch.
- A hull holding station in a current **yaws** — it must, to stay put. Folding `r` into
  the gate would fine a correctly-behaving controller exactly when it is working hardest.
  With `ref = 0.4`, `(r/0.4)^2` makes a 0.5 rad/s spin a 1.5× larger exponent than the
  same 0.5 m/s of drift; the term would be dominated by turn-in-place.
- `|v_body|` is what the R7 measurement already reports (1.002 m/s mean at entry vs
  `max_surge` 1.0), so the before/after comparison is like-for-like.
- Yaw is not left unsupervised: `w_heading` (0.3) and `w_near_goal_speed` (now 0.3) still
  price it, and the R5 hysteresis work already showed yaw chatter is the failure mode to
  watch.

If a future run shows a policy that settles in translation while spinning, the fix is a
separate `w_spin * (r / spin_ref)^2` term, not a redefinition of `speed` — changing
`speed` would move the observation-side meaning of the metric the whole project reports.

## 5. Where the spec was ambiguous, and what I chose

1. **Floor at the final waypoint.** The brief states two rules that cannot both hold for
   one `arrive_evt`: "intermediate: `w_arrival * (floor + (1-floor) * gate)`" and "final
   (first entry, latch unchanged): `w_arrival * gate`". If the floor is applied to
   `arrive_evt` as a whole (what I implemented, because `_arrive_evt` returns exactly one
   array and the brief says the floor's only job is to keep a fast *intermediate* crossing
   paying something), the final goal also keeps the floor — which is the same reward a
   fly-through used to get, just 3.0 instead of 10.0. Divergence at the final goal:

   | speed at entry | floor on both (implemented) | gate only at final |
   |---|---|---|
   | 0.0 | 10.000 | 10.000 |
   | 0.4 | 5.575 | 3.679 |
   | 0.7 | 3.327 | 0.468 |
   | 1.0 | 3.014 | 0.019 |
   | 1.468 | 3.000 | 0.000 |

   **I implemented the literal common formula.** Switching to gate-only-at-final needs no
   new key — `adv` already excludes the final WP, so `evt * (1 - final)` is the
   intermediate subset:

   ```python
   g = self._gate(self.speed_arrival, ref)
   bonus = (w["w_arrival"] * arrive_evt * final * g
            + w["w_arrival"] * arrive_evt * (1.0 - final)
                             * (w["arrive_gate_floor"] + (1.0 - w["arrive_gate_floor"]) * g))
   ```

   This is the one decision in the report that the orchestrator should overrule or
   confirm before training: §6 shows the residual 3.0 is small against the ~60 the adopted
   terms offer, so it does not by itself reopen fly-through.
2. **Gate speed at the final waypoint is `speed_at_entry`** — the latched entry, not the
   current speed. A latched bonus cannot be re-gated later; that is why
   `speed_arrival` exists at all. It is set on **any** event (intermediate advance or
   final first-entry) and reset to 0 on `reset(ids)`.
3. **`speed_arrival` staleness.** It holds the last event's speed. Only read where the
   current `arrive_evt` is 1, so a stale value can never leak into a bonus. It is *not*
   reset when the pointer advances, which is deliberate: one field, one meaning ("speed at
   the most recent arrival").
4. **`hold_radius_m` vs `arrival_radius_m`.** 1.5 < 2.0 as specified, so entering the hold
   zone necessarily happens *inside* the arrival radius. The policy collects it on the way
   in, which is the intent (no dead zone).
5. **Saturation penalty** is `mean over axes of ((|a| - 0.8)+ / 0.2)^2` — squared per axis
   *then* averaged, so one axis pinned at 1.0 contributes `0.04/3 ≈ 0.0133` at `w_sat = 0.1`
   → 0.0013, and all three pinned gives 0.1. That is cheap next to `w_energy`; it is a
   tie-breaker, not a constraint. If the intent was a hard "don't ride the rail" term, the
   magnitude needs T8 tuning.
6. **`w_hold` under `final`.** Applied as `(d < hold_radius_m) * final * gate(speed)` with
   `final` as a float mask, so the guards `if w["w_hold"] != 0.0` / `if w["w_sat"] != 0.0`
   keep the control arm free of the new arithmetic *and* free of the extra multiply — the
   bitwise control run confirms it either way.
7. **README of the arm.** `configs/r8/env_control_A.yaml` is a new file (name chosen to
   avoid `env_control.yaml`, which another agent owns).

## 6. Can `w_arrival 10.0` (one-shot) vs `w_hold 0.5` (dense) change behaviour?

Discount from `configs/train.yaml`: `gamma = 0.99`, `policy_dt = 0.1 s` →
**effective horizon `1/(1-gamma) = 100 steps = 10 s`**, `Σ γ^t` over a 1200-step episode
= **100.0**. So a per-step reward of 0.5 held forever is worth `0.5 × 100 = 50`, five
times one 10.0 prize. The dense term is *not* outgunned.

**Final goal**, `w_arrival = 10`, `ref = 0.4`, full episode (1200 steps):

| speed at entry | pass through (one-shot + floor) | stop at rest at t=600 | ratio |
|---|---|---|---|
| 0.4 | 5.58 | 60.0 | **10.8×** |
| 1.0 | 3.01 | 60.0 | **19.9×** |
| 1.335 (p95 of R7) | 3.00 | 60.0 | **20.0×** |
| 1.468 (max of R7) | 3.00 | 60.0 | **20.0×** |

where "stop" = the 10.0 gated one-shot at rest + `0.5 × 100 = 50` of hold; the position
kernels (≈1.5/step at `d ≈ 0`) are on both sides and are omitted. **The reward can work**:
at the speeds the R7 policy actually arrives at, settling is worth **~20×** more than
sailing through, and even a policy arriving at a perfect crawl gains 6× by staying. Under
the gate-only-at-final reading the ratio is far larger (fly-through ≈ 0.0).

**Intermediate waypoint** (no dense hold there by design): crossing at 1.0–1.47 m/s pays
the 3.0 floor; parking 3–5 m short earns heading only, `0.3/step`, so the floor beat a park
after **10 steps (1.0 s)**. And parking forfeits every later leg: one 30 m leg at 0.55 m/s
is ≈545 steps earning heading + progress ≈ `0.8/step` = **~436**. Crossing wins by ~150×.

**Where it is lopsided the other way.** `w_hold` alone must be given `duty > 0.2` (20 % of
the episode inside 1.5 m at rest) to match a single 10.0 arrival; at a 10 % duty cycle the
hold is worth 5.0, half a single arrival. So the hold bonus is a *settling incentive*, not
a substitute for the arrival bonus — which is the intended split and why I did not reduce
`w_arrival`.

**Honest risk, stated plainly.** `hold_radius_m = 1.5` against `spawn/waypoint_radius_m =
35 / geofence 80`: a hold zone covers `π·1.5² ≈ 7 m²` of a `~π·35² ≈ 3800 m²` waypoint box,
0.19 %. `w_hold` is a sparse-reward term the policy must *stumble into* before it can be
shaped by it; on top of a fresh policy it may contribute nothing for a long time, and its
measured contribution in the R7-policy smoke is only `+0.062/step` overall (band max 0.5,
reached from t ≈ 800 onward, see §7). If training shows `hold` flat at 0, the fix is
`hold_radius_m 1.5 → 2.0` (must stay inside `arrival_radius_m`) or `w_hold` up — not a
larger radius *outside* the arrival radius, which would teach parking short of the goal.

## 7. Smoke: finite and sane (16 envs, CPU, no training)

Driver: the **R7 checkpoint policy** (`runs/r7_300m/ckpt/ckpt_last.pt`) as the action
window, 1200 policy steps = one full episode, `torch.manual_seed(1234)` pinned so the
pre-R8 and new runs share the identical trajectory. This is the case that matters — it is
the policy that flies through.

Headline: `finite = True`, reward range `[-0.7487, +4.6183]`, mean `+0.7413`
(pre-R8 mean `+0.7366`, mean |Δ| `0.0996`). Max per-step reward dropped from `+11.41`
(the old 10.0 spike) to `+4.62`.

Per-component **mean per iteration** (`t` = policy step; the first four rows are startup):

| t | pos | pos_fine | heading | progress | xtrack | near_spd | act_rate | energy | arrival | hold | sat | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.0081 | +0.0000 | +0.0656 | +0.0001 | −0.0000 | +0.0000 | −0.0060 | −0.0008 | +0.0000 | +0.0000 | +0.0000 | +0.0687 |
| 1 | +0.0081 | +0.0000 | +0.0650 | +0.0008 | −0.0000 | +0.0000 | −0.0060 | −0.0032 | +0.0000 | +0.0000 | +0.0000 | +0.0664 |
| 2 | +0.0081 | +0.0000 | +0.0636 | +0.0020 | −0.0000 | +0.0000 | −0.0060 | −0.0072 | +0.0000 | +0.0000 | +0.0000 | +0.0623 |
| 3 | +0.0081 | +0.0000 | +0.0630 | +0.0036 | −0.0000 | +0.0000 | −0.0057 | −0.0121 | +0.0000 | +0.0000 | +0.0000 | +0.0574 |
| 4 | +0.0081 | +0.0000 | +0.0615 | +0.0057 | −0.0001 | +0.0000 | −0.0036 | −0.0158 | +0.0000 | +0.0000 | −0.0492 | +0.0071 |
| 99 | +0.0178 | +0.0000 | +0.1330 | +0.4832 | −0.1207 | +0.0000 | −0.0023 | −0.0099 | +0.0000 | +0.0000 | −0.0228 | +0.4859 |
| 199 | +0.0490 | +0.0014 | +0.1159 | +0.4138 | −0.1783 | +0.0000 | −0.0021 | −0.0099 | +0.0000 | +0.0000 | −0.0258 | +0.3642 |
| 399 | +0.1428 | +0.0246 | +0.0913 | +0.3482 | −0.1263 | −0.0124 | −0.0019 | −0.0088 | +0.0000 | +0.0014 | −0.0181 | +0.4428 |
| 599 | +0.2467 | +0.0808 | +0.1063 | +0.3329 | −0.1480 | −0.0281 | −0.0020 | −0.0106 | +0.0000 | +0.0497 | −0.0189 | +0.6245 |
| 799 | +0.4013 | +0.1315 | +0.1772 | +0.3287 | −0.0979 | −0.0311 | −0.0026 | −0.0106 | +0.0000 | +0.1152 | −0.0210 | +0.9908 |
| 999 | +0.5747 | +0.1859 | +0.1722 | +0.2146 | −0.0508 | −0.0750 | −0.0023 | −0.0102 | +0.0000 | +0.1478 | −0.0198 | +1.1371 |
| 1199 | +0.3681 | +0.1421 | +0.0755 | +0.1706 | +0.0000 | +0.0000 | −0.0023 | −0.0115 | +0.0000 | +0.1875 | −0.0222 | +1.3721 |

Overall mean: `pos +0.2913, pos_fine +0.0929, heading +0.1374, progress +0.3108,
xtrack −0.1086, near_spd −0.0228, act_rate −0.0024, energy −0.0102, arrival +0.0054,
hold +0.0617, sat −0.0228`.

Bounds over all 19200 env-steps (each inside its designed range):

| component | min | max | designed range |
|---|---|---|---|
| pos | 0.000 | +0.999 | [0, `w_position`] |
| pos_fine | 0.000 | +0.498 | [0, `w_position_fine`] |
| heading | 0.000 | +0.300 | [0, `w_heading`] |
| progress | −0.500 | +0.500 | ±`w_progress` |
| xtrack | −0.400 | −0.000 | [−`w_cross_track`, 0] |
| near_spd | −0.690 | −0.000 | [−`w_near_goal_speed`, 0] |
| act_rate | −0.006 | −0.000 | (−∞, 0] |
| energy | −0.020 | −0.001 | [−`w_energy`, 0] |
| arrival | 0.000 | **+3.616** | [0, `w_arrival`] — 3.6 not 10.0: gated |
| hold | 0.000 | +0.500 | [0, `w_hold`] |
| sat | −0.100 | −0.000 | [−`w_sat`, 0] |

The two numbers that matter: **arrival max 3.62 (was a flat 10.0 for every event)** and
**34 events at mean speed 1.117 m/s** — the gate is live and the old fly-through payoff is
gone. `hold` is zero until the policy first parks inside 1.5 m at rest (t ≈ 400) and then
grows, i.e. the term is reachable but sparse, as noted in §6.

Bit-compatibility and parity, same script:

```
[1] pre-R8 (HEAD code + HEAD yaml) vs new code + configs/r8/env_control_A.yaml, 1200x16
    obs bitwise equal = True;  reward bitwise equal = True (max|diff| 0)
[2] obs identical, pre-R8 vs new default config = True  (hash e35fb025208473c761a58438de0ce479)
tests/test_obs_parity.py ................................ 5 passed
```

### Test-suite note (not mine to fix)

`tests/test_reward.py::test_station_keeping_pays_no_arrival_and_keeps_the_fine_kernel`
now **fails** — by exactly `w_hold`:

```
expected gain (0.5 m vs 3.0 m, position kernels only) = 0.48825
observed                                              = 0.98825      Δ = 0.50000 = w_hold
```

The test asserts the position kernels' gain in isolation at station keeping, and the new
dense hold bonus legitimately adds `w_hold × gate(0)` while `d = 0.5 < 1.5`, dropping to 0
at `d = 3.0`. That is the intended R8 change, not a regression. The test needs
`hold = w["w_hold"] * (0.5 < hold_radius) * gate(0)` added to `want` (or `w_hold: 0.0` for
that case). `tests/` is outside my write scope. Everything else passes: `test_reward.py`
(the other 6), `test_env.py`, `test_env_mode_fraction.py`, `test_env_randomization.py` —
32 passed, 1 failed (the one above).

## 8. What I did not do

- No training launched.
- No changes to `tests/`, `evaluate.py`, `evalkit/`, `sim/`, `train.py`, `models.py`,
  `export_onnx.py`, `runtime/`, `common/`, `configs/vehicle.yaml`, `configs/sensors.yaml`,
  `configs/disturbance.yaml`.
- No commits.
- Tuning: every new number is a placeholder. The 20× settlement ratio in §6 is computed at
  the *specified* defaults and is robust to ±50 % on `w_hold`; it is not a claim that these
  weights are right, only that the term is not outgunned by the one-shot bonus.
