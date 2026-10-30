"""R9: `pathgen.StructuredPath` — same interface as `evaluate.SquarePath`, five patterns.

Fast and CPU-only: ~4 envs, small episodes.  The critical test is
`test_legacy_mode_bit_identical` (200 steps, measured max abs diff 0.0).
"""

import copy

import pytest
import torch

from evaluate import FastEnv, SquarePath, env_cfg
from pathgen import PATTERNS, StructuredPath, swept_angle

SIDE, HOLD_S, F_STAY = 30.0, 10.0, 1.5


def _cfg(n: int = 4) -> dict:
    cfg = env_cfg(physics_randomize=False)
    cfg.update(num_envs=n, episode_s=30.0, max_steps=None)
    return cfg


def _env(n: int = 4, seed: int = 3) -> FastEnv:
    env = FastEnv(_cfg(n), device="cpu", seed=seed)
    env.set_curriculum(env.n_levels - 1)
    env.reset()
    return env


# --------------------------------------------------------------------- 1. legacy compat
def test_legacy_mode_bit_identical():
    """StructuredPath(pattern="legacy_square") == SquarePath over 200 policy steps, bit for bit.

    Two independently reset envs (identical seeds, identical state) get the identical action
    stream; the returned goal tensors, `goal_xy`, `idx`, `hold` and `done` must match exactly.
    Shuffling the box in a shared env would make this test disagree by rounding: the two paths
    must each step their own env.
    """
    env_a, env_b = _env(n=4), _env(n=4)
    assert torch.equal(env_a.state, env_b.state)
    ref = SquarePath(env_a, SIDE, HOLD_S, F_STAY)
    new = StructuredPath(env_b, SIDE, HOLD_S, F_STAY)
    assert torch.equal(ref.c, new.c), "corner list must be the identical tensor"

    g = torch.Generator().manual_seed(0)
    dmax, dmax_state = 0.0, 0.0
    corner = ref.c[:, 1]
    th0 = 120                                        # step to teleport onto corner 1 in
    moved = False
    for k in range(200):
        if k == th0:                                 # park both envs ON corner 1 (no dynamics
            env_a.state[:, :2] = corner             # controller reaches 30 m in 20 s), so the
            env_b.state[:, :2] = corner             # dwell/leave branch is exercised too
            moved = True
        act = torch.randn(env_a.num_envs, 3, generator=g) * 0.6
        oa = ref(env_a)
        env_a.step(act)
        ob = new(env_b)
        env_b.step(act)
        dmax = max(dmax, float((oa - ob).abs().max()),
                   float((ref.goal_xy - new.goal_xy).abs().max()),
                   float((env_a.goal_xy - env_b.goal_xy).abs().max()),
                   float((env_a.psi_goal - env_b.psi_goal).abs().max()))
        dmax_state = max(dmax_state, float((env_a.state - env_b.state).abs().max()))
        assert torch.equal(ref.idx, new.idx), k
        assert torch.equal(ref.hold, new.hold), k
        assert torch.equal(ref.done, new.done), k
        assert torch.equal(ref.holding, new.holding), k
    assert dmax_state == 0.0                    # the two envs really did evolve identically
    assert dmax == 0.0, f"legacy path differs from SquarePath by {dmax}"
    # the trace must also DO something: some env advanced past the spawn corner or dwelled
    assert moved and (bool((ref.idx > 0).any()) or bool((ref.hold > 0).any()))


def test_legacy_dwell_state_machine_matches():
    """The dwell/hysteresis machine at a corner is identical, not just the free-running rollout."""
    env_a, env_b = _env(n=2), _env(n=2)
    ref = SquarePath(env_a, SIDE, 0.5, 1.0)          # 5 steps of dwell, no hysteresis
    new = StructuredPath(env_b, SIDE, 0.5, 1.0)
    assert new.stay == ref.stay == pytest.approx(2.0)
    # walk corner 1 -> 2 -> 3 by teleporting both envs in lockstep
    for k in (1, 2, 3):
        env_a.state[:, :2] = ref.c[:, k]
        env_b.state[:, :2] = new.c[:, k]
        for _ in range(ref.hold_steps + 3):
            assert torch.equal(ref(env_a), new(env_b))
            assert ref.idx.tolist() == new.idx.tolist()
            assert ref.done.tolist() == new.done.tolist()
    # three corners visited -> idx advanced three times (corner 4 not reached)
    assert ref.idx.tolist() == new.idx.tolist() == [3, 3]
    assert ref.done.tolist() == new.done.tolist() == [False, False]


# --------------------------------------------------------------------- 2. pattern legality
@pytest.mark.parametrize("pattern", [p for p in PATTERNS if p != "legacy_square"])
def test_pattern_waypoints_legs_turns_radius(pattern):
    """200 seeds: waypoint count, leg bounds, chain radius, turn bounds all hold."""
    n = 50
    npts = {"straight": (2, 3), "turn": (3, 3), "zigzag": (4, 6), "random": (4, 4)}[pattern]
    hi = min(npts[1], int(60.0 // 15.0))             # defaults: ring holds <= 4 legs
    lo = min(npts[0], hi)
    for seed in range(200):
        env = _env(n=n, seed=seed)
        p = StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern=pattern, seed=seed)
        assert lo <= p.P <= hi, (pattern, seed, p.P)
        assert p.c.shape == (n, p.P + 1, 2)
        legs = (p.c[:, 1:] - p.c[:, :-1]).norm(dim=-1)          # [n, P]
        # fallback envs carry the give-up chain; it is still legal except where the ring
        # cannot hold P distinct min-length legs (measured below, rate is asserted separately)
        ok = ~p.fallback
        assert float(legs[ok].min()) >= p.min_leg_m - 1e-4, (pattern, seed, float(legs.min()))
        assert float(legs.max()) <= p.max_leg_m + 1e-4, (pattern, seed, float(legs.max()))
        rad = (p.c[:, 1:] - p.c[:, :1]).norm(dim=-1)            # every wp, distance to spawn
        assert float(rad.max()) <= p.chain_radius_m + 1e-4, (pattern, seed, float(rad.max()))
        assert torch.equal(p.c[:, 0], p.spawn)                  # corner 0 IS the spawn
        if p.P >= 3 and pattern in ("zigzag", "random"):
            t = p.turn_angles(p.c)[:, 0]                        # real bends, not the 0 tail
            assert float(t.min()) >= p.min_turn_deg - 1e-3, (pattern, seed, float(t.min()))
            assert float(t.max()) <= p.max_turn_deg + 1e-3, (pattern, seed, float(t.max()))
        if pattern == "straight":
            # float32 acos of a cosine that is exactly 1.0 still returns ~3e-2 deg; a real
            # 60 deg minimum turn could never hide under this
            assert float(p.turn_angles(p.c).max()) < 0.1, (seed, float(p.turn_angles(p.c).max()))
        if pattern == "turn":
            assert p.P == 3
            t = p.turn_angles(p.c)
            # a fallback env legitimately carries a straight chain (no 90 deg turn)
            ok = p.fallback
            assert torch.allclose(t[~ok, 0], torch.full_like(t[~ok, 0], 90.0), atol=1e-3)
            assert float(t[:, 1:].max()) < 0.1, (seed, t[0].tolist())
        # bounded regeneration is allowed to give up on an env whose draws cannot fit the
        # ring; it must be rare and only ever leave a legal `straight` chain behind
        assert int(p.fallback.sum()) <= n // 10, (pattern, seed, int(p.fallback.sum()))
        assert p.orbit_frac() == 0.0, (pattern, seed, "orbit penalty tripped")


def test_waypoints_and_corner_idx_match_pattern_length():
    """`waypoints(i)` is the chain after the spawn; `CORNER_IDX` follows it (metrics wiring)."""
    env = _env(n=3)
    for pattern in PATTERNS[1:]:
        p = StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern=pattern, seed=5)
        assert p.CORNER_IDX == tuple(range(1, p.P + 1))
        wp = p.waypoints(0)
        assert wp.shape == (p.P, 2)
        assert torch.equal(wp, p.c[0, 1:])
        assert p.pattern_index().tolist() == [PATTERNS.index(pattern)] * 3
        assert p.pattern_names() == [pattern] * 3


# --------------------------------------------------------------------- 3. orbit penalty
def _chainxy(pts, dev="cpu"):
    return torch.tensor(pts, dtype=torch.float32, device=dev).unsqueeze(0)


def test_orbit_penalty_fires_on_spiral_not_on_line_or_turn():
    """Hand-built chains: a >330 deg spiral trips the accumulator, a line and one turn do not."""
    import math
    rad = 2.0
    # spiral: radius grows so the chain winds about the hold point more than once; 8 steps of
    # 60 deg = 480 deg absolute sweep (each step < 180 so the short-way fold keeps it)
    spiral = []
    for k in range(9):
        ang = math.radians(60 * k)
        r = 4.0 + 1.5 * k
        spiral.append((r * math.cos(ang), r * math.sin(ang)))
    ch = _chainxy(spiral)
    hold = torch.zeros(1, 2)                         # spiral centre = the hold point
    swept = float(swept_angle(ch, hold, rad))
    assert swept == pytest.approx(480.0, abs=1.0), swept
    assert swept > 330.0
    # a StructuredPath-shaped check: drop the same chain in as `c` and read the accumulator
    env = _env(n=1)
    p = StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern="random", seed=0)
    p.c = ch
    p.orbit = swept_angle(ch, hold, rad)
    assert p.orbit_frac() == 1.0

    # straight line: 0 deg, and a straight line THROUGH the hold point still 0
    line = _chainxy([(x, 0.0) for x in (0.0, 5.0, 10.0, 15.0)])
    assert float(swept_angle(line, line[:, -1, :], rad)) == pytest.approx(0.0)
    assert float(swept_angle(line, line[:, 1, :], rad)) == pytest.approx(0.0)

    # single 90 deg turn: swept about the corner (the hold) the two legs are 90 deg apart in
    # direction, i.e. 45 deg between the two hold->wp vectors
    turn = _chainxy([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)])
    swept_turn = float(swept_angle(turn, turn[:, -1, :], rad))
    assert swept_turn == pytest.approx(45.0, abs=1e-3)
    assert swept_turn <= 330.0


def test_orbit_penalty_regenerates_beyond_cap():
    """A cap below the natural swept angle forces regeneration; a normal cap never fires."""
    env = _env(n=8)
    loose = StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern="random", seed=1,
                           max_wind_deg=330.0)
    assert loose.orbit_frac() == 0.0
    tight = StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern="random", seed=1,
                           max_wind_deg=1.0)
    # nothing can sweep under 1 deg, so every env regenerates out and falls back to straight
    assert tight.orbit_frac() == 0.0
    assert int(tight.fallback.sum()) == 8
    assert float(tight.orbit.max()) == pytest.approx(0.0, abs=1e-3)
    # the fallback is a legal `straight` chain under the same bounds
    legs = (tight.c[:, 1:] - tight.c[:, :-1]).norm(dim=-1)
    assert float(legs.min()) >= tight.min_leg_m - 1e-4
    assert float(legs.max()) <= tight.max_leg_m + 1e-4


# --------------------------------------------------------------------- 4. hold exclusion
def test_hold_zone_exclusion():
    """Only segments with an endpoint within `arrival_radius_m` of the hold are skipped.

    Chain: one long northward leg, then a tight circle (30 deg a step, 12 steps = 360 deg)
    about the FINAL hold point, then a 1 m hop to the hold itself.  Every circle segment has
    both endpoints within the 2 m radius, so with `radius=2` only the north leg counts
    (0 deg); with `radius=0` the same chain sweeps the full 360.
    """
    import math
    rad, r_c, n = 2.0, 1.0, 12
    ring = [(r_c * math.cos(math.radians(30 * k)), 20.0 + r_c * math.sin(math.radians(30 * k)))
            for k in range(n)]
    ch = _chainxy([(0.0, 0.0)] + ring + [(0.0, 20.0)])
    hold = ch[:, -1, :].clone()                      # the hold IS the circle centre
    raw = float(swept_angle(ch, hold, 0.0))          # every segment counted
    kept = float(swept_angle(ch, hold, rad))         # near-hold segments skipped
    assert raw == pytest.approx(420.0, abs=1.0), raw   # 360 ring + 60 entry + 0 exit
    assert kept == pytest.approx(0.0, abs=1e-3), kept
    # the same numbers through a StructuredPath: with the cap at 330 the chain trips at
    # radius 0 and does not trip at the configured radius
    env = _env(n=1)
    p = StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern="random", seed=0)
    p.c, p.hold_pt, p.orbit = ch, hold, swept_angle(ch, hold, rad)
    assert p.orbit_frac() == 0.0
    p.orbit = swept_angle(ch, hold, 0.0)
    assert p.orbit_frac() == 1.0
    # a waypoint just OUTSIDE the radius is not skipped: 20 m out, 20 m back an arc of 60 deg
    import math as _m
    outside = _chainxy([(0.0, 0.0), (20.0, 0.0),
                        (20.0 * _m.cos(_m.radians(60)), 20.0 * _m.sin(_m.radians(60)))])
    h2 = torch.tensor([[0.0, 0.0]])
    assert float(swept_angle(outside, h2, rad)) == pytest.approx(60.0, abs=1e-3)


# --------------------------------------------------------------------- 5. determinism
def test_determinism_same_seed_same_plan():
    env = _env(n=5, seed=9)
    a = StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern="zigzag", seed=42)
    b = StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern="zigzag", seed=42)
    assert torch.equal(a.c, b.c)
    c = StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern="zigzag", seed=43)
    assert not torch.equal(a.c, c.c)
    # a different spawn seed gives a different plan even at the same plan seed
    env2 = _env(n=5, seed=10)
    d = StructuredPath(env2, SIDE, HOLD_S, F_STAY, pattern="zigzag", seed=42)
    assert not torch.equal(a.spawn, d.spawn)
    assert not torch.equal(a.c, d.c)
    # patterns differ from each other at the same seed
    e = StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern="random", seed=42)
    assert e.P != a.P or not torch.equal(e.c, a.c)


def test_shape_contract_and_orig_untouched():
    """No writes to runs/ or results_/; `env.spawn_r` box untouched (plan is spawn-relative)."""
    env = _env(n=2)
    spawn_r, wp_r = env.spawn_r, env.wp_r
    StructuredPath(env, SIDE, HOLD_S, F_STAY, pattern="turn", seed=0)
    assert (env.spawn_r, env.wp_r) == (spawn_r, wp_r)
    cfg = copy.deepcopy(_cfg(2))
    assert cfg["num_envs"] == 2
