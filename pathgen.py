"""R9: `StructuredPath` — waypoint patterns as an `evaluate.SquarePath` drop-in.

Every waypoint number so far was one fixed 30 m square, so "waypoint tracking" and "one square"
were never separated.  `StructuredPath` keeps the SquarePath state machine VERBATIM (one copy,
in `_StateMachine` below) and only replaces the corner list `c` [N, P+1, 2]: corner 0 = spawn,
corners 1..P in visit order.

    from pathgen import StructuredPath
    StructuredPath(env, side, hold_s, dwell_stay)                       # legacy_square
    StructuredPath(env, side, hold_s, dwell_stay, pattern="zigzag", seed=7)

`pattern="legacy_square"` is `evaluate.SquarePath` bit for bit (tests/test_pathgen.py measures
the max abs diff over 200 steps).  Public surface, same as SquarePath: `.c`, `.idx`, `.hold`,
`.done`, `.holding`, `.goal_xy`, `.arrival`, `.hold_steps`, `waypoints(i)`, `__call__(env)`.
"""

from __future__ import annotations

import math

import torch

from evaluate import SquarePath

PATTERNS = ("legacy_square", "straight", "turn", "zigzag", "random")
NPTS = {"straight": (2, 3), "turn": (3, 3), "zigzag": (4, 6), "random": (4, 4)}
TODO_LEG_M = 30.0      # non-legacy leg length when no `side` is meaningful
TRIES = 32             # plan regenerations per env before falling back to "straight"


# --------------------------------------------------------------------------- geometry
def _rot(v: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
    """Rotate (..., 2) heading by angle `deg` (..., ) rad; positive = CCW."""
    cs, sn = torch.cos(deg).unsqueeze(-1), torch.sin(deg).unsqueeze(-1)
    x, y = v[..., 0:1], v[..., 1:2]
    return torch.cat((x * cs - y * sn, x * sn + y * cs), dim=-1)


def swept_angle(chain: torch.Tensor, hold: torch.Tensor, radius: float) -> torch.Tensor:
    """Absolute swept angle (deg) of `chain` (..., M, 2) about `hold` (..., 2).

    Sum of |signed angle| between consecutive hold->p vectors.  Segments with an endpoint
    within `radius` of the hold point are skipped.  A sweep > 180 deg counts the short way
    round (2pi - |dtheta|), otherwise an atan2 branch cut alone reports a whole extra turn;
    exact zero stays zero, so a straight line through the hold point sweeps 0.
    """
    v = chain - hold.unsqueeze(-2)
    dth = torch.atan2(v[..., 1:, 0] * v[..., :-1, 1] - v[..., 1:, 1] * v[..., :-1, 0],
                      (v[..., 1:, :] * v[..., :-1, :]).sum(-1)).abs()
    dth = torch.where(dth > math.pi, 2.0 * math.pi - dth, dth)
    dth = torch.where(dth < 1e-12, torch.zeros_like(dth), dth)
    near = v.norm(dim=-1) < radius
    return torch.rad2deg(torch.where(near[..., 1:] | near[..., :-1],
                                     torch.zeros_like(dth), dth).sum(-1))


# --------------------------------------------------------------------------- state machine
class _StateMachine(SquarePath):
    """SquarePath's state machine, inherited unchanged; only `__init__`/`__call__` differ.

    `__init__` does not build a square: `_build(env)` sets `self.c` [N, P+1, 2] (corner 0 =
    spawn) and may set `self.arrival_r`.  `__call__` is SquarePath's body with the hardcoded 4
    replaced by `c.shape[1] - 1`, which is what makes it work for any P.
    """

    arrival_r: float | None = None

    def _build(self, env) -> None:                               # pragma: no cover
        raise NotImplementedError

    def __init__(self, env, side, hold_s, dwell_stay=1.5) -> None:
        self._build(env)
        self.P = self.c.shape[1] - 1
        self.CORNER_IDX = tuple(range(1, self.P + 1))
        self.hold_pt = self.c[:, -1].clone()                     # FINAL hold point
        r = float(env.arrival if self.arrival_r is None else self.arrival_r)
        self.orbit = swept_angle(self.c, self.hold_pt, r)
        self.hold_steps = max(1, int(round(hold_s / env.policy_dt)))
        z = lambda: torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # noqa: E731
        self.idx, self.hold = z(), z()
        self.done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.arrival = float(env.arrival)
        self.dwell_stay = float(dwell_stay)
        self.stay = self.arrival * self.dwell_stay
        self.holding, self.goal_xy = self.done.clone(), self.c[:, 0].clone()

    def waypoints(self, i: int) -> torch.Tensor:
        """Chain after the spawn, [P, 2], in visit order."""
        return self.c[i, list(self.CORNER_IDX)]

    def __call__(self, env):
        c = self.c
        last = c.shape[1] - 1
        a = torch.arange(env.num_envs, device=c.device)
        pos = env.state[:, :2]
        nxt = (self.idx + 1).clamp(max=last)
        dist = (c[a, nxt] - pos).norm(dim=1)
        live = ~self.done
        dwell = torch.where(dist < self.arrival, self.hold + 1,
                            torch.where((self.hold > 0) & (dist < self.stay), self.hold + 1,
                                        torch.zeros_like(self.hold)))
        self.hold = torch.where(live, dwell, self.hold)
        leave = live & (self.hold > self.hold_steps)
        self.hold = torch.where(leave, torch.zeros_like(self.hold), self.hold)
        is_last = leave & (nxt == last)
        self.idx = torch.where(leave & ~is_last, nxt, self.idx)
        self.done = self.done | is_last

        holding = live & (self.hold > 0)
        goal_xy = c[a, torch.where(holding, nxt, (self.idx + 1).clamp(max=last))]
        leg_start = c[a, self.idx.clamp(max=last - 1)]
        d = goal_xy - pos
        bearing = torch.atan2(d[:, 0], d[:, 1])
        env.hold_psi.copy_(torch.where(holding & (self.hold <= 1), env.state[:, 2],
                                       env.hold_psi))
        env.goal_xy.copy_(goal_xy)
        env.leg_start.copy_(torch.where(holding.unsqueeze(1), goal_xy, leg_start))
        env.mode.copy_(holding.to(env.dtype))
        env.hold_xy.copy_(goal_xy)
        env.psi_goal.copy_(torch.where(holding, env.hold_psi, bearing))
        self.holding, self.goal_xy = holding, goal_xy.clone()
        return torch.stack([goal_xy[:, 0], goal_xy[:, 1], env.psi_goal, env.mode], dim=1)


class StructuredPath(_StateMachine):
    """Waypoint chain per env, one of `PATTERNS`.  Same interface as `SquarePath`.

    env   : AsvEnv — read for the spawn pose/heading, `arrival`, `dtype`, `device`.
    side  : float — `legacy_square` leg length (m); other patterns use `TODO_LEG_M`.
    hold_s, dwell_stay — straight to the state machine.

    `orient_deg=None` (default) starts each chain along the env's spawn heading, so the plan
    does not hand the policy a free initial turn; `legacy_square` ignores it.
    """

    def __init__(self, env, side, hold_s, dwell_stay=1.5, *,
                 pattern: str = "legacy_square", seed: int = 0,
                 min_leg_m: float = 15.0, max_leg_m: float = 45.0,
                 chain_radius_m: float = 60.0, min_turn_deg: float = 60.0,
                 max_turn_deg: float = 120.0, max_wind_deg: float = 330.0,
                 arrival_radius_m: float = 2.0, orient_deg: float | None = None) -> None:
        if pattern not in PATTERNS:
            raise ValueError(f"pattern {pattern!r} not in {PATTERNS}")
        self.pattern, self.seed, self.side = pattern, int(seed), float(side)
        self.min_leg_m, self.max_leg_m = float(min_leg_m), float(max_leg_m)
        self.chain_radius_m, self.arrival_r = float(chain_radius_m), float(arrival_radius_m)
        self.min_turn_deg, self.max_turn_deg = float(min_turn_deg), float(max_turn_deg)
        self.max_wind_deg, self.orient_deg = float(max_wind_deg), orient_deg
        super().__init__(env, side, hold_s, dwell_stay)

    # ------------------------------------------------------------------ plan
    def _build(self, env) -> None:
        n, dev, dt = env.num_envs, env.device, env.dtype
        self.arrival_r = float(self.arrival_r)          # narrows the Optional for `_sample`
        self.spawn = env.state[:, :2].clone()
        self.spawn_psi = env.state[:, 2].clone()
        # [N, 2]: spawn point per env, shaped for the chain arithmetic below
        self.origin = self.spawn
        if self.pattern == "legacy_square":
            off = torch.tensor([[0.0, 0.0], [self.side, 0.0], [self.side, self.side],
                                [0.0, self.side], [0.0, 0.0]], device=dev, dtype=dt)
            self.c = self.spawn.unsqueeze(1) + off.unsqueeze(0)
            self.fallback = torch.zeros(n, dtype=torch.bool, device=dev)
            return
        g = torch.Generator(device=dev)
        # fold the pattern into the plan seed: otherwise `random` and a k=4 `zigzag` at the
        # same seed draw the identical stream and produce identical plans
        g.manual_seed(self.seed * len(PATTERNS) + PATTERNS.index(self.pattern))
        lo, hi = NPTS[self.pattern]
        # the ring can hold at most floor(chain_radius_m / min_leg_m) distinct legs
        hi = min(hi, max(1, int(self.chain_radius_m // self.min_leg_m)))
        lo = min(lo, hi)
        k = int(torch.randint(lo, hi + 1, (1,), generator=g, device=dev)) if hi > lo else lo
        self.c, self.fallback = self._sample(g, n, k, dev, dt)

    def _draw(self, g, shape, lo, hi, dev, dt):
        """Uniform [lo, hi] floats from the plan generator `g`."""
        return lo + (hi - lo) * torch.rand(shape, generator=g, device=dev, dtype=dt)

    def _sample(self, g, n: int, k: int, dev, dt):
        """Bounded regeneration.  Returns (chain [N, k+1, 2], fallback mask [N])."""
        if self.orient_deg is None:
            th = self.spawn_psi
        else:
            th = torch.full((n,), math.radians(self.orient_deg), device=dev, dtype=dt)
        u = torch.stack((torch.sin(th), torch.cos(th)), dim=1)   # compass bearing -> e/n
        chain, fallback = None, torch.ones(n, dtype=torch.bool, device=dev)
        if self.pattern == "straight":
            # Collinear chains cannot be re-rolled into a different plan: a straight leg past
            # `chain_radius_m` is infeasible, not unlucky, so clamp the legs instead.
            k_max = max(1, min(k, int(self.chain_radius_m // self.min_leg_m)))
            legs = self._draw(g, (n, k_max), self.min_leg_m, self.max_leg_m, dev, dt)
            legs = legs.clamp(max=self.chain_radius_m / k_max)   # last wp lands on the ring
            pts = [self.origin]
            for j in range(k_max):
                pts.append(pts[-1] + u * legs[:, j:j + 1])
            return torch.stack(pts, dim=1), torch.zeros(n, dtype=torch.bool, device=dev)
        for _ in range(TRIES):
            legs = self._draw(g, (n, k), self.min_leg_m, self.max_leg_m, dev, dt)
            signs = torch.where(torch.rand((n, k), generator=g, device=dev) < 0.5, -1.0, 1.0)
            deg = self._draw(g, (n, k), self.min_turn_deg, self.max_turn_deg, dev, dt)
            if self.pattern == "turn":
                deg[:] = 0.0                             # exactly one turn, deg[:, 0], no other
                deg[:, 0] = 90.0
            # turn j bends leg j+1 away from leg j; deg[:, -1] is unused (no leg after it)
            d, p, pts = u, self.origin, [self.origin]
            for j in range(k):
                p = p + d * legs[:, j:j + 1]
                pts.append(p)
                if j < k - 1:
                    d = _rot(d, torch.deg2rad(deg[:, j]) * signs[:, j])
            cand = torch.stack(pts, dim=1)
            ok = (cand - self.origin.unsqueeze(1)).norm(dim=-1).max(dim=1).values \
                <= self.chain_radius_m
            ok &= swept_angle(cand, cand[:, -1, :], self.arrival_r) <= self.max_wind_deg
            chain = cand if chain is None else torch.where(ok[:, None, None], cand, chain)
            fallback = fallback & ~ok
            if not bool(fallback.any()):
                break
        if bool(fallback.any()):
            chain = self._straight(chain, fallback, k)
        return chain, fallback

    def _straight(self, chain, mask, k):
        """`straight` over the envs whose draws never produced a legal plan."""
        chain = chain.clone()
        if self.orient_deg is None:
            th = self.spawn_psi[mask]
        else:
            th = torch.full((int(mask.sum()),), math.radians(self.orient_deg),
                            device=chain.device, dtype=chain.dtype)
        u = torch.stack((torch.sin(th), torch.cos(th)), dim=1)
        # as many min-leg legs as the ring holds; any surplus waypoints repeat the last point
        # (degenerate, but `_straight` is the give-up path and every wp stays inside the ring)
        k_leg = max(1, min(k, int(self.chain_radius_m // self.min_leg_m)))
        leg = max(self.min_leg_m, min(TODO_LEG_M, self.chain_radius_m / k_leg))
        for j in range(1, k + 1):
            chain[mask, j, :] = chain[mask, 0, :] + u * (leg * min(j, k_leg))
        return chain

    @staticmethod
    def turn_angles(chain: torch.Tensor) -> torch.Tensor:
        """Interior turn angles (deg) per chain, [N, P-1]; diagnostics/tests."""
        d1, d2 = chain[:, 1:-1, :] - chain[:, :-2, :], chain[:, 2:, :] - chain[:, 1:-1, :]
        cos = (d1 * d2).sum(-1) / (d1.norm(dim=-1).clamp(min=1e-9)
                                   * d2.norm(dim=-1).clamp(min=1e-9))
        return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))

    # ------------------------------------------------------------------ logging
    def pattern_names(self) -> list[str]:
        """`self.pattern` repeated per env (for a log column)."""
        return [self.pattern] * self.c.shape[0]

    def pattern_index(self) -> torch.Tensor:
        """[N] long tensor: index of `self.pattern` in `PATTERNS`."""
        return torch.full((self.c.shape[0],), PATTERNS.index(self.pattern),
                          dtype=torch.long, device=self.c.device)

    def orbit_frac(self) -> float:
        """Fraction of envs whose chain swept more than `max_wind_deg` about the final hold."""
        return float((self.orbit > self.max_wind_deg).to(torch.float64).mean())
