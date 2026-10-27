"""R5: SquarePath state machine (reports/R5_E.md, finding 4).

A scripted agent teleports a real `FastEnv` between waypoints, so the path logic is exercised
with the env's own `state` / `goal_xy` / `mode` plumbing but no dynamics.  The spawn poses stay
random (physics randomisation off), which is the point: R4 pinned the hold goal to `c[:, 1]` for
every env, so `waypoints_reached` could never exceed 1.

Cheap enough for the default suite: ~60 policy steps per rollout, 4 envs.
"""

import copy

import numpy as np
import pytest
import torch

from evaluate import (SCENARIOS, TW_COLS, FastEnv, SquarePath, episode_arrays, run_rollout,
                      score, tw_columns)


def _env(n: int = 4, device: str = "cpu") -> FastEnv:
    from evaluate import env_cfg
    cfg = env_cfg(physics_randomize=False)
    cfg.update(num_envs=n, episode_s=10.0, max_steps=None)
    env = FastEnv(cfg, device=device, seed=3)
    env.set_curriculum(env.n_levels - 1)
    env.reset()
    return env


def _lap(env: FastEnv, hold_s: float, dwell_stay: float = 1.5):
    """Teleport ON each corner in turn, wait out the dwell, walk to the next; return the trace."""
    path = SquarePath(env, 30.0, hold_s, dwell_stay)
    trace = {"xy": [], "goal_xy": [], "mode": []}
    step = path.hold_steps + 2                       # dwell plus the leave step
    for k in range(1, 5):
        env.state[:, :2] = path.c[:, k]
        for _ in range(step):
            path(env)
            trace["xy"].append(env.state[:, :2].clone())
            trace["goal_xy"].append(env.goal_xy.clone())
            trace["mode"].append(env.mode.clone())
    for key in trace:
        trace[key] = torch.stack(trace[key])
    return path, trace


@pytest.mark.parametrize("hold_s", [0.0, 10.0])
def test_full_lap_reaches_all_four_corners(hold_s):
    env = _env()
    path, tr = _lap(env, hold_s)
    dt = env.policy_dt
    for i in range(env.num_envs):
        wps = path.waypoints(i).numpy().astype(np.float64)
        # the teleport parks on the tracked goal, so xy == goal_xy at every sample: the
        # walk in waypoint_progress sees the corner the moment the goal becomes that corner
        m = score({"t": np.arange(tr["goal_xy"].shape[0]) * dt,
                   "xy": tr["xy"][:, i].numpy(),
                   "goal_xy": tr["goal_xy"][:, i].numpy(),
                   "psi": np.zeros(tr["goal_xy"].shape[0]),
                   "action": np.zeros((tr["goal_xy"].shape[0], 3)),
                   "mode": tr["mode"][:, i].numpy()},
                  tol=1.0, hold_s=5.0, max_thrust=1.0,
                  path=path.waypoints(i).numpy(),      # cross-track against corners 1..4
                  waypoints=wps, arrival=path.arrival)
        assert m["waypoints_reached"] == 4, (hold_s, i, m["waypoints_reached"])
        assert m["lap_completed"] is True
        assert m["lap_pct"] == 100.0
    # tw_columns is fed by summarize(), which score() pops after use
    from evalkit.metrics import summarize
    raw = summarize({"t": np.arange(tr["goal_xy"].shape[0]) * dt,
                     "xy": tr["xy"][:, 0].numpy(), "goal_xy": tr["goal_xy"][:, 0].numpy(),
                     "psi": np.zeros(tr["goal_xy"].shape[0]),
                     "action": np.zeros((tr["goal_xy"].shape[0], 3)),
                     "mode": tr["mode"][:, 0].numpy()},
                    waypoints=path.waypoints(0).numpy(), arrival_radius=path.arrival)
    cols = tw_columns(raw)
    assert set(cols) == set(TW_COLS)
    assert np.all(np.isfinite(np.array(list(cols.values()), dtype=float)))
    assert set(tw_columns({})) == set(TW_COLS)
    assert all(np.isnan(v) for v in tw_columns({}).values())


def test_dwell_hysteresis_stay_factor():
    """Parking 2.5 m off corner 1: F=1.0 never advances, F=1.5 advances (touch < 2.0 m once)."""
    # --- F=1.0: 2.5 m > arrival 2.0 m -> the counter never starts, no advance ---
    env = _env(n=2)
    path = SquarePath(env, 30.0, 0.5, 1.0)             # 5 steps of dwell
    assert path.stay == pytest.approx(2.0)
    env.state[:, :2] = path.c[:, 1] + torch.tensor([[2.5, 0.0]])
    for _ in range(20):
        path(env)
    assert path.idx.tolist() == [0, 0] and path.hold.tolist() == [0, 0]
    assert path.holding.tolist() == [False, False]

    # --- F=1.5: touch < 2.0 m once (starts the counter), then hold at 2.5 m < 3.0 m ---
    env2 = _env(n=2)
    p2 = SquarePath(env2, 30.0, 0.5, 1.5)
    assert p2.stay == pytest.approx(3.0)
    env2.state[:, :2] = p2.c[:, 1] + torch.tensor([[2.5, 0.0]])
    p2(env2)                                           # no touch yet: not started
    assert p2.hold.tolist() == [0, 0]
    env2.state[:, :2] = p2.c[:, 1] + torch.tensor([[1.5, 0.0]])     # one touch below arrival
    p2(env2)
    assert p2.hold.tolist() == [1, 1]
    env2.state[:, :2] = p2.c[:, 1] + torch.tensor([[2.5, 0.0]])     # back to 2.5 m < 3.0 m
    for k in range(2, 6):
        p2(env2)
        assert p2.hold.tolist() == [k, k]              # keeps counting: hysteresis
    p2(env2)                                           # hold 6 > hold_steps 5 -> leave
    assert p2.idx.tolist() == [1, 1]                   # advanced to the next corner
    # --- F=1.0 with the same trace: the 2.5 m step resets the counter, it never leaves ---
    env3 = _env(n=2)
    p3 = SquarePath(env3, 30.0, 0.5, 1.0)
    env3.state[:, :2] = p3.c[:, 1] + torch.tensor([[1.5, 0.0]])
    p3(env3)
    env3.state[:, :2] = p3.c[:, 1] + torch.tensor([[2.5, 0.0]])
    for _ in range(10):
        p3(env3)
    assert p3.idx.tolist() == [0, 0] and p3.hold.tolist() == [0, 0]
    # --- beyond dwell_stay*arrival (3.0 m) even F=1.5 resets ---
    env4 = _env(n=2)
    p4 = SquarePath(env4, 30.0, 0.5, 1.5)
    env4.state[:, :2] = p4.c[:, 1] + torch.tensor([[1.5, 0.0]])
    p4(env4)
    env4.state[:, :2] = p4.c[:, 1] + torch.tensor([[3.5, 0.0]])
    p4(env4)
    assert p4.hold.tolist() == [0, 0]


def test_goal_after_first_leave_is_corner_2_not_corner_1():
    env = _env()
    path = SquarePath(env, 30.0, 1.0)
    env.state[:, :2] = path.c[:, 1]
    for _ in range(11):                     # 1.0 s dwell = 10 steps, leave on the 11th
        path(env)
    assert path.idx.tolist() == [1] * env.num_envs
    assert path.done.tolist() == [False] * env.num_envs       # done means LAP finished
    assert path.holding.tolist() == [False] * env.num_envs    # back to waypoint mode
    env.state[:, :2] = path.c[:, 1] + torch.tensor([[5.0, 0.0]])
    goal = path(env)
    d0 = (goal[:, :2] - path.c[:, 2]).norm(dim=1)
    d1 = (goal[:, :2] - path.c[:, 1]).norm(dim=1)
    assert torch.all(d0 == 0) and torch.all(d1 > 0)           # pinned c[:,1] bug is gone
    assert env.mode.tolist() == [0.0] * env.num_envs
    # the hold itself also points at the ACTIVE corner, not c[:, 1]
    env.state[:, :2] = path.c[:, 2]
    path(env)
    assert torch.all(path.holding)
    assert torch.all((path.goal_xy - path.c[:, 2]).norm(dim=1) == 0)


def test_path_has_five_points_and_starts_at_spawn():
    env = _env(n=2)
    cfg = dict(SCENARIOS["wp_square"])
    rec = run_rollout("pid", None, torch.device("cpu"), cfg,
                      seed=5, episodes=1, num_envs=2, rough_scale=2.0,
                      square_side=30.0, hold_s=0.0, dwell_stay=1.5)
    p = rec["_path"]
    assert p.shape == (2, 5, 2)
    # first rollout sample = the spawn pose the reset drew; corner 0 IS that sample
    assert np.allclose(p[:, 0, :], rec["xy"][0], atol=1e-3)    # xy[0] is one step after spawn
    assert np.allclose(p[:, 4, :], p[:, 0, :])          # square closes on the spawn point
    legs = np.linalg.norm(np.diff(p, axis=1), axis=2)
    assert np.allclose(legs, 30.0, atol=1e-4)
    # waypoint list for the R4 counters is still corners 1..4
    assert [w.shape for w in rec["_waypoints"]] == [(4, 2), (4, 2)]
    assert np.allclose(rec["_waypoints"][0][0], p[0, 1])
    assert TW_COLS == [f"time_to_waypoint_{k}" for k in range(1, 5)]
    assert set(tw_columns({"time_to_waypoint": [1.0, float("nan")]})) == set(TW_COLS)
