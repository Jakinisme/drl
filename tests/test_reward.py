"""R4: the waypoint "parking" exploit — reward arithmetic only, no training.

Hand-set state, `_reward`/`step` called directly. Every test names the number it expects, so
these test the formula in CONTRACT §3.4 (reports/R4.md), not the vessel: the physics is
placeholder-parameterised, these rewards say nothing about the real ASV.
"""

import copy
import math

import pytest
import torch

from envs.asv_env import AsvEnv, load_config

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _env(n=2, **over):
    cfg = copy.deepcopy(load_config())
    for path, val in over.items():                      # "reward.w_arrival": 10.0
        node = cfg
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = val
    return AsvEnv(cfg=cfg, num_envs=n, device=DEV, seed=11)


def _wp_env(n, n_wp=2, max_wp=3):
    """Waypoint mode, every env at the origin facing north, so psi_goal=0 (no heading
    penalty) and leg 1 runs north along y with waypoint 0 at (0, 30)."""
    env = _env(n, **{"episode.n_waypoints": [max_wp, max_wp],
                     "episode.station_keeping_fraction": 0.0})
    env.mode[:] = 0.0
    env.n_wp[:] = n_wp
    env.wp_idx[:] = 0
    env.state[:] = 0.0
    env.wps[:, 0] = torch.tensor([0.0, 30.0], device=DEV)
    env.wps[:, 1] = torch.tensor([0.0, 60.0], device=DEV)
    env.goal_xy[:] = env.wps[:, 0]
    env.leg_start[:] = 0.0
    env.psi_goal[:] = 0.0
    env.final_latch[:] = False
    return env


def _old_reward(env, d, dd, a, a_prev, pos_sigma_m=10.0, xtrack_sigma_m=None):
    """The pre-R4 formula (reports/R4.md §1), for recording how the numbers moved."""
    w = env.w
    speed = env.state[:, 3:5].norm(dim=1)
    perr = (env.psi_goal - env.state[:, 2]).abs()
    near = (d < env.slowdown).to(env.dtype)
    sig = pos_sigma_m if xtrack_sigma_m is None else xtrack_sigma_m
    return (w["w_position"] * torch.exp(-d / pos_sigma_m)
            + w["w_heading"] * torch.exp(-perr / w["head_sigma_rad"])
            + w["w_progress"] * (dd / (env.max_surge * env.policy_dt)).clamp(-1.0, 1.0)
            - w["w_cross_track"] * (env._xtrack() / sig).pow(2).clamp(max=1.0)
            - w["w_near_goal_speed"] * (speed / env.max_surge).pow(2) * near
            - w["w_action_rate"] * (a - a_prev).pow(2).mean(dim=1)
            - w["w_energy"] * a.pow(2).mean(dim=1))


def _parked_state(env, d=3.4):
    """Stopped 3.4 m short of the first (intermediate) waypoint, bow on the goal."""
    env.state[:, :2] = env.goal_xy - torch.tensor([0.0, d], device=DEV)
    env.state[:, 3:5] = 0.0
    return env.state[:, :2]


def _moving_state(env, d=15.0, u=0.5, psi=0.0):
    """`d` m short of the first waypoint, doing 0.5 m/s along heading `psi` (0 = north)."""
    env.state[:, :2] = env.goal_xy - torch.tensor([0.0, d], device=DEV)
    env.state[:, 2] = psi
    env.state[:, 3] = u * math.cos(psi)
    env.state[:, 4] = u * math.sin(psi)
    return env.state[:, :2]


# ---------------------------------------------------------------- (1) parking vs moving
def test_parking_at_intermediate_waypoint_beats_moving_on_the_old_reward():
    """Proof the old formula is the bug, with the geometry of the measured run
    (reports/R4.md §1): parked 3.4 m short of waypoint 0, bow 20 deg off the line, earns
    exp(-3.4/10) = 0.71/step; still 15 m out, 0.5 m/s, aimed at the waypoint, earns
    exp(-15/10) + 0.5*1 + 0.3 = 1.02/step. Parking wins, and parking never ends."""
    env = _wp_env(1)
    a = torch.zeros(1, 3, device=DEV)
    _parked_state(env)
    d_park = env._goal_dist()
    park_old = _old_reward(env, d_park, torch.zeros(1, device=DEV), a, a)
    _moving_state(env, d=14.5, psi=math.radians(20.0))  # 0.5 m/s * 0.1 s, 20 deg off
    d_move = env._goal_dist()
    move_old = _old_reward(env, d_move, torch.full((1,), 0.5, device=DEV), a, a)
    assert float(park_old) > float(move_old)            # old reward prefers the stall
    assert float(park_old) == pytest.approx(math.exp(-3.4 / 10.0) + 0.3, abs=1e-5)


def test_parking_at_intermediate_waypoint_loses_to_moving_now():
    env = _wp_env(2, n_wp=3)                            # waypoint 0 is not the last
    a = torch.zeros(2, 3, device=DEV)
    _parked_state(env)
    park = env._reward(env._goal_dist(), torch.zeros(2, device=DEV), a, a)
    _moving_state(env, d=14.5)
    move = env._reward(env._goal_dist(), torch.full((2,), 0.5, device=DEV), a, a)
    assert (move > park).all(), (park, move)
    # where the gap comes from: only the heading + progress terms are live off the final goal
    exp_park = env.w["w_heading"] * math.exp(-0.0 / env.w["head_sigma_rad"])
    assert float(park[0]) == pytest.approx(exp_park, abs=1e-5)


# ---------------------------------------------------------------- (2) arrival bonus
def test_arrival_bonus_fires_once_on_advance_and_equals_w_arrival():
    env = _wp_env(3, n_wp=3)
    a = torch.zeros(3, 3, device=DEV)
    env.goal_xy[:] = torch.tensor([0.0, 1.0], device=DEV)        # inside arrival_radius_m
    idx0 = env.wp_idx.clone()
    base = env._reward(env._goal_dist(), torch.zeros(3, device=DEV), a, a)
    _, rew, done, info = env.step(a)
    assert done.max() == 0.0                                     # no termination in the way
    assert (env.wp_idx == idx0 + 1).all()
    assert torch.allclose(info["arrive_evt"], torch.ones(3, device=DEV))
    assert (info["wp_idx"] == env.wp_idx).all() and (info["n_wp"] == 3).all()
    # exactly w_arrival more than the same state without the event
    assert torch.allclose(rew - base - env.w["w_arrival"],
                          torch.zeros(3, device=DEV), atol=0.2)
    _, _, _, info2 = env.step(a)                                 # sitting at wp1 pays nothing
    assert info2["arrive_evt"].abs().max() < 1e-6


# ---------------------------------------------------------------- (3) final waypoint
def test_final_waypoint_bonus_is_latched():
    env = _wp_env(2, n_wp=1)                                     # waypoint 0 IS the final goal
    a = torch.zeros(2, 3, device=DEV)
    env.goal_xy[:] = torch.tensor([0.0, 1.0], device=DEV)
    _, _, _, info = env.step(a)
    assert torch.allclose(info["arrive_evt"], torch.ones(2, device=DEV))
    env.state[:, :2] = env.goal_xy + torch.tensor([0.0, 10.0], device=DEV)  # leave
    _, _, _, info = env.step(a)
    assert info["arrive_evt"].abs().max() < 1e-6
    env.state[:, :2] = env.goal_xy - torch.tensor([0.0, 1.0], device=DEV)   # re-enter
    _, _, _, info = env.step(a)
    assert info["arrive_evt"].abs().max() < 1e-6
    env.reset()                                                 # new episode, new coupon
    assert not bool(env.final_latch.any())
    env.mode[:] = 0.0
    env.n_wp[:] = 1
    env.psi_goal[:] = 0.0
    env.goal_xy[:] = env.state[:, :2] + torch.tensor([0.0, 1.0], device=DEV)
    _, _, _, info = env.step(a)
    assert torch.allclose(info["arrive_evt"], torch.ones(2, device=DEV))


# ---------------------------------------------------------------- (4) station keeping
def test_station_keeping_pays_no_arrival_and_keeps_the_fine_kernel():
    env = _wp_env(2, n_wp=2)
    env.mode[:] = 1.0
    env.n_wp[:] = 0
    env.psi_goal[:] = 0.0
    a = torch.zeros(2, 3, device=DEV)
    env.goal_xy[:] = torch.tensor([0.0, 1.0], device=DEV)        # inside arrival radius
    _, _, _, info = env.step(a)
    assert info["arrive_evt"].abs().max() < 1e-6                 # never an arrival event

    def at(d):
        env.state[:, :2] = env.goal_xy - torch.tensor([0.0, d], device=DEV)
        env.state[:, 3:5] = 0.0
        return env._reward(env._goal_dist(), torch.zeros(2, device=DEV), a, a)

    near, far = at(0.5), at(3.0)
    gain = float(near[0] - far[0])
    w = env.w
    want = (w["w_position"] * (math.exp(-0.5 / w["pos_sigma_m"]) - math.exp(-3.0 / w["pos_sigma_m"]))
            + w["w_position_fine"] * (math.exp(-0.5 / w["pos_sigma_fine_m"])
                                      - math.exp(-3.0 / w["pos_sigma_fine_m"])))
    assert gain == pytest.approx(want, abs=1e-4)
    assert gain > w["w_position"] * (math.exp(-0.5 / 10.0) - math.exp(-3.0 / 10.0))  # fine kernel adds


# ---------------------------------------------------------------- (5) intermediate WP
def test_intermediate_waypoint_has_no_position_or_near_goal_speed_term():
    env = _wp_env(2, n_wp=3)
    a = torch.zeros(2, 3, device=DEV)
    for d in (0.5, 3.0, 30.0):
        env.state[:, :2] = env.goal_xy - torch.tensor([0.0, d], device=DEV)
        env.state[:, 3] = 0.0
        r = env._reward(env._goal_dist(), torch.zeros(2, device=DEV), a, a)
        assert float(r[0]) == pytest.approx(env.w["w_heading"], abs=1e-5)
    # and a fast approach inside slowdown_radius is not fined (was: w_near_goal_speed);
    # the only speed-dependent terms left are action-rate and energy, both on `a`
    env.state[:, :2] = env.goal_xy - torch.tensor([0.0, 4.0], device=DEV)
    env.state[:, 3] = env.max_surge
    fast = env._reward(env._goal_dist(), torch.zeros(2, device=DEV), a, a)
    env.state[:, 3] = 0.0
    slow = env._reward(env._goal_dist(), torch.zeros(2, device=DEV), a, a)
    assert torch.allclose(fast, slow, atol=1e-6)


# ---------------------------------------------------------------- (6) cross-track
def test_cross_track_15m_off_the_leg_costs_10x_more():
    env = _wp_env(1, n_wp=2)
    a = torch.zeros(1, 3, device=DEV)
    env.state[:, :2] = torch.tensor([[1.5, 15.0]], device=DEV)  # 1.5 m east of the leg
    xt = float(env._xtrack()[0])
    assert xt == pytest.approx(1.5, abs=1e-5)
    w = env.w
    new = w["w_cross_track"] * min((xt / w["xtrack_sigma_m"]) ** 2, 1.0)
    old = 0.2 * min((xt / 10.0) ** 2, 1.0)      # pre-R4 term (reports/R4.md §2)
    assert new >= 10.0 * old
