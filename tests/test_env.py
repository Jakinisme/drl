"""Tests for envs/asv_env.py — shapes, rate limit, clock, termination, randomisation."""

import copy

import torch

from envs.asv_env import AsvEnv, load_config

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = 16


def _cfg(**over):
    cfg = copy.deepcopy(load_config())
    for path, val in over.items():
        node = cfg
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = val
    return cfg


def _env(**over):
    return AsvEnv(cfg=_cfg(**over), num_envs=N, device=DEV, seed=7)


def test_dims_and_dtypes():
    env = _env()
    obs = env.reset()
    assert env.obs_dim == 150 and env.act_dim == 3
    assert env.dt == 0.02 and abs(env.policy_dt - 0.1) < 1e-9
    assert env.substeps == 5 and env.max_steps == 1200
    assert obs.shape == (N, 150) and obs.dtype == torch.float32
    obs, rew, done, info = env.step(torch.zeros(N, 3, device=DEV))
    assert obs.shape == (N, 150) and obs.dtype == torch.float32
    assert rew.shape == (N,) and rew.dtype == torch.float32 and torch.isfinite(rew).all()
    assert done.shape == (N,) and set(done.unique().tolist()) <= {0.0, 1.0}
    for key in ("xy", "goal_xy", "psi", "mode", "time"):
        assert key in info
    assert info["xy"].shape == (N, 2) and info["psi"].shape == (N,)
    assert isinstance(info["time"], float)
    assert env.get_privileged_state().shape == (N, 6)


def test_reset_fills_all_history_slots():
    env = _env()
    obs = env.reset().reshape(N, 10, 15)
    for k in range(10):
        assert torch.allclose(obs[:, k], obs[:, 0]), f"history slot {k} differs at reset"


def test_history_shifts_after_step():
    env = _env()
    obs0 = env.reset().reshape(N, 10, 15)
    obs1, _, _, _ = env.step(torch.zeros(N, 3, device=DEV))
    obs1 = obs1.reshape(N, 10, 15)
    assert torch.allclose(obs1[:, :-1], obs0[:, 1:])
    assert not torch.allclose(obs1[:, -1], obs0[:, -1])


def test_action_rate_limit():
    env = _env()
    env.reset()
    lim = env.rate_limit * env.policy_dt          # 2.0 / s * 0.1 s = 0.2
    env.step(torch.ones(N, 3, device=DEV))        # wants +1 from 0
    assert env.prev_action.abs().max() <= lim + 1e-6
    env.step(torch.ones(N, 3, device=DEV))
    assert env.prev_action.abs().max() <= 2 * lim + 1e-6
    env.step(-torch.ones(N, 3, device=DEV))       # full reversal is also rate limited
    assert (env.prev_action - 2 * lim).abs().max() <= lim + 1e-6


def test_action_is_clipped():
    env = _env()
    env.reset()
    for _ in range(20):
        env.step(torch.full((N, 3), 5.0, device=DEV))
    assert env.prev_action.abs().max() <= 1.0 + 1e-6


def test_est_clock_is_control_loop_time():
    """est['t'] = control-loop monotonic clock, NOT the GPS fix timestamp (CONTRACT §4.8)."""
    env = _env()
    env.reset()
    t0 = env.get_estimator_state()["t"]
    assert t0 == 0.0 and env.t_loop == 0.0
    a = torch.zeros(N, 3, device=DEV)
    prev = t0
    for i in range(1, 13):
        _, _, _, info = env.step(a)
        t = env.get_estimator_state()["t"]
        assert abs(t - prev - env.policy_dt) < 1e-6      # exactly one policy_dt
        assert abs(t - info["t"]) < 1e-9 and abs(t - env.t_loop) < 1e-9
        assert abs(t - i * env.policy_dt) < 1e-6
        prev = t
    # a reset does not rewind the clock (watchdog would else see a 40 s jump backwards)
    env.reset()
    assert env.get_estimator_state()["t"] == prev


def test_est_clock_never_trips_the_runtime_watchdog():
    """The env clock is what runtime/policy.py compares with now - est['t'] <= watchdog_s."""
    from runtime.policy import DEFAULT_SAFETY

    wd = float(DEFAULT_SAFETY["watchdog_s"])
    env = _env()
    env.reset()
    a = torch.zeros(N, 3, device=DEV)
    for _ in range(10):
        est_t = env.get_estimator_state()["t"]   # what the controller last received
        _, _, _, info = env.step(a)
        now = info["time"]                       # control-loop monotonic time
        assert 0.0 <= now - est_t <= wd


def test_est_dict_has_runtime_keys():
    env = _env()
    env.reset()
    env.step(torch.zeros(N, 3, device=DEV))
    d = env.get_estimator_state()
    for key in ("lat", "lon", "heading_deg", "yaw_rate_deg_s", "speed_mps",
                "course_deg", "gps_valid", "sensor_ok", "est", "t"):
        assert key in d
    assert d["est"].shape == (N, 6) and d["t"] == env.t_loop


def test_timeout_termination_and_auto_reset():
    env = _env(episode_s=0.3)                    # -> max_steps = 3
    assert env.max_steps == 3
    env.reset()
    a = torch.zeros(N, 3, device=DEV)
    done = None
    for _ in range(3):
        obs, _, done, info = env.step(a)
    assert env.max_steps == 3 and done.min() == 1.0
    assert info["timeout"].min() == 1.0
    # auto-reset: returned rows are the fresh episode, step counter back to 0
    assert env.ep_step.max() == 0
    assert torch.allclose(obs, env.obs_buf)
    assert env.state[:, :2].norm(dim=1).max() <= env.spawn_r * 2 ** 0.5 + 1e-3


def test_geofence_termination():
    env = _env()
    env.reset()
    env.state[:, 0] = env.geofence + 5.0
    obs, _, done, info = env.step(torch.zeros(N, 3, device=DEV))
    assert done.min() == 1.0 and info["geofence"].min() == 1.0
    assert torch.allclose(obs, env.obs_buf)


def test_episode_types_both_present():
    env = _env(station_keeping_fraction=0.5)
    env.reset()
    modes = env.mode
    assert set(modes.unique().tolist()) == {0.0, 1.0}
    sk = modes > 0.5
    assert torch.allclose(env.goal_xy[sk], env.hold_xy[sk])
    assert (env.n_wp[sk] == 0).all()
    wp = ~sk
    assert (env.n_wp[wp] >= 1).all() and (env.n_wp[wp] <= env.max_wp).all()


def test_waypoint_advances_on_arrival():
    env = _env(station_keeping_fraction=0.0)
    env.reset()
    env.state[:, :2] = env.goal_xy                     # sit on the first waypoint
    before = env.wp_idx.clone()
    env.step(torch.zeros(N, 3, device=DEV))
    moved = env.n_wp > 1
    assert (env.wp_idx[moved] == before[moved] + 1).all()
    assert (env.wp_idx[~moved] == before[~moved]).all()


def test_deadband_is_esp32_level():
    env = _env()
    env.reset()
    assert abs(env.thrusters.deadband0 - 0.03) < 1e-12          # ESP32, never the Pi 0.05
    db = env.thrusters.deadband
    assert db.min() >= 0.0
    assert (db - 0.03).abs().max() <= env.r_db + 1e-6


def test_domain_randomization_ranges():
    env = _env()
    env.reset()
    ratio = env.dynamics.M_env / env.dynamics.M
    assert (ratio[:, 0] - 1.0).abs().max() <= env.r_mass + 1e-5
    assert (ratio[:, 1] - 1.0).abs().max() <= env.r_mass + 1e-5
    assert (ratio[:, 2] - 1.0).abs().max() <= env.r_inertia + 1e-5
    # `set_params` couples mass_scale into the surge/sway linear drag (reports/R2.md §2)
    dr = env.dynamics.D_lin_env / env.dynamics.D_lin
    assert (dr - 1.0).abs().max() <= env.r_mass + env.r_drag + env.r_mass * env.r_drag + 1e-5
    assert (env.thrusters.lag_tau / env.lag0 - 1.0).abs().max() <= env.r_lag + 1e-5
    sf = env.thrusters.scale_fwd / env.sf0
    assert sf.max() <= 1.0 + env.r_gain + 1e-5
    assert sf.min() < 0.75                                     # one weakened motor exists


def test_curriculum_levels():
    env = _env()
    env.reset()
    env.set_curriculum(0)
    zero = env.disturbance.sample(env.t_loop, env.dt, env.state)
    assert torch.all(zero == 0.0)                              # level 0 = exact zero
    env.set_curriculum(len(env.scale_by_level) - 1)
    dist = env.disturbance.sample(env.t_loop, env.dt, env.state)
    assert dist.abs().max() > 0.0
    assert env.level_index == len(env.scale_by_level) - 1


def test_reward_weights_all_come_from_config():
    cfg = _cfg()
    for key in cfg["reward"]:
        if key.startswith("w_"):
            cfg["reward"][key] = 0.0
    env = AsvEnv(cfg=cfg, num_envs=N, device=DEV, seed=3)
    env.reset()
    _, rew, _, _ = env.step(torch.zeros(N, 3, device=DEV))
    assert torch.all(rew == 0.0)


def test_reward_favours_being_at_the_goal():
    env = _env()
    env.reset()
    a = torch.zeros(N, 3, device=DEV)
    env.state[:, :2] = env.goal_xy
    env.state[:, 3:5] = 0.0
    env._update_goals()
    d = env._goal_dist()
    near = env._reward(d, torch.zeros(N, device=DEV), a, a)
    env.state[:, :2] = env.goal_xy + 25.0
    d = env._goal_dist()
    far = env._reward(d, torch.zeros(N, device=DEV), a, a)
    assert (near > far).all()
