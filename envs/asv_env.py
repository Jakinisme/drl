"""Vectorised ASV environment (torch, GPU) — CONTRACT.md §3.4, §4.5, §5, §6.

One policy step = `substeps` sim steps (`dt` 0.02 s, 5 per 0.1 s policy step, TASK.md).
The observation is the 15-value sample of §5, normalized per step and stacked K=10
oldest-first (150 values). `raw_sample_torch`/`_sample` repeat the formulas of
`runtime/obs_np.py` line for line; `tests/test_obs_parity.py` enforces
`max |obs_torch - obs_numpy| <= 1e-5`.

Clock (binding decision, see drl/reports/B1.md): `est["t"]` is `self.t_loop`, the
CONTROL-LOOP monotonic clock in seconds — advanced by exactly `policy_dt` per policy
step, never reset per episode, never derived from the GPS fix age or a GPS timestamp.
`runtime/policy.py` compares `est["t"]` against `time.monotonic()` under a 0.5 s
watchdog, so a per-episode clock (or a fix timestamp, which lags by `latency_s` and
only updates at 5 Hz) would trip it spuriously. `get_estimator_state()` exposes the
runtime-shaped dict with that same clock.

Episodes: waypoint (`mode` 0, goal = current waypoint of a random 1..MAXWP list, wp1/wp2
= the two following waypoints, 0 when absent) and station keeping (`mode` 1, goal =
random hold point, heading held at the initial hull heading). Terminate on timeout
(`max_steps`) or on distance from the ENU origin > `geofence_m`. **Auto-reset:** `step()`
resets finished envs before returning, so the returned obs row is the post-reset sample;
`evaluate.py` must read the terminal state from `info`, not from `obs`.

Only `est` enters the observation; `state` is used for reward, termination and
`get_privileged_state()` (§7.4).

Randomisation (CONTRACT §3.4) is per env on **every** reset (partial resets included):
mass/inertia/drag go through `dynamics.set_params`, thruster gain/lag/deadband/asymmetry
through `thrusters.set_gains`, sensor noise/bias/latency through `sensors.set_noise`.
Zeroing the whole `randomization:` block in the config therefore switches every draw off.
TODO(CONTRACT T8): geofence radius, episode length, spawn/waypoint boxes are un-tuned.
"""

import math
from pathlib import Path

import torch
import yaml

from sim.disturbance import Disturbance, load_config as load_disturbance_cfg
from sim.dynamics import Dynamics, load_config as load_vehicle_cfg
from sim.sensors import SensorModel, load_config as load_sensors_cfg
from sim.thrusters import ThrusterModel

__all__ = ["AsvEnv", "RAW_DIM", "load_config", "raw_sample_torch"]

RAW_DIM = 15


def load_config(path=None) -> dict:
    """Load `configs/env.yaml` (default) or an explicit file."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "env.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    """Wrap to [-pi, pi). Same formula as `common.frames.wrap_pi` (numpy)."""
    return torch.remainder(a + math.pi, 2.0 * math.pi) - math.pi


def raw_sample_torch(est, goal, prev_action=None, wp1=None, wp2=None) -> torch.Tensor:
    """Torch twin of `runtime/obs_np.py::raw_sample` — same 15 values, same order.

    Args:
        est: [N, 6] `(x, y, psi, u, v, r)`, estimator output only.
        goal: [N, 4] `(x_goal_e, y_goal_n, psi_goal_rad, mode)`.
        prev_action: [N, 3] last applied action, zeros at reset.
        wp1, wp2: [N, 2] ENU waypoints after the current goal, or None -> zeros.
    Returns:
        [N, 15], dtype of `est`.
    """
    n = est.shape[0]
    out = torch.empty(n, RAW_DIM, device=est.device, dtype=est.dtype)
    psi = est[:, 2]
    s, c = torch.sin(psi), torch.cos(psi)

    de = goal[:, 0] - est[:, 0]
    dn = goal[:, 1] - est[:, 1]
    out[:, 0] = de * s + dn * c
    out[:, 1] = de * c - dn * s

    perr = _wrap_pi(goal[:, 2] - psi)
    out[:, 2] = torch.sin(perr)
    out[:, 3] = torch.cos(perr)
    out[:, 4:7] = est[:, 3:6]
    out[:, 7:10] = 0.0 if prev_action is None else prev_action

    for col, wp in ((10, wp1), (12, wp2)):
        if wp is None:
            out[:, col] = 0.0
            out[:, col + 1] = 0.0
        else:
            we = wp[:, 0] - est[:, 0]
            wn = wp[:, 1] - est[:, 1]
            out[:, col] = we * s + wn * c
            out[:, col + 1] = we * c - wn * s

    out[:, 14] = goal[:, 3]
    return out


class AsvEnv:
    """Batched ASV env. `reset()`/`step()` per CONTRACT §4.5."""

    def __init__(self, cfg: dict | None = None, num_envs: int | None = None,
                 device: str = "cuda", dtype=torch.float32, seed: int | None = None,
                 vehicle_cfg: dict | None = None,
                 disturbance_cfg: dict | None = None,
                 sensors_cfg: dict | None = None) -> None:
        self.cfg = cfg = cfg or load_config()
        self.vehicle = vehicle_cfg or load_vehicle_cfg()
        self.num_envs = int(num_envs if num_envs is not None else cfg["num_envs"])
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"                       # keeps tests runnable on a CPU-only host
        self.device = torch.device(device)
        self.dtype = dtype = torch.as_tensor(0.0, dtype=dtype).dtype
        dev = str(self.device)

        self.dt = float(cfg["dt_s"])
        self.policy_hz = float(cfg["policy_hz"])
        self.substeps = max(1, int(round((1.0 / self.policy_hz) / self.dt)))
        self.policy_dt = self.substeps * self.dt
        self.max_steps = int(cfg["max_steps"] or round(float(cfg["episode_s"]) * self.policy_hz))
        obs_cfg = cfg["obs"]
        self.k = int(obs_cfg["k_history"])
        self.obs_dim = RAW_DIM * self.k
        self.act_dim = 3
        self.obs_clip = float(obs_cfg["clip_norm"])
        self.obs_offset = torch.as_tensor(obs_cfg["offset"], device=self.device, dtype=dtype)
        self.obs_scale = torch.as_tensor(obs_cfg["scale"], device=self.device, dtype=dtype)
        self.clip = float(cfg["action"]["clip"])
        self.rate_limit = float(cfg["action"]["rate_limit_per_s"])

        ep = cfg["episode"]
        self.arrival = float(ep["arrival_radius_m"])
        self.slowdown = float(ep["slowdown_radius_m"])
        self.geofence = float(ep["geofence_m"])
        self.sk_frac = float(ep["station_keeping_fraction"])
        self.wp_lo, self.wp_hi = (int(x) for x in ep["n_waypoints"])
        self.max_wp = max(1, self.wp_hi)
        self.spawn_r = float(ep.get("spawn_radius_m", 30.0))
        self.wp_r = float(ep.get("waypoint_radius_m", 35.0))

        self.w = {key: float(val) for key, val in cfg["reward"].items()}
        self.max_surge = float(self.vehicle["max_surge_mps"])

        rd = cfg["randomization"]
        self.r_mass = float(rd["mass"])
        self.r_inertia = float(rd["inertia"])
        self.r_drag = float(rd["drag"])
        self.r_gain = float(rd["thruster_gain"])
        self.r_lag = float(rd["thruster_lag"])
        self.r_db = float(rd["thruster_deadband"])
        self.r_asym = float(rd["fwd_rev_asymmetry"])
        self.r_weak_prob = float(rd["weak_thruster_prob"])
        self.weak_frac = float(rd.get("weak_factor", 0.6))
        self.r_noise = float(rd["sensor_noise"])
        self.r_latency = float(rd["sensor_latency_s"])
        # one source of truth for the magnetometer bias bound: sensors.yaml `mag.bias_deg`
        # (CONTRACT §3.3); env.yaml `sensor_bias_deg` is no longer read (reports/R3.md §3)
        # all ranges 0 -> "randomisation off": skip every draw and keep the nominal
        # yaml parameters (see reports/R2.md §3)
        self.randomize = any(v != 0.0 for v in (
            self.r_mass, self.r_inertia, self.r_drag, self.r_gain, self.r_lag,
            self.r_db, self.r_asym, self.r_weak_prob, self.r_noise, self.r_latency))

        self.dynamics = Dynamics(self.vehicle, self.num_envs, device=dev, dtype=dtype, dt=self.dt)
        self.thrusters = ThrusterModel(self.vehicle, self.num_envs, device=dev, dtype=dtype)
        self.disturbance = Disturbance(disturbance_cfg or load_disturbance_cfg(),
                                      self.num_envs, device=dev, dtype=dtype)
        sen_cfg = sensors_cfg or load_sensors_cfg()
        self.sensors = SensorModel(sen_cfg, self.num_envs, device=dev, dtype=dtype,
                                   # largest latency a draw can reach -> the GPS ring is
                                   # sized once and never reallocated (reports/R3.md §2)
                                   max_latency_s=float(sen_cfg["gps"]["latency_s"]) + self.r_latency)

        # nominal copies the randomisation is drawn around
        self.sf0 = self.thrusters.nominal["scale_fwd"].clone()
        self.sr0 = self.thrusters.nominal["scale_rev"].clone()
        self.lag0 = float(self.thrusters.lag_tau0)
        self.db0 = float(self.thrusters.deadband0)
        sen = self.sensors
        self._sensor_base = {key: float(getattr(sen, key)) for key in
                             ("gps_pos_sigma", "gps_vel_sigma", "gps_cog_sigma",
                              "gyro_sigma", "mag_sigma")}
        self._gyro_bias0 = float(sen.gyro_bias_max)
        self._mag_bias0 = float(sen.mag_bias_max)    # sensors.yaml mag.bias_deg, the one source
        self._latency0 = float(sen.gps_latency)

        cur = self.disturbance.cfg.get("curriculum", {}) or {}
        self.scale_by_level = [float(v) for v in cur.get("scale_by_level", [0.0])]
        self.n_levels = len(self.scale_by_level)
        self.start_level = int(cfg["curriculum"]["start_level"])
        self.level_up_steps = max(1, int(round(
            float(cfg["curriculum"]["level_up_fraction"]) * self.max_steps)))
        self._level_override = None
        self.level_index = min(self.start_level, self.n_levels - 1)

        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed if seed is not None else cfg.get("seed", 0)))

        z = lambda *sh: torch.zeros(*sh, device=self.device, dtype=dtype)   # noqa: E731
        n = self.num_envs
        self.state = z(n, 6)
        self.mode = z(n)
        self.prev_action = z(n, 3)
        self.goal_xy = z(n, 2)
        self.hold_xy = z(n, 2)
        self.hold_psi = z(n)
        self.psi_goal = z(n)
        self.leg_start = z(n, 2)
        self.wps = z(n, self.max_wp, 2)
        lz = lambda: torch.zeros(n, dtype=torch.long, device=self.device)   # noqa: E731
        self.wp_idx, self.n_wp, self.ep_step = lz(), lz(), lz()
        self.obs_buf = z(n, self.obs_dim)
        self.meas = None
        self.t_loop = 0.0          # control-loop monotonic clock [s], never reset
        self.total_steps = 0
        self.closed = False
        self.reset()

    # ------------------------------------------------------------------ helpers
    def _unif(self, *shape, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
        r = torch.rand(*shape, device=self.device, dtype=self.dtype, generator=self._gen)
        return lo + (hi - lo) * r

    def _level_value(self) -> float:
        return self.scale_by_level[min(self.level_index, self.n_levels - 1)]

    def _sync_level(self, force: bool = False) -> None:
        if self._level_override is not None:
            i = int(round(self._level_override))
        else:
            i = self.start_level + self.total_steps // self.level_up_steps
        i = max(0, min(i, self.n_levels - 1))
        if force or i != self.level_index:
            self.level_index = i
            self.disturbance.set_level(self._level_value())

    def _all(self):
        return torch.arange(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Reset `env_ids` (default all). Returns their rows, oldest history first."""
        full = env_ids is None
        idx = self._all() if full else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long).reshape(-1)
        k = int(idx.numel())
        dev, dtp = self.device, self.dtype

        s = torch.zeros(k, 6, device=dev, dtype=dtp)
        s[:, 0] = self._unif(k) * self.spawn_r
        s[:, 1] = self._unif(k) * self.spawn_r
        s[:, 2] = self._unif(k) * math.pi
        self.state[idx] = s

        self.mode[idx] = (self._unif(k) < self.sk_frac).to(dtp)
        hold = torch.stack([self._unif(k) * self.wp_r, self._unif(k) * self.wp_r], dim=1)
        wps = torch.stack([self._unif(k, self.max_wp) * self.wp_r,
                           self._unif(k, self.max_wp) * self.wp_r], dim=2)  # [k, max_wp, 2]
        nw = torch.randint(self.wp_lo, self.wp_hi + 1, (k,), device=dev, generator=self._gen)
        self.wps[idx] = wps
        wp_mode = self.mode[idx] < 0.5
        self.n_wp[idx] = torch.where(wp_mode, nw, torch.zeros_like(nw))
        self.wp_idx[idx] = 0
        self.hold_xy[idx] = hold
        self.hold_psi[idx] = s[:, 2]
        self.goal_xy[idx] = torch.where(wp_mode.unsqueeze(1), wps[:, 0], hold)
        self.leg_start[idx] = s[:, :2]
        gx = self.goal_xy[idx, 0] - s[:, 0]
        gy = self.goal_xy[idx, 1] - s[:, 1]
        self.psi_goal[idx] = torch.where(wp_mode, torch.atan2(gx, gy), s[:, 2])

        self.dynamics.reset(idx)
        self.thrusters.reset(idx)
        self.disturbance.reset(idx, level=self._level_value())
        # randomise BEFORE sensors.reset: the gyro/mag bias draw of this episode must use
        # the per-env bounds set here (CONTRACT §3.4)
        if self.randomize:
            self._randomize(idx, k)
            self._sensor_randomization(idx, k)
        self.sensors.reset(idx, pos=self.state[idx, :2], heading=self.state[idx, 2])
        self.ep_step[idx] = 0
        self.prev_action[idx] = 0.0

        # prime the estimator so the first obs of the episode carries a fix, not zeros
        # (advances the sensor clock one extra dt for envs NOT reset here — harmless)
        self.meas = self.sensors.step(self.state, self.dt)
        sample = self._sample()
        self.obs_buf[idx] = sample[idx].repeat(1, self.k)
        return self.obs_buf.clone() if full else self.obs_buf[idx].clone()

    def _randomize(self, idx, k: int) -> None:
        """Per-env domain randomisation on reset (CONTRACT §3.4 randomization block)."""
        ms = 1.0 + self._unif(k) * self.r_mass
        is_ = 1.0 + self._unif(k) * self.r_inertia
        ds = 1.0 + self._unif(k) * self.r_drag
        # independent factors: mass_scale -> M(x,y), drag_scale -> D_lin/D_quad only
        # (reports/R3.md §4; the R2.md §2 coupling is gone)
        self.dynamics.set_params(idx, mass_scale=ms, inertia_scale=is_, drag_scale=ds)

        sf = self.sf0.unsqueeze(0) * (1.0 + self._unif(k, 4) * self.r_gain)
        sr = self.sr0.unsqueeze(0) * (1.0 + self._unif(k, 4) * self.r_asym)
        weak = self._unif(k) < (2.0 * self.r_weak_prob - 1.0)   # U(-1,1) < p  <=>  U(0,1) < p
        if bool(weak.any()):
            wm = torch.randint(0, 4, (k,), device=self.device, generator=self._gen)
            mask = torch.zeros(k, 4, dtype=torch.bool, device=self.device)
            mask[torch.arange(k, device=self.device), wm] = weak
            sf = torch.where(mask, sf * self.weak_frac, sf)
            sr = torch.where(mask, sr * self.weak_frac, sr)
        lag = self.lag0 * (1.0 + self._unif(k) * self.r_lag)
        db = (self.db0 + self._unif(k) * self.r_db).clamp(min=0.0)
        self.thrusters.set_gains(idx, scale_fwd=sf, scale_rev=sr, lag_tau=lag, deadband=db)

    def _sensor_randomization(self, idx, k: int) -> None:
        """Per-env sensor noise/bias/latency draw for `idx` (CONTRACT §3.4)."""
        b = self._sensor_base
        f = 1.0 + self._unif(k) * self.r_noise
        self.sensors.set_noise(
            idx,
            pos_noise_m=b["gps_pos_sigma"] * f,
            vel_noise_mps=b["gps_vel_sigma"] * f,
            cog_noise_deg=b["gps_cog_sigma"] * f,
            gyro_noise_dps=b["gyro_sigma"] * f,
            mag_noise_deg=b["mag_sigma"] * f,
            gyro_bias_dps=self._gyro_bias0 * (1.0 + self._unif(k) * self.r_noise),
            mag_bias_deg=self._mag_bias0 * (1.0 + self._unif(k) * self.r_noise),
            gps_latency_s=(self._latency0 + self._unif(k) * self.r_latency).clamp(min=0.0),
        )

    # ------------------------------------------------------------------ obs
    def _next_wps(self):
        n = self.num_envs
        zeros = torch.zeros(n, 2, device=self.device, dtype=self.dtype)
        a = self._all()
        i1 = (self.wp_idx + 1).clamp(max=self.max_wp - 1)
        i2 = (self.wp_idx + 2).clamp(max=self.max_wp - 1)
        wp_mode = self.mode < 0.5
        m1 = (wp_mode & (self.wp_idx + 1 < self.n_wp)).unsqueeze(1)
        m2 = (wp_mode & (self.wp_idx + 2 < self.n_wp)).unsqueeze(1)
        return (torch.where(m1, self.wps[a, i1], zeros),
                torch.where(m2, self.wps[a, i2], zeros))

    def _sample(self, action: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized [N, 15] sample of the current `est`/goal/action (CONTRACT §5).

        `action` is the action applied during the step the sample describes (the
        "previous action" the runtime puts in `obs`, i.e. a_{t-1} at policy step t).
        """
        goal = torch.stack([self.goal_xy[:, 0], self.goal_xy[:, 1], self.psi_goal,
                            self.mode], dim=1)
        wp1, wp2 = self._next_wps()
        act = self.prev_action if action is None else action
        raw = raw_sample_torch(self.meas["est"], goal, act, wp1, wp2)
        return ((raw - self.obs_offset) / self.obs_scale).clamp(-self.obs_clip, self.obs_clip)

    # ------------------------------------------------------------------ physics
    def _goal_dist(self) -> torch.Tensor:
        return (self.goal_xy - self.state[:, :2]).norm(dim=1)

    def _xtrack(self) -> torch.Tensor:
        """Distance to the current leg; 0 in station keeping (no leg)."""
        leg = self.goal_xy - self.leg_start
        per = self.state[:, :2] - self.leg_start
        xt = (per[:, 0] * leg[:, 1] - per[:, 1] * leg[:, 0]).abs() / leg.norm(dim=1).clamp(min=1e-6)
        return torch.where(self.mode > 0.5, torch.zeros_like(xt), xt)

    def _update_goals(self) -> None:
        pos = self.state[:, :2]
        d = (self.goal_xy - pos).norm(dim=1)
        adv = (self.mode < 0.5) & (self.wp_idx + 1 < self.n_wp) & (d < self.arrival)
        if bool(adv.any()):
            ids = adv.nonzero(as_tuple=True)[0]
            self.wp_idx[ids] += 1
            self.leg_start[ids] = pos[ids]
            self.goal_xy[ids] = self.wps[ids, self.wp_idx[ids]]
        dx = self.goal_xy[:, 0] - pos[:, 0]
        dy = self.goal_xy[:, 1] - pos[:, 1]
        bearing = torch.atan2(dx, dy)                      # compass bearing to the goal
        bearing = torch.where(torch.hypot(dx, dy) < 0.5, self.state[:, 2], bearing)
        self.psi_goal = torch.where(self.mode < 0.5, bearing, self.hold_psi)

    def _reward(self, d, dd, a, a_prev) -> torch.Tensor:
        w = self.w
        speed = self.state[:, 3:5].norm(dim=1)
        perr = _wrap_pi(self.psi_goal - self.state[:, 2]).abs()
        near = (d < self.slowdown).to(self.dtype)
        r = (w["w_position"] * torch.exp(-d / w["pos_sigma_m"])
             + w["w_heading"] * torch.exp(-perr / w["head_sigma_rad"])
             + w["w_progress"] * (dd / (self.max_surge * self.policy_dt)).clamp(-1.0, 1.0)
             - w["w_cross_track"] * (self._xtrack() / w["pos_sigma_m"]).pow(2).clamp(max=1.0)
             - w["w_near_goal_speed"] * (speed / self.max_surge).pow(2) * near
             - w["w_action_rate"] * (a - a_prev).pow(2).mean(dim=1)
             - w["w_energy"] * a.pow(2).mean(dim=1))
        return r.to(self.dtype)

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor):
        """One policy step. Returns (obs [N,150], reward [N], done [N] float 0/1, info)."""
        a = torch.as_tensor(action, device=self.device, dtype=self.dtype).reshape(self.num_envs, 3)
        a = a.clamp(-self.clip, self.clip)
        a_prev = self.prev_action.clone()
        lim = self.rate_limit * self.policy_dt
        a = (a_prev + (a - a_prev).clamp(-lim, lim)).clamp(-self.clip, self.clip)

        goal0 = self.goal_xy
        d_prev = (goal0 - self.state[:, :2]).norm(dim=1)
        for _ in range(self.substeps):
            tau = self.thrusters.step(a, self.dt) \
                + self.disturbance.sample(self.t_loop, self.dt, self.state)
            self.state = self.dynamics.step(self.state, tau, self.dt)
            self.meas = self.sensors.step(self.state, self.dt)
        self.t_loop += self.policy_dt
        self.total_steps += 1
        self.ep_step += 1
        d_prog = (goal0 - self.state[:, :2]).norm(dim=1)

        self._sync_level()
        self._update_goals()
        d = self._goal_dist()
        reward = self._reward(d, d_prev - d_prog, a, a_prev)

        timeout = self.ep_step >= self.max_steps
        outside = self.state[:, :2].norm(dim=1) > self.geofence
        done = timeout | outside

        sample = self._sample(a_prev)      # obs carries a_{t-1}, as the runtime does
        self.prev_action = a.clone()       # last APPLIED action
        self.obs_buf = torch.cat([self.obs_buf[:, RAW_DIM:], sample], dim=1)
        obs = self.obs_buf.clone()
        info = {
            "xy": self.state[:, :2].clone(),
            "goal_xy": self.goal_xy.clone(),
            "psi": self.state[:, 2].clone(),
            "mode": self.mode.clone(),
            "time": float(self.t_loop),
            "t": float(self.t_loop),            # control-loop clock, same as est["t"]
            "ep_step": self.ep_step.clone(),
            "dist": d.clone(),
            "arrived": (d < self.arrival).to(self.dtype),
            "timeout": timeout.to(self.dtype),
            "geofence": outside.to(self.dtype),
        }
        if bool(done.any()):
            ids = done.nonzero(as_tuple=True)[0]
            obs[ids] = self.reset(ids)          # post-reset obs rows, auto-reset §4.5
        return obs, reward, done.to(self.dtype), info

    # ------------------------------------------------------------------ access
    def get_privileged_state(self) -> torch.Tensor:
        """Ground truth [N, 6] — evaluation/metrics only, never an observation."""
        return self.state

    def get_estimator_state(self) -> dict:
        """Runtime-shaped dict (`runtime/policy.py`, CONTRACT §4.8).

        Array quantities are tensors [N]; `t` is the control-loop monotonic clock.
        `imu_valid`/`mag_valid` are folded into `sensor_ok` by `sim/sensors.py`.
        """
        m = self.meas
        return {
            "lat": m["gps_lat"], "lon": m["gps_lon"],
            "heading_deg": m["heading_deg"], "yaw_rate_deg_s": m["yaw_rate_deg_s"],
            "speed_mps": m["gps_sog"], "course_deg": m["gps_cog_deg"],
            "u": m["u"], "v": m["v"], "r": m["r"], "est": m["est"],
            "gps_valid": m["gps_valid"], "sensor_ok": m["sensor_ok"],
            "t": float(self.t_loop),
        }

    def set_curriculum(self, level: float) -> None:
        """Pin the level index (into `disturbance.curriculum.scale_by_level`)."""
        self._level_override = float(level)
        self._sync_level(force=True)

    def close(self) -> None:
        self.closed = True
