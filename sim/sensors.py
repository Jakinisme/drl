"""Sensor model: GPS, IMU gyro, magnetometer + complementary heading fusion (torch, batched).

CONTRACT.md §3.3 (config), §4.4 (interface), §2 (`est` layout), §1 (frames/signs),
§4.10 (`common/estimator.py`, the one velocity estimator shared with the Pi runtime).

Structure reproduces what the policy will meet on the water (INSPECTION §3, `fusion.py`):

* every source accumulates its own `dt` per env and takes a sample on its `rate_hz`
  epoch; between epochs the last sample is held (`gps.hold_last` repeats the fix);
* GPS latency is a rolling window of past *true* positions — the fix is what the
  receiver computed `latency_s` ago, not what is true now. The window holds the batch
  maximum lag and each env indexes its own delay (`set_noise(gps_latency_s=[...])`), so
  latency can be randomised per env. lat/lon are always derived back from the noisy ENU
  fix, so `gps_lat/lon`, `gps_sog`, `gps_cog_deg` stay mutually consistent even while the
  fix is stale;
* heading = mag-primary + gyro propagation, low gain `mag_weight/(mag_weight+gyro_weight)`
  (the measured 0.02 of `fusion.py:54`); GPS COG pulls the heading only when
  `sog >= gps_heading_min_speed_mps`;
* position is low-passed with `gps_alpha`; `(u, v)` come from `estimate_body_velocity`
  so a lost fix decays the estimate instead of snapping it to zero.

`est` is the only thing the policy may observe (CONTRACT §4.4); `true_state` is used
here only to synthesise the measurements.

Outputs are `[N]` float32 on `self.device` (`est` is `[N, 6]`); `gps_valid` and
`sensor_ok` are bool. `heading_deg`, `gps_cog_deg` are compass degrees, 0 = north,
clockwise (CONTRACT §1), wrapped to [-180, 180).
"""

import math
from pathlib import Path

import torch
import yaml

from common.estimator import estimate_body_velocity
from common.frames import enu_to_latlon

__all__ = ["SensorModel", "load_config"]


def load_config(path: str | Path | None = None) -> dict:
    """Load `configs/sensors.yaml` (default) or an explicit file."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "sensors.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _wrap180(a):
    """Wrap degrees to [-180, 180). Same formula as `common/estimator.py:_wrap180`."""
    return (a + 180.0) % 360.0 - 180.0


class SensorModel:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda",
                 dtype=torch.float32) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        n = self.num_envs

        g = cfg.get("gps", {}) or {}
        im = cfg.get("imu", {}) or {}
        mg = cfg.get("mag", {}) or {}
        es = cfg.get("estimator", {}) or {}

        # --- gps (CONTRACT §3.3). Every noise value is a TODO SYSID placeholder.
        self.periods = {
            "gps": 1.0 / float(g.get("rate_hz", 5.0)),
            "imu": 1.0 / float(im.get("rate_hz", 50.0)),
            "mag": 1.0 / float(mg.get("rate_hz", 15.0)),
        }
        self.gps_pos_sigma = float(g.get("pos_noise_m", 0.4))
        self.gps_vel_sigma = float(g.get("vel_noise_mps", 0.05))
        self.gps_cog_sigma = float(g.get("cog_noise_deg", 3.0))
        self.gps_latency = float(g.get("latency_s", 0.2))
        self.gps_hold = bool(g.get("hold_last", True))
        self.gps_stale = float(g.get("stale_after_s", 2.0))
        # null in the config -> sim fallback 0.0 (runtime uses the first real fix)
        self.origin_lat = 0.0 if g.get("origin_lat") is None else float(g["origin_lat"])
        self.origin_lon = 0.0 if g.get("origin_lon") is None else float(g["origin_lon"])

        # --- imu
        self.gyro_sigma = float(im.get("gyro_noise_dps", 0.3))
        self.gyro_bias_max = float(im.get("gyro_bias_dps", 0.2))
        # accel_noise_mps2 has no consumer in the 3-DOF planar sim (no accelerometer in
        # the output); parsed so the key stays wired for the interface only.
        self.accel_sigma = float(im.get("accel_noise_mps2", 0.05))
        self.imu_stale = float(im.get("stale_after_s", 0.5))

        # --- mag
        self.mag_sigma = float(mg.get("noise_deg", 2.0))
        self.mag_bias_max = float(mg.get("bias_deg", 5.0))
        self.mag_stale = float(mg.get("stale_after_s", 1.0))

        # --- estimator (measured Pi values, not TODO)
        self.estimator_type = str(es.get("type", "complementary"))
        self.mag_weight = float(es.get("mag_weight", 0.02))
        self.gyro_weight = float(es.get("gyro_weight", 0.98))
        self.gps_alpha = float(es.get("gps_alpha", 0.7))
        self.gps_min_speed = float(es.get("gps_heading_min_speed_mps", 0.3))
        w = self.mag_weight + self.gyro_weight
        # Neutral ratio of the two measured weights == fusion.py's gain (0.02), and the
        # gain goes to 1.0 (pure mag) if gyro_weight is 0.
        self.mag_gain = self.mag_weight / w if w > 0.0 else 1.0

        z = lambda *s: torch.zeros(*s, device=self.device, dtype=dtype)  # noqa: E731
        # per-env noise/latency (CONTRACT §3.4 `randomization`): nominal from yaml,
        # overwritten by `set_noise()` on reset. The float attributes above stay the
        # nominal values (the env reads them once, at construction).
        full = lambda v: torch.full((n,), float(v), device=self.device, dtype=dtype)  # noqa: E731
        self.pos_sigma_env = full(self.gps_pos_sigma)
        self.vel_sigma_env = full(self.gps_vel_sigma)
        self.cog_sigma_env = full(self.gps_cog_sigma)
        self.gyro_sigma_env = full(self.gyro_sigma)
        self.gyro_bias_max_env = full(self.gyro_bias_max)
        self.mag_sigma_env = full(self.mag_sigma)
        self.mag_bias_max_env = full(self.mag_bias_max)
        self.latency_env = full(self.gps_latency)
        self.acc = {k: z(n) for k in self.periods}      # time since last epoch [s]
        self.age = {k: z(n) for k in self.periods}      # time since last sample [s]
        self.gyro_bias = z(n)                          # deg/s, per episode
        self.mag_bias = z(n)                           # deg, per episode
        self.gyro = z(n)                               # deg/s, held between epochs
        self.mag = z(n)                                # deg, held between epochs
        self.heading = z(n)                            # deg, fused, [-180, 180)
        self.gps_enu = z(n, 2)                         # last noisy fix, ENU [m]
        self.sog = z(n)                                # m/s
        self.cog = z(n)                                # deg
        self.pos_hat = z(n, 2)                         # low-passed ENU position [m]
        self.u = z(n)
        self.v = z(n)
        self.pos_init = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.mag_init = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._buf = None                               # rolling window of true positions

        self.reset()

    # ------------------------------------------------------------------ public API
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """New episode: redraw gyro/mag biases for `env_ids` and drop all samples."""
        idx = slice(None) if env_ids is None else env_ids
        k = self.num_envs if env_ids is None else int(env_ids.numel())
        sgn = lambda *s: torch.empty(*s, device=self.device, dtype=self.dtype).uniform_(-1.0, 1.0)  # noqa: E731

        self.gyro_bias[idx] = sgn(k) * self.gyro_bias_max_env[idx]
        self.mag_bias[idx] = sgn(k) * self.mag_bias_max_env[idx]
        for key, per in self.periods.items():
            self.acc[key][idx] = per          # fire on the first step, no dead time
            self.age[key][idx] = 1e9          # nothing is fresh before the first sample
        self.gyro[idx] = 0.0
        self.mag[idx] = 0.0
        self.heading[idx] = 0.0
        self.gps_enu[idx] = 0.0
        self.sog[idx] = 0.0
        self.cog[idx] = 0.0
        self.pos_hat[idx] = 0.0
        self.u[idx] = 0.0
        self.v[idx] = 0.0
        self.pos_init[idx] = False
        self.mag_init[idx] = False
        if self._buf is not None:
            self._buf[idx] = 0.0

    def set_noise(self, env_ids=None, pos_noise_m=None, vel_noise_mps=None,
                  cog_noise_deg=None, gyro_noise_dps=None, gyro_bias_dps=None,
                  mag_noise_deg=None, mag_bias_deg=None,
                  gps_latency_s=None) -> None:
        """Per-env measurement noise/latency (CONTRACT §3.4 `randomization`).

        Each value is a scalar or a `[k]` tensor/array (one per selected env);
        `None` leaves that quantity unchanged. `gyro_bias_dps`/`mag_bias_deg` are
        the +/- uniform bound and are drawn at the **next** `reset()`; the rest
        take effect on the next `step()`. `gps_latency_s` is per env, resampled
        into the delay ring on the next `step()`.
        """
        idx = slice(None) if env_ids is None else env_ids
        k = self.num_envs if env_ids is None or isinstance(env_ids, slice) \
            else int(torch.as_tensor(env_ids).reshape(-1).numel())
        for attr, val in (("pos_sigma_env", pos_noise_m), ("vel_sigma_env", vel_noise_mps),
                          ("cog_sigma_env", cog_noise_deg), ("gyro_sigma_env", gyro_noise_dps),
                          ("gyro_bias_max_env", gyro_bias_dps),
                          ("mag_sigma_env", mag_noise_deg),
                          ("mag_bias_max_env", mag_bias_deg),
                          ("latency_env", gps_latency_s)):
            if val is None:
                continue
            got = torch.as_tensor(val, device=self.device, dtype=self.dtype).reshape(-1)
            if got.numel() not in (1, k):
                raise ValueError(f"{attr} needs 1 or {k} values, got {got.numel()}")
            getattr(self, attr)[idx] = got.expand(k) if got.numel() == 1 else got

    def step(self, true_state: torch.Tensor, dt: float) -> dict[str, torch.Tensor]:
        """One sim step -> measurement dict (CONTRACT §4.4 table)."""
        dt = float(dt)
        dev, dtp = self.device, self.dtype
        n = true_state.shape[0]
        pos = true_state[:, :2]

        # ---- GPS latency: one rolling window of past true positions (col 0 = now,
        # col j = j steps ago) with a PER-ENV delay index; the window holds the batch
        # maximum lag, so a slow env does not force a longer queue on the others.
        lag_i = torch.round(self.latency_env / dt).long().clamp(1, 1024)   # steps
        lag = int(lag_i.max())
        if self._buf is None or self._buf.shape != (n, lag + 1, 2):
            self._buf = pos.unsqueeze(1).repeat(1, lag + 1, 1)
        else:
            self._buf = torch.roll(self._buf, shifts=1, dims=1)
            self._buf[:, 0] = pos
        delayed = torch.gather(self._buf, 1, lag_i.view(n, 1, 1).expand(n, 1, 2)).squeeze(1)

        def epoch(key: str) -> torch.Tensor:
            a = self.acc[key] + dt
            fire = a >= self.periods[key]
            self.acc[key] = torch.where(fire, a - self.periods[key], a)
            self.age[key] = self.age[key] + dt
            return fire

        g_new = epoch("gps")
        i_new = epoch("imu")
        m_new = epoch("mag")

        # ---- gps fix (position via latency window; velocity of the true state now)
        if bool(g_new.any()):
            noise = torch.randn(n, 2, device=dev, dtype=dtp)
            fix = delayed + noise * self.pos_sigma_env.unsqueeze(1)
            psi = true_state[:, 2]
            u = true_state[:, 3]
            v = true_state[:, 4]
            ve = (u * torch.sin(psi) + v * torch.cos(psi)) + torch.randn(n, device=dev, dtype=dtp) * self.vel_sigma_env
            vn = (u * torch.cos(psi) - v * torch.sin(psi)) + torch.randn(n, device=dev, dtype=dtp) * self.vel_sigma_env
            sog = torch.sqrt(ve * ve + vn * vn)
            cog = _wrap180(torch.rad2deg(torch.atan2(ve, vn))
                           + torch.randn(n, device=dev, dtype=dtp) * self.cog_sigma_env)

            self.gps_enu = torch.where(g_new.unsqueeze(1), fix, self.gps_enu)
            self.sog = torch.where(g_new, sog, self.sog)
            self.cog = torch.where(g_new, cog, self.cog)
            self.age["gps"] = torch.where(g_new, torch.zeros_like(self.age["gps"]), self.age["gps"])

            # position low-pass (gps_alpha), snapped on the first fix after reset
            first = (g_new & ~self.pos_init).unsqueeze(1)
            lp = self.pos_hat + self.gps_alpha * (fix - self.pos_hat)
            self.pos_hat = torch.where(g_new.unsqueeze(1), torch.where(first, fix, lp), self.pos_hat)
            self.pos_init = self.pos_init | g_new

        # ---- imu gyro sample
        if bool(i_new.any()):
            meas = torch.rad2deg(true_state[:, 5]) + self.gyro_bias \
                + torch.randn(n, device=dev, dtype=dtp) * self.gyro_sigma_env
            self.gyro = torch.where(i_new, meas, self.gyro)
            self.age["imu"] = torch.where(i_new, torch.zeros_like(self.age["imu"]), self.age["imu"])

        # ---- magnetometer sample
        if bool(m_new.any()):
            meas = _wrap180(_wrap180(torch.rad2deg(true_state[:, 2])) + self.mag_bias
                            + torch.randn(n, device=dev, dtype=dtp) * self.mag_sigma_env)
            self.mag = torch.where(m_new, meas, self.mag)
            self.age["mag"] = torch.where(m_new, torch.zeros_like(self.age["mag"]), self.age["mag"])

        # ---- freshness (sample age vs stale_after_s)
        gps_valid = self.age["gps"] <= self.gps_stale
        if not self.gps_hold:                     # a real receiver without hold repeats nothing
            gps_valid = gps_valid & g_new
        imu_ok = self.age["imu"] <= self.imu_stale
        mag_ok = self.age["mag"] <= self.mag_stale
        sensor_ok = gps_valid & imu_ok & mag_ok

        # ---- heading fusion: gyro propagation every step, mag-primary, GPS course when fast
        h = _wrap180(self.heading + self.gyro * dt)
        ready = self.mag_init | m_new                       # mag is meaningful only after a sample
        gain = torch.where(self.mag_init, torch.full_like(h, self.mag_gain), torch.ones_like(h))
        h = _wrap180(h + ready.to(dtp) * gain * _wrap180(self.mag - h))   # snap on first sample
        self.mag_init = ready
        use_cog = g_new & gps_valid & (self.sog >= self.gps_min_speed)
        h = _wrap180(h + use_cog.to(dtp) * self.mag_gain * _wrap180(self.cog - h))
        self.heading = h

        # ---- body velocity from estimator (decays when the fix is stale)
        self.u, self.v = estimate_body_velocity(
            self.sog, self.cog, h, dt, self.u, self.v, gps_valid.to(dtp),
            min_speed_mps=self.gps_min_speed,
        )
        r = torch.deg2rad(self.gyro)

        # lat/lon always derived back from the noisy ENU fix -> consistent with sog/cog
        lat, lon = enu_to_latlon(self.gps_enu[:, 0].cpu().double().numpy(),
                                 self.gps_enu[:, 1].cpu().double().numpy(),
                                 self.origin_lat, self.origin_lon)

        est = torch.stack([self.pos_hat[:, 0], self.pos_hat[:, 1],
                           torch.deg2rad(h), self.u, self.v, r], dim=1)

        return {
            "gps_lat": torch.as_tensor(lat, device=dev, dtype=dtp),
            "gps_lon": torch.as_tensor(lon, device=dev, dtype=dtp),
            "gps_sog": self.sog,
            "gps_cog_deg": self.cog,
            "gps_valid": gps_valid,
            "heading_deg": h,
            "yaw_rate_deg_s": self.gyro,
            "u": self.u,
            "v": self.v,
            "r": r,
            "est": est,
            "sensor_ok": sensor_ok,
        }
