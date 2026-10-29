"""Environmental disturbances: wind, current, waves (torch, batched over N envs).

CONTRACT.md §3.2 (config), §4.3 (interface). Returns a *body-frame* wrench
`[surge N, sway N, yaw N·m]`, added by `sim/dynamics.py` as `tau_env`.

Physics, in the order the curriculum scales them (`level` = curriculum level value,
NOT the level index):

* wind     `F = level^2 * 0.5*rho_air*CdA * |V_aw| * V_aw` in the world frame, with
           `V_aw = wind - v_hull` (relative wind). Quadratic in relative speed; the
           `level^2` factor scales the wind *speed* by `level`, so the force goes as
           `level^2`.
* current  `F = level * drag_linear * (c_body - v_body)`. The hull is dragged along by
           the water; at rest with current `c` the push equals `drag_linear*c`.
           Approximation: the hull's own drag in `sim/dynamics.py` already acts on the
           absolute body velocity, so current is modelled as a relative-flow drag term
           rather than by shifting the whole drag model. TODO: fold into the Fossen
           relative-velocity form once A1's drag coefficients are identified.
* waves    sum of `n_components` sinusoids, random phase and direction per episode,
           linear in `level`: horizontal force amplitude `amp_n[i]` and yaw moment
           `yaw_amp_nm[i]`, at `freq_hz[i]`.

`level = 0` gives exact zeros (every term is multiplied by `level` or `level^2`).
Stateless in `t` except for the wave phases and wind direction drawn at `reset()`.

Units: SI (m, s, rad internally; `direction_deg` in the config is compass degrees,
0 = north, clockwise = CONTRACT §1).
"""

import math
from pathlib import Path

import torch
import yaml

__all__ = ["Disturbance", "load_config"]


def load_config(path: str | Path | None = None) -> dict:
    """Load `configs/disturbance.yaml` (default) or an explicit file."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "disturbance.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Disturbance:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda",
                 dtype=torch.float32) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        n = self.num_envs

        w = cfg.get("wind", {}) or {}
        self.wind_on = bool(w.get("enabled", True))
        self.wind_coef = 0.5 * float(w.get("rho_air_kgm3", 1.225)) * float(w.get("cd_area_m2", 0.35))
        self.wind_max = float(w.get("max_speed_mps", 8.0))

        c = cfg.get("current", {}) or {}
        self.current_on = bool(c.get("enabled", True))
        self.current_speed = float(c.get("speed_mps", 0.25))
        self.current_drag = float(c.get("drag_linear", 6.0))
        cd = math.radians(float(c.get("direction_deg", 0.0)))  # compass CW from north
        self.current_enu = torch.tensor(
            [self.current_speed * math.sin(cd), self.current_speed * math.cos(cd)],
            device=self.device, dtype=dtype,
        )

        wv = cfg.get("waves", {}) or {}
        self.waves_on = bool(wv.get("enabled", True))
        self.n_comp = int(wv.get("n_components", 4))
        freq = [float(x) for x in wv.get("freq_hz", [0.15, 0.25, 0.4, 0.6])][: self.n_comp]
        amp = [float(x) for x in wv.get("amp_n", [2.0, 1.5, 1.0, 0.6])][: self.n_comp]
        yaw = [float(x) for x in wv.get("yaw_amp_nm", [0.3, 0.2, 0.1, 0.05])][: self.n_comp]
        if not (len(freq) == len(amp) == len(yaw) == self.n_comp):
            raise ValueError("waves: n_components must match freq_hz/amp_n/yaw_amp_nm lengths")
        self.wave_freq = torch.tensor(freq, device=self.device, dtype=dtype)
        self.wave_amp = torch.tensor(amp, device=self.device, dtype=dtype)
        self.wave_yaw_amp = torch.tensor(yaw, device=self.device, dtype=dtype)

        self.level = torch.zeros(n, device=self.device, dtype=dtype)

        # Per-episode random quantities (drawn in reset()).
        self.wind_speed = torch.zeros(n, device=self.device, dtype=dtype)
        self.wind_dir = torch.zeros(n, device=self.device, dtype=dtype)      # rad, compass
        self.wave_dir = torch.zeros(n, device=self.device, dtype=dtype)      # rad, compass
        self.wave_phase = torch.zeros(n, self.n_comp, device=self.device, dtype=dtype)
        self.wave_phase_yaw = torch.zeros(n, self.n_comp, device=self.device, dtype=dtype)

        self._gen = torch.Generator(device=self.device)
        # R8_D: additive seed hook.  Absent/0 -> identical stream to before (default).
        # evaluate.py sets `seed_offset = seed*1000 + 17` so --seed also moves wind/waves;
        # train.py never sets it, so training is unchanged.
        self._gen.manual_seed(int(cfg.get("seed_offset", 0)))

        self.reset()

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, device=self.device, dtype=self.dtype, generator=self._gen)

    def _set_level(self, idx, level) -> None:
        lv = torch.as_tensor(level, device=self.device, dtype=self.dtype)
        if lv.numel() == 1:
            self.level[idx] = lv.reshape(())
        elif lv.numel() == self.num_envs:
            self.level[idx] = lv[idx]
        else:
            raise ValueError(f"level must be scalar or [{self.num_envs}], got {tuple(lv.shape)}")

    # ------------------------------------------------------------------ public API
    def reset(self, env_ids: torch.Tensor | None = None, level=0.0) -> None:
        """Resample wind/wave randomness for `env_ids` and set their curriculum level."""
        idx = slice(None) if env_ids is None else env_ids
        k = self.num_envs if env_ids is None else int(env_ids.numel())
        two_pi = 2.0 * math.pi

        self.wind_speed[idx] = self._rand(k) * self.wind_max
        self.wind_dir[idx] = self._rand(k) * two_pi
        self.wave_dir[idx] = self._rand(k) * two_pi
        self.wave_phase[idx] = self._rand(k, self.n_comp) * two_pi
        self.wave_phase_yaw[idx] = self._rand(k, self.n_comp) * two_pi
        self._set_level(idx, level)

    def set_level(self, level) -> None:
        self._set_level(slice(None), level)

    def sample(self, t: float, dt: float, state: torch.Tensor) -> torch.Tensor:
        """Disturbance wrench for the given *true* state.

        Args:
            t: absolute episode time [s] (drives the wave phase).
            dt: sim step [s] — unused: the model is an algebraic function of `t` and
                `state`, with no internal integration, so it cannot drift with dt.
            state: `[N, 6]` = (x, y, psi, u, v, r).
        Returns:
            `[N, 3]` body wrench `[surge_force, sway_force, yaw_moment]`.
        """
        N = state.shape[0]
        psi = state[:, 2]
        u = state[:, 3]
        v = state[:, 4]
        sp = torch.sin(psi)
        cp = torch.cos(psi)
        # body -> world (CONTRACT §1)
        ve = u * sp + v * cp
        vn = u * cp - v * sp

        fe = torch.zeros(N, device=self.device, dtype=self.dtype)
        fn = torch.zeros(N, device=self.device, dtype=self.dtype)
        yaw = torch.zeros(N, device=self.device, dtype=self.dtype)

        lv = self.level

        if self.wind_on:
            # Quadratic in the *relative* wind speed; level scales the force as level^2
            # (equivalent to scaling wind speed by level, but keeps level 0 exactly 0
            # even while the hull is moving).
            we = self.wind_speed * torch.sin(self.wind_dir)
            wn = self.wind_speed * torch.cos(self.wind_dir)
            re = we - ve
            rn = wn - vn
            k = (self.wind_coef * lv * lv) * torch.sqrt(re * re + rn * rn)
            fe = fe + k * re
            fn = fn + k * rn

        if self.current_on:
            ce = self.current_enu[0] * lv
            cn = self.current_enu[1] * lv
            # relative water velocity in body frame; drag pulls the hull with the water
            cbe = ce * sp + cn * cp
            cbn = ce * cp - cn * sp
            d = self.current_drag * lv
            cur_bx = d * (cbe - u)
            cur_by = d * (cbn - v)
        else:
            cur_bx = torch.zeros(N, device=self.device, dtype=self.dtype)
            cur_by = torch.zeros(N, device=self.device, dtype=self.dtype)

        if self.waves_on:
            wfe = torch.zeros(N, device=self.device, dtype=self.dtype)
            wfn = torch.zeros(N, device=self.device, dtype=self.dtype)
            for i in range(self.n_comp):
                ph = 2.0 * math.pi * float(self.wave_freq[i].item()) * float(t) + self.wave_phase[:, i]
                a = float(self.wave_amp[i].item()) * torch.sin(ph)
                wfe = wfe + a * torch.sin(self.wave_dir)
                wfn = wfn + a * torch.cos(self.wave_dir)
                yaw = yaw + float(self.wave_yaw_amp[i].item()) * torch.sin(ph + self.wave_phase_yaw[:, i])
            fe = fe + wfe * lv        # waves scale linearly with level
            fn = fn + wfn * lv
            yaw = yaw * lv

        # world -> body
        bx = fe * sp + fn * cp + cur_bx
        by = fe * cp - fn * sp + cur_by
        return torch.stack([bx, by, yaw], dim=1)
