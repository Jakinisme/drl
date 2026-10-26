"""Per-motor thruster model: mixing, deadband, lag, asymmetric thrust, body wrench.

CONTRACT.md §3.1 (config), §4.2 (interface + exact per-motor order).

The mixing matrix and MOTOR_INVERT are the firmware's, verbatim. Nothing here
"corrects" them: the physical sense of sway/yaw is unresolved (INSPECTION.md H6), so
the blind sign factors live in `vehicle.yaml: signs.*` and default to identity.

order per motor i (NW, NE, SW, SE):
    cmd_eff = cmd * [1, sway_sign, yaw_sign]
    m_i     = motor_sign[i] * flip_i * (MIX[i] . cmd_eff)      flip = -1 if motor_invert
    m_i     = 0 if |m_i| <= deadband                          deadband = 0.03 (ESP32)
    target  = clamp(m_i, -1, 1)
    m_i    += (target - m_i) * dt / lag_tau                   first-order lag
    F_i     = m_i >= 0 ? m_i * max_fwd * scale_fwd[i] : m_i * max_rev * scale_rev[i]
    tau     = [ sum F_i*dir_i , sum (p_i x F_i*dir_i)_z ]
"""

from pathlib import Path

import torch
import yaml

__all__ = ["ThrusterModel", "load_config"]


def load_config(path: str | Path | None = None) -> dict:
    """Load `configs/vehicle.yaml` (default) or an explicit file."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "vehicle.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ThrusterModel:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda",
                 dtype=torch.float32) -> None:
        t = cfg["thrusters"]
        s = cfg.get("signs", {}) or {}
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.order = list(t["order"])
        n = self.num_envs
        k = lambda x: torch.as_tensor(x, device=self.device, dtype=dtype)  # noqa: E731

        self.mix = k(t["mix_matrix"])                                  # [4, 3]
        flip = torch.tensor([-1.0 if bool(inv) else 1.0 for inv in t["motor_invert"]],
                            device=self.device, dtype=dtype)
        motor_sign = k([s.get("motor_sign", {}).get(m, 1.0) for m in self.order])
        self.row_sign = flip * motor_sign                              # [4]
        self.axis_sign = k([1.0, float(s.get("sway_sign", 1.0)),
                            float(s.get("yaw_sign", 1.0))])            # [3]

        pos = k([t["positions_m"][m] for m in self.order])             # [4, 2] x_fwd, y_stbd
        dr = k([t["thrust_dir"][m] for m in self.order])               # [4, 2]
        self.pos = pos
        self.dir = dr

        self.max_fwd = float(t["max_thrust_fwd_n"])
        self.max_rev = float(t["max_thrust_rev_n"])
        self.deadband0 = float(t["deadband"])
        self.lag_tau0 = float(t["lag_tau_s"])

        self.nominal = {
            "scale_fwd": k([s.get("thrust_scale_fwd", {}).get(m, 1.0) for m in self.order]),
            "scale_rev": k([s.get("thrust_scale_rev", {}).get(m, 0.5) for m in self.order]),
        }

        # per-env state + randomisable gains (the env overwrites the gains per episode)
        self.m = torch.zeros(n, 4, device=self.device, dtype=dtype)          # motor command state
        self.scale_fwd = self.nominal["scale_fwd"].expand(n, 4).clone()
        self.scale_rev = self.nominal["scale_rev"].expand(n, 4).clone()
        self.lag_tau = torch.full((n, 1), self.lag_tau0, device=self.device, dtype=dtype)
        self.deadband = torch.full((n, 1), self.deadband0, device=self.device, dtype=dtype)

    # ------------------------------------------------------------------ gains
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the motor state and restore nominal gains for `env_ids` (default: all)."""
        idx = slice(None) if env_ids is None else env_ids
        self.m[idx] = 0.0
        self.scale_fwd[idx] = self.nominal["scale_fwd"]
        self.scale_rev[idx] = self.nominal["scale_rev"]
        self.lag_tau[idx] = self.lag_tau0
        self.deadband[idx] = self.deadband0

    def set_gains(self, env_ids, scale_fwd=None, scale_rev=None, lag_tau=None,
                  deadband=None, weakened_motor=None, weak_factor=0.6) -> None:
        """Per-env randomisation, called by the env on reset (CONTRACT §3.4)."""
        idx = slice(None) if env_ids is None else env_ids
        if scale_fwd is not None:
            self.scale_fwd[idx] = torch.as_tensor(scale_fwd, device=self.device, dtype=self.dtype)
        if scale_rev is not None:
            self.scale_rev[idx] = torch.as_tensor(scale_rev, device=self.device, dtype=self.dtype)
        if lag_tau is not None:
            self.lag_tau[idx] = torch.as_tensor(lag_tau, device=self.device,
                                                dtype=self.dtype).reshape(-1, 1)
        if deadband is not None:
            self.deadband[idx] = torch.as_tensor(deadband, device=self.device,
                                                dtype=self.dtype).reshape(-1, 1)
        if weakened_motor is not None:
            # per-env motor index (int tensor) that gets a factor on both directions
            self.scale_fwd[idx] = self.scale_fwd[idx].clone()
            self.scale_rev[idx] = self.scale_rev[idx].clone()
            wm = torch.as_tensor(weakened_motor, device=self.device).long().reshape(-1, 1)
            fwd = self.scale_fwd[idx].clone().scatter(1, wm, weak_factor)
            rev = self.scale_rev[idx].clone().scatter(1, wm, weak_factor)
            self.scale_fwd[idx] = fwd
            self.scale_rev[idx] = rev

    # ------------------------------------------------------------------ model
    def motor_thrust(self, cmd: torch.Tensor) -> torch.Tensor:
        """[N, 3] body command -> [N, 4] motor command after mixing, deadband and clamp."""
        cmd_eff = cmd * self.axis_sign
        m = cmd_eff @ self.mix.T * self.row_sign                       # [N, 4]
        m = torch.where(m.abs() <= self.deadband, torch.zeros_like(m), m)
        return m.clamp(-1.0, 1.0)

    def wrench(self, motor_thrust: torch.Tensor) -> torch.Tensor:
        """[N, 4] motor command -> [N, 3] body wrench [N, N, N*m]."""
        fwd = motor_thrust.clamp(min=0.0) * self.max_fwd * self.scale_fwd
        rev = motor_thrust.clamp(max=0.0) * self.max_rev * self.scale_rev
        f = fwd + rev                                                  # [N, 4], signed N
        fx = (f * self.dir[:, 0]).sum(dim=1)
        fy = (f * self.dir[:, 1]).sum(dim=1)
        yaw = (self.pos[:, 0] * f * self.dir[:, 1]).sum(dim=1) - \
              (self.pos[:, 1] * f * self.dir[:, 0]).sum(dim=1)
        return torch.stack([fx, fy, yaw], dim=1)

    def step(self, cmd: torch.Tensor, dt: float) -> torch.Tensor:
        """Apply lag and return the body wrench. `cmd` is saturated to [-1, 1]."""
        target = self.motor_thrust(cmd.clamp(-1.0, 1.0))
        alpha = (float(dt) / self.lag_tau.clamp(min=1e-3)).clamp(max=1.0)
        self.m = self.m + (target - self.m) * alpha
        return self.wrench(self.m)
