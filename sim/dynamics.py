"""3-DOF Fossen-style ASV dynamics, vectorised over N envs (torch, GPU).

CONTRACT.md §1 (frames), §2 (state layout), §4.1 (interface).

    M_eff * nu_dot = tau - D_lin*nu - D_quad*|nu|*nu - C(nu)*nu        nu = (u, v, r)
    x_dot = u*sin(psi) + v*cos(psi)      (east,   psi = 0 -> bow north)
    y_dot = u*cos(psi) - v*sin(psi)      (north)
    psi_dot = r

M_eff = M_rb + M_A is diagonal: no off-diagonal added-mass coupling is modelled
(ponytail: add the full 3x3 M_A with cross terms when SYSID.md experiment D2 measures
them; the interface does not change). `tau` is the TOTAL body wrench in N / N*m, i.e.
thrusters plus disturbance — the env adds them.

Integration: semi-implicit Euler with the linear drag term treated implicitly, which
keeps free decay monotone and stable for any dt * D_lin / M; the quadratic drag and
Coriolis terms use the old velocity. Everything is differentiable w.r.t. `tau`.
"""

from pathlib import Path

import torch
import yaml

__all__ = ["Dynamics", "load_config"]


def load_config(path: str | Path | None = None) -> dict:
    """Load `configs/vehicle.yaml` (default) or an explicit file."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "vehicle.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + torch.pi, 2.0 * torch.pi) - torch.pi


class Dynamics:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda",
                 dtype=torch.float32, dt: float = 0.02) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.dt = float(dt)

        k = lambda x: torch.as_tensor(x, device=self.device, dtype=dtype)  # noqa: E731
        am, dl, dq = cfg["added_mass"], cfg["drag_linear"], cfg["drag_quadratic"]
        m_rb = [float(cfg["mass_kg"]), float(cfg["mass_kg"]), float(cfg["inertia_izz_kgm2"])]
        m_a = [float(am["x"]), float(am["y"]), float(am["n"])]

        self.M = k([m_rb[i] + m_a[i] for i in range(3)])            # [3]
        self.D_lin = k([dl["x"], dl["y"], dl["n"]])                 # [3]
        self.D_quad = k([dq["x"], dq["y"], dq["n"]])                # [3]

        # per-env randomised parameters (the env overwrites these on reset)
        self.M_env = self.M.expand(self.num_envs, 3).clone()
        self.D_lin_env = self.D_lin.expand(self.num_envs, 3).clone()
        self.D_quad_env = self.D_quad.expand(self.num_envs, 3).clone()

    # ------------------------------------------------------------------ params
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restore nominal mass/drag for `env_ids` (default: all)."""
        idx = slice(None) if env_ids is None else env_ids
        self.M_env[idx] = self.M
        self.D_lin_env[idx] = self.D_lin
        self.D_quad_env[idx] = self.D_quad

    def set_params(self, env_ids, mass_scale=None, inertia_scale=None,
                   drag_scale=None, quad_drag_scale=None) -> None:
        """Per-env randomisation (CONTRACT §3.4).

        `env_ids` selects rows (None = all). Every `*_scale` is a scalar, a 0-d
        tensor or a `[k]` tensor/array with one value per selected env; `None`
        restores that quantity to nominal. `mass_scale` scales both surge/sway
        inertia and their linear drag (as before); `quad_drag_scale` is an extra
        factor on the quadratic drag only.
        """
        idx = slice(None) if env_ids is None else env_ids
        k = self.num_envs if env_ids is None or isinstance(env_ids, slice) \
            else int(torch.as_tensor(env_ids).reshape(-1).numel())
        ms = self._scales(mass_scale, k)
        ins = self._scales(inertia_scale, k)
        ds = self._scales(drag_scale, k)
        qs = self._scales(quad_drag_scale, k)

        M = self.M.expand(k, 3).clone()
        d = torch.ones(k, 3, device=self.device, dtype=self.dtype)
        if ms is not None:
            M[:, :2] = M[:, :2] * ms.unsqueeze(1)
            d[:, :2] = d[:, :2] * ms.unsqueeze(1)
        if ins is not None:
            M[:, 2] = M[:, 2] * ins
        if ds is not None:
            d = d * ds.unsqueeze(1)
        self.M_env[idx] = M
        self.D_lin_env[idx] = self.D_lin * d
        self.D_quad_env[idx] = self.D_quad * (d if qs is None else d * qs.unsqueeze(1))

    def _scales(self, val, k: int) -> torch.Tensor | None:
        """Broadcast a scalar / `[k]` scale to `[k]` on this module's device+dtype."""
        if val is None:
            return None
        t = torch.as_tensor(val, device=self.device, dtype=self.dtype).reshape(-1)
        if t.numel() == 1:
            return t.expand(k)
        if t.numel() != k:
            raise ValueError(f"scale must have 1 or {k} values, got {t.numel()}")
        return t

    # ------------------------------------------------------------------ physics
    def derivative(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """[N, 6] state, [N, 3] total body wrench -> [N, 6] time derivative."""
        psi, nu = state[:, 2], state[:, 3:6]
        u, v, r = nu[:, 0], nu[:, 1], nu[:, 2]
        M, D_lin, D_quad = self.M_env, self.D_lin_env, self.D_quad_env

        drag = D_lin * nu + D_quad * nu.abs() * nu
        # 3-DOF rigid-body Coriolis/centripetal with diagonal M_eff:
        #   m_u*u_dot - m_v*v*r = X ,  m_v*v_dot + m_u*u*r = Y ,  m_r*r_dot + (m_v-m_u)*u*v = N
        cor = torch.stack([
            -M[:, 1] * v * r,
            M[:, 0] * u * r,
            (M[:, 1] - M[:, 0]) * u * v,
        ], dim=1)
        nu_dot = (tau - drag - cor) / M

        sp, cp = torch.sin(psi), torch.cos(psi)
        pose_dot = torch.stack([
            u * sp + v * cp,
            u * cp - v * sp,
            r,
        ], dim=1)
        return torch.cat([pose_dot, nu_dot], dim=1)

    def step(self, state: torch.Tensor, tau: torch.Tensor,
             dt: float | None = None) -> torch.Tensor:
        """Advance one step (semi-implicit Euler, implicit linear drag)."""
        h = self.dt if dt is None else float(dt)
        psi, nu = state[:, 2], state[:, 3:6]
        u, v, r = nu[:, 0], nu[:, 1], nu[:, 2]
        M, D_lin, D_quad = self.M_env, self.D_lin_env, self.D_quad_env

        cor = torch.stack([
            -M[:, 1] * v * r,
            M[:, 0] * u * r,
            (M[:, 1] - M[:, 0]) * u * v,
        ], dim=1)
        rhs = (M / h) * nu + tau - D_quad * nu.abs() * nu - cor
        nu_new = rhs / (M / h + D_lin)

        sp, cp = torch.sin(psi), torch.cos(psi)
        pos_new = torch.stack([
            state[:, 0] + (u * sp + v * cp) * h,
            state[:, 1] + (u * cp - v * sp) * h,
            _wrap_pi(psi + r * h),
        ], dim=1)
        return torch.cat([pos_new, nu_new], dim=1)
