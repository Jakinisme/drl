"""Dynamics tests: free decay, steady state, psi wrap, frames, differentiability."""

import torch

from sim.dynamics import Dynamics, load_config

CFG = load_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dyn(n: int = 2, dt: float = 0.02) -> Dynamics:
    return Dynamics(CFG, n, device=DEVICE, dt=dt)


def state(u=0.0, v=0.0, r=0.0, psi=0.0, x=0.0, y=0.0, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor([[x, y, psi, u, v, r]], device=DEVICE, dtype=dtype)


def test_free_decay_is_monotone_and_positive():
    d = dyn(1)
    s = state(u=1.0, v=0.4, r=0.3)
    tau = torch.zeros(1, 3, device=DEVICE)
    speeds = [s[0, 3].item()]
    for _ in range(400):
        s = d.step(s, tau)
        speeds.append(s[0, 3].item())
        assert torch.isfinite(s).all()
    assert all(b <= a + 1e-7 for a, b in zip(speeds, speeds[1:]))   # never grows
    assert speeds[-1] < 0.1 * speeds[0]                             # actually decays


def test_steady_state_matches_drag_balance():
    d = dyn(1)
    s = state(u=0.0)
    tau = torch.tensor([[3.0, 0.0, 0.0]], device=DEVICE)            # pure surge force, N
    for _ in range(20000):
        s = d.step(s, tau)
    m = CFG["mass_kg"] + CFG["added_mass"]["x"]
    dl, dq = CFG["drag_linear"]["x"], CFG["drag_quadratic"]["x"]
    u = s[0, 3].item()
    assert abs(dl * u + dq * u * abs(u) - 3.0) < 1e-3               # drag == applied force
    assert u > 0 and u < CFG["max_surge_mps"] * 2
    assert abs(s[0, 4].item()) < 1e-4 and abs(s[0, 5].item()) < 1e-4
    assert m > 0


def test_psi_wraps_after_step():
    d = dyn(1)
    s = state(r=10.0, psi=3.0)
    s = d.step(s, torch.zeros(1, 3, device=DEVICE))
    assert -torch.pi <= s[0, 2] <= torch.pi


def test_surge_at_zero_heading_moves_north():
    d = dyn(1)
    s = state(u=0.8)
    s = d.step(s, torch.zeros(1, 3, device=DEVICE))
    assert s[0, 1].item() > 0          # y = north increases
    assert abs(s[0, 0].item()) < 1e-6  # no east motion


def test_sway_at_zero_heading_moves_east():
    d = dyn(1)
    s = state(v=0.5)
    s = d.step(s, torch.zeros(1, 3, device=DEVICE))
    assert s[0, 0].item() > 0          # y_b = starboard = east at psi = 0
    assert abs(s[0, 1].item()) < 1e-6


def test_step_is_differentiable_in_tau():
    d = dyn(1)
    s = state(u=0.1)
    tau = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE, requires_grad=True)
    out = d.step(s, tau)
    out[0, 3].backward()
    assert tau.grad is not None and torch.isfinite(tau.grad).all()
    assert tau.grad[0, 0].item() > 0


def test_implicit_drag_is_stable_with_large_drag_and_dt():
    d = dyn(1, dt=0.5)
    s = state(u=1.0)
    tau = torch.zeros(1, 3, device=DEVICE)
    for _ in range(50):
        s = d.step(s, tau)
    assert torch.isfinite(s).all()
    assert s[0, 3].item() >= 0.0


def test_batch_shapes_and_derivative():
    d = dyn(4)
    s = torch.zeros(4, 6, device=DEVICE)
    s[:, 3] = 0.2
    tau = torch.zeros(4, 3, device=DEVICE)
    assert d.step(s, tau).shape == (4, 6)
    assert d.derivative(s, tau).shape == (4, 6)
