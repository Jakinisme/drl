"""Thruster model tests: mixing vs firmware, deadband, lag, saturation, asymmetry."""

import re
from pathlib import Path

import pytest
import torch

from sim.thrusters import ThrusterModel, load_config

CFG = load_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_H = Path(__file__).resolve().parents[2] / "firmware-esp" / "src" / "header" / "config.h"


def model(n: int = 2) -> ThrusterModel:
    return ThrusterModel(CFG, n, device=DEVICE)


def test_mix_matrix_matches_firmware():
    """The sim matrix must be the firmware's, verbatim (config.h:85-90)."""
    mix = torch.tensor(CFG["thrusters"]["mix_matrix"])
    expected = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
    ])
    assert torch.equal(mix, expected)
    assert CFG["thrusters"]["order"] == ["NW", "NE", "SW", "SE"]

    if CONFIG_H.exists():  # cross-check against the actual source of truth
        src = CONFIG_H.read_text(encoding="utf-8", errors="replace")
        rows = re.findall(r"/\*\s*(NW|NE|SW|SE)\s*\*/\s*\{\s*([-+0-9.f,\s]+)\}", src)
        assert len(rows) == 4, "could not parse MIX_MATRIX from config.h"
        parsed = {name: [float(x.strip().rstrip("f")) for x in vals.split(",")]
                  for name, vals in rows}
        for name in CFG["thrusters"]["order"]:
            assert parsed[name] == expected[CFG["thrusters"]["order"].index(name)].tolist()


def test_ne_motor_is_inverted_by_firmware():
    m = model(1)
    out = m.motor_thrust(torch.tensor([[0.8, 0.0, 0.0]], device=DEVICE))[0].tolist()
    # pure surge: every motor gets +surge, NE is flipped by MOTOR_INVERT[NE] = true
    assert out[0] == pytest.approx(0.8) and out[2] == pytest.approx(0.8)
    assert out[1] == pytest.approx(-0.8)


def test_deadband_is_esp32_level_per_motor():
    m = model(1)
    db = CFG["thrusters"]["deadband"]
    assert db == pytest.approx(0.03)  # config.h:110, not the Pi's 0.05
    # a command whose mixed value sits below the deadband is zeroed per motor
    cmd = torch.tensor([[0.02, 0.0, 0.0]], device=DEVICE)
    assert torch.count_nonzero(m.motor_thrust(cmd)).item() == 0
    cmd = torch.tensor([[0.05, 0.0, 0.0]], device=DEVICE)
    out = m.motor_thrust(cmd)[0].tolist()
    assert out[0] == pytest.approx(0.05) and out[2] == pytest.approx(0.05)


def test_mixing_saturates_per_motor():
    m = model(1)
    out = m.motor_thrust(torch.tensor([[1.0, 1.0, 1.0]], device=DEVICE))
    assert out.abs().max().item() <= 1.0 + 1e-6
    assert out.abs().max().item() == pytest.approx(1.0)


def test_first_order_lag_reaches_target():
    m = model(1)
    cmd = torch.tensor([[0.6, 0.0, 0.0]], device=DEVICE)
    dt, tau = 0.02, CFG["thrusters"]["lag_tau_s"]
    out = m.step(cmd, dt)
    first = m.m[0, 0].item()
    assert 0.0 < first < 0.6  # not instantaneous
    assert first == pytest.approx(0.6 * min(dt / tau, 1.0), rel=1e-4)
    for _ in range(500):
        out = m.step(cmd, dt)
    assert m.m[0, 0].item() == pytest.approx(0.6, rel=1e-3)
    assert torch.isfinite(out).all()


def test_asymmetric_forward_reverse_thrust():
    m = model(1)
    fwd = m.wrench(torch.tensor([[0.5, 0.0, 0.0, 0.0]], device=DEVICE))
    rev = m.wrench(torch.tensor([[-0.5, 0.0, 0.0, 0.0]], device=DEVICE))
    t = CFG["thrusters"]
    s = CFG["signs"]
    ratio = (t["max_thrust_fwd_n"] * s["thrust_scale_fwd"]["NW"]) / \
            (t["max_thrust_rev_n"] * s["thrust_scale_rev"]["NW"])
    assert fwd.norm().item() / rev.norm().item() == pytest.approx(ratio, rel=1e-4)
    assert (fwd * rev).sum().item() < 0  # opposite directions


def test_body_axes_are_independent():
    """cmd -> wrench must be invertible: distinct commands, distinct wrenches.

    This vehicle does NOT decouple axes: MIX's sway column `[+,-,-,+]` combined with
    `MOTOR_INVERT[NE]` gives three motors forward and one reverse, so every axis couples
    (reports/A1.md open issues 1-2, INSPECTION H6). What must never happen is a *collapsed*
    map — e.g. diagonal-pairing `thrust_dir` with NW and SE sharing a direction makes pure
    surge and pure sway produce the identical wrench (rank 2), i.e. an uncontrollable sim.
    """
    t, s = CFG["thrusters"], CFG["signs"]
    d = t["thrust_dir"]
    for m in t["order"]:
        assert (d[m][0] ** 2 + d[m][1] ** 2) ** 0.5 == pytest.approx(1.0, abs=1e-3)
        assert d[m][0] > 0.0, "45-degree X mount: every prop pushes forward-ish"

    m = model(1)
    eye = torch.eye(3, device=DEVICE)
    T = torch.stack([m.wrench(m.motor_thrust(eye[i:i + 1] * 0.8))[0] for i in range(3)], dim=1)
    assert torch.linalg.matrix_rank(T).item() == 3
    # the intended axis responds strongest to its own command
    assert T[2, 2].abs() > 3.0 * T[2, 0].abs()        # yaw command -> dominant moment
    assert T[1, 0] * T[1, 1] < 0.0                    # surge and sway push oppositely sideways
    assert not torch.allclose(T[:, 0], T[:, 1], atol=1e-3)


def test_reset_zeroes_motor_state_and_gains():
    m = model(3)
    m.step(torch.full((3, 3), 0.5, device=DEVICE), 0.02)
    assert torch.count_nonzero(m.m) > 0
    m.set_gains(None, lag_tau=0.5, deadband=0.01)
    m.reset(torch.tensor([1], device=DEVICE))
    assert torch.count_nonzero(m.m[1]) == 0
    assert m.lag_tau[1].item() == pytest.approx(CFG["thrusters"]["lag_tau_s"])
    assert m.lag_tau[0].item() == pytest.approx(0.5)
