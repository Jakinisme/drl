"""Sensor model tests: keys/shapes, noise levels, rate gating, heading wrap, est layout,
stale-fix decay, numerical stability. CONTRACT.md §4.4."""

import copy
import math

import torch

from common.frames import latlon_to_enu
from sim.sensors import SensorModel, load_config

BASE = load_config()
DT = 0.02
KEYS = {"gps_lat", "gps_lon", "gps_sog", "gps_cog_deg", "gps_valid", "heading_deg",
        "yaw_rate_deg_s", "u", "v", "r", "est", "sensor_ok"}


def cfg(**over):
    c = copy.deepcopy(BASE)
    for key, val in over.items():
        c[key].update(val)
    return c


def state(n=2, psi_deg=0.0, u=0.0, v=0.0, r_dps=0.0, x=0.0, y=0.0):
    s = torch.zeros(n, 6)
    s[:, 0] = x
    s[:, 1] = y
    s[:, 2] = math.radians(psi_deg)
    s[:, 3] = u
    s[:, 4] = v
    s[:, 5] = math.radians(r_dps)
    return s


def test_output_keys_shapes_dtypes():
    m = SensorModel(cfg(), 3, device="cpu")
    out = m.step(state(3), DT)
    assert set(out) == KEYS
    for k, v in out.items():
        assert v.shape == ((3, 6) if k == "est" else (3,)), k
        want = torch.bool if k in ("gps_valid", "sensor_ok") else torch.float32
        assert v.dtype == want, k


def test_noise_std_matches_config():
    """Loose 1-sigma check against configs/sensors.yaml (placeholders, not calibration)."""
    c = cfg(gps={"rate_hz": 50.0, "latency_s": 0.0},
            imu={"gyro_bias_dps": 0.0},
            mag={"bias_deg": 0.0})
    n = 4
    m = SensorModel(c, n, device="cpu")
    s = state(n, psi_deg=30.0, u=0.6)
    for _ in range(5):                       # past the first-fix snap
        m.step(s, DT)

    east, north, sog, gyro = [], [], [], []
    for _ in range(400):
        out = m.step(s, DT)
        e, nn = latlon_to_enu(out["gps_lat"].numpy(), out["gps_lon"].numpy(), 0.0, 0.0)
        east.append(torch.as_tensor(e))
        north.append(torch.as_tensor(nn))
        sog.append(out["gps_sog"])
        gyro.append(out["yaw_rate_deg_s"])

    pos_std = torch.cat([torch.cat(east), torch.cat(north)]).std().item()
    assert 0.28 < pos_std < 0.55, pos_std          # config pos_noise_m = 0.4
    sog_std = torch.cat(sog).std().item()
    assert 0.02 < sog_std < 0.10, sog_std          # config vel_noise_mps = 0.05
    gyro_std = torch.cat(gyro).std().item()
    assert 0.20 < gyro_std < 0.45, gyro_std        # config gyro_noise_dps = 0.3


def test_zero_noise_reproduces_truth():
    c = cfg(gps={"latency_s": 0.0, "pos_noise_m": 0.0, "vel_noise_mps": 0.0,
                 "cog_noise_deg": 0.0},
            imu={"gyro_noise_dps": 0.0, "gyro_bias_dps": 0.0},
            mag={"noise_deg": 0.0, "bias_deg": 0.0})
    m = SensorModel(c, 2, device="cpu")
    s = state(2, psi_deg=-20.0, u=0.5, x=3.0, y=-2.0)
    for _ in range(10):
        out = m.step(s, DT)
    assert torch.allclose(out["heading_deg"], torch.full((2,), -20.0), atol=1e-4)
    assert torch.allclose(out["yaw_rate_deg_s"], torch.zeros(2), atol=1e-6)
    assert torch.allclose(out["gps_sog"], torch.full((2,), 0.5), atol=1e-4)
    assert torch.allclose(out["est"][:, 0], torch.full((2,), 3.0), atol=1e-3)
    assert torch.allclose(out["est"][:, 1], torch.full((2,), -2.0), atol=1e-3)
    assert torch.allclose(out["est"][:, 3], torch.full((2,), 0.5), atol=1e-3)
    assert torch.allclose(out["est"][:, 4], torch.zeros(2), atol=1e-3)


def test_gps_rate_gating_and_hold_last():
    c = cfg(gps={"rate_hz": 5.0, "latency_s": 0.0})
    m = SensorModel(c, 1, device="cpu")
    s = state(1, u=0.5)
    lat = torch.stack([m.step(s, DT)["gps_lat"] for _ in range(100)])   # 2 s at 50 Hz
    epochs = int((lat[1:] != lat[:-1]).sum().item())                    # 5 Hz -> ~10 fixes
    assert 8 <= epochs <= 12, epochs

    # hold_last = false: the fix is only valid on the epoch itself
    m2 = SensorModel(cfg(gps={"rate_hz": 5.0, "latency_s": 0.0, "hold_last": False}),
                     1, device="cpu")
    valid = torch.stack([m2.step(state(1, u=0.5), DT)["gps_valid"] for _ in range(12)])
    assert bool(valid[0]) and not bool(valid[1:10].any()) and bool(valid[10])


def test_heading_wrap_stays_in_range():
    m = SensorModel(cfg(), 2, device="cpu")
    psi_deg = 170.0
    hs = []
    for _ in range(400):                 # sweeps past the +/-180 seam several times
        psi_deg = (psi_deg + 2.5) % 360.0
        hs.append(m.step(state(2, psi_deg=psi_deg, u=0.6), DT)["heading_deg"])
    h = torch.stack(hs)
    assert bool((h >= -180.0).all()) and bool((h < 180.0).all())
    # a wrapped heading must be self-consistent with the est layout (radians)
    assert float(h.abs().max()) > 90.0


def test_est_layout_matches_state_convention():
    m = SensorModel(cfg(), 3, device="cpu")
    s = state(3, psi_deg=45.0, u=0.6, v=0.05, r_dps=3.0, x=1.0, y=2.0)
    for _ in range(50):
        out = m.step(s, DT)
    est = out["est"]
    assert est.shape == (3, 6) and est.dtype == torch.float32
    assert torch.equal(est[:, 3], out["u"]) and torch.equal(est[:, 4], out["v"])
    assert torch.equal(est[:, 5], out["r"])
    assert torch.allclose(est[:, 2], torch.deg2rad(out["heading_deg"]))
    assert bool((est[:, 2] >= -math.pi).all()) and bool((est[:, 2] < math.pi).all())
    assert bool(torch.isfinite(est).all())


def test_stale_fix_flags_invalid_and_decays_velocity():
    """GPS at 0.2 Hz, stale after 1 s: the fix dies, (u, v) decay instead of snapping."""
    c = cfg(gps={"rate_hz": 0.2, "latency_s": 0.0, "stale_after_s": 1.0})
    m = SensorModel(c, 2, device="cpu")
    s = state(2, u=0.8)
    out = m.step(s, DT)
    assert bool(out["gps_valid"].all())
    u0 = out["u"].clone()
    assert torch.allclose(u0, torch.full((2,), 0.8), atol=0.15)

    us, valid = [], []
    for _ in range(100):                       # next fix would be at t = 5 s
        out = m.step(s, DT)
        us.append(out["u"])
        valid.append(out["gps_valid"])
    u = torch.stack(us)                        # [100, 2]
    assert not bool(torch.stack(valid)[-1].any())
    steps = (u[1:] - u[:-1]).abs().max().item()
    assert steps < 0.05, steps                 # no snap to zero
    assert 0.10 < float(u[-1].mean()) < 0.45   # first-order decay, tau = 1 s


def test_no_nan_over_500_steps():
    n = 4
    m = SensorModel(cfg(), n, device="cpu")
    g = torch.Generator().manual_seed(0)
    for i in range(500):
        s = torch.stack([
            torch.rand(n, generator=g) * 20.0 - 10.0,
            torch.rand(n, generator=g) * 20.0 - 10.0,
            torch.rand(n, generator=g) * 2.0 * math.pi - math.pi,
            torch.randn(n, generator=g) * 1.2,
            torch.randn(n, generator=g) * 0.4,
            torch.randn(n, generator=g) * 1.0,
        ], dim=1)
        if i == 250:
            m.reset(torch.tensor([1, 3]))
        out = m.step(s, DT)
        for k, v in out.items():
            assert bool(torch.isfinite(v.to(torch.float32)).all()), f"{k} not finite at {i}"
