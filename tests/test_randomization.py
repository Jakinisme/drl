"""R1: per-env randomisation of `sim/dynamics.py` and `sim/sensors.py` (CONTRACT §3.4).

(a) two envs with different mass/drag/latency solve different dynamics in one batch,
    proven from the numeric response of `step()`, not from attributes;
(b) two envs with different sensor noise/latency give different statistics;
(c) the un-randomised path is bit-for-bit what it was before the change — the
    reference values below were captured from the pre-change code (`float.hex()` of
    each float32 output, i.e. exact) and are compared with `torch.equal`.
"""

import copy

import torch

from sim.dynamics import Dynamics, load_config as load_vehicle
from sim.sensors import SensorModel, load_config as load_sensors
from sim.thrusters import ThrusterModel

VEH = load_vehicle()
SEN = load_sensors()
DEV = "cpu"          # bit-exact snapshots were taken on CPU
DT = 0.02

# --- (c) reference: 60 nominal dynamics steps, 2 envs -------------------------
DYN_REF = [
    "0x1.26062a0000000p-2", "0x1.358bfa0000000p-1", "0x1.aa53900000000p-2",
    "0x1.c92d620000000p-2", "0x1.dafacc0000000p-7", "0x1.8a5ef00000000p-5",
    "-0x1.e067200000000p-3", "-0x1.0fd5300000000p-3", "-0x1.3bb2b00000000p+0",
    "0x1.80f88a0000000p-4", "-0x1.5f1b3e0000000p-4", "0x1.69acd00000000p-9",
]
# --- (c) reference: 25 sensor steps from seed 1234, no set_noise() ------------
GEO_REF = [   # gps_lat[2], gps_lon[2], heading_deg[2], yaw_rate_dps[2], sog[2], cog[2]
    "0x1.5cb00e0000000p-18", "0x1.c02c120000000p-16", "0x1.e5cdb20000000p-18",
    "-0x1.6d04540000000p-16", "0x1.6ff0800000000p+1", "0x1.a1c6500000000p+6",
    "0x1.442d060000000p+2", "-0x1.7283480000000p+3", "0x1.ffef160000000p-2",
    "0x1.31f5320000000p-5", "0x1.072ce00000000p+3", "0x1.18a5140000000p+7",
]
EST_REF = [
    "0x1.d621d60000000p-2", "0x1.a441980000000p-2", "0x1.9afe080000000p-5",
    "0x1.fdb43c0000000p-2", "0x1.7dd6a00000000p-5", "0x1.6a1ba00000000p-4",
    "-0x1.cd8fd40000000p+0", "0x1.0d38800000000p+1", "0x1.d2a8a60000000p+0",
    "0x0.0p+0", "0x0.0p+0", "-0x1.9ddde60000000p-3",
]


def _t(ref, shape):
    """Exact float32 tensor from the hex snapshot (every value is a float32)."""
    return torch.tensor([float.fromhex(h) for h in ref], dtype=torch.float64).to(torch.float32).reshape(shape)


def _nominal_steps(d):
    s = torch.zeros(2, 6)
    s[0, 3], s[0, 4], s[0, 5], s[0, 2] = 0.7, 0.1, 0.2, 0.3
    s[1, 3], s[1, 4], s[1, 5], s[1, 2] = 0.2, -0.3, -0.1, -1.2
    for i in range(60):
        tau = torch.tensor([[1.0 + 0.01 * i, 0.2, 0.05], [-0.5, 0.1, -0.02]])
        s = d.step(s, tau)
    return s


def _sensor_run(m):
    for i in range(25):
        st = torch.zeros(2, 6)
        st[0, 0], st[0, 1], st[0, 2] = 0.1 * i, 0.05 * i, 0.01 * i
        st[0, 3], st[0, 4], st[0, 5] = 0.5, 0.02, 0.1
        st[1, 0], st[1, 1], st[1, 2] = -0.2 * i, 0.3 * i, 2.0 - 0.02 * i
        st[1, 3], st[1, 4], st[1, 5] = 0.1, -0.05, -0.2
        out = m.step(st, DT)
    return out


def test_no_randomisation_is_bit_identical_to_pre_change_snapshot():
    d = Dynamics(VEH, 2, device=DEV, dt=DT)
    assert torch.equal(_nominal_steps(d), _t(DYN_REF, (2, 6)))

    # set_params with explicit 1.0 (the nominal scales the env uses at level 0) must
    # leave the same trajectory untouched, bit for bit
    d2 = Dynamics(VEH, 2, device=DEV, dt=DT)
    d2.set_params(None, mass_scale=1.0, inertia_scale=1.0, drag_scale=1.0)
    assert torch.equal(_nominal_steps(d2), _t(DYN_REF, (2, 6)))

    torch.manual_seed(1234)
    m = SensorModel(SEN, 2, device=DEV)
    out = _sensor_run(m)
    geo = torch.stack([out["gps_lat"], out["gps_lon"], out["heading_deg"],
                       out["yaw_rate_deg_s"], out["gps_sog"], out["gps_cog_deg"]])
    assert torch.equal(geo, _t(GEO_REF, (6, 2)))
    assert torch.equal(out["est"], _t(EST_REF, (2, 6)))


def test_dynamics_per_env_mass_and_drag_change_step_response():
    """Same tau, same nu, different scale per env -> different nu_dot (checked vs. the
    closed-form implicit-Euler solution, so it cannot pass on attribute writes alone)."""
    d = Dynamics(VEH, 2, device=DEV, dt=DT)
    ms = torch.tensor([1.0, 1.5])
    ds = torch.tensor([1.0, 2.0])
    d.set_params(None, mass_scale=ms, drag_scale=ds)

    s = torch.zeros(2, 6)
    s[:, 3] = 0.5                                  # surge 0.5 m/s, no sway/yaw
    tau = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    out = d.step(s, tau)
    assert not torch.allclose(out[0, 3], out[1, 3])            # envs really differ

    m0 = VEH["mass_kg"] + VEH["added_mass"]["x"]
    dl0, dq0 = VEH["drag_linear"]["x"], VEH["drag_quadratic"]["x"]
    for i in range(2):
        m = m0 * float(ms[i])
        dl = dl0 * float(ms[i]) * float(ds[i])     # mass_scale also scales D_lin, as before
        dq = dq0 * float(ms[i]) * float(ds[i])
        u = 0.5
        want = (m / DT * u + 3.0 - dq * u * abs(u)) / (m / DT + dl)
        assert abs(float(out[i, 3]) - want) < 1e-5, (i, float(out[i, 3]), want)

    # derivative uses the same per-env M/D
    a = d.derivative(s, tau)[:, 3]
    assert not torch.allclose(a[0], a[1])

    # `_scales` rejects a wrong-length tensor instead of silently broadcasting
    try:
        d.set_params(None, mass_scale=torch.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-length scale must raise")


def test_dynamics_per_env_reset_keeps_other_envs():
    d = Dynamics(VEH, 3, device=DEV, dt=DT)
    d.set_params(torch.tensor([1]), mass_scale=torch.tensor([2.0]))
    assert float(d.M_env[1, 0]) == 2.0 * float(VEH["mass_kg"] + VEH["added_mass"]["x"])
    assert torch.equal(d.M_env[0], d.M) and torch.equal(d.M_env[2], d.M)
    d.reset(torch.tensor([1]))
    assert torch.equal(d.M_env[1], d.M)


def test_per_env_thruster_wrench_differs():
    """tau entering `Dynamics.step` is per env once gains are (env already does this)."""
    th = ThrusterModel(VEH, 2, device=DEV)
    th.set_gains(None, scale_fwd=torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                               [0.5, 0.5, 0.5, 0.5]]),
                 lag_tau=torch.tensor([0.12, 0.36]))
    cmd = torch.tensor([[0.5, 0.2, 0.1], [0.5, 0.2, 0.1]])
    tau = torch.stack([th.step(cmd, DT) for _ in range(10)])[-1]
    assert not torch.allclose(tau[0], tau[1])


def test_sensor_per_env_noise_statistics_differ():
    n, steps = 2, 600
    m = SensorModel(SEN, n, device=DEV)
    m.set_noise(None, pos_noise_m=torch.tensor([0.0, 2.0]),
                gps_latency_s=torch.tensor([0.0, 0.0]),
                vel_noise_mps=0.0, cog_noise_deg=0.0,
                gyro_noise_dps=torch.tensor([0.0, 3.0]), gyro_bias_dps=0.0,
                mag_noise_deg=0.0, mag_bias_deg=0.0)
    torch.manual_seed(7)
    st = torch.zeros(n, 6)
    st[:, 3] = 0.5                                  # same truth for both envs
    xs, gyros = [], []
    for _ in range(steps):
        out = m.step(st, DT)
        xs.append(out["est"][:, 0].clone())
        gyros.append(out["yaw_rate_deg_s"].clone())
    x = torch.stack(xs)                              # [steps, n]
    g = torch.stack(gyros)
    assert float(x[:, 0].std()) < 1e-3               # env 0: no position noise
    assert float(x[:, 1].std()) > 0.2                # env 1: pos_noise_m = 2.0
    assert float(g[:, 0].std()) < 1e-3               # env 0: no gyro noise
    assert float(g[:, 1].std()) > 2.0                # env 1: gyro_noise_dps = 3.0


def test_sensor_per_env_latency_differs():
    """No noise at all: the only difference is the delay, so the error is the lag."""
    c = copy.deepcopy(SEN)
    c["gps"] = dict(c["gps"], rate_hz=50.0)          # a fix every step, so the lag is visible
    n, u = 2, 1.0                                    # 1 m/s east
    m = SensorModel(c, n, device=DEV)
    m.set_noise(None, pos_noise_m=0.0, vel_noise_mps=0.0, cog_noise_deg=0.0,
                gyro_noise_dps=0.0, gyro_bias_dps=0.0, mag_noise_deg=0.0,
                mag_bias_deg=0.0, gps_latency_s=torch.tensor([0.0, 0.2]))
    for i in range(40):                              # 0.8 s, past the 0.2 s delay
        st = torch.zeros(n, 6)
        st[:, 0] = u * i * DT
        st[:, 3] = u
        out = m.step(st, DT)
    x = out["est"][:, 0]
    # env 0 has the minimum 1-step delay, env 1 the requested 10 steps
    assert abs(float(x[0] - x[1]) - (10 - 1) * u * DT) < 0.03

    # a single-env model with the same latency reproduces env 1 exactly
    m1 = SensorModel(c, 1, device=DEV)
    m1.set_noise(None, pos_noise_m=0.0, vel_noise_mps=0.0, cog_noise_deg=0.0,
                 gyro_noise_dps=0.0, gyro_bias_dps=0.0, mag_noise_deg=0.0,
                 mag_bias_deg=0.0, gps_latency_s=0.2)
    for i in range(40):
        st = torch.zeros(1, 6)
        st[:, 0] = u * i * DT
        st[:, 3] = u
        out1 = m1.step(st, DT)
    assert abs(float(out1["est"][0, 0]) - float(x[1])) < 1e-6
