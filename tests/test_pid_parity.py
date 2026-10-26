"""Parity and stability tests for `baseline_pid.PIDBaseline`.

Parity is against the *real* Pi code, imported read-only from
`pi-control/src/control/thrust.py`. That file cannot be imported as-is on this
box: it pulls `src.sensors.fusion`, which needs pyserial/smbus/pigpio
(INSPECTION H12). So three hardware-only modules are stubbed before the import:

    src.sensors        - package shell
    src.sensors.fusion - `FusedState` dataclass (thrust.py uses it as a type hint)
    src.utils.logger   - `setup_logger()` without the rotating file handler

Everything that computes the command (`src.config`, `src.utils.geo`,
`src.control.pid`, `src.control.navigation`, `src.control.thrust`) is the real,
unmodified file, so the comparison is code-vs-code, not copy-vs-code.

Parity levels:
  * single step (`dt = 0.05`, the default first-call dt of `PIDController`) -
    full chain: navigation -> PID -> NED/body rotation -> deadband -> clamp.
  * multi step (`dt = 0.1`, wall clock frozen at a known value) - check that the
    integrator, the filtered derivative and the anti-windup track the original
    recurrence over several cycles.

Tolerances are explicit: `atol = 1e-4` on the normalised command (float32 in the
port vs float64 in the reference, plus a rad->deg round trip on the heading).

Stability tests close the loop on a toy first-order body-velocity plant - enough
to show the ported controller still converges; it is NOT the sim and says
nothing about the real vessel.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from baseline_pid import PIDBaseline
from common.frames import enu_to_latlon

PI_ROOT = Path(__file__).resolve().parents[2] / "pi-control"
ATOL = 1e-4
LAT0, LON0 = -6.200000, 106.816666   # arbitrary site, only used to talk lat/lon to the navigator


def _load_pi_stack():
    """Import the real thrust.py with the hardware modules stubbed out."""
    if str(PI_ROOT) not in sys.path:
        sys.path.insert(0, str(PI_ROOT))

    sensors = types.ModuleType("src.sensors")
    sensors.__path__ = [str(PI_ROOT / "src" / "sensors")]
    sys.modules.setdefault("src.sensors", sensors)

    if "src.sensors.fusion" not in sys.modules:
        fusion = types.ModuleType("src.sensors.fusion")

        @dataclass
        class FusedState:
            latitude: float = 0.0
            longitude: float = 0.0
            heading: float = 0.0
            speed: float = 0.0

        fusion.FusedState = FusedState
        sys.modules["src.sensors.fusion"] = fusion

    if "src.utils.logger" not in sys.modules:
        logger = types.ModuleType("src.utils.logger")
        logger.setup_logger = lambda name="asv": __import__("logging").getLogger(name)
        sys.modules["src.utils.logger"] = logger

    import src.config as cfg
    from src.control.navigation import (  # noqa: F401
        NavigationOutput,
        NavigationState,
        Waypoint,
        WaypointNavigator,
    )
    from src.control.pid import PIDController
    from src.control.thrust import ThrustComputer

    return types.SimpleNamespace(
        cfg=cfg, PIDController=PIDController, ThrustComputer=ThrustComputer,
        NavigationOutput=NavigationOutput, NavigationState=NavigationState,
        Waypoint=Waypoint, WaypointNavigator=WaypointNavigator,
        FusedState=sys.modules["src.sensors.fusion"].FusedState,
    )


@pytest.fixture(scope="module")
def pi():
    return _load_pi_stack()


def _ref_station_keeping_cycle(FusedState, nav, comp, hold_e, hold_n, north_m, east_m,
                               hold_heading_deg, heading_deg):
    """One real cycle: WaypointNavigator + ThrustComputer on lat/lon inputs.

    The 0.5 m DP deadband lives in `navigation.py`, so the navigator has to be in
    the loop or the reference would silently skip it.
    """
    lat, lon = enu_to_latlon(hold_e + east_m, hold_n + north_m, LAT0, LON0)
    out = nav.update(lat, lon, heading_deg, gps_valid=True)
    return comp.compute(out, FusedState(heading=heading_deg))


def _ref_station_keeping(pi, north_m, east_m, hold_heading_deg, heading_deg):
    """Fresh navigator + thrust computer, one cycle (single-step reference)."""
    nav = pi.WaypointNavigator()
    nav.start_station_keeping(LAT0, LON0, hold_heading_deg)
    return _ref_station_keeping_cycle(pi.FusedState, nav, pi.ThrustComputer(), 0.0, 0.0,
                                      north_m, east_m, hold_heading_deg, heading_deg)


SK_CASES = [
    # (north_m actual-hold, east_m actual-hold, hold heading deg, hull heading deg)
    (3.0, -4.0, 30.0, 350.0),      # heading error across the 0/360 seam
    (-1.2, 0.4, 120.0, 120.0),     # east error inside the 0.5 m DP deadband
    (8.0, 6.0, 200.0, 20.0),       # 180 deg heading error
    (0.2, 0.1, 0.0, 0.0),          # everything inside a deadband -> zero command
    (-6.5, 2.0, 359.0, 1.0),       # near-north small error
]


def _est_from_sk(hold_e, hold_n, hold_psi, north_m, east_m, heading_deg):
    """Estimator state [1,6] that reproduces (actual - hold) = (east_m, north_m)."""
    return torch.tensor([[hold_e + east_m, hold_n + north_m,
                          math.radians(heading_deg), 0.0, 0.0, 0.0]], dtype=torch.float32)


def test_station_keeping_single_step_parity(pi):
    """All five SK cases in one batched act() call vs five fresh Pi cycles."""
    hold_e, hold_n = 0.0, 0.0
    est = torch.cat([_est_from_sk(hold_e, hold_n, 0.0, n, e, h) for n, e, _, h in SK_CASES])
    goal = torch.tensor([[hold_e, hold_n, math.radians(hh), 1.0] for _, _, hh, _ in SK_CASES])

    got = PIDBaseline(dt=0.05).act(est, goal)
    ref = torch.tensor([[c.surge, c.sway, c.yaw] for c in
                        (_ref_station_keeping(pi, n, e, hh, h) for n, e, hh, h in SK_CASES)],
                       dtype=torch.float32)
    torch.testing.assert_close(got, ref, atol=ATOL, rtol=0.0)


def test_waypoint_single_step_parity(pi):
    """Waypoint mode: speed taper, turn-in-place, surge falloff, bearing heading."""
    cases = [  # (x_e, y_n, heading deg, goal_e, goal_n)
        (0.0, 0.0, 0.0, 0.0, 30.0),      # far, dead ahead -> full surge
        (0.0, 0.0, 0.0, 8.0, 3.0),       # 69 deg off -> tapered surge
        (2.0, 1.0, 90.0, 4.0, 3.0),      # inside slowdown radius
        (0.0, 0.0, 0.0, 1.0, -12.0),     # goal behind -> turn in place, surge 0
        (-20.0, 5.0, 200.0, -18.0, 8.0),  # short hop, ~45 deg bearing change
    ]
    est = torch.tensor([[x, y, math.radians(h), 0.0, 0.0, 0.0] for x, y, h, _, _ in cases],
                       dtype=torch.float32)
    goal = torch.tensor([[ge, gn, 0.0, 0.0] for _, _, _, ge, gn in cases], dtype=torch.float32)

    got = PIDBaseline(dt=0.05).act(est, goal)

    ref = []
    for x, y, h, ge, gn in cases:
        lat, lon = enu_to_latlon(x, y, LAT0, LON0)
        glat, glon = enu_to_latlon(ge, gn, LAT0, LON0)
        nav = pi.WaypointNavigator()
        nav.start_waypoint_nav([pi.Waypoint(latitude=glat, longitude=glon)])
        out = nav.update(lat, lon, h, gps_valid=True)
        assert out.mode == pi.NavigationState.WAYPOINT_NAV
        c = pi.ThrustComputer().compute(out, pi.FusedState(heading=h))
        ref.append([c.surge, c.sway, c.yaw])
    torch.testing.assert_close(got, torch.tensor(ref, dtype=torch.float32), atol=ATOL, rtol=0.0)


class _Clock:
    """Frozen wall clock; pid.py calls time.time() on its own module reference."""
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def frozen_clock(monkeypatch, pi):
    """Replace pid.py's `time` reference and prime `_last_time` so dt is exact."""
    import src.control.pid as pid_mod

    clock = _Clock()
    monkeypatch.setattr(pid_mod, "time", types.SimpleNamespace(time=clock))
    return clock


def test_station_keeping_multistep_parity(pi, frozen_clock):
    """Integrator / derivative filter / anti-windup over 8 cycles at dt = 0.1 s."""
    dt = 0.1
    comp = pi.ThrustComputer()
    for pid in (comp.pid_north, comp.pid_east, comp.pid_heading):
        pid._last_time = frozen_clock.t               # first clock advance then gives exactly dt

    # Pi output deadband stays ON: thrust.py zeroes |cmd| < 0.05 (H4), so turning it
    # off here would break parity on exactly the steps that cross that threshold.
    ctrl = PIDBaseline(dt=dt)
    nav = pi.WaypointNavigator()
    nav.start_station_keeping(LAT0, LON0, 90.0)
    hold_e, hold_n = 0.0, 0.0
    north_m, east_m, hull = 9.0, -7.0, 340.0

    for k in range(8):
        # a hand-made trajectory: error shrinks and the hull swings across north
        north_m, east_m, hull = north_m * 0.8, east_m * 0.75, (hull + 12.0) % 360.0
        frozen_clock.t += dt
        ref = _ref_station_keeping_cycle(pi.FusedState, nav, comp, hold_e, hold_n,
                                        north_m, east_m, 90.0, hull)
        got = ctrl.act(_est_from_sk(hold_e, hold_n, 0.0, north_m, east_m, hull),
                       torch.tensor([[hold_e, hold_n, math.radians(90.0), 1.0]]))
        torch.testing.assert_close(
            got[0], torch.tensor([ref.surge, ref.sway, ref.yaw], dtype=torch.float32),
            atol=ATOL, rtol=0.0,
            msg=f"step {k}")


def test_waypoint_multistep_parity(pi, frozen_clock):
    """Same, waypoint mode: the navigator drives heading and speed each cycle."""
    dt = 0.1
    comp = pi.ThrustComputer()
    for pid in (comp.pid_north, comp.pid_east, comp.pid_heading):
        pid._last_time = frozen_clock.t               # first clock advance then gives exactly dt

    glat, glon = enu_to_latlon(15.0, 20.0, LAT0, LON0)
    nav = pi.WaypointNavigator()
    nav.start_waypoint_nav([pi.Waypoint(latitude=glat, longitude=glon)])

    ctrl = PIDBaseline(dt=dt)                          # Pi output deadband on, see above
    x, y, hull = 0.0, 0.0, 10.0
    goal = torch.tensor([[15.0, 20.0, 0.0, 0.0]])

    for k in range(8):
        x, y, hull = x + 0.5, y + 0.8, hull + 18.0
        lat, lon = enu_to_latlon(x, y, LAT0, LON0)
        frozen_clock.t += dt
        out = nav.update(lat, lon, hull, gps_valid=True)
        assert out.mode == pi.NavigationState.WAYPOINT_NAV, "navigator dropped out"
        ref = comp.compute(out, pi.FusedState(heading=hull))
        got = ctrl.act(torch.tensor([[x, y, math.radians(hull), 0.0, 0.0, 0.0]]), goal)
        torch.testing.assert_close(
            got[0], torch.tensor([ref.surge, ref.sway, ref.yaw], dtype=torch.float32),
            atol=ATOL, rtol=0.0,
            msg=f"step {k}")


def test_output_contract():
    """Shape, dtype, range, deadband and mode separation as CONTRACT §4.6 requires."""
    est = torch.tensor([[0.0, 0.0, 0.0, 0.3, -0.2, 0.01],
                        [30.0, -40.0, 2.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    goal = torch.tensor([[0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [1.02, 1.01, 0.0, 1.0]], dtype=torch.float64)
    ctrl = PIDBaseline(dt=0.1)
    a = ctrl.act(est, goal, prev_action=None)
    assert a.shape == (3, 3) and a.dtype == torch.float32
    assert torch.all(a.abs() <= 1.0)
    # inside the DP deadband and inside the Pi output deadband -> exactly zero
    assert torch.count_nonzero(a[2]) == 0
    # a big goal in waypoint mode gives extra sway = 0 (thrust.py has no sway there)
    assert a[1, 1].item() == 0.0

    # reset() clears the integrators: the same input after reset must repeat
    first = ctrl.act(est[:1], goal[:1]).clone()
    ctrl.reset()
    torch.testing.assert_close(ctrl.act(est[:1], goal[:1]), first, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------- stability ---
class _ToyPlant:
    """First-order body-velocity plant, just enough to close the loop."""

    def __init__(self, x=0.0, y=0.0, psi_deg=0.0):
        self.x, self.y, self.psi = x, y, math.radians(psi_deg)
        self.u = self.v = self.r = 0.0

    def step(self, action, dt=0.02):
        a = action.tolist()
        tau = 0.4
        u_t, v_t, r_t = a[0] * 0.7, a[1] * 0.5, a[2] * 0.6
        self.u += (u_t - self.u) * min(dt / tau, 1.0)
        self.v += (v_t - self.v) * min(dt / tau, 1.0)
        self.r += (r_t - self.r) * min(dt / tau, 1.0)
        self.x += (self.u * math.sin(self.psi) + self.v * math.cos(self.psi)) * dt
        self.y += (self.u * math.cos(self.psi) - self.v * math.sin(self.psi)) * dt
        self.psi = (self.psi + self.r * dt + math.pi) % (2.0 * math.pi) - math.pi

    def est(self):
        return torch.tensor([[self.x, self.y, self.psi, self.u, self.v, self.r]],
                            dtype=torch.float32)


def _run(ctrl, plant, goal, n_steps, policy_dt=0.1, sub=5):
    for _ in range(n_steps):
        a = ctrl.act(plant.est(), goal)
        for _ in range(sub):
            plant.step(a[0], dt=policy_dt / sub)


def test_station_keeping_converges():
    """Step input to a fixed hold point settles and stays there."""
    plant = _ToyPlant()
    goal = torch.tensor([[5.0, 3.0, 0.0, 1.0]])
    ctrl = PIDBaseline(dt=0.1)
    _run(ctrl, plant, goal, n_steps=600)          # 60 s of policy steps
    # The 0.5 m DP deadband (navigation.py) stops correcting inside half a metre,
    # so "converged" here means ~1 m, not zero: measured 0.6-0.8 m on this plant.
    err = math.hypot(plant.x - 5.0, plant.y - 3.0)
    assert err < 1.0, f"radial error {err:.3f} m"

    # and it holds: no growth over the next 10 s
    _run(ctrl, plant, goal, n_steps=100)
    err2 = math.hypot(plant.x - 5.0, plant.y - 3.0)
    assert err2 < 1.0, f"radial error drifted to {err2:.3f} m"
    assert abs(err2 - err) < 0.25

    # integrator is what removes the steady-state offset, so it must be non-zero
    assert ctrl._pos["i"].abs().sum() > 1e-3
    # hold heading 0 -> hull within a few degrees of north
    assert abs(math.degrees(plant.psi)) < 10.0, math.degrees(plant.psi)


def test_waypoint_converges_into_arrival_radius():
    """Waypoint 15 m away: the boat travels there and arrives inside 2 m."""
    plant = _ToyPlant(x=-5.0, y=-2.0, psi_deg=200.0)
    goal = torch.tensor([[10.0, 13.0, 0.0, 0.0]])
    ctrl = PIDBaseline(dt=0.1)
    dists = []
    for _ in range(900):
        _run(ctrl, plant, goal, n_steps=1)
        dists.append(math.hypot(plant.x - 10.0, plant.y - 13.0))
    assert min(dists) < 2.0, f"closest approach {min(dists):.3f} m"
    assert dists[-1] < 2.5, f"final distance {dists[-1]:.3f} m"
    # it got there by moving, not by staying put
    assert math.hypot(plant.x + 5.0, plant.y + 2.0) > 10.0
