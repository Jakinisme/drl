"""PID baseline controller for the DRL ASV comparison (CONTRACT.md §4.6).

Faithful, batched port of `pi-control/src/control/thrust.py` plus the parts of
`pi-control/src/control/navigation.py` that feed it, re-expressed in the drl/
conventions of CONTRACT.md §1-§2 (psi radians CW from north, ENU metres):

    in : est  [B, 6] = (x_e, y_n, psi_rad, u, v, r)          estimator output
         goal [B, 4] = (x_goal_e, y_goal_n, psi_goal_rad, mode)
                       mode 0 = waypoint following, 1 = station keeping
    out: action [B, 3] = (surge, sway, yaw) in [-1, 1]  -> ESP32 mixing unchanged

What is ported verbatim (same numbers as `config.py`):

* position PID (north, east): kp 0.5, ki 0.05, kd 0.1, i_limit 2.0, out 1.0
* heading PID: kp 1.0, ki 0.1, kd 0.3, i_limit 1.0, out 1.0, heading wrap mode
* derivative low-pass coeff 0.1, dt clamp [0.001, 0.5]        (pid.py:32, 78-81)
* waypoint mode: 0.55 m/s max speed, slowdown radius 5 m (ratio floor 0.2),
  surge falloff 150 deg, min surge 0.25, turn-in-place 100 deg (config.py:63-75)
* station keeping: 0.5 m position deadband on (actual - hold), PID setpoint 0 so
  the error is (hold - actual), then NED -> body rotation        (navigation.py:206-225,
  thrust.py:161-172)
* output: per-axis 0.05 deadband, then clamp to [-1, 1]         (thrust.py:112-124)

Deliberate differences and open items (details in `reports/A3.md`):

* `u, v` are ignored - thrust.py has no velocity term at all (INSPECTION H7).
* `dt` is an argument (default 0.1 s = policy rate); the Pi version uses the wall
  clock inside `PIDController.compute()`.
* `mode` arrives per env and there is no waypoint list and no arrival advance: the
  env owns the goal (CONTRACT §4.6), so the navigator state machine is not ported.
  In waypoint mode the desired heading is recomputed as the bearing to the goal,
  which is exactly what `navigation.py:172` does and what CONTRACT §5 says
  `psi_goal` holds; `goal[2]` is used only for station keeping (hold heading).
* No action-rate limit: CONTRACT §4.6 says the env applies it.
* TODO(H4): the 0.05 output deadband is a Pi-side convenience kept **only** for
  parity. It is NOT the actuator deadband - that is 0.03 per mixed motor
  (`config.h:110`, owned by sim/thrusters.py per CONTRACT §4.2). Pass
  `output_deadband=0.0` to evaluate a controller whose small commands survive.
"""

from __future__ import annotations

import math

import torch

__all__ = ["PIDBaseline"]

# --- mirrored from pi-control/src/config.py (read-only upstream, do not tune here)
POSITION_PID_KP, POSITION_PID_KI, POSITION_PID_KD = 0.5, 0.05, 0.1
POSITION_PID_INTEGRAL_LIMIT, POSITION_PID_OUTPUT_LIMIT = 2.0, 1.0
HEADING_PID_KP, HEADING_PID_KI, HEADING_PID_KD = 1.0, 0.1, 0.3
HEADING_PID_INTEGRAL_LIMIT, HEADING_PID_OUTPUT_LIMIT = 1.0, 1.0
DERIVATIVE_FILTER_COEFF = 0.1
WAYPOINT_MAX_SPEED = 0.55
WAYPOINT_SLOWDOWN_RADIUS_M = 5.0
WAYPOINT_SLOWDOWN_MIN_RATIO = 0.2
WAYPOINT_SURGE_FALLOFF_DEG = 150.0
WAYPOINT_MIN_SURGE = 0.25
WAYPOINT_TURN_IN_PLACE_DEG = 100.0
STATION_KEEPING_DEADBAND_M = 0.5
THRUST_DEADZONE = 0.05


class PIDBaseline:
    """Batched PID baseline with the CONTRACT §4.6 controller interface.

    State (integrators, filtered derivatives, previous errors) is per env and is
    only advanced for the axes the current mode uses, exactly like
    `ThrustComputer`: waypoint mode steps the heading PID only, station keeping
    steps all three. `reset()` clears everything.
    """

    def __init__(
        self,
        dt: float = 0.1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        pos_gains: tuple[float, float, float] = (POSITION_PID_KP, POSITION_PID_KI, POSITION_PID_KD),
        pos_integral_limit: float = POSITION_PID_INTEGRAL_LIMIT,
        pos_output_limit: float = POSITION_PID_OUTPUT_LIMIT,
        head_gains: tuple[float, float, float] = (HEADING_PID_KP, HEADING_PID_KI, HEADING_PID_KD),
        head_integral_limit: float = HEADING_PID_INTEGRAL_LIMIT,
        head_output_limit: float = HEADING_PID_OUTPUT_LIMIT,
        output_deadband: float = THRUST_DEADZONE,   # TODO(H4): Pi convenience, not the actuator deadband
        derivative_filter_coeff: float = DERIVATIVE_FILTER_COEFF,
    ) -> None:
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = dtype
        self.pos_gains = tuple(float(g) for g in pos_gains)
        self.pos_integral_limit = float(pos_integral_limit)
        self.pos_output_limit = float(pos_output_limit)
        self.head_gains = tuple(float(g) for g in head_gains)
        self.head_integral_limit = float(head_integral_limit)
        self.head_output_limit = float(head_output_limit)
        self.output_deadband = float(output_deadband)
        self.d_filter = float(derivative_filter_coeff)
        self._B = 0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        dev, dt_ = self.device, self.dtype
        # position PIDs share gains -> one [B, 2] block, columns = (east, north)
        self._pos = {
            "i": torch.zeros(self._B, 2, device=dev, dtype=dt_),
            "e": torch.zeros(self._B, 2, device=dev, dtype=dt_),
            "d": torch.zeros(self._B, 2, device=dev, dtype=dt_),
        }
        self._head = {
            "i": torch.zeros(self._B, device=dev, dtype=dt_),
            "e": torch.zeros(self._B, device=dev, dtype=dt_),
            "d": torch.zeros(self._B, device=dev, dtype=dt_),
        }

    def _ensure(self, B: int) -> None:
        if B != self._B:
            self._B = B
            self.reset()

    # -------------------------------------------------------------------- PID
    def _pid(self, err, st, gains, i_limit, out_limit, dt):
        """One PID step, vectorised. Recurrence copied from pid.py:89-113."""
        kp, ki, kd = gains
        st["i"] = torch.clamp(st["i"] + err * dt, -i_limit, i_limit)
        raw_d = (err - st["e"]) / dt
        st["d"] = self.d_filter * raw_d + (1.0 - self.d_filter) * st["d"]
        st["e"] = err
        out = kp * err + ki * st["i"] + kd * st["d"]
        return torch.clamp(out, -out_limit, out_limit)

    # -------------------------------------------------------------------- act
    def act(self, est: torch.Tensor, goal: torch.Tensor,
            prev_action: torch.Tensor | None = None) -> torch.Tensor:
        """est [B, 6], goal [B, 4] -> action [B, 3] in [-1, 1]. See module docstring."""
        est = est.to(device=self.device, dtype=self.dtype)
        goal = goal.to(device=self.device, dtype=self.dtype)
        B = est.shape[0]
        self._ensure(B)

        dt = min(max(self.dt, 0.001), 0.5)   # pid.py:80-81
        rad2deg = 180.0 / math.pi
        deg2rad = math.pi / 180.0

        x, y, psi = est[:, 0], est[:, 1], est[:, 2]
        heading_deg = torch.remainder(psi * rad2deg, 360.0)     # fusion.py heading domain [0,360)

        gx, gy, gpsi, mode = goal[:, 0], goal[:, 1], goal[:, 2], goal[:, 3]
        sk = mode > 0.5

        # --- waypoint mode: navigation.py:163-204 -------------------------------
        de, dn = gx - x, gy - y
        dist = torch.hypot(de, dn)
        bearing_deg = torch.remainder(torch.atan2(de, dn) * rad2deg, 360.0)
        ratio = torch.clamp(dist / WAYPOINT_SLOWDOWN_RADIUS_M, min=WAYPOINT_SLOWDOWN_MIN_RATIO)
        speed = torch.where(dist < WAYPOINT_SLOWDOWN_RADIUS_M,
                            WAYPOINT_MAX_SPEED * ratio,
                            torch.full_like(dist, WAYPOINT_MAX_SPEED))
        # heading error in degrees, wrapped like geo.heading_error() -> [-180, 180)
        herr = torch.remainder(bearing_deg - heading_deg + 180.0, 360.0) - 180.0
        hfac = torch.clamp(1.0 - herr.abs() / WAYPOINT_SURGE_FALLOFF_DEG, min=WAYPOINT_MIN_SURGE)
        speed_norm = WAYPOINT_MAX_SPEED if WAYPOINT_MAX_SPEED > 0 else 1.0
        surge_wp = torch.clamp(speed * hfac / speed_norm, -1.0, 1.0)
        surge_wp = torch.where(herr.abs() >= WAYPOINT_TURN_IN_PLACE_DEG, torch.zeros_like(surge_wp), surge_wp)

        # --- station keeping: navigation.py:206-225, thrust.py:161-172 ----------
        north_m, east_m = y - gy, x - gx                  # actual - hold
        north_m = torch.where(north_m.abs() < STATION_KEEPING_DEADBAND_M, torch.zeros_like(north_m), north_m)
        east_m = torch.where(east_m.abs() < STATION_KEEPING_DEADBAND_M, torch.zeros_like(east_m), east_m)
        pos_cmd = self._pid(-torch.stack([east_m, north_m], dim=1), self._pos,   # error = 0 - meas
                            self.pos_gains, self.pos_integral_limit,
                            self.pos_output_limit, dt)
        east_cmd, north_cmd = pos_cmd[:, 0], pos_cmd[:, 1]
        cp, sp = torch.cos(psi), torch.sin(psi)
        surge_sk = north_cmd * cp + east_cmd * sp
        sway_sk = -north_cmd * sp + east_cmd * cp

        # --- heading PID: one instance for both modes ---------------------------
        psi_err = torch.remainder(gpsi - psi + math.pi, 2.0 * math.pi) - math.pi
        head_err_deg = torch.where(sk, psi_err * rad2deg, herr)               # thrust.py:135,170
        yaw = self._pid(head_err_deg, self._head, self.head_gains,
                        self.head_integral_limit, self.head_output_limit, dt)

        surge = torch.where(sk, surge_sk, surge_wp)
        sway = torch.where(sk, sway_sk, torch.zeros_like(sway_sk))

        # --- per-axis deadband then clamp: thrust.py:112-124 --------------------
        action = torch.stack([surge, sway, yaw], dim=1)
        action = torch.where(action.abs() < self.output_deadband, torch.zeros_like(action), action)
        return torch.clamp(action, -1.0, 1.0)
