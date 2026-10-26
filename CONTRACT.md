# CONTRACT.md — shared interface for the DRL ASV controller

Every module in `drl/` codes against this file. If an implementation needs something
that is not here, add it here first (Wave-0 owner), do not invent a private convention.
Sources: `drl/TASK.md`, `drl/AGENTS.md`, `drl/INSPECTION.md` (measured facts and open
questions about the existing vessel).

Anything marked **TODO** is a placeholder that is not derivable from the repo. Do not
present a TODO value as identified, and do not change the key name.

---

## 0. Import and layout rules

- Repository root for imports is `drl/` itself. There is **no** `drl/__init__.py`
  (deliberate). All cross-module imports are top-level packages:
  `from common.frames import ...`, `from sim.dynamics import Dynamics`,
  `from envs.asv_env import AsvEnv`, `from runtime.policy import RLController`.
- Tests run from `drl/`: `py -3.13 -m pytest tests -q` (cwd on `sys.path`).
- Units: SI everywhere — m, s, rad, kg, N, N·m, N/(m/s), N/(m/s)². Degrees only where
  a value crosses the Pi/ONNX/JSON boundary and is documented as degrees.
- Batch dimension is always first and named `N` (envs) / `B` (controllers).
- `dtype`: `torch.float32` for all tensors on the training side; `np.float32` for the
  ONNX boundary; `np.float64` inside `common/frames.py` math (functions accept and
  return float64 arrays/scalars, cast at the call site).

## 1. Frames and sign conventions

World frame: local ENU tangent plane, **origin = first valid GPS fix** (`lat0, lon0`).
`x = east [m]`, `y = north [m]`, `z = up` (unused, 3-DOF planar).

Heading `psi`: radians, **clockwise from +y (north)**:
`psi = 0` → bow north, `psi = +pi/2` → bow east.
`psi = deg2rad(compass_heading_deg)`; this is the convention the existing navigation,
PID, `geo.bearing()` and body-frame rotation use (INSPECTION.md §1). The magnetometer
driver's own angle is a different (CCW) mathematical angle adjusted by two fudge
constants — that is a vessel bug, not a drl/ convention (INSPECTION.md H1–H3).

Body frame: `x_b` = bow (forward), `y_b` = starboard (right), `z_b` = down is **not**
used. Body velocity: `u` surge [m/s] forward, `v` sway [m/s] starboard,
`r` yaw rate [rad/s], `r = dpsi/dt`, positive = bow turns to starboard (clockwise seen
from above).

Kinematics (the sim must use exactly this):

```
x_dot = u·sin(psi) + v·cos(psi)
y_dot = u·cos(psi) − v·sin(psi)
psi_dot = r
```

Vector rotation (implemented once, in `common/frames.py`):

```
enu_to_body(e, n, psi) -> (u, v) = ( e·sin psi + n·cos psi ,  e·cos psi − n·sin psi )
body_to_enu(u, v, psi) -> (e, n) = ( u·sin psi + v·cos psi ,  u·cos psi − v·sin psi )
```

Heading error: `psi_err = wrap_pi(psi_goal − psi)`, positive = turn starboard. Degrees
equivalent `(desired − current + 180) mod 360 − 180` (matches `geo.py:108`).

`common/frames.py` public API (only these names; extended only by editing this file):
`EARTH_RADIUS_M`, `wrap_deg`, `wrap_pi`, `heading_error_deg`, `heading_error_rad`,
`heading_deg_to_psi`, `psi_to_heading_deg`, `latlon_to_enu`, `enu_to_latlon`,
`enu_to_body`, `body_to_enu`, `goal_in_body`.

Projection: equirectangular, `R = 6 371 000 m`, `east` scaled by `cos(lat0)` — the same
numbers as `geo.lat_lon_offset()` and `navigation.py:211-212`. Valid to a few km.
**TODO**: revisit if the operating area exceeds ~10 km (then use a proper geodetic
conversion; the API stays the same).

NED is not used inside `drl/`. If a module ever needs it (e.g. to talk to a
north-east-down consumer), convert at the boundary: `N = y`, `E = x`, yaw NED =
`−psi` (CCW), and document it there.

## 2. State and command layout

Simulator state tensor `state: [N, 6]`, `torch.float32`:

| idx | symbol | meaning | unit |
|---|---|---|---|
| 0 | `x` | east position, local ENU | m |
| 1 | `y` | north position, local ENU | m |
| 2 | `psi` | heading, clockwise from north, wrapped to `[-pi, pi)` | rad |
| 3 | `u` | surge velocity, body | m/s |
| 4 | `v` | sway velocity, body (starboard +) | m/s |
| 5 | `r` | yaw rate, `dpsi/dt` | rad/s |

Action `action: [N, 3]`, `torch.float32`, each component in `[-1, 1]`:
`[surge, sway, yaw]` = the existing ESP32 body command (`esp32_link.py:96-105`), i.e.
the same 3 numbers the PID produces today. Saturation at ±1 is applied by the env
before the thruster model; the mixer inside the policy network already bounds it.

Body wrench `tau: [N, 3]` = `[surge_force N, sway_force N, yaw_moment N·m]`.

## 3. Configuration files (paths and keys are fixed)

All configs are YAML, parsed with `PyYAML`, loaded relative to `drl/` (e.g.
`Path(__file__).with_name("configs")`). Every physical value that has not been
identified on the real vessel must carry an inline `# TODO SYSID` comment. Values below
marked `TODO` are placeholders: pick something plausible, flag it, and list it in
`SYSID.md`.

### 3.1 `configs/vehicle.yaml` (owner: A1 `sim/dynamics.py`, `sim/thrusters.py`)

```yaml
mass_kg: 12.0                 # TODO SYSID
inertia_izz_kgm2: 1.2         # TODO SYSID
added_mass: {x: 3.0, y: 8.0, n: 0.6}          # TODO SYSID, kg / kg·m^2
drag_linear: {x: 6.0, y: 18.0, n: 1.5}        # TODO SYSID, N/(m/s), N·m/(rad/s)
drag_quadratic: {x: 4.0, y: 12.0, n: 0.8}     # TODO SYSID, N/(m/s)^2
max_surge_mps: 1.0            # soft cap used only for reward/obs scaling
thrusters:
  order: [NW, NE, SW, SE]     # MUST equal config.h MotorID order
  positions_m:                # body frame [x_fwd, y_stbd] relative to CG
    NW: [0.45, -0.30]         # TODO SYSID (NW = front-left)
    NE: [0.45,  0.30]         # TODO SYSID
    SW: [-0.45, -0.30]        # TODO SYSID
    SE: [-0.45,  0.30]        # TODO SYSID
  thrust_dir:                 # unit body-frame direction of each prop's thrust
    NW: [0.707, -0.707]       # TODO SYSID — see INSPECTION H6
    NE: [0.707,  0.707]       # TODO SYSID
    SW: [0.707, -0.707]       # TODO SYSID
    SE: [0.707,  0.707]       # TODO SYSID
  max_thrust_fwd_n: 9.8       # TODO SYSID bollard pull per motor
  max_thrust_rev_n: 4.9       # TODO SYSID (assumed ~half of forward)
  deadband: 0.03              # config.h:110 MOTOR_DEADZONE (measured fact)
  lag_tau_s: 0.12             # TODO SYSID first-order motor lag
  mix_matrix:                 # config.h:85-90 verbatim, rows NW,NE,SW,SE
    - [ 1.0,  1.0,  1.0]
    - [ 1.0, -1.0, -1.0]
    - [ 1.0, -1.0,  1.0]
    - [ 1.0,  1.0, -1.0]
  motor_invert: [false, true, false, false]   # config.h:100-105 verbatim
```

`mix_matrix` and `motor_invert` are **verbatim from firmware** and must never be
"corrected" here; INSPECTION.md H6 explains why the physical sign of sway/yaw is still
open (the sim geometry must be fixed by a bench test, not by editing the matrix).

### 3.2 `configs/disturbance.yaml` (owner: A2 `sim/disturbance.py`)

```yaml
wind:      {enabled: true, rho_air_kgm3: 1.225, cd_area_m2: 0.35, max_speed_mps: 8.0}   # TODO cd_area
current:   {enabled: true, speed_mps: 0.25, direction_deg: 0.0, drag_linear: 6.0}      # TODO
waves:     {enabled: true, n_components: 4, freq_hz: [0.15, 0.25, 0.4, 0.6],
            amp_n: [2.0, 1.5, 1.0, 0.6], yaw_amp_nm: [0.3, 0.2, 0.1, 0.05]}               # TODO
curriculum: {n_levels: 6, scale_by_level: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]}
seed_offset: 0
```

`level 0` must produce exactly zero disturbance (tested by A2).

### 3.3 `configs/sensors.yaml` (owner: A2 `sim/sensors.py`)

```yaml
gps:      {rate_hz: 5.0, pos_noise_m: 0.4, vel_noise_mps: 0.05, cog_noise_deg: 3.0,
           latency_s: 0.2, hold_last: true, stale_after_s: 2.0,  # TODO all noise values
           origin_lat: null, origin_lon: null}                    # TODO: real site or random
imu:      {rate_hz: 50.0, gyro_noise_dps: 0.3, gyro_bias_dps: 0.2,
           accel_noise_mps2: 0.05, stale_after_s: 0.5}            # TODO
mag:      {rate_hz: 15.0, noise_deg: 2.0, bias_deg: 5.0, stale_after_s: 1.0}  # TODO
estimator: {type: complementary, mag_weight: 0.02, gyro_weight: 0.98,
            gps_alpha: 0.7, gps_heading_min_speed_mps: 0.3}      # from config.py:38-40, fusion.py:54
```

The `estimator` keys mirror the measured Pi values and are the only ones that are not
TODO. The sim estimator must reproduce the *structure* of `fusion.py` (mag-primary +
gyro propagation + GPS-course correction when fast, position low-pass with
`gps_alpha`), so the policy sees the same imperfections it will meet on the water.
Policy observations may only use estimator output, never `state`.

### 3.4 `configs/env.yaml` (owner: B1 `envs/asv_env.py`)

```yaml
num_envs: 2048
dt_s: 0.02                 # sim step (TASK.md)
policy_hz: 10.0            # -> policy_dt = 0.1 s, 5 sim steps per action
episode_s: 120.0
max_steps: null            # derived: episode_s * policy_hz
action:
  clip: 1.0
  rate_limit_per_s: 2.0    # |a_t - a_{t-1}| <= rate_limit_per_s * policy_dt
obs:
  k_history: 10
  offset: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  scale:  [10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 1.0]
  clip_norm: 5.0           # normalized values clipped to [-5, 5]
episode:
  arrival_radius_m: 2.0    # config.py:63 WAYPOINT_ARRIVAL_RADIUS_M
  slowdown_radius_m: 5.0   # config.py:68
  geofence_m: 80.0         # TODO
  station_keeping_fraction: 0.3
  n_waypoints: [1, 4]      # range per episode
reward:
  w_position: 1.0
  w_heading: 0.3
  w_progress: 0.5
  w_cross_track: 0.2
  w_near_goal_speed: 0.1
  w_action_rate: 0.05
  w_energy: 0.01
  pos_sigma_m: 10.0        # exponential kernels
  head_sigma_rad: 0.5
randomization:
  mass: 0.30               # +/- fraction
  inertia: 0.30
  drag: 0.30
  thruster_gain: 0.20
  thruster_lag: 0.30
  thruster_deadband: 0.01  # additive
  fwd_rev_asymmetry: 0.20
  weak_thruster_prob: 0.1
  sensor_noise: 0.30
  sensor_bias_deg: 5.0
  sensor_latency_s: 0.1
curriculum:
  start_level: 0
  level_up_fraction: 0.6   # of max_steps
seed: 0
```

### 3.5 `configs/train.yaml` (owner: B2 `train.py`, `models.py`)

```yaml
algo: ppo                  # rsl_rl if the installed version fits, else built-in PPO
total_steps: 20000000
num_envs: 2048
rollout_steps: 32          # per env, per iteration
gamma: 0.99
lam: 0.95
clip_ratio: 0.2
lr: 3.0e-4
epochs: 5
minibatches: 4
entropy_coef: 0.005
value_coef: 0.5
max_grad_norm: 1.0
hidden: [256, 128, 128]
activation: elu
init_log_std: -0.5
device: cuda
dtype: float32
log_dir: runs
ckpt_dir: runs/checkpoints
save_interval_steps: 500000
seed: 0
env_config: configs/env.yaml
vehicle_config: configs/vehicle.yaml
disturbance_config: configs/disturbance.yaml
sensors_config: configs/sensors.yaml
```

## 4. Module interfaces

Signatures are binding. Extra keyword arguments with defaults are allowed; changing
existing names, order or shapes is not.

### 4.1 `sim/dynamics.py`

```python
class Dynamics:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda",
                 dtype=torch.float32, dt: float = 0.02) -> None: ...
    def derivative(self, state: Tensor[N, 6], tau: Tensor[N, 3]) -> Tensor[N, 6]: ...
    def step(self, state: Tensor[N, 6], tau: Tensor[N, 3],
             dt: float | None = None) -> Tensor[N, 6]: ...
```

Model: `M_eff · nu_dot = tau + tau_env − D_lin·nu − D_quad·|nu|·nu − C(nu)·nu`, with
`M_eff = M_rb + M_A`. Integration: semi-implicit Euler at `dt` (explicit on drag is
unstable for large `D_quad`; use the implicit linear-drag form or sub-step — A1's
choice, but `test_dynamics.py` must show a free-decay test that is monotone and
`step()` must be differentiable w.r.t. `tau`). `psi` is wrapped to `[-pi, pi)` after
every step; `u, v, r` are **not** clamped (that is physics, not safety).

### 4.2 `sim/thrusters.py`

```python
class ThrusterModel:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda",
                 dtype=torch.float32) -> None: ...
    def reset(self, env_ids: Tensor | None = None) -> None: ...
    def motor_thrust(self, cmd: Tensor[N, 3]) -> Tensor[N, 4]: ...   # normalized [-1,1]
    def wrench(self, motor_thrust: Tensor[N, 4]) -> Tensor[N, 3]: ...  # N, N, N·m
    def step(self, cmd: Tensor[N, 3], dt: float) -> Tensor[N, 3]: ...  # applies lag
```

Per motor: `m_i = flip_i · (MIX[i] · cmd)` (with `flip = +1` unless `motor_invert[i]`),
deadband → 0, clamp ±1, first-order lag `dm/dt = (m_target − m)/lag_tau`, then
`F_i = m_i ≥ 0 ? m_i·max_fwd : m_i·max_rev` (asymmetric), and
`tau = Σ_i [ F_i·dir_i , (p_i × F_i·dir_i)_z ]` with `p_i` from `thrusters.positions_m`.
Static randomisation knobs (`step` must honour per-env gains) are set by the env.

### 4.3 `sim/disturbance.py`

```python
class Disturbance:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda",
                 dtype=torch.float32) -> None: ...
    def reset(self, env_ids: Tensor | None = None,
              level: Tensor[N] | float = 0.0) -> None: ...
    def set_level(self, level: Tensor[N] | float) -> None: ...
    def sample(self, t: float, dt: float, state: Tensor[N, 6]) -> Tensor[N, 3]: ...
```

Returns body-frame `[surge_force N, sway_force N, yaw_moment N·m]`. Wind is quadratic in
the *relative* wind speed, current acts on the relative velocity, waves are a sum of
sinusoids. `level = 0` ⇒ exact zeros. Stateless w.r.t. `t` except for the wave phases
fixed at `reset()`.

### 4.4 `sim/sensors.py`

```python
class SensorModel:
    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda",
                 dtype=torch.float32) -> None: ...
    def reset(self, env_ids: Tensor | None = None) -> None: ...
    def step(self, true_state: Tensor[N, 6], dt: float) -> dict[str, Tensor]: ...
```

Returned dict (exact keys; all tensors `[N]` except `est`):

| key | shape | unit |
|---|---|---|
| `gps_lat`, `gps_lon` | `[N]` | deg |
| `gps_sog` | `[N]` | m/s |
| `gps_cog_deg` | `[N]` | deg, 0 = north, CW |
| `gps_valid` | `[N]` bool | — |
| `heading_deg` | `[N]` | deg, 0 = north, CW (fused) |
| `yaw_rate_deg_s` | `[N]` | deg/s |
| `u`, `v` | `[N]` | m/s (estimator, body) |
| `r` | `[N]` | rad/s |
| `est` | `[N, 6]` | state layout of §2, estimator output |
| `sensor_ok` | `[N]` bool | all sources fresh |

`est` is the only thing the policy may observe. `true_state` is for the reward and for
evaluation metrics only.

### 4.5 `envs/asv_env.py`

```python
class AsvEnv:
    num_envs: int            # N
    obs_dim: int             # 150
    act_dim: int             # 3
    dt: float                # 0.02
    policy_dt: float         # 0.1
    device: str

    def reset(self, env_ids: Tensor | None = None) -> Tensor[N, 150]: ...
    def step(self, action: Tensor[N, 3]
             ) -> tuple[Tensor[N, 150], Tensor[N], Tensor[N], dict]: ...
    def get_privileged_state(self) -> Tensor[N, 6]: ...   # evaluation/metrics only
    def set_curriculum(self, level: float) -> None: ...
    def close(self) -> None: ...
```

`step` returns `(obs, reward, done, info)`; `done` is `float32` 0/1 (or bool, documented
in the module), `info` holds at least `{"xy": [N,2] ground truth, "goal_xy": [N,2],
"psi": [N], "mode": [N], "time": float}` for logging and `evalkit`. Auto-reset on
`done` (the returned obs is the post-reset observation) — this must be documented in
the module docstring and matched by `evaluate.py`.

### 4.6 Controller interface (policy, PID baseline, runtime)

```python
class Controller(Protocol):
    def reset(self) -> None: ...
    def act(self, est: Tensor[B, 6], goal: Tensor[B, 4],
            prev_action: Tensor[B, 3] | None = None) -> Tensor[B, 3]: ...
```

- `est`: `[B, 6]` state layout of §2 (estimator output only).
- `goal`: `[B, 4]` = `[x_goal_e, y_goal_n, psi_goal_rad, mode]`,
  `mode` 0 = waypoint following, 1 = station keeping (goal = hold point, target
  velocity 0, `psi_goal` = hold heading).
- Returns `[B, 3]` body command in `[-1, 1]`. The env applies the action-rate limit;
  a controller may also limit when `prev_action` is given.
- History (`K = 10`) is internal state of the policy controller: `act()` must be called
  once per policy step, and `reset()` clears the ring buffer.
- `baseline_pid.py` implements the same interface with the same `est`/`goal` semantics
  so `evaluate.py` can swap controllers without branching.

### 4.7 `models.py` (B2)

```python
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = 150, act_dim: int = 3,
                 hidden: tuple[int, ...] = (256, 128, 128),
                 activation: str = "elu", init_log_std: float = -0.5,
                 obs_mean: Tensor | None = None, obs_std: Tensor | None = None) -> None: ...
    def forward(self, obs: Tensor[B, 150]
                ) -> tuple[Tensor[B, 3], Tensor[B, 1], Tensor[B, 3]]: ...
    # returns (mean_action in [-1,1] via tanh, value, std) where std = exp(log_std)
```

Sampling for PPO: `a = clamp(mean + std · eps, -1, 1)`, `eps ~ N(0, I)`, state-independent
`log_std` (3 values). No tanh-Jacobian correction (documented simplification; the
clamp is a projection, not a squashing bijection). `PolicyNet.size_in = 150`.

### 4.8 `export_onnx.py` / `runtime/policy.py` (C3, A4)

ONNX graph:

| item | value |
|---|---|
| input | `obs`, `float32[B, 150]`, dynamic batch axis `B` |
| output | `action`, `float32[B, 3]`, deterministic mean action (tanh applied) |
| opset | 17 |
| normalization | `(obs − obs_mean) / obs_std` baked in as graph constants before the MLP |
| parity | max abs diff torch vs onnxruntime **< 1e-4** on 256 random obs in `[-3, 3]` |

Next to the `.onnx` file, `export_onnx.py` writes `policy_meta.json`:

```json
{
  "obs_dim": 150, "act_dim": 3, "k_history": 10,
  "obs_offset": [ ...15 floats... ], "obs_scale": [ ...15 floats... ],
  "obs_clip_norm": 5.0, "input_name": "obs", "output_name": "action",
  "policy_dt_s": 0.1,
  "safety": {"clip": 1.0, "rate_limit_per_s": 2.0, "geofence_m": 80.0,
             "gps_timeout_s": 2.0, "watchdog_s": 0.5}
}
```

`runtime/policy.py` (Pi 3B, numpy + onnxruntime only):

```python
class RLController:
    def __init__(self, onnx_path: str, meta_path: str | None = None,
                 fallback: Callable | None = None) -> None: ...
    def reset(self) -> None: ...
    def step(self, estimator_state: dict, goal: dict) -> tuple[float, float, float]: ...
```

`estimator_state` keys (Pi units, i.e. what `FusedState` can actually provide):

| key | unit | source today |
|---|---|---|
| `lat`, `lon` | deg | GPS |
| `heading_deg` | deg | fused mag+gyro |
| `yaw_rate_deg_s` | deg/s | gyro |
| `speed_mps` | m/s | GPS SOG |
| `course_deg` | deg | GPS COG |
| `gps_valid`, `imu_valid`, `mag_valid` | bool | flags |
| `t` | s | `time.monotonic()` |

`goal` keys: `lat`, `lon`, `heading_deg` (optional, `None` = no heading hold),
`mode` (`"waypoint"` | `"station_keeping"`). Returns `(surge, sway, yaw)` clipped to
±1. The controller keeps its own ENU origin (`lat0, lon0` = first valid fix) and its own
`K = 10` history; `u, v` come from
`u = speed·cos(course − heading)`, `v = speed·sin(course − heading)` gated on
`speed > 0.3 m/s`, else decayed — **TODO** (INSPECTION H7, owned by A4).

### 4.9 `evalkit/metrics.py` (B3)

```python
def radial_error(xy: np.ndarray[T, 2], goal_xy: np.ndarray[T, 2]) -> np.ndarray[T]
def mean_p95(err: np.ndarray) -> tuple[float, float]
def cross_track_rmse(xy: np.ndarray[T, 2], path: np.ndarray[P, 2]) -> float
def settling_time(t: np.ndarray[T], err: np.ndarray[T], tol: float,
                  hold_s: float = 5.0) -> float
def energy(t: np.ndarray[T], action: np.ndarray[T, 3]) -> float   # ∫ Σ|a|² dt, normalized units
def action_rate(t: np.ndarray[T], action: np.ndarray[T, 3]) -> float
def summarize(traj: dict) -> dict
```

`traj` dict keys: `t`, `xy`, `psi`, `action`, `goal_xy`, `mode`. `energy` is in
normalized-command units (dimensionless·s); if joule-equivalent numbers are wanted,
multiply per-motor thrust² by `max_thrust**2` — **TODO** decide in Wave 4 and document
in the results table.

## 5. Observation layout

One policy step produces a raw vector `s_t ∈ R^15` (index order fixed):

| idx | name | unit | notes |
|---|---|---|---|
| 0 | `goal_dx` | m | current goal in body frame, along bow |
| 1 | `goal_dy` | m | current goal in body frame, starboard |
| 2 | `sin(psi_err)` | — | `psi_err = wrap_pi(psi_goal − psi)` |
| 3 | `cos(psi_err)` | — | 1.0 when aligned |
| 4 | `u` | m/s | estimator |
| 5 | `v` | m/s | estimator |
| 6 | `r` | rad/s | estimator (gyro) |
| 7 | `a_prev_surge` | — | last applied action, zeros at reset |
| 8 | `a_prev_sway` | — | |
| 9 | `a_prev_yaw` | — | |
| 10 | `wp1_dx` | m | **next** waypoint after the current goal, body frame |
| 11 | `wp1_dy` | m | |
| 12 | `wp2_dx` | m | the waypoint after that, body frame |
| 13 | `wp2_dy` | m | |
| 14 | `mode` | — | 0.0 = waypoint, 1.0 = station keeping |

Rules that make this unambiguous:

- Normalization happens **per step, before** buffering:
  `s_norm = clip((s_raw − obs.offset) / obs.scale, −obs.clip_norm, +obs.clip_norm)`
  with `obs.scale` from `configs/env.yaml` (divisor, not multiplier).
- `K = 10` samples are concatenated **oldest first, newest last**:
  `obs = [s_{t−9}, …, s_{t−1}, s_t]`, `obs_dim = 15·10 = 150`. At reset all 10 slots
  are filled with the initial sample.
- `wp1`/`wp2` are the waypoints *after* the current goal; when there is none (last
  waypoint, or station keeping) the values are `0.0`. In station keeping
  `goal = hold point` and `psi_goal = hold heading`.
- In waypoint mode `psi_goal` is the bearing to the current goal (the navigator's
  desired heading), not the hull heading.
- No ground-truth state, no wind/current truth, no disturbance level may enter `obs`.
- Distances are **not** clipped in raw units; the normalized clip of ±5 covers ±50 m
  at the default 10 m scale.

## 6. numpy ↔ torch observation parity (binding)

`runtime/obs_np.py::build_obs(...)` (numpy, Pi) and the env's torch builder must
compute the same 15 values in the same order with the same formulas from
`common/frames.py`, and produce bit-comparable normalized outputs:

- same inputs (`est`, `goal`, `wp1`, `wp2`, `mode`, `prev_action`, history buffer),
- parity test (`tests/test_obs_parity.py`, B1/A4): random inputs, `max |obs_torch −
  obs_numpy| ≤ 1e-5` after casting both to `float32`, all 150 components,
- the numpy path computes in `float64` and casts **once** at the end; the torch path
  uses `float32` throughout. No different formulas (e.g. no `atan2` in one and
  `sin/cos` in the other for the same quantity).
- Frame math must go through the same helper module on both sides. The torch model
  cannot import numpy helpers per-sample, so `envs/` may reimplement them with torch
  ops — but the formulas in §1 are the only allowed source, and the parity test is the
  enforcement.

## 7. Behavioural requirements that later waves must not break

1. `config.h` mixing matrix and `MOTOR_INVERT` are firmware-owned. `sim/` consumes them
   from `configs/vehicle.yaml`; `tests/test_thrusters.py` must assert equality with the
   `config.h` literals (parsed from the file or hard-coded with a comment citing
   `config.h:85-105`).
2. Policy output ⇒ body command `[surge, sway, yaw]` in `[-1, 1]` and nothing else. No
   module downstream of the policy may change the mixing or send per-motor values
   directly; the ESP32 keeps mixing.
3. Sim policy rate 10 Hz, sim dt 0.02 s (5 sub-steps per action). The real loop is
   20 Hz (INSPECTION §3) — the 10 Hz policy rate is the design choice from
   `TASK.md:24`, not a measurement.
4. `est` is the only policy input; `state` is reward/metrics-only.
5. Nothing may claim sim-to-real validity. Metrics are reported as simulator numbers.

## 8. TODO register (open at Wave 0 close)

| id | item | owner |
|---|---|---|
| T1 | Real mag/IMU mounting rotation and whether heading is north-referenced (H1–H3) | SYSID / A1 note |
| T2 | Physical sign of sway and yaw as felt by the hull (H6) | bench test, A1 geometry |
| T3 | All `vehicle.yaml` physical parameters (mass, Izz, added mass, drag, thrust, positions, directions, lag) | SYSID.md |
| T4 | `u, v` estimator on the Pi from SOG/COG/heading (H7) | A4 |
| T5 | GPS NMEA output rate on the NEO-M8N | SYSID |
| T6 | Sim ENU origin (`lat0, lon0`) policy: fixed site vs random per episode | B1 |
| T7 | `energy` unit convention (normalized vs N²·s) | B3/C2 |
| T8 | Geofence radius and episode length tuning values | B1 |
| T9 | rsl_rl API compatibility; fallback to built-in PPO | B2 |
| T10 | torch installation repair (see INSPECTION §8) | orchestrator/human |
