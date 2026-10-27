"""Wave 4 / C2 — PPO checkpoint vs PID baseline, identical seeds and identical episodes.

Report with all numbers, caveats and open issues: drl/reports/C2.md.
Metrics are `evalkit.metrics.summarize` (unchanged); plots are `evalkit.plots` (unchanged).
NOTHING here is valid for the real vessel: the physics is placeholder (SYSID.md).

Usage (run from drl/ with the venv python — `python` on PATH is 3.14 without torch):

    .venv/Scripts/python.exe evaluate.py                       # default sizes -> results/
    .venv/Scripts/python.exe evaluate.py --ckpt runs/checkpoints/ckpt_last.pt
    .venv/Scripts/python.exe evaluate.py --envs 8 --episodes 1 --scenarios sk_wind
    .venv/Scripts/python.exe evaluate.py --out results/smoke --envs 4 --episodes 1 --no-plots

CLI
---
--ckpt PATH        checkpoint dict {"model", "optimizer", "steps", "iteration", "cfg"};
                   default runs/checkpoints/ckpt_last.pt.  The policy is built with
                   `models.build_policy(ckpt["cfg"])` then `load_state_dict(ckpt["model"])`,
                   which restores trunk/heads/log_std AND the obs_mean/obs_std buffers, so the
                   normalisation is exactly training's (here both are 0/1 and the env's own
                   per-value normalisation of CONTRACT §5 is the only one).  Actions are the
                   deterministic tanh mean, no sampling.
--envs N           episodes simulated in parallel (default 16).  See "Speed": wall time is
                   ~constant per policy step regardless of N.
--episodes K       episodes per env (default 2).  Only the LAST one per env is measured;
                   earlier ones are warm-up that is discarded.  The controller is re-created
                   and the env re-reset between episodes (clean start, see below).
--seed S           base seed (default 0).  Scenario index s and controller c in {ppo, pid}
                   use seed = S + 1000*s + 10*c, so the two controllers of one scenario get
                   the SAME env seed: same spawn poses, waypoints, randomisation and
                   disturbance draws.
--scenarios LIST   comma list, default sk_wind,wp_square,rough_sea (definitions in
                   SCENARIOS and reports/C2.md §2; "sk_pid_soft" is an auxiliary probe).
--rough-scale F    rough_sea only: multiplies disturbance.yaml wind.max_speed_mps,
                   current.speed_mps, waves.amp_n, waves.yaw_amp_nm (default 2.0).  All four
                   are "TODO SYSID" placeholders; the scenario asks "what if they were F
                   times larger" — it is NOT a measurement.
--square-side M    wp_square side length (default 30 m; half-diagonal stays inside the 80 m
                   geofence).  Leg time follows the env's slowdown radius and the PID's own
                   0.55 m/s waypoint speed (baseline_pid.WAYPOINT_MAX_SPEED).
--device d         cuda|cpu (default cuda if available).
--out DIR          output dir (default results/).  Writes <scenario>_metrics.csv,
                   summary.csv, summary.png, <scenario>_track.png, <scenario>_error.png.
--no-plots         skip matplotlib.
--tol M            settled threshold for the "% settled" column (default 1.0 m): the GPS noise
                   floor (configs/sensors.yaml).  The env arrival radius is 2.0 m but that is
                   a WAYPOINT radius (env.yaml, quoted as measured).
--hold-s S         evalkit settling hold time (default 5.0 s).

Episode accounting (CONTRACT §4.5 — the env auto-resets inside step())
---------------------------------------------------------------------
`AsvEnv.step` resets finished envs before returning, so a rollout is a MIXTURE of episodes.
`split_episodes()` cuts it by `info["ep_step"]`, which counts 1..max_steps inside one episode
and restarts at 1 after the auto-reset, so every non-increment is a boundary.  `episode_arrays`
stacks those blocks into [E, N, T]; metrics are summarised per episode and averaged over
episodes, never over a mixed stream.  An episode that did not end on a timeout (geofence exit)
is excluded from the report and counted (`n_complete`).

Known artefact, NOT worked around (env-side, reported in reports/C2.md §6): the env re-draws
the spawn pose and goal *before* sampling the observation of the last step of an episode, so
that obs row carries the NEW goal distance while `info["xy"]` is the terminal one.  The PPO
sees that one garbage observation as the first input of the next episode.  Fixing it is an
env change (`envs/asv_env.py`), outside this file's ownership.

Speed (measured, reports/C2.md §5)
---------------------------------
The env is CPU-synchronous: `sim/sensors.py::step` calls `.item()` 12x per policy step (the
wave loop in `sim/disturbance.py::sample` another 8) and each one is a CUDA sync, so a step
costs ~60-85 ms at ANY batch size.  `FastEnv` overrides `step()` and removes only those `.item()`
calls: the wave frequencies/amplitudes are cached as Python floats (same float `.item()`
returned — 1-element tensors already inferred the same dtype) and `lat`/`lon` are not computed
because `AsvEnv` never reads them.  `check_fast_env()` asserts the obs/state stream equals
`AsvEnv`'s.  It is a harness speed-up, not a model change; it is also why the default is
K=2, N=16.  Delete `FastEnv` once A2 caches these constants inside `sim/`.

Clean start per episode
-----------------------
The PPO's K=10 history lives inside the 150-value observation the env keeps, so `env.reset()`
clears it.  `PIDBaseline` keeps integrators per env and never sees a reset, so sharing one
instance across episodes would leak an integrator across the auto-reset: a new controller
instance is created per episode.  Both controllers of a scenario therefore start every
episode from the identical env state.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_pid import THRUST_DEADZONE, PIDBaseline               # noqa: E402
from envs.asv_env import AsvEnv                                     # noqa: E402
from evalkit.metrics import summarize                               # noqa: E402
from sim.disturbance import Disturbance                             # noqa: E402
from models import build_policy                                     # noqa: E402

__all__ = ["SCENARIOS", "FastEnv", "check_fast_env", "split_episodes", "episode_arrays",
           "run_rollout", "score", "main"]

DEFAULT_CKPT = ROOT / "runs" / "checkpoints" / "ckpt_last.pt"
TOL_M = 1.0             # % settled threshold; GPS noise floor (configs/sensors.yaml)
HOLD_S = 5.0            # evalkit settling hold time
SQUARE_SIDE_M = 30.0    # 90 m legs; half-diagonal 21 m < 80 m geofence

# mode 1 = station keeping, 0 = waypoints (CONTRACT §4.6)
SCENARIOS: dict[str, dict] = {
    "sk_wind": dict(mode=1, episode_s=120.0, square=False, rough=False, pid_soft=False),
    "wp_square": dict(mode=0, episode_s=200.0, square=True, rough=False, pid_soft=False),
    "rough_sea": dict(mode=1, episode_s=120.0, square=False, rough=True, pid_soft=False),
    # auxiliary probe: the C1 coupling claim with the Pi-side 0.05 output deadband removed and
    # the physics randomisation off.  Not a fair PID setting, not headline.
    "sk_pid_soft": dict(mode=1, episode_s=120.0, square=False, rough=False, pid_soft=True),
}

SCENARIO_NOTES = {
    "sk_wind": "station keeping, 120 s, wind+current+waves at curriculum level 1.0",
    "wp_square": "one lap of a square (leg 1 east from the spawn point, then north, west, "
                 "south; corner held inside the 5 m slowdown radius), level 1.0",
    "rough_sea": "station keeping, 120 s, level 1.0 amplitudes x --rough-scale "
                 "(wind/current/wave TODO SYSID keys)",
    "sk_pid_soft": "AUX: station keeping, 120 s, level 1.0, physics randomisation OFF, "
                   "PID 0.05 output deadband OFF",
}

METRIC_COLS = ["radial_error_mean", "radial_error_p95", "radial_error_max",
               "cross_track_rmse", "settling_time_s", "settling_p95_s", "energy",
               "energy_j_proxy", "action_rate", "settled_pct"]


# --------------------------------------------------------------------------- config helpers
def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env_cfg(physics_randomize: bool = True) -> dict:
    """configs/env.yaml; physics draws optionally off (sensor draws stay = measurement realism)."""
    cfg = _load_yaml(ROOT / "configs" / "env.yaml")
    if not physics_randomize:
        for key in ("mass", "inertia", "drag", "thruster_gain", "thruster_lag",
                    "thruster_deadband", "fwd_rev_asymmetry", "weak_thruster_prob"):
            cfg["randomization"][key] = 0.0
    return cfg


def rough_disturbance(scale: float) -> dict:
    """configs/disturbance.yaml with the TODO SYSID amplitudes x scale.  Not a measurement."""
    cfg = _load_yaml(ROOT / "configs" / "disturbance.yaml")
    cfg["wind"]["max_speed_mps"] = float(cfg["wind"]["max_speed_mps"]) * scale
    cfg["current"]["speed_mps"] = float(cfg["current"]["speed_mps"]) * scale
    cfg["waves"]["amp_n"] = [float(x) * scale for x in cfg["waves"]["amp_n"]]
    cfg["waves"]["yaw_amp_nm"] = [float(x) * scale for x in cfg["waves"]["yaw_amp_nm"]]
    return cfg


# --------------------------------------------------------------------------- FastEnv
def _fast_disturbance_sample(d, t: float, dt: float, state: torch.Tensor) -> torch.Tensor:
    """`sim/disturbance.py::Disturbance.sample` with the per-step `.item()` calls removed.

    Same expressions, same order, except: the wave frequency/amplitude triples are Python
    floats cached on `d._fast_wave` (the value `.item()` returned on every step — the source
    tensors are 1-element float32), the level multiplies the wave sum once instead of per
    component, and the loop index is enumerated.  `check_fast_env()["dist"] == 0.0` shows the
    wrench is bit-identical to the original on random states and times.
    """
    N = state.shape[0]
    psi, u, v = state[:, 2], state[:, 3], state[:, 4]
    sp, cp = torch.sin(psi), torch.cos(psi)
    ve, vn = u * sp + v * cp, u * cp - v * sp
    z = torch.zeros(N, device=d.device, dtype=d.dtype)
    fe, fn, yaw = z, z, z
    lv = d.level
    if d.wind_on:
        we, wn = d.wind_speed * torch.sin(d.wind_dir), d.wind_speed * torch.cos(d.wind_dir)
        re, rn = we - ve, wn - vn
        k = (d.wind_coef * lv * lv) * torch.sqrt(re * re + rn * rn)
        fe, fn = fe + k * re, fn + k * rn
    if d.current_on:
        ce, cn = d.current_enu[0] * lv, d.current_enu[1] * lv
        cbe, cbn = ce * sp + cn * cp, ce * cp - cn * sp
        dd = d.current_drag * lv
        cur_bx, cur_by = dd * (cbe - u), dd * (cbn - v)
    else:
        cur_bx, cur_by = torch.zeros_like(z), torch.zeros_like(z)
    if d.waves_on:
        wfe, wfn, wyaw = z, z, z
        for i, (f, am, ya) in enumerate(d._fast_wave):
            ph = 2.0 * math.pi * f * float(t) + d.wave_phase[:, i]
            a = am * torch.sin(ph)
            wfe = wfe + a * torch.sin(d.wave_dir)
            wfn = wfn + a * torch.cos(d.wave_dir)
            wyaw = wyaw + ya * torch.sin(ph + d.wave_phase_yaw[:, i])
        fe, fn, yaw = fe + wfe * lv, fn + wfn * lv, wyaw * lv
    bx = fe * sp + fn * cp + cur_bx
    by = fe * cp - fn * sp + cur_by
    return torch.stack([bx, by, yaw], dim=1)



class FastEnv(AsvEnv):
    """`AsvEnv` with the per-step host synchronisations of `sim/` removed.

    Measured (reports/C2.md §5): `AsvEnv.step` costs ~80 ms at 4 envs AND at 512 envs, because
    `sim/disturbance.py::sample` does 8 `.item()` calls and `sim/sensors.py::step` 12 more plus
    a `gps_lat/gps_lon` CPU round trip per policy step — each a CUDA synchronisation of a
    ~50 us kernel.  Two object-level substitutions, everything else inherited unchanged:

    * `env.disturbance.sample` -> `_fast_disturbance_sample` (wave constants as Python floats).
    * `env.sensors.step` -> the original with the unused `gps_lat`/`gps_lon` entries dropped;
      `AsvEnv` never reads them (only `runtime/policy.py` would, and this harness uses `est`).

    `check_fast_env()` asserts the obs/state/reward stream is bit-identical to `AsvEnv`.
    `ponytail:` delete once A2 caches the constants and makes lat/lon lazy inside `sim/`.
    """

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        d = self.disturbance
        d._fast_wave = [(float(f), float(am), float(ya))
                        for f, am, ya in zip(d.wave_freq.tolist(), d.wave_amp.tolist(),
                                             d.wave_yaw_amp.tolist())]
        d.sample = _fast_disturbance_sample.__get__(d, type(d))

        sen = self.sensors
        orig = sen.step

        def step(state, dt):
            out = orig(state, dt)
            out.pop("gps_lat", None)
            out.pop("gps_lon", None)
            return out

        sen.step = step


def check_fast_env(n: int = 3, steps: int = 40, seed: int = 1234) -> dict:
    """Compare `AsvEnv` and `FastEnv` on the same seed and zero actions.

    Returns the max |difference| of obs / state / reward, the disturbance wrench parity, and
    the step time before/after.  Note `obs`/`state` compare two *independently constructed*
    envs: `sim/sensors.py` draws the per-episode gyro/mag bias and every noise sample from the
    global torch RNG (not an env generator), so the two streams differ by design and only the
    disturbance parity (`dist`) is an equality claim.
    """
    cfg = env_cfg()
    cfg["num_envs"] = n
    cfg["episode_s"] = steps * 0.1
    kw = dict(device="cuda" if torch.cuda.is_available() else "cpu", seed=seed)
    slow, fast = AsvEnv(copy.deepcopy(cfg), **kw), FastEnv(copy.deepcopy(cfg), **kw)
    # NOTE: set_curriculum takes the level INDEX into scale_by_level, not the scale itself.
    # Index 1 is scale 0.1; the top row (n_levels-1) is scale 1.0.
    slow.set_curriculum(slow.n_levels - 1)
    fast.set_curriculum(fast.n_levels - 1)
    act = torch.zeros(n, 3, device=slow.device)
    d_obs = d_st = d_rew = 0.0
    obs_s, obs_f = slow.reset(), fast.reset()
    t0 = time.time()
    for _ in range(steps):
        obs_s, r_s, _d, _i = slow.step(act)
        d_obs = max(d_obs, float((obs_s - obs_f).abs().max()))
        obs_f, r_f, _d, _i = fast.step(act)
        d_st = max(d_st, float((slow.state - fast.state).abs().max()))
        d_rew = max(d_rew, float((r_s - r_f).abs().max()))
    t_slow = time.time() - t0
    d = fast.disturbance
    obs_f = fast.reset()
    t0 = time.time()
    for _ in range(steps):
        obs_f, _r, _d, _i = fast.step(act)
    t_fast = time.time() - t0
    d_w = 0.0
    gen = torch.Generator(device=d.device).manual_seed(7)
    for _ in range(20):
        st = torch.randn(n, 6, device=d.device, generator=gen) * torch.tensor(
            [20.0, 20.0, 3.0, 0.5, 0.3, 0.2], device=d.device)
        st[:, 2] = torch.remainder(st[:, 2], 2 * math.pi) - math.pi
        tt = float(torch.rand(1, device=d.device, generator=gen)) * 50.0
        slow_s = _orig_disturbance_sample(d, tt, 0.02, st)
        d_w = max(d_w, float((slow_s - _fast_disturbance_sample(d, tt, 0.02, st)).abs().max()))
    slow.close()
    fast.close()
    return dict(obs=d_obs, state=d_st, reward=d_rew, dist=d_w,
                ms_slow=1000 * t_slow / steps, ms_fast=1000 * t_fast / steps)


_orig_disturbance_sample = Disturbance.sample


# --------------------------------------------------------------------------- episode split
def split_episodes(ep_step) -> list[np.ndarray]:
    """Episode row-index blocks of one rollout for ONE env (`info["ep_step"]`, [T]).

    `ep_step` counts 1..max_steps inside an episode and restarts at 1 after the auto-reset
    (zeroed inside `step()`, incremented before the sample), so every non-increment is a
    boundary.
    """
    s = np.asarray(ep_step).ravel()
    if s.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(s) != 1) + 1        # first index of every new episode
    return np.split(np.arange(s.size), breaks)


def episode_arrays(rec: dict, max_steps: int, measured: int | None = None) -> dict:
    """[T, N, ...] rollout -> [E, N, T_ep, ...]; episodes right-aligned per env.

    Samples past a short block stay NaN.  `complete[e, i]` = env i's episode e ended on a
    timeout (a geofence exit or a cut block is incomplete).  `measured=None` keeps every
    episode; the caller that knows the run length passes `episodes`.
    """
    T, N = rec["ep_step"].shape
    blocks = [split_episodes(rec["ep_step"][:, i]) for i in range(N)]
    E = min(len(b) for b in blocks) if measured is None else int(measured)
    out: dict[str, np.ndarray] = {"complete": np.zeros((E, N), dtype=bool)}
    for key, val in rec.items():
        if key == "ep_step":
            continue
        vals = np.asarray(val)
        acc = np.full((E, N, max_steps) + vals.shape[2:], np.nan, dtype=np.float64)
        for i, bl in enumerate(blocks):
            last = bl[-E:]
            for j, ids in enumerate(last):
                ids = ids[:max_steps]
                acc[E - len(last) + j, i, :ids.size] = vals[ids, i]
        out[key] = acc
    # right-aligned: the last E blocks of every env sit at indices E-len(bl) .. E-1
    to, gf = np.asarray(rec["timeout"]), np.asarray(rec["geofence"])
    for i, bl in enumerate(blocks):
        last = bl[-E:]
        for j, ids in enumerate(last):
            e = E - len(last) + j
            out["complete"][e, i] = bool(
                ids.size == max_steps and to[ids[-1], i] and not gf[ids[-1], i])
    return out


# --------------------------------------------------------------------------- controllers
class PPOController:
    """CONTRACT §4.6 controller on a frozen checkpoint (deterministic tanh mean).

    The observation — K=10 history included — is environment state, not controller state: the
    rollout assigns `ctrl.obs = obs` once per policy step, exactly like `runtime/policy.py`
    appends its own sample.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 dtype: torch.dtype = torch.float32) -> None:
        self.model = model.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype
        self.obs: torch.Tensor | None = None

    def reset(self) -> None:
        self.obs = None

    @torch.no_grad()
    def act(self, est: torch.Tensor, goal: torch.Tensor,
            prev_action: torch.Tensor | None = None) -> torch.Tensor:
        if self.obs is None:
            raise RuntimeError("PPOController.act called before the first observation")
        mean, _value, _std = self.model(self.obs)
        return mean


def load_policy(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_policy(ck["cfg"])
    model.load_state_dict(ck["model"])          # restores obs_mean/obs_std buffers too
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def make_controller(kind: str, policy, device: torch.device, scen: dict, dt: float):
    if kind == "ppo":
        return PPOController(policy, device)
    dead = 0.0 if scen.get("pid_soft") else THRUST_DEADZONE
    return PIDBaseline(dt=float(dt), device=str(device), output_deadband=dead)


# --------------------------------------------------------------------------- square scenario
def square_corners(env: AsvEnv, side: float) -> torch.Tensor:
    """Planned square [N, 5, 2] per env: spawn point = corner 0, then +east, +north, -east."""
    off = torch.tensor([[0.0, 0.0], [side, 0.0], [side, side], [0.0, side], [0.0, 0.0]],
                       device=env.device, dtype=env.dtype)
    return env.state[:, :2].unsqueeze(1) + off.unsqueeze(0)


class SquarePath:
    """The 4-leg square as an `env.goal_xy` writer, one instance per rollout.

    Only the env's random goal/leg is replaced.  Everything else stays the env's own: the
    arrival test, the per-leg slowdown factor in `_reward`, the cross-track term (it reads
    `goal_xy`/`leg_start`) and `mode` — waypoint while travelling, station keeping (hold point
    + the heading at arrival) for `hold_steps` steps once inside `arrival`.  The dwell timer
    is what makes the hold stick: a pure radius test drops out of the hold on the next
    overshoot step (first version did, reports/C2.md §4).
    """

    def __init__(self, env: AsvEnv, side: float, hold_s: float) -> None:
        self.c = square_corners(env, side)                        # [N, 5, 2]
        self.hold_steps = max(1, int(round(hold_s / env.policy_dt)))
        z = lambda: torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # noqa: E731
        self.idx, self.hold, self.done = z(), z(), torch.zeros(env.num_envs, dtype=torch.bool,
                                                               device=env.device)

    def __call__(self, env: AsvEnv) -> torch.Tensor:
        c = self.c
        a = torch.arange(env.num_envs, device=c.device)
        pos = env.state[:, :2]
        inside = (c[a, (self.idx + 1).clamp(max=4)] - pos).norm(dim=1) < env.arrival
        # dwell bookkeeping (before the goal is written, so step k+1 sees the new leg)
        self.hold = torch.where(inside & ~self.done, self.hold + 1, torch.zeros_like(self.hold))
        leave = (self.hold > self.hold_steps) & ~self.done
        self.idx = torch.where(leave, (self.idx + 1).clamp(max=4), self.idx)
        self.done = self.done | leave
        self.hold = torch.where(leave, torch.zeros_like(self.hold), self.hold)

        holding = self.done | (self.hold > 0)                     # station keeping
        goal_xy = torch.where(holding.unsqueeze(1), c[:, 1], c[a, (self.idx + 1).clamp(max=4)])
        leg_start = c[a, self.idx.clamp(max=3)]
        d = goal_xy - pos
        bearing = torch.atan2(d[:, 0], d[:, 1])
        env.hold_psi.copy_(torch.where(holding & (self.hold <= 1), env.state[:, 2],
                                       env.hold_psi))
        env.goal_xy.copy_(goal_xy)
        env.leg_start.copy_(torch.where(holding.unsqueeze(1), goal_xy, leg_start))
        env.mode.copy_(holding.to(env.dtype))                      # 1 = station keeping
        env.hold_xy.copy_(goal_xy)
        env.psi_goal.copy_(torch.where(holding, env.hold_psi, bearing))
        return torch.stack([goal_xy[:, 0], goal_xy[:, 1], env.psi_goal, env.mode], dim=1)


# --------------------------------------------------------------------------- rollout
@torch.no_grad()
def run_rollout(kind: str, policy, device: torch.device, scen: dict, seed: int,
                episodes: int, num_envs: int, rough_scale: float,
                square_side: float = SQUARE_SIDE_M, hold_s: float = 10.0) -> dict:
    """Simulate `episodes` episodes on `num_envs` envs; return the [T, N, ...] record."""
    cfg = env_cfg(physics_randomize=not scen.get("pid_soft", False))
    cfg["num_envs"] = int(num_envs)
    cfg["episode_s"] = float(scen["episode_s"])
    cfg["max_steps"] = None                              # derived: episode_s * policy_hz
    kw = {"disturbance_cfg": rough_disturbance(rough_scale)} if scen["rough"] else {}
    env = FastEnv(cfg, device=str(device), seed=int(seed), **kw)
    assert env.max_steps == int(round(scen["episode_s"] * env.policy_hz))
    # pin the last row of disturbance.curriculum.scale_by_level (1.0).  NOT via
    # set_curriculum(1.0): that takes the level INDEX, and index 1 is scale 0.1.
    env.set_curriculum(env.n_levels - 1)
    assert env._level_value() == 1.0, f"curriculum scale is {env._level_value()}, not 1.0"
    B, T = env.num_envs, env.max_steps
    rec: dict[str, list[torch.Tensor]] = {k: [] for k in
                                          ("xy", "goal_xy", "psi", "action", "ep_step",
                                           "timeout", "geofence", "mode", "reward")}
    t_wall = time.time()
    path = None
    for _ep in range(int(episodes)):
        obs = env.reset()                       # all envs: identical start for both controllers
        ctrl = make_controller(kind, policy, device, scen, env.policy_dt)
        path = SquarePath(env, square_side, hold_s) if scen["square"] else None
        for _k in range(T):
            est = env.meas["est"]
            if path is not None:
                goal = path(env)
            else:
                gpsi = torch.where(env.mode > 0.5, env.hold_psi, env.psi_goal)
                goal = torch.stack([env.goal_xy[:, 0], env.goal_xy[:, 1], gpsi, env.mode],
                                   dim=1)
            if kind == "ppo":
                ctrl.obs = obs                    # type: ignore[attr-defined]
            act = ctrl.act(est, goal, env.prev_action)
            obs, rew, _done, info = env.step(act)
            rec["xy"].append(info["xy"])
            rec["goal_xy"].append(info["goal_xy"])
            rec["psi"].append(info["psi"])
            rec["action"].append(env.prev_action.clone())   # last APPLIED action (§4.5)
            rec["ep_step"].append(info["ep_step"])
            rec["timeout"].append(info["timeout"])
            rec["geofence"].append(info["geofence"])
            rec["mode"].append(info["mode"])
            rec["reward"].append(rew)
    out = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
    out["_meta"] = dict(wall_s=time.time() - t_wall, max_steps=T, num_envs=B,
                        policy_dt=env.policy_dt, level=env._level_value())
    out["_path"] = (path.c[:, 1:, :].cpu().numpy() if path is not None else None)
    env.close()
    return out


# --------------------------------------------------------------------------- scoring
def score(traj: dict, tol: float, hold_s: float, max_thrust: float, path=None) -> dict:
    """`evalkit.metrics.summarize` for ONE episode (already [T, ...], NaNs trimmed).

    `energy` stays in NORMALISED command units (CONTRACT §4.9 / evalkit default);
    `energy_j_proxy = 2 * max_thrust_fwd_n**2 * energy` is the documented joule PROXY,
    printed for scale only (reports/C2.md §4).
    """
    m = summarize(traj, tol=tol, hold_s=hold_s) if path is None else \
        summarize({**traj, "path": path}, tol=tol, hold_s=hold_s)
    m["settled_pct"] = 100.0 if m["settling_time_s"] < np.inf else 0.0
    m["energy_j_proxy"] = 2.0 * max_thrust ** 2 * m["energy"]
    return m


def aggregate(per_ep: list[dict]) -> dict:
    """Mean over episodes of every scalar metric (`settling_time_s` keeps inf -> inf)."""
    agg: dict[str, float] = {}
    if per_ep:
        for key in per_ep[0]:
            vals = np.array([m[key] for m in per_ep], dtype=np.float64)
            agg[key] = float(np.mean(vals))
            if key == "settling_time_s":
                agg["settling_p95_s"] = float(np.percentile(vals, 95))
        agg["n_episodes"] = len(per_ep)
    return agg


def clean_episode(ep: dict, e: int, i: int, dt: float) -> dict | None:
    """One [T, ...] episode slice with the NaN padding trimmed, or None if incomplete."""
    if not ep["complete"][e, i]:
        return None
    good = np.isfinite(ep["xy"][e, i, :, 0])
    return {"t": np.arange(good.size)[good] * dt,
            "xy": ep["xy"][e, i][good], "goal_xy": ep["goal_xy"][e, i][good],
            "psi": ep["psi"][e, i][good], "action": ep["action"][e, i][good],
            "mode": ep["mode"][e, i][good]}


# --------------------------------------------------------------------------- output
def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c))
                        for c in cols})


def fmt(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "inf" if not np.isfinite(v) else f"{float(v):.3g}"
    return str(v)


def markdown(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    return head + "\n".join("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows)


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PPO vs PID on identical seeds (C2)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=4,
                    help="seeds per scenario; both controllers run each seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", default="sk_wind,wp_square,rough_sea")
    ap.add_argument("--rough-scale", type=float, default=2.0)
    ap.add_argument("--square-side", type=float, default=SQUARE_SIDE_M)
    ap.add_argument("--square-hold", type=float, default=10.0,
                    help="seconds of station keeping at each corner")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--tol", type=float, default=TOL_M)
    ap.add_argument("--hold-s", type=float, default=HOLD_S)
    args = ap.parse_args(argv)
    if args.episodes != 1:
        # the measured episode is the first of a seed; only K=1 makes both controllers score
        # the same episode, and extra episodes only differ through the global-RNG sensors
        print("--episodes forced to 1 (see the module docstring)")
        args.episodes = 1
    torch.set_num_threads(1)        # the env is launch-latency bound: threads add sync cost

    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in names if s not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s) {unknown}; choose from {sorted(SCENARIOS)}")
        return 2
    t_start = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if not str(args.device).startswith("cuda")
                          or torch.cuda.is_available() else "cpu")

    fast = check_fast_env(n=3, steps=40)
    print(f"FastEnv parity vs AsvEnv: obs {fast['obs']:.3g} state {fast['state']:.3g} "
          f"reward {fast['reward']:.3g} dist {fast['dist']:.3g} | step "
          f"{fast['ms_slow']:.1f} ms -> {fast['ms_fast']:.1f} ms "
          f"(3 envs, {device.type})")

    ckpt_path = Path(args.ckpt)
    policy, ck = load_policy(ckpt_path, device)
    max_thrust = float(_load_yaml(ROOT / "configs" / "vehicle.yaml")["thrusters"]
                       ["max_thrust_fwd_n"])
    print(f"checkpoint {ckpt_path} steps={ck.get('steps')} iteration={ck.get('iteration')} "
          f"obs_mean|max|={float(ck['model']['obs_mean'].abs().max()):.3g} "
          f"obs_std=[{float(ck['model']['obs_std'].min()):.3g},"
          f"{float(ck['model']['obs_std'].max()):.3g}]")

    # ---- one row per (scenario, seed, controller); the comparable unit is one episode at
    # one seed.  Several seeds: a 4-env sample is a smoke number, and C1 showed a handful of
    # PID episodes DO converge, so a single batch would mis-report the baseline.
    rows = []
    for si, name in enumerate(names):
        scen = SCENARIOS[name]
        for k in range(args.seeds):
            seed = int(args.seed) + 1000 * si + k
            for kind in ("ppo", "pid"):
                t0 = time.time()
                rec = run_rollout(kind, policy, device, scen, seed, args.episodes, args.envs,
                                  args.rough_scale, args.square_side, args.square_hold)
                meta = rec.pop("_meta")
                path = rec.pop("_path", None)
                ep = episode_arrays(rec, meta["max_steps"], args.episodes)
                per_ep, eps_rows = [], []
                for e in range(ep["xy"].shape[0]):
                    for i in range(ep["xy"].shape[1]):
                        tr = clean_episode(ep, e, i, meta["policy_dt"])
                        if tr is None:
                            continue
                        p = path[i] if path is not None else None
                        m = score(tr, args.tol, args.hold_s, max_thrust, path=p)
                        per_ep.append(m)
                        eps_rows.append({"scenario": name, "controller": kind,
                                         "seed": seed, "episode": e, "env": i, **m})
                agg = aggregate(per_ep)
                rows.append({"scenario": name, "controller": kind, "seed": seed,
                             "seeds": args.seeds, "episodes": args.episodes,
                             "n_episodes": len(per_ep), "n_envs": meta["num_envs"],
                             "duration_s": meta["max_steps"] * meta["policy_dt"],
                             "wall_s": round(meta["wall_s"], 1),
                             "level": meta["level"], **agg})
                write_csv(out_dir / f"{name}_{kind}_{seed}_per_episode.csv", eps_rows,
                          ["scenario", "controller", "seed", "episode", "env"] + list(agg))
                print(f"  {name:12s} {kind} seed {seed:5d} n_ep {len(per_ep)}/"
                      f"{meta['num_envs']} radial {agg.get('radial_error_mean', float('nan')):6.2f} m"
                      f" p95 {agg.get('radial_error_p95', float('nan')):6.2f} m settled "
                      f"{agg.get('settled_pct', 0.0):5.1f}% ({time.time() - t0:.0f} s)")
                if not args.no_plots:
                    plots(out_dir, name, kind, ep, path, args.tol, meta["policy_dt"])

    # ---- pooled over seeds: mean of the per-seed episode means + spread
    pooled = []
    for name in names:
        for kind in ("ppo", "pid"):
            sel = [r for r in rows if r["scenario"] == name and r["controller"] == kind]
            if not sel:
                continue
            row = {c: sel[0][c] for c in ("scenario", "controller", "seeds", "episodes",
                                          "n_envs", "duration_s", "level")}
            row["n_episodes"] = int(sum(r["n_episodes"] for r in sel))
            for c in METRIC_COLS:
                vals = np.array([r.get(c, np.nan) for r in sel], dtype=np.float64)
                n_fin = int(np.isfinite(vals).sum())
                row[c] = float(np.nanmean(vals)) if n_fin else float("nan")
                row[c + "_sd"] = float(np.nanstd(vals)) if n_fin else float("nan")
            pooled.append(row)

    cols = ["scenario", "controller", "seeds", "episodes", "n_episodes", "n_envs",
            "duration_s", "level"] + METRIC_COLS
    write_csv(out_dir / "per_seed.csv", rows, ["scenario", "controller", "seed"] + cols[2:])
    write_csv(out_dir / "summary.csv", pooled,
              cols + [c + "_sd" for c in METRIC_COLS])
    (out_dir / "summary.json").write_text(json.dumps(
        {"ckpt": str(ckpt_path), "steps": ck.get("steps"), "iteration": ck.get("iteration"),
         "args": vars(args), "fast_env_parity": fast, "pooled": pooled, "per_seed": rows},
        indent=1, default=str), encoding="utf-8")
    table = markdown(pooled, cols)
    print("\n" + table)
    if not args.no_plots:
        from evalkit.plots import summary_table
        flat = {}
        for r in pooled:
            flat[f"{r['scenario']}/{r['controller']}"] = {k: r.get(k, np.nan)
                                                          for k in METRIC_COLS}
        summary_table(flat, out=str(out_dir / "summary.png"),
                      title="PPO vs PID (sim, placeholder physics)")
    (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir} ({time.time() - t_start:.0f} s total, device {device})")
    return 0


def plots(out_dir: Path, name: str, kind: str, ep: dict, path, tol: float, dt: float) -> None:
    """Trajectory + error plot of the FIRST complete episode (evalkit.plots)."""
    from evalkit.plots import plot_error, plot_trajectory
    tr = None
    for e in range(ep["xy"].shape[0]):
        for i in range(ep["xy"].shape[1]):
            tr = clean_episode(ep, e, i, dt)
            if tr is not None:
                p = path[i] if path is not None else None
                if p is not None:
                    tr["path"] = p
                break
        if tr is not None:
            break
    if tr is None:
        return
    plot_trajectory(tr, out=str(out_dir / f"{name}_{kind}_track.png"))
    plot_error(tr, tol=tol, out=str(out_dir / f"{name}_{kind}_error.png"))


if __name__ == "__main__":
    raise SystemExit(main())
