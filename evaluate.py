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
--hold-s S         evalkit settling hold time (default 5.0 s).  NOT the corner dwell.
--dwell-stay-factor F  wp_square dwell hysteresis (default 1.5): the dwell counter starts
                   below the arrival radius, keeps counting while within F*arrival and resets
                   only beyond that.  F=1.0 reproduces the pre-R5 behaviour exactly.
--episode-s S      wp_square ONLY: episode length in s (default 200.0).  Other scenarios keep
                   their SCENARIOS length.  Both new flags are recorded in summary.json.

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
               "energy_j_proxy", "action_rate", "settled_pct",
               # R4: parking 3.4 m short of a corner looks fine to the radial columns
               "waypoints_reached", "lap_pct", "lap_completed",
               # R8: appended at the END, nothing above reordered (R8_D)
               "sat_frac", "reversal_rate", "arrival_speed_mean_mps",
               "hold_rms_m", "hold_p95_m"]

# The printed per-seed line carries the SAME numbers as the summary.csv columns, in the same
# order (tests/test_disturbance_seed.py checks the two agree with METRIC_COLS).
PRINT_COLS = list(METRIC_COLS)


def disturbance_seed_offset(seed: int, env_index: int = 0) -> int:
    """Disturbance-generator seed for one eval run: ``seed*1000 + 17``, one stream per seed.

    `AsvEnv` builds ONE `Disturbance` for the whole batch, so the wind/wave stream is shared by
    the envs of a rollout: `env_index` only distinguishes rollouts, it does not decorrelate the
    envs.  Without this the generator seed was the yaml constant, so `--seed` changed the spawn
    pose but NOT wind/waves (measured: max abs diff 0.0 between --seed 0 and --seed 1234).
    """
    return int(seed) * 1000 + 17


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

    Corner 0 = spawn, corners 1..4 = east, north, west, back to the spawn point.  Only the
    env's random goal/leg is replaced; the arrival test, the slowdown factor, the cross-track
    term (it reads `goal_xy`/`leg_start`) and `mode` stay the env's own.

    State machine per env (R5, reports/R5_E.md):
    * travelling (`idx` = leg, 0..3): goal = `c[idx+1]`, mode 0.
    * within `arrival` of `c[idx+1]`: station keeping at that corner, mode 1, dwell counter
      running.  Hysteresis: the counter starts below `arrival`, keeps counting below
      `dwell_stay * arrival`, resets only beyond that.
    * `hold_steps` of dwell done -> `leave`: at the LAST corner that is the lap finished
      (`done`, stays mode 1 on `c[4]`); at any other corner the counter restarts and mode
      returns to waypoint (0) toward `c[idx+1]`.
    Holding never uses `c[:, 1]`: before `leave` the goal is the ACTIVE corner `c[idx+1]`,
    after it `c[idx]`.  (R4 pinned `c[:, 1]` for every env after the first leave, so
    `waypoints_reached` could never exceed 1 — finding 4, reports/R5_E.md.)
    """

    # R4 metric wiring: the four legs of the lap, in the order SquarePath visits them
    # (corner 0 = spawn, then corners 1..4 = east, north, west, back south).
    CORNER_IDX = (1, 2, 3, 4)

    def __init__(self, env: AsvEnv, side: float, hold_s: float,
                 dwell_stay: float = 1.5) -> None:
        self.c = square_corners(env, side)                        # [N, 5, 2]
        self.hold_steps = max(1, int(round(hold_s / env.policy_dt)))
        z = lambda: torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # noqa: E731
        self.idx, self.hold = z(), z()
        self.done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.arrival = float(env.arrival)
        self.dwell_stay = float(dwell_stay)
        self.stay = self.arrival * self.dwell_stay
        # diagnostics of the last call (tests, reports)
        self.holding, self.goal_xy = self.done.clone(), self.c[:, 0].clone()

    def waypoints(self, i: int) -> torch.Tensor:
        """Planned waypoint list for env `i`, [4, 2] in visit order (R4 metrics)."""
        return self.c[i, list(self.CORNER_IDX)]

    def __call__(self, env: AsvEnv) -> torch.Tensor:
        c = self.c
        a = torch.arange(env.num_envs, device=c.device)
        pos = env.state[:, :2]
        nxt = (self.idx + 1).clamp(max=4)                         # corner being approached
        dist = (c[a, nxt] - pos).norm(dim=1)
        # dwell bookkeeping BEFORE the goal is written, so step k+1 sees the new leg
        live = ~self.done
        dwell = torch.where(dist < self.arrival, self.hold + 1,
                            torch.where((self.hold > 0) & (dist < self.stay), self.hold + 1,
                                        torch.zeros_like(self.hold)))
        self.hold = torch.where(live, dwell, self.hold)
        leave = live & (self.hold > self.hold_steps)
        self.hold = torch.where(leave, torch.zeros_like(self.hold), self.hold)
        last = leave & (nxt == c.shape[1] - 1)                    # dwell of the final corner
        self.idx = torch.where(leave & ~last, nxt, self.idx)      # next leg
        self.done = self.done | last

        holding = live & (self.hold > 0)                          # station keeping
        # travelling: the corner of leg `idx` is c[idx+1]; holding: the active corner
        goal_xy = c[a, torch.where(holding, nxt, (self.idx + 1).clamp(max=4))]
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
        self.holding, self.goal_xy = holding, goal_xy.clone()
        return torch.stack([goal_xy[:, 0], goal_xy[:, 1], env.psi_goal, env.mode], dim=1)


# --------------------------------------------------------------------------- conditions
def capture_conditions(env: AsvEnv, path) -> dict[str, torch.Tensor]:
    """Per-env condition snapshot of the CURRENT episode (call right after `env.reset()`).

    Reads the spawn pose/heading off `env.state`, the wind/current/wave draws off
    `env.disturbance`, and the domain-randomisation draws off `env.dynamics`,
    `env.thrusters` and `env.sensors` — all live per-env tensors written by the reset
    path, so this is the exact value the episode runs with.  Returns [N] / [N,k] torch
    tensors on `env.device` (the caller `.cpu()`s them).  `path` is the `SquarePath`
    (or None) so the "bearing to corner 1" column has something to point at.
    """
    n = env.num_envs
    d = env.disturbance
    thr = env.thrusters
    dyn = env.dynamics
    sen = env.sensors
    s = env.state
    # spawn pose + heading (state[:, :3])
    spawn_x, spawn_y, spawn_psi = s[:, 0], s[:, 1], s[:, 2]
    # bearing to corner 1 = planned square corner 1 (30 m east of spawn) when square;
    # otherwise the episode's current goal.  Heading error = wrap_pi(bearing - psi).
    if path is not None:
        c1 = path.c[:, 1]                       # [N, 2] corner 1
    else:
        c1 = env.goal_xy
    dxy = c1 - s[:, :2]
    bearing = torch.atan2(dxy[:, 0], dxy[:, 1])           # compass bearing (CONTRACT §1)
    head_err = torch.remainder(bearing - spawn_psi + math.pi, 2 * math.pi) - math.pi
    # wind / current / waves
    wind_speed, wind_dir = d.wind_speed, d.wind_dir         # rad compass, speed m/s
    cur_enu = d.current_enu                                # [2], e/n
    cur_speed = cur_enu.norm()                             # m/s (config value, not per-env)
    cur_dir = torch.atan2(cur_enu[0], cur_enu[1])          # compass rad
    wave_dir = d.wave_dir                                  # rad compass
    # physics randomisation (per-env tensors)
    mass = dyn.M_env[:, :2].mean(dim=1)                    # M(x,y) after mass_scale
    inertia = dyn.M_env[:, 2]                              # M(n) after inertia_scale
    drag = dyn.D_lin_env.mean(dim=1)                       # D_lin after drag_scale
    # scale factors vs nominal (dyn.M / dyn.D_lin are the nominal [3] vectors)
    mass_scale = mass / dyn.M[:2].mean()
    inertia_scale = inertia / dyn.M[2]
    drag_scale = drag / dyn.D_lin.mean()
    # thruster draws: scale_fwd/rev [N,4], lag [N,1], deadband [N,1]; weakened motor
    # is the one whose fwd scale dropped by weak_frac (or -1 if none)
    sf, sr = thr.scale_fwd, thr.scale_rev
    lag, db = thr.lag_tau[:, 0], thr.deadband[:, 0]
    # a "weakened" motor is scaled by weak_frac (env.yaml randomization.weak_factor, ~0.6),
    # far below the 1 +/- thruster_gain band a normal gain draw lands in.
    weak_thr = 0.5 * (1.0 + float(getattr(env, "weak_frac", 0.6))) - 0.02
    weak_mask = sf < (thr.nominal["scale_fwd"].max() * weak_thr)
    weak_motor = torch.where(weak_mask.any(dim=1),
                             weak_mask.float().argmax(dim=1).float(),
                             torch.full((n,), -1.0, device=env.device))
    # sensor draws
    out = dict(
        spawn_x=spawn_x, spawn_y=spawn_y, spawn_psi=spawn_psi,
        bearing_c1=bearing, heading_err=head_err,
        wind_speed=wind_speed, wind_dir=wind_dir,
        cur_speed=cur_speed.expand(n), cur_dir=cur_dir.expand(n), wave_dir=wave_dir,
        mass_scale=mass_scale, inertia_scale=inertia_scale, drag_scale=drag_scale,
        thr_fwd_mean=sf.mean(dim=1), thr_rev_mean=sr.mean(dim=1),
        thr_lag=lag, thr_deadband=db, weak_motor=weak_motor,
        gps_pos_sigma=sen.pos_sigma_env, gps_latency=sen.latency_env,
        mag_bias=sen.mag_bias_max_env, gyro_bias=sen.gyro_bias_max_env,
    )
    # per-motor forward scales, so "which thruster is weakened and by how much" is
    # visible (a weakened motor's scale = weak_frac, the others ~1 +/- thruster_gain)
    for j in range(sf.shape[1]):
        out[f"thr_fwd_{j}"] = sf[:, j]
    return out


# --------------------------------------------------------------------------- rollout
@torch.no_grad()
def run_rollout(kind: str, policy, device: torch.device, scen: dict, seed: int,
                episodes: int, num_envs: int, rough_scale: float,
                square_side: float = SQUARE_SIDE_M, hold_s: float = 10.0,
                dwell_stay: float = 1.5, dump: bool = False,
                env_index: int = 0) -> dict:
    """Simulate `episodes` episodes on `num_envs` envs; return the [T, N, ...] record.

    `env_index` (> 0) is set by the CLI to stamp `seed_offset = disturbance_seed_offset(seed,
    env_index)` into the disturbance config, so `--seed` changes wind/waves too; the default
    0 keeps this function's old behaviour for tests and callers that share a policy rollout.

    `dump=True` (--dump-conditions) additionally records, per step, the observation the
    policy saw (`obs` [T,N,150]), the raw policy action before the env's rate clamp
    (`action_raw` [T,N,3], the deterministic tanh mean), the policy log_std [3], the
    applied action (`action`, already recorded), and `mode` (already recorded); and, per
    episode, the condition snapshot of `capture_conditions` keyed by the episode's start
    step.  These ride in `rec["_dump"]` so the main path stays byte-compatible when the
    flag is off.
    """
    cfg = env_cfg(physics_randomize=not scen.get("pid_soft", False))
    cfg["num_envs"] = int(num_envs)
    cfg["episode_s"] = float(scen["episode_s"])
    cfg["max_steps"] = None                              # derived: episode_s * policy_hz
    dcfg = rough_disturbance(rough_scale) if scen["rough"] else None
    if env_index > 0:
        dcfg = dcfg if dcfg is not None else _load_yaml(ROOT / "configs" / "disturbance.yaml")
        dcfg["seed_offset"] = disturbance_seed_offset(seed, env_index)
    kw = {"disturbance_cfg": dcfg} if dcfg is not None else {}
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
    if dump:
        # extra per-step tracks (only allocated when --dump-conditions is on)
        rec["obs"] = []
        rec["action_raw"] = []      # pre-clamp tanh mean, before the env's rate limit
        cond_eps: list[dict] = []   # one dict per (episode, env) with captured conditions
    log_std = None                  # [3] policy log_std (constant per rollout), numpy
    t_wall = time.time()
    path = None
    for _ep in range(int(episodes)):
        obs = env.reset()                       # all envs: identical start for both controllers
        ctrl = make_controller(kind, policy, device, scen, env.policy_dt)
        path = (SquarePath(env, square_side, hold_s, dwell_stay)
                if scen["square"] else None)
        # capture the episode's conditions AFTER SquarePath, so the spawn pose, the
        # randomisation draws and the corner list all match what the episode runs with
        if dump:
            cond = {k: v.cpu().numpy() for k, v in capture_conditions(env, path).items()}
            for i in range(B):
                cond_eps.append({**{k: float(v[i]) for k, v in cond.items()},
                                 "episode": _ep, "env": i})
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
            if dump:
                rec["obs"].append(obs.clone())
                rec["action_raw"].append(act.detach().clone())   # tanh mean, pre rate-limit
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
        if dump and kind == "ppo":
            # state-independent log_std, one triple for the whole rollout
            log_std = policy.log_std.detach().cpu().numpy()
    out = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
    out["_meta"] = dict(wall_s=time.time() - t_wall, max_steps=T, num_envs=B,
                        policy_dt=env.policy_dt, level=env._level_value())
    out["_path"] = (path.c.cpu().numpy() if path is not None else None)  # 5 pts incl corner 0
    out["_waypoints"] = ([path.waypoints(i).cpu().numpy() for i in range(env.num_envs)]
                         if path is not None else None)
    out["_arrival"] = (path.arrival if path is not None else float(env.arrival))
    if dump:
        out["_conditions"] = cond_eps          # [episodes * num_envs] dicts
        out["_log_std"] = log_std              # numpy [3] or None (PID)
    env.close()
    return out


# --------------------------------------------------------------------------- scoring
def score(traj: dict, tol: float, hold_s: float, max_thrust: float, path=None,
          waypoints=None, arrival: float = 2.0) -> dict:
    """`evalkit.metrics.summarize` for ONE episode (already [T, ...], NaNs trimmed).

    `energy` stays in NORMALISED command units (CONTRACT §4.9 / evalkit default);
    `energy_j_proxy = 2 * max_thrust_fwd_n**2 * energy` is the documented joule PROXY,
    printed for scale only (reports/C2.md §4).
    """
    m = summarize(traj, tol=tol, hold_s=hold_s, waypoints=waypoints,
                  arrival_radius=arrival) if path is None else \
        summarize({**traj, "path": path}, tol=tol, hold_s=hold_s, waypoints=waypoints,
                  arrival_radius=arrival)
    m["settled_pct"] = 100.0 if m["settling_time_s"] < np.inf else 0.0
    m["energy_j_proxy"] = 2.0 * max_thrust ** 2 * m["energy"]
    m["lap_pct"] = 100.0 * float(m["waypoints_reached"]) / len(waypoints) \
        if waypoints is not None and len(waypoints) else float("nan")
    m["time_to_waypoint"] = m.pop("time_to_waypoint", None)   # R5: exported, not aggregated
    return m
TW_COLS = [f"time_to_waypoint_{k}" for k in range(1, 5)]   # R5: appended to the per-ep CSV only


def tw_columns(m: dict) -> dict:
    """`time_to_waypoint_1..4` (s, NaN when never reached) from a `score()` row."""
    tw = list(m.get("time_to_waypoint") or [])
    return {c: (float(tw[k - 1]) if k <= len(tw) else float("nan")) for k, c in
            enumerate(TW_COLS, start=1)}


def aggregate(per_ep: list[dict]) -> dict:
    """Mean over episodes of every scalar metric (`settling_time_s` keeps inf -> inf)."""
    agg: dict[str, float] = {}
    if per_ep:
        for key in per_ep[0]:
            if key == "time_to_waypoint":          # R5: list per episode, not aggregated
                continue
            vals = np.array([m[key] for m in per_ep], dtype=np.float64)
            agg[key] = float(np.mean(vals))           # bool -> fraction of laps (R4)
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
    ap.add_argument("--dwell-stay-factor", type=float, default=1.5,
                    help="wp_square dwell hysteresis: once the dwell counter started below the "
                         "arrival radius it keeps counting while within factor*arrival and only "
                         "resets beyond that (default 1.5; 1.0 = no hysteresis)")
    ap.add_argument("--episode-s", type=float, default=None,
                    help="wp_square ONLY: episode length in s (default 200.0 = SCENARIOS)")
    ap.add_argument("--dump-conditions", action="store_true",
                    help="write <scenario>_<ctrl>_conditions.csv (one row per seed/env with "
                         "spawn pose, heading error, wind/current/wave draws and every "
                         "randomised parameter) plus a per-seed .npz of the raw trajectories "
                         "and, for failing episodes, the first 200 steps of obs/action/log_std")
    ap.add_argument("--plot-env", type=int, default=None,
                    help="force the overlay plot to a specific env index")
    ap.add_argument("--plot-seed", type=int, default=None,
                    help="force the overlay plot to a specific seed index (0-based within a "
                         "scenario; requires --plot-env)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--tol", type=float, default=TOL_M)
    ap.add_argument("--skip-pid", action="store_true",
                    help="run only the ppo controller; skip the PID baseline (halves the wall "
                         "time and writes no *_pid rows/CSVs). PPO and PID roll the same "
                         "episode, so a PPO-only run is comparable to an earlier full run")
    ap.add_argument("--hold-s", type=float, default=HOLD_S,
                    help="evalkit SETTLING hold time (default 5.0 s); NOT the corner dwell "
                         "(that is --square-hold)")
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
    controllers = ("ppo",) if args.skip_pid else ("ppo", "pid")
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
    all_cond: dict[tuple[str, str], list[dict]] = {}   # R6: (scenario, ctrl) -> rows across seeds
    for si, name in enumerate(names):
        scen = dict(SCENARIOS[name])
        if name == "wp_square" and args.episode_s is not None:    # wp_square only
            scen["episode_s"] = float(args.episode_s)
        for k in range(args.seeds):
            seed = int(args.seed) + 1000 * si + k
            for kind in controllers:
                t0 = time.time()
                rec = run_rollout(kind, policy, device, scen, seed, args.episodes, args.envs,
                                  args.rough_scale, args.square_side, args.square_hold,
                                  args.dwell_stay_factor, dump=args.dump_conditions,
                                  env_index=si + 1)
                meta = rec.pop("_meta")
                path = rec.pop("_path", None)
                wps = rec.pop("_waypoints", None)          # R4: [N][4,2] or None
                arrival = rec.pop("_arrival", 2.0)
                cond = rec.pop("_conditions", None)        # R6: list of per-(ep,env) dicts
                log_std = rec.pop("_log_std", None)        # R6: [3] numpy or None
                ep = episode_arrays(rec, meta["max_steps"], args.episodes)
                if cond is not None:
                    # default outcomes for every (episode, env); the scoring loop overwrites
                    # the ones it can score.  An incomplete episode (geofence exit) keeps
                    # nan outcomes but always has its condition columns (no NaN there).
                    for c in cond:
                        c.setdefault("waypoints_reached", float("nan"))
                        c.setdefault("lap_completed", False)
                        c.setdefault("radial_error_max", float("nan"))
                per_ep, eps_rows = [], []
                for e in range(ep["xy"].shape[0]):
                    for i in range(ep["xy"].shape[1]):
                        tr = clean_episode(ep, e, i, meta["policy_dt"])
                        if tr is None:
                            continue
                        p = path[i] if path is not None else None
                        m = score(tr, args.tol, args.hold_s, max_thrust, path=p,
                                  waypoints=None if wps is None else wps[i], arrival=arrival)
                        per_ep.append(m)
                        eps_rows.append({"scenario": name, "controller": kind,
                                         "seed": seed, "episode": e, "env": i, **m,
                                         **tw_columns(m)})      # R5: appended columns only
                        if cond is not None:
                            # conditions captured at the start of this (episode, env): attach
                            # the episode's outcomes to the same row for the correlation dump
                            cj = cond[e * ep["xy"].shape[1] + i]
                            cj["waypoints_reached"] = m["waypoints_reached"]
                            cj["lap_completed"] = m["lap_completed"]
                            cj["radial_error_max"] = m["radial_error_max"]
                agg = aggregate(per_ep)
                rows.append({"scenario": name, "controller": kind, "seed": seed,
                             "seeds": args.seeds, "episodes": args.episodes,
                             "n_episodes": len(per_ep), "n_envs": meta["num_envs"],
                             "duration_s": meta["max_steps"] * meta["policy_dt"],
                             "wall_s": round(meta["wall_s"], 1),
                             "level": meta["level"],
                             # R8_D: the disturbance-generator seed this rollout actually used
                             "disturbance_seed":
                                 disturbance_seed_offset(seed, si + 1),
                             **agg})
                write_csv(out_dir / f"{name}_{kind}_{seed}_per_episode.csv", eps_rows,
                          ["scenario", "controller", "seed", "episode", "env"] + list(agg)
                          + TW_COLS)          # R5: appended at the END, nothing reordered
                if cond is not None:
                    # R6: --dump-conditions.  ONE <scenario>_<ctrl>_conditions.csv per
                    # (scenario, controller) covering EVERY seed (one row per seed x env), so
                    # the seed loop must accumulate instead of overwriting.  Rows were filled
                    # with outcomes above; incomplete episodes keep nan outcomes but always
                    # carry their condition columns.
                    all_cond.setdefault((name, kind), []).extend(
                        {"scenario": name, "controller": kind, "seed": seed,
                         "episode": c["episode"], "env": c["env"], **c} for c in cond)
                    # raw trajectories + failing-episode obs/action (per seed -> per-seed npz)
                    _write_dump_npz(out_dir, name, kind, seed, rec, ep, meta, path,
                                    args.dump_conditions, log_std)
                fin = [[r[c] for r in eps_rows if np.isfinite(r[c])] for c in TW_COLS]
                tw_mean = [float(np.mean(f)) if f else float("nan") for f in fin]
                print(f"  {name:12s} {kind} seed {seed:5d} n_ep {len(per_ep)}/"
                      f"{meta['num_envs']} radial {agg.get('radial_error_mean', float('nan')):6.2f} m"
                      f" p95 {agg.get('radial_error_p95', float('nan')):6.2f} m settled "
                      f"{agg.get('settled_pct', 0.0):5.1f}% wp "
                      f"{agg.get('waypoints_reached', float('nan')):4.2f} lap "
                      f"{agg.get('lap_pct', float('nan')):5.1f}% ({time.time() - t0:.0f} s)"
                      # R8: same numbers and order as PRINT_COLS/summary.csv
                      + " r8 " + " ".join(f"{c}={fmt(agg.get(c))}" for c in PRINT_COLS[13:])
                      + ("" if not tw_mean else " tw "
                         + str([None if np.isnan(t) else round(t, 1) for t in tw_mean])))
                if not args.no_plots:
                    plots(out_dir, name, kind, ep, path, args.tol, meta["policy_dt"],
                          seed_index=k, force_env=args.plot_env, force_seed=args.plot_seed)

    # ---- R6: one conditions CSV per (scenario, controller) covering every seed
    for (name, kind), cond_rows in all_cond.items():
        cond_cols = (["scenario", "controller", "seed", "episode", "env"]
                     + [k for k in cond_rows[0]
                        if k not in ("scenario", "controller", "seed", "episode", "env")])
        write_csv(out_dir / f"{name}_{kind}_conditions.csv", cond_rows, cond_cols)
        print(f"wrote {name}_{kind}_conditions.csv ({len(cond_rows)} rows)")

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
         "args": vars(args), "fast_env_parity": fast, "pooled": pooled, "per_seed": rows,
         # R5: effective wp_square parameters (args already carry the two flags)
         "square": {"episode_s": float(SCENARIOS["wp_square"]["episode_s"])
                    if args.episode_s is None else float(args.episode_s),
                    "dwell_stay_factor": float(args.dwell_stay_factor),
                    "square_hold_s": float(args.square_hold),
                    "square_side_m": float(args.square_side)},
         # R8_D: disturbance seeding — before this, `--seed` left wind/waves identical
         "disturbance_seeding": {
             "scheme": "seed_offset = seed*1000 + 17",
             "applies_to": "evaluate.py rollouts only (train.py unchanged)",
             "per_seed": [{"scenario": r["scenario"], "seed": r["seed"],
                           "seed_offset": r["disturbance_seed"]}
                          for r in rows if r["controller"] == "ppo"]}},
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


def _write_dump_npz(out_dir: Path, name: str, kind: str, seed: int, rec: dict,
                    ep: dict, meta: dict, path, dump: bool, log_std=None) -> None:
    """`<scenario>_<ctrl>_seed<S>.npz`: raw trajectories for replotting + the observation /
    action dump.  Trajectory arrays are [E, N, T, ...] (NaN padded); the observation stream
    `obs` [T, N, 150] and the pre-clamp `action_raw` [T, N, 3] are stored whole (the report
    slices the first 200 steps of the failing envs).  `_failing` lists (episode, env,
    complete) for every non-clean episode so a replot can locate the failures without
    rerunning the 15-minute eval.
    """
    if not dump:
        return
    dt = meta["policy_dt"]
    npz = {}
    for key in ("xy", "goal_xy", "psi", "action", "mode"):
        if key in ep:
            npz[key] = ep[key]
    npz["_path"] = path if path is not None else np.zeros((0, 2))
    fails = []
    for e in range(ep["xy"].shape[0]):
        for i in range(ep["xy"].shape[1]):
            complete = ep["complete"][e, i]
            if not complete or clean_episode(ep, e, i, dt) is None:
                fails.append((e, i, int(bool(complete))))
    npz["_failing"] = np.array(fails, dtype=np.int64).reshape(-1, 3)
    obs = rec.get("obs")
    araw = rec.get("action_raw")
    if obs is not None:
        npz["obs"] = obs                    # [T, N, 150], as the policy saw it
        npz["action_raw"] = araw            # [T, N, 3], pre-clamp tanh mean
        npz["log_std"] = (np.zeros(3, dtype=np.float32) if log_std is None
                          else np.asarray(log_std, dtype=np.float32))   # [3], constant
    np.savez_compressed(out_dir / f"{name}_{kind}_seed{seed}.npz", **npz)


def plots(out_dir: Path, name: str, kind: str, ep: dict, path, tol: float, dt: float,
          seed_index: int = 0, force_env: int | None = None,
          force_seed: int | None = None) -> None:
    """Per-seed overlay plots + best/median/worst tracks, ranked by `waypoints_reached`.

    Spec (R6): filenames carry the seed `<scenario>_<ctrl>_seed<S>_{best,median,worst}_
    {track,error}.png`, ranked by `waypoints_reached` (ties -> lower `radial_error_mean`);
    the old `<scenario>_<ctrl>_track.png`/`_error.png` are written for the LAST seed so
    existing tooling keeps working.  One overlay plot per seed colours every env's track by
    `waypoints_reached`, labels the env index at the start, and draws the planned square
    (5 points).  `force_env`/`force_seed` pin one episode and put it in the title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evalkit.metrics import waypoint_progress

    E, N = ep["xy"].shape[0], ep["xy"].shape[1]
    # collect one clean episode per env with its ranking key
    eps: list[dict] = []
    for e in range(E):
        for i in range(N):
            tr = clean_episode(ep, e, i, dt)
            if tr is None:
                continue
            p = path[i] if path is not None else None
            if p is not None:
                tr["path"] = p
            # waypoints_reached via the metric (ties -> radial_error_mean)
            wp = (waypoint_progress(tr["t"], tr["xy"], tr["goal_xy"], p)
                  if p is not None else {"waypoints_reached": float("nan"),
                                         "lap_completed": False})
            err = np.sqrt(((tr["xy"] - tr["goal_xy"]) ** 2).sum(-1))
            tr["_wp"] = int(wp["waypoints_reached"]) if np.isfinite(wp["waypoints_reached"]) \
                else -1                            # sk_wind: no plan -> not a waypoint episode
            tr["_err"] = float(err.mean())
            tr["_lap"] = bool(wp["lap_completed"])
            tr["_e"], tr["_i"] = e, i
            eps.append(tr)
    if not eps:
        return
    # overlay plot: one per seed, all envs coloured by waypoints_reached
    if force_env is not None and force_seed is not None and force_seed == seed_index:
        sel = [t for t in eps if t["_i"] == force_env]
    elif force_env is not None and force_seed is None:
        sel = [t for t in eps if t["_i"] == force_env]
    else:
        sel = eps
    if sel:
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.get_cmap("viridis")
        wpmax = max(t["_wp"] for t in sel) or 1
        for tr in sel:
            c = cmap(tr["_wp"] / wpmax)
            ax.plot(tr["xy"][:, 0], tr["xy"][:, 1], lw=0.9, color=c)
            ax.text(tr["xy"][0, 0], tr["xy"][0, 1], str(tr["_i"]), fontsize=7)
        if path is not None and len(sel):
            p = sel[0].get("path")
            if p is not None:
                ax.plot(p[:, 0], p[:, 1], "k--", lw=0.8, label="planned square")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("east [m]"); ax.set_ylabel("north [m]")
        ax.grid(alpha=0.3)
        title = f"seed {seed_index} env overlay ({name}/{kind})"
        if force_env is not None:
            t = sel[0] if sel else None
            title = f"seed {seed_index} env {force_env}, waypoints {t['_wp']}, " \
                    f"{'lap completed' if t and t['_lap'] else 'no lap'}"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_{kind}_seed{seed_index}_overlay.png", dpi=120)
        plt.close(fig)
    # best / median / worst, ranked by waypoints_reached (ties: lower radial_error_mean)
    ranked = sorted(eps, key=lambda t: (t["_wp"], -t["_err"]))
    order = {"best": ranked[-1], "median": ranked[len(ranked) // 2], "worst": ranked[0]}
    for tag, tr in order.items():
        from evalkit.plots import plot_error, plot_trajectory
        base = f"{name}_{kind}_seed{seed_index}_{tag}"
        fig = plot_trajectory(tr, out=str(out_dir / f"{base}_track.png"))
        plt.close(fig)
        fig = plot_error(tr, tol=tol, out=str(out_dir / f"{base}_error.png"))
        plt.close(fig)
    # legacy names: written on every call, so the LAST seed of the loop overwrites them —
    # exactly the pre-R6 behaviour existing tooling relies on (reports/C2.md filenames).
    from evalkit.plots import plot_error as _pe, plot_trajectory as _pt
    fig = _pt(ranked[-1], out=str(out_dir / f"{name}_{kind}_track.png"))
    plt.close(fig)
    fig = _pe(ranked[-1], tol=tol, out=str(out_dir / f"{name}_{kind}_error.png"))
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
