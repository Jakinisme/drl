"""Wave 3 integration smoke test — the whole stack on the real env (CONTRACT §4.5/§4.6/§4.9).

Two things this file pins so they cannot regress (reports/C1.md):

1. **Estimator error at the first step of an episode is small.** R3 seeded the sensor
   model with the episode's true pose, so `est[:2]` after `reset()` differs from the truth
   by the GPS noise floor (~0.5 m), not by the spawn distance (~20 m). Before R3 the GPS
   latency ring was zero-filled and the policy saw a 38 m phantom error at t=0.
2. **The effective drag factor of 1000 sampled episodes stays inside the `configs/env.yaml`
   range** (`randomization.drag` 0.30 -> [0.70, 1.30] on `drag_linear` and
   `drag_quadratic`, decoupled from mass since R3).

Run `python tests/test_integration.py` for the full-length PID baseline measurement
(1200 policy steps = one 120 s episode, station keeping and waypoint, `evalkit.metrics`).
The pytest entry points are the short, fast versions.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # allow direct script use

from baseline_pid import PIDBaseline                          # noqa: E402
from envs.asv_env import AsvEnv, load_config                   # noqa: E402
from evalkit.metrics import (                                  # noqa: E402
    action_rate, cross_track_rmse, energy, mean_p95, radial_error, settling_time,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = 16                     # small batch: the env is overhead-bound below ~256 envs
RANDOM_KEYS = ("mass", "inertia", "drag", "thruster_gain", "thruster_lag",
               "thruster_deadband", "fwd_rev_asymmetry", "weak_thruster_prob",
               "sensor_noise", "sensor_latency_s")


def _cfg(randomize: bool = False, **over):
    """`env.yaml` copy; `randomize=False` zeroes every range (repeatable physics)."""
    cfg = copy.deepcopy(load_config())
    if not randomize:
        for key in RANDOM_KEYS:
            cfg["randomization"][key] = 0.0
    for path, val in over.items():
        node, keys = cfg, path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = val
    return cfg


def _env(num_envs: int = N, randomize: bool = False, seed: int = 0, **over) -> AsvEnv:
    env = AsvEnv(cfg=_cfg(randomize, **over), num_envs=num_envs, device=DEV, seed=seed)
    env.set_curriculum(0)          # level 0 = exact zero disturbance, auto-advance pinned off
    return env


def _goal(env: AsvEnv) -> torch.Tensor:
    return torch.stack([env.goal_xy[:, 0], env.goal_xy[:, 1], env.psi_goal, env.mode], dim=1)


def run_pid(mode: str, steps: int = 1200, num_envs: int = N, seed: int = 0) -> dict:
    """Drive the env with `baseline_pid.PIDBaseline`; return a `summarize`-shaped traj.

    `mode` "sk" = station keeping only, "wp" = waypoint only (overrides
    `episode.station_keeping_fraction` for that run). Physics randomisation is off so the
    numbers are repeatable; the controller is the faithful default (0.05 output deadband).
    """
    frac = 1.0 if mode == "sk" else 0.0
    env = _env(num_envs=num_envs, seed=seed, **{"episode.station_keeping_fraction": frac})
    pid = PIDBaseline(dt=env.policy_dt, device=DEV)
    env.reset()
    pid.reset()

    xy, goal, act, t = [], [], [], []
    for _ in range(steps):
        a = pid.act(env.meas["est"], _goal(env))
        env.step(a)
        xy.append(env.state[:, :2].clone())
        goal.append(env.goal_xy.clone())
        act.append(a.clone())
        t.append(env.t_loop)

    return {
        "t": np.asarray(t, dtype=np.float64),
        "xy": torch.stack(xy).cpu().numpy(),           # [T, N, 2]
        "goal_xy": torch.stack(goal).cpu().numpy(),    # [T, N, 2]
        "action": torch.stack(act).cpu().numpy(),      # [T, N, 3]
        "mode": mode,
    }


def summarize_pid(traj: dict) -> dict:
    """`evalkit.metrics` on every env, then averaged (one number per metric).

    `settling_s` is `inf` unless the env settles, so it is reported as the fraction of
    envs that settled plus the mean settling time of those (both in `settled_frac`).
    """
    t, xy, goal, act = traj["t"], traj["xy"], traj["goal_xy"], traj["action"]
    n = xy.shape[1]
    err = radial_error(xy, goal)                       # [T, N]
    # SK: the path degenerates to the single hold point (distance-to-point RMSE);
    # WP: the polyline of that env's goals, so the leg offset is the real cross-track error
    sk = traj["mode"] == "sk"
    path = [goal[0, i].reshape(1, 2) if sk else goal[:, i] for i in range(n)]
    m, p95 = mean_p95(err)
    xt = [cross_track_rmse(xy[:, i], path[i]) for i in range(n)]
    st = [settling_time(t, err[:, i], tol=2.0, hold_s=5.0) for i in range(n)]
    settled = [s for s in st if np.isfinite(s)]
    return {
        "radial_mean_m": m,
        "radial_p95_m": p95,
        "xtrack_rmse_m": float(np.mean(xt)),
        "settled_frac": len(settled) / n,
        "settling_s": float(np.mean(settled)) if settled else float("inf"),
        "energy": float(np.mean([energy(t, act[:, i]) for i in range(n)])),
        "action_rate_1_s": float(np.mean([action_rate(t, act[:, i]) for i in range(n)])),
        "err_start_m": float(err[:10].mean()),         # first 1 s, all envs
        "err_end_m": float(err[-60:].mean()),          # last 6 s, all envs
    }


# --------------------------------------------------------------------- (1) estimator
def test_estimator_error_small_at_first_step():
    """Reset + first policy step: est vs truth = GPS noise floor, not the spawn distance."""
    env = _env(num_envs=256, randomize=True, seed=0)   # R3 path: seeded with the true pose
    env.reset()
    err0 = (env.meas["est"][:, :2] - env.state[:, :2]).norm(dim=1)
    spawn = env.state[:, :2].norm(dim=1)
    assert err0.max().item() < 5.0, err0.max().item()          # R3 measured 1.53 m max
    assert err0.mean().item() < 1.5, err0.mean().item()
    assert err0.max().item() < 0.25 * spawn.mean().item()      # far below the spawn radius

    env.step(torch.zeros(env.num_envs, 3, device=DEV))         # step 1
    err1 = (env.meas["est"][:, :2] - env.state[:, :2]).norm(dim=1)
    assert err1.max().item() < 5.0, err1.max().item()
    assert err1.max().item() < 0.25 * spawn.mean().item()


# ------------------------------------------------------------------------ (2) drag
def test_drag_factor_of_1000_sampled_episodes_inside_config_range():
    """1000 sampled episodes: D_lin_env/D_lin and D_quad_env/D_quad in [0.70, 1.30]."""
    cfg = load_config()
    r_drag = float(cfg["randomization"]["drag"])
    assert abs(r_drag - 0.30) < 1e-9
    lo, hi = 1.0 - r_drag, 1.0 + r_drag                   # 0.70 .. 1.30 from env.yaml

    env = _env(num_envs=1000, randomize=True, seed=0)
    seen = 0
    for _ in range(2):                                    # 2 x 1000 resets = 2000 samples
        env.reset()
        for name, num, den in (("D_lin", env.dynamics.D_lin_env, env.dynamics.D_lin),
                               ("D_quad", env.dynamics.D_quad_env, env.dynamics.D_quad)):
            ratio = (num / den).cpu()
            assert ratio.shape == (1000, 3)
            assert ratio.min().item() >= lo - 1e-4, (name, ratio.min().item())
            assert ratio.max().item() <= hi + 1e-4, (name, ratio.max().item())
        # mass must NOT leak into drag any more (reports/R3.md §4)
        mass_ratio = (env.dynamics.M_env[:, 0] / env.dynamics.M[0]).cpu()
        drag_ratio = (env.dynamics.D_lin_env / env.dynamics.D_lin)[:, 0].cpu()
        assert not torch.allclose(mass_ratio, drag_ratio)
        seen += 1000
    assert seen >= 1000


# -------------------------------------------------------------------- (3) PID in the env
# The PID does NOT converge in this env (reports/C1.md §2: the placeholder thruster
# geometry cross-couples badly, so the heading PID saturates every step and limit-cycles).
# These tests therefore pin the *interface* and the metric plumbing, not control quality;
# they fail if a controller/env interface changes, not because the numbers are good.
def test_pid_station_keeping_runs_through_the_env():
    traj = run_pid("sk", steps=300, num_envs=N)
    s = summarize_pid(traj)
    for key in ("radial_mean_m", "radial_p95_m", "xtrack_rmse_m", "energy",
                "action_rate_1_s", "err_start_m", "err_end_m"):
        assert np.isfinite(s[key]), (key, s[key])
    assert 0.0 <= s["settled_frac"] <= 1.0
    assert s["radial_p95_m"] >= s["radial_mean_m"] > 0.0
    assert s["energy"] > 0.0 and s["action_rate_1_s"] > 0.0
    assert np.abs(traj["action"]).max() <= 1.0 + 1e-6          # PID output stays clipped


def test_pid_waypoint_runs_through_the_env():
    traj = run_pid("wp", steps=300, num_envs=N)
    s = summarize_pid(traj)
    for key in ("radial_mean_m", "radial_p95_m", "xtrack_rmse_m", "energy", "action_rate_1_s"):
        assert np.isfinite(s[key]), (key, s[key])
    assert np.abs(traj["action"]).max() <= 1.0 + 1e-6
    assert np.all(np.isfinite(traj["xy"]))                     # no NaN blown up physics


# --------------------------------------------------------------------- (4) PPO plumbing
def test_ppo_loop_runs_on_the_real_env():
    """Two PPO iterations on `envs/asv_env.py` itself: finite losses, no NaN reward."""
    from train import PPOTrainer, load_config as load_train_cfg

    cfg = load_train_cfg(Path(__file__).resolve().parent.parent / "configs" / "train.yaml")
    cfg.update({"num_envs": 32, "device": DEV, "seed": 0, "rollout_steps": 8,
                "minibatches": 4, "epochs": 1, "log_dir": "runs/c1_smoke",
                "ckpt_dir": "runs/c1_smoke/ckpt", "save_interval_steps": 0})
    env = _env(num_envs=32)
    tr = PPOTrainer(cfg, env=env)
    last = tr.train(iterations=2)
    assert len(tr.history) == 2
    assert np.isfinite(last["reward"]) and np.isfinite(last["pi_loss"]) and np.isfinite(last["v_loss"])
    assert last["steps"] == 2 * 8 * 32


# ------------------------------------------------------------------------- CLI report
def main() -> int:
    print(f"device={DEV}  physics randomisation OFF  curriculum level 0  "
          f"seed 0  PID default (0.05 output deadband)")
    for mode, label in (("sk", "station keeping"), ("wp", "waypoint")):
        traj = run_pid(mode, steps=1200, num_envs=N)
        s = summarize_pid(traj)
        print(f"\nPID {label} | 1200 policy steps = 120 s, {N} envs")
        print(f"  radial error  mean {s['radial_mean_m']:.2f} m   p95 {s['radial_p95_m']:.2f} m")
        print(f"  cross-track RMSE (path)   {s['xtrack_rmse_m']:.2f} m")
        print(f"  settled (tol 2.0 m)       {100*s['settled_frac']:.0f} % of envs, "
              f"mean {s['settling_s']:.1f} s")
        print(f"  energy (normalized)       {s['energy']:.1f}")
        print(f"  action rate               {s['action_rate_1_s']:.3f} 1/s")
        print(f"  batch-mean error  first 1 s {s['err_start_m']:.2f} m -> last 6 s {s['err_end_m']:.2f} m")
    print("\nAll numbers are simulator outputs with placeholder vehicle parameters "
          "(configs/vehicle.yaml TODO SYSID), not measurements of the real vessel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
