"""Compact PPO trainer for the ASV env (CONTRACT §3.5, §4.5, §4.7).

Why not rsl_rl: `rsl-rl-lib` is **not installed** in `drl/.venv`, and
`pip install --dry-run rsl-rl-lib` resolves to `rsl-rl-lib 5.5.1` **plus torch-2.14.1**
(it requires `torch>=2.6.0` and pip picks the newest), which would replace the working
`torch 2.6.0+cu124`. Installing is forbidden for this wave, and rsl_rl 5.x also expects
an Isaac-style `VecEnv` wrapper (`get_observations() -> dict`, gym-spaces) that the plain
`AsvEnv` of CONTRACT §4.5 does not provide. Measured check: `reports/B2.md` §1.

Environment contract used here (CONTRACT §4.5): `reset() -> obs[N,150]`,
`step(a[N,3]) -> (obs, reward[N], done[N], info)`, auto-reset on done.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from models import PolicyNet, build_policy, gaussian_log_prob

ROOT = Path(__file__).resolve().parent

# Every number that is a hyperparameter lives in configs/train.yaml; these are the
# fallbacks for keys a hand-written config may omit, and they mirror that file.
DEFAULTS: dict = {
    "algo": "ppo",
    "total_steps": 20_000_000,
    "num_envs": 2048,
    "rollout_steps": 32,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_ratio": 0.2,
    "lr": 3.0e-4,
    "epochs": 5,
    "minibatches": 4,
    "entropy_coef": 0.005,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "hidden": [256, 128, 128],
    "activation": "elu",
    "init_log_std": -0.5,
    "device": "cuda",
    "dtype": "float32",
    "log_dir": "runs",
    "ckpt_dir": "runs/checkpoints",
    "save_interval_steps": 500_000,
    "seed": 0,
    "env_config": "configs/env.yaml",
    "vehicle_config": "configs/vehicle.yaml",
    "disturbance_config": "configs/disturbance.yaml",
    "sensors_config": "configs/sensors.yaml",
    # B2 additions (not in CONTRACT §3.5, no key there renamed)
    "lr_schedule": "linear",   # linear anneal lr -> lr*lr_min_frac
    "lr_min_frac": 0.05,
    "normalize_advantage": True,
    "log_interval": 1,
    "obs_mean": None,
    "obs_std": None,
}

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def load_config(path: str | Path) -> dict:
    cfg = dict(DEFAULTS)
    with open(path, "r", encoding="utf-8") as fh:
        cfg.update(yaml.safe_load(fh) or {})
    if cfg["algo"] not in ("ppo",):
        raise ValueError(f"algo {cfg['algo']!r} unsupported (rsl_rl not installable, see report)")
    if cfg["dtype"] not in DTYPES:
        raise ValueError(f"dtype {cfg['dtype']!r} not in {sorted(DTYPES)}")
    if (int(cfg["num_envs"]) * int(cfg["rollout_steps"])) % int(cfg["minibatches"]):
        raise ValueError("minibatches must divide num_envs*rollout_steps")
    return cfg


def build_env(cfg: dict):
    """Real env. Imported lazily so the smoke test can pass its own dummy env."""
    from envs.asv_env import AsvEnv  # B1

    env_cfg = cfg.get("_env_yaml")
    if env_cfg is None:
        with open(ROOT / cfg["env_config"], "r", encoding="utf-8") as fh:
            env_cfg = yaml.safe_load(fh)
    # TODO(B2): the ctor belongs to B1; CONTRACT §4.5 pins attributes, not the signature.
    # This call matches envs/asv_env.py as landed (vehicle/disturbance/sensors configs
    # default to the same files named in configs/train.yaml).
    # R6: hand the curriculum its budget in ENV-STEPS so `level_up_fraction` is a fraction
    # of the actual run, not of max_steps (see the ctor comment in envs/asv_env.py).
    # `total_steps` is ALREADY the env-step budget (the trainer accumulates T*N per
    # iteration), so do NOT multiply it by num_envs.
    env_cfg.setdefault("curriculum", {})
    if env_cfg["curriculum"].get("total_train_steps") is None:
        env_cfg["curriculum"]["total_train_steps"] = int(cfg["total_steps"])
    return AsvEnv(env_cfg, num_envs=int(cfg["num_envs"]), device=str(cfg["device"]),
                  dtype=DTYPES[cfg["dtype"]], seed=int(cfg["seed"]))


class PPOTrainer:
    def __init__(self, cfg: dict, env=None, writer=None) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu")
        self.dtype = DTYPES[cfg["dtype"]]
        torch.manual_seed(int(cfg["seed"]))

        self.env = env if env is not None else build_env(cfg)
        self.env_device = str(getattr(self.env, "device", self.device))
        self.N = int(getattr(self.env, "num_envs", cfg["num_envs"]))
        self.obs_dim = int(getattr(self.env, "obs_dim", PolicyNet.size_in))
        self.act_dim = int(getattr(self.env, "act_dim", 3))
        self.T = int(cfg["rollout_steps"])

        self.model = build_policy(cfg, self.obs_dim, self.act_dim).to(self.device, self.dtype)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(cfg["lr"]))
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(cfg["seed"]))

        # Rollout storage: [T, N, ...]. At 2048x32x150 float32 this is 39 MB.
        shape = (self.T, self.N)
        self.b_obs = torch.zeros(*shape, self.obs_dim, device=self.device, dtype=self.dtype)
        self.b_act = torch.zeros(*shape, self.act_dim, device=self.device, dtype=self.dtype)
        self.b_logp = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.b_val = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.b_rew = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.b_done = torch.zeros(shape, device=self.device, dtype=self.dtype)

        self.obs = self._to(self.env.reset())
        # R5-A: stagger episode phase so the N envs do not all time out on one iteration.
        # Dummy envs in tests have no phase counter -> optional method.
        desync = getattr(self.env, "desync_episodes", None)
        if desync is not None:
            desync(self.gen)
        self.steps = 0
        self.it = 0
        self.history: list[dict] = []
        self.writer = writer
        self.ckpt_dir = Path(cfg["ckpt_dir"])

    # --- plumbing -------------------------------------------------------------
    def _to(self, x) -> torch.Tensor:
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def _lr(self, frac: float) -> float:
        base = float(self.cfg["lr"])
        if self.cfg["lr_schedule"] != "linear":
            return base
        return base * (1.0 - (1.0 - float(self.cfg["lr_min_frac"])) * min(max(frac, 0.0), 1.0))

    def _log(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    # --- rollout --------------------------------------------------------------
    def collect(self) -> dict:
        T, N = self.T, self.N
        gamma = float(self.cfg["gamma"])
        boot = torch.zeros(T, N, device=self.device, dtype=self.dtype)   # R5 timeout bootstrap
        run_ret = torch.zeros(N, device=self.device, dtype=self.dtype)   # R5 per-env episode return
        n_fin = 0
        ep_ret, ep_len, n_arr = 0.0, 0.0, 0.0
        with torch.no_grad():
            for t in range(T):
                action, logp, value = self.model.act(self.obs, generator=self.gen)
                obs, rew, done, info = self.env.step(action)
                rew = self._to(rew)
                done_t = self._to(done).clamp_(0.0, 1.0)
                run_ret += rew
                # R5: time-limit truncation -> bootstrap gamma*V(terminal obs); geofence stays a
                # true terminal. done stays 1 either way, so GAE never bleeds across episodes.
                if info.get("terminated") is not None and info.get("truncated") is not None:
                    trunc = self._to(info["truncated"]) * (1.0 - self._to(info["terminated"]))
                    sel = trunc > 0.5
                    if bool(sel.any()):
                        term_obs = info.get("terminal_obs")
                        term_obs = obs if term_obs is None else self._to(term_obs)
                        _m, v_next, _s = self.model(term_obs[sel])
                        boot[t, sel] = v_next.squeeze(-1).to(self.dtype)
                self.obs = self._to(obs)
                # R5 episode stats: episodes that FINISHED in this iteration
                if "arrive_evt" in info:
                    n_arr += float(self._to(info["arrive_evt"]).sum())
                fin = done_t.nonzero(as_tuple=True)[0]
                if fin.numel():
                    n_fin += int(fin.numel())
                    ep_ret += float(run_ret[fin].sum())
                    # `ep_step` is optional: a CONTRACT §4.5-shaped dummy env (tests) may not
                    # report it, so the length metric degrades to NaN instead of raising.
                    if "ep_step" in info:
                        ep_len += float(self._to(info["ep_step"])[fin].sum())
                    else:
                        ep_len = float("nan")
                    run_ret[fin] = 0.0
                self.b_obs[t] = self.obs
                self.b_act[t] = action
                self.b_logp[t] = logp
                self.b_val[t] = value
                self.b_rew[t] = rew
                self.b_done[t] = done_t
            _mean, last_val, _std = self.model(self.obs)
        self.b_rew += gamma * boot
        self.steps += T * N
        stats = self._gae(last_val.squeeze(-1))
        nan = float("nan")
        stats.update({
            "ep_return_mean": ep_ret / n_fin if n_fin else nan,
            "ep_len_mean": ep_len / n_fin if n_fin else nan,
            "arrivals_per_ep": n_arr / n_fin if n_fin else nan,
            "action_std_mean": float(self.model.log_std.exp().mean()),
            "episodes": float(n_fin),
            "arrivals": n_arr,
        })
        return stats

    def _gae(self, last_val: torch.Tensor) -> dict:
        cfg = self.cfg
        gamma, lam = float(cfg["gamma"]), float(cfg["lam"])
        adv = torch.zeros_like(self.b_rew)
        running = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        for t in reversed(range(self.T)):
            next_val = last_val if t == self.T - 1 else self.b_val[t + 1]
            live = 1.0 - self.b_done[t]           # auto-reset: next obs is a new episode
            delta = self.b_rew[t] + gamma * next_val * live - self.b_val[t]
            running = delta + gamma * lam * live * running
            adv[t] = running
        ret = adv + self.b_val
        self._adv, self._ret = adv, ret
        return {
            "reward": float(self.b_rew.mean()),
            "return": float(ret.mean()),
            "adv_std": float(adv.std()),
        }

    # --- update ---------------------------------------------------------------
    def update(self) -> dict:
        cfg = self.cfg
        flat = self.T * self.N
        obs = self.b_obs.reshape(flat, self.obs_dim)
        act = self.b_act.reshape(flat, self.act_dim)
        old_logp = self.b_logp.reshape(flat)
        adv = self._adv.reshape(flat)
        ret = self._ret.reshape(flat)
        if cfg["normalize_advantage"]:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        mb = flat // int(cfg["minibatches"])
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0, "n": 0}
        for _epoch in range(int(cfg["epochs"])):
            idx = torch.randperm(flat, device=self.device)
            for k in range(int(cfg["minibatches"])):
                sel = idx[k * mb:(k + 1) * mb]
                o, a, lp_old = obs[sel], act[sel], old_logp[sel]
                mean, value, std = self.model(o)
                lp = gaussian_log_prob(a, mean, std)
                ratio = (lp - lp_old).exp()
                clipped = torch.clamp(ratio, 1.0 - float(cfg["clip_ratio"]), 1.0 + float(cfg["clip_ratio"]))
                pi_loss = -torch.min(ratio * adv[sel], clipped * adv[sel]).mean()
                v_loss = (value.squeeze(-1) - ret[sel]).pow(2).mean()
                entropy = self.model.entropy()
                loss = pi_loss + float(cfg["value_coef"]) * v_loss - float(cfg["entropy_coef"]) * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg["max_grad_norm"]))
                self.opt.step()

                stats["pi_loss"] += float(pi_loss)
                stats["v_loss"] += float(v_loss)
                stats["entropy"] += float(entropy)
                stats["kl"] += float((lp_old - lp).mean())
                stats["clip_frac"] += float(((ratio - 1.0).abs() > float(cfg["clip_ratio"])).float().mean())
                stats["n"] += 1
        n = max(stats.pop("n"), 1)
        return {k: v / n for k, v in stats.items()}

    # --- loop -----------------------------------------------------------------
    def train(self, iterations: int | None = None) -> dict:
        cfg = self.cfg
        total = int(cfg["total_steps"])
        per_iter = self.T * self.N
        n_iter = iterations if iterations is not None else max(1, total // per_iter)
        lr = self._lr(0.0)
        for p in self.opt.param_groups:
            p["lr"] = lr
        t0 = time.time()
        last = {}
        for it in range(n_iter):
            roll = self.collect()
            lr = self._lr(self.steps / total)
            for p in self.opt.param_groups:
                p["lr"] = lr
            upd = self.update()
            last = {**roll, **upd, "lr": lr, "steps": self.steps,
                    "steps_per_s": self.steps / max(time.time() - t0, 1e-6)}
            self.history.append(last)
            if self.it % int(cfg["log_interval"]) == 0:
                self._log_train(last)
            self.it += 1
            if int(cfg["save_interval_steps"]) > 0 and \
                    (self.steps // int(cfg["save_interval_steps"])) > ((self.steps - per_iter) // int(cfg["save_interval_steps"])):
                self.save(f"ckpt_{self.steps:09d}")
        self.save("ckpt_last")
        if self.writer is not None:
            self.writer.flush()
        return last

    def _log_train(self, d: dict) -> None:
        # one stdout line per iteration: the smoke run is read from a log file (reports/C1.md)
        print(f"it {self.it:4d} steps {d['steps']:8d} reward {d['reward']:+.4f} "
              f"return {d['return']:+.3f} pi_loss {d['pi_loss']:+.4f} lr {d['lr']:.2e} "
              f"steps/s {d['steps_per_s']:.0f} ep_ret {d.get('ep_return_mean', float('nan')):+.3f} "
              f"ep_len {d.get('ep_len_mean', float('nan')):.1f} arrivals/ep {d.get('arrivals_per_ep', float('nan')):.3f} "
              f"std {d.get('action_std_mean', float('nan')):.3f}", flush=True)
        if self.writer is None:
            return
        for key in ("reward", "return", "pi_loss", "v_loss", "entropy", "kl", "clip_frac", "lr", "steps_per_s",
                    "ep_return_mean", "ep_len_mean", "arrivals_per_ep", "action_std_mean"):
            if key in d:
                self._log(f"train/{key}", d[key], self.steps)
        self._log("train/iteration", self.it, self.steps)

    # --- checkpoint -----------------------------------------------------------
    def save(self, tag: str) -> Path:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "steps": self.steps,
            "iteration": self.it,
            "cfg": self.cfg,
            # R5: curriculum state lives in the env, not in the trainer
            "env_total_steps": int(getattr(self.env, "total_steps", 0)),
            "env_level_index": int(getattr(self.env, "level_index", 0)),
            "env_level_override": getattr(self.env, "_level_override", None),
        }, path)
        return path

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
        self.steps = int(ckpt.get("steps", 0))
        self.it = int(ckpt.get("iteration", 0))
        self.obs = self._to(self.env.reset())
        if "env_total_steps" in ckpt:
            self._apply_env_state(ckpt["env_total_steps"], ckpt.get("env_level_override"),
                                  ckpt.get("env_level_index"))
        else:
            print(f"[warn] {path}: no env curriculum state (env_total_steps) in checkpoint; "
                  f"env starts at total_steps={getattr(self.env, 'total_steps', 0)} "
                  f"level={getattr(self.env, 'level_index', 0)}", flush=True)

    def _apply_env_state(self, total_steps, level_override, level_index) -> None:
        """Restore the env curriculum state saved by `save()` (level override wins)."""
        env = self.env
        if hasattr(env, "total_steps"):
            env.total_steps = int(total_steps)
        if hasattr(env, "_level_override"):
            env._level_override = None if level_override is None else float(level_override)
        if hasattr(env, "_sync_level"):
            env._sync_level(force=True)
        elif level_index is not None and hasattr(env, "set_curriculum"):
            env.set_curriculum(float(level_index))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PPO trainer for the ASV env (compact, no rsl_rl)")
    ap.add_argument("--config", default=str(ROOT / "configs/train.yaml"))
    ap.add_argument("--total-steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--iterations", type=int, default=None, help="override the iteration count")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--curriculum-level", type=float, default=None,
                    help="pin the disturbance curriculum level INDEX (not the scale); "
                         "disables auto-advance. Index i -> "
                         "disturbance.curriculum.scale_by_level[i], so 0 is no disturbance "
                         "and n_levels-1 is full scale")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    for key in ("num_envs", "device", "seed", "log_dir", "ckpt_dir", "total_steps"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(ROOT / cfg["log_dir"]))
    except ImportError as exc:  # pragma: no cover - tensorboard is installed in .venv
        print(f"[warn] tensorboard unavailable ({exc}); logging disabled")

    trainer = PPOTrainer(cfg, writer=writer)
    if args.curriculum_level is not None:
        trainer.env.set_curriculum(float(args.curriculum_level))
    if args.resume:
        trainer.load(args.resume)
    print(f"PPO | envs={trainer.N} rollout={trainer.T} obs_dim={trainer.obs_dim} "
          f"act_dim={trainer.act_dim} device={trainer.device} params={sum(p.numel() for p in trainer.model.parameters())}")
    last = trainer.train(args.iterations)
    print("last:", {k: round(v, 6) if isinstance(v, float) else v for k, v in last.items()})
    if torch.cuda.is_available():
        print(f"peak_vram_mib {torch.cuda.max_memory_allocated() / 2**20:.1f} "
              f"reserved {torch.cuda.max_memory_reserved() / 2**20:.1f}", flush=True)
    if writer is not None:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
