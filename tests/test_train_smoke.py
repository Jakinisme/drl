"""Smoke tests for the compact PPO trainer (B2).

`envs/asv_env.py` (B1) does not exist yet, so these tests drive a DUMMY env that follows
CONTRACT §4.5: obs_dim 150, act_dim 3, auto-reset on done, `step -> (obs, rew, done, info)`.
Nothing here touches the real sim, hardware, or a GPU: `device: cpu`.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from models import PolicyNet, build_policy, gaussian_log_prob
from train import DEFAULTS, PPOTrainer, load_config

ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    """CONTRACT §4.5-shaped toy task: drive the action to a fixed target."""

    def __init__(self, num_envs: int = 8, done_every: int = 0, seed: int = 0) -> None:
        self.num_envs = num_envs
        self.obs_dim = 150
        self.act_dim = 3
        self.dt = 0.02
        self.policy_dt = 0.1
        self.device = "cpu"
        self.done_every = done_every
        self.target = torch.tensor([0.5, -0.5, 0.0])
        self.gen = torch.Generator().manual_seed(seed)
        self.t = 0
        self.action = torch.zeros(num_envs, 3)
        self.history = torch.zeros(num_envs, 10, 15)

    def _slot(self) -> torch.Tensor:
        noise = 0.01 * torch.randn(self.num_envs, 9, generator=self.gen)
        return torch.cat([self.action, self.target.expand(self.num_envs, 3), noise], dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        slot = self._slot()
        if env_ids is None:
            self.action.zero_()
            self.history = slot.unsqueeze(1).expand(self.num_envs, 10, 15).clone()
            self.t = 0
        else:
            self.action[env_ids] = 0.0
            self.history[env_ids] = slot[env_ids].unsqueeze(1).expand(len(env_ids), 10, 15)
        return self.history.reshape(self.num_envs, self.obs_dim).clone()

    def step(self, action: torch.Tensor):
        action = torch.clamp(torch.as_tensor(action, dtype=torch.float32), -1.0, 1.0)
        self.action = action
        reward = 1.0 - (action - self.target).pow(2).sum(-1)      # in [-3, 1]
        self.history = torch.cat([self.history[:, 1:], self._slot().unsqueeze(1)], dim=1)
        obs = self.history.reshape(self.num_envs, self.obs_dim).clone()
        self.t += 1
        done = torch.zeros(self.num_envs)
        if self.done_every and self.t % self.done_every == 0:
            done[:] = 1.0
            obs = self.reset()          # auto-reset, post-reset obs (CONTRACT §4.5)
        return obs, reward, done, {"time": self.t * self.policy_dt}

    def get_privileged_state(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 6)

    def set_curriculum(self, level: float) -> None:
        self.level = float(level)

    def close(self) -> None:
        pass


def make_cfg(**over) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update({
        "device": "cpu", "num_envs": 8, "rollout_steps": 32, "total_steps": 8 * 32 * 20,
        "epochs": 5, "minibatches": 4, "lr": 3.0e-3, "lr_min_frac": 0.2,
        "log_dir": "runs_smoke", "ckpt_dir": "runs_smoke/checkpoints",
        "save_interval_steps": 10_000, "seed": 0,
    })
    cfg.update(over)
    return cfg


def test_config_file_holds_every_hyperparameter():
    cfg = load_config(ROOT / "configs" / "train.yaml")
    raw = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    for key in ("algo", "total_steps", "num_envs", "rollout_steps", "gamma", "lam",
                "clip_ratio", "lr", "epochs", "minibatches", "entropy_coef",
                "value_coef", "max_grad_norm", "hidden", "activation", "init_log_std",
                "device", "dtype", "log_dir", "ckpt_dir", "save_interval_steps", "seed"):
        assert key in raw, key
    assert cfg["algo"] == "ppo"
    assert cfg["hidden"] == [256, 128, 128]
    assert cfg["obs_mean"] is None and cfg["obs_std"] is None


def test_policy_contract_shapes_and_size():
    net = build_policy(DEFAULTS)
    assert PolicyNet.size_in == 150
    obs = torch.randn(7, 150)
    mean, value, std = net(obs)
    assert mean.shape == (7, 3) and value.shape == (7, 1) and std.shape == (7, 3)
    assert torch.all(mean.abs() <= 1.0)
    assert torch.allclose(std, torch.full((7, 3), math.exp(DEFAULTS["init_log_std"])), rtol=1e-6)
    n_params = sum(p.numel() for p in net.parameters())
    assert 50_000 < n_params < 150_000, n_params          # ~95k for [256,128,128]
    assert not hasattr(net, "unexpected_state")


def test_obs_normalization_is_baked_in():
    net = PolicyNet(obs_dim=150, act_dim=3, hidden=(16,))
    obs = torch.zeros(2, 150)
    base, _, _ = net(obs)
    net.obs_mean.fill_(1.0)
    net.obs_std.fill_(0.5)
    shifted, _, _ = net(obs)
    assert not torch.allclose(base, shifted)


def test_gaussian_log_prob_matches_torch_distribution():
    mean = torch.zeros(4, 3)
    std = torch.full((4, 3), 0.6)
    action = torch.randn(4, 3)
    ours = gaussian_log_prob(action, mean, std)
    ref = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_ppo_improves_and_saves_checkpoint(tmp_path):
    cfg = make_cfg(ckpt_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"))
    trainer = PPOTrainer(cfg, env=DummyEnv(num_envs=8))
    trainer.train(iterations=20)
    history = [float(h["reward"]) for h in trainer.history]
    assert len(history) == 20
    assert all(math.isfinite(h["return"]) and math.isfinite(h["entropy"]) for h in trainer.history)

    mean_first = sum(history[:3]) / 3
    mean_last = sum(history[-5:]) / 5
    # Task: reward finite, and rising or at least never NaN.
    assert all(math.isfinite(r) for r in history), history
    assert mean_last > mean_first, (mean_first, mean_last)
    assert mean_last <= 1.0 + 1e-5
    # lr annealed, not constant
    assert trainer.history[-1]["lr"] < trainer.history[0]["lr"]

    # Checkpoint saved and loadable, small on disk.
    ckpts = sorted((tmp_path / "ckpt").glob("*.pt"))
    assert ckpts, "no checkpoint written"
    size_mb = ckpts[-1].stat().st_size / 1e6
    assert size_mb < 5.0, size_mb
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    assert set(state) >= {"model", "optimizer", "steps", "iteration", "cfg"}
    assert state["steps"] == trainer.steps > 0
    fresh = PPOTrainer(cfg, env=DummyEnv(num_envs=8))
    fresh.load(ckpts[-1])
    for a, b in zip(fresh.model.parameters(), trainer.model.parameters()):
        assert torch.equal(a, b)


def test_done_masking_and_reset_path():
    cfg = make_cfg(rollout_steps=40, total_steps=8 * 40 * 3)
    trainer = PPOTrainer(cfg, env=DummyEnv(num_envs=8, done_every=5), writer=None)
    for _ in range(3):
        roll = trainer.collect()
        upd = trainer.update()
        assert math.isfinite(roll["return"]) and math.isfinite(roll["adv_std"])
        assert all(math.isfinite(v) for v in upd.values())
    assert trainer.b_done.sum() > 0          # the mask was actually exercised
    assert torch.isfinite(trainer.obs).all()
