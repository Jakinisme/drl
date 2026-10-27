"""R5 training-loop tests: info["terminated"/"truncated"], timeout bootstrap, resume state,
per-iteration episode metrics. CPU only, small dummy envs, no real training."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from train import DEFAULTS, PPOTrainer

ROOT = Path(__file__).resolve().parents[1]


class ScriptEnv:
    """CONTRACT §4.5-shaped env whose `done` is scripted per (step index, env).

    `kind[i][t]` is "t" (time limit -> truncated), "g" (geofence -> terminated) or None.
    obs row 0 = env id (so the terminal obs is identifiable after the auto-reset).
    """

    def __init__(self, num_envs: int = 4, obs_dim: int = 4, script: dict | None = None,
                 keys: bool = True, reward: float = 1.0) -> None:
        self.num_envs, self.obs_dim, self.act_dim = num_envs, obs_dim, 3
        self.device = "cpu"
        self.total_steps = 0
        self.level_index = 0
        self._level_override = None
        self.script = script or {}
        self.keys = keys
        self.reward = reward
        self.t = 0
        self.ep_step = torch.zeros(num_envs)
        self.episode = torch.zeros(num_envs)      # episode counter, also recorded in obs

    def _obs(self) -> torch.Tensor:
        o = torch.zeros(self.num_envs, self.obs_dim)
        o[:, 0] = torch.arange(self.num_envs, dtype=torch.float32)
        o[:, 1] = self.episode
        return o

    def reset(self, env_ids=None) -> torch.Tensor:
        o = self._obs()
        return o if env_ids is None else o[env_ids]

    def step(self, action):
        n = self.num_envs
        cur_t = self.t                          # script is keyed by this step's index
        obs = self._obs()
        rew = torch.full((n,), float(self.reward))
        kind = [self.script.get((cur_t, i)) for i in range(n)] if self.keys else [None] * n
        done = torch.tensor([1.0 if k else 0.0 for k in kind])
        info = {
            "ep_step": self.ep_step.clone() + 1.0,
            "arrive_evt": torch.zeros(n),
            "terminal_obs": obs.clone(),
        }
        if self.keys:
            info["terminated"] = torch.tensor([k == "g" for k in kind])
            info["truncated"] = torch.tensor([k == "t" for k in kind])
        self.ep_step += 1.0
        ids = (done > 0.5).nonzero().squeeze(-1)
        if ids.numel():                             # auto-reset: post-reset rows in `obs`
            obs[ids, 1] = self.episode[ids] + 1.0
            self.episode[ids] += 1.0
            self.ep_step[ids] = 0.0
        self.t += 1
        self.total_steps += 1
        return obs, rew, done, info

    def set_curriculum(self, level):              # trainer fallback path
        self._level_override = float(level)


def make_cfg(tmp_path, **over) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update({"device": "cpu", "num_envs": 1, "rollout_steps": 3, "total_steps": 3,
                "epochs": 1, "minibatches": 1, "lr": 1e-3, "log_interval": 1,
                "ckpt_dir": str(tmp_path / "ckpt"), "log_dir": str(tmp_path / "log"),
                "save_interval_steps": 0, "seed": 0, "gamma": 0.9})
    cfg.update(over)
    return cfg


def frozen_value_zero_trainer(cfg, env, value_bias: float) -> PPOTrainer:
    """Trainer whose critic returns exactly `value_bias` and cannot change (all params frozen).

    `train()` is never called; `collect()` reads the actor under no_grad, so the critic value
    of the TRUNCATED step (computed after `update()` in real training) equals `value_bias`.
    """
    tr = PPOTrainer(cfg, env=env)
    with torch.no_grad():
        tr.model.trunk[-2].weight.zero_()                       # last trunk Linear
        tr.model.trunk[-2].bias.zero_()
        tr.model.value_head.weight.zero_()                      # value == bias everywhere
        tr.model.value_head.bias.fill_(value_bias)
    return tr


def last_step_reward(tr: PPOTrainer) -> float:
    return float(tr.b_rew[-1, 0])


# ---------------------------------------------------------------- info keys
def test_info_keys_real_env():
    from envs.asv_env import load_config
    from envs.asv_env import AsvEnv

    env = AsvEnv(load_config(), num_envs=16, device="cpu", seed=0)
    zip_ = dict(env.cfg["episode"])
    zip_["max_steps"] = 2                       # force the time limit fast
    env.max_steps = 2
    a = torch.zeros(16, 3)
    seen = {"term": False, "trunc": False}
    for _ in range(6):
        obs, rew, done, info = env.step(a)
        assert info["terminated"].dtype == torch.bool and info["terminated"].shape == (16,)
        assert info["truncated"].dtype == torch.bool and info["truncated"].shape == (16,)
        assert not bool((info["terminated"] & info["truncated"]).any())
        assert not bool((done > 0.5)[~(info["terminated"] | info["truncated"])].any()), \
            "done set without terminated|truncated"
        assert not bool((info["timeout"] > 0.5)[~info["truncated"]].any())
        assert not bool((info["geofence"] > 0.5)[~info["terminated"]].any())
        assert info["terminal_obs"].shape == obs.shape
        seen["trunc"] |= bool(info["truncated"].any())

    # geofence -> terminated: put one env far outside the fence
    env.reset()
    env.state[0, 0] = env.geofence + 50.0
    _obs, _rew, done, info = env.step(a)
    assert info["geofence"][0] == 1.0 and info["terminated"][0] and done[0] == 1.0
    seen["term"] = True
    assert seen == {"term": True, "trunc": True}


# ---------------------------------------------------------------- GAE bootstrap
def test_gae_bootstrap_truncated_vs_terminated(tmp_path):
    g, V, r = 0.9, 2.5, 1.0
    cfg = make_cfg(tmp_path, rollout_steps=1, minibatches=1, lam=0.0)
    # NOTE: lam=0 makes the TD target exact and non-recursive: ret_t = r + g*V(1-done) - ... check
    out = {}
    for kind in ("t", "g"):
        env = ScriptEnv(num_envs=1, script={(0, 0): kind}, reward=r)
        tr = frozen_value_zero_trainer(cfg, env, value_bias=V)
        tr.collect()
        out[kind] = (float(tr.b_rew[-1, 0]), float(tr._ret[-1, 0]), float(tr.b_done[-1, 0]))

    r_t, ret_t, done_t = out["t"]
    r_g, ret_g, done_g = out["g"]
    assert r_t == pytest.approx(r + g * V, abs=1e-5), out        # truncated: bootstraps
    assert r_g == pytest.approx(r, abs=1e-5), out                # terminated: no bootstrap
    # lam=0, V(s_t)=V: ret = r_t (+ g*V if truncated)
    assert ret_t == pytest.approx(r + g * V, abs=1e-4), out
    assert ret_g == pytest.approx(r, abs=1e-4), out
    assert done_t == 1.0 and done_g == 1.0                       # no GAE bleed either way


def test_truncation_keeps_done_mask_and_no_bootstrap_on_terminated(tmp_path):
    """b_done must stay 1 on truncated steps; no terminal gets the bonus when geofence."""
    cfg = make_cfg(tmp_path, rollout_steps=3)
    script = {(1, 0): "t"}
    env = ScriptEnv(num_envs=1, script=script, reward=0.5)
    tr = frozen_value_zero_trainer(cfg, env, value_bias=1.0)
    tr.collect()
    rew = tr.b_rew[:, 0]
    assert rew[0] == pytest.approx(0.5) and rew[2] == pytest.approx(0.5)
    assert rew[1] == pytest.approx(0.5 + 0.9 * 1.0)
    assert tr.b_done[:, 0].tolist() == [0.0, 1.0, 0.0]

    env2 = ScriptEnv(num_envs=1, script={(1, 0): "g"}, reward=0.5)
    tr2 = frozen_value_zero_trainer(cfg, env2, value_bias=1.0)
    tr2.collect()
    assert tr2.b_rew[:, 0].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_env_without_keys_keeps_old_behaviour(tmp_path):
    cfg = make_cfg(tmp_path, rollout_steps=2)
    env = ScriptEnv(num_envs=1, script={(0, 0): "t"}, keys=False, reward=0.25)
    tr = frozen_value_zero_trainer(cfg, env, value_bias=3.0)
    tr.collect()
    assert tr.b_rew[:, 0].tolist() == pytest.approx([0.25, 0.25])


# ---------------------------------------------------------------- episode metrics
def test_episode_stats_nan_and_finite(tmp_path):
    cfg = make_cfg(tmp_path, num_envs=2, rollout_steps=4)
    # no episode finishes in 4 steps -> NaN
    tr = PPOTrainer(cfg, env=ScriptEnv(num_envs=2, reward=1.0))
    st = tr.collect()
    for k in ("ep_return_mean", "ep_len_mean", "arrivals_per_ep"):
        assert math.isnan(st[k]), (k, st)
    assert math.isfinite(st["action_std_mean"])
    assert st["action_std_mean"] == pytest.approx(math.exp(DEFAULTS["init_log_std"]), rel=1e-6)

    # env 0 truncates at t=2 (ep_step 3), env 1 never -> mean over 1 finished episode
    tr = PPOTrainer(cfg, env=ScriptEnv(num_envs=2, script={(2, 0): "t"}, reward=2.0))
    st = tr.collect()
    assert st["ep_return_mean"] == pytest.approx(6.0)        # 3 steps * 2.0
    assert st["ep_len_mean"] == pytest.approx(3.0)
    assert st["arrivals_per_ep"] == pytest.approx(0.0)


# ---------------------------------------------------------------- resume (item C)
class LevelEnv(ScriptEnv):
    """ScriptEnv + the AsvEnv curriculum API (total_steps drives the level)."""

    def __init__(self, num_envs=2, level_up=True, **kw):
        super().__init__(num_envs=num_envs, **kw)
        self.start_level = 0
        self.n_levels = 4
        if level_up:
            self._sync_level = self._sync

    def _sync(self, force=False):
        if self._level_override is not None:
            i = int(round(self._level_override))
        else:
            i = self.start_level + self.total_steps // 10
        self.level_index = max(0, min(i, self.n_levels - 1))


def test_resume_restores_total_steps_and_level(tmp_path):
    cfg = make_cfg(tmp_path, num_envs=2, rollout_steps=2)
    env = LevelEnv(num_envs=2)
    tr = PPOTrainer(cfg, env=env)
    for _ in range(7):                      # 14 env-steps * ... -> total_steps = 14 > 10
        tr.collect()
        tr.update()
    tr.env.total_steps = 23                  # auto-advance level 2 (start_level + 23//10)
    tr.env._sync_level(force=True)
    assert tr.env.level_index == 2
    path = tr.save("ckpt_last")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["env_total_steps"] == 23 and ck["env_level_index"] == 2

    fresh = PPOTrainer(cfg, env=LevelEnv(num_envs=2))
    assert fresh.env.total_steps == 0 and fresh.env.level_index == 0
    fresh.load(path)
    assert fresh.env.total_steps == 23
    assert fresh.env.level_index == 2
    assert fresh.steps == tr.steps and fresh.it == tr.it


def test_old_checkpoint_without_env_state_warns(tmp_path, capsys):
    cfg = make_cfg(tmp_path, num_envs=2, rollout_steps=2)
    tr = PPOTrainer(cfg, env=LevelEnv(num_envs=2))
    path = tr.save("old")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("env_total_steps", "env_level_index", "env_level_override"):
        ck.pop(k, None)
    torch.save(ck, path)

    fresh = PPOTrainer(cfg, env=LevelEnv(num_envs=2))
    fresh.load(path)                          # no crash
    out = capsys.readouterr().out
    assert out.count("[warn]") == 1 and "env_total_steps" in out
    assert fresh.env.total_steps == 0 and fresh.env.level_index == 0


def test_curriculum_override_wins_over_checkpoint(tmp_path):
    cfg = make_cfg(tmp_path, num_envs=2, rollout_steps=2)
    tr = PPOTrainer(cfg, env=LevelEnv(num_envs=2))
    tr.env.total_steps = 35
    tr.env._sync_level(force=True)
    path = tr.save("ck")
    fresh = PPOTrainer(cfg, env=LevelEnv(num_envs=2))
    fresh.load(path)
    fresh.env.set_curriculum(0.0)             # what main() does for --curriculum-level
    fresh.env._sync_level(force=True)         # _apply_env_state already forced; set_curriculum re-forces
    assert fresh.env.level_index == 0
    assert fresh.env.total_steps == 35


# ---------------------------------------------------------------- end to end
def test_train_logs_new_metrics(tmp_path, capsys):
    # env 0 truncates every 4 steps, env 1 geofence-terminates at t=3 (episode 1) and t=7 (2)
    script = {(t, 0): "t" for t in (3, 7)} | {(3, 1): "g", (7, 1): "g"}
    cfg = make_cfg(tmp_path, num_envs=2, rollout_steps=4, total_steps=8 * 4, epochs=1,
                   minibatches=2, save_interval_steps=0)
    tr = PPOTrainer(cfg, env=ScriptEnv(num_envs=2, script=script, reward=0.5))
    last = tr.train(iterations=2)
    for k in ("ep_return_mean", "ep_len_mean", "arrivals_per_ep", "action_std_mean",
              "reward", "return", "pi_loss", "v_loss", "entropy", "kl", "clip_frac", "lr"):
        assert k in last, k
    # iteration 2: 4 finished episodes, each 4 steps * 0.5
    assert last["ep_return_mean"] == pytest.approx(2.0)
    assert last["ep_len_mean"] == pytest.approx(4.0)
    assert last["arrivals_per_ep"] == pytest.approx(0.0)
    assert tr.history[0]["ep_return_mean"] == pytest.approx(2.0)
    console = capsys.readouterr().out
    assert "ep_ret" in console and "arrivals/ep" in console and "std" in console
