"""Policy network shared by training (`train.py`) and ONNX export (`export_onnx.py`).

Spec: CONTRACT.md §4.7.  Observation normalization is baked into the module as buffers
so that the exported ONNX graph can use the same constants (CONTRACT §4.8).

    forward(obs) -> (mean_action in [-1, 1] via tanh, value, std)

PPO sampling (CONTRACT §4.7): ``a = clamp(mean + std * eps, -1, 1)``, state-independent
``log_std`` (one per action component).  No tanh-Jacobian correction: the clamp is a
projection, not a squashing bijection — a documented simplification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PolicyNet", "ACTS", "build_policy"]

HALF_LOG_2PI = 0.9189385332046727        # 0.5 * log(2*pi), Gaussian log-density constant
HALF_LOG_2PIE = 1.4189385332046727       # 0.5 * log(2*pi*e), Gaussian entropy constant

ACTS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Diagonal Gaussian log-density, summed over the action dims.  [B] out."""
    var = std.pow(2)
    return (-((action - mean).pow(2) / (2.0 * var)) - torch.log(std) - HALF_LOG_2PI).sum(-1)


class PolicyNet(nn.Module):
    """MLP actor-critic with tanh-bounded mean and state-independent log_std."""

    size_in = 150  # CONTRACT §5: 15 raw values x K=10 history

    def __init__(
        self,
        obs_dim: int = 150,
        act_dim: int = 3,
        hidden: tuple[int, ...] = (256, 128, 128),
        activation: str = "elu",
        init_log_std: float = -0.5,
        obs_mean: torch.Tensor | None = None,
        obs_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if activation not in ACTS:
            raise ValueError(f"activation {activation!r} not in {sorted(ACTS)}")
        hidden = tuple(int(h) for h in hidden)
        if not hidden:
            raise ValueError("hidden must contain at least one layer")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = hidden
        self.activation = activation

        # Normalization constants: (obs - mean) / std, CPU float32 buffers so a
        # state_dict carries them and the ONNX exporter can read them.
        mean = torch.zeros(self.obs_dim) if obs_mean is None else torch.as_tensor(obs_mean, dtype=torch.float32)
        std = torch.ones(self.obs_dim) if obs_std is None else torch.as_tensor(obs_std, dtype=torch.float32)
        if mean.shape != (self.obs_dim,) or std.shape != (self.obs_dim,):
            raise ValueError(f"obs_mean/obs_std must have shape ({self.obs_dim},)")
        if torch.any(std <= 0):
            raise ValueError("obs_std must be strictly positive")
        self.register_buffer("obs_mean", mean.clone())
        self.register_buffer("obs_std", std.clone())

        act = ACTS[activation]
        layers: list[nn.Module] = []
        prev = self.obs_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), act()]
            prev = width
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, self.act_dim)
        self.value_head = nn.Linear(prev, 1)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), float(init_log_std)))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2.0**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """obs [B, obs_dim] -> (mean [B, act_dim] in [-1,1], value [B,1], std [B,act_dim])."""
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"expected obs_dim={self.obs_dim}, got {obs.shape[-1]}")
        h = self.trunk(self.normalize(obs))
        mean = torch.tanh(self.mean_head(h))
        value = self.value_head(h)
        std = self.log_std.exp().expand(obs.shape[0], self.act_dim)
        return mean, value, std

    # --- PPO helpers (training side only; export uses forward()[0]) -------------
    def act(
        self,
        obs: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a clipped action. Returns (action in [-1,1], log_prob, value)."""
        mean, value, std = self(obs)
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        action = torch.clamp(mean + std * eps, -1.0, 1.0)
        return action, self.log_prob(obs, action), value.squeeze(-1)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, _value, std = self(obs)
        return gaussian_log_prob(action, mean, std)

    def entropy(self) -> torch.Tensor:
        """Differential entropy of the (unclipped) Gaussian, summed over action dims.

        H = sum(log std + 0.5*log(2*pi*e))  [nats].
        """
        return (self.log_std + HALF_LOG_2PIE).sum()


def build_policy(cfg: dict, obs_dim: int = PolicyNet.size_in, act_dim: int = 3) -> PolicyNet:
    """Construct the policy from a `configs/train.yaml` dict (empty section = defaults)."""
    return PolicyNet(
        obs_dim=int(cfg.get("obs_dim", obs_dim)),
        act_dim=int(cfg.get("act_dim", act_dim)),
        hidden=tuple(cfg.get("hidden", (256, 128, 128))),
        activation=str(cfg.get("activation", "elu")),
        init_log_std=float(cfg.get("init_log_std", -0.5)),
        obs_mean=cfg.get("obs_mean"),
        obs_std=cfg.get("obs_std"),
    )
