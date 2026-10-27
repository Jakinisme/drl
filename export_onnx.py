"""Export a trained PPO checkpoint to ONNX + `policy_meta.json` — CONTRACT.md §4.8.

    python export_onnx.py [--ckpt runs/checkpoints/ckpt_last.pt] [--out runs/policy.onnx]
                          [--meta runs/policy_meta.json] [--env configs/env.yaml]

Graph: input `obs` float32 [B, 150] -> output `action` float32 [B, 3] (deterministic
tanh mean), opset 17, dynamic batch. `(obs - obs_mean) / obs_std` is baked in as graph
constants before the MLP. `PolicyNet` itself is not modified: a thin wrapper module
holds the buffers and calls the same `trunk` / `mean_head` it uses at training time.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

import models

OPSET = 17
INPUT_NAME = "obs"
OUTPUT_NAME = "action"


class _Export(nn.Module):
    """`(obs - mean) / std -> trunk -> tanh(mean_head)`.  mean/std are graph constants."""

    def __init__(self, policy):
        super().__init__()
        self.register_buffer("mean", policy.obs_mean.detach().clone())
        self.register_buffer("std", policy.obs_std.detach().clone())
        self.trunk = policy.trunk
        self.mean_head = policy.mean_head

    def forward(self, obs):
        return torch.tanh(self.mean_head(self.trunk((obs - self.mean) / self.std)))


# obs_offset / obs_scale / obs_clip_norm values fed to the *policy* graph are the
# identity/none defaults: `envs/asv_env.py` already normalizes by `configs/env.yaml`
# obs.scale before the MLP. The export therefore bakes only obs_mean/obs_std (identity
# for this checkpoint) — exactly the normalization training applied.
# ponytail: if the env ever stops normalizing, bakes will extend to (obs-offset)/scale
# here; ceiling = one graph revision, upgrade = fold scale/offset into `mean`/`std`.

REQUIRED_META_KEYS = (
    "obs_dim", "act_dim", "k_history", "obs_offset", "obs_scale", "obs_clip_norm",
    "input_name", "output_name", "policy_dt_s", "safety",
)


def load_policy(ckpt_path):
    """Checkpoint -> eval-mode `PolicyNet` (built through `models.build_policy`)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy = models.build_policy(dict(ck.get("cfg") or {}), act_dim=3)
    policy.load_state_dict(ck["model"])
    policy.eval()
    return policy, ck


def export(ckpt_path="runs/checkpoints/ckpt_last.pt", out_path="runs/policy.onnx",
           meta_path=None, env_cfg="configs/env.yaml"):
    """Write `out_path` and `meta_path` (default: `policy_meta.json` beside the .onnx)."""
    out_path = Path(out_path)
    meta_path = Path(meta_path) if meta_path else out_path.with_name("policy_meta.json")
    policy, ck = load_policy(ckpt_path)
    env = yaml.safe_load(Path(env_cfg).read_text(encoding="utf-8"))

    dummy = torch.zeros(1, policy.obs_dim, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _Export(policy), dummy, str(out_path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "B"}, OUTPUT_NAME: {0: "B"}},
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )

    obs = env.get("obs") or {}
    action = env.get("action") or {}
    meta = {
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
        "k_history": int(obs.get("k_history", 10)),
        "obs_offset": [float(v) for v in obs.get("offset")],
        "obs_scale": [float(v) for v in obs.get("scale")],
        "obs_clip_norm": float(obs.get("clip_norm", 5.0)),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "policy_dt_s": 1.0 / float(env.get("policy_hz", 10.0)),
        "safety": {
            "clip": float(action.get("clip", 1.0)),
            "rate_limit_per_s": float(action.get("rate_limit_per_s", 2.0)),
            "geofence_m": float((env.get("episode") or {}).get("geofence_m", 80.0)),
            "watchdog_s": 0.5,   # runtime/policy.DEFAULT_SAFETY: no config key exists
        },
        # Provenance, ignored by RLController (it reads only the keys above).
        "source_ckpt": str(ckpt_path), "steps": ck.get("steps"),
        "iteration": ck.get("iteration"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_path, meta_path, meta


def main(argv=None):
    ap = argparse.ArgumentParser(prog="export_onnx", description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="runs/checkpoints/ckpt_last.pt")
    ap.add_argument("--out", default="runs/policy.onnx")
    ap.add_argument("--meta", default=None, help="default: policy_meta.json beside --out")
    ap.add_argument("--env", default="configs/env.yaml")
    a = ap.parse_args(argv)
    out, meta, m = export(a.ckpt, a.out, a.meta, a.env)
    print(f"wrote {out} ({out.stat().st_size} B) and {meta}")
    print(json.dumps({k: m[k] for k in REQUIRED_META_KEYS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
