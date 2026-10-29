"""torch vs onnxruntime parity of the exported policy — CONTRACT.md §4.8.

Runs against the BEST trained checkpoint (`runs/r7_300m/ckpt/ckpt_last.pt`, 300M steps),
falling back to the Wave 4 `runs/checkpoints/ckpt_last.pt` when that run is absent. Re-exports
on the fly so nothing depends on a stale artifact. CPU only.
"""

import json
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

import export_onnx
from runtime.policy import RLController

DRL = Path(__file__).resolve().parents[1]
# R7 (300M) is the model that ships; tests must guard THAT artifact, not an obsolete one.
CKPT = DRL / "runs" / "r7_300m" / "ckpt" / "ckpt_last.pt"
if not CKPT.is_file():
    CKPT = DRL / "runs" / "checkpoints" / "ckpt_last.pt"
ENV = DRL / "configs" / "env.yaml"
TOL = 1e-4
LAT0, LON0 = -6.2, 106.8


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """(onnx_path, meta_path, policy, session, meta) — real checkpoint, real export."""
    if not CKPT.is_file():
        pytest.skip(f"no trained checkpoint at {CKPT}")
    d = tmp_path_factory.mktemp("export")
    out, meta_path, meta = export_onnx.export(CKPT, d / "policy.onnx", env_cfg=ENV)
    policy, _ = export_onnx.load_policy(CKPT)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    return out, meta_path, policy, sess, meta


def _obs(n, seed=0):
    return np.random.default_rng(seed).uniform(-3.0, 3.0, (n, 150)).astype(np.float32)


def _torch(policy, x):
    with torch.no_grad():
        return policy(torch.from_numpy(np.ascontiguousarray(x)))[0].numpy()


def _ort(sess, x):
    return sess.run([export_onnx.OUTPUT_NAME], {export_onnx.INPUT_NAME: x})[0]


# ------------------------------------------------------------------- graph shape
def test_graph_signature_and_opset(exported):
    out, _meta_path, policy, sess, _meta = exported
    model = onnx.load(str(out))
    assert [o.version for o in model.opset_import] == [17]
    (inp,) = sess.get_inputs()
    (outp,) = sess.get_outputs()
    assert (inp.name, inp.type) == ("obs", "tensor(float)")
    assert inp.shape == ["B", 150]
    assert (outp.name, outp.type, outp.shape) == ("action", "tensor(float)", ["B", 3])
    # graph constants carry the checkpoint's normalization buffers, pre-MLP
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    baked = [v for v in inits.values() if v.shape == (150,)]
    assert len(baked) >= 2
    assert any(np.allclose(v, policy.obs_mean.numpy(), atol=0) for v in baked)
    assert any(np.allclose(v, policy.obs_std.numpy(), atol=0) for v in baked)
    assert model.graph.node[0].op_type in ("Sub", "Div")


# ------------------------------------------------------------------------ parity
def test_parity_torch_vs_onnxruntime(exported):
    _out, _mp, policy, sess, _meta = exported
    x = _obs(256)
    diff = float(np.abs(_torch(policy, x) - _ort(sess, x)).max())
    print(f"max abs diff torch vs onnxruntime on 256 obs: {diff:.3e}")
    assert diff < TOL, f"parity {diff:.3e} >= {TOL:.0e}"


def test_dynamic_batch(exported):
    _out, _mp, policy, sess, _meta = exported
    for b in (1, 7):
        x = _obs(b, seed=b)
        a = _ort(sess, x)
        assert a.shape == (b, 3) and a.dtype == np.float32
        assert float(np.abs(a - _torch(policy, x)).max()) < TOL
        assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0)


def test_output_is_deterministic_mean(exported):
    _out, _mp, policy, sess, _meta = exported
    x = _obs(64, seed=7)
    a1, a2 = _ort(sess, x), _ort(sess, x)
    assert np.array_equal(a1, a2), "ONNX output varies on identical input"
    # == tanh(mean_head(trunk(obs))) of the checkpoint, i.e. no sampling and tanh applied
    with torch.no_grad():
        ref = torch.tanh(policy.mean_head(policy.trunk(torch.from_numpy(x)))).numpy()
    assert float(np.abs(a1 - ref).max()) < TOL


# -------------------------------------------------------------------------- meta
def test_policy_meta_has_required_keys(exported):
    out, meta_path, _policy, _sess, _meta = exported
    assert meta_path.is_file() and meta_path.parent == out.parent
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for k in export_onnx.REQUIRED_META_KEYS:
        assert k in meta, f"policy_meta.json missing {k!r}"
    assert (meta["obs_dim"], meta["act_dim"], meta["k_history"]) == (150, 3, 10)
    assert len(meta["obs_offset"]) == 15 and len(meta["obs_scale"]) == 15
    assert meta["obs_scale"][0] == 10.0 and meta["obs_scale"][6] == 0.5
    assert meta["obs_clip_norm"] == 5.0
    assert meta["policy_dt_s"] == pytest.approx(0.1)
    assert set(meta["safety"]) >= {"clip", "rate_limit_per_s", "geofence_m", "watchdog_s"}
    assert meta["safety"]["geofence_m"] == 80.0
    assert meta["safety"]["rate_limit_per_s"] == 2.0


# ------------------------------------------------------- runtime consumer loads pair
def test_rlcontroller_loads_exported_pair(exported):
    out, meta_path, _policy, _sess, _meta = exported
    clock_t = [0.0]

    def clock():
        return clock_t[0]

    ctl = RLController(onnx_path=str(out), meta_path=str(meta_path), clock=clock)
    assert ctl.meta["obs_dim"] == 150 and ctl.policy_dt_s == pytest.approx(0.1)

    from common.frames import enu_to_latlon

    lat, lon = enu_to_latlon(5.0, 5.0, LAT0, LON0)
    est = {"lat": lat, "lon": lon, "heading_deg": 10.0, "yaw_rate_deg_s": 0.5,
           "speed_mps": 1.2, "course_deg": 15.0, "gps_valid": True,
           "imu_valid": True, "mag_valid": True, "t": 0.0}
    goal = {"lat": LAT0, "lon": LON0, "heading_deg": 0.0, "mode": "waypoint"}
    for k in range(12):
        clock_t[0] = 0.1 * k
        est["t"] = clock_t[0]
        a = ctl.step(est, goal)
        assert isinstance(a, tuple) and len(a) == 3
        assert all(isinstance(v, float) and math.isfinite(v) for v in a)
        assert all(abs(v) <= 1.0 for v in a)
    assert ctl.status["source"] == "policy" and ctl.status["fault"] is None
