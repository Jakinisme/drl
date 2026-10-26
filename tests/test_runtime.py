"""Tests for the Pi runtime: numpy obs builder, ONNX controller, offline replay.

Wave 1 has no trained model and will not have one: every test builds its own dummy
ONNX MLP (random weights, `onnx.helper` — no torch needed) and asserts plumbing only:
shapes, history order, the §4.6 safety wrapper and the replay exit code. Nothing here
says anything about behaviour on the water.

Contract references: CONTRACT.md §1 (frames), §4.4 (`gps_valid`/`sensor_ok`, not the
legacy `has_fix`), §4.6/§4.8 (controller), §5 (observation), §6 (numpy/torch parity).
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from common.frames import enu_to_latlon, heading_deg_to_psi
from runtime.infer_replay import main as replay_main
from runtime.obs_np import OBS_DIM, RAW_DIM, ObsBuilder, normalize, raw_sample
from runtime.policy import RLController

DRL = Path(__file__).resolve().parents[1]
LAT0, LON0 = -6.2, 106.8          # arbitrary site; tests only use offsets from it


# ------------------------------------------------------------------ dummy model
def dummy_onnx(seed=0):
    """150 -> 8 tanh -> 3 tanh MLP with CONTRACT §4.8 signature, opset 17."""
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((OBS_DIM, 8)) * 0.1).astype(np.float32)
    w2 = (rng.standard_normal((8, 3)) * 0.5).astype(np.float32)
    nodes = [
        helper.make_node("MatMul", ["obs", "W1"], ["h1"]),
        helper.make_node("Add", ["h1", "B1"], ["h1b"]),
        helper.make_node("Tanh", ["h1b"], ["h1t"]),
        helper.make_node("MatMul", ["h1t", "W2"], ["h2"]),
        helper.make_node("Add", ["h2", "B2"], ["h2b"]),
        helper.make_node("Tanh", ["h2b"], ["action"]),
    ]
    inits = [numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(np.zeros(8, np.float32), "B1"),
             numpy_helper.from_array(w2, "W2"), numpy_helper.from_array(np.zeros(3, np.float32), "B2")]
    graph = helper.make_graph(
        nodes, "dummy_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["B", OBS_DIM])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["B", 3])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.fixture(scope="module")
def onnx_bytes():
    return dummy_onnx()


@pytest.fixture(scope="module")
def ort_session(onnx_bytes):
    return ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])


class RecordingSession:
    """ORT session that keeps every `obs` it was fed — proves the model is used."""

    def __init__(self, inner):
        self.inner = inner
        self.obs = []

    def get_inputs(self):
        return self.inner.get_inputs()

    def get_outputs(self):
        return self.inner.get_outputs()

    def run(self, output_names, feed):
        self.obs.append(np.array(feed["obs"], copy=True))
        return self.inner.run(output_names, feed)


class FakeSession:
    """Returns a fixed output; the smallest thing `RLController` accepts."""

    def __init__(self, out):
        self.out = np.asarray(out, dtype=np.float32)

    def get_inputs(self):
        return [SimpleNamespace(name="obs")]

    def get_outputs(self):
        return [SimpleNamespace(name="action")]

    def run(self, output_names, feed):
        return [self.out]


class Clock:
    """Monotonic fake clock: `dt` per call (0.0 = the test sets `.t` by hand)."""

    def __init__(self, dt=0.1):
        self.t, self.dt = 0.0, dt

    def __call__(self):
        self.t += self.dt
        return self.t


def _ctl(onnx_bytes=None, session=None, **kw):
    """Controller with the slew limiter out of the way unless a test overrides it."""
    kw.setdefault("safety", {"rate_limit_per_s": 1e9})
    kw.setdefault("clock", Clock())
    if session is not None:
        return RLController(session=session, **kw)
    return RLController(onnx_bytes=onnx_bytes, **kw)


def est(east=0.0, north=0.0, **kw):
    """`estimator_state` of CONTRACT §4.8 at a given ENU offset."""
    lat, lon = enu_to_latlon(east, north, LAT0, LON0)
    d = dict(lat=float(lat), lon=float(lon), heading_deg=0.0, yaw_rate_deg_s=0.0,
             speed_mps=1.0, course_deg=0.0, gps_valid=True, imu_valid=True,
             mag_valid=True)
    d.update(kw)
    return d


def goal(east=0.0, north=0.0, **kw):
    lat, lon = enu_to_latlon(east, north, LAT0, LON0)
    d = dict(lat=float(lat), lon=float(lon), mode="waypoint")
    d.update(kw)
    return d


# ------------------------------------------------------- (a) observation layout
def test_raw_sample_matches_contract_layout():
    psi = 0.0
    est_vec = [[0.0, 0.0, psi, 1.0, -0.5, 0.2]]
    goal_vec = [[10.0, 0.0, np.pi / 2, 0.0]]
    prev = [[0.1, 0.2, 0.3]]
    wp1 = [[0.0, 20.0]]
    wp2 = [[-5.0, 0.0]]
    s = raw_sample(est_vec, goal_vec, prev, wp1, wp2)
    assert s.shape == (1, RAW_DIM) and s.dtype == np.float64
    s = s[0]
    # bow north, goal 10 m east -> purely starboard offset
    assert np.allclose([s[0], s[1]], [0.0, 10.0], atol=1e-9)
    # psi_err = +90 deg -> sin 1, cos 0
    assert np.allclose([s[2], s[3]], [1.0, 0.0], atol=1e-9)
    assert np.allclose(s[4:7], [1.0, -0.5, 0.2], atol=1e-12)
    assert np.allclose(s[7:10], [0.1, 0.2, 0.3], atol=1e-12)
    # wp1 20 m north of a north-facing hull -> along the bow; wp2 5 m west
    assert np.allclose(s[10:12], [20.0, 0.0], atol=1e-9)
    assert np.allclose(s[12:14], [0.0, -5.0], atol=1e-9)
    assert s[14] == 0.0
    # missing waypoints and prev_action -> zeros, mode flag passes through
    bare = raw_sample(est_vec, [[0.0, 0.0, 0.0, 1.0]])[0]
    assert np.allclose(bare[7:10], 0.0) and np.allclose(bare[10:14], 0.0)
    assert bare[14] == 1.0


def test_obs_history_oldest_first_and_deterministic():
    builder = ObsBuilder(k_history=10)
    est_b = np.zeros((1, 6))
    samples = []
    for i in range(12):
        g = np.array([[10.0 + i, 0.0, 0.0, 0.0]])
        samples.append(builder.build(est_b, g, np.zeros((1, 3))).copy())

    last = samples[-1]
    assert last.shape == (1, OBS_DIM) and last.dtype == np.float32
    # slot i holds sample t-9+i: oldest first, newest last
    for k in range(10):
        assert np.allclose(last[0, k * RAW_DIM:(k + 1) * RAW_DIM],
                           samples[2 + k][0, -RAW_DIM:], atol=0.0)
    # the 10 slots are 10 different samples (no accidental duplication)
    slots = last[0].reshape(10, RAW_DIM)
    assert not np.allclose(slots[0], slots[-1])

    # reset() re-tiles the next sample into all K slots
    builder.reset()
    g = np.array([[10.0, 0.0, 0.0, 0.0]])
    fresh = builder.build(est_b, g, np.zeros((1, 3)))
    assert np.allclose(fresh[0].reshape(10, RAW_DIM), fresh[0, -RAW_DIM:], atol=0.0)

    # same input sequence -> bit-identical output
    again = ObsBuilder(k_history=10)
    reps = [again.build(est_b, np.array([[10.0 + i, 0.0, 0.0, 0.0]]),
                        np.zeros((1, 3))).copy() for i in range(12)]
    assert np.array_equal(reps[-1], last)


# ------------------------------------------- (b) controller really runs the ONNX
def test_controller_uses_model_and_builds_contract_obs(onnx_bytes, ort_session):
    rec = RecordingSession(ort_session)
    ctl = _ctl(session=rec)
    e = est(heading_deg=30.0, speed_mps=1.2, course_deg=60.0, yaw_rate_deg_s=5.0)
    g = goal(east=10.0, north=-5.0, heading_deg=10.0)

    a1 = np.array(ctl.step(e, g))
    assert ctl.status["source"] == "policy" and ctl.status["fault"] is None
    assert a1.shape == (3,) and np.abs(a1).max() <= 1.0
    assert len(rec.obs) == 1
    obs = rec.obs[0]
    assert obs.shape == (1, OBS_DIM) and obs.dtype == np.float32

    # hand-computed §5 sample: body frame from (0,0,psi=30deg), (u,v) from SOG/COG
    psi = float(heading_deg_to_psi(30.0))
    dg = np.deg2rad(60.0 - 30.0)
    est_vec = [[0.0, 0.0, psi, 1.2 * np.cos(dg), 1.2 * np.sin(dg), np.deg2rad(5.0)]]
    goal_vec = [[10.0, -5.0, float(heading_deg_to_psi(10.0)), 0.0]]
    want = normalize(raw_sample(est_vec, goal_vec, np.zeros((1, 3))))[0]
    assert np.allclose(obs[0, -RAW_DIM:], want, atol=1e-5)
    assert np.allclose(obs[0, :-RAW_DIM], np.tile(want, 9), atol=1e-5)

    # the command is the model output (clip and slew are the identity here)
    model_out = ort_session.run(None, {"obs": obs})[0].reshape(3)
    assert np.allclose(a1, model_out, atol=1e-6)

    # step 2: ring buffer shifts by one sample, a_prev is the applied command
    a2 = np.array(ctl.step(e, g))
    assert a2.shape == (3,) and np.abs(a2).max() <= 1.0
    assert np.array_equal(rec.obs[1][0, :-RAW_DIM], obs[0, RAW_DIM:])
    assert np.allclose(rec.obs[1][0, -8:-5], a1, atol=1e-6)


def test_controller_reads_policy_meta(tmp_path, onnx_bytes):
    onnx_path = tmp_path / "policy.onnx"
    onnx_path.write_bytes(onnx_bytes)
    meta_path = tmp_path / "policy_meta.json"
    meta_path.write_text(json.dumps({
        "obs_dim": OBS_DIM, "act_dim": 3, "k_history": 10,
        "input_name": "obs", "output_name": "action",
        "obs_clip_norm": 5.0, "policy_dt_s": 0.1,
        "safety": {"clip": 1.0, "rate_limit_per_s": 2.0, "geofence_m": 80.0,
                   "gps_timeout_s": 2.0, "watchdog_s": 0.5},
    }), encoding="utf-8")

    ctl = RLController(onnx_path=str(onnx_path), meta_path=str(meta_path), clock=Clock())
    assert ctl.safety["rate_limit_per_s"] == 2.0
    assert (ctl._input_name, ctl._output_name) == ("obs", "action")
    assert np.abs(np.array(ctl.step(est(), goal()))).max() <= 2.0 * 0.1 + 1e-9


def test_goal_mode_flag_and_missing_heading(onnx_bytes, ort_session):
    rec = RecordingSession(ort_session)
    ctl = _ctl(session=rec)
    wlat, wlon = enu_to_latlon(0.0, 10.0, LAT0, LON0)
    ctl.step(est(), goal(heading_deg=None, mode="station_keeping",
                         next_lat=float(wlat), next_lon=float(wlon)))
    obs = rec.obs[0][0]
    assert obs[-1] == 1.0                      # mode flag, §5 idx 14
    assert np.allclose(obs[-13:-11], [0.0, 1.0], atol=1e-6)   # psi_goal = psi
    assert np.isclose(obs[-5], 1.0, atol=1e-6)  # wp1 10 m ahead -> 10/10 m scale
    assert np.allclose(obs[-4], 0.0, atol=1e-6)


def test_controller_clips_to_unit_range(onnx_bytes):
    ctl = _ctl(session=FakeSession([[5.0, -9.0, 0.25]]))
    assert np.allclose(ctl.step(est(), goal()), [1.0, -1.0, 0.25], atol=1e-12)


# ----------------------------------------------------- (c) safety wrapper, §4.6
def test_slew_rate_limit(onnx_bytes):
    ctl = _ctl(onnx_bytes, clock=Clock(), safety={"rate_limit_per_s": 2.0})
    prev, deltas = np.zeros(3), []
    for i in range(8):
        a = np.array(ctl.step(est(), goal(east=5.0 * i)))
        assert np.abs(a).max() <= 1.0 + 1e-12
        deltas.append(np.abs(a - prev).max())
        prev = a
    # CONTRACT §3.4: |da| <= rate_limit_per_s * policy_dt (0.1 s here)
    assert max(deltas) <= 2.0 * 0.1 + 1e-9
    assert max(deltas) > 0.05, "model output never moved, slew test is vacuous"


def test_geofence_zeroes_the_command(onnx_bytes):
    ctl = _ctl(onnx_bytes, clock=Clock())
    ctl.step(est(), goal())
    assert np.abs(np.array(ctl.step(est(east=79.0), goal()))).max() > 0.0
    assert ctl.status["source"] == "policy"

    out = np.array(ctl.step(est(east=81.0), goal()))   # > geofence_m 80
    assert ctl.status["source"] == "geofence" and ctl.status["fault"] is None
    assert np.allclose(out, 0.0, atol=1e-12)


def test_watchdog_on_stale_sample(onnx_bytes):
    clock = Clock()
    ctl = _ctl(onnx_bytes, clock=clock)
    ctl.step(est(t=clock.t), goal())
    ctl.step(est(t=clock.t - 1.0), goal())            # 1.0 s > watchdog_s 0.5
    assert ctl.status["source"] == "fallback" and ctl.status["fault"] == "watchdog"


def test_watchdog_on_gap_between_calls(onnx_bytes):
    ctl = _ctl(onnx_bytes, clock=Clock(dt=1.0))
    ctl.step(est(), goal())
    ctl.step(est(), goal())                           # caller stalled 1.0 s
    assert ctl.status["source"] == "fallback" and ctl.status["fault"] == "watchdog"


def test_gps_invalid_uses_fallback_hook(onnx_bytes):
    calls = []

    def fallback(e, g):
        calls.append((e, g))
        return (0.9, -0.8, 0.7)

    ctl = _ctl(onnx_bytes, clock=Clock(), fallback=fallback)
    ctl.step(est(), goal())
    assert np.allclose(ctl.step(est(gps_valid=False), goal()), (0.9, -0.8, 0.7))
    assert ctl.status["source"] == "fallback" and ctl.status["fault"] == "gps"
    assert len(calls) == 1

    # no hook -> zeros; a missing `gps_valid` key means no fix
    for bad in (est(gps_valid=False), est(gps_valid=None), est(lat=None)):
        ctl = _ctl(onnx_bytes, clock=Clock())
        ctl.step(est(), goal())
        assert np.allclose(ctl.step(bad, goal()), 0.0)
        assert ctl.status["source"] == "fallback"


def test_fallback_output_is_slewed_too(onnx_bytes):
    ctl = _ctl(onnx_bytes, clock=Clock(), fallback=lambda e, g: (1.0, 1.0, 1.0),
               safety={"rate_limit_per_s": 2.0})
    ctl.step(est(), goal())
    prev = np.array(ctl._last_action)
    out = np.array(ctl.step(est(gps_valid=False), goal()))
    # the handover is rate-limited, not a step to the PID value
    assert np.abs(out - prev).max() <= 2.0 * 0.1 + 1e-9
    assert np.abs(out).max() < 1.0 - 1e-9


def test_sensor_validity_semantics(onnx_bytes):
    # explicit `sensor_ok` (CONTRACT §4.4) wins over the per-sensor flags
    ctl = _ctl(onnx_bytes, clock=Clock())
    ctl.step(est(), goal())
    ctl.step(est(sensor_ok=False), goal())
    assert ctl.status["source"] == "fallback" and ctl.status["fault"] == "sensor"

    # derived from imu_valid & mag_valid when the key is absent
    for bad in (est(imu_valid=False), est(mag_valid=False)):
        ctl = _ctl(onnx_bytes, clock=Clock())
        ctl.step(est(), goal())
        ctl.step(bad, goal())
        assert ctl.status["fault"] == "sensor"


def test_legacy_has_fix_is_not_used(onnx_bytes):
    # INSPECTION H9 / CONTRACT §4.4: gps_valid decides, never the legacy has_fix
    ctl = _ctl(onnx_bytes, clock=Clock())
    ctl.step(est(has_fix=False, gps_valid=True), goal())
    assert ctl.status["source"] == "policy" and ctl.status["fault"] is None


def test_non_finite_policy_output_is_zeroed(onnx_bytes):
    for out in ([[np.nan, 0.1, 0.2]], [[0.1, np.inf, 0.2]], [[0.1, 0.2]]):
        ctl = _ctl(session=FakeSession(out))
        assert np.allclose(ctl.step(est(), goal()), 0.0)
        assert ctl.status["source"] == "zero" and ctl.status["fault"] == "nan"


def test_history_is_reset_after_a_fault(onnx_bytes, ort_session):
    rec = RecordingSession(ort_session)
    ctl = _ctl(session=rec)
    ctl.step(est(), goal())
    ctl.step(est(east=2.0), goal())
    ctl.step(est(gps_valid=False), goal())            # fallback, buffer discarded
    ctl.step(est(), goal())
    obs = rec.obs[-1][0]
    assert np.array_equal(obs[:RAW_DIM], obs[-RAW_DIM:])


def test_no_model_raises():
    with pytest.raises(ValueError):
        RLController()


# --------------------------------------------------- (d) offline replay, exit 0
REPLAY_HEADER = ("t,lat,lon,heading_deg,yaw_rate_deg_s,speed_mps,course_deg,"
                 "gps_valid,imu_valid,mag_valid,est_age_s,goal_lat,goal_lon,mode\n")


def write_log(path, n=20):
    glat, glon = enu_to_latlon(20.0, 5.0, LAT0, LON0)
    rows = [REPLAY_HEADER]
    for i in range(n):
        lat, lon = enu_to_latlon(0.3 * i, 0.0, LAT0, LON0)
        rows.append(
            f"{0.1 * i:.2f},{lat:.9f},{lon:.9f},10.0,1.0,1.0,15.0,"
            f"{0 if i == 3 else 1},1,1,{0.9 if i == 7 else 0.0:.2f},"
            f"{glat:.9f},{glon:.9f},waypoint\n")
    path.write_text("".join(rows), encoding="utf-8")
    return path


def test_infer_replay_runs_on_a_logged_csv(tmp_path, onnx_bytes):
    csv_path = write_log(tmp_path / "log.csv")
    onnx_path = tmp_path / "policy.onnx"
    onnx_path.write_bytes(onnx_bytes)

    proc = subprocess.run(
        [sys.executable, "-m", "runtime.infer_replay",
         "--csv", str(csv_path), "--onnx", str(onnx_path)],
        cwd=str(DRL), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "steps: 20" in out
    assert "surge" in out and "yaw" in out and "min" in out and "max" in out
    # row 3 has no GPS fix, row 7 carries a 0.9 s old sample -> watchdog
    assert "fallback=2" in out and "gps=1" in out and "watchdog=1" in out


def test_infer_replay_rejects_a_missing_csv(tmp_path, onnx_bytes):
    onnx_path = tmp_path / "policy.onnx"
    onnx_path.write_bytes(onnx_bytes)
    assert replay_main(["--csv", str(tmp_path / "nope.csv"),
                        "--onnx", str(onnx_path)]) == 2


# ------------------------------------------------ (e) runtime stays torch-free
def test_runtime_imports_without_torch():
    code = (
        "import sys\n"
        "import runtime.obs_np, runtime.policy, runtime.infer_replay\n"
        "assert 'torch' not in sys.modules, [m for m in sys.modules if 'torch' in m]\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(DRL),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
