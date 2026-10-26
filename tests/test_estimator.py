"""Cross-backend and contract tests for common.estimator."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from common.estimator import estimate_body_velocity


def test_numpy_and_torch_inputs_match():
    values = dict(
        sog_mps=np.array([1.0, 2.0, 0.1, 3.0]),
        cog_deg=np.array([0.0, 90.0, 180.0, 350.0]),
        heading_deg=np.array([0.0, 0.0, 10.0, 10.0]),
        dt_s=0.1,
        u_prev=np.array([0.2, -0.3, 0.4, 0.5]),
        v_prev=np.array([-0.1, 0.2, 0.3, -0.4]),
        gps_valid=np.array([True, True, True, False]),
    )
    np_out = estimate_body_velocity(**values)
    torch_out = estimate_body_velocity(
        **{k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
           for k, v in values.items()}
    )
    assert np.allclose(np_out[0], torch_out[0].numpy(), atol=1e-6)
    assert np.allclose(np_out[1], torch_out[1].numpy(), atol=1e-6)


def test_module_has_no_backend_branching_or_imports():
    source = Path(inspect.getfile(estimate_body_velocity)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
    assert not any(
        isinstance(node, ast.Call) and getattr(node.func, "id", None) == "isinstance"
        for node in ast.walk(tree)
    )


def test_known_value_and_decay():
    u, v = estimate_body_velocity(
        2.0, 90.0, 0.0, 0.1, 0.0, 0.0, True,
    )
    assert abs(float(u)) < 1e-8  # east course is pure sway (series error ~1e-10)
    assert float(v) == pytest.approx(2.0, abs=1e-8)

    u, v = estimate_body_velocity(0.1, 0.0, 0.0, 0.25, 0.8, -0.4, True)
    assert float(u) == pytest.approx(0.6, abs=1e-8)
    assert float(v) == pytest.approx(-0.3, abs=1e-8)
