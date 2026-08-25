"""Regression tests for audit-fix Workstream G wrapper hygiene (WRAP-04/05).

* WRAP-04: MC samplers must return exactly ``(n_test, n_samples)`` — a
  wrong-shaped sampler now raises ``AssertionError`` instead of being
  silently reshaped by heuristics.
* WRAP-05: NIG constraint is a single shared ``_constrain_nig`` helper used
  by both fit and predict, so train/test parameterizations cannot drift.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harness.wrappers import MDNRegressor, NormalizingFlowRegressor, _constrain_nig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


# ---------------------------------------------------------------------------
# WRAP-05 — shared NIG parameter constraint
# ---------------------------------------------------------------------------


def test_constrain_nig_parameter_ranges() -> None:
    raw = torch.tensor([[-2.0, -3.0, -4.0, -5.0], [1.0, 2.0, 3.0, 4.0]])
    gamma, nu, alpha, beta = _constrain_nig(raw)
    # gamma passes through unbounded (raw).
    assert torch.allclose(gamma, raw[:, 0:1])
    # softplus(x) > 0 ⇒ floors are strictly respected.
    assert (nu > 0.0).all()
    assert (alpha > 1.01).all()
    assert (beta > 0.0).all()
    # Asymptotes: large positive raw → softplus ≈ raw (float32 tolerance).
    big = _constrain_nig(torch.tensor([[0.0, 10.0, 10.0, 10.0]]))
    assert big[1].item() == pytest.approx(10.0005, abs=1e-2)
    assert big[2].item() == pytest.approx(11.0005, abs=1e-2)
    assert big[3].item() == pytest.approx(10.0005, abs=1e-2)


# ---------------------------------------------------------------------------
# WRAP-04 — MDN sampler shape contract
# ---------------------------------------------------------------------------


def test_mdn_predict_diagnostics_sample_shape(rng: np.random.Generator) -> None:
    reg = MDNRegressor()
    X = rng.standard_normal((12, 3)).astype(np.float32)
    reg._build_model(in_dim=3, hidden=8)
    reg._inference_seed = 0

    diag = reg._predict_diagnostics(X)
    assert diag["samples"].shape == (12, 256)
    assert diag["mean"].shape == (12,)
    assert diag["std"].shape == (12,)


def test_mdn_bad_sampler_shape_raises(rng: np.random.Generator) -> None:
    reg = MDNRegressor()
    X = rng.standard_normal((12, 3)).astype(np.float32)
    reg._build_model(in_dim=3, hidden=8)
    reg._inference_seed = 0

    # Sampler returns the WRONG number of samples — must trip the contract.
    reg._loss_fn.sample = lambda out, n_samples=256: torch.zeros(4, 12, 1)
    with pytest.raises(AssertionError, match="sampler returned"):
        reg._predict_diagnostics(X)


def test_mdn_bad_batch_dimension_raises(rng: np.random.Generator) -> None:
    """A transposed-looking (S≠n_test, n_test) layout must not be swapped in."""
    reg = MDNRegressor()
    X = rng.standard_normal((9, 3)).astype(np.float32)
    reg._build_model(in_dim=3, hidden=8)
    reg._inference_seed = 0

    reg._loss_fn.sample = lambda out, n_samples=256: torch.zeros(5, 9, 1)
    with pytest.raises(AssertionError, match="sampler returned"):
        reg._predict_diagnostics(X)


# ---------------------------------------------------------------------------
# WRAP-04 — NormalizingFlow sampler shape contract
# ---------------------------------------------------------------------------


def test_flow_good_and_bad_sampler_shapes(rng: np.random.Generator) -> None:
    flow = NormalizingFlowRegressor()
    X = rng.standard_normal((7, 2)).astype(np.float32)
    flow._build_model(in_dim=2, hidden=8)
    flow._inference_seed = 0

    # Good path: (n_test, NF_N_SAMPLES).
    diag = flow._predict_diagnostics(X)
    assert diag["samples"].shape == (7, 256)

    # Bad path: wrong sample count must raise, not silently reshape.
    flow._loss.sample = lambda context, n_samples=256: torch.zeros(7, 5)
    with pytest.raises(AssertionError, match="sampler returned"):
        flow._predict_diagnostics(X)
