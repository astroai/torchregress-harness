"""Tests for the astronomical flow empirical-Bayes EIV suite."""

from __future__ import annotations

import time

import numpy as np
from suites.tabular.astro_flow_eiv import (
    DATASET_NAME,
    TRUE_SLOPE,
    AstroFlowEIVSuite,
    _gauss_hermite,
    _generate_splits,
)

from harness.runners import HarnessConfig


def test_gauss_hermite_rule_normalisation() -> None:
    """The GH rule must integrate a unit Gaussian exactly (the EIV core maths)."""
    nodes, weights = _gauss_hermite(32)
    assert np.isclose(np.sum(weights), np.sqrt(np.pi), rtol=1e-12)
    # (1/sqrt(pi)) sum_j w_j f(sqrt(2) t_j) with f(x)=exp(-x^2/2) is
    # int N(x;0,1) exp(-x^2/2) dx = 1/sqrt(2).
    f_at_scaled_nodes = np.exp(-(nodes**2))
    integral = float(np.sum(weights * f_at_scaled_nodes) / np.sqrt(np.pi))
    assert np.isclose(integral, 1.0 / np.sqrt(2.0), rtol=1e-10)


def test_astro_flow_eiv_metadata() -> None:
    suite = AstroFlowEIVSuite()
    assert suite.name == "astro_flow_eiv"
    assert isinstance(suite.description, str) and suite.description
    assert suite.supported_datasets == [DATASET_NAME]
    assert suite.methods() == ["OLS", "WLS", "ForwardInversion", "FlowEB"]
    assert suite.library_name("OLS") == "numpy"
    assert suite.library_name("WLS") == "numpy"
    assert suite.library_name("ForwardInversion") == "torchregress"
    assert suite.library_name("FlowEB") == "harness (zuko)"
    assert suite.library_version("OLS") == np.__version__
    for method in suite.methods():
        assert suite.method_notes(method), method
    assert "simplified" in suite.method_notes("FlowEB").lower()
    assert TRUE_SLOPE > 0


def test_astro_flow_eiv_generator_is_truly_extrapolating() -> None:
    """The OOD split must be outside the training support, with correlated errors."""
    cfg = HarnessConfig(seed=0, epochs=2, smoke=True, batch_size=64)
    splits = _generate_splits(cfg)
    x_train = np.asarray(splits["X_train_true"]).reshape(-1)
    x_ood = np.asarray(splits["X_test_ood_true"]).reshape(-1)
    sigma_x = np.asarray(splits["sigma_x_train"]).reshape(-1)
    sigma_y = np.asarray(splits["sigma_y_train"]).reshape(-1)

    assert x_ood.min() > x_train.max()
    assert splits["X_train"].shape == (240, 1)
    assert splits["X_test"].shape == (80, 1)
    assert splits["X_test_ood"].shape == (60, 1)
    # The paper's realistic-error premise: errors correlate with the latent x.
    assert np.corrcoef(x_train, sigma_x)[0, 1] > 0.8
    assert np.corrcoef(x_train, sigma_y)[0, 1] < -0.8

    # Same seed -> same design (deterministic suite-generated data).
    again = _generate_splits(cfg)
    np.testing.assert_array_equal(splits["X_train"], again["X_train"])
    np.testing.assert_array_equal(splits["X_test_ood"], again["X_test_ood"])


def test_astro_flow_eiv_smoke_run_is_finite_and_fast() -> None:
    """Metadata plus the requested tiny smoke protocol: finite metrics, <40 s."""
    cfg = HarnessConfig(seed=0, epochs=2, smoke=True, batch_size=64)
    suite = AstroFlowEIVSuite()

    start = time.perf_counter()
    rows = suite.run(DATASET_NAME, config=cfg)
    elapsed = time.perf_counter() - start

    assert elapsed < 40.0, f"smoke run took {elapsed:.1f}s"
    assert len(rows) == len(suite.methods())
    by_method = {row["Method"].split("/")[-1]: row for row in rows}
    assert set(by_method) == set(suite.methods())

    id_keys = (
        "RMSE",
        "MAE",
        "NLL",
        "CRPS",
        "Coverage",
        "Slope",
        "SlopeBias",
        "Attenuation",
        "Slope_Effective",
    )
    ood_keys = (
        "RMSE_OOD",
        "MAE_OOD",
        "NLL_OOD",
        "CRPS_OOD",
        "Coverage_OOD",
        "Slope_OOD",
        "SlopeBias_OOD",
        "Attenuation_OOD",
    )
    for method in suite.methods():
        row = by_method[method]
        assert "failed:" not in str(row.get("Notes", "")).lower(), (method, row["Notes"])
        assert "skipped:" not in str(row.get("Notes", "")).lower(), (method, row["Notes"])
        for key in id_keys + ood_keys:
            assert key in row, (method, key)
            assert np.isfinite(row[key]), (method, key, row[key])
        assert 0.0 <= row["Coverage"] <= 1.0
        assert 0.0 <= row["Coverage_OOD"] <= 1.0

    # FlowEB additionally reports the exact mixture NLL on both splits.
    assert np.isfinite(by_method["FlowEB"]["NLL_Mixture"])
    assert np.isfinite(by_method["FlowEB"]["NLL_Mixture_OOD"])
