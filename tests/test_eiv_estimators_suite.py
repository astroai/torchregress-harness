"""Tests for the forward-inversion / attenuation-free EIV estimators suite."""

from __future__ import annotations

import numpy as np
from suites.tabular.eiv_estimators import EIVEstimatorsSuite
from tools._suite_registry import ALL_SUITES

from harness.datasets import load_dataset
from harness.runners import HarnessConfig


def test_eiv_estimators_suite_registered() -> None:
    assert ALL_SUITES["eiv_estimators"] is EIVEstimatorsSuite
    suite = EIVEstimatorsSuite()
    assert suite.name == "eiv_estimators"
    assert suite.library_name("sklearn_OLS") == "sklearn"
    assert suite.library_name("SplitMeasurement") == "torchregress"
    assert "ForwardInversion_GN" in suite.methods()
    assert "ForwardInversion_LBFGS" in suite.methods()
    assert set(suite.supported_datasets) == {
        "synthetic_eiv",
        "synthetic_eiv_highnoise",
        "synthetic_eiv_nonlinear",
    }


def test_eiv_estimators_smoke_run() -> None:
    splits = load_dataset("synthetic_eiv", n_samples=320, seed=0)
    suite = EIVEstimatorsSuite()
    cfg = HarnessConfig(seed=0, epochs=2, smoke=True, batch_size=64)
    rows = suite.run("synthetic_eiv", config=cfg, splits=splits)
    assert len(rows) == len(suite.methods())
    by_method = {row["Method"].split("/")[1]: row for row in rows}
    for method in ("MSE", "sklearn_OLS", "SplitMeasurement", "CorrectedScore"):
        row = by_method[method]
        assert np.isfinite(row["RMSE"]), (method, row)
        assert "failed" not in str(row.get("Notes", "")), (method, row.get("Notes"))
    # Forward inversion is a generative MLE: in a 2-epoch smoke run we only
    # require the inversion to produce finite predictions and covariance.
    for method in ("ForwardInversion_GN", "ForwardInversion_LBFGS"):
        row = by_method[method]
        assert np.isfinite(row["RMSE"]), (method, row)
        assert "failed" not in str(row.get("Notes", "")), (method, row.get("Notes"))
