"""Rows that give suite coverage to 0.3.0 exports (smoke-sized, synthetic data).

* cell_counting ``CNN+PoissonGaussianConv`` -> ``PoissonGaussianLoss``
* distributional_bins / ordinal ``CumulativeCDF`` -> ``CumulativeCDFLoss``,
  ``binned_cdf``, ``crps_from_binned_cdf``
* robust_regression ``IRLS_HuberTukey`` (``IRLSContinuationStage``,
  ``IRLSBreakdownError``) and ``IRLS_ReweightFixed`` (``reweight_fixed_predictions``)
* sklearn_baselines ``FTTransformer`` and likelihoods ``FTT_GaussianNLL``
  -> ``FTTransformer``

(imbalanced_representation rows are covered in
``tests/test_imbalanced_representation_suite.py``.)
"""

from __future__ import annotations

import numpy as np
import pytest
from suites._ft_transformer import token_budget_epochs
from suites.image.cell_counting import CellCountingSuite
from suites.tabular.distributional_bins import DistributionalBinsSuite
from suites.tabular.likelihoods import LikelihoodsSuite
from suites.tabular.ordinal import OrdinalSuite
from suites.tabular.robust_regression import RobustRegressionSuite
from suites.tabular.sklearn_baselines import SklearnBaselinesSuite

from harness.datasets import load_dataset
from harness.devices import current_device
from harness.runners import HarnessConfig


def _ok(row: dict) -> bool:
    return not str(row.get("Notes", "")).startswith("failed")


def _only(suite: object, methods: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(suite, "methods", lambda: list(methods))


def _tabular(n: int = 600, d: int = 5, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * np.sin(2 * X[:, 1]) + 0.3 * rng.normal(size=n)).astype(np.float32)
    k = int(0.8 * n)
    return {"X_train": X[:k], "y_train": y[:k], "X_test": X[k:], "y_test": y[k:]}


def test_cell_counting_poisson_gaussian_conv_row(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    n, h = 160, 32
    y = rng.poisson(rng.uniform(0.5, 30, size=n)).astype(np.float32)
    X = np.zeros((n, 1, h, h), np.float32)
    for i in range(n):
        idx = rng.integers(0, h, size=(int(y[i]), 2))
        X[i, 0, idx[:, 0], idx[:, 1]] += 1.0
    splits = {"X_train": X[:120], "y_train": y[:120], "X_test": X[120:], "y_test": y[120:]}
    suite = CellCountingSuite()
    _only(suite, ["CNN+PoissonGaussianConv"], monkeypatch)
    cfg = HarnessConfig(seed=0, epochs=3, batch_size=32, standardize_targets=False)
    (row,) = suite.run("synthetic_counts", cfg, splits=splits)
    assert _ok(row), row
    for key in ("RMSE", "NLL", "CRPS", "ExactNLL"):
        assert np.isfinite(row[key]), row


def test_distributional_bins_cumulative_cdf_row(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = DistributionalBinsSuite()
    _only(suite, ["Split_CP", "CumulativeCDF"], monkeypatch)
    rows = suite.run(
        "synthetic_hetero",
        HarnessConfig(seed=0, epochs=3, smoke=True),
        splits=load_dataset("synthetic_hetero"),
    )
    row = next(r for r in rows if r["Method"] == "distributional_bins/CumulativeCDF")
    assert _ok(row), row
    assert np.isfinite(row["CRPS"]) and row["CRPS"] > 0
    assert 0.0 <= row["Coverage"] <= 1.0
    assert "PIT_KS" in row


def test_ordinal_cumulative_cdf_row(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = OrdinalSuite()
    _only(suite, ["CumulativeCDF"], monkeypatch)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    y = np.digitize(X[:, 0] + 0.3 * rng.normal(size=600), [-1.0, -0.3, 0.3, 1.0])
    splits = {
        "X_train": X[:400],
        "y_train": y[:400].astype(np.float32),
        "X_test": X[400:],
        "y_test": y[400:].astype(np.float32),
    }
    cfg = HarnessConfig(seed=0, epochs=5, batch_size=64, smoke=True, standardize_targets=False)
    (row,) = suite.run("synthetic_ordinal", cfg, splits=splits)
    assert _ok(row), row
    assert row["Accuracy"] > 0.4  # chance is ~0.2 for 5 classes
    assert row["MAE"] < 1.0


def test_robust_regression_irls_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = RobustRegressionSuite()
    _only(
        suite,
        ["MSE_Linear", "IRLS_HuberTukey", "statsmodels_RLM_HuberTukey", "IRLS_ReweightFixed"],
        monkeypatch,
    )
    rows = {r["Method"].split("/", 1)[1]: r for r in suite.run("synthetic_outliers")}
    assert all(_ok(r) for r in rows.values()), rows
    tukey, sm_tukey = rows["IRLS_HuberTukey"], rows["statsmodels_RLM_HuberTukey"]
    assert tukey["IRLS_Iterations"] > 2
    assert abs(tukey["MAE"] - sm_tukey["MAE"]) < 0.05 * sm_tukey["MAE"]
    fixed = rows["IRLS_ReweightFixed"]
    # Reweighting fixed predictions does not refit: same predictions as least squares.
    assert fixed["MAE"] == pytest.approx(rows["MSE_Linear"]["MAE"], rel=1e-6)
    assert 0.0 < fixed["Reweight_ZeroWeightFrac"] < fixed["Reweight_DownweightedFrac"] < 0.5


def test_sklearn_baselines_ft_transformer_row(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = SklearnBaselinesSuite()
    _only(suite, ["FTTransformer"], monkeypatch)
    (row,) = suite.run("synthetic", HarnessConfig(seed=0, epochs=2), splits=_tabular())
    assert _ok(row), row
    assert np.isfinite(row["RMSE"])
    assert row["FT_EpochsRun"] <= row["FT_EpochsBudget"] == 2
    assert row["Device"] == str(current_device())


def test_likelihoods_ft_transformer_row(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = LikelihoodsSuite()
    _only(suite, ["FTT_GaussianNLL"], monkeypatch)
    (row,) = suite.run("synthetic", HarnessConfig(seed=0, epochs=2), splits=_tabular(800))
    assert _ok(row), row
    for key in ("NLL", "CRPS", "Coverage"):
        assert np.isfinite(row[key]), row


def test_token_budget_caps_wide_tables_only() -> None:
    import torch

    cpu = torch.device("cpu")
    assert token_budget_epochs(60, 16_000, 10, cpu) == 60
    assert token_budget_epochs(60, 16_000, 987, cpu) == 12
    assert token_budget_epochs(60, 1_200_000, 71, torch.device("cuda")) == 60
    assert token_budget_epochs(60, 10**9, 10**4, cpu) == 1
