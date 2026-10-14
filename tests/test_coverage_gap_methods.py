"""Smoke contracts for methods added to existing suites for the 0.3.0 coverage gaps.

* label_shift: test_time BBSE / EM target-prior rows and ShiftFactoredTransport rows
* ensembles: HeteroBatchEnsemble (BatchEnsembleLinear stack) and VariationalMLP
* age_regression: CNN+RankNContrast on the synthetic image fallback
"""

from __future__ import annotations

import numpy as np
import pytest
from suites.image.age_regression import AgeRegressionSuite
from suites.tabular.ensembles import EnsemblesSuite
from suites.tabular.label_shift import (
    LabelShiftSuite,
    _finite_interval_stats,
    prior_ratio_weights,
)

from harness.datasets import load_dataset
from harness.runners import HarnessConfig


def _ok(row: dict) -> bool:
    return not str(row.get("Notes", "")).startswith("failed")


# ---------------------------------------------------------------------------
# label_shift
# ---------------------------------------------------------------------------


def _shifted_problem(seed: int = 0) -> dict[str, np.ndarray]:
    """Gaussian regression with a label-shifted target and an informative predictor."""
    rng = np.random.default_rng(seed)
    y_src = rng.normal(size=3000)
    p_src = y_src + 0.3 * rng.normal(size=3000)
    y_tgt = rng.normal(loc=0.8, size=2000)  # shifted label distribution
    p_tgt = y_tgt + 0.3 * rng.normal(size=2000)
    return {"y_train": y_src, "p_src": p_src, "p_tgt": p_tgt, "y_tgt": y_tgt}


@pytest.mark.parametrize("method", ["TT_BBSE", "TT_EM"])
def test_prior_ratio_weights_track_the_shift(method: str) -> None:
    w, extra = prior_ratio_weights(method, **_shifted_problem())
    assert w.mean() == pytest.approx(1.0)
    d = _shifted_problem()
    # Up-weights the high-label rows the target over-represents.
    assert w[d["y_tgt"] > 1.0].mean() > w[d["y_tgt"] < -0.5].mean()
    # Estimated target prior is closer to the oracle than "no shift" (TV 0.32).
    # BBSE only needs a confusion matrix; EM assumes calibrated posteriors and
    # the Gaussian-bin probabilities are not, so it stops at a biased fixed point.
    assert extra["PriorTV"] < (0.15 if method == "TT_BBSE" else 0.30)


def test_finite_interval_stats_handle_inf() -> None:
    out = _finite_interval_stats(
        np.array([0.0, -np.inf]), np.array([1.0, np.inf]), np.array([0.5, 9.0])
    )
    assert out == {"ShiftCoverage": 1.0, "ShiftWidth": 1.0, "InfiniteIntervalFrac": 0.5}


def test_label_shift_new_rows_smoke() -> None:
    suite = LabelShiftSuite()
    methods = ["TT_EM", "ShiftFactoredTransport", "ShiftFactoredTransport+Conformal"]
    suite.methods = lambda: methods  # type: ignore[method-assign]
    splits = load_dataset("diabetes", n_samples=200)
    rows = {
        r["Method"].split("/")[1]: r
        for r in suite.run("diabetes", HarnessConfig(seed=0, smoke=True), splits=splits)
    }
    assert set(rows) == set(methods)
    for name, row in rows.items():
        assert _ok(row), (name, row["Notes"])
    assert np.isfinite(rows["TT_EM"]["WeightRMSE"])
    for name in ("ShiftFactoredTransport", "ShiftFactoredTransport+Conformal"):
        assert 0.0 <= rows[name]["ShiftCoverage"] <= 1.0
        assert rows[name]["InfiniteIntervalFrac"] >= 0.0
        assert "TransportApplied" in rows[name]


# ---------------------------------------------------------------------------
# ensembles
# ---------------------------------------------------------------------------


def test_rank1_and_variational_ensembles_smoke() -> None:
    suite = EnsemblesSuite()
    suite.methods = lambda: ["HeteroBatchEnsemble", "VariationalMLP"]  # type: ignore[method-assign]
    splits = load_dataset("diabetes", n_samples=200)
    rows = suite.run("diabetes", HarnessConfig(seed=0, smoke=True, epochs=2), splits=splits)
    assert len(rows) == 2
    for row in rows:
        assert _ok(row), row["Notes"]
        assert np.isfinite(row["NLL"]) and 0.0 <= row["Coverage"] <= 1.0
        assert row["EpistemicVar"] >= 0.0 and row["AleatoricVar"] > 0.0


# ---------------------------------------------------------------------------
# age_regression
# ---------------------------------------------------------------------------


def test_rank_n_contrast_smoke_on_synthetic_fallback() -> None:
    suite = AgeRegressionSuite()
    assert "CNN+RankNContrast" in suite.methods()
    suite.methods = lambda: ["CNN+RankNContrast"]  # type: ignore[method-assign]
    splits = load_dataset("synthetic_image_regression", n_samples=120)
    rows = suite.run(
        "synthetic_image_regression", HarnessConfig(seed=0, smoke=True, epochs=2), splits=splits
    )
    assert len(rows) == 1 and _ok(rows[0]), rows[0]["Notes"]
    assert np.isfinite(rows[0]["RMSE"])
