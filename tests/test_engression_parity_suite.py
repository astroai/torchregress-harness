"""Tests for the Engression parity suite.

The metadata/gating tests run even when the optional ``engression`` package is
not installed.  The end-to-end smoke test is skipped in that case and, when it
runs, uses a small split of the smallest supported dataset so it stays well
under the suite's 30 s CI budget.
"""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular.engression_parity import (
    _ENGRESSION_AVAILABLE,
    _N_SAMPLES_SMOKE,
    EngressionParitySuite,
)

from harness.datasets import DATASET_REGISTRY, load_dataset
from harness.runners import HarnessConfig

_REQUIRED_METHODS = {
    "Engression",
    "MSE_MLP",
    "GaussianNLL_MLP",
    "Quantile_MLP",
    "NormalizingFlow_NLL",
}


# ---------------------------------------------------------------------------
# Metadata / registration (new-only files: the shared registry entry is wired
# by the Lead, so this test does not import tools._suite_registry).
# ---------------------------------------------------------------------------


def test_engression_parity_metadata() -> None:
    suite = EngressionParitySuite()
    assert suite.name == "engression_parity"
    assert suite.description
    assert suite.supported_datasets == [
        "stress_noisy_inputs",
        "synthetic_hetero",
        "california_housing",
    ]
    # Existing registry datasets only — no new loaders were added.
    for dataset in suite.supported_datasets:
        assert dataset in DATASET_REGISTRY

    assert _REQUIRED_METHODS.issubset(set(suite.methods()))
    assert suite.library_name("Engression") == "engression"
    assert suite.library_name("MSE_MLP") == "torchregress"
    assert suite.library_name("GaussianNLL_MLP") == "torchregress"

    version = suite.library_version("Engression")
    assert version.startswith("engression=="), version
    if _ENGRESSION_AVAILABLE:
        assert "unknown" not in version

    notes = suite.method_notes("Engression").lower()
    assert "engression" in notes
    assert "energy loss" in notes
    assert "pre-additive-noise" in notes
    assert "2307.00835" in notes
    for method in suite.methods():
        assert suite.method_notes(method), f"missing method notes for {method}"


def test_engression_missing_dependency_is_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The external row raises a clean ImportError when the package is absent.

    ``HarnessRunner.run`` converts ImportError into a "skipped: ..." row, so
    this is the exact gating contract the suite relies on.
    """
    import suites.tabular.engression_parity as module

    monkeypatch.setattr(module, "_ENGRESSION_AVAILABLE", False)
    splits = load_dataset("synthetic_hetero", n_samples=120, standardize_targets=True)
    with pytest.raises(ImportError, match="engression"):
        module.EngressionParitySuite().fit(
            "Engression",
            splits,
            HarnessConfig(seed=0, epochs=1, smoke=True),
        )


# ---------------------------------------------------------------------------
# End-to-end smoke: all five rows fit/predict/evaluate on a small split.
# ---------------------------------------------------------------------------


def test_engression_sample_prediction_contract() -> None:
    """The official ``.sample`` layout is (n_test, response_dim, n_samples).

    The suite squeezes the response axis; getting that layout wrong would
    silently transpose test points and samples, so lock the contract here.
    """
    pytest.importorskip("engression")

    splits = load_dataset("synthetic_hetero", n_samples=160, standardize_targets=True)
    suite = EngressionParitySuite()
    cfg = HarnessConfig(seed=0, epochs=1, smoke=True)
    state = suite.fit("Engression", splits, cfg)
    X_test = np.asarray(splits["X_test"])[:20]
    preds = suite.predict("Engression", state, X_test, cfg)

    assert preds["samples"].shape == (20, _N_SAMPLES_SMOKE), preds["samples"].shape
    for key in ("y_pred", "mean", "std", "lower", "upper"):
        assert preds[key].shape == (20,), (key, preds[key].shape)
    assert np.allclose(preds["mean"], preds["samples"].mean(axis=1))
    assert np.all(preds["lower"] <= preds["upper"])


def test_engression_parity_smoke_run() -> None:
    pytest.importorskip("engression")

    # Smallest supported dataset; a 240-row draw keeps the full fit/predict/
    # evaluate loop fast while smoke=True caps the test split at 50 points.
    splits = load_dataset("synthetic_hetero", n_samples=240, standardize_targets=True)
    suite = EngressionParitySuite()
    rows = suite.run(
        "synthetic_hetero",
        config=HarnessConfig(seed=0, epochs=2, smoke=True, batch_size=64),
        splits=splits,
    )

    by_method = {row["Method"].split("/")[-1]: row for row in rows}
    assert _REQUIRED_METHODS.issubset(set(by_method))
    assert len(rows) == len(suite.methods())

    for method, row in by_method.items():
        notes = str(row.get("Notes", "")).lower()
        assert "failed:" not in notes, (method, row.get("Notes"))
        assert "skipped:" not in notes, (method, row.get("Notes"))
        assert np.isfinite(row["RMSE"]), (method, "RMSE", row["RMSE"])
        assert np.isfinite(row["MAE"]), (method, "MAE", row["MAE"])

    # Engression must report the full metric contract: point error, Gaussian
    # NLL/CRPS from sample moments, 90% interval metrics, PIT and pinball.
    eng = by_method["Engression"]
    assert eng["Library"] == "engression"
    assert eng["LibraryVersion"].startswith("engression==")
    for metric in (
        "RMSE",
        "MAE",
        "NLL",
        "CRPS",
        "Coverage",
        "Width",
        "IntervalScore",
        "Pinball",
        "PIT_KS",
    ):
        assert np.isfinite(eng[metric]), (metric, eng.get(metric))

    # Distributional torchregress baselines expose NLL/CRPS and intervals.
    for method in ("GaussianNLL_MLP", "NormalizingFlow_NLL"):
        row = by_method[method]
        for metric in ("NLL", "CRPS", "Coverage", "Width", "IntervalScore"):
            assert np.isfinite(row[metric]), (method, metric, row.get(metric))

    # Quantile baseline reports its own pinball + interval metrics.
    quantile = by_method["Quantile_MLP"]
    for metric in ("Pinball", "Coverage", "Width", "IntervalScore"):
        assert np.isfinite(quantile[metric]), (metric, quantile.get(metric))
