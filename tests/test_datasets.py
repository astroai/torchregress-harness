"""Tests for the dataset registry."""

import pytest

from harness.datasets import (
    DATASET_REGISTRY,
    DatasetInfo,
    list_datasets,
    load_dataset,
)


def test_registry_has_expected_datasets() -> None:
    expected = [
        "california_housing",
        "diabetes",
        "openml_year",
        "synthetic_hetero",
        "synthetic_outliers",
        "synthetic_tweedie",
        "synthetic_censored",
        "synthetic_eiv",
    ]
    for name in expected:
        assert name in DATASET_REGISTRY, f"Missing dataset: {name}"


def test_load_dataset_returns_splits() -> None:
    splits = load_dataset("synthetic_hetero", n_samples=200)
    assert "X_train" in splits
    assert "X_test" in splits
    assert "y_train" in splits
    assert "y_test" in splits
    assert splits["X_train"].shape[0] + splits["X_test"].shape[0] == 200


def test_load_dataset_unknown_raises() -> None:
    with pytest.raises(KeyError):
        load_dataset("nonexistent_dataset")


def test_list_datasets_all() -> None:
    all_datasets = list_datasets()
    assert len(all_datasets) >= 5
    assert all(isinstance(d, DatasetInfo) for d in all_datasets)


def test_list_datasets_by_category() -> None:
    tabular = list_datasets(category="tabular")
    synthetic = list_datasets(category="synthetic")
    assert len(tabular) >= 2
    assert len(synthetic) >= 3


# ===========================================================================
# End-to-end PIT diagnostics wiring (Proposal #9 + #10)
# ===========================================================================
#
# Verifies that the DistributionalBinsSuite.evaluate() path calls
# compute_pit_values + pit_uniformity_tests and merges ``PIT_Mean``,
# ``PIT_Std``, ``PIT_KS``, ``PIT_CvM`` into the suite row dict.
# Without this wiring, the benchmark_pit_ensemble.py script would parse
# rows that contain only ``Coverage``/``Width``/``IntervalScore``.


def test_distributional_bins_emits_pit_diagnostics() -> None:
    """End-to-end: DistributionalBinsSuite must include PIT_* metrics in /dev_summary rows.

    Runs a 3-epoch smoke bench on ``synthetic_hetero`` (fastest dataset
    in the suite's list) and confirms that NF + NF_Ensemble_K5 + MDN
    rows carry the four calibration diagnostics.
    """
    from harness.runners import HarnessConfig
    from suites.tabular.distributional_bins import DistributionalBinsSuite

    suite = DistributionalBinsSuite()
    cfg = HarnessConfig(seed=42, epochs=3, smoke=True)
    rows = suite.run("synthetic_hetero", config=cfg)

    # Find dev_summary rows for NormalizingFlow and NF_Ensemble_K5
    nf_summary = None
    ens_summary = None
    for r in rows:
        method_full = r.get("Method", "")
        if method_full == "distributional_bins/NormalizingFlow/dev_summary":
            nf_summary = r
        elif method_full == "distributional_bins/NF_Ensemble_K5/dev_summary":
            ens_summary = r

    assert nf_summary is not None, "NF /dev_summary row not produced"
    assert ens_summary is not None, "NF_Ensemble_K5 /dev_summary row not produced"

    # The /dev_summary row from evaluate() inherits CoverageDeviationAbsMean
    # but PIT_* come from per-method evaluate() rather than the bin
    # aggregation.  Verify by checking the per-method source row.
    nf_method_row = next(
        r for r in rows if r.get("Method") == "distributional_bins/NormalizingFlow"
    )
    assert "PIT_Mean" in nf_method_row, (
        "DistributionalBinsSuite.evaluate() must emit PIT_Mean (Proposal #9 wiring regression)"
    )
    assert "PIT_Std" in nf_method_row
    assert "PIT_KS" in nf_method_row
    assert "PIT_CvM" in nf_method_row

    # Ensemble's PIT diagnostics should also be present.
    ens_method_row = next(
        r for r in rows if r.get("Method") == "distributional_bins/NF_Ensemble_K5"
    )
    assert "PIT_Mean" in ens_method_row
    assert "PIT_KS" in ens_method_row
