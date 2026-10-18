"""Tests for the PRIME / ConR / group-DRO representation benchmark suite."""

from __future__ import annotations

import numpy as np
from suites.tabular.imbalanced_representation import ImbalancedRepresentationSuite
from tools._suite_registry import ALL_SUITES

from harness.datasets import load_dataset
from harness.runners import HarnessConfig


def test_imbalanced_representation_registered() -> None:
    assert ALL_SUITES["imbalanced_representation"] is ImbalancedRepresentationSuite
    suite = ImbalancedRepresentationSuite()
    assert suite.supported_datasets == ["stress_imbalanced", "california_housing"]
    assert {"MSE_PRIME", "MSE_ConR", "MSE_GroupDRO", "VIR"} <= set(suite.methods())


def test_imbalanced_representation_smoke() -> None:
    splits = load_dataset("stress_imbalanced", n_samples=400, seed=0)
    suite = ImbalancedRepresentationSuite()
    rows = suite.run(
        "stress_imbalanced",
        config=HarnessConfig(seed=0, epochs=2, smoke=True, batch_size=64),
        splits=splits,
    )
    assert len(rows) == len(suite.methods())
    for row in rows:
        assert "failed" not in str(row.get("Notes", "")), row.get("Notes")
        assert np.isfinite(row["RMSE"]), row
        assert np.isfinite(row["WorstGroupMAE"]), row
