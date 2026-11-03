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
    assert {
        "MSE_PRIME",
        "MSE_ConR",
        "MSE_UAConR",
        "MSE_GroupDRO",
        "MSE_BMC",
        "MSE_RankSim",
        "VIR",
    } <= set(suite.methods())


def test_imbalanced_representation_smoke() -> None:
    splits = load_dataset("stress_imbalanced", n_samples=400, seed=0)
    suite = ImbalancedRepresentationSuite()
    rows = suite.run(
        "stress_imbalanced",
        config=HarnessConfig(seed=0, epochs=2, smoke=True, batch_size=64),
        splits=splits,
    )
    method_rows = [r for r in rows if r["Method"].split("/", 1)[1] in suite.methods()]
    assert len(method_rows) == len(suite.methods())
    rep_keys = (
        "Rep_neighbour_target_gap",
        "Rep_target_ordering",
        "Rep_ols_RMSE",
        "Rep_knn_RMSE",
    )
    for row in rows:
        assert "failed" not in str(row.get("Notes", "")), row.get("Notes")
    for row in method_rows:
        assert np.isfinite(row["RMSE"]), row
        assert np.isfinite(row["WorstGroupMAE"]), row
        assert all(np.isfinite(row[k]) for k in rep_keys), row
    by_name = {r["Method"]: r for r in rows}
    raw = by_name["imbalanced_representation/rep_geometry/raw_inputs"]
    assert all(np.isfinite(raw[k]) for k in rep_keys), raw
    paired = [r for r in rows if "/paired_abs_error/" in r["Method"]]
    assert len(paired) == 3
    for row in paired:
        assert row["MAE_Diff_Lo"] <= row["MAE_Diff"] <= row["MAE_Diff_Hi"], row
    assert len(rows) == len(method_rows) + 1 + len(paired)
