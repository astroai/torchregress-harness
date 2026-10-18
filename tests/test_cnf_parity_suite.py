"""Tests for the Contrastive Normalizing Flow (CNF) reproduction suite."""

from __future__ import annotations

import numpy as np
from suites.tabular.cnf_parity import CNFParitySuite
from tools._suite_registry import ALL_SUITES

from harness.datasets import load_dataset
from harness.runners import HarnessConfig


def test_cnf_parity_registered() -> None:
    assert ALL_SUITES["cnf_parity"] is CNFParitySuite
    suite = CNFParitySuite()
    assert suite.methods() == ["DNN", "NF_NLL", "CNF"]
    assert {"cnf_toy_gaussian", "fair_universe_sample"} == set(suite.supported_datasets)


def test_cnf_parity_toy_smoke() -> None:
    splits = load_dataset("cnf_toy_gaussian", standardize_targets=False)
    suite = CNFParitySuite()
    rows = suite.run(
        "cnf_toy_gaussian",
        config=HarnessConfig(seed=0, epochs=2, smoke=True, batch_size=256),
        splits=splits,
    )
    assert len(rows) == len(suite.methods())
    for row in rows:
        assert "failed" not in str(row.get("Notes", "")), row.get("Notes")
        for metric in ("Accuracy", "AUC", "LogLoss", "Rotation30_Accuracy"):
            assert np.isfinite(row[metric]), (row["Method"], metric)
