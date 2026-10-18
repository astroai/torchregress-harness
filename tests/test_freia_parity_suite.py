"""Tests for the FrEIA/cINN conditional-density parity suite."""

from __future__ import annotations

import numpy as np
import pytest
from tools._suite_registry import ALL_SUITES

from harness.datasets import load_dataset
from harness.runners import HarnessConfig

pytest.importorskip("FrEIA")

from suites.tabular.freia_parity import FreiaParitySuite  # noqa: E402


def test_freia_parity_registered() -> None:
    assert ALL_SUITES["freia_parity"] is FreiaParitySuite
    suite = FreiaParitySuite()
    assert suite.library_name("FrEIA_cINN") == "FrEIA"
    assert {"GaussianNLL", "NSF", "FrEIA_cINN"} == set(suite.methods())
    assert "diabetes" in suite.supported_datasets


def test_freia_parity_smoke() -> None:
    splits = load_dataset("diabetes", standardize_targets=True)
    suite = FreiaParitySuite()
    rows = suite.run(
        "diabetes",
        config=HarnessConfig(seed=0, epochs=2, smoke=True, batch_size=64),
        splits=splits,
    )
    by_method = {row["Method"].split("/")[1]: row for row in rows}
    for method in ("GaussianNLL", "NSF", "FrEIA_cINN"):
        row = by_method[method]
        assert "failed" not in str(row.get("Notes", "")), (method, row.get("Notes"))
        for metric in ("RMSE", "NLL", "CRPS"):
            assert np.isfinite(row[metric]), (method, metric, row[metric])
