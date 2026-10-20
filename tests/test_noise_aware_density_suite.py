"""Smoke tests for the known-noise conditional-density suite."""

from __future__ import annotations

from suites.tabular.noise_aware_density import NoiseAwareDensitySuite
from tools._suite_registry import ALL_SUITES

from harness.runners import HarnessConfig


def test_registered() -> None:
    assert ALL_SUITES["noise_aware_density"] is NoiseAwareDensitySuite


def test_smoke_scalar_and_multi() -> None:
    suite = NoiseAwareDensitySuite()
    cfg = HarnessConfig(seed=0, epochs=1, smoke=True)
    for dataset in suite.supported_datasets:
        rows = suite.run(dataset, cfg)
        assert rows
        for row in rows:
            assert row["Dataset"] == dataset
            assert row["Method"].startswith("noise_aware_density/")
            assert row.get("Notes")
            assert row.get("RMSE") is not None
