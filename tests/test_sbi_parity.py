"""Focused contracts for SBI task expansion and calibration diagnostics."""

import numpy as np
from suites.tabular.sbi_parity import SBIParitySuite

from harness.datasets import DATASET_REGISTRY
from harness.runners import HarnessConfig


def test_sbi_parity_registers_stage_three_tasks() -> None:
    expected = {
        "sbibm_gaussian_linear",
        "sbibm_gaussian_mixture",
        "sbibm_bernoulli_glm",
    }
    suite = SBIParitySuite()
    assert expected <= set(DATASET_REGISTRY)
    assert expected <= set(suite.supported_datasets)


def test_sbc_diagnostics_use_held_out_truth_and_posterior_draws(monkeypatch) -> None:
    suite = SBIParitySuite()
    draws = np.tile(np.array([[0.0, 0.0], [0.4, 0.4], [0.6, 0.6], [1.0, 1.0]]), (50, 1))
    monkeypatch.setattr(suite, "_posterior_samples", lambda *args, **kwargs: draws)

    diagnostics = suite._sbc_diagnostics(
        {},
        {
            "sbc_x": np.zeros((2, 2), dtype=np.float32),
            "sbc_theta": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        },
        HarnessConfig(smoke=True),
    )

    assert diagnostics["SBC_Coverage95"] == 1.0
    assert diagnostics["SBC_RankMean"] == 0.5
    assert 0.0 <= diagnostics["SBC_RankKS"] <= 1.0
