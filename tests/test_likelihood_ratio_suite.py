"""Tests for the likelihood-ratio / SBI parity suite and its synthetic tasks."""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular.likelihood_ratio import LikelihoodRatioSuite
from tools._suite_registry import ALL_SUITES

from harness.datasets import load_dataset
from harness.runners import HarnessConfig
from harness.synthetic_sbi import (
    BimodalTask,
    GaussianLinearTask,
    make_bimodal_splits,
    reference_samples,
    task_from_splits,
)


def test_likelihood_ratio_suite_registered() -> None:
    assert ALL_SUITES["likelihood_ratio"] is LikelihoodRatioSuite
    suite = LikelihoodRatioSuite()
    assert suite.name == "likelihood_ratio"
    assert suite.description
    assert suite.methods() == list(dict.fromkeys(suite.methods()))
    assert "sklearn_NRE" in suite.methods()
    assert suite.library_name("sklearn_NRE") == "sklearn"
    assert suite.library_name("torchregress_NRE") == "torchregress"
    assert set(suite.supported_datasets) == {
        "synthetic_sbi_gaussian_linear",
        "synthetic_sbi_bimodal",
    }


def test_synthetic_sbi_datasets_register_and_load() -> None:
    for name in ("synthetic_sbi_gaussian_linear", "synthetic_sbi_bimodal"):
        splits = load_dataset(name, n_samples=64, n_observations=2, n_sbc_simulations=8, seed=3)
        assert splits["X_train"].shape == (64, 4 if "gaussian" in name else 2)
        assert splits["y_train"].shape == (64, 2)
        assert splits["X_test"].shape == (2, splits["X_train"].shape[1])
        assert splits["y_test"].shape == (2, 2)
        assert splits["n_targets"] == 2
        assert splits["_skip_y_standardization"] is True
        assert splits["sbc_x"].shape[0] == 8
        assert splits["sbc_theta"].shape == (8, 2)
        task = task_from_splits(splits)
        assert task is not None


def test_gaussian_linear_reference_posterior_matches_analytic() -> None:
    splits = load_dataset(
        "synthetic_sbi_gaussian_linear",
        n_samples=200,
        n_observations=2,
        n_sbc_simulations=10,
        seed=5,
    )
    task = task_from_splits(splits)
    assert isinstance(task, GaussianLinearTask)
    x = np.asarray(splits["X_test"][0], dtype=np.float64)
    mean, cov = task.posterior_mean_cov(x)
    rng = np.random.default_rng(0)
    samples = reference_samples(task, x, rng, 4000)
    # Rejection sampling from the prior-truncated posterior: mean within MC error.
    assert samples.shape == (4000, 2)
    # Prior truncation at +/-2 shifts the mean slightly from the untruncated value.
    assert np.allclose(samples.mean(axis=0), mean, atol=0.1)
    assert np.all((samples >= task.prior_low) & (samples <= task.prior_high))

    weights = task.reference_grid_weights(
        np.stack(np.meshgrid(np.linspace(-2, 2, 21), np.linspace(-2, 2, 21)), -1).reshape(-1, 2),
        x,
    )
    assert weights.shape == (441,)
    assert weights.sum() == pytest.approx(1.0)


def test_bimodal_reference_posterior_is_bimodal_and_normalized() -> None:
    splits = make_bimodal_splits(n_samples=100, n_observations=1, n_sbc_simulations=8, seed=7)
    task = task_from_splits(splits)
    assert isinstance(task, BimodalTask)
    x = np.asarray(splits["X_test"][0], dtype=np.float64)
    rng = np.random.default_rng(1)
    samples = reference_samples(task, x, rng, 5000)
    assert samples.shape == (5000, 2)
    # Bimodal marginal: both signs should be populated and the mean near zero.
    assert np.all(samples >= task.prior_low)
    assert np.all(samples <= task.prior_high)
    for dim in range(2):
        assert np.mean(samples[:, dim] > 0.3) > 0.1
        assert np.mean(samples[:, dim] < -0.3) > 0.1
    grid = np.stack(np.meshgrid(np.linspace(-2, 2, 31), np.linspace(-2, 2, 31)), -1).reshape(-1, 2)
    weights = task.reference_grid_weights(grid, x)
    assert weights.shape == (961,)
    assert np.isclose(weights.sum(), 1.0)


@pytest.mark.parametrize(
    "dataset",
    ["synthetic_sbi_gaussian_linear", "synthetic_sbi_bimodal"],
)
def test_likelihood_ratio_smoke_run(dataset: str) -> None:
    splits = load_dataset(dataset, n_samples=128, n_observations=2, n_sbc_simulations=24, seed=11)
    suite = LikelihoodRatioSuite()
    cfg = HarnessConfig(seed=0, epochs=2, smoke=True, batch_size=64)
    rows = suite.run(dataset, config=cfg, splits=splits)
    assert len(rows) == len(suite.methods())
    by_method = {row["Method"].split("/")[1]: row for row in rows}
    for method in ("torchregress_NRE", "sklearn_NRE"):
        row = by_method[method]
        assert "failed" not in str(row.get("Notes", "")), row.get("Notes")
        assert "skipped" not in str(row.get("Notes", "")), row.get("Notes")
        for metric in ("PostMeanRMSE", "W1_mean", "Coverage95", "train_s", "eval_s"):
            assert row[metric] is not None and np.isfinite(row[metric]), (method, metric)
    # The reproducibility contract: a second identical run gives the same metrics.
    second = suite.run(dataset, config=cfg, splits=splits)
    by_method_2 = {row["Method"].split("/")[1]: row for row in second}
    assert by_method_2["torchregress_NRE"]["PostMeanRMSE"] == pytest.approx(
        by_method["torchregress_NRE"]["PostMeanRMSE"], rel=1e-6
    )
