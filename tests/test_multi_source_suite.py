"""Smoke and contract tests for the multi-survey shared-latent suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from suites.tabular.multi_source import (
    _CORR,
    _DIM,
    _K,
    _METHODS_BY_DATASET,
    MultiSourceSuite,
    batched_marginal_nll,
    make_data,
)
from tools._suite_registry import ALL_SUITES

from harness.runners import HarnessConfig


def test_registered() -> None:
    assert ALL_SUITES["multi_source"] is MultiSourceSuite


def test_generator_ground_truth_contract() -> None:
    data = make_data("multisurvey_biased", HarnessConfig(seed=3, smoke=True))
    train = data["train"]
    mask = train["mask"]
    assert train["values"].shape == (len(train["x"]), _K)
    # Missing measurements are NaN (value and reported error), never zero-noise zeros.
    assert np.isnan(train["values"][~mask]).all() and np.isnan(train["sigma"][~mask]).all()
    assert np.isfinite(train["values"][mask]).all() and (train["sigma"][mask] > 0).all()
    assert 0 < mask.mean() < 1
    assert np.linalg.eigvalsh(_CORR).min() > 0 and (_CORR[np.triu_indices(_K, 1)] != 0).any()
    assert np.abs(train["bias"]).max() > 0


def test_batched_reference_matches_library() -> None:
    from torchregress.losses import MarginalizedMultiSourceGaussianNLLLoss, PackedObservations

    test = make_data("multisurvey", HarnessConfig(seed=1, smoke=True))["test"]
    n = 32
    g = torch.Generator().manual_seed(0)
    mean = torch.randn(n, 3, generator=g, dtype=torch.float64)
    lower = torch.eye(3, dtype=torch.float64).expand(n, -1, -1) * 0.5
    jac = torch.zeros(n, _K, 3, dtype=torch.float64)
    jac[:, torch.arange(_K), torch.as_tensor(_DIM)] = 1.0
    mapped = torch.einsum("bkd,bd->bk", jac, mean)
    sigma = torch.from_numpy(test["sigma"][:n]).double()
    cov = sigma[:, :, None] * torch.from_numpy(_CORR) * sigma[:, None, :]
    values = torch.from_numpy(test["values"][:n]).double()
    mask = torch.from_numpy(test["mask"][:n])
    lib = MarginalizedMultiSourceGaussianNLLLoss("none").per_star(
        mean, lower, PackedObservations(values, mapped, jac, cov, mask)
    )
    ref = batched_marginal_nll(mean, lower, values, mapped, jac, cov, mask)
    torch.testing.assert_close(ref, lib, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dataset", list(_METHODS_BY_DATASET))
def test_smoke_rows(dataset: str) -> None:
    suite = MultiSourceSuite()
    rows = suite.run(dataset, HarnessConfig(seed=0, epochs=1, smoke=True))
    assert [r["Method"] for r in rows] == [
        f"multi_source/{m}" for m in _METHODS_BY_DATASET[dataset]
    ]
    for row in rows:
        assert row["Dataset"] == dataset and row["SyntheticFallback"] is False
        assert "failed:" not in row["Notes"], row["Notes"]
        if row["Method"].endswith(("Exactness_LinearGaussian", "SplineBasis_ScipyParity")):
            continue
        assert np.isfinite(row["RMSE"]) and np.isfinite(row["NLL"])
        assert 0.0 <= row["Coverage90"] <= 1.0 and np.isfinite(row["ObsNLL"])
    if dataset == "multisurvey":
        exact = rows[0]
        assert exact["ExactnessPass"] is True, exact
    if dataset == "multisurvey_biased":
        parity = rows[0]
        assert parity["MaxAbsErr_Basis"] < 1e-12 and parity["MaxAbsErr_Derivative"] < 1e-12
        assert parity["OODFlagMismatches"] == 0
        assert all("BiasRMSE" in r for r in rows[1:])


def test_failing_method_becomes_failed_row(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = MultiSourceSuite()

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("suites.tabular.multi_source._exactness", boom)
    rows = suite.run("multisurvey", HarnessConfig(seed=0, epochs=1, smoke=True))
    assert rows[0]["Notes"] == "failed: RuntimeError: boom" and rows[0]["RMSE"] is None
    assert "failed:" not in rows[1]["Notes"]
