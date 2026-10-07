"""Contracts for the synthetic likelihood-family designs of the likelihoods suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from suites.tabular import _likelihood_designs as L
from suites.tabular.likelihoods import LikelihoodsSuite

from harness.runners import HarnessConfig

FAMILIES = list(L.FAMILIES)


def test_designs_are_supported_datasets() -> None:
    supported = set(LikelihoodsSuite().supported_datasets)
    assert {f"lik_{f}" for f in FAMILIES} <= supported
    assert {"folktables_all", "tlc_fare", "insurance_claims"} <= supported


@pytest.mark.parametrize("family", FAMILIES)
def test_library_density_matches_scipy_at_truth(family: str) -> None:
    """The torchregress NLL functional equals the scipy.stats log-density."""
    fam = L.FAMILIES[family]
    d = L.draw_design(f"lik_{family}", seed=1, smoke=True)
    lib = L.library_nll_rows(fam, d["raw_test"], d["y_test"])
    ref = -fam.logpdf(d["y_test"], fam.constrain(d["raw_test"]))
    assert np.allclose(lib, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_raw_shapes_round_trip(family: str) -> None:
    fam = L.FAMILIES[family]
    raw = L.true_raw(fam, np.zeros((2, L.N_FEATURES)))
    p = fam.constrain(raw)
    n_shape = len(fam.shapes)
    if n_shape:
        assert np.allclose(p[0, fam.k - n_shape :], fam.shapes, atol=1e-9)


def test_beta_draws_stay_inside_unit_interval() -> None:
    d = L.draw_design("lik_beta", seed=0, smoke=True)
    assert (d["y_train"] > 0).all() and (d["y_train"] < 1).all()


def test_nll_summary_reports_infinite_rows() -> None:
    out = L.nll_summary(np.array([1.0, 3.0, np.inf, np.inf]))
    assert out["NLL"] == pytest.approx(2.0)
    assert out["FracInfNLL"] == pytest.approx(0.5)


@pytest.mark.parametrize("family", ["skewnormal", "johnsonsu", "gev", "beta"])
def test_smoke_rows_linear_fit_matches_scipy(family: str) -> None:
    rows = LikelihoodsSuite().run(f"lik_{family}", config=HarnessConfig(seed=0, smoke=True))
    by = {r["Method"].split("/")[1]: r for r in rows}
    assert set(by) == set(L.METHODS)
    for name, row in by.items():
        assert not str(row["Notes"]).startswith("failed"), (name, row["Notes"])
        assert row["SyntheticFallback"] is False
    assert by["TrueModel"]["TruthNLLParity"] < 1e-6
    # Same conditional MLE through the library loss and through scipy.
    assert abs(by["tr_Linear"]["NLLGapToScipyLinear"]) < 0.02
    # The well-specified conditional fit beats the unconditional scipy MLE.
    assert by["tr_Linear"]["NLL"] < by["scipy_MLE_Unconditional"]["NLL"]
    assert by["tr_Linear"]["LocRMSE"] < 0.2


def test_skew_t_alpha_gradient_at_zero() -> None:
    """Regression test for torchregress e02d186: the skewness gradient is
    non-zero at alpha = 0, so skew-t fits can start from a symmetric model."""
    from torchregress.losses import skew_t_nll

    y = torch.tensor([[0.7], [-1.2], [2.5]], dtype=torch.float64)
    raw = torch.tensor([[0.0, 0.5, 0.0, 2.0]] * 3, dtype=torch.float64, requires_grad=True)
    skew_t_nll(raw, y).backward()
    assert raw.grad is not None
    assert float(raw.grad[:, 2].abs().sum()) > 0.0
