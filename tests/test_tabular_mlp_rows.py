"""Smoke contracts for the rows that wire in the torchregress 0.3 features.

* ``torchregress.models`` TabularMLP rows (``sklearn_baselines``, ``uq``, ``uq_layer``, ``conformal``),
* ``orthogonal_inference`` RFF nuisances from ``inference.random_fourier_features``,
* the documented NSF training recipe (``pzflow_parity``, ``distributional_bins``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from suites._flow_recipe import fit_nsf_recipe
from suites._tabular_mlp import LIBRARY_EPOCHS, fit_tabmlp, predict_gaussian, tabular_epochs
from suites.tabular.conformal import ConformalSuite
from suites.tabular.distributional_bins import DistributionalBinsSuite
from suites.tabular.orthogonal_inference import OrthogonalInferenceSuite
from suites.tabular.pzflow_parity import PzflowParitySuite
from suites.tabular.sklearn_baselines import SklearnBaselinesSuite
from suites.tabular.uq import UQSuite

from harness.runners import HarnessConfig

CFG = HarnessConfig(seed=0, epochs=2, batch_size=64, smoke=True, standardize_targets=True)


def _run(suite: object, dataset: str, methods: list[str], cfg: HarnessConfig = CFG) -> dict:
    suite.methods = lambda: methods  # type: ignore[attr-defined,method-assign]
    rows = suite.run(dataset, cfg)  # type: ignore[attr-defined]
    return {r["Method"].split("/", 1)[1]: r for r in rows}


def test_epochs_are_capped_by_config() -> None:
    assert tabular_epochs(HarnessConfig(epochs=2)) == 2
    assert tabular_epochs(HarnessConfig(epochs=10_000)) == LIBRARY_EPOCHS == 100


def test_fit_tabmlp_heads_and_determinism() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=120)
    a = fit_tabmlp(X, y, CFG, kind="gaussian", n_members=2)
    b = fit_tabmlp(X, y, CFG, kind="gaussian", n_members=2)
    ma, sa = predict_gaussian(a, X)
    mb, sb = predict_gaussian(b, X)
    assert np.allclose(ma, mb) and np.allclose(sa, sb)
    assert (sa > 0).all() and len(a.members) == 2
    q = fit_tabmlp(X, y, CFG, kind="quantile", quantiles=(0.05, 0.95))
    assert q.predict(X).shape == (120, 2)
    with pytest.raises(ValueError, match="unknown head"):
        fit_tabmlp(X, y, CFG, kind="nope")


def test_sklearn_baselines_rows() -> None:
    assert {"TabularMLP", "TabularMLP_Ens5"} <= set(SklearnBaselinesSuite().methods())
    rows = _run(SklearnBaselinesSuite(), "diabetes", ["TabularMLP", "TabularMLP_Ens5", "MSE"])
    for m in ("TabularMLP", "TabularMLP_Ens5"):
        assert np.isfinite(rows[m]["RMSE"]) and not str(rows[m]["Notes"]).startswith("failed")
        assert rows[m]["Library"] == "torchregress"


def test_uq_rows_and_vts_control() -> None:
    names = [
        "TabularMLP_GaussianNLL",
        "TabularMLP_Ens5_GaussianNLL",
        "TabularMLP_Ens5_GaussianNLL_carve",
        "TabularMLP_Ens5_GaussianNLL+VTS",
    ]
    assert set(names) <= set(UQSuite().methods())
    rows = _run(UQSuite(), "synthetic_hetero", names)
    for m in names:
        assert all(
            rows[m][k] is not None and np.isfinite(rows[m][k]) for k in ("NLL", "CRPS", "ECE")
        )
    # VTS only rescales the std: same mean predictions as its carved-ensemble control.
    assert rows["TabularMLP_Ens5_GaussianNLL+VTS"]["RMSE"] == pytest.approx(
        rows["TabularMLP_Ens5_GaussianNLL_carve"]["RMSE"]
    )


def test_conformal_cqr_tabular_row() -> None:
    assert "CQR+TabularMLP" in ConformalSuite().methods()
    row = _run(ConformalSuite(), "synthetic_hetero", ["CQR+TabularMLP"])["CQR+TabularMLP"]
    assert 0.0 <= row["Coverage"] <= 1.0 and row["Width"] > 0
    assert not str(row["Notes"]).startswith("failed")


def test_orthogonal_rff_rows() -> None:
    suite = OrthogonalInferenceSuite()
    assert {"tr_Orthogonal_RFF", "tr_Orthogonal_RFF_legacy"} <= set(suite.methods())
    assert "random_fourier_features" in suite.method_notes("tr_Orthogonal_RFF")
    assert "gcv" in suite.method_notes("tr_Orthogonal_RFF")
    rows = _run(suite, "plr_cubic", ["tr_Orthogonal_RFF", "tr_Orthogonal_RFF_legacy", "tr_Naive"])
    for m in ("tr_Orthogonal_RFF", "tr_Orthogonal_RFF_legacy"):
        assert np.isfinite(rows[m]["Bias"]) and rows[m]["Replications"] == 10
    # Both orthogonal variants beat the confounded naive slope even on 10 replications.
    assert abs(rows["tr_Orthogonal_RFF"]["Bias"]) < abs(rows["tr_Naive"]["Bias"])


def test_nsf_recipe_helper_stops_early_and_widens_bound() -> None:
    torch.manual_seed(0)
    X = torch.randn(200, 2)
    heavy = torch.distributions.StudentT(2.0).sample((200, 1))
    y = (heavy - heavy.mean()) / heavy.std()
    cfg = HarnessConfig(seed=0, epochs=3, batch_size=64, smoke=True)
    _, loss_fn, info = fit_nsf_recipe(X, y, cfg, in_dim=2, hidden=8)
    assert info["bound"] >= 5.0 and info["n_val"] == 30
    assert 1 <= info["epochs_run"] <= 3 and info["best_epoch"] < info["epochs_run"]
    assert hasattr(loss_fn, "sample")


def test_nsf_recipe_rows() -> None:
    assert "torchregress_NSF_recipe" in PzflowParitySuite().methods()
    assert "NormalizingFlow_recipe" in DistributionalBinsSuite().methods()
    rows = _run(
        PzflowParitySuite(),
        "synthetic_cond_gaussian",
        ["torchregress_NSF", "torchregress_NSF_recipe"],
    )
    for m in ("torchregress_NSF", "torchregress_NSF_recipe"):
        assert np.isfinite(rows[m]["NLL"]) and not str(rows[m]["Notes"]).startswith("failed")
    dist = _run(DistributionalBinsSuite(), "diabetes", ["NormalizingFlow_recipe"])
    assert not str(dist["NormalizingFlow_recipe"]["Notes"]).startswith("failed")
