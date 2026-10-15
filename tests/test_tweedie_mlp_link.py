"""Regression test: tweedie-suite MLP rows must feed log(mu) to the loss.

TweedieLoss / CompoundPoissonLoss / GammaLoss / InverseGaussianLoss default to
``link='log'`` (y_pred is log(mu); the loss exponentiates internally). The MLP
rows used to train with ``transform_output=torch.exp``, so the loss fitted
mu = exp(exp(f)) while predict() reported exp(f) = log(mu): R2 ~ -0.15 and
deviance ~50 on synthetic_tweedie vs GLM R2 0.55 / deviance 0.72.
"""

from __future__ import annotations

import math

import pytest
import torch
from suites.tabular.tweedie import TweedieSuite
from torchregress.losses import (
    CompoundPoissonLoss,
    GammaLoss,
    InverseGaussianLoss,
    TweedieLoss,
)

from harness.runners import HarnessConfig


@pytest.mark.parametrize(
    "loss_fn",
    [TweedieLoss(p=1.5), CompoundPoissonLoss(p=1.5), GammaLoss(), InverseGaussianLoss()],
)
def test_tweedie_family_losses_take_log_mean(loss_fn):
    # Gamma unit deviance is zero iff mu == y; passing log(y) must give ~0.
    y = torch.tensor([[1.0], [2.0], [3.0]])
    if loss_fn.p == 2 or loss_fn.p == 3:
        assert loss_fn(torch.log(y), y).item() == pytest.approx(0.0, abs=1e-5)
    assert loss_fn.link == "log"


@pytest.mark.parametrize("method", ["Tweedie", "CompoundPoisson"])
def test_tweedie_mlp_rows_learn_on_synthetic(method):
    torch.set_num_threads(1)
    suite = TweedieSuite()
    suite.methods = lambda: [method, "Tweedie_Linear"]
    rows = suite.run("synthetic_tweedie", HarnessConfig(seed=0, epochs=10))
    by = {r["Method"].split("/")[-1]: r for r in rows}
    mlp, lin = by[method], by["Tweedie_Linear"]
    assert math.isfinite(mlp["R2"]) and mlp["R2"] > 0.3, mlp
    assert mlp["TweedieDeviance"] < 1.5 * lin["TweedieDeviance"], (mlp, lin)


def test_positive_support_rows_skipped_on_zero_inflated_data():
    """Gamma / InverseGaussian (support y > 0) are not fitted on data with zeros.

    The library losses raise on y <= 0; the suite must neither drop nor shift
    the zero rows (that would change the estimand) but report a skipped row.
    """
    suite = TweedieSuite()
    suite.methods = lambda: ["Gamma", "Gamma_Linear", "InverseGaussian"]
    rows = suite.run("synthetic_tweedie", HarnessConfig(seed=0, epochs=1, smoke=True))
    assert len(rows) == 3
    for r in rows:
        assert r["Notes"].startswith("skipped: "), r
        assert "requires strictly positive targets" in r["Notes"], r
        assert r["RMSE"] is None and r["MAE"] is None, r


def test_positive_support_rows_fit_on_positive_data():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    y = rng.gamma(2.0, np.exp(0.3 * X[:, 0]) / 2.0).astype(np.float32) + 1e-3
    splits = {"X_train": X[:150], "y_train": y[:150], "X_test": X[150:], "y_test": y[150:]}
    suite = TweedieSuite()
    suite.methods = lambda: ["Gamma_Linear", "InverseGaussian"]
    rows = suite.run("positive_gamma", HarnessConfig(seed=0, epochs=2), splits=splits)
    for r in rows:
        assert not r["Notes"].startswith(("skipped", "failed")), r
        assert math.isfinite(r["RMSE"]), r


def _claims_standin(seed: int = 123, n: int = 6000, d: int = 6):
    """Claims-like stand-in: ~95% zeros, lognormal(sigma=1.8) severities, 5 extreme values."""
    import numpy as np

    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d)).astype(np.float32)
    pos = r.random(n) < np.exp(-3.0 + 0.4 * X[:, 0] - 0.3 * X[:, 1])
    y = np.zeros(n)
    y[pos] = np.exp(1.0 + 0.3 * X[pos, 2] + r.normal(0, 1.8, pos.sum()))
    y[r.choice(np.flatnonzero(pos), 5, replace=False)] *= 200
    X = (X - X.mean(0)) / X.std(0)
    k = int(0.7 * n)
    y = y.astype(np.float32)
    return {
        "X_train": X[:k],
        "y_train": y[:k],
        "X_test": X[k:],
        "y_test": y[k:],
        "_skip_y_standardization": True,
    }


@pytest.mark.parametrize("method", ["Tweedie", "CompoundPoisson"])
def test_tweedie_mlp_rows_stable_on_heavy_tailed_claims(method):
    """insurance_claims regression: MLP rows diverged (R2 ~ -30, deviance ~187 vs GLM 1.41).

    The log-mean MLP recipe (bias = log mean(y), zero head, grad clipping,
    early stopping on a train carve) must stay near the intercept-only/GLM level.
    """
    torch.set_num_threads(1)
    suite = TweedieSuite()
    suite.methods = lambda: [method, "Tweedie_Linear"]
    rows = suite.run("claims_standin", HarnessConfig(seed=0, epochs=15), splits=_claims_standin())
    by = {r["Method"].split("/")[-1]: r for r in rows}
    mlp, lin = by[method], by["Tweedie_Linear"]
    assert math.isfinite(mlp["R2"]) and mlp["R2"] > -0.1, mlp
    assert mlp["TweedieDeviance"] < 1.5 * lin["TweedieDeviance"], (mlp, lin)
