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


@pytest.mark.parametrize("method", ["Tweedie", "CompoundPoisson", "Gamma"])
def test_tweedie_mlp_rows_learn_on_synthetic(method):
    torch.set_num_threads(1)
    suite = TweedieSuite()
    suite.methods = lambda: [method, "Tweedie_Linear"]
    rows = suite.run("synthetic_tweedie", HarnessConfig(seed=0, epochs=10))
    by = {r["Method"].split("/")[-1]: r for r in rows}
    mlp, lin = by[method], by["Tweedie_Linear"]
    assert math.isfinite(mlp["R2"]) and mlp["R2"] > 0.3, mlp
    assert mlp["TweedieDeviance"] < 1.5 * lin["TweedieDeviance"], (mlp, lin)
