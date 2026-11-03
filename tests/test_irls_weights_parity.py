"""Parity: IRLS robust weights vs statsmodels norms and the M-estimator definition w = psi(r) / r.

* ``robust_weights(..., "huber" | "tukey")`` vs ``statsmodels.robust.norms``
  ``HuberT(t).weights`` / ``TukeyBiweight(c).weights``.
* ``"pseudo_huber"``, ``"cauchy"``, ``"barron"`` (no statsmodels norm) vs
  ``psi(r) / r`` with ``psi = rho'`` taken by autograd of the textbook
  ``rho`` written here (pseudo-Huber; Cauchy/Lorentzian
  ``c^2/2 log(1 + (r/c)^2)``; Barron 2019 general loss with
  ``|alpha - 2|`` regularised as in the library's ``beta``).
* ``reweight_fixed_predictions`` (no refit) vs statsmodels: Tukey weights of
  the fixed residuals scaled by the normalised MAD.

Float64 throughout; agreement is to round-off (``rtol = 1e-12``) except where
a documented constant differs (see the reweighting test).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from statsmodels.robust import norms
from statsmodels.robust.scale import mad
from torchregress.algorithms import IRLSConfig, reweight_fixed_predictions, robust_weights

RTOL = 1e-12
ATOL = 1e-14


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _residuals(device: str) -> torch.Tensor:
    rng = np.random.default_rng(0)
    r = np.concatenate([rng.standard_t(df=2, size=500) * 3.0, [0.0, 1e-9, -1.345, 4.685, -50.0]])
    return torch.as_tensor(r, dtype=torch.float64, device=device)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("delta", [0.5, 1.345, 3.0])
def test_huber_weights_match_statsmodels(device: str, delta: float) -> None:
    r = _residuals(device)
    got = robust_weights(r, "huber", {"delta": delta}).cpu().numpy()
    np.testing.assert_allclose(
        got, norms.HuberT(t=delta).weights(r.cpu().numpy()), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("c", [2.0, 4.685, 6.0])
def test_tukey_weights_match_statsmodels(device: str, c: float) -> None:
    r = _residuals(device)
    got = robust_weights(r, "tukey", {"c": c}).cpu().numpy()
    np.testing.assert_allclose(
        got, norms.TukeyBiweight(c=c).weights(r.cpu().numpy()), rtol=RTOL, atol=ATOL
    )


def _rho_pseudo_huber(r: torch.Tensor, delta: float) -> torch.Tensor:
    return delta**2 * (torch.sqrt(1.0 + (r / delta) ** 2) - 1.0)


def _rho_cauchy(r: torch.Tensor, c: float) -> torch.Tensor:
    return 0.5 * c**2 * torch.log1p((r / c) ** 2)


def _rho_barron(r: torch.Tensor, alpha: float, scale: float) -> torch.Tensor:
    # Barron (2019), eq. 1, with |alpha - 2| -> beta = sqrt((alpha - 2)^2 + 1e-8)
    # (the library's regularisation); alpha = 0 is the limit beta/2 log(1 + x^2/beta).
    x2 = (r / scale) ** 2
    beta = float(np.sqrt((alpha - 2.0) ** 2 + 1e-8))
    if alpha == 0.0:
        return 0.5 * beta * torch.log1p(x2 / beta)
    return beta / alpha * ((x2 / beta + 1.0) ** (alpha / 2.0) - 1.0)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    ("name", "params", "rho"),
    [
        ("pseudo_huber", {"delta": 1.345}, lambda r: _rho_pseudo_huber(r, 1.345)),
        ("cauchy", {"c": 2.3849}, lambda r: _rho_cauchy(r, 2.3849)),
        ("barron", {"alpha": 0.0, "scale": 1.0}, lambda r: _rho_barron(r, 0.0, 1.0)),
        ("barron", {"alpha": 1.0, "scale": 0.7}, lambda r: _rho_barron(r, 1.0, 0.7)),
        ("barron", {"alpha": -2.0, "scale": 1.5}, lambda r: _rho_barron(r, -2.0, 1.5)),
    ],
    ids=["pseudo_huber", "cauchy", "barron_a0", "barron_a1", "barron_am2"],
)
def test_weights_are_psi_over_r(
    device: str,
    name: str,
    params: dict[str, float],
    rho: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    r = _residuals(device)
    r = r[r.abs() > 1e-6].clone().requires_grad_(True)  # psi(r)/r is 0/0 at r = 0
    (psi,) = torch.autograd.grad(rho(r).sum(), r)
    expected = (psi / r).detach()
    if name == "barron":
        # Barron psi(r)/r = (1 / scale^2) (x^2/beta + 1)^(alpha/2 - 1): the library
        # returns the scale-free factor (the 1/scale^2 is absorbed in the precision).
        expected = expected * params["scale"] ** 2
    got = robust_weights(r.detach(), name, params)
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("device", _devices())
def test_reweight_fixed_predictions_matches_statsmodels_tukey(device: str) -> None:
    """Tukey weights of fixed residuals with MAD scale; the model is NOT refit.

    Odd ``n`` so ``torch.median`` (lower middle for even counts) and
    ``np.median`` agree.  The library normalises the MAD with the rounded
    constant 1.4826, statsmodels with ``1 / Phi^-1(0.75) = 1.482602218...``;
    the reference uses 1.4826 so the comparison is exact to round-off.
    """
    rng = np.random.default_rng(1)
    n, d = 401, 3
    X = rng.normal(size=(n, d))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.3 * rng.normal(size=n)
    y[rng.choice(n, 40, replace=False)] += rng.uniform(5, 15, size=40)

    model = torch.nn.Linear(d, 1).double().to(device)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.9, -1.8, 0.4]], dtype=torch.float64))
        model.bias.fill_(0.2)
    before = [p.detach().clone() for p in model.parameters()]
    xt = torch.as_tensor(X, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y, dtype=torch.float64, device=device).unsqueeze(1)
    y_pred, history, precision = reweight_fixed_predictions(
        model,
        xt,
        yt,
        config=IRLSConfig(
            weight_fn="tukey",
            weight_params={"c": 4.685},
            variance_type="robust",
            max_iter=4,
            batch_size=64,  # exercise the mini-batched inference path
        ),
    )
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))  # no refit
    resid = y - (X @ np.array([0.9, -1.8, 0.4]) + 0.2)
    np.testing.assert_allclose(y_pred.cpu().numpy().ravel(), y - resid, rtol=RTOL, atol=1e-12)
    scale = 1.4826 * mad(resid, c=1.0, center=np.median(resid))
    expected = norms.TukeyBiweight(c=4.685).weights(resid / (scale + 1e-8))
    np.testing.assert_allclose(precision.cpu().numpy().ravel(), expected, rtol=1e-10, atol=1e-14)
    # Fixed residuals: the weights (and the weighted loss) cannot change between iterations.
    assert len(history) >= 2 and history[-1] == pytest.approx(history[1], rel=1e-12)
