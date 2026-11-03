"""Parity: ``PoissonGaussianLoss`` vs a direct Poisson (x) Gaussian convolution in SciPy.

The detector law is ``K ~ Poisson(rate)``, ``y = gain * K + offset + eps`` with
``eps ~ N(0, read_var)``, so

    p(y) = sum_k Pois(k; rate) N(y; gain * k + offset, read_var).

The reference sums that series in float64 with ``scipy.stats`` log-pmf/log-pdf
and ``scipy.special.logsumexp`` over ``k = 0 .. rate + 40 sqrt(rate + 1) + 400``
(plus the counts needed to reach far outliers), i.e. far beyond the library's
12-sigma window, so the reference truncation error is below float64 round-off.
The grid covers zero and sub-unit rates (low counts), Gaussian-limit rates,
gains below and above one, negative/large offsets, near-noiseless and
read-noise-dominated detectors, and targets from -6 to +30 predictive standard
deviations plus exact integer-count positions.

Tolerance: a forward-error model, not a fudge factor.  The NLL is a log-sum-exp
whose inputs carry the rounding of ``(y - gain k - offset) / sigma`` and a
cumulative sum of ~``24 sqrt(rate + 1)`` Poisson log-ratios, so

    |err| <= C * eps * (1 + |NLL| + (|y| + |offset| + gain * rate) |dNLL/dy| + 24 sqrt(rate + 1))

with ``eps`` the dtype's machine epsilon and ``dNLL/dy`` from the reference by
central differences.  The observed maximum of ``|err| / bound`` is ~1.3 in
float32 and far below 1 in float64; ``C = 8`` leaves headroom for platform
libm differences only.  Float32 inputs are rounded first and the reference is
evaluated at the rounded values, so input quantisation is not counted as error.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import norm, poisson
from torchregress.losses import PoissonGaussianLoss

RATES = (0.0, 0.03, 0.4, 1.7, 9.0, 85.0, 900.0)
GAINS = (0.4, 1.0, 2.7)
OFFSETS = (-3.0, 0.0, 100.0)
READ_VARS = (0.01, 0.3, 4.0, 40.0)
Z_POSITIONS = (-6.0, -2.5, -1.0, 0.0, 0.3, 1.0, 2.5, 6.0, 30.0)
C_FORWARD = 8.0


def _devices() -> list[str]:
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    return out


def reference_nll(y: float, rate: float, gain: float, offset: float, read_var: float) -> float:
    """-log sum_k Pois(k; rate) N(y; gain k + offset, read_var), float64."""
    sd = float(np.sqrt(read_var))
    if rate == 0.0:
        return float(-norm.logpdf(y, offset, sd))
    k_max = int(rate + 40.0 * np.sqrt(rate + 1.0) + 400.0 + 2.0 * max(0.0, (y - offset) / gain))
    k = np.arange(k_max + 1)
    return float(-logsumexp(poisson.logpmf(k, rate) + norm.logpdf(y, gain * k + offset, sd)))


def _targets(rate: float, gain: float, offset: float, read_var: float) -> np.ndarray:
    sd_tot = np.sqrt(gain * gain * rate + read_var)
    around = gain * rate + offset + sd_tot * np.asarray(Z_POSITIONS)
    counts = offset + gain * np.arange(4.0)  # exact low-count positions
    return np.concatenate([around, counts])


def _cast(v: float, dtype: torch.dtype) -> float:
    return float(np.float32(v)) if dtype == torch.float32 else float(v)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("rate", RATES)
def test_nll_matches_scipy_convolution(rate: float, dtype: torch.dtype, device: str) -> None:
    eps = float(torch.finfo(dtype).eps)
    worst = 0.0
    for gain, offset, read_var in itertools.product(GAINS, OFFSETS, READ_VARS):
        r, g, b, v = (_cast(x, dtype) for x in (rate, gain, offset, read_var))
        ys = np.array([_cast(y, dtype) for y in _targets(r, g, b, v)])
        ref = np.array([reference_nll(y, r, g, b, v) for y in ys])
        h = 1e-6 * np.maximum(1.0, np.abs(ys))
        dref_dy = np.array(
            [
                (reference_nll(y + hh, r, g, b, v) - reference_nll(y - hh, r, g, b, v)) / (2 * hh)
                for y, hh in zip(ys, h)
            ]
        )
        loss = PoissonGaussianLoss(gain=g, offset=b, read_noise_variance=v, reduction="none")
        got = (
            loss(
                torch.full((len(ys),), r, dtype=dtype, device=device),
                torch.as_tensor(ys, dtype=dtype, device=device),
            )
            .double()
            .cpu()
            .numpy()
        )
        bound = eps * (
            1.0
            + np.abs(ref)
            + (np.abs(ys) + abs(b) + g * r) * np.abs(dref_dy)
            + 24.0 * np.sqrt(r + 1.0)
        )
        ratio = np.abs(got - ref) / bound
        worst = max(worst, float(ratio.max()))
        assert np.all(ratio <= C_FORWARD), (
            f"rate={r} gain={g} offset={b} read_var={v} dtype={dtype}: "
            f"max |err|/bound = {ratio.max():.3g} at y={ys[ratio.argmax()]} "
            f"(got {got[ratio.argmax()]!r}, ref {ref[ratio.argmax()]!r})"
        )
    assert worst <= C_FORWARD


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("rate", (0.03, 0.4, 1.7, 9.0, 85.0))
def test_first_derivatives_match_scipy_finite_differences(rate: float, device: str) -> None:
    """Analytic posterior-moment gradients vs Richardson central differences of the reference.

    Steps are scaled to each parameter's natural resolution; Richardson
    extrapolation makes the truncation error O(h^4) and the round-off term is
    ~``eps |NLL| / h`` <= 1e-7, so ``1e-6 (1 + |grad|)`` is a float64 bound.
    """
    for gain, offset, read_var in itertools.product(GAINS, (-3.0, 100.0), (0.01, 0.3, 4.0)):
        sd = np.sqrt(read_var)
        sd_tot = np.sqrt(gain * gain * rate + read_var)
        for y in gain * rate + offset + sd_tot * np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 10.0]):
            params = [
                torch.tensor([v], dtype=torch.float64, device=device, requires_grad=True)
                for v in (rate, gain, offset, read_var)
            ]
            loss = PoissonGaussianLoss(reduction="sum")
            out = loss(
                params[0],
                torch.tensor([y], dtype=torch.float64, device=device),
                gain=params[1],
                offset=params[2],
                read_noise_variance=params[3],
            )
            grads = [gr.item() for gr in torch.autograd.grad(out, params)]
            base = [rate, gain, offset, read_var]
            scales = [min(rate, sd / gain), sd / (rate + 1.0), sd, read_var]
            for i, scale in enumerate(scales):

                def central(h: float, i: int = i) -> float:
                    up, dn = list(base), list(base)
                    up[i] += h
                    dn[i] -= h
                    return (reference_nll(y, *up) - reference_nll(y, *dn)) / (2.0 * h)

                h = 1e-3 * scale
                fd = (4.0 * central(h / 2.0) - central(h)) / 3.0
                assert abs(grads[i] - fd) <= 1e-6 * (1.0 + abs(fd)), (
                    f"d/d{['rate', 'gain', 'offset', 'read_var'][i]} at rate={rate} gain={gain} "
                    f"offset={offset} read_var={read_var} y={y}: autograd {grads[i]!r} vs FD {fd!r}"
                )


def test_limits_and_learned_read_noise_parameter() -> None:
    """Gaussian/Poisson limits and the learned-variance parameterisation."""
    # sigma -> 0, gain 1, offset 0: Poisson pmf on integer targets (mass, not density:
    # NLL = -log Pois(k) + 0.5 log(2 pi sigma^2) at the count itself).
    v = 1e-6
    k = torch.arange(0.0, 12.0, dtype=torch.float64)
    nll = PoissonGaussianLoss(read_noise_variance=v, reduction="none")(torch.full_like(k, 3.5), k)
    expect = -poisson.logpmf(k.numpy(), 3.5) + 0.5 * np.log(2 * np.pi * v)
    np.testing.assert_allclose(nll.numpy(), expect, rtol=0, atol=1e-9)

    # learn_read_noise stores log(var) as a module parameter.  It is created in
    # float32 (torch default dtype), so even after ``.double()`` it holds
    # float32(log 2.5); the fixed-variance reference uses that same value.
    learned = PoissonGaussianLoss(read_noise_variance=2.5, learn_read_noise=True).double()
    stored_var = float(np.exp(np.float64(np.float32(np.log(2.5)))))
    fixed = PoissonGaussianLoss(read_noise_variance=stored_var)
    rate = torch.tensor([[0.2], [4.0], [30.0]], dtype=torch.float64)
    y = torch.tensor([[1.0], [3.0], [41.0]], dtype=torch.float64)
    assert torch.allclose(learned(rate, y), fixed(rate, y), rtol=1e-12, atol=0)
    learned(rate, y).backward()
    assert learned.log_read_noise_variance.grad is not None
    assert torch.isfinite(learned.log_read_noise_variance.grad)
