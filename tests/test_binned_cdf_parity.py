"""Parity: binned-distribution helpers and the cumulative-CDF loss vs independent references.

* ``binned_cdf`` vs ``scipy.stats.rv_histogram(...).cdf`` at the bin edges.
* ``crps_from_binned_cdf`` (closed-form piecewise-linear CRPS) vs
  ``scipy.integrate.quad`` of ``(F(z) - 1{z >= y})^2`` with ``F`` the SciPy
  histogram CDF; the integrand is a piecewise quadratic, so Gauss-Kronrod with
  every edge and ``y`` as a breakpoint integrates it exactly up to round-off.
* ``crps_from_binned_cdf`` on a finely binned Gaussian vs the harness's own
  closed-form Gaussian CRPS (``harness.metrics.compute_distribution_metrics``),
  with the rigorous linear-interpolation error bound as tolerance.
* ``cumulative_cdf_loss`` / ``CumulativeCDFLoss`` vs ``scoringrules.rps_score``:
  on centers ``c_k`` the loss is ``mean_width * RPS(softmax(logits), y)`` with
  the observed category the first center ``>= y``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scoringrules as sr
import torch
from scipy import integrate
from scipy.stats import norm, rv_histogram
from torchregress.losses import CumulativeCDFLoss, cumulative_cdf_loss
from torchregress.metrics import binned_cdf, crps_from_binned_cdf

from harness.metrics import compute_distribution_metrics

# Float64 round-off on O(1)-O(10) quantities summed over <= 100 bins.
RTOL = 1e-10
ATOL = 1e-12


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.fixture(params=[0, 1, 2], ids=lambda s: f"seed{s}")
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    return np.random.default_rng(request.param)


def _random_histograms(
    rng: np.random.Generator, n: int, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unequal-width edges, sparse/spiky masses (some empty bins) and targets in and out of range."""
    edges = np.cumsum(np.concatenate([[rng.normal()], rng.uniform(0.05, 1.5, size=k)]))
    probs = rng.dirichlet(np.full(k, 0.4), size=n)
    probs[:, rng.integers(0, k, size=3)] = 0.0
    probs[probs.sum(axis=1) == 0.0, 0] = 1.0
    span = edges[-1] - edges[0]
    y = rng.uniform(edges[0] - 0.3 * span, edges[-1] + 0.3 * span, size=n)
    y[:3] = edges[[0, k // 2, -1]]  # exactly on edges
    return edges, probs, y


def test_binned_cdf_matches_scipy_histogram(rng: np.random.Generator) -> None:
    edges, probs, _ = _random_histograms(rng, 40, 17)
    unnormalised = probs * rng.uniform(0.5, 3.0, size=(len(probs), 1))
    got = binned_cdf(unnormalised)
    assert got.shape == (len(probs), len(edges))
    for i in range(len(probs)):
        ref = rv_histogram((probs[i], edges), density=False).cdf(edges)
        np.testing.assert_allclose(got[i], ref, rtol=RTOL, atol=ATOL)


def _quad_crps(edges: np.ndarray, probs: np.ndarray, y: float) -> float:
    dist = rv_histogram((probs, edges), density=False)
    lo, hi = min(edges[0], y), max(edges[-1], y)
    points = np.unique(np.concatenate([edges, [y]]))

    def integrand(z: float) -> float:
        return (float(dist.cdf(z)) - float(z >= y)) ** 2

    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        if b <= lo or a >= hi:
            continue
        val, _ = integrate.quad(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=200)
        total += val
    return total


def test_crps_from_binned_cdf_matches_quadrature(rng: np.random.Generator) -> None:
    edges, probs, y = _random_histograms(rng, 30, 12)
    got = crps_from_binned_cdf(edges, binned_cdf(probs), y, reduction="none")
    ref = np.array([_quad_crps(edges, probs[i], y[i]) for i in range(len(y))])
    # quad's own error estimate is <= 1e-13 relative on these exact polynomials.
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
    assert crps_from_binned_cdf(edges, binned_cdf(probs), y) == pytest.approx(
        ref.mean(), rel=1e-9, abs=1e-12
    )
    assert crps_from_binned_cdf(edges, binned_cdf(probs), y, reduction="sum") == pytest.approx(
        ref.sum(), rel=1e-9, abs=1e-12
    )


def test_crps_from_binned_cdf_converges_to_harness_gaussian_crps(
    rng: np.random.Generator,
) -> None:
    """Fine bins on N(mu, sd) vs the harness closed-form Gaussian CRPS.

    ``F_h`` linearly interpolates the Gaussian CDF ``F`` on edges spaced ``h``
    over ``mu +- 10 sd``; ``|F_h - F| <= h^2 / 8 * max|f'|`` with
    ``max|f'| = 1 / (sd^2 sqrt(2 pi e))``.  Because both CDFs lie in [0, 1],
    ``|CRPS_h - CRPS| <= 2 * int |F_h - F| + tails <= 2 * 20 sd * h^2 / 8 * max|f'|``
    (the tail mass beyond 10 sd is < 1e-22).
    """
    n = 25
    mu = rng.normal(size=n)
    sd = rng.uniform(0.3, 2.0, size=n)
    y = mu + sd * rng.standard_t(df=5, size=n)
    k = 2000
    got = np.empty(n)
    bound = np.empty(n)
    for i in range(n):
        edges = np.linspace(mu[i] - 10 * sd[i], mu[i] + 10 * sd[i], k + 1)
        cdf = norm.cdf(edges, mu[i], sd[i])
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        got[i] = crps_from_binned_cdf(edges, cdf[None, :], y[i : i + 1])
        h = edges[1] - edges[0]
        bound[i] = 2.0 * 20.0 * sd[i] * h**2 / 8.0 / (sd[i] ** 2 * np.sqrt(2 * np.pi * np.e))
    for i in range(n):
        ref = compute_distribution_metrics(mu[i : i + 1], sd[i : i + 1], y[i : i + 1])["CRPS"]
        assert abs(got[i] - ref) <= bound[i] + 1e-12, (got[i], ref, bound[i])


@pytest.mark.parametrize("device", _devices())
def test_cumulative_cdf_loss_is_scaled_rps(rng: np.random.Generator, device: str) -> None:
    k, n = 9, 64
    centers = np.cumsum(rng.uniform(0.2, 1.0, size=k))
    logits = rng.normal(scale=2.0, size=(n, k))
    y = rng.uniform(centers[0] - 1.0, centers[-1] + 1.0, size=n)
    y[:k] = centers  # exactly on centers
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    category = np.minimum(np.searchsorted(centers, y, side="left"), k)  # first c_k >= y
    above_all = category == k
    onehot = np.zeros((n, k))
    onehot[~above_all, category[~above_all]] = 1.0
    rps = np.asarray(sr.rps_score(onehot, probs, onehot=True), dtype=np.float64)
    # Above the last center the step CDF is 0 everywhere: sum_k F_k^2.
    rps[above_all] = (np.cumsum(probs[above_all], axis=1) ** 2).sum(axis=1)
    mean_width = (centers[-1] - centers[0]) / (k - 1)
    expected = mean_width * rps

    lg = torch.as_tensor(logits, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y, dtype=torch.float64, device=device)
    ct = torch.as_tensor(centers, dtype=torch.float64, device=device)
    got = cumulative_cdf_loss(lg, yt, ct)
    np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=RTOL, atol=ATOL)

    # Module: float32 buffer, so compare in float32 against the float64 reference
    # (softmax/cumsum of K=9 values: a few float32 ulps of O(10) quantities).
    module = CumulativeCDFLoss(centers.tolist(), reduction="none").to(device)
    got32 = module(lg.float(), yt.float()).double().cpu().numpy()
    np.testing.assert_allclose(got32, expected, rtol=1e-5, atol=1e-6)
    weights = torch.as_tensor(rng.uniform(0.1, 2.0, size=n), dtype=torch.float32, device=device)
    mask = torch.as_tensor(rng.uniform(size=n) > 0.3, device=device)
    reduced = CumulativeCDFLoss(centers.tolist(), reduction="mean").to(device)(
        lg.float(), yt.float(), mask=mask, weights=weights
    )
    w = weights.double().cpu().numpy() * mask.cpu().numpy()
    assert reduced.item() == pytest.approx(float((w * expected).sum() / w.sum()), rel=1e-5)
