"""Parity: representation-geometry diagnostics vs scikit-learn / SciPy, and the paired bootstrap.

``torchregress.metrics`` representation diagnostics (used by the
``imbalanced_representation`` suite's ``Rep_*`` columns) against independent
implementations, all in float64:

* ``neighbour_target_gap``: ``sklearn.neighbors.NearestNeighbors`` on the
  standardized features (``ddof=1``, the library convention), self excluded;
  median over points of the mean |target gap| to the k neighbours, divided by
  ``std(y, ddof=1)``.
* ``target_ordering``: ``scipy.stats.spearmanr`` of ``scipy.spatial.distance.pdist``
  feature distances against target distances (all pairs; same pair order).
* ``frozen_readout_metrics``: ``sklearn.linear_model.LinearRegression`` and
  ``sklearn.neighbors.KNeighborsRegressor`` (brute force) on the same
  deterministic split (``np.random.default_rng(seed).permutation``; the split
  is the documented contract of the function), train-standardized features.
* ``representation_geometry_report``: equals its components.

Continuous random features make neighbour ties have probability zero, so the
k-NN sets agree exactly; the remaining differences are float64 round-off of
distance and least-squares computations (``rtol = 1e-9``).

Medians are textbook medians (``np.median``, the average of the two middle values
for an even count) and NMAD uses ``1 / Phi^-1(0.75)``, as in torchregress >= 0.3.0.

``paired_bootstrap_ci`` vs ``scipy.stats.bootstrap(paired=True, method="percentile")``:
the two use different random streams, so the comparison is Monte-Carlo, with a
tolerance from the standard error of a bootstrap quantile estimate.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import bootstrap, norm, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from torchregress.metrics import (
    frozen_readout_metrics,
    neighbour_target_gap,
    paired_bootstrap_ci,
    representation_geometry_report,
    target_ordering,
)

RTOL = 1e-9
ATOL = 1e-12


@pytest.fixture(params=[0, 1, 2], ids=lambda s: f"seed{s}")
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    return np.random.default_rng(request.param)


_MAD_SCALE = 1.482602218505602  # 1 / Phi^-1(0.75)


def _features(rng: np.random.Generator, n: int = 300, d: int = 6) -> tuple[np.ndarray, np.ndarray]:
    scales = rng.uniform(0.1, 10.0, size=d)
    f = rng.normal(size=(n, d)) * scales + rng.normal(size=d)
    y = f[:, 0] / scales[0] + 0.5 * np.sin(f[:, 1] / scales[1]) + 0.3 * rng.normal(size=n)
    return f, y


def _standardize(f: np.ndarray) -> np.ndarray:
    return (f - f.mean(axis=0)) / f.std(axis=0, ddof=1)


@pytest.mark.parametrize("n", [301, 300], ids=["odd_n", "even_n"])
@pytest.mark.parametrize("k", [1, 5, 10])
def test_neighbour_target_gap_matches_sklearn(rng: np.random.Generator, k: int, n: int) -> None:
    f, y = _features(rng, n=n)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(_standardize(f))
    _, idx = nn.kneighbors(_standardize(f))
    assert np.all(idx[:, 0] == np.arange(len(y)))  # self first: no exact ties
    gaps = np.abs(y[idx[:, 1:]] - y[:, None]).mean(axis=1)
    median = np.median(gaps)
    expected = median / y.std(ddof=1)
    assert neighbour_target_gap(f, y, k=k) == pytest.approx(expected, rel=RTOL, abs=ATOL)


def test_target_ordering_matches_scipy(rng: np.random.Generator) -> None:
    f, y = _features(rng, n=250)
    expected = spearmanr(pdist(_standardize(f)), pdist(y[:, None])).statistic
    assert target_ordering(f, y) == pytest.approx(expected, rel=RTOL, abs=ATOL)


def test_frozen_readout_metrics_match_sklearn(rng: np.random.Generator) -> None:
    f, y = _features(rng, n=400)
    seed, k, test_fraction = 3, 25, 0.2
    order = np.random.default_rng(seed).permutation(len(y))
    n_test = max(1, int(round(len(y) * test_fraction)))
    test, train = order[:n_test], order[n_test:]
    mu, sd = f[train].mean(axis=0), f[train].std(axis=0, ddof=1)
    ft, fe = (f[train] - mu) / sd, (f[test] - mu) / sd
    preds = {
        "ols": LinearRegression().fit(ft, y[train]).predict(fe),
        "knn": KNeighborsRegressor(n_neighbors=k, algorithm="brute").fit(ft, y[train]).predict(fe),
    }
    got = frozen_readout_metrics(f, y, k=k, test_fraction=test_fraction, seed=seed)
    for name, pred in preds.items():
        err = pred - y[test]
        assert got[f"{name}_RMSE"] == pytest.approx(np.sqrt(np.mean(err**2)), rel=RTOL)
        assert got[f"{name}_MAE"] == pytest.approx(np.mean(np.abs(err)), rel=RTOL)
        # NMAD = median|err| / Phi^-1(0.75); 80 test points (even count).
        assert len(err) % 2 == 0
        assert got[f"{name}_NMAD"] == pytest.approx(_MAD_SCALE * np.median(np.abs(err)), rel=RTOL)


def test_report_equals_components(rng: np.random.Generator) -> None:
    f, y = _features(rng, n=200)
    report = representation_geometry_report(f, y, k=7, readouts=True, seed=5)
    expected = {
        "neighbour_target_gap": neighbour_target_gap(f, y, k=7),
        "target_ordering": target_ordering(f, y, seed=5),
        **frozen_readout_metrics(f, y, seed=5),
    }
    assert report.keys() == expected.keys()
    for key, value in expected.items():
        assert report[key] == pytest.approx(value, rel=RTOL, abs=ATOL), key


@pytest.mark.parametrize("alpha", [0.32, 0.05])
def test_paired_bootstrap_ci_matches_scipy(rng: np.random.Generator, alpha: float) -> None:
    """Percentile bootstrap of mean(a - b): torchregress vs SciPy (paired=True).

    Both estimate the same bootstrap-distribution quantiles from independent
    resamples.  For ``B`` resamples the standard error of the ``p`` quantile is
    ``sqrt(p (1 - p) / B) / f(q_p)``; the bootstrap distribution of the mean is
    ~Normal(mean, s / sqrt(n)), so ``f(q_p) = phi(z_p) / (s / sqrt(n))``.  The
    difference of two independent estimates has SE ``sqrt(2)`` times that; we
    allow 6 SE (two-sided miss probability ~2e-9 per bound under the normal
    approximation).  The seeds are fixed, so the test is deterministic.
    """
    n, n_boot = 300, 20_000
    a = rng.gamma(2.0, 1.0, size=n)
    b = a + rng.normal(0.1, 0.5, size=n)
    diff, lo, hi = paired_bootstrap_ci(a, b, n_boot=n_boot, alpha=alpha, seed=11)
    assert diff == pytest.approx(np.mean(a - b), rel=1e-12, abs=1e-15)
    ref = bootstrap(
        (a, b),
        lambda x, y, axis=-1: np.mean(x - y, axis=axis),
        paired=True,
        vectorized=True,
        n_resamples=n_boot,
        confidence_level=1.0 - alpha,
        method="percentile",
        random_state=np.random.default_rng(12),
    ).confidence_interval
    s = np.std(a - b, ddof=1)
    p = alpha / 2.0
    density = norm.pdf(norm.ppf(p)) / (s / np.sqrt(n))
    tol = 6.0 * np.sqrt(2.0) * np.sqrt(p * (1 - p) / n_boot) / density
    assert abs(lo - ref.low) <= tol, (lo, ref.low, tol)
    assert abs(hi - ref.high) <= tol, (hi, ref.high, tol)
    assert lo < diff < hi
