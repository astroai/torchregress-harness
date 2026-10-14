"""Metrics parity: torchregress scoring rules vs ``scoringrules`` on random inputs.

Every comparison runs in float64 on seeded random forecasts, so agreement is
to round-off (``RTOL``/``ATOL`` below), not to a statistical tolerance.  Where
the two libraries define a score with a different convention the test pins
the exact relation instead of loosening the tolerance:

* variogram score: torchregress sums the unordered pairs ``i < j``;
  ``scoringrules.vs_ensemble`` sums the ordered pairs ``i != j`` (factor 2).
* quantile CRPS: torchregress integrates ``2 * QS(tau)`` with trapezoid
  weights on ``[0, levels..., 1]``; ``scoringrules.crps_quantile`` uses equal
  weights ``2 / M``.  On the equispaced grid ``tau_i = i / (M + 1)`` the two
  differ by exactly ``M / (M + 1)``; both converge to ``crps_normal``.
* vario score (Zamo & Naveau) is positively oriented: at ``rho = 1`` it is
  ``-CRPS`` with the fair ensemble estimator.
* Dawid-Sebastiani: ``scoringrules.dssuv_ensemble(bias=True)`` on the
  ensemble's population moments.

Metrics with no ``scoringrules`` counterpart that the export-coverage
allowlist schedules for these parity tests (2-Wasserstein, robust point
metrics) are checked against an independent closed form or SciPy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchmetrics import Metric

sr = pytest.importorskip("scoringrules")

from torchregress import metrics as M  # noqa: E402

RTOL = 1e-10
ATOL = 1e-12
N, S, D = 64, 48, 4


@pytest.fixture(params=[0, 1, 2], ids=lambda s: f"seed{s}")
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    return np.random.default_rng(request.param)


def _np(x: object) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _close(actual: object, expected: object, rtol: float = RTOL, atol: float = ATOL) -> None:
    np.testing.assert_allclose(_np(actual), _np(expected), rtol=rtol, atol=atol)


def _f64(metric: Metric) -> Metric:
    """torchmetrics states default to float32; parity is checked in float64."""
    metric.set_dtype(torch.float64)
    return metric


def _gaussian(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = rng.normal(size=N)
    sd = rng.uniform(0.2, 3.0, size=N)
    y = mu + sd * rng.standard_t(df=4, size=N)  # heavier tails than the forecast
    return mu, sd, y


# ---------------------------------------------------------------- univariate


def test_crps_gaussian(rng: np.random.Generator) -> None:
    mu, sd, y = _gaussian(rng)
    ref = sr.crps_normal(y, mu, sd)
    _close(M.crps_gaussian(mu, y, sd, reduction="none"), ref)
    _close(M.crps_gaussian(mu, y, sd), ref.mean())


def test_crps_ensemble_fair(rng: np.random.Generator) -> None:
    ens = rng.normal(size=(S, N)) * rng.uniform(0.5, 2.0, size=N) + rng.normal(size=N)
    y = rng.normal(size=N)
    ref = sr.crps_ensemble(y, ens, m_axis=0, estimator="fair")
    _close(M.crps_from_samples(ens, y, reduction="none"), ref)
    _close(M.crps_from_samples(ens, y), ref.mean())


def test_log_score_gaussian(rng: np.random.Generator) -> None:
    mu, sd, y = _gaussian(rng)
    ref = sr.logs_normal(y, mu, sd)
    _close(M.gaussian_nll(mu, y, sd**2, reduction="none"), ref)
    _close(M.gaussian_nll(mu, y, sd**2), ref.mean())


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.32])
def test_interval_score(rng: np.random.Generator, alpha: float) -> None:
    mu, sd, y = _gaussian(rng)
    half = sd * rng.uniform(0.5, 2.0, size=N)
    lo, hi = mu - half, mu + half
    ref = sr.interval_score(y, lo, hi, alpha)
    _close(M.interval_score(lo, hi, y, alpha=alpha, reduction="none"), ref)
    metric = _f64(M.IntervalScore(alpha=alpha))
    metric.update(torch.from_numpy(lo), torch.from_numpy(hi), torch.from_numpy(y))
    _close(metric.compute(), ref.mean())


def test_dawid_sebastiani(rng: np.random.Generator) -> None:
    ens = rng.normal(size=(S, N)) * rng.uniform(0.5, 2.0, size=N) + rng.normal(size=N)
    y = rng.normal(size=N)
    ref = sr.dssuv_ensemble(y, ens, m_axis=0, bias=True)
    m, s = ens.mean(axis=0), ens.std(axis=0)
    _close(M.dss_score(m, s, y), ref.mean())
    metric = _f64(M.DawidSebastianiScore())
    metric.update(torch.from_numpy(m[:20]), torch.from_numpy(s[:20]), torch.from_numpy(y[:20]))
    metric.update(torch.from_numpy(m[20:]), torch.from_numpy(s[20:]), torch.from_numpy(y[20:]))
    _close(metric.compute(), ref.mean())


@pytest.mark.parametrize("level", [0.05, 0.3, 0.5, 0.9])
def test_pinball_single_level(rng: np.random.Generator, level: float) -> None:
    mu, sd, y = _gaussian(rng)
    q = mu + sd * rng.normal(size=N)
    ref = sr.quantile_score(y, q, level)
    _close(M.pinball_loss(level, torch.from_numpy(q), torch.from_numpy(y)), ref)


def _quantile_forecast(
    rng: np.random.Generator, levels: np.ndarray
) -> tuple[dict[float, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.stats import norm

    mu, sd, y = _gaussian(rng)
    qs = mu[:, None] + sd[:, None] * norm.ppf(levels)[None, :]
    return {float(t): qs[:, i] for i, t in enumerate(levels)}, qs, mu, sd, y


def test_pinball_metric_mean_over_levels(rng: np.random.Generator) -> None:
    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    qd, qs, _, _, y = _quantile_forecast(rng, levels)
    ref = np.mean([sr.quantile_score(y, qs[:, i], t) for i, t in enumerate(levels)])
    _close(M.pinball_metric(qd, y), ref)
    metric = _f64(M.PinballMetric())
    metric.update({k: torch.from_numpy(v) for k, v in qd.items()}, torch.from_numpy(y))
    _close(metric.compute(), ref)


@pytest.mark.parametrize("m", [3, 9, 49])
def test_quantile_crps_equispaced_relation(rng: np.random.Generator, m: int) -> None:
    levels = np.arange(1, m + 1) / (m + 1)
    qd, qs, _, _, y = _quantile_forecast(rng, levels)
    ref = sr.crps_quantile(y, qs, levels, m_axis=-1)
    ratio = m / (m + 1)  # trapezoid on [0, tau..., 1] vs equal 2/M weights
    _close(M.continuous_ranked_probability_score(qd, y, reduction="none"), ratio * ref)
    metric = _f64(M.ContinuousRankedProbabilityScore())
    metric.update({k: torch.from_numpy(v) for k, v in qd.items()}, torch.from_numpy(y))
    _close(metric.compute(), ratio * ref.mean())


def test_quantile_crps_converges_to_analytic(rng: np.random.Generator) -> None:
    levels = np.arange(1, 2000) / 2000
    qd, _, mu, sd, y = _quantile_forecast(rng, levels)
    exact = sr.crps_normal(y, mu, sd)
    got = _np(M.continuous_ranked_probability_score(qd, y, reduction="none"))
    # Discretisation error, not round-off: the dense grid gets within 0.5%.
    np.testing.assert_allclose(got, exact, rtol=5e-3, atol=1e-4)


@pytest.mark.parametrize("rho", [1.0, 0.5, 2.0])
def test_vario_score(rng: np.random.Generator, rho: float) -> None:
    ens = rng.normal(size=(S, N)) * rng.uniform(0.5, 2.0, size=N) + rng.normal(size=N)
    y = rng.normal(size=N)
    # nu_rho = 0.5 E|X - X'|^rho - E|X - y|^rho, fair pair estimator.
    term_obs = np.mean(np.abs(ens - y[None]) ** rho, axis=0)
    pair = np.abs(ens[:, None, :] - ens[None, :, :]) ** rho
    term_spread = pair.sum(axis=(0, 1)) / (S * (S - 1)) / 2.0
    expected = (term_spread - term_obs).mean()
    _close(M.vario_score(ens, y, rho=rho), expected)
    if rho == 1.0:
        _close(
            M.vario_score(ens, y, rho=1.0),
            -sr.crps_ensemble(y, ens, m_axis=0, estimator="fair").mean(),
        )
    metric = _f64(M.VarioScore(rho=rho))
    metric.update(torch.from_numpy(ens[:, :30]), torch.from_numpy(y[:30]))
    metric.update(torch.from_numpy(ens[:, 30:]), torch.from_numpy(y[30:]))
    _close(metric.compute(), expected)


# -------------------------------------------------------------- multivariate


def _mv_ensemble(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = rng.normal(size=(D, D))
    ens = rng.normal(size=(S, N, D)) @ a.T + rng.normal(size=(N, D))
    y = rng.normal(size=(N, D)) * 1.3
    return ens, y


def test_energy_score_fair(rng: np.random.Generator) -> None:
    ens, y = _mv_ensemble(rng)
    ref = sr.es_ensemble(y, ens, m_axis=0, v_axis=-1, estimator="fair")
    _close(M.energy_score(ens, y, reduction="none"), ref)
    metric = _f64(M.EnergyScore())
    metric.update(torch.from_numpy(ens), torch.from_numpy(y))
    _close(metric.compute(), ref.mean())


def test_energy_score_univariate_is_crps(rng: np.random.Generator) -> None:
    ens, y = _mv_ensemble(rng)
    ref = sr.crps_ensemble(y[:, 0], ens[:, :, 0], m_axis=0, estimator="fair")
    _close(M.energy_score(ens[..., :1], y[:, :1], reduction="none"), ref)


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_variogram_score(rng: np.random.Generator, p: float) -> None:
    ens, y = _mv_ensemble(rng)
    ref = sr.vs_ensemble(y, ens, m_axis=0, v_axis=-1, p=p, estimator="nrg")
    # scoringrules sums ordered pairs i != j; torchregress the pairs i < j.
    _close(M.variogram_score(ens, y, p=p, reduction="none"), 0.5 * ref)
    metric = _f64(M.VariogramScore(p=p))
    metric.update(torch.from_numpy(ens), torch.from_numpy(y))
    _close(metric.compute(), 0.5 * ref.mean())


# ------------------------------------- allowlisted metrics without an sr twin


def test_wasserstein_gaussian_p2_diag_quadrature(rng: np.random.Generator) -> None:
    from scipy.integrate import quad
    from scipy.stats import norm

    m1, m2 = rng.normal(size=D), rng.normal(size=D)
    s1, s2 = rng.uniform(0.3, 2.0, size=D), rng.uniform(0.3, 2.0, size=D)
    # W2^2 = sum_d int_0^1 (F1^-1(u) - F2^-1(u))^2 du (independent dims).
    w2_sq = sum(
        quad(
            lambda u, d=d: (m1[d] + s1[d] * norm.ppf(u) - m2[d] - s2[d] * norm.ppf(u)) ** 2,
            0.0,
            1.0,
            epsabs=1e-13,
            epsrel=1e-13,
            limit=200,
        )[0]
        for d in range(D)
    )
    _close(M.wasserstein_gaussian_p2(m1, s1, m2, s2), np.sqrt(w2_sq), rtol=1e-8)


def test_wasserstein_gaussian_p2_full_cov(rng: np.random.Generator) -> None:
    from scipy.linalg import sqrtm

    a, b = rng.normal(size=(D, D)), rng.normal(size=(D, D))
    c1, c2 = a @ a.T + 0.1 * np.eye(D), b @ b.T + 0.1 * np.eye(D)
    m1, m2 = rng.normal(size=D), rng.normal(size=D)
    r1 = sqrtm(c1).real
    cross = sqrtm(r1 @ c2 @ r1).real
    ref = np.sqrt(np.sum((m1 - m2) ** 2) + np.trace(c1 + c2 - 2.0 * cross))
    _close(M.wasserstein_gaussian_p2(m1, c1, m2, c2), ref, rtol=1e-8)


def test_wasserstein_gaussian_metric_to_dirac(rng: np.random.Generator) -> None:
    mu, sd, y = _gaussian(rng)
    metric = _f64(M.WassersteinGaussian())
    metric.update(torch.from_numpy(mu), torch.from_numpy(sd), torch.from_numpy(y))
    per_point = [M.wasserstein_gaussian_p2(mu[i], sd[i], y[i], 0.0) for i in range(N)]
    _close(metric.compute(), np.mean(per_point))


def test_robust_point_metrics_vs_scipy(rng: np.random.Generator) -> None:
    from scipy.special import huber
    from scipy.stats import median_abs_deviation, trim_mean

    y = rng.standard_t(df=2, size=N + 1)  # odd count, heavy tails
    yp = y + rng.standard_t(df=2, size=N + 1)
    e = y - yp
    t_yp, t_y = torch.from_numpy(yp), torch.from_numpy(y)

    for delta in (0.5, 1.0, 2.0):
        ref = huber(delta, e).mean()
        _close(M.huber_loss(yp, y, delta=delta), ref)
        hm = _f64(M.HuberMetric(delta=delta))
        hm.update(t_yp, t_y)
        _close(hm.compute(), ref)

    for prop in (0.0, 0.1, 0.25):
        ref = trim_mean(e**2, prop)
        _close(M.trimmed_mean_squared_error(yp, y, proportion=prop), ref)
        tm = _f64(M.TrimmedMeanSquaredError(proportion=prop))
        tm.update(t_yp, t_y)
        _close(tm.compute(), ref)

    ref = 1.4826 * median_abs_deviation(e, scale=1.0)
    _close(M.median_absolute_deviation(yp, y), ref)
    mad = _f64(M.MedianAbsoluteDeviation())
    mad.update(t_yp, t_y)
    _close(mad.compute(), ref)

    rmse = np.sqrt(np.mean(e**2))
    q75, q25 = np.quantile(y, [0.75, 0.25])
    norms = {
        "std": np.std(y, ddof=1),
        "range": np.ptp(y),
        "mean": np.mean(np.abs(y)),
        "iqr": q75 - q25,
    }
    for name, denom in norms.items():
        _close(M.normalized_rmse(yp, y, normalization=name), rmse / denom)
        nr = _f64(M.NormalizedRMSE(normalization=name))
        nr.update(t_yp, t_y)
        _close(nr.compute(), rmse / denom)
