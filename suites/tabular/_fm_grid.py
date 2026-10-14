"""Shared quantile-grid bases for the foundation-model UQ suites.

Every base model (tabular foundation models and two references) is reduced to
the same object: a predictive quantile grid ``q[n, K]`` at the shared levels
``LEVELS``.  The grid is the model's own predictive distribution (TabICL /
TabPFN / sdm quantile heads, LightGBM quantile regressors, a Gaussian
TabularMLP ensemble), so distribution scores (CRPS, PIT) and every interval
level come from one representation, and the torchregress layer acts on it.

Used by ``fm_uq_scenarios`` (single target, stress scenarios) and
``fm_multitarget`` (per-target marginals joined by a copula).
"""

from __future__ import annotations

import os
from statistics import NormalDist
from time import perf_counter
from typing import Any

import numpy as np
import torch

from harness.devices import current_device
from harness.runners import HarnessConfig

# 99 equal-mass midpoints plus four tail levels (the 95% / 99% intervals and the
# PIT recalibration need levels beyond 0.005 / 0.995).
LEVELS = np.unique(np.r_[0.002, 0.005, (np.arange(99) + 0.5) / 99, 0.995, 0.998])
_GBM_LEVELS = np.array(
    [0.002, 0.01, 0.025, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.975, 0.99, 0.998]
)


def tabpfn_enabled() -> bool:
    """TabPFN rows run when opted in or when a Prior Labs API key is present."""
    return os.environ.get("TORCHREGRESS_HARNESS_WITH_TABPFN") == "1" or bool(
        os.environ.get("TABPFN_TOKEN")
    )


def gbm_base_name() -> str:
    from suites.tabular.uq_layer import _GBM_BASE

    return _GBM_BASE


def available_bases(*, references: bool = True) -> list[str]:
    """Foundation models (latest of each family) plus, optionally, the references."""
    from suites.tabular.foundation_model import _SDM_MODELS

    out = [gbm_base_name(), "TabularMLP"] if references else []
    out += ["TabICL", *_SDM_MODELS]
    if tabpfn_enabled():
        out.append("TabPFN")
    return out


def library_of(base: str) -> str:
    from suites.tabular.foundation_model import _SDM_MODELS

    if base in _SDM_MODELS:
        return "sdm"
    return {"TabICL": "tabicl", "TabPFN": "tabpfn", "TabularMLP": "torchregress"}.get(
        base, "lightgbm"
    )


# ---------------------------------------------------------------------------
# Fitting: base -> quantile grids on each evaluation matrix
# ---------------------------------------------------------------------------


def _interp_levels(q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Re-express a sorted grid at ``levels`` on ``LEVELS`` (flat beyond the ends)."""
    q = np.sort(np.asarray(q, dtype=np.float64), axis=1)
    out = np.empty((q.shape[0], len(LEVELS)))
    j = np.clip(np.searchsorted(levels, LEVELS), 1, len(levels) - 1)
    w = np.clip((LEVELS - levels[j - 1]) / (levels[j] - levels[j - 1]), 0.0, 1.0)
    out[:] = q[:, j - 1] * (1 - w) + q[:, j] * w
    return out


def fit_grids(
    base: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: list[np.ndarray],
    cfg: HarnessConfig,
) -> tuple[list[np.ndarray], float]:
    """Fit ``base`` on (X_train, y_train); quantile grids [n, K] on each X_eval."""
    t0 = perf_counter()
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    seed = int(cfg.seed)
    from suites.tabular.foundation_model import _SDM_MODELS

    if base == gbm_base_name():
        from suites.tabular.uq_layer import _gbm_estimator

        models = [
            _gbm_estimator(seed, cfg.smoke, quantile=float(a)).fit(X_train, y_train)
            for a in _GBM_LEVELS
        ]
        grids = [
            _interp_levels(np.stack([m.predict(Xe) for m in models], axis=1), _GBM_LEVELS)
            for Xe in X_eval
        ]
    elif base == "TabularMLP":
        from suites._tabular_mlp import fit_tabmlp, predict_gaussian

        fit = fit_tabmlp(X_train, y_train, cfg, kind="gaussian", n_members=5)
        z = np.array([NormalDist().inv_cdf(float(t)) for t in LEVELS])
        grids = []
        for Xe in X_eval:
            mu, sd = predict_gaussian(fit, Xe)
            grids.append(np.asarray(mu).reshape(-1, 1) + np.asarray(sd).reshape(-1, 1) * z)
    elif base == "TabICL":
        from tabicl import TabICLRegressor

        est = TabICLRegressor(device=str(current_device()), random_state=seed)
        est.fit(X_train, y_train)
        grids = [
            np.sort(
                np.asarray(
                    est.predict(Xe, output_type="quantiles", alphas=LEVELS.tolist())
                ).reshape(len(Xe), -1),
                axis=1,
            )
            for Xe in X_eval
        ]
    elif base == "TabPFN":
        from tabpfn import TabPFNRegressor

        est = TabPFNRegressor(device=str(current_device()), random_state=seed)
        est.fit(X_train, y_train)
        grids = []
        for Xe in X_eval:
            q = est.predict(Xe, output_type="quantiles", quantiles=LEVELS.tolist())
            grids.append(np.sort(np.stack([np.asarray(c) for c in q], axis=1), axis=1))
    elif base in _SDM_MODELS:
        import sdm.models
        from sdm import TableTensor

        cls_name, kwargs = _SDM_MODELS[base]
        dev = current_device()
        model = getattr(sdm.models, cls_name)(task="regression", device=dev, **kwargs)
        model.eval()
        model.fit(
            TableTensor.from_tensor(torch.as_tensor(X_train)).to(dev),
            TableTensor.from_tensor(
                torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
            ).to(dev),
            generator=torch.Generator(device=dev).manual_seed(seed),
        )
        grids = []
        for Xe in X_eval:
            parts = []
            for start in range(0, len(Xe), 2048):  # bounded GPU memory on 6k-row evals
                xb = torch.as_tensor(np.asarray(Xe[start : start + 2048], dtype=np.float32))
                with torch.no_grad():
                    out = model.predict(TableTensor.from_tensor(xb).to(dev))
                parts.append(out.numerical.detach().float().cpu().numpy().reshape(len(xb), -1))
            q = np.concatenate(parts)
            if q.shape[1] == 1:  # point head only (TabFM): a point mass, see is_point()
                grids.append(np.repeat(q, len(LEVELS), axis=1))
                continue
            lv = (np.arange(q.shape[1]) + 1) / (q.shape[1] + 1)
            grids.append(_interp_levels(q, lv))
    else:
        raise ValueError(f"unknown base {base!r}")
    return [np.sort(np.asarray(g, dtype=np.float64), axis=1) for g in grids], perf_counter() - t0


# ---------------------------------------------------------------------------
# Grid maths (rows share LEVELS)
# ---------------------------------------------------------------------------


_Z = np.array([NormalDist().inv_cdf(float(t)) for t in LEVELS])
_TAU_MIN = 1e-6  # deepest tail level reached by extrapolation (|z| ~ 4.75)


def _tail_slopes(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row quantile slope dq/dz on the outermost grid interval of each tail."""
    lo = (q[:, 1] - q[:, 0]) / (_Z[1] - _Z[0])
    hi = (q[:, -1] - q[:, -2]) / (_Z[-1] - _Z[-2])
    return np.maximum(lo, 1e-12), np.maximum(hi, 1e-12)


def quantile_at(q: np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    """Quantile at level ``tau``: a scalar -> [n]; an array [n, S] -> [n, S] (per row).

    Linear between grid levels; beyond the outermost levels the grid is extended
    linearly in normal-score space (Gaussian-shaped tails with the slope of the
    last grid interval), so recalibration can widen a too-narrow distribution.
    """
    tau = np.clip(np.asarray(tau, dtype=np.float64), _TAU_MIN, 1 - _TAU_MIN)
    j = np.clip(np.searchsorted(LEVELS, tau), 1, len(LEVELS) - 1)
    w = np.clip((tau - LEVELS[j - 1]) / (LEVELS[j] - LEVELS[j - 1]), 0.0, 1.0)
    s_lo, s_hi = _tail_slopes(q)
    from scipy.special import ndtri

    z = ndtri(tau)
    if tau.ndim == 0:
        out = q[:, j - 1] * (1 - w) + q[:, j] * w
        if tau < LEVELS[0]:
            out = q[:, 0] + s_lo * (z - _Z[0])
        elif tau > LEVELS[-1]:
            out = q[:, -1] + s_hi * (z - _Z[-1])
        return out
    rows = np.arange(q.shape[0])[:, None]
    out = q[rows, j - 1] * (1 - w) + q[rows, j] * w
    out = np.where(tau < LEVELS[0], q[:, :1] + s_lo[:, None] * (z - _Z[0]), out)
    return np.where(tau > LEVELS[-1], q[:, -1:] + s_hi[:, None] * (z - _Z[-1]), out)


def cdf_at(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Predictive CDF F(y | x) per row (linear inside the grid, Gaussian tails outside)."""
    from scipy.special import ndtr

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    k = (q < y[:, None]).sum(1)
    kk = np.clip(k, 1, q.shape[1] - 1)
    rows = np.arange(len(y))
    lo, hi = q[rows, kk - 1], q[rows, kk]
    w = np.where(hi > lo, (y - lo) / np.where(hi > lo, hi - lo, 1.0), 0.5)
    F = LEVELS[kk - 1] + np.clip(w, 0, 1) * (LEVELS[kk] - LEVELS[kk - 1])
    s_lo, s_hi = _tail_slopes(q)
    F = np.where(k == 0, ndtr(_Z[0] + (y - q[:, 0]) / s_lo), F)
    F = np.where(k == q.shape[1], ndtr(_Z[-1] + (y - q[:, -1]) / s_hi), F)
    return np.clip(F, _TAU_MIN, 1 - _TAU_MIN)


def crps(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per row as twice the integrated pinball loss over LEVELS."""
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    u = y - q
    pin = np.maximum(LEVELS * u, (LEVELS - 1) * u)
    return 2.0 * np.trapezoid(pin, LEVELS, axis=1)


def pit_ks(pit: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of PIT values from Uniform(0, 1)."""
    p = np.sort(np.asarray(pit, dtype=np.float64).reshape(-1))
    n = len(p)
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(n) / n
    return float(max(np.max(np.abs(ecdf_hi - p)), np.max(np.abs(p - ecdf_lo))))


def is_point(q: np.ndarray) -> bool:
    """True when a grid is a point mass (the model has no predictive distribution)."""
    return bool(np.allclose(q[:, 0], q[:, -1]))


def residual_grid(point_cal: np.ndarray, y_cal: np.ndarray, point_eval: np.ndarray) -> np.ndarray:
    """Split conformal predictive distribution: point + calibration-residual quantiles.

    ``F(y | x) = #{i : point(x) + r_i <= y} / (n + 1)`` (Vovk et al. 2017),
    represented on LEVELS; gives a distribution to point-only models and a
    homoscedastic baseline for every model.
    """
    r = np.sort(np.asarray(y_cal, dtype=np.float64) - np.asarray(point_cal, dtype=np.float64))
    n = len(r)
    lv = (np.arange(1, n + 1)) / (n + 1)
    qr = np.interp(LEVELS, lv, r)
    return np.asarray(point_eval, dtype=np.float64).reshape(-1, 1) + qr[None, :]


def pit_recalibrate(q_cal: np.ndarray, y_cal: np.ndarray, q_test: np.ndarray) -> np.ndarray:
    """Quantile recalibration (Kuleshov et al. 2018) via torchregress ``PITCalibrator``.

    The calibrated CDF is ``G(F(y | x))`` with ``G`` the empirical CDF of the
    calibration PITs; its quantile at level ``tau`` is ``F^{-1}(G^{-1}(tau))``.
    """
    from torchregress.calibration import PITCalibrator

    cal = PITCalibrator().fit(torch.as_tensor(cdf_at(q_cal, y_cal), dtype=torch.float64))
    fine = np.r_[
        np.geomspace(1e-6, 1e-3, 200),
        np.linspace(1e-3, 1 - 1e-3, 4001)[1:-1],
        1 - np.geomspace(1e-3, 1e-6, 200),
    ]
    g = cal.transform(torch.as_tensor(fine, dtype=torch.float64)).numpy()
    # Invert G in normal-score space; beyond the range G reaches (0.5/n .. 1-0.5/n
    # for n calibration rows) shift by a constant z offset (unit slope).
    from scipy.special import ndtr, ndtri

    g, first = np.unique(np.maximum.accumulate(g), return_index=True)
    zg, zx, zl = ndtri(g), ndtri(fine[first]), ndtri(LEVELS)
    zt = np.interp(zl, zg, zx)
    zt = np.where(zl < zg[0], zx[0] + (zl - zg[0]), zt)
    zt = np.where(zl > zg[-1], zx[-1] + (zl - zg[-1]), zt)
    tau_new = ndtr(zt)
    return np.sort(np.stack([quantile_at(q_test, t) for t in tau_new], axis=1), axis=1)


def interval(q: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    return quantile_at(q, alpha / 2), quantile_at(q, 1 - alpha / 2)


def ppf_normal(u: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri

    return ndtri(np.clip(u, 1e-6, 1 - 1e-6))


def cdf_normal(z: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return ndtr(z)
