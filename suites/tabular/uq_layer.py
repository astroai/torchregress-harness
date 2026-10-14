"""torchregress as the UQ layer on top of any base model.

Plan Phase 2b, item A2.  The August leaderboard showed that on interval
metrics torchregress loses mainly because its 60-epoch MLP is a weaker *base
model* than TabICL or gradient boosting, not because its conformal or
calibration layer is worse.  This suite separates the two: each base model is
fitted **once**, and every UQ method -- torchregress, MAPIE and crepes -- is
applied to the **same** base predictions on the **same** train /
calibration / test split.

Base models
-----------
* ``LightGBM`` -- ``LGBMRegressor`` point model (L2) plus quantile models at
  ``alpha/2``, ``0.5`` and ``1 - alpha/2`` (sklearn
  ``HistGradientBoostingRegressor`` when lightgbm is missing).
* ``RF`` -- ``RandomForestRegressor``: point = forest mean, quantiles and
  spread from the per-tree predictions (epistemic spread only, so the raw
  intervals under-cover; that is what the UQ layer has to fix).
* ``TabICL`` -- point and ``alpha/2, 0.5, 1 - alpha/2`` predictive quantiles;
  every row fails with the reason when the checkpoint cannot be downloaded.
* ``DeepEnsK5`` -- 5 Gaussian MLPs (``BetaNLLLoss(beta=0.5)``, the ``uq``
  suite recipe) combined as a Gaussian mixture.
* ``TabularMLP`` -- 5 ``torchregress.models.TabularMLP`` Gaussian members
  (``fit_tabular_ensemble`` with library defaults, seeds ``seed..seed+4``)
  combined as a Gaussian mixture; the "strong by default" base.

UQ methods (row ``<method>+<base>``)
-----------------------------------
* ``Raw`` -- the base model's own 90% interval and its Gaussian view.
* ``tr_Split`` / ``MAPIE_Split`` / ``crepes_Split`` -- absolute-residual
  split conformal (``SplitConformal``; MAPIE ``SplitConformalRegressor``
  prefit; crepes ``ConformalRegressor``).
* ``tr_Normalized`` / ``crepes_Normalized`` -- residual / spread scores
  (``SplitConformal(normalize_fn=...)``; crepes ``sigmas=``), spread = the
  base's predictive std or quantile width.
* ``tr_CQR`` / ``MAPIE_CQR`` / ``MAPIE_CQR_asym`` -- conformalized quantile
  regression on the base quantiles (MAPIE ``ConformalizedQuantileRegressor``
  prefit, symmetric correction for parity and its default asymmetric one).
  crepes has no CQR and its ``ConformalRegressor`` takes ``|score|`` (checked:
  ``np.abs(residuals)`` in ``fit``), so feeding it signed CQR scores gives
  wider intervals; there is deliberately no crepes CQR row.
* ``tr_CVPlus`` / ``MAPIE_CVPlus`` -- CV+ (K=5, same folds) on train + cal
  pooled (refits the base; LightGBM and RF only).
* ``tr_JackknifePlus`` / ``MAPIE_JackknifePlus`` -- Jackknife+ (LOO refits;
  LightGBM, pooled n <= 500 only).
* ``tr_VTS`` / ``tr_IsoMean+VTS`` -- ``VarianceTemperatureScaler`` on the
  base's Gaussian view (and ``IsotonicMeanCalibrator`` on the mean first,
  fitted on one half of the calibration split, VTS on the other).

"Gaussian view": ``DeepEnsK5`` and ``TabularMLP`` are Gaussian; RF uses the forest mean and the
per-tree std; quantile bases use the point prediction and
``(q_hi - q_lo) / (2 z_{1-alpha/2})``.  Rows with a Gaussian view get
NLL / CRPS / ECE (the harness Gaussian PIT, as in ``uq``); interval-only
rows get Coverage / Width / IntervalScore / InfiniteIntervalFrac and point
metrics.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.metrics import (
    compute_interval_metrics,
    compute_point_metrics,
    infinite_interval_note,
)
from harness.runners import (
    HarnessConfig,
    HarnessRunner,
    _inverse_transform_predictions,
    _inverse_transform_y,
)
from suites._backbones import (
    _MLP,
    _carve_calibration_split,
    _gaussian_nll_predict,
    _hidden_size,
    _mixture_gaussian_stats,
    _train_one,
)
from suites._tabular_mlp import fit_tabmlp, predict_gaussian
from suites.tabular.conformal import (
    _as_float64,
    _mapie_coverage_and_width,
    _mapie_intervals_to_bounds,
    _mapie_version,
    mapie_predict_blocked,
)

try:
    import lightgbm  # noqa: F401

    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

try:
    import mapie  # noqa: F401

    _MAPIE_AVAILABLE = True
except ImportError:
    _MAPIE_AVAILABLE = False

try:
    from crepes import ConformalRegressor

    _CREPES_AVAILABLE = True
except ImportError:
    _CREPES_AVAILABLE = False

try:
    import tabicl  # noqa: F401

    _TABICL_AVAILABLE = True
except ImportError:
    _TABICL_AVAILABLE = False

_GBM_BASE = "LightGBM" if _LIGHTGBM_AVAILABLE else "HistGB"
_N_ENSEMBLE = 5
_N_FOLDS = 5
_JACKKNIFE_MAX_N = 500  # Jackknife+ refits n models: small datasets only
_REFIT_BASES = (_GBM_BASE, "RF")  # bases cheap enough to refit for CV+

# Methods, in row order, with what each needs from the base model.
_SPLIT_METHODS = ("Raw", "tr_Split", "MAPIE_Split", "crepes_Split")
_NORMALIZED_METHODS = ("tr_Normalized", "crepes_Normalized")
# crepes has no CQR: its ConformalRegressor takes |score|, which would discard the
# sign of the CQR score and widen the interval, so no crepes CQR row is reported.
_CQR_METHODS = ("tr_CQR", "MAPIE_CQR", "MAPIE_CQR_asym")
_CV_METHODS = ("tr_CVPlus", "MAPIE_CVPlus")
_JACKKNIFE_METHODS = ("tr_JackknifePlus", "MAPIE_JackknifePlus")
_CALIBRATION_METHODS = ("tr_VTS", "tr_IsoMean+VTS")


class _Skip(Exception):
    """A method that does not apply to this dataset size (skipped row, not failed)."""


# ---------------------------------------------------------------------------
# Base predictions shared by every UQ method
# ---------------------------------------------------------------------------


@dataclass
class _BasePreds:
    """One base model's predictions on the calibration and test splits."""

    point_cal: np.ndarray
    point_test: np.ndarray
    q_cal: np.ndarray  # [n, 3] sorted (lo, median, hi)
    q_test: np.ndarray
    mean_cal: np.ndarray  # Gaussian view
    std_cal: np.ndarray
    mean_test: np.ndarray
    std_test: np.ndarray
    spread_cal: np.ndarray  # difficulty for normalized conformal
    spread_test: np.ndarray
    factory: Callable[[], Any] | None = None  # sklearn point estimator, for CV+
    library: str = "torchregress"
    fit_s: float = 0.0


def _z(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _floor_spread(spread: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Floor a spread at 5% of the calibration median so no score divides by ~0."""
    floor = max(0.05 * float(np.median(ref)), 1e-6)
    return np.maximum(np.asarray(spread, dtype=np.float64), floor)


def _from_quantiles(
    point_cal: np.ndarray,
    point_test: np.ndarray,
    q_cal: np.ndarray,
    q_test: np.ndarray,
    *,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Sort quantiles (crossing fix, applied once for every library) and derive the views."""
    q_cal = np.sort(np.asarray(q_cal, dtype=np.float64), axis=1)
    q_test = np.sort(np.asarray(q_test, dtype=np.float64), axis=1)
    w_cal = q_cal[:, 2] - q_cal[:, 0]
    w_test = q_test[:, 2] - q_test[:, 0]
    spread_cal = _floor_spread(w_cal, w_cal)
    spread_test = _floor_spread(w_test, w_cal)
    z2 = 2.0 * _z(alpha)
    return {
        "point_cal": np.asarray(point_cal, dtype=np.float64).reshape(-1),
        "point_test": np.asarray(point_test, dtype=np.float64).reshape(-1),
        "q_cal": q_cal,
        "q_test": q_test,
        "mean_cal": np.asarray(point_cal, dtype=np.float64).reshape(-1),
        "std_cal": spread_cal / z2,
        "mean_test": np.asarray(point_test, dtype=np.float64).reshape(-1),
        "std_test": spread_test / z2,
        "spread_cal": spread_cal,
        "spread_test": spread_test,
    }


def _gbm_estimator(seed: int, smoke: bool, *, quantile: float | None = None) -> Any:
    """LightGBM (or sklearn HistGB) point / quantile regressor, single-threaded."""
    n_estimators = 50 if smoke else 300
    if _LIGHTGBM_AVAILABLE:
        from lightgbm import LGBMRegressor

        kw: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 10,
            "subsample": 1.0,
            "n_jobs": 1,
            "random_state": seed,
            "verbose": -1,
        }
        if quantile is not None:
            kw.update(objective="quantile", alpha=quantile)
        return LGBMRegressor(**kw)
    from sklearn.ensemble import HistGradientBoostingRegressor

    kw2: dict[str, Any] = {"max_iter": n_estimators, "learning_rate": 0.05, "random_state": seed}
    if quantile is not None:
        kw2.update(loss="quantile", quantile=quantile)
    return HistGradientBoostingRegressor(**kw2)


def _fit_gbm(s: dict[str, Any], *, alpha: float, seed: int, smoke: bool) -> _BasePreds:
    X, y = s["X_train"], s["y_train"].reshape(-1)
    point = _gbm_estimator(seed, smoke).fit(X, y)
    qs = [
        _gbm_estimator(seed, smoke, quantile=a).fit(X, y) for a in (alpha / 2, 0.5, 1 - alpha / 2)
    ]

    def _q(Xe: np.ndarray) -> np.ndarray:
        return np.stack([m.predict(Xe) for m in qs], axis=1)

    views = _from_quantiles(
        point.predict(s["X_cal"]),
        point.predict(s["X_test"]),
        _q(s["X_cal"]),
        _q(s["X_test"]),
        alpha=alpha,
    )
    return _BasePreds(
        **views,
        factory=lambda: _gbm_estimator(seed, smoke),
        library="lightgbm" if _LIGHTGBM_AVAILABLE else "sklearn",
    )


def _rf_estimator(seed: int, smoke: bool) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=50 if smoke else 200,
        min_samples_leaf=5,
        n_jobs=1,
        random_state=seed,
    )


def _fit_rf(s: dict[str, Any], *, alpha: float, seed: int, smoke: bool) -> _BasePreds:
    rf = _rf_estimator(seed, smoke).fit(s["X_train"], s["y_train"].reshape(-1))

    def _trees(Xe: np.ndarray) -> np.ndarray:
        return np.stack([t.predict(np.asarray(Xe, dtype=np.float32)) for t in rf.estimators_])

    t_cal, t_test = _trees(s["X_cal"]), _trees(s["X_test"])
    levels = (alpha / 2, 0.5, 1 - alpha / 2)
    q_cal = np.quantile(t_cal, levels, axis=0).T
    q_test = np.quantile(t_test, levels, axis=0).T
    std_cal = t_cal.std(axis=0)
    std_test = t_test.std(axis=0)
    spread_cal = _floor_spread(std_cal, std_cal)
    spread_test = _floor_spread(std_test, std_cal)
    return _BasePreds(
        point_cal=t_cal.mean(axis=0),
        point_test=t_test.mean(axis=0),
        q_cal=q_cal,
        q_test=q_test,
        mean_cal=t_cal.mean(axis=0),
        std_cal=spread_cal,
        mean_test=t_test.mean(axis=0),
        std_test=spread_test,
        spread_cal=spread_cal,
        spread_test=spread_test,
        factory=lambda: _rf_estimator(seed, smoke),
        library="sklearn",
    )


def _fit_tabicl(s: dict[str, Any], *, alpha: float, seed: int, smoke: bool) -> _BasePreds:
    del seed, smoke
    from tabicl import TabICLRegressor

    est = TabICLRegressor(device=str(current_device()))
    est.fit(s["X_train"], s["y_train"].reshape(-1))
    levels = [alpha / 2, 0.5, 1 - alpha / 2]

    def _q(Xe: np.ndarray) -> np.ndarray:
        q = np.asarray(est.predict(Xe, output_type="quantiles", alphas=levels))
        return q.reshape(len(Xe), -1)

    views = _from_quantiles(
        est.predict(s["X_cal"]),
        est.predict(s["X_test"]),
        _q(s["X_cal"]),
        _q(s["X_test"]),
        alpha=alpha,
    )
    return _BasePreds(**views, library="tabicl")


def _fit_deep_ensemble(s: dict[str, Any], *, alpha: float, config: HarnessConfig) -> _BasePreds:
    from torchregress.losses import BetaNLLLoss

    in_dim = s["X_train"].shape[1]
    hidden = _hidden_size(in_dim)
    X = torch.from_numpy(np.asarray(s["X_train"])).float().to(current_device())
    y = torch.from_numpy(np.asarray(s["y_train"]).reshape(-1, 1)).float().to(current_device())
    members = []
    for i in range(_N_ENSEMBLE):
        set_harness_seed(config.seed + i * 1000)
        m = _MLP(in_dim, hidden, 2)
        _train_one(m, BetaNLLLoss(beta=0.5), X, y, config)
        m.eval()
        members.append(m)

    def _mix(Xe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        stats = [_gaussian_nll_predict(m, np.asarray(Xe, dtype=np.float32)) for m in members]
        return _mixture_gaussian_stats([a for a, _ in stats], [b for _, b in stats])

    mu_c, sd_c = _mix(s["X_cal"])
    mu_t, sd_t = _mix(s["X_test"])
    z = _z(alpha)
    return _BasePreds(
        point_cal=mu_c,
        point_test=mu_t,
        q_cal=np.stack([mu_c - z * sd_c, mu_c, mu_c + z * sd_c], axis=1),
        q_test=np.stack([mu_t - z * sd_t, mu_t, mu_t + z * sd_t], axis=1),
        mean_cal=mu_c,
        std_cal=sd_c,
        mean_test=mu_t,
        std_test=sd_t,
        spread_cal=sd_c,
        spread_test=sd_t,
        library="torchregress",
    )


def _fit_tabular_mlp(s: dict[str, Any], *, alpha: float, config: HarnessConfig) -> _BasePreds:
    """5-member TabularMLP Gaussian ensemble (library defaults) as a Gaussian base."""
    fit = fit_tabmlp(
        np.asarray(s["X_train"]),
        np.asarray(s["y_train"]).reshape(-1),
        config,
        kind="gaussian",
        n_members=_N_ENSEMBLE,
    )
    mu_c, sd_c = predict_gaussian(fit, s["X_cal"])
    mu_t, sd_t = predict_gaussian(fit, s["X_test"])
    z = _z(alpha)
    return _BasePreds(
        point_cal=mu_c,
        point_test=mu_t,
        q_cal=np.stack([mu_c - z * sd_c, mu_c, mu_c + z * sd_c], axis=1),
        q_test=np.stack([mu_t - z * sd_t, mu_t, mu_t + z * sd_t], axis=1),
        mean_cal=mu_c,
        std_cal=sd_c,
        mean_test=mu_t,
        std_test=sd_t,
        spread_cal=sd_c,
        spread_test=sd_t,
        library="torchregress",
    )


# ---------------------------------------------------------------------------
# MAPIE prefit adapter: replays the cached base predictions
# ---------------------------------------------------------------------------


def _key(X: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    return f"{arr.shape}:{hashlib.sha1(arr.tobytes()).hexdigest()}"


class _CachedRegressor(RegressorMixin, BaseEstimator):
    """Prefit sklearn regressor returning cached base predictions.

    MAPIE's prefit path calls ``predict`` on the calibration and test
    features; replaying the cached arrays guarantees MAPIE sees exactly the
    numbers torchregress and crepes see (no second TabICL / ensemble pass).
    Any other input raises, so a silent refit cannot happen.
    """

    def __init__(self, table: dict[str, np.ndarray] | None = None) -> None:
        self.table = table

    fitted_ = True

    def fit(self, X: Any, y: Any = None) -> _CachedRegressor:  # pragma: no cover - guard
        raise RuntimeError("_CachedRegressor is prefit; fit() must not be called")

    def predict(self, X: Any) -> np.ndarray:
        table = self.table or {}
        k = _key(X)
        if k not in table:
            raise KeyError("features not seen by the cached base model")
        return table[k].copy()

    def __sklearn_is_fitted__(self) -> bool:
        return True


def _cached(s: dict[str, Any], cal: np.ndarray, test: np.ndarray) -> _CachedRegressor:
    return _CachedRegressor(
        {
            _key(s["X_cal"]): np.asarray(cal, dtype=np.float64),
            _key(s["X_test"]): np.asarray(test, dtype=np.float64),
        }
    )


# ---------------------------------------------------------------------------
# UQ methods
# ---------------------------------------------------------------------------


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


def _np(a: torch.Tensor) -> np.ndarray:
    return a.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _interval(lo: Any, hi: Any, point: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "y_pred": np.asarray(point, dtype=np.float64).reshape(-1),
        "lower": np.asarray(lo, dtype=np.float64).reshape(-1),
        "upper": np.asarray(hi, dtype=np.float64).reshape(-1),
    }


def _gaussian(mean: np.ndarray, std: np.ndarray, *, alpha: float) -> dict[str, np.ndarray]:
    z = _z(alpha)
    return {
        "mean": mean,
        "std": std,
        "lower": mean - z * std,
        "upper": mean + z * std,
    }


def _cv_folds(n: int, *, seed: int, loo: bool) -> tuple[Any, np.ndarray]:
    from sklearn.model_selection import KFold, LeaveOneOut

    cv: Any = LeaveOneOut() if loo else KFold(n_splits=_N_FOLDS, shuffle=True, random_state=seed)
    fold = np.empty(n, dtype=np.int64)
    for k, (_, te) in enumerate(cv.split(np.zeros((n, 1)))):
        fold[te] = k
    return cv, fold


def _apply(
    method: str,
    base: _BasePreds,
    s: dict[str, Any],
    *,
    alpha: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run one UQ method on one base model's cached predictions."""
    y_cal = np.asarray(s["y_cal"], dtype=np.float64).reshape(-1)
    conf = 1.0 - alpha

    if method == "Raw":
        out = _gaussian(base.mean_test, base.std_test, alpha=alpha)
        out["lower"], out["upper"] = base.q_test[:, 0], base.q_test[:, 2]
        return out

    # --- absolute-residual split conformal ---
    if method == "tr_Split":
        from torchregress.losses.conformal import SplitConformal

        cp = SplitConformal(alpha=alpha)
        cp.calibrate(_t(base.point_cal), _t(y_cal))
        lo, hi = cp.predict_interval(_t(base.point_test))
        return _interval(_np(lo), _np(hi), base.point_test)
    if method == "MAPIE_Split":
        from mapie.regression import SplitConformalRegressor

        scr = SplitConformalRegressor(
            estimator=_cached(s, base.point_cal, base.point_test),
            confidence_level=conf,
            conformity_score="absolute",
            prefit=True,
        )
        scr.conformalize(_as_float64(s["X_cal"]), y_cal)
        _, iv = scr.predict_interval(_as_float64(s["X_test"]))
        lo, hi = _mapie_intervals_to_bounds(iv)
        return _interval(lo, hi, base.point_test)
    if method == "crepes_Split":
        cr = ConformalRegressor().fit(y_cal - base.point_cal)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iv = cr.predict_int(base.point_test, confidence=conf)
        return _interval(iv[:, 0], iv[:, 1], base.point_test)

    # --- normalized (locally weighted) split conformal ---
    if method == "tr_Normalized":
        from torchregress.losses.conformal import SplitConformal

        cp = SplitConformal(alpha=alpha, normalize_fn=lambda _pred, spread: spread)
        cp.calibrate(_t(base.point_cal), _t(y_cal), x=_t(base.spread_cal))
        lo, hi = cp.predict_interval(_t(base.point_test), x=_t(base.spread_test))
        return _interval(_np(lo), _np(hi), base.point_test)
    if method == "crepes_Normalized":
        cr = ConformalRegressor().fit(y_cal - base.point_cal, sigmas=base.spread_cal)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iv = cr.predict_int(base.point_test, sigmas=base.spread_test, confidence=conf)
        return _interval(iv[:, 0], iv[:, 1], base.point_test)

    # --- CQR on the base quantiles ---
    if method == "tr_CQR":
        from torchregress.losses.conformal import CQR

        cqr = CQR(alpha=alpha)
        cqr.calibrate(_t(base.q_cal[:, [0, 2]]), _t(y_cal))
        lo, hi = cqr.predict_interval(_t(base.q_test[:, [0, 2]]))
        return _interval(_np(lo), _np(hi), base.point_test)
    if method in ("MAPIE_CQR", "MAPIE_CQR_asym"):
        from mapie.regression import ConformalizedQuantileRegressor

        ests = [_cached(s, base.q_cal[:, j], base.q_test[:, j]) for j in (0, 2, 1)]
        cqr_m = ConformalizedQuantileRegressor(estimator=ests, confidence_level=conf, prefit=True)
        cqr_m.conformalize(_as_float64(s["X_cal"]), y_cal)
        _, iv = cqr_m.predict_interval(
            _as_float64(s["X_test"]), symmetric_correction=method == "MAPIE_CQR"
        )
        lo, hi = _mapie_intervals_to_bounds(iv)
        return _interval(lo, hi, base.point_test)

    # --- CV+ / Jackknife+ (refit the base point model on train + cal) ---
    if method in _CV_METHODS + _JACKKNIFE_METHODS:
        if base.factory is None:
            raise _Skip("base model is not refittable")
        X_pool = np.vstack([_as_float64(s["X_train"]), _as_float64(s["X_cal"])])
        y_pool = np.concatenate([np.asarray(s["y_train"]).reshape(-1), y_cal]).astype(np.float64)
        loo = method in _JACKKNIFE_METHODS
        if loo and len(y_pool) > _JACKKNIFE_MAX_N:
            raise _Skip(f"Jackknife+ refits n={len(y_pool)} models (> {_JACKKNIFE_MAX_N})")
        cv, fold = _cv_folds(len(y_pool), seed=seed, loo=loo)
        X_test = _as_float64(s["X_test"])
        if method.startswith("MAPIE"):
            from mapie.regression import CrossConformalRegressor

            ccr = CrossConformalRegressor(
                estimator=base.factory(),
                confidence_level=conf,
                conformity_score="absolute",
                method="plus",
                cv=cv,
                random_state=seed,
            )
            ccr.fit_conformalize(X_pool, y_pool)
            # CV+ aggregates (n_test, n_pool) scores: predict in row blocks.
            pred, lo, hi = mapie_predict_blocked(ccr, X_test, n_ref=len(y_pool))
            return _interval(lo, hi, pred)
        from torchregress.losses.conformal import CVPlus, JackknifePlus

        n_folds = int(fold.max()) + 1
        oof = np.empty_like(y_pool)
        member_test = np.empty((n_folds, len(X_test)))
        for k in range(n_folds):
            tr = fold != k
            m = base.factory().fit(X_pool[tr], y_pool[tr])
            oof[~tr] = m.predict(X_pool[~tr])
            member_test[k] = m.predict(X_test)
        cp_cls = JackknifePlus if loo else CVPlus
        cvp = cp_cls(alpha=alpha)
        cvp.calibrate_ensemble(_t(oof[:, None]), _t(y_pool[:, None]), torch.as_tensor(fold))
        lo, hi = cvp.predict_interval(_t(member_test[:, :, None]))
        return _interval(_np(lo), _np(hi), member_test.mean(axis=0))

    # --- post-hoc Gaussian calibration ---
    if method in _CALIBRATION_METHODS:
        from torchregress.calibration import IsotonicMeanCalibrator, VarianceTemperatureScaler

        mean_cal, var_cal = base.mean_cal, base.std_cal**2
        mean_test = base.mean_test
        idx_vts = np.arange(len(y_cal))
        if method == "tr_IsoMean+VTS":
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(y_cal))
            idx_iso, idx_vts = perm[: len(perm) // 2], perm[len(perm) // 2 :]
            iso = IsotonicMeanCalibrator().fit(_t(mean_cal[idx_iso]), _t(y_cal[idx_iso]))
            mean_cal = _np(iso.transform(_t(mean_cal)))
            mean_test = _np(iso.transform(_t(mean_test)))
        vts = VarianceTemperatureScaler().fit(
            _t(mean_cal[idx_vts]), _t(var_cal[idx_vts]), _t(y_cal[idx_vts])
        )
        std_test = np.sqrt(_np(vts.transform(_t(base.std_test**2))))
        out = _gaussian(mean_test, std_test, alpha=alpha)
        out["_temperature"] = np.asarray([vts.temperature])
        return out

    raise ValueError(f"Unknown UQ method: {method}")


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class UQLayerSuite(HarnessRunner):
    """torchregress / MAPIE / crepes UQ layers on shared base-model predictions."""

    CALIBRATION_FRAC = 0.25
    tabred_samples: int = 10000

    @property
    def name(self) -> str:
        return "uq_layer"

    @property
    def description(self) -> str:
        return (
            "UQ layer on any base model (LightGBM, RF, TabICL, deep ensemble): "
            "torchregress conformal / calibration vs MAPIE / crepes on the same predictions"
        )

    @property
    def supported_datasets(self) -> list[str]:
        # The conformal suite's tabular set.
        return [
            "california_housing",
            "diabetes",
            "synthetic_hetero",
            "house_sales",
            "insurance_claims",
            "openml_year",
            "tabred_delivery",
            "tabred_cooking",
            "tabred_weather",
            "zurich_delays",
        ]

    def bases(self) -> list[str]:
        out = [_GBM_BASE, "RF"]
        if _TABICL_AVAILABLE:
            out.append("TabICL")
        out.append("DeepEnsK5")
        out.append("TabularMLP")
        return out

    def base_methods(self, base: str) -> list[str]:
        """UQ methods applied to ``base`` (external ones only when installed)."""
        keep: list[str] = []
        groups: list[tuple[str, ...]] = [
            _SPLIT_METHODS,
            _NORMALIZED_METHODS,
            _CQR_METHODS,
        ]
        if base in _REFIT_BASES:
            groups.append(_CV_METHODS)
        if base == _GBM_BASE:
            groups.append(_JACKKNIFE_METHODS)
        groups.append(_CALIBRATION_METHODS)
        for group in groups:
            for m in group:
                if m.startswith("MAPIE") and not _MAPIE_AVAILABLE:
                    continue
                if m.startswith("crepes") and not _CREPES_AVAILABLE:
                    continue
                keep.append(m)
        return keep

    def methods(self) -> list[str]:
        return [f"{m}+{b}" for b in self.bases() for m in self.base_methods(b)]

    @staticmethod
    def _split_method(method: str) -> tuple[str, str]:
        uq, _, base = method.rpartition("+")
        return uq, base

    def library_name(self, method: str) -> str:
        uq, base = self._split_method(method)
        if uq.startswith("MAPIE"):
            return "MAPIE"
        if uq.startswith("crepes"):
            return "crepes"
        if uq == "Raw":
            return {
                "RF": "sklearn",
                "TabICL": "tabicl",
                "DeepEnsK5": "torchregress",
                "TabularMLP": "torchregress",
            }.get(base, "lightgbm" if _LIGHTGBM_AVAILABLE else "sklearn")
        return "torchregress"

    def library_version(self, method: str) -> str:
        return _mapie_version() if method.startswith("MAPIE") else ""

    def method_notes(self, method: str) -> str:
        uq, base = self._split_method(method)
        base_note = {
            "LightGBM": "LGBMRegressor L2 point + quantile(alpha/2, 0.5, 1-alpha/2)",
            "HistGB": "HistGradientBoostingRegressor point + quantile models",
            "RF": "RandomForest mean; quantiles/std over per-tree predictions",
            "TabICL": "TabICLRegressor point + predictive quantiles",
            "DeepEnsK5": "5x Gaussian MLP (BetaNLL 0.5) mixture",
            "TabularMLP": "5x torchregress.models TabularMLP Gaussian head (library defaults) mixture",
        }.get(base, "")
        uq_note = {
            "Raw": "uncalibrated base interval; NLL/CRPS/ECE from the base's Gaussian view",
            "tr_Split": "torchregress SplitConformal (absolute residual)",
            "MAPIE_Split": "mapie SplitConformalRegressor(prefit, absolute) on the cached base",
            "crepes_Split": "crepes ConformalRegressor on the cached base residuals",
            "tr_Normalized": "torchregress SplitConformal(normalize_fn=spread) (residual/spread)",
            "crepes_Normalized": "crepes ConformalRegressor(sigmas=spread)",
            "tr_CQR": "torchregress CQR on the base quantiles",
            "MAPIE_CQR": "mapie ConformalizedQuantileRegressor(prefit), symmetric_correction=True",
            "MAPIE_CQR_asym": "mapie ConformalizedQuantileRegressor(prefit), default asymmetric correction",
            "tr_CVPlus": "torchregress CVPlus, K=5 refits on train+cal (same KFold as MAPIE)",
            "MAPIE_CVPlus": "mapie CrossConformalRegressor(method='plus', same KFold) on train+cal",
            "tr_JackknifePlus": "torchregress JackknifePlus, LOO refits on train+cal",
            "MAPIE_JackknifePlus": "mapie CrossConformalRegressor(plus, LeaveOneOut) on train+cal",
            "tr_VTS": "torchregress VarianceTemperatureScaler fitted on the calibration split",
            "tr_IsoMean+VTS": "IsotonicMeanCalibrator on cal half A, VarianceTemperatureScaler on half B",
        }.get(uq, "")
        return f"{uq_note} | base: {base_note}"

    # ------------------------------------------------------------------
    # Base fitting / HarnessRunner protocol
    # ------------------------------------------------------------------

    def fit_base(self, base: str, s: dict[str, Any], config: HarnessConfig) -> _BasePreds:
        alpha = float(s.get("_alpha", 0.1))
        set_harness_seed(config.seed)
        t0 = perf_counter()
        if base in ("LightGBM", "HistGB"):
            out = _fit_gbm(s, alpha=alpha, seed=config.seed, smoke=config.smoke)
        elif base == "RF":
            out = _fit_rf(s, alpha=alpha, seed=config.seed, smoke=config.smoke)
        elif base == "TabICL":
            out = _fit_tabicl(s, alpha=alpha, seed=config.seed, smoke=config.smoke)
        elif base == "DeepEnsK5":
            out = _fit_deep_ensemble(s, alpha=alpha, config=config)
        elif base == "TabularMLP":
            out = _fit_tabular_mlp(s, alpha=alpha, config=config)
        else:
            raise ValueError(f"Unknown base model: {base}")
        out.fit_s = perf_counter() - t0
        return out

    def fit(self, method: str, splits: dict[str, Any], config: HarnessConfig) -> dict[str, Any]:
        """Fit base + UQ method on a carved split (``X_cal``/``y_cal`` required)."""
        uq, base = self._split_method(method)
        bp = self.fit_base(base, splits, config)
        alpha = float(splits.get("_alpha", 0.1))
        preds = _apply(uq, bp, splits, alpha=alpha, seed=config.seed)
        return {"predictions": preds, "X_test_key": _key(splits["X_test"])}

    def predict(
        self, method: str, model: dict[str, Any], X_test: np.ndarray, config: HarnessConfig
    ) -> dict[str, np.ndarray]:
        if _key(X_test) != model["X_test_key"]:
            raise ValueError("uq_layer predictions are computed for the carved test split only")
        return dict(model["predictions"])

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        preds = {k: v for k, v in predictions.items() if not k.startswith("_") or k == "_y_scaler"}
        if "std" in preds:
            # Distribution rows: harness Gaussian NLL/CRPS/PIT-ECE, as in ``uq``.
            metrics = super().evaluate(method, preds, y_test, config)
        else:
            y_scaler = preds.pop("_y_scaler", None)
            y_true = np.asarray(y_test).reshape(-1)
            if y_scaler is not None:
                preds = _inverse_transform_predictions(preds, y_scaler)
                y_true = _inverse_transform_y(y_true, y_scaler)
            metrics = compute_point_metrics(np.asarray(preds["y_pred"]).reshape(-1), y_true)
            metrics.update(compute_interval_metrics(preds["lower"], preds["upper"], y_true))
            if method.startswith("MAPIE_"):
                cw = _mapie_coverage_and_width(preds["lower"], preds["upper"], y_true)
                if cw is not None:
                    metrics["MAPIE_Coverage"], metrics["MAPIE_Width"] = cw
        if "_temperature" in predictions:
            metrics["VTS_Temperature"] = float(np.asarray(predictions["_temperature"])[0])
        return metrics

    # ------------------------------------------------------------------
    # run(): one base fit, many UQ layers
    # ------------------------------------------------------------------

    def _row(
        self,
        method: str,
        dataset_name: str,
        splits: dict[str, Any],
        **fields: Any,
    ) -> dict[str, Any]:
        uq, base = self._split_method(method)
        return {
            "Method": f"{self.name}/{method}",
            "Library": self.library_name(method),
            "LibraryVersion": self.library_version(method),
            "Dataset": dataset_name,
            "Suite": self.name,
            "Base": base,
            "UQMethod": uq,
            **fields,
            "SourceURL": splits.get("_source_url", ""),
            "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
            "Device": str(current_device()),
        }

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, Any] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        cfg = config or HarnessConfig()
        dataset_name, splits = self._prepare_scenario_run(
            scenario_name, dataset_name, splits, standardize_targets=cfg.standardize_targets
        )
        if splits is None:
            from harness.datasets import load_dataset

            kw: dict[str, Any] = {}
            if dataset_name.startswith("tabred_"):
                kw["n_samples"] = self.tabred_samples
            splits = load_dataset(dataset_name, standardize_targets=cfg.standardize_targets, **kw)

        s = _carve_calibration_split(
            splits,
            cal_frac=float(splits.get("_calibration_fraction", self.CALIBRATION_FRAC)),
            seed=cfg.seed,
            smoke=cfg.smoke,
            suite_name=self.name,
            forward_keys=("_alpha", "_scenario", "_source_url"),
        )
        alpha = float(s.get("_alpha", 0.1))
        y_test = s["y_test"]
        rows: list[dict[str, Any]] = []
        failed_fields = {"Coverage": None, "Width": None, "IntervalScore": None}

        for base in self.bases():
            uq_methods = self.base_methods(base)
            try:
                bp = self.fit_base(base, s, cfg)
            except Exception as exc:  # noqa: BLE001 - e.g. TabICL checkpoint download refused
                note = f"failed: base {base} could not be fitted: {type(exc).__name__}: {exc}"
                for uq in uq_methods:
                    rows.append(
                        self._row(f"{uq}+{base}", dataset_name, s, **failed_fields, Notes=note)
                    )
                continue

            for uq in uq_methods:
                method = f"{uq}+{base}"
                set_harness_seed(cfg.seed)
                try:
                    t0 = perf_counter()
                    preds = _apply(uq, bp, s, alpha=alpha, seed=cfg.seed)
                    uq_s = perf_counter() - t0
                    if "_y_scaler" in s:
                        preds["_y_scaler"] = s["_y_scaler"]
                    metrics = self.evaluate(method, preds, y_test, cfg)
                    notes = "; ".join(
                        n for n in (self.method_notes(method), infinite_interval_note(metrics)) if n
                    )
                    rows.append(
                        self._row(
                            method,
                            dataset_name,
                            s,
                            **metrics,
                            train_s=bp.fit_s + uq_s,
                            eval_s=0.0,
                            BaseFit_s=bp.fit_s,
                            UQ_s=uq_s,
                            NCal=len(s["y_cal"]),
                            Notes=notes,
                        )
                    )
                except _Skip as exc:
                    rows.append(
                        self._row(method, dataset_name, s, **failed_fields, Notes=f"skipped: {exc}")
                    )
                except Exception as exc:  # noqa: BLE001 - one method must not drop the rest
                    rows.append(
                        self._row(
                            method,
                            dataset_name,
                            s,
                            **failed_fields,
                            Notes=f"failed: {type(exc).__name__}: {exc}",
                        )
                    )
        return rows
