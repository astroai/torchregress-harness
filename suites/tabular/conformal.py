"""Conformal prediction suite: torchregress vs MAPIE vs crepes vs torchcp.

Ported from ``torchregress/examples/external_comparison_conformal_vs_mapie.py``.

Compares split-conformal and CQR-based prediction intervals across four
conformal-prediction libraries on a shared train/calibration/test split.

Canonical scenarios
-------------------
* ``mapie_reference``
  https://mapie.readthedocs.io/en/latest/examples_regression/1-tutorial.html
* ``crepes_reference``
  https://crepes.readthedocs.io/en/latest/

Architecture
------------
Conformal prediction requires a *calibration split* in addition to train/test.
The HarnessRunner protocol expects only train/test from the dataset registry,
so this suite overrides ``run()`` to carve a calibration set from the training
data and passes it through the model state dict so that ``predict()`` can
access it for interval construction.

MAPIE metrics
-------------
MAPIE methods are evaluated using MAPIE's own ``regression_coverage_score``
and ``regression_mean_width_score`` (from ``mapie.metrics``) alongside the
harness's standard interval metrics.  This ensures we measure the external
library on its own canonical terms.

MAPIE integration
-----------------
``MAPIE_*`` rows call the **real** ``mapie.regression`` API
(``SplitConformalRegressor`` / ``ConformalizedQuantileRegressor`` with
``fit -> conformalize -> predict_interval``, and git-main
``CrossConformalizedQuantileRegressor`` with its one-shot
``fit_conformalize``) — no manual reimplementation.
The installed version (and git commit for pip git installs) is recorded in
each row's ``LibraryVersion`` field; see ``EXTERNAL_PINS.md`` for the pinned
upstream commit used for paper comparisons.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchregress.losses.conformal import finite_sample_quantile

from harness.core import pip_git_commit, set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner

# ---------------------------------------------------------------------------
# Optional external conformal libraries
# ---------------------------------------------------------------------------

try:
    from crepes import ConformalRegressor

    _CREPES_AVAILABLE = True
except ImportError:
    _CREPES_AVAILABLE = False

try:
    import mapie  # noqa: F401

    _MAPIE_AVAILABLE = True
except ImportError:
    _MAPIE_AVAILABLE = False

# ``CrossConformalizedQuantileRegressor`` is a git-main addition (post-1.4.1,
# included in the EXTERNAL_PINS.md pin); gate the row on its availability so
# the suite still runs against older MAPIE builds.
try:
    from mapie.regression import CrossConformalizedQuantileRegressor  # noqa: F401

    _MAPIE_CROSS_CQR_AVAILABLE = True
except (ImportError, AttributeError):
    _MAPIE_CROSS_CQR_AVAILABLE = False

try:
    from torchcp.regression.predictor import SplitPredictor as _SplitPredictor

    _TORCHCP_AVAILABLE = True
except ImportError:
    _TORCHCP_AVAILABLE = False

try:
    import tabicl  # noqa: F401

    _TABICL_AVAILABLE = True
except ImportError:
    _TABICL_AVAILABLE = False

from suites._backbones import (
    _MLP,
    _carve_calibration_split,
    _hidden_size,
    _Linear,
    _train_lbfgs,
    _train_one,
)

# ---------------------------------------------------------------------------
# Interval helpers — torchregress
# ---------------------------------------------------------------------------


def _torch_split_intervals(
    model: nn.Module,
    X_cal: torch.Tensor,
    y_cal: torch.Tensor,
    X_test: torch.Tensor,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred_cal = model(X_cal).squeeze(-1)
        pred_test = model(X_test).squeeze(-1)
    residuals = (y_cal - pred_cal).abs()
    q = finite_sample_quantile(residuals, alpha).item()
    return (pred_test - q).cpu().numpy(), (pred_test + q).cpu().numpy()


def _torch_cqr_intervals(
    model: nn.Module,
    X_cal: torch.Tensor,
    y_cal: torch.Tensor,
    X_test: torch.Tensor,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred_cal = model(X_cal)
        pred_test = model(X_test)
    q_lo_cal, q_hi_cal = pred_cal[:, 0], pred_cal[:, 1]
    q_lo_test, q_hi_test = pred_test[:, 0], pred_test[:, 1]
    score = torch.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    q = finite_sample_quantile(score, alpha).item()
    return (q_lo_test - q).cpu().numpy(), (q_hi_test + q).cpu().numpy()


# ---------------------------------------------------------------------------
# Interval helpers — external libraries
# ---------------------------------------------------------------------------


def _as_float64(arr: np.ndarray) -> np.ndarray:
    """Upcast a harness split array to float64 for MAPIE.

    MAPIE's conformity-score consistency check uses an ``eps`` of 1e-8 and
    trips on float32 rounding (~1.5e-8) when the base regressor fits the
    calibration split nearly exactly.  sklearn estimators convert to float64
    internally anyway, so upcasting at the MAPIE boundary is lossless.
    """
    return np.asarray(arr, dtype=np.float64)


def _mapie_intervals_to_bounds(intervals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise MAPIE ``predict_interval`` output to (lower, upper) 1-D arrays.

    MAPIE v1 returns intervals of shape ``(n_samples, 2, n_confidence_levels)``;
    with a single float ``confidence_level`` the last axis has length 1.
    """
    arr = np.asarray(intervals, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(f"Unexpected MAPIE interval shape: {np.asarray(intervals).shape}")
    return arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)


def _mapie_coverage_and_width(
    lo: np.ndarray,
    hi: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float] | None:
    """MAPIE's canonical coverage/width for whichever ``mapie.metrics`` API is installed.

    MAPIE git main moved the regression metrics into ``mapie.metrics.regression``
    and changed both signatures to take the ``(n_samples, 2[, n_levels])``
    interval array (``regression_coverage_score(y_true, intervals)`` /
    ``regression_mean_width_score(intervals)``).  Older 1.4.x builds expose the
    same names at the top-level ``mapie.metrics`` with ``(y_true, lo, hi)`` /
    ``(lo, hi)`` signatures.  Returns ``None`` when neither is importable so the
    harness's standard interval metrics still stand alone.
    """
    intervals = np.stack([np.asarray(lo), np.asarray(hi)], axis=1)[:, :, None]
    try:  # git-main API (pinned in EXTERNAL_PINS.md)
        from mapie.metrics.regression import (
            regression_coverage_score,
            regression_mean_width_score,
        )

        cov = float(np.asarray(regression_coverage_score(y_true, intervals)).reshape(-1)[0])
        w = float(np.asarray(regression_mean_width_score(intervals)).reshape(-1)[0])
        return cov, w
    except (ImportError, TypeError):
        pass
    try:  # legacy v1.4.x API
        from mapie.metrics import (
            regression_coverage_score,
            regression_mean_width_score,
        )

        cov = float(regression_coverage_score(y_true, lo, hi))
        w = float(regression_mean_width_score(lo, hi))
        return cov, w
    except (ImportError, TypeError):
        return None


def _mapie_version() -> str:
    """Installed MAPIE version, annotated with the git commit when pip-installed from git.

    e.g. ``"1.4.1"`` for a PyPI wheel or ``"1.5.0 (git 6478f16c9a)"`` for a
    ``pip install git+https://github.com/scikit-learn-contrib/MAPIE.git`` install.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        ver = version("mapie")
    except PackageNotFoundError:
        return "unknown"
    commit = pip_git_commit("mapie")
    return f"{ver} (git {commit})" if commit else ver


def _require_mapie_version() -> None:
    """Raise a clear ``ImportError`` when MAPIE is missing or predates the v1 API (< 1.4.1)."""
    import re

    try:
        import mapie
    except ImportError as exc:
        raise ImportError(
            "mapie not installed — install with: "
            "pip install 'git+https://github.com/scikit-learn-contrib/MAPIE.git@6478f16c9a' "
            "(see EXTERNAL_PINS.md)"
        ) from exc
    raw = getattr(mapie, "__version__", None)
    if raw is None:
        return
    m = re.match(r"\s*v?(\d+)\.(\d+)", raw)
    if m is None:
        raise ImportError(
            f"cannot parse mapie version {raw!r} — cannot verify the >= 1.4.1 "
            "v1-API requirement.  Install from git main per EXTERNAL_PINS.md."
        )
    if (int(m.group(1)), int(m.group(2))) < (1, 4):
        raise ImportError(
            f"mapie {raw} predates the v1 API (needs >= 1.4.1 for "
            "SplitConformalRegressor / ConformalizedQuantileRegressor). "
            "Install from git main per EXTERNAL_PINS.md."
        )


def _mapie_fit(
    splits: dict[str, np.ndarray],
    estimator: Any,
    *,
    alpha: float,
) -> Any:
    """Fit-half: ``fit -> conformalize`` a MAPIE v1 ``SplitConformalRegressor``.

    Returns the fitted/conformalized predictor; interval construction happens
    in :func:`_mapie_predict` so the runner's fit/eval timing is truthful.
    """
    from mapie.regression import SplitConformalRegressor

    scr = SplitConformalRegressor(
        estimator=estimator,
        confidence_level=1.0 - alpha,
        conformity_score="absolute",
        prefit=False,
    )
    scr.fit(_as_float64(splits["X_train"]), _as_float64(splits["y_train"]).reshape(-1))
    scr.conformalize(_as_float64(splits["X_cal"]), _as_float64(splits["y_cal"]).reshape(-1))
    return scr


def _mapie_predict(pred_obj: Any, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict-half: interval prediction from a fitted MAPIE v1 predictor."""
    _, intervals = pred_obj.predict_interval(_as_float64(X_test))
    return _mapie_intervals_to_bounds(intervals)


def _mapie_split_linear_fit(
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
) -> Any:
    """Fit-half for ``MAPIE_Split+Linear`` (sklearn LinearRegression base)."""
    from sklearn.linear_model import LinearRegression

    return _mapie_fit(splits, LinearRegression(), alpha=alpha)


def _mapie_split_mlp_fit(splits: dict[str, np.ndarray], *, alpha: float, seed: int) -> Any:
    """Fit-half for ``MAPIE_Split+MLP`` (sklearn MLPRegressor base).

    Uses the same 3-layer ReLU architecture heuristic as torchregress's
    ``_MLP`` (hidden = max(16, min(in_dim*4, 64))).
    """
    from sklearn.neural_network import MLPRegressor

    in_dim = splits["X_train"].shape[1]
    hidden = max(16, min(in_dim * 4, 64))

    est = MLPRegressor(
        hidden_layer_sizes=(hidden, hidden),
        activation="relu",
        solver="adam",
        alpha=0.0,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=60,
        tol=1e-10,
        random_state=seed,
        early_stopping=False,
    )
    return _mapie_fit(splits, est, alpha=alpha)


def _mapie_cqr_fit(splits: dict[str, np.ndarray], *, alpha: float, seed: int) -> Any:
    """Fit-half for ``MAPIE_CQR+GBR``: fit + conformalize the real CQR."""
    from mapie.regression import ConformalizedQuantileRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    est = GradientBoostingRegressor(loss="quantile", random_state=seed)
    cqr = ConformalizedQuantileRegressor(
        estimator=est,
        confidence_level=1.0 - alpha,
        prefit=False,
    )
    cqr.fit(_as_float64(splits["X_train"]), _as_float64(splits["y_train"]).reshape(-1))
    cqr.conformalize(_as_float64(splits["X_cal"]), _as_float64(splits["y_cal"]).reshape(-1))
    return cqr


def _mapie_cross_cqr_fit(splits: dict[str, np.ndarray], *, alpha: float, seed: int) -> Any:
    """Fit-half for ``MAPIE_CrossCQR+GBR`` (git-main cross-conformal CQR).

    Cross-conformal needs no separate calibration split: one-shot
    ``fit_conformalize`` on the FULL training data via K-fold out-of-fold
    scores (Romano et al., 2019).
    """
    from mapie.regression import CrossConformalizedQuantileRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    est = GradientBoostingRegressor(loss="quantile", random_state=seed)
    ccqr = CrossConformalizedQuantileRegressor(
        estimator=est,
        confidence_level=1.0 - alpha,
        cv=5,
        random_state=seed,
        n_jobs=-1,
    )
    X_all = np.vstack([_as_float64(splits["X_train"]), _as_float64(splits["X_cal"])])
    y_all = np.concatenate(
        [_as_float64(splits["y_train"]).reshape(-1), _as_float64(splits["y_cal"]).reshape(-1)]
    )
    ccqr.fit_conformalize(X_all, y_all)
    return ccqr


def _crepes_split_fit(splits: dict[str, np.ndarray], *, alpha: float) -> dict[str, Any]:
    """Fit-half for ``crepes_Split+Linear``: point model + crepes calibration."""
    del alpha
    from sklearn.linear_model import LinearRegression

    point = LinearRegression().fit(splits["X_train"], splits["y_train"])
    pred_cal = point.predict(splits["X_cal"])
    cr = ConformalRegressor()
    cr.fit(np.asarray(splits["y_cal"]).reshape(-1) - pred_cal.reshape(-1))
    return {"point": point, "cr": cr}


def _crepes_split_predict(
    state: dict[str, Any],
    X_test: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict-half for ``crepes_Split+Linear``."""
    pred_test = state["point"].predict(X_test).reshape(-1)
    intervals = state["cr"].predict_int(pred_test, confidence=1.0 - alpha)
    return np.asarray(intervals[:, 0]), np.asarray(intervals[:, 1])


def _crepes_cqr_fit(splits: dict[str, np.ndarray], *, alpha: float, seed: int) -> dict[str, Any]:
    """Fit-half for ``crepes_CQR+GBR``: quantile GBRs + crepes score calibration."""
    from sklearn.ensemble import GradientBoostingRegressor

    lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, random_state=seed)
    hi = GradientBoostingRegressor(loss="quantile", alpha=1.0 - alpha / 2, random_state=seed)
    lo.fit(splits["X_train"], splits["y_train"])
    hi.fit(splits["X_train"], splits["y_train"])
    q_lo_cal = lo.predict(splits["X_cal"])
    q_hi_cal = hi.predict(splits["X_cal"])
    y_cal = np.asarray(splits["y_cal"]).reshape(-1)
    score = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    cr = ConformalRegressor()
    cr.fit(score)
    return {"lo": lo, "hi": hi, "cr": cr}


def _crepes_cqr_predict(
    state: dict[str, Any],
    X_test: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict-half for ``crepes_CQR+GBR``."""
    q_lo_test = state["lo"].predict(X_test)
    q_hi_test = state["hi"].predict(X_test)
    intervals = np.asarray(state["cr"].predict_int(np.zeros(1), confidence=1.0 - alpha))
    q = float(intervals[0, 1])
    return q_lo_test - q, q_hi_test + q


def _torchcp_split_fit(splits: dict[str, np.ndarray], *, alpha: float) -> Any:
    """Fit-half for ``torchcp_Split+Linear``: adapter + calibration.

    WRAP-07: no fake device-marker Parameter — the ``SplitPredictor`` is
    constructed with an explicit ``device`` so it never probes model
    parameters to locate the device.
    """
    from sklearn.linear_model import LinearRegression
    from torch.utils.data import DataLoader, TensorDataset
    from torchcp.regression.score import ABS as _ABS

    # --- fit sklearn point estimator (apples-to-apples with MAPIE/crepes) ---
    point = LinearRegression().fit(splits["X_train"], splits["y_train"])

    # --- wrap sklearn estimator in torch.nn.Module so SplitPredictor accepts it ---
    class _SklearnAdapter(torch.nn.Module):
        def __init__(self, est: Any) -> None:
            super().__init__()
            self.est = est

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            arr = x.detach().cpu().numpy()
            pred = np.asarray(self.est.predict(arr)).reshape(-1, 1)
            return torch.from_numpy(pred).float().to(current_device())

    model = _SklearnAdapter(point)
    model.eval()

    # --- calibrate via torch DataLoader; alpha lives on constructor (1.x API) ---
    X_cal_t = torch.from_numpy(np.asarray(splits["X_cal"])).float().to(current_device())
    y_cal_t = torch.from_numpy(np.asarray(splits["y_cal"]).reshape(-1)).float().to(current_device())
    cal_loader = DataLoader(
        TensorDataset(X_cal_t, y_cal_t),
        batch_size=min(len(X_cal_t), 1024),
    )
    # ABS = plain absolute-residual score `|y - f(x)|` — approximately
    # equivalent to MAPIE's ``SplitConformalRegressor(conformity_score='absolute')``
    # and to crepes' ``ConformalRegressor().fit(y_cal - pred_cal)``.
    cp = _SplitPredictor(_ABS(), model=model, alpha=alpha, device=str(current_device()))
    cp.calibrate(cal_loader)
    return cp


def _torchcp_split_predict(cp: Any, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict-half for ``torchcp_Split+Linear``; returns a (N, 1, 2)-style tensor."""
    X_test_t = torch.from_numpy(np.asarray(X_test)).float().to(current_device())
    out = cp.predict(X_test_t)
    arr = out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)
    if arr.ndim == 3 and arr.shape[-2:] == (1, 2):
        arr = arr.squeeze(axis=1)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)
    raise RuntimeError(f"torchcp SplitPredictor unexpected output shape: {arr.shape}")


def _tabicl_split_fit(splits: dict[str, np.ndarray], *, alpha: float) -> dict[str, Any]:
    """Fit-half for TabICL split rows: backbone fit + conformal calibration."""
    from tabicl import TabICLRegressor

    est = TabICLRegressor(device=str(current_device()))
    est.fit(splits["X_train"], splits["y_train"])

    pred_cal = est.predict(splits["X_cal"]).reshape(-1)
    residuals = np.abs(np.asarray(splits["y_cal"]).reshape(-1) - pred_cal)
    q = finite_sample_quantile(torch.as_tensor(residuals), alpha).item()
    return {"est": est, "q": q}


def _tabicl_split_predict(
    state: dict[str, Any],
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict-half for TabICL split rows."""
    pred_test = state["est"].predict(X_test).reshape(-1)
    return pred_test - state["q"], pred_test + state["q"]


def _crepes_tabicl_fit(splits: dict[str, np.ndarray], *, alpha: float) -> dict[str, Any]:
    """Fit-half for ``crepes_Split+TabICL``: TabICL backbone + crepes calibrator."""
    del alpha
    from tabicl import TabICLRegressor

    est = TabICLRegressor(device=str(current_device()))
    est.fit(splits["X_train"], splits["y_train"])

    pred_cal = est.predict(splits["X_cal"]).reshape(-1)
    cr = ConformalRegressor()
    cr.fit(np.asarray(splits["y_cal"]).reshape(-1) - pred_cal.reshape(-1))
    return {"est": est, "cr": cr}


def _crepes_tabicl_predict(
    state: dict[str, Any],
    X_test: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict-half for ``crepes_Split+TabICL``."""
    pred_test = state["est"].predict(X_test).reshape(-1)
    intervals = state["cr"].predict_int(pred_test, confidence=1.0 - alpha)
    return np.asarray(intervals[:, 0]), np.asarray(intervals[:, 1])


# ---------------------------------------------------------------------------
# External fit/predict dispatch
# ---------------------------------------------------------------------------


def _external_fit(
    method: str,
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
    seed: int,
) -> Any:
    """Fit-half dispatch for external conformal builders.

    All expensive work (estimator fitting, conformalization/calibration)
    happens here and the returned fitted object is carried through the suite
    state under the ``"predictor"`` key, so runner ``train_s`` vs ``eval_s``
    timings are truthful (HN-WRAP-01 companion refactor).
    """
    if method == "MAPIE_Split+Linear":
        return _mapie_split_linear_fit(splits, alpha=alpha)
    if method == "MAPIE_Split+MLP":
        return _mapie_split_mlp_fit(splits, alpha=alpha, seed=seed)
    if method == "MAPIE_CQR+GBR":
        return _mapie_cqr_fit(splits, alpha=alpha, seed=seed)
    if method == "MAPIE_CrossCQR+GBR":
        return _mapie_cross_cqr_fit(splits, alpha=alpha, seed=seed)
    if method == "crepes_Split+Linear":
        return _crepes_split_fit(splits, alpha=alpha)
    if method == "crepes_CQR+GBR":
        return _crepes_cqr_fit(splits, alpha=alpha, seed=seed)
    if method == "torchcp_Split+Linear":
        return _torchcp_split_fit(splits, alpha=alpha)
    if method in ("Split+TabICL", "MAPIE_Split+TabICL"):
        return _tabicl_split_fit(splits, alpha=alpha)
    if method == "crepes_Split+TabICL":
        return _crepes_tabicl_fit(splits, alpha=alpha)
    raise ValueError(f"Unknown external method: {method}")


def _external_predict(
    method: str,
    predictor: Any,
    X_test: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict-half dispatch for external conformal builders (inference only)."""
    if method.startswith("MAPIE") and not method.endswith("TabICL"):
        return _mapie_predict(predictor, X_test)
    if method == "crepes_Split+Linear":
        return _crepes_split_predict(predictor, X_test, alpha=alpha)
    if method == "crepes_CQR+GBR":
        return _crepes_cqr_predict(predictor, X_test, alpha=alpha)
    if method == "torchcp_Split+Linear":
        return _torchcp_split_predict(predictor, X_test)
    if method in ("Split+TabICL", "MAPIE_Split+TabICL"):
        return _tabicl_split_predict(predictor, X_test)
    if method == "crepes_Split+TabICL":
        return _crepes_tabicl_predict(predictor, X_test, alpha=alpha)
    raise ValueError(f"Unknown external method: {method}")


# ---------------------------------------------------------------------------
# One-shot compositions (backward-compatible helpers)
# ---------------------------------------------------------------------------


def _mapie_split_intervals(
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """MAPIE split-conformal + LinearRegression, one-shot (fit half + predict half)."""
    del seed
    return _mapie_predict(_mapie_split_linear_fit(splits, alpha=alpha), splits["X_test"])


def _mapie_split_mlp_intervals(
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """MAPIE split-conformal + MLPRegressor, one-shot (fit half + predict half)."""
    pred_obj = _mapie_split_mlp_fit(splits, alpha=alpha, seed=seed)
    return _mapie_predict(pred_obj, splits["X_test"])


def _mapie_cqr_intervals(
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """MAPIE CQR + GradientBoostingRegressor, one-shot (fit half + predict half)."""
    pred_obj = _mapie_cqr_fit(splits, alpha=alpha, seed=seed)
    return _mapie_predict(pred_obj, splits["X_test"])


def _mapie_cross_cqr_intervals(
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """MAPIE cross-conformal CQR, one-shot (fit half + predict half)."""
    pred_obj = _mapie_cross_cqr_fit(splits, alpha=alpha, seed=seed)
    return _mapie_predict(pred_obj, splits["X_test"])


def _crepes_split_intervals(
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """crepes split-conformal + LinearRegression, one-shot (fit half + predict half)."""
    return _crepes_split_predict(
        _crepes_split_fit(splits, alpha=alpha), splits["X_test"], alpha=alpha
    )


def _crepes_cqr_intervals(
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """crepes CQR + GradientBoostingRegressor, one-shot (fit half + predict half)."""
    state = _crepes_cqr_fit(splits, alpha=alpha, seed=seed)
    return _crepes_cqr_predict(state, splits["X_test"], alpha=alpha)


def _torchcp_split_intervals(
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """torchcp split-conformal + LinearRegression, one-shot (fit half + predict half)."""
    return _torchcp_split_predict(_torchcp_split_fit(splits, alpha=alpha), splits["X_test"])


def _tabicl_split_intervals(
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """TabICL + absolute-residual split conformal, one-shot (fit half + predict half)."""
    del seed
    return _tabicl_split_predict(_tabicl_split_fit(splits, alpha=alpha), splits["X_test"])


def _crepes_split_tabicl_intervals(
    splits: dict[str, np.ndarray],
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """TabICL + crepes, one-shot (fit half + predict half)."""
    state = _crepes_tabicl_fit(splits, alpha=alpha)
    return _crepes_tabicl_predict(state, splits["X_test"], alpha=alpha)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class ConformalSuite(HarnessRunner):
    """Conformal prediction benchmark suite.

    Overrides ``run()`` to carve a calibration split from training data
    before calling the fit/predict/evaluate protocol.
    """

    CALIBRATION_FRAC = 0.25  # portion of training data used for calibration

    # Override for TabReD sample sizes (set by leaderboard)
    tabred_samples: int = 10000

    @property
    def name(self) -> str:
        return "conformal"

    @property
    def description(self) -> str:
        return "Conformal prediction intervals: torchregress vs MAPIE / crepes / torchcp"

    @property
    def supported_datasets(self) -> list[str]:
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
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """MAPIE and crepes canonical conformal prediction examples."""
        return ["mapie_reference", "crepes_reference"]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        methods = [
            "Split+MLP",
            "CQR+MLP",
            "Split+Linear",
            "Split+Linear(post-hoc)",  # post-hoc cal → parity with MAPIE/crepes/torchcp
        ]
        if _TABICL_AVAILABLE:
            methods.append("Split+TabICL")
        if _MAPIE_AVAILABLE:
            methods.extend(["MAPIE_Split+Linear", "MAPIE_Split+MLP", "MAPIE_CQR+GBR"])
            if _MAPIE_CROSS_CQR_AVAILABLE:
                methods.append("MAPIE_CrossCQR+GBR")
            if _TABICL_AVAILABLE:
                methods.append("MAPIE_Split+TabICL")
        if _CREPES_AVAILABLE:
            methods.extend(["crepes_Split+Linear", "crepes_CQR+GBR"])
            if _TABICL_AVAILABLE:
                methods.append("crepes_Split+TabICL")
        if _TORCHCP_AVAILABLE:
            methods.append("torchcp_Split+Linear")
        return methods

    def library_name(self, method: str) -> str:
        if "TabICL" in method:
            return "tabicl"
        if method.startswith("MAPIE"):
            return "MAPIE"
        if method.startswith("crepes"):
            return "crepes"
        if method.startswith("torchcp"):
            return "torchcp"
        return "torchregress"

    def library_version(self, method: str) -> str:
        """Version of the external library backing a method row.

        MAPIE rows self-report the installed version (plus git commit when
        pip-installed from git) so artifacts record the exact upstream code
        they were compared against — see ``EXTERNAL_PINS.md``.
        """
        if method.startswith("MAPIE"):
            return _mapie_version()
        return ""

    def method_notes(self, method: str) -> str:
        notes_map: dict[str, str] = {
            "Split+MLP": "point MLP + MSE → post-hoc residual calibration (ConformalLoss(split) ≡ MSE by design)",
            "CQR+MLP": "quantile MLP + MultiQuantileLoss + post-hoc CQR calibration (ConformalLoss(cqr) not used — it would be 2× loss scaling, same pinball formula)",
            "Split+Linear": "single linear layer + ConformalLoss(split); matched-capacity baseline",
            "Split+Linear(post-hoc)": "single linear layer + MSE → post-hoc residual calibration; parity with MAPIE/crepes/torchcp",
            "MAPIE_Split+Linear": "real mapie.regression.SplitConformalRegressor (absolute-residual score) on sklearn LinearRegression — MAPIE canonical",
            "MAPIE_Split+MLP": "real mapie.regression.SplitConformalRegressor (absolute-residual score) on sklearn MLPRegressor (matched 3-layer ReLU)",
            "MAPIE_CQR+GBR": "real mapie.regression.ConformalizedQuantileRegressor on sklearn GradientBoostingRegressor(loss='quantile') — MAPIE canonical CQR",
            "MAPIE_CrossCQR+GBR": "real mapie.regression.CrossConformalizedQuantileRegressor (K-fold cross-conformal CQR, Romano et al. 2019) on sklearn GradientBoostingRegressor(loss='quantile') — no calibration split: CV out-of-fold scores over the full train+cal data; requires MAPIE git main",
            "crepes_Split+Linear": "sklearn LinearRegression + crepes.ConformalRegressor",
            "crepes_CQR+GBR": "sklearn GradientBoostingRegressor + crepes ConformalRegressor",
            "torchcp_Split+Linear": (
                "sklearn LinearRegression + torchcp.regression.predictor.SplitPredictor "
                "(wrapped in torch nn.Module adapter for 1.x API)"
            ),
            "Split+TabICL": "TabICL in-context learner + absolute-residual split conformal",
            "MAPIE_Split+TabICL": "TabICL point predictions + absolute-residual split conformal",
            "crepes_Split+TabICL": "TabICL point predictions + crepes.ConformalRegressor",
        }
        return notes_map.get(method, "")

    # ------------------------------------------------------------------
    # fit / predict — torchregress methods
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit model. ``splits`` must contain ``X_train``, ``y_train``, ``X_cal``, ``y_cal``.

        The calibration data is carried through to ``predict()`` via the
        returned model state dict.
        """
        set_harness_seed(config.seed)
        alpha = float(splits.get("_alpha", 0.1))

        dev = current_device()
        X_train = torch.from_numpy(splits["X_train"]).float().to(dev)
        y_train = torch.from_numpy(splits["y_train"]).float().unsqueeze(1).to(dev)
        X_cal = torch.from_numpy(splits["X_cal"]).float().to(dev)
        y_cal = torch.from_numpy(splits["y_cal"]).float().to(dev)
        in_dim = X_train.shape[1]
        hidden = _hidden_size(in_dim)

        # --- torchregress: Split + MLP + MSE → post-hoc calibration ---
        if method == "Split+MLP":
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=1)
            loss_fn = nn.MSELoss()
            _train_one(model, loss_fn, X_train, y_train, config)
            model.eval()

            return {
                "type": "split_posthoc",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "alpha": alpha,
            }

        # --- torchregress: Split + Linear + ConformalLoss(split) ---
        elif method == "Split+Linear":
            from torchregress.losses.conformal import ConformalLoss

            model = _Linear(in_dim, out_dim=1)
            loss_fn = ConformalLoss(method="split", alpha=alpha)
            _train_one(model, loss_fn, X_train, y_train, config)

            return {
                "type": "split",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "alpha": alpha,
            }

        # --- torchregress: Split + Linear + MSE → post-hoc calibration (LBFGS for OLS parity) ---
        elif method == "Split+Linear(post-hoc)":
            model = _Linear(in_dim, out_dim=1)
            loss_fn = nn.MSELoss()
            _train_lbfgs(model, loss_fn, X_train, y_train, max_iter=500, tolerance_grad=1e-7)

            return {
                "type": "split_posthoc",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "alpha": alpha,
            }

        # --- torchregress: CQR + MLP quantile head (MultiQuantileLoss + post-hoc CQR cal) ---
        elif method == "CQR+MLP":
            from torchregress.losses.quantile import MultiQuantileLoss

            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=2)
            loss_fn = MultiQuantileLoss(quantiles=[alpha / 2, 1.0 - alpha / 2])

            _train_one(model, loss_fn, X_train, y_train, config)

            return {
                "type": "cqr",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "alpha": alpha,
            }

        # --- External: run the fit-half here and carry the fitted predictor ---
        # (conformalization/calibration) through the state dict, so predict()
        # performs only inference and runner train_s/eval_s timings are truthful.
        elif (
            method.startswith("MAPIE_")
            or method.startswith("crepes_")
            or method.startswith("torchcp_")
            or "TabICL" in method
        ):
            predictor = _external_fit(method, splits, alpha=alpha, seed=config.seed)
            return {
                "type": "external",
                "method": method,
                "predictor": predictor,
                "alpha": alpha,
            }

        raise ValueError(f"Unknown method: {method}")

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        alpha = model_state.get("alpha", 0.1)
        model_type = model_state["type"]

        # --- torchregress: Split ---
        if model_type == "split":
            model = model_state["model"]
            X_cal = model_state["X_cal"]
            y_cal = model_state["y_cal"]
            X_test_t = torch.from_numpy(X_test).float().to(current_device())
            lo, hi = _torch_split_intervals(model, X_cal, y_cal, X_test_t, alpha=alpha)
            return {"lower": lo, "upper": hi}

        # --- torchregress: Split + Linear(post-hoc) → residual calibration ---
        elif model_type == "split_posthoc":
            model = model_state["model"]
            X_cal = model_state["X_cal"]
            y_cal = model_state["y_cal"]
            X_test_t = torch.from_numpy(X_test).float().to(current_device())
            lo, hi = _torch_split_intervals(model, X_cal, y_cal, X_test_t, alpha=alpha)
            return {"lower": lo, "upper": hi}

        # --- torchregress: CQR ---
        elif model_type == "cqr":
            model = model_state["model"]
            X_cal = model_state["X_cal"]
            y_cal = model_state["y_cal"]
            X_test_t = torch.from_numpy(X_test).float().to(current_device())
            lo, hi = _torch_cqr_intervals(model, X_cal, y_cal, X_test_t, alpha=alpha)
            return {"lower": lo, "upper": hi}

        # --- External: predict-half only; all fitting happened in fit() ---
        elif model_type == "external":
            external_method = model_state["method"]
            predictor = model_state["predictor"]

            lo, hi = _external_predict(external_method, predictor, X_test, alpha=alpha)

            return {"lower": lo, "upper": hi}

        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # evaluate — use MAPIE's own metrics for MAPIE methods
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        """Compute interval metrics, dispatching to MAPIE's own functions."""
        from harness.runners import _inverse_transform_predictions, _inverse_transform_y

        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_test = _inverse_transform_y(y_test, y_scaler)

        lo = np.asarray(predictions["lower"]).reshape(-1)
        hi = np.asarray(predictions["upper"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)

        # Always compute harness standard interval metrics
        from harness.metrics import compute_interval_metrics

        metrics = compute_interval_metrics(lo, hi, y_true)

        # For MAPIE methods, also compute MAPIE's canonical metrics
        if method.startswith("MAPIE_"):
            cw = _mapie_coverage_and_width(lo, hi, y_true)
            if cw is not None:
                metrics["MAPIE_Coverage"], metrics["MAPIE_Width"] = cw

        return metrics

    # ------------------------------------------------------------------
    # Override run() to create calibration split
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the conformal suite, carving a cal split from training data."""
        from time import perf_counter

        cfg = config or HarnessConfig()

        # --- Resolve canonical scenario (reusable helper) ---
        dataset_name, splits = self._prepare_scenario_run(
            scenario_name,
            dataset_name,
            splits,
            standardize_targets=cfg.standardize_targets,
        )

        if splits is None:
            from harness.datasets import load_dataset

            loader_kwargs: dict[str, Any] = {}
            if dataset_name.startswith("tabred_"):
                loader_kwargs["n_samples"] = self.tabred_samples
            splits = load_dataset(
                dataset_name, standardize_targets=cfg.standardize_targets, **loader_kwargs
            )

        # Read scenario-driven parameters (with suite defaults as fallback)
        cal_frac = float(splits.get("_calibration_fraction", self.CALIBRATION_FRAC))

        # Carve calibration split from training data
        augmented_splits = _carve_calibration_split(
            splits,
            cal_frac=cal_frac,
            seed=cfg.seed,
            smoke=cfg.smoke,
            suite_name="conformal",
            forward_keys=("_alpha", "_scenario", "_source_url"),
        )

        X_test = augmented_splits["X_test"]
        y_test = augmented_splits["y_test"]

        rows: list[dict[str, Any]] = []

        for method in self.methods():
            set_harness_seed(cfg.seed)

            try:
                t0 = perf_counter()
                model_state = self.fit(method, augmented_splits, cfg)
                fit_time = perf_counter() - t0

                t0 = perf_counter()
                predictions = self.predict(method, model_state, X_test, cfg)
                pred_time = perf_counter() - t0

                if "_y_scaler" in augmented_splits:
                    predictions["_y_scaler"] = augmented_splits["_y_scaler"]

                metrics = self.evaluate(method, predictions, y_test, cfg)

                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    **metrics,
                    "train_s": fit_time,
                    "eval_s": pred_time,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": self.method_notes(method),
                }

            except ImportError as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "Coverage": None,
                    "Width": None,
                    "IntervalScore": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": f"skipped: {type(exc).__name__}: {exc}",
                }

            except (
                ValueError,
                RuntimeError,
                ImportError,
                KeyError,
                TypeError,
                OSError,
                NameError,
            ) as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "Coverage": None,
                    "Width": None,
                    "IntervalScore": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }

            rows.append(row)

        return rows
