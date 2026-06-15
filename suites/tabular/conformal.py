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
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import set_harness_seed
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
    from mapie.regression import MapieQuantileRegressor, MapieRegressor

    _MAPIE_AVAILABLE = True
except ImportError:
    _MAPIE_AVAILABLE = False

try:
    from torchcp.regression import SplitCP

    _TORCHCP_AVAILABLE = True
except ImportError:
    _TORCHCP_AVAILABLE = False

from suites._backbones import _MLP, _carve_calibration_split, _hidden_size, _Linear, _train_one

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
    q = torch.quantile(residuals, 1.0 - alpha).item()
    return (pred_test - q).numpy(), (pred_test + q).numpy()


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
    q = torch.quantile(score, 1.0 - alpha).item()
    return (q_lo_test - q).numpy(), (q_hi_test + q).numpy()


# ---------------------------------------------------------------------------
# Interval helpers — external libraries
# ---------------------------------------------------------------------------


def _mapie_split_intervals(
    splits: dict[str, np.ndarray], *, alpha: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LinearRegression

    mapie = MapieRegressor(
        estimator=LinearRegression(),
        method="base",
        cv="split",
        random_state=seed,
    )
    mapie.fit(splits["X_train"], splits["y_train"])
    _, y_pis = mapie.predict(splits["X_test"], alpha=alpha)
    return np.asarray(y_pis[:, 0, 0]), np.asarray(y_pis[:, 1, 0])


def _mapie_cqr_intervals(
    splits: dict[str, np.ndarray], *, alpha: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import GradientBoostingRegressor

    lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, random_state=seed)
    hi = GradientBoostingRegressor(loss="quantile", alpha=1.0 - alpha / 2, random_state=seed)
    mapie = MapieQuantileRegressor(
        [lo, hi],
        method="quantile",
        cv="split",
        alpha=alpha,
        random_state=seed,
    )
    mapie.fit(splits["X_train"], splits["y_train"])
    mapie.calibrate(splits["X_cal"], splits["y_cal"])
    _, y_pis = mapie.predict(splits["X_test"])
    return np.asarray(y_pis[:, 0, 0]), np.asarray(y_pis[:, 1, 0])


def _crepes_split_intervals(
    splits: dict[str, np.ndarray], *, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LinearRegression

    point = LinearRegression().fit(splits["X_train"], splits["y_train"])
    pred_cal = point.predict(splits["X_cal"])
    pred_test = point.predict(splits["X_test"])
    cr = ConformalRegressor()
    cr.fit(np.asarray(splits["y_cal"]).reshape(-1) - pred_cal.reshape(-1))
    intervals = cr.predict(pred_test.reshape(-1), significance=alpha)
    return np.asarray(intervals[:, 0]), np.asarray(intervals[:, 1])


def _crepes_cqr_intervals(
    splits: dict[str, np.ndarray], *, alpha: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import GradientBoostingRegressor

    lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, random_state=seed)
    hi = GradientBoostingRegressor(loss="quantile", alpha=1.0 - alpha / 2, random_state=seed)
    lo.fit(splits["X_train"], splits["y_train"])
    hi.fit(splits["X_train"], splits["y_train"])
    q_lo_cal = lo.predict(splits["X_cal"])
    q_hi_cal = hi.predict(splits["X_cal"])
    q_lo_test = lo.predict(splits["X_test"])
    q_hi_test = hi.predict(splits["X_test"])
    y_cal = np.asarray(splits["y_cal"]).reshape(-1)
    score = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    cr = ConformalRegressor()
    cr.fit(score)
    _, intervals = cr.predict(np.zeros(1), significance=alpha)
    q = float(intervals[0, 1])
    return q_lo_test - q, q_hi_test + q


def _torchcp_split_intervals(
    splits: dict[str, np.ndarray], *, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LinearRegression

    point = LinearRegression().fit(splits["X_train"], splits["y_train"])
    pred_cal = point.predict(splits["X_cal"])
    cp = SplitCP(point)
    if hasattr(cp, "calibrate"):
        cp.calibrate(splits["X_cal"], splits["y_cal"])
    elif hasattr(cp, "fit"):
        cp.fit(residuals=np.asarray(splits["y_cal"]).reshape(-1) - pred_cal.reshape(-1))

    if hasattr(cp, "predict"):
        out = cp.predict(splits["X_test"], alpha=alpha)
    elif hasattr(cp, "predict_intervals"):
        out = cp.predict_intervals(splits["X_test"], alpha=alpha)

    if isinstance(out, (tuple, list)) and len(out) == 2:
        lo, hi = np.asarray(out[0]).reshape(-1), np.asarray(out[1]).reshape(-1)
    elif hasattr(out, "ndim") and out.ndim == 2:
        arr = np.asarray(out)
        if arr.shape[-1] == 2 and arr.shape[0] != 2:
            lo, hi = arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)
        else:
            lo, hi = arr[0].reshape(-1), arr[1].reshape(-1)
    else:
        raise RuntimeError(f"torchcp SplitCP unexpected output shape: {getattr(out, 'shape', '?')}")
    return lo, hi


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class ConformalSuite(HarnessRunner):
    """Conformal prediction benchmark suite.

    Overrides ``run()`` to carve a calibration split from training data
    before calling the fit/predict/evaluate protocol.
    """

    CALIBRATION_FRAC = 0.25  # portion of training data used for calibration

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
        ]
        if _MAPIE_AVAILABLE:
            methods.extend(["MAPIE_Split+Linear", "MAPIE_CQR+GBR"])
        if _CREPES_AVAILABLE:
            methods.extend(["crepes_Split+Linear", "crepes_CQR+GBR"])
        if _TORCHCP_AVAILABLE:
            methods.append("torchcp_Split+Linear")
        return methods

    def library_name(self, method: str) -> str:
        if method.startswith("MAPIE"):
            return "MAPIE"
        if method.startswith("crepes"):
            return "crepes"
        if method.startswith("torchcp"):
            return "torchcp"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        notes_map: dict[str, str] = {
            "Split+MLP": "point MLP + torchregress.ConformalLoss(split)",
            "CQR+MLP": "quantile MLP + torchregress.ConformalLoss(cqr)",
            "Split+Linear": "single linear layer + ConformalLoss(split); matched-capacity baseline",
            "MAPIE_Split+Linear": "sklearn LinearRegression + mapie.MapieRegressor(method='base')",
            "MAPIE_CQR+GBR": "sklearn GradientBoostingRegressor + mapie.MapieQuantileRegressor",
            "crepes_Split+Linear": "sklearn LinearRegression + crepes.ConformalRegressor",
            "crepes_CQR+GBR": "sklearn GradientBoostingRegressor + crepes ConformalRegressor",
            "torchcp_Split+Linear": "sklearn LinearRegression + torchcp.regression.SplitCP",
        }
        return notes_map.get(method, "")

    # ------------------------------------------------------------------
    # fit / predict — torchregress methods
    # ------------------------------------------------------------------

    def fit(
        self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit model. ``splits`` must contain ``X_train``, ``y_train``, ``X_cal``, ``y_cal``.

        The calibration data is carried through to ``predict()`` via the
        returned model state dict.
        """
        set_harness_seed(config.seed)
        alpha = float(splits.get("_alpha", 0.1))

        X_train = torch.from_numpy(splits["X_train"]).float()
        y_train = torch.from_numpy(splits["y_train"]).float().unsqueeze(1)
        X_cal = torch.from_numpy(splits["X_cal"]).float()
        y_cal = torch.from_numpy(splits["y_cal"]).float()
        in_dim = X_train.shape[1]
        hidden = _hidden_size(in_dim)

        # --- torchregress: Split + MLP point head ---
        if method == "Split+MLP" or method == "Split+Linear":
            from torchregress.losses.conformal import ConformalLoss

            out_dim = 1
            if method == "Split+MLP":
                model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
            else:
                model = _Linear(in_dim, out_dim)

            loss_fn = ConformalLoss(method="split", alpha=alpha)
            _train_one(model, loss_fn, X_train, y_train, config)

            return {
                "type": "split",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "alpha": alpha,
            }

        # --- torchregress: CQR + MLP quantile head ---
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

        # --- External: store the splits dict so predict() can use native APIs ---
        elif method.startswith("MAPIE_") or method.startswith("crepes_") or method.startswith("torchcp_"):
            return {
                "type": "external",
                "method": method,
                "splits": splits,
                "alpha": alpha,
                "seed": config.seed,
            }

        raise ValueError(f"Unknown method: {method}")

    def predict(
        self, method: str, model_state: dict[str, Any], X_test: np.ndarray, config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        alpha = model_state.get("alpha", 0.1)
        model_type = model_state["type"]

        # --- torchregress: Split ---
        if model_type == "split":
            model = model_state["model"]
            X_cal = model_state["X_cal"]
            y_cal = model_state["y_cal"]
            X_test_t = torch.from_numpy(X_test).float()
            lo, hi = _torch_split_intervals(model, X_cal, y_cal, X_test_t, alpha=alpha)
            return {"lower": lo, "upper": hi}

        # --- torchregress: CQR ---
        elif model_type == "cqr":
            model = model_state["model"]
            X_cal = model_state["X_cal"]
            y_cal = model_state["y_cal"]
            X_test_t = torch.from_numpy(X_test).float()
            lo, hi = _torch_cqr_intervals(model, X_cal, y_cal, X_test_t, alpha=alpha)
            return {"lower": lo, "upper": hi}

        # --- External ---
        elif model_type == "external":
            external_method = model_state["method"]
            splits = model_state["splits"]
            seed = model_state["seed"]

            if external_method == "MAPIE_Split+Linear":
                lo, hi = _mapie_split_intervals(splits, alpha=alpha, seed=seed)
            elif external_method == "MAPIE_CQR+GBR":
                lo, hi = _mapie_cqr_intervals(splits, alpha=alpha, seed=seed)
            elif external_method == "crepes_Split+Linear":
                lo, hi = _crepes_split_intervals(splits, alpha=alpha)
            elif external_method == "crepes_CQR+GBR":
                lo, hi = _crepes_cqr_intervals(splits, alpha=alpha, seed=seed)
            elif external_method == "torchcp_Split+Linear":
                lo, hi = _torchcp_split_intervals(splits, alpha=alpha)
            else:
                raise ValueError(f"Unknown external method: {external_method}")

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
            try:
                from mapie.metrics import (
                    regression_coverage_score,
                    regression_mean_width_score,
                )

                metrics["MAPIE_Coverage"] = float(
                    regression_coverage_score(y_true, lo, hi)
                )
                metrics["MAPIE_Width"] = float(
                    regression_mean_width_score(lo, hi)
                )
            except ImportError:
                pass  # MAPIE not installed — harness metrics suffice

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
            scenario_name, dataset_name, splits,
        )

        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name)

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
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    **metrics,
                    "train_s": fit_time,
                    "eval_s": pred_time,
                    "SourceURL": splits.get("_source_url", ""),
                    "Notes": self.method_notes(method),
                }

            except ImportError as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "Coverage": None,
                    "Width": None,
                    "IntervalScore": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "Notes": f"skipped: {type(exc).__name__}: {exc}",
                }

            except Exception as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "Coverage": None,
                    "Width": None,
                    "IntervalScore": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }

            rows.append(row)

        return rows
