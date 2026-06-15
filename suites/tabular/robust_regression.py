"""Robust regression suite: torchregress robust losses vs LightGBM vs XGBoost.

Compares a broad family of robust loss functions (Huber, Cauchy, Barron,
AdaptiveRobust, LogCosh, Charbonnier, Tukey, PseudoHuber) against
gradient-boosting baselines on data with outliers.

Canonical scenarios
-------------------
* ``sklearn_synthetic_reference``
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

Ported and generalized from:
- ``torchregress/examples/comprehensive_comparison.py`` (robustness scenario)
- ``torchregress/examples/loss_comparison.py`` (loss comparison)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Torchregress robust losses — all hard dependencies (from torchregress)
from torchregress.losses import (
    AdaptiveRobustLoss,
    BarronLoss,
    CauchyLoss,
    CharbonnierLoss,
    LogCoshLoss,
    PseudoHuberLoss,
    TukeyBiweightLoss,
    WeightedHuberLoss,
    WeightedL1Loss,
    WeightedMSELoss,
)

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _train_one

# ---------------------------------------------------------------------------
# Optional external gradient-boosting libraries
# ---------------------------------------------------------------------------

try:
    import lightgbm as lgb

    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Loss registry: maps method name → (loss factory, notes)
# ---------------------------------------------------------------------------

_TORCHREGRESS_LOSSES: dict[str, tuple] = {
    "MSE": (lambda: WeightedMSELoss(reduction="mean"), "outlier-sensitive baseline"),
    "MAE": (lambda: WeightedL1Loss(reduction="mean"), "L1 absolute error baseline"),
    "Huber": (lambda: WeightedHuberLoss(delta=1.0, reduction="mean"), "balanced robustness default"),
    "Cauchy": (lambda: CauchyLoss(c=0.5, reduction="mean"), "very robust, non-convex"),
    "Barron": (lambda: BarronLoss(alpha=1.0, scale=1.0, reduction="mean"), "general Barron family α=1"),
    "AdaptiveRobust": (
        lambda: AdaptiveRobustLoss(alpha_init=1.0, scale_init=1.0, reduction="mean"),
        "learns α dynamically",
    ),
    "LogCosh": (lambda: LogCoshLoss(scale=1.0, reduction="mean"), "smooth log(cosh) approximation"),
    "Charbonnier": (lambda: CharbonnierLoss(eps=1e-3, reduction="mean"), "smooth L1 alternative"),
    "Tukey": (lambda: TukeyBiweightLoss(c=4.685, reduction="mean"), "redescending, ignores far outliers"),
    "PseudoHuber": (lambda: PseudoHuberLoss(delta=1.0, reduction="mean"), "smooth Huber approximation"),
}


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class RobustRegressionSuite(HarnessRunner):
    """Robust regression benchmark suite.

    Compares torchregress robust losses against LightGBM/XGBoost on
    real tabular and synthetic outlier datasets.
    """

    @property
    def name(self) -> str:
        return "robust_regression"

    @property
    def description(self) -> str:
        return "Robust regression losses vs LightGBM / XGBoost on outlier data"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "synthetic_outliers",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """sklearn synthetic regression benchmark."""
        return ["sklearn_synthetic_reference"]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        methods = list(_TORCHREGRESS_LOSSES.keys())
        if _LIGHTGBM_AVAILABLE:
            methods.append("LightGBM")
        if _XGBOOST_AVAILABLE:
            methods.append("XGBoost")
        return methods

    def library_name(self, method: str) -> str:
        if method == "LightGBM":
            return "LightGBM"
        if method == "XGBoost":
            return "XGBoost"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        if method in _TORCHREGRESS_LOSSES:
            return _TORCHREGRESS_LOSSES[method][1]
        if method == "LightGBM":
            return "gradient boosting (objective='regression_l1' for Huber-like)"
        if method == "XGBoost":
            return "gradient boosting (objective='reg:pseudohubererror')"
        return ""

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(
        self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"].reshape(-1)

        # --- Torchregress methods ---
        if method in _TORCHREGRESS_LOSSES:
            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            in_dim = X.shape[1]
            hidden = _hidden_size(in_dim)
            model = _MLP(in_dim, hidden, out_dim=1)

            loss_factory = _TORCHREGRESS_LOSSES[method][0]
            loss_fn = loss_factory()
            _train_one(model, loss_fn, X, y, config, loss_params=loss_fn.parameters())
            model.eval()

            return {"type": "torch", "model": model}

        # --- LightGBM ---
        elif method == "LightGBM":
            train_data = lgb.Dataset(X_train_np, label=y_train_np)
            params = {
                "objective": "regression_l1",
                "metric": "l1",
                "verbosity": -1,
                "seed": config.seed,
                "num_leaves": 31,
                "learning_rate": 0.05,
            }
            booster = lgb.train(params, train_data, num_boost_round=100)
            return {"type": "lgbm", "model": booster}

        # --- XGBoost ---
        elif method == "XGBoost":
            params = {
                "objective": "reg:pseudohubererror",
                "eval_metric": "mae",
                "random_state": config.seed,
                "verbosity": 0,
                "max_depth": 6,
                "learning_rate": 0.05,
            }
            booster = xgb.train(params, xgb.DMatrix(X_train_np, y_train_np), num_boost_round=100)
            return {"type": "xgb", "model": booster}

        raise ValueError(f"Unknown method: {method}")

    def predict(
        self, method: str, model_state: dict[str, Any], X_test: np.ndarray, config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        model_type = model_state["type"]

        if model_type == "torch":
            m = model_state["model"]
            with torch.no_grad():
                X = torch.from_numpy(X_test).float()
                y_pred = m(X).squeeze(-1).numpy()
            return {"y_pred": y_pred}

        elif model_type == "lgbm":
            booster = model_state["model"]
            y_pred = booster.predict(X_test, num_iteration=booster.best_iteration)
            return {"y_pred": np.asarray(y_pred).reshape(-1)}

        elif model_type == "xgb":
            booster = model_state["model"]
            y_pred = booster.predict(xgb.DMatrix(X_test))
            return {"y_pred": np.asarray(y_pred).reshape(-1)}

        raise ValueError(f"Unknown model type: {model_type}")
