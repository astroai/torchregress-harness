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
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Torchregress robust losses — all hard dependencies (from torchregress)
from torchregress.losses import (
    AdaptiveRobustLoss,
    BarronLoss,
    CauchyLoss,
    CharbonnierLoss,
    LogCoshLoss,
    MultiQuantileLoss,
    PseudoHuberLoss,
    TukeyBiweightLoss,
    WeightedHuberLoss,
    WeightedL1Loss,
    WeightedMSELoss,
)

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _Linear, _train_lbfgs, _train_one

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

try:
    import statsmodels.api as sm

    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Loss registry: maps method name → (loss factory, notes)
# ---------------------------------------------------------------------------

_TORCHREGRESS_LOSSES: dict[str, tuple] = {
    "MSE": (lambda: WeightedMSELoss(reduction="mean"), "outlier-sensitive baseline"),
    "MAE": (lambda: WeightedL1Loss(reduction="mean"), "L1 absolute error baseline"),
    "Huber": (
        lambda: WeightedHuberLoss(delta=1.0, reduction="mean"),
        "balanced robustness default",
    ),
    "Cauchy": (lambda: CauchyLoss(c=0.5, reduction="mean"), "very robust, non-convex"),
    "Barron": (
        lambda: BarronLoss(alpha=1.0, scale=1.0, reduction="mean"),
        "general Barron family α=1",
    ),
    "AdaptiveRobust": (
        lambda: AdaptiveRobustLoss(alpha_init=1.0, scale_init=1.0, reduction="mean"),
        "learns α dynamically",
    ),
    "LogCosh": (lambda: LogCoshLoss(scale=1.0, reduction="mean"), "smooth log(cosh) approximation"),
    "Charbonnier": (lambda: CharbonnierLoss(eps=1e-3, reduction="mean"), "smooth L1 alternative"),
    "Tukey": (
        lambda: TukeyBiweightLoss(c=4.685, reduction="mean"),
        "redescending, ignores far outliers",
    ),
    "PseudoHuber": (
        lambda: PseudoHuberLoss(delta=1.0, reduction="mean"),
        "smooth Huber approximation",
    ),
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
        return "Robust regression: Huber_Linear parity (LBFGS) vs statsmodels RLM vs MLP losses"

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
        # Linear + LBFGS parity variants for every loss (matches statsmodels RLM)
        linear_methods = [f"{name}_Linear" for name in _TORCHREGRESS_LOSSES]
        # MLP + Adam variants (original)
        mlp_methods = list(_TORCHREGRESS_LOSSES.keys())
        methods = (
            linear_methods
            + mlp_methods
            + [
                "Quantile_Linear",  # torchregress: _Linear + MultiQuantileLoss via LBFGS
                "sklearn_QuantileRegressor",  # sklearn: QuantileRegressor via LP
            ]
        )
        if _STATSMODELS_AVAILABLE:
            methods.append("statsmodels_RLM")
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
        if method == "statsmodels_RLM":
            return "statsmodels"
        if method == "sklearn_QuantileRegressor":
            return "sklearn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        if method == "statsmodels_RLM":
            return "IRLS + HuberT(delta=1.345) — gold-standard ground truth"
        if method == "Quantile_Linear":
            return "linear model + MultiQuantileLoss(q=0.5) via LBFGS — parity with sklearn QuantileRegressor"
        if method == "sklearn_QuantileRegressor":
            return "linear median regression via linear programming (solver=highs)"
        if method.endswith("_Linear"):
            base = method[:-7]  # strip "_Linear" suffix
            if base in _TORCHREGRESS_LOSSES:
                return f"linear model + {base}Loss via LBFGS — parity reference"
            return "linear model via LBFGS"
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
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"].reshape(-1)

        # --- sklearn QuantileRegressor (LP-based median regression) ---
        if method == "sklearn_QuantileRegressor":
            qr = QuantileRegressor(quantile=0.5, alpha=0.0, fit_intercept=True, solver="highs")
            qr.fit(X_train_np, y_train_np)
            return {"type": "sklearn_qr", "model": qr}

        # --- Torchregress: Quantile_Linear (LBFGS parity with sklearn QuantileRegressor) ---
        elif method == "Quantile_Linear":
            X = torch.from_numpy(X_train_np).float().to(current_device())
            y = torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1)

            in_dim = X.shape[1]
            model = _Linear(in_dim, out_dim=1)  # single Linear with bias
            loss_fn = MultiQuantileLoss(quantiles=[0.5])
            _train_lbfgs(model, loss_fn, X, y, max_iter=500, tolerance_grad=1e-7)

            return {"type": "torch", "model": model}

        # --- Torchregress: _Linear variants (LBFGS, parity with statsmodels RLM) ---
        elif method.endswith("_Linear"):
            base = method[:-7]  # strip "_Linear" suffix
            if base not in _TORCHREGRESS_LOSSES:
                raise ValueError(f"Unknown _Linear variant: {method}")

            X = torch.from_numpy(X_train_np).float().to(current_device())
            y = torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1)

            in_dim = X.shape[1]
            model = _Linear(in_dim, out_dim=1)  # single Linear with bias
            loss_factory = _TORCHREGRESS_LOSSES[base][0]
            loss_fn = loss_factory()
            _train_lbfgs(model, loss_fn, X, y, max_iter=500, tolerance_grad=1e-7)

            return {"type": "torch", "model": model}

        # --- Torchregress MLP losses ---
        elif method in _TORCHREGRESS_LOSSES:
            X = torch.from_numpy(X_train_np).float().to(current_device())
            y = torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1)

            in_dim = X.shape[1]
            hidden = _hidden_size(in_dim)
            model = _MLP(in_dim, hidden, out_dim=1)

            loss_factory = _TORCHREGRESS_LOSSES[method][0]
            loss_fn = loss_factory()
            _train_one(model, loss_fn, X, y, config, loss_params=loss_fn.parameters())
            model.eval()

            return {"type": "torch", "model": model}

        # --- statsmodels RLM (ground truth) ---
        elif method == "statsmodels_RLM":
            # Add intercept column (statsmodels convention)
            X_const = sm.add_constant(X_train_np, prepend=True)
            rlm = sm.RLM(
                y_train_np,
                X_const,
                M=sm.robust.norms.HuberT(t=1.345),
            )
            result = rlm.fit(maxiter=100)
            return {"type": "statsmodels_rlm", "model": result}

        # --- LightGBM ---
        elif method == "LightGBM":
            # WRAP-03: carve an 80/20 train/valid split so early stopping
            # has a held-out signal; ``num_iteration=booster.best_iteration``
            # at predict time is now meaningful.
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_np, y_train_np, test_size=0.2, random_state=config.seed
            )
            train_data = lgb.Dataset(X_tr, label=y_tr)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            params = {
                "objective": "regression_l1",
                "metric": "l1",
                "verbosity": -1,
                "seed": config.seed,
                "num_leaves": 31,
                "learning_rate": 0.05,
            }
            booster = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            return {"type": "lgbm", "model": booster}

        # --- XGBoost ---
        elif method == "XGBoost":
            # WRAP-03 sibling: same 80/20 split + early stopping contract.
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_np, y_train_np, test_size=0.2, random_state=config.seed
            )
            params = {
                "objective": "reg:pseudohubererror",
                "eval_metric": "mae",
                "random_state": config.seed,
                "verbosity": 0,
                "max_depth": 6,
                "learning_rate": 0.05,
            }
            booster = xgb.train(
                params,
                xgb.DMatrix(X_tr, y_tr),
                num_boost_round=100,
                eval_set=[(xgb.DMatrix(X_val, y_val), "valid")],
                early_stopping_rounds=20,
            )
            return {"type": "xgb", "model": booster}

        raise ValueError(f"Unknown method: {method}")

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        model_type = model_state["type"]

        if model_type == "torch":
            m = model_state["model"]
            with torch.no_grad():
                X = torch.from_numpy(X_test).float().to(current_device())
                y_pred = m(X).squeeze(-1).detach().cpu().numpy()
            return {"y_pred": y_pred}

        elif model_type == "sklearn_qr":
            qr = model_state["model"]
            y_pred = qr.predict(X_test).reshape(-1)
            return {"y_pred": y_pred}

        elif model_type == "lgbm":
            booster = model_state["model"]
            y_pred = booster.predict(X_test, num_iteration=booster.best_iteration)
            return {"y_pred": np.asarray(y_pred).reshape(-1)}

        elif model_type == "xgb":
            booster = model_state["model"]
            y_pred = booster.predict(xgb.DMatrix(X_test))
            return {"y_pred": np.asarray(y_pred).reshape(-1)}

        elif model_type == "statsmodels_rlm":
            result = model_state["model"]
            X_const = sm.add_constant(X_test, prepend=True)
            y_pred = result.predict(X_const)
            return {"y_pred": np.asarray(y_pred).reshape(-1)}

        raise ValueError(f"Unknown model type: {model_type}")
