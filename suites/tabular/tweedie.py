"""Tweedie / compound-Poisson regression suite: torchregress vs scikit-lego.

Ported from ``torchregress/examples/external_comparison_tweedie_vs_sklego.py``.

Compares torchregress's ``TweedieLoss`` and ``CompoundPoissonLoss`` against
scikit-lego's ``GLMRegressor(distribution="tweedie")`` on zero-inflated
compound Poisson-Gamma data.

Canonical scenarios
-------------------
* ``sklearn_tweedie_reference``
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_tweedie_deviance
from torchregress.losses import CompoundPoissonLoss, TweedieLoss

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _Linear, _train_lbfgs, _train_one

try:
    from sklego.linear_model import GLMRegressor

    _SKLEGO_AVAILABLE = True
except ImportError:
    _SKLEGO_AVAILABLE = False

try:
    import statsmodels.api as sm

    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Tweedie deviance helper — matches sklearn convention for cross-library
# apples-to-apples comparison
# ---------------------------------------------------------------------------


def _tweedie_deviance(y: np.ndarray, mu: np.ndarray, p: float) -> float:
    """Mean Tweedie unit deviance using the sklearn / lightgbm convention."""
    return float(mean_tweedie_deviance(y, mu, sample_weight=None, power=p))


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class TweedieSuite(HarnessRunner):
    """Tweedie / compound-Poisson regression benchmark suite."""

    @property
    def name(self) -> str:
        return "tweedie"

    @property
    def description(self) -> str:
        return "Tweedie / compound-Poisson: torchregress vs statsmodels GLM (ground truth) vs scikit-lego"

    @property
    def supported_datasets(self) -> list[str]:
        return ["synthetic_tweedie"]

    @property
    def canonical_scenarios(self) -> list[str]:
        """sklearn TweedieRegressor canonical example."""
        return ["sklearn_tweedie_reference"]

    def methods(self) -> list[str]:
        # Linear + LBFGS parity variants (match statsmodels GLM)
        linear_methods = ["Tweedie_Linear", "CompoundPoisson_Linear"]
        # MLP + Adam variants (original)
        mlp_methods = ["Tweedie", "CompoundPoisson"]
        methods = linear_methods + mlp_methods
        if _STATSMODELS_AVAILABLE:
            methods.append("statsmodels_GLM")
        if _SKLEGO_AVAILABLE:
            methods.append("sklego_GLMRegressor")
        return methods

    def library_name(self, method: str) -> str:
        if method.startswith("sklego"):
            return "scikit-lego"
        if method.startswith("statsmodels"):
            return "statsmodels"
        return "torchregress"

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"]

        if method in ("Tweedie", "CompoundPoisson", "Tweedie_Linear", "CompoundPoisson_Linear"):
            # Tweedie / compound Poisson require non-negative targets
            if y_train_np.min() < 0:
                raise ValueError(
                    f"{method} requires non-negative targets "
                    f"(min={y_train_np.min():.2f}). "
                    f"Use a Tweedie dataset (synthetic_tweedie, insurance_claims) "
                    f"or disable y-standardisation."
                )

            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            p_power = float(splits.get("p_power", 1.5))
            if "Tweedie" in method:
                loss_fn = TweedieLoss(p=p_power)
            else:
                loss_fn = CompoundPoissonLoss(p=p_power)

            if method.endswith("_Linear"):
                # Linear model + LBFGS — parity with statsmodels GLM
                in_dim = X.shape[1]
                model = _Linear(in_dim, out_dim=1)
                _train_lbfgs(model, loss_fn, X, y, max_iter=500, tolerance_grad=1e-7)
            else:
                # MLP + Adam — original torchregress approach
                in_dim = X.shape[1]
                hidden = _hidden_size(in_dim)
                model = _MLP(in_dim, hidden, out_dim=1)
                _train_one(model, loss_fn, X, y, config, transform_output=torch.exp)
                model.eval()

            return {"type": "torch", "model": model, "p_power": p_power}

        elif method == "sklego_GLMRegressor":
            p_power = float(splits.get("p_power", 1.5))
            glm = GLMRegressor(
                distribution="tweedie",
                power=p_power,
                alpha=1.0,
                fit_intercept=True,
                solver="lbfgs",
                max_iter=200,
            )
            glm.fit(X_train_np, y_train_np)

            return {"type": "sklego", "model": glm}

        elif method == "statsmodels_GLM":
            p_power = float(splits.get("p_power", 1.5))
            X_const = sm.add_constant(X_train_np, prepend=True)
            glm = sm.GLM(
                y_train_np,
                X_const,
                family=sm.families.Tweedie(
                    var_power=p_power,
                    link=sm.families.links.Log(),
                ),
            )
            result = glm.fit()
            return {"type": "statsmodels_glm", "model": result}

        raise ValueError(f"Unknown method: {method}")

    def predict(
        self,
        method: str,
        model: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        if model["type"] == "torch":
            m = model["model"]
            with torch.no_grad():
                X = torch.from_numpy(X_test).float()
                mu = torch.exp(m(X)).squeeze(-1).numpy()
            return {"y_pred": mu}

        elif model["type"] == "sklego":
            glm = model["model"]
            mu = np.asarray(glm.predict(X_test), dtype=np.float64)
            return {"y_pred": mu}

        elif model["type"] == "statsmodels_glm":
            result = model["model"]
            X_const = sm.add_constant(X_test, prepend=True)
            mu = np.asarray(result.predict(X_const), dtype=np.float64)
            return {"y_pred": mu}

        raise ValueError(f"Unknown method: {method}")

    def method_notes(self, method: str) -> str:
        if method == "Tweedie_Linear":
            return "linear model + TweedieLoss via LBFGS — parity with statsmodels GLM"
        elif method == "CompoundPoisson_Linear":
            return "linear model + CompoundPoissonLoss via LBFGS — parity with statsmodels GLM"
        elif method == "Tweedie":
            return "MLP + torchregress.TweedieLoss on log-mean"
        elif method == "CompoundPoisson":
            return "MLP + torchregress.CompoundPoissonLoss on log-mean"
        elif method == "statsmodels_GLM":
            return "GLM(family=Tweedie, link=log) via IRLS — gold-standard ground truth"
        elif method == "sklego_GLMRegressor":
            return "log-link GLM with tweedie deviance; capacity not matched"
        return ""

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        from harness.runners import _inverse_transform_predictions, _inverse_transform_y

        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_test = _inverse_transform_y(y_test, y_scaler)

        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)

        # Standard point metrics (always computable)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

        # Tweedie-specific metrics — only meaningful for non-negative targets
        p_power = float(predictions.get("p_power", 1.5))
        has_nonneg = bool(y_true.min() >= 0)

        if has_nonneg:
            try:
                deviance = _tweedie_deviance(y_true, y_pred, p_power)
            except ValueError:
                deviance = float("nan")
            zero_frac = float(np.mean(y_pred <= 1e-3))
        else:
            deviance = float("nan")
            zero_frac = float("nan")

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "TweedieDeviance": deviance,
            "ZeroFracPred": zero_frac,
        }
