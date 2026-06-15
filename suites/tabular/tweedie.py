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
from suites._backbones import _MLP, _hidden_size, _train_one

try:
    from sklego.linear_model import GLMRegressor

    _SKLEGO_AVAILABLE = True
except ImportError:
    _SKLEGO_AVAILABLE = False

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
        return "Tweedie / compound-Poisson regression: torchregress vs scikit-lego"

    @property
    def supported_datasets(self) -> list[str]:
        return ["synthetic_tweedie"]

    @property
    def canonical_scenarios(self) -> list[str]:
        """sklearn TweedieRegressor canonical example."""
        return ["sklearn_tweedie_reference"]

    def methods(self) -> list[str]:
        methods = ["Tweedie", "CompoundPoisson"]
        if _SKLEGO_AVAILABLE:
            methods.append("sklego_GLMRegressor")
        return methods

    def library_name(self, method: str) -> str:
        if method.startswith("sklego"):
            return "scikit-lego"
        return "torchregress"

    def fit(
        self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"]

        if method in ("Tweedie", "CompoundPoisson"):
            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            in_dim = X.shape[1]
            hidden = _hidden_size(in_dim)
            model = _MLP(in_dim, hidden, out_dim=1)

            p_power = float(splits.get("p_power", 1.5))
            if method == "Tweedie":
                loss_fn = TweedieLoss(p=p_power)
            else:
                loss_fn = CompoundPoissonLoss(p=p_power)

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

        raise ValueError(f"Unknown method: {method}")

    def predict(
        self, method: str, model: dict[str, Any], X_test: np.ndarray, config: HarnessConfig,
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

        raise ValueError(f"Unknown method: {method}")

    def method_notes(self, method: str) -> str:
        if method == "Tweedie":
            return "MLP + torchregress.TweedieLoss on log-mean"
        elif method == "CompoundPoisson":
            return "MLP + torchregress.CompoundPoissonLoss on log-mean"
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

        # Tweedie datasets skip y-standardisation, but handle gracefully
        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_test = _inverse_transform_y(y_test, y_scaler)

        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)

        # Read p_power from model state if available; default to 1.5
        p_power = float(predictions.get("p_power", 1.5))

        mae = float(np.mean(np.abs(y_pred - y_true)))
        try:
            deviance = _tweedie_deviance(y_true, y_pred, p_power)
        except ValueError:
            deviance = float("nan")
        zero_frac = float(np.mean(y_pred <= 1e-3))

        return {
            "MAE": mae,
            "TweedieDeviance": deviance,
            "ZeroFracPred": zero_frac,
        }
