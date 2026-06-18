"""Sklearn baselines suite: torchregress vs all standard sklearn regressors.

Compares three torchregress probabilistic methods (GaussianNLL, MultiQuantile,
MSE) against every sklearn regression estimator on all registered tabular
datasets.  This is the definitive ``does torchregress beat sklearn?`` suite.

Canonical scenarios
-------------------
* ``sklearn_diabetes_reference``
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
* ``sklearn_synthetic_reference``
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

Torchregress methods produce ``mean``/``std`` or ``y_pred``/``lower``/``upper``,
enabling both point and distribution/interval metrics.  Sklearn methods produce
``y_pred`` only (point metrics).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Sklearn regressors — all are hard dependencies (sklearn is a core harness dep)
# ---------------------------------------------------------------------------
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Torchregress losses
# ---------------------------------------------------------------------------
from torchregress.losses import BetaNLLLoss, MultiQuantileLoss

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import (
    _MLP,
    _gaussian_nll_predict,
    _hidden_size,
    _Linear,
    _train_lbfgs,
    _train_one,
)

# ---------------------------------------------------------------------------
# Registry: sklearn methods
# ---------------------------------------------------------------------------

_SKLEARN_REGRESSORS: dict[str, tuple] = {
    "sklearn_LinearRegression": (
        lambda seed: LinearRegression(),
        "OLS linear regression",
    ),
    "sklearn_Ridge": (
        lambda seed: Ridge(alpha=1.0, random_state=seed),
        "L2-regularized linear regression",
    ),
    "sklearn_Lasso": (
        lambda seed: Lasso(alpha=0.1, random_state=seed, max_iter=5000),
        "L1-regularized linear regression",
    ),
    "sklearn_ElasticNet": (
        lambda seed: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=seed, max_iter=5000),
        "L1+L2-regularized linear regression",
    ),
    "sklearn_BayesianRidge": (
        lambda seed: BayesianRidge(),
        "Bayesian linear regression with automatic relevance determination",
    ),
    "sklearn_HuberRegressor": (
        lambda seed: HuberRegressor(epsilon=1.35, max_iter=200),
        "robust linear regression with Huber loss",
    ),
    "sklearn_DecisionTree": (
        lambda seed: DecisionTreeRegressor(max_depth=10, random_state=seed),
        "single decision tree",
    ),
    "sklearn_RandomForest": (
        lambda seed: RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1,
        ),
        "bagged trees (100 estimators)",
    ),
    "sklearn_ExtraTrees": (
        lambda seed: ExtraTreesRegressor(
            n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1,
        ),
        "extremely randomized trees (100 estimators)",
    ),
    "sklearn_GradientBoosting": (
        lambda seed: GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed,
        ),
        "classic gradient boosting (100 estimators)",
    ),
    "sklearn_HistGradientBoosting": (
        lambda seed: HistGradientBoostingRegressor(
            max_iter=100, max_depth=6, learning_rate=0.1, random_state=seed,
        ),
        "histogram-based gradient boosting (100 iterations)",
    ),
    "sklearn_AdaBoost": (
        lambda seed: AdaBoostRegressor(
            n_estimators=50, learning_rate=0.5, random_state=seed,
        ),
        "AdaBoost on shallow decision stubs",
    ),
    "sklearn_SVR": (
        lambda seed: SVR(kernel="rbf", C=1.0, epsilon=0.1, max_iter=5000),
        "support vector regression (RBF kernel)",
    ),
    "sklearn_KNeighbors": (
        lambda seed: KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1),
        "k-nearest neighbors (k=5, distance-weighted)",
    ),
    "sklearn_MLPRegressor": (
        None,  # handled in its own elif branch — needs config-aware hyperparams
        "3-layer MLP (ReLU) + Adam — direct parity with torchregress MSE",
    ),
}

# ---------------------------------------------------------------------------
# Registry: sklearn methods
# ---------------------------------------------------------------------------


class SklearnBaselinesSuite(HarnessRunner):
    """Torchregress vs 14 standard sklearn tabular regressors.

    The big comparison suite — provides a definitive answer to
    "does torchregress beat sklearn?" on every registered dataset.
    """

    @property
    def name(self) -> str:
        return "sklearn_baselines"

    @property
    def description(self) -> str:
        return "Torchregress (OLS / GaussianNLL / MultiQuantile / MSE) vs all 14 sklearn regressors"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "diabetes",
            "openml_year",
            "insurance_claims",
            "house_sales",
            "make_regression_synthetic",
            "synthetic_hetero",
            "synthetic_outliers",
            "synthetic_tweedie",
            "synthetic_censored",
            "synthetic_eiv",
            "synthetic_sinusoidal",
            "tabred_sberbank",
            "tabred_cooking",
            "tabred_delivery",
            "tabred_maps",
            "tabred_weather",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """sklearn canonical regression examples."""
        return ["sklearn_diabetes_reference", "sklearn_synthetic_reference"]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    # Torchregress method keys (always available)
    _TR_METHODS: list[str] = ["OLS", "GaussianNLL_Linear", "GaussianNLL", "MultiQuantile", "MSE"]

    def methods(self) -> list[str]:
        # torchregress methods first, then all sklearn methods
        return self._TR_METHODS + list(_SKLEARN_REGRESSORS.keys())

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn_"):
            return "sklearn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        if method == "OLS":
            return "analytic linear regression (torch.linalg.lstsq) → point prediction"
        if method == "GaussianNLL_Linear":
            return "_Linear + BetaNLLLoss(beta=0.0) via two-stage: MSE→OLS (LBFGS) + analytic residual variance; exact mean parity with sklearn LinearRegression"
        if method == "GaussianNLL":
            return "MLP + BetaNLLLoss(beta=0.5) → mean & std"
        if method == "MultiQuantile":
            return "MLP + MultiQuantileLoss (q=0.1,0.5,0.9) → median & quantile intervals"
        if method == "MSE":
            return "3-layer MLP + Adam + MSE — Kaiming init (sklearn uses Glorot); converges to same optimum at 200+ epochs"
        if method == "sklearn_MLPRegressor":
            return "3-layer MLP (ReLU) + Adam + MSE — Glorot init (torchregress uses Kaiming); converges to same optimum at 200+ epochs"
        if method in _SKLEARN_REGRESSORS:
            return _SKLEARN_REGRESSORS[method][1]
        return ""

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"].reshape(-1)
        in_dim = X_train_np.shape[1]

        # --- torchregress: OLS (analytic closed-form, matches sklearn LinearRegression) ---
        if method == "OLS":
            X_t = torch.from_numpy(X_train_np).float()
            y_t = torch.from_numpy(y_train_np).float().unsqueeze(1)

            # Design matrix [X | 1] with intercept column
            ones = torch.ones(X_t.shape[0], 1, dtype=X_t.dtype, device=X_t.device)
            A = torch.cat([X_t, ones], dim=-1)

            # Analytic OLS via lstsq:  w = argmin ||A w - y||₂
            solution = torch.linalg.lstsq(A, y_t)
            coef = solution.solution.squeeze(-1)  # (in_dim + 1,)

            return {"type": "ols", "coef": coef}

        # --- torchregress: GaussianNLL_Linear (OLS parity + analytic variance) ---
        elif method == "GaussianNLL_Linear":
            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            # Stage 1: _Linear(→1) + MSE via LBFGS → exact OLS optimum
            # (MSE is strictly convex for a linear model)
            model_mse = _Linear(in_dim, out_dim=1)
            _train_lbfgs(model_mse, nn.MSELoss(), X, y, max_iter=500, tolerance_grad=1e-7)

            # Stage 2: Clone to 2-output model with analytic residual variance
            # No further training needed — mean is exactly OLS, variance is
            # the homoscedastic residual variance (MLE for constant σ²)
            model = _Linear(in_dim, out_dim=2)
            with torch.no_grad():
                model.fc.weight[0, :].copy_(model_mse.fc.weight[0, :])
                model.fc.bias[0].copy_(model_mse.fc.bias[0])
                residuals = (y - model_mse(X)).squeeze(-1)
                model.fc.weight[1, :].zero_()
                model.fc.bias[1].fill_(torch.log(residuals.var().clamp_min(1e-8)).item())

            return {"type": "gaussian_nll", "model": model}

        # --- torchregress: GaussianNLL ---
        elif method == "GaussianNLL":
            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            hidden = _hidden_size(in_dim)
            model = _MLP(in_dim, hidden, 2)  # mean, log_variance
            loss_fn = BetaNLLLoss(beta=0.5)
            _train_one(model, loss_fn, X, y, config)
            model.eval()

            return {"type": "gaussian_nll", "model": model}

        # --- torchregress: MultiQuantile ---
        elif method == "MultiQuantile":
            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            hidden = _hidden_size(in_dim)
            quantiles = [0.1, 0.5, 0.9]
            model = _MLP(in_dim, hidden, len(quantiles))
            loss_fn = MultiQuantileLoss(quantiles=quantiles)
            _train_one(model, loss_fn, X, y, config)
            model.eval()

            return {"type": "multi_quantile", "model": model, "quantiles": quantiles}

        # --- torchregress: MSE ---
        elif method == "MSE":
            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            hidden = _hidden_size(in_dim)
            model = _MLP(in_dim, hidden, 1)
            loss_fn = nn.MSELoss()
            _train_one(model, loss_fn, X, y, config)
            model.eval()

            return {"type": "mse", "model": model}

        # --- sklearn MLPRegressor — direct parity with torchregress MSE ---
        elif method == "sklearn_MLPRegressor":
            hidden = _hidden_size(in_dim)
            est = MLPRegressor(
                hidden_layer_sizes=(hidden, hidden),
                activation="relu",
                solver="adam",
                alpha=0.0,
                batch_size=min(config.batch_size, len(X_train_np)),
                learning_rate_init=config.lr,
                max_iter=config.epochs,
                tol=1e-10,  # disable early stopping
                random_state=config.seed,
                early_stopping=False,
            )
            est.fit(X_train_np, y_train_np)
            return {"type": "sklearn", "model": est}

        # --- sklearn methods ---
        elif method in _SKLEARN_REGRESSORS:
            factory, _ = _SKLEARN_REGRESSORS[method]
            est = factory(config.seed)

            # Scale y for SVR (sensitive to target scale)
            if method == "sklearn_SVR":
                from sklearn.preprocessing import StandardScaler as SkScaler

                y_scaler = SkScaler()
                y_scaled = y_scaler.fit_transform(y_train_np.reshape(-1, 1)).ravel()
                est.fit(X_train_np, y_scaled)
                return {"type": "sklearn", "model": est, "y_scaler": y_scaler}

            est.fit(X_train_np, y_train_np)
            return {"type": "sklearn", "model": est}

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self, method: str, model_state: dict[str, Any],
        X_test: np.ndarray, config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        model_type = model_state["type"]

        # --- torchregress: GaussianNLL → mean, std ---
        if model_type == "gaussian_nll":
            mean, std = _gaussian_nll_predict(model_state["model"], X_test)
            return {"mean": mean, "std": std}

        # --- torchregress: MultiQuantile → y_pred, lower, upper ---
        elif model_type == "multi_quantile":
            m = model_state["model"]
            with torch.no_grad():
                out = m(torch.from_numpy(X_test).float())
                # out shape: (N, 3) for quantiles [0.1, 0.5, 0.9]
                lo = out[:, 0].numpy().reshape(-1)
                med = out[:, 1].numpy().reshape(-1)
                hi = out[:, 2].numpy().reshape(-1)
            return {"y_pred": med, "lower": lo, "upper": hi}

        # --- torchregress: OLS → y_pred (analytic, matches sklearn LinearRegression) ---
        elif model_type == "ols":
            coef = model_state["coef"]
            X_t = torch.from_numpy(X_test).float()
            ones = torch.ones(X_t.shape[0], 1, dtype=X_t.dtype, device=coef.device)
            A = torch.cat([X_t, ones], dim=-1)
            y_pred = (A @ coef).numpy().reshape(-1)
            return {"y_pred": y_pred}

        # --- torchregress: MSE → y_pred ---
        elif model_type == "mse":
            m = model_state["model"]
            with torch.no_grad():
                y_pred = m(torch.from_numpy(X_test).float()).squeeze(-1).numpy().reshape(-1)
            return {"y_pred": y_pred}

        # --- sklearn → y_pred ---
        elif model_type == "sklearn":
            est = model_state["model"]
            y_pred = est.predict(X_test).reshape(-1)

            # Un-scale SVR predictions
            if "y_scaler" in model_state:
                y_pred = model_state["y_scaler"].inverse_transform(
                    y_pred.reshape(-1, 1),
                ).ravel()

            return {"y_pred": y_pred}

        raise ValueError(f"Unknown model type: {model_type}")
