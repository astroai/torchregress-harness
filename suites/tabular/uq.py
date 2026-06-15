"""Uncertainty quantification suite: torchregress vs sklearn GP + conformal.

Compares three torchregress UQ strategies (single GaussianNLL, MC-dropout
with 20 forward passes, and a 10-member deep ensemble) against sklearn's
``GaussianProcessRegressor`` and a quantile gradient-boosting conformal
baseline on small-to-medium tabular datasets.

Canonical scenarios
-------------------
* ``sklearn_diabetes_reference``
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
* ``sklearn_synthetic_reference``
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

Architecture
------------ * ``GaussianNLL`` → single MLP (mean, log_var) + BetaNLLLoss → ``{"mean", "std"}``
* ``MC_Dropout`` → MLP with 50% dropout, 20 stochastic forward passes at
  test time → empirical mean & std → ``{"mean", "std"}``
* ``DeepEnsemble_K10`` → 10 independently-trained MLPs → mixture mean &
  total variance → ``{"mean", "std"}``

Sklearn baselines produce ``{"mean", "std"}`` (GP) or ``{"y_pred", "lower", "upper"}`` (quantile GBR).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Sklearn — hard deps
# ---------------------------------------------------------------------------
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# ---------------------------------------------------------------------------
# Torchregress losses
# ---------------------------------------------------------------------------
from torchregress.losses import BetaNLLLoss

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import (
    _MLP,
    _gaussian_nll_predict,
    _hidden_size,
    _mixture_gaussian_stats,
    _train_one,
)

# ---------------------------------------------------------------------------
# MC-dropout forward pass
# ---------------------------------------------------------------------------

_MC_SAMPLES = 20  # number of stochastic forward passes


def _mc_predict(
    model: _MLP, X: torch.Tensor, n_samples: int = _MC_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Run n_samples stochastic forward passes, return empirical mean & std.

    Temporarily enables training mode so dropout is active, then restores
    the original mode before returning.
    """
    was_training = model.training
    model.train()  # enable dropout for stochastic forward passes
    try:
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = model(X)
                preds.append(out[:, 0].numpy().reshape(-1))
        stacked = np.stack(preds, axis=0)  # (n_samples, n_test)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=0).clip(1e-8)
        return mean, std
    finally:
        if not was_training:
            model.eval()


# ---------------------------------------------------------------------------
# Deep ensemble size (larger than ensembles suite's K=5)
# ---------------------------------------------------------------------------

_N_ENSEMBLE = 10


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class UQSuite(HarnessRunner):
    """Uncertainty quantification benchmark suite.

    Compares torchregress UQ methods (GaussianNLL, MC-dropout,
    deep ensemble K=10) against sklearn GaussianProcessRegressor
    and quantile gradient boosting.
    """

    # GP doesn't scale to large N — cap training samples
    _GP_MAX_SAMPLES = 2000

    @property
    def name(self) -> str:
        return "uq"

    @property
    def description(self) -> str:
        return "UQ methods: GaussianNLL / MC-dropout / DeepEnsemble_K10 vs sklearn GP & quantile GBR"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "diabetes",
            "make_regression_synthetic",
            "synthetic_hetero",
            "synthetic_sinusoidal",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """sklearn canonical regression benchmarks."""
        return ["sklearn_diabetes_reference", "sklearn_synthetic_reference"]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    # Note: BetaNLL is not yet available as a standalone loss in torchregress.
    # When torchregress exposes a BetaNLL loss (e.g., torchregress.losses.BetaNLLLoss),
    # add a "BetaNLL" method here that trains _MLP(in_dim, hidden, 2) with it.

    _TR_METHODS: list[str] = [
        "GaussianNLL",
        "MC_Dropout",
        "DeepEnsemble_K10",
    ]

    _SKLEARN_METHODS: list[str] = [
        "sklearn_GP",
        "sklearn_QuantileGBR",
    ]

    def methods(self) -> list[str]:
        return self._TR_METHODS + self._SKLEARN_METHODS

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn_"):
            return "sklearn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        notes = {
            "GaussianNLL": "single MLP with BetaNLLLoss(beta=0.5) → aleatoric uncertainty",
            "MC_Dropout": f"MLP with 50% dropout, {_MC_SAMPLES} stochastic passes → empirical mean & std",
            "DeepEnsemble_K10": f"{_N_ENSEMBLE}× MLP + GaussianNLL → mixture mean & total variance",
            "sklearn_GP": "GaussianProcessRegressor(RBF+WhiteKernel) → posterior mean & std",
            "sklearn_QuantileGBR": "GradientBoostingRegressor(q=0.05,0.5,0.95) → median & 90% intervals",
        }
        return notes.get(method, "")

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
        hidden = _hidden_size(in_dim)

        # --- GaussianNLL: single MLP with mean + log_var output ---
        if method == "GaussianNLL":
            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            model = _MLP(in_dim, hidden, 2)  # mean, log_variance
            loss_fn = BetaNLLLoss(beta=0.5)
            _train_one(model, loss_fn, X, y, config)
            model.eval()

            return {"type": "gaussian_nll", "model": model}

        # --- MC_Dropout: MLP with dropout, train normally ---
        elif method == "MC_Dropout":
            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            model = _MLP(in_dim, hidden, 1, dropout_p=0.5)
            loss_fn = nn.MSELoss()
            _train_one(model, loss_fn, X, y, config)
            # Store in eval mode; _mc_predict() will temporarily enable train mode
            model.eval()

            return {"type": "mc_dropout", "model": model}

        # --- DeepEnsemble_K10: 10 independent MLPs ---
        elif method == "DeepEnsemble_K10":
            models: list[nn.Module] = []

            for i in range(_N_ENSEMBLE):
                member_seed = config.seed + i * 1000
                set_harness_seed(member_seed)

                X_t = torch.from_numpy(X_train_np).float()
                y_t = torch.from_numpy(y_train_np).float().unsqueeze(1)

                model = _MLP(in_dim, hidden, 2)
                loss_fn = BetaNLLLoss(beta=0.5)
                _train_one(model, loss_fn, X_t, y_t, config)
                model.eval()
                models.append(model)

            return {"type": "deep_ensemble", "models": models}

        # --- sklearn GP: subsample if dataset is large ---
        elif method == "sklearn_GP":
            n_train = len(X_train_np)
            if n_train > self._GP_MAX_SAMPLES:
                rng = np.random.default_rng(config.seed)
                idx = rng.choice(n_train, self._GP_MAX_SAMPLES, replace=False)
                X_gp = X_train_np[idx]
                y_gp = y_train_np[idx]
            else:
                X_gp = X_train_np
                y_gp = y_train_np

            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=config.seed,
            )
            gp.fit(X_gp, y_gp)

            return {
                "type": "gp",
                "model": gp,
                "n_train": n_train,
                "n_used": len(X_gp),
            }

        # --- sklearn quantile GBR: three separate models per quantile ---
        elif method == "sklearn_QuantileGBR":
            quantiles = [0.05, 0.50, 0.95]
            models = {}
            for q in quantiles:
                gbr = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=config.seed,
                )
                gbr.fit(X_train_np, y_train_np)
                models[q] = gbr

            return {"type": "quantile_gbr", "models": models, "quantiles": quantiles}

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self, method: str, model_state: dict[str, Any],
        X_test: np.ndarray, config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        model_type = model_state["type"]

        # --- GaussianNLL → mean & std ---
        if model_type == "gaussian_nll":
            mean, std = _gaussian_nll_predict(model_state["model"], X_test)
            return {"mean": mean, "std": std}

        # --- MC_Dropout → empirical mean & std ---
        elif model_type == "mc_dropout":
            m = model_state["model"]
            mean, std = _mc_predict(m, torch.from_numpy(X_test).float())
            return {"mean": mean, "std": std}

        # --- DeepEnsemble_K10 → mixture mean & total variance ---
        elif model_type == "deep_ensemble":
            means = []
            stds = []
            for m in model_state["models"]:
                mu, std = _gaussian_nll_predict(m, X_test)
                means.append(mu)
                stds.append(std)
            mixture_mean, mixture_std = _mixture_gaussian_stats(means, stds)
            return {"mean": mixture_mean, "std": mixture_std}

        # --- sklearn GP → posterior mean & std ---
        elif model_type == "gp":
            gp = model_state["model"]
            mean, std = gp.predict(X_test, return_std=True)
            return {"mean": mean.reshape(-1), "std": std.clip(1e-8)}

        # --- sklearn quantile GBR → median + 90% intervals ---
        elif model_type == "quantile_gbr":
            lo = model_state["models"][0.05].predict(X_test).reshape(-1)
            med = model_state["models"][0.50].predict(X_test).reshape(-1)
            hi = model_state["models"][0.95].predict(X_test).reshape(-1)
            return {"y_pred": med, "lower": lo, "upper": hi}

        raise ValueError(f"Unknown model type: {model_type}")
