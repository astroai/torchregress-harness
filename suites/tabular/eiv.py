"""Errors-in-Variables (EIV) regression suite.

Compares torchregress EIV losses against a plain MSE baseline on synthetic
data with known feature-measurement noise.  The dataset provides both noisy
observed features (X) and clean true features (X_true); the suite trains on
the noisy data and evaluates point-prediction quality.

Methods
------
- MSE              : MLP + WeightedMSELoss (no EIV correction — baseline)
- FunctionalEIV    : MLP + FunctionalEIVLoss (analytical gradient-based)
- StructuralEIV    : MLP + StructuralEIVLoss (full cross-covariance model)
- ODR              : MLP + OrthogonalDistanceRegressionLoss (latent-x opt)
- EnsembleEIV      : MLP + EnsembleEIVLoss (Monte-Carlo input perturbation)
- TLS              : Total Least Squares via SVD (orthogonal distance, external)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from suites._backbones import _MLP, _hidden_size, _train_one
from torchregress.losses import (
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
    WeightedMSELoss,
)

# ---------------------------------------------------------------------------
# Custom training helper for EIV losses
#
# EIV losses are called with (x_obs, y_obs), NOT (model(x), y).  The model
# is passed at construction time and the loss calls it internally.
# ---------------------------------------------------------------------------


def _train_eiv(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    config: HarnessConfig | None = None,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
) -> None:
    """Mini-batch training loop for EIV losses (loss_fn takes x_obs, y_obs)."""
    if config is not None:
        n_epochs = epochs if epochs is not None else config.epochs
        bs = batch_size if batch_size is not None else config.batch_size
        learning_rate = lr if lr is not None else config.lr
    else:
        n_epochs = epochs or 50
        bs = batch_size or 64
        learning_rate = lr or 1e-3

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _epoch in range(n_epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            # EIV losses: forward(x_obs, target) — the loss internally calls model(x_obs)
            loss = loss_fn(X_batch, y_batch)
            loss.backward()
            optimizer.step()
    model.eval()


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class EIVSuite(HarnessRunner):
    """Errors-in-Variables regression benchmark suite."""

    @property
    def name(self) -> str:
        return "eiv"

    @property
    def description(self) -> str:
        return (
            "Errors-in-Variables: torchregress EIV losses vs plain MSE baseline "
            "on synthetic data with known feature noise"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["synthetic_eiv"]

    @property
    def canonical_scenarios(self) -> list[str]:
        return []

    def methods(self) -> list[str]:
        return ["MSE", "FunctionalEIV", "StructuralEIV", "ODR", "EnsembleEIV", "TLS"]

    def library_name(self, method: str) -> str:
        if method == "TLS":
            return "numpy"
        return "torchregress"

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"]

        X = torch.from_numpy(X_train_np).float()
        y = torch.from_numpy(y_train_np).float()
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Known noise parameters from the dataset generator
        sigma_x = float(splits.get("sigma_x", 0.5))
        sigma_y = float(splits.get("sigma_y", 0.1))

        in_dim = X.shape[1]
        out_dim = y.shape[1]
        hidden = _hidden_size(in_dim)

        if method == "MSE":
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = WeightedMSELoss()
            _train_one(model, loss_fn, X, y, config)
            return {"type": "torch", "model": model}

        elif method == "FunctionalEIV":
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = FunctionalEIVLoss(model=model, sigma_x=sigma_x, sigma_y=sigma_y)
            _train_eiv(model, loss_fn, X, y, config)
            return {"type": "torch", "model": model}

        elif method == "StructuralEIV":
            model = _MLP(in_dim, hidden, out_dim)
            # Uncorrelated feature/target noise → zero cross-covariance
            sigma_xy = torch.zeros(out_dim, in_dim)
            loss_fn = StructuralEIVLoss(
                model=model, sigma_x=sigma_x, sigma_y=sigma_y, sigma_xy=sigma_xy
            )
            _train_eiv(model, loss_fn, X, y, config)
            return {"type": "torch", "model": model}

        elif method == "ODR":
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = OrthogonalDistanceRegressionLoss(
                model=model, sigma_x=sigma_x, sigma_y=sigma_y
            )
            _train_eiv(model, loss_fn, X, y, config)
            return {"type": "torch", "model": model}

        elif method == "EnsembleEIV":
            model = _MLP(in_dim, hidden, out_dim)
            loss_fn = EnsembleEIVLoss(model=model, sigma_x=sigma_x)
            _train_eiv(model, loss_fn, X, y, config)
            return {"type": "torch", "model": model}

        elif method == "TLS":
            # Total Least Squares via SVD — minimises orthogonal distances
            X_np = X_train_np.astype(np.float64)
            y_np = y_train_np.ravel().astype(np.float64)
            # Centre the data
            X_mean = X_np.mean(axis=0)
            y_mean = y_np.mean()
            X_c = X_np - X_mean
            y_c = y_np - y_mean
            # SVD of the augmented matrix [X_c | y_c]
            Z = np.column_stack([X_c, y_c])
            _, _, Vt = np.linalg.svd(Z, full_matrices=False)
            # Solution from the right singular vector of the smallest singular value
            V = Vt[-1]  # last row = smallest singular vector
            beta = -V[:-1] / V[-1]
            intercept = y_mean - X_mean @ beta
            return {"type": "tls", "coef": beta, "intercept": intercept}

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        del method, config
        model_type = model_state.get("type", "torch")

        if model_type == "tls":
            beta = model_state["coef"]
            intercept = model_state["intercept"]
            y_pred = X_test @ beta + intercept
            return {"y_pred": y_pred}

        # Torch models
        m = model_state["model"]
        with torch.no_grad():
            X = torch.from_numpy(X_test).float()
            y_pred = m(X).squeeze(-1).numpy()
        return {"y_pred": y_pred}

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        del method, config
        from harness.runners import _inverse_transform_predictions, _inverse_transform_y

        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_test = _inverse_transform_y(y_test, y_scaler)

        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)

        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    # ------------------------------------------------------------------
    # notes
    # ------------------------------------------------------------------

    def method_notes(self, method: str) -> str:
        notes: dict[str, str] = {
            "MSE": "MLP + WeightedMSELoss — plain baseline (no EIV correction)",
            "FunctionalEIV": (
                "MLP + FunctionalEIVLoss — analytical gradient-based variance propagation"
            ),
            "StructuralEIV": (
                "MLP + StructuralEIVLoss — full cross-covariance structural model"
            ),
            "ODR": (
                "MLP + OrthogonalDistanceRegressionLoss — iterative latent-x optimisation"
            ),
            "EnsembleEIV": (
                "MLP + EnsembleEIVLoss — Monte-Carlo input perturbation averaging"
            ),
            "TLS": (
                "Total Least Squares via SVD — minimises orthogonal distances (self-contained, numpy-only)"
            ),
        }
        return notes.get(method, "")
