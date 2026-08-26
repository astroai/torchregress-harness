"""Censored regression suite.

Compares torchregress censored-regression losses against a plain MSE baseline
on synthetic right-censored data.  The dataset provides observed (censored)
targets and binary censoring indicators; the suite trains with censoring-aware
losses and evaluates point-prediction quality on the observed data.

Methods
------
- MSE                      : MLP + WeightedMSELoss (ignores censoring — baseline)
- CensoredGaussianNLL      : MLP + CensoredGaussianNLLLoss (censored Gaussian NLL)
- CensoredQuantile         : MLP + CensoredQuantileLoss (censored quantile / median)
- AFT                      : MLP + AFTLoss (log-normal accelerated failure time)
- lifelines_LogNormalAFT   : lifelines.LogNormalAFTFitter (log-normal AFT, external)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchregress.losses import (
    AFTLoss,
    CensoredGaussianNLLLoss,
    CensoredQuantileLoss,
    WeightedMSELoss,
)

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _train_one

# Optional external censored regression packages
try:
    import pandas as pd
    from lifelines import LogNormalAFTFitter

    _LIFELINES_AVAILABLE = True
except ImportError:
    _LIFELINES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom training helper for censored losses
#
# Censored losses need the ``censoring`` kwarg on every call.  The standard
# _train_one calls ``criterion(out, y)`` with no kwargs, so we need a thin
# wrapper that forwards the censoring tensor.
# ---------------------------------------------------------------------------


def _train_censored(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    censoring: torch.Tensor,
    config: HarnessConfig | None = None,
    *,
    transform_output: Any = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
) -> None:
    """Mini-batch training loop for censored losses (passes ``censoring`` kwarg)."""
    if config is not None:
        n_epochs = epochs if epochs is not None else config.epochs
        bs = batch_size if batch_size is not None else config.batch_size
        learning_rate = lr if lr is not None else config.lr
    else:
        n_epochs = epochs or 50
        bs = batch_size or 64
        learning_rate = lr or 1e-3

    y_2d = y if y.dim() >= 2 else y.unsqueeze(1)
    c_2d = censoring if censoring.dim() >= 2 else censoring.unsqueeze(1)
    dataset = TensorDataset(X, y_2d, c_2d)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _epoch in range(n_epochs):
        for X_batch, y_batch, c_batch in loader:
            optimizer.zero_grad()
            out = model(X_batch)
            if transform_output is not None:
                out = transform_output(out)
            # Some censored losses expect (mean, log_var) tuple, not concatenated
            if isinstance(out, torch.Tensor) and out.shape[-1] >= 2:
                # Split into (mean, log_var) tuple for GaussianNLL / AFT losses
                y_pred = (out[..., 0:1], out[..., 1:2])
            else:
                y_pred = out
            loss = loss_fn(y_pred, y_batch, censoring=c_batch)
            loss.backward()
            optimizer.step()
    model.eval()


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class CensoredSuite(HarnessRunner):
    """Censored regression benchmark suite."""

    @property
    def name(self) -> str:
        return "censored"

    @property
    def description(self) -> str:
        return (
            "Censored regression: torchregress censored losses vs MSE baseline "
            "on synthetic right-censored data"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["synthetic_censored"]

    @property
    def canonical_scenarios(self) -> list[str]:
        return []

    def methods(self) -> list[str]:
        methods = ["MSE", "CensoredGaussianNLL", "CensoredQuantile", "AFT"]
        if _LIFELINES_AVAILABLE:
            methods.append("lifelines_LogNormalAFT")
        return methods

    def library_name(self, method: str) -> str:
        if method.startswith("lifelines"):
            return "lifelines"
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
        censoring_np = splits["censoring_train"]

        X = torch.from_numpy(X_train_np).float().to(current_device())
        y = torch.from_numpy(y_train_np).float().to(current_device())
        censoring = torch.from_numpy(censoring_np.astype(int)).to(current_device()).long()

        in_dim = X.shape[1]
        hidden = _hidden_size(in_dim)
        out_dim = 2  # mean + log_var / log_scale for censored losses

        if method == "MSE":
            model = _MLP(in_dim, hidden, out_dim=1)
            _train_one(model, WeightedMSELoss(), X, y.unsqueeze(1), config)
            return {"type": "torch", "model": model}

        elif method == "CensoredGaussianNLL":
            model = _MLP(in_dim, hidden, out_dim=out_dim)
            loss_fn = CensoredGaussianNLLLoss()
            _train_censored(model, loss_fn, X, y, censoring, config)
            return {"type": "torch", "model": model}

        elif method == "CensoredQuantile":
            model = _MLP(in_dim, hidden, out_dim=1)
            loss_fn = CensoredQuantileLoss(quantile=0.5)
            _train_censored(model, loss_fn, X, y, censoring, config)
            return {"type": "torch", "model": model}

        elif method == "AFT":
            model = _MLP(in_dim, hidden, out_dim=out_dim)
            loss_fn = AFTLoss()
            _train_censored(model, loss_fn, X, y, censoring, config)
            return {"type": "torch", "model": model}

        elif method == "lifelines_LogNormalAFT":
            # lifelines uses 1=observed, 0=censored (opposite of torchregress convention)
            feature_cols = [f"x{i}" for i in range(X_train_np.shape[1])]
            event = 1 - censoring_np.astype(int)
            y_raw = y_train_np.ravel()
            # LogNormalAFTFitter requires strictly positive durations — shift if needed
            y_min = float(y_raw.min())
            y_shift = 0.0
            if y_min <= 0:
                y_shift = abs(y_min) + 1.0
            y_pos = y_raw + y_shift
            df = pd.DataFrame(X_train_np, columns=feature_cols)
            df["duration"] = y_pos
            df["event"] = event.ravel()
            aft = LogNormalAFTFitter()
            aft.fit(df, duration_col="duration", event_col="event")
            return {
                "type": "lifelines",
                "model": aft,
                "feature_cols": feature_cols,
                "y_shift": y_shift,
            }

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

        if model_type == "lifelines":
            m = model_state["model"]
            feature_cols = model_state["feature_cols"]
            y_shift = model_state.get("y_shift", 0.0)
            df_test = pd.DataFrame(X_test, columns=feature_cols)
            y_pred = np.asarray(m.predict_expectation(df_test), dtype=np.float64)
            if y_shift != 0.0:
                y_pred = y_pred - y_shift
            return {"y_pred": y_pred}

        # Torch models
        m = model_state["model"]
        with torch.no_grad():
            X = torch.from_numpy(X_test).float().to(current_device())
            raw = m(X)
            # For 2-output models (GaussianNLL, AFT), extract the mean
            if raw.shape[-1] >= 2:
                y_pred = raw[..., 0].squeeze(-1).detach().cpu().numpy()
            else:
                y_pred = raw.squeeze(-1).detach().cpu().numpy()
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
            "MSE": "MLP + WeightedMSELoss — plain baseline (ignores censoring)",
            "CensoredGaussianNLL": (
                "MLP + CensoredGaussianNLLLoss — censored Gaussian NLL with right-censoring support"
            ),
            "CensoredQuantile": ("MLP + CensoredQuantileLoss (τ=0.5) — censored median regression"),
            "AFT": ("MLP + AFTLoss — log-normal accelerated failure time model"),
            "lifelines_LogNormalAFT": (
                "lifelines.LogNormalAFTFitter — log-normal AFT (parametric, external)"
            ),
        }
        return notes.get(method, "")
