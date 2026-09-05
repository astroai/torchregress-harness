"""Censored-imaging suite — right-censored regression on microscopy counts.

Mirrors tabular ``CensoredSuite`` on images: detection-limit censoring
(``censoring_train/test`` codes, 1 = right-censored) with CNN backbones.
Trains with censoring-aware torchregress losses and evaluates point quality
on the observed targets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchregress.losses import (
    AFTLoss,
    CensoredGaussianNLLLoss,
    CensoredQuantileLoss,
    WeightedMSELoss,
)

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig

from ._base import ImageHarnessRunner, _image_to_dataloader

try:
    import pandas as pd
    from lifelines import LogNormalAFTFitter

    _LIFELINES_AVAILABLE = True
except ImportError:
    _LIFELINES_AVAILABLE = False


def _train_censored_image(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    censoring: torch.Tensor,
    config: HarnessConfig,
) -> None:
    """Mini-batch training forwarding the ``censoring`` kwarg (CNN version)."""
    from torch.optim import Adam

    y_2d = y if y.dim() >= 2 else y.unsqueeze(1)
    c_2d = censoring if censoring.dim() >= 2 else censoring.unsqueeze(1)
    loader = DataLoader(TensorDataset(X, y_2d, c_2d), batch_size=config.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=config.lr)
    model.train()
    for _ in range(config.epochs):
        for Xb, yb, cb in loader:
            optimizer.zero_grad()
            out = model(Xb)
            y_pred = (out[..., 0:1], out[..., 1:2]) if out.shape[-1] >= 2 else out
            loss = loss_fn(y_pred, yb, censoring=cb)
            loss.backward()
            optimizer.step()
    model.eval()


class CensoredImagingSuite(ImageHarnessRunner):
    """Right-censored count regression on microscopy images."""

    @property
    def name(self) -> str:
        return "censored_imaging"

    @property
    def description(self) -> str:
        return "Detection-limit censored counts: CNN censored losses vs MSE"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "bbbc_censored",
            "bbbc_censored_tiny",
            "tcga_survival",
            "tcga_survival_tiny",
        ]

    def methods(self) -> list[str]:
        m = ["MSE", "CensoredGaussianNLL", "CensoredQuantile", "AFT"]
        if _LIFELINES_AVAILABLE:
            m.append("lifelines_LogNormalAFT")
        return m

    def library_name(self, method: str) -> str:
        return "lifelines" if method.startswith("lifelines") else "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "MSE": "CNN + WeightedMSELoss — ignores censoring (baseline)",
            "CensoredGaussianNLL": "CNN + censored Gaussian NLL (detection-limit aware)",
            "CensoredQuantile": "CNN + censored median regression (τ=0.5)",
            "AFT": "CNN + log-normal AFT on counts",
            "lifelines_LogNormalAFT": "lifelines LogNormalAFT on flattened pixels (external)",
        }.get(method, "")

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
        X = torch.from_numpy(np.asarray(splits["X_train"])).float().to(current_device())
        y = torch.from_numpy(np.asarray(splits["y_train"])).float().to(current_device())
        c = (
            torch.from_numpy(np.asarray(splits["censoring_train"]).astype(int))
            .to(current_device())
            .long()
        )
        in_channels = int(np.asarray(splits["X_train"]).shape[1])

        if method == "MSE":
            model = self._build_cnn(in_channels=in_channels, out_dim=1)
            loader = self._make_train_loader(splits, batch_size=config.batch_size)
            from ._base import _train_image_model

            _train_image_model(model, WeightedMSELoss(), loader, config)
            return {"type": "torch", "model": model}
        if method == "CensoredGaussianNLL":
            model = self._build_cnn(in_channels=in_channels, out_dim=2)
            _train_censored_image(model, CensoredGaussianNLLLoss(), X, y, c, config)
            return {"type": "torch", "model": model}
        if method == "CensoredQuantile":
            model = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_censored_image(model, CensoredQuantileLoss(quantile=0.5), X, y, c, config)
            return {"type": "torch", "model": model}
        if method == "AFT":
            model = self._build_cnn(in_channels=in_channels, out_dim=2)
            _train_censored_image(model, AFTLoss(), X, y, c, config)
            return {"type": "torch", "model": model}
        if method == "lifelines_LogNormalAFT":
            Xf = np.asarray(splits["X_train"]).reshape(len(X), -1)
            event = 1 - np.asarray(splits["censoring_train"]).astype(int)
            y_raw = np.asarray(splits["y_train"]).ravel()
            y_shift = float(abs(y_raw.min()) + 1.0) if y_raw.min() <= 0 else 0.0
            df = pd.DataFrame(Xf, columns=[f"x{i}" for i in range(Xf.shape[1])])
            df["duration"] = y_raw + y_shift
            df["event"] = event.ravel()
            aft = LogNormalAFTFitter()
            aft.fit(df, duration_col="duration", event_col="event")
            return {
                "type": "lifelines",
                "model": aft,
                "n_features": Xf.shape[1],
                "y_shift": y_shift,
            }
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict / evaluate
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        if model_state.get("type") == "lifelines":
            m = model_state["model"]
            Xf = np.asarray(X_test).reshape(len(np.asarray(X_test)), -1)
            df = pd.DataFrame(Xf, columns=[f"x{i}" for i in range(Xf.shape[1])])
            y_pred = np.asarray(m.predict_expectation(df), dtype=np.float64)
            return {"y_pred": y_pred - model_state.get("y_shift", 0.0)}
        m = model_state["model"]
        loader = _image_to_dataloader(X_test, y=None, batch_size=64, shuffle=False)
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                raw = m(xb.to(current_device()))
                outs.append(
                    raw[..., 0].reshape(-1).detach().cpu().numpy()
                    if raw.shape[-1] >= 2
                    else raw.reshape(-1).detach().cpu().numpy()
                )
        return {"y_pred": np.clip(np.concatenate(outs), 0, None)}

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        return {"RMSE": rmse, "MAE": mae, "R2": r2}
