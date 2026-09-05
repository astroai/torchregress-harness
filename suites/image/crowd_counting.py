"""Crowd-counting suite — dense density-map regression (ShanghaiTech).

Trains UNet density estimators: the count is the spatial integral of the
density map. Primary task metric is count MAE (reported alongside standard
flattened-density metrics). Robust losses matter here: extreme-density heads
in congested scenes are the outliers of this domain.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import set_harness_seed
from harness.runners import HarnessConfig

from ._base import (
    ImageHarnessRunner,
    _image_to_dataloader,
    _train_image_model,
)


class CrowdCountingSuite(ImageHarnessRunner):
    """Dense crowd-density regression with count-level evaluation."""

    Z_SCORE = 1.645
    DENSE_BATCH_SIZE = 16

    @property
    def name(self) -> str:
        return "crowd_counting"

    @property
    def description(self) -> str:
        return "Crowd density maps: UNet MSE/Huber/GaussianNLL with count MAE"

    @property
    def supported_datasets(self) -> list[str]:
        return ["shanghai_crowd", "shanghai_crowd_tiny"]

    def methods(self) -> list[str]:
        return ["UNet+MSE", "UNet+Huber", "UNet+GaussianNLL"]

    def library_name(self, method: str) -> str:
        return "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "UNet+MSE": "UNet density map + MSE (point baseline)",
            "UNet+Huber": "UNet density map + PseudoHuber (robust to congested outliers)",
            "UNet+GaussianNLL": "UNet 2ch density mean+logvar per pixel",
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
        in_channels, _, _ = self._image_shape(splits)
        train_loader = self._make_train_loader(splits, batch_size=config.batch_size)
        if method == "UNet+MSE":
            m = self._build_unet(in_channels=in_channels, out_channels=1, base_filters=32)
            _train_image_model(m, nn.MSELoss(), train_loader, config)
            return {"type": "mse", "model": m}
        if method == "UNet+Huber":
            from torchregress.losses import PseudoHuberLoss

            m = self._build_unet(in_channels=in_channels, out_channels=1, base_filters=32)
            _train_image_model(m, PseudoHuberLoss(), train_loader, config)
            return {"type": "mse", "model": m}
        if method == "UNet+GaussianNLL":
            from torchregress.losses import GaussianNLLLoss

            m = self._build_unet(in_channels=in_channels, out_channels=2, base_filters=32)
            _train_image_model(m, GaussianNLLLoss(split_dim=1), train_loader, config)
            return {"type": "gaussian_nll", "model": m}
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
        t = model_state["type"]
        model = model_state["model"]
        device = next(model.parameters()).device
        loader = _image_to_dataloader(
            X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False
        )
        if t == "mse":
            preds = []
            with torch.no_grad():
                for (xb,) in loader:
                    preds.append(model(xb.to(device)).cpu().numpy())
            dens = np.concatenate(preds, axis=0).reshape(len(np.asarray(X_test)), -1)
            dens = np.clip(dens, 0, None)
            return {"y_pred": dens.reshape(-1), "_density": dens}
        # gaussian_nll: 2ch (mean, logvar) per pixel.
        means, stds = [], []
        with torch.no_grad():
            for (xb,) in loader:
                out = model(xb.to(device))
                means.append(out[:, 0:1, :, :].cpu().numpy())
                var = torch.exp(out[:, 1:2, :, :]).clamp_min(1e-8).cpu().numpy()
                stds.append(np.sqrt(var))
        mean = np.concatenate(means, axis=0).reshape(len(np.asarray(X_test)), -1)
        std = np.concatenate(stds, axis=0).reshape(len(np.asarray(X_test)), -1)
        mean = np.clip(mean, 0, None)
        z = self.Z_SCORE
        return {
            "lower": np.clip(mean - z * std, 0, None).reshape(-1),
            "upper": (mean + z * std).reshape(-1),
            "y_pred": mean.reshape(-1),
            "mean": mean.reshape(-1),
            "std": std.reshape(-1),
            "_density": mean,
        }

    # ------------------------------------------------------------------
    # evaluate — density metrics + count MAE/RMSE from spatial sums
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        y_flat = np.asarray(y_test).reshape(-1)
        metrics = super().evaluate(method, predictions, y_flat, config)
        # Count-level metrics need the test counts; the runner passes only
        # y_test, so recover counts as density sums (exact for GT maps).
        dens_pred = predictions.get(
            "_density", np.asarray(predictions["y_pred"]).reshape(len(y_flat), -1)
        )
        dens_pred = np.asarray(dens_pred).reshape(-1, y_flat.size // max(len(dens_pred), 1))
        n = dens_pred.shape[0]
        pred_counts = dens_pred.sum(axis=1)
        true_counts = np.asarray(y_test).reshape(n, -1).sum(axis=1)
        metrics["CountMAE"] = float(np.mean(np.abs(pred_counts - true_counts)))
        metrics["CountRMSE"] = float(np.sqrt(np.mean((pred_counts - true_counts) ** 2)))
        return metrics
