"""Photo-z imaging suite — redshift from galaxy cutouts (GalaxiesML).

Subclasses the age-regression engine (CNN backbones + UQ heads) on HSC
grizy cutouts, with photo-z-native metrics layered on: bias, NMAD and
catastrophic-outlier rate in Δz/(1+z), alongside the standard
RMSE/coverage. Rows: distributional CNN heads, Laplace-posthoc BNN,
measured-noise EIV training. No pixel-RF (uninformative at 20k dims).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig
from suites.image.age_regression import AgeRegressionSuite


class PhotozImagingSuite(AgeRegressionSuite):
    """Redshift regression on GalaxiesML cutouts + photo-z metrics."""

    @property
    def name(self) -> str:
        return "photoz_imaging"

    @property
    def description(self) -> str:
        return "Photo-z from HSC cutouts (GalaxiesML): CNN UQ + Laplace BNN + noise-aware EIV"

    @property
    def supported_datasets(self) -> list[str]:
        return ["galaxiesml_cutouts"]

    def methods(self) -> list[str]:
        # Compact flagship set: distributional heads, measured-noise EIV,
        # MC-Dropout (the image BNN row). No sklearn_RF (uninformative on
        # pixels). No Laplace: FullNetworkLaplace's functorch/vmap Fisher
        # assumes MLP-rank modules and fails on conv nets (verified minimal
        # repro) — BNN-on-images is covered by MC_Dropout instead.
        return [
            "CNN+GaussianNLL",
            "CNN+MDN_K3",
            "CNN+MultiQuantile",
            "CNN+NormalizingFlow",
            "CNN+MC_Dropout",
            "CNN+NoiseAware",
        ]

    def method_notes(self, method: str) -> str:
        base = super().method_notes(method)
        tag = {
            "CNN+GaussianNLL": " (photo-z baseline head)",
            "CNN+MDN_K3": " (multimodal posteriors for color degeneracies)",
            "CNN+NoiseAware": " (measured sky-sigma input augmentation)",
        }.get(method, "")
        return f"{base}{tag}"

    # ------------------------------------------------------------------
    # fit — new rows here, everything else delegates to the age engine
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        if method == "CNN+NoiseAware":
            return self._fit_noiseaware(splits, config)
        return super().fit(method, splits, config)

    def _fit_noiseaware(
        self, splits: dict[str, np.ndarray], config: HarnessConfig
    ) -> dict[str, Any]:
        """EIV training: per-image measured sky-sigma input augmentation."""
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        set_harness_seed(config.seed)
        in_channels, _, _ = self._image_shape(splits)
        model = self._build_cnn(in_channels=in_channels, out_dim=1)
        dev = current_device()
        model.to(dev)
        sig = np.asarray(
            splits.get("sigma_image_train", np.full(len(splits["X_train"]), 0.01)),
            dtype=np.float32,
        ).reshape(-1)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.asarray(splits["X_train"])).float(),
                torch.from_numpy(np.asarray(splits["y_train"])).float().unsqueeze(1),
                torch.from_numpy(sig).float(),
            ),
            batch_size=config.batch_size,
            shuffle=True,
        )
        opt = optim.Adam(model.parameters(), lr=config.lr)
        model.train()
        for _ in range(config.epochs):
            for Xb, yb, sb in loader:
                Xb, yb, sb = Xb.to(dev), yb.to(dev), sb.to(dev)
                opt.zero_grad(set_to_none=True)
                noisy = Xb + sb.view(-1, 1, 1, 1) * torch.randn_like(Xb)
                nn.MSELoss()(model(noisy), yb).backward()
                opt.step()
        model.eval()
        return {"type": "point", "model": model}

    # ------------------------------------------------------------------
    # predict — NoiseAware is point-type (parent handles); delegate all
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        return super().predict(method, model_state, X_test, config)

    # ------------------------------------------------------------------
    # evaluate — base metrics + photo-z native diagnostics
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        metrics = super().evaluate(method, predictions, y_test, config)
        try:
            y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
            y_true = np.asarray(y_test).reshape(-1)
            # Native redshift units (runner standardizes targets).
            y_scaler = predictions.get("_y_scaler")
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
            dz = (y_pred - y_true) / (1.0 + np.abs(y_true))
            metrics["Bias"] = float(np.mean(dz))
            metrics["NMAD"] = float(1.4826 * np.median(np.abs(dz - np.median(dz))))
            metrics["OutlierFrac"] = float(np.mean(np.abs(dz) > 0.15))
        except Exception:
            pass
        return metrics
