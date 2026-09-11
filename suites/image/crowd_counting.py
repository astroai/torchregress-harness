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
from harness.devices import current_device
from harness.runners import HarnessConfig

from ._base import (
    ImageHarnessRunner,
    _image_to_dataloader,
    _train_image_model,
)


def _vgg_block(in_ch: int, out_ch: int, n_convs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(n_convs):
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)]
        in_ch = out_ch
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class _CSRNet(nn.Module):
    """CSRNet density estimator (Li, Zhang, Chen, CVPR 2016), unpretrained.

    Frontend mirrors VGG-16 conv1-conv3 (3 poolings → 1/8 resolution, 256
    channels); backend is 6 dilated (rate 2) convolutions followed by a 1×1
    output head. Pure ``nn`` modules — no torchvision dependency, so the row
    smokes on CPU. Randomly initialised (like the timm comparator rows);
    comparability notes record the no-pretraining caveat.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            _vgg_block(in_channels, 64, 2),
            _vgg_block(64, 128, 2),
            _vgg_block(128, 256, 3),
        )
        backend_cfg = [(256, 512), (512, 512), (512, 512), (512, 256), (256, 128), (128, 64)]
        backend: list[nn.Module] = []
        for in_ch, out_ch in backend_cfg:
            backend += [nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2), nn.ReLU(inplace=True)]
        self.backend = nn.Sequential(*backend)
        self.output_layer = nn.Conv2d(64, 1, 1)
        self.to(current_device())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x).squeeze(1)  # [B, H/8, W/8]


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
        return ["shanghai_crowd", "shanghai_crowd_tiny", "ucf_qnrf"]

    def methods(self) -> list[str]:
        return ["UNet+MSE", "UNet+Huber", "UNet+GaussianNLL", "CSRNet", "sklearn_RF"]

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn_"):
            return "scikit-learn"
        if method == "CSRNet":
            return "csrnet-paper"
        return "torchregress"

    def library_version(self, method: str) -> str:
        if method == "CSRNet":
            return "Li-Zhang-Chen-2016-native-unpretrained"
        return super().library_version(method)

    def method_notes(self, method: str) -> str:
        return {
            "UNet+MSE": "UNet density map + MSE (point baseline)",
            "UNet+Huber": "UNet density map + PseudoHuber (robust to congested outliers)",
            "UNet+GaussianNLL": "UNet 2ch density mean+logvar per pixel",
            "CSRNet": "native CSRNet reimplementation (VGG-16 frontend + dilated backend, unpretrained) at 1/8 resolution; counts via map sums",
            "sklearn_RF": "RF count regressor → uniform maps (count metrics honest; no spatial model)",
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
        if method == "sklearn_RF":
            from sklearn.ensemble import RandomForestRegressor

            Xf = np.asarray(splits["X_train"]).reshape(len(splits["X_train"]), -1)
            # Train on counts (density sums); predict uniform maps below.
            ytr = np.asarray(splits["y_train"]).reshape(len(splits["X_train"]), -1).sum(axis=1)
            est = RandomForestRegressor(
                n_estimators=50, max_depth=12, random_state=config.seed, n_jobs=-1
            )
            est.fit(Xf, ytr)
            return {"type": "sklearn", "estimator": est}
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
        if method == "CSRNet":
            import torch.utils.data as _data

            m = _CSRNet(in_channels=in_channels)
            # CSRNet predicts at 1/8 resolution: sum-pool the GT density
            # maps (count-preserving) for the MSE target.
            y_small = (
                torch.nn.functional.avg_pool2d(
                    torch.from_numpy(np.asarray(splits["y_train"])).float().unsqueeze(1),
                    kernel_size=8,
                ).squeeze(1)
                * 64.0
            )
            loader = _data.DataLoader(
                _data.TensorDataset(
                    torch.from_numpy(np.asarray(splits["X_train"])).float(), y_small
                ),
                batch_size=min(config.batch_size, 8),
                shuffle=True,
            )
            _train_image_model(m, nn.MSELoss(), loader, config)
            _ = current_device
            return {"type": "csrnet", "model": m}
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
        if t == "sklearn":
            X_list = list(np.asarray(X_test, dtype=object))
            Xf = np.stack([np.asarray(x).reshape(-1) for x in X_list], axis=0).astype(np.float64)
            counts = np.clip(
                np.asarray(model_state["estimator"].predict(Xf), dtype=np.float64).reshape(-1),
                0,
                None,
            )
            # Uniform maps sized per image (variable resolutions possible).
            dens_list = []
            for c, x in zip(counts, X_list):
                xa = np.asarray(x)
                hw = int(np.prod(xa.shape[-2:])) if xa.ndim >= 3 else 1
                dens_list.append(np.full(hw, c / max(hw, 1)))
            # Pad to rectangular for the artifact contract.
            max_hw = max(len(d) for d in dens_list)
            dens = np.stack([np.pad(d, (0, max_hw - len(d))) for d in dens_list], axis=0)
            return {"y_pred": dens.reshape(-1), "_density": dens}
        model = model_state["model"]
        device = next(model.parameters()).device
        loader = _image_to_dataloader(
            X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False
        )
        if t == "csrnet":
            # Native 1/8-resolution maps; nearest-upsample and renormalise
            # by 64 so flattened densities keep count-preserving sums.
            preds = []
            with torch.no_grad():
                for (xb,) in loader:
                    small = model(xb.to(device))
                    full = (
                        torch.nn.functional.interpolate(
                            small.unsqueeze(1), scale_factor=8, mode="nearest"
                        ).squeeze(1)
                        / 64.0
                    )
                    preds.append(full.cpu().numpy())
            dens = np.concatenate(preds, axis=0).reshape(len(np.asarray(X_test)), -1)
            dens = np.clip(dens, 0, None)
            return {"y_pred": dens.reshape(-1), "_density": dens}
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
