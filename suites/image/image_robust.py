"""Image robust suite — robust + imbalanced losses on CNN + boosting.

Mirrors ``tabular.robust_regression`` on images: point predictors with
robust (Huber/Barron/Cauchy) and imbalanced (Density/LDS) losses, plus
conditional LightGBM/XGBoost comparators on flattened pixels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn as nn

from harness.core import set_harness_seed
from harness.runners import HarnessConfig

from ._base import ImageHarnessRunner, _image_to_dataloader, _train_image_model

try:
    import lightgbm  # noqa: F401

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    import xgboost  # noqa: F401

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


class ImageRobustSuite(ImageHarnessRunner):
    """Robust regression on images."""

    @property
    def name(self) -> str:
        return "image_robust"

    @property
    def description(self) -> str:
        return "Robust/imbalanced losses on CNN + LightGBM/XGBoost baselines"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "synthetic_image_regression",
            "imdb_wiki",
            "utkface_age",
            "utkface_age_tiny",
            "ixi_brainage",
            "ixi_brainage_tiny",
            "udacity_steering",
            "udacity_steering_tiny",
            "boneage",
        ]

    def methods(self) -> list[str]:
        m = [
            "MSE",
            "Huber",
            "Barron",
            "Cauchy",
            "LogCosh",
            "DensityWeighted",
            "LDS",
            "BalancedMSE",
            "FocalR",
            "FDS",
        ]
        if _LGBM_AVAILABLE:
            m.append("LightGBM")
        if _XGB_AVAILABLE:
            m.append("XGBoost")
        return m

    def library_name(self, method: str) -> str:
        if method == "LightGBM":
            return "lightgbm"
        if method == "XGBoost":
            return "xgboost"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "MSE": "CNN + MSE (non-robust baseline)",
            "Huber": "CNN + PseudoHuber (delta=1.0)",
            "Barron": "CNN + Barron adaptive robust loss",
            "Cauchy": "CNN + Cauchy robust loss",
            "LogCosh": "CNN + LogCosh robust loss",
            "DensityWeighted": "CNN + MSE with density-based reweighting",
            "LDS": "CNN + label-distribution-smoothing reweighting",
            "BalancedMSE": "CNN + binned inverse-frequency MSE (fit on train targets)",
            "FocalR": "CNN + focal-R down-weighted easy samples (mse base)",
            "FDS": "CNN features → FDS smoothing → Linear head (train-time calibration)",
            "LightGBM": "LightGBM on flattened pixels — model-family gap vs CNN",
            "XGBoost": "XGBoost on flattened pixels — model-family gap vs CNN",
        }.get(method, "")

    def fit(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        set_harness_seed(config.seed)
        if method in ("LightGBM", "XGBoost"):
            Xf = np.asarray(splits["X_train"]).reshape(len(splits["X_train"]), -1)
            y = np.asarray(splits["y_train"]).reshape(-1)
            if method == "LightGBM":
                import lightgbm as lgb

                ds = lgb.Dataset(Xf, label=y)
                params = {"objective": "regression", "verbosity": -1, "seed": config.seed}
                model = lgb.train(params, ds, num_boost_round=50)
            else:
                import xgboost as xgb

                model = xgb.XGBRegressor(
                    n_estimators=50, max_depth=4, random_state=config.seed, verbosity=0
                )
                model.fit(Xf, y)
            return {"type": "boosting", "model": model, "method": method}
        in_channels, _, _ = self._image_shape(splits)
        if method == "FDS":
            return self._fit_fds(splits, in_channels, config)
        loader = self._make_train_loader(splits, batch_size=config.batch_size)
        # Imbalanced losses need the train-target density first (tabular
        # stress_matrix precedent: fit_density()/fit() before training).
        # BalancedMSE builds its loss after bin edges are known (no early
        # _loss_for call — it requires bin_edges).
        loss: nn.Module | None = None
        if method != "BalancedMSE":
            loss = self._loss_for(method)
        if method in ("DensityWeighted", "LDS", "BalancedMSE"):
            import torch

            from harness.devices import current_device

            y_tr = (
                torch.from_numpy(np.asarray(splits["y_train"]).reshape(-1))
                .float()
                .to(current_device())
            )
            if method == "DensityWeighted":
                loss.fit_density(y_tr.reshape(-1))
            elif method == "LDS":
                loss.fit(y_tr.reshape(-1))
            else:  # BalancedMSE: quantile bin edges from train targets.
                q = torch.quantile(
                    y_tr.reshape(-1).float(),
                    torch.linspace(0, 1, 16, device=y_tr.device),
                )
                for i in range(1, len(q)):
                    q[i] = torch.maximum(q[i], q[i - 1] + 1e-6)
                loss = self._loss_for(method, bin_edges=q)
                loss.fit(y_tr.reshape(-1))
        model = self._build_cnn(in_channels=in_channels, out_dim=1)
        _train_image_model(model, loss, loader, config)
        model.eval()
        return {"type": "point", "model": model}

    def _fit_fds(self, splits: Any, in_channels: int, config: HarnessConfig) -> Any:
        """Split-head CNN → FDS → Linear training for FeatureDistributionSmoother."""
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchregress.losses import FeatureDistributionSmoother

        from harness.devices import current_device
        from suites.image._base import _image_to_dataloader, _ImageCNN

        device = current_device()
        feat_dim = 64  # _ImageCNN(base_filters=16) head dim (validate guard)
        backbone = _ImageCNN(in_channels=in_channels, out_dim=feat_dim).to(device)
        # Replace the Linear head with identity: we need raw features.
        backbone.head = nn.Identity().to(device)
        fds = FeatureDistributionSmoother(feature_dim=feat_dim).to(device)
        head = nn.Linear(feat_dim, 1).to(device)
        y_tr = np.asarray(splits["y_train"]).reshape(-1)
        fds.fit(torch.from_numpy(y_tr).float().to(device))
        loader = _image_to_dataloader(
            np.asarray(splits["X_train"]),
            y_tr.astype(np.float32),
            batch_size=config.batch_size,
            shuffle=True,
        )
        opt = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=config.lr)
        criterion = nn.MSELoss()
        backbone.train()
        head.train()
        for epoch in range(config.epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                feats = backbone(Xb)
                fds.update_running_stats(feats.detach(), yb.reshape(-1), epoch)
                out = head(fds(feats, yb.reshape(-1), epoch)).reshape(-1)
                loss = criterion(out, yb.reshape(-1))
                loss.backward()
                opt.step()
            fds.update_last_epoch_stats(epoch)
        backbone.eval()
        head.eval()
        return {"type": "fds", "backbone": backbone, "head": head}

    def _loss_for(self, method: str, bin_edges: Any = None) -> nn.Module:
        if method == "MSE":
            return nn.MSELoss()
        if method == "Huber":
            from torchregress.losses import PseudoHuberLoss

            return PseudoHuberLoss()
        if method == "Barron":
            from torchregress.losses import BarronLoss

            return BarronLoss()
        if method == "Cauchy":
            from torchregress.losses import CauchyLoss

            return CauchyLoss()
        if method == "LogCosh":
            from torchregress.losses import LogCoshLoss

            return LogCoshLoss()
        if method == "DensityWeighted":
            from torchregress.losses import DensityWeightedLoss

            return DensityWeightedLoss()
        if method == "LDS":
            from torchregress.losses import LDSLoss

            return LDSLoss()
        if method == "BalancedMSE":
            from torchregress.losses import BalancedMSELoss

            if bin_edges is None:
                raise ValueError("BalancedMSE needs bin_edges from fit()")
            return BalancedMSELoss(bin_edges)
        if method == "FocalR":
            from torchregress.losses import FocalRLoss

            return FocalRLoss()
        raise ValueError(f"Unknown method: {method}")

    def predict(self, method: str, state: Any, X_test: np.ndarray, config: HarnessConfig) -> Any:
        if state["type"] == "boosting":
            Xf = np.asarray(X_test).reshape(len(X_test), -1)
            m = state["model"]
            pred = m.predict(Xf) if state["method"] == "LightGBM" else m.predict(Xf)
            return {"y_pred": np.asarray(pred).reshape(-1)}
        if state["type"] == "fds":
            import torch

            backbone, head = state["backbone"], state["head"]
            device = next(backbone.parameters()).device
            loader = _image_to_dataloader(X_test, y=None, batch_size=64, shuffle=False)
            outs = []
            with torch.no_grad():
                for (xb,) in loader:
                    outs.append(head(backbone(xb.to(device))).cpu().numpy().reshape(-1))
            return {"y_pred": np.concatenate(outs)}
        import torch

        model = state["model"]
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=64, shuffle=False)
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                outs.append(model(xb.to(device)).cpu().numpy().reshape(-1))
        return {"y_pred": np.concatenate(outs)}
