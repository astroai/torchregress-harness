"""Depth estimation suite — pixel-level dense regression.

Trains and compares methods for per-pixel regression on image datasets
where the target is a dense map of the same spatial dimensions as the input
(e.g., depth estimation from RGB images).

Methods
-------
* ``UNet+GaussianNLL`` — UNet backbone + per-pixel Gaussian NLL
* ``UNet+Evidential`` — UNet backbone + per-pixel NIG evidential regression
* ``UNet+MSE`` — UNet backbone + MSE loss (point prediction baseline)
* ``sklearn_RF`` — RandomForestRegressor on flattened pixels (baseline)

Evaluation
----------
Predictions are flattened to (N*H*W,) so existing point + distribution
metrics (RMSE, NLL, CRPS, PIT, ECE) work per-pixel out-of-the-box.
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

# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class DepthEstimationSuite(ImageHarnessRunner):
    """Pixel-level dense regression suite.

    Compares UNet-based torchregress methods (MSE, GaussianNLL, Evidential)
    against sklearn baselines on flattened pixels for per-pixel depth/map
    regression.
    """

    Z_SCORE = 1.645  # 90% prediction interval
    DENSE_BATCH_SIZE = 16  # smaller batches for 64×64 dense prediction (more memory)

    @property
    def name(self) -> str:
        return "depth_estimation"

    @property
    def description(self) -> str:
        return (
            "Pixel-level dense regression: UNet backbones with "
            "MSE, GaussianNLL, Evidential vs sklearn baselines"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "synthetic_image_depth",
            "nyu_depth_v2",
        ]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        return [
            "UNet+MSE",
            "UNet+GaussianNLL",
            "UNet+Evidential",
            "sklearn_RF",
        ]

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn_"):
            return "scikit-learn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        notes = {
            "UNet+MSE": "Lightweight UNet backbone + MSE loss (point prediction, no uncertainty)",
            "UNet+GaussianNLL": "UNet backbone + per-pixel Gaussian NLL (mean + log-var per pixel)",
            "UNet+Evidential": "UNet backbone + per-pixel NIG evidential regression",
            "sklearn_RF": "RandomForestRegressor on flattened pixels (100 trees, 10 depth) — point prediction only",
        }
        return notes.get(method, "")

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

        in_channels, img_h, img_w = self._image_shape(splits)
        train_loader = self._make_train_loader(splits, batch_size=config.batch_size)

        # --- UNet-based methods ---
        if method in ("UNet+MSE", "UNet+GaussianNLL", "UNet+Evidential"):
            return self._fit_unet_method(method, in_channels, train_loader, config)

        # --- sklearn baseline on flattened pixels ---
        if method.startswith("sklearn_"):
            return self._fit_sklearn_dense(method, splits, config)

        raise ValueError(f"Unknown method: {method}")

    def _fit_unet_method(
        self,
        method: str,
        in_channels: int,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit a UNet-based method for dense regression."""

        # --- UNet+MSE ---
        if method == "UNet+MSE":
            model = self._build_unet(in_channels=in_channels, out_channels=1, base_filters=32)
            _train_image_model(model, nn.MSELoss(), train_loader, config)
            return {"type": "mse", "model": model}

        # --- UNet+GaussianNLL ---
        elif method == "UNet+GaussianNLL":
            from torchregress.losses import GaussianNLLLoss

            # 2 output channels: mean + log-variance per pixel
            model = self._build_unet(in_channels=in_channels, out_channels=2, base_filters=32)
            _train_image_model(model, GaussianNLLLoss(), train_loader, config)
            return {"type": "gaussian_nll", "model": model}

        # --- UNet+Evidential ---
        elif method == "UNet+Evidential":
            from torchregress.losses import EvidentialRegressionLoss

            # 4 output channels: gamma + nu + alpha + beta per pixel
            model = self._build_unet(in_channels=in_channels, out_channels=4, base_filters=32)
            loss_fn = EvidentialRegressionLoss(coeff_nig=0.01)
            self._train_evidential_unet(model, loss_fn, train_loader, config)
            return {"type": "evidential", "model": model, "loss_fn": loss_fn}

        raise ValueError(f"Unknown UNet method: {method}")

    def _fit_sklearn_dense(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit sklearn baseline on flattened pixel data for dense regression.

        For dense targets (N, H, W), we flatten both X and y to (N*H*W, D)
        and (N*H*W,) so the estimator learns per-pixel prediction.
        """
        X_train = splits["X_train"]  # (N, C, H, W)
        y_train = splits["y_train"]  # (N, H, W)

        # Flatten to per-pixel samples
        N, C, H, W = X_train.shape
        X_flat = X_train.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        y_flat = y_train.reshape(-1)  # (N*H*W,)

        if method == "sklearn_RF":
            from sklearn.ensemble import RandomForestRegressor

            est = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=config.seed, n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown sklearn method: {method}")

        est.fit(X_flat, y_flat)
        return {"type": "sklearn_dense", "estimator": est}

    # ------------------------------------------------------------------
    # Custom evidential training for UNet
    # ------------------------------------------------------------------

    @staticmethod
    def _train_evidential_unet(
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
    ) -> None:
        """Train evidential UNet with softplus constraints per-pixel."""
        import torch.nn.functional as F
        import torch.optim as optim

        params = list(model.parameters())
        for p in loss_fn.parameters():
            if p.requires_grad:
                params.append(p)

        opt = optim.Adam(params, lr=config.lr)
        device = next(model.parameters()).device
        model.train()

        for _ in range(config.epochs):
            for batch in train_loader:
                X_batch = batch[0].to(device)
                y_batch = batch[1].to(device)  # (B, 1, H, W)

                opt.zero_grad(set_to_none=True)
                raw = model(X_batch)  # (B, 4, H, W)

                # Apply evidential constraints per-pixel
                gamma = raw[:, 0:1, :, :]
                nu = F.softplus(raw[:, 1:2, :, :]) + 0.01
                alpha = F.softplus(raw[:, 2:3, :, :]) + 1.01
                beta = F.softplus(raw[:, 3:4, :, :]) + 0.01
                constrained = torch.cat([gamma, nu, alpha, beta], dim=1)

                loss = loss_fn(constrained, y_batch)
                loss.backward()
                opt.step()

        model.eval()

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
        model_type = model_state["type"]

        # --- UNet+MSE ---
        if model_type == "mse":
            return self._predict_mse(model_state["model"], X_test)

        # --- UNet+GaussianNLL ---
        elif model_type == "gaussian_nll":
            return self._predict_gaussian_nll(model_state["model"], X_test)

        # --- UNet+Evidential ---
        elif model_type == "evidential":
            return self._predict_evidential(model_state["model"], X_test)

        # --- sklearn dense ---
        elif model_type == "sklearn_dense":
            return self._predict_sklearn_dense(model_state["estimator"], X_test)

        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # Prediction helpers — all flatten to (N*H*W,) for unified metrics
    # ------------------------------------------------------------------

    def _predict_mse(
        self,
        model: nn.Module,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """UNet+MSE: point prediction only (no uncertainty)."""
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False)

        preds = []
        with torch.no_grad():
            for (X_batch,) in loader:
                out = model(X_batch.to(device))  # (B, 1, H, W)
                preds.append(out.cpu().numpy())

        y_pred = np.concatenate(preds, axis=0)  # (N, 1, H, W)
        # Flatten to (N*H*W,) for existing metrics
        y_pred_flat = y_pred.reshape(-1)
        return {"y_pred": y_pred_flat}

    def _predict_gaussian_nll(
        self,
        model: nn.Module,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """UNet+GaussianNLL: per-pixel mean ± z*std, flattened."""
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False)

        means, stds = [], []
        with torch.no_grad():
            for (X_batch,) in loader:
                out = model(X_batch.to(device))  # (B, 2, H, W)
                mean = out[:, 0:1, :, :].cpu().numpy()
                var = torch.exp(out[:, 1:2, :, :]).clamp_min(1e-8).cpu().numpy()
                means.append(mean)
                stds.append(np.sqrt(var))

        mean = np.concatenate(means, axis=0).reshape(-1)
        std = np.concatenate(stds, axis=0).reshape(-1)
        z = self.Z_SCORE
        return {
            "lower": mean - z * std,
            "upper": mean + z * std,
            "y_pred": mean,
            "mean": mean,
            "std": std,
        }

    def _predict_evidential(
        self,
        model: nn.Module,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """UNet+Evidential: per-pixel Student-t intervals via NIG, flattened."""
        import torch.nn.functional as F

        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False)

        gammas, nus, alphas, betas = [], [], [], []
        with torch.no_grad():
            for (X_batch,) in loader:
                raw = model(X_batch.to(device))  # (B, 4, H, W)
                gammas.append(raw[:, 0, :, :].cpu().numpy())
                nus.append((F.softplus(raw[:, 1, :, :]) + 0.01).cpu().numpy())
                alphas.append((F.softplus(raw[:, 2, :, :]) + 1.01).cpu().numpy())
                betas.append((F.softplus(raw[:, 3, :, :]) + 0.01).cpu().numpy())

        gamma = np.concatenate(gammas, axis=0).reshape(-1)
        nu = np.concatenate(nus, axis=0).reshape(-1)
        alpha = np.concatenate(alphas, axis=0).reshape(-1)
        beta = np.concatenate(betas, axis=0).reshape(-1)

        # Student-t intervals per-pixel
        from scipy.stats import t as student_t

        df = 2.0 * alpha
        scale = np.sqrt(beta * (1.0 + nu) / (nu * alpha))
        lo = student_t.ppf(0.05, df=df, loc=gamma, scale=scale)
        hi = student_t.ppf(0.95, df=df, loc=gamma, scale=scale)

        return {
            "lower": lo,
            "upper": hi,
            "y_pred": gamma,
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
        }

    def _predict_sklearn_dense(
        self,
        estimator: Any,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """sklearn on flattened pixels for dense regression."""
        N, C, H, W = X_test.shape
        X_flat = X_test.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        y_pred = estimator.predict(X_flat).reshape(-1).astype(np.float32)
        return {"y_pred": y_pred}

    # ------------------------------------------------------------------
    # evaluate — flatten targets to match predictions
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        """Flatten dense targets then delegate to standard metrics."""
        # y_test is (N, H, W) for dense regression — flatten to (N*H*W,)
        y_flat = np.asarray(y_test).reshape(-1)
        return super().evaluate(method, predictions, y_flat, config)
