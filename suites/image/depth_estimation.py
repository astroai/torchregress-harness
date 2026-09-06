"""Depth estimation suite — pixel-level dense regression.

Trains and compares methods for per-pixel regression on image datasets
where the target is a dense map of the same spatial dimensions as the input
(e.g., depth estimation from RGB images).

Methods
-------
* ``UNet+GaussianNLL`` — UNet backbone + per-pixel Gaussian NLL
* ``UNet+Evidential`` — UNet backbone + per-pixel NIG evidential regression
* ``UNet+MSE`` — UNet backbone + MSE loss (point prediction baseline)
* ``UNet+MDN_K3`` — UNet + 3-component per-pixel MDN
* ``UNet+MultiQuantile`` — UNet + 3-quantile head per pixel
* ``UNet+StudentT`` — UNet + Student-t NLL per pixel (heavy-tailed)
* ``UNet+Huber`` — UNet + PseudoHuber robust loss per pixel
* ``UNet+DeepEnsemble_K3`` — 3 UNets + MSE, mean/var across members
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
    EVIDENTIAL_COEFF_NIG,
    EVIDENTIAL_EPS,
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

    ALPHA = 0.1  # nominal miscoverage for 90% intervals
    Z_SCORE = 1.645  # 90% Gaussian interval (scipy norm.ppf(0.95))
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
            "nyu_depth_v2_tiny",
            "nyu_depth_v2_fullres",
            "kitti_depth",
            "kitti_depth_tiny",
        ]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        return [
            "UNet+MSE",
            "UNet+GaussianNLL",
            "UNet+BetaNLL",
            "UNet+FaithfulGaussian",
            "UNet+CRPS",
            "UNet+Evidential",
            "UNet+MDN_K3",
            "UNet+MultiQuantile",
            "UNet+StudentT",
            "UNet+Huber",
            "UNet+DeepEnsemble_K3",
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
            "UNet+BetaNLL": "UNet + per-pixel BetaNLL(beta=0.5) anti-collapse variant",
            "UNet+FaithfulGaussian": "UNet + per-pixel Faithful decoupled Gaussian NLL",
            "UNet+CRPS": "UNet + per-pixel Gaussian CRPS (tail-robust proper score)",
            "UNet+Evidential": "UNet backbone + per-pixel NIG evidential regression",
            "UNet+MDN_K3": "UNet + 3-component per-pixel mixture density network",
            "UNet+MultiQuantile": "UNet + 3-quantile head per pixel (0.05/0.5/0.95)",
            "UNet+StudentT": "UNet + Student-t NLL per pixel (heavy-tailed)",
            "UNet+Huber": "UNet + PseudoHuber robust loss per pixel (point prediction)",
            "UNet+DeepEnsemble_K3": "3 independent UNets + MSE; mean/var across members",
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
        sparse = bool(splits.get("_sparse_depth")) and splits.get("mask_train") is not None
        if sparse:
            train_loader = self._make_masked_loader(splits, batch_size=config.batch_size)
        else:
            train_loader = self._make_train_loader(splits, batch_size=config.batch_size)

        # --- UNet-based methods ---
        if method.startswith("UNet+"):
            return self._fit_unet_method(method, in_channels, train_loader, config, masked=sparse)

        # --- sklearn baseline on flattened pixels ---
        if method.startswith("sklearn_"):
            return self._fit_sklearn_dense(method, splits, config)

        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _make_masked_loader(
        splits: dict[str, Any],
        batch_size: int = 32,
    ) -> torch.utils.data.DataLoader:
        """Training loader yielding (X, y, mask) for sparse depth."""
        import torch.utils.data as _data

        from harness.devices import current_device

        device = current_device()
        X_t = torch.from_numpy(np.asarray(splits["X_train"])).float().to(device)
        y_t = torch.from_numpy(np.asarray(splits["y_train"])).float().to(device).unsqueeze(1)
        m_t = torch.from_numpy(np.asarray(splits["mask_train"])).float().to(device).unsqueeze(1)
        return _data.DataLoader(
            _data.TensorDataset(X_t, y_t, m_t), batch_size=batch_size, shuffle=True
        )

    @staticmethod
    def _train_masked(
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
    ) -> None:
        """Masked dense training on valid pixels only.

        Losses with a ``mask`` kwarg (torchregress BaseLoss family) receive
        the (B,1,H,W) mask directly. Plain losses (e.g. ``nn.MSELoss``) fall
        back to flattened valid-pixel subsets.
        """
        import inspect as _inspect

        import torch.optim as optim

        accepts_mask = "mask" in _inspect.signature(loss_fn.forward).parameters
        opt = optim.Adam(
            list(model.parameters()) + [p for p in loss_fn.parameters() if p.requires_grad],
            lr=config.lr,
        )
        device = next(model.parameters()).device
        loss_fn.to(device)
        model.train()
        for _ in range(config.epochs):
            for batch in train_loader:
                Xb, yb, mb = (t.to(device) for t in batch)
                opt.zero_grad(set_to_none=True)
                out = model(Xb)
                if accepts_mask:
                    loss = loss_fn(out, yb, mask=mb)
                else:
                    valid = mb.reshape(-1) > 0
                    b, c, h, w = out.shape
                    loss = loss_fn(
                        out.permute(0, 2, 3, 1).reshape(b * h * w, c)[valid],
                        yb.permute(0, 2, 3, 1).reshape(b * h * w, 1)[valid],
                    )
                loss.backward()
                opt.step()
        model.eval()

    @staticmethod
    def _train_masked_pixelwise(
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
        evidential: bool = False,
    ) -> None:
        """Masked variant of _train_pixelwise_head: subset valid pixels.

        Flattens (B,C,H,W) → (N_valid, C) so scalar losses (MDN,
        MultiQuantile) and the layout-agnostic evidential tuple form work
        unchanged on sparse depth.
        """
        import torch.nn.functional as F
        import torch.optim as optim

        opt = optim.Adam(
            list(model.parameters()) + [p for p in loss_fn.parameters() if p.requires_grad],
            lr=config.lr,
        )
        device = next(model.parameters()).device
        loss_fn.to(device)
        model.train()
        for _ in range(config.epochs):
            for Xb, yb, mb in train_loader:
                Xb, yb, mb = Xb.to(device), yb.to(device), mb.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(Xb)  # (B, C, H, W)
                b, c, h, w = out.shape
                valid = mb.reshape(-1) > 0
                if int(valid.sum()) == 0:
                    continue
                if evidential:
                    gamma = out[:, 0:1, :, :].reshape(-1)[valid].unsqueeze(1)
                    nu = (
                        (F.softplus(out[:, 1:2, :, :]) + EVIDENTIAL_EPS)
                        .reshape(-1)[valid]
                        .unsqueeze(1)
                    )
                    alpha = (
                        (F.softplus(out[:, 2:3, :, :]) + 1.0 + EVIDENTIAL_EPS)
                        .reshape(-1)[valid]
                        .unsqueeze(1)
                    )
                    beta = (
                        (F.softplus(out[:, 3:4, :, :]) + EVIDENTIAL_EPS)
                        .reshape(-1)[valid]
                        .unsqueeze(1)
                    )
                    loss = loss_fn((gamma, nu, alpha, beta), yb.reshape(-1)[valid].unsqueeze(1))
                else:
                    out_flat = out.permute(0, 2, 3, 1).reshape(b * h * w, c)[valid]
                    y_flat = yb.permute(0, 2, 3, 1).reshape(b * h * w, 1)[valid]
                    loss = loss_fn(out_flat, y_flat)
                loss.backward()
                opt.step()
        model.eval()

    def _fit_unet_method(
        self,
        method: str,
        in_channels: int,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
        masked: bool = False,
    ) -> dict[str, Any]:
        """Fit a UNet-based method for dense regression."""

        # --- UNet+MSE ---
        if method == "UNet+MSE":
            model = self._build_unet(in_channels=in_channels, out_channels=1, base_filters=32)
            if masked:
                self._train_masked(model, nn.MSELoss(), train_loader, config)
            else:
                _train_image_model(model, nn.MSELoss(), train_loader, config)
            return {"type": "mse", "model": model, "sparse": masked}

        # --- UNet+GaussianNLL ---
        elif method == "UNet+GaussianNLL":
            from torchregress.losses import GaussianNLLLoss

            # 2 output channels: mean + log-variance per pixel.
            # split_dim=1: dense layout is (B, 2, H, W), not last-dim.
            model = self._build_unet(in_channels=in_channels, out_channels=2, base_filters=32)
            if masked:
                self._train_masked(model, GaussianNLLLoss(split_dim=1), train_loader, config)
            else:
                _train_image_model(model, GaussianNLLLoss(split_dim=1), train_loader, config)
            return {"type": "gaussian_nll", "model": model, "sparse": masked}

        # --- UNet+BetaNLL / FaithfulGaussian (same 2ch dense layout) ---
        elif method == "UNet+BetaNLL":
            from torchregress.losses import BetaNLLLoss

            model = self._build_unet(in_channels=in_channels, out_channels=2, base_filters=32)
            if masked:
                self._train_masked(model, BetaNLLLoss(beta=0.5, split_dim=1), train_loader, config)
            else:
                _train_image_model(model, BetaNLLLoss(beta=0.5, split_dim=1), train_loader, config)
            return {"type": "gaussian_nll", "model": model, "sparse": masked}
        elif method == "UNet+FaithfulGaussian":
            from torchregress.losses import FaithfulGaussianLoss

            model = self._build_unet(in_channels=in_channels, out_channels=2, base_filters=32)
            if masked:
                self._train_masked(model, FaithfulGaussianLoss(split_dim=1), train_loader, config)
            else:
                _train_image_model(model, FaithfulGaussianLoss(split_dim=1), train_loader, config)
            return {"type": "gaussian_nll", "model": model, "sparse": masked}

        # --- UNet+CRPS (same 2ch dense layout) ---
        elif method == "UNet+CRPS":
            from torchregress.losses import GaussianCRPSLoss

            model = self._build_unet(in_channels=in_channels, out_channels=2, base_filters=32)
            if masked:
                self._train_masked(model, GaussianCRPSLoss(split_dim=1), train_loader, config)
            else:
                _train_image_model(model, GaussianCRPSLoss(split_dim=1), train_loader, config)
            return {"type": "gaussian_nll", "model": model, "sparse": masked}

        # --- UNet+Evidential ---
        elif method == "UNet+Evidential":
            from torchregress.losses import EvidentialRegressionLoss

            # 4 output channels: gamma + nu + alpha + beta per pixel
            model = self._build_unet(in_channels=in_channels, out_channels=4, base_filters=32)
            loss_fn = EvidentialRegressionLoss(coeff_nig=EVIDENTIAL_COEFF_NIG)
            if masked:
                self._train_masked_pixelwise(model, loss_fn, train_loader, config, evidential=True)
            else:
                self._train_evidential_unet(model, loss_fn, train_loader, config)
            return {"type": "evidential", "model": model, "loss_fn": loss_fn, "sparse": masked}

        # --- UNet+MDN_K3 (per-pixel mixture over scalar depth) ---
        elif method == "UNet+MDN_K3":
            from torchregress.losses import MixtureDensityLoss

            loss_fn = MixtureDensityLoss(n_components=3, n_features=1)
            # MDN head emits (B, out_dim) per location via 1×1 conv path:
            # reuse UNet with out_channels=out_dim; loss broadcasts over HW.
            model = self._build_unet(
                in_channels=in_channels,
                out_channels=loss_fn.expected_output_size,
                base_filters=32,
            )
            self._train_mdn_unet(model, loss_fn, train_loader, config, masked=masked)
            return {"type": "mdn", "model": model, "loss_fn": loss_fn, "sparse": masked}

        # --- UNet+MultiQuantile (3 quantiles per pixel) ---
        elif method == "UNet+MultiQuantile":
            from torchregress.losses import MultiQuantileLoss

            quantiles = [0.05, 0.5, 0.95]
            model = self._build_unet(
                in_channels=in_channels, out_channels=len(quantiles), base_filters=32
            )
            self._train_quantile_unet(
                model,
                MultiQuantileLoss(quantiles=quantiles),
                train_loader,
                config,
                masked=masked,
            )
            return {
                "type": "multi_quantile",
                "model": model,
                "quantiles": quantiles,
                "sparse": masked,
            }

        # --- UNet+StudentT / Huber (single-channel point-ish heads) ---
        # NOTE: distinct types (not "mse") so metrics correctly treat these
        # as point-only and reports don't mislabel them as MSE baselines.
        elif method == "UNet+StudentT":
            from torchregress.losses import StudentTLoss

            model = self._build_unet(in_channels=in_channels, out_channels=1, base_filters=32)
            if masked:
                self._train_masked(model, StudentTLoss(), train_loader, config)
            else:
                _train_image_model(model, StudentTLoss(), train_loader, config)
            return {"type": "studentt", "model": model, "sparse": masked}
        elif method == "UNet+Huber":
            from torchregress.losses import PseudoHuberLoss

            model = self._build_unet(in_channels=in_channels, out_channels=1, base_filters=32)
            if masked:
                self._train_masked(model, PseudoHuberLoss(), train_loader, config)
            else:
                _train_image_model(model, PseudoHuberLoss(), train_loader, config)
            return {"type": "huber", "model": model, "sparse": masked}

        # --- UNet+DeepEnsemble_K3 ---
        elif method == "UNet+DeepEnsemble_K3":
            members = []
            for _ in range(3):
                m = self._build_unet(in_channels=in_channels, out_channels=1, base_filters=32)
                if masked:
                    self._train_masked(m, nn.MSELoss(), train_loader, config)
                else:
                    _train_image_model(m, nn.MSELoss(), train_loader, config)
                m.eval()
                members.append(m)
            return {"type": "deep_ensemble", "members": members, "sparse": masked}

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
        if splits.get("mask_train") is not None:
            valid = np.asarray(splits["mask_train"]).reshape(-1) > 0
            X_flat, y_flat = X_flat[valid], y_flat[valid]

        if method == "sklearn_RF":
            from sklearn.ensemble import RandomForestRegressor

            est = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=config.seed,
                n_jobs=-1,
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
        loss_fn.to(device)
        model.train()

        for _ in range(config.epochs):
            for batch in train_loader:
                X_batch = batch[0].to(device)
                y_batch = batch[1].to(device)  # (B, 1, H, W)

                opt.zero_grad(set_to_none=True)
                raw = model(X_batch)  # (B, 4, H, W)

                # Apply evidential constraints per-pixel. The loss accepts a
                # pre-split (gamma, nu, alpha, beta) tuple, which is layout
                # agnostic — required for dense (B, 1, H, W) targets.
                gamma = raw[:, 0:1, :, :]
                nu = F.softplus(raw[:, 1:2, :, :]) + EVIDENTIAL_EPS
                alpha = F.softplus(raw[:, 2:3, :, :]) + 1.0 + EVIDENTIAL_EPS
                beta = F.softplus(raw[:, 3:4, :, :]) + EVIDENTIAL_EPS

                loss = loss_fn((gamma, nu, alpha, beta), y_batch)
                loss.backward()
                opt.step()

        model.eval()

    @staticmethod
    def _train_pixelwise_head(
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
    ) -> None:
        """Train a per-pixel head whose loss expects (B', out_dim)/(B', 1).

        Flattens ``(B, C, H, W)`` → ``(B*H*W, C)`` so scalar losses
        (MDN, MultiQuantile) work on dense outputs without a custom kernel.
        """
        import torch.optim as optim

        opt = optim.Adam(list(model.parameters()), lr=config.lr)
        device = next(model.parameters()).device
        loss_fn.to(device)
        model.train()
        for _ in range(config.epochs):
            for batch in train_loader:
                Xb = batch[0].to(device)
                yb = batch[1].to(device)  # (B, 1, H, W)
                opt.zero_grad(set_to_none=True)
                out = model(Xb)  # (B, C, H, W)
                b, c, h, w = out.shape
                out_flat = out.permute(0, 2, 3, 1).reshape(b * h * w, c)
                y_flat = yb.permute(0, 2, 3, 1).reshape(b * h * w, 1)
                loss = loss_fn(out_flat, y_flat)
                loss.backward()
                opt.step()
        model.eval()

    def _train_mdn_unet(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
        masked: bool = False,
    ) -> None:
        if masked:
            self._train_masked_pixelwise(model, loss_fn, train_loader, config)
        else:
            self._train_pixelwise_head(model, loss_fn, train_loader, config)

    def _train_quantile_unet(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
        masked: bool = False,
    ) -> None:
        if masked:
            self._train_masked_pixelwise(model, loss_fn, train_loader, config)
        else:
            self._train_pixelwise_head(model, loss_fn, train_loader, config)

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
        sparse = bool(model_state.get("sparse"))

        def _tag(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            if sparse:
                out["_sparse"] = np.array([True])
            return out

        # --- UNet+MSE / StudentT / Huber (point-only heads) ---
        if model_type in ("mse", "studentt", "huber"):
            return _tag(self._predict_mse(model_state["model"], X_test))

        # --- UNet+GaussianNLL ---
        elif model_type == "gaussian_nll":
            return _tag(self._predict_gaussian_nll(model_state["model"], X_test))

        # --- UNet+Evidential ---
        elif model_type == "evidential":
            return _tag(self._predict_evidential(model_state["model"], X_test))

        # --- UNet+MDN_K3 ---
        elif model_type == "mdn":
            return _tag(
                self._predict_mdn(
                    model_state["model"],
                    model_state["loss_fn"],
                    X_test,
                    seed=config.seed,
                )
            )

        # --- UNet+MultiQuantile ---
        elif model_type == "multi_quantile":
            return _tag(
                self._predict_multi_quantile(model_state["model"], model_state["quantiles"], X_test)
            )

        # --- UNet+DeepEnsemble_K3 ---
        elif model_type == "deep_ensemble":
            return _tag(self._predict_deep_ensemble(model_state["members"], X_test))

        # --- sklearn dense ---
        elif model_type == "sklearn_dense":
            out = self._predict_sklearn_dense(model_state["estimator"], X_test)
            return _tag(out)

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
        loader = _image_to_dataloader(
            X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False
        )

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
        loader = _image_to_dataloader(
            X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False
        )

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
        loader = _image_to_dataloader(
            X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False
        )

        gammas, nus, alphas, betas = [], [], [], []
        with torch.no_grad():
            for (X_batch,) in loader:
                raw = model(X_batch.to(device))  # (B, 4, H, W)
                gammas.append(raw[:, 0, :, :].cpu().numpy())
                nus.append((F.softplus(raw[:, 1, :, :]) + EVIDENTIAL_EPS).cpu().numpy())
                alphas.append((F.softplus(raw[:, 2, :, :]) + 1.0 + EVIDENTIAL_EPS).cpu().numpy())
                betas.append((F.softplus(raw[:, 3, :, :]) + EVIDENTIAL_EPS).cpu().numpy())

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

    def _predict_mdn(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        X_test: np.ndarray,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        """UNet+MDN: per-pixel mixture mean + MC quantile intervals, flattened."""
        device = next(model.parameters()).device
        loader = _image_to_dataloader(
            X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False
        )
        means, lowers, uppers = [], [], []
        with torch.no_grad():
            for (Xb,) in loader:
                out = model(Xb.to(device))  # (B, C, H, W)
                b, c, h, w = out.shape
                flat = out.permute(0, 2, 3, 1).reshape(b * h * w, c)
                mean, _ = loss_fn.predict_mean_std(flat)
                mean_np = mean.squeeze(-1).cpu().numpy().reshape(b, h, w)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(seed))
                    s = loss_fn.sample(flat, n_samples=64).squeeze(-1).cpu().numpy()
                if s.ndim == 2 and s.shape[0] == 64:
                    s = s.T
                s = s.reshape(b, h, w, -1)
                means.append(mean_np.reshape(-1))
                lowers.append(np.quantile(s, 0.05, axis=-1).reshape(-1))
                uppers.append(np.quantile(s, 0.95, axis=-1).reshape(-1))
        mean = np.concatenate(means)
        return {
            "lower": np.concatenate(lowers),
            "upper": np.concatenate(uppers),
            "y_pred": mean,
            "mean": mean,
        }

    def _predict_multi_quantile(
        self, model: nn.Module, quantiles: list[float], X_test: np.ndarray
    ) -> dict[str, np.ndarray]:
        device = next(model.parameters()).device
        loader = _image_to_dataloader(
            X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False
        )
        los, meds, his = [], [], []
        with torch.no_grad():
            for (Xb,) in loader:
                out = model(Xb.to(device)).cpu().numpy()  # (B, 3, H, W)
                los.append(out[:, 0].reshape(-1))
                meds.append(out[:, len(quantiles) // 2].reshape(-1))
                his.append(out[:, -1].reshape(-1))
        return {
            "lower": np.concatenate(los),
            "upper": np.concatenate(his),
            "y_pred": np.concatenate(meds),
        }

    def _predict_deep_ensemble(
        self, members: list[nn.Module], X_test: np.ndarray
    ) -> dict[str, np.ndarray]:
        device = next(members[0].parameters()).device
        loader = _image_to_dataloader(
            X_test, y=None, batch_size=self.DENSE_BATCH_SIZE, shuffle=False
        )
        preds = []
        with torch.no_grad():
            for m in members:
                part = []
                for (Xb,) in loader:
                    part.append(m(Xb.to(device)).cpu().numpy())
                preds.append(np.concatenate(part, axis=0))
        stack = np.stack(preds, axis=0)  # (K, N, 1, H, W)
        mean = stack.mean(axis=0).reshape(-1)
        std = stack.std(axis=0).reshape(-1).clip(min=1e-6)
        z = self.Z_SCORE
        return {
            "lower": mean - z * std,
            "upper": mean + z * std,
            "y_pred": mean,
            "mean": mean,
            "std": std,
        }

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
        """Flatten dense targets then delegate to standard metrics.

        Sparse depth (KITTI): restrict to valid pixels (y > 0). Invalid
        pixels are exactly 0 in both the real loader and the synthetic
        proxy; dense datasets never take this branch (no ``_sparse`` tag).
        """
        # y_test is (N, H, W) for dense regression — flatten to (N*H*W,)
        y_flat = np.asarray(y_test).reshape(-1)
        preds = {k: v for k, v in predictions.items() if k != "_sparse"}
        valid_mask = None
        if "_sparse" in predictions:
            valid_mask = y_flat > 0
            if int(valid_mask.sum()) == 0:
                return {"Coverage": float("nan"), "Width": float("nan")}
            y_flat = y_flat[valid_mask]
            for k, v in list(preds.items()):
                arr = np.asarray(v)
                if arr.size == valid_mask.size:
                    preds[k] = arr.reshape(-1)[valid_mask]
        out = super().evaluate(method, preds, y_flat, config)
        if valid_mask is not None:
            out["ValidPixelFrac"] = float(np.mean(valid_mask))
        return out
