"""Keypoint / multitarget image regression suite.

Targets are ``(N, D)`` vectors: ``(x, y)`` coords for K joints
(``synthetic_image_keypoint``, D=2K) or yaw/pitch/roll (``biwi_head_pose``,
D=3). Mirrors ``multivariate_intervals`` (tabular) on CNN/ResNet backbones:
diagonal GaussianNLL, full-covariance MultivariateGaussian, conditional NF.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import set_harness_seed
from harness.runners import HarnessConfig

from ._base import (
    _RESNET_AVAILABLE,
    ImageHarnessRunner,
    _image_to_dataloader,
    _train_image_model,
)


def _cov_from_chol_params(L_params: torch.Tensor, D: int) -> torch.Tensor:
    N = L_params.shape[0]
    L = torch.zeros(N, D, D, device=L_params.device, dtype=L_params.dtype)
    idx = 0
    for i in range(D):
        L[:, i, i] = torch.nn.functional.softplus(L_params[:, idx]) + 1e-4
        idx += 1
        for j in range(i):
            L[:, i, j] = L_params[:, idx]
            idx += 1
    return L @ L.transpose(-1, -2)


class _CholeskyWrapper(nn.Module):
    """Adapt (mean+chol_params) head output to MultivariateGaussianLoss."""

    def __init__(self, D: int, mv_loss: nn.Module) -> None:
        super().__init__()
        self.D = D
        self.mv = mv_loss

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        D = self.D
        mean = y_pred[:, :D]
        cov = _cov_from_chol_params(y_pred[:, D:], D)
        return self.mv(mean, target, covariance_matrices=cov)


class _LowRankWrapper(nn.Module):
    """Adapt (mean+factor+diag) head output to LowRankGaussianLoss (rank r)."""

    def __init__(self, D: int, rank: int, lr_loss: nn.Module) -> None:
        super().__init__()
        self.D = D
        self.rank = rank
        self.lr = lr_loss

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        D, r = self.D, self.rank
        mean = y_pred[:, :D]
        factor = y_pred[:, D : D + D * r].reshape(-1, D, r)
        diag = F.softplus(y_pred[:, D + D * r :]) + 1e-4
        return self.lr(mean, target, cov_factor=factor, cov_diag=diag)


class KeypointRegressionSuite(ImageHarnessRunner):
    """Multitarget image regression (keypoints / head-pose)."""

    # Shared constants live in ._base; kept as aliases for back-compat.
    ALPHA = 0.1
    Z_SCORE = 1.645

    LOW_RANK = 1  # factor rank for LowRankGaussian (D + D*r + D head)

    @property
    def name(self) -> str:
        return "keypoint_regression"

    @property
    def description(self) -> str:
        return "Multitarget image regression: CNN/ResNet + GaussianNLL/MV-Gaussian/LowRank/NF"

    @property
    def supported_datasets(self) -> list[str]:
        return ["synthetic_image_keypoint", "biwi_head_pose", "biwi_head_pose_tiny"]

    def methods(self) -> list[str]:
        m = [
            "CNN+MSE",
            "CNN+GaussianNLL",
            "CNN+MultivariateGaussian",
            "CNN+LowRankGaussian",
            "CNN+NF",
            "sklearn_RF",
        ]
        if _RESNET_AVAILABLE:
            m.extend(
                [
                    "ResNet+GaussianNLL",
                    "ResNet+MultivariateGaussian",
                    "ResNet+LowRankGaussian",
                ]
            )
        return m

    def library_name(self, method: str) -> str:
        return "scikit-learn" if method.startswith("sklearn_") else "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "CNN+MSE": "CNN + MSE on (N,D) targets (point baseline)",
            "CNN+GaussianNLL": "CNN + diagonal Gaussian NLL (2D outputs per dim)",
            "CNN+MultivariateGaussian": "CNN + full-covariance Gaussian via Cholesky params",
            "CNN+LowRankGaussian": "CNN + low-rank (r=1) + diag Gaussian (3D head, scalable MV)",
            "CNN+NF": "CNN context + zuko NSF flow on D-dim targets",
            "ResNet+GaussianNLL": "ResNet18 + diagonal Gaussian NLL",
            "ResNet+MultivariateGaussian": "ResNet18 + full-covariance Gaussian",
            "ResNet+LowRankGaussian": "ResNet18 + low-rank (r=1) + diag Gaussian",
            "sklearn_RF": "RandomForest on flattened pixels, multi-output (point only)",
        }.get(method, "")

    def fit(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        set_harness_seed(config.seed)
        in_channels, _, _ = self._image_shape(splits)
        y_train = np.asarray(splits["y_train"])
        D = int(y_train.shape[1]) if y_train.ndim == 2 else 1
        loader = self._make_train_loader(splits, batch_size=config.batch_size)
        if method.startswith("sklearn_"):
            from sklearn.ensemble import RandomForestRegressor

            Xf = np.asarray(splits["X_train"]).reshape(len(splits["X_train"]), -1)
            est = RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=config.seed, n_jobs=-1
            )
            est.fit(Xf, y_train)
            return {"type": "sklearn", "estimator": est, "D": D}
        backbone = "resnet" if method.startswith("ResNet+") else "cnn"
        head = method.split("+", 1)[1]
        return self._fit_head(backbone, head, in_channels, D, loader, config)

    def _backbone(self, kind: str, in_channels: int, out_dim: int) -> nn.Module:
        if kind == "resnet":
            from ._base import _make_resnet_backbone

            m = _make_resnet_backbone(in_channels=in_channels, out_dim=out_dim)
            if m is None:
                raise ImportError("torchvision missing")
            return m
        return self._build_cnn(in_channels=in_channels, out_dim=out_dim)

    def _fit_head(
        self,
        backbone: str,
        head: str,
        in_channels: int,
        D: int,
        loader: Any,
        config: HarnessConfig,
    ) -> Any:
        if head == "MSE":
            m = self._backbone(backbone, in_channels, D)
            _train_image_model(m, nn.MSELoss(), loader, config)
            return {"type": "point", "model": m, "D": D}
        if head == "GaussianNLL":
            from torchregress.losses import GaussianNLLLoss

            m = self._backbone(backbone, in_channels, 2 * D)
            _train_image_model(m, GaussianNLLLoss(), loader, config)
            m.eval()
            return {"type": "gaussian_nll", "model": m, "D": D}
        if head == "MultivariateGaussian":
            from torchregress.losses import MultivariateGaussianLoss

            n_params = D + D * (D + 1) // 2
            m = self._backbone(backbone, in_channels, n_params)
            mv = MultivariateGaussianLoss(n_features=D)
            _train_image_model(m, _CholeskyWrapper(D, mv), loader, config)
            m.eval()
            return {"type": "mv_gaussian", "model": m, "D": D}
        if head == "LowRankGaussian":
            from torchregress.losses import LowRankGaussianLoss

            r = self.LOW_RANK
            m = self._backbone(backbone, in_channels, D + D * r + D)
            lr = LowRankGaussianLoss()
            _train_image_model(m, _LowRankWrapper(D, r, lr), loader, config)
            m.eval()
            return {"type": "lowrank_gaussian", "model": m, "D": D}
        if head == "NF":
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            ctx = 64
            m = self._backbone(backbone, in_channels, ctx)
            flow = create_flow_model(
                n_features=D,
                context_dim=ctx,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            loss = NormalizingFlowLoss(flow=flow)
            _train_image_model(m, loss, loader, config, loss_params=list(flow.parameters()))
            m.eval()
            return {"type": "nflow", "model": m, "flow": loss, "D": D}
        raise ValueError(f"Unknown head: {head}")

    def predict(
        self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig
    ) -> Any:
        # Contract: ALL D dims flattened to (N*D,) so predictions align with
        # the flattened target in evaluate().
        t = model_state["type"]
        D = int(model_state.get("D", 1))
        n_test = len(np.asarray(X_test))
        if t == "sklearn":
            Xf = np.asarray(X_test).reshape(n_test, -1)
            pred = model_state["estimator"].predict(Xf)
            return {"y_pred": np.asarray(pred).reshape(n_test, -1).reshape(-1)}
        model = model_state["model"]
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)
        if t == "point":
            outs = []
            with torch.no_grad():
                for (xb,) in loader:
                    outs.append(model(xb.to(device)).cpu().numpy())
            arr = np.concatenate(outs, axis=0).reshape(n_test, -1)
            return {"y_pred": arr.reshape(-1)}
        if t in ("gaussian_nll", "mv_gaussian", "lowrank_gaussian"):
            means, stds = [], []
            with torch.no_grad():
                for (xb,) in loader:
                    out = model(xb.to(device))
                    if t == "gaussian_nll":
                        B = out.shape[0]
                        o = out.reshape(B, 2, D)
                        means.append(o[:, 0, :].cpu().numpy())
                        stds.append(
                            torch.exp(o[:, 1, :]).clamp_min(1e-8).sqrt().detach().cpu().numpy()
                        )
                    elif t == "mv_gaussian":
                        mean = out[:, :D].cpu().numpy()
                        cov = _cov_from_chol_params(out[:, D:], D).cpu().numpy()
                        var = np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), 1e-8)
                        means.append(mean)
                        stds.append(np.sqrt(var))
                    else:  # lowrank_gaussian: marginal var = sum(factor^2) + diag
                        import torch.nn.functional as F

                        r = self.LOW_RANK
                        mean = out[:, :D].cpu().numpy()
                        factor = out[:, D : D + D * r].reshape(-1, D, r).cpu().numpy()
                        diag = F.softplus(out[:, D + D * r :]).clamp_min(1e-8).cpu().numpy()
                        var = np.maximum((factor**2).sum(-1) + diag, 1e-8)
                        means.append(mean)
                        stds.append(np.sqrt(var))
            mean = np.concatenate(means, axis=0)
            std = np.concatenate(stds, axis=0)
            # Flatten dims for unified metrics; report per-dim mean intervals.
            flat_mean, flat_std = mean.reshape(-1), std.reshape(-1)
            return {
                "lower": flat_mean - self.Z_SCORE * flat_std,
                "upper": flat_mean + self.Z_SCORE * flat_std,
                "y_pred": flat_mean,
                "mean": flat_mean,
                "std": flat_std,
            }
        if t == "nflow":
            flow = model_state["flow"]
            pools = []
            with torch.no_grad():
                for (xb,) in loader:
                    ctx = model(xb.to(device))
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(config.seed)
                        s = flow.sample(ctx, n_samples=128)
                    pools.append(s.cpu().numpy())
            samples = np.concatenate(pools, axis=0)  # (N, S, D)
            flat_samples = samples.transpose(0, 2, 1).reshape(n_test * D, -1)
            flat_mean = flat_samples.mean(axis=1)
            return {
                "lower": np.quantile(flat_samples, 0.05, axis=1),
                "upper": np.quantile(flat_samples, 0.95, axis=1),
                "y_pred": flat_mean,
                "samples": flat_samples,
            }
        raise ValueError(f"Unknown model type: {t}")

    def evaluate(
        self, method: str, predictions: Any, y_test: np.ndarray, config: HarnessConfig
    ) -> Any:
        y_flat = np.asarray(y_test).reshape(-1)
        preds = dict(predictions)
        return super().evaluate(method, preds, y_flat, config)
