"""Sklearn-style wrappers for image regression (CNN/UNet backbones).

Mirrors :mod:`harness.wrappers` (tabular MLP) but accepts ``(N, C, H, W)``
image arrays. Each wrapper exposes ``fit(X, y) / predict(X) /
predict_interval(X, confidence=0.9) / sample(X, n_samples)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import set_harness_seed
from harness.runners import HarnessConfig


def _coerce_image(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Image X must be 4D (N, C, H, W), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Image X contains NaN or infinity.")
    return arr


def _coerce_y(y: Any, n: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == n:
        return arr
    flat = arr.reshape(n, -1)
    if flat.shape[0] != n:
        raise ValueError(f"y length {len(arr)} != X length {n}")
    return flat.astype(np.float32)


class _ImageRegressorBase:
    """Shared fit/predict protocol for image wrappers."""

    _model: Any = None

    def __init__(
        self,
        *,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        backbone: str = "cnn",
        in_channels: int = 1,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.backbone = backbone
        self.in_channels = in_channels
        self._fitted = False
        self._inference_seed = 0

    @property
    def name(self) -> str:
        return type(self).__name__

    def _build_backbone(self, out_dim: int) -> nn.Module:
        from suites.image._base import _ImageCNN

        channels = int(self.in_channels)
        return _ImageCNN(in_channels=channels, out_dim=out_dim)

    def _fit_impl(self, X_t: torch.Tensor, y_t: torch.Tensor, cfg: HarnessConfig) -> None:
        raise NotImplementedError

    def fit(self, X: Any, y: Any) -> "_ImageRegressorBase":
        X_arr = _coerce_image(X)
        y_arr = _coerce_y(y, len(X_arr))
        self.in_channels = int(X_arr.shape[1])
        set_harness_seed(self.seed)
        from suites.image._base import _image_to_dataloader

        cfg = HarnessConfig(
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, seed=self.seed
        )
        self._inference_seed = int(self.seed)
        X_t = torch.from_numpy(X_arr).float()
        y_t = torch.from_numpy(np.asarray(y_arr, dtype=np.float32)).float()
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)
        self._loader = _image_to_dataloader(X_arr, np.asarray(y_arr), cfg.batch_size, True)
        # Move handled inside suite trainers via current_device; keep CPU refs here.
        self._fit_impl(X_t, y_t, cfg)
        self._fitted = True
        return self

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def predict(self, X: Any) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{self.name} not fitted; call .fit(X, y) first.")
        from harness.wrappers import _diagnostics_to_mean

        return _diagnostics_to_mean(self._predict_diagnostics(_coerce_image(X)))

    def predict_interval(self, X: Any, *, confidence: float = 0.9) -> tuple[np.ndarray, ...]:
        if not self._fitted:
            raise RuntimeError(f"{self.name} not fitted; call .fit(X, y) first.")
        from harness.wrappers import _diagnostics_to_interval

        return _diagnostics_to_interval(
            self._predict_diagnostics(_coerce_image(X)), confidence=confidence
        )

    def sample(self, X: Any, *, n_samples: int = 100) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{self.name} not fitted; call .fit(X, y) first.")
        from harness.wrappers import _diagnostics_to_samples

        return _diagnostics_to_samples(
            self._predict_diagnostics(_coerce_image(X)), n_samples=n_samples
        )


class ImageGaussianNLLRegressor(_ImageRegressorBase):
    """CNN + diagonal Gaussian NLL for images."""

    def _fit_impl(self, X_t: torch.Tensor, y_t: torch.Tensor, cfg: HarnessConfig) -> None:
        from suites.image._base import _train_image_model
        from torchregress.losses import GaussianNLLLoss

        self._model = self._build_backbone(out_dim=2)
        _train_image_model(self._model, GaussianNLLLoss(), self._loader, cfg)
        self._model.eval()

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        from suites.image._base import _image_to_dataloader

        device = next(self._model.parameters()).device
        loader = _image_to_dataloader(X, y=None, batch_size=32, shuffle=False)
        means, stds = [], []
        with torch.no_grad():
            for (xb,) in loader:
                out = self._model(xb.to(device))
                means.append(out[:, 0].cpu().numpy())
                stds.append(np.sqrt(np.exp(out[:, 1]).clamp_min(1e-8).cpu().numpy()))
        return {"mean": np.concatenate(means), "std": np.concatenate(stds)}


class ImageMDNRegressor(_ImageRegressorBase):
    """CNN + mixture density network for images."""

    def __init__(self, *, n_components: int = 3, **kw: Any) -> None:
        super().__init__(**kw)
        self.n_components = n_components
        self._loss_fn: Any = None

    def _fit_impl(self, X_t: torch.Tensor, y_t: torch.Tensor, cfg: HarnessConfig) -> None:
        from suites.image._base import _train_image_model
        from torchregress.losses import MixtureDensityLoss

        self._loss_fn = MixtureDensityLoss(n_components=self.n_components, n_features=1)
        self._model = self._build_backbone(out_dim=self._loss_fn.expected_output_size)
        _train_image_model(self._model, self._loss_fn, self._loader, cfg)
        self._model.eval()

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        from suites.image._base import _image_to_dataloader

        device = next(self._model.parameters()).device
        loader = _image_to_dataloader(X, y=None, batch_size=32, shuffle=False)
        pools = []
        with torch.no_grad():
            for (xb,) in loader:
                out = self._model(xb.to(device))
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self._inference_seed)
                    s = self._loss_fn.sample(out, n_samples=256).squeeze(-1)
                s_np = s.cpu().numpy()
                if s_np.ndim == 2 and s_np.shape[0] == 256:
                    s_np = s_np.T
                pools.append(s_np)
        samples = np.concatenate(pools, axis=0)
        return {"samples": samples}


class ImageFlowRegressor(_ImageRegressorBase):
    """CNN context + zuko NSF flow for images (scalar targets)."""

    def _fit_impl(self, X_t: torch.Tensor, y_t: torch.Tensor, cfg: HarnessConfig) -> None:
        from suites.image._base import _train_image_model
        from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

        ctx = 64
        self._flow = create_flow_model(
            n_features=1,
            context_dim=ctx,
            flow_type="nsf",
            n_transforms=3,
            hidden_features=[64, 64],
        )
        self._loss = NormalizingFlowLoss(flow=self._flow)
        self._model = self._build_backbone(out_dim=ctx)
        _train_image_model(
            self._model,
            self._loss,
            self._loader,
            cfg,
            loss_params=list(self._flow.parameters()),
        )
        self._model.eval()

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        from suites.image._base import _image_to_dataloader

        device = next(self._model.parameters()).device
        loader = _image_to_dataloader(X, y=None, batch_size=32, shuffle=False)
        pools = []
        with torch.no_grad():
            for (xb,) in loader:
                ctx = self._model(xb.to(device))
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self._inference_seed)
                    s = self._loss.sample(ctx, n_samples=256).squeeze(-1)
                s_np = s.cpu().numpy()
                if s_np.ndim == 2 and s_np.shape[0] == 256:
                    s_np = s_np.T
                pools.append(s_np)
        return {"samples": np.concatenate(pools, axis=0)}


class ImageEvidentialRegressor(_ImageRegressorBase):
    """CNN + NIG evidential regression for images."""

    def _fit_impl(self, X_t: torch.Tensor, y_t: torch.Tensor, cfg: HarnessConfig) -> None:
        import torch.nn.functional as F
        import torch.optim as optim
        from torchregress.losses import EvidentialRegressionLoss

        self._loss_fn = EvidentialRegressionLoss(coeff_nig=0.01)
        self._model = self._build_backbone(out_dim=4)
        opt = optim.Adam(list(self._model.parameters()), lr=cfg.lr)
        device = next(self._model.parameters()).device
        self._model.train()
        for _ in range(cfg.epochs):
            for batch in self._loader:
                xb, yb = batch[0].to(device), batch[1].to(device)
                opt.zero_grad(set_to_none=True)
                raw = self._model(xb)
                constrained = torch.cat(
                    [
                        raw[:, 0:1],
                        F.softplus(raw[:, 1:2]) + 0.01,
                        F.softplus(raw[:, 2:3]) + 1.01,
                        F.softplus(raw[:, 3:4]) + 0.01,
                    ],
                    dim=1,
                )
                loss = self._loss_fn(constrained, yb)
                loss.backward()
                opt.step()
        self._model.eval()

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        import torch.nn.functional as F
        from suites.image._base import _image_to_dataloader

        device = next(self._model.parameters()).device
        loader = _image_to_dataloader(X, y=None, batch_size=32, shuffle=False)
        g, nu, al, be = [], [], [], []
        with torch.no_grad():
            for (xb,) in loader:
                raw = self._model(xb.to(device))
                g.append(raw[:, 0].cpu().numpy())
                nu.append((F.softplus(raw[:, 1:2]) + 0.01).cpu().numpy().reshape(-1))
                al.append((F.softplus(raw[:, 2:3]) + 1.01).cpu().numpy().reshape(-1))
                be.append((F.softplus(raw[:, 3:4]) + 0.01).cpu().numpy().reshape(-1))
        return {
            "gamma": np.concatenate(g).reshape(-1),
            "nu": np.concatenate(nu).reshape(-1),
            "alpha": np.concatenate(al).reshape(-1),
            "beta": np.concatenate(be).reshape(-1),
        }


class ImageQuantileRegressor(_ImageRegressorBase):
    """CNN + MultiQuantile head for images (median + intervals)."""

    def __init__(self, *, quantiles: list[float] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.quantiles = list(quantiles or [0.05, 0.5, 0.95])

    def _fit_impl(self, X_t: torch.Tensor, y_t: torch.Tensor, cfg: HarnessConfig) -> None:
        from suites.image._base import _train_image_model
        from torchregress.losses import MultiQuantileLoss

        self._loss_fn = MultiQuantileLoss(quantiles=self.quantiles)
        self._model = self._build_backbone(out_dim=len(self.quantiles))
        _train_image_model(self._model, self._loss_fn, self._loader, cfg)
        self._model.eval()

    def _predict_diagnostics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        from suites.image._base import _image_to_dataloader

        device = next(self._model.parameters()).device
        loader = _image_to_dataloader(X, y=None, batch_size=32, shuffle=False)
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                outs.append(self._model(xb.to(device)).cpu().numpy())
        arr = np.concatenate(outs, axis=0)
        # Convention: quantiles sorted ascending; median is middle column.
        mid = len(self.quantiles) // 2
        return {
            "y_pred": arr[:, mid].reshape(-1),
            "lower": arr[:, 0].reshape(-1),
            "upper": arr[:, -1].reshape(-1),
        }


__all__ = [
    "ImageGaussianNLLRegressor",
    "ImageMDNRegressor",
    "ImageFlowRegressor",
    "ImageEvidentialRegressor",
    "ImageQuantileRegressor",
]
