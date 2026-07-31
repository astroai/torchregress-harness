"""Shared image backbones and training utilities for image regression suites.

Provides:
- ``_ImageCNN`` — lightweight 4-layer CNN for small image regression
- ``_ImageResNet`` — torchvision ResNet18 backbone with custom head
- ``_UNet`` — simple UNet for pixel-level (dense) regression
- ``ImageHarnessRunner`` — ABC for image suites extending ``HarnessRunner``
- ``_train_image_model`` — Adam mini-batch training loop on DataLoaders
- ``_image_to_dataloader`` — convert numpy image arrays to PyTorch DataLoader
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from harness.runners import HarnessConfig, HarnessRunner

# ---------------------------------------------------------------------------
# CNN backbone — lightweight, reproducible, no pretrained weights
# ---------------------------------------------------------------------------


class _ImageCNN(nn.Module):
    """Lightweight 4-layer CNN for small image regression.

    Architecture: Conv→ReLU→MaxPool → Conv→ReLU→MaxPool →
    Conv→ReLU→MaxPool → Conv→ReLU→AdaptiveAvgPool → Flatten → Linear head.

    Designed for 32×32 or 64×64 input images.  ~30-50k parameters.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_dim: int = 1,
        base_filters: int = 16,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2
            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8
            nn.Conv2d(base_filters * 4, base_filters * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # → (B, C, 1, 1)
        )
        self.head = nn.Linear(base_filters * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x).flatten(1)
        return self.head(feats)


# ---------------------------------------------------------------------------
# ResNet backbone — torchvision ResNet18 with custom head
# ---------------------------------------------------------------------------

_RESNET_AVAILABLE: bool
try:
    from torchvision.models import ResNet18_Weights, resnet18

    _RESNET_AVAILABLE = True
except ImportError:
    _RESNET_AVAILABLE = False


def _make_resnet_backbone(
    in_channels: int = 1,
    out_dim: int = 1,
    pretrained: bool = False,
) -> nn.Module | None:
    """Build a ResNet18 backbone with a custom head for regression.

    Returns None if torchvision is not available.
    """
    if not _RESNET_AVAILABLE:
        return None

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = resnet18(weights=weights)

    # Replace first conv if input channels differ from 3
    if in_channels != 3:
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained and in_channels == 1:
            # Average RGB weights for single-channel input
            with torch.no_grad():
                resnet.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

    # Replace final FC layer with regression head
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, out_dim)

    return resnet


# ---------------------------------------------------------------------------
# Simple UNet for pixel-level (dense) regression
# ---------------------------------------------------------------------------


class _UNet(nn.Module):
    """Lightweight UNet for pixel-level regression (depth estimation etc.).

    Encoder: 4 down-sampling blocks.  Decoder: 4 up-sampling blocks
    with skip connections.  Output: single-channel at original resolution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 32,
    ) -> None:
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = self._conv_block(base_filters * 16, base_filters * 8)
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)

        self.out_conv = nn.Conv2d(base_filters, out_channels, 1)

    @staticmethod
    def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# DataLoader utilities
# ---------------------------------------------------------------------------


def _image_to_dataloader(
    X: np.ndarray,
    y: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Convert numpy image array to a PyTorch DataLoader.

    X shape: (N, C, H, W) — image tensor in CHW format.
    y shape: (N,) for scalar, (N, H, W) for dense, or None.

    Returns a DataLoader yielding (X_batch,) or (X_batch, y_batch).
    """
    X_t = torch.from_numpy(X).float()
    if y is not None:
        if y.ndim == 1:
            y_t = torch.from_numpy(y).float().unsqueeze(1)
        elif y.ndim == 3:
            # Dense (N, H, W) — keep as 3D for spatial loss functions
            y_t = torch.from_numpy(y).float().unsqueeze(1)  # (N, 1, H, W)
        else:
            y_t = torch.from_numpy(y).float()
        dataset = TensorDataset(X_t, y_t)
    else:
        dataset = TensorDataset(X_t)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_image_model(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    config: HarnessConfig,
    *,
    loss_params: Any = None,
) -> None:
    """Train an image model with Adam mini-batch SGD on DataLoaders.

    Parameters
    ----------
    model : nn.Module
    loss_fn : nn.Module
    train_loader : DataLoader yielding (X_batch, y_batch)
    config : HarnessConfig
        Uses ``config.epochs``, ``config.lr``.
    loss_params : iterable of nn.Parameter or None
        Additional parameters to optimise (e.g., flow coupling network).
    """
    params = list(model.parameters())
    if loss_params is not None:
        params.extend(p for p in loss_params if p.requires_grad)

    opt = optim.Adam(params, lr=config.lr)
    device = next(model.parameters()).device
    model.train()

    for _ in range(config.epochs):
        for batch in train_loader:
            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(X_batch)
            loss = loss_fn(out, y_batch)
            loss.backward()
            opt.step()

    model.eval()


# ---------------------------------------------------------------------------
# ImageHarnessRunner — base ABC for image regression suites
# ---------------------------------------------------------------------------


class ImageHarnessRunner(HarnessRunner):
    """Abstract base class for image regression suites.

    Extends ``HarnessRunner`` with:
    - Image-specific DataLoader creation from numpy arrays
    - Shared backbones (CNN, ResNet, UNet) via helper methods
    - Training loop adapted for image batches

    Subclasses implement:
    - ``fit()`` — train a model given a method name and dataset splits
    - ``predict()`` — produce predictions dict from a fitted model
    - ``methods()`` — return list of method identifiers
    - Properties (``name``, ``description``, ``supported_datasets``)
    """

    # ------------------------------------------------------------------
    # Shared backbone builders (convenience for subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cnn(
        in_channels: int,
        out_dim: int,
        base_filters: int = 16,
    ) -> _ImageCNN:
        """Build a lightweight CNN backbone."""
        return _ImageCNN(
            in_channels=in_channels,
            out_dim=out_dim,
            base_filters=base_filters,
        )

    @staticmethod
    def _build_resnet(
        in_channels: int,
        out_dim: int,
        pretrained: bool = False,
    ) -> nn.Module | None:
        """Build a ResNet18 backbone.  Returns None if torchvision missing."""
        return _make_resnet_backbone(
            in_channels=in_channels,
            out_dim=out_dim,
            pretrained=pretrained,
        )

    @staticmethod
    def _build_unet(
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 32,
    ) -> _UNet:
        """Build a lightweight UNet for pixel-level regression."""
        return _UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
        )

    # ------------------------------------------------------------------
    # Image metadata extraction from splits dict
    # ------------------------------------------------------------------

    @staticmethod
    def _image_shape(splits: dict[str, Any]) -> tuple[int, int, int]:
        """Extract (C, H, W) from image splits dict."""
        X_train = splits["X_train"]
        if X_train.ndim == 4:  # (N, C, H, W)
            return int(X_train.shape[1]), int(X_train.shape[2]), int(X_train.shape[3])
        raise ValueError(f"Expected 4D image array (N, C, H, W), got shape {X_train.shape}")

    @staticmethod
    def _make_train_loader(
        splits: dict[str, Any],
        batch_size: int = 32,
    ) -> DataLoader:
        """Create a training DataLoader from splits dict."""
        return _image_to_dataloader(
            splits["X_train"],
            splits["y_train"],
            batch_size=batch_size,
            shuffle=True,
        )

    @staticmethod
    def _make_test_loader(
        splits: dict[str, Any],
        batch_size: int = 32,
    ) -> DataLoader:
        """Create a test DataLoader from splits dict."""
        X_test = splits["X_test"]
        y_test = splits.get("y_test")
        return _image_to_dataloader(
            X_test,
            y_test,
            batch_size=batch_size,
            shuffle=False,
        )
