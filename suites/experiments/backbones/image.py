"""ResNet18 feature extractor for image regression multi-target."""

from __future__ import annotations

import torch.nn as nn

try:
    from torchvision.models import resnet18

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    resnet18 = None  # type: ignore[assignment]


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, d_out: int = 64, hidden: int = 128) -> None:
        super().__init__()
        if HAS_TORCHVISION:
            base = resnet18(weights=None)
            base.fc = nn.Identity()
            self.feature_extractor = nn.Sequential(
                base,
                nn.Linear(512, hidden),
                nn.ReLU(),
                nn.Linear(hidden, d_out),
            )
        else:
            # fallback: simple MLP on flattened image
            self.feature_extractor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 64 * 64, hidden),
                nn.ReLU(),
                nn.Linear(hidden, d_out),
            )
        self.d_out = d_out

    def forward(self, x):
        return self.feature_extractor(x)
