"""ResNet18 feature extractor for image regression multi-target."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, d_out: int = 64, hidden: int = 128) -> None:
        super().__init__()
        base = resnet18(weights=None)
        base.fc = nn.Identity()
        self.feature_extractor = nn.Sequential(
            base,
            nn.Linear(512, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )
        self.d_out = d_out

    def forward(self, x):
        return self.feature_extractor(x)
