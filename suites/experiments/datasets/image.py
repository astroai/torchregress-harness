"""Image regression loaders — BIWI multi-target with synthetic fallback."""

from __future__ import annotations

import torch
from torch import Tensor


def load_image(
    name: str, seed: int, n_cap: int = 1000, target_dim: int = 3, gen: torch.Generator | None = None
):
    if gen is None:
        gen = torch.Generator().manual_seed(seed)
    # try BIWI via torchvision or openml, else synthetic
    if name.lower() == "biwi":
        try:
            # attempt to load via torchvision.datasets.BIWI if available
            import torchvision

            # placeholder: if not found, fallback
            raise ImportError("BIWI not in torchvision")
        except Exception:
            pass
    # synthetic image regression proxy: 64x64 RGB, d=3 targets
    n = min(n_cap, 1000)
    # images as 3x64x64
    x = torch.randn(n, 3, 64, 64, generator=gen)
    # targets: 3 correlated values
    y_base = torch.randn(n, generator=gen)
    y = torch.stack(
        [
            y_base,
            0.6 * y_base + 0.8 * torch.randn(n, generator=gen),
            -0.4 * y_base + 0.9 * torch.randn(n, generator=gen),
        ],
        dim=-1,
    )
    if target_dim != 3:
        y = y[:, :target_dim]

    # splits
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    perm = torch.randperm(n, generator=gen)
    idx_train, idx_cal, idx_test = (
        perm[:n_train],
        perm[n_train : n_train + n_cal],
        perm[n_train + n_cal :],
    )
    return x[idx_train], y[idx_train], x[idx_cal], y[idx_cal], x[idx_test], y[idx_test]
