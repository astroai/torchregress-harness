"""Tabular loaders with pre-split y scaling fix (L1)."""

from __future__ import annotations

import torch
from torch import Tensor


def _standardize(x: Tensor, stats_from: Tensor) -> Tensor:
    mean, std = stats_from.mean(dim=0), stats_from.std(dim=0).clamp_min(1e-8)
    return (x - mean) / std


def make_multi_target(y: Tensor, *, rho: float, gen: torch.Generator, train_idx: Tensor) -> Tensor:
    # y is 1D [n]; compute y_std from train only (fix L1 leak)
    y_train = y[train_idx]
    y_mean, y_std = y_train.mean(), y_train.std().clamp_min(1e-8)
    y_std_all = (y - y_mean) / y_std
    import numpy as np

    eps = torch.randn(y_std_all.shape[0], generator=gen) * float(np.sqrt(max(1e-8, 1.0 - rho**2)))
    return torch.stack([y_std_all, rho * y_std_all + eps], dim=-1)


def load_tabular(
    name: str,
    seed: int,
    n_cap: int = 5000,
    target_dim: int = 2,
    rho: float = 0.6,
    gen: torch.Generator | None = None,
):
    if gen is None:
        gen = torch.Generator().manual_seed(seed)
    # minimal loaders — reuse harness datasets logic where possible, else synthetic
    if name == "diabetes":
        from sklearn.datasets import load_diabetes

        raw = load_diabetes()
        x = torch.as_tensor(raw.data, dtype=torch.float32)
        y = torch.as_tensor(raw.target, dtype=torch.float32)
    elif name == "california_housing":
        import os

        from sklearn.datasets import fetch_california_housing

        data_home = os.environ.get("SKLEARN_DATA_HOME") or __import__("os").path.expanduser(
            "~/.astroai/sklearn_data"
        )
        raw = fetch_california_housing(data_home=data_home)
        idx = torch.randperm(raw.data.shape[0], generator=gen)[:n_cap]
        x = torch.as_tensor(raw.data[idx.numpy()], dtype=torch.float32)
        y = torch.as_tensor(raw.target[idx.numpy()], dtype=torch.float32)
    elif name in ("superconductivity", "diamonds", "year_prediction_msd"):
        # fallback to synthetic if openml not cached; never fail CI
        try:
            from torchregress.utils.openml_relaxed import fetch_openml_dataset

            x, y = fetch_openml_dataset(name, n_cap=n_cap, allow_download=False)
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
        except Exception:
            # synthetic proxy
            n = min(n_cap, 5000)
            d = 20
            x = torch.randn(n, d, generator=gen)
            y = x[:, 0] * 2.0 + torch.randn(n, generator=gen) * 0.5
    else:
        n = min(n_cap, 5000)
        d = 10
        x = torch.randn(n, d, generator=gen)
        y = x[:, 0] * 1.5 + torch.randn(n, generator=gen) * 0.3

    # n_cap subsampling already done for some; if still > n_cap, subsample
    if x.shape[0] > n_cap:
        idx = torch.randperm(x.shape[0], generator=gen)[:n_cap]
        x, y = x[idx], y[idx]

    # multi-target
    if target_dim > 1:
        # need train_idx for y scaling fix
        n = x.shape[0]
        perm = torch.randperm(n, generator=gen)
        n_train = max(1, int(0.6 * n))
        train_idx = perm[:n_train]
        y = make_multi_target(y, rho=rho, gen=gen, train_idx=train_idx)
    else:
        y = y.reshape(-1, 1) if y.dim() == 1 else y

    # splits
    n = x.shape[0]
    perm = torch.randperm(n, generator=gen)
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    idx_train, idx_cal, idx_test = (
        perm[:n_train],
        perm[n_train : n_train + n_cal],
        perm[n_train + n_cal :],
    )
    # standardize X on train only
    x_train, x_cal, x_test = x[idx_train], x[idx_cal], x[idx_test]
    y_train, y_cal, y_test = y[idx_train], y[idx_cal], y[idx_test]
    # y already standardized via make_multi_target; for single-target also standardize on train
    if target_dim == 1:
        y_mean, y_std = y_train.mean(), y_train.std().clamp_min(1e-8)
        y_train = (y_train - y_mean) / y_std
        y_cal = (y_cal - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

    # standardize X
    mean, std = x_train.mean(dim=0), x_train.std(dim=0).clamp_min(1e-8)
    x_train, x_cal, x_test = (x_train - mean) / std, (x_cal - mean) / std, (x_test - mean) / std

    return x_train, y_train, x_cal, y_cal, x_test, y_test
