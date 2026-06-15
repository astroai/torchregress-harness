"""Shared neural-network backbones and training helpers for all suites.

Import from here instead of duplicating ``_MLP`` and ``_train_one`` in every
suite file.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from harness.runners import HarnessConfig


class _MLP(nn.Module):
    """Three-layer MLP backbone (Linear→ReLU→Linear→ReLU→Linear).

    Optional dropout after each hidden layer for MC-dropout use cases.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        layers.extend([
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ])
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Linear(nn.Module):
    """Single linear layer — lightweight capacity-matched baseline."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _train_one(
    model: nn.Module,
    loss_fn: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    config: HarnessConfig | None = None,
    *,
    loss_params: Any = None,
    transform_output: Any = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
) -> None:
    """Train one model with Adam mini-batch SGD.

    Parameters
    ----------
    model : nn.Module
    loss_fn : nn.Module
    X : (N, D) float tensor
    y : (N, 1) float tensor
    config : HarnessConfig | None
        If provided, epochs/batch_size/lr default from config.
    loss_params : iterable of nn.Parameter or None
        Additional parameters to optimise alongside model.parameters()
        (e.g., ``loss_fn.parameters()`` for learnable-loss methods such as
        ``AdaptiveRobustLoss``).
    transform_output : callable or None
        If not None, applied to the raw model output before computing the loss.
        Used for log-mean parameterisations (``transform_output=torch.exp``).
    epochs : int | None
        Overrides ``config.epochs``.
    batch_size : int | None
        Overrides ``config.batch_size``.
    lr : float | None
        Overrides ``config.lr``.
    """
    _epochs = epochs if epochs is not None else config.epochs if config is not None else 60
    _batch = batch_size if batch_size is not None else config.batch_size if config is not None else 64
    _lr = lr if lr is not None else config.lr if config is not None else 1e-3

    params = list(model.parameters())
    if loss_params is not None:
        params.extend(p for p in loss_params if p.requires_grad)

    opt = optim.Adam(params, lr=_lr)
    n = X.shape[0]

    for _ in range(_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, _batch):
            idx = perm[i : i + _batch]
            opt.zero_grad(set_to_none=True)
            out = model(X[idx])
            if transform_output is not None:
                out = transform_output(out)
            loss = loss_fn(out, y[idx])
            loss.backward()
            opt.step()


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _hidden_size(in_dim: int) -> int:
    """Heuristic hidden-layer size for tabular MLPs."""
    return max(16, min(in_dim * 4, 64))


def _gaussian_nll_predict(
    model: nn.Module, X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract predictive mean & std from a 2-output GaussianNLL model.

    Parameters
    ----------
    model : nn.Module (eval mode expected)
        Outputs (N, 2) tensor — column 0 = mean, column 1 = log-variance.
    X : (N, D) numpy array

    Returns
    -------
    mean : (N,) numpy array
    std  : (N,) numpy array
    """
    with torch.no_grad():
        out = model(torch.from_numpy(X).float())
        mean = out[:, 0].numpy().reshape(-1)
        var = torch.exp(out[:, 1]).clamp_min(1e-8).numpy().reshape(-1)
    return mean, np.sqrt(var)


def _mixture_gaussian_stats(
    means: list[np.ndarray],
    stds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine ensemble member means & stds into mixture mean & std.

    Mixture mean  = average of member means.
    Mixture variance = average of member variances
                     + variance of member means.

    Parameters
    ----------
    means : list of (N,) numpy arrays, one per ensemble member.
    stds  : list of (N,) numpy arrays, one per ensemble member.

    Returns
    -------
    mixture_mean : (N,) numpy array
    mixture_std  : (N,) numpy array
    """
    means_s = np.stack(means, axis=0)  # (K, N)
    vars_s = np.stack([s ** 2 for s in stds], axis=0)
    mixture_mean = means_s.mean(axis=0)
    mixture_var = vars_s.mean(axis=0) + means_s.var(axis=0, ddof=0)
    mixture_std = np.sqrt(np.maximum(mixture_var, 1e-8))
    return mixture_mean, mixture_std


def _carve_calibration_split(
    splits: dict[str, Any],
    *,
    cal_frac: float,
    seed: int,
    smoke: bool,
    suite_name: str,
    forward_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Carve a calibration split from the training data.

    Used by suites that need a train/calibration/test split (conformal,
    calibration).  Returns a new ``splits``-like dict with ``X_cal``,
    ``y_cal`` added and ``X_train``/``y_train`` reduced accordingly.

    Parameters
    ----------
    splits : dict
        Must contain ``X_train``, ``y_train``, ``X_test``, ``y_test``.
    cal_frac : float
        Fraction of training data to reserve for calibration.
    seed : int
        Random seed for the permutation.
    smoke : bool
        If True, cap sizes for a fast smoke check (train≤100, cal≤30, test≤50).
    suite_name : str
        Used in the error message when the dataset is too small.
    forward_keys : tuple[str, ...]
        Metadata keys to copy from ``splits`` into the returned dict
        (e.g. ``("_alpha", "_source_url")``).

    Returns
    -------
    dict[str, Any]
        A new dict with ``X_train``, ``y_train``, ``X_cal``, ``y_cal``,
        ``X_test``, ``y_test``, plus any forwarded metadata keys.
    """
    X_all = splits["X_train"]
    y_all = splits["y_train"]

    n_cal = max(50, int(len(X_all) * cal_frac))
    n_cal = min(n_cal, max(1, len(X_all) - 20))
    n_fit = len(X_all) - n_cal

    if n_fit < 10:
        raise ValueError(
            f"Dataset too small for {suite_name} suite: "
            f"{len(X_all)} total, {n_fit} fit, {n_cal} cal (need >=10 fit)"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_all))

    augmented: dict[str, Any] = {
        "X_train": X_all[perm[:n_fit]],
        "y_train": y_all[perm[:n_fit]],
        "X_cal": X_all[perm[n_fit:]],
        "y_cal": y_all[perm[n_fit:]],
        "X_test": splits["X_test"],
        "y_test": splits["y_test"],
    }

    # Always forward the y-scaler so evaluate() can inverse-transform
    if "_y_scaler" in splits:
        augmented["_y_scaler"] = splits["_y_scaler"]

    for key in forward_keys:
        if key in splits:
            augmented[key] = splits[key]

    if smoke:
        n_fit_smoke = min(100, n_fit)
        n_cal_smoke = min(30, n_cal)
        n_test_smoke = min(50, len(augmented["X_test"]))
        augmented["X_train"] = augmented["X_train"][:n_fit_smoke]
        augmented["y_train"] = augmented["y_train"][:n_fit_smoke]
        augmented["X_cal"] = augmented["X_cal"][:n_cal_smoke]
        augmented["y_cal"] = augmented["y_cal"][:n_cal_smoke]
        augmented["X_test"] = augmented["X_test"][:n_test_smoke]
        augmented["y_test"] = augmented["y_test"][:n_test_smoke]

    return augmented
