"""Shared helpers for the ``torchregress.models`` TabularMLP rows.

The point of these rows is "strong by default": the model, preprocessor and
training recipe are the library defaults (``TabularMLP`` hidden ``(256, 256,
256)``, periodic embeddings, ``fit_tabular`` AdamW + one-cycle + early stopping
on a 15% validation split).  The harness only supplies

* the loss head (MSE, Gaussian NLL or multi-quantile),
* the seed (``config.seed``; ensemble members use ``seed + m``),
* ``epochs = min(library default, config.epochs)`` so smoke runs and the
  release epoch budget still bound the cost.

The harness-wide ``config.lr`` / ``config.batch_size`` are deliberately *not*
forwarded: they belong to the harness's generic MLP, not to this recipe.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch
from torchregress.losses import GaussianNLLLoss, MultiQuantileLoss, WeightedMSELoss
from torchregress.models import (
    TabularEnsembleFit,
    TabularFit,
    TabularMLP,
    fit_tabular,
    fit_tabular_ensemble,
)

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig
from suites._backbones import _mixture_gaussian_stats

N_MEMBERS = 5

# Library default for ``fit_tabular(epochs=...)``; the cap is min(this, config.epochs).
LIBRARY_EPOCHS: int = int(inspect.signature(fit_tabular).parameters["epochs"].default)


def tabular_epochs(config: HarnessConfig) -> int:
    """Epoch budget: the library default, capped by the harness config."""
    return max(1, min(LIBRARY_EPOCHS, int(config.epochs)))


def _onecycle_safe_epochs(epochs: int, n_fit: int, batch_size: int = 256) -> int:
    """Dodge a ZeroDivisionError in torch ``OneCycleLR(pct_start=0.1)`` at exactly 10 total steps.

    ``fit_tabular`` schedules ``epochs * ceil(n_fit / batch_size)`` steps; with 10 the
    first phase ends at step 0 and the scheduler divides by zero (torchregress 0.3 dev,
    reported in RELEASE_0.3_STATUS.md).  One extra epoch changes the step count.
    """
    if epochs * -(-n_fit // batch_size) == 10:
        return epochs + 1
    return epochs


def _head(kind: str, quantiles: tuple[float, ...]) -> tuple[Any, int]:
    if kind == "mse":
        return WeightedMSELoss(), 1
    if kind == "gaussian":
        return GaussianNLLLoss(), 2
    if kind == "quantile":
        return MultiQuantileLoss(quantiles=list(quantiles)), len(quantiles)
    raise ValueError(f"unknown head {kind!r}")


def fit_tabmlp(
    X: np.ndarray,
    y: np.ndarray,
    config: HarnessConfig,
    *,
    kind: str,
    n_members: int = 1,
    quantiles: tuple[float, ...] = (),
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> TabularFit | TabularEnsembleFit:
    """Fit TabularMLP(s) with library defaults.

    ``n_members == 1`` returns a :class:`TabularFit`, otherwise a
    :class:`TabularEnsembleFit` (members seeded ``config.seed + m``).
    """
    loss_fn, out = _head(kind, quantiles)
    set_harness_seed(config.seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    dev = current_device()
    n_fit = len(X) if X_val is not None else len(X) - max(1, int(round(0.15 * len(X))))
    kw: dict[str, Any] = {
        "epochs": _onecycle_safe_epochs(tabular_epochs(config), n_fit),
        "device": dev,
    }
    if X_val is not None and y_val is not None:
        kw.update(X_val=np.asarray(X_val, dtype=np.float32), y_val=np.asarray(y_val, np.float32))
    if n_members == 1:
        from torchregress.models import TabularPreprocessor

        prep = TabularPreprocessor().fit(X)
        torch.manual_seed(config.seed)
        model = TabularMLP(prep.n_features_out_, out)
        return fit_tabular(model, loss_fn, X, y, seed=config.seed, preprocessor=prep, **kw)
    return fit_tabular_ensemble(
        lambda d: TabularMLP(d, out), loss_fn, X, y, n_members=n_members, seed=config.seed, **kw
    )


def predict_point(fit: TabularFit | TabularEnsembleFit, X: np.ndarray) -> np.ndarray:
    return fit.predict(np.asarray(X, dtype=np.float32))[:, 0].cpu().numpy().astype(np.float64)


def predict_gaussian(
    fit: TabularFit | TabularEnsembleFit, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std in target units; ensembles combine members as a Gaussian mixture."""
    X = np.asarray(X, dtype=np.float32)
    members = fit.members if isinstance(fit, TabularEnsembleFit) else [fit]
    means, stds = [], []
    for m in members:
        out = m.predict(X).cpu().numpy().astype(np.float64)
        means.append(out[:, 0])
        stds.append(np.sqrt(np.exp(out[:, 1])))
    if len(members) == 1:
        return means[0], np.maximum(stds[0], 1e-8)
    return _mixture_gaussian_stats(means, stds)


def predict_quantiles(fit: TabularFit | TabularEnsembleFit, X: np.ndarray) -> np.ndarray:
    """Quantile head outputs ``[n, len(quantiles)]`` (ensembles average the heads)."""
    return fit.predict(np.asarray(X, dtype=np.float32)).cpu().numpy().astype(np.float64)
