"""Shared FT-Transformer training/inference for the ``torchregress.models.FTTransformer`` rows.

``FTTransformer`` (Gorishniy et al., 2021) tokenizes every numeric feature
(``FeatureTokenizer``: per-feature scale and bias), prepends a CLS token and runs
a pre-norm Transformer encoder.  The rows use the library default architecture
(``d_token=128``, 3 layers, 4 heads, dropout 0.1) and the paper's default
optimiser (AdamW, lr ``1e-4``, weight decay ``1e-5``).  Inputs go through the
library ``TabularPreprocessor`` (robust scaling + smooth clipping) fitted on the
training rows, like the ``TabularMLP`` rows.

Cost is quadratic in the number of feature tokens, so two things are bounded
explicitly and deterministically:

* inference and validation run in chunks of at most ``ACTIVATION_BUDGET``
  activation floats (``batch * tokens * 4 * d_token``), so a 300k-row test set
  or a 1000-feature table never materialises one huge activation tensor;
* the number of epochs is capped by a device-dependent token-pass budget
  (``epochs * n_fit * n_tokens <= TOKEN_BUDGET[device]``), so wide tables on CPU
  finish inside the per-job wall clock.  The epochs actually used are reported.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torchregress.models import FTTransformer, TabularPreprocessor

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig

LR = 1e-4
WEIGHT_DECAY = 1e-5
ACTIVATION_BUDGET = 1 << 28  # floats per inference chunk (1 GiB in float32)
# Token-passes (epochs * training rows * tokens) per fit.  CUDA: ~1 h of an
# A100-class GPU at the default width; CPU: ~2 h of an 8-core session.
TOKEN_BUDGET = {"cuda": 1e10, "cpu": 2e8, "mps": 2e8}


@dataclass
class FTFit:
    """A trained FT-Transformer plus its input preprocessing and training record."""

    model: FTTransformer
    preprocessor: TabularPreprocessor
    epochs_budget: int
    epochs_run: int = 0
    stopped_early: bool = False
    capped_by_token_budget: bool = False

    @property
    def n_tokens(self) -> int:
        return int(self.preprocessor.n_features_out_) + 1  # features + CLS

    def chunk_size(self) -> int:
        d_token = int(self.model.cls.shape[-1])
        return max(1, ACTIVATION_BUDGET // (self.n_tokens * 4 * d_token))

    @torch.no_grad()
    def forward(self, X: np.ndarray) -> torch.Tensor:
        """Head outputs for raw features ``X`` (chunked; returned on CPU)."""
        device = next(self.model.parameters()).device
        Xp = _to_tensor(self.preprocessor.transform(np.asarray(X, dtype=np.float32)))
        self.model.eval()
        step = self.chunk_size()
        outs = [self.model(Xp[i : i + step].to(device)).cpu() for i in range(0, len(Xp), step)]
        return torch.cat(outs) if outs else torch.empty(0, self.model.out_dim or 0)

    def notes(self) -> str:
        """Epochs actually trained, and why fewer than the epoch budget."""
        capped = ", capped by the device token-pass budget" if self.capped_by_token_budget else ""
        stop = ", early-stopped on validation" if self.stopped_early else ""
        return f"{self.epochs_run}/{self.epochs_budget} epochs{capped}{stop}"


def _to_tensor(x: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


def token_budget_epochs(epochs: int, n_fit: int, n_tokens: int, device: torch.device) -> int:
    """Epochs allowed by the device token-pass budget (at least one)."""
    budget = TOKEN_BUDGET.get(device.type, TOKEN_BUDGET["cpu"])
    return max(1, min(int(epochs), int(budget // max(1, n_fit * n_tokens))))


def fit_ft_transformer(
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: nn.Module,
    out_dim: int,
    config: HarnessConfig,
    *,
    epochs: int,
    batch_size: int = 256,
    val_fraction: float = 0.15,
    patience: int = 10,
) -> FTFit:
    """Train ``FTTransformer(n_features, out_dim=out_dim)`` with mini-batch AdamW.

    ``y`` is used as given (callers standardise it when their peers do).  With
    ``val_fraction > 0`` a seeded share of the training rows is held out, the
    validation loss is checked every epoch and the best weights are restored
    after ``patience`` epochs without improvement.
    """
    set_harness_seed(config.seed)
    device = current_device()
    X = np.asarray(X, dtype=np.float32)
    y2 = np.asarray(y, dtype=np.float32).reshape(len(X), -1)
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(X))
    n_val = int(round(val_fraction * len(X))) if val_fraction > 0 else 0
    val_idx, fit_idx = order[:n_val], order[n_val:]
    prep = TabularPreprocessor().fit(X[fit_idx])
    torch.manual_seed(config.seed)
    model = FTTransformer(int(prep.n_features_out_), out_dim=out_dim).to(device)
    n_tokens = int(prep.n_features_out_) + 1
    run_epochs = token_budget_epochs(epochs, len(fit_idx), n_tokens, device)

    Xf = _to_tensor(prep.transform(X[fit_idx])).to(device)
    yf = torch.as_tensor(y2[fit_idx]).to(device)
    Xv = _to_tensor(prep.transform(X[val_idx])) if n_val else None
    yv = torch.as_tensor(y2[val_idx]) if n_val else None
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = loss_fn.to(device)
    fit = FTFit(model, prep, epochs_budget=int(epochs))
    fit.capped_by_token_budget = run_epochs < int(epochs)
    step = fit.chunk_size()
    best, bad, best_state = math.inf, 0, None
    gen = torch.Generator().manual_seed(int(config.seed) + 17)
    bs = max(1, min(int(batch_size), len(Xf)))
    for epoch in range(run_epochs):
        model.train()
        perm = torch.randperm(len(Xf), generator=gen).to(device)
        for i in range(0, len(Xf), bs):
            idx = perm[i : i + bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xf[idx]), yf[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        fit.epochs_run = epoch + 1
        if Xv is None or yv is None:
            continue
        model.eval()
        with torch.no_grad():
            total = 0.0
            for i in range(0, len(Xv), step):
                xb, yb = Xv[i : i + step].to(device), yv[i : i + step].to(device)
                total += float(loss_fn(model(xb), yb)) * len(xb)
        val = total / len(Xv)
        if not math.isfinite(val):
            raise FloatingPointError(f"FT-Transformer validation loss is {val} at epoch {epoch}")
        if val < best:
            best, bad, best_state = val, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                fit.stopped_early = True
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return fit
