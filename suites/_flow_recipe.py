"""NSF training recipe from ``docs/losses/nflows.md`` ("Training recipe for NLL and calibration").

* hold out 15% of the training rows and stop on their NLL (patience 10,
  improvement threshold 1e-4, best weights restored),
* standardised target with ``bound=recommended_tail_bound(y_train_std)``,
* ``bins=16``.

The 3-transform ``[64, 64]`` coupling networks, Adam at ``config.lr`` and
``config.batch_size`` match the plain flow rows, so only the recipe differs.
Targets must already be standardised by the caller (the harness runner does so
by default; ``pzflow_parity`` standardises explicitly).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from harness.runners import HarnessConfig
from suites._backbones import _MLP

VAL_FRACTION = 0.15
PATIENCE = 10
MIN_DELTA = 1e-4
BINS = 16
MAX_EPOCHS = 200  # ceiling; early stopping ends the run far sooner


def fit_nsf_recipe(
    X: torch.Tensor,
    y_std: torch.Tensor,
    config: HarnessConfig,
    *,
    in_dim: int,
    hidden: int,
    n_transforms: int = 3,
    hidden_features: tuple[int, ...] = (64, 64),
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Train an ``_MLP`` backbone + NSF with the documented recipe.

    Returns ``(model, flow_loss, info)`` with ``info`` holding ``bound``,
    ``best_epoch``, ``epochs_run`` and ``n_val``.
    """
    from torchregress.losses.nflows import (
        NormalizingFlowLoss,
        create_flow_model,
        recommended_tail_bound,
    )

    n = X.shape[0]
    n_val = max(1, int(round(VAL_FRACTION * n)))
    if n - n_val < 2:
        raise ValueError(f"training split too small for a validation carve: n={n}")
    gen = torch.Generator().manual_seed(int(config.seed))
    perm = torch.randperm(n, generator=gen)
    va, tr = perm[:n_val].to(X.device), perm[n_val:].to(X.device)
    X_tr, y_tr, X_va, y_va = X[tr], y_std[tr], X[va], y_std[va]

    bound = float(recommended_tail_bound(y_tr))
    flow = create_flow_model(
        n_features=y_std.shape[1],
        context_dim=hidden,
        flow_type="nsf",
        n_transforms=n_transforms,
        hidden_features=list(hidden_features),
        bins=BINS,
        bound=bound,
    )
    model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=hidden).to(X.device)
    loss_fn = NormalizingFlowLoss(flow=flow).to(X.device)
    opt = torch.optim.Adam(
        list(model.parameters()) + [p for p in loss_fn.parameters() if p.requires_grad],
        lr=config.lr,
    )
    max_epochs = config.epochs if config.smoke else MAX_EPOCHS

    best, best_state, best_epoch, bad, ran = float("inf"), None, 0, 0, 0
    for epoch in range(max_epochs):
        ran = epoch + 1
        model.train()
        loss_fn.train()
        for idx in torch.randperm(len(X_tr), generator=gen).to(X.device).split(config.batch_size):
            opt.zero_grad()
            loss_fn(model(X_tr[idx]), y_tr[idx]).backward()
            opt.step()
        model.eval()
        loss_fn.eval()
        with torch.no_grad():
            val = float(loss_fn(model(X_va), y_va))
        if val < best - MIN_DELTA:
            best, bad, best_epoch = val, 0, epoch
            best_state = copy.deepcopy((model.state_dict(), loss_fn.state_dict()))
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state[0])
        loss_fn.load_state_dict(best_state[1])
    model.eval()
    loss_fn.eval()
    info = {"bound": bound, "best_epoch": best_epoch, "epochs_run": ran, "n_val": n_val}
    return model, loss_fn, info
