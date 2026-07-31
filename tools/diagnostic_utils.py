"""Shared training utilities for EIV diagnostic scripts.

Extracted from diagnose_hybrid_mc.py and diagnose_mc_warmup.py to eliminate
cross-module duplication.  Provides data generators, model factory, hybrid
loss, training loops with optional external-optimizer support, evaluation
helpers, and warmup trainers.

Usage from diagnostic scripts::

    from diagnostic_utils import (
        make_mlp, train_with_loss, train_baseline, train_warmup_mc,
        train_with_warmup, hybrid_loss,
        generate_linear_eiv_data, generate_nonlinear_eiv_data,
    )

Usage from tests::

    from tools.diagnostic_utils import train_with_loss, train_baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# ── torchregress import path ────────────────────────────────────────────
# Same pattern as the diagnostic scripts: resolve torchregress/src relative
# to the tools/ directory.
_HERE = Path(__file__).resolve().parent
_TORCHREGRESS_SRC = _HERE.parent.parent / "torchregress" / "src"
if str(_TORCHREGRESS_SRC) not in sys.path:
    sys.path.insert(0, str(_TORCHREGRESS_SRC))

from torchregress.losses import FunctionalEIVLoss, WeightedMSELoss  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# Data generators
# ═══════════════════════════════════════════════════════════════════════


def generate_nonlinear_eiv_data(
    n_samples: int = 2000,
    sigma_x: float = 0.5,
    sigma_y: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_features = 6
    X_true = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    X_obs = (X_true + sigma_x * rng.standard_normal((n_samples, n_features))).astype(np.float32)
    y = (
        np.sin(2.0 * np.pi * X_true[:, 0])
        + 0.5 * np.cos(4.0 * np.pi * X_true[:, 1])
        + 0.3 * np.tanh(2.0 * X_true[:, 2])
        + sigma_y * rng.standard_normal(n_samples).astype(np.float32)
    )
    n_train = n_samples // 2
    return {
        "X_train": X_obs[:n_train],
        "y_train": y[:n_train].reshape(-1, 1),
        "X_test": X_obs[n_train:],
        "y_test": y[n_train:].reshape(-1, 1),
        "X_test_true": X_true[n_train:],
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
    }


def generate_linear_eiv_data(
    n_samples: int = 2000,
    sigma_x: float = 0.5,
    sigma_y: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Linear EIV data: y = X·β + ε, β ~ U(-1, 1) per feature.

    Matches the torchregress-harness synthetic_eiv dataset generator.
    """
    rng = np.random.default_rng(seed)
    n_features = 6
    beta = rng.uniform(-1.0, 1.0, (n_features, 1)).astype(np.float32)

    X_true = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    X_obs = (X_true + sigma_x * rng.standard_normal((n_samples, n_features))).astype(np.float32)
    y_clean = X_true @ beta
    y = (y_clean + sigma_y * rng.standard_normal((n_samples, 1)).astype(np.float32)).astype(
        np.float32
    )
    n_train = n_samples // 2
    return {
        "X_train": X_obs[:n_train],
        "y_train": y[:n_train],
        "X_test": X_obs[n_train:],
        "y_test": y[n_train:],
        "X_test_true": X_true[n_train:],
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
    }


# ═══════════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════════


def make_mlp(in_dim: int, hidden: int = 32, out_dim: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ═══════════════════════════════════════════════════════════════════════
# Hybrid loss: Jacobian variance + MC mean
# ═══════════════════════════════════════════════════════════════════════


def hybrid_loss(
    x_obs: torch.Tensor,
    target: torch.Tensor,
    model: nn.Module,
    sigma_x: float,
    sigma_y: float,
    n_samples: int = 20,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Hybrid EIV loss: analytical Jacobian variance + MC perturbation mean.

    Args:
        x_obs: Noisy inputs [B, D]
        target: Observed targets [B, 1]
        model: Differentiable model
        sigma_x: Input noise stddev (scalar or per-feature)
        sigma_y: Output noise stddev (scalar)
        n_samples: Number of MC perturbations for mean prediction
        eps: Numerical stability jitter

    Returns:
        Scalar loss (mean over batch)
    """
    B, D = x_obs.shape
    device = x_obs.device

    # ── 1. Analytical Jacobian variance (stable) ──────────
    x_grad = x_obs.detach().clone().requires_grad_(True)
    f_x = model(x_grad)  # [B, 1]

    grad_list: list[torch.Tensor] = []
    for j in range(f_x.shape[1]):
        g = torch.autograd.grad(
            f_x[:, j].sum(),
            x_grad,
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, D]
        grad_list.append(g)
    J = torch.stack(grad_list, dim=-1)  # [B, D, out_dim]
    J = J.squeeze(-1)  # [B, D]

    var_jac = (J**2).sum(dim=-1) * (sigma_x**2) + sigma_y**2  # [B]
    var_jac = var_jac.clamp_min(eps)

    # ── 2. MC perturbation mean (bias-corrected) ─────────
    noise = torch.randn(n_samples, B, D, device=device, dtype=x_obs.dtype)
    if isinstance(sigma_x, (int, float)):
        noise = noise * sigma_x
    else:
        noise = noise * torch.tensor(sigma_x, device=device).view(1, 1, D)

    x_perturbed = x_obs.unsqueeze(0) + noise
    x_flat = x_perturbed.reshape(-1, D)
    y_preds_flat = model(x_flat)
    y_preds = y_preds_flat.reshape(n_samples, B, -1)
    mean_pred = y_preds.mean(dim=0).squeeze(-1)  # [B]

    # ── 3. Gaussian NLL ──
    target_s = target.squeeze(-1)
    nll = 0.5 * (torch.log(var_jac) + (target_s - mean_pred) ** 2 / var_jac)
    return nll.mean()


# ═══════════════════════════════════════════════════════════════════════
# Low-level training helpers (from diagnose_mc_warmup)
# ═══════════════════════════════════════════════════════════════════════


def _batch_loss_mse(model: nn.Module, Xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    return ((model(Xb) - yb) ** 2).mean()


def _batch_loss_mc(loss_fn: FunctionalEIVLoss, Xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    return loss_fn(Xb, yb)


def _train_loop(
    model: nn.Module,
    opt: torch.optim.Optimizer | None,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    batch_size: int,
    mc_loss_fn: FunctionalEIVLoss | None = None,
    lr: float = 1e-3,
) -> None:
    """Generic training loop supporting both MSE and MC loss.

    Args:
        opt: Optimizer to use. If None, creates a new Adam internally.
             Passing one allows sharing optimizer state across calls.
        lr: Learning rate used only when opt is None (creates new Adam).
    """
    if opt is None:
        opt = Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    model.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            if mc_loss_fn is not None:
                loss = _batch_loss_mc(mc_loss_fn, X[idx], y[idx])
            else:
                loss = _batch_loss_mse(model, X[idx], y[idx])
            loss.backward()
            opt.step()


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════


def eval_rmse_clean(
    model: nn.Module,
    X_test_true: torch.Tensor,
    y_test_true: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(X_test_true).squeeze(-1)
        return float(torch.sqrt(((pred - y_test_true.squeeze(-1)) ** 2).mean()).item())


def eval_pred_range(
    model: nn.Module,
    X: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(X).squeeze(-1)
        return float(pred.max().item() - pred.min().item())


# ═══════════════════════════════════════════════════════════════════════
# General-purpose training (from diagnose_hybrid_mc)
# ═══════════════════════════════════════════════════════════════════════


def train_with_loss(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    X_test_true: torch.Tensor,
    y_test_true: torch.Tensor,
    *,
    loss_type: str = "mse",
    sigma_x: float = 0.5,
    sigma_y: float = 0.1,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    opt: Adam | None = None,
) -> dict:
    """Train model with specified loss type, tracking RMSE_clean trajectory.

    Args:
        opt: External optimizer to use. If None, creates a new Adam.
             Passing one allows sharing optimizer state across calls.
    """
    if opt is None:
        opt = Adam(model.parameters(), lr=lr)
    n = X.shape[0]

    # Prepare loss functions
    if loss_type == "mse":
        loss_fn = WeightedMSELoss()
    elif loss_type == "feiv":
        loss_fn = FunctionalEIVLoss(
            model=model,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            monte_carlo=False,
        )
    elif loss_type == "mc":
        loss_fn = FunctionalEIVLoss(
            model=model,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            monte_carlo=True,
            n_samples=20,
        )
    elif loss_type == "hybrid":
        loss_fn = None  # We'll call hybrid_loss directly
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    history: list[dict] = []
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_losses: list[float] = []

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)

            if loss_type == "hybrid":
                loss = hybrid_loss(
                    X[idx],
                    y[idx],
                    model,
                    sigma_x,
                    sigma_y,
                    n_samples=20,
                )
            elif loss_type in ("feiv", "mc"):
                loss = loss_fn(X[idx], y[idx])
            elif loss_type == "mse":
                loss = loss_fn(model(X[idx]), y[idx])
            else:
                raise RuntimeError

            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))

        # Track every 10 epochs + key milestones
        if epoch % 10 == 0 or epoch in (0, 9, 19, 29, 49, 74, 99):
            model.eval()
            with torch.no_grad():
                pred_clean = model(X_test_true).squeeze(-1)
                rmse = float(
                    torch.sqrt(((pred_clean - y_test_true.squeeze(-1)) ** 2).mean()).item()
                )
                pred_range = float(pred_clean.max().item() - pred_clean.min().item())
            history.append(
                {
                    "epoch": epoch,
                    "rmse_clean": rmse,
                    "pred_range": pred_range,
                }
            )
            model.train()

    # Final eval
    model.eval()
    with torch.no_grad():
        pred_clean = model(X_test_true).squeeze(-1)
        rmse_final = float(torch.sqrt(((pred_clean - y_test_true.squeeze(-1)) ** 2).mean()).item())
        pred_range = float(pred_clean.max().item() - pred_clean.min().item())

    return {
        "loss_type": loss_type,
        "rmse_clean": rmse_final,
        "pred_range": pred_range,
        "history": history,
    }


# ═══════════════════════════════════════════════════════════════════════
# Baseline training (from diagnose_mc_warmup)
# ═══════════════════════════════════════════════════════════════════════


def train_baseline(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    X_test_true: torch.Tensor,
    y_test_true: torch.Tensor,
    *,
    label: str,
    sigma_x: float = 0.5,
    sigma_y: float = 0.1,
    total_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    opt: Adam | None = None,
) -> dict:
    """Train a pure baseline (MSE or FunctionalEIV_MC from scratch).

    Args:
        opt: External optimizer to use. If None, _train_loop creates
             a new Adam internally. Passing one allows sharing optimizer
             state across calls (e.g. MSE warmup → MC training).
    """
    if label == "MSE":
        _train_loop(model, opt, X, y, total_epochs, batch_size, lr=lr)
    elif label == "FuncEIV":
        loss_fn = FunctionalEIVLoss(
            model=model,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            monte_carlo=False,
        )
        _train_loop(
            model,
            opt,
            X,
            y,
            total_epochs,
            batch_size,
            mc_loss_fn=loss_fn,
            lr=lr,
        )
    elif label == "FuncEIV_MC":
        loss_fn = FunctionalEIVLoss(
            model=model,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            monte_carlo=True,
            n_samples=20,
        )
        _train_loop(
            model,
            opt,
            X,
            y,
            total_epochs,
            batch_size,
            mc_loss_fn=loss_fn,
            lr=lr,
        )
    else:
        raise ValueError(f"Unknown baseline: {label}")

    rmse = eval_rmse_clean(model, X_test_true, y_test_true)
    p_range = eval_pred_range(model, X)
    return {"label": label, "rmse_clean": rmse, "pred_range": p_range}


# ═══════════════════════════════════════════════════════════════════════
# Warmup trainers
# ═══════════════════════════════════════════════════════════════════════


def train_with_warmup(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    X_test_true: torch.Tensor,
    y_test_true: torch.Tensor,
    *,
    warmup_epochs: int,
    hybrid_epochs: int,
    sigma_x: float = 0.5,
    sigma_y: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> dict:
    """MSE warmup → hybrid training, tracking RMSE_clean trajectory.

    Reuses train_with_loss() with a shared Adam optimizer for the MSE
    warmup phase so momentum carries over into hybrid training.
    """
    opt = Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    total_epochs = warmup_epochs + hybrid_epochs
    history: list[dict] = []

    # Phase 1: MSE warmup — delegate to train_with_loss with shared optimizer
    if warmup_epochs > 0:
        mse_result = train_with_loss(
            model,
            X,
            y,
            X_test_true,
            y_test_true,
            loss_type="mse",
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            epochs=warmup_epochs,
            lr=lr,
            batch_size=batch_size,
            opt=opt,
        )
        history.extend(mse_result["history"])
        history.append(
            {
                "epoch": warmup_epochs,
                "rmse_clean": mse_result["rmse_clean"],
                "phase": "warmup_end",
            }
        )

    # Phase 2: Hybrid training — continues with same optimizer
    model.train()
    for epoch in range(hybrid_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = hybrid_loss(
                X[idx],
                y[idx],
                model,
                sigma_x,
                sigma_y,
                n_samples=20,
            )
            loss.backward()
            opt.step()

    # Final eval
    model.eval()
    with torch.no_grad():
        pred_clean = model(X_test_true).squeeze(-1)
        rmse_final = float(torch.sqrt(((pred_clean - y_test_true.squeeze(-1)) ** 2).mean()).item())
        pred_range = float(pred_clean.max().item() - pred_clean.min().item())
    history.append({"epoch": total_epochs, "rmse_clean": rmse_final, "phase": "hybrid_end"})

    return {
        "loss_type": "hybrid_warmup",
        "rmse_clean": rmse_final,
        "pred_range": pred_range,
        "history": history,
    }


def train_warmup_mc(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    X_test_true: torch.Tensor,
    y_test_true: torch.Tensor,
    *,
    warmup_epochs: int,
    mc_epochs: int,
    lr: float = 1e-3,
    batch_size: int = 64,
    sigma_x: float = 0.5,
    sigma_y: float = 0.1,
    n_mc_samples: int = 20,
    reset_optimizer: bool = False,
    track_every: int = 5,
) -> dict:
    """MSE warmup → FunctionalEIV_MC switch, tracking RMSE_clean trajectory.

    Reuses train_baseline() with a shared Adam optimizer for the MSE
    warmup phase so momentum carries over into MC training.
    """
    total_epochs = warmup_epochs + mc_epochs
    opt = Adam(model.parameters(), lr=lr)
    history: list[dict] = []

    # Phase 1: MSE warmup — delegate to train_baseline with shared optimizer
    if warmup_epochs > 0:
        mse_result = train_baseline(
            model,
            X,
            y,
            X_test_true,
            y_test_true,
            label="MSE",
            sigma_x=sigma_x,
            total_epochs=warmup_epochs,
            lr=lr,
            batch_size=batch_size,
            opt=opt,
        )
        history.append(
            {
                "epoch": warmup_epochs,
                "phase": "warmup_end",
                "rmse_clean": mse_result["rmse_clean"],
                "pred_range": mse_result["pred_range"],
            }
        )

    # Phase 2: FunctionalEIV_MC
    if reset_optimizer:
        opt = Adam(model.parameters(), lr=lr * 0.1)

    mc_loss = FunctionalEIVLoss(
        model=model,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        monte_carlo=True,
        n_samples=n_mc_samples,
    )

    _train_loop(model, opt, X, y, mc_epochs, batch_size, mc_loss_fn=mc_loss)

    # Evaluate after MC phase
    rmse = eval_rmse_clean(model, X_test_true, y_test_true)
    p_range = eval_pred_range(model, X)
    history.append(
        {
            "epoch": total_epochs,
            "phase": "mc_end",
            "rmse_clean": rmse,
            "pred_range": p_range,
        }
    )

    return {
        "warmup_epochs": warmup_epochs,
        "mc_epochs": mc_epochs,
        "rmse_clean": rmse,
        "pred_range": p_range,
        "reset_optimizer": reset_optimizer,
        "history": history,
        "rmse_after_warmup": history[0]["rmse_clean"] if warmup_epochs > 0 else None,
    }
