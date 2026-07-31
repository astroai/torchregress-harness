#!/usr/bin/env python3
"""Diagnostic: Does TLS_Init+FunctionalEIV preserve the structural TLS β?

This script generates synthetic linear EIV data, computes TLS β, initializes
a linear model with those weights, fine-tunes with FunctionalEIVLoss, and
tracks the β trajectory to determine whether the loss preserves or drifts
from the TLS solution.  Also compares RC+FunctionalEIV (the recommended
composability method) as a positive control.

Run: uv run python tools/diagnose_tls_eiv_drift.py [--sigma_x 0.2|0.5|1.0|2.0]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torchregress.algorithms import RegressionCalibration
from torchregress.losses import FunctionalEIVLoss

# ── synthetic data generator ──────────────────────────────────────────


def generate_eiv_data(
    n_samples: int = 2000,
    d: int = 3,
    sigma_x: float = 0.5,
    sigma_y: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate linear EIV data with known true coefficients."""
    rng = np.random.default_rng(seed)

    # Ground-truth coefficients: y = X_true @ beta + intercept + noise
    beta_true = rng.normal(1.0, 0.5, size=(d,)).astype(np.float64)
    intercept_true = 0.5

    # Clean inputs
    X_true = rng.normal(0.0, 1.0, size=(n_samples, d)).astype(np.float64)

    # Clean targets
    y_true = X_true @ beta_true + intercept_true
    y_obs = y_true + rng.normal(0.0, sigma_y, size=n_samples).astype(np.float64)

    # Noisy observed inputs
    X_obs = X_true + rng.normal(0.0, sigma_x, size=(n_samples, d)).astype(np.float64)

    return {
        "X_train": X_obs[: n_samples // 2],
        "y_train": y_obs[: n_samples // 2].reshape(-1, 1),
        "X_test": X_obs[n_samples // 2 :],
        "y_test": y_obs[n_samples // 2 :].reshape(-1, 1),
        "X_test_true": X_true[n_samples // 2 :],
        "y_test_true": y_true[n_samples // 2 :].reshape(-1, 1),
        "beta_true": beta_true,
        "intercept_true": intercept_true,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
    }


# ── TLS solver (numpy SVD) ────────────────────────────────────────────


def tls_solve(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Total Least Squares via SVD of augmented [X | y]."""
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Z = np.column_stack([X - X_mean, y.ravel() - y_mean])
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt[-1]
    beta = -V[:-1] / V[-1]
    intercept = float(y_mean - X_mean @ beta)
    return beta, intercept


# ── OLS solver ────────────────────────────────────────────────────────


def ols_solve(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Ordinary Least Squares (naive, attenuated)."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y.ravel(), rcond=None)
    return coef[1:], float(coef[0])


# ── training loop ─────────────────────────────────────────────────────


def train_eiv(
    model: nn.Module,
    loss_fn: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    track_beta_every: int = 5,
) -> list[dict]:
    """Train with EIV loss, tracking β and metrics each epoch."""
    n = X.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_losses = []
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(X[idx], y[idx])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        if epoch % track_beta_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                w = model.weight.detach().cpu().numpy().ravel()
                b = model.bias.detach().cpu().item()
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(epoch_losses)),
                    "beta": w.copy(),
                    "intercept": b,
                }
            )

    return history


# ── metrics ───────────────────────────────────────────────────────────


def eval_rmse(beta: np.ndarray, intercept: float, X: np.ndarray, y: np.ndarray) -> float:
    """RMSE of linear model β, intercept on (X, y)."""
    pred = X @ beta + intercept
    return float(np.sqrt(np.mean((pred - y.ravel()) ** 2)))


def beta_distance(beta: np.ndarray, beta_ref: np.ndarray) -> float:
    """Euclidean distance between two β vectors."""
    return float(np.sqrt(np.sum((beta - beta_ref) ** 2)))


# ── main diagnostic ───────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="TLS_Init+FunctionalEIV β-drift diagnostic")
    parser.add_argument("--sigma_x", type=float, default=0.5, help="Input noise std (default: 0.5)")
    parser.add_argument(
        "--sigma_y", type=float, default=0.1, help="Target noise std (default: 0.1)"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs (default: 200)")
    parser.add_argument("--d", type=int, default=3, help="Feature dimension (default: 3)")
    args = parser.parse_args()

    torch.manual_seed(123)
    np.random.seed(123)

    sigma_x = args.sigma_x
    sigma_y = args.sigma_y
    n_epochs = args.epochs
    d = args.d

    data = generate_eiv_data(n_samples=4000, d=d, sigma_x=sigma_x, sigma_y=sigma_y)
    beta_true = data["beta_true"]
    intercept_true = data["intercept_true"]

    X_train_np = data["X_train"]
    y_train_np = data["y_train"]
    X_test_true = data["X_test_true"]
    y_test_true = data["y_test_true"]

    # ── Reference solutions ───────────────────────────────────────
    beta_tls, intercept_tls = tls_solve(X_train_np, y_train_np)
    beta_ols, intercept_ols = ols_solve(X_train_np, y_train_np)

    print("=" * 75)
    print("  TLS_Init+FunctionalEIV β-Drift Diagnostic")
    print("=" * 75)
    print(f"  n_train={len(X_train_np)}, d={d}, σx={sigma_x}, σy={sigma_y}, epochs={n_epochs}")
    print(f"  True β:       {np.array2string(beta_true, precision=4)}")
    print(f"  True intercept: {intercept_true:.4f}")
    print(
        f"  TLS β:        {np.array2string(beta_tls, precision=4)} (dist to true: {beta_distance(beta_tls, beta_true):.6f})"
    )
    print(
        f"  OLS β:        {np.array2string(beta_ols, precision=4)} (dist to true: {beta_distance(beta_ols, beta_true):.6f})"
    )
    print()

    X_t = torch.from_numpy(X_train_np).float()
    y_t = torch.from_numpy(y_train_np).float()

    # ── Method 1: TLS_Init+FunctionalEIV ───────────────────────────
    model = nn.Linear(d, 1)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(beta_tls).float().unsqueeze(0))
        model.bias.copy_(torch.tensor([intercept_tls]).float())

    loss_fn = FunctionalEIVLoss(model=model, sigma_x=sigma_x, sigma_y=sigma_y)
    track_every = max(1, n_epochs // 20)
    history = train_eiv(
        model, loss_fn, X_t, y_t, epochs=n_epochs, lr=1e-4, track_beta_every=track_every
    )
    beta_tls_init = history[-1]["beta"]
    intercept_tls_init = history[-1]["intercept"]

    # ── Method 2: RC+FunctionalEIV (positive control — ROADMAP recommends) ─
    rc = RegressionCalibration(sigma_u=sigma_x)
    X_cal_train = rc.fit_transform(X_t)
    model_rc = nn.Linear(d, 1)
    loss_fn_rc = FunctionalEIVLoss(model=model_rc, sigma_x=sigma_x * 0.3, sigma_y=sigma_y)
    hist_rc = train_eiv(
        model_rc,
        loss_fn_rc,
        X_cal_train,
        y_t,
        epochs=n_epochs,
        lr=1e-3,
        track_beta_every=track_every,
    )
    beta_rc_feiv = hist_rc[-1]["beta"]
    intercept_rc_feiv = hist_rc[-1]["intercept"]

    # ── Method 3: MSE from scratch (attenuation baseline) ──────────
    model_mse = nn.Linear(d, 1)
    opt_mse = torch.optim.Adam(model_mse.parameters(), lr=1e-3)
    for _ in range(n_epochs):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 64):
            idx = perm[i : i + 64]
            opt_mse.zero_grad(set_to_none=True)
            loss = ((model_mse(X_t[idx]) - y_t[idx]) ** 2).mean()
            loss.backward()
            opt_mse.step()
    with torch.no_grad():
        beta_mse = model_mse.weight.numpy().ravel()
        intercept_mse = model_mse.bias.item()

    # ── Results ────────────────────────────────────────────────────
    print("-" * 75)
    print("  FINAL RESULTS (after training)")
    print("-" * 75)
    print(f"  {'Method':<32} {'β dist to true':>14} {'β dist to TLS':>14} {'RMSE_clean':>12}")
    print(f"  {'─' * 32} {'─' * 14} {'─' * 14} {'─' * 12}")

    methods = [
        ("TLS (no training)", beta_tls, intercept_tls),
        ("OLS (no training)", beta_ols, intercept_ols),
        ("MSE trained (from scratch)", beta_mse, intercept_mse),
        ("TLS_Init+FunctionalEIV", beta_tls_init, intercept_tls_init),
        ("RC+FunctionalEIV (recommended)", beta_rc_feiv, intercept_rc_feiv),
    ]

    for name, beta, intercept in methods:
        d_true = beta_distance(beta, beta_true)
        d_tls = beta_distance(beta, beta_tls)
        rmse = eval_rmse(beta, intercept, X_test_true, y_test_true)
        print(f"  {name:<32} {d_true:14.6f} {d_tls:14.6f} {rmse:12.6f}")

    # ── Trajectory ─────────────────────────────────────────────────
    print()
    print("-" * 75)
    print("  β TRAJECTORY (TLS_Init+FunctionalEIV)")
    print("-" * 75)
    print(
        f"  {'Epoch':>6} {'Loss':>10} {'β dist to TLS':>14} {'β dist to True':>14} {'RMSE_clean':>12}"
    )
    for h in history:
        d_tls = beta_distance(h["beta"], beta_tls)
        d_true = beta_distance(h["beta"], beta_true)
        rmse = eval_rmse(h["beta"], h["intercept"], X_test_true, y_test_true)
        print(f"  {h['epoch']:6d} {h['loss']:10.4f} {d_tls:14.6f} {d_true:14.6f} {rmse:12.6f}")

    # ── Verdict ────────────────────────────────────────────────────
    print()
    print("-" * 75)
    print("  VERDICT")
    print("-" * 75)

    drift = beta_distance(beta_tls_init, beta_tls)
    initial_drift = beta_distance(history[0]["beta"], beta_tls)

    if drift < 0.01:
        status = "✅ TLS β PRESERVED — drift < 0.01"
    elif drift < 0.05:
        status = "⚠️  TLS β mostly preserved — minor drift"
    elif drift < 0.20:
        status = "⚠️  Moderate drift from TLS β"
    else:
        status = "❌ TLS β NOT preserved — significant drift"

    d_to_tls = beta_distance(beta_tls_init, beta_tls)
    d_to_ols = beta_distance(beta_tls_init, beta_ols)
    attraction = "closer to TLS" if d_to_tls < d_to_ols else "closer to OLS (attenuation direction)"

    print(f"  Initial β distance to TLS:   {initial_drift:.6f}")
    print(f"  Final β distance to TLS:     {drift:.6f}")
    print(f"  Status: {status}")
    print(f"  Attraction direction: {attraction}")

    rmse_tls = eval_rmse(beta_tls, intercept_tls, X_test_true, y_test_true)
    rmse_tls_init = eval_rmse(beta_tls_init, intercept_tls_init, X_test_true, y_test_true)
    rmse_mse_val = eval_rmse(beta_mse, intercept_mse, X_test_true, y_test_true)
    rmse_rc_feiv = eval_rmse(beta_rc_feiv, intercept_rc_feiv, X_test_true, y_test_true)

    print()
    print(f"  TLS RMSE_clean:                    {rmse_tls:.6f}")
    print(f"  TLS_Init+FuncEIV RMSE_clean:       {rmse_tls_init:.6f}")
    print(f"  MSE (scratch) RMSE_clean:          {rmse_mse_val:.6f}")
    print(f"  RC+FunctionalEIV RMSE_clean:       {rmse_rc_feiv:.6f}")

    # Compare RC+FunctionalEIV vs TLS
    if rmse_rc_feiv <= rmse_tls * 1.05:
        print("  → RC+FunctionalEIV nearly matches TLS on clean data (RC does the work)")
    elif rmse_rc_feiv <= rmse_mse_val * 0.80:
        print("  → RC+FunctionalEIV significantly improves over MSE")
    else:
        print("  → RC+FunctionalEIV partially corrects attenuation")

    if rmse_tls_init <= rmse_tls * 1.01:
        print("  → TLS_Init+FunctionalEIV matches TLS on clean data")
    elif rmse_tls_init <= rmse_mse_val * 0.95:
        print("  → TLS_Init+FunctionalEIV slightly improves over MSE, some drift from TLS")
    else:
        print("  → TLS_Init+FunctionalEIV drifts toward MSE solution — NOT recommended")

    # Summary
    print()
    improvement_over_mse = (rmse_mse_val - rmse_rc_feiv) / rmse_mse_val * 100
    if improvement_over_mse > 0:
        print(f"  RC+FunctionalEIV improves RMSE_clean by {improvement_over_mse:.1f}% over MSE")
    tls_init_improvement = (rmse_mse_val - rmse_tls_init) / rmse_mse_val * 100
    if tls_init_improvement > 0:
        print(f"  TLS_Init+FuncEIV improves RMSE_clean by {tls_init_improvement:.1f}% over MSE")
    else:
        print("  TLS_Init+FuncEIV DEGRADES RMSE_clean vs MSE (the drift is harmful)")


if __name__ == "__main__":
    main()
