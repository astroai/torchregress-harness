#!/usr/bin/env python3
"""Diagnostic: Why does RC degrade at 60 epochs vs 30 on linear EIV at σx=2.0?

60-epoch benchmark: RC RMSE_clean = 0.452
30-epoch benchmark: RC RMSE_clean = 0.367
That's a 23% degradation with 2× training time.

Hypotheses:
  1. Classic overfitting — train loss drops while test (clean) error rises
  2. Adam drift — optimizer overshoots the optimum with fixed LR
  3. Stochastic noise — difference is within run-to-run variance

This diagnostic:
  - Trains an MLP on RC-calibrated data for 200 epochs
  - Tracks train loss and RMSE_clean at every epoch
  - Identifies the optimal early-stopping epoch
  - Reports RMSE_clean at epochs 30, 60, 120, 200
  - Runs multiple seeds to check variance

Run: uv run python tools/diagnose_rc_overfitting.py [--sigma_x 2.0] [--n_runs 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Allow running from either torchregress or torchregress-harness directory
_HERE = Path(__file__).resolve().parent
_TORCHREGRESS_SRC = _HERE.parent.parent / "torchregress" / "src"
if str(_TORCHREGRESS_SRC) not in sys.path:
    sys.path.insert(0, str(_TORCHREGRESS_SRC))

from torchregress.algorithms.rc import RegressionCalibration

# ── data generator (mirrors _load_synthetic_eiv in harness/datasets.py) ────────


def generate_linear_eiv_data(
    n_samples: int = 2000,
    n_features: int = 6,
    sigma_x: float = 2.0,
    sigma_y: float = 0.1,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_true = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    X_obs = (X_true + sigma_x * rng.standard_normal((n_samples, n_features))).astype(np.float32)
    y = (
        0.8 * X_true[:, 0]
        - 0.6 * X_true[:, 1]
        + 0.4 * X_true[:, 2]
        + sigma_y * rng.standard_normal(n_samples)
    ).astype(np.float32)

    n_train = int(n_samples * 0.7)
    return {
        "X_train": X_obs[:n_train],
        "y_train": y[:n_train].reshape(-1, 1),
        "X_test": X_obs[n_train:],
        "y_test": y[n_train:].reshape(-1, 1),
        "X_train_true": X_true[:n_train],
        "X_test_true": X_true[n_train:],
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
    }


# ── model (mirrors _MLP from harness suites/_backbones.py) ─────────────────────


def _hidden_size(n_features: int) -> int:
    return max(16, min(64, 2 ** (n_features - 1).bit_length()))


def make_mlp(in_dim: int, hidden: int, out_dim: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ── single-run diagnostic ─────────────────────────────────────────────────────


def run_one(
    data: dict[str, np.ndarray],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 123,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_np = data["X_train"]
    y_train_np = data["y_train"].ravel()
    X_test_true_np = data["X_test_true"]
    y_test_true_np = data["y_test"].ravel()
    sigma_x = float(data["sigma_x"])

    in_dim = X_train_np.shape[1]
    hidden = _hidden_size(in_dim)

    # ── 1. Fit RC ────────────────────────────────────────────
    X = torch.from_numpy(X_train_np).float()
    y = torch.from_numpy(y_train_np.reshape(-1, 1)).float()
    rc = RegressionCalibration(sigma_u=sigma_x)
    X_cal = rc.fit_transform(X)

    # ── 2. Train MLP on RC-calibrated data ───────────────────
    model = make_mlp(in_dim, hidden)
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(X_cal, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Pre-compute clean-X tensor for eval
    X_clean_t = torch.from_numpy(X_test_true_np).float()
    y_clean_t = torch.from_numpy(y_test_true_np).float()

    history: list[dict] = []

    model.train()
    for epoch in range(epochs):
        epoch_train_losses: list[float] = []
        for Xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_train_losses.append(float(loss.item()))

        # Evaluate every epoch for fine-grained trajectory
        model.eval()
        with torch.no_grad():
            pred_train = model(X_cal).squeeze(-1)
            train_loss = float(loss_fn(pred_train, y.squeeze(-1)).item())
            pred_clean = model(X_clean_t).squeeze(-1)
            rmse_clean = float(torch.sqrt(((pred_clean - y_clean_t) ** 2).mean()).item())

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "rmse_clean": rmse_clean,
            }
        )
        model.train()

    # Find best epoch by RMSE_clean
    best = min(history, key=lambda h: h["rmse_clean"])

    return {
        "best_epoch": best["epoch"],
        "best_rmse_clean": best["rmse_clean"],
        "rmse30": history[29]["rmse_clean"] if len(history) > 29 else None,
        "rmse60": history[59]["rmse_clean"] if len(history) > 59 else None,
        "rmse120": history[119]["rmse_clean"] if len(history) > 119 else None,
        "rmse200": history[-1]["rmse_clean"],
        "history": history,
    }


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose RC overfitting on linear EIV data",
    )
    parser.add_argument("--sigma_x", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 78)
    print("  RC Overfitting Diagnostic — Linear EIV (σx=%.1f, σy=0.1)" % args.sigma_x)
    print("=" * 78)
    print(f"  n_features=6, n_train=1400, n_test=600, epochs={args.epochs}")
    print(f"  lr={args.lr}, n_runs={args.n_runs}")
    print()

    results = []
    for run_idx in range(args.n_runs):
        seed = 123 + run_idx * 7
        data = generate_linear_eiv_data(sigma_x=args.sigma_x, seed=seed, n_samples=2000)
        r = run_one(data, epochs=args.epochs, lr=args.lr, seed=seed)
        results.append(r)

    # ── Aggregate results ────────────────────────────────────
    print(
        f"  {'Run':<6} {'Best(epoch)':<15} {'RMSE@30':<12} {'RMSE@60':<12} {'RMSE@120':<12} {'RMSE@200':<12}"
    )
    print(f"  {'─' * 6} {'─' * 15} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12}")
    for i, r in enumerate(results):
        best_str = f"{r['best_rmse_clean']:.4f}@{r['best_epoch']}"
        print(
            f"  {i + 1:<6} {best_str:<15} {r['rmse30'] or 0:.4f}       "
            f"  {r['rmse60'] or 0:.4f}       "
            f"  {r['rmse120'] or 0:.4f}       "
            f"  {r['rmse200']:.4f}"
        )

    # ── Summary statistics ───────────────────────────────────
    rmse30s = [r["rmse30"] for r in results if r["rmse30"] is not None]
    rmse60s = [r["rmse60"] for r in results if r["rmse60"] is not None]
    rmse200s = [r["rmse200"] for r in results]
    best_ep = [r["best_epoch"] for r in results]
    best_rmse = [r["best_rmse_clean"] for r in results]

    print()
    print("-" * 78)
    print("  AGGREGATE SUMMARY")
    print("-" * 78)
    print(
        f"  RMSE_clean @ epoch 30:  mean={np.mean(rmse30s):.4f}  "
        f"std={np.std(rmse30s):.4f}  range=[{min(rmse30s):.4f}, {max(rmse30s):.4f}]"
    )
    print(
        f"  RMSE_clean @ epoch 60:  mean={np.mean(rmse60s):.4f}  "
        f"std={np.std(rmse60s):.4f}  range=[{min(rmse60s):.4f}, {max(rmse60s):.4f}]"
    )
    print(
        f"  RMSE_clean @ epoch 200: mean={np.mean(rmse200s):.4f}  "
        f"std={np.std(rmse200s):.4f}  range=[{min(rmse200s):.4f}, {max(rmse200s):.4f}]"
    )
    print(
        f"  Best RMSE_clean:        mean={np.mean(best_rmse):.4f}  "
        f"std={np.std(best_rmse):.4f}  range=[{min(best_rmse):.4f}, {max(best_rmse):.4f}]"
    )
    print(
        f"  Best epoch:              mean={np.mean(best_ep):.1f}  "
        f"std={np.std(best_ep):.1f}  range=[{min(best_ep)}, {max(best_ep)}]"
    )

    # ── Trajectory for first run ─────────────────────────────
    print()
    print("-" * 78)
    print("  TRAJECTORY (Run 1): Train Loss vs RMSE_clean")
    print("-" * 78)
    h = results[0]["history"]
    # Show every 10 epochs
    print(f"  {'Epoch':>6} {'Train Loss':>12} {'RMSE_clean':>12}")
    print(f"  {'─' * 6} {'─' * 12} {'─' * 12}")
    for pt in h:
        if pt["epoch"] % 10 == 0 or pt["epoch"] < 10 or pt["epoch"] in (29, 59, 119, 199):
            marker = ""
            if pt["epoch"] == 29:
                marker = "  ← 30-epoch benchmark"
            elif pt["epoch"] == 59:
                marker = "  ← 60-epoch benchmark"
            elif pt["epoch"] == results[0]["best_epoch"]:
                marker = "  ← BEST"
            print(f"  {pt['epoch']:6d} {pt['train_loss']:12.6f} {pt['rmse_clean']:12.4f}{marker}")

    # ── Verdict ──────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)

    mean30 = np.mean(rmse30s)
    mean60 = np.mean(rmse60s)
    mean200 = np.mean(rmse200s)
    mean_best = np.mean(best_rmse)
    mean_best_ep = np.mean(best_ep)

    # Overfitting: does RMSE_clean increase while train loss decreases?
    last_train = h[-1]["train_loss"]
    mid_train = h[min(60, len(h) - 1)]["train_loss"]
    early_train = h[min(30, len(h) - 1)]["train_loss"]

    print(
        f"  Train loss:  epoch 30={early_train:.6f}  →  epoch 60={mid_train:.6f}  →  final={last_train:.6f}"
    )
    print(f"  RMSE_clean:  epoch 30={mean30:.4f}  →  epoch 60={mean60:.4f}  →  final={mean200:.4f}")
    print(f"  Best RMSE_clean: {mean_best:.4f} at epoch ≈ {mean_best_ep:.0f}")

    deg_30_60 = (mean60 - mean30) / mean30 * 100 if mean30 > 0 else 0
    deg_30_200 = (mean200 - mean30) / mean30 * 100 if mean30 > 0 else 0

    print()
    if last_train < mid_train < early_train and mean200 > mean60 > mean30:
        print(
            f"  ❌ CLASSIC OVERFITTING: Train loss keeps dropping "
            f"({early_train:.6f}→{last_train:.6f}) but RMSE_clean rises "
            f"({mean30:.4f}→{mean200:.4f}, +{deg_30_200:.0f}%)"
        )
        print(
            f"  → Early stopping at epoch ≈ {mean_best_ep:.0f} would recover "
            f"RMSE_clean ≈ {mean_best:.4f}"
        )
    elif abs(mean60 - mean30) < 0.02:
        print(
            f"  ✅ NO OVERFITTING: RMSE_clean stable "
            f"(epoch 30={mean30:.4f}, 60={mean60:.4f}) — "
            f"the 30-vs-60 benchmark difference may be within run-to-run variance"
        )
    elif mean200 > mean30 * 1.1:
        print(
            f"  ⚠️  MODERATE DEGRADATION: RMSE_clean rises {deg_30_200:.0f}% "
            f"from epoch 30→200 with continued training"
        )
        if mean_best_ep < 60:
            print(f"  → Early stopping around epoch {mean_best_ep:.0f} recommended")
        else:
            print(
                f"  → Best epoch ({mean_best_ep:.0f}) is after 60 — degradation may "
                f"not be monotonic"
            )
    else:
        print(
            f"  ⚠️  AMBIGUOUS: Degradation from 30→60 epochs ({deg_30_60:.1f}%) but "
            f"not clearly overfitting. Run-to-run variance may dominate."
        )

    # ── RC diagnostics ───────────────────────────────────────
    print()
    print("-" * 78)
    print("  RC DIAGNOSTICS")
    print("-" * 78)

    # Show reliability and shrinkage for a representative run
    data0 = generate_linear_eiv_data(sigma_x=args.sigma_x, seed=123, n_samples=2000)
    X_t = torch.from_numpy(data0["X_train"]).float()
    rc0 = RegressionCalibration(sigma_u=args.sigma_x)
    rc0.fit(X_t)

    reliability_diag = torch.diagonal(rc0.reliability_matrix)
    signal_diag = torch.diagonal(rc0.signal_covariance)
    print(f"  Reliability diagonal:     {reliability_diag.numpy()}")
    print(f"  Signal covariance diag:   {signal_diag.numpy()}")
    print(
        f"  Avg reliability:          {float(reliability_diag.mean()):.4f} "
        f"(={float(reliability_diag.mean()) * 100:.1f}% shrinkage toward mean)"
    )

    # Check variance of calibrated inputs vs original
    X_orig = data0["X_train"]
    var_orig = float(np.var(X_orig))
    X_cal = rc0.transform(X_t).numpy()
    var_cal = float(np.var(X_cal))
    print(f"  Var(X_obs):               {var_orig:.4f}")
    print(
        f"  Var(X_cal):               {var_cal:.4f} ({var_cal / var_orig * 100:.1f}% of original)"
    )


if __name__ == "__main__":
    main()
