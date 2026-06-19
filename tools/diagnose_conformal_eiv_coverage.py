#!/usr/bin/env python3
"""Diagnostic: Does SplitCP+InputNoiseMC undercover on nonlinear EIV data?

ROADMAP issue 6: "InputNoiseMC may undercover on nonlinear data due to
test-time MC variance inflation."

This script loads nonlinear EIV data, trains InputNoiseMC and FunctionalEIV
models, computes split-conformal prediction intervals at α=0.1, and reports
empirical coverage vs nominal 90% target.  InputNoiseMC's train-time MC
perturbations may create a distribution mismatch vs. test-time predictions,
inflating residuals on the calibration set and producing over-wide intervals
that still achieve good coverage — or, if the mismatch is asymmetric,
producing undercovered intervals.

Run: uv run python tools/diagnose_conformal_eiv_coverage.py [--sigma_x 0.5|1.0|2.0]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from torchregress.losses import (
    FunctionalEIVLoss,
    InputNoiseMarginalizationLoss,
    WeightedMSELoss,
)


# ── data generator ────────────────────────────────────────────────────


def generate_nonlinear_eiv_data(
    n_samples: int = 3000,
    sigma_x: float = 0.5,
    sigma_y: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate nonlinear EIV data matching the harness dataset generator.

    y = sin(2π·x₀) + 0.5·cos(4π·x₁) + 0.3·tanh(2·x₂) + ε
    """
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
    n_cal = int(n_train * 0.25)

    return {
        "X_train": X_obs[: n_train - n_cal],
        "y_train": y[: n_train - n_cal].reshape(-1, 1),
        "X_cal": X_obs[n_train - n_cal : n_train],
        "y_cal": y[n_train - n_cal : n_train].reshape(-1, 1),
        "X_test": X_obs[n_train:],
        "y_test": y[n_train:].reshape(-1, 1),
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
    }


# ── training helpers ──────────────────────────────────────────────────


def _hidden_size(in_dim: int) -> int:
    return max(16, min(in_dim * 4, 64))


def train_input_noise_mc(
    X: torch.Tensor, y: torch.Tensor, sigma_x: float, epochs: int = 100, lr: float = 1e-3
) -> nn.Module:
    """Train an MLP with InputNoiseMarginalizationLoss + WeightedMSE."""
    d = X.shape[1]
    hidden = _hidden_size(d)
    model = nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )

    loss_fn = InputNoiseMarginalizationLoss(
        model=model, base_loss=WeightedMSELoss(),
        sigma_x=sigma_x, n_samples=20,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(X[idx], y[idx])
            loss.backward()
            opt.step()

    model.eval()
    return model


def train_functional_eiv(
    X: torch.Tensor, y: torch.Tensor, sigma_x: float, sigma_y: float = 0.1,
    epochs: int = 100, lr: float = 1e-3,
) -> nn.Module:
    """Train an MLP with FunctionalEIVLoss (analytical, gradient-based)."""
    d = X.shape[1]
    hidden = _hidden_size(d)
    model = nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )

    loss_fn = FunctionalEIVLoss(model=model, sigma_x=sigma_x, sigma_y=sigma_y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(X[idx], y[idx])
            loss.backward()
            opt.step()

    model.eval()
    return model


# ── prediction helpers ────────────────────────────────────────────────


def predict_point(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Standard point prediction on X."""
    with torch.no_grad():
        X_t = torch.from_numpy(X).float()
        return model(X_t).squeeze(-1).numpy()


# ── conformal evaluation ──────────────────────────────────────────────


def evaluate_conformal(
    model: nn.Module,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float = 0.10,
) -> dict[str, float]:
    """Split-conformal: compute q from calibration residuals, evaluate on test.

    Returns coverage (actual fraction in [pred-Δ, pred+Δ]) and interval width.
    """
    y_pred_cal = predict_point(model, X_cal)
    residuals_cal = np.abs(y_cal.ravel() - y_pred_cal)
    q = float(np.quantile(residuals_cal, 1.0 - alpha))

    y_pred_test = predict_point(model, X_test)
    lower = y_pred_test - q
    upper = y_pred_test + q
    y_test_vec = y_test.ravel()

    covered = (lower <= y_test_vec) & (y_test_vec <= upper)
    coverage = float(covered.mean())
    width = float((upper - lower).mean())

    return {"coverage": coverage, "interval_width": width, "q": q, "n_cal": len(X_cal)}


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose SplitCP+InputNoiseMC coverage on nonlinear EIV data"
    )
    parser.add_argument("--sigma_x", type=float, default=0.5, help="Input noise std (default: 0.5)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of independent runs (default: 3)")
    args = parser.parse_args()

    sigma_x = args.sigma_x
    n_epochs = args.epochs
    n_runs = args.n_runs

    print("=" * 75)
    print("  SplitCP Coverage Diagnostic — Nonlinear EIV, σx={:.1f}".format(sigma_x))
    print("=" * 75)
    print(f"  Data: y = sin(2π·x₀) + 0.5·cos(4π·x₁) + 0.3·tanh(2·x₂) + ε")
    print(f"  σx={sigma_x}, σy=0.1, n_runs={n_runs}")
    print()

    coverage_input_noise: list[float] = []
    width_input_noise: list[float] = []
    coverage_func_eiv: list[float] = []
    width_func_eiv: list[float] = []

    for run in range(n_runs):
        np.random.seed(run)
        torch.manual_seed(run)

        data = generate_nonlinear_eiv_data(
            n_samples=3000, sigma_x=sigma_x, seed=run
        )

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_cal = data["X_cal"]
        y_cal = data["y_cal"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        X_t = torch.from_numpy(X_train).float()
        y_t = torch.from_numpy(y_train).float()

        # ── InputNoiseMC ─────────────────────────────────────────
        model_mc = train_input_noise_mc(
            X_t, y_t, sigma_x=sigma_x, epochs=n_epochs
        )
        r_mc = evaluate_conformal(model_mc, X_cal, y_cal, X_test, y_test)
        coverage_input_noise.append(r_mc["coverage"])
        width_input_noise.append(r_mc["interval_width"])

        # ── FunctionalEIV ────────────────────────────────────────
        model_feiv = train_functional_eiv(
            X_t, y_t, sigma_x=sigma_x, epochs=n_epochs
        )
        r_feiv = evaluate_conformal(model_feiv, X_cal, y_cal, X_test, y_test)
        coverage_func_eiv.append(r_feiv["coverage"])
        width_func_eiv.append(r_feiv["interval_width"])

        print(f"  Run {run + 1}:  CP@InputNoiseMC  coverage={r_mc['coverage']:.4f}  width={r_mc['interval_width']:.4f}")
        print(f"         CP@FunctionalEIV  coverage={r_feiv['coverage']:.4f}  width={r_feiv['interval_width']:.4f}")

    print()
    print("-" * 75)
    print("  AGGREGATE SUMMARY ({n} runs)".format(n=n_runs))
    print("-" * 75)
    print(f"  {'Method':<28} {'Coverage':>12} {'Width':>12}")
    print(f"  {'─'*28} {'─'*12} {'─'*12}")

    mean_cov_mc = float(np.mean(coverage_input_noise))
    std_cov_mc = float(np.std(coverage_input_noise))
    mean_width_mc = float(np.mean(width_input_noise))
    print(f"  SplitCP+InputNoiseMC      {mean_cov_mc:8.4f}±{std_cov_mc:.4f}  {mean_width_mc:8.4f}")

    mean_cov_feiv = float(np.mean(coverage_func_eiv))
    std_cov_feiv = float(np.std(coverage_func_eiv))
    mean_width_feiv = float(np.mean(width_func_eiv))
    print(f"  SplitCP+FunctionalEIV     {mean_cov_feiv:8.4f}±{std_cov_feiv:.4f}  {mean_width_feiv:8.4f}")

    print()
    print("-" * 75)
    print("  VERDICT")
    print("-" * 75)
    nominal = 0.90
    tol = 0.02  # allow ±2% tolerance around nominal

    mc_ok = abs(mean_cov_mc - nominal) <= tol
    feiv_ok = abs(mean_cov_feiv - nominal) <= tol

    if mc_ok:
        print(f"  ✅ SplitCP+InputNoiseMC: {mean_cov_mc:.3f} ≈ nominal {nominal} — coverage OK")
    elif mean_cov_mc < nominal - tol:
        under = (nominal - mean_cov_mc) * 100
        print(f"  ❌ SplitCP+InputNoiseMC UNDERCOVERS: {mean_cov_mc:.3f} vs nominal {nominal} ({under:.1f}% gap)")
    else:
        over = (mean_cov_mc - nominal) * 100
        print(f"  ⚠️  SplitCP+InputNoiseMC OVERCOVERS: {mean_cov_mc:.3f} vs nominal {nominal} ({over:.1f}% excess)")

    if feiv_ok:
        print(f"  ✅ SplitCP+FunctionalEIV: {mean_cov_feiv:.3f} ≈ nominal {nominal} — coverage OK")
    elif mean_cov_feiv < nominal - tol:
        under = (nominal - mean_cov_feiv) * 100
        print(f"  ❌ SplitCP+FunctionalEIV UNDERCOVERS: {mean_cov_feiv:.3f} vs nominal {nominal} ({under:.1f}% gap)")
    else:
        over = (mean_cov_feiv - nominal) * 100
        print(f"  ⚠️  SplitCP+FunctionalEIV OVERCOVERS: {mean_cov_feiv:.3f} vs nominal {nominal} ({over:.1f}% excess)")

    # Compare widths
    width_diff_pct = (mean_width_mc - mean_width_feiv) / mean_width_feiv * 100
    if width_diff_pct > 0:
        print(f"  InputNoiseMC intervals are {width_diff_pct:.1f}% WIDER than FunctionalEIV")
    else:
        print(f"  InputNoiseMC intervals are {abs(width_diff_pct):.1f}% NARROWER than FunctionalEIV")


if __name__ == "__main__":
    main()
