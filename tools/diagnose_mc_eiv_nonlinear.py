#!/usr/bin/env python3
"""Diagnostic: Why does FunctionalEIV_MC fail on nonlinear EIV data at σx=0.5?

RMSE_clean: 1.406 (MC) vs 0.827 (MSE) vs 0.827 (analytical FunctionalEIV).
This is a 70% degradation.  Investigate whether the MC variance estimates
create a pathological loss landscape for ReLU networks.

Run: uv run python tools/diagnose_mc_eiv_nonlinear.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchregress.losses import FunctionalEIVLoss, WeightedMSELoss

# ── data generator ────────────────────────────────────────────────────


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


# ── model ─────────────────────────────────────────────────────────────


def make_mlp(in_dim: int, hidden: int = 32, out_dim: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


# ── training with diagnostics ────────────────────────────────────────


def train_and_diagnose(
    model: nn.Module,
    loss_fn: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    X_test_true: torch.Tensor,
    y_test_true: torch.Tensor,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    label: str = "",
) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    history: list[dict] = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_losses = []
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            opt.zero_grad(set_to_none=True)
            # EIV losses (FunctionalEIVLoss, etc.) take (x_obs, target)
            # Standard losses (MSELoss, etc.) take (prediction, target)
            if isinstance(loss_fn, FunctionalEIVLoss):
                loss = loss_fn(X[idx], y[idx])
            else:
                loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_train = model(X).squeeze(-1)
                pred_clean = model(X_test_true).squeeze(-1)
                rmse_clean = float(torch.sqrt(((pred_clean - y_test_true.squeeze(-1)) ** 2).mean()))
                # Weight norms
                w_norms = []
                for p in model.parameters():
                    if p.dim() >= 2:
                        w_norms.append(float(p.norm().item()))

            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(epoch_losses)),
                    "rmse_clean": rmse_clean,
                    "weight_norm_L1": w_norms[0] if w_norms else 0,
                    "weight_norm_L2": w_norms[1] if len(w_norms) > 1 else 0,
                    "weight_norm_L3": w_norms[2] if len(w_norms) > 2 else 0,
                }
            )
            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred_clean = model(X_test_true).squeeze(-1)
        pred_noisy = model(X).squeeze(-1)
        rmse_clean = float(torch.sqrt(((pred_clean - y_test_true.squeeze(-1)) ** 2).mean()))
        rmse_noisy = float(torch.sqrt(((pred_noisy - y.squeeze(-1)) ** 2).mean()))
        pred_std = float(pred_clean.std())
        pred_range = float(pred_clean.max() - pred_clean.min())

    return {
        "label": label,
        "rmse_clean": rmse_clean,
        "rmse_noisy": rmse_noisy,
        "pred_std": pred_std,
        "pred_range": pred_range,
        "history": history,
    }


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    torch.manual_seed(123)

    data = generate_nonlinear_eiv_data(n_samples=2000, sigma_x=0.5, seed=123)
    sigma_x = float(data["sigma_x"])
    sigma_y = float(data["sigma_y"])
    in_dim = data["X_train"].shape[1]

    X_t = torch.from_numpy(data["X_train"]).float()
    y_t = torch.from_numpy(data["y_train"]).float()
    X_test_true_t = torch.from_numpy(data["X_test_true"]).float()
    y_test_true_t = torch.from_numpy(data["y_test"]).float()

    results = []

    # ── Method 1: MSE ───────────────────────────────────────────────
    print("Training MSE...")
    model_mse = make_mlp(in_dim)
    loss_mse = WeightedMSELoss()
    r = train_and_diagnose(
        model_mse,
        loss_mse,
        X_t,
        y_t,
        X_test_true_t,
        y_test_true_t,
        epochs=100,
        lr=1e-3,
        label="MSE",
    )
    results.append(r)

    # ── Method 2: FunctionalEIV (analytical) ─────────────────────────
    print("Training FunctionalEIV (analytical)...")
    model_feiv = make_mlp(in_dim)
    loss_feiv = FunctionalEIVLoss(
        model=model_feiv,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        monte_carlo=False,
    )
    r = train_and_diagnose(
        model_feiv,
        loss_feiv,
        X_t,
        y_t,
        X_test_true_t,
        y_test_true_t,
        epochs=100,
        lr=1e-3,
        label="FunctionalEIV",
    )
    results.append(r)

    # ── Method 3: FunctionalEIV_MC n_samples=20 ─────────────────────
    print("Training FunctionalEIV_MC (n_samples=20)...")
    model_mc20 = make_mlp(in_dim)
    loss_mc20 = FunctionalEIVLoss(
        model=model_mc20,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        monte_carlo=True,
        n_samples=20,
    )
    r = train_and_diagnose(
        model_mc20,
        loss_mc20,
        X_t,
        y_t,
        X_test_true_t,
        y_test_true_t,
        epochs=100,
        lr=1e-3,
        label="FuncEIV_MC_n20",
    )
    results.append(r)

    # ── Method 4: FunctionalEIV_MC n_samples=200 ────────────────────
    print("Training FunctionalEIV_MC (n_samples=200)...")
    model_mc200 = make_mlp(in_dim)
    loss_mc200 = FunctionalEIVLoss(
        model=model_mc200,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        monte_carlo=True,
        n_samples=200,
    )
    r = train_and_diagnose(
        model_mc200,
        loss_mc200,
        X_t,
        y_t,
        X_test_true_t,
        y_test_true_t,
        epochs=100,
        lr=1e-3,
        label="FuncEIV_MC_n200",
    )
    results.append(r)

    # ── Method 5: FunctionalEIV_MC n_samples=200, GELU activations ──
    print("Training FunctionalEIV_MC (GELU, n_samples=200)...")
    model_gelu = nn.Sequential(
        nn.Linear(in_dim, 32),
        nn.GELU(),
        nn.Linear(32, 32),
        nn.GELU(),
        nn.Linear(32, 1),
    )
    loss_gelu = FunctionalEIVLoss(
        model=model_gelu,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        monte_carlo=True,
        n_samples=200,
    )
    r = train_and_diagnose(
        model_gelu,
        loss_gelu,
        X_t,
        y_t,
        X_test_true_t,
        y_test_true_t,
        epochs=100,
        lr=1e-3,
        label="FuncEIV_MC_GELU_n200",
    )
    results.append(r)

    # ── Print results ──────────────────────────────────────────────
    print()
    print("=" * 75)
    print("  FunctionalEIV_MC Nonlinear Failure Diagnostic")
    print("=" * 75)
    print("  Data: y = sin(2π·x₀) + 0.5·cos(4π·x₁) + 0.3·tanh(2·x₂) + ε")
    print("  σx=0.5, σy=0.1, n_train=1000, epochs=100")
    print()
    print(
        f"  {'Method':<28} {'RMSE_clean':>12} {'RMSE_noisy':>12} {'pred_std':>10} {'pred_range':>10}"
    )
    print(f"  {'─' * 28} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 10}")
    for r in results:
        print(
            f"  {r['label']:<28} {r['rmse_clean']:12.4f} {r['rmse_noisy']:12.4f} {r['pred_std']:10.4f} {r['pred_range']:10.4f}"
        )

    # Check for collapsed predictions
    print()
    print("-" * 75)
    print("  DIAGNOSIS")
    print("-" * 75)
    for r in results:
        collapse = r["pred_range"] < 0.1 or r["pred_std"] < 0.05
        if collapse:
            print(
                f"  ❌ {r['label']}: COLLAPSED — pred_range={r['pred_range']:.4f}, pred_std={r['pred_std']:.4f}"
            )
        else:
            normal = r["pred_range"] > 1.0
            if normal:
                print(
                    f"  ✅ {r['label']}: Normal — pred_range={r['pred_range']:.4f}, pred_std={r['pred_std']:.4f}"
                )
            else:
                print(
                    f"  ⚠️  {r['label']}: Low-variance — pred_range={r['pred_range']:.4f}, pred_std={r['pred_std']:.4f}"
                )

    # Compare n_samples effect
    mc20 = results[2]
    mc200 = results[3]
    gelu = results[4]
    mse = results[0]
    feiv = results[1]

    print()
    print(f"  MSE baseline RMSE_clean:            {mse['rmse_clean']:.4f}")
    print(f"  Analytical FuncEIV RMSE_clean:       {feiv['rmse_clean']:.4f}")
    print(f"  MC n=20 RMSE_clean:                  {mc20['rmse_clean']:.4f}")
    print(f"  MC n=200 RMSE_clean:                 {mc200['rmse_clean']:.4f}")
    print(f"  MC GELU n=200 RMSE_clean:            {gelu['rmse_clean']:.4f}")

    if mc200["rmse_clean"] < mc20["rmse_clean"] * 0.8:
        print("  → More MC samples SIGNIFICANTLY help — noise in variance estimates is the issue")
    elif mc200["rmse_clean"] < mc20["rmse_clean"] * 0.95:
        print("  → More MC samples slightly help — some benefit from reduced variance noise")
    else:
        print("  → More MC samples don't help — the issue is deeper than sample count")

    if gelu["rmse_clean"] < mc200["rmse_clean"] * 0.9:
        print("  → GELU activations SIGNIFICANTLY help — ReLU threshold crossing is the issue")
    elif gelu["rmse_clean"] < mc200["rmse_clean"] * 0.98:
        print("  → GELU activations slightly help — some benefit from smooth activations")
    else:
        print("  → GELU activations don't help — the issue is not ReLU-specific")

    # Weight norm trajectory
    print()
    print("-" * 75)
    print("  WEIGHT NORM TRAJECTORY (L1 layer)")
    print("-" * 75)
    print(f"  {'Epoch':>6} {'MSE':>10} {'FuncEIV':>10} {'MC_n20':>10} {'MC_n200':>10} {'GELU':>10}")
    for i in range(len(mse["history"])):
        e = mse["history"][i]["epoch"]
        w_mse = mse["history"][i].get("weight_norm_L1", 0)
        w_feiv = feiv["history"][i].get("weight_norm_L1", 0)
        w_mc20 = mc20["history"][i].get("weight_norm_L1", 0)
        w_mc200 = mc200["history"][i].get("weight_norm_L1", 0)
        w_gelu = gelu["history"][i].get("weight_norm_L1", 0)
        print(
            f"  {e:6d} {w_mse:10.4f} {w_feiv:10.4f} {w_mc20:10.4f} {w_mc200:10.4f} {w_gelu:10.4f}"
        )

    # Final RMSE_clean trajectory
    print()
    print("-" * 75)
    print("  RMSE_clean TRAJECTORY")
    print("-" * 75)
    print(f"  {'Epoch':>6} {'MSE':>10} {'FuncEIV':>10} {'MC_n20':>10} {'MC_n200':>10} {'GELU':>10}")
    for i in range(len(mse["history"])):
        e = mse["history"][i]["epoch"]
        print(
            f"  {e:6d} {mse['history'][i]['rmse_clean']:10.4f} {feiv['history'][i]['rmse_clean']:10.4f} {mc20['history'][i]['rmse_clean']:10.4f} {mc200['history'][i]['rmse_clean']:10.4f} {gelu['history'][i]['rmse_clean']:10.4f}"
        )


if __name__ == "__main__":
    main()
