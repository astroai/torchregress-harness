#!/usr/bin/env python3
"""Diagnostic: Hybrid FunctionalEIV — Jacobian variance + MC mean.

The pure MC path fails on nonlinear EIV because empirical variance from
perturbed outputs creates a pathological loss landscape.  The analytical
Jacobian path is stable but has no bias correction.

Hybrid approach: compute variance via analytical Jacobian (stable), compute
mean via MC perturbation averaging (bias-corrected), then Gaussian NLL.

Run: uv run python tools/diagnose_hybrid_mc.py [--sigma_x 0.5] [--linear] [--n_runs 3]
     uv run python tools/diagnose_hybrid_mc.py --warmup_epochs 20  # MSE warmup → hybrid
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from diagnostic_utils import (
    generate_linear_eiv_data,
    generate_nonlinear_eiv_data,
    make_mlp,
    train_with_loss,
    train_with_warmup,
)


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Jacobian-variance + MC-mean EIV diagnostic",
    )
    parser.add_argument("--sigma_x", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--linear", action="store_true",
        help="Use linear data (y = X·β) instead of nonlinear",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=None,
        help="MSE warmup epochs before hybrid training (e.g. 20 from "
             "HarnessConfig.warmup_epochs). When provided, runs only the "
             "hybrid loss with MSE pre-training instead of all four losses.",
    )
    args = parser.parse_args()

    sigma_x = args.sigma_x
    sigma_y = 0.1
    use_warmup = args.warmup_epochs is not None
    if use_warmup:
        warmup = min(args.warmup_epochs, args.epochs - 1)
        loss_types = ["hybrid_warmup"]
    else:
        warmup = 0
        loss_types = ["mse", "feiv", "mc", "hybrid"]
    data_type = "linear" if args.linear else "nonlinear"
    gen_fn = generate_linear_eiv_data if args.linear else generate_nonlinear_eiv_data

    print("=" * 78)
    print(
        f"  Hybrid EIV Diagnostic — Jacobian Variance + MC Mean  ({data_type} data)"
    )
    print(f"  σx={sigma_x}, σy={sigma_y}, epochs={args.epochs}, n_runs={args.n_runs}")
    if args.linear:
        print(f"  Linear model: y = X·β + ε  (β ~ U(-1,1))")
    else:
        print(f"  Nonlinear: y = sin(2π·x₀) + 0.5·cos(4π·x₁) + 0.3·tanh(2·x₂) + ε")
    if use_warmup:
        hybrid_eps = args.epochs - warmup
        print(f"  Warmup: {warmup} epochs MSE → {hybrid_eps} epochs hybrid")
    else:
        print(f"  Full comparison: {loss_types}")
    print("=" * 78)
    print()

    # ── Run experiments ──────────────────────────────────────
    all_results: dict[str, list[dict]] = {k: [] for k in loss_types}
    if use_warmup:
        all_results["mse"] = []  # always include baseline for comparison

    for run_idx in range(args.n_runs):
        seed = 123 + run_idx * 17
        torch.manual_seed(seed)
        np.random.seed(seed)
        data = gen_fn(
            n_samples=2000, sigma_x=sigma_x, sigma_y=sigma_y, seed=seed,
        )
        in_dim = data["X_train"].shape[1]
        X_t = torch.from_numpy(data["X_train"]).float()
        y_t = torch.from_numpy(data["y_train"]).float()
        X_clean_t = torch.from_numpy(data["X_test_true"]).float()
        y_clean_t = torch.from_numpy(data["y_test"]).float()

        # Always train MSE baseline for comparison
        if use_warmup:
            m_mse = make_mlp(in_dim)
            r_mse = train_with_loss(
                m_mse, X_t, y_t, X_clean_t, y_clean_t,
                loss_type="mse", sigma_x=sigma_x, sigma_y=sigma_y,
                epochs=args.epochs, lr=args.lr,
            )
            all_results["mse"].append(r_mse)

        for lt in loss_types:
            if lt == "hybrid_warmup":
                # MSE warmup → hybrid training
                r = train_with_warmup(
                    make_mlp(in_dim), X_t, y_t, X_clean_t, y_clean_t,
                    warmup_epochs=warmup,
                    hybrid_epochs=args.epochs - warmup,
                    sigma_x=sigma_x, sigma_y=sigma_y,
                    lr=args.lr,
                )
                all_results[lt].append(r)
            else:
                m = make_mlp(in_dim)
                r = train_with_loss(
                    m, X_t, y_t, X_clean_t, y_clean_t,
                    loss_type=lt, sigma_x=sigma_x, sigma_y=sigma_y,
                    epochs=args.epochs, lr=args.lr,
                )
                all_results[lt].append(r)

    # ── Results table ───────────────────────────────────────
    print(f"  {'Method':<18} {'RMSE_clean':>12} {'±std':>8} {'pred_range':>10} {'vs_MSE':>8}")
    print(f"  {'─'*18} {'─'*12} {'─'*8} {'─'*10} {'─'*8}")

    mse_mean = np.mean([r["rmse_clean"] for r in all_results["mse"]])

    labels = {
        "mse": "MSE (baseline)",
        "feiv": "FuncEIV (analyt.)",
        "mc": "FuncEIV_MC (pure)",
        "hybrid": "HYBRID (J+MC)",
        "hybrid_warmup": "HYBRID+warmup",
    }

    for lt in loss_types:
        values = [r["rmse_clean"] for r in all_results[lt]]
        ranges = [r["pred_range"] for r in all_results[lt]]
        mean_v = np.mean(values)
        std_v = np.std(values)
        mean_r = np.mean(ranges)
        delta = (mean_v - mse_mean) / mse_mean * 100

        marker = ""
        if lt == "hybrid":
            mc_mean = np.mean([r["rmse_clean"] for r in all_results["mc"]])
            if mean_v < mse_mean * 0.98:
                marker = f"  ← BEATS MSE! (vs MC {mc_mean:.3f})"
            elif mean_v < mc_mean * 0.9:
                marker = f"  ← RESCUES MC! (vs {mc_mean:.3f})"
            elif mean_v < mse_mean * 0.95:
                marker = "  ← better than MSE"
        elif lt == "hybrid_warmup":
            w_eps = warmup
            if mean_v < mse_mean * 0.98:
                marker = f"  ← BEATS MSE! (warmup={w_eps})"
            elif mean_v < mse_mean:
                marker = "  ← better than MSE"

        print(
            f"  {labels[lt]:<18} {mean_v:12.4f} {std_v:>8.4f} {mean_r:10.4f} {delta:+8.1f}%{marker}"
        )

    # ── Cross-comparison ────────────────────────────────────
    if not use_warmup:
        hybrid_vals = [r["rmse_clean"] for r in all_results["hybrid"]]
        mc_vals = [r["rmse_clean"] for r in all_results["mc"]]
        feiv_vals = [r["rmse_clean"] for r in all_results["feiv"]]

        print()
        print("-" * 78)
        print("  CROSS-COMPARISON")
        print("-" * 78)
        print(f"  Pure MC:       {np.mean(mc_vals):.4f} ± {np.std(mc_vals):.4f}")
        print(f"  Analytical:    {np.mean(feiv_vals):.4f} ± {np.std(feiv_vals):.4f}")
        print(f"  HYBRID:        {np.mean(hybrid_vals):.4f} ± {np.std(hybrid_vals):.4f}")
        print(f"  MSE baseline:  {mse_mean:.4f}")

        hybrid_vs_mc = (np.mean(hybrid_vals) - np.mean(mc_vals)) / np.mean(mc_vals) * 100
        hybrid_vs_feiv = (np.mean(hybrid_vals) - np.mean(feiv_vals)) / np.mean(feiv_vals) * 100
        print(f"  Hybrid vs MC:         {hybrid_vs_mc:+.1f}%")
        print(f"  Hybrid vs Analytical: {hybrid_vs_feiv:+.1f}%")
    else:
        warmup_vals = [r["rmse_clean"] for r in all_results["hybrid_warmup"]]
        print()
        print("-" * 78)
        print("  WARMUP RESULT")
        print("-" * 78)
        print(f"  HYBRID+warmup_{warmup}: {np.mean(warmup_vals):.4f} ± {np.std(warmup_vals):.4f}")
        print(f"  MSE baseline:  {mse_mean:.4f}")
        print(f"  vs MSE:        {(np.mean(warmup_vals) - mse_mean) / mse_mean * 100:+.1f}%")

    # ── Trajectory (first run, only in full comparison mode) ─
    if not use_warmup:
        print()
        print("-" * 78)
        print("  RMSE_clean TRAJECTORY (Run 1)")
        print("-" * 78)

        histories = {lt: all_results[lt][0]["history"] for lt in loss_types}
        all_epochs = sorted(set(
            h["epoch"] for hl in histories.values() for h in hl
        ))

        header = f"  {'Epoch':>6}"
        for lt in loss_types:
            header += f" {labels[lt]:>18}"
        print(header)
        print(f"  {'─'*6}" + "".join(f" {'─'*18}" for _ in loss_types))

        for e in all_epochs:
            line = f"  {e:6d}"
            for lt in loss_types:
                match = [h for h in histories[lt] if h["epoch"] == e]
                if match:
                    line += f" {match[0]['rmse_clean']:18.4f}"
                else:
                    line += f" {'':>18}"
            print(line)

    # ── Verdict ─────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)

    if use_warmup:
        warmup_mean = np.mean([r["rmse_clean"] for r in all_results["hybrid_warmup"]])
        if warmup_mean < mse_mean * 0.98:
            print(f"  ✅ HYBRID+warmup_{warmup} beats MSE ({mse_mean:.4f} → {warmup_mean:.4f})")
        elif warmup_mean < mse_mean:
            print(f"  → HYBRID+warmup_{warmup} slightly better than MSE ({mse_mean:.4f} → {warmup_mean:.4f})")
        else:
            print(f"  ❌ HYBRID+warmup_{warmup} worse than MSE ({mse_mean:.4f} → {warmup_mean:.4f})")
        print(f"  Run full comparison (without --warmup_epochs) to compare vs pure MC/analytical")
    else:
        if hybrid_vs_mc < -20:
            print(f"  ✅ HYBRID STRONGLY RESCUES MC: {abs(hybrid_vs_mc):.0f}% better than pure MC")
            print(f"     Jacobian variance stabilises the MC path on nonlinear data")
        elif hybrid_vs_mc < -5:
            print(f"  ⚠️  HYBRID MODERATELY IMPROVES MC: {abs(hybrid_vs_mc):.0f}% better")
            print(f"     Jacobian variance helps but doesn't fully fix the MC path")
        elif abs(hybrid_vs_mc) < 5:
            print(f"  → HYBRID ≈ PURE MC: Jacobian variance doesn't change MC outcome")
        else:
            print(f"  ❌ HYBRID WORSE than MC: Jacobian variance hurts MC path")

        if np.mean(hybrid_vals) < mse_mean * 0.98:
            print(f"  ✅ HYBRID beats MSE ({mse_mean:.4f} → {np.mean(hybrid_vals):.4f})")
        elif np.mean(hybrid_vals) < mse_mean:
            print(f"  → HYBRID slightly better than MSE ({mse_mean:.4f} → {np.mean(hybrid_vals):.4f})")
        else:
            print(f"  → HYBRID worse than MSE ({mse_mean:.4f} → {np.mean(hybrid_vals):.4f})")


if __name__ == "__main__":
    main()
