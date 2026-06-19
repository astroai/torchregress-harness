#!/usr/bin/env python3
"""Diagnostic: Does MSE warmup prevent FunctionalEIV_MC from exploding on nonlinear data?

The MC path catastrophically fails on nonlinear EIV (σx=0.5: +91% RMSE_clean vs MSE).
Hypothesis: pre-training with MSE for N epochs gets the model into a reasonable basin,
then switching to FunctionalEIV_MC avoids the pathological bad basin.

Tests: warmup ∈ {0, 5, 10, 20, 50} epochs at σx ∈ {0.5, 1.0}.
Also tests: keep vs reset optimizer at switch point.

Run: uv run python tools/diagnose_mc_warmup.py [--sigma_x 0.5] [--total_epochs 100] [--n_runs 3]
     uv run python tools/diagnose_mc_warmup.py --warmup_epochs 20  # single config value like HarnessConfig
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from diagnostic_utils import (
    generate_linear_eiv_data,
    generate_nonlinear_eiv_data,
    make_mlp,
    train_baseline,
    train_warmup_mc,
)


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MSE warmup → FunctionalEIV_MC diagnostic",
    )
    parser.add_argument("--sigma_x", type=float, default=0.5)
    parser.add_argument("--total_epochs", type=int, default=100)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--warmup_epochs", type=int, default=None,
        help="Single warmup duration (e.g. 20 from HarnessConfig.warmup_epochs). "
             "When provided, runs only this level instead of the full sweep.",
    )
    parser.add_argument(
        "--linear", action="store_true",
        help="Use linear data (y = X·β) instead of nonlinear",
    )
    args = parser.parse_args()

    sigma_x = args.sigma_x
    sigma_y = 0.1
    total_epochs = args.total_epochs
    data_type = "linear" if args.linear else "nonlinear"
    gen_fn = generate_linear_eiv_data if args.linear else generate_nonlinear_eiv_data

    if args.warmup_epochs is not None:
        warmup_levels = [args.warmup_epochs]
    else:
        warmup_levels = [0, 5, 10, 20, 50]

    print("=" * 78)
    print(
        f"  MSE Warmup → FunctionalEIV_MC Diagnostic  "
        f"(σx={sigma_x}, σy={sigma_y}, {data_type} data)"
    )
    print("=" * 78)
    if args.linear:
        print(f"  Data: y = X·β + ε  (β ~ U(-1,1))")
    else:
        print(f"  Data: y = sin(2π·x₀) + 0.5·cos(4π·x₁) + 0.3·tanh(2·x₂) + ε")
    print(f"  n_train=1000, total_epochs={total_epochs}, n_runs={args.n_runs}")
    if args.warmup_epochs is not None:
        print(f"  Warmup epochs: {args.warmup_epochs} (single level from config)")
    else:
        print(f"  Warmup levels: {warmup_levels} (full sweep)")
    print(f"  MC samples: 20, keep optimizer across switch (no reset)")
    print()

    # ── Run baselines and warmup experiments ─────────────────────
    all_results: dict[str, list[dict]] = {
        "MSE": [],
        "FuncEIV": [],
        "FuncEIV_MC": [],
    }
    for w in warmup_levels:
        all_results[f"Warmup_{w}"] = []

    for run_idx in range(args.n_runs):
        seed = 123 + run_idx * 13
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

        # Baselines
        for label in ("MSE", "FuncEIV", "FuncEIV_MC"):
            m = make_mlp(in_dim)
            r = train_baseline(
                m, X_t, y_t, X_clean_t, y_clean_t,
                label=label, sigma_x=sigma_x, total_epochs=total_epochs,
                lr=args.lr,
            )
            all_results[label].append(r)

        # Warmup experiments
        for w in warmup_levels:
            mc_epochs = total_epochs - w
            m = make_mlp(in_dim)
            r = train_warmup_mc(
                m, X_t, y_t, X_clean_t, y_clean_t,
                warmup_epochs=w, mc_epochs=mc_epochs,
                lr=args.lr, sigma_x=sigma_x, reset_optimizer=False,
            )
            all_results[f"Warmup_{w}"].append(r)

    # ── Aggregate and print ────────────────────────────────────
    print(f"  {'Method':<18} {'RMSE_clean':>12} {'±std':>8} {'pred_range':>10} {'vs_MSE':>8}")
    print(f"  {'─'*18} {'─'*12} {'─'*8} {'─'*10} {'─'*8}")

    mse_mean = np.mean([r["rmse_clean"] for r in all_results["MSE"]])

    for key in ["MSE", "FuncEIV", "FuncEIV_MC"] + [f"Warmup_{w}" for w in warmup_levels]:
        values = [r["rmse_clean"] for r in all_results[key]]
        ranges = [r.get("pred_range", 0) for r in all_results[key]]
        mean_v = np.mean(values)
        std_v = np.std(values)
        mean_r = np.mean(ranges)
        delta = (mean_v - mse_mean) / mse_mean * 100
        marker = ""
        if key == "MSE":
            marker = "  ← baseline"
        elif key == "FuncEIV_MC":
            marker = "  ← pure MC (no warmup)"
        elif key.startswith("Warmup_") and key != "Warmup_0":
            w = int(key.split("_")[1])
            if mean_v < mse_mean * 0.98:
                marker = f"  ← BEATS MSE!"
        print(
            f"  {key:<18} {mean_v:12.4f} {std_v:>8.4f} {mean_r:10.4f} {delta:+8.1f}%{marker}"
        )

    # ── Warmup effectiveness analysis ──────────────────────────
    print()
    print("-" * 78)
    print("  WARMUP EFFECTIVENESS")
    print("-" * 78)

    mc_mean = np.mean([r["rmse_clean"] for r in all_results["FuncEIV_MC"]])
    mc_std = np.std([r["rmse_clean"] for r in all_results["FuncEIV_MC"]])

    for w in warmup_levels:
        values = [r["rmse_clean"] for r in all_results[f"Warmup_{w}"]]
        mean_v = np.mean(values)
        delta_mc = (mean_v - mc_mean) / mc_mean * 100

        if delta_mc < -20:
            effect = "STRONG RESCUE — warmup dramatically reduces MC failure"
        elif delta_mc < -5:
            effect = "MODERATE — warmup helps but doesn't fully rescue"
        elif delta_mc < 5:
            effect = "NEUTRAL — warmup doesn't change MC outcome much"
        else:
            effect = "HARMFUL — warmup makes MC worse"

        print(f"  Warmup={w:3d}:  RMSE_clean={mean_v:.4f}  ΔMC={delta_mc:+6.1f}%  {effect}")

        # Check for post-warmup RMSE trajectory
        if w > 0:
            warmup_end_vals = [
                r["rmse_after_warmup"]
                for r in all_results[f"Warmup_{w}"]
                if r["rmse_after_warmup"] is not None
            ]
            final_vals = [r["rmse_clean"] for r in all_results[f"Warmup_{w}"]]
            if warmup_end_vals:
                w_end = np.mean(warmup_end_vals)
                w_final = np.mean(final_vals)
                after_delta = (w_final - w_end) / w_end * 100
                print(f"           After warmup={w_end:.4f} → MC final={w_final:.4f} ({after_delta:+.1f}%)")

    # ── Best method ────────────────────────────────────────────
    print()
    print("-" * 78)
    print("  SUMMARY")
    print("-" * 78)

    best_key = None
    best_val = float("inf")
    for key in ["MSE", "FuncEIV", "FuncEIV_MC"] + [f"Warmup_{w}" for w in warmup_levels]:
        mean_v = np.mean([r["rmse_clean"] for r in all_results[key]])
        if mean_v < best_val:
            best_val = mean_v
            best_key = key

    print(f"  Best method: {best_key} (RMSE_clean={best_val:.4f})")
    print(f"  MSE baseline: {mse_mean:.4f}")
    print(f"  Pure MC: {mc_mean:.4f} (±{mc_std:.4f})")
    improvement = (mse_mean - best_val) / mse_mean * 100
    print(f"  Best improvement over MSE: {improvement:+.1f}%")

    if best_key.startswith("Warmup_"):
        w_opt = int(best_key.split("_")[1])
        print(f"  → Optimal warmup: {w_opt} epochs MSE → {total_epochs - w_opt} epochs MC")
        print(f"  ✅ WARMUP STRATEGY WORKS — MSE pre-training avoids the MC bad basin")
    elif best_key == "MSE":
        print(f"  ❌ WARMUP DOESN'T HELP — MSE alone is best; MC degrades regardless")
    else:
        print(f"  ⚠️  AMBIGUOUS — {best_key} is best but not clearly better than MSE")


if __name__ == "__main__":
    main()
