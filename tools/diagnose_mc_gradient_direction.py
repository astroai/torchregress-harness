#!/usr/bin/env python3
"""Diagnostic: Does the MC loss gradient point in the wrong direction?

We know MSE warmup gets the model to a good RMSE (~0.82), then MC destroys it.
This diagnostic tests WHY: at the post-MSE optimum, we compute gradients from:
  - MC loss (noisy X)    — the "bad" direction
  - MSE loss (clean X)   — the "true" gradient (oracle)
  - MSE loss (noisy X)   — the training direction that works
  - Analytical FEIV (noisy X) — the stable EIV direction

We compare cosine similarities to see whether MC pushes the model away from
the true optimum, and whether any single MC step degrades RMSE_clean.

Run: uv run python tools/diagnose_mc_gradient_direction.py [--sigma_x 0.5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# Allow running from either torchregress or torchregress-harness directory
_HERE = Path(__file__).resolve().parent
_TORCHREGRESS_SRC = _HERE.parent.parent / "torchregress" / "src"
if str(_TORCHREGRESS_SRC) not in sys.path:
    sys.path.insert(0, str(_TORCHREGRESS_SRC))

from torchregress.losses import FunctionalEIVLoss

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
        "X_train_true": X_true[:n_train],
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


def eval_rmse_clean(
    model: nn.Module,
    X_test_true: torch.Tensor,
    y_test_true: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(X_test_true).squeeze(-1)
        return float(torch.sqrt(((pred - y_test_true.squeeze(-1)) ** 2).mean()).item())


# ── gradient collection ──────────────────────────────────────────────


def flatten_grad(model: nn.Module) -> torch.Tensor:
    """Collect all parameter gradients into a single flat vector."""
    grads: list[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads)


def per_layer_cosine_sims(
    model: nn.Module,
    compute_grad_a: callable,
    compute_grad_b: callable,
) -> dict[str, float]:
    """Compute per-layer cosine similarity between two gradient sources.

    Returns dict mapping layer name to cosine similarity.
    """
    # Get named parameters with gradients
    named_params = [(n, p) for n, p in model.named_parameters() if p.dim() >= 2]

    sims: dict[str, float] = {}
    for name, _ in named_params:
        # Compute grad_a for this layer only
        grad_a_fn = compute_grad_a()
        grad_b_fn = compute_grad_b()

        # Extract this layer's gradient from the model
        pa = dict(model.named_parameters())[name]
        pb = dict(model.named_parameters())[name]

        ga = pa.grad
        gb = pb.grad

        if ga is None or gb is None:
            sims[name] = float("nan")
            continue

        ga_f = ga.detach().flatten()
        gb_f = gb.detach().flatten()
        sims[name] = float((ga_f @ gb_f).item() / (ga_f.norm().item() * gb_f.norm().item() + 1e-12))

    return sims


def zero_grad_all(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two gradient vectors."""
    return float((a @ b).item() / (a.norm().item() * b.norm().item() + 1e-12))


def compute_mc_gradient(
    model: nn.Module,
    x_noisy: torch.Tensor,
    y: torch.Tensor,
    sigma_x: float,
    sigma_y: float,
) -> torch.Tensor:
    """Compute MC loss gradient w.r.t. all parameters."""
    zero_grad_all(model)
    loss_fn = FunctionalEIVLoss(
        model=model,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        monte_carlo=True,
        n_samples=20,
    )
    loss = loss_fn(x_noisy, y)
    loss.backward()
    g = flatten_grad(model)
    return g, float(loss.item())


def compute_analytical_gradient(
    model: nn.Module,
    x_noisy: torch.Tensor,
    y: torch.Tensor,
    sigma_x: float,
    sigma_y: float,
) -> torch.Tensor:
    """Compute analytical FunctionalEIV gradient."""
    zero_grad_all(model)
    loss_fn = FunctionalEIVLoss(
        model=model,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        monte_carlo=False,
    )
    loss = loss_fn(x_noisy, y)
    loss.backward()
    g = flatten_grad(model)
    return g, float(loss.item())


def compute_mse_gradient(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE gradient on given (x, y)."""
    zero_grad_all(model)
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    g = flatten_grad(model)
    return g, float(loss.item())


def copy_model(model: nn.Module) -> nn.Module:
    """Deep copy model weights."""
    clone = make_mlp(6, 32, 1)
    clone.load_state_dict({k: v.clone().detach() for k, v in model.state_dict().items()})
    clone.to(next(model.parameters()).device)
    return clone


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose MC gradient direction at MSE optimum",
    )
    parser.add_argument("--sigma_x", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    sigma_x = args.sigma_x
    sigma_y = 0.1

    print("=" * 78)
    print(f"  MC Gradient Direction Diagnostic  (σx={sigma_x}, σy={sigma_y})")
    print("=" * 78)
    print(f"  n_runs={args.n_runs}, MSE pre-train epochs={args.epochs}")
    print()

    all_sims: dict[str, list[float]] = {
        "MC_vs_CleanMSE": [],
        "MC_vs_NoisyMSE": [],
        "MC_vs_Analytical": [],
        "Analytical_vs_CleanMSE": [],
        "Analytical_vs_NoisyMSE": [],
    }
    all_deltas: list[float] = []
    all_rmse_mse: list[float] = []
    all_grad_norms: dict[str, list[float]] = {
        "MC": [],
        "Analytical": [],
        "CleanMSE": [],
        "NoisyMSE": [],
    }

    for run_idx in range(args.n_runs):
        seed = 123 + run_idx * 31
        torch.manual_seed(seed)
        np.random.seed(seed)

        data = generate_nonlinear_eiv_data(
            n_samples=2000,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            seed=seed,
        )
        in_dim = 6
        X_noisy_t = torch.from_numpy(data["X_train"]).float()
        y_t = torch.from_numpy(data["y_train"]).float()
        X_clean_t = torch.from_numpy(data["X_train_true"]).float()
        X_test_clean_t = torch.from_numpy(data["X_test_true"]).float()
        y_test_t = torch.from_numpy(data["y_test"]).float()

        # ── Train MSE to convergence ─────────────────────
        model = make_mlp(in_dim)
        opt = Adam(model.parameters(), lr=args.lr)
        n = X_noisy_t.shape[0]
        bs = args.batch_size

        model.train()
        for epoch in range(args.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                opt.zero_grad(set_to_none=True)
                loss = ((model(X_noisy_t[idx]) - y_t[idx]) ** 2).mean()
                loss.backward()
                opt.step()

        rmse_mse = eval_rmse_clean(model, X_test_clean_t, y_test_t)
        all_rmse_mse.append(rmse_mse)
        model.eval()

        # ── Compute all gradients at MSE optimum ──────────
        # Use a small batch for gradient comparison (all methods see same data)
        idx_batch = torch.randperm(n)[:bs]
        x_batch_noisy = X_noisy_t[idx_batch]
        x_batch_clean = X_clean_t[idx_batch]
        y_batch = y_t[idx_batch]

        g_mc, loss_mc = compute_mc_gradient(
            model,
            x_batch_noisy,
            y_batch,
            sigma_x,
            sigma_y,
        )
        g_anal, loss_anal = compute_analytical_gradient(
            model,
            x_batch_noisy,
            y_batch,
            sigma_x,
            sigma_y,
        )
        g_clean, loss_clean = compute_mse_gradient(
            model,
            x_batch_clean,
            y_batch,
        )
        g_noisy, loss_noisy = compute_mse_gradient(
            model,
            x_batch_noisy,
            y_batch,
        )

        # ── Cosine similarities ───────────────────────────
        all_sims["MC_vs_CleanMSE"].append(cosine_sim(g_mc, g_clean))
        all_sims["MC_vs_NoisyMSE"].append(cosine_sim(g_mc, g_noisy))
        all_sims["MC_vs_Analytical"].append(cosine_sim(g_mc, g_anal))
        all_sims["Analytical_vs_CleanMSE"].append(cosine_sim(g_anal, g_clean))
        all_sims["Analytical_vs_NoisyMSE"].append(cosine_sim(g_anal, g_noisy))

        all_grad_norms["MC"].append(float(g_mc.norm().item()))
        all_grad_norms["Analytical"].append(float(g_anal.norm().item()))
        all_grad_norms["CleanMSE"].append(float(g_clean.norm().item()))
        all_grad_norms["NoisyMSE"].append(float(g_noisy.norm().item()))

        # ── One-step degradation test ─────────────────────
        model_test = copy_model(model)
        opt_test = Adam(model_test.parameters(), lr=args.lr * 0.1)
        model_test.train()

        # One MC step
        loss_fn = FunctionalEIVLoss(
            model=model_test,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            monte_carlo=True,
            n_samples=20,
        )
        opt_test.zero_grad()
        l = loss_fn(x_batch_noisy, y_batch)
        l.backward()
        opt_test.step()

        rmse_after_mc_step = eval_rmse_clean(model_test, X_test_clean_t, y_test_t)
        delta = rmse_after_mc_step - rmse_mse
        all_deltas.append(delta)

    # ── Aggregate results ─────────────────────────────────────────
    print(f"  MSE baseline RMSE_clean: {np.mean(all_rmse_mse):.4f} ± {np.std(all_rmse_mse):.4f}")
    print()
    print(f"  {'Gradient Pair':<28} {'CosSim(mean)':>12} {'±std':>8}  {'Interpretation'}")
    print(f"  {'─' * 28} {'─' * 12} {'─' * 8}  {'─' * 30}")

    for key, label in [
        ("MC_vs_CleanMSE", "MC vs CLEAN MSE (truth)"),
        ("MC_vs_NoisyMSE", "MC vs NOISY MSE (works)"),
        ("MC_vs_Analytical", "MC vs Analytical FEIV"),
        ("Analytical_vs_CleanMSE", "Analytical vs CLEAN MSE"),
        ("Analytical_vs_NoisyMSE", "Analytical vs NOISY MSE"),
    ]:
        vals = all_sims[key]
        mean_v = np.mean(vals)
        std_v = np.std(vals)

        if mean_v > 0.7:
            interp = "← strongly aligned (same direction)"
        elif mean_v > 0.3:
            interp = "← weakly aligned"
        elif mean_v > -0.3:
            interp = "← nearly orthogonal (unrelated)"
        elif mean_v > -0.7:
            interp = "← weakly opposed"
        else:
            interp = "← STRONGLY OPPOSED (wrong direction!)"

        print(f"  {label:<28} {mean_v:12.4f} {std_v:>8.4f}  {interp}")

    # ── Gradient norms ────────────────────────────────────────────
    print()
    print(f"  {'Gradient Source':<22} {'Norm(mean)':>12} {'±std':>8}")
    print(f"  {'─' * 22} {'─' * 12} {'─' * 8}")
    for key, label in [
        ("MC", "MC loss (noisy X)"),
        ("Analytical", "Analytical FEIV (noisy X)"),
        ("CleanMSE", "Clean MSE (oracle)"),
        ("NoisyMSE", "Noisy MSE (works)"),
    ]:
        vals = all_grad_norms[key]
        print(f"  {label:<22} {np.mean(vals):12.4f} {np.std(vals):>8.4f}")

    # ── One-step degradation ───────────────────────────────────────
    print()
    print(
        f"  One MC step at MSE optimum: ΔRMSE_clean = {np.mean(all_deltas):+.4f} ± {np.std(all_deltas):.4f}"
    )
    if np.mean(all_deltas) > 0.05:
        print("  → SINGLE MC STEP DESTROYS the MSE solution — the gradient pushes away")
    elif np.mean(all_deltas) > 0.01:
        print("  → MC step mildly degrades MSE solution")
    else:
        print("  → MC step has negligible immediate effect")

    # ── Verdict ────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)

    mc_vs_clean = np.mean(all_sims["MC_vs_CleanMSE"])
    mc_vs_noisy = np.mean(all_sims["MC_vs_NoisyMSE"])
    anal_vs_clean = np.mean(all_sims["Analytical_vs_CleanMSE"])
    one_step = np.mean(all_deltas)

    if mc_vs_clean < 0:
        print(f"  ❌ MC gradient is OPPOSED to clean MSE gradient (cos={mc_vs_clean:.3f})")
        print("     → Following MC gradient moves AWAY from the true optimum")
    elif mc_vs_clean < 0.3:
        print(
            f"  ⚠️  MC gradient is nearly ORTHOGONAL to clean MSE gradient (cos={mc_vs_clean:.3f})"
        )
        print("     → MC gradient doesn't help; it explores irrelevant directions")
    else:
        print(f"  ✅ MC gradient is ALIGNED with clean MSE gradient (cos={mc_vs_clean:.3f})")

    if anal_vs_clean > mc_vs_clean + 0.2:
        print(
            f"  ✅ Analytical FEIV is MUCH better aligned with truth ({anal_vs_clean:.3f} vs {mc_vs_clean:.3f})"
        )
    elif anal_vs_clean > mc_vs_clean + 0.05:
        print(
            f"  → Analytical FEIV is slightly better aligned ({anal_vs_clean:.3f} vs {mc_vs_clean:.3f})"
        )
    else:
        print("  → MC and Analytical have similar alignment to truth")

    if one_step > 0.02:
        print(f"  ❌ Single MC step degrades RMSE_clean by {one_step:.4f}")
        print("     → Even ONE MC gradient step at the MSE optimum is harmful")

    mc_norm = np.mean(all_grad_norms["MC"])
    clean_norm = np.mean(all_grad_norms["CleanMSE"])
    noisy_norm = np.mean(all_grad_norms["NoisyMSE"])
    print(
        f"  Gradient norms: MC={mc_norm:.2f}  CleanMSE={clean_norm:.2f}  NoisyMSE={noisy_norm:.2f}"
    )
    ratio = mc_norm / clean_norm if clean_norm > 0 else 0
    if ratio > 5:
        print(f"  ⚠️  MC gradient norm is {ratio:.1f}× larger than clean MSE gradient —")
        print("     the MC loss is much steeper/louder than the true loss")


if __name__ == "__main__":
    main()
