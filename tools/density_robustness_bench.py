#!/usr/bin/env python3
"""Factorial density-head benchmark: clean/noisy labels, imbalance, label budget.

Compares a conditional NSF, a diagonal mixture density network (MDN), a smoothed
binned-CDF head, and their noise-aware likelihood variants on a scalar,
multimodal regression problem.  Evaluation is on a large clean test set; the
noise-aware training methods only see noisy labels.

Run: .pixi/envs/default/bin/python tools/density_robustness_bench.py --out reports/density_robustness.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchregress.losses import (
    BinnedDensityLoss,
    MixtureDensityLoss,
    NoiseAwareDensityLoss,
    NormalizingFlowLoss,
    create_flow_model,
)
from torchregress.metrics import crps_from_samples


class _Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _data(
    n: int, imbalance: str, seed: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar target with conditional bimodality and rare bright/faint region."""
    rare = 0.03 if imbalance == "severe" else 0.15
    is_rare = rng.random(n) < rare
    x = rng.normal(size=(n, 1))
    x[is_rare] = rng.normal(2.5, 0.35, size=(int(is_rare.sum()), 1))
    mean = 1.2 * np.sin(x[:, 0]) + 0.5 * x[:, 0]
    mode = rng.choice([-1.0, 1.0], size=n, p=[0.7, 0.3])
    y = mean + mode * (0.9 + 0.2 * np.abs(x[:, 0])) + rng.normal(0, 0.35, size=n)
    return x.astype(np.float32), y.astype(np.float32)


def _train(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y).unsqueeze(-1)
    s_t = torch.from_numpy(sigma)
    in_dim = x.shape[1]
    if method.startswith("NSF"):
        flow = create_flow_model(
            n_features=1,
            context_dim=48,
            flow_type="nsf",
            n_transforms=3,
            hidden_features=[64, 64],
            bound=8.0,
        )
        model = _Net(in_dim, 48)
        base = NormalizingFlowLoss(flow=flow)
        if method == "NSF_noise":
            loss_fn = NoiseAwareDensityLoss(base, mode="quadrature")
        else:
            loss_fn = base
        params = list(model.parameters()) + list(loss_fn.parameters())
    elif method.startswith("MDN"):
        mdn = MixtureDensityLoss(n_components=5, n_features=1)
        model = _Net(in_dim, mdn.expected_output_size)
        base = mdn
        loss_fn = NoiseAwareDensityLoss(mdn, mode="analytic") if method == "MDN_noise" else mdn
        params = list(model.parameters())
    elif method.startswith("Binned"):
        edges = torch.linspace(-6.0, 6.0, 65)
        base = BinnedDensityLoss(edges)
        model = _Net(in_dim, base.n_bins)
        loss_fn = base
        params = list(model.parameters())
    else:
        raise ValueError(method)
    opt = torch.optim.Adam(params, lr=1e-3)
    n = len(x_t)
    bs = min(64, n)
    for _ in range(int(epochs)):
        perm = torch.randperm(n)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            opt.zero_grad()
            ctx = model(x_t[idx])
            if method.endswith("_noise") and not method.startswith("Binned"):
                loss = loss_fn(ctx, y_t[idx], sigma=s_t[idx])
            elif method == "Binned_noise":
                loss = loss_fn(ctx, y_t[idx], sigma=s_t[idx])
            else:
                loss = loss_fn(ctx, y_t[idx])
            loss.backward()
            opt.step()
    return model, loss_fn


def _eval(
    model: Any, loss_fn: Any, x: np.ndarray, y_clean: np.ndarray, sig: np.ndarray | None = None
) -> dict[str, float]:
    model.eval()
    # Evaluate the clean conditional density even when training used a
    # NoiseAwareDensityLoss wrapper.
    if isinstance(loss_fn, NoiseAwareDensityLoss):
        loss_fn = loss_fn.base_loss
    with torch.no_grad():
        ctx = model(torch.from_numpy(x))
        y_t = torch.from_numpy(y_clean).unsqueeze(-1)
        if isinstance(loss_fn, NormalizingFlowLoss):
            clean_nll = -loss_fn.log_prob(ctx, y_t).mean().item()
            samples = loss_fn.sample(ctx, n_samples=256)[..., 0]
        elif isinstance(loss_fn, MixtureDensityLoss):
            clean_nll = loss_fn(ctx, y_t).item()
            # sample via mixture; approximate with component draw using model parameters
            log_w, means, stds = loss_fn._extract_distribution_parameters(ctx)
            comp = torch.distributions.Categorical(logits=log_w).sample((256,)).T  # [B, S]
            means_c = means[:, :, 0]  # [B, K]
            stds_c = stds[:, :, 0]
            means_s = means_c.gather(1, comp)
            stds_s = stds_c.gather(1, comp)
            samples = means_s + stds_s * torch.randn_like(means_s)
        elif isinstance(loss_fn, BinnedDensityLoss):
            logits = ctx
            y_idx = (
                torch.searchsorted(loss_fn.bin_edges, y_t.reshape(-1)).clamp(1, loss_fn.n_bins) - 1
            )
            clean_nll = -torch.log_softmax(logits, -1)[torch.arange(len(y_t)), y_idx].mean().item()
            samples = loss_fn.sample(logits, n_samples=256)
        else:
            raise TypeError(type(loss_fn))
        y_np = y_clean.astype(np.float64)
        s_bs = samples.detach().cpu().numpy().astype(np.float64)  # [B, S]
        crps = float(crps_from_samples(s_bs.T, y_np))  # metric expects [S, B]
        lo = np.quantile(s_bs, 0.05, axis=1)
        hi = np.quantile(s_bs, 0.95, axis=1)
        cov90 = float(np.mean((y_np >= lo) & (y_np <= hi)))
        pit = np.mean(s_bs <= y_np[:, None], axis=1)
        pit_ks = float(np.max(np.abs(np.sort(pit) - (np.arange(len(pit)) + 0.5) / len(pit))))
        rare = x[:, 0] > 1.5
        rare_nll = None
        if np.any(rare) and hasattr(loss_fn, "log_prob"):
            with torch.no_grad():
                rare_ctx = model(torch.from_numpy(x[rare]))
                rare_y = torch.from_numpy(y_clean[rare]).unsqueeze(-1)
                rare_nll = float(-loss_fn.log_prob(rare_ctx, rare_y).mean())
        return {
            "NLL": clean_nll,
            "CRPS": crps,
            "Coverage90": cov90,
            "PITKS": pit_ks,
            "RareNLL": rare_nll,
        }


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/density_robustness.json"))
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    methods = ["NSF", "NSF_noise", "MDN", "MDN_noise", "Binned", "Binned_noise"]
    budgets = [200, 1000]
    noises = [0.0, 0.3]
    imbalances = ["mild", "severe"]
    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(100 + seed)
        x_test, y_test = _data(5000, "mild", seed, rng)
        for imbalance in imbalances:
            for budget in budgets:
                for noise in noises:
                    rng = np.random.default_rng(1000 * seed + budget + int(noise * 10))
                    x_tr, y_clean = _data(budget, imbalance, seed, rng)
                    sigma = np.full(budget, noise, dtype=np.float32)
                    y_tr = y_clean + sigma * rng.normal(size=budget).astype(np.float32)
                    for method in methods:
                        try:
                            model, loss_fn = _train(
                                method,
                                x_tr,
                                y_tr,
                                sigma,
                                epochs=40,
                                seed=seed,
                            )
                            metrics = _eval(model, loss_fn, x_test, y_test)
                            rows.append(
                                {
                                    "seed": seed,
                                    "imbalance": imbalance,
                                    "budget": budget,
                                    "noise": noise,
                                    "method": method,
                                    **metrics,
                                }
                            )
                            print(
                                f"{seed} {imbalance:6s} n={budget:4d} noise={noise:.1f} "
                                f"{method:12s} NLL={metrics['NLL']:.3f} CRPS={metrics['CRPS']:.3f}",
                                flush=True,
                            )
                            args.out.parent.mkdir(parents=True, exist_ok=True)
                            args.out.write_text(json.dumps(rows, indent=2))
                        except Exception as exc:  # noqa: BLE001
                            print(f"FAIL {method} {imbalance} {budget} {noise}: {exc}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
