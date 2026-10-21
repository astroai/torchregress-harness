#!/usr/bin/env python3
"""Conformalized conditional density: raw vs CTI vs Mondrian vs WALDO.

Synthetic scalar/multi-target data with a known conditional density shape.  A
shared MLP encoder is paired with an NSF (NLL) or a ContrastiveFlowLoss positive
flow; calibration uses a held-out split and coverage is measured on a clean test
set.  This is a controlled benchmark of the conformal machinery, not a claim
about astronomy data.
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
    ConformalizedDensity,
    ContrastiveFlowLoss,
    NormalizingFlowLoss,
    WaldoDensityConformal,
    create_flow_model,
)


def _data(n: int, multi: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    mean = np.stack(
        [np.sin(1.2 * x[:, 0]) + 0.4 * x[:, 1], np.cos(1.1 * x[:, 1]) - 0.3 * x[:, 0]], axis=1
    )
    mode = rng.choice([-1.0, 1.0], size=(n, 1) if multi else (n,))
    if multi:
        y = mean + mode * np.array([0.8, 0.5]) + rng.normal(0, 0.4, size=(n, 2))
    else:
        y = mean[:, 0] + mode * 0.8 + rng.normal(0, 0.4, size=n)
    return x, y.astype(np.float32)


class _Encoder(nn.Module):
    def __init__(self, d: int = 2, h: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train(
    enc: nn.Module, loss_fn: Any, X: np.ndarray, Y: np.ndarray, *, epochs: int, contrastive: bool
) -> None:
    dev = next(enc.parameters()).device
    xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    if yt.ndim == 1:
        yt = yt.unsqueeze(-1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(loss_fn.parameters()), lr=1e-3)
    n = len(xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for start in range(0, n, 256):
            idx = perm[start : start + 256]
            opt.zero_grad(set_to_none=True)
            ctx = enc(xt[idx])
            if contrastive:
                neg = ctx.roll(1, dims=0).unsqueeze(1)
                loss = loss_fn(ctx, yt[idx], negative_context=neg)
            else:
                loss = loss_fn(ctx, yt[idx])
            loss.backward()
            opt.step()


def _sample_fn(loss_fn: Any, enc: nn.Module):
    def sample(context: torch.Tensor, n: int) -> torch.Tensor:
        with torch.no_grad():
            return loss_fn.sample(enc(context), n_samples=n)

    return sample


@torch.no_grad()
def _metrics(
    enc: nn.Module, loss_fn: Any, X: np.ndarray, Y: np.ndarray, *, groups: np.ndarray | None = None
) -> dict[str, float]:
    dev = next(enc.parameters()).device
    xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    if yt.ndim == 1:
        yt = yt.unsqueeze(-1)
    ctx = enc(xt)
    nll = float(-loss_fn.log_prob(ctx, yt).mean())
    samples = loss_fn.sample(ctx, n_samples=256)
    # 5/95 marginal intervals from samples.
    lo = samples.quantile(0.05, dim=1)
    hi = samples.quantile(0.95, dim=1)
    raw_cov = float(((yt >= lo) & (yt <= hi)).double().mean())
    raw_width = float((hi - lo).mean())
    # Raw CRPS via samples per dimension.
    from torchregress.metrics import crps_from_samples

    crps = float(
        np.mean(
            [
                crps_from_samples(samples[:, :, d].T.cpu().numpy(), yt[:, d].cpu().numpy())
                for d in range(yt.shape[1])
            ]
        )
    )
    return {
        "NLL": nll,
        "CRPS": crps,
        "SampleCov90": raw_cov,
        "SampleWidth90": raw_width,
        "_ctx": ctx,
    }


def _cti_metrics(
    loss_fn: Any,
    ctx_cal: torch.Tensor,
    y_cal: torch.Tensor,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    enc: nn.Module,
    *,
    groups_cal: np.ndarray | None = None,
    groups_test: np.ndarray | None = None,
) -> tuple[dict[str, float], ConformalizedDensity]:
    def negative_log_prob(context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -loss_fn.log_prob(context, target)

    cp = ConformalizedDensity(alpha=0.1, grid_size=100)
    cp.calibrate(
        negative_log_prob,
        ctx_cal,
        y_cal,
        groups=None if groups_cal is None else torch.as_tensor(groups_cal),
    )
    dev = next(enc.parameters()).device
    xt = torch.as_tensor(X_test, dtype=torch.float32, device=dev)
    with torch.no_grad():
        ctx = enc(xt)
    y = torch.as_tensor(Y_test, dtype=torch.float32, device=dev)
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    groups_t = None if groups_test is None else torch.as_tensor(groups_test, device=dev)
    inside = cp.contains(negative_log_prob, ctx, y, groups=groups_t)
    if y.shape[1] == 1:
        n_iv = min(100, len(y))
        lo, hi = cp.predict_interval(
            negative_log_prob,
            ctx[:n_iv],
            float(y.min()) - 4.0,
            float(y.max()) + 4.0,
            groups=None if groups_t is None else groups_t[:n_iv],
        )
        cov = float(inside.double().mean())
        width = float((hi - lo).mean())
    else:
        cov = float(inside.double().mean())
        width = float("nan")
    return {"CTICov": cov, "CTIWidth": width, "CTILevelSet": float(inside.double().mean())}, cp


def _waldo_metrics(
    loss_fn: Any,
    sample_fn: Any,
    ctx_cal: torch.Tensor,
    y_cal: torch.Tensor,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    enc: nn.Module,
) -> dict[str, float]:
    cp = WaldoDensityConformal(alpha=0.1, covariance_type="full", n_samples=128)

    # Clarify that sample_fn consumes encoded context, not raw features.
    def encoded_sample_fn(context: torch.Tensor, n_samples: int) -> torch.Tensor:
        return loss_fn.sample(context, n_samples=n_samples)

    cp.calibrate(encoded_sample_fn, ctx_cal, y_cal)
    dev = next(enc.parameters()).device
    with torch.no_grad():
        ctx = enc(torch.as_tensor(X_test, dtype=torch.float32, device=dev))
    y = torch.as_tensor(Y_test, dtype=torch.float32, device=dev)
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    inside = cp.contains(encoded_sample_fn, ctx, y)
    out = {"WaldoCov": float(inside.double().mean())}
    if y.shape[1] == 1:
        n_iv = min(100, len(y))
        lo, hi = cp.predict_interval(
            encoded_sample_fn,
            ctx[:n_iv],
            y_min=float(y.min()) - 4.0,
            y_max=float(y.max()) + 4.0,
        )
        out["WaldoWidth"] = float((hi - lo).mean())
        out["WaldoCov"] = float(inside.double().mean())
    else:
        out["WaldoWidth"] = float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("reports/conformal_density_bench.json"))
    p.add_argument("--epochs", type=int, default=80)
    args = p.parse_args()
    rows = []
    for multi in (False, True):
        name = "cd_multi" if multi else "cd_scalar"
        x_fit, y_fit = _data(1500, multi, 0)
        x_cal, y_cal = _data(600, multi, 1)
        x_test, y_test = _data(600, multi, 2)
        y_dim = 2 if multi else 1
        for model_name in ("NSF", "ContrastiveFlow"):
            torch.manual_seed(0)
            enc = _Encoder(2, 64)
            flow = create_flow_model(
                n_features=y_dim,
                context_dim=64,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            if model_name == "NSF":
                loss_fn = NormalizingFlowLoss(flow=flow)
                contrastive = False
            else:
                loss_fn = ContrastiveFlowLoss(flow=flow, temperature=1.0, margin=0.0)
                contrastive = True
            _train(enc, loss_fn, x_fit, y_fit, epochs=args.epochs, contrastive=contrastive)
            enc.eval()
            loss_fn.eval()
            with torch.no_grad():
                ctx_cal = enc(torch.as_tensor(x_cal, dtype=torch.float32))
                y_cal_t = torch.as_tensor(y_cal, dtype=torch.float32)
                if y_cal_t.ndim == 1:
                    y_cal_t = y_cal_t.unsqueeze(-1)
            base = {"Dataset": name, "Model": model_name, "Epochs": args.epochs}
            raw = _metrics(enc, loss_fn, x_test, y_test)
            raw.pop("_ctx")
            base.update(raw)
            # Marginal vs Mondrian: groups are calibration/test target-mean bins.
            groups_cal = np.digitize(
                y_cal.reshape(len(y_cal), -1).mean(1),
                np.quantile(y_cal.reshape(len(y_cal), -1).mean(1), [0.33, 0.67]),
            )
            groups_test = np.digitize(
                y_test.reshape(len(y_test), -1).mean(1),
                np.quantile(y_cal.reshape(len(y_cal), -1).mean(1), [0.33, 0.67]),
            )
            cti, _ = _cti_metrics(loss_fn, ctx_cal, y_cal_t, x_test, y_test, enc)
            base.update(cti)
            cti_g, _ = _cti_metrics(
                loss_fn,
                ctx_cal,
                y_cal_t,
                x_test,
                y_test,
                enc,
                groups_cal=groups_cal,
                groups_test=groups_test,
            )
            base.update({f"{k}_Mondrian": v for k, v in cti_g.items()})
            sample_fn = _sample_fn(loss_fn, enc)
            base.update(_waldo_metrics(loss_fn, sample_fn, ctx_cal, y_cal_t, x_test, y_test, enc))
            rows.append(base)
            print(json.dumps(base), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
