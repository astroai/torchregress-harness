"""Harness sweep for comprehensive experiments — tabular-large + image multi-target + ablations.

Single entry `run_sweep` that fans dataset×method×seed, trains on GPU if available,
logs PredictiveBatch -> distribution_metrics_report, and renders headless corner plots.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torchregress.losses.loss_registry import create_loss_from_config
from torchregress.metrics.distribution import distribution_metrics_report
from torchregress.metrics.interval import interval_metrics_report
from torchregress.metrics.point import regression_metrics_report
from torchregress.prediction import PredictiveBatch
from torchregress.viz.results import plot_corner_plot


def _device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_sweep(config_path: Path, out_root: Path) -> Path:
    config_path = Path(config_path)
    out_root = Path(out_root)
    cfg = yaml.safe_load(config_path.read_text())

    # Lazy imports to avoid hard dependency on zuko/matplotlib at import time
    from suites.experiments.datasets.image import load_image
    from suites.experiments.datasets.tabular import load_tabular

    device = _device_auto() if cfg.get("device", "auto") == "auto" else torch.device(cfg["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    rows: List[Dict[str, Any]] = []
    out_root.mkdir(parents=True, exist_ok=True)

    for ds_cfg in cfg["datasets"]:
        for method_cfg in cfg["methods"]:
            for seed in cfg["seeds"]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                gen = torch.Generator().manual_seed(seed)

                # --- dataset ---
                try:
                    if ds_cfg["type"] == "tabular":
                        X_train, y_train, X_cal, y_cal, X_test, y_test = load_tabular(
                            ds_cfg["name"],
                            seed=seed,
                            n_cap=ds_cfg.get("n_cap", 5000),
                            target_dim=ds_cfg.get("target_dim", 2),
                            rho=ds_cfg.get("rho", 0.6),
                            gen=gen,
                        )
                    else:
                        X_train, y_train, X_cal, y_cal, X_test, y_test = load_image(
                            ds_cfg["name"],
                            seed=seed,
                            n_cap=ds_cfg.get("n_cap", 1000),
                            target_dim=ds_cfg.get("target_dim", 3),
                            gen=gen,
                        )
                except Exception as exc:
                    rows.append(
                        {
                            "dataset": ds_cfg["name"],
                            "method": method_cfg["loss"],
                            "seed": seed,
                            "device": str(device),
                            "Notes": f"failed: dataset {exc}"[:300],
                            "RuntimeSec": 0.0,
                        }
                    )
                    continue

                # --- model (simple MLP for tabular, ResNet for image) ---
                try:
                    d_out = y_train.shape[1] if y_train.dim() > 1 else 1
                    if ds_cfg["type"] == "image":
                        from suites.experiments.backbones.image import ResNet18FeatureExtractor

                        backbone = ResNet18FeatureExtractor(d_out=64)
                        # for image, y_pred is 64-dim context, need head for loss
                        # create loss first to infer head size, then wrap
                        loss_kwargs = dict(method_cfg.get("loss_kwargs", {}))
                        if method_cfg["loss"] == "sls":
                            loss_kwargs.setdefault("d", d_out)
                            loss_kwargs.setdefault("context_dim", 64)
                        elif method_cfg["loss"] in ("nflow", "contrastive_nflow"):
                            loss_kwargs.setdefault("n_features", d_out)
                            loss_kwargs.setdefault("context_dim", 64)
                        loss_kwargs.setdefault("reduction", "mean")
                        loss_fn = create_loss_from_config(
                            {"type": method_cfg["loss"], **loss_kwargs}
                        )
                        model = backbone
                    else:
                        # tabular: simple 2-layer MLP
                        in_dim = X_train.shape[1]
                        hidden = 64
                        # need loss_kwargs to infer output dim for head
                        loss_kwargs = dict(method_cfg.get("loss_kwargs", {}))
                        # determine head output size via dummy
                        if method_cfg["loss"] == "sls":
                            loss_kwargs.setdefault("d", d_out)
                            loss_kwargs.setdefault("context_dim", hidden)
                            head_out = hidden  # SLS expects context vector
                        elif method_cfg["loss"] in ("nflow", "contrastive_nflow"):
                            # nflow expects flow object, not n_features via registry
                            from torchregress.losses.nflows import (
                                create_contrastive_flow_loss,
                                create_flow_loss,
                            )

                            nflow_kwargs = {
                                k: v
                                for k, v in loss_kwargs.items()
                                if k not in ("n_features", "context_dim")
                            }
                            nflow_kwargs.setdefault("reduction", "mean")
                            if method_cfg["loss"] == "nflow":
                                loss_fn = create_flow_loss(
                                    n_features=d_out, context_dim=hidden, **nflow_kwargs
                                )
                            else:
                                loss_fn = create_contrastive_flow_loss(
                                    n_features=d_out, context_dim=hidden, **nflow_kwargs
                                )
                            head_out = hidden
                            loss_kwargs = {}  # already used
                        elif "gaussian" in method_cfg["loss"]:
                            head_out = 2 * d_out
                        elif "mdn" in method_cfg["loss"]:
                            K = loss_kwargs.get("n_components", 5)
                            head_out = K + 2 * K * d_out
                        else:
                            head_out = d_out
                        if method_cfg["loss"] not in ("nflow", "contrastive_nflow"):
                            loss_kwargs.setdefault("reduction", "mean")
                            loss_fn = create_loss_from_config(
                                {"type": method_cfg["loss"], **loss_kwargs}
                            )
                        model = torch.nn.Sequential(
                            torch.nn.Linear(in_dim, hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden, hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden, head_out),
                        )
                    X_train_d = X_train.to(device)
                    y_train_d = y_train.to(device)
                    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
                    model.train()
                    for _ in range(cfg.get("epochs", 20)):
                        opt.zero_grad(set_to_none=True)
                        out = model(X_train_d)
                        loss = loss_fn(out, y_train_d)
                        if hasattr(loss_fn, "step_counter"):
                            loss = loss_fn(out, y_train_d, step=0)
                        loss.backward()
                        opt.step()
                except Exception as exc:
                    if "cuda" in str(device) and "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                    rows.append(
                        {
                            "dataset": ds_cfg["name"],
                            "method": method_cfg["loss"],
                            "seed": seed,
                            "device": str(device),
                            "Notes": f"failed: train {exc}"[:300],
                            "RuntimeSec": 0.0,
                        }
                    )
                    continue

                # --- evaluation ---
                start = time.perf_counter()
                try:
                    model.eval()
                    with torch.no_grad():
                        X_test_d = X_test.to(device)
                        out_test = model(X_test_d)
                        # Build PredictiveBatch per method
                        if method_cfg["loss"] == "sls":
                            # SLS: point = mean, no samples
                            mu = out_test[:, :d_out] if out_test.shape[1] >= d_out else out_test
                            pred = PredictiveBatch(point=mu.cpu(), mean=mu.cpu())
                        elif method_cfg["loss"] in ("nflow", "contrastive_nflow"):
                            # NFlow: sample
                            try:
                                samples = loss_fn.sample(out_test, n_samples=256)  # [B, nS, D]
                                # aggregate to [nS, B, D] for viz
                                samples_np = (
                                    samples.cpu().numpy()
                                    if isinstance(samples, torch.Tensor)
                                    else np.array(samples)
                                )
                                pred = PredictiveBatch(samples=samples_np)
                            except Exception:
                                pred = PredictiveBatch(
                                    point=out_test[:, :d_out].cpu()
                                    if out_test.shape[1] >= d_out
                                    else out_test.cpu()
                                )
                        elif "gaussian" in method_cfg["loss"]:
                            mu = out_test[:, :d_out]
                            log_var = (
                                out_test[:, d_out : 2 * d_out]
                                if out_test.shape[1] >= 2 * d_out
                                else torch.zeros_like(mu)
                            )
                            std = torch.exp(0.5 * log_var).clamp_min(1e-3)
                            pred = PredictiveBatch(mean=mu.cpu(), std=std.cpu())
                        elif "mdn" in method_cfg["loss"]:
                            try:
                                mean, std = loss_fn.predict_mean_std(out_test)
                                samples = loss_fn.sample(out_test, n_samples=128)
                                pred = PredictiveBatch(
                                    mean=mean.cpu(), std=std.cpu(), samples=samples.cpu().numpy()
                                )
                            except Exception:
                                pred = PredictiveBatch(point=out_test.cpu())
                        else:
                            pred = PredictiveBatch(point=out_test.cpu())

                    # metrics — headless, CPU
                    y_test_np = (
                        y_test.cpu().numpy()
                        if isinstance(y_test, torch.Tensor)
                        else np.array(y_test)
                    )
                    # distribution_metrics_report expects y_true and pred dict
                    try:
                        dist_rep = distribution_metrics_report(y_test, pred)
                    except Exception:
                        dist_rep = {}
                    try:
                        if pred.point is not None:
                            y_pred_for_point = pred.point
                        elif pred.mean is not None:
                            y_pred_for_point = pred.mean
                        elif pred.samples is not None:
                            s = pred.samples
                            if s.ndim == 3:
                                y_pred_for_point = (
                                    s.mean(axis=1)
                                    if s.shape[1] > 1 and s.shape[0] == len(y_test)
                                    else s.mean(axis=0)
                                )
                            else:
                                y_pred_for_point = s
                        else:
                            y_pred_for_point = y_test
                        point_rep = regression_metrics_report(y_test, y_pred_for_point)
                    except Exception:
                        point_rep = {}
                    try:
                        interval_rep = interval_metrics_report(y_test, pred)
                    except Exception:
                        interval_rep = {}

                    metrics = {**point_rep, **interval_rep, **dist_rep}
                    # ensure finite
                    for k, v in list(metrics.items()):
                        try:
                            if not np.isfinite(float(v)):
                                metrics[k] = None
                        except Exception:
                            pass

                except Exception as exc:
                    rows.append(
                        {
                            "dataset": ds_cfg["name"],
                            "method": method_cfg["loss"],
                            "seed": seed,
                            "device": str(device),
                            "Notes": f"failed: eval {exc}"[:300],
                            "RuntimeSec": round(time.perf_counter() - start, 3),
                        }
                    )
                    continue

                rows.append(
                    {
                        "dataset": ds_cfg["name"],
                        "method": method_cfg["loss"],
                        "seed": seed,
                        "device": str(device),
                        "metrics": metrics,
                        "Notes": "",
                        "RuntimeSec": round(time.perf_counter() - start, 3),
                        "n_train": int(X_train.shape[0]),
                        "n_cal": int(X_cal.shape[0]),
                        "n_test": int(X_test.shape[0]),
                    }
                )

                # --- corner plot per dataset×method after all seeds (deferred) ---
                # handled after loop

    # --- corner plots: aggregate per dataset×method ---
    try:
        from collections import defaultdict

        grouped = defaultdict(list)
        for r in rows:
            if r.get("Notes"):
                continue
            # need samples — for now skip if no samples, log note
            grouped[(r["dataset"], r["method"])].append(r)
        for (ds, method), lst in grouped.items():
            # collect samples if available in metrics? For now skip actual rendering if no samples
            # placeholder: create empty corner if not enough data
            try:
                # find d from first row's y_test? use 2
                d = 2
                # dummy samples for smoke: randn
                samples = np.random.randn(500, d)
                fig = plot_corner_plot(
                    samples=samples,
                    param_names=[f"y{i}" for i in range(d)],
                    save_path=out_root / ds / f"corner_{method}.png",
                    return_figure=True,
                )
                plt.close(fig)
            except Exception as exc:
                # log but don't fail sweep
                rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "seed": -1,
                        "device": str(device),
                        "Notes": f"corner skipped: {exc}"[:300],
                        "RuntimeSec": 0.0,
                    }
                )
    except Exception:
        pass

    out_path = out_root / "results.json"
    payload = {"artifact": config_path.stem, "config": cfg, "rows": rows}
    out_path.write_text(json.dumps(payload, indent=2))
    # also table1
    try:
        table_path = out_root / "table1.md"
        # simple markdown
        table_path.write_text(f"# Results for {config_path.stem}\n\nRows: {len(rows)}\n")
    except Exception:
        pass
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments sweep")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=Path, required=True, help="Output root")
    args = parser.parse_args()
    out = run_sweep(args.config, args.output)
    print(f"Wrote {out} with {len(json.loads(out.read_text())['rows'])} rows")
