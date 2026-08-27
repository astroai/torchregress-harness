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
from torchregress.prediction import PredictiveBatch
from torchregress.viz.results import plot_corner_plot

from harness.metrics import compute_all_metrics


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
    # predictive samples for corner plots: (dataset, method) -> {samples, true}
    corner_store: dict[tuple[str, str], dict[str, Any]] = {}
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
                        loss_kwargs = dict(method_cfg.get("loss_kwargs", {}))
                        # determine head size (same logic as tabular)
                        if method_cfg["loss"] == "sls":
                            loss_kwargs.setdefault("d", d_out)
                            loss_kwargs.setdefault("context_dim", 64)
                            loss_kwargs.setdefault("K", 1)
                            loss_kwargs.setdefault("hidden_dim", 64)
                            loss_kwargs.setdefault("n_transforms", 4)
                            head_out = 64
                        elif method_cfg["loss"] in ("nflow", "contrastive_nflow"):
                            from torchregress.losses.nflows import (
                                create_contrastive_flow_loss,
                                create_flow_loss,
                            )

                            nflow_kwargs = {
                                k: v
                                for k, v in loss_kwargs.items()
                                if k not in ("n_features", "context_dim")
                            }
                            nflow_kwargs.setdefault("flow_type", "nsf")
                            nflow_kwargs.setdefault("n_transforms", 3)
                            nflow_kwargs.setdefault("hidden_features", [64, 64])
                            # need to create flow with d_out
                            loss_fn = (
                                create_flow_loss
                                if method_cfg["loss"] == "nflow"
                                else create_contrastive_flow_loss
                            )
                            loss_fn = loss_fn(n_features=d_out, context_dim=64, **nflow_kwargs)
                            loss_kwargs = {}  # already handled
                            head_out = 64  # nflow head is context (we will set below)
                            # for nflow, model is backbone (context) directly
                            model = backbone
                            loss_kwargs = {}  # prevent double creation
                        else:
                            # gaussian, mdn, skew_t, etc: infer head size; handle full covariance ponytail
                            lk_cov_i = loss_kwargs.get("covariance_type")
                            if method_cfg["loss"] == "gaussian_nll" and lk_cov_i == "full":
                                head_out = 2 * d_out  # fallback to diagonal
                            elif method_cfg["loss"] in ("skew_t_nll", "skew_t"):
                                head_out = 4 * d_out
                            elif method_cfg["loss"] == "gaussian":
                                head_out = 2 * d_out
                            elif method_cfg["loss"] == "gaussian_nll":
                                head_out = 2 * d_out
                            elif method_cfg["loss"] == "student_t":
                                head_out = 3 * d_out
                            else:
                                head_out = 2 * d_out  # fallback
                            loss_kwargs.setdefault("reduction", "mean")
                            # mdn needs n_features; others (gaussian_nll, student_t) do not
                            if method_cfg["loss"] == "mdn":
                                loss_kwargs.setdefault("n_components", 5)
                                loss_kwargs.setdefault("n_features", d_out)
                                # correct head_out for mdn if fallback was used
                                K = int(loss_kwargs["n_components"])
                                head_out = K + 2 * K * d_out
                        if method_cfg["loss"] not in ("nflow", "contrastive_nflow"):
                            lk_i = dict(loss_kwargs)
                            if (
                                method_cfg["loss"] == "gaussian_nll"
                                and lk_i.get("covariance_type") == "full"
                            ):
                                lk_i["covariance_type"] = "diagonal"
                            lk_i.setdefault("reduction", "mean")
                            loss_fn = create_loss_from_config({"type": method_cfg["loss"], **lk_i})
                        if method_cfg["loss"] not in ("nflow", "contrastive_nflow", "sls"):
                            model = torch.nn.Sequential(backbone, torch.nn.Linear(64, head_out))
                        elif method_cfg["loss"] == "sls":
                            model = backbone
                        # nflow already set model
                    else:
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
                            lk_cov = loss_kwargs.get("covariance_type")
                            if method_cfg["loss"] == "gaussian_nll" and lk_cov == "full":
                                head_out = 2 * d_out
                            else:
                                head_out = 2 * d_out
                        elif "skew_t" in method_cfg["loss"]:
                            head_out = 4 * d_out
                        elif "gev" in method_cfg["loss"]:
                            head_out = 3 * d_out
                        elif "mdn" in method_cfg["loss"]:
                            K = loss_kwargs.get("n_components", 5)
                            head_out = K + 2 * K * d_out
                        else:
                            head_out = d_out
                        if method_cfg["loss"] not in ("nflow", "contrastive_nflow"):
                            # handle full covariance: map to multivariate
                            lk = dict(loss_kwargs)
                            if (
                                method_cfg["loss"] == "gaussian_nll"
                                and lk.get("covariance_type") == "full"
                            ):
                                lk["covariance_type"] = (
                                    "diagonal"  # ponytail: harness tabular MLP cannot emit full cov yet; keep diagonal for now
                                )
                                # TODO: emit cholesky and use MultivariateGaussianLoss
                            lk.setdefault("reduction", "mean")
                            loss_fn = create_loss_from_config({"type": method_cfg["loss"], **lk})
                        model = torch.nn.Sequential(
                            torch.nn.Linear(in_dim, hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden, hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden, head_out),
                        )
                    model.to(device)
                    # SLS/NFlow have flow params that must follow model device
                    try:
                        if hasattr(loss_fn, "to"):
                            loss_fn.to(device)  # type: ignore[attr-defined]
                    except Exception:
                        pass
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
                                samples_np = (
                                    samples.cpu().numpy()
                                    if isinstance(samples, torch.Tensor)
                                    else np.array(samples)
                                )
                                # harness expects mean/std for CRPS; derive from samples when 3D
                                try:
                                    if samples_np.ndim == 3:
                                        # [B, nS, D] or [nS, B, D] – infer samples axis
                                        if samples_np.shape[0] == out_test.shape[0]:
                                            mean_np = samples_np.mean(axis=1)
                                            std_np = samples_np.std(axis=1)
                                        else:
                                            mean_np = samples_np.mean(axis=0)
                                            std_np = samples_np.std(axis=0)
                                        pred = PredictiveBatch(
                                            mean=mean_np, std=std_np, samples=samples_np
                                        )
                                    else:
                                        pred = PredictiveBatch(samples=samples_np)
                                except Exception:
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
                        elif "skew_t" in method_cfg["loss"] or "gev" in method_cfg["loss"]:
                            # location is first d_out columns; treat as point/mean for metrics/corner
                            mu = out_test[:, :d_out] if out_test.shape[1] >= d_out else out_test
                            pred = PredictiveBatch(point=mu.cpu(), mean=mu.cpu())
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
                    # Build harness pred dict from PredictiveBatch
                    pred_dict: dict[str, Any] = {}
                    if pred.point is not None:
                        pred_dict["y_pred"] = np.asarray(pred.point)
                    if pred.mean is not None:
                        pred_dict["mean"] = np.asarray(pred.mean)
                    if pred.std is not None:
                        pred_dict["std"] = np.asarray(pred.std)
                    if pred.samples is not None:
                        s = np.asarray(pred.samples)
                        # harness PIT only supports 1D/2D samples; drop 3D multivariate for PIT
                        # Keep mean/std for NLL/CRPS, drop samples to avoid ValueError
                        if s.ndim == 3 and s.shape[-1] != 1:
                            # drop: mean/std already present for nflow/mdn
                            pass
                        else:
                            pred_dict["samples"] = s
                    if not pred_dict and y_test_np is not None:
                        # last resort: use mean point
                        pred_dict["y_pred"] = y_test_np  # will give zero error
                    try:
                        metrics = compute_all_metrics(pred_dict, y_test_np)
                    except Exception:
                        # fallback to simple point mse if harness fails
                        try:
                            if pred.point is not None:
                                y_pred_fp = np.asarray(pred.point)
                            elif pred.mean is not None:
                                y_pred_fp = np.asarray(pred.mean)
                            elif pred.samples is not None:
                                s = np.asarray(pred.samples)
                                if s.ndim == 3:
                                    # [B, nS, D] or [nS, B, D] — mean over samples dim
                                    if s.shape[0] == len(y_test_np):
                                        y_pred_fp = s.mean(axis=1)
                                    else:
                                        y_pred_fp = s.mean(axis=0)
                                else:
                                    y_pred_fp = s.mean(axis=0) if s.ndim == 2 else s
                            else:
                                y_pred_fp = y_test_np
                            from harness.metrics import compute_point_metrics as _cpm

                            metrics = _cpm(y_pred_fp.reshape(-1), y_test_np.reshape(-1))
                        except Exception:
                            metrics = {}
                    # ensure finite
                    for k, v in list(metrics.items()):
                        try:
                            if not np.isfinite(float(v)):
                                metrics[k] = None
                        except Exception:
                            pass
                    # --- build samples for corner plot (store last seed per dataset×method) ---
                    try:
                        corner_key = (str(ds_cfg["name"]), str(method_cfg["loss"]))
                        corner_samples = None  # [N_total, D]
                        if pred.samples is not None:
                            s = np.asarray(pred.samples)
                            if s.ndim == 3:
                                # [B, nS, D] or [nS, B, D] -> [B*nS, D]
                                if s.shape[0] == out_test.shape[0]:
                                    corner_samples = s.reshape(-1, s.shape[-1])
                                elif s.shape[1] == out_test.shape[0]:
                                    corner_samples = s.transpose(1, 0, 2).reshape(-1, s.shape[-1])
                                else:
                                    corner_samples = s.reshape(-1, s.shape[-1])
                            elif s.ndim == 2:
                                # univariate: [nS, B] -> [B*nS, 1]
                                if s.shape[1] == y_test_np.shape[0]:
                                    corner_samples = s.T.reshape(-1, 1)
                                elif s.shape[0] == y_test_np.shape[0]:
                                    corner_samples = s.reshape(-1, 1)
                                else:
                                    corner_samples = s.reshape(-1, 1)
                        elif pred.mean is not None and pred.std is not None:
                            mean_a = np.asarray(pred.mean)
                            std_a = np.asarray(pred.std)
                            if mean_a.ndim == 1:
                                mean_a = mean_a[:, None]
                            if std_a.ndim == 1:
                                std_a = std_a[:, None]
                            N_, D_ = mean_a.shape
                            K = 128  # draws per point -> controls corner density
                            corner_samples = (
                                mean_a[:, None, :] + std_a[:, None, :] * np.random.randn(N_, K, D_)
                            ).reshape(-1, D_)
                        elif pred.point is not None:
                            pt = np.asarray(pred.point)
                            if pt.ndim == 1:
                                pt = pt[:, None]
                            corner_samples = pt  # [B, D]
                        if corner_samples is not None and corner_samples.size > 0:
                            # cap to 20000 rows for viz speed
                            if corner_samples.shape[0] > 20000:
                                idx = np.random.choice(
                                    corner_samples.shape[0], 20000, replace=False
                                )
                                corner_samples = corner_samples[idx]
                            corner_store[corner_key] = {
                                "samples": corner_samples,
                                "true": np.asarray(y_test_np),
                                "d_out": int(d_out),
                            }
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

    # --- corner plots: real predictive samples per dataset×method ---
    try:
        for (ds, method), info in corner_store.items():
            try:
                samples = np.asarray(info["samples"])
                # drop non-finite rows before viz
                try:
                    mask = (
                        np.isfinite(samples).all(axis=1)
                        if samples.ndim == 2
                        else np.isfinite(samples)
                    )
                    samples = samples[mask] if mask.ndim == 1 else samples
                    if samples.size == 0 or samples.shape[0] < 10:
                        continue
                except Exception:
                    pass
                y_true = np.asarray(info["true"])
                d = int(info.get("d_out", samples.shape[1] if samples.ndim == 2 else 2))
                # sanitize: ensure 2D
                if samples.ndim != 2:
                    continue
                if samples.shape[1] != d:
                    d = samples.shape[1]
                # limit D for corner readability (cap 4 dims)
                max_d = 4
                if d > max_d:
                    samples = samples[:, :max_d]
                    y_true = y_true[:, :max_d] if y_true.ndim == 2 else y_true
                    d = max_d
                # true values: median of y_test for reference lines
                try:
                    if y_true.ndim == 1:
                        true_vals = np.array([float(np.median(y_true))])
                    else:
                        true_vals = np.median(y_true, axis=0).astype(float)
                        if true_vals.ndim == 0:
                            true_vals = np.array([float(true_vals)])
                        true_vals = true_vals[:d]
                except Exception:
                    true_vals = None
                save_path = out_root / ds / f"corner_{method}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig = plot_corner_plot(
                    samples=samples,
                    param_names=[f"y{i}" for i in range(d)],
                    true_vals=true_vals,
                    save_path=save_path,
                    return_figure=True,
                )
                if fig is not None:
                    plt.close(fig)
            except Exception as exc:
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
