"""Harness core: reproducible comparison helpers.

These are extracted and generalized from ``torchregress.comparison``
so the harness is self-contained for its infrastructure layer, but the
actual regression methods are always called through ``torchregress`` itself.
"""

from __future__ import annotations

import json
import platform
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch

# Keys that are row metadata, not performance metrics.
# (Device/SourceURL/SyntheticFallback are provenance fields present on every
# row — treating them as metrics made error rows look like successes.)
_NON_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "Method",
        "Library",
        "LibraryVersion",
        "Dataset",
        "Suite",
        "Notes",
        "Device",
        "SourceURL",
        "SyntheticFallback",
    }
)


def set_harness_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch RNGs for reproducible harness runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timed_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, elapsed_seconds)``."""
    start = perf_counter()
    result = fn(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, elapsed


def pip_git_commit(dist_name: str) -> str | None:
    """Return the git commit recorded by pip for ``dist_name`` when installed from git.

    pip writes ``direct_url.json`` into the package's ``.dist-info`` directory
    for VCS installs (``pip install git+https://...@<sha>``); PyPI wheels have
    no such file and this returns ``None``.

    Used to self-report the exact upstream commit a benchmark row was run
    against (MAPIE / torch-uncertainty) so external-library comparisons are
    reproducible from the artifacts alone.
    """
    try:
        from importlib.metadata import distribution

        raw = distribution(dist_name).read_text("direct_url.json")
        if not raw:
            return None
        import json

        vcs = json.loads(raw).get("vcs_info") or {}
        return vcs.get("commit_id")
    except Exception:
        return None


def compute_point_metrics(
    y_pred: np.ndarray | torch.Tensor, y_true: np.ndarray | torch.Tensor
) -> dict[str, float]:
    """Compute common point-prediction regression metrics.

    Accepts both numpy arrays and torch tensors.
    """
    y_pred_np = np.asarray(y_pred).reshape(-1)
    y_true_np = np.asarray(y_true).reshape(-1)

    mse = float(np.mean((y_pred_np - y_true_np) ** 2))
    mae = float(np.mean(np.abs(y_pred_np - y_true_np)))
    rmse = float(np.sqrt(mse))
    medae = float(np.median(np.abs(y_pred_np - y_true_np)))

    y_true_mean = float(np.mean(y_true_np))
    ss_res = float(np.sum((y_true_np - y_pred_np) ** 2))
    ss_tot = float(np.sum((y_true_np - y_true_mean) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Pearson correlation (uncertainty-toolbox parity: corr).
    if len(y_pred_np) >= 2 and np.std(y_pred_np) > 0 and np.std(y_true_np) > 0:
        corr = float(np.corrcoef(y_pred_np, y_true_np)[0, 1])
    else:
        corr = float("nan")

    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "MedAE": medae, "R2": r2, "Corr": corr}


def print_fairness_notes(
    *,
    title: str,
    seed_policy: str,
    train_budget: str,
    metric_policy: str,
) -> None:
    """Print a compact comparability statement for harness outputs."""
    print(f"\n[{title} | harness fairness]")
    print(f"  seeds:   {seed_policy}")
    print(f"  budget:  {train_budget}")
    print(f"  metrics: {metric_policy}")


def print_comparison_summary(
    title: str,
    rows: list[dict[str, Any]],
    metric_order: list[str] | None = None,
) -> None:
    """Print an aligned summary table for a harness run."""
    if not rows:
        print(f"\n{title}\n(no rows)")
        return

    # Auto-discover all metric keys from rows, then reorder by metric_order preference.
    all_metric_keys: set[str] = set()
    for row in rows:
        for key in row:
            if key not in _NON_METRIC_KEYS:
                all_metric_keys.add(key)

    metric_order = metric_order or [
        "Dataset",
        "RMSE",
        "MAE",
        "MedAE",
        "R2",
        "Corr",
        "NLL",
        "CRPS",
        "Sharpness",
        "Coverage",
        "Width",
        "IntervalScore",
        "ECE",
        "MACE",
        "RMSCE",
        # PIT calibration diagnostics (Proposal #9)
        "PIT_Mean",
        "PIT_Std",
        "PIT_KS",
        "PIT_CvM",
        # dev_summary / bin metrics
        "AvgBinCoverage",
        "AvgBinWidth",
        "CoverageDeviationMean",
        "CoverageDeviationAbsMean",
        "CoverageDeviationRMSE",
        "CoverageDeviation",
        "BinCenter",
        "BinSize",
        "BinCoverage",
        "BinMeanWidth",
        # timing
        "train_s",
        "eval_s",
    ]
    # Start with ordered metrics that are present, then append any remaining
    present_metrics: list[str] = []
    for m in metric_order:
        if m in all_metric_keys:
            present_metrics.append(m)
    # Append any metrics not in the preferred order (future-proofing)
    for m in sorted(all_metric_keys - set(present_metrics)):
        present_metrics.append(m)
    headers = ["Method", "Library", *present_metrics]
    include_notes = any(row.get("Notes") for row in rows)
    if include_notes:
        headers.append("Notes")

    table_rows: list[list[str]] = []
    for row in rows:
        vals = [
            str(row.get("Method", "")),
            str(row.get("Library", "")),
        ]
        for key in present_metrics:
            value = row.get(key)
            if value is None:
                vals.append("-")
            elif isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        if include_notes:
            vals.append(str(row.get("Notes", "")))
        table_rows.append(vals)

    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(f"\n{title}")
    print(_fmt(headers))
    print("-+".join("-" * w for w in widths))
    for r in table_rows:
        print(_fmt(r))


def write_harness_json(
    path: str | Path,
    *,
    config: Any,
    rows: list[dict[str, Any]],
    suites: list[str] | None = None,
    datasets: list[str] | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Write a machine-readable harness run artifact.

    Produces a unified JSON artifact following the harness output schema.
    """
    import datetime

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload: Any
    if is_dataclass(config) and not isinstance(config, type):
        config_payload = asdict(config)
    elif hasattr(config, "__dict__"):
        config_payload = dict(vars(config))
    else:
        config_payload = config

    # Classification precedence: an error row may still carry metric-shaped
    # keys (Device, timings) or even stale metric values, so classify by Notes
    # first and only fall back to the any-metric heuristic otherwise.
    n_skipped = 0
    n_error = 0
    n_ok = 0
    for r in rows:
        note = str(r.get("Notes", "")).lower()
        if "error" in note or "failed" in note:
            n_error += 1
        elif "skipped" in note:
            n_skipped += 1
        elif any(r.get(m) is not None and m not in _NON_METRIC_KEYS for m in r):
            n_ok += 1
        else:
            n_error += 1  # no note, no metrics — treat as failure, not success

    payload: dict[str, Any] = {
        "artifact": "harness_run",
        "version": 1,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "suites": suites or [],
        "datasets": datasets or [],
        "config": config_payload,
        "rows": rows,
        "notes": notes or [],
        "aggregate": {
            "n_total": len(rows),
            "n_ok": n_ok,
            "n_skipped": n_skipped,
            "n_error": n_error,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
