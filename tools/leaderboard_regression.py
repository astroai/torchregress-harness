#!/usr/bin/env python
"""Comprehensive regression leaderboard across every suite × dataset.

Mirrors the shape of :mod:`leaderboard_conformal` but generalizes to every
suite registered in ``tools/_suite_registry.ALL_SUITES`` (12 suites at the time
of writing — conformal, bayesian_linear, sklearn_baselines,
distributional_bins, multivariate_intervals, calibration, tweedie, ensembles,
foundation_model, robust_regression, uq, pzflow_parity).

For each (suite, dataset) cell, the script:

1. Instantiates the suite (``suite_cls()``) and sets ``suite.tabred_samples``
   where supported (so the synthetic market-scale TabReD datasets stay
   tractable on a full leaderboard run).
2. Calls ``suite.run(dataset, config=cfg)`` and collects the raw result rows.
3. Per-method aggregation: strips the suite prefix from ``Method``
   (``"conformal/Split+MLP"`` → ``"Split+MLP"``), groups by ``(Method, Library)``,
   and computes the suite-mean of every leaderboard metric over all
   (suite, dataset) cells the method ran in.
4. Ranks methods by **IntervalScore** (the canonical proper scoring rule for
   prediction intervals — penalises both width and miss-rate) when available,
   falling back to **RMSE** for methods that produce only point predictions.

Outputs::

    # Default JSON:   leaderboard_results.json
    # Default Markdown: leaderboard_results.md

Both paths are configurable via ``--output`` / ``--markdown``.  JSON contains
the full per-cell result rows plus a compact ``leaderboard`` summary; Markdown
renders the same data as readable tables (a single ranked leaderboard plus a
per-suite × per-dataset breakdown for forensic inspection).

Usage examples::

    # Default datasets × all suites
    uv run python tools/leaderboard_regression.py

    # Custom dataset shortlist + fast smoke
    uv run python tools/leaderboard_regression.py --smoke \\
        --datasets diabetes,california_housing

    # Subset of suites (e.g. for ablation runs)
    uv run python tools/leaderboard_regression.py \\
        --suites conformal,bayesian_linear,distributional_bins

Argument format (--datasets / --suites)
---------------------------------------
Both flags accept **comma-separated** names only.  Each flag consumes ONE
``argparse`` value; space-separated positional arguments (``--suites a b``)
are rejected with ``unrecognized arguments: b``.  Single names also work
(e.g. ``--suites bayesian_linear`` for one suite).  Unknown names
(typos, dependencies not installed) are skipped with a ``WARNING: unknown
suite '...', skipping`` line printed to stdout; the rest of the
leaderboard still runs.

For a registered-name list:
    - Suites:    ``tools/_suite_registry.ALL_SUITES.keys()``
    - Datasets:  ``harness.datasets.DATASET_REGISTRY.keys()``
"""

from __future__ import annotations

import os

# Must precede the first torch import (suite registry pulls it in): MIG
# partitions break torch's NVML-based allocator query otherwise.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Make `tools/_suite_registry` and `harness` importable when invoked as a
# script directly via `python tools/leaderboard_regression.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tools._suite_registry import ALL_SUITES  # noqa: E402

from harness.runners import HarnessConfig  # noqa: E402

# Keys that are row metadata, not performance metrics.  Mirrors the
# ``_NON_METRIC_KEYS`` set in ``src/harness/core.py`` (re-declared locally
# to keep this script dependency-light — no cross-module import).
_NON_METRIC_KEYS = frozenset(
    {
        "Method",
        "Library",
        "Dataset",
        "Suite",
        "Notes",
        "SourceURL",
        "train_s",
        "eval_s",
    }
)

# Metrics featured on the leaderboard surface (display + ranking).
# Order is the table-column order in the Markdown output.
LEADERBOARD_METRICS = ("RMSE", "IntervalScore", "Coverage", "ECE")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Comprehensive regression leaderboard — ranks every method "
            "from every suite by suite-mean RMSE / IntervalScore / "
            "Coverage / ECE."
        ),
    )
    p.add_argument(
        "--datasets",
        default="diabetes,california_housing,synthetic_hetero,house_sales",
        help=(
            "Comma-separated dataset names.  ONE argparse value — "
            "space-separated positional arguments are rejected with "
            "'unrecognized arguments'.  Default targets small → medium "
            "tabular datasets that all suites support.  Use --smoke to "
            "trim training cost before scale-out."
        ),
    )
    p.add_argument(
        "--suites",
        default=",".join(ALL_SUITES.keys()),
        help=(
            "Comma-separated suite names.  ONE argparse value — "
            "space-separated positional arguments are rejected with "
            "'unrecognized arguments'.  Single names also work "
            "('--suites bayesian_linear').  Default: every suite in "
            "tools/_suite_registry.py.  Unknown names are skipped with a "
            "warning."
        ),
    )
    p.add_argument("--epochs", type=int, default=60, help="Training epochs for MLP methods.")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for training: auto | cpu | cuda | cuda:0 | mps (default: auto — detects the session)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test (reduced data, 10 epochs) for fast iteration.",
    )
    p.add_argument(
        "--output",
        default="leaderboard_results.json",
        help="Output JSON artifact path (default: leaderboard_results.json).",
    )
    p.add_argument(
        "--markdown",
        default=None,
        help="Output Markdown path (default: same name as JSON with .md extension).",
    )
    p.add_argument(
        "--tabred-samples",
        type=int,
        default=10000,
        help="Max samples for TabReD datasets (default: 10000).",
    )
    p.add_argument(
        "--target-coverage",
        type=float,
        default=0.9,
        help=(
            "Nominal interval coverage target.  Used in the Markdown header "
            "and to compute CoverageDeviation (mean |Coverage - target|)."
        ),
    )
    return p.parse_args()


def _is_skipped(row: dict[str, Any]) -> bool:
    """True when a row's Notes flag a skip/failure (per the rows-from-run() contract).

    Known-bad rows (intrinsic library bugs surfaced via the ``_KNOWN_BAD``
    registry pattern — see ``PzflowParitySuite._KNOWN_BAD`` and
    ``PARITY_REPORT.md`` §12) are NOT skipped: they carry a deterministic
    ``Notes="known-bad: …"`` prefix and are designed to surface in the
    leaderboard as a documented limitation, not as a transient failure.
    """
    notes = str(row.get("Notes", "")).lower()
    if notes.startswith("known-bad:"):
        return False
    # Colon-suffixed markers only (see tools/run_all.py): bare "error"
    # false-positives on notes like "error-aware feature encoder".
    return any(token in notes for token in ("skipped:", "failed:", "error:"))


def _has_any_metric(row: dict[str, Any]) -> bool:
    """True when a row carries at least one non-null metric value."""
    return any(v is not None and k not in _NON_METRIC_KEYS for k, v in row.items())


def run_leaderboard(
    *,
    datasets: list[str],
    suites: list[str],
    epochs: int,
    smoke: bool,
    output_path: str,
    markdown_path: str | None,
    tabred_samples: int = 10000,
    target_coverage: float = 0.9,
    device: str = "auto",
) -> None:
    """Run every requested suite × dataset pair and emit the leaderboard artifact."""
    from harness.devices import set_default_device

    resolved_device = set_default_device(device)
    cfg = HarnessConfig(seed=42, smoke=smoke, standardize_targets=True, device=str(resolved_device))
    if smoke:
        cfg.epochs = 10

    all_rows: list[dict[str, Any]] = []
    n_attempts = 0
    n_completed = 0

    for suite_name in suites:
        if suite_name not in ALL_SUITES:
            print(f"  WARNING: unknown suite {suite_name!r}, skipping", flush=True)
            continue
        suite_cls = ALL_SUITES[suite_name]
        suite = suite_cls()
        # TabReD sample-size override (mirrors leaderboard_conformal.py)
        if hasattr(suite, "tabred_samples"):
            suite.tabred_samples = tabred_samples

        for ds in datasets:
            print(f"  Suite={suite_name}  Dataset={ds} ...", flush=True)
            try:
                rows = suite.run(ds, config=cfg)
                all_rows.extend(rows)
                ok = [r for r in rows if not _is_skipped(r) and _has_any_metric(r)]
                n_completed += len(ok)
                print(f"    {len(ok)} methods completed", flush=True)
            except Exception as exc:  # noqa: BLE001 — broad to keep leadboard robust
                print(f"    ERROR: {exc}", flush=True)
            n_attempts += 1

    if not all_rows:
        print("No results collected; aborting.")
        return

    summary = aggregate_per_method(all_rows, target_coverage=target_coverage)
    artifact = {
        "metadata": {
            "datasets": datasets,
            "suites": suites,
            "epochs": cfg.epochs,
            "smoke": smoke,
            "target_coverage": target_coverage,
            "device": str(resolved_device),
            "n_raw_rows": len(all_rows),
            "n_methods_total": len(summary),
            "n_attempts": n_attempts,
            "n_completed": n_completed,
        },
        "rows": all_rows,
        "leaderboard": summary,
    }
    Path(output_path).write_text(
        json.dumps(artifact, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\nFull results saved to {output_path}")

    md_path = Path(markdown_path) if markdown_path else Path(output_path).with_suffix(".md")
    md = render_leaderboard_markdown(summary, all_rows, target_coverage=target_coverage)
    Path(md_path).write_text(md, encoding="utf-8")
    print(f"Markdown leaderboard saved to {md_path}")

    print_leaderboard_stdout(summary, target_coverage=target_coverage)


def aggregate_per_method(
    rows: list[dict[str, Any]],
    *,
    target_coverage: float = 0.9,
) -> list[dict[str, Any]]:
    """Aggregate raw rows into a per-(Method, Library) leaderboard summary.

    Strips the suite prefix from ``Method`` (e.g. ``"conformal/Split+MLP"`` →
    ``"Split+MLP"``) so the same method exposed by multiple suites is
    averaged across them.  Methods without intervals fall back to ranking
    by ``RMSE``.

    Sort key
    --------
    Primary = ``mean_IntervalScore`` when available, else ``mean_RMSE``
    (lower is better in both cases).  Ties break on ``CoverageDeviation``
    → ``mean_ECE`` → ``N_Runs`` (descending — more runs is preferable).
    """
    bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _is_skipped(row):
            continue
        method_full = str(row.get("Method", "?"))
        # Strip suite prefix ("suite/method" → "method"); preserve the bare
        # method name when no "/" is present.
        method = method_full.split("/", 1)[1] if "/" in method_full else method_full
        library = str(row.get("Library", "?"))
        bucket[(method, library)].append(row)

    summaries: list[dict[str, Any]] = []

    for (method, library), method_rows in bucket.items():
        suites_seen = sorted(
            {r.get("Suite", "?") for r in method_rows}
        )  # Distinct (suite, dataset) cells — a method may run across
        # multiple suites on the same dataset (e.g. ``BayesianRidge`` in
        # both ``conformal`` and ``bayesian_linear`` ⇒ 2 cells, 1 dataset).
        # ``N_Cells`` is the canonical count; ``N_Datasets_Distinct`` is
        # exposed separately for forensic inspection.
        n_cells = len({(r.get("Suite", "?"), r.get("Dataset", "?")) for r in method_rows})
        n_datasets_distinct = len({r.get("Dataset", "?") for r in method_rows})
        rmse_vals = [r["RMSE"] for r in method_rows if r.get("RMSE") is not None]
        is_vals = [r["IntervalScore"] for r in method_rows if r.get("IntervalScore") is not None]
        cov_vals = [r["Coverage"] for r in method_rows if r.get("Coverage") is not None]
        ece_vals = [r["ECE"] for r in method_rows if r.get("ECE") is not None]

        coverages_np = np.asarray(cov_vals, dtype=np.float64) if cov_vals else None
        cov_dev = (
            float(np.mean(np.abs(coverages_np - target_coverage)))
            if coverages_np is not None
            else float("nan")
        )

        if is_vals:
            primary_metric = "IntervalScore"
            primary_value = float(np.mean(is_vals))
        elif rmse_vals:
            primary_metric = "RMSE"
            primary_value = float(np.mean(rmse_vals))
        else:
            primary_metric = "RMSE"
            primary_value = float("nan")

        summaries.append(
            {
                "Method": method,
                "Library": library,
                "Suites": suites_seen,
                "N_Suites": len(suites_seen),
                "N_Cells": n_cells,
                "N_Datasets_Distinct": n_datasets_distinct,
                "N_Runs": len(method_rows),
                "mean_RMSE": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
                "mean_IntervalScore": (float(np.mean(is_vals)) if is_vals else float("nan")),
                "mean_Coverage": (float(np.mean(cov_vals)) if cov_vals else float("nan")),
                "CoverageDeviation": cov_dev,
                "mean_ECE": float(np.mean(ece_vals)) if ece_vals else float("nan"),
                "n_metric_rmse": len(rmse_vals),
                "n_metric_is": len(is_vals),
                "n_metric_cov": len(cov_vals),
                "n_metric_ece": len(ece_vals),
                "primary_metric": primary_metric,
                "primary_value": primary_value,
            },
        )

    def _safe_finite(value: float) -> float:
        """NaN-safe sort-key mapper: NaN → +inf so they sort to the bottom."""
        return value if not np.isnan(value) else float("inf")

    # Primary-rank key: ``IntervalScore`` methods (canonical proper-score
    # for prediction intervals) come before RMSE-only point methods — the
    # ``IntervalScore`` value range (~0–∞) and ``RMSE`` value range (~0–∞)
    # are NOT directly comparable and a method with ``RMSE = 0.5`` would
    # otherwise sort ahead of one with ``IntervalScore = 250`` simply
    # because 0.5 < 250 numerically.
    _METRIC_RANK = {"IntervalScore": 0, "RMSE": 1}

    def sort_key(s: dict[str, Any]) -> tuple[int, float, float, float, int]:
        # More runs (-N_Runs) is "better" so less-noise methods win ties
        # all the way down.  Numerically we encode this as a smaller sort
        # key — e.g. N_Runs=10 sorts before N_Runs=2.
        return (
            _METRIC_RANK.get(s["primary_metric"], 2),
            _safe_finite(s["primary_value"]),
            _safe_finite(s["CoverageDeviation"]),
            _safe_finite(s["mean_ECE"]),
            -s["N_Runs"],
        )

    summaries.sort(key=sort_key)
    return summaries


def render_leaderboard_markdown(
    summary: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    target_coverage: float,
) -> str:
    """Render the leaderboard as a Markdown document.

    Structure:
    - Header with metadata (rank count, target coverage, primary metric).
    - **Main leaderboard table** — one row per method, ranked by primary
      metric, showing all four leaderboard metrics (suite-mean across
      every (suite, dataset) cell).
    - **Per-suite breakdown** — for each suite, a sub-section per
      dataset with a 6-column table (Method | Library | RMSE |
      IntervalScore | Coverage | ECE) sorted by IntervalScore-then-RMSE.
    """
    out: list[str] = []
    out.append("# Comprehensive Regression Leaderboard")
    out.append("")
    out.append(f"- Target coverage: **{target_coverage:.2f}**")
    out.append(f"- Methods ranked: **{len(summary)}**")
    if summary:
        primary_metric = summary[0]["primary_metric"]
        out.append(
            f"- Primary ranking metric: **{primary_metric}** (lower is better)",
        )
    out.append("")

    # ----- Main leaderboard table -----
    out.append("## Leaderboard (suite-mean by method)")
    out.append("")
    out.append(
        "| Rank | Method | Library | N Suites | N Runs | "
        "RMSE | IntervalScore | Coverage | Coverage Dev | ECE |",
    )
    out.append("|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|")

    for i, s in enumerate(summary, start=1):
        out.append(
            f"| {i} | {s['Method']} | {s['Library']} | "
            f"{s['N_Suites']} | {s['N_Runs']} | "
            f"{_fmt(s['mean_RMSE'])} | {_fmt(s['mean_IntervalScore'])} | "
            f"{_fmt(s['mean_Coverage'])} | {_fmt(s['CoverageDeviation'])} | "
            f"{_fmt(s['mean_ECE'])} |",
        )
    out.append("")

    # ----- Per-suite × per-dataset breakdown -----
    out.append("## Per-Suite Breakdown")
    out.append("")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for row in raw_rows:
        if _is_skipped(row):
            continue
        suite = str(row.get("Suite", "?"))
        ds = str(row.get("Dataset", "?"))
        grouped[suite][ds].append(row)

    for suite_name in sorted(grouped):
        out.append(f"### Suite: `{suite_name}`")
        out.append("")
        for ds_name in sorted(grouped[suite_name]):
            ds_rows = list(grouped[suite_name][ds_name])

            # Sort the per-dataset table by IntervalScore-then-RMSE for
            # forensic inspection.  NaN-aware: missing values sink to the
            # bottom.
            def _sort_key(row: dict[str, Any]) -> tuple[float, float]:
                is_val = row.get("IntervalScore")
                rmse_val = row.get("RMSE")
                return (
                    is_val
                    if is_val is not None
                    else (rmse_val if rmse_val is not None else float("inf")),
                )

            ds_rows.sort(key=_sort_key)
            out.append(f"#### Dataset: `{ds_name}`")
            out.append("")
            out.append(
                "| Method | Library | RMSE | IntervalScore | Coverage | ECE |",
            )
            out.append("|:---|:---|---:|---:|---:|---:|")
            for row in ds_rows:
                method = row.get("Method", "?")
                if "/" in str(method):
                    method = str(method).split("/", 1)[1]
                out.append(
                    f"| {method} | {row.get('Library', '?')} | "
                    f"{_fmt(row.get('RMSE'))} | "
                    f"{_fmt(row.get('IntervalScore'))} | "
                    f"{_fmt(row.get('Coverage'))} | "
                    f"{_fmt(row.get('ECE'))} |",
                )
            out.append("")
    return "\n".join(out)


def print_leaderboard_stdout(
    summary: list[dict[str, Any]],
    *,
    target_coverage: float,
) -> None:
    """Print a compact leaderboard table to stdout for log-friendly tailing."""
    if not summary:
        return
    print()
    print("--- Comprehensive Regression Leaderboard (lower = better) ---")
    print(
        f"{'Rank':<5s} {'Method':<32s} {'Library':<14s} "
        f"{'N':>3s} {'RMSE':>10s} {'IntervalScore':>14s} "
        f"{'Coverage':>10s} {'CovDev':>10s} {'ECE':>10s}",
    )
    print("-" * 122)
    for i, s in enumerate(summary, start=1):
        print(
            f"{i:<5d} {s['Method'][:32]:<32s} {s['Library'][:14]:<14s} "
            f"{s['N_Runs']:>3d} {_fmt(s['mean_RMSE']):>10s} "
            f"{_fmt(s['mean_IntervalScore']):>14s} "
            f"{_fmt(s['mean_Coverage']):>10s} "
            f"{_fmt(s['CoverageDeviation']):>10s} "
            f"{_fmt(s['mean_ECE']):>10s}",
        )

    n_with_is = sum(1 for s in summary if not np.isnan(s["mean_IntervalScore"]))
    n_with_rmse = sum(1 for s in summary if not np.isnan(s["mean_RMSE"]))
    print(
        f"\n{n_with_is} methods produce intervals (IntervalScore available), "
        f"{n_with_rmse} methods produce point predictions (RMSE available) — "
        f"ranked by primary metric (IntervalScore where available, else RMSE).",
    )


def _fmt(value: Any) -> str:
    """Format a scalar metric value as 4 decimals, '-' for None/NaN.

    Coerces ``int`` to ``float`` for uniform formatting (``_fmt(0)`` returns
    ``"0.0000"`` rather than ``"0"``) so columns line up in Markdown.
    Booleans are treated as non-numeric (rendered via ``str``).
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        f_value = float(value)
        if np.isnan(f_value):
            return "-"
        return f"{f_value:.4f}"
    return str(value)


if __name__ == "__main__":
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    print(f"Datasets: {datasets}")
    print(f"Suites: {suites}")
    print(f"Epochs: {args.epochs}, Smoke: {args.smoke}")
    print()
    run_leaderboard(
        datasets=datasets,
        suites=suites,
        epochs=args.epochs,
        smoke=args.smoke,
        output_path=args.output,
        markdown_path=args.markdown,
        device=args.device,
        target_coverage=args.target_coverage,
    )
