"""Conformal prediction leaderboard — rank all methods across all libraries.

Runs the conformal suite on 2-6 datasets, collects Coverage / Width / IntervalScore,
and produces a ranked table sorted by IntervalScore (lower is better).

Usage:
    uv run python leaderboard_conformal.py \\
        --datasets DATASET1,DATASET2,... [--epochs N] [--output FILE]

Outputs:
    - Ranked table to stdout
    - JSON file with full results (for downstream analysis)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from harness.runners import HarnessConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conformal prediction leaderboard")
    p.add_argument(
        "--datasets",
        default="california_housing,diabetes,synthetic_hetero,house_sales",
        help="Comma-separated dataset names",
    )
    p.add_argument("--epochs", type=int, default=60, help="Training epochs for MLP methods")
    p.add_argument("--output", default="leaderboard_results.json", help="Output JSON path")
    p.add_argument("--smoke", action="store_true", help="Smoke test (reduced data, 10 epochs)")
    p.add_argument(
        "--tabred-samples", type=int, default=10000,
        help="Max samples for TabReD datasets (default: 10000)",
    )
    return p.parse_args()


def run_leaderboard(
    datasets: list[str],
    epochs: int,
    smoke: bool,
    output_path: str,
    *,
    tabred_samples: int = 10000,
) -> None:
    """Run conformal suite on all datasets and produce a ranked table."""
    from suites.tabular.conformal import ConformalSuite

    config = HarnessConfig(seed=42, smoke=smoke, standardize_targets=True)
    if smoke:
        config.epochs = 10  # type: ignore[assignment]

    suite = ConformalSuite()
    suite.tabred_samples = tabred_samples  # used by run() for TabReD datasets
    all_rows: list[dict] = []

    for ds in datasets:
        print(f"  Dataset: {ds} ...", flush=True)
        try:
            rows = suite.run(ds, config=config)
            all_rows.extend(rows)
            methods_ran = [r["Method"] for r in rows if r.get("Coverage") is not None]
            print(f"    {len(methods_ran)} methods completed", flush=True)
        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)

    if not all_rows:
        print("No results collected.")
        return

    # Save full results
    with open(output_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=float)
    print(f"\nFull results saved to {output_path}")

    # --- Ranked table (by IntervalScore, then Width) ---
    valid = [r for r in all_rows if r.get("IntervalScore") is not None]
    if not valid:
        print("No valid rows with IntervalScore.")
        return

    # Round metrics for display
    for r in valid:
        r["_cov"] = round(r["Coverage"], 4)
        r["_w"] = round(r["Width"], 4)
        r["_is"] = round(r["IntervalScore"], 4)

    # Sort: lower IntervalScore is better; break ties with Width
    valid.sort(key=lambda r: (r["_is"], r["_w"]))

    # Print header
    print()
    print(f"{'Rank':<5s} {'Method':<32s} {'Library':<12s} {'Dataset':<22s} "
          f"{'Coverage':>9s} {'Width':>10s} {'IntervalScore':>14s}")
    print("-" * 110)

    for i, r in enumerate(valid, start=1):
        method = r["Method"].split("/", 1)[1] if "/" in r.get("Method", "") else r.get("Method", "")
        ds = r.get("Dataset", "")
        lib = r.get("Library", "")
        print(
            f"{i:<5d} {method:<32s} {lib:<12s} {ds:<22s} "
            f"{r['_cov']:>9.4f} {r['_w']:>10.4f} {r['_is']:>14.4f}"
        )
        ds = r.get("Dataset", "")
        lib = r.get("Library", "")
        print(
            f"{i:<5d} {method:<32s} {lib:<12s} {ds:<22s} "
            f"{r['_cov']:>9.4f} {r['_w']:>10.4f} {r['_is']:>14.4f}"
        )

    # Summary by method (average IntervalScore across datasets)
    print()
    print("--- Method Summary (avg IntervalScore across datasets, lower = better) ---")

    method_scores: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        method_scores[r["Method"]].append(r["_is"])

    ranked_methods = sorted(method_scores.items(), key=lambda x: np.mean(x[1]))
    print(f"{'Rank':<5s} {'Method':<32s} {'Avg IS':>10s} {'N datasets':>12s}")
    print("-" * 65)
    for i, (method, scores) in enumerate(ranked_methods, start=1):
        print(f"{i:<5d} {method:<32s} {np.mean(scores):>10.4f} {len(scores):>12d}")


if __name__ == "__main__":
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"Datasets: {datasets}")
    print(f"Epochs: {args.epochs}, Smoke: {args.smoke}")
    print()
    run_leaderboard(datasets, args.epochs, args.smoke, args.output,
                    tabred_samples=args.tabred_samples)
