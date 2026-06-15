#!/usr/bin/env python
"""Run all harness suites and write a unified JSON artifact.

Usage::

    # CI smoke (tiny data, 2 epochs)
    uv run python tools/run_all.py --smoke --output results/latest/harness_smoke.json

    # Full run
    uv run python tools/run_all.py --output results/latest/harness_full.json

    # Specific suites and datasets
    uv run python tools/run_all.py --suites conformal,tweedie --datasets diabetes --output results/latest/selected.json

    # Validate torchregress install before running
    uv run python tools/run_all.py --validate --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure harness and suites are importable from repo root
_repo_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _repo_root)
sys.path.insert(0, str(Path(_repo_root) / "src"))

from harness import (
    HarnessConfig,
    print_comparison_summary,
    print_fairness_notes,
    write_harness_json,
)
from harness.datasets import load_dataset
from harness.runners import HarnessRunner
from tools._suite_registry import ALL_SUITES
from tools._validate import validate_suites, validate_torchregress


def main() -> None:
    parser = argparse.ArgumentParser(description="Run torchregress-harness suites.")
    parser.add_argument(
        "--suites",
        type=str,
        default=None,
        help="Comma-separated suite names (default: all)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names (default: each suite's supported datasets)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny data / few epochs for CI smoke check",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Training epochs per method",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run torchregress + suite validation checks before benchmarks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest/harness.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    # --- Validation mode ---
    if args.validate:

        print("=== torchregress validation ===")
        tr_ok, tr_msgs = validate_torchregress()
        for m in tr_msgs:
            print(m)
        print()

        print("=== Suite validation ===")
        suite_ok, suite_msgs = validate_suites(ALL_SUITES)
        for m in suite_msgs:
            print(m)
        print()

        if not tr_ok or not suite_ok:
            print("Validation FAILED — fix the issues above before running benchmarks.")
            sys.exit(1)
        print("Validation PASSED — all checks green.\n")

    # Select suites
    suite_names = args.suites.split(",") if args.suites else list(ALL_SUITES.keys())
    suites: list[HarnessRunner] = []
    for name in suite_names:
        if name not in ALL_SUITES:
            print(f"Warning: unknown suite {name!r}, skipping")
            continue
        suites.append(ALL_SUITES[name]())

    if not suites:
        print("No suites selected — exiting.")
        sys.exit(1)

    config = HarnessConfig(
        seed=args.seed,
        epochs=2 if args.smoke else args.epochs,
        smoke=args.smoke,
    )

    # Run each suite on its supported datasets
    dataset_names = args.datasets.split(",") if args.datasets else None
    all_rows: list[dict[str, object]] = []

    for suite in suites:
        datasets = dataset_names or suite.supported_datasets
        for dataset_name in datasets:
            print(f"\n=== Suite: {suite.name} · Dataset: {dataset_name} ===")
            try:
                try:
                    split = load_dataset(dataset_name, n_samples=200 if args.smoke else None)
                except KeyError:
                    print(f"  Dataset {dataset_name!r} not found — skipping")
                    continue

                rows = suite.run(dataset_name, config, splits=split)
                all_rows.extend(rows)

                for row in rows:
                    # Interval/conformal suites don't produce RMSE — use relevant metric
                    has_result = (
                        row.get("RMSE") is not None
                        or row.get("Coverage") is not None
                        or row.get("MAE") is not None
                        or row.get("TweedieDeviance") is not None
                    )
                    status = "✓" if has_result else "✗"
                    print(f"  {status} {row['Method']}")

            except NotImplementedError:
                print(f"  Suite {suite.name!r} not yet implemented — skipping")
                continue

    if not all_rows:
        print("\nNo rows produced — exiting.")
        sys.exit(0)

    print_fairness_notes(
        title="Harness run",
        seed_policy=f"fixed seed ({config.seed})",
        train_budget=f"{config.epochs} epochs, batch={config.batch_size}, lr={config.lr}",
        metric_policy="RMSE, MAE, R2, NLL, CRPS, Coverage, Width, IntervalScore, ECE, runtime",
    )
    print_comparison_summary("Harness Results", all_rows)

    out_path = write_harness_json(
        args.output,
        config=config,
        rows=all_rows,
        suites=[s.name for s in suites],
        datasets=sorted({r.get("Dataset", "?") for r in all_rows}),
    )
    print(f"\nWrote harness artifact: {out_path}")


if __name__ == "__main__":
    main()
