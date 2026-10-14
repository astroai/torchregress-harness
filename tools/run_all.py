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

import os

# Must precede the first torch import (suite registry pulls it in): MIG
# partitions break torch's NVML-based allocator query otherwise.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import argparse
import sys
from pathlib import Path

# Ensure harness and suites are importable from repo root
_repo_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _repo_root)
sys.path.insert(0, str(Path(_repo_root) / "src"))

from tools._suite_registry import ALL_SUITES, split_supported
from tools._validate import (
    validate_gaussiannll_coverage,
    validate_image_backbones,
    validate_multivariate_gaussian_coverage,
    validate_nf_architecture,
    validate_sls_architecture,
    validate_sls_coverage,
    validate_suites,
    validate_torchregress,
)

from harness import (
    HarnessConfig,
    print_comparison_summary,
    print_fairness_notes,
    write_harness_json,
)
from harness.datasets import load_dataset
from harness.devices import set_default_device
from harness.runners import HarnessRunner


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
        "--device",
        type=str,
        default="auto",
        help="Torch device for training: auto | cpu | cuda | cuda:0 | mps (default: auto — detects the session)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Training epochs per method",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override image size for image datasets (e.g. 64 smoke, 128 default, 256 fullres)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained ResNet/timm weights for image backbones (default: random init)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run torchregress + suite validation checks before benchmarks",
    )
    parser.add_argument(
        "--standardize-targets",
        dest="standardize_targets",
        action="store_true",
        default=True,
        help="Standardize targets with StandardScaler (default: True)",
    )
    parser.add_argument(
        "--no-standardize-targets",
        dest="standardize_targets",
        action="store_false",
        help="Skip target standardization",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest/harness.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()
    resolved_device = set_default_device(args.device)
    print(f"Device: {resolved_device}")

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

        print()
        print("=== NF architecture smoketest ===")
        nf_ok, nf_msgs = validate_nf_architecture()
        for m in nf_msgs:
            print(m)
        print()

        print("=== SLS coverage smoketest ===")
        sls_ok, sls_msgs = validate_sls_coverage()
        for m in sls_msgs:
            print(m)
        print()

        print()
        print("=== SLS architecture smoketest ===")
        sls_arch_ok, sls_arch_msgs = validate_sls_architecture()
        for m in sls_arch_msgs:
            print(m)
        print()

        print()
        print("=== GaussianNLL coverage smoketest ===")
        gnll_ok, gnll_msgs = validate_gaussiannll_coverage()
        for m in gnll_msgs:
            print(m)
        print()

        print("=== MultivariateGaussian coverage smoketest ===")
        mvg_ok, mvg_msgs = validate_multivariate_gaussian_coverage()
        for m in mvg_msgs:
            print(m)
        print()

        print("=== Image backbone smoketest ===")
        img_ok, img_msgs = validate_image_backbones()
        for m in img_msgs:
            print(m)
        print()

        if (
            not tr_ok
            or not suite_ok
            or not nf_ok
            or not sls_ok
            or not sls_arch_ok
            or not gnll_ok
            or not mvg_ok
            or not img_ok
        ):
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
        standardize_targets=args.standardize_targets,
        device=str(resolved_device),
    )

    # Run each suite on its supported datasets
    dataset_names = args.datasets.split(",") if args.datasets else None
    all_rows: list[dict[str, object]] = []

    skipped_unsupported: list[tuple[str, str]] = []

    for suite in suites:
        datasets = dataset_names or suite.supported_datasets
        # A suite only runs on the datasets it declares: an explicit
        # --datasets list is intersected per suite, so e.g.
        # multivariate_intervals is never handed a univariate dataset.
        datasets, unsupported = split_supported(suite, list(datasets))
        for dataset_name in unsupported:
            skipped_unsupported.append((suite.name, dataset_name))
            print(
                f"\n=== Suite: {suite.name} · Dataset: {dataset_name} === skipped (not supported)"
            )
        for dataset_name in datasets:
            print(f"\n=== Suite: {suite.name} · Dataset: {dataset_name} ===")
            try:
                try:
                    load_kw: dict[str, object] = {
                        "standardize_targets": config.standardize_targets,
                    }
                    if args.smoke:
                        load_kw["n_samples"] = 200
                    if args.img_size is not None and "image" in dataset_name:
                        load_kw["img_size"] = args.img_size
                    split = load_dataset(dataset_name, **load_kw)  # type: ignore[arg-type]
                except KeyError:
                    # Suite-generated designs (orthogonal_inference,
                    # constrained_outputs) are not registry datasets: the
                    # suite draws them itself from the seed.
                    split = None
                except Exception as exc:
                    # Real datasets (NYU/IMDB-WIKI downloads, corrupt caches)
                    # must not kill the whole run — report and continue.
                    print(f"  Dataset {dataset_name!r} unavailable — skipping ({exc})")
                    continue

                rows = suite.run(dataset_name, config, splits=split)
                all_rows.extend(rows)

                for row in rows:
                    note = str(row.get("Notes", "")).lower()
                    # Colon-suffixed markers only: bare-substring matching
                    # false-positives on innocent notes such as
                    # "error-aware feature encoder". All failure notes use
                    # the "failed: …" / "skipped: …" / "error: …" convention.
                    # "known-bad: …" rows stay visibly flagged (documented
                    # limitation, not a transient failure).
                    has_result = not (
                        "error:" in note
                        or "failed:" in note
                        or "skipped:" in note
                        or note.startswith("known-bad:")
                    )
                    status = "✓" if has_result else "✗"
                    suffix = f"  ({row['Notes'][:80]})" if not has_result else ""
                    print(f"  {status} {row['Method']}{suffix}")

            except NotImplementedError:
                print(f"  Suite {suite.name!r} not yet implemented — skipping")
                continue

    if skipped_unsupported:
        print(
            f"\nSkipped {len(skipped_unsupported)} unsupported suite x dataset pair(s): "
            + ", ".join(f"{s}/{d}" for s, d in skipped_unsupported)
        )

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
        notes=[f"skipped (not supported): {s}/{d}" for s, d in skipped_unsupported],
    )
    print(f"\nWrote harness artifact: {out_path}")


if __name__ == "__main__":
    main()
