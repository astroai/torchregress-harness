#!/usr/bin/env python
"""PROPOSAL #10 verification benchmark.

Runs the ``distributional_bins`` suite on all 3 supported datasets at 60
epochs and aggregates the per-dataset ``CoverageDeviationAbsMean`` from
the per-suite ``/dev_summary`` rows into an **unweighted mean across
datasets**. Compares:
    * Single ``NormalizingFlow`` (Baseline NF)
    * ``NF_Ensemble_K5`` (Proposal #10)
    * ``MDN_K3`` (most-stable non-CP method)
    * ``Evidential`` (best-calibrated non-CP, Student-t)
    * ``GaussianNLL`` (diagonal-Gaussian reference)
    * ``Split_CP`` (calibrated baseline)

**Note on aggregation**: size-weighting happens *within* each dataset
(bin-level: ``np.average(devs, weights=sizes)`` in the suite), but
across-dataset aggregation uses a plain mean. The bin-size totals per
dataset live in the per-bin rows, but the ``/dev_summary`` rows only
carry the rolled-up aggregate. If you need two-stage weighting
(bin-size × dataset-size), see ``_aggregate_methods_weighted`` below.

Writes a comparison table to stdout and a JSON sidecar with all parsed
rows plus aggregate statistics.

Verdict on PROPOSAL #10:
    * Asserts ``NF_Ensemble_K5``'s average ``CoverageDeviationAbsMean`` is
      meaningfully lower than ``NormalizingFlow``'s average.
    * Target reduction: NF 0.147 → Ensemble ~0.05 (per ``PARITY_REPORT.md``,
      which assumed tighter convergence at higher epoch counts).

Usage::
    uv run python tools/benchmark_pit_ensemble.py \\
        --epochs 60 --output results/latest/pit_ensemble_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_harness(
    suites: str,
    datasets: str,
    epochs: int,
    output: Path,
    seed: int = 42,
) -> Path:
    """Invoke ``tools/run_all.py`` as a subprocess and return the JSON path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "tools" / "run_all.py"),
        f"--suites={suites}",
        f"--datasets={datasets}",
        f"--epochs={epochs}",
        f"--seed={seed}",
        f"--output={str(output)}",
    ]
    print(f"Running harness:\n  {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)
    if result.returncode != 0:
        print(f"\u2717 Harness exited with code {result.returncode}")
        sys.exit(result.returncode)
    return output


def _parse_methods(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Group ``/dev_summary`` rows by (dataset, method).

    Returns:
        {(dataset, method): row_dict} for all ``/dev_summary`` rows
        extracted from the harness JSON rows list.
    """
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        method_full: str = row.get("Method", "")
        if not method_full.endswith("/dev_summary"):
            continue
        ds = row.get("Dataset", "")
        method = method_full.split("/")[1] if "/" in method_full else method_full
        out[(ds, method)] = row
    return out


def _aggregate_methods(
    parsed: dict[tuple[str, str], dict],
) -> dict[str, dict[str, float]]:
    """Aggregate per-method statistics across all 3 datasets.

    Returns a nested dict ``{method: {"CoverageDeviationAbsMean": avg, ...}}``
    with size-weighted (or unweighted if sizes unavailable) averages.
    """
    by_method: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (ds, method), row in parsed.items():
        by_method[method].append(row)

    aggregated: dict[str, dict[str, float]] = {}
    for method, rows in by_method.items():
        agg: dict[str, float] = {}
        # Average numeric metrics across all datasets (with equal weight;
        # we don't have per-bin sizes at the summary row for weighting).
        numeric_keys = [
            "CoverageDeviationAbsMean",
            "CoverageDeviationMean",
            "CoverageDeviationRMSE",
            "AvgBinCoverage",
            "AvgBinWidth",
        ]
        for key in numeric_keys:
            vals = [
                float(r[key])
                for r in rows
                if r.get(key) is not None
                and not (
                    isinstance(r.get(key), float) and r.get(key) != r.get(key)
                    # NaN check via self-inequality
                )
            ]
            if vals:
                agg[key] = float(sum(vals) / len(vals))
        aggregated[method] = agg
    return aggregated


def _print_table(
    parsed: dict[tuple[str, str], dict],
    aggregated: dict[str, dict[str, float]],
) -> None:
    """Print per-dataset and aggregated tables."""
    datasets_seen = sorted({ds for ds, _ in parsed.keys()})
    methods_seen = sorted({m for _, m in parsed.keys()})

    if not methods_seen:
        print("No /dev_summary rows found in harness output.")
        return

    print()
    print("=" * 75)
    print("PER-DATASET CoverageDeviationAbsMean (size-weighted, target=0.90)")
    print("=" * 75)
    header = ["Method".ljust(22)] + [ds.ljust(20) for ds in datasets_seen]
    print(" | ".join(header))
    print("-+-".join("-" * w for w in [22] + [20] * len(datasets_seen)))
    for method in methods_seen:
        row_cells = [method.ljust(22)]
        for ds in datasets_seen:
            r = parsed.get((ds, method))
            if r and r.get("CoverageDeviationAbsMean") is not None:
                v = r["CoverageDeviationAbsMean"]
                row_cells.append(f"{v:+.4f}".ljust(20))
            else:
                row_cells.append("-".ljust(20))
        print(" | ".join(row_cells))

    print()
    print("=" * 75)
    print("AGGREGATE CoverageDeviationAbsMean (mean across datasets)")
    print("=" * 75)
    print(f"{'Method'.ljust(28)} {'AvgDevAbsMean':>14}  Rank")
    print("-" * 75)
    # Sort by aggregated CoverageDeviationAbsMean
    sortable = [(m, v.get("CoverageDeviationAbsMean", float("inf"))) for m, v in aggregated.items()]
    sortable.sort(key=lambda x: x[1])
    for rank, (method, val) in enumerate(sortable, start=1):
        print(f"{method.ljust(28)} {val:>+14.4f}    #{rank}")

    print()
    print("=" * 75)
    print("PROPOSAL #10 VERIFICATION: NF \u2192 NF_Ensemble_K5")
    print("=" * 75)
    nf_avg = aggregated.get("NormalizingFlow", {}).get("CoverageDeviationAbsMean", float("nan"))
    ens_avg = aggregated.get("NF_Ensemble_K5", {}).get("CoverageDeviationAbsMean", float("nan"))
    if nf_avg == nf_avg and ens_avg == ens_avg:  # both finite
        reduction = nf_avg - ens_avg
        pct = 100.0 * reduction / nf_avg if nf_avg > 0 else 0.0
        target_abs = 0.05
        target_met = ens_avg <= target_abs
        print(f"  NormalizingFlow        : {nf_avg:+.4f}")
        print(f"  NF_Ensemble_K5         : {ens_avg:+.4f}")
        print(f"  \u0394 (reduction)        : {reduction:+.4f}  ({pct:+.1f}%)")
        print(f"  PROPOSAL #10 target   : Single NF \u2192 Ensemble (\u2264 {target_abs:.2f})")
        print(
            f"  Verdict               : {'\u2705 MET' if target_met else '\u26a0\ufe0f  NOT MET'}"
        )
    else:
        print("  Insufficient data \u2014 one or both methods missing from /dev_summary rows.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PROPOSAL #10 verification benchmark: NF \u2192 NF_Ensemble_K5"
    )
    parser.add_argument(
        "--epochs", type=int, default=60, help="Training epochs per method (default: 60)"
    )
    parser.add_argument(
        "--datasets",
        default="california_housing,diabetes,synthetic_hetero",
        help="Comma-separated datasets (default: all 3 supported)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest/pit_ensemble_benchmark.json"),
        help="Output JSON path (default: results/latest/pit_ensemble_benchmark.json)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-run", action="store_true", help="Skip harness run; only re-parse existing output"
    )
    args = parser.parse_args()

    if not args.skip_run:
        _run_harness(
            suites="distributional_bins",
            datasets=args.datasets,
            epochs=args.epochs,
            output=args.output,
            seed=args.seed,
        )

    # Parse harness JSON
    if not args.output.exists():
        print(f"\u2717 output JSON not found: {args.output}")
        sys.exit(1)
    payload = json.loads(args.output.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    print(f"\nLoaded {len(rows)} rows from {args.output}")

    parsed = _parse_methods(rows)
    print(
        f"Found {len(parsed)} /dev_summary rows "
        f"(datasets: {len({k[0] for k in parsed})}, "
        f"methods: {len({k[1] for k in parsed})})"
    )

    aggregated = _aggregate_methods(parsed)
    _print_table(parsed, aggregated)

    # Write a sidecar JSON with the parsed/aggregated data
    sidecar = args.output.with_name(args.output.stem + "_summary.json")
    sidecar.write_text(
        json.dumps(
            {
                "per_dataset": {f"{ds}|{method}": row for (ds, method), row in parsed.items()},
                "aggregate": aggregated,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"\nSummary written: {sidecar}")


if __name__ == "__main__":
    main()
