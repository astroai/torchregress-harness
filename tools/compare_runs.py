#!/usr/bin/env python
"""Compare two harness run artifacts to highlight changes.

Usage::

    uv run python tools/compare_runs.py results/archive/run_v1.json results/latest/harness.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.report import load_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two harness run artifacts.")
    parser.add_argument("baseline", type=Path, help="Baseline harness JSON")
    parser.add_argument("current", type=Path, help="Current harness JSON")
    parser.add_argument("--metric", type=str, default="RMSE", help="Metric to compare")
    parser.add_argument("--diff-threshold", type=float, default=0.05, help="Relative diff threshold to flag")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit non-zero when changed rows are found")
    args = parser.parse_args()

    baseline = load_run(args.baseline)
    current = load_run(args.current)

    baseline_rows = {_row_key(r): r for r in baseline.get("rows", [])}
    current_rows = {_row_key(r): r for r in current.get("rows", [])}

    metric = args.metric
    threshold = args.diff_threshold

    print(f"Comparing {len(baseline_rows)} vs {len(current_rows)} rows on metric '{metric}'")
    print()

    # New rows (methods added)
    new_keys = set(current_rows) - set(baseline_rows)
    if new_keys:
        print(f"** New methods ({len(new_keys)}):")
        for key in sorted(new_keys):
            row = current_rows[key]
            print(f"  + {key}: {metric}={_fmt(row.get(metric))}")
        print()

    # Removed rows
    removed_keys = set(baseline_rows) - set(current_rows)
    if removed_keys:
        print(f"** Removed methods ({len(removed_keys)}):")
        for key in sorted(removed_keys):
            row = baseline_rows[key]
            print(f"  - {key}: {metric}={_fmt(row.get(metric))}")
        print()

    # Changed rows
    common_keys = set(baseline_rows) & set(current_rows)
    changed = []
    for key in sorted(common_keys):
        b_val = baseline_rows[key].get(metric)
        c_val = current_rows[key].get(metric)
        if b_val is not None and c_val is not None and b_val != 0:
            rel_diff = abs(c_val - b_val) / abs(b_val)
            if rel_diff > threshold:
                changed.append((key, b_val, c_val, rel_diff))

    if changed:
        print(f"** Changed (>{threshold:.0%} relative diff, {len(changed)} rows):")
        for key, b_val, c_val, rel_diff in changed:
            direction = "↑" if c_val > b_val else "↓"
            print(f"  {key}: {_fmt(b_val)} → {_fmt(c_val)} ({direction} {rel_diff:.1%})")

    regressions = bool(removed_keys or changed)
    if regressions and args.fail_on_regression:
        print()
        print("Regression detected — exiting with code 1")
        sys.exit(1)


def _row_key(row: dict) -> str:
    return f"{row.get('Method', '?')}|{row.get('Dataset', '?')}|{row.get('Suite', '?')}"


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
