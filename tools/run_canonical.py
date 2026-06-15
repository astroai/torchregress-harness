#!/usr/bin/env python
"""Run a named canonical scenario across all suites that support it.

Usage::

    # List available scenarios
    uv run python tools/run_canonical.py --list

    # Run a specific scenario across all matching suites
    uv run python tools/run_canonical.py --scenario mapie_reference

    # Run with CI smoke settings
    uv run python tools/run_canonical.py --scenario mapie_reference --smoke

    # Output with specific paths
    uv run python tools/run_canonical.py --scenario mapie_reference \
        --output results/latest/canonical_mapie.json \
        --report results/latest/canonical_mapie.md

The tool discovers every suite whose ``canonical_scenarios`` property
includes the requested scenario name, runs each one, collects the
result rows, and renders a unified comparison report with source URLs
linking back to upstream library documentation.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Ensure harness and suites are importable from repo root
_repo_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _repo_root)
sys.path.insert(0, str(Path(_repo_root) / "src"))

from harness import (
    CANONICAL_SCENARIOS,
    CanonicalScenario,
    HarnessConfig,
    print_comparison_summary,
    print_fairness_notes,
    write_harness_json,
)
from harness.runners import HarnessRunner
from tools._suite_registry import ALL_SUITES
from tools._validate import validate_suites, validate_torchregress

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_compatible_suites(
    scenario_name: str,
) -> list[HarnessRunner]:
    """Return instantiated suite objects whose ``canonical_scenarios``
    include ``scenario_name``."""
    matches: list[HarnessRunner] = []
    for name, cls in ALL_SUITES.items():
        suite = cls()
        if scenario_name in suite.canonical_scenarios:
            matches.append(suite)
    return matches


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_scenario_header(scenario: CanonicalScenario) -> str:
    """Render a header block describing the canonical scenario."""
    lines = [
        f"# Canonical Scenario: `{scenario.scenario_name}`",
        "",
        f"- **Source library:** {scenario.source_library}",
        f"- **Canonical dataset:** `{scenario.dataset_key}`",
        f"- **Documentation:** {scenario.source_url}",
        f"- **Description:** {scenario.description}",
    ]
    if scenario.alpha is not None:
        lines.append(f"- **Alpha (conformal):** {scenario.alpha}")
    if scenario.calibration_fraction is not None:
        lines.append(f"- **Calibration fraction:** {scenario.calibration_fraction}")
    if scenario.loader_kwargs:
        lines.append(f"- **Loader kwargs:** `{scenario.loader_kwargs}`")
    lines.append("")
    return "\n".join(lines)


def _render_scenario_report(
    scenario: CanonicalScenario,
    all_rows: list[dict],
    suite_names: list[str],
) -> str:
    """Render a full Markdown report for a canonical scenario run."""
    lines: list[str] = [_render_scenario_header(scenario)]

    if not all_rows:
        lines.append("_(no result rows produced)_")
        return "\n".join(lines)

    # Group rows by suite
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        suite = str(row.get("Suite", "?"))
        grouped[suite].append(row)

    for suite in suite_names:
        if suite not in grouped:
            continue
        suite_rows = grouped[suite]
        lines.append(f"## Suite: `{suite}` ({len(suite_rows)} methods)")
        lines.append("")

        # Collect all metric keys
        metric_keys: list[str] = []
        seen: set[str] = set()
        header_keys = {"Method", "Library", "Dataset", "Suite", "SourceURL", "Notes"}
        for row in suite_rows:
            for key in row:
                if key not in header_keys and key not in seen:
                    metric_keys.append(key)
                    seen.add(key)

        # Build table
        headers = ["Method", *metric_keys, "SourceURL"]
        widths = [len(h) for h in headers]
        table_rows: list[list[str]] = []

        for row in suite_rows:
            vals = [str(row.get("Method", ""))]
            for key in metric_keys:
                v = row.get(key)
                if v is None:
                    vals.append("-")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            # SourceURL — show last path segment only (full URL in header)
            url = str(row.get("SourceURL", ""))
            short_url = url.rsplit("/", 1)[-1] if "/" in url else url
            vals.append(short_url)
            table_rows.append(vals)

        # Recompute widths
        for r in table_rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(cell))

        def _fmt(cells: list[str]) -> str:
            return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

        sep = "|".join("-" * w for w in widths)
        lines.append(_fmt(headers))
        lines.append(sep)
        for r in table_rows:
            lines.append(_fmt(r))
        lines.append("")

    # Notes
    lines.append("## Reference")
    lines.append("")
    lines.append(f"- Canonical documentation: {scenario.source_url}")
    lines.append(
        "- All methods were tested on the upstream library's own dataset, split "
        "configuration, and metric convention."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a canonical scenario across all compatible suites.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Canonical scenario name (e.g., mapie_reference)",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List all available canonical scenarios and exit",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run torchregress + suite validation checks before benchmarks",
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
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: results/latest/canonical_<scenario>.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output Markdown report path (default: derived from --output)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the Markdown report",
    )
    args = parser.parse_args()

    # --list: instant, no deps needed
    if args.list:
        print("Canonical scenarios in DATASET_REGISTRY:\n")
        for name in sorted(CANONICAL_SCENARIOS):
            sc = CANONICAL_SCENARIOS[name]
            print(f"  {name}")
            print(f"    desc:     {sc.description}")
            print(f"    library:  {sc.source_library}")
            print(f"    dataset:  {sc.dataset_key}")
            print(f"    url:      {sc.source_url}")
            if sc.alpha is not None:
                print(f"    alpha:    {sc.alpha}")
            if sc.calibration_fraction is not None:
                print(f"    cal_frac: {sc.calibration_fraction}")
            # Show compatible suites
            compat = _discover_compatible_suites(name)
            suite_names = [s.name for s in compat]
            print(f"    suites:   {', '.join(suite_names) if suite_names else '(none)'}")
            print()
        return

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

    scenario_name: str = args.scenario  # type: ignore[assignment]

    # Validate scenario
    if scenario_name not in CANONICAL_SCENARIOS:
        print(f"Unknown scenario: {scenario_name!r}")
        print(f"Available: {sorted(CANONICAL_SCENARIOS.keys())}")
        print("Use --list to see all scenarios with metadata.")
        sys.exit(1)

    scenario = CANONICAL_SCENARIOS[scenario_name]

    # Discover compatible suites
    suites = _discover_compatible_suites(scenario_name)
    if not suites:
        print(f"No suites support scenario {scenario_name!r}.")
        print("Available scenarios per suite:")
        for name, cls in ALL_SUITES.items():
            s = cls()
            if s.canonical_scenarios:
                print(f"  {name}: {s.canonical_scenarios}")
        sys.exit(1)

    print(f"Scenario: {scenario_name}")
    print(f"  Source:    {scenario.source_library}")
    print(f"  Dataset:   {scenario.dataset_key}")
    print(f"  URL:       {scenario.source_url}")
    print(f"  Suites:    {', '.join(s.name for s in suites)}")
    print()

    config = HarnessConfig(
        seed=args.seed,
        epochs=2 if args.smoke else args.epochs,
        smoke=args.smoke,
    )

    # Run each matching suite
    all_rows: list[dict] = []
    suite_names: list[str] = []

    for suite in suites:
        suite_names.append(suite.name)
        print(f"=== Suite: {suite.name} ({scenario_name}) ===")
        try:
            rows = suite.run(scenario_name=scenario_name, config=config)
            all_rows.extend(rows)

            for row in rows:
                status = "✓" if any(
                    row.get(k) is not None
                    for k in ("RMSE", "MAE", "Coverage", "Width")
                ) else "✗"
                print(f"  {status} {row['Method']}")
        except NotImplementedError:
            print(f"  Suite {suite.name!r} not yet implemented — skipping")
            continue
        except Exception as exc:
            print(f"  Suite {suite.name!r} failed: {type(exc).__name__}: {exc}")
            continue

    if not all_rows:
        print("\nNo rows produced — exiting.")
        sys.exit(0)

    print()

    # Print fairness notes
    print_fairness_notes(
        title=f"Canonical scenario: {scenario_name}",
        seed_policy=f"fixed seed ({config.seed})",
        train_budget=f"{config.epochs} epochs, batch={config.batch_size}, lr={config.lr}",
        metric_policy="RMSE, MAE, R2, NLL, CRPS, Coverage, Width, IntervalScore, ECE, runtime",
    )

    # Print comparison summary
    print_comparison_summary(
        f"Canonical Scenario: {scenario_name}",
        all_rows,
    )

    # Determine output paths
    out_json = args.output or Path(f"results/latest/canonical_{scenario_name}.json")
    out_report = args.report or out_json.with_suffix(".md")

    # Write JSON artifact
    out_path = write_harness_json(
        out_json,
        config=config,
        rows=all_rows,
        suites=suite_names,
        datasets=[scenario.dataset_key],
    )
    print(f"\nWrote harness artifact: {out_path}")

    # Write Markdown report
    if not args.no_report:
        md = _render_scenario_report(scenario, all_rows, suite_names)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(md, encoding="utf-8")
        print(f"Wrote scenario report:  {out_report}")

    # Print summary
    ok = sum(
        1 for r in all_rows
        if any(r.get(k) is not None for k in ("RMSE", "MAE", "Coverage"))
    )
    skipped = sum(1 for r in all_rows if r.get("Notes", "").startswith("skipped"))
    failed = sum(1 for r in all_rows if r.get("Notes", "").startswith("failed"))
    print(
        f"\nDone. {ok} ok, {skipped} skipped, {failed} failed "
        f"across {len(suite_names)} suites."
    )


if __name__ == "__main__":
    main()
