#!/usr/bin/env python
"""Find flaky harness rows: metrics whose spread across seeds is too large to rank on.

Give the same run repeated with different seeds (``run_all.py --seed``)::

    python tools/seed_spread.py results/seed0.json results/seed1.json results/seed2.json \\
        --metrics RMSE IntervalScore NLL --rel-tol 0.05 --json-out results/latest/seed_spread.json

For every (Suite, Dataset, Method) row present in at least two runs, the tool
reports the mean, standard deviation and relative spread (std / |mean|) of each
metric.  A row is *flaky* when any metric's relative spread exceeds
``--rel-tol``.  Flaky rows need more seeds or must be reported with error bars,
never as a single-seed win.  Exit status is 1 with ``--fail-on-flaky`` and at
least one flaky row.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_METRICS: tuple[str, ...] = ("RMSE", "IntervalScore", "NLL", "CRPS", "Coverage")

RowKey = tuple[str, str, str]


@dataclass
class MetricSpread:
    metric: str
    n: int
    mean: float
    std: float
    rel_spread: float


@dataclass
class RowSpread:
    suite: str
    dataset: str
    method: str
    n_runs: int
    flaky: bool
    metrics: list[MetricSpread] = field(default_factory=list)


def _rows(payload: Any) -> list[Mapping[str, Any]]:
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else payload
    return [r for r in rows if isinstance(r, Mapping) and "Method" in r]


def _key(row: Mapping[str, Any]) -> RowKey:
    method = str(row["Method"])
    suite = str(row.get("Suite") or method.split("/", 1)[0])
    return suite, str(row.get("Dataset", "?")), method


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def collect(runs: Iterable[Sequence[Mapping[str, Any]]]) -> dict[RowKey, list[Mapping[str, Any]]]:
    """Group rows from several runs by (suite, dataset, method)."""
    grouped: dict[RowKey, list[Mapping[str, Any]]] = defaultdict(list)
    for rows in runs:
        for row in rows:
            grouped[_key(row)].append(row)
    return grouped


def spread(
    grouped: Mapping[RowKey, Sequence[Mapping[str, Any]]],
    metrics: Sequence[str],
    rel_tol: float,
    abs_floor: float = 1e-12,
) -> list[RowSpread]:
    """Per-row spread of each metric; rows seen in fewer than two runs are skipped."""
    out: list[RowSpread] = []
    for (suite, dataset, method), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        result = RowSpread(suite, dataset, method, n_runs=len(rows), flaky=False)
        for metric in metrics:
            values = [v for v in (_finite(r.get(metric)) for r in rows) if v is not None]
            if len(values) < 2:
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values)
            rel = std / max(abs(mean), abs_floor)
            result.metrics.append(MetricSpread(metric, len(values), mean, std, rel))
            if rel > rel_tol:
                result.flaky = True
        if result.metrics:
            out.append(result)
    return out


def render_markdown(results: Sequence[RowSpread], rel_tol: float) -> str:
    flaky = [r for r in results if r.flaky]
    lines = [
        f"# Seed spread ({len(flaky)} flaky of {len(results)} rows, rel-tol {rel_tol:g})",
        "",
        "| Suite | Dataset | Method | Runs | Metric | Mean | Std | Rel. spread |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for r in flaky:
        for m in r.metrics:
            if m.rel_spread > rel_tol:
                lines.append(
                    f"| {r.suite} | {r.dataset} | {r.method} | {r.n_runs} | {m.metric} "
                    f"| {m.mean:.4g} | {m.std:.3g} | {m.rel_spread:.1%} |"
                )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="Run artifacts (one per seed).")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--rel-tol", type=float, default=0.05)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--fail-on-flaky", action="store_true")
    args = parser.parse_args(argv)
    if len(args.runs) < 2:
        parser.error("need at least two run artifacts")

    grouped = collect(_rows(json.loads(p.read_text(encoding="utf-8"))) for p in args.runs)
    results = spread(grouped, args.metrics, args.rel_tol)
    md = render_markdown(results, args.rel_tol)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "runs": [str(p) for p in args.runs],
                    "rel_tol": args.rel_tol,
                    "rows": [asdict(r) for r in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.markdown_out:
        args.markdown_out.write_text(md, encoding="utf-8")
    print(md)
    n_flaky = sum(r.flaky for r in results)
    return 1 if args.fail_on_flaky and n_flaky else 0


if __name__ == "__main__":
    sys.exit(main())
