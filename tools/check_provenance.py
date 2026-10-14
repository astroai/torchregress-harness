#!/usr/bin/env python
"""Release gate: refuse result artifacts whose real-data rows silently used synthetic data.

Every harness row carries provenance fields (``SyntheticFallback``,
``SourceURL``, ``Device``).  A row may be synthetic only when its dataset is
synthetic *by design* (``DatasetInfo.category == "synthetic"``).  A real
dataset that fell back to a synthetic proxy (download failure, missing staged
mirror, 404'd upstream) must be restored or dropped before a report is
published, never reported as if it were the real dataset.

Usage::

    python tools/check_provenance.py leaderboard_results.json results/latest/*.json

Exit status is 1 when any violation is found, 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.datasets import DATASET_REGISTRY

PROVENANCE_FIELDS: tuple[str, ...] = ("SyntheticFallback",)


@dataclass(frozen=True)
class Violation:
    """One row that must not reach a published report."""

    source: str
    method: str
    dataset: str
    reason: str

    def render(self) -> str:
        return f"{self.source}: {self.method} on {self.dataset}: {self.reason}"


def _extract_rows(payload: Any) -> list[Mapping[str, Any]] | None:
    """Return result rows from a harness JSON artifact, or ``None`` if it has none."""
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        return None
    return [r for r in rows if isinstance(r, Mapping) and "Method" in r]


def _is_error_row(row: Mapping[str, Any]) -> bool:
    notes = str(row.get("Notes", ""))
    return notes.startswith(("SKIPPED", "ERROR"))


def check_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str,
    synthetic_datasets: frozenset[str],
    known_datasets: frozenset[str],
) -> list[Violation]:
    """Return every provenance violation in ``rows``.

    Error and skip rows are ignored: they carry no metrics to misreport.
    """
    found: list[Violation] = []
    for row in rows:
        if _is_error_row(row):
            continue
        method = str(row.get("Method", "?"))
        dataset = str(row.get("Dataset", "?"))
        missing = [f for f in PROVENANCE_FIELDS if f not in row]
        if missing:
            found.append(Violation(source, method, dataset, f"missing {', '.join(missing)}"))
            continue
        if not row["SyntheticFallback"]:
            continue
        if dataset in synthetic_datasets:
            continue
        if dataset in known_datasets:
            reason = "real dataset fell back to synthetic data"
        else:
            reason = "synthetic data on a dataset that is not registered as synthetic"
        found.append(Violation(source, method, dataset, reason))
    return found


def registry_sets() -> tuple[frozenset[str], frozenset[str]]:
    """``(synthetic-by-design dataset names, all registered dataset names)``."""
    synthetic = frozenset(n for n, info in DATASET_REGISTRY.items() if info.category == "synthetic")
    return synthetic, frozenset(DATASET_REGISTRY)


def check_files(paths: Iterable[Path]) -> tuple[list[Violation], int]:
    """Check every artifact; return violations and the number of rows inspected."""
    synthetic, known = registry_sets()
    violations: list[Violation] = []
    n_rows = 0
    for path in paths:
        rows = _extract_rows(json.loads(path.read_text(encoding="utf-8")))
        if rows is None:
            continue
        n_rows += len(rows)
        violations += check_rows(
            rows, source=str(path), synthetic_datasets=synthetic, known_datasets=known
        )
    return violations, n_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifacts", nargs="+", type=Path, help="Harness JSON artifacts.")
    args = parser.parse_args(argv)

    violations, n_rows = check_files(args.artifacts)
    for v in violations:
        print(v.render(), file=sys.stderr)
    if violations:
        print(f"FAIL: {len(violations)} of {n_rows} rows violate provenance.", file=sys.stderr)
        return 1
    print(f"OK: {n_rows} rows, no synthetic fallback on real datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
