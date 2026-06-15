"""Report rendering: harness JSON artifacts → readable Markdown tables."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_run(path: str | Path) -> dict[str, Any]:
    """Load a harness run JSON artifact."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def _format_val(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(run: dict[str, Any]) -> str:
    """Render a harness run artifact as a Markdown summary.

    Produces a header block, a per-suite-per-dataset table, and an
    aggregate summary.
    """
    artifact = run.get("artifact", "harness_run")
    env = run.get("environment", {})
    config = run.get("config", {})
    rows: list[dict[str, Any]] = run.get("rows", [])
    aggregate = run.get("aggregate", {})

    lines: list[str] = [
        f"# Harness Run Report ({artifact})",
        "",
        f"- **Timestamp:** `{run.get('timestamp', '-')}`",
        f"- **Python:** `{env.get('python', '-')}` · **PyTorch:** `{env.get('torch', '-')}`",
        f"- **Suites:** `{', '.join(run.get('suites', [])) or '-'}`",
        f"- **Datasets:** `{', '.join(run.get('datasets', [])) or '-'}`",
        f"- **Cases:** {aggregate.get('n_total', '?')} total, "
        f"{aggregate.get('n_ok', '?')} ok, "
        f"{aggregate.get('n_skipped', '?')} skipped, "
        f"{aggregate.get('n_error', '?')} error",
        "",
    ]

    # Config block
    if isinstance(config, dict) and config:
        lines.append("## Configuration")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(config, indent=2))
        lines.append("```")
        lines.append("")

    if not rows:
        lines.append("_(no result rows)_")
        return "\n".join(lines)

    # Group by suite, then dataset
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        suite = str(row.get("Suite", "?"))
        dataset = str(row.get("Dataset", "?"))
        grouped[suite][dataset].append(row)

    for suite in sorted(grouped):
        lines.append(f"## Suite: `{suite}`")
        lines.append("")
        for dataset in sorted(grouped[suite]):
            dataset_rows = grouped[suite][dataset]
            lines.append(f"### Dataset: `{dataset}` ({len(dataset_rows)} methods)")
            lines.append("")
            table_lines = _render_table(dataset_rows)
            lines.extend(table_lines)
            lines.append("")

    # Notes
    notes = run.get("notes", [])
    if notes:
        lines.append("## Run Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def _render_table(rows: list[dict[str, Any]]) -> list[str]:
    """Render a single comparison table for a set of rows."""
    # Collect all metric keys across rows
    metric_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in ("Method", "Library", "Dataset", "Suite", "Notes") and key not in seen:
                metric_keys.append(key)
                seen.add(key)

    headers = ["Method", "Library", *metric_keys]
    include_notes = any(row.get("Notes") for row in rows)
    if include_notes:
        headers.append("Notes")

    table_rows: list[list[str]] = []
    for row in rows:
        vals = [
            str(row.get("Method", "")),
            str(row.get("Library", "")),
        ]
        for key in metric_keys:
            vals.append(_format_val(row.get(key)))
        if include_notes:
            vals.append(str(row.get("Notes", "")))
        table_rows.append(vals)

    # Compute column widths
    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [_fmt(headers), "|".join("-" * w for w in widths)]
    for r in table_rows:
        lines.append(_fmt(r))

    return lines


def render_report_file(
    run_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Load a harness run artifact and write a Markdown report.

    If ``output_path`` is None, writes to the same directory as the input
    with a ``.md`` extension.
    """
    run = load_run(run_path)
    md = render_markdown(run)

    out = Path(output_path) if output_path else Path(run_path).with_suffix(".md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return out
