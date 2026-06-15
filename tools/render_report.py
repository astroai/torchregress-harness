#!/usr/bin/env python
"""Render harness JSON artifacts as Markdown reports.

Usage::

    uv run python tools/render_report.py results/latest/harness_smoke.json
    uv run python tools/render_report.py results/latest/harness_smoke.json --output results/latest/harness_smoke.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.report import render_report_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Render harness JSON as Markdown report.")
    parser.add_argument("run_path", type=Path, help="Path to harness JSON artifact")
    parser.add_argument("--output", type=Path, default=None, help="Output Markdown path")
    args = parser.parse_args()

    out = render_report_file(args.run_path, args.output)
    print(f"Wrote report: {out}")


if __name__ == "__main__":
    main()
