"""Tests for :mod:`tools.leaderboard_regression`.

Covers:

* ``_is_skipped`` / ``_has_any_metric`` / ``_fmt`` — string-metric edge cases.
* ``aggregate_per_method`` — ranking correctness, skipped-row handling,
  fallback to RMSE for methods without ``IntervalScore``, mean computation,
  CoverageDeviation.
* ``render_leaderboard_markdown`` — header structure, table column order,
  per-suite breakdown, empty-input safety.
* ``print_leaderboard_stdout`` — basic smoke (calls don't crash on empty).
* End-to-end CLI smoke run on a small dataset / single suite / ``--smoke``,
  verifying JSON + Markdown artifacts contain the expected structure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Make tools/ + src/ importable so we can import `tools.leaderboard_regression`
# directly without spawning a subprocess for the unit tests.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from tools.leaderboard_regression import (  # noqa: E402  # noqa: E402
    _fmt,
    _has_any_metric,
    _is_skipped,
    aggregate_per_method,
    print_leaderboard_stdout,
    render_leaderboard_markdown,
)

# ---------------------------------------------------------------------------
# Synthetic rows
# ---------------------------------------------------------------------------


def _row(
    method: str,
    suite: str,
    library: str,
    dataset: str,
    **metrics: Any,
) -> dict[str, Any]:
    """Build a synthetic harness row mirroring the schema from ``HarnessRunner.run``."""
    return {
        "Method": f"{suite}/{method}",
        "Library": library,
        "Dataset": dataset,
        "Suite": suite,
        **{k: v for k, v in metrics.items()},
    }


# Mixed dataset coverage: some methods produce intervals, some are point-only,
# some are skipped.  Designed so that ranking is unambiguous.
FAKE_ROWS: list[dict[str, Any]] = [
    _row(
        "Split+MLP",
        "conformal",
        "torchregress",
        "diabetes",
        RMSE=0.42,
        MAE=0.30,
        IntervalScore=2.5,
        Coverage=0.91,
        Width=0.8,
        ECE=0.05,
    ),
    _row(
        "Split+MLP",
        "conformal",
        "torchregress",
        "house_sales",
        RMSE=0.51,
        MAE=0.40,
        IntervalScore=3.0,
        Coverage=0.92,
        Width=0.9,
        ECE=0.04,
    ),
    _row(
        "CQR+MLP",
        "conformal",
        "torchregress",
        "diabetes",
        RMSE=0.45,
        MAE=0.32,
        IntervalScore=2.3,
        Coverage=0.90,
        Width=0.7,
        ECE=0.06,
    ),
    _row(
        "Ridge",
        "sklearn_baselines",
        "sklearn",
        "diabetes",
        RMSE=0.50,
        MAE=0.35,
    ),  # No interval metrics — RMSE-only fallback
    _row(
        "Ridge",
        "sklearn_baselines",
        "sklearn",
        "house_sales",
        RMSE=0.55,
        MAE=0.38,
    ),
    _row(
        "BayesianRidge",
        "bayesian_linear",
        "sklearn",
        "diabetes",
        RMSE=0.48,
    ),
    # Skipped row (should be ignored by the aggregator)
    _row(
        "MissingLib",
        "conformal",
        "external",
        "diabetes",
        Notes="skipped: ImportError: missing",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_is_skipped_handles_variants() -> None:
    assert _is_skipped({"Notes": "skipped: foo"})
    assert _is_skipped({"Notes": "failed: foo"})
    assert _is_skipped({"Notes": "error: foo"})
    assert _is_skipped({"Notes": "FAILED BAR"})
    assert not _is_skipped({"Notes": ""})
    assert not _is_skipped({})
    assert not _is_skipped({"Notes": "ok"})


def test_has_any_metric_true_when_metric_present() -> None:
    assert _has_any_metric({"RMSE": 0.5, "Library": "sklearn"})
    assert not _has_any_metric({"RMSE": None, "Library": "sklearn"})
    # Row containing only structural keys → no actual metric signal.
    assert not _has_any_metric(
        {"Method": "foo", "Library": "torchregress", "Suite": "x", "Dataset": "y"},
    )


def test_fmt_handles_nan_and_none() -> None:
    assert _fmt(0.123456) == "0.1235"
    assert _fmt(None) == "-"
    assert _fmt(float("nan")) == "-"
    assert _fmt("text") == "text"
    assert _fmt(0) == "0.0000"
    assert _fmt(1) == "1.0000"


# ---------------------------------------------------------------------------
# Aggregator behaviour
# ---------------------------------------------------------------------------


def test_aggregate_per_method_groups_by_stripped_method() -> None:
    summary = aggregate_per_method(FAKE_ROWS, target_coverage=0.9)
    methods = [s["Method"] for s in summary]
    # Suite prefix must be stripped → "Split+MLP", not "conformal/Split+MLP".
    assert all("/" not in m for m in methods)
    assert "Split+MLP" in methods
    assert "Ridge" in methods
    assert "BayesianRidge" in methods
    # Skipped rows dropped.
    assert "MissingLib" not in methods


def test_aggregate_per_method_ranks_by_interval_score_then_rmse() -> None:
    summary = aggregate_per_method(FAKE_ROWS, target_coverage=0.9)

    # IS-methods (CQR+MLP, Split+MLP) — mean IS values:
    #   CQR+MLP      = 2.30
    #   Split+MLP    = 2.75
    # RMSE-only methods (Ridge, BayesianRidge):
    #   Ridge        = 0.525  (sklearn)
    #   BayesianRidge= 0.48
    primary = [(s["Method"], s["primary_metric"]) for s in summary]

    # IS-methods come first (lower IS beats higher IS).
    assert primary[0] == ("CQR+MLP", "IntervalScore")
    assert primary[1] == ("Split+MLP", "IntervalScore")
    # RMSE-only methods come after, sorted by mean RMSE.
    rmse_only = [m for m, p in primary if p == "RMSE"]
    assert rmse_only == ["BayesianRidge", "Ridge"]  # 0.48 < 0.525


def test_aggregate_per_method_means_computed_correctly() -> None:
    summary = aggregate_per_method(FAKE_ROWS, target_coverage=0.9)
    by_name = {s["Method"]: s for s in summary}

    split_mlp = by_name["Split+MLP"]
    assert split_mlp["N_Runs"] == 2
    assert split_mlp["N_Suites"] == 1
    assert split_mlp["mean_RMSE"] == pytest.approx((0.42 + 0.51) / 2)
    assert split_mlp["mean_IntervalScore"] == pytest.approx((2.5 + 3.0) / 2)
    assert split_mlp["mean_Coverage"] == pytest.approx((0.91 + 0.92) / 2)
    assert split_mlp["mean_ECE"] == pytest.approx((0.05 + 0.04) / 2)

    # Methods without IntervalScore have it as NaN, RMSE available.
    cqr = by_name["CQR+MLP"]
    assert not np.isnan(cqr["mean_IntervalScore"])
    ridge = by_name["Ridge"]
    assert np.isnan(ridge["mean_IntervalScore"])
    assert ridge["mean_RMSE"] == pytest.approx((0.50 + 0.55) / 2)


def test_aggregate_per_method_coverage_deviation_computed() -> None:
    # Target 0.9, Coverages 0.91 + 0.92 → deviation (0.01 + 0.02)/2 = 0.015.
    summary = aggregate_per_method(FAKE_ROWS, target_coverage=0.9)
    by_name = {s["Method"]: s for s in summary}
    assert by_name["Split+MLP"]["CoverageDeviation"] == pytest.approx(0.015)
    # Methods without coverage → NaN.
    assert np.isnan(by_name["Ridge"]["CoverageDeviation"])


def test_aggregate_per_method_handles_empty_input() -> None:
    assert aggregate_per_method([]) == []
    assert (
        aggregate_per_method(
            [_row("M", "S", "L", "D", Notes="skipped: x")],
        )
        == []
    )


def test_aggregate_per_method_no_intervals_falls_back_to_rmse() -> None:
    """Methods without IntervalScore excluded from primary_metric=IS ranking."""
    rows = [
        _row("RidgeFoo", "sklearn_baselines", "sklearn", "diabetes", RMSE=0.6),
        _row("BayesianRidgeFoo", "bayesian_linear", "sklearn", "diabetes", RMSE=0.5),
    ]
    summary = aggregate_per_method(rows, target_coverage=0.9)
    # Order: BayesianRidgeFoo first (RMSE 0.5), RidgeFoo second.
    assert [(s["Method"], s["primary_metric"]) for s in summary] == [
        ("BayesianRidgeFoo", "RMSE"),
        ("RidgeFoo", "RMSE"),
    ]


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def test_render_leaderboard_markdown_structure() -> None:
    summary = aggregate_per_method(FAKE_ROWS, target_coverage=0.9)
    md = render_leaderboard_markdown(summary, FAKE_ROWS, target_coverage=0.9)
    assert md.startswith("# Comprehensive Regression Leaderboard")
    assert "Target coverage: **0.90**" in md
    assert "## Leaderboard (suite-mean by method)" in md
    assert "## Per-Suite Breakdown" in md
    # Main leaderboard: column headers present
    assert "IntervalScore" in md
    assert "Coverage Dev" in md
    # Per-suite breakdown contains both suites
    assert "conformal" in md
    assert "sklearn_baselines" in md
    # Rank ordering: CQR+MLP precedes Split+MLP in the table
    assert md.index("| 1 | CQR+MLP") < md.index("| 2 | Split+MLP")
    # Method names appear without suite prefix in the per-dataset tables.
    assert "conformal/Split+MLP" not in md


def test_render_leaderboard_markdown_empty() -> None:
    md = render_leaderboard_markdown([], [], target_coverage=0.9)
    assert "Methods ranked: **0**" in md
    assert "## Leaderboard (suite-mean by method)" in md
    assert "## Per-Suite Breakdown" in md


def test_print_leaderboard_stdout_does_not_crash_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_leaderboard_stdout([], target_coverage=0.9)
    # Nothing should be printed when summary is empty.
    captured = capsys.readouterr()
    assert "Comprehensive Regression Leaderboard" not in captured.out


# ---------------------------------------------------------------------------
# End-to-end CLI smoke run
# ---------------------------------------------------------------------------


def test_cli_smoke_run_produces_artifact(tmp_path: Path) -> None:
    """Launch ``tools/leaderboard_regression.py --smoke`` on diabetes+conformal
    and verify JSON + Markdown are produced and well-formed.
    """
    output = tmp_path / "lb.json"
    markdown = tmp_path / "lb.md"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "tools/leaderboard_regression.py",
            "--datasets",
            "diabetes",
            "--suites",
            "conformal",
            "--smoke",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, (
        f"smoke run failed:\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
    )
    assert output.exists()
    artifact = json.loads(output.read_text())
    # Top-level keys
    assert "metadata" in artifact and "rows" in artifact and "leaderboard" in artifact
    assert isinstance(artifact["leaderboard"], list)
    assert isinstance(artifact["rows"], list)
    # Metadata echoes CLI args
    meta = artifact["metadata"]
    assert meta["datasets"] == ["diabetes"]
    assert meta["suites"] == ["conformal"]
    assert meta["smoke"] is True
    assert meta["epochs"] == 10  # smoke mode forces 10
    # Leaderboard rows are sorted by primary metric ascending
    primary_values = [s["primary_value"] for s in artifact["leaderboard"]]
    assert primary_values == sorted(primary_values)
    # Markdown file exists and has the expected header
    assert markdown.exists()
    md_text = markdown.read_text()
    assert md_text.startswith("# Comprehensive Regression Leaderboard")
