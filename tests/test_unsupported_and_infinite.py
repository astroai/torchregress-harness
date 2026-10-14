"""Drivers skip unsupported suite x dataset pairs; metrics/renderers survive infinite intervals."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tools import leaderboard_regression as lb
from tools import run_all
from tools._suite_registry import ALL_SUITES, split_supported, supports_dataset

from harness.metrics import (
    compute_all_metrics,
    compute_distribution_metrics,
    compute_interval_metrics,
    compute_pit_values,
    infinite_interval_note,
)
from harness.report import render_markdown
from harness.runners import HarnessConfig, HarnessRunner


class _FakeSuite(HarnessRunner):
    """Records which datasets it was asked to run."""

    calls: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def description(self) -> str:
        return "fake suite"

    @property
    def supported_datasets(self) -> list[str]:
        return ["fake_design"]

    def methods(self) -> list[str]:
        return ["M"]

    def fit(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        raise NotImplementedError

    def predict(self, method: str, model: Any, X_test: np.ndarray, config: HarnessConfig) -> Any:
        raise NotImplementedError

    def run(  # type: ignore[override]
        self, dataset_name: str = "", config: Any = None, splits: Any = None, **_: Any
    ) -> list[dict[str, Any]]:
        type(self).calls.append(dataset_name)
        return [
            {
                "Method": "fake/M",
                "Library": "torchregress",
                "Dataset": dataset_name,
                "Suite": "fake",
                "RMSE": 1.0,
                "Notes": "",
            }
        ]


# ---------------------------------------------------------------------------
# unsupported pairs
# ---------------------------------------------------------------------------


def test_split_supported_keeps_order() -> None:
    suite = ALL_SUITES["multivariate_intervals"]()
    ok, skipped = split_supported(suite, ["diabetes", "synthetic_mv_gaussian", "house_sales"])
    assert ok == ["synthetic_mv_gaussian"]
    assert skipped == ["diabetes", "house_sales"]
    assert not supports_dataset(suite, "diabetes")


def test_leaderboard_skips_unsupported_pairs(tmp_path: Path, monkeypatch) -> None:
    _FakeSuite.calls = []
    monkeypatch.setattr(lb, "ALL_SUITES", {"fake": _FakeSuite})
    out = tmp_path / "lb.json"
    lb.run_leaderboard(
        datasets=["diabetes", "fake_design"],
        suites=["fake"],
        epochs=1,
        smoke=True,
        output_path=str(out),
        markdown_path=None,
        device="cpu",
    )
    assert _FakeSuite.calls == ["fake_design"]
    meta = json.loads(out.read_text())["metadata"]
    assert meta["skipped_unsupported"] == [["fake", "diabetes"]]
    assert meta["n_attempts"] == 1
    rows = json.loads(out.read_text())["rows"]
    assert all("failed" not in str(r.get("Notes", "")) for r in rows)


def test_run_all_skips_unsupported_pairs(tmp_path: Path, monkeypatch) -> None:
    _FakeSuite.calls = []
    monkeypatch.setattr(run_all, "ALL_SUITES", {"fake": _FakeSuite})
    out = tmp_path / "run.json"
    argv = [
        "run_all.py",
        "--suites",
        "fake",
        "--datasets",
        "diabetes,fake_design",
        "--smoke",
        "--device",
        "cpu",
        "--output",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    run_all.main()
    assert _FakeSuite.calls == ["fake_design"]
    artifact = json.loads(out.read_text())
    assert [r["Dataset"] for r in artifact["rows"]] == ["fake_design"]
    assert artifact["notes"] == ["skipped (not supported): fake/diabetes"]


# ---------------------------------------------------------------------------
# infinite conformal intervals
# ---------------------------------------------------------------------------


def test_interval_metrics_with_infinite_bounds() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    lo = np.array([-1.0, 0.5, -np.inf, -np.inf])
    hi = np.array([1.0, 0.8, np.inf, 5.0])
    m = compute_interval_metrics(lo, hi, y)
    assert m["Coverage"] == pytest.approx(0.75)  # infinite intervals cover
    assert m["InfiniteIntervalFrac"] == pytest.approx(0.5)
    assert m["Width"] == pytest.approx((2.0 + 0.3) / 2)  # finite rows only
    # Row 2 misses by 0.2: score 0.3 + (2/0.1) * 0.2 = 4.3.
    assert m["IntervalScore"] == pytest.approx((2.0 + 4.3) / 2)
    assert all(math.isfinite(m[k]) for k in ("Coverage", "Width", "IntervalScore"))
    assert "inf-intervals: 50.0%" in infinite_interval_note(m)


def test_all_infinite_intervals_give_nan_width_not_inf() -> None:
    y = np.zeros(3)
    m = compute_interval_metrics(np.full(3, -np.inf), np.full(3, np.inf), y)
    assert m["Coverage"] == 1.0 and m["InfiniteIntervalFrac"] == 1.0
    assert math.isnan(m["Width"]) and math.isnan(m["IntervalScore"])


def test_finite_intervals_unchanged() -> None:
    m = compute_interval_metrics(np.array([0.0]), np.array([2.0]), np.array([1.0]))
    assert m == {"Coverage": 1.0, "Width": 2.0, "IntervalScore": 2.0, "InfiniteIntervalFrac": 0.0}
    assert infinite_interval_note(m) == ""


def test_pit_proxy_skips_infinite_intervals() -> None:
    pit = compute_pit_values(
        {"lower": np.array([0.0, -np.inf]), "upper": np.array([2.0, np.inf])},
        np.array([1.0, 0.0]),
    )
    assert np.array_equal(pit, np.array([0.5]))


def test_distribution_metrics_skip_infinite_std() -> None:
    m = compute_distribution_metrics(np.zeros(4), np.array([1.0, 1.0, np.inf, 1.0]), np.zeros(4))
    assert math.isfinite(m["NLL"]) and m["NonFiniteStdFrac"] == pytest.approx(0.25)


def test_compute_all_metrics_with_infinite_conformal_rows() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    lo, hi = y - 1.0, y + 1.0
    lo[:5], hi[:5] = -np.inf, np.inf
    m = compute_all_metrics({"y_pred": y, "lower": lo, "upper": hi}, y)
    assert m["InfiniteIntervalFrac"] == pytest.approx(0.1)
    finite = {k: v for k, v in m.items() if isinstance(v, float)}
    assert all(math.isfinite(v) for v in finite.values()), finite


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "Method": "conformal/Finite",
            "Library": "torchregress",
            "Suite": "conformal",
            "Dataset": "d",
            "RMSE": 1.0,
            "IntervalScore": 5.0,
            "Coverage": 0.9,
            "InfiniteIntervalFrac": 0.0,
            "Notes": "",
        },
        {
            "Method": "conformal/SmallCal",
            "Library": "torchregress",
            "Suite": "conformal",
            "Dataset": "d",
            "RMSE": 1.0,
            "IntervalScore": 2.0,  # finite rows only: looks better than it is
            "Coverage": 1.0,
            "InfiniteIntervalFrac": 0.4,
            "Notes": "",
        },
        {
            "Method": "conformal/LegacyInf",
            "Library": "torchregress",
            "Suite": "conformal",
            "Dataset": "d",
            "RMSE": float("nan"),
            "IntervalScore": float("inf"),
            "Coverage": 1.0,
            "Notes": "",
        },
    ]


def test_leaderboard_aggregation_is_finite_and_ranks_infinite_last() -> None:
    summary = lb.aggregate_per_method(_rows(), target_coverage=0.9)
    order = [s["Method"] for s in summary]
    assert order[0] == "Finite"
    by = {s["Method"]: s for s in summary}
    assert by["SmallCal"]["mean_InfiniteIntervalFrac"] == pytest.approx(0.4)
    assert by["LegacyInf"]["n_nonfinite_is"] == 1
    assert math.isnan(by["LegacyInf"]["mean_IntervalScore"])  # no finite cell, not inf
    md = lb.render_leaderboard_markdown(summary, _rows(), target_coverage=0.9)
    assert "Inf. Interval Frac" in md and lb.INFINITE_INTERVAL_NOTE in md
    lb.print_leaderboard_stdout(summary, target_coverage=0.9)


def test_report_renders_inf_and_footnote() -> None:
    rows = _rows()
    md = render_markdown({"rows": rows, "suites": ["conformal"], "datasets": ["d"]})
    assert "InfiniteIntervalFrac" in md
    assert "infinite bound" in md
    assert "| inf" in md or " inf " in md
