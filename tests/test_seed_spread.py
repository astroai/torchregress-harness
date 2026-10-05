"""Tests for tools/seed_spread.py (flaky-row detection across seeds)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.seed_spread import collect, main, spread


def _row(method: str, rmse: float, dataset: str = "diabetes") -> dict[str, object]:
    return {"Method": f"conformal/{method}", "Dataset": dataset, "RMSE": rmse}


def test_stable_row_is_not_flaky() -> None:
    runs = [[_row("A", 1.00)], [_row("A", 1.01)], [_row("A", 0.99)]]
    (r,) = spread(collect(runs), ["RMSE"], rel_tol=0.05)
    assert not r.flaky
    assert r.metrics[0].mean == pytest.approx(1.0)


def test_noisy_row_is_flaky() -> None:
    runs = [[_row("A", 1.0)], [_row("A", 1.5)], [_row("A", 0.6)]]
    (r,) = spread(collect(runs), ["RMSE"], rel_tol=0.05)
    assert r.flaky
    assert r.suite == "conformal"


def test_rows_seen_once_and_nonfinite_values_are_skipped() -> None:
    runs = [[_row("A", 1.0), _row("B", 2.0)], [_row("A", float("nan"))]]
    assert spread(collect(runs), ["RMSE"], rel_tol=0.05) == []


def test_cli_fail_on_flaky(tmp_path: Path) -> None:
    paths = []
    for i, v in enumerate([1.0, 2.0]):
        p = tmp_path / f"s{i}.json"
        p.write_text(json.dumps({"rows": [_row("A", v)]}))
        paths.append(str(p))
    out = tmp_path / "spread.json"
    assert main([*paths, "--json-out", str(out)]) == 0
    assert main([*paths, "--fail-on-flaky"]) == 1
    assert json.loads(out.read_text())["rows"][0]["flaky"] is True
