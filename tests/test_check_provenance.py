"""Tests for the synthetic-fallback release gate (tools/check_provenance.py)."""

from __future__ import annotations

import json
from pathlib import Path

from tools.check_provenance import check_rows, main, registry_sets

SYNTH = frozenset({"synthetic_hetero"})
KNOWN = frozenset({"synthetic_hetero", "diabetes"})


def _row(dataset: str, fallback: bool | None, notes: str = "") -> dict[str, object]:
    row: dict[str, object] = {"Method": "suite/M", "Dataset": dataset, "Notes": notes}
    if fallback is not None:
        row["SyntheticFallback"] = fallback
    return row


def _check(rows: list[dict[str, object]]):
    return check_rows(rows, source="x.json", synthetic_datasets=SYNTH, known_datasets=KNOWN)


def test_real_data_row_passes() -> None:
    assert _check([_row("diabetes", False)]) == []


def test_synthetic_by_design_passes() -> None:
    assert _check([_row("synthetic_hetero", True)]) == []


def test_real_dataset_fallback_fails() -> None:
    (v,) = _check([_row("diabetes", True)])
    assert "fell back" in v.reason


def test_unregistered_synthetic_fails() -> None:
    (v,) = _check([_row("mystery", True)])
    assert "not registered" in v.reason


def test_missing_provenance_fails() -> None:
    (v,) = _check([_row("diabetes", None)])
    assert "missing SyntheticFallback" in v.reason


def test_error_and_skip_rows_ignored() -> None:
    assert (
        _check([_row("diabetes", True, "SKIPPED: no lib"), _row("diabetes", None, "ERROR")]) == []
    )


def test_registry_marks_stress_datasets_synthetic() -> None:
    synthetic, known = registry_sets()
    assert "stress_ood" in synthetic
    assert "california_housing" in known
    assert "california_housing" not in synthetic


def test_cli_exit_codes(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"rows": [_row("california_housing", False)]}))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([_row("california_housing", True)]))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"seeds": [0, 1]}))
    assert main([str(good), str(other)]) == 0
    assert main([str(good), str(bad)]) == 1


def test_published_leaderboard_passes() -> None:
    """The canonical leaderboard must never carry a silent synthetic proxy."""
    root = Path(__file__).resolve().parents[1]
    assert main([str(root / "leaderboard_results.json")]) == 0


# ── strict data mode (load_dataset refuses synthetic proxies) ────────


def _fake_loader(**_: object) -> dict[str, object]:
    import numpy as np

    x = np.zeros((4, 2), dtype=np.float32)
    y = np.arange(4, dtype=np.float32)
    return {
        "X_train": x,
        "X_test": x,
        "y_train": y,
        "y_test": y,
        "_synthetic_fallback": True,
        "_fallback_reason": "mirror 404",
    }


def _register(monkeypatch, name: str, category: str) -> None:
    from harness import datasets

    info = datasets.DatasetInfo(
        name=name,
        category=category,
        n_features=2,
        n_samples=4,
        description="t",
        loader=_fake_loader,
    )
    monkeypatch.setitem(datasets.DATASET_REGISTRY, name, info)


def test_strict_mode_refuses_real_dataset_fallback(monkeypatch) -> None:
    import pytest

    from harness import datasets

    _register(monkeypatch, "fake_real", "tabular")
    monkeypatch.setenv(datasets.STRICT_DATA_ENV, "1")
    with pytest.raises(datasets.SyntheticFallbackError, match="mirror 404"):
        datasets.load_dataset("fake_real")


def test_strict_mode_allows_synthetic_by_design(monkeypatch) -> None:
    from harness import datasets

    _register(monkeypatch, "fake_synth", "synthetic")
    monkeypatch.setenv(datasets.STRICT_DATA_ENV, "1")
    assert datasets.load_dataset("fake_synth")["_synthetic_fallback"] is True


def test_default_mode_keeps_fallback_flagged(monkeypatch) -> None:
    from harness import datasets

    _register(monkeypatch, "fake_real2", "tabular")
    monkeypatch.delenv(datasets.STRICT_DATA_ENV, raising=False)
    assert datasets.load_dataset("fake_real2")["_synthetic_fallback"] is True
