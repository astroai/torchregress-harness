"""Metadata and smoke tests for the ``uq_shift`` suite.

The smoke run is a tiny ``stress_ood`` fit (2 epochs, 200 fit / 50 evaluation
rows) and must finish well inside the CI budget.  It checks that every method
produces every shift view, that the metrics and OOD gaps are finite, and that
the deterministic shift builder respects the covariate-shift definition.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from suites.tabular.uq_shift import UQShiftSuite, _build_views

from harness.runners import HarnessConfig

_METHODS = ["MSE", "GaussianNLL", "BetaNLL", "MDN_K3"]
_SHIFTS = {"id", "covariate", "noise_shift"}
_FINITE_METRICS = (
    "RMSE",
    "MAE",
    "NLL",
    "CRPS",
    "Sharpness",
    "Coverage",
    "Width",
    "Coverage68",
    "Width68",
    "Coverage90",
    "Width90",
    "Coverage95",
    "Width95",
    "ECE",
    "MACE",
    "RMSCE",
    "PIT_Mean",
    "PIT_Std",
    "PIT_KS",
    "PIT_CvM",
)
_GAP_METRICS = ("NLL", "CRPS", "RMSE", "ECE", "Coverage68", "Coverage95", "Width95")


@pytest.fixture(scope="module")
def smoke_rows() -> list[dict]:
    suite = UQShiftSuite()
    return suite.run("stress_ood", config=HarnessConfig(seed=42, epochs=2, smoke=True))


def test_metadata() -> None:
    suite = UQShiftSuite()
    assert suite.name == "uq_shift"
    assert suite.description
    assert suite.supported_datasets == [
        "california_housing",
        "stress_ood",
        "synthetic_eiv",
        "synthetic_hetero",
    ]
    assert suite.methods() == _METHODS
    assert suite.supported_methods == _METHODS
    for method in _METHODS:
        assert suite.library_name(method) == "torchregress"
        assert suite.library_version(method)
        notes = suite.method_notes(method)
        assert len(notes) > 20, f"{method} has no useful notes"
        assert "MLP" in notes or "mixture" in notes.lower()


def test_smoke_rows_cover_every_method_and_shift(smoke_rows: list[dict]) -> None:
    assert smoke_rows, "smoke run produced no rows"
    pairs = {(row["Method"].split("/", 1)[1], row["Shift"]) for row in smoke_rows}
    assert pairs == {(method, shift) for method in _METHODS for shift in _SHIFTS}
    for row in smoke_rows:
        assert row["Dataset"] == "stress_ood"
        assert row["Suite"] == "uq_shift"
        assert row["Library"] == "torchregress"
        assert row["ShiftSpec"], f"missing ShiftSpec on {row['Method']}/{row['Shift']}"
        note = str(row.get("Notes", "")).lower()
        for marker in ("failed:", "error:", "skipped:"):
            assert marker not in note, f"{row['Method']}/{row['Shift']}: {row['Notes']}"


def test_smoke_metrics_are_finite(smoke_rows: list[dict]) -> None:
    for row in smoke_rows:
        where = f"{row['Method']}/{row['Shift']}"
        for key in _FINITE_METRICS:
            value = row.get(key)
            assert value is not None, f"{where}: missing {key}"
            assert math.isfinite(float(value)), f"{where}: {key}={value!r} is not finite"
        for key in ("Coverage68", "Coverage90", "Coverage95"):
            assert 0.0 <= float(row[key]) <= 1.0, f"{where}: {key} outside [0, 1]"
        for key in ("Width68", "Width90", "Width95"):
            assert float(row[key]) >= 0.0, f"{where}: negative {key}"
        assert float(row["NLL"]) > -50.0, f"{where}: implausible NLL"


def test_ood_gaps_are_shift_minus_id(smoke_rows: list[dict]) -> None:
    by_key = {(row["Method"], row["Shift"]): row for row in smoke_rows}
    for method in _METHODS:
        full = f"uq_shift/{method}"
        id_row = by_key[(full, "id")]
        for key in _GAP_METRICS:
            assert id_row[f"OOD_gap_{key}"] == 0.0, f"{method}: id gap for {key} is not 0"
        for shift in sorted(_SHIFTS - {"id"}):
            shift_row = by_key[(full, shift)]
            for key in _GAP_METRICS:
                gap = float(shift_row[f"OOD_gap_{key}"])
                expected = float(shift_row[key]) - float(id_row[key])
                assert math.isfinite(gap), f"{method}/{shift}: non-finite gap for {key}"
                assert math.isclose(gap, expected, rel_tol=1e-9, abs_tol=1e-12), (
                    f"{method}/{shift}: OOD_gap_{key}={gap} != {expected}"
                )


def _synthetic_splits() -> dict:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=400)).astype(np.float32)
    return {
        "X_train": X[:300],
        "y_train": y[:300],
        "X_test": X[300:],
        "y_test": y[300:],
    }


def test_shift_builder_is_deterministic_and_pairs_rows() -> None:
    splits = _synthetic_splits()

    fit_1, y_fit_1, views_1 = _build_views("synthetic_hetero", splits, seed=7, smoke=False)
    fit_2, y_fit_2, views_2 = _build_views("synthetic_hetero", splits, seed=7, smoke=False)

    assert np.array_equal(fit_1, fit_2)
    assert np.array_equal(y_fit_1, y_fit_2)
    assert [view.name for view in views_1] == ["id", "mean_shift", "noise_shift"]

    for left, right in zip(views_1, views_2, strict=True):
        assert np.array_equal(left.X, right.X), f"{left.name} X is not deterministic"
        assert np.array_equal(left.y, right.y), f"{left.name} y is not deterministic"

    id_view, mean_view, noise_view = views_1
    # Non-stress datasets use the native random test split as the id view.
    assert np.array_equal(id_view.X, splits["X_test"])
    assert np.array_equal(id_view.y, splits["y_test"])
    # Mean shift moves x0 only and leaves the labels untouched.
    assert not np.array_equal(mean_view.X, id_view.X)
    assert np.array_equal(mean_view.X[:, 1:], id_view.X[:, 1:])
    delta = mean_view.X[:, 0] - id_view.X[:, 0]
    assert np.allclose(delta, 1.5 * np.std(splits["X_train"][:, 0]), rtol=1e-5)
    assert np.array_equal(mean_view.y, id_view.y)
    # Noise shift perturbs labels only, on the same features.
    assert np.array_equal(noise_view.X, id_view.X)
    assert not np.array_equal(noise_view.y, id_view.y)


def test_synthetic_run_covers_mean_shift_path() -> None:
    """The non-stress dataset path (id / mean_shift / noise_shift) runs clean."""
    rows = UQShiftSuite().run(
        "unit",
        config=HarnessConfig(seed=42, epochs=2, smoke=True),
        splits=_synthetic_splits(),
    )
    pairs = {(row["Method"].split("/", 1)[1], row["Shift"]) for row in rows}
    assert pairs == {
        (method, shift) for method in _METHODS for shift in {"id", "mean_shift", "noise_shift"}
    }
    for row in rows:
        assert math.isfinite(float(row["RMSE"]))
        assert math.isfinite(float(row["NLL"]))
        assert math.isfinite(float(row["OOD_gap_NLL"]))
        assert row["ShiftSpec"]


def test_stress_ood_views_split_by_x0_sign() -> None:
    from harness.datasets import load_dataset

    splits = load_dataset("stress_ood", standardize_targets=True)
    X_fit, _y_fit, views = _build_views("stress_ood", splits, seed=42, smoke=False)
    by_name = {view.name: view for view in views}

    assert np.all(X_fit[:, 0] < 0.0), "stress_ood fit rows must come from x0 < 0"
    assert np.all(by_name["id"].X[:, 0] < 0.0), "stress_ood id rows must come from x0 < 0"
    assert np.all(by_name["covariate"].X[:, 0] >= 0.0), "stress_ood covariate rows must be x0 >= 0"
    assert 0 < len(by_name["id"].X) < len(splits["X_train"])
    # The noise shift is paired with the id view: same features, new labels.
    assert np.array_equal(by_name["noise_shift"].X, by_name["id"].X)
    assert not np.array_equal(by_name["noise_shift"].y, by_name["id"].y)
