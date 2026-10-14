"""Contracts for the semi-supervised suite (smoke mode)."""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular.semi_supervised import (
    N_LABELED,
    SKLEARN_METHODS,
    TORCH_METHODS,
    SemiSupervisedSuite,
    draw_friedman,
    labeled_split,
)
from tools._suite_registry import ALL_SUITES

from harness.runners import HarnessConfig


def test_registered() -> None:
    assert ALL_SUITES["semi_supervised"] is SemiSupervisedSuite
    assert set(SemiSupervisedSuite().supported_datasets) == set(N_LABELED)


def test_labeled_split_is_disjoint_and_seeded() -> None:
    lab, unl = labeled_split(100, 20, seed=3)
    assert len(lab) == 20 and len(unl) == 80
    assert not set(lab) & set(unl)
    assert np.array_equal(lab, labeled_split(100, 20, seed=3)[0])
    with pytest.raises(ValueError):
        labeled_split(10, 10, seed=0)


def test_friedman_design_is_standardized() -> None:
    d = draw_friedman(0, smoke=True)
    assert d["X_train"].shape == (400, 10)
    assert abs(float(d["y_train"].mean())) < 1e-5
    assert d["X_test"].shape[0] == len(d["y_test"]) == 200


@pytest.fixture(scope="module")
def smoke_rows() -> dict[str, dict]:
    rows = SemiSupervisedSuite().run("ssl_friedman", config=HarnessConfig(seed=0, smoke=True))
    return {r["Method"].split("/")[1]: r for r in rows}


def test_smoke_rows_all_succeed(smoke_rows: dict[str, dict]) -> None:
    assert set(smoke_rows) == {*TORCH_METHODS, *SKLEARN_METHODS}
    for name, row in smoke_rows.items():
        assert not str(row["Notes"]).startswith("failed"), (name, row["Notes"])
        assert np.isfinite(row["RMSE"]), name
        assert row["SyntheticFallback"] is False


def test_labeled_budget_is_shared(smoke_rows: dict[str, dict]) -> None:
    n_lab = N_LABELED["ssl_friedman"][1]
    for name, row in smoke_rows.items():
        if name != "Oracle_FullyLabeled":
            assert row["NLabeled"] == n_lab, name
    assert smoke_rows["Oracle_FullyLabeled"]["NUnlabeled"] == 0


def test_teacher_student_reports_trust_weights(smoke_rows: dict[str, dict]) -> None:
    for name in ("TeacherStudent_SAGEReg", "TeacherStudent_BatchRelative"):
        assert 0.0 <= smoke_rows[name]["MeanTrustWeight"] <= 1.0
        assert smoke_rows[name]["MeanDisagreement"] >= 0.0
    # Gaussian rows carry interval metrics with the infinite-interval column.
    assert smoke_rows["Supervised_GaussianNLL"]["InfiniteIntervalFrac"] == 0.0


def test_unsupported_dataset_gives_failed_rows() -> None:
    rows = SemiSupervisedSuite().run("california_housing", config=HarnessConfig(smoke=True))
    assert rows and all(str(r["Notes"]).startswith("failed") for r in rows)


def test_real_dataset_respects_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """diabetes is bundled with scikit-learn: strict mode loads it, no synthetic stand-in."""
    from harness.datasets import STRICT_DATA_ENV

    monkeypatch.setenv(STRICT_DATA_ENV, "1")
    suite = SemiSupervisedSuite()
    suite.methods = lambda: ["Ridge_labeled"]  # type: ignore[method-assign]
    rows = suite.run("diabetes", config=HarnessConfig(seed=0, smoke=True))
    assert rows[0]["SyntheticFallback"] is False
    assert np.isfinite(rows[0]["RMSE"])
