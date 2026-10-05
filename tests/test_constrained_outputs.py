"""Contracts for the constrained-outputs suite."""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular.constrained_outputs import (
    DESIGNS,
    METHODS,
    ConstrainedOutputsSuite,
    pinball,
    violation_rate,
)
from tools._suite_registry import ALL_SUITES

from harness.runners import HarnessConfig


def test_registered() -> None:
    assert ALL_SUITES["constrained_outputs"] is ConstrainedOutputsSuite


@pytest.mark.parametrize(
    ("kind", "pred", "rate"),
    [
        ("nonneg", np.array([[0.0], [-0.1], [2.0]]), 1 / 3),
        ("bounded", np.array([[0.0], [1.0], [1.2]]), 1 / 3),
        ("simplex", np.array([[0.5, 0.5], [0.7, 0.4], [-0.1, 1.1]]), 2 / 3),
        ("quantiles", np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 1.0]]), 1 / 2),
    ],
)
def test_violation_rate(kind: str, pred: np.ndarray, rate: float) -> None:
    assert violation_rate(kind, pred) == pytest.approx(rate)


def test_pinball_of_exact_median_prediction() -> None:
    y = np.array([[1.0], [3.0]])
    assert pinball(np.array([[2.0], [2.0]]), y, levels=(0.5,)) == pytest.approx(0.5)


@pytest.mark.parametrize("name", list(DESIGNS))
def test_designs_obey_their_constraint(name: str) -> None:
    design = DESIGNS[name]
    data = design.draw(np.random.default_rng(0))
    y = data["y_train"]
    if design.kind != "quantiles":
        assert violation_rate(design.kind, y) == 0.0
    assert data["X_train"].shape[0] == len(y)


@pytest.mark.parametrize("name", ["con_nonneg", "con_simplex", "con_quantiles"])
def test_builtin_constraint_never_violates(name: str, monkeypatch) -> None:
    from suites.tabular import constrained_outputs

    # One LightGBM model per quantile level dominates the runtime; drop it here.
    trimmed = {k: tuple(m for m in v if m != "LightGBM_quantile") for k, v in METHODS.items()}
    monkeypatch.setattr(constrained_outputs, "METHODS", trimmed)
    suite = ConstrainedOutputsSuite()
    rows = suite.run(name, config=HarnessConfig(seed=0, smoke=True, epochs=2))
    by_method = {r["Method"].split("/")[1]: r for r in rows}
    assert set(by_method) == set(trimmed[DESIGNS[name].kind])
    builtin = {"con_nonneg": "NonNegativeHead", "con_simplex": "SimplexHead"}.get(
        name, "NonCrossingSort"
    )
    assert by_method[builtin]["ViolationRate"] == 0.0
