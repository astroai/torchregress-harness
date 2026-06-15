"""Tests for harness core utilities."""

import numpy as np
import pytest

from harness.core import (
    compute_point_metrics,
    set_harness_seed,
    timed_call,
)


def test_set_harness_seed_reproducible() -> None:
    set_harness_seed(42)
    a = np.random.randn(5)
    set_harness_seed(42)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_timed_call_returns_result_and_time() -> None:
    def fn(x: int) -> int:
        return x * 2

    result, elapsed = timed_call(fn, 21)
    assert result == 42
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_compute_point_metrics_perfect() -> None:
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.0, 2.0, 3.0])
    metrics = compute_point_metrics(y_pred, y_true)
    assert metrics["MSE"] == 0.0
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == pytest.approx(1.0)


def test_compute_point_metrics_numpy_array() -> None:
    y_pred = np.array([0.0, 1.0, 0.0])
    y_true = np.array([1.0, 1.0, 1.0])
    metrics = compute_point_metrics(y_pred, y_true)
    assert metrics["MSE"] == pytest.approx(2.0 / 3.0)


def test_compute_point_metrics_constant_target() -> None:
    y_pred = np.array([5.0, 5.0])
    y_true = np.array([5.0, 5.0])
    metrics = compute_point_metrics(y_pred, y_true)
    assert metrics["R2"] != metrics["R2"]  # NaN due to zero variance (SS_tot = 0)
