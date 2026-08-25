"""Regression tests for the honest interval-position PIT proxy.

Audit finding (r9 / TR-TT PIT-noise): ``compute_pit_values`` used to return a
seeded Uniform(0, 1) draw for interval-only predictions, laundering any
interval method — including catastrophically wrong ones — into "perfectly
calibrated" diagnostics.
"""

import numpy as np
import pytest

from harness.metrics import compute_pit_values, pit_uniformity_tests


def _intervals(lower, upper):
    return {"lower": np.asarray(lower, dtype=float), "upper": np.asarray(upper, dtype=float)}


def test_interval_pit_position_inside() -> None:
    preds = _intervals([0.0, 0.0], [10.0, 10.0])
    y = np.array([2.5, 7.5])
    pit = compute_pit_values(preds, y)
    assert np.allclose(pit, [0.25, 0.75])


def test_interval_pit_saturates_outside() -> None:
    preds = _intervals(np.full(3, -11.0), np.full(3, -10.0))
    y = np.zeros(3)  # y above every interval bound
    pit = compute_pit_values(preds, y)
    assert np.allclose(pit, 1.0)


def test_miscalibrated_intervals_no_longer_laundered() -> None:
    n = 200
    y = np.zeros(n)
    wrong_low = _intervals(np.full(n, 10.0), np.full(n, 11.0))  # misses low entirely
    wrong_high = _intervals(np.full(n, -11.0), np.full(n, -10.0))  # misses high entirely

    t_low = pit_uniformity_tests(compute_pit_values(wrong_low, y))
    t_high = pit_uniformity_tests(compute_pit_values(wrong_high, y))

    # Both must be flagged as grossly non-uniform, not fake-perfect.
    assert abs(t_low["PIT_Mean"] - 0.5) > 0.4
    assert abs(t_high["PIT_Mean"] - 0.5) > 0.4
    assert t_low["PIT_KS"] > 0.5 and t_high["PIT_KS"] > 0.5


def test_degenerate_interval_raises() -> None:
    preds = _intervals([1.0], [1.0])  # upper == lower
    with pytest.raises(ValueError, match="upper > lower"):
        compute_pit_values(preds, np.zeros(1))


def test_well_placed_intervals_look_reasonable() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(size=400)
    preds = _intervals(y - 2.0, y + 2.0)  # centered intervals
    pit = compute_pit_values(preds, y)
    assert abs(pit.mean() - 0.5) < 0.05
