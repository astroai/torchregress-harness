"""Tests for harness core utilities."""

import numpy as np
import pytest

from harness.core import (
    compute_point_metrics,
    set_harness_seed,
    timed_call,
)
from harness.metrics import (
    compute_all_metrics,
    compute_pit_values,
    pit_uniformity_tests,
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


# ===========================================================================
# PIT (Probability Integral Transform) diagnostics — Proposal #9
# ===========================================================================


def test_pit_gaussian_matches_scipy() -> None:
    """PIT from mean+std must match scipy.stats.norm.cdf exactly.

    Parity test against scipy — defensive check that our PIT implementation
    doesn't drift from the reference.  For perfectly specified Gaussian
    predictive, PIT values should be ~ Uniform(0, 1).
    """
    from scipy.stats import norm

    rng = np.random.default_rng(42)
    n = 200
    mean = np.zeros(n)
    std = np.ones(n)
    y_true = rng.standard_normal(n)  # y | X ~ N(0, 1)

    pit = compute_pit_values({"mean": mean, "std": std}, y_true)

    expected = norm.cdf(y_true, loc=mean, scale=std)
    assert pit.shape == (n,)
    assert np.allclose(pit, expected)


def test_pit_uniformity_on_uniform_samples() -> None:
    """PIT from Uniform(0, 1) samples must yield uniformity statistics.

    Uniform input → uniform ECDF → KS and Cramér-von Mises T-statistic
    bounded by the asymptotic null distribution.  This is the
    *well-calibrated* baseline.  PIT_Mean ≈ 0.5, PIT_Std ≈ 0.289 (1/√12).

    Note on PIT_CvM: the harness reports n·ω² (the T-statistic, scipy's
    convention).  For n=500 perfect uniform inputs the asymptotic mean
    is ≈ 0.166 with std ≈ 0.15 (Anderson 1962, Table 1).  We allow
    generous headroom to avoid flagging statistical flukes.
    """
    rng = np.random.default_rng(0)
    pit_values = rng.uniform(0.0, 1.0, size=500)

    tests = pit_uniformity_tests(pit_values)
    assert tests["PIT_Mean"] == pytest.approx(0.5, abs=0.05)
    assert tests["PIT_Std"] == pytest.approx(0.289, abs=0.02)
    # KS ≤ 1/√n for the empirical uniform CDF.
    assert tests["PIT_KS"] < 0.10
    # T = n·ω² has 99th-percentile < 1.0 for n=500 under the null.
    assert tests["PIT_CvM"] < 1.0


def test_pit_uniformity_detects_overconfident_peaked() -> None:
    """Overconfident (peaked) PIT histogram has U-shape → KS significantly > 0.

    A model that's too confident produces PIT values clustered near 0 and 1.
    We simulate this by pushing PIT values to the extremes: PIT ∈ {0.05, 0.95}.
    """
    pit_values = np.array([0.05] * 50 + [0.95] * 50, dtype=np.float64)

    tests = pit_uniformity_tests(pit_values)
    # Mean is 0.5 (symmetric) but std is much larger than uniform's 0.289
    assert tests["PIT_Mean"] == pytest.approx(0.5, abs=0.01)
    assert tests["PIT_Std"] > 0.40  # well above uniform's 0.289
    # KS hits the bimodal pattern: should be ~0.45
    assert tests["PIT_KS"] > 0.30
    assert tests["PIT_CvM"] > 1.0


def test_pit_from_samples_empirical_cdf() -> None:
    """Sample-based PIT = empirical CDF at y_true.

    For samples from N(0, 1), PIT at y_true (drawn from N(0, 1)) should be
    close to Uniform(0, 1).  We draw many samples (n=2048) so the empirical
    CDF approximates the true CDF.
    """
    rng = np.random.default_rng(7)
    n_test = 100
    n_samples = 2048

    # y_true drawn from N(0, 1); samples also from N(0, 1)
    y_true = rng.standard_normal(n_test)
    samples = rng.standard_normal((n_test, n_samples))

    pit = compute_pit_values({"samples": samples}, y_true)
    assert pit.shape == (n_test,)
    assert np.all(pit >= 0.0) and np.all(pit <= 1.0)

    tests = pit_uniformity_tests(pit)
    # Empirical PIT from well-specified flow should be near-uniform
    assert tests["PIT_Mean"] == pytest.approx(0.5, abs=0.1)
    assert tests["PIT_KS"] < 0.15  # not too far from uniform


def test_pit_from_samples_transposed_layout() -> None:
    """Sample-based PIT must accept (S, N) layout (transpose on the fly)."""
    rng = np.random.default_rng(11)
    n_test = 50
    n_samples = 256

    y_true = rng.standard_normal(n_test)
    samples_n_s = rng.standard_normal((n_test, n_samples))
    samples_s_n = samples_n_s.T  # (S, N)

    pit_n_s = compute_pit_values({"samples": samples_n_s}, y_true)
    pit_s_n = compute_pit_values({"samples": samples_s_n}, y_true)

    assert np.allclose(pit_n_s, pit_s_n, atol=1e-12)


def test_pit_1d_samples_returns_single_value() -> None:
    """Edge case: 1D sample array returns a single PIT value."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(100)
    y_true = np.array([0.0])

    pit = compute_pit_values({"samples": samples}, y_true)
    assert pit.shape == (1,)
    assert 0.0 <= float(pit[0]) <= 1.0


def test_pit_evidential_student_t() -> None:
    """NIG predictive PIT must match scipy.stats.t at the same params.

    For an evidential regression that has converged to the trivial case
    (γ=μ, ν=∞ approximates Gaussian), the Student-t should approach N.
    """
    from scipy.stats import t as student_t

    rng = np.random.default_rng(99)
    n = 100
    gamma = np.zeros(n)
    nu = np.full(n, 1e6)  # very large ν → near-Gaussian
    alpha = np.full(n, 1e6)
    beta = np.full(n, 1e6)
    y_true = rng.standard_normal(n)

    pit = compute_pit_values(
        {
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
        },
        y_true,
    )

    expected_scale = np.sqrt(beta * (1.0 + nu) / (nu * alpha))
    expected = student_t.cdf(
        y_true,
        df=2.0 * alpha,
        loc=gamma,
        scale=expected_scale,
    )
    assert np.allclose(pit, expected, atol=1e-6)


def test_pit_uniformity_tests_empty_input_returns_nan() -> None:
    """Empty / too-small input → NaN statistics (not error)."""
    tests_0 = pit_uniformity_tests(np.array([]))
    assert all(np.isnan(v) for v in tests_0.values())

    tests_2 = pit_uniformity_tests(np.array([0.5, 0.5]))
    assert all(np.isnan(v) for v in tests_2.values())


def test_compute_all_metrics_auto_adds_pit_for_samples() -> None:
    """compute_all_metrics must include PIT diagnostics when samples present."""
    rng = np.random.default_rng(0)
    n = 100
    samples = rng.standard_normal((n, 256))
    y_true = rng.standard_normal(n)

    metrics = compute_all_metrics({"samples": samples}, y_true)
    # Point, distribution, AND PIT calibration metrics must all be present
    assert "RMSE" in metrics
    assert "NLL" in metrics
    assert "PIT_Mean" in metrics
    assert "PIT_Std" in metrics
    assert "PIT_KS" in metrics
    assert "PIT_CvM" in metrics


def test_compute_all_metrics_auto_adds_pit_for_nig() -> None:
    """compute_all_metrics must include PIT diagnostics for NIG params."""
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.standard_normal(n)

    metrics = compute_all_metrics(
        {
            "gamma": np.zeros(n),
            "nu": np.full(n, 10.0),
            "alpha": np.full(n, 5.0),
            "beta": np.full(n, 2.0),
        },
        y_true,
    )
    assert "PIT_Mean" in metrics
    assert "PIT_KS" in metrics


def test_compute_all_metrics_no_pit_for_intervals_only() -> None:
    """Intervals-only predictions fall back to uniform PIT placeholder
    (conformal intervals are uniform-by-design).

    With n >= 3 test points the placeholder uniform PIT contributes the
    PIT_Mean/PIT_KS/PIT_CvM statistics.  Below n=3 we skip uniformity tests
    (statistically meaningless with so few points).
    """
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.standard_normal(n)
    preds = rng.standard_normal(n)
    lower = preds - 1.0
    upper = preds + 1.0

    metrics = compute_all_metrics({"lower": lower, "upper": upper}, y_true)
    assert "Coverage" in metrics
    assert "IntervalScore" in metrics
    # PIT placeholder is uniform → PIT_Mean ≈ 0.5
    assert "PIT_Mean" in metrics
    assert metrics["PIT_Mean"] == pytest.approx(0.5, abs=0.10)
    assert "PIT_KS" in metrics
