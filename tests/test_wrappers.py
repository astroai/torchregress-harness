"""Smoke tests for PROPOSAL #3 sklearn-style wrappers in :mod:`harness.wrappers`.

Each wrapper is exercised through the full
``fit → predict → predict_interval → sample`` chain on a tiny synthetic dataset
(50 samples, 3 features, fast epochs).  Validates:

* ``.fit(X, y)`` returns self and sets ``_fitted = True``
* ``.predict(X)`` returns shape ``(N,)`` numeric arrays
* ``.predict_interval(X)`` returns ``(lower, upper)`` both of shape ``(N,)``
* ``.sample(X, n_samples)`` returns shape ``(N, n_samples)``

Plus shared protocol checks: unfitted-call raises, input validation accepts
1D ``X`` coerced to ``(N, 1)``, confidence bounds validation raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness.wrappers import (
    EvidentialRegressor,
    GaussianNLLRegressor,
    MDNRegressor,
    NormalizingFlowRegressor,
)

# All four estimators share the same protocol surface.
ALL_ESTIMATORS = [
    GaussianNLLRegressor,
    MDNRegressor,
    EvidentialRegressor,
    NormalizingFlowRegressor,
]


# ---------------------------------------------------------------------------
# Shared synthetic data fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_xy() -> tuple[np.ndarray, np.ndarray]:
    """50-sample, 3-feature linear-ish regression problem with mild noise."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((50, 3)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] + 0.1 * rng.standard_normal(50)).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Sklearn-style protocol smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_fit_returns_self_for_chaining(synthetic_xy, estimator_cls) -> None:
    """``.fit`` returns ``self`` (sklearn convention)."""
    X, y = synthetic_xy
    est = estimator_cls(epochs=2, batch_size=16, seed=42)
    out = est.fit(X, y)
    assert out is est
    assert est._fitted is True


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_predict_shape_and_finite(synthetic_xy, estimator_cls) -> None:
    """``.predict(X)`` returns shape ``(N,)`` with all finite values."""
    X, y = synthetic_xy
    est = estimator_cls(epochs=2, batch_size=16, seed=42).fit(X, y)
    pred = est.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_predict_interval_shapes_and_order(synthetic_xy, estimator_cls) -> None:
    """``.predict_interval(X)`` returns ``(lo, hi)`` both ``(N,)`` with ``hi >= lo``."""
    X, y = synthetic_xy
    est = estimator_cls(epochs=2, batch_size=16, seed=42).fit(X, y)
    lo, hi = est.predict_interval(X, confidence=0.9)
    assert lo.shape == (X.shape[0],)
    assert hi.shape == (X.shape[0],)
    assert np.all(np.isfinite(lo))
    assert np.all(np.isfinite(hi))
    assert np.all(hi >= lo - 1e-6)


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_predict_interval_confidence_widens_strict(synthetic_xy, estimator_cls) -> None:
    """Wider confidence ⇒ STRICTLY wider intervals (regression guard).

    The previous `>= - 1e-6` allowed equality and would let through an
    EvidentialRegress implementation that hard-coded ``confidence=0.9``.
    Strict-magnitude assertion (``+1e-3`` margin) catches the regression.
    """
    X, y = synthetic_xy
    est = estimator_cls(epochs=3, batch_size=16, seed=42).fit(X, y)
    lo50, hi50 = est.predict_interval(X, confidence=0.5)
    lo90, hi90 = est.predict_interval(X, confidence=0.9)
    width_50 = float(np.mean(hi50 - lo50))
    width_90 = float(np.mean(hi90 - lo90))
    assert width_90 > width_50 + 1e-3, (
        # 1e-3 margin is tight but safe here: MDN/NF sample means/uniform
        # Gaussian both produce well-separated 50%/90% quantiles (~5× wider
        # in expectation); Evidential uses 512 samples (low MC variance) so
        # 1e-3 is reliably distinguishable from equality even with df≈2.
        f"{estimator_cls.__name__}: 90% width ({width_90:.4f}) should be "
        f"STRICTLY wider than 50% width ({width_50:.4f}); sampler is "
        "ignoring the confidence kwarg (regression guard)."
    )


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_sample_shape(synthetic_xy, estimator_cls) -> None:
    """``.sample(X, n_samples)`` returns ``(N, n_samples)``."""
    X, y = synthetic_xy
    est = estimator_cls(epochs=2, batch_size=16, seed=42).fit(X, y)
    s = est.sample(X, n_samples=10)
    assert s.shape == (X.shape[0], 10)
    assert np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# Shared protocol behaviour
# ---------------------------------------------------------------------------


def test_unfitted_predict_raises() -> None:
    """Calling ``.predict`` before ``.fit`` raises ``RuntimeError``."""
    est = GaussianNLLRegressor(epochs=2, batch_size=16)
    X = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="not fitted"):
        est.predict(X)


def test_unfitted_sample_raises() -> None:
    """Calling ``.sample`` before ``.fit`` raises ``RuntimeError``."""
    est = GaussianNLLRegressor(epochs=2, batch_size=16)
    X = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="not fitted"):
        est.sample(X, n_samples=5)


def test_unfitted_predict_interval_raises() -> None:
    """Calling ``.predict_interval`` before ``.fit`` raises ``RuntimeError``."""
    est = GaussianNLLRegressor(epochs=2, batch_size=16)
    X = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="not fitted"):
        est.predict_interval(X)


def test_one_dimensional_X_coerced_to_2d() -> None:
    """1D ``X`` (single feature) coerced to ``(N, 1)`` — sklearn convention."""
    X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    est = GaussianNLLRegressor(epochs=2, batch_size=16, seed=42).fit(X, y)
    pred = est.predict(X)
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))


def test_three_dimensional_X_raises() -> None:
    """``X`` with rank > 2 raises a clear ``ValueError``."""
    X = np.zeros((5, 2, 2), dtype=np.float32)  # 3-D input is invalid
    y = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="1D or 2D"):
        GaussianNLLRegressor(epochs=1, batch_size=16, seed=42).fit(X, y)


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_confidence_validation_raises(synthetic_xy, estimator_cls) -> None:
    """``confidence`` outside (0, 1) raises ``ValueError``."""
    X, y = synthetic_xy
    est = estimator_cls(epochs=2, batch_size=16, seed=42).fit(X, y)
    with pytest.raises(ValueError, match="confidence must be in"):
        est.predict_interval(X, confidence=1.5)
    with pytest.raises(ValueError, match="confidence must be in"):
        est.predict_interval(X, confidence=-0.1)


# ---------------------------------------------------------------------------
# Estimator-specific guards
# ---------------------------------------------------------------------------


def test_mdn_n_components_propagates_to_layer() -> None:
    """``n_components`` argument of MDNRegressor is honoured."""
    X = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    y = X[:, 0].astype(np.float32)
    for K in (2, 5):
        est = MDNRegressor(n_components=K, epochs=1, batch_size=16, seed=42).fit(X, y)
        pred = est.predict(X)
        assert pred.shape == (20,)
        assert np.all(np.isfinite(pred))


def test_evidential_predict_interval_is_conservative() -> None:
    """Evidential predictive intervals should give reasonable (non-NaN) bounds."""
    X = np.random.default_rng(0).standard_normal((30, 2)).astype(np.float32)
    y = (X[:, 0] + 0.1 * np.random.default_rng(1).standard_normal(30)).astype(np.float32)
    est = EvidentialRegressor(epochs=2, batch_size=16, seed=42).fit(X, y)
    lo, hi = est.predict_interval(X, confidence=0.9)
    width = float(np.mean(hi - lo))
    # Evidential intervals are typically conservative — ``width > 0`` and finite.
    # We don't cap the upper bound because heavy-tailed Student-t quantiles
    # can blow up on tiny synthetic datasets with very few epochs.
    assert width > 0.0 and np.isfinite(width), f"unexpected evidential width: {width}"


def test_evidential_respects_confidence_arg() -> None:
    """Evidential ``predict_interval`` MUST respect the ``confidence`` kwarg.

    Regression guard: a previous implementation hardcoded
    ``confidence=0.9`` inside ``_predict_diagnostics`` and returned only
    ``lower``/``upper`` keys, so every ``predict_interval`` call returned
    the 90% interval regardless of the user's chosen confidence.  This test
    exercises 50% vs 99% and asserts STRICT widening beyond noise.
    """
    X = np.random.default_rng(0).standard_normal((60, 3)).astype(np.float32)
    y = (X[:, 0] + 0.5 * np.random.default_rng(2).standard_normal(60)).astype(np.float32)
    est = EvidentialRegressor(epochs=4, batch_size=16, seed=42).fit(X, y)
    lo50, hi50 = est.predict_interval(X, confidence=0.50)
    lo99, hi99 = est.predict_interval(X, confidence=0.99)
    width_50 = float(np.mean(hi50 - lo50))
    width_99 = float(np.mean(hi99 - lo99))
    assert width_99 > width_50 + 1e-3, (
        f"Evidential ignores confidence arg — width_99 ({width_99:.4f}) "
        f"should be STRICTLY wider than width_50 ({width_50:.4f}) "
        "(regression: _predict_diagnostics was hardcoding confidence=0.9)"
    )


def test_normalizingflow_sample_count_consistent() -> None:
    """NormalizingFlow sample shape matches the documented ``NF_N_SAMPLES``."""
    X = np.random.default_rng(0).standard_normal((30, 3)).astype(np.float32)
    y = (X[:, 0] + 0.1 * np.random.default_rng(1).standard_normal(30)).astype(np.float32)
    est = NormalizingFlowRegressor(epochs=2, batch_size=16, seed=42).fit(X, y)
    s = est.sample(X, n_samples=20)
    assert s.shape == (30, 20)


# ---------------------------------------------------------------------------
# Determinism hardening — repeated inference calls must be bit-identical
# ---------------------------------------------------------------------------

# (Pure-deterministic GaussianNLL excluded from this group: its
# ``_predict_diagnostics`` does no MC sampling, so the wrapper is already
# deterministic across calls by construction.)

_DETERMINISTIC_STOCHASTIC_WRAPPERS = [
    MDNRegressor,
    EvidentialRegressor,
    NormalizingFlowRegressor,
]


@pytest.mark.parametrize("estimator_cls", _DETERMINISTIC_STOCHASTIC_WRAPPERS)
def test_repeated_predict_is_bit_identical(synthetic_xy, estimator_cls) -> None:
    """Repeated ``.predict()`` / ``.predict_interval()`` / ``.sample()``
    calls must produce bit-identical MC samples even when the global
    torch RNG has been advanced in between.

    Regression guard: ``_predict_diagnostics`` must wrap MC sampling in
    ``torch.random.fork_rng`` + ``torch.manual_seed(self._inference_seed)``
    so each call re-asserts the cached fit-time seed.  Without this,
    Adam step shuffles + intervening torch ops would produce different
    MC draws on successive ``.predict()`` invocations.
    """
    import torch  # local import — deterministic tests use torch state probing

    X, y = synthetic_xy
    est = estimator_cls(epochs=3, batch_size=16, seed=42).fit(X, y)

    # Pollute global torch RNG between predict calls — without
    # ``fork_rng`` + ``manual_seed`` re-assertion this would cause
    # MC samples to differ between calls.
    torch.manual_seed(999)
    for _ in range(7):
        _ = torch.randn(128)

    pred1 = est.predict(X).copy()
    lo1, hi1 = est.predict_interval(X, confidence=0.9)
    s1 = est.sample(X, n_samples=20)

    # Pollute again between second batch of calls.
    torch.manual_seed(2025)
    for _ in range(11):
        _ = torch.randn(128)

    pred2 = est.predict(X).copy()
    lo2, hi2 = est.predict_interval(X, confidence=0.9)
    s2 = est.sample(X, n_samples=20)

    np.testing.assert_array_equal(
        pred1,
        pred2,
        err_msg=(
            f"{estimator_cls.__name__}: .predict() not bit-identical across "
            f"calls (got diff norm {float(np.linalg.norm(pred1 - pred2)):.3e}); "
            "fork_rng + manual_seed missing in _predict_diagnostics"
        ),
    )
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)
    np.testing.assert_array_equal(s1, s2)


@pytest.mark.parametrize("estimator_cls", _DETERMINISTIC_STOCHASTIC_WRAPPERS)
def test_predict_does_not_leak_global_rng_state(synthetic_xy, estimator_cls) -> None:
    """After ``.predict()`` returns, the global torch RNG state must be
    unchanged from before the call (``torch.random.fork_rng`` restores it).
    """
    import torch

    X, y = synthetic_xy
    est = estimator_cls(epochs=2, batch_size=16, seed=42).fit(X, y)

    # Snapshot the global torch RNG state before any sampling call.
    torch.manual_seed(31415)
    state_before = torch.random.get_rng_state().clone()

    # Trigger ``_predict_diagnostics`` internally (consumes
    # ``fork_rng`` scope).
    _ = est.predict(X)

    state_after = torch.random.get_rng_state().clone()
    assert torch.equal(state_before, state_after), (
        f"{estimator_cls.__name__}: fork_rng did not restore the global "
        "torch RNG state — repeated inference calls leak RNG state into "
        "the surrounding harness"
    )


# ---------------------------------------------------------------------------
# PROPOSAL #8 — pandas DataFrame input adapters
# ---------------------------------------------------------------------------


def _make_synthetic_df(n: int = 50, seed: int = 7) -> pd.DataFrame:
    """Build a 3-column numeric DataFrame for adapter tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.standard_normal(n).astype(np.float32),
            "x2": rng.standard_normal(n).astype(np.float32),
            "x3": rng.standard_normal(n).astype(np.float32),
        },
    )


def test_dataframe_input_adapter_fit_captures_introspection_attrs() -> None:
    """``fit(pd.DataFrame, pd.Series)`` populates sklearn-style introspection
    attrs: ``n_features_in_``, ``feature_names_in_`` (column order preserved),
    ``target_name_``.
    """

    df = _make_synthetic_df()
    y = pd.Series(
        df["x1"].to_numpy()
        - 0.5 * df["x2"].to_numpy()
        + 0.1 * np.random.default_rng(2).standard_normal(len(df)).astype(np.float32),
        name="y_target",
    )
    est = GaussianNLLRegressor(epochs=2, batch_size=16, seed=42).fit(df, y)
    assert est.feature_names_in_ == ["x1", "x2", "x3"]
    assert est.n_features_in_ == 3
    assert est.target_name_ == "y_target"


def test_dataframe_input_adapter_predict_reorders_columns() -> None:
    """``predict(X_df_reordered)`` produces the same predictions (modulo
    noise) as ``predict(X_df)`` when the column set is identical and only
    the order differs — column-set validation rather than position validation.
    """

    df = _make_synthetic_df(n=40)
    y_np = df["x1"].to_numpy()
    est = GaussianNLLRegressor(epochs=2, batch_size=16, seed=42).fit(df, y_np)
    # Reorder columns
    df_reordered = df[["x3", "x1", "x2"]]
    pred_orig = est.predict(df)
    pred_reordered = est.predict(df_reordered)
    np.testing.assert_allclose(
        pred_orig,
        pred_reordered,
        rtol=1e-6,
        err_msg=(
            "predict(df_reordered) must match predict(df) when columns "
            "are reordered but to the same feature set"
        ),
    )


def test_dataframe_input_adapter_predict_subset_raises() -> None:
    """``predict(X_df)`` with X_df missing fit-time columns raises a clear
    ``ValueError`` mentioning the missing columns.
    """

    df = _make_synthetic_df()
    y_np = df["x1"].to_numpy()
    est = GaussianNLLRegressor(epochs=2, batch_size=16, seed=42).fit(df, y_np)
    df_subset = df[["x1", "x2"]]
    with pytest.raises(ValueError, match="missing"):
        est.predict(df_subset)


def test_dataframe_input_adapter_predict_extra_columns_raises() -> None:
    """``predict(X_df)`` with X_df containing columns not seen at fit-time
    raises ``ValueError``.
    """

    df = _make_synthetic_df()
    y_np = df["x1"].to_numpy()
    est = GaussianNLLRegressor(epochs=2, batch_size=16, seed=42).fit(df, y_np)
    df_extra = df.copy()
    df_extra["x_extra"] = np.zeros(len(df), dtype=np.float32)
    with pytest.raises(ValueError, match="extra"):
        est.predict(df_extra)


def test_dataframe_input_adapter_fit_non_numeric_raises() -> None:
    """``fit(X_df)`` with non-numeric dtype columns raises ``TypeError``
    naming the offending columns.
    """

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(20).astype(np.float32),
            "x2_str": ["a"] * 20,
        }
    )
    y_np = rng.standard_normal(20).astype(np.float32)
    with pytest.raises(TypeError, match="non-numeric"):
        GaussianNLLRegressor(epochs=1, batch_size=16, seed=42).fit(df, y_np)


def test_dataframe_input_adapter_pandas_series_target_preserves_name() -> None:
    """When ``y`` is a ``pd.Series`` with ``name``, ``reg.target_name_`` is
    that string.  Predictions remain numpy.
    """

    df = _make_synthetic_df(n=30)
    y = pd.Series(
        df["x1"].to_numpy().astype(np.float32),
        name="price",
    )
    est = MDNRegressor(epochs=2, batch_size=16, seed=42).fit(df, y)
    assert est.target_name_ == "price"
    pred = est.predict(df)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (30,)


def test_dataframe_input_adapter_dataframe_target_one_column() -> None:
    """Single-column ``pd.DataFrame`` ``y`` is accepted; ``target_name_``
    is the column label.
    """

    df = _make_synthetic_df(n=30)
    y = pd.DataFrame({"price": df["x1"].to_numpy().astype(np.float32)})
    est = GaussianNLLRegressor(epochs=2, batch_size=16, seed=42).fit(df, y)
    assert est.target_name_ == "price"
    pred = est.predict(df)
    assert pred.shape == (30,)


def test_dataframe_input_adapter_dataframe_target_multi_column_raises() -> None:
    """Multi-column ``pd.DataFrame`` ``y`` is rejected (scalar wrappers
    only).  Multivariate targets live in the multivariate suite.
    """

    df = _make_synthetic_df(n=20)
    y = pd.DataFrame(
        {
            "y0": df["x1"].to_numpy().astype(np.float32),
            "y1": df["x2"].to_numpy().astype(np.float32),
        }
    )
    with pytest.raises(ValueError, match="scalar"):
        GaussianNLLRegressor(epochs=1, batch_size=16, seed=42).fit(df, y)


def test_dataframe_input_adapter_heterogeneous_numpy_to_dataframe() -> None:
    """Fit on numpy, predict on DataFrame: sklearn-style heterogeneous path.
    Columns of predict-time DataFrame are not validated (no feature_names_in_
    capture), so any numeric DataFrame works.
    """

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 2)).astype(np.float32)
    y = rng.standard_normal(30).astype(np.float32)
    est = GaussianNLLRegressor(epochs=2, batch_size=16, seed=42).fit(X, y)
    df_pred = pd.DataFrame(
        {
            "anything": rng.standard_normal(10).astype(np.float32),
            "goes": rng.standard_normal(10).astype(np.float32),
        },
    )
    pred = est.predict(df_pred)
    assert pred.shape == (10,)
    assert np.all(np.isfinite(pred))


def test_dataframe_input_adapter_heterogeneous_dataframe_to_numpy() -> None:
    """Fit on DataFrame, predict on numpy: feature_names_in_ is set but
    predict-time validation only kicks in when predict(X) is a DataFrame.
    numpy predict-time is a no-op for the column check.
    """
    df = _make_synthetic_df(n=30)
    rng = np.random.default_rng(5)
    y_np = (df["x1"].to_numpy() + 0.1 * rng.standard_normal(30)).astype(np.float32)
    est = GaussianNLLRegressor(epochs=2, batch_size=16, seed=42).fit(df, y_np)
    X_np = df.to_numpy(dtype=np.float32)
    pred = est.predict(X_np)
    assert pred.shape == (30,)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_dataframe_input_adapter_all_four_wrappers_match_numpy_path(
    synthetic_xy,
    estimator_cls,
) -> None:
    """Cross-validate: ``fit(pd.DataFrame, pd.Series).predict(pd.DataFrame)``
    produces the same numerical predictions (modulo MC stochasticity) as
    ``fit(np.ndarray, np.ndarray).predict(np.ndarray)`` for an identical-row
    DataFrame/ndarray view.
    """

    X_np, y_np = synthetic_xy
    df = pd.DataFrame(X_np, columns=["x1", "x2", "x3"])
    y_series = pd.Series(y_np, name="y")
    # Fit on DataFrame path
    est_df = estimator_cls(epochs=2, batch_size=16, seed=42).fit(df, y_series)
    # Fit on numpy path
    est_np = estimator_cls(epochs=2, batch_size=16, seed=42).fit(X_np, y_np)
    pred_df = est_df.predict(df)
    pred_np = est_np.predict(X_np)
    assert isinstance(pred_df, np.ndarray)
    assert pred_df.shape == pred_np.shape
    # The two paths MUST agree numerically (same data → same predictions)
    np.testing.assert_allclose(
        pred_df,
        pred_np,
        rtol=1e-6,
        atol=1e-6,
        err_msg=(
            f"{estimator_cls.__name__}: DataFrame path and numpy path must "
            "produce identical predictions for identical inputs"
        ),
    )
