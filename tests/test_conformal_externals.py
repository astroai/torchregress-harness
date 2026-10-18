"""External conformal comparator rows: puncc, conformal-tights, CQR reference.

Covers the rows added alongside the existing MAPIE / crepes / torchcp
comparators:

* ``puncc_Split+Linear`` and ``puncc_CQR+GBR`` — the real ``deel.puncc``
  ``SplitCP`` / ``CQR`` wrappers, gated on puncc being importable.
* ``conformal_tights_CQR+GBR`` — ``conformal_tights``'s coherent conformal
  quantile meta-estimator, gated on conformal-tights (and its Darts/XGBoost
  imports) being importable.
* ``CQR+GBR(reference)`` — the native reimplementation of the official
  Romano et al. (2019) CQR routine (https://github.com/yromano/cqr); it has no
  optional dependency, so its tests always run.

The optional-library tests use ``pytest.importorskip`` so this file still
collects and runs in an environment without the ``conformal`` extra
(see OPTIONAL_DEPS.md).
"""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular.conformal import (
    _CONFORMAL_TIGHTS_AVAILABLE,
    _CQR_REFERENCE_METHOD,
    _CQR_REFERENCE_VERSION,
    _PUNCC_AVAILABLE,
    ConformalSuite,
    _cqr_reference_fit,
    _external_fit,
    _external_predict,
)

from harness.runners import HarnessConfig

CONFORMAL_TIGHTS_METHOD = "conformal_tights_CQR+GBR"
PUNCC_METHODS = ("puncc_Split+Linear", "puncc_CQR+GBR")


def _make_splits(
    *,
    n_train: int = 240,
    n_cal: int = 80,
    n_test: int = 60,
    n_features: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray | float]:
    """Small linear-plus-noise regression problem on a fixed train/cal/test split."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_cal + n_test, n_features))
    coef = np.array([1.5, -2.0, 0.7, 0.3])[:n_features]
    y = X @ coef + rng.normal(scale=0.5, size=X.shape[0])
    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_cal": X[n_train : n_train + n_cal],
        "y_cal": y[n_train : n_train + n_cal],
        "X_test": X[n_train + n_cal :],
        "y_test": y[n_train + n_cal :],
        "_alpha": 0.1,
    }


def _assert_intervals(
    lo: np.ndarray,
    hi: np.ndarray,
    n_test: int,
    *,
    max_inversion_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    assert lo.shape == (n_test,)
    assert hi.shape == (n_test,)
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    inversion_frac = float(np.mean(lo > hi))
    assert inversion_frac <= max_inversion_frac, (
        f"{inversion_frac:.1%} of intervals are inverted (allowed {max_inversion_frac:.1%})"
    )
    assert float(np.mean(hi - lo)) > 0.0, "intervals must have positive mean width"
    return lo, hi


# ---------------------------------------------------------------------------
# Official CQR reference (native numpy/sklearn row — no optional dependency)
# ---------------------------------------------------------------------------


def test_cqr_reference_row_registered_and_documented() -> None:
    suite = ConformalSuite()
    assert _CQR_REFERENCE_METHOD in suite.methods()
    assert suite.library_name(_CQR_REFERENCE_METHOD) == "cqr-reference"
    assert suite.library_version(_CQR_REFERENCE_METHOD) == _CQR_REFERENCE_VERSION
    notes = suite.method_notes(_CQR_REFERENCE_METHOD)
    assert "Romano" in notes
    assert "github.com/yromano/cqr" in notes
    assert "arxiv.org/abs/1905.03222" in notes


def test_cqr_reference_matches_paper_calibration_rule() -> None:
    """q must be the reference's higher order statistic at (1-alpha)(1+1/n)."""
    from sklearn.ensemble import GradientBoostingRegressor

    splits = _make_splits(seed=1)
    alpha, seed = 0.1, 7
    state = _cqr_reference_fit(splits, alpha=alpha, seed=seed)
    lo, hi = _external_predict(_CQR_REFERENCE_METHOD, state, splits["X_test"], alpha=alpha)

    # Independent recomputation of the published CQR routine.
    q_lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, random_state=seed)
    q_hi = GradientBoostingRegressor(loss="quantile", alpha=1.0 - alpha / 2, random_state=seed)
    q_lo.fit(splits["X_train"], splits["y_train"])
    q_hi.fit(splits["X_train"], splits["y_train"])
    y_cal = np.asarray(splits["y_cal"]).reshape(-1)
    scores = np.maximum(
        q_lo.predict(splits["X_cal"]) - y_cal,
        y_cal - q_hi.predict(splits["X_cal"]),
    )
    level = (1.0 - alpha) * (1.0 + 1.0 / len(y_cal))
    expected_q = float(np.quantile(scores, level, method="higher"))

    assert state["q"] == pytest.approx(expected_q, rel=0.0, abs=1e-12)
    expected_lo = q_lo.predict(splits["X_test"]) - expected_q
    expected_hi = q_hi.predict(splits["X_test"]) + expected_q
    assert np.allclose(lo, expected_lo, atol=1e-10, rtol=0.0)
    assert np.allclose(hi, expected_hi, atol=1e-10, rtol=0.0)


def test_cqr_reference_row_runs_through_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = ConformalSuite()
    monkeypatch.setattr(ConformalSuite, "methods", lambda self: [_CQR_REFERENCE_METHOD])
    rows = suite.run(
        "unit",
        config=HarnessConfig(seed=0, smoke=True),
        splits=_make_splits(seed=2),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["Method"] == f"conformal/{_CQR_REFERENCE_METHOD}"
    assert row["Library"] == "cqr-reference"
    assert row["LibraryVersion"] == _CQR_REFERENCE_VERSION
    assert not str(row["Notes"]).startswith(("failed:", "skipped:"))
    assert row["Coverage"] is not None and 0.0 <= row["Coverage"] <= 1.0
    assert row["Width"] is not None and row["Width"] > 0.0


# ---------------------------------------------------------------------------
# puncc (deel.puncc SplitCP / CQR)
# ---------------------------------------------------------------------------


def test_puncc_rows_registered_and_versioned() -> None:
    pytest.importorskip("deel.puncc.regression")
    assert _PUNCC_AVAILABLE

    suite = ConformalSuite()
    registered = [m for m in suite.methods() if m.startswith("puncc_")]
    assert set(registered) == set(PUNCC_METHODS)
    for method in registered:
        assert suite.library_name(method) == "puncc"
        assert suite.library_version(method), f"{method} must self-report its puncc version"
        assert suite.library_version(method) != "unknown"
        assert suite.method_notes(method), f"{method} must carry method notes"


@pytest.mark.parametrize("method", PUNCC_METHODS)
def test_puncc_rows_end_to_end(method: str) -> None:
    pytest.importorskip("deel.puncc.regression")

    splits = _make_splits(seed=3)
    alpha = 0.1
    state = _external_fit(method, splits, alpha=alpha, seed=42)
    lo, hi = _external_predict(method, state, splits["X_test"], alpha=alpha)
    _assert_intervals(lo, hi, len(splits["X_test"]))


def test_puncc_cqr_agrees_with_native_reference() -> None:
    """Matched quantile GBRs must give puncc's CQR and the reference the same intervals.

    This is the parity contract behind the independent CQR reference row: the
    real ``deel.puncc.regression.CQR`` calibration and the native reimplementation
    of Romano et al. (2019) use the same finite-sample quantile rule, so with
    identical base learners (same seed) they must produce identical bands.
    """
    pytest.importorskip("deel.puncc.regression")

    splits = _make_splits(seed=4)
    alpha, seed = 0.1, 42
    ref_state = _cqr_reference_fit(splits, alpha=alpha, seed=seed)
    ref_lo, ref_hi = _external_predict(
        _CQR_REFERENCE_METHOD, ref_state, splits["X_test"], alpha=alpha
    )
    puncc_state = _external_fit("puncc_CQR+GBR", splits, alpha=alpha, seed=seed)
    puncc_lo, puncc_hi = _external_predict(
        "puncc_CQR+GBR", puncc_state, splits["X_test"], alpha=alpha
    )
    assert np.allclose(puncc_lo, ref_lo, atol=1e-9, rtol=0.0)
    assert np.allclose(puncc_hi, ref_hi, atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------------------
# conformal-tights (ConformalCoherentQuantileRegressor)
# ---------------------------------------------------------------------------


def test_conformal_tights_row_registered_and_documented() -> None:
    pytest.importorskip("conformal_tights")
    assert _CONFORMAL_TIGHTS_AVAILABLE

    suite = ConformalSuite()
    assert CONFORMAL_TIGHTS_METHOD in suite.methods()
    assert suite.library_name(CONFORMAL_TIGHTS_METHOD) == "conformal-tights"
    version = suite.library_version(CONFORMAL_TIGHTS_METHOD)
    assert version and version != "unknown"
    assert "calibration split" in suite.method_notes(CONFORMAL_TIGHTS_METHOD)


def test_conformal_tights_row_end_to_end() -> None:
    pytest.importorskip("conformal_tights")

    splits = _make_splits(seed=5)
    alpha = 0.1
    state = _external_fit(CONFORMAL_TIGHTS_METHOD, splits, alpha=alpha, seed=42)
    lo, hi = _external_predict(CONFORMAL_TIGHTS_METHOD, state, splits["X_test"], alpha=alpha)
    # The library's coherence constraint is fitted on its own calibration
    # design, so a test point can occasionally land outside the fitted
    # monotone cone.  The row reports the library's raw interval; tolerate a
    # small inversion fraction (documented in method_notes) instead of hiding it.
    _assert_intervals(lo, hi, len(splits["X_test"]), max_inversion_frac=0.1)

    # Coherent Conformal Prediction contract on the calibration design:
    # predicted quantiles are monotonically increasing (they never cross).
    quantiles = np.asarray(state.predict_quantiles(state.X_calib_, quantiles=(0.05, 0.5, 0.95)))
    assert quantiles.shape == (len(state.X_calib_), 3)
    assert np.all(np.diff(quantiles, axis=1) >= -1e-9)
