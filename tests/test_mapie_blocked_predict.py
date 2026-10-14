"""MAPIE predictions in test-row blocks equal one-shot predictions."""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular import conformal as C


@pytest.mark.skipif(not C._MAPIE_CROSS_CQR_AVAILABLE, reason="needs MAPIE git main")
def test_cross_cqr_blocked_predict_matches_single_shot() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = X[:, 0] + 0.5 * rng.normal(size=400)
    splits = {"X_train": X[:250], "y_train": y[:250], "X_cal": X[250:300], "y_cal": y[250:300]}
    pred = C._mapie_cross_cqr_fit(splits, alpha=0.1, seed=0)
    X_test = X[300:]
    _, ref = pred.predict_interval(X_test)
    lo_ref, hi_ref = C._mapie_intervals_to_bounds(ref)
    pred._n_conformalize_rows = (1 << 24) // 7  # 7-row blocks
    lo, hi = C._mapie_predict(pred, X_test)
    np.testing.assert_array_equal(lo, lo_ref)
    np.testing.assert_array_equal(hi, hi_ref)
