"""Inverse target scaling keeps the +-inf conformal bounds (Zoo_Prevalence failed on them)."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from harness.runners import _inverse_transform_predictions, _inverse_transform_y


def test_inverse_transform_keeps_infinite_bounds() -> None:
    sc = StandardScaler().fit(np.array([[1.0], [3.0], [5.0]]))
    y = np.array([-np.inf, 0.0, 1.0, np.inf])
    out = _inverse_transform_y(y, sc)
    np.testing.assert_array_equal(np.isinf(out), [True, False, False, True])
    np.testing.assert_allclose(out[1:3], sc.inverse_transform(y[1:3, None]).ravel())
    preds = _inverse_transform_predictions(
        {"lower": np.array([-np.inf, 0.0]), "upper": np.array([np.inf, 1.0])}, sc
    )
    assert preds["lower"][0] == -np.inf and preds["upper"][0] == np.inf
