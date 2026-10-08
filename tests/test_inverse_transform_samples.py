"""Regression test: predictive samples must be mapped back to the target scale."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from harness.runners import _inverse_transform_predictions


def test_samples_inverse_transformed_like_mean():
    rng = np.random.default_rng(0)
    y = rng.normal(150.0, 70.0, size=200)
    scaler = StandardScaler().fit(y.reshape(-1, 1))
    z = scaler.transform(y.reshape(-1, 1)).ravel()
    samples_z = z[:, None] + rng.normal(0, 0.1, size=(200, 99))
    out = _inverse_transform_predictions(
        {"samples": samples_z, "mean": samples_z.mean(axis=1)}, scaler
    )
    np.testing.assert_allclose(out["samples"].mean(axis=1), out["mean"], rtol=1e-10)
    assert abs(np.median(out["samples"]) - np.median(y)) < 10.0
