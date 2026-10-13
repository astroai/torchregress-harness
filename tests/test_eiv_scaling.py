"""eiv: equation-error estimate for exact-target datasets and analytic-Jacobian ODR."""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular.eiv import EIVSuite, equation_error_sd

from harness.runners import HarnessConfig


def _eiv(n: int, d: int, eq_sd: float, sx: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xt = rng.standard_normal((n, d))
    y = Xt @ rng.standard_normal(d) / np.sqrt(d) + eq_sd * rng.standard_normal(n)
    return Xt + sx * rng.standard_normal(Xt.shape), y


@pytest.mark.parametrize("eq_sd", [0.2, 0.5])
def test_equation_error_sd_recovers_truth_from_noisy_inputs(eq_sd: float) -> None:
    X, y = _eiv(20000, 8, eq_sd, sx=0.4)
    est = equation_error_sd(X, y, 0.4)
    assert est == pytest.approx(eq_sd, rel=0.1)
    # Ignoring the input noise inflates the estimate (attenuated fit).
    assert equation_error_sd(X, y, 0.0) > est


def test_exact_target_dataset_runs_tls_and_odr() -> None:
    """sigma_y = 0 (exact targets) used to make TLS raise and ODRPACK crash."""
    X, y = _eiv(3000, 6, 0.3, sx=0.2)
    k = 2400
    splits = {
        "X_train": X[:k].astype(np.float32),
        "X_test": X[k:].astype(np.float32),
        "y_train": y[:k].astype(np.float32),
        "y_test": y[k:].astype(np.float32),
        "sigma_x": 0.2,
        "sigma_y": 0.0,
    }
    suite = EIVSuite()
    cfg = HarnessConfig(seed=0, epochs=2)
    for method in ("TLS", "SciPyODR", "LatentNN"):
        state = suite.fit(method, splits, cfg)
        pred = suite.predict(method, state, splits["X_test"], cfg)
        metrics = suite.evaluate(method, pred, splits["y_test"], cfg)
        assert np.isfinite(metrics["RMSE"]), (method, metrics)
        if method != "LatentNN":  # 2 epochs: only finiteness is meaningful for the NN
            assert metrics["RMSE"] < 0.6, (method, metrics)
