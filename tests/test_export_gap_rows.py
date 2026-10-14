"""Rows that give suite coverage to previously allowlisted torchregress exports.

* multivariate_intervals ``MultivariateGaussian+LevelSetCP`` -> ``LevelSetConformalPredictor``
* test_time_adaptation ``SemiConformal`` -> ``SemiConformalCalibrator``
* ordinal ``DiscreteWasserstein1`` -> ``DiscreteWasserstein1Loss``
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from suites.tabular import test_time_adaptation as tta
from suites.tabular.multivariate_intervals import (
    MultivariateIntervalsSuite,
    _covariance_from_cholesky_params,
    _gaussian_density_level_set_cp,
)
from suites.tabular.ordinal import OrdinalSuite

from harness.runners import HarnessConfig

D = 3
N_PARAMS = D + D * (D + 1) // 2


def _ok(row: dict) -> bool:
    return not str(row.get("Notes", "")).startswith("failed")


# ---------------------------------------------------------------------------
# LevelSetConformalPredictor (multivariate_intervals)
# ---------------------------------------------------------------------------


def _gaussian_heads(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random full-covariance Gaussian head outputs and targets drawn from them."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(n, N_PARAMS, generator=g, dtype=torch.float64)
    mu = raw[:, :D]
    cov = _covariance_from_cholesky_params(raw[:, D:], D)
    y = (
        mu
        + (torch.linalg.cholesky(cov) @ torch.randn(n, D, 1, generator=g, dtype=raw.dtype))[..., 0]
    )
    return raw, cov, y


def test_level_set_cp_is_library_subclass() -> None:
    from torchregress.losses.conformal import LevelSetConformalPredictor

    assert isinstance(_gaussian_density_level_set_cp(D), LevelSetConformalPredictor)


def test_level_set_box_matches_analytic_hpd_projection() -> None:
    raw_c, _, y_c = _gaussian_heads(400, seed=0)
    raw_t, cov_t, _ = _gaussian_heads(50, seed=1)
    grid = 4096
    cp = _gaussian_density_level_set_cp(D, alpha=0.1, grid_size=grid)
    cp.calibrate(raw_c, y_c)
    lo, hi = cp.predict_interval(raw_t)
    q = float(cp.q_hat)
    mu = raw_t[:, :D]
    log_norm = 0.5 * (D * np.log(2 * np.pi) + torch.logdet(cov_t))
    r = (2.0 * (q - log_norm)).clamp_min(0.0)
    var = torch.diagonal(cov_t, dim1=-2, dim2=-1)
    half = torch.sqrt(r.unsqueeze(1) * var)
    # Exact box up to one grid step of the base class's level-set search.
    span = ((mu + 1.05 * torch.sqrt(r.max() * var)).max(0).values) - (
        (mu - 1.05 * torch.sqrt(r.max() * var)).min(0).values
    )
    step = (span / (grid - 1)).unsqueeze(0)
    assert torch.all((lo - (mu - half)).abs() <= step + 1e-6)
    assert torch.all((hi - (mu + half)).abs() <= step + 1e-6)


def test_level_set_region_is_calibrated() -> None:
    raw_c, _, y_c = _gaussian_heads(1000, seed=2)
    raw_t, _, y_t = _gaussian_heads(4000, seed=3)
    cp = _gaussian_density_level_set_cp(D, alpha=0.1)
    cp.calibrate(raw_c, y_c)
    region = cp._compute_scores(raw_t, y_t) <= cp.q_hat
    lo, hi = cp.predict_interval(raw_t)
    in_box = ((y_t >= lo) & (y_t <= hi)).all(dim=1)
    assert 0.87 <= float(region.double().mean()) <= 0.93
    # The box contains the region (up to grid resolution at the boundary).
    assert float(in_box.double().mean()) >= float(region.double().mean()) - 0.01


def test_level_set_row_shares_base_model_with_mahalanobis_row() -> None:
    suite = MultivariateIntervalsSuite()
    assert "MultivariateGaussian+LevelSetCP" in suite.methods()
    assert "calibration carve" in suite.method_notes("MultivariateGaussian+LevelSetCP")
    suite.methods = lambda: [  # type: ignore[method-assign]
        "MultivariateGaussian+Conformal",
        "MultivariateGaussian+LevelSetCP",
    ]
    cfg = HarnessConfig(seed=0, epochs=2, batch_size=64, smoke=True, standardize_targets=True)
    rows = {r["Method"].split("/", 1)[1]: r for r in suite.run("synthetic_mv_gaussian", cfg)}
    ls, mh = rows["MultivariateGaussian+LevelSetCP"], rows["MultivariateGaussian+Conformal"]
    assert _ok(ls), ls
    assert 0.0 <= ls["JointCoverage"] <= 1.0
    assert np.isfinite(ls["Width"]) and ls["Width"] > 0
    assert _ok(mh), mh
    # Same seed, carve and model: identical point predictions, different regions.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 4))
    y = (X[:, :D] + 0.3 * rng.normal(size=(160, D))).reshape(-1)
    split = {"X_train": X, "y_train": y, "X_test": X[:20], "y_test": y[: 20 * D]}
    preds = {
        m: suite.predict(m, suite.fit(m, split, cfg), X[:20], cfg)
        for m in ("MultivariateGaussian+Conformal", "MultivariateGaussian+LevelSetCP")
    }
    a, b = preds["MultivariateGaussian+Conformal"], preds["MultivariateGaussian+LevelSetCP"]
    np.testing.assert_array_equal(a["y_pred"], b["y_pred"])
    assert not np.allclose(a["upper"] - a["lower"], b["upper"] - b["lower"])


# ---------------------------------------------------------------------------
# SemiConformalCalibrator (test_time_adaptation)
# ---------------------------------------------------------------------------


def _shift_splits(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_tr, n_te = 1200, 1200
    X = rng.normal(size=(n_tr + n_te, 4)).astype(np.float32)
    noise = 0.2 + 0.5 * np.abs(X[:, 0])  # heteroscedastic: shift changes residual scale
    y = (X[:, 0] + 0.5 * X[:, 1] + noise * rng.normal(size=len(X))).astype(np.float32)
    return {"X_train": X[:n_tr], "y_train": y[:n_tr], "X_test": X[n_tr:], "y_test": y[n_tr:]}


def test_tta_evaluate_scores_predictions_against_their_own_rows() -> None:
    """Predictions live on the shifted rows already; evaluate must not re-index them."""
    suite = tta.TestTimeAdaptationSuite()
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    idx = rng.choice(200, size=200, replace=True)
    pred = {
        "y_pred": y[idx],
        "mean": y[idx],
        "std": np.full(200, 0.1),
        "lower": y[idx] - 0.1,
        "upper": y[idx] + 0.1,
        "tta_tgt_idx": idx.astype(np.float64),
    }
    m = suite.evaluate("Static", pred, y, HarnessConfig(seed=0))
    assert m["RMSE"] == pytest.approx(0.0, abs=1e-12)
    assert m["Coverage"] == pytest.approx(1.0)


def test_semi_conformal_row_smoke() -> None:
    suite = tta.TestTimeAdaptationSuite()
    assert "SemiConformal" in suite.methods()
    splits = _shift_splits()
    cfg = HarnessConfig(seed=0, epochs=15, batch_size=64, smoke=True, standardize_targets=False)
    state = suite.fit("SemiConformal", splits, cfg)
    assert len(state["X_cal"]) == len(splits["X_train"]) // 5
    pred = suite.predict("SemiConformal", state, splits["X_test"], cfg)
    m = suite.evaluate("SemiConformal", pred, splits["y_test"], cfg)
    assert 0.8 <= m["Coverage"] <= 1.0, m
    assert 0.0 <= m["Coverage_Unweighted"] <= 1.0
    assert m["InfiniteFraction"] == 0.0
    assert 1.0 < m["WeightESS"] <= len(state["X_cal"])
    assert 0.0 < m["WeightMax"] < 1.0
    assert np.all(pred["upper"] >= pred["lower"])


# ---------------------------------------------------------------------------
# DiscreteWasserstein1Loss (ordinal)
# ---------------------------------------------------------------------------


def test_discrete_wasserstein1_ordinal_row() -> None:
    suite = OrdinalSuite()
    assert "DiscreteWasserstein1" in suite.methods()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    y = np.digitize(X[:, 0] + 0.3 * rng.normal(size=600), [-1.0, -0.3, 0.3, 1.0])
    splits = {
        "X_train": X[:400],
        "y_train": y[:400].astype(np.float32),
        "X_test": X[400:],
        "y_test": y[400:].astype(np.float32),
    }
    suite.methods = lambda: ["DiscreteWasserstein1"]  # type: ignore[method-assign]
    cfg = HarnessConfig(seed=0, epochs=5, batch_size=64, smoke=True, standardize_targets=False)
    (row,) = suite.run("synthetic_ordinal", cfg, splits=splits)
    assert _ok(row), row
    assert row["Accuracy"] > 0.4  # chance is ~0.2 for 5 classes
    assert row["MAE"] < 1.0
