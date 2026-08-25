"""Workstream B audit-fix regression tests for the conformal suite (B5).

Covers:
- torchregress split/CQR interval helpers use the shared finite-sample
  quantile rule (HN-WRAP-01) instead of linear-interpolated ``np/torch.quantile``.
- The external fit-half / predict-half restructure produces intervals
  identical to the old one-shot path on a fixed seed (skipped when the
  corresponding external library is not installed).
- ``ConformalSuite.fit`` carries the fitted external predictor under the
  ``"predictor"`` key so ``predict()`` performs inference only.
"""

import numpy as np
import pytest
import torch
from suites.tabular.conformal import (
    _MAPIE_AVAILABLE,
    _TORCHCP_AVAILABLE,
    ConformalSuite,
    _external_fit,
    _external_predict,
    _tabicl_split_intervals,
    _torch_cqr_intervals,
    _torch_split_intervals,
)

try:
    import tabicl  # noqa: F401

    _TABICL_AVAILABLE = True
except ImportError:
    _TABICL_AVAILABLE = False


def _make_splits(n_train=200, n_cal=80, n_test=50, n_features=3, seed=0):
    generator = np.random.default_rng(seed)
    X = generator.normal(size=(n_train + n_cal + n_test, n_features))
    coef = np.array([1.5, -2.0, 0.7])
    y = X @ coef + generator.normal(scale=0.5, size=X.shape[0])
    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_cal": X[n_train : n_train + n_cal],
        "y_cal": y[n_train : n_train + n_cal],
        "X_test": X[n_train + n_cal :],
        "y_test": y[n_train + n_cal :],
    }


class _TinyMLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 16), torch.nn.ReLU(), torch.nn.Linear(16, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# HN-WRAP-01 -- finite-sample quantile rule in the torchregress helpers
# ---------------------------------------------------------------------------


def test_torch_split_intervals_match_finite_sample_rule() -> None:
    """q must equal the ceil((n+1)*(1-alpha)) order statistic of residuals."""
    import math

    splits = _make_splits()
    alpha = 0.1
    model = _TinyMLP(3, 1)
    model.eval()
    with torch.no_grad():
        X_cal_t = torch.from_numpy(splits["X_cal"]).float()
        pred_cal = model(X_cal_t).squeeze(-1)
    X_test_t = torch.from_numpy(splits["X_test"]).float()
    lo, hi = _torch_split_intervals(
        model, X_cal_t, torch.from_numpy(splits["y_cal"]).float(), X_test_t, alpha=alpha
    )

    residuals = torch.abs(torch.from_numpy(splits["y_cal"]).float() - pred_cal)
    k = min(math.ceil((residuals.numel() + 1) * (1.0 - alpha)), residuals.numel())
    expected_q = float(torch.sort(residuals).values[k - 1])
    with torch.no_grad():
        pred_test = model(X_test_t).squeeze(-1)
    assert np.allclose(lo, (pred_test - expected_q).numpy())
    assert np.allclose(hi, (pred_test + expected_q).numpy())


def test_torch_cqr_intervals_match_finite_sample_rule() -> None:
    import math

    splits = _make_splits()
    alpha = 0.15
    model = _TinyMLP(3, 2)
    model.eval()
    with torch.no_grad():
        X_cal_t = torch.from_numpy(splits["X_cal"]).float()
        pred_cal = model(X_cal_t)
    X_test_t = torch.from_numpy(splits["X_test"]).float()
    lo, hi = _torch_cqr_intervals(
        model, X_cal_t, torch.from_numpy(splits["y_cal"]).float(), X_test_t, alpha=alpha
    )

    y_cal_t = torch.from_numpy(splits["y_cal"]).float()
    score = torch.maximum(pred_cal[:, 0] - y_cal_t, y_cal_t - pred_cal[:, 1])
    k = min(math.ceil((score.numel() + 1) * (1.0 - alpha)), score.numel())
    expected_q = float(torch.sort(score).values[k - 1])
    with torch.no_grad():
        pred_test = model(X_test_t)
    assert np.allclose(lo, (pred_test[:, 0] - expected_q).numpy())
    assert np.allclose(hi, (pred_test[:, 1] + expected_q).numpy())


# ---------------------------------------------------------------------------
# Fit/predict split equivalence -- external builders
# ---------------------------------------------------------------------------


def test_mapie_fit_predict_split_equivalent_to_one_shot() -> None:
    """Fit-half + predict-half must reproduce the old one-shot intervals."""
    from suites.tabular.conformal import _mapie_split_intervals

    if not _MAPIE_AVAILABLE:
        pytest.skip("mapie not installed -- install per EXTERNAL_PINS.md")

    splits = _make_splits(seed=1)
    predictor = _external_fit("MAPIE_Split+Linear", splits, alpha=0.1, seed=42)
    lo_split, hi_split = _external_predict(
        "MAPIE_Split+Linear", predictor, splits["X_test"], alpha=0.1
    )
    lo_one, hi_one = _mapie_split_intervals(splits, alpha=0.1, seed=42)
    # Deterministic base estimator: both paths must agree bit-for-bit.
    assert np.allclose(lo_split, lo_one)
    assert np.allclose(hi_split, hi_one)


@pytest.mark.skipif(not _TABICL_AVAILABLE, reason="tabicl not installed")
def test_tabicl_fit_predict_split_equivalent_to_one_shot() -> None:
    splits = _make_splits(seed=2)
    predictor = _external_fit("Split+TabICL", splits, alpha=0.1, seed=42)
    lo_split, hi_split = _external_predict("Split+TabICL", predictor, splits["X_test"], alpha=0.1)
    lo_one, hi_one = _tabicl_split_intervals(splits, alpha=0.1, seed=42)
    assert np.allclose(lo_split, lo_one)
    assert np.allclose(hi_split, hi_one)


@pytest.mark.skipif(not _TORCHCP_AVAILABLE, reason="torchcp not installed")
def test_torchcp_row_end_to_end() -> None:
    splits = _make_splits(seed=3)
    predictor = _external_fit("torchcp_Split+Linear", splits, alpha=0.1, seed=42)
    lo, hi = _external_predict("torchcp_Split+Linear", predictor, splits["X_test"], alpha=0.1)
    assert lo.shape == (len(splits["X_test"]),)
    assert np.all(lo <= hi)


# ---------------------------------------------------------------------------
# Suite state plumbing -- fitted predictor carried through fit()
# ---------------------------------------------------------------------------


def test_suite_external_fit_stores_predictor_and_predict_is_inference_only() -> None:
    if not _MAPIE_AVAILABLE:
        pytest.skip("mapie not installed -- install per EXTERNAL_PINS.md")

    from harness.runners import HarnessConfig

    suite = ConformalSuite()
    splits = _make_splits(seed=4)
    cfg = HarnessConfig(seed=42, epochs=1)
    state = suite.fit("MAPIE_Split+Linear", splits, cfg)
    assert state["type"] == "external"
    assert "predictor" in state, "fit() must store the fitted external predictor"

    preds = suite.predict("MAPIE_Split+Linear", state, splits["X_test"], cfg)
    n_test = len(splits["X_test"])
    assert preds["lower"].shape == (n_test,)
    assert np.all(preds["lower"] <= preds["upper"])
