"""Comparator rows must be on the same scale/targets as the torchregress rows.

Regression tests for two August-leaderboard artefacts:

* ``ordinal``: ``statsmodels_OrderedModel`` returned the argmax *column index*
  of ~N distinct standardised float levels (RMSE ~100-1300 vs ~0.8) because
  regression targets were never discretised into shared classes.
* ``eiv``: numpy TLS ran on an unweighted ``[X | y]`` SVD (equal-noise
  assumption violated, collinear features) and produced RMSE 4-5 orders
  above the baselines.
"""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular.eiv import EIVSuite, TLSIllConditionedError, tls_solve
from suites.tabular.ordinal import OrdinalSuite, discretize_splits

from harness.runners import HarnessConfig

pytest.importorskip("statsmodels")


def _latent_logit_splits(n: int = 3000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2)).astype(np.float32)
    latent = 1.5 * X[:, 0] - 1.0 * X[:, 1] + rng.logistic(size=n)
    y = np.digitize(latent, [-1.5, 0.0, 1.5]).astype(np.int64)  # 4 classes
    h = int(0.7 * n)
    return {
        "X_train": X[:h],
        "y_train": y[:h],
        "X_test": X[h:],
        "y_test": y[h:],
        "n_classes": 4,
    }


def test_ordered_model_matches_cumulative_link_on_latent_logit():
    suite = OrdinalSuite()
    cfg = HarnessConfig(epochs=30, smoke=False)
    splits = discretize_splits(_latent_logit_splits())
    res = {}
    for m in ("CumulativeLink", "statsmodels_OrderedModel"):
        state = suite.fit(m, splits, cfg)
        pred = suite.predict(m, state, splits["X_test"], cfg)
        res[m] = suite.evaluate(m, pred, splits["y_test"], cfg)
    a, b = res["CumulativeLink"], res["statsmodels_OrderedModel"]
    assert b["RMSE"] < 1.5  # class units, never ~1e2-1e3
    assert abs(a["RMSE"] - b["RMSE"]) < 0.1
    assert abs(a["Accuracy"] - b["Accuracy"]) < 0.05


def test_ordinal_run_shares_discretised_classes_on_regression_targets():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=400)) * 80.0 + 150.0  # original units
    mu, sd = y[:300].mean(), y[:300].std()
    raw = {"X_train": X[:300], "y_train": y[:300], "X_test": X[300:], "y_test": y[300:]}
    std = dict(raw, y_train=(y[:300] - mu) / sd, y_test=(y[300:] - mu) / sd)
    d_raw, d_std = discretize_splits(raw, 5), discretize_splits(std, 5)
    np.testing.assert_array_equal(d_raw["y_train"], d_std["y_train"])
    np.testing.assert_array_equal(d_raw["y_test"], d_std["y_test"])
    assert d_raw["y_train"].min() == 0 and d_raw["y_train"].max() == 4
    assert "_y_scaler" not in discretize_splits(dict(std, _y_scaler=object()), 5)

    cfg = HarnessConfig(epochs=2, smoke=True)
    suite = OrdinalSuite()
    suite.methods = lambda: ["MSE", "CumulativeLink", "statsmodels_OrderedModel"]  # type: ignore[method-assign]
    rows = suite.run("custom", cfg, splits=std)
    assert len(rows) == 3
    for r in rows:
        assert r["RMSE"] is not None and r["RMSE"] <= 4.0, r  # <= K-1 class units


def test_ordered_model_maps_missing_class_back_to_label():
    s = _latent_logit_splits(n=1500)
    s["y_train"] = np.where(s["y_train"] == 1, 2, s["y_train"])  # class 1 absent in train
    cfg = HarnessConfig(epochs=1)
    suite = OrdinalSuite()
    state = suite.fit("statsmodels_OrderedModel", s, cfg)
    pred = suite.predict("statsmodels_OrderedModel", state, s["X_test"], cfg)["y_pred"]
    assert set(np.unique(pred)) <= {0, 2, 3}


def _eiv_design(n=6000, sx=0.3, sy=0.15, seed=0):
    rng = np.random.default_rng(seed)
    xt = rng.normal(size=n)
    x = xt + sx * rng.normal(size=n)
    y = 2.0 * xt + 1.0 + sy * rng.normal(size=n)
    return x, y, sx, sy


def test_tls_matches_deming_closed_form():
    x, y, sx, sy = _eiv_design()
    coef, icpt = tls_solve(x[:, None], y, sx, sy)
    delta = sy**2 / sx**2
    sxx, syy = np.var(x), np.var(y)
    sxy = np.cov(x, y, bias=True)[0, 1]
    slope = (syy - delta * sxx + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy**2)) / (2 * sxy)
    assert coef[0] == pytest.approx(slope, rel=1e-6)
    assert icpt == pytest.approx(y.mean() - slope * x.mean(), abs=1e-6)
    assert coef[0] == pytest.approx(2.0, abs=0.1)  # unbiased vs attenuated OLS (~1.8)
    # unit invariance: y in other units with matching sigma_y rescales the fit
    coef_k, _ = tls_solve(x[:, None], 1000 * y, sx, 1000 * sy)
    assert coef_k[0] == pytest.approx(1000 * coef[0], rel=1e-6)


def test_tls_row_close_to_torchregress_baseline_in_same_units():
    x, y, sx, sy = _eiv_design(n=2000)
    X = x[:, None].astype(np.float32)
    h = 1400
    splits = {
        "X_train": X[:h],
        "y_train": y[:h].astype(np.float32),
        "X_test": X[h:],
        "y_test": y[h:].astype(np.float32),
        "sigma_x": sx,
        "sigma_y": sy,
    }
    cfg = HarnessConfig(epochs=40)
    suite = EIVSuite()
    out = {}
    for m in ("MSE", "TLS"):
        st = suite.fit(m, splits, cfg)
        out[m] = suite.evaluate(
            m, suite.predict(m, st, splits["X_test"], cfg), splits["y_test"], cfg
        )
    assert out["TLS"]["RMSE"] < 1.5 * out["MSE"]["RMSE"]
    assert out["TLS"]["RMSE"] > 0.5 * out["MSE"]["RMSE"]


def test_tls_reports_ill_conditioned_instead_of_huge_rmse():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(300, 1))
    X = np.hstack([base, base + 1e-6 * rng.normal(size=(300, 1))])  # collinear
    y = base[:, 0] + 0.5 * rng.normal(size=300)
    with pytest.raises(TLSIllConditionedError):
        tls_solve(X, y, 0.2, 0.1)
    splits = {
        "X_train": X.astype(np.float32),
        "y_train": y.astype(np.float32),
        "X_test": X[:50].astype(np.float32),
        "y_test": y[:50].astype(np.float32),
    }
    suite = EIVSuite()
    suite.methods = lambda: ["TLS"]  # type: ignore[method-assign]
    (row,) = suite.run("collinear", HarnessConfig(epochs=1), splits=splits)
    assert row["RMSE"] is None and "ill-conditioned" in row["Notes"]
