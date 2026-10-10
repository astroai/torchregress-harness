"""fm_uq_scenarios / fm_multitarget: grid maths, scenarios and CPU smoke runs."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pytest
from suites.tabular import _fm_grid as G
from suites.tabular import fm_multitarget as MT
from suites.tabular import fm_uq_scenarios as M

from harness.runners import HarnessConfig


def _data(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * rng.normal(size=n)
    return X, (y - y.mean()) / y.std()


def _gauss_grid(n: int, mu: float = 0.0, sd: float = 1.0) -> np.ndarray:
    z = np.array([NormalDist().inv_cdf(float(t)) for t in G.LEVELS])
    return np.tile(mu + sd * z, (n, 1))


# --- grid maths -----------------------------------------------------------------


def test_grid_quantile_cdf_roundtrip_and_crps() -> None:
    q = _gauss_grid(5)
    np.testing.assert_allclose(G.quantile_at(q, 0.5), 0.0, atol=1e-9)
    np.testing.assert_allclose(G.cdf_at(q, G.quantile_at(q, 0.3)), 0.3, atol=1e-6)
    # CRPS of N(0, 1) at y = 0 is (sqrt(2) - 1) / sqrt(pi) ~ 0.2337
    assert G.crps(q[:1], np.zeros(1))[0] == pytest.approx(0.2337, abs=5e-3)


def test_pit_recalibration_fixes_overconfidence() -> None:
    rng = np.random.default_rng(0)
    y_cal, y_test = rng.normal(0, 2, 2000), rng.normal(0, 2, 2000)
    q = _gauss_grid(2000, sd=1.0)  # half the true spread
    lo, hi = G.interval(q, 0.1)
    assert np.mean((y_test >= lo) & (y_test <= hi)) < 0.6
    qr = G.pit_recalibrate(q, y_cal, q)
    lo, hi = G.interval(qr, 0.1)
    assert 0.86 < np.mean((y_test >= lo) & (y_test <= hi)) < 0.94
    assert G.pit_ks(G.cdf_at(qr, y_test)) < G.pit_ks(G.cdf_at(q, y_test))


# --- scenarios ------------------------------------------------------------------


def test_outliers_only_in_train_and_cal() -> None:
    X, y = _data()
    clean, dirty = M.scenario_splits(X, y, "clean", 0), M.scenario_splits(X, y, "outliers", 0)
    np.testing.assert_array_equal(clean["y_test"], dirty["y_test"])
    for k in ("y_train", "y_cal"):
        assert 0.03 < np.mean(dirty[k] != clean[k]) < 0.07


def test_extrapolate_noise_imbalance_heavy_tail() -> None:
    X, y = _data(3000)
    ex = M.scenario_splits(X, y, "extrapolate", 0)
    # every test row lies beyond every train row along PC1
    proj = lambda A: (A - X.mean(0)) @ np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]  # noqa: E731
    assert len(ex["y_test"]) == int(0.2 * len(X))
    t, r = proj(ex["X_test"]), proj(ex["X_train"])
    assert (t.min() >= r.max()) or (t.max() <= r.min())

    clean, nt = M.scenario_splits(X, y, "clean", 0), M.scenario_splits(X, y, "noise_test", 0)
    np.testing.assert_array_equal(clean["X_train"], nt["X_train"])
    assert not np.allclose(clean["X_test"], nt["X_test"])
    na = M.scenario_splits(X, y, "noise_all", 0)
    assert not np.allclose(clean["X_train"], na["X_train"])

    im = M.scenario_splits(X, y, "imbalance", 0)
    thr = np.quantile(y, 0.8)
    assert np.mean(im["y_train"] > thr) < 0.05 and np.mean(im["y_test"] > thr) > 0.1

    ht = M.scenario_splits(X, y, "heavy_tail", 0)
    kurt = lambda a: np.mean(((a - a.mean()) / a.std()) ** 4)  # noqa: E731
    assert kurt(ht["y_train"]) > kurt(clean["y_train"]) + 1


def test_shift_weights_track_the_shift() -> None:
    X, y = _data(3000)
    s = M.scenario_splits(X, y, "shift", 0)
    w_cal, w_test = M.density_ratio(s["X_cal"], s["X_test"], 0)
    assert np.median(w_test) > np.median(w_cal)


# --- smoke runs (reference bases only: no foundation-model weights on CI) --------


@pytest.mark.parametrize("scenario", ["clean", "shift", "extrapolate"])
def test_scenarios_smoke(monkeypatch: pytest.MonkeyPatch, scenario: str) -> None:
    monkeypatch.setattr(M.FMUQScenariosSuite, "bases", lambda self: [G.gbm_base_name()])
    X, y = _data(800)
    splits = {"X_train": X[:600], "y_train": y[:600], "X_test": X[600:], "y_test": y[600:]}
    rows = M.FMUQScenariosSuite().run(
        "toy", HarnessConfig(smoke=True, seed=0), splits=splits, scenario_name=scenario
    )
    n_expected = 2 + 5 + (2 if scenario == "shift" else 0)
    assert len(rows) == n_expected
    for r in rows:
        assert not str(r.get("Notes", "")).startswith("failed"), r["Notes"]
        assert 0.0 <= r["Coverage90"] <= 1.0 and r["Coverage50"] <= r["Coverage95"]
        dist = r["Method"].split("/")[1].split("+")[0] in M.DIST_METHODS
        assert (r["CRPS"] is not None) == dist


def test_multitarget_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MT.FMMultiTargetSuite, "bases", lambda self: [G.gbm_base_name()])
    monkeypatch.setattr(MT, "DENSITY_REFS", ())
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3)).astype(np.float32)
    e = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=500)
    Y = np.c_[X[:, 0] + 0.5 * e[:, 0], X[:, 1] + 0.5 * e[:, 1]]
    splits = {"X_train": X[:400], "y_train": Y[:400], "X_test": X[400:], "y_test": Y[400:]}
    rows = MT.FMMultiTargetSuite().run("toy", HarnessConfig(smoke=True, seed=0), splits=splits)
    assert len(rows) == len(MT.JOINT_METHODS)
    by = {r["Method"].split("/")[1].split("+")[0]: r for r in rows}
    for r in rows:
        assert not str(r.get("Notes", "")).startswith("failed"), r["Notes"]
        assert r["ES"] > 0 and 0 <= r["Cov90_ell"] <= 1
    # residual correlation 0.8: the copula recovers it, independence cannot
    assert by["GaussCopula"]["CorrErr"] < by["Indep"]["CorrErr"]


def test_copula_sampler_matches_correlation() -> None:
    q = [_gauss_grid(50), _gauss_grid(50)]
    R = np.array([[1.0, 0.7], [0.7, 1.0]])
    S = MT.sample_joint(q, 400, np.random.default_rng(0), R)
    r = np.mean([np.corrcoef(S[i].T)[0, 1] for i in range(50)])
    assert r == pytest.approx(0.7, abs=0.05)
