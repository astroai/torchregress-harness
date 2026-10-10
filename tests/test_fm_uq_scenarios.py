"""fm_uq_scenarios: scenario construction, density-ratio weights and a CPU smoke run."""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular import fm_uq_scenarios as M

from harness.runners import HarnessConfig


def _data(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * rng.normal(size=n)
    return X, (y - y.mean()) / y.std()


def test_outliers_only_in_train_and_cal() -> None:
    X, y = _data()
    clean, dirty = M._scenario_splits(X, y, "clean", 0), M._scenario_splits(X, y, "outliers", 0)
    np.testing.assert_array_equal(clean["y_test"], dirty["y_test"])
    for k in ("y_train", "y_cal"):
        frac = np.mean(np.abs(dirty[k] - clean[k]) > 0)
        assert 0.03 < frac < 0.07
    np.testing.assert_array_equal(dirty["y_train_clean"], clean["y_train"])


def test_shift_moves_test_along_pc1_and_weights_track_it() -> None:
    X, y = _data(3000)
    s = M._scenario_splits(X, y, "shift", 0)
    sizes = [len(s[k]) for k in ("X_train", "X_cal", "X_test")]
    assert sum(sizes) == len(X) and sizes[2] == len(X) // 5
    w_cal, w_test = M._density_ratio(s["X_cal"], s["X_test"], 0)
    assert w_cal.shape == (len(s["X_cal"]),) and np.all(w_test > 0)
    assert np.median(w_test) > np.median(w_cal)  # test rows look more "test-like"


def test_smoke_run_reference_bases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(M.FMUQScenariosSuite, "bases", lambda self: [M._GBM_BASE])
    X, y = _data(800)
    splits = {"X_train": X[:600], "y_train": y[:600], "X_test": X[600:], "y_test": y[600:]}
    rows = M.FMUQScenariosSuite().run("toy", HarnessConfig(smoke=True, seed=0), splits=splits)
    assert len(rows) == 5 + 5 + 7
    ok = [r for r in rows if not str(r.get("Notes", "")).startswith("failed")]
    assert len(ok) == len(rows)
    for r in ok:
        if r["Method"].split("/")[1].startswith("tr_"):
            assert 0.75 <= r["Coverage"] <= 1.0 and np.isfinite(r["Width"])
