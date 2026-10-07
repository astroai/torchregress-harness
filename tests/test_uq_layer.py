"""Phase 2b: ``uq_layer`` suite and the recalibrated-ensemble recipe rows (smoke)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from suites.tabular import uq_layer
from suites.tabular.bayesian_linear import BayesianLinearSuite
from suites.tabular.uq import UQSuite
from suites.tabular.uq_layer import UQLayerSuite

from harness.runners import HarnessConfig

CFG = HarnessConfig(seed=0, epochs=2, batch_size=64, smoke=True, standardize_targets=True)


def _by_method(rows: list[dict]) -> dict[str, dict]:
    return {r["Method"].split("/", 1)[1]: r for r in rows}


@pytest.fixture(scope="module")
def rows() -> dict[str, dict]:
    return _by_method(UQLayerSuite().run("synthetic_hetero", CFG))


def test_registered_and_scheduled() -> None:
    from tools._suite_registry import ALL_SUITES
    from tools.release_jobs import jobs

    assert ALL_SUITES["uq_layer"] is UQLayerSuite
    conformal = set(ALL_SUITES["conformal"]().supported_datasets)
    assert set(UQLayerSuite().supported_datasets) == conformal
    assert {d for s, d in jobs("tabular") if s == "uq_layer"} == conformal


def test_method_grid() -> None:
    suite = UQLayerSuite()
    gbm = uq_layer._GBM_BASE
    assert "tr_JackknifePlus" in suite.base_methods(gbm)
    assert "tr_CVPlus" in suite.base_methods("RF")
    assert "tr_JackknifePlus" not in suite.base_methods("RF")
    assert "tr_CVPlus" not in suite.base_methods("DeepEnsK5")
    # crepes would take |CQR score|: no crepes CQR row.
    assert not any("crepes_CQR" in m for m in suite.methods())
    assert len(set(suite.methods())) == len(suite.methods())


def test_same_base_same_numbers(rows: dict[str, dict]) -> None:
    """torchregress, MAPIE and crepes see identical base predictions: split rows agree."""
    gbm = uq_layer._GBM_BASE
    for base in (gbm, "RF", "DeepEnsK5"):
        tr, mp, cr = (rows[f"{m}+{base}"] for m in ("tr_Split", "MAPIE_Split", "crepes_Split"))
        assert tr["RMSE"] == mp["RMSE"] == cr["RMSE"]
        for key in ("Coverage", "Width", "IntervalScore"):
            assert tr[key] == pytest.approx(mp[key], rel=1e-6, abs=1e-9), (base, key)
            assert tr[key] == pytest.approx(cr[key], rel=1e-6, abs=1e-9), (base, key)
        assert rows[f"tr_CQR+{base}"]["Width"] == pytest.approx(
            rows[f"MAPIE_CQR+{base}"]["Width"], rel=1e-6
        )
        assert rows[f"tr_Normalized+{base}"]["Width"] == pytest.approx(
            rows[f"crepes_Normalized+{base}"]["Width"], rel=1e-6
        )


def test_cv_plus_matches_mapie(rows: dict[str, dict]) -> None:
    for base in (uq_layer._GBM_BASE, "RF"):
        a, b = rows[f"tr_CVPlus+{base}"], rows[f"MAPIE_CVPlus+{base}"]
        assert a["Width"] == pytest.approx(b["Width"], rel=1e-6)
        assert a["Coverage"] == b["Coverage"]


def test_distribution_metrics_only_where_a_distribution_exists(rows: dict[str, dict]) -> None:
    for m in ("Raw+DeepEnsK5", "tr_VTS+RF", "tr_IsoMean+VTS+DeepEnsK5"):
        assert all(rows[m][k] is not None for k in ("NLL", "CRPS", "ECE")), m
    assert rows["tr_Split+RF"].get("NLL") is None
    for r in rows.values():
        if r["Coverage"] is not None:
            assert "InfiniteIntervalFrac" in r


def test_tabicl_download_failure_is_a_failed_row(rows: dict[str, dict]) -> None:
    if not uq_layer._TABICL_AVAILABLE:
        pytest.skip("tabicl not installed")
    tab = [r for m, r in rows.items() if m.endswith("+TabICL")]
    assert tab
    # No network in CI: every TabICL row is a failed row, never a dropped dataset.
    if all(str(r["Notes"]).startswith("failed: base TabICL") for r in tab):
        assert all(r["Coverage"] is None for r in tab)
    assert any(m.endswith("+DeepEnsK5") for m in rows)


def test_tabicl_quantile_path_with_fake_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, X: np.ndarray, y: np.ndarray) -> Fake:
            self.mu = float(np.mean(y))
            self.sd = float(np.std(y))
            return self

        def predict(self, X, output_type="mean", alphas=None):  # noqa: ANN001, ANN201
            base = np.full(len(X), self.mu)
            if output_type == "mean":
                return base
            return np.stack([base + self.sd * (a - 0.5) * 4 for a in alphas], axis=1)

    mod = types.ModuleType("tabicl")
    mod.TabICLRegressor = Fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tabicl", mod)
    monkeypatch.setattr(uq_layer, "_TABICL_AVAILABLE", True)
    out = _by_method(UQLayerSuite().run("synthetic_hetero", CFG))
    for uq in ("tr_Split", "MAPIE_CQR", "tr_CQR", "crepes_Normalized"):
        r = out[f"{uq}+TabICL"]
        assert r["Coverage"] is not None, r["Notes"]


def test_jackknife_skipped_beyond_size_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uq_layer, "_JACKKNIFE_MAX_N", 10)
    out = _by_method(UQLayerSuite().run("synthetic_hetero", CFG))
    r = out[f"tr_JackknifePlus+{uq_layer._GBM_BASE}"]
    assert str(r["Notes"]).startswith("skipped:") and r["Coverage"] is None


def test_cached_regressor_rejects_unseen_features() -> None:
    reg = uq_layer._CachedRegressor({uq_layer._key(np.zeros((2, 2))): np.zeros(2)})
    assert reg.predict(np.zeros((2, 2))).shape == (2,)
    with pytest.raises(KeyError):
        reg.predict(np.ones((2, 2)))


# --- A3 recipe rows -------------------------------------------------------


def test_uq_recipe_rows() -> None:
    suite = UQSuite()
    out = _by_method(suite.run("synthetic_hetero", CFG))
    base = out["DeepEnsemble_K5"]
    for m in ("DeepEnsemble_K5+VTS", "DeepEnsemble_K5+CQR"):
        r = out[m]
        assert r["RMSE"] == pytest.approx(
            base["RMSE"], rel=1e-5
        )  # same ensemble, only the UQ layer differs
        assert r["ECE"] is not None and r["NLL"] is not None
    assert out["DeepEnsemble_K5+CQR"]["Coverage"] is not None
    assert (
        out["DeepEnsemble_K5+VTS"]["ECE"] != base["ECE"]
        or out["DeepEnsemble_K5+VTS"]["NLL"] != base["NLL"]
    )


def test_blr_vts_row() -> None:
    suite = BayesianLinearSuite()
    out = _by_method(suite.run("synthetic_hetero", CFG))
    r = out["BLR+VTS"]
    assert r["ECE"] is not None and r["NLL"] is not None
    assert out["BayesianLinearHead"]["ECE"] is not None
