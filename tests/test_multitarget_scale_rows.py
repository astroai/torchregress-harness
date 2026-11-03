"""Multi-target density / conformal rows and the >= 1M-row synthetic scale dataset.

* ``synthetic_multitarget_big``: registry metadata, deterministic on-the-fly draw.
* ``density_parity/torchregress_CopulaSpline``: ``GaussianCopulaSplineHead`` row and
  its missing-coordinate marginalization check; large-data schedule.
* ``conformal_density``: ``ContrastiveFlowConformal`` / ``WaldoDensityConformal`` rows.
* ``multivariate_intervals/MultivariateGaussian+WALDO``: ``WaldoConformal`` +
  ``waldo_statistic`` parity.
* ``sbi_parity``: ``WaldoPosteriorCalibrator`` diagnostics from posterior draws.
"""

from __future__ import annotations

import numpy as np
import pytest
import suites.tabular.density_parity as density_parity
from suites.tabular.conformal_density import ConformalDensitySuite
from suites.tabular.density_parity import DensityParitySuite
from suites.tabular.multivariate_intervals import MultivariateIntervalsSuite
from suites.tabular.sbi_parity import _waldo_posterior_diagnostics
from tools.release_jobs import jobs

from harness.datasets import DATASET_REGISTRY, load_dataset
from harness.runners import HarnessConfig

BIG = "synthetic_multitarget_big"
SMOKE = HarnessConfig(seed=0, smoke=True, batch_size=128)


@pytest.fixture(scope="module")
def small_big_draw() -> dict:
    return load_dataset(BIG, n_samples=3000)


def test_big_dataset_metadata() -> None:
    info = DATASET_REGISTRY[BIG]
    assert info.category == "synthetic"
    assert info.n_samples >= 1_000_000
    assert info.n_features == 20


def test_big_dataset_draw_is_deterministic(small_big_draw: dict) -> None:
    again = load_dataset(BIG, n_samples=3000)
    for key in ("X_train", "y_train", "X_test", "y_test"):
        np.testing.assert_array_equal(small_big_draw[key], again[key])
    assert small_big_draw["X_train"].shape == (2400, 20)
    assert small_big_draw["y_train"].shape == (2400, 6)
    assert small_big_draw["y_test"].shape == (600, 6)
    assert np.isfinite(small_big_draw["y_train"]).all()
    assert "_y_scaler" not in small_big_draw  # raw [N, D] targets


def test_big_jobs_run_in_the_gpu_group() -> None:
    big = set(jobs("tabular_big"))
    assert {("density_parity", BIG), ("conformal_density", BIG)} <= big
    assert not {d for _, d in jobs("tabular")} & {BIG}


def test_copula_spline_row_and_marginalization(
    small_big_draw: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TORCHREGRESS_DENSITY_METHODS", "torchregress_CopulaSpline")
    (row,) = DensityParitySuite().run(BIG, SMOKE, splits=small_big_draw)
    assert not str(row["Notes"]).startswith("failed"), row["Notes"]
    assert row["NTargets"] == 6
    for key in ("NLL", "ES", "CRPS", "MargNLL"):
        assert np.isfinite(row[key]), key
    # Exact marginalization: masked log-density equals the quadrature marginal.
    assert row["MargQuadErr"] < 1e-3
    assert "PeakGPUMemMB" in row and "train_s" in row


def test_big_schedule_skips_full_data_trainers(
    small_big_draw: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TORCHREGRESS_DENSITY_METHODS", "pzflow_NSF,torchregress_MAF")
    monkeypatch.setattr(density_parity, "is_big", lambda n: True)
    rows = {
        r["Method"].split("/", 1)[1]: r
        for r in DensityParitySuite().run(BIG, SMOKE, splits=small_big_draw)
    }
    assert rows["pzflow_NSF"]["Notes"].startswith("skipped: pzflow trains with its own")
    assert rows["pzflow_NSF"]["NLL"] is None
    assert not str(rows["torchregress_MAF"]["Notes"]).startswith("failed")
    assert np.isfinite(rows["torchregress_MAF"]["NLL"])


@pytest.mark.parametrize("dataset", ["cd_multi", BIG])
def test_conformal_density_rows(dataset: str, small_big_draw: dict) -> None:
    splits = small_big_draw if dataset == BIG else None
    rows = {
        r["Method"].split("/", 1)[1]: r
        for r in ConformalDensitySuite().run(dataset, SMOKE, splits=splits)
    }
    assert set(rows) == set(ConformalDensitySuite().methods())
    for name, row in rows.items():
        assert not str(row["Notes"]).startswith("failed"), (name, row["Notes"])
        assert 0.0 <= row["Coverage90"] <= 1.0
        assert row["CoverageKind"] == ("marginal" if "_" not in name else "joint")
    for name in ("ContrastiveFlow_CTI", "ContrastiveFlow_CTI_Mondrian"):
        assert rows[name]["CPParityGap"] == 0.0
    assert np.isfinite(rows["NSF_WALDO"]["WaldoThreshold"])


def test_multivariate_waldo_row(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = MultivariateIntervalsSuite()
    monkeypatch.setattr(suite, "methods", lambda: ["MultivariateGaussian+WALDO"])
    (row,) = suite.run("synthetic_mv_gaussian", HarnessConfig(seed=0, smoke=True))
    assert not str(row["Notes"]).startswith("failed"), row["Notes"]
    assert row["WaldoStatParityErr"] < 1e-4
    assert 0.0 <= row["RegionCoverage"] <= 1.0
    assert np.isfinite(row["RegionLogVol"])
    assert row["WaldoThr"] > 0 and row["MVScoreThr"] > 0


def test_waldo_posterior_diagnostics() -> None:
    rng = np.random.default_rng(0)
    n, s, d = 200, 400, 3
    m = rng.normal(size=(n, d))
    theta = m + rng.normal(size=(n, d))
    calibrated = m[None] + rng.normal(size=(s, n, d))
    over = m[None] + 0.4 * rng.normal(size=(s, n, d))
    good = _waldo_posterior_diagnostics(theta, calibrated)
    bad = _waldo_posterior_diagnostics(theta, over)
    assert set(good) == {"WALDO_Cov90", "WALDO_Thr", "Wald_Cov90_chi2"}
    # Calibration repairs an overconfident posterior; the raw chi2 region does not.
    assert bad["Wald_Cov90_chi2"] < 0.5
    assert bad["WALDO_Cov90"] > 0.8
    assert abs(good["Wald_Cov90_chi2"] - 0.9) < 0.1
    with pytest.raises(ValueError, match="needs >= 4 SBC cases"):
        _waldo_posterior_diagnostics(theta[:3], calibrated[:, :3])
