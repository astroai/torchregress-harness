"""Tests for the LSST DESC PZ Data Challenge suite."""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular.pz_challenge import PZChallengeSuite, _build_features


def test_metadata() -> None:
    suite = PZChallengeSuite()
    assert suite.name == "pz_challenge"
    expected = {
        "MSE_MLP", "GaussianNLL_MLP", "MDN_K3", "Quantile_MLP",
        "NormalizingFlow", "TabM_MSE", "TabM_GaussianNLL", "TabM_SplitCP",
        "SplitCP_MLP", "CQR_MLP", "MondrianCP_MLP", "Evidential",
        "BetaNLL_MLP", "InputNoiseAug_MLP", "DensityWeighted_MLP", "BMC_MLP",
    }
    assert expected <= set(suite.methods())
    assert "pz_challenge_cardinal_10yr" in suite.supported_datasets
    assert suite.library_version("TabM_MSE")


def test_feature_builder_shape_and_finiteness() -> None:
    raw = np.full((5, 18), np.nan)
    raw[:, :9] = 20.0
    raw[:, 9:] = 0.1
    raw[0, 0] = np.nan
    raw[1, 10] = np.nan
    feats = _build_features(raw)
    assert feats.shape == (5, 37)
    assert np.isfinite(feats).all()
    # Missing u detection must not leak a finite u-g colour (colours start at 27).
    assert feats[0, 27] == 0.0


def test_loader_and_smoke(tmp_path, monkeypatch) -> None:
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("tabm")
    from harness.datasets import load_dataset
    from harness.runners import HarnessConfig

    bands = [
        ("u", "lsst"),
        ("g", "lsst"),
        ("r", "lsst"),
        ("i", "lsst"),
        ("z", "lsst"),
        ("y", "lsst"),
        ("Y", "roman"),
        ("J", "roman"),
        ("H", "roman"),
    ]
    path = tmp_path / "pz_challenge_taskset_1_cardinal_training_10yr.hdf5"
    rng = np.random.default_rng(0)
    n = 180
    with h5py.File(path, "w") as f:
        for i, (band, survey) in enumerate(bands):
            f[f"mag_{band}_{survey}"] = rng.normal(20 + i, 1, n)
            f[f"mag_{band}_{survey}_err"] = rng.uniform(0.02, 0.2, n)
        f["redshift"] = rng.uniform(0.01, 2.5, n)
    monkeypatch.setenv("TORCHREGRESS_PZ_CHALLENGE_DIR", str(tmp_path))
    splits = load_dataset("pz_challenge_cardinal_10yr", n_samples=180, seed=0)
    assert splits["X_train"].shape[1] == 18
    rows = PZChallengeSuite().run(
        "pz_challenge_cardinal_10yr",
        config=HarnessConfig(seed=0, epochs=2, smoke=True),
        splits=splits,
    )
    assert len(rows) == len(PZChallengeSuite().methods())
    for row in rows:
        assert "Notes" in row
        if "failed" not in str(row.get("Notes", "")):
            assert row["RMSE"] is not None and np.isfinite(row["RMSE"])
