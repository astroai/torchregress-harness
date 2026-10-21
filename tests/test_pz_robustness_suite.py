"""Tests for PZ challenge removal/OOD robustness suites."""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular.pz_robustness import PZOODSuite, PZRemovalSuite


def _write_fixture(path, n=160, seed=0):
    h5py = pytest.importorskip("h5py")
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
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        for i, (band, survey) in enumerate(bands):
            f[f"mag_{band}_{survey}"] = rng.normal(20 + i, 1, n)
            f[f"mag_{band}_{survey}_err"] = rng.uniform(0.02, 0.2, n)
        f["redshift"] = rng.uniform(0.01, 2.5, n)
        f["ra"] = rng.uniform(0, 10, n)
        f["dec"] = rng.uniform(-2, 2, n)


def test_metadata() -> None:
    assert PZRemovalSuite().name == "pz_removal"
    assert PZOODSuite().name == "pz_ood"
    assert "SplitCP_MLP" in PZRemovalSuite().methods()


def test_removal_and_ood_smoke(tmp_path, monkeypatch) -> None:
    pytest.importorskip("tabm")
    from harness.datasets import load_dataset
    from harness.runners import HarnessConfig

    names = [
        "pz_challenge_taskset_1_cardinal_training_10yr.hdf5",
        "pz_challenge_taskset_2_cardinal_training_10yr.hdf5",
        "pz_challenge_taskset_3_cardinal_training_10yr.hdf5",
        "pz_challenge_taskset_4_cardinal_training_10yr.hdf5",
    ]
    for i, name in enumerate(names):
        _write_fixture(tmp_path / name, seed=i)
    monkeypatch.setenv("TORCHREGRESS_PZ_CHALLENGE_DIR", str(tmp_path))
    cfg = HarnessConfig(seed=0, epochs=1, smoke=True)
    splits = load_dataset(
        "pz_challenge_cardinal_10yr", n_samples=120, seed=0, standardize_targets=False
    )
    rows = PZRemovalSuite().run("pz_challenge_cardinal_10yr", cfg, splits)
    assert len(rows) == 20
    assert any(r["Scenario"] == "omit_field" for r in rows)
    rows2 = PZOODSuite().run("pz_challenge_cardinal_10yr", cfg)
    assert len(rows2) == 20
    assert {r["Scenario"] for r in rows2} == {"ID_task1", "OOD_task2", "OOD_task3", "OOD_task4"}
