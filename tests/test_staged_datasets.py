"""Contracts for staged real-image corpora (IXI / Udacity / KITTI).

Readers must resolve the exact staged layouts (Windows log paths,
single-column Eigen manifests, .nii-vs-.nii.gz spelling) and refuse to
silently substitute synthetic proxies when an explicit root is set.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from harness import datasets as _datasets


def test_resolve_udacity_frame_handles_windows_paths(tmp_path: Path) -> None:
    img = tmp_path / "IMG"
    img.mkdir()
    frame = img / "center_2022_04_10_12_44_27_913.jpg"
    frame.write_bytes(b"fake")
    resolved = _datasets._resolve_udacity_frame(
        tmp_path, r"C:\Users\Andy\Desktop\x\IMG\center_2022_04_10_12_44_27_913.jpg"
    )
    assert resolved == frame


def test_udacity_headerless_driving_log(tmp_path: Path) -> None:
    img = tmp_path / "IMG"
    img.mkdir()
    (img / "center_a.jpg").write_bytes(b"fake")
    (tmp_path / "driving_log.csv").write_text(
        "C:\\x\\IMG\\center_a.jpg,C:\\x\\IMG\\left_a.jpg,C:\\x\\IMG\\right_a.jpg,0.25,1,0,25.2\n"
    )
    pairs = _datasets._udacity_base_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][0].name == "center_a.jpg"
    assert pairs[0][1] == pytest.approx(0.25)


def test_kitti_single_column_gt_manifest(tmp_path: Path) -> None:
    rgb = tmp_path / "2011_09_26" / "d_sync" / "image_02" / "data"
    depth = tmp_path / "2011_09_26" / "d_sync" / "proj_depth" / "groundtruth" / "image_02"
    rgb.mkdir(parents=True)
    depth.mkdir(parents=True)
    (rgb / "0000000001.jpg").write_bytes(b"fake")
    (depth / "0000000001.png").write_bytes(b"fake")
    (tmp_path / "gt_locations.txt").write_text(
        "2011_09_26/d_sync/proj_depth/groundtruth/image_02/data/0000000001.jpg\n"
    )
    train, _ = _datasets._read_kitti_eigen_pairs(tmp_path)
    assert len(train) == 1
    assert train[0][0].name == "0000000001.jpg"
    assert train[0][1].name == "0000000001.png"


def test_ixi_subjects_filters_sentinel_ages(tmp_path: Path) -> None:
    (tmp_path / "subjects.csv").write_text(
        "subject_id,age,registered_mni\n"
        "IXI001,35.8,sub-IXI001/anat/vol.nii\n"
        "IXI002,-1.0,sub-IXI002/anat/vol.nii\n"
    )
    (tmp_path / "sub-IXI001" / "anat").mkdir(parents=True)
    (tmp_path / "sub-IXI001" / "anat" / "vol.nii").write_bytes(b"fake")
    (tmp_path / "sub-IXI002" / "anat").mkdir(parents=True)
    (tmp_path / "sub-IXI002" / "anat" / "vol.nii").write_bytes(b"fake")
    pairs = _datasets._read_ixi_staged_subjects(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][1] == pytest.approx(35.8)


def test_ucf_fallback_is_pure_synthetic(tmp_path: Path, monkeypatch) -> None:
    """UCF-QNRF fallback must be the seeded generator — never another corpus.

    Regression test: the old fallback routed through the Shanghai loader,
    whose real-data path silently substituted real ShanghaiTech data while
    the artifact claimed a synthetic proxy (bit-identical metrics across
    supposedly different datasets).
    """
    import numpy as np

    monkeypatch.setenv("TORCHREGRESS_HARNESS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("TORCHREGRESS_HARNESS_STAGED_ROOT", str(tmp_path / "arc"))
    monkeypatch.delenv("QNRF_ROOT", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    splits = _datasets._load_ucf_qnrf(n_samples=20, img_size=32)
    assert splits.get("_synthetic_fallback") is True
    expected = _datasets._synthetic_crowd_splits(20, 32, 0.2, 42)
    assert np.array_equal(np.asarray(splits["X_train"]), np.asarray(expected["X_train"]))
    assert np.array_equal(np.asarray(splits["y_test"]), np.asarray(expected["y_test"]))


def _parquet_engine_available() -> bool:
    for name in ("pyarrow", "fastparquet"):
        try:
            importlib.import_module(name)
            return True
        except ImportError:
            continue
    return False


def test_tlc_fare_real_data_contract() -> None:
    """NYC TLC loader reads the staged parquet with raw skewed targets."""
    import numpy as np

    from harness.datasets import _load_tlc_fare

    if not _parquet_engine_available():
        pytest.skip("no parquet engine (pyarrow/fastparquet) installed")

    splits = _load_tlc_fare(n_samples=5000)
    assert splits.get("_synthetic_fallback", False) is False
    assert splits["X_train"].shape[1] == 8
    y = np.concatenate([np.asarray(splits["y_train"]), np.asarray(splits["y_test"])]).astype(
        np.float64
    )
    # Right-skewed fares on raw scale (no y standardization for likelihoods).
    assert np.median(y) < y.mean()
    assert float(y.min()) > 0.0
    assert splits.get("_skip_y_standardization", False) is True


def test_tlc_explicit_root_refuses_fallback(tmp_path: Path, monkeypatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("TORCHREGRESS_HARNESS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("TORCHREGRESS_HARNESS_STAGED_ROOT", str(tmp_path / "arc"))
    monkeypatch.setenv("TLC_ROOT", str(empty))

    def _boom(*args: object, **kwargs: object) -> None:
        raise ConnectionError("offline")

    monkeypatch.setattr(_datasets, "_download_with_progress", _boom)
    with pytest.raises(RuntimeError):
        _datasets._load_tlc_fare(n_samples=8)


def test_folktables_all_registered() -> None:
    from harness.datasets import _US_STATES, DATASET_REGISTRY

    assert "folktables_all" in DATASET_REGISTRY
    assert "tlc_fare" in DATASET_REGISTRY
    assert len(_US_STATES) == 51


def test_explicit_roots_refuse_synthetic_fallback(tmp_path: Path, monkeypatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("TORCHREGRESS_HARNESS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("TORCHREGRESS_HARNESS_STAGED_ROOT", str(tmp_path / "arc"))
    monkeypatch.setenv("IXI_ROOT", str(empty))
    monkeypatch.setenv("UDACITY_ROOT", str(empty))
    monkeypatch.setenv("KITTI_ROOT", str(empty))
    with pytest.raises(RuntimeError):
        _datasets._load_ixi_brainage(n_samples=8, img_size=8)
    with pytest.raises(RuntimeError):
        _datasets._load_udacity_steering(n_samples=8, img_size=8)
    with pytest.raises(RuntimeError):
        _datasets._load_kitti_depth(n_samples=8, img_size=8)
