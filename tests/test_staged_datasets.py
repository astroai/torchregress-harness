"""Contracts for staged real-image corpora (IXI / Udacity / KITTI).

Readers must resolve the exact staged layouts (Windows log paths,
single-column Eigen manifests, .nii-vs-.nii.gz spelling) and refuse to
silently substitute synthetic proxies when an explicit root is set.
"""

from __future__ import annotations

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
