"""The n(z) loader reads its test set from the reserved WFD file only.

The public and reserved WFD files share names (the reserved archive unpacks to
``<root>/reserved``) but not object ids or photometry, so the loader must never
pick a file by ``glob`` order or try to join the two.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

from harness import datasets


def _write(path: pathlib.Path, n: int, *, z_offset: float, oid_start: int, with_z: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(oid_start)
    with h5py.File(path, "w") as f:
        for b, s in datasets.NZ_BANDS:
            f[f"mag_{b}_{s}"] = rng.normal(24.0, 1.0, n)
            f[f"mag_{b}_{s}_err"] = rng.uniform(0.01, 0.1, n)
        f["object_id"] = np.arange(oid_start, oid_start + n, dtype=np.int64)
        if with_z:
            f["redshift"] = z_offset + np.linspace(0.1, 1.0, n)


@pytest.fixture
def nz_root(tmp_path, monkeypatch):
    stem = "nz_challenge_taskset_1_flagship_4yr"
    _write(tmp_path / "public" / f"{stem}_ddf_0.hdf5", 50, z_offset=0.0, oid_start=1, with_z=True)
    _write(
        tmp_path / "public" / f"{stem}_wfd.hdf5",
        40,
        z_offset=0.0,
        oid_start=3_000_000,
        with_z=False,
    )
    # Same file names in reserved/: a DDF copy that must not be read twice, and
    # the WFD truth file with its own ids and photometry.
    _write(tmp_path / "reserved" / f"{stem}_ddf_0.hdf5", 50, z_offset=0.0, oid_start=1, with_z=True)
    _write(
        tmp_path / "reserved" / f"{stem}_wfd.hdf5",
        30,
        z_offset=2.0,
        oid_start=7_000_000,
        with_z=True,
    )
    monkeypatch.setenv(datasets.NZ_CHALLENGE_ENV, str(tmp_path))
    # Archives are "present": nothing is downloaded or extracted.
    monkeypatch.setattr(datasets, "_ensure_nz_archive", lambda root, taskset: root)
    monkeypatch.setattr(datasets, "_ensure_nz_reserved", lambda root: root)
    return tmp_path


def test_test_set_is_the_reserved_wfd_file(nz_root):
    s = datasets._load_nz_challenge(taskset=1, simulation="flagship", scenario="4yr")
    stem = "nz_challenge_taskset_1_flagship_4yr_wfd.hdf5"
    with h5py.File(nz_root / "reserved" / stem, "r") as f:
        z = np.asarray(f["redshift"][:], dtype=np.float32)
        mags = np.stack([f[f"mag_{b}_{s}"][:] for b, s in datasets.NZ_BANDS], axis=1)
    assert s["X_test"].shape == (30, 2 * len(datasets.NZ_BANDS))
    np.testing.assert_array_equal(s["y_test"], z)
    np.testing.assert_allclose(s["X_test"][:, : len(datasets.NZ_BANDS)], mags.astype(np.float32))


def test_training_reads_public_ddf_once(nz_root):
    s = datasets._load_nz_challenge(taskset=1, simulation="flagship", scenario="4yr")
    assert len(s["y_train"]) == 50


def test_public_file_lookup_ignores_reserved_copies(nz_root):
    found = datasets._nz_public_files(nz_root, "nz_challenge_taskset_1_*.hdf5")
    assert found and all("reserved" not in p.parts for p in found)
    assert len(found) == 2
