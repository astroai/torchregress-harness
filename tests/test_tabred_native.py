"""TabReD native layout (official downloader) is read with its time-based split."""

from pathlib import Path

import numpy as np
import pytest

from harness.datasets import _load_tabred


def _write(base: Path, upper: bool) -> None:
    rng = np.random.default_rng(0)
    n = 500
    base.mkdir(parents=True)
    num = rng.normal(size=(n, 4))
    num[::7, 1] = np.nan
    names = ("X_num", "X_bin", "X_cat", "Y") if upper else ("x_num", "x_bin", "x_cat", "y")
    np.save(base / f"{names[0]}.npy", num.astype(np.float32))
    np.save(base / f"{names[1]}.npy", rng.integers(0, 2, (n, 2)).astype(np.float32))
    np.save(base / f"{names[2]}.npy", rng.integers(0, 5, (n, 1)).astype(np.int64))
    np.save(base / f"{names[3]}.npy", (num[:, 0] + rng.normal(size=n)).astype(np.float32))
    idx = np.arange(n)
    parts = {"train": idx[:300], "val": idx[300:400], "test": idx[400:]}
    for k, v in parts.items():
        if upper:
            (base / "split-default").mkdir(exist_ok=True)
            np.save(base / "split-default" / f"{k}_idx.npy", v)
        else:
            (base / "splits" / "default").mkdir(parents=True, exist_ok=True)
            np.save(base / "splits" / "default" / f"{k}.npy", v)


@pytest.mark.parametrize("upper", [False, True])
def test_native_tabred_time_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upper: bool
) -> None:
    _write(tmp_path / "weather", upper)
    monkeypatch.setenv("TORCHREGRESS_HARNESS_TABRED_ROOT", str(tmp_path))
    s = _load_tabred("weather", n_rows=500, n_features=7, n_samples=10_000, data_dir="nowhere")
    assert "_synthetic_fallback" not in s
    assert s["X_train"].shape == (400, 7) and s["X_test"].shape == (100, 7)  # train+val / test
    assert np.isfinite(s["X_train"]).all() and np.isfinite(s["X_test"]).all()
    s2 = _load_tabred("weather", n_rows=500, n_features=7, n_samples=200, data_dir="nowhere")
    assert len(s2["y_train"]) == 200 and len(s2["y_test"]) == 60
