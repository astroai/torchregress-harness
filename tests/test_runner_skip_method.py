"""HarnessRunner.run reports SkipMethod as a "skipped: <reason>" row."""

from __future__ import annotations

from typing import Any

import numpy as np
from tools.leaderboard_regression import _is_skipped

from harness.runners import HarnessConfig, HarnessRunner, SkipMethod


class _SkipSuite(HarnessRunner):
    @property
    def name(self) -> str:
        return "skipdemo"

    @property
    def description(self) -> str:
        return "SkipMethod demo"

    @property
    def supported_datasets(self) -> list[str]:
        return ["demo"]

    def methods(self) -> list[str]:
        return ["Skips", "Fails"]

    def fit(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        if method == "Skips":
            raise SkipMethod("needs y > 0; y_train has zeros")
        raise ValueError("boom")

    def predict(self, method: str, model: Any, X_test: np.ndarray, config: HarnessConfig) -> Any:
        raise AssertionError("predict must not run")


def test_skip_method_produces_skipped_row() -> None:
    splits = {
        "X_train": np.zeros((4, 1)),
        "y_train": np.zeros(4),
        "X_test": np.zeros((2, 1)),
        "y_test": np.zeros(2),
    }
    rows = {r["Method"]: r for r in _SkipSuite().run("demo", HarnessConfig(), splits=splits)}
    skipped = rows["skipdemo/Skips"]
    assert skipped["Notes"] == "skipped: needs y > 0; y_train has zeros"
    assert skipped["RMSE"] is None and skipped["train_s"] is None
    assert _is_skipped(skipped)
    assert rows["skipdemo/Fails"]["Notes"].startswith("failed: ValueError")


class _ImportSuite(_SkipSuite):
    """ImportErrors: only an absent package is a skip; API drift in an installed one fails."""

    def methods(self) -> list[str]:
        return ["NotInstalledMsg", "MissingPackage", "InstalledApiDrift"]

    def fit(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        if method == "NotInstalledMsg":
            raise ImportError("somelib not installed")
        if method == "MissingPackage":
            raise ModuleNotFoundError(
                "No module named 'surely_not_installed_pkg_0'", name="surely_not_installed_pkg_0"
            )
        # numpy is installed: a missing submodule means its API moved.
        raise ModuleNotFoundError("No module named 'numpy.no_such_api'", name="numpy.no_such_api")


def test_import_errors_skip_only_for_absent_packages() -> None:
    splits = {
        "X_train": np.zeros((4, 1)),
        "y_train": np.zeros(4),
        "X_test": np.zeros((2, 1)),
        "y_test": np.zeros(2),
    }
    rows = {r["Method"]: r for r in _ImportSuite().run("demo", HarnessConfig(), splits=splits)}
    assert rows["skipdemo/NotInstalledMsg"]["Notes"].startswith("skipped: ImportError")
    assert rows["skipdemo/MissingPackage"]["Notes"].startswith("skipped: ModuleNotFoundError")
    drift = rows["skipdemo/InstalledApiDrift"]
    assert drift["Notes"].startswith("failed: ModuleNotFoundError")
