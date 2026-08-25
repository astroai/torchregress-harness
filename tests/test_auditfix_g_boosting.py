"""Regression tests for audit-fix Workstream G boosting hygiene (WRAP-03).

LightGBM / XGBoost fits must train against a held-out validation set with
early stopping (20 rounds) so ``num_iteration=booster.best_iteration`` at
predict time is meaningful.  The external libraries are faked via
monkeypatching so the test asserts on the exact kwargs that reach
``lgb.train`` / ``xgb.train`` (lightgbm/xgboost are optional extras and are
not installed in the default pixi environment).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import suites.tabular.robust_regression as rr
from suites.tabular.robust_regression import RobustRegressionSuite

from harness.runners import HarnessConfig


@pytest.fixture
def small_splits() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    return {
        "X_train": rng.standard_normal((40, 4)).astype(np.float32),
        "y_train": rng.standard_normal(40).astype(np.float32),
        "X_test": rng.standard_normal((10, 4)).astype(np.float32),
        "y_test": rng.standard_normal(10).astype(np.float32),
    }


class _FakeBooster:
    def __init__(self) -> None:
        self.best_iteration = 7


class _FakeDataset:
    instances: list["_FakeDataset"] = []

    def __init__(self, data: Any = None, label: Any = None, **kwargs: Any) -> None:
        self.data = data
        self.label = label
        self.kwargs = kwargs
        _FakeDataset.instances.append(self)


class _FakeDMatrix:
    pass


class _FakeLgb:
    early_stopping_calls: list[tuple[int, dict]] = []
    train_kwargs: dict[str, Any] = {}

    @staticmethod
    def Dataset(data: Any, label: Any = None, **kwargs: Any) -> _FakeDataset:
        return _FakeDataset(data, label, **kwargs)

    @staticmethod
    def early_stopping(rounds: int, **kwargs: Any) -> tuple[str, int]:
        _FakeLgb.early_stopping_calls.append((rounds, kwargs))
        return ("early_stopping", rounds)

    @staticmethod
    def train(params: dict, train_data: Any, num_boost_round: int = 100, **kwargs: Any) -> Any:
        _FakeLgb.train_kwargs.update(kwargs)
        _FakeLgb.train_kwargs["num_boost_round"] = num_boost_round
        return _FakeBooster()


class _FakeXgb:
    train_kwargs: dict[str, Any] = {}

    @staticmethod
    def DMatrix(data: Any, label: Any = None) -> _FakeDMatrix:
        return _FakeDMatrix()

    @staticmethod
    def train(params: dict, dtrain: Any, num_boost_round: int = 100, **kwargs: Any) -> Any:
        _FakeXgb.train_kwargs.update(kwargs)
        _FakeXgb.train_kwargs["num_boost_round"] = num_boost_round
        return _FakeBooster()


def test_lightgbm_fit_uses_valid_set_and_early_stopping(
    monkeypatch: pytest.MonkeyPatch, small_splits: dict[str, np.ndarray]
) -> None:
    _FakeDataset.instances = []
    _FakeLgb.early_stopping_calls = []
    _FakeLgb.train_kwargs = {}
    # raising=False: lightgbm may not be installed in the default env.
    monkeypatch.setattr(rr, "lgb", _FakeLgb, raising=False)

    state = RobustRegressionSuite().fit("LightGBM", small_splits, HarnessConfig())
    assert isinstance(state["model"], _FakeBooster)

    # 80/20 split carved from training data ⇒ two datasets (valid references train).
    assert len(_FakeDataset.instances) == 2
    valid = _FakeDataset.instances[1]
    assert valid.kwargs.get("reference") is _FakeDataset.instances[0]
    assert len(valid.data) == 8  # 20% of 40 rows
    assert len(_FakeDataset.instances[0].data) == 32

    kwargs = _FakeLgb.train_kwargs
    assert valid in kwargs["valid_sets"]
    callbacks = kwargs["callbacks"]
    assert ("early_stopping", 20) in callbacks
    assert _FakeLgb.early_stopping_calls[0][0] == 20
    assert kwargs["num_boost_round"] == 100


def test_xgboost_fit_uses_eval_set_and_early_stopping(
    monkeypatch: pytest.MonkeyPatch, small_splits: dict[str, np.ndarray]
) -> None:
    _FakeXgb.train_kwargs = {}
    monkeypatch.setattr(rr, "xgb", _FakeXgb, raising=False)

    state = RobustRegressionSuite().fit("XGBoost", small_splits, HarnessConfig())
    assert isinstance(state["model"], _FakeBooster)

    kwargs = _FakeXgb.train_kwargs
    eval_set = kwargs["eval_set"]
    assert len(eval_set) == 1
    assert eval_set[0][1] == "valid"
    assert isinstance(eval_set[0][0], _FakeDMatrix)
    assert kwargs["early_stopping_rounds"] == 20
    assert kwargs["num_boost_round"] == 100
