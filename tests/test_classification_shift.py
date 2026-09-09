"""Execution coverage for supervised binary classification label-shift APIs."""

from __future__ import annotations

from typing import cast

import numpy as np
from sklearn.datasets import make_classification
from suites.tabular.classification_shift import ClassificationShiftSuite
from tools._suite_registry import ALL_SUITES

from harness.datasets import SplitsDict
from harness.runners import HarnessConfig


def test_classification_shift_registered() -> None:
    assert ALL_SUITES["classification_shift"] is ClassificationShiftSuite


def test_classification_shift_api_methods_smoke() -> None:
    X, y = make_classification(
        n_samples=240,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.75, 0.25],
        class_sep=1.5,
        random_state=7,
    )
    splits = cast(
        SplitsDict,
        {
            "X_train": X[:180].astype(np.float32),
            "y_train": y[:180].astype(np.int64),
            "X_test": X[180:].astype(np.float32),
            "y_test": y[180:].astype(np.int64),
        },
    )
    config = HarnessConfig(seed=13, epochs=1, batch_size=32, standardize_targets=False)
    suite = ClassificationShiftSuite()
    state = suite.fit("EM", splits, config)

    for method in suite.methods():
        predictions = suite.predict(method, state, splits["X_test"], config)
        metrics = suite.evaluate(method, predictions, splits["y_test"], config)
        for name in ("Accuracy", "BalancedAccuracy", "LogLoss", "Brier", "PriorRatioRMSE"):
            assert np.isfinite(metrics[name]), f"{method} produced invalid {name}"
        assert metrics["TargetPositivePrior"] == 0.7

    oracle_predictions = suite.predict("Oracle", state, splits["X_test"], config)
    oracle_metrics = suite.evaluate("Oracle", oracle_predictions, splits["y_test"], config)
    assert oracle_metrics["TargetPriorL1"] == 0.0
