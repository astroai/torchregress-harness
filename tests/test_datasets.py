"""Tests for the dataset registry."""

import pytest

from harness.datasets import (
    DATASET_REGISTRY,
    DatasetInfo,
    list_datasets,
    load_dataset,
)


def test_registry_has_expected_datasets() -> None:
    expected = [
        "california_housing",
        "diabetes",
        "openml_year",
        "synthetic_hetero",
        "synthetic_outliers",
        "synthetic_tweedie",
        "synthetic_censored",
        "synthetic_eiv",
    ]
    for name in expected:
        assert name in DATASET_REGISTRY, f"Missing dataset: {name}"


def test_load_dataset_returns_splits() -> None:
    splits = load_dataset("synthetic_hetero", n_samples=200)
    assert "X_train" in splits
    assert "X_test" in splits
    assert "y_train" in splits
    assert "y_test" in splits
    assert splits["X_train"].shape[0] + splits["X_test"].shape[0] == 200


def test_load_dataset_unknown_raises() -> None:
    with pytest.raises(KeyError):
        load_dataset("nonexistent_dataset")


def test_list_datasets_all() -> None:
    all_datasets = list_datasets()
    assert len(all_datasets) >= 5
    assert all(isinstance(d, DatasetInfo) for d in all_datasets)


def test_list_datasets_by_category() -> None:
    tabular = list_datasets(category="tabular")
    synthetic = list_datasets(category="synthetic")
    assert len(tabular) >= 2
    assert len(synthetic) >= 3
