"""Smoke tests for harness suites.

Each test imports a suite class and verifies it is properly structured
(methods, supported_datasets, name, description).
"""

import pytest

# All suites should be importable
from suites.tabular.bayesian_linear import BayesianLinearSuite
from suites.tabular.calibration import CalibrationSuite
from suites.tabular.conformal import ConformalSuite
from suites.tabular.ensembles import EnsemblesSuite
from suites.tabular.foundation_model import FoundationModelSuite
from suites.tabular.robust_regression import RobustRegressionSuite
from suites.tabular.sklearn_baselines import SklearnBaselinesSuite
from suites.tabular.tweedie import TweedieSuite
from suites.tabular.uq import UQSuite

ALL_SUITES = [
    ConformalSuite,
    BayesianLinearSuite,
    TweedieSuite,
    FoundationModelSuite,
    RobustRegressionSuite,
    CalibrationSuite,
    SklearnBaselinesSuite,
    EnsemblesSuite,
    UQSuite,
]


@pytest.mark.parametrize("suite_cls", ALL_SUITES)
def test_suite_has_name_and_description(suite_cls) -> None:
    suite = suite_cls()
    assert isinstance(suite.name, str) and len(suite.name) > 0
    assert isinstance(suite.description, str) and len(suite.description) > 0


@pytest.mark.parametrize("suite_cls", ALL_SUITES)
def test_suite_has_supported_datasets(suite_cls) -> None:
    suite = suite_cls()
    assert isinstance(suite.supported_datasets, list)
    assert len(suite.supported_datasets) > 0
    assert all(isinstance(d, str) for d in suite.supported_datasets)


@pytest.mark.parametrize("suite_cls", ALL_SUITES)
def test_suite_has_methods(suite_cls) -> None:
    suite = suite_cls()
    methods = suite.methods()
    assert isinstance(methods, list)
    assert len(methods) > 0
    assert all(isinstance(m, str) for m in methods)
