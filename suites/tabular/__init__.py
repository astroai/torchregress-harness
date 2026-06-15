"""Tabular regression suites.

Each submodule implements a ``HarnessRunner`` subclass that compares
torchregress methods against external software on tabular datasets.

Suites are imported directly from their modules, e.g.::

    from suites.tabular.conformal import ConformalSuite
"""

from .bayesian_linear import BayesianLinearSuite
from .calibration import CalibrationSuite
from .conformal import ConformalSuite
from .ensembles import EnsemblesSuite
from .foundation_model import FoundationModelSuite
from .robust_regression import RobustRegressionSuite
from .sklearn_baselines import SklearnBaselinesSuite
from .tweedie import TweedieSuite
from .uq import UQSuite

__all__ = [
    "BayesianLinearSuite",
    "CalibrationSuite",
    "ConformalSuite",
    "EnsemblesSuite",
    "FoundationModelSuite",
    "RobustRegressionSuite",
    "SklearnBaselinesSuite",
    "TweedieSuite",
    "UQSuite",
]
