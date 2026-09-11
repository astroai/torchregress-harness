"""Tabular regression suites.

Each submodule implements a ``HarnessRunner`` subclass that compares
torchregress methods against external software on tabular datasets.

Suites are imported directly from their modules, e.g.::

    from suites.tabular.conformal import ConformalSuite
"""

from .bayesian_linear import BayesianLinearSuite
from .calibration import CalibrationSuite
from .conformal import ConformalSuite
from .distributional_bins import DistributionalBinsSuite
from .ensembles import EnsemblesSuite
from .foundation_model import FoundationModelSuite
from .multivariate_intervals import MultivariateIntervalsSuite
from .pzflow_parity import PzflowParitySuite
from .robust_regression import RobustRegressionSuite
from .sklearn_baselines import SklearnBaselinesSuite
from .stress_matrix import StressMatrixSuite
from .tweedie import TweedieSuite
from .uq import UQSuite

__all__ = [
    "BayesianLinearSuite",
    "CalibrationSuite",
    "ConformalSuite",
    "DistributionalBinsSuite",
    "EnsemblesSuite",
    "FoundationModelSuite",
    "MultivariateIntervalsSuite",
    "PzflowParitySuite",
    "RobustRegressionSuite",
    "SklearnBaselinesSuite",
    "StressMatrixSuite",
    "TweedieSuite",
    "UQSuite",
]
