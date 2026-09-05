"""Shared suite registry for CLI tools.

Provides the canonical ``ALL_SUITES`` dict so that ``run_all.py``
and ``run_canonical.py`` don't need to duplicate the import list.
"""

from __future__ import annotations

from suites.image.age_regression import AgeRegressionSuite
from suites.image.cell_counting import CellCountingSuite
from suites.image.depth_estimation import DepthEstimationSuite
from suites.image.image_calibration import ImageCalibrationSuite
from suites.image.image_conformal import ImageConformalSuite
from suites.image.image_ensembles_uq import ImageEnsemblesUQSuite
from suites.image.image_robust import ImageRobustSuite
from suites.image.keypoint_regression import KeypointRegressionSuite
from suites.tabular.bayesian_linear import BayesianLinearSuite
from suites.tabular.calibration import CalibrationSuite
from suites.tabular.censored import CensoredSuite
from suites.tabular.conformal import ConformalSuite
from suites.tabular.distributional_bins import DistributionalBinsSuite
from suites.tabular.eiv import EIVSuite
from suites.tabular.ensembles import EnsemblesSuite
from suites.tabular.foundation_model import FoundationModelSuite
from suites.tabular.multivariate_intervals import MultivariateIntervalsSuite
from suites.tabular.ordinal import OrdinalSuite
from suites.tabular.pzflow_parity import PzflowParitySuite
from suites.tabular.r2c_conformal import R2CConformalSuite
from suites.tabular.robust_regression import RobustRegressionSuite
from suites.tabular.sklearn_baselines import SklearnBaselinesSuite
from suites.tabular.stress_matrix import StressMatrixSuite
from suites.tabular.tweedie import TweedieSuite
from suites.tabular.uq import UQSuite

from harness.runners import HarnessRunner

ALL_SUITES: dict[str, type[HarnessRunner]] = {
    "conformal": ConformalSuite,
    "bayesian_linear": BayesianLinearSuite,
    "censored": CensoredSuite,
    "tweedie": TweedieSuite,
    "foundation_model": FoundationModelSuite,
    "robust_regression": RobustRegressionSuite,
    "calibration": CalibrationSuite,
    "sklearn_baselines": SklearnBaselinesSuite,
    "stress_matrix": StressMatrixSuite,
    "ensembles": EnsemblesSuite,
    "uq": UQSuite,
    "multivariate_intervals": MultivariateIntervalsSuite,
    "ordinal": OrdinalSuite,
    "distributional_bins": DistributionalBinsSuite,
    "eiv": EIVSuite,
    "pzflow_parity": PzflowParitySuite,
    "r2c_conformal": R2CConformalSuite,
    "age_regression": AgeRegressionSuite,
    "cell_counting": CellCountingSuite,
    "depth_estimation": DepthEstimationSuite,
    "keypoint_regression": KeypointRegressionSuite,
    "image_conformal": ImageConformalSuite,
    "image_ensembles_uq": ImageEnsemblesUQSuite,
    "image_robust": ImageRobustSuite,
    "image_calibration": ImageCalibrationSuite,
}
