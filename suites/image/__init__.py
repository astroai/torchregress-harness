"""Image regression suites.

Each submodule implements a ``HarnessRunner`` subclass that compares
torchregress methods against external software on image regression datasets
(depth estimation, age estimation, keypoint regression).
"""

from .age_regression import AgeRegressionSuite
from .cell_counting import CellCountingSuite
from .censored_imaging import CensoredImagingSuite
from .crowd_counting import CrowdCountingSuite
from .depth_estimation import DepthEstimationSuite
from .image_calibration import ImageCalibrationSuite
from .image_conformal import ImageConformalSuite
from .image_ensembles_uq import ImageEnsemblesUQSuite
from .image_robust import ImageRobustSuite
from .keypoint_regression import KeypointRegressionSuite

__all__ = [
    "AgeRegressionSuite",
    "CellCountingSuite",
    "CensoredImagingSuite",
    "CrowdCountingSuite",
    "DepthEstimationSuite",
    "ImageCalibrationSuite",
    "ImageConformalSuite",
    "ImageEnsemblesUQSuite",
    "ImageRobustSuite",
    "KeypointRegressionSuite",
]
