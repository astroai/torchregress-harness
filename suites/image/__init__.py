"""Image regression suites.

Each submodule implements a ``HarnessRunner`` subclass that compares
torchregress methods against external software on image regression datasets
(depth estimation, age estimation, keypoint regression).

Suites
------
* ``AgeRegressionSuite`` — image-to-scalar regression (CNN/ResNet + GaussianNLL/MDN/NF/Evidential)
* ``DepthEstimationSuite`` — pixel-level dense regression (UNet + MSE/GaussianNLL/Evidential)
"""

from .age_regression import AgeRegressionSuite
from .depth_estimation import DepthEstimationSuite

__all__ = [
    "AgeRegressionSuite",
    "DepthEstimationSuite",
]
