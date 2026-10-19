"""Shared suite registry for CLI tools.

Provides the canonical ``ALL_SUITES`` dict so that ``run_all.py``
and ``run_canonical.py`` don't need to duplicate the import list.
"""

from __future__ import annotations

from suites.image.age_regression import AgeRegressionSuite
from suites.image.cell_counting import CellCountingSuite
from suites.image.censored_imaging import CensoredImagingSuite
from suites.image.crowd_counting import CrowdCountingSuite
from suites.image.depth_estimation import DepthEstimationSuite
from suites.image.image_calibration import ImageCalibrationSuite
from suites.image.image_conformal import ImageConformalSuite
from suites.image.image_ensembles_uq import ImageEnsemblesUQSuite
from suites.image.image_robust import ImageRobustSuite
from suites.image.keypoint_regression import KeypointRegressionSuite
from suites.image.photoz_imaging import PhotozImagingSuite
from suites.tabular.bayesian_linear import BayesianLinearSuite
from suites.tabular.calibration import CalibrationSuite
from suites.tabular.causal import CausalSuite
from suites.tabular.censored import CensoredSuite
from suites.tabular.classification_shift import ClassificationShiftSuite
from suites.tabular.cnf_parity import CNFParitySuite
from suites.tabular.conformal import ConformalSuite
from suites.tabular.constrained_outputs import ConstrainedOutputsSuite
from suites.tabular.density_parity import DensityParitySuite, ImageDensityParitySuite
from suites.tabular.distributional_bins import DistributionalBinsSuite
from suites.tabular.eiv import EIVSuite
from suites.tabular.eiv_estimators import EIVEstimatorsSuite
from suites.tabular.engression_parity import EngressionParitySuite
from suites.tabular.ensembles import EnsemblesSuite
from suites.tabular.flow_eiv import FlowEIVSuite
from suites.tabular.fm_multitarget import FMMultiTargetSuite
from suites.tabular.fm_uq_scenarios import FMUQScenariosSuite
from suites.tabular.foundation_model import FoundationModelSuite
from suites.tabular.freia_parity import FreiaParitySuite
from suites.tabular.imbalanced_representation import ImbalancedRepresentationSuite
from suites.tabular.label_shift import LabelShiftSuite
from suites.tabular.likelihood_ratio import LikelihoodRatioSuite
from suites.tabular.likelihoods import LikelihoodsSuite
from suites.tabular.missing_data import MissingDataSuite
from suites.tabular.multivariate_intervals import MultivariateIntervalsSuite
from suites.tabular.ordinal import OrdinalSuite
from suites.tabular.orthogonal_inference import OrthogonalInferenceSuite
from suites.tabular.ppi import PPISuite
from suites.tabular.pz_challenge import PZChallengeSuite, PZCPExtendedSuite
from suites.tabular.pz_robustness import PZOODSuite, PZRemovalSuite
from suites.tabular.pzflow_parity import PzflowParitySuite
from suites.tabular.r2c_conformal import R2CConformalSuite
from suites.tabular.robust_regression import RobustRegressionSuite
from suites.tabular.sbi_parity import SBIParitySuite
from suites.tabular.semi_supervised import SemiSupervisedSuite
from suites.tabular.sklearn_baselines import SklearnBaselinesSuite
from suites.tabular.streaming_tta import StreamingTTASuite
from suites.tabular.stress_matrix import StressMatrixSuite
from suites.tabular.test_time_adaptation import TestTimeAdaptationSuite
from suites.tabular.tweedie import TweedieSuite
from suites.tabular.uq import UQSuite
from suites.tabular.uq_layer import UQLayerSuite
from suites.tabular.uq_shift import UQShiftSuite

from harness.runners import HarnessRunner

ALL_SUITES: dict[str, type[HarnessRunner]] = {
    "conformal": ConformalSuite,
    "cnf_parity": CNFParitySuite,
    "bayesian_linear": BayesianLinearSuite,
    "censored": CensoredSuite,
    "tweedie": TweedieSuite,
    "foundation_model": FoundationModelSuite,
    "freia_parity": FreiaParitySuite,
    "robust_regression": RobustRegressionSuite,
    "calibration": CalibrationSuite,
    "sklearn_baselines": SklearnBaselinesSuite,
    "stress_matrix": StressMatrixSuite,
    "test_time_adaptation": TestTimeAdaptationSuite,
    "ensembles": EnsemblesSuite,
    "imbalanced_representation": ImbalancedRepresentationSuite,
    "uq": UQSuite,
    "uq_layer": UQLayerSuite,
    "uq_shift": UQShiftSuite,
    "fm_uq_scenarios": FMUQScenariosSuite,
    "fm_multitarget": FMMultiTargetSuite,
    "multivariate_intervals": MultivariateIntervalsSuite,
    "ordinal": OrdinalSuite,
    "ppi": PPISuite,
    "sbi_parity": SBIParitySuite,
    "likelihood_ratio": LikelihoodRatioSuite,
    "streaming_tta": StreamingTTASuite,
    "causal": CausalSuite,
    "orthogonal_inference": OrthogonalInferenceSuite,
    "constrained_outputs": ConstrainedOutputsSuite,
    "missing_data": MissingDataSuite,
    "label_shift": LabelShiftSuite,
    "likelihoods": LikelihoodsSuite,
    "semi_supervised": SemiSupervisedSuite,
    "classification_shift": ClassificationShiftSuite,
    "flow_eiv": FlowEIVSuite,
    "distributional_bins": DistributionalBinsSuite,
    "engression_parity": EngressionParitySuite,
    "eiv": EIVSuite,
    "eiv_estimators": EIVEstimatorsSuite,
    "pz_challenge": PZChallengeSuite,
    "pz_cp": PZCPExtendedSuite,
    "pz_ood": PZOODSuite,
    "pz_removal": PZRemovalSuite,
    "pzflow_parity": PzflowParitySuite,
    "density_parity": DensityParitySuite,
    "image_density_parity": ImageDensityParitySuite,
    "r2c_conformal": R2CConformalSuite,
    "age_regression": AgeRegressionSuite,
    "cell_counting": CellCountingSuite,
    "censored_imaging": CensoredImagingSuite,
    "crowd_counting": CrowdCountingSuite,
    "depth_estimation": DepthEstimationSuite,
    "keypoint_regression": KeypointRegressionSuite,
    "photoz_imaging": PhotozImagingSuite,
    "image_conformal": ImageConformalSuite,
    "image_ensembles_uq": ImageEnsemblesUQSuite,
    "image_robust": ImageRobustSuite,
    "image_calibration": ImageCalibrationSuite,
}


def supports_dataset(suite: HarnessRunner, dataset: str) -> bool:
    """True when ``dataset`` is one of ``suite.supported_datasets``.

    Drivers (``run_all.py``, ``leaderboard_regression.py``) consult this before
    calling ``suite.run``: a suite handed a dataset it does not declare (e.g. a
    univariate dataset for ``multivariate_intervals``) is skipped, not run into
    a column of failed rows.
    """
    return dataset in suite.supported_datasets


def split_supported(suite: HarnessRunner, datasets: list[str]) -> tuple[list[str], list[str]]:
    """Partition ``datasets`` into (supported, unsupported) for ``suite``, order kept."""
    supported = set(suite.supported_datasets)
    return [d for d in datasets if d in supported], [d for d in datasets if d not in supported]
