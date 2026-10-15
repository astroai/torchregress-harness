"""torchregress exports the harness deliberately does not benchmark in a suite.

``tests/test_export_coverage.py`` fails when a public export (``__all__`` of
the modules in ``tools/export_coverage.MODULES``) is neither referenced by a
registered suite nor listed here.  Each entry is ``name -> one-line reason``.
An entry that becomes covered (or stops being exported) is stale and also
fails the test: delete it.

Reasons fall in a few groups:

* base classes, protocols, configs and result containers that a covered
  estimator consumes or returns (they have no behaviour of their own to score);
* functional twins / aliases of a covered class (same kernel);
* metrics the harness computes itself in ``src/harness`` (outside the suite
  scan) or that wait for the metrics parity suite against ``scoringrules``
  (RELEASE_0.3_STATUS.md, still to do);
* methods with no suite yet, named with the suite that should take them.
"""

from __future__ import annotations

_BASE = "abstract base class; exercised through its covered subclasses"
_CONFIG = "configuration dataclass consumed by a covered estimator; no behaviour to score"
_RESULT = "result container returned by a covered estimator; not called directly"
_FACTORY = "factory/registry helper; the losses it builds are benchmarked directly"
_PARITY = (
    "metric not scored by any suite; checked against scoringrules / scipy references "
    "in tests/test_metrics_scoringrules_parity.py"
)
_HARNESS_POINT = (
    "harness computes this point metric itself (harness.core.compute_point_metrics); "
    "library parity pending the metrics parity tests"
)
_HARNESS_SRC = (
    "called for every row by harness.metrics.compute_all_metrics (src/, outside the suite scan)"
)
_OOD = (
    "OOD score; only the aggregate ood_metrics_report is used (test_time_adaptation shift severity)"
)
_ENSEMBLE_STATS = (
    "ensemble statistics helper; ensemble suites compute mixture moments in "
    "suites/_backbones._mixture_gaussian_stats and score them via harness metrics"
)
_CALIB_QUANTILE = (
    "quantile-calibration error; harness reports ECE/MACE/RMSCE from PIT values "
    "(harness.metrics.compute_calibration_metrics); parity pending metrics parity tests"
)
_TTA_INTERNAL = "low-level helper called inside covered adapters (ShiftFactoredPredictiveTransport, Gaussian label-shift correction)"
_CLASSIFICATION = "classification-only test-time helper; regression suites have no use for it"

ALLOWLIST: dict[str, str] = {
    # ------------------------------------------------------------------ losses
    "BaseLoss": _BASE,
    "RegressionLoss": _BASE,
    "DistributionLoss": _BASE,
    "BaseEIVLoss": "abstract EIV base class; exercised through FunctionalEIVLoss/StructuralEIVLoss (covered in eiv)",
    "ConformalPredictor": _BASE,
    "WeightedLossWrapper": "masking/weighting wrapper around torch losses; weighting is exercised via covered losses",
    "WeightedNLLLoss": "classification NLL wrapper (torch.nn.NLLLoss); no regression benchmark",
    "AsymmetricLeastSquaresLoss": "alias of ExpectileLoss (covered in robust_regression)",
    "ConsistencyRegLoss": "uncertain-ground-truth teacher/student loss; semi_supervised covers TeacherStudentTrainer/SAGEReg instead",
    "PseudoLabelConsistencyLoss": "uncertain-ground-truth pseudo-label loss; no pseudo-label suite beyond semi_supervised yet",
    "discrete_wasserstein1": "functional twin of DiscreteWasserstein1Loss (covered in ordinal)",
    "MahalanobisFrontier": "SLS frontier component built inside SLSLoss/SLSConformal (covered)",
    "UnionFrontier": "SLS multimodal frontier component built inside SLSLoss (covered)",
    "VolumePreservingFlow": "SLS flow component built inside SLSLoss (covered)",
    "TransformedTargetLoss": "generic target-transform wrapper; its concrete LogTransform/BoxCox/YeoJohnson losses are covered",
    "asymmetric_laplace_nll": "functional twin of AsymmetricLaplaceNLLLoss (covered in likelihoods)",
    "beta_nll_loss": "functional twin of BetaNLLLoss (covered)",
    "expectile_loss": "functional twin of ExpectileLoss (covered)",
    "quantile_loss": "functional twin of QuantileLoss (covered)",
    "tweedie_loss": "functional twin of TweedieLoss (covered in tweedie)",
    "sqr_loss": "functional twin of SQRLoss (covered)",
    "gaussian_wasserstein_bound_loss": "functional twin of GaussianWassersteinBoundLoss (covered)",
    "rank_n_contrast_loss": "functional kernel of RankNContrastLoss (covered in age_regression)",
    "create_flow_loss": _FACTORY,
    "create_contrastive_flow_loss": _FACTORY,
    "create_gaussian_nll": _FACTORY,
    "create_mdn_loss": _FACTORY,
    "create_loss_from_config": _FACTORY,
    "get_regression_loss": _FACTORY,
    "list_regression_losses": _FACTORY,
    "low_rank_output_dim": "shape helper for LowRankGaussianLoss heads (covered in multivariate_intervals)",
    "split_low_rank_gaussian_output": "output-parsing helper used inside LowRankGaussianLoss (covered)",
    "symmetric_spd_matrix_sqrt": "linear-algebra helper inside GaussianWassersteinBoundLoss (covered)",
    # ----------------------------------------------------------------- metrics
    "ContinuousRankedProbabilityScore": _PARITY,
    "continuous_ranked_probability_score": _PARITY,
    "DawidSebastianiScore": _PARITY,
    "PinballMetric": _PARITY,
    "pinball_loss": _PARITY,
    "VarioScore": _PARITY,
    "vario_score": _PARITY,
    "WassersteinGaussian": _PARITY,
    "wasserstein_gaussian_p2": _PARITY,
    "probability_integral_transform": "harness computes PIT itself (harness.metrics.compute_pit_values); parity pending metrics parity tests",
    "highest_posterior_density_level": "HPD level helper; suites use highest_posterior_density_coverage (covered)",
    "distribution_metrics_report": "report aggregator over covered distribution metrics",
    "interval_metrics_report": "report aggregator; harness computes interval metrics in harness.metrics.compute_interval_metrics",
    "regression_metrics_report": "report aggregator; harness computes point metrics in harness.core",
    "calibration_metrics_report": "report aggregator over calibration metrics; harness reports PIT-based ECE/MACE",
    "calibration_score": _CALIB_QUANTILE,
    "ExpectedCalibrationError": _CALIB_QUANTILE,
    "expected_calibration_error": _CALIB_QUANTILE,
    "MarginalCalibrationError": _CALIB_QUANTILE,
    "marginal_calibration_error": _CALIB_QUANTILE,
    "MeanPredictionIntervalWidth": "harness computes width in harness.metrics.compute_interval_metrics (inf-aware)",
    "PredictionIntervalCoverageProbability": "harness computes coverage in harness.metrics.compute_interval_metrics",
    "prediction_interval_coverage_probability": "harness computes coverage in harness.metrics.compute_interval_metrics",
    "prediction_interval_coverage": "compatibility alias of prediction_interval_coverage_probability",
    "Sharpness": "harness reports Sharpness (mean std) in harness.metrics.compute_distribution_metrics",
    "sharpness": "functional twin of Sharpness; harness computes it itself",
    "mean_squared_error": _HARNESS_POINT,
    "mean_absolute_error": _HARNESS_POINT,
    "median_absolute_error": _HARNESS_POINT,
    "MedianAbsoluteError": _HARNESS_POINT,
    "r2_score": _HARNESS_POINT,
    "huber_loss": _PARITY,
    "HuberMetric": _PARITY,
    "MedianAbsoluteDeviation": _PARITY,
    "median_absolute_deviation": _PARITY,
    "NormalizedRMSE": _PARITY,
    "normalized_rmse": _PARITY,
    "TrimmedMeanSquaredError": _PARITY,
    "trimmed_mean_squared_error": _PARITY,
    "tail_mae": _HARNESS_SRC,
    "tail_rmse": _HARNESS_SRC,
    "risk_coverage_curve": _HARNESS_SRC,
    "RiskCoverageCurve": "class twin of risk_coverage_curve (AURC reported for every Gaussian row by harness.metrics)",
    "RejectionPolicy": "fixed-threshold rejection evaluation; harness reports the threshold-free AURC instead",
    "MultivariateMAE": "multivariate point metric; multivariate_intervals scores regions (volume/coverage), not vector MAE",
    "MultivariateRMSE": "multivariate point metric; multivariate_intervals scores regions (volume/coverage), not vector RMSE",
    "TaskAgnosticCorrelations": "representation-quality metric (TAC); no representation-learning suite yet",
    "EntropyScore": _OOD,
    "entropy_score": _OOD,
    "KernelDensityScore": _OOD,
    "kernel_density_score": _OOD,
    "MahalanobisDistance": _OOD,
    "mahalanobis_distance": _OOD,
    "TypicalityScore": _OOD,
    "typicality_score": _OOD,
    "EnsembleIntervalMetrics": _ENSEMBLE_STATS,
    "ensemble_interval_bounds": _ENSEMBLE_STATS,
    "ensemble_interval_metrics": _ENSEMBLE_STATS,
    "ensemble_mean": _ENSEMBLE_STATS,
    "ensemble_std": _ENSEMBLE_STATS,
    "ensemble_statistics": _ENSEMBLE_STATS,
    "ensemble_variance_decomposition": "alias of uncertainty_decomposition (covered in ensembles)",
    "gaussian_nll_ensemble": "functional twin of GaussianNLLEnsemble (covered in ensembles)",
    "censoring_rate": "censored-data descriptor (fraction censored), not a model score",
    "interval_overlap_rate": "censored-interval overlap diagnostic; censored suite scores C-index and observed MAE",
    "noisy_target_gaussian_nll": "functional twin of NoisyTargetGaussianNLL (covered in age_regression)",
    "consistency_error": "teacher/student diagnostic for the uncertain-GT losses (not benchmarked yet)",
    "pseudo_label_acceptance_rate": "pseudo-label diagnostic for the uncertain-GT losses (not benchmarked yet)",
    # ------------------------------------------------------------- calibration
    # ---------------------------------------------------------------- ensemble
    "BaseEnsembleModel": _BASE,
    "EnsembleFitConfig": _CONFIG,
    "BatchEnsembleOutput": "structured return type of BatchEnsembleRegressor.predict_output; suite uses predict()",
    "SoftmaxModelCombiner": "learned softmax combiner; ensembles suite benchmarks StackingEnsemble as the learned combiner",
    "enable_dropout": "helper used inside MCDropoutWrapper (covered)",
    "parse_heteroscedastic_output": "output-parsing helper inside heteroscedastic ensembles (covered)",
    # -------------------------------------------------------------- algorithms
    "AdaptivePriorGuide": "guide network built inside VIDSRegressor (covered in ensembles)",
    "sample_synthetic_environments": "environment sampler used inside VIDSRegressor (covered)",
    "NaturalHeteroscedasticHead": "head module inside HeteroscedasticLaplaceRegressor (covered)",
    "NaturalReparamHead": "reparameterization helper inside HeteroscedasticLaplaceRegressor (covered)",
    "NeighborhoodCovarianceConfig": _CONFIG,
    "ErrorAwareFeatureEncoder": "feature encoder for inputs with known errors; eiv benchmarks NoiseAwareRegressor/SIMEX instead",
    # --------------------------------------------------------------- test_time
    "AdaptationBatch": _RESULT,
    "JointTTAResult": _RESULT,
    "LabelShiftEstimate": "EM result container (target_prior/iterations read in label_shift TT_EM)",
    "ShiftFactoredTransportState": _RESULT,
    "SubspaceAlignmentState": _RESULT,
    "SupportsPredictiveBatch": "typing Protocol; no behaviour",
    "GaussianLabelShiftConfig": _CONFIG,
    "LocalConsistencyConfig": _CONFIG,
    "ParameterEMA": "parameter-EMA utility for user-written adaptation loops; no behaviour to score on its own",
    "ScoreCDFReweighter": "base of the OT weight estimator behind JointDistributionalTTA(weight_estimator='ot') (JointTTA_OTScore row)",
    "WeightedSplitConformalAdapter": _CLASSIFICATION,
    "weighted_split_classification_predictive_batch": _CLASSIFICATION,
    "confidence_scores": _CLASSIFICATION,
    "entropy_scores": _CLASSIFICATION,
    "pseudo_label_targets": "pseudo-label helper for user loops; test_time_adaptation runs JointTTA with pseudo-label rounds disabled",
    "select_high_confidence": _TTA_INTERNAL,
    "local_consistency_weights": _TTA_INTERNAL,
    "flatten_adaptation_parameters": "parameter-collection utility for user adaptation loops; no behaviour to score",
    "gaussian_moments_from_binned_probabilities": "inverse map used inside correct_gaussian_predictions_for_label_shift (covered)",
    # --------------------------------------------------------------- inference
    "OrthogonalEstimate": _RESULT,
    "median_heuristic_bandwidth": "bandwidth rule called inside random_fourier_features(bandwidth='median') (covered by orthogonal_inference)",
    # ------------------------------------------------------------------ models
    "PeriodicEmbedding": "embedding layer built inside TabularMLP (embedding='periodic', covered)",
    # --------------------------------------------------------- semi_supervised
    "SAGERegAgreement": _RESULT,
    "SAGERegOutput": _RESULT,
    "SelfAgreementTrainer": "alias of TeacherStudentTrainer (same class; covered in semi_supervised)",
    "build_consensus_predictive_batch": "consensus builder called inside TeacherStudentTrainer and SAGERegLoss (covered)",
    "distributional_pseudo_loss": "pseudo-label loss called inside TeacherStudentTrainer and SAGERegLoss (covered)",
    "conformal_width_to_weight": "needs per-batch conformal intervals on unlabeled rows; no conformal pseudo-labeling row yet",
}
