"""torchregress-harness — Benchmarking harness for torchregress.

Provides a unified framework for comparing every torchregress regression
method against external software on known tabular and image datasets.
"""

from harness.core import (
    compute_point_metrics,
    print_comparison_summary,
    print_fairness_notes,
    set_harness_seed,
    timed_call,
    write_harness_json,
)
from harness.datasets import (
    CANONICAL_SCENARIOS,
    DATASET_REGISTRY,
    CanonicalScenario,
    DatasetInfo,
    list_datasets,
    load_dataset,
)

# Apply PROPOSAL #15 protocol patches to torchregress distribution losses.
# Idempotent — guarded by the _PATCH_SENTINEL attribute on each class.
# Importing this module therefore completes the ``log_prob`` /
# ``predict_interval`` surface on MixtureDensityLoss / NormalizingFlowLoss
# before any test or suite runs.
from harness.loss_augments import install_loss_protocol_augments  # noqa: E402, F401
from harness.metrics import (
    compute_all_metrics,
    compute_calibration_metrics,
    compute_distribution_metrics,
    compute_interval_metrics,
)
from harness.report import load_run, render_markdown, render_report_file
from harness.runners import HarnessConfig, HarnessRunner
from harness.wrappers import (
    EvidentialRegressor,
    GaussianNLLRegressor,
    MDNRegressor,
    NormalizingFlowRegressor,
)

install_loss_protocol_augments()

__all__ = [
    # core
    "compute_point_metrics",
    "print_comparison_summary",
    "print_fairness_notes",
    "set_harness_seed",
    "timed_call",
    "write_harness_json",
    # datasets
    "CANONICAL_SCENARIOS",
    "CanonicalScenario",
    "DATASET_REGISTRY",
    "DatasetInfo",
    "list_datasets",
    "load_dataset",
    # metrics
    "compute_all_metrics",
    "compute_calibration_metrics",
    "compute_distribution_metrics",
    "compute_interval_metrics",
    # runners
    "HarnessConfig",
    "HarnessRunner",
    # wrappers (Proposal #3 — sklearn-style fit/predict API)
    "GaussianNLLRegressor",
    "MDNRegressor",
    "EvidentialRegressor",
    "NormalizingFlowRegressor",
    # report
    "load_run",
    "render_markdown",
    "render_report_file",
]
