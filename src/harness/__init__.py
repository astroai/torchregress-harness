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
from harness.metrics import (
    compute_all_metrics,
    compute_calibration_metrics,
    compute_distribution_metrics,
    compute_interval_metrics,
)
from harness.report import load_run, render_markdown, render_report_file
from harness.runners import HarnessConfig, HarnessRunner

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
    # report
    "load_run",
    "render_markdown",
    "render_report_file",
]
