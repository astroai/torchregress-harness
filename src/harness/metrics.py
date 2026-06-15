"""Unified metric computation for the harness.

Every suite calls these functions to produce a consistent metrics dictionary
regardless of whether the method returns point predictions, distributions,
intervals, or samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import compute_point_metrics


def compute_interval_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: np.ndarray,
    target_coverage: float = 0.9,
) -> dict[str, float]:
    """Compute coverage, mean width, and interval score for prediction intervals.

    Interval score is the proper scoring rule for (1-alpha) prediction intervals:
        S = (u - l) + (2/alpha)*(l - y)_+ + (2/alpha)*(y - u)_+
    """
    lower_np = np.asarray(lower).reshape(-1)
    upper_np = np.asarray(upper).reshape(-1)
    y_true_np = np.asarray(y_true).reshape(-1)

    coverage = float(np.mean((y_true_np >= lower_np) & (y_true_np <= upper_np)))
    width = float(np.mean(upper_np - lower_np))
    alpha = 1.0 - target_coverage

    below = np.maximum(lower_np - y_true_np, 0)
    above = np.maximum(y_true_np - upper_np, 0)
    interval_score_val = float(np.mean(
        (upper_np - lower_np) + (2.0 / alpha) * (below + above)
    ))

    return {
        "Coverage": coverage,
        "Width": width,
        "IntervalScore": interval_score_val,
    }


def compute_distribution_metrics(
    mean: np.ndarray,
    std: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Compute NLL and CRPS for Gaussian predictive distributions."""
    mean_np = np.asarray(mean).reshape(-1)
    std_np = np.asarray(std).reshape(-1).clip(1e-8)
    y_true_np = np.asarray(y_true).reshape(-1)

    # Gaussian NLL (up to additive constant from log(2π))
    nll = float(np.mean(
        0.5 * np.log(2 * np.pi) + np.log(std_np) + 0.5 * ((y_true_np - mean_np) / std_np) ** 2
    ))

    # CRPS for Gaussian (closed form)
    from scipy.special import ndtr
    z = (y_true_np - mean_np) / std_np
    crps = float(np.mean(std_np * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        - 1.0 / np.sqrt(np.pi)
    )))

    return {"NLL": nll, "CRPS": crps}


def compute_calibration_metrics(
    pit_values: np.ndarray,
) -> dict[str, float]:
    """Compute calibration metrics from PIT (Probability Integral Transform) values.

    PIT values should be approximately Uniform(0, 1) for a well-calibrated model.
    """
    pit = np.asarray(pit_values).ravel()
    n = len(pit)

    # ECE via equal-mass binning
    n_bins = min(10, n // 10)
    if n_bins < 2:
        return {"ECE": float("nan"), "PIT_KS": float("nan")}

    bins = np.percentile(pit, np.linspace(0, 100, n_bins + 1))
    ece = 0.0

    for i in range(n_bins):
        mask = (pit >= bins[i]) & (pit < bins[i + 1])
        if np.any(mask):
            observed = np.mean(pit[mask])
            expected = (bins[i] + bins[i + 1]) / 2.0
            ece += float(len(pit[mask]) / n * abs(observed - expected))

    # Kolmogorov-Smirnov against Uniform(0, 1)
    sorted_pit = np.sort(pit)
    cdf_uniform = np.linspace(0, 1, n)
    ks = float(np.max(np.abs(sorted_pit - cdf_uniform)))

    return {"ECE": ece, "PIT_KS": ks}


def compute_all_metrics(
    predictions: dict[str, Any],
    y_true: np.ndarray,
) -> dict[str, float]:
    """Dispatch to the right metrics based on prediction type.

    Args:
        predictions: Dict with keys indicating prediction type:
            - ``y_pred``: point predictions → point metrics
            - ``lower`` + ``upper``: intervals → interval metrics
            - ``mean`` + ``std``: Gaussian distribution → point + distribution metrics
            - ``samples``: MC samples → derives mean/std from samples
        y_true: Ground truth targets (numpy array).

    Returns:
        Flattened dictionary of all applicable metrics.
    """
    y_true_np = np.asarray(y_true).reshape(-1)
    metrics: dict[str, float] = {}

    if "y_pred" in predictions:
        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        metrics.update(compute_point_metrics(y_pred, y_true_np))

    if "lower" in predictions and "upper" in predictions:
        metrics.update(
            compute_interval_metrics(
                np.asarray(predictions["lower"]),
                np.asarray(predictions["upper"]),
                y_true_np,
            )
        )

    if "mean" in predictions and "std" in predictions:
        mean = np.asarray(predictions["mean"]).reshape(-1)
        std = np.asarray(predictions["std"]).reshape(-1)
        if "y_pred" not in predictions:
            metrics.update(compute_point_metrics(mean, y_true_np))
        metrics.update(compute_distribution_metrics(mean, std, y_true_np))

        # PIT values for calibration
        from scipy.stats import norm
        pit = norm.cdf(y_true_np, loc=mean, scale=std)
        metrics.update(compute_calibration_metrics(pit))

    if "samples" in predictions and "y_pred" not in predictions:
        samples = np.asarray(predictions["samples"])
        if samples.ndim == 2:
            mean = samples.mean(axis=0)
            std = samples.std(axis=0)
        else:
            mean = samples.mean()
            std = samples.std()
        metrics.update(compute_point_metrics(mean, y_true_np))
        metrics.update(compute_distribution_metrics(mean, std, y_true_np))

    return metrics
