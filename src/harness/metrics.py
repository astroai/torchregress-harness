"""Unified metric computation for the harness.

Every suite calls these functions to produce a consistent metrics dictionary
regardless of whether the method returns point predictions, distributions,
intervals, or samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import compute_point_metrics

# ---------------------------------------------------------------------------
# PIT (Probability Integral Transform) — calibration diagnostics
# ---------------------------------------------------------------------------
#
# A model is *well-calibrated* iff PIT_i = F(y_i | X_i) ~ Uniform(0, 1), where
# F is the model's predictive CDF for the i-th test point.
#
# Empirical histogram of PIT values:
#   * Uniform → model is calibrated
#   * U-shaped (mass near 0 and 1) → model is overconfident
#   * Bell-shaped (mass near 0.5) → model is underconfident
#   * Skewed left/right → model has systematic bias
#
# Statistical tests detect deviations from Uniform(0, 1):
#   * KS (Kolmogorov-Smirnov) — max vertical distance between empirical and
#     uniform CDFs; sensitive to any location/scale distortion.
#   * CvM (Cramér-von Mises) — integrated squared difference; tighter on
#     near-uniform distributions than KS.
#
# Reference:
#   Dawid, A. P. (1984). "Present position and potential developments:
#   Some personal views. Statistical theory. The Prequential Approach".
#   Gneiting, T., Balabdaoui, F., Raftery, A. E. (2007). "Probabilistic
#   forecasts, calibration and sharpness". JRSS B.


def _pit_from_gaussian(
    y_true: np.ndarray,
    predictions: dict[str, Any],
) -> np.ndarray:
    """PIT values from a Gaussian predictive (mean+std)."""
    from scipy.stats import norm

    mean = np.asarray(predictions["mean"]).reshape(-1)
    std = np.asarray(predictions["std"]).reshape(-1).clip(min=1e-8)
    return norm.cdf(np.asarray(y_true).reshape(-1), loc=mean, scale=std)


def _pit_from_student_t(
    y_true: np.ndarray,
    predictions: dict[str, Any],
) -> np.ndarray:
    """PIT values from a NIG/Student-t predictive (gamma, nu, alpha, beta).

    Evidential regression returns a Normal-Inverse-Gamma posterior predictive,
    which is a Student-t distribution with:
        df       = 2 * alpha
        loc      = gamma
        scale^2  = beta * (1 + nu) / (nu * alpha)
    """
    from scipy.stats import t as student_t

    gamma = np.asarray(predictions["gamma"]).reshape(-1)
    nu = np.asarray(predictions["nu"]).reshape(-1).clip(min=1e-8)
    alpha = np.asarray(predictions["alpha"]).reshape(-1).clip(min=1.0 + 1e-8)
    beta = np.asarray(predictions["beta"]).reshape(-1).clip(min=1e-8)

    df = 2.0 * alpha
    scale = np.sqrt(beta * (1.0 + nu) / (nu * alpha))
    return student_t.cdf(np.asarray(y_true).reshape(-1), df=df, loc=gamma, scale=scale)


def _pit_from_samples(
    y_true: np.ndarray,
    predictions: dict[str, Any],
) -> np.ndarray:
    """PIT values from MC samples (e.g. NormalizingFlow, MDN).

    For each test point, the empirical PIT equals the fraction of samples
    that are less than or equal to the true value.  Fully vectorised via
    NumPy broadcasting.

    Accepts both ``(N, S)`` (per-row samples) and ``(S, N)`` (per-column)
    layouts — auto-detected by matching the first axis to ``len(y_true)``.
    """
    samples = np.asarray(predictions["samples"])
    y_arr = np.asarray(y_true).reshape(-1)

    if samples.ndim == 1:
        return np.array([float(np.mean(samples <= y_arr[0]))])

    if samples.ndim != 2:
        raise ValueError(f"Samples must be 1D or 2D, got shape {samples.shape}")

    n_test = len(y_arr)
    s = samples if samples.shape[0] == n_test else samples.T
    return (s <= y_arr[:, None]).mean(axis=1).astype(np.float64)


def compute_pit_values(
    predictions: dict[str, Any],
    y_true: np.ndarray,
    *,
    loss_fn: Any | None = None,
    context: Any | None = None,
    target: np.ndarray | Any | None = None,
    n_samples: int = 512,
) -> np.ndarray:
    """Compute PIT values from a prediction dictionary OR raw loss outputs.

    Two dispatch modes (mutually exclusive; ``loss_fn`` wins if both are set):

    **Dict-passthrough** (legacy, fast):
        Dispatches based on what keys ``predictions`` contains:
            1. ``samples`` — empirical PIT (NF, MDN, ensemble)
            2. ``mean`` + ``std`` — Gaussian PIT (GaussianNLL, MultivariateGaussian)
            3. ``gamma``+``nu``+``alpha``+``beta`` — Student-t PIT (NIG evidential)
            4. ``lower``+``upper`` — interval-position proxy: relative position
               of ``y`` inside its interval, clipped to [0, 1] (0 = at/below
               lower, 1 = at/above upper).  Miscalibrated intervals saturate
               and fail uniformity instead of being reported as calibrated.

    **Loss-fn dispatch** (PROPOSAL #15):
        If ``loss_fn`` is provided (alongside ``context`` and ``target``),
        delegates to :func:`harness.loss_augments.compute_pit_from_loss_fn` to
        draw MC samples from the fitted distribution loss and compute the
        empirical CDF at each ``target`` value.  This is the path that lets
        callers pass *raw loss outputs* (computed in the suite's training
        loop) rather than pre-flattening them into a predictions dict.

    Backwards compatible: existing callers passing only ``predictions`` and
    ``y_true`` keep the legacy dict-passthrough path.  No torch import is
    pulled in unless the ``loss_fn`` arg is supplied.

    Returns
    -------
    np.ndarray
        PIT values, one per test point.  Empty array if no supported key found.
    """
    y_arr = np.asarray(y_true).reshape(-1)

    # ----- PROPOSAL #15 dispatch path: fitted loss_fn + context + target -----
    if loss_fn is not None:
        # Lazy import keeps the legacy dict-passthrough path torch-free:
        # callers who never use the ``loss_fn`` kwarg do not pay any
        # torchregress import cost.
        from .loss_augments import compute_pit_from_loss_fn

        # The loss_fn dispatch path is torch-side: ``context`` and ``target``
        # are torch tensors consumed by ``loss_fn.sample()``.  When the
        # caller supplies only ``loss_fn + context`` (the headline new entry
        # point: ``compute_pit_values(y_true=np.asarray(y_test), loss_fn,
        # context=X_test_tensor)``), ``target`` falls back to ``y_arr`` —
        # we bridge from NumPy to torch, matching the context's device AND
        # dtype so a GPU/float64 loss_fn doesn't drift to CPU/float32.
        if context is None:
            raise TypeError(
                "compute_pit_values: the ``loss_fn`` dispatch requires "
                "``context`` (and ``target``, which may fall back to y_true)."
            )
        import torch as _torch

        tgt = target if isinstance(target, _torch.Tensor) else None
        if tgt is None:
            # Bridge from NumPy (or whatever non-tensor the caller
            # supplied; ``None`` falls back to ``y_true``) to torch,
            # matching the context's device AND dtype so a GPU/float64
            # loss_fn doesn't drift to CPU/float32.
            src = y_arr if target is None else np.asarray(target)
            tgt = _torch.as_tensor(src)
            ctx_device = getattr(context, "device", None)
            ctx_dtype = getattr(context, "dtype", None)
            if ctx_device is not None and tgt.device != ctx_device:
                tgt = tgt.to(ctx_device)
            if ctx_dtype is not None and tgt.dtype != ctx_dtype:
                tgt = tgt.to(ctx_dtype)
        return compute_pit_from_loss_fn(
            loss_fn,
            context=context,
            target=tgt,
            n_samples=n_samples,
        )

    # ----- Legacy dict-passthrough path (fast, pure NumPy) -----
    if "samples" in predictions:
        return _pit_from_samples(y_arr, predictions)

    if "mean" in predictions and "std" in predictions:
        return _pit_from_gaussian(y_arr, predictions)

    nig_keys = {"gamma", "nu", "alpha", "beta"}
    if nig_keys.issubset(predictions.keys()):
        return _pit_from_student_t(y_arr, predictions)

    if "lower" in predictions and "upper" in predictions:
        # Interval-only descriptors carry no full predictive distribution, so
        # no exact PIT exists.  The previous seeded-Uniform(0,1) placeholder
        # laundered *any* interval method — including catastrophically wrong
        # ones — into "perfectly calibrated" PIT diagnostics.  Use the honest
        # interval-position proxy instead: 0 = y at/below the lower bound,
        # 1 = y at/above the upper bound, relative position inside otherwise.
        # Well-calibrated intervals concentrate near-uniformly; miscalibrated
        # ones saturate and fail the uniformity tests.
        lower = np.asarray(predictions["lower"], dtype=float).reshape(-1)
        upper = np.asarray(predictions["upper"], dtype=float).reshape(-1)
        width = upper - lower
        if len(lower) != len(y_arr) or np.any(width <= 0):
            raise ValueError("interval predictions require upper > lower per test point")
        return np.clip((y_arr - lower) / width, 0.0, 1.0)

    return np.array([], dtype=np.float64)


def pit_uniformity_tests(
    pit_values: np.ndarray | list[float],
) -> dict[str, float]:
    """Statistical uniformity tests for PIT values.

    A well-calibrated model's PIT values are Uniform(0, 1), with:
        * Mean = 0.5
        * Std  = 1/sqrt(12) ≈ 0.2887

    Tests:
        * **PIT_Mean** — average PIT; 0.5 means no bias.  Values >0.5 mean
          predictions are systematically low (over-shoot upward); <0.5
          means predictions are systematically high.
        * **PIT_Std** — std of PIT; 0.2887 means perfect calibration.  Values
          <0.288 indicate overconfidence (PIT mass near 0 and 1); >0.288
          indicate underconfidence (PIT bunched near 0.5).
        * **PIT_KS** — Kolmogorov-Smirnov statistic against Uniform(0, 1).
          Sensitive to any deviation from uniformity.  Lower = better.
        * **PIT_CvM** — Cramér-von Mises statistic (ω²) against Uniform(0, 1).
          Tighter on near-uniform distributions than KS.  Lower = better.

    Returns
    -------
    dict[str, float]
        ``PIT_Mean``, ``PIT_Std``, ``PIT_KS``, ``PIT_CvM``.  NaN for empty input.
    """
    pit = np.asarray(pit_values, dtype=np.float64).ravel()
    n = len(pit)

    if n < 3:
        return {
            "PIT_Mean": float("nan"),
            "PIT_Std": float("nan"),
            "PIT_KS": float("nan"),
            "PIT_CvM": float("nan"),
        }

    # Clip to (0, 1) open interval to avoid log(0) in any future extensions
    pit_clipped = np.clip(pit, 1e-10, 1.0 - 1e-10)

    mean_pit = float(np.mean(pit_clipped))
    std_pit = float(np.std(pit_clipped, ddof=1))

    # KS: max vertical distance between sorted PIT (ECDF) and uniform CDF
    sorted_pit = np.sort(pit_clipped)
    cdf_uniform = (np.arange(1, n + 1) - 0.5) / n  # standard ECDF midpoints
    ks = float(np.max(np.abs(sorted_pit - cdf_uniform)))

    # CvM: Cramér-von Mises ω² statistic against Uniform(0, 1).
    # Standard textbook form:
    #     ω² = (1/(12n)) + Σ_{i=1}^n [F_n(x_{(i)}) − F(x_{(i)})]²
    # where F_n(x_{(i)}) = i/n is the empirical CDF at the i-th order
    # statistic and F(x_{(i)}) = x_{(i)} for a Uniform(0, 1) target.
    # Asymptotically the diff term vanishes as O(1/n); the (12n)^-1
    # term keeps ω² strictly positive for finite n reference comparisons.
    ecdf_i = np.arange(1, n + 1, dtype=np.float64) / n
    diffs = ecdf_i - sorted_pit
    cvm = float(np.sum(diffs**2) + 1.0 / (12 * n))

    return {
        "PIT_Mean": mean_pit,
        "PIT_Std": std_pit,
        "PIT_KS": ks,
        "PIT_CvM": cvm,
    }


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
    interval_score_val = float(np.mean((upper_np - lower_np) + (2.0 / alpha) * (below + above)))

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
    nll = float(
        np.mean(
            0.5 * np.log(2 * np.pi) + np.log(std_np) + 0.5 * ((y_true_np - mean_np) / std_np) ** 2
        )
    )

    # CRPS for Gaussian (closed form)
    from scipy.special import ndtr

    z = (y_true_np - mean_np) / std_np
    crps = float(
        np.mean(
            std_np
            * (
                z * (2.0 * ndtr(z) - 1.0)
                + 2.0 * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
                - 1.0 / np.sqrt(np.pi)
            )
        )
    )

    # Sharpness: mean predictive std (uncertainty-toolbox parity).
    sharpness = float(np.mean(std_np))

    return {"NLL": nll, "CRPS": crps, "Sharpness": sharpness}


def compute_calibration_metrics(
    pit_values: np.ndarray,
) -> dict[str, float]:
    """Compute calibration metrics from PIT (Probability Integral Transform) values.

    PIT values should be approximately Uniform(0, 1) for a well-calibrated model.
    Adds uncertainty-toolbox parity metrics:
    - ECE (legacy equal-mass binning, kept for back-compat)
    - MACE: mean absolute calibration error vs uniform (equal-width bins)
    - RMSCE: root-mean-square calibration error vs uniform
    """
    pit = np.asarray(pit_values).ravel()
    n = len(pit)

    # ECE via equal-mass binning
    n_bins = min(10, n // 10)
    if n_bins < 2:
        return {
            "ECE": float("nan"),
            "PIT_KS": float("nan"),
            "MACE": float("nan"),
            "RMSCE": float("nan"),
        }

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

    # MACE/RMSCE: equal-width bins on [0,1], |empirical mass - 1/B|.
    n_eq = 10
    hist, _ = np.histogram(pit, bins=n_eq, range=(0.0, 1.0))
    freq = hist.astype(float) / max(n, 1)
    dev = np.abs(freq - 1.0 / n_eq)
    mace = float(np.mean(dev))
    rmsce = float(np.sqrt(np.mean(dev**2)))

    return {"ECE": ece, "PIT_KS": ks, "MACE": mace, "RMSCE": rmsce}


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
            - ``gamma``+``nu``+``alpha``+``beta``: NIG → Student-t PIT
            - Calibration metrics (PIT_Mean, PIT_Std, PIT_KS, PIT_CvM) are
              added automatically for any prediction that supports PIT.
        y_true: Ground truth targets (numpy array).

    Returns:
        Flattened dictionary of all applicable metrics.
    """
    y_true_np = np.asarray(y_true).reshape(-1)
    metrics: dict[str, float] = {}

    if "y_pred" in predictions:
        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        metrics.update(compute_point_metrics(y_pred, y_true_np))
        # Tail-risk point metrics (library; default 90% upper tail).
        try:
            from torchregress.metrics import tail_mae, tail_rmse

            metrics["TailMAE"] = float(tail_mae(y_pred, y_true_np))
            metrics["TailRMSE"] = float(tail_rmse(y_pred, y_true_np))
        except Exception:
            pass

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
        # Selective-prediction quality: area under the risk-coverage curve
        # (library RiskCoverageCurve; lower = better uncertainty ranking).
        try:
            from torchregress.metrics import risk_coverage_curve

            aurc = risk_coverage_curve(mean, y_true_np, std).get("aurc")
            if aurc is not None:
                metrics["AURC"] = float(aurc.item() if hasattr(aurc, "item") else aurc)
        except Exception:
            pass

    if "samples" in predictions and "y_pred" not in predictions:
        samples = np.asarray(predictions["samples"])
        # Handle both (N, S) and (S, N) orientations — use the same logic
        # as compute_pit_values for consistency.
        n_test = len(y_true_np)
        if samples.ndim == 2 and samples.shape == (n_test, samples.shape[1]):
            # (N, S) — mean/std over samples-per-test-point (axis=1)
            mean = samples.mean(axis=1)
            std = samples.std(axis=1)
        elif samples.ndim == 2 and samples.shape[1] == n_test:
            # (S, N) — transpose, then mean over axis=1
            mean = samples.T.mean(axis=1)
            std = samples.T.std(axis=1)
        else:
            # Fallback: maintain legacy (S, N) behaviour
            if samples.ndim == 2:
                mean = samples.mean(axis=0)
                std = samples.std(axis=0)
            else:
                mean = samples.mean()
                std = samples.std()
        metrics.update(compute_point_metrics(mean, y_true_np))
        metrics.update(compute_distribution_metrics(mean, std, y_true_np))

    # --- PIT calibration diagnostics (added whenever supported) ---
    pit = compute_pit_values(predictions, y_true_np)
    if pit.size >= 3:
        pit_metrics = pit_uniformity_tests(pit)
        # Keep ECE parity for downstream metric_order consumers: re-use the
        # legacy calibration binning (still meaningful for PIT diagnostics).
        try:
            cal = compute_calibration_metrics(pit)
            pit_metrics["ECE"] = float(cal.get("ECE", float("nan")))
            pit_metrics["MACE"] = float(cal.get("MACE", float("nan")))
            pit_metrics["RMSCE"] = float(cal.get("RMSCE", float("nan")))
        except Exception:
            pit_metrics["ECE"] = float("nan")
            pit_metrics["MACE"] = float("nan")
            pit_metrics["RMSCE"] = float("nan")
        metrics.update(pit_metrics)

    # --- Density-grid diagnostics (CDE loss + HPD coverage) ---
    # Rows emitting a 1-D predictive density on a support grid
    # (binned-PDF ensembles, flow grids) get proper density scores.
    if "density_support" in predictions and "density_values" in predictions:
        try:
            from torchregress.metrics import (
                conditional_density_estimation_loss as _cde,
            )
            from torchregress.metrics import (
                highest_posterior_density_coverage as _hpd,
            )

            support = np.asarray(predictions["density_support"], dtype=np.float64)
            dens = np.asarray(predictions["density_values"], dtype=np.float64)
            if dens.ndim == 1:
                dens = dens[None, :]
            if dens.shape[0] == len(y_true_np) and support.ndim == 1:
                metrics["CDELoss"] = float(_cde(support, dens, y_true_np))
                metrics["HPDCoverage"] = float(_hpd(support, dens, y_true_np))
        except Exception:
            pass

    return metrics
