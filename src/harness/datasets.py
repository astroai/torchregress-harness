"""Dataset registry for the harness.

Provides a unified API for loading known regression datasets — both real
tabular benchmarks and synthetic stress tests, plus image regression datasets.
Every entry returns ``(X_train, X_test, y_train, y_test)`` as numpy arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SplitArrays(TypedDict):
    """The four canonical arrays every dataset loader returns."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class SplitMetadata(TypedDict, total=False):
    """Optional auxiliary arrays and piggybacked loader/scenario metadata.

    Loaders and :func:`load_dataset` attach metadata using leading-underscore
    keys (e.g. ``_synthetic_fallback``); consumers read them with ``dict.get``.
    """

    X_train_true: np.ndarray
    X_test_true: np.ndarray
    _y_true_sigma: float
    censoring_train: np.ndarray
    censoring_test: np.ndarray
    exposure_train: np.ndarray
    exposure_test: np.ndarray
    sigma_x: float
    sigma_y: float
    n_classes: int
    _multi_output: bool
    _skip_y_standardization: bool
    _synthetic_fallback: bool
    _fallback_reason: str
    _image_data: bool
    _image_size: int
    _n_channels: int
    _scenario: str
    _source_url: str
    _alpha: float
    _calibration_fraction: float
    _dense_output: bool
    _y_scaler: StandardScaler


class SplitsDict(SplitArrays, SplitMetadata):
    """Full splits dict: the four canonical arrays plus optional extras."""


# ---------------------------------------------------------------------------
# Dataset descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a registered dataset."""

    name: str
    category: str  # "tabular", "image", "synthetic"
    n_features: int
    n_samples: int
    description: str
    loader: Callable[..., "SplitsDict"]


# ---------------------------------------------------------------------------
# Real tabular datasets
# ---------------------------------------------------------------------------


def _load_california_housing(
    n_samples: int = 5000, test_size: float = 0.3, seed: int = 42
) -> SplitsDict:
    """Load California Housing dataset."""
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    X, y = data.data, data.target

    if n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
    }


def _load_diabetes(
    n_samples: int = 442,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Load sklearn Diabetes dataset."""
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)

    if n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
    }


def _load_openml_year(
    data_id: int | None = None,
    n_samples: int = 10000,
    test_size: float = 0.3,
    seed: int = 42,
    cache_path: str | None = None,
) -> SplitsDict:
    """Load OpenML YearPredictionMSD dataset.

    Falls back to synthetic data if OpenML fetch fails.
    """
    splits: SplitMetadata = {}
    try:
        from sklearn.datasets import fetch_openml

        data = fetch_openml(data_id=data_id or 44031, parser="pandas", as_frame=True)
        df = data.frame
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col]).values.astype(np.float64)
        y = df[target_col].values.astype(np.float64)
    except Exception:
        # Fallback: synthetic data with similar dimensions (DATA-01/02:
        # provenance flags let result rows disclose non-real data).
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((515345, 90)).astype(np.float64)
        y = 0.7 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.3, 515345).astype(np.float64)
        splits["_synthetic_fallback"] = True
        splits["_fallback_reason"] = "OpenML YearPredictionMSD fetch failed"

    if n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        **splits,
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Synthetic tabular datasets
# ---------------------------------------------------------------------------


def _load_synthetic_heteroscedastic(
    n_samples: int = 2000,
    n_features: int = 8,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic heteroscedastic data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y_mean = 0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * np.sin(1.6 * X[:, 2])
    noise_std = 0.15 + 0.25 * np.abs(X[:, 0])
    y = (y_mean + noise_std * rng.standard_normal(n_samples)).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _load_synthetic_outliers(
    n_samples: int = 2000,
    n_features: int = 8,
    outlier_fraction: float = 0.15,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic data with outliers."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (
        2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] ** 2 + 0.3 * rng.standard_normal(n_samples)
    ).astype(np.float32)

    n_outliers = int(n_samples * outlier_fraction)
    outlier_idx = rng.choice(n_samples, n_outliers, replace=False)
    y[outlier_idx] += rng.uniform(-5, 5, n_outliers).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _load_synthetic_tweedie(
    n_samples: int = 2000,
    n_features: int = 6,
    p_power: float = 1.5,
    phi: float = 0.6,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic compound Poisson-Gamma (Tweedie) data."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(n_samples, n_features)).astype(np.float32)
    log_mu = 0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    mu = np.exp(log_mu).astype(np.float32)

    shape = (2.0 - p_power) / (p_power - 1.0)
    y = np.zeros(n_samples, dtype=np.float32)
    lam = mu ** (2.0 - p_power) / (phi * (2.0 - p_power))
    scale = phi * (p_power - 1.0) * mu ** (p_power - 1.0)
    for i in range(n_samples):
        n_events = int(rng.poisson(lam[i]) if lam[i] > 0 else 0)
        if n_events > 0:
            y[i] = float(rng.gamma(shape, scale[i], size=n_events).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_skip_y_standardization": True,  # preserve zero-inflated Tweedie distribution
    }


def _load_synthetic_censored(
    n_samples: int = 2000,
    n_features: int = 8,
    censoring_rate: float = 0.3,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic data with censored targets."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y_true = (
        1.0 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2] + 0.2 * rng.standard_normal(n_samples)
    ).astype(np.float32)

    # Right-censor: observed = min(true, threshold)
    threshold = float(np.quantile(y_true, 1.0 - censoring_rate))
    y_observed = np.minimum(y_true, threshold).astype(np.float32)
    censoring_codes = (y_true > threshold).astype(np.float32)  # 1 = right-censored

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_observed, test_size=test_size, random_state=seed
    )
    _, _, codes_train, codes_test = train_test_split(
        X, censoring_codes, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "censoring_train": codes_train,
        "censoring_test": codes_test,
        "_skip_y_standardization": True,  # preserve censoring threshold semantics
    }


def _load_synthetic_multivariate(
    n_samples: int = 2000,
    n_features: int = 8,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic data with 3 correlated output dimensions.

    Targets are correlated: y1 depends on y0, y2 depends on both y0 and y1.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    y0 = (
        0.7 * X[:, 0]
        - 0.5 * X[:, 1]
        + 0.3 * np.sin(1.6 * X[:, 2])
        + 0.15 * rng.standard_normal(n_samples)
    ).astype(np.float32)
    y1 = (0.5 * y0 + 0.4 * X[:, 3] - 0.3 * X[:, 4] + 0.2 * rng.standard_normal(n_samples)).astype(
        np.float32
    )
    y2 = (
        -0.3 * y0 + 0.4 * y1 + 0.6 * X[:, 5] - 0.2 * X[:, 6] + 0.25 * rng.standard_normal(n_samples)
    ).astype(np.float32)

    y = np.column_stack([y0, y1, y2]).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_multi_output": True,
        "_skip_y_standardization": True,  # preserve (N, D) shape for multi-output
    }


def _load_synthetic_mv_gaussian(
    n_samples: int = 2000,
    n_features: int = 8,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic 3D targets from a purely multivariate Gaussian.

    y | X ~ N(μ(X), Σ) where Σ is a fixed covariance matrix with
    cross-dimension correlations.  This is the correctly-specified
    setting for GaussianNLL and MultivariateGaussian — they should
    achieve optimal IntervalScore here.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # Conditional mean depends on X
    mu0 = 0.7 * X[:, 0] - 0.5 * X[:, 1]
    mu1 = 0.4 * X[:, 2] - 0.3 * X[:, 3]
    mu2 = 0.6 * X[:, 4] - 0.2 * X[:, 5]
    mu = np.column_stack([mu0, mu1, mu2])

    # Fixed covariance with correlations
    cov = np.array(
        [
            [1.0, 0.6, -0.3],
            [0.6, 0.8, 0.2],
            [-0.3, 0.2, 1.2],
        ],
        dtype=np.float32,
    )
    L = np.linalg.cholesky(cov)

    y = (mu + rng.standard_normal((n_samples, 3)) @ L.T).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_multi_output": True,
        "_skip_y_standardization": True,
    }


def _load_synthetic_mv_nongaussian(
    n_samples: int = 2000,
    n_features: int = 8,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic 3D targets with non-Gaussian conditional distributions.

    y0: μ₀(X) + t₃(ν=4) noise  — heavy-tailed
    y1: μ₁(X) + χ²(3) noise    — right-skewed
    y2: μ₂(X) + bimodal        — 0.5·N(-1,0.3²) + 0.5·N(1,0.3²)

    This is the *misspecified* setting for Gaussian methods — the true
    conditional p(y|X) is non-Gaussian, so NormalizingFlow and SLS
    should outperform.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # Conditional means (same structure as Gaussian dataset for fair comparison)
    mu0 = 0.7 * X[:, 0] - 0.5 * X[:, 1]
    mu1 = 0.4 * X[:, 2] - 0.3 * X[:, 3]
    mu2 = 0.6 * X[:, 4] - 0.2 * X[:, 5]

    # Non-Gaussian noise — each dimension scaled to unit variance
    #   t₄ variance = ν/(ν-2) = 2   → scale = 1/√2 ≈ 0.707
    #   χ²(3) variance = 2·ν = 6    → scale = 1/√6 ≈ 0.408
    #   bimodal already ≈ 1.09 → close enough, leave unscaled
    y0 = mu0 + rng.standard_t(df=4, size=n_samples).astype(np.float32) * 0.707
    y1 = mu1 + (rng.chisquare(df=3, size=n_samples) - 3).astype(np.float32) * 0.408
    bimodal_comp = rng.choice(2, size=n_samples)
    bimodal_noise = np.where(
        bimodal_comp == 0,
        rng.normal(-1.0, 0.3, n_samples),
        rng.normal(1.0, 0.3, n_samples),
    ).astype(np.float32)
    y2 = mu2 + bimodal_noise

    y = np.column_stack([y0, y1, y2]).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_multi_output": True,
        "_skip_y_standardization": True,
    }


def _load_synthetic_mv_bimodal(
    n_samples: int = 2000,
    n_features: int = 8,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic 3D targets from a mixture of two multivariate Gaussians.

    y | X ~ 0.5·N(μ₁(X), Σ₁) + 0.5·N(μ₂(X), Σ₂)

    The two components have different mean offsets and covariance matrices,
    creating a bimodal conditional distribution.  Gaussian methods (GaussianNLL,
    MultivariateGaussian) can only fit a single Gaussian — they will place their
    mean between the two modes and inflate variance.  NormalizingFlow and SLS
    should capture the bimodality and produce tighter, better-calibrated intervals.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    # Shared base mean (identical to other MV datasets for comparability)
    mu0 = 0.7 * X[:, 0] - 0.5 * X[:, 1]
    mu1 = 0.4 * X[:, 2] - 0.3 * X[:, 3]
    mu2 = 0.6 * X[:, 4] - 0.2 * X[:, 5]
    mu_base = np.column_stack([mu0, mu1, mu2])

    # Two modes with distinct means and covariances
    # 3.5σ separation → <1% overlap, definitive stress test
    offset = np.array([3.5, -2.5, 2.5], dtype=np.float32)
    mu_a = mu_base
    mu_b = mu_base + offset[np.newaxis, :]

    # Different covariance per component
    cov_a = np.array(
        [
            [1.0, 0.3, 0.0],
            [0.3, 0.6, 0.0],
            [0.0, 0.0, 0.8],
        ],
        dtype=np.float32,
    )
    cov_b = np.array(
        [
            [0.5, -0.2, 0.1],
            [-0.2, 0.7, 0.0],
            [0.1, 0.0, 0.6],
        ],
        dtype=np.float32,
    )

    L_a = np.linalg.cholesky(cov_a)
    L_b = np.linalg.cholesky(cov_b)

    # Sample from mixture
    comp = rng.choice(2, size=n_samples, p=[0.5, 0.5])
    noise = rng.standard_normal((n_samples, 3))
    y = np.empty((n_samples, 3), dtype=np.float32)
    mask_a = comp == 0
    mask_b = comp == 1
    y[mask_a] = mu_a[mask_a] + noise[mask_a] @ L_a.T
    y[mask_b] = mu_b[mask_b] + noise[mask_b] @ L_b.T
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_multi_output": True,
        "_skip_y_standardization": True,
    }


# ---------------------------------------------------------------------------
# Conditional flow benchmark datasets (pzflow parity)
# ---------------------------------------------------------------------------


def _load_synthetic_cond_gaussian(
    n_samples: int = 2000,
    n_features: int = 5,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate 1D target with Gaussian conditional distribution.

    y | X ~ N(μ(X), σ²)  where μ(X) = X·β and σ = 0.3.

    This is the correctly-specified setting for all flow methods —
    the true conditional is exactly a Normal distribution, so flows
    should converge to the theoretical NLL lower bound.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    beta = np.array([1.0, -2.0, 0.5, 1.5, -0.8], dtype=np.float32)
    mu = X @ beta
    sigma = 0.3
    y = (mu + sigma * rng.standard_normal(n_samples)).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_y_true_sigma": sigma,  # for computing theoretical NLL lower bound
    }


def _load_synthetic_cond_nongaussian(
    n_samples: int = 2000,
    n_features: int = 5,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate 1D target with non-Gaussian conditional distribution.

    y | X = μ(X) + ε  where ε are i.i.d. draws from a 3-component
    Gaussian mixture: 0.3·N(-0.8, 0.15²) + 0.4·N(0.2, 0.3²) + 0.3·N(0.9, 0.12²).

    The mixture has mean ≈ 0, variance ≈ 1 (after scaling).
    This is *misspecified* for diagonal Gaussian — flows should capture
    the multi-modal conditional density.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    beta = np.array([1.0, -2.0, 0.5, 1.5, -0.8], dtype=np.float32)
    mu = X @ beta

    # 3-component Gaussian mixture noise (unit variance after scaling)
    comp = rng.choice(3, size=n_samples, p=[0.3, 0.4, 0.3])
    eps = np.where(
        comp == 0,
        rng.normal(-0.8, 0.15, n_samples),
        np.where(
            comp == 1,
            rng.normal(0.2, 0.3, n_samples),
            rng.normal(0.9, 0.12, n_samples),
        ),
    ).astype(np.float32)
    # Scale to unit variance
    eps = eps / np.std(eps).item()
    y = (mu + eps).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _load_synthetic_eiv(
    n_samples: int = 2000,
    n_features: int = 6,
    sigma_x: float = 0.2,
    sigma_y: float = 0.1,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic data with errors-in-variables (noisy inputs)."""
    rng = np.random.default_rng(seed)
    X_true = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    X_obs = (X_true + sigma_x * rng.standard_normal((n_samples, n_features))).astype(np.float32)
    y = (
        0.8 * X_true[:, 0]
        - 0.6 * X_true[:, 1]
        + 0.4 * X_true[:, 2]
        + sigma_y * rng.standard_normal(n_samples)
    ).astype(np.float32)

    X_train_obs, X_test_obs, y_train, y_test = train_test_split(
        X_obs, y, test_size=test_size, random_state=seed
    )
    X_train_true, X_test_true, _, _ = train_test_split(
        X_true, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train_obs,
        "X_test": X_test_obs,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_true": X_train_true,
        "X_test_true": X_test_true,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
    }


def _load_synthetic_eiv_nonlinear(
    n_samples: int = 2000,
    n_features: int = 6,
    sigma_x: float = 0.5,
    sigma_y: float = 0.1,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic EIV data with nonlinear ground truth.

    The true relationship is sinusoidal — a linear model (TLS / OLS)
    cannot fit it, forcing neural EIV methods to earn their keep.

    Ground truth:  y = sin(2π·x₀) + 0.5·cos(4π·x₁) + 0.3·tanh(2·x₂) + ε
    where ε ∼ 𝒩(0, σy²).  Features 3..n are noise.
    """
    rng = np.random.default_rng(seed)
    X_true = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    X_obs = (X_true + sigma_x * rng.standard_normal((n_samples, n_features))).astype(np.float32)
    y = (
        np.sin(2.0 * np.pi * X_true[:, 0])
        + 0.5 * np.cos(4.0 * np.pi * X_true[:, 1])
        + 0.3 * np.tanh(2.0 * X_true[:, 2])
        + sigma_y * rng.standard_normal(n_samples).astype(np.float32)
    )

    X_train_obs, X_test_obs, y_train, y_test = train_test_split(
        X_obs, y, test_size=test_size, random_state=seed
    )
    X_train_true, X_test_true, _, _ = train_test_split(
        X_true, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train_obs,
        "X_test": X_test_obs,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_true": X_train_true,
        "X_test_true": X_test_true,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
    }


# ---------------------------------------------------------------------------
# Upstream canonical datasets (from external library documentation)
# ---------------------------------------------------------------------------


def _load_insurance_claims(
    n_samples: int = 20000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Load the French Motor Insurance Claims dataset (OpenML ID 41214).

    This is the canonical sklearn Tweedie regression dataset — used in
    ``sklearn.linear_model.TweedieRegressor`` documentation and examples.
    Falls back to synthetic Tweedie data if OpenML fetch fails.

    Canonical source:
    https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
    """
    splits: SplitMetadata = {}
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        df = fetch_openml(data_id=41214, parser="auto", as_frame=True)
        data = df.frame if hasattr(df, "frame") else df.data

        # Extract target and exposure column
        target_col = "ClaimNb"
        exposure_col = "Exposure"

        # Separate numerical and categorical features
        drop_cols = {"IDpol", target_col, exposure_col}
        feature_cols = [c for c in data.columns if c not in drop_cols]

        # Identify categorical columns (object-typed in pandas)
        categorical_cols = [
            c for c in feature_cols if data[c].dtype == object or data[c].dtype.name == "category"
        ]
        numerical_cols = [c for c in feature_cols if c not in categorical_cols]

        # Subset
        if n_samples < len(data):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(data), n_samples, replace=False)
            data = data.iloc[idx]

        y = data[target_col].values.astype(np.float64)
        exposure = data[exposure_col].values.astype(np.float64)

        # One-hot encode categorical, concatenate numerical
        X_num = data[numerical_cols].values.astype(np.float64)
        if categorical_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_cat = ohe.fit_transform(data[categorical_cols])
            X = np.hstack([X_num, X_cat])
        else:
            X = X_num

    except Exception:
        # Fallback: synthetic Tweedie with insurance-claim-like characteristics
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1.5, 1.5, size=(n_samples, 9)).astype(np.float64)
        lin = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
        mu = np.exp(lin)
        y = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            lam = mu[i] ** (0.5) / (0.8 * 0.5)
            n_events = int(rng.poisson(lam) if lam > 0 else 0)
            if n_events > 0:
                scale = 0.8 * 0.5 * mu[i] ** 0.5
                y[i] = float(rng.gamma(2.0, scale, size=n_events).sum())
        exposure = np.ones_like(y)
        splits["_synthetic_fallback"] = True
        splits["_fallback_reason"] = "OpenML French Motor Claims fetch failed"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    _, _, exp_train, exp_test = train_test_split(
        X, exposure, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        **splits,
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "exposure_train": exp_train.astype(np.float32),
        "exposure_test": exp_test.astype(np.float32),
        "_skip_y_standardization": True,  # Tweedie claim counts — preserve zero-inflation
    }


def _load_house_sales(
    n_samples: int = 20000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Load the King County House Sales dataset (OpenML ``house_sales``).

    This is the canonical crepes conformal prediction example dataset.
    Falls back to California Housing if OpenML fetch fails.

    Canonical source:
    https://crepes.readthedocs.io/en/latest/ (examples / quick-start notebooks)
    """
    try:
        from sklearn.datasets import fetch_openml

        data = fetch_openml(name="house_sales", version=3, parser="auto", as_frame=True)
        df = data.frame if hasattr(data, "frame") else data.data

        # Target is expected to be named "price" (crepes canonical name).
        # Fall back to California Housing if not found.
        target_col = None
        for cand in ["price", "SalePrice"]:
            if cand in df.columns:
                target_col = cand
                break
        if target_col is None:
            raise KeyError(
                "house_sales target column not found — falling back to California Housing"
            )

        numeric_df = df.select_dtypes(include=[np.number])
        X = numeric_df.drop(columns=[target_col]).values.astype(np.float64)
        y = numeric_df[target_col].values.astype(np.float64)
    except Exception:
        # Fallback: California Housing substitute (flagged so result rows
        # disclose that the canonical crepes dataset was unavailable).
        splits = _load_california_housing(n_samples=n_samples, test_size=test_size, seed=seed)
        splits["_synthetic_fallback"] = True
        splits["_fallback_reason"] = (
            "OpenML house_sales unavailable; substituted California Housing"
        )
        return splits

    if n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
    }


def _load_make_regression_synthetic(
    n_samples: int = 5000,
    n_features: int = 10,
    n_informative: int = 6,
    noise: float = 15.0,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate data via sklearn's ``make_regression``.

    This is the canonical sklearn synthetic regression benchmark — used
    across most sklearn estimator documentation examples.

    Canonical source:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
    """
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
    }


def _load_synthetic_ordinal(
    n_samples: int = 2000,
    n_features: int = 6,
    n_classes: int = 5,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """Synthetic ordinal regression data.

    Generates continuous targets as a linear combination of features
    plus noise, then discretises into ``n_classes`` ordinal bins
    using quantile-based thresholds.  Returns class indices 0..K-1.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # True continuous target
    y_cts = (
        0.8 * X[:, 0]
        - 0.6 * X[:, 1]
        + 0.4 * X[:, 2]
        + 0.3 * rng.standard_normal(n_samples).astype(np.float32)
    )
    # Discretise into ordinal classes using quantile thresholds
    thresholds = np.quantile(y_cts, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.digitize(y_cts, thresholds).astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "n_classes": n_classes,
        "_skip_y_standardization": True,
    }


def _load_synthetic_sinusoidal(
    n_samples: int = 2000,
    n_features: int = 4,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate sinusoidal test-function data with homoscedastic Gaussian noise.

    Target:  y = sin(2π⋅x₀) + 0.3⋅cos(4π⋅x₁) + 0.2⋅tanh(3⋅x₂) + ε
    where ε ∼ 𝒩(0, 0.15²).  The remaining features are noise.

    This mirrors the low-dimensional synthetic test functions used in
    BoTorch GP regression tutorials (Branin / sinusoidal style).

    Canonical source:
    https://botorch.org/tutorials/fit_model_with_torch_optimizer
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n_samples, n_features)).astype(np.float32)

    y_clean = (
        np.sin(2.0 * np.pi * X[:, 0])
        + 0.3 * np.cos(4.0 * np.pi * X[:, 1])
        + 0.2 * np.tanh(3.0 * X[:, 2])
    )
    y = (y_clean + 0.15 * rng.standard_normal(n_samples)).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Canonical scenario references
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalScenario:
    """A canonical benchmarking scenario from an upstream library's own docs.

    When a suite declares support for a canonical scenario, the harness
    automatically loads the correct dataset with the library's own split
    parameters, alpha level, and calibration fraction — reproducing the
    upstream example exactly rather than inventing new test conditions.
    """

    scenario_name: str
    source_library: str
    dataset_key: str
    source_url: str
    description: str

    # --- Structured kwargs merged into load_dataset() call ---
    loader_kwargs: dict[str, Any] | None = None
    #   e.g. {"n_samples": 20640, "test_size": 0.2}

    # --- Conformal / calibration scenario parameters ---
    alpha: float | None = None
    #   e.g. 0.1 for 90% prediction intervals (MAPIE convention)

    calibration_fraction: float | None = None
    #   e.g. 0.2 = 20% of training data carved off for calibration


CANONICAL_SCENARIOS: dict[str, CanonicalScenario] = {
    "mapie_reference": CanonicalScenario(
        scenario_name="mapie_reference",
        source_library="MAPIE",
        dataset_key="california_housing",
        source_url="https://mapie.readthedocs.io/en/latest/examples_regression/1-tutorial.html",
        description=(
            "MAPIE CQR tutorial: LGBMRegressor base on California Housing "
            "with 80/20 split and α=0.1."
        ),
        loader_kwargs={"n_samples": 16512, "test_size": 0.2},
        alpha=0.1,
        calibration_fraction=0.2,
    ),
    "crepes_reference": CanonicalScenario(
        scenario_name="crepes_reference",
        source_library="crepes",
        dataset_key="house_sales",
        source_url="https://crepes.readthedocs.io/en/latest/",
        description=(
            "crepes quick-start: RandomForestRegressor with DifficultyEstimator on house_sales."
        ),
        loader_kwargs={"n_samples": 20000, "test_size": 0.3},
        alpha=0.1,
        calibration_fraction=0.3,
    ),
    "sklearn_tweedie_reference": CanonicalScenario(
        scenario_name="sklearn_tweedie_reference",
        source_library="scikit-learn",
        dataset_key="insurance_claims",
        source_url="https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html",
        description=(
            "sklearn TweedieRegressor example: French Motor Insurance Claims with p=1.5 log-link."
        ),
        loader_kwargs={"n_samples": 20000, "test_size": 0.3},
    ),
    "sklearn_synthetic_reference": CanonicalScenario(
        scenario_name="sklearn_synthetic_reference",
        source_library="scikit-learn",
        dataset_key="make_regression_synthetic",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html",
        description="sklearn make_regression: 5K samples, 10 features (6 informative), noise=15.",
        loader_kwargs={"n_samples": 5000, "test_size": 0.25},
    ),
    "sklearn_diabetes_reference": CanonicalScenario(
        scenario_name="sklearn_diabetes_reference",
        source_library="scikit-learn",
        dataset_key="diabetes",
        source_url="https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html",
        description="sklearn OLS tutorial: load_diabetes with standard 70/30 split.",
        loader_kwargs={"test_size": 0.3},
    ),
    "botorch_reference": CanonicalScenario(
        scenario_name="botorch_reference",
        source_library="BoTorch",
        dataset_key="synthetic_sinusoidal",
        source_url="https://botorch.org/tutorials/fit_model_with_torch_optimizer",
        description=(
            "BoTorch SingleTaskGP tutorial: sinusoidal test function "
            "with 2-4D input and homoscedastic noise."
        ),
        loader_kwargs={"n_samples": 2000, "test_size": 0.2},
    ),
}


# ---------------------------------------------------------------------------
# TabReD industrial-scale datasets
#   https://github.com/yandex-research/tabred
#
# TabReD requires preprocessing scripts from the upstream repository.
# Each loader tries to read from a local data directory; if the data isn't
# available, a synthetic dataset with matching dimensions is generated.
# ---------------------------------------------------------------------------


def _load_tabred(
    dataset_id: str,
    n_rows: int,
    n_features: int,
    *,
    data_dir: str = "datasets/tabred",
    n_samples: int = 50000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generic TabReD loader: try local data, fall back to synthetic."""
    import pandas as pd

    csv_path = Path(data_dir) / f"{dataset_id}.csv"
    splits: SplitMetadata = {}

    # Existence check stays OUTSIDE the parse handler: a missing CSV is an
    # expected fallback trigger, not a parsing failure to swallow (DATA-02).
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            target_col = df.columns[-1]
            X = df.drop(columns=[target_col]).values.astype(np.float64)
            y = df[target_col].values.astype(np.float64)
        except (pd.errors.ParserError, OSError):
            X = None  # unreadable local file — fall back to synthetic below
    else:
        X = None

    if X is None:
        # Fallback: synthetic data with TabReD dimensions
        splits["_synthetic_fallback"] = True
        splits["_fallback_reason"] = f"TabReD {dataset_id} CSV missing or unreadable"
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_rows, n_features)).astype(np.float64)
        # First 4 features are informative (if they exist); rest are noise
        feat_idx = [0, 1, 2, min(3, n_features - 1)] if n_features >= 4 else list(range(n_features))
        coefs = [0.6, -0.4, 0.3, 0.2][: len(feat_idx)]
        y = sum(c * X[:, i] for c, i in zip(coefs, feat_idx))
        y = (y + 0.1 * rng.standard_normal(n_rows)).astype(np.float64)

    if n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        **splits,
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Image dataset cache helpers
# ---------------------------------------------------------------------------


def _image_cache_dir() -> Path:
    """Return the harness image dataset cache directory."""
    cache = Path(
        os.environ.get(
            "TORCHREGRESS_HARNESS_CACHE",
            os.path.expanduser("~/.cache/torchregress-harness"),
        )
    )
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_with_progress(url: str, dest: Path, desc: str = "") -> None:
    """Download a file from *url* to *dest* with a tqdm progress bar.

    Skips the download if *dest* already exists and has non-zero size.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return

    try:
        import requests  # ty: ignore[unresolved-import]  # optional dep, imported lazily with a friendly ImportError
    except ImportError:
        raise ImportError(
            "requests is required to download image datasets. Install with: uv pip install requests"
        )

    from tqdm import tqdm

    print(f"Downloading {desc or url.split('/')[-1]} …")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=desc or "download") as pbar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


# ---------------------------------------------------------------------------
# Image regression datasets
# ---------------------------------------------------------------------------


def _load_synthetic_image_regression(
    n_samples: int = 1000,
    img_size: int = 32,
    n_channels: int = 1,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic image regression data.

    Each image contains a 2D Gaussian blob whose centre coordinates
    (cx, cy) determine the scalar regression target.

    Target: y = cx / img_size + 0.5 * cy / img_size + noise

    The image is (n_channels, img_size, img_size).  For multi-channel,
    all channels contain the same blob (useful for testing 3-channel RGB
    pipelines without requiring actual colour data).
    """
    rng = np.random.default_rng(seed)

    # Random blob parameters
    cx = rng.uniform(img_size * 0.15, img_size * 0.85, n_samples).astype(np.float32)
    cy = rng.uniform(img_size * 0.15, img_size * 0.85, n_samples).astype(np.float32)
    sigma = rng.uniform(2.0, 6.0, n_samples).astype(np.float32)
    amplitude = rng.uniform(0.6, 1.4, n_samples).astype(np.float32)

    # Build coordinate grid
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)

    images = np.zeros((n_samples, n_channels, img_size, img_size), dtype=np.float32)
    for i in range(n_samples):
        blob = amplitude[i] * np.exp(
            -((xs - cx[i]) ** 2 + (ys - cy[i]) ** 2) / (2.0 * sigma[i] ** 2)
        )
        for c in range(n_channels):
            images[i, c] = blob

    # Target: normalised centre coordinates + noise
    y_clean = cx / img_size + 0.5 * cy / img_size
    y = (y_clean + 0.05 * rng.standard_normal(n_samples)).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        images, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": n_channels,
    }


def _load_synthetic_image_depth(
    n_samples: int = 600,
    img_size: int = 64,
    n_channels: int = 3,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Generate synthetic depth-estimation data.

    Each input is an RGB image with a simple geometric scene (gradient
    background + coloured rectangle).  The output is a single-channel
    depth map (img_size, img_size) where each pixel's value is its
    distance from a virtual camera.

    Target shape: (N, img_size, img_size) — per-pixel depth values.

    The ground-truth depth is a simple planar ramp plus a rectangular
    foreground object at a different depth.
    """
    rng = np.random.default_rng(seed)

    # Input images: gradient sky + coloured foreground rectangle
    X = np.zeros((n_samples, n_channels, img_size, img_size), dtype=np.float32)
    y_depth = np.zeros((n_samples, img_size, img_size), dtype=np.float32)

    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)

    for i in range(n_samples):
        # Background gradient (simulates sky/floor)
        sky_colour = rng.uniform(0.3, 0.8, 3).astype(np.float32)
        horizon = rng.integers(int(img_size * 0.3), int(img_size * 0.7))
        for c in range(n_channels):
            grad = np.where(
                ys < horizon,
                sky_colour[c] * (1.0 - ys / horizon),
                sky_colour[c] * 0.3,
            )
            X[i, c] = grad

        # Foreground rectangle
        rect_x = rng.integers(img_size // 4, 3 * img_size // 4)
        rect_y = rng.integers(img_size // 3, 3 * img_size // 4)
        rect_w = rng.integers(img_size // 6, img_size // 4)
        rect_h = rng.integers(img_size // 6, img_size // 4)
        rect_colour = rng.uniform(0.1, 0.9, 3).astype(np.float32)
        x0, x1 = max(0, rect_x - rect_w // 2), min(img_size, rect_x + rect_w // 2)
        y0, y1 = max(0, rect_y - rect_h // 2), min(img_size, rect_y + rect_h // 2)
        for c in range(n_channels):
            X[i, c, y0:y1, x0:x1] = rect_colour[c]

        # Ground-truth depth: planar ramp + foreground offset
        base_depth = 0.3 + 0.5 * (ys / img_size).astype(np.float32)
        depth = base_depth.copy()
        depth[y0:y1, x0:x1] = (0.15 + 0.15 * rng.standard_normal((y1 - y0, x1 - x0))).astype(
            np.float32
        )
        y_depth[i] = depth

    X = X.astype(np.float32)
    y_depth = y_depth.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_depth, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": n_channels,
        "_dense_output": True,
        "_skip_y_standardization": True,  # dense (N, H, W) targets
    }


# ---------------------------------------------------------------------------
# Real image dataset loaders (NYU Depth v2, IMDB-WIKI)
# ---------------------------------------------------------------------------

_NYU_DEPTH_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
# Official NYU mirror (used when the horatio host is unreachable).
_NYU_DEPTH_MIRROR_URLS = (
    "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat",
)

_IMDB_CROP_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"
_WIKI_CROP_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"
_IMDB_META_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar"
_WIKI_META_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_meta.tar"
# HuggingFace mirror fallback for IMDB-WIKI metadata (ETH mirror 404s
# wiki_meta.tar as of 2026-08 — PARITY_REPORT header). Tars already cached
# locally are reused without any download.
_IMDB_WIKI_HF_MIRROR = "https://huggingface.co/datasets/ashvala/imdb-wiki/resolve/main"


def _load_nyu_depth_v2(
    n_samples: int = 1449,
    img_size: int = 128,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """Load the NYU Depth v2 labeled dataset.

    Downloads the official ~2.7 GB MATLAB file, caches it locally,
    extracts RGB images and depth maps, resizes them to ``img_size``,
    and returns train/test splits.

    Parameters
    ----------
    n_samples : int
        Number of samples to load (max 1449).  1449 by default.
    img_size : int
        Target square size for resized images (RGB) and depth maps.
        Original is 480×640; resize makes training tractable.
    test_size : float
        Fraction of data reserved for testing.
    seed : int
        Random seed for the train/test split.

    Returns
    -------
    splits : dict
        ``X_train`` (N_train, 3, img_size, img_size) float32 RGB,
        ``X_test`` (N_test, 3, img_size, img_size) float32 RGB,
        ``y_train`` (N_train, img_size, img_size) float32 depth in metres,
        ``y_test`` (N_test, img_size, img_size) float32 depth in metres.
    """
    cache_dir = _image_cache_dir()
    mat_path = cache_dir / "nyu_depth_v2_labeled.mat"

    # Download if needed — try each mirror in order; cached files are reused.
    if not (mat_path.exists() and mat_path.stat().st_size > 0):
        last_exc: Exception | None = None
        for url in _NYU_DEPTH_MIRROR_URLS:
            try:
                _download_with_progress(url, mat_path, desc="NYU Depth v2 (.mat)")
                break
            except Exception as exc:  # noqa: BLE001 — try next mirror
                last_exc = exc
                if mat_path.exists():
                    try:
                        mat_path.unlink()
                    except OSError:
                        pass
        else:
            raise RuntimeError(
                "NYU Depth v2 download failed from all mirrors "
                f"({_NYU_DEPTH_MIRROR_URLS}). Last error: {last_exc}. "
                f"Place nyu_depth_v2_labeled.mat in {cache_dir} manually."
            ) from last_exc

    # Load .mat file
    try:
        import h5py  # ty: ignore[unresolved-import]  # optional dep, imported lazily with a friendly ImportError
    except ImportError:
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError(
                "h5py or scipy is required to load NYU Depth v2 (.mat format). "
                "Install with: uv pip install h5py"
            )
        mat = sio.loadmat(str(mat_path))
        images = mat["images"]  # (H, W, 3, N)  uint8
        depths = mat["depths"]  # (H, W, N)     float64
        images = np.transpose(images, (3, 2, 0, 1)).astype(np.float32)  # → (N, 3, H, W)
        depths = np.transpose(depths, (2, 0, 1)).astype(np.float32)  # → (N, H, W)
    else:
        with h5py.File(str(mat_path), "r") as f:
            images = np.array(f["images"])  # (N, 3, H, W) or (H, W, 3, N)  uint8
            depths = np.array(f["depths"])  # (N, H, W) or (H, W, N)     float64

        # h5py stores in MATLAB order — find the channel axis (size 3)
        # and transpose to (N, 3, H, W).
        if images.ndim == 4:
            if 3 not in images.shape:
                raise ValueError(
                    f"NYUv2: expected a channel axis of size 3, got shape {images.shape}"
                )
            chan_axis = int(np.where(np.array(images.shape) == 3)[0][0])
            if chan_axis != 1:
                src = list(range(4))
                dst = src.copy()
                dst.pop(chan_axis)
                dst.insert(1, chan_axis)
                images = np.transpose(images, dst)

        if depths.ndim == 3 and depths.shape[1] < depths.shape[2]:
            depths = np.transpose(depths, (2, 0, 1))  # (H, W, N) → (N, H, W)

        images = images.astype(np.float32) / 255.0
        depths = depths.astype(np.float32)

    # Subset
    if n_samples < len(images):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(images), n_samples, replace=False)
        idx.sort()
        images, depths = images[idx], depths[idx]

    # Resize in parallel via numpy + PIL (torchvision not guaranteed)
    try:
        from PIL import Image

        H_out, W_out = img_size, img_size
        X_resized = np.empty((len(images), 3, H_out, W_out), dtype=np.float32)
        y_resized = np.empty((len(depths), H_out, W_out), dtype=np.float32)

        for i in range(len(images)):
            # RGB: (3, H, W) → (H, W, 3)
            rgb = images[i].transpose(1, 2, 0)
            rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
            rgb_resized = (
                np.array(
                    rgb_pil.resize((W_out, H_out), Image.Resampling.BILINEAR), dtype=np.float32
                )
                / 255.0
            )
            X_resized[i] = rgb_resized.transpose(2, 0, 1)  # back to (3, H, W)

            # Depth: (H, W)
            depth_pil = Image.fromarray(depths[i])
            y_resized[i] = np.array(
                depth_pil.resize((W_out, H_out), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )

        images = X_resized
        depths = y_resized
    except ImportError:
        # No PIL — keep original resolution
        pass

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        depths,
        test_size=test_size,
        random_state=seed,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": 3,
        "_dense_output": True,
        "_skip_y_standardization": True,  # dense (N, H, W) — preserve metres
    }


def _load_imdb_wiki(
    n_samples: int = 20000,
    img_size: int = 128,
    test_size: float = 0.2,
    min_face_score: float = 2.0,
    min_age: int = 0,
    max_age: int = 100,
    seed: int = 42,
) -> SplitsDict:
    """Load the IMDB-WIKI face age dataset.

    Downloads cropped face images (~8 GB total) and metadata from ETH Zurich,
    computes ages from date-of-birth and photo-taken year, filters by face
    score and age range, resizes to ``img_size``, and returns train/test splits.

    Parameters
    ----------
    n_samples : int
        Maximum number of samples to load.  Default 20 000.
    img_size : int
        Target square size for resized face images.
    test_size : float
        Fraction of data reserved for testing.
    min_face_score : float
        Minimum face detection score (higher = better quality crops).
        IMDB-WIKI computed scores; ≥ 2.0 filters obvious false positives.
    min_age, max_age : int
        Valid age range in years.  Ages outside this are dropped.
    seed : int
        Random seed for the train/test split.

    Returns
    -------
    splits : dict
        ``X_train`` (N_train, 3, img_size, img_size) float32 face images,
        ``X_test`` (N_test, 3, img_size, img_size) float32 face images,
        ``y_train`` (N_train,) float32 ages in years,
        ``y_test`` (N_test,) float32 ages in years.
    """
    from PIL import Image

    cache_dir = _image_cache_dir()

    # -- Step 1: download crop tars and meta tars (if not cached) ------------
    # Cached tars are reused without download. The ETH Zurich mirror 404s
    # wiki_meta.tar (2026-08); fall back to the HuggingFace mirror for
    # metadata before failing. Crop tars (6.6G/774M) are the common case
    # already cached — only missing files trigger downloads.
    sources = [
        ("imdb_crop.tar", _IMDB_CROP_URL, None),
        ("wiki_crop.tar", _WIKI_CROP_URL, None),
        ("imdb_meta.tar", _IMDB_META_URL, f"{_IMDB_WIKI_HF_MIRROR}/imdb_meta.tar"),
        ("wiki_meta.tar", _WIKI_META_URL, f"{_IMDB_WIKI_HF_MIRROR}/wiki_meta.tar"),
    ]
    for fname, url, mirror in sources:
        dest = cache_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            _download_with_progress(url, dest, desc=fname)
        except Exception as exc_primary:  # noqa: BLE001 — mirror fallback
            if mirror is not None:
                try:
                    _download_with_progress(mirror, dest, desc=fname)
                    continue
                except Exception:
                    pass
            # Metadata tars are optional: a missing wiki_meta only drops the
            # wiki subset (imdb subset still yields a valid benchmark); crop
            # tars are mandatory.
            if fname in ("wiki_meta.tar", "imdb_meta.tar"):
                print(f"Warning: {fname} unavailable ({exc_primary}); subset may be skipped.")
                continue
            raise

    # -- Step 2: extract tars (one-time, cached) ----------------------------
    import tarfile

    crop_dir = cache_dir / "imdb_wiki_crop"
    meta_dir = cache_dir / "imdb_wiki_meta"

    for fname in ["imdb_crop.tar", "wiki_crop.tar"]:
        tar_path = cache_dir / fname
        if not tar_path.exists():
            raise FileNotFoundError(
                f"IMDB-WIKI: required {fname} missing in {cache_dir}. "
                "Download it manually or check network access."
            )
        if not (crop_dir / fname.replace(".tar", "")).is_dir():
            print(f"Extracting {fname} …")
            with tarfile.open(tar_path) as tf:
                tf.extractall(path=crop_dir, filter="data")

    for fname in ["imdb_meta.tar", "wiki_meta.tar"]:
        tar_path = cache_dir / fname
        if not tar_path.exists():
            # Missing metadata tar (e.g. wiki_meta.tar 404s upstream):
            # skip extraction; the parse step below skips that subset.
            print(f"Warning: {fname} not cached — its subset will be skipped.")
            continue
        if not (meta_dir / fname.replace(".tar", "")).is_dir():
            print(f"Extracting {fname} …")
            with tarfile.open(tar_path) as tf:
                tf.extractall(path=meta_dir, filter="data")

    # -- Step 3: parse metadata and compute ages -----------------------------
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError(
            "scipy is required to load IMDB-WIKI metadata (.mat format). "
            "Install with: uv pip install scipy"
        )

    all_images = []
    all_ages = []

    for subset in ["imdb", "wiki"]:
        mat_path = meta_dir / subset / f"{subset}.mat"
        if not mat_path.exists():
            mat_path = meta_dir / f"{subset}_meta" / subset / f"{subset}.mat"
        if not mat_path.exists():
            # Try glob
            candidates = sorted(meta_dir.rglob(f"{subset}.mat"))
            if candidates:
                mat_path = candidates[0]
            else:
                print(f"Warning: metadata not found for {subset}, skipping.")
                continue

        meta = sio.loadmat(str(mat_path))[subset][0, 0]

        # Fields: dob (serial date number), photo_taken (year),
        #         full_path (relative), face_score, face_location, gender, …
        dobs = meta["dob"][0]  # MATLAB serial date numbers
        years = meta["photo_taken"][0].astype(np.int32)  # year photo was taken
        paths = meta["full_path"][0]
        scores = meta["face_score"][0] if "face_score" in meta.dtype.names else None

        # Compute age: photo_taken_year - birth_year.
        # MATLAB serial date numbers → proper calendar year via datetime.
        from datetime import datetime, timedelta

        birth_years = np.empty(len(dobs), dtype=np.int32)
        for j, dob in enumerate(dobs):
            birth_date = datetime(1, 1, 1) + timedelta(days=int(dob) - 1)
            birth_years[j] = birth_date.year
        ages = years - birth_years

        crop_subdir = crop_dir / f"{subset}_crop"

        for i in range(len(paths)):
            if len(all_images) >= n_samples:
                break

            age = ages[i]
            if age < min_age or age > max_age:
                continue
            if scores is not None and scores[i] < min_face_score:
                continue

            img_path = crop_subdir / str(
                paths[i][0] if isinstance(paths[i], np.ndarray) else paths[i]
            )
            if not img_path.is_file():
                # Try alternate path construction
                path_str = str(paths[i][0] if isinstance(paths[i], np.ndarray) else paths[i])
                fname_only = Path(path_str).name
                alt = crop_subdir / fname_only
                if alt.is_file():
                    img_path = alt
                else:
                    continue

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
                all_images.append(arr.transpose(2, 0, 1))  # → (3, H, W)
                all_ages.append(float(age))
            except Exception:
                continue

        if len(all_images) >= n_samples:
            break

    if len(all_images) == 0:
        raise RuntimeError(
            "IMDB-WIKI: no valid face images found after filtering. "
            "Check the dataset was downloaded correctly to "
            f"{cache_dir} and min_face_score={min_face_score} "
            "is not too strict."
        )

    X = np.stack(all_images, axis=0).astype(np.float32)
    y = np.array(all_ages, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": 3,
    }


def _load_biwi_head_pose(
    n_samples: int = 1000,
    img_size: int = 64,
    n_channels: int = 3,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """Load BIWI head-pose (multi-target: yaw/pitch/roll) or a proxy.

    Tries ``torchvision.datasets``-style local cache first; BIWI is not
    distributed with torchvision, so the common path is a synthetic proxy
    with matching geometry: ``X=(N,3,H,W)`` face-like blobs,
    ``y=(N,3)`` yaw/pitch/roll in degrees. The proxy preserves the
    (N,3) multi-output contract so keypoint/multitarget suites run
    hermetically in CI; headless GPU jobs with the real BIWI tar override
    via ``BIWI_ROOT`` env (directory of ``*.jpg`` + ``*.txt`` poses).
    """
    import os as _os

    biwi_root = _os.environ.get("BIWI_ROOT", "")
    if biwi_root:
        root = Path(biwi_root)
        jpgs = sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png"))
        if jpgs:
            try:
                from PIL import Image

                imgs, poses = [], []
                for jp in jpgs[:n_samples]:
                    txt = jp.with_suffix(".txt")
                    if not txt.is_file():
                        continue
                    pose = np.loadtxt(str(txt), dtype=np.float32).ravel()[:3]
                    img = Image.open(jp).convert("RGB").resize((img_size, img_size))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    imgs.append(arr.transpose(2, 0, 1))
                    poses.append(pose)
                    if len(imgs) >= n_samples:
                        break
                if imgs:
                    X = np.stack(imgs).astype(np.float32)
                    y = np.stack(poses).astype(np.float32)
                    Xtr, Xte, ytr, yte = train_test_split(
                        X, y, test_size=test_size, random_state=seed
                    )
                    return {
                        "X_train": Xtr,
                        "X_test": Xte,
                        "y_train": ytr,
                        "y_test": yte,
                        "_image_data": True,
                        "_image_size": img_size,
                        "_n_channels": 3,
                        "_multi_output": True,
                        "_skip_y_standardization": True,
                    }
            except Exception:
                pass  # fall through to synthetic proxy

    # Synthetic proxy: correlated yaw/pitch/roll + blob images.
    rng = np.random.default_rng(seed)
    n = n_samples
    yaw = rng.uniform(-75, 75, n).astype(np.float32)
    pitch = (0.5 * yaw + rng.normal(0, 10, n)).astype(np.float32)
    roll = (0.3 * yaw - 0.2 * pitch + rng.normal(0, 8, n)).astype(np.float32)
    y = np.column_stack([yaw, pitch, roll]).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    X = np.zeros((n, n_channels, img_size, img_size), dtype=np.float32)
    for i in range(n):
        cx = img_size / 2 + yaw[i] / 75 * img_size * 0.3
        cy = img_size / 2 + pitch[i] / 60 * img_size * 0.3
        sigma = img_size / 8
        blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
        for c in range(n_channels):
            X[i, c] = (blob * (0.6 + 0.1 * c)).astype(np.float32)
    X += 0.05 * rng.standard_normal(X.shape).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    splits: SplitsDict = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": n_channels,
        "_multi_output": True,
        "_skip_y_standardization": True,
        "_synthetic_fallback": not bool(biwi_root),
    }
    if not biwi_root:
        splits["_fallback_reason"] = "BIWI_ROOT unset; synthetic head-pose proxy"
    return splits


def _load_synthetic_image_keypoint(
    n_samples: int = 800,
    img_size: int = 64,
    n_channels: int = 1,
    n_keypoints: int = 4,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """Synthetic keypoint regression: blobs at K joints, y=(N,2K) coords."""
    rng = np.random.default_rng(seed)
    K = n_keypoints
    cx = rng.uniform(img_size * 0.15, img_size * 0.85, (n_samples, K)).astype(np.float32)
    cy = rng.uniform(img_size * 0.15, img_size * 0.85, (n_samples, K)).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    X = np.zeros((n_samples, n_channels, img_size, img_size), dtype=np.float32)
    for i in range(n_samples):
        canvas = np.zeros((img_size, img_size), dtype=np.float32)
        for k in range(K):
            canvas += np.exp(-((xs - cx[i, k]) ** 2 + (ys - cy[i, k]) ** 2) / 18.0)
        for c in range(n_channels):
            X[i, c] = canvas
    X += 0.03 * rng.standard_normal(X.shape).astype(np.float32)
    y = np.concatenate([cx / img_size, cy / img_size], axis=1).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": n_channels,
        "_multi_output": True,
        "_skip_y_standardization": True,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _stress_registry_loader(name: str) -> Callable[..., "SplitsDict"]:
    """Build a registry loader for a synthetic stress-matrix dataset.

    Lazy import: ``suites.tabular.stress_matrix`` pulls in torchregress and
    the rest of the harness stack, which must not happen at datasets.py
    import time (circular import).
    """

    def _loader(**kwargs: Any) -> "SplitsDict":
        from suites.tabular.stress_matrix import load_stress_dataset

        n = int(kwargs.get("n_samples") or 4000)
        return load_stress_dataset(name, n=n)

    return _loader


def _registry_loader(
    fn: Callable[..., SplitsDict],
    **defaults: Any,
) -> Callable[..., SplitsDict]:
    """Wrap a loader so extra kwargs can override defaults."""

    def _loader(**kwargs: Any) -> SplitsDict:
        merged = {**defaults, **{k: v for k, v in kwargs.items() if v is not None}}
        return fn(**merged)

    return _loader


DATASET_REGISTRY: dict[str, DatasetInfo] = {
    # --- Real tabular ---
    "california_housing": DatasetInfo(
        name="california_housing",
        category="tabular",
        n_features=8,
        n_samples=20640,
        description="California Housing: median house value prediction from census features.",
        loader=_registry_loader(_load_california_housing, n_samples=5000),
    ),
    "diabetes": DatasetInfo(
        name="diabetes",
        category="tabular",
        n_features=10,
        n_samples=442,
        description="Sklearn Diabetes: disease progression after one year.",
        loader=_registry_loader(_load_diabetes),
    ),
    "openml_year": DatasetInfo(
        name="openml_year",
        category="tabular",
        n_features=90,
        n_samples=515345,
        description="OpenML YearPredictionMSD: song release year from audio features.",
        loader=_registry_loader(_load_openml_year, n_samples=10000),
    ),
    # --- Synthetic tabular ---
    "synthetic_hetero": DatasetInfo(
        name="synthetic_hetero",
        category="synthetic",
        n_features=8,
        n_samples=2000,
        description="Synthetic heteroscedastic: noise varies with input magnitude.",
        loader=_registry_loader(_load_synthetic_heteroscedastic),
    ),
    "synthetic_outliers": DatasetInfo(
        name="synthetic_outliers",
        category="synthetic",
        n_features=8,
        n_samples=2000,
        description="Synthetic data with 15% extreme outliers.",
        loader=_registry_loader(_load_synthetic_outliers),
    ),
    "synthetic_tweedie": DatasetInfo(
        name="synthetic_tweedie",
        category="synthetic",
        n_features=6,
        n_samples=2000,
        description="Synthetic compound Poisson-Gamma (Tweedie p=1.5) with zero inflation.",
        loader=_registry_loader(_load_synthetic_tweedie),
    ),
    "synthetic_censored": DatasetInfo(
        name="synthetic_censored",
        category="synthetic",
        n_features=8,
        n_samples=2000,
        description="Synthetic right-censored targets at 70th percentile of true values.",
        loader=_registry_loader(_load_synthetic_censored),
    ),
    "synthetic_ordinal": DatasetInfo(
        name="synthetic_ordinal",
        category="synthetic",
        n_features=6,
        n_samples=2000,
        description=(
            "Synthetic ordinal regression (5 classes from quantile-binned continuous target)"
        ),
        loader=_registry_loader(_load_synthetic_ordinal),
    ),
    "synthetic_eiv": DatasetInfo(
        name="synthetic_eiv",
        category="synthetic",
        n_features=6,
        n_samples=2000,
        description="Synthetic errors-in-variables with noisy input measurements.",
        loader=_registry_loader(_load_synthetic_eiv),
    ),
    "synthetic_eiv_highnoise": DatasetInfo(
        name="synthetic_eiv_highnoise",
        category="synthetic",
        n_features=6,
        n_samples=2000,
        description="Synthetic EIV with high input noise (σx=1.0) — stress test for EIV losses.",
        loader=_registry_loader(_load_synthetic_eiv, sigma_x=1.0),
    ),
    "synthetic_eiv_nonlinear": DatasetInfo(
        name="synthetic_eiv_nonlinear",
        category="synthetic",
        n_features=6,
        n_samples=2000,
        description="Synthetic EIV with nonlinear ground-truth (sin/cos/tanh) — TLS cannot fit.",
        loader=_registry_loader(_load_synthetic_eiv_nonlinear, sigma_x=0.5),
    ),
    "synthetic_eiv_nonlinear_highnoise": DatasetInfo(
        name="synthetic_eiv_nonlinear_highnoise",
        category="synthetic",
        n_features=6,
        n_samples=2000,
        description="Nonlinear EIV with high input noise (σx=1.0) — definitive EIV stress test.",
        loader=_registry_loader(_load_synthetic_eiv_nonlinear, sigma_x=1.0),
    ),
    "synthetic_eiv_veryhighnoise": DatasetInfo(
        name="synthetic_eiv_veryhighnoise",
        category="synthetic",
        n_features=6,
        n_samples=2000,
        description=(
            "Linear EIV with extreme input noise (σx=2.0) — maximum stress test for EIV methods."
        ),
        loader=_registry_loader(_load_synthetic_eiv, sigma_x=2.0),
    ),
    "synthetic_eiv_nonlinear_veryhighnoise": DatasetInfo(
        name="synthetic_eiv_nonlinear_veryhighnoise",
        category="synthetic",
        n_features=6,
        n_samples=2000,
        description="Nonlinear EIV with extreme input noise (σx=2.0) — ultimate EIV stress test.",
        loader=_registry_loader(_load_synthetic_eiv_nonlinear, sigma_x=2.0),
    ),
    "synthetic_multivariate": DatasetInfo(
        name="synthetic_multivariate",
        category="synthetic",
        n_features=8,
        n_samples=2000,
        description=(
            "Synthetic 3D correlated targets: y0, y1=0.5*y0+f1(X), y2=-0.3*y0+0.4*y1+f2(X)."
        ),
        loader=_registry_loader(_load_synthetic_multivariate),
    ),
    "synthetic_mv_gaussian": DatasetInfo(
        name="synthetic_mv_gaussian",
        category="synthetic",
        n_features=8,
        n_samples=2000,
        description=(
            "Purely multivariate Gaussian 3D targets: "
            "y|X ~ N(μ(X), Σ). Correctly specified for Gaussian methods."
        ),
        loader=_registry_loader(_load_synthetic_mv_gaussian),
    ),
    "synthetic_mv_nongaussian": DatasetInfo(
        name="synthetic_mv_nongaussian",
        category="synthetic",
        n_features=8,
        n_samples=2000,
        description=(
            "Non-Gaussian 3D targets: t₃ heavy tails + χ² skew + bimodal. "
            "Misspecified for Gaussian methods."
        ),
        loader=_registry_loader(_load_synthetic_mv_nongaussian),
    ),
    "synthetic_mv_bimodal": DatasetInfo(
        name="synthetic_mv_bimodal",
        category="synthetic",
        n_features=8,
        n_samples=2000,
        description=(
            "Bimodal 3D targets: 0.5·N(μ₁,Σ₁) + 0.5·N(μ₂,Σ₂). "
            "Gaussian methods fail — NF/SLS should excel."
        ),
        loader=_registry_loader(_load_synthetic_mv_bimodal),
    ),
    # --- Conditional flow benchmarks (pzflow parity) ---
    "synthetic_cond_gaussian": DatasetInfo(
        name="synthetic_cond_gaussian",
        category="synthetic",
        n_features=5,
        n_samples=2000,
        description=(
            "1D target with Gaussian conditional: y|X ~ N(X·β, 0.3²). "
            "Correctly specified for all flow methods."
        ),
        loader=_registry_loader(_load_synthetic_cond_gaussian),
    ),
    "synthetic_cond_nongaussian": DatasetInfo(
        name="synthetic_cond_nongaussian",
        category="synthetic",
        n_features=5,
        n_samples=2000,
        description=(
            "1D target with non-Gaussian conditional: y|X has 3-component "
            "mixture noise. Misspecified for Gaussian — flows should excel."
        ),
        loader=_registry_loader(_load_synthetic_cond_nongaussian),
    ),
    # --- Upstream canonical datasets ---
    "insurance_claims": DatasetInfo(
        name="insurance_claims",
        category="tabular",
        n_features=12,
        n_samples=678013,
        description=(
            "French Motor Insurance Claims (OpenML 41214): Tweedie "
            "claim count with exposure offset.  ~12 features after OHE.  "
            "Canonical sklearn TweedieRegressor example."
        ),
        loader=_registry_loader(_load_insurance_claims, n_samples=20000),
    ),
    "house_sales": DatasetInfo(
        name="house_sales",
        category="tabular",
        n_features=19,
        n_samples=21613,
        description=(
            "King County House Sales (OpenML house_sales v3): price "
            "prediction.  Canonical crepes conformal prediction example."
        ),
        loader=_registry_loader(_load_house_sales, n_samples=20000),
    ),
    "make_regression_synthetic": DatasetInfo(
        name="make_regression_synthetic",
        category="synthetic",
        n_features=10,
        n_samples=5000,
        description=(
            "sklearn make_regression: 5K samples, 10 features "
            "(6 informative), noise=15.  Standard sklearn estimator "
            "benchmark."
        ),
        loader=_registry_loader(_load_make_regression_synthetic),
    ),
    "synthetic_sinusoidal": DatasetInfo(
        name="synthetic_sinusoidal",
        category="synthetic",
        n_features=4,
        n_samples=2000,
        description=(
            "Sinusoidal test function: sin(2πx₀) + 0.3cos(4πx₁) "
            "+ 0.2tanh(3x₂) + noise.  Canonical BoTorch GP regression "
            "example."
        ),
        loader=_registry_loader(_load_synthetic_sinusoidal),
    ),
    # --- TabReD industrial-scale datasets ---
    "tabred_sberbank": DatasetInfo(
        name="tabred_sberbank",
        category="tabular",
        n_features=392,
        n_samples=28321,
        description=(
            "TabReD Sberbank Russian Housing: 28K rows, 392 features.  "
            "Place preprocessed CSV at datasets/tabred/sberbank.csv "
            "or pass data_dir= to load_dataset()."
        ),
        loader=_registry_loader(
            _load_tabred,
            dataset_id="sberbank",
            n_rows=28321,
            n_features=392,
            n_samples=20000,
        ),
    ),
    "tabred_cooking": DatasetInfo(
        name="tabred_cooking",
        category="tabular",
        n_features=192,
        n_samples=319986,
        description=(
            "TabReD Yandex Cooking Time: 320K rows, 192 features.  "
            "Place preprocessed CSV at datasets/tabred/cooking.csv "
            "or pass data_dir= to load_dataset()."
        ),
        loader=_registry_loader(
            _load_tabred,
            dataset_id="cooking",
            n_rows=319986,
            n_features=192,
            n_samples=10000,
        ),
    ),
    "tabred_delivery": DatasetInfo(
        name="tabred_delivery",
        category="tabular",
        n_features=223,
        n_samples=416451,
        description=(
            "TabReD Yandex Delivery ETA: 416K rows, 223 features.  "
            "Place preprocessed CSV at datasets/tabred/delivery.csv "
            "or pass data_dir= to load_dataset()."
        ),
        loader=_registry_loader(
            _load_tabred,
            dataset_id="delivery",
            n_rows=416451,
            n_features=223,
            n_samples=10000,
        ),
    ),
    "tabred_maps": DatasetInfo(
        name="tabred_maps",
        category="tabular",
        n_features=986,
        n_samples=340981,
        description=(
            "TabReD Yandex Maps Routing: 341K rows, 986 features.  "
            "Place preprocessed CSV at datasets/tabred/maps.csv "
            "or pass data_dir= to load_dataset()."
        ),
        loader=_registry_loader(
            _load_tabred,
            dataset_id="maps",
            n_rows=340981,
            n_features=986,
            n_samples=20000,
        ),
    ),
    "tabred_weather": DatasetInfo(
        name="tabred_weather",
        category="tabular",
        n_features=103,
        n_samples=423795,
        description=(
            "TabReD Yandex Weather: 424K rows, 103 features.  "
            "Place preprocessed CSV at datasets/tabred/weather.csv "
            "or pass data_dir= to load_dataset()."
        ),
        loader=_registry_loader(
            _load_tabred,
            dataset_id="weather",
            n_rows=423795,
            n_features=103,
            n_samples=10000,
        ),
    ),
    # --- Image regression datasets ---
    "synthetic_image_regression": DatasetInfo(
        name="synthetic_image_regression",
        category="image",
        n_features=-1,  # image data — feature count is img_size²
        n_samples=1000,
        description=(
            "Synthetic image regression: 32×32 grayscale images with a "
            "Gaussian blob whose centre coordinates determine the scalar target."
        ),
        loader=_registry_loader(_load_synthetic_image_regression),
    ),
    "synthetic_image_depth": DatasetInfo(
        name="synthetic_image_depth",
        category="image",
        n_features=-1,  # image data
        n_samples=600,
        description=(
            "Synthetic depth estimation: 64×64 RGB images with simple "
            "geometric scenes.  Target is a per-pixel depth map."
        ),
        loader=_registry_loader(_load_synthetic_image_depth),
    ),
    # --- Real image datasets ---
    "nyu_depth_v2": DatasetInfo(
        name="nyu_depth_v2",
        category="image",
        n_features=-1,  # image data — 3×img_size×img_size
        n_samples=1449,
        description=(
            "NYU Depth V2 labeled subset: 1449 RGB-D indoor scenes "
            "(480×640). Resized to img_size×img_size. "
            "X=(N,3,H,W) images, y=(N,H,W) depth maps in metres. "
            "Auto-downloads from official source (~2.7 GB)."
        ),
        loader=_registry_loader(_load_nyu_depth_v2),
    ),
    "imdb_wiki": DatasetInfo(
        name="imdb_wiki",
        category="image",
        n_features=-1,  # image data — 3×img_size×img_size
        n_samples=523051,
        description=(
            "IMDB-WIKI face age dataset: ~523K celebrity face images "
            "with age labels. X=(N,3,H,W) cropped faces, y=(N,) ages "
            "in years. Auto-downloads from ETH Zurich (~8 GB). "
            "Default filters: face_score ≥ 2.0, ages 0–100."
        ),
        loader=_registry_loader(_load_imdb_wiki),
    ),
    "imdb_wiki_fullres": DatasetInfo(
        name="imdb_wiki_fullres",
        category="image",
        n_features=-1,
        n_samples=523051,
        description=(
            "IMDB-WIKI at 256×256 (full-real nightly/CUDA track). "
            "Same loader as imdb_wiki with img_size=256."
        ),
        loader=_registry_loader(_load_imdb_wiki, img_size=256),
    ),
    "nyu_depth_v2_fullres": DatasetInfo(
        name="nyu_depth_v2_fullres",
        category="image",
        n_features=-1,
        n_samples=1449,
        description=(
            "NYU Depth V2 at 256×256 (full-real nightly/CUDA track). "
            "Same loader as nyu_depth_v2 with img_size=256."
        ),
        loader=_registry_loader(_load_nyu_depth_v2, img_size=256),
    ),
    "biwi_head_pose": DatasetInfo(
        name="biwi_head_pose",
        category="image",
        n_features=-1,
        n_samples=1000,
        description=(
            "BIWI head-pose multitarget (yaw/pitch/roll, N×3). Real data "
            "via BIWI_ROOT env; synthetic proxy otherwise (flagged)."
        ),
        loader=_registry_loader(_load_biwi_head_pose),
    ),
    "synthetic_image_keypoint": DatasetInfo(
        name="synthetic_image_keypoint",
        category="image",
        n_features=-1,
        n_samples=800,
        description=(
            "Synthetic keypoint regression: K Gaussian blobs, y=(N,2K) normalized coordinates."
        ),
        loader=_registry_loader(_load_synthetic_image_keypoint),
    ),
    # --- Stress-matrix (suites/tabular/stress_matrix.py) ---
    # Thin registry wrappers over the synthetic stress generators so that
    # tools/run_all.py can resolve ``stress_*`` dataset names through the
    # standard DATASET_REGISTRY path. Lazy import avoids a circular dep
    # (datasets.py must stay importable without the suites package).
    "stress_imbalanced": DatasetInfo(
        name="stress_imbalanced",
        category="synthetic",
        n_features=12,
        n_samples=4000,
        description="Stress axis: long-tailed (exponential) target density.",
        loader=_stress_registry_loader("stress_imbalanced"),
    ),
    "stress_noisy_inputs": DatasetInfo(
        name="stress_noisy_inputs",
        category="synthetic",
        n_features=12,
        n_samples=4000,
        description="Stress axis: additive observation noise on train features (EIV); test features clean.",
        loader=_stress_registry_loader("stress_noisy_inputs"),
    ),
    "stress_noisy_labels": DatasetInfo(
        name="stress_noisy_labels",
        category="synthetic",
        n_features=12,
        n_samples=4000,
        description="Stress axis: heteroscedastic Gaussian label noise with known per-sample variance.",
        loader=_stress_registry_loader("stress_noisy_labels"),
    ),
    "stress_sparse_labels": DatasetInfo(
        name="stress_sparse_labels",
        category="synthetic",
        n_features=12,
        n_samples=4000,
        description="Stress axis: sparse supervision (20% labeled) with pseudo-targets and known variance.",
        loader=_stress_registry_loader("stress_sparse_labels"),
    ),
    "stress_ood": DatasetInfo(
        name="stress_ood",
        category="synthetic",
        n_features=12,
        n_samples=4000,
        description="Stress axis: covariate shift (train/test split by x0 sign, not random).",
        loader=_stress_registry_loader("stress_ood"),
    ),
    "stress_semi_sup": DatasetInfo(
        name="stress_semi_sup",
        category="synthetic",
        n_features=12,
        n_samples=4000,
        description="Stress axis: small labeled pool (200) + large unlabeled pool (SAGE regime).",
        loader=_stress_registry_loader("stress_semi_sup"),
    ),
}


def load_dataset(name: str, **kwargs: Any) -> SplitsDict:
    """Load a dataset by name from the registry.

    Targets (y) are automatically standardised (zero-mean, unit-variance)
    using ``StandardScaler`` fitted on ``y_train`` unless the dataset
    sets ``_skip_y_standardization = True`` in its returned dict
    (Tweedie / censored datasets do this to preserve distribution semantics).

    The fitted scaler is stored as ``splits["_y_scaler"]`` so that
    ``evaluate()`` can inverse-transform back to original units before
    computing metrics.

    Args:
        name: Dataset key from ``DATASET_REGISTRY``.
        standardize_targets: If False, skip target standardisation.
            Overrides ``_skip_y_standardization`` in dataset loaders.
        **kwargs: Override default loader parameters (e.g., ``n_samples=500``).

    Returns:
        Dictionary with ``X_train``, ``X_test``, ``y_train``, ``y_test``
        as numpy arrays (y arrays are standardised unless opt-out).
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name!r}. Available: {sorted(DATASET_REGISTRY.keys())}")
    # Pop harness-level kwargs before passing through to the dataset loader
    standardize = bool(kwargs.pop("standardize_targets", True))
    splits = DATASET_REGISTRY[name].loader(**kwargs)

    # Standardise targets unless the dataset opts out (Tweedie, censored)
    # or the harness is explicitly configured to skip standardisation.
    if standardize and not splits.get("_skip_y_standardization"):
        y_scaler = StandardScaler()
        y_scaler.fit(splits["y_train"].reshape(-1, 1))
        splits["y_train"] = (
            y_scaler.transform(splits["y_train"].reshape(-1, 1)).ravel().astype(np.float32)
        )
        splits["y_test"] = (
            y_scaler.transform(splits["y_test"].reshape(-1, 1)).ravel().astype(np.float32)
        )
        splits["_y_scaler"] = y_scaler

    return splits


def list_datasets(category: str | None = None) -> list[DatasetInfo]:
    """List all registered datasets, optionally filtered by category."""
    datasets = list(DATASET_REGISTRY.values())
    if category is not None:
        datasets = [d for d in datasets if d.category == category]
    return sorted(datasets, key=lambda d: d.name)
