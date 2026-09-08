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
    _count_data: bool
    _observed_std: bool
    y_std_train: np.ndarray
    y_std_test: np.ndarray
    count_train: np.ndarray
    count_test: np.ndarray
    mask_train: np.ndarray
    mask_test: np.ndarray
    _sparse_depth: bool
    _n_real_slides: int
    male_train: np.ndarray
    male_test: np.ndarray
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


def _load_survset_hdfail(
    n_samples: int = 20000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Backblaze hard-drive failure survival (SurvSet ``hdfail``, 52k drives).

    Real right-censored time-to-event data: days-to-failure with temperature +
    drive model/firmware codes. From the SurvSet time-to-event repository
    (Drysdale 2022, compiled from R survival packages + modern sources) —
    a credible modern survival benchmark, no simulation. ``event=1`` means
    failure observed; censoring codes follow the harness convention
    (1 = right-censored). Falls back to synthetic censored data if the
    GitHub fetch fails.
    """
    splits: SplitMetadata = {}
    try:
        from sklearn.preprocessing import OneHotEncoder
        from SurvSet.data import SurvLoader

        d = SurvLoader().load_dataset(ds_name="hdfail")
        df, _ = d.values()
        y_time = df["time"].to_numpy(dtype=np.float64)
        event = df["event"].to_numpy(dtype=np.float64)
        codes = (1.0 - event).astype(np.float32)  # 1 = right-censored
        feat_cols = [c for c in df.columns if c not in ("pid", "event", "time")]
        num_cols = [c for c in feat_cols if c.startswith("num_")]
        cat_cols = [c for c in feat_cols if c.startswith("fac_")]
        X_num = df[num_cols].to_numpy(dtype=np.float64)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = ohe.fit_transform(df[cat_cols].astype(str))
        X = np.hstack([X_num, X_cat])
        y = y_time
        if n_samples < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n_samples, replace=False)
            X, y, codes = X[idx], y[idx], codes[idx]
    except Exception:
        synth = _load_synthetic_censored(n_samples=n_samples, seed=seed)
        synth["_synthetic_fallback"] = True
        synth["_fallback_reason"] = "SurvSet hdfail fetch failed"
        return synth

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    _, _, codes_train, codes_test = train_test_split(
        X, codes, test_size=test_size, random_state=seed
    )
    return {
        **splits,
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "censoring_train": codes_train.astype(np.float32),
        "censoring_test": codes_test.astype(np.float32),
        "_skip_y_standardization": True,
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


def _fetch_folktables_xy(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Shared ACS 2018 CA fetch + encoding → (X features, continuous PINCP)."""
    import folktables
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder

    root = str(_image_cache_dir() / "folktables")
    ds = folktables.ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person", root_dir=root
    )
    acs = ds.get_data(states=["CA"], download=True)
    df = acs.copy()
    filt = (df["AGEP"] > 16) & (df["WKHP"] > 0) & (df["PINCP"] >= 100)
    df = df.loc[filt].reset_index(drop=True)
    y = df["PINCP"].to_numpy(dtype=np.float64)
    low_card = ["COW", "SCHL", "MAR", "SEX", "RAC1P", "RELP"]
    high_card = ["OCCP", "POBP"]
    numeric = ["AGEP", "WKHP"]
    X_num = df[numeric].to_numpy(dtype=np.float64)
    X_high = df[high_card].apply(lambda s: pd.Categorical(s).codes).to_numpy(dtype=np.float64)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(df[low_card].astype(str))
    return np.hstack([X_num, X_high, X_cat]), y


def _load_folktables_income(
    n_samples: int = 20000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """US Census income regression (Folktables ACS 2018, California).

    Same cohort definition as the published ACSIncome task (Ding et al.,
    NeurIPS 2021 Datasets & Benchmarks: age>16, worked>0h, income>=100)
    but with CONTINUOUS personal income (PINCP) instead of the $50k binary
    label — a modern, large-scale regression benchmark straight from the
    prediction-powered inference literature's home turf. Downloads ACS PUMS
    microdata via the folktables package (cached under the harness cache);
    falls back to California Housing if the fetch fails.

    Categorical codes: low-cardinality one-hot (COW/SCHL/MAR/SEX/RAC1P/RELP),
    high-cardinality ordinal (OCCP/POBP); numeric AGEP/WKHP; all standardized.
    """
    splits: SplitMetadata = {}
    try:
        from sklearn.preprocessing import StandardScaler

        X, y = _fetch_folktables_xy(seed)

        if n_samples < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n_samples, replace=False)
            X, y = X[idx], y[idx]
    except Exception:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        X, y = data.data.astype(np.float64), data.target.astype(np.float64)
        if n_samples < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n_samples, replace=False)
            X, y = X[idx], y[idx]
        splits["_synthetic_fallback"] = True
        splits["_fallback_reason"] = "Folktables ACS fetch failed"

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


def _load_folktables_decile(
    n_samples: int = 20000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """US Census income-BRACKET ordinal task (Folktables ACS 2018, California).

    Same cohort/features as :func:`_load_folktables_income`, but the target
    is the income DECILE (0-9, ordered) — a real ordinal task with natural
    class imbalance (credit-band style). ``n_classes=10`` rides in splits
    for the ordinal suite. Falls back to synthetic ordinal data if the
    fetch fails.
    """
    splits: SplitMetadata = {}
    try:
        from sklearn.preprocessing import StandardScaler

        X, y_cont = _fetch_folktables_xy(seed)
        edges = np.quantile(y_cont, np.linspace(0, 1, 11))
        edges[0] -= 1.0
        edges[-1] += 1.0
        y = np.clip(np.digitize(y_cont, edges) - 1, 0, 9).astype(np.float64)
        if n_samples < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n_samples, replace=False)
            X, y = X[idx], y[idx]
    except Exception:
        synth = _load_synthetic_ordinal(n_samples=n_samples, n_classes=10, seed=seed)
        synth["_synthetic_fallback"] = True
        synth["_fallback_reason"] = "Folktables ACS fetch failed"
        return synth

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
        "y_train": y_train.astype(np.int64),
        "y_test": y_test.astype(np.int64),
        "n_classes": 10,
        "_skip_y_standardization": True,  # class labels — never standardize
    }


def _load_energy_efficiency(
    n_samples: int = 768,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """UCI Energy Efficiency (Tsanas & Xifara, 2012): heating/cooling loads.

    768 buildings × 8 features → 2 targets (Y1 heating, Y2 cooling). The
    standard real-data benchmark for MULTI-TARGET regression. Reads
    ENB2012_data.xlsx (double header row) via openpyxl, cached under the
    harness cache; falls back to synthetic multivariate data if the fetch
    fails. ``n_targets=2`` rides in splits for the MV suite.
    """
    splits: SplitMetadata = {}
    try:
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        path = _image_cache_dir() / "ENB2012_data.xlsx"
        if not (path.exists() and path.stat().st_size > 0):
            import urllib.request

            urllib.request.urlretrieve(url, path)
        df = pd.read_excel(path, engine="openpyxl")
        # Flatten possible multi-row header; keep numeric columns only.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join(str(c) for c in col).strip() for col in df.columns]
        num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=1, how="all")
        arr = num.to_numpy(dtype=np.float64)
        X, y = arr[:, :8], arr[:, 8:10]
        if n_samples < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n_samples, replace=False)
            X, y = X[idx], y[idx]
    except Exception:
        synth = _load_synthetic_multivariate(n_samples=n_samples, seed=seed)
        synth["_synthetic_fallback"] = True
        synth["_fallback_reason"] = "UCI Energy Efficiency fetch failed"
        return synth

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
        "n_targets": 2,
        "_multi_output": True,
        "_skip_y_standardization": True,  # preserve (N, D) shape for multi-output
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

# BBBC005 synthetic cell counting (Broad Bioimage Benchmark Collection,
# CC-BY). 19.2k 696×520 8-bit TIFs, counts 1–100 encoded in filenames.
# Verified 2026-09-05: HTTP 200, 1,882,973,059 bytes (~1.8 GB).
_BBBC005_URL = "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip"

# APPA-REAL apparent age (ChaLearn LAP, 7,591 faces + ~250k votes).
# Verified 2026-09-05: direct zip, no login, 885,442,010 bytes (~845 MB).
_APPA_URL = "https://data.chalearnlap.cvc.uab.cat/AppaRealAge/appa-real-release.zip"

# ShanghaiTech crowd counting (CVPR 2016; Part-B street scenes default).
# Verified 2026-09-05: 2024 Dropbox mirror supports direct range download.
_SHANGHAI_URL = (
    "https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/"
    "ShanghaiTech_Crowd_Counting_Dataset.zip?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=1"
)

# IXI brain age (T1, ~600 healthy subjects, CC BY-SA 3.0, DLTK age stats:
# mean 47.4 ± 16.8, range 20–82). NOTE: the IC download server 403s
# automated clients (verified 2026-09-05), so headless auto-download is NOT
# attempted — place IXI-T1.tar (or extracted dir) + IXI.xls under IXI_ROOT.
_IXI_T1_URL = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
_IXI_XLS_URL = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls"

# Udacity steering (Challenge 2, dashboard-cam + steering angle). Raw data
# are ROS bags needing the udacity-driving-reader — NOT headless-feasible —
# so no auto-download: place extracted frames (center/*.jpg) + steering.csv
# (or SullyChen-style data.txt) under UDACITY_ROOT. Heavily imbalanced
# (~0 straight-driving bias) by construction.
_UDACITY_INFO_URL = "https://github.com/udacity/self-driving-car/tree/master/datasets/CH2"

# KITTI depth prediction (93k annotated maps + raw LiDAR/RGB, CC BY-NC-SA).
# Annotated zip alone is 14 GB and RGB lives in per-drive raw zips
# (~175 GB total) — over any sane headless auto-download. NO auto-download:
# stage torch-uncertainty KITTIDepth layout (train/val with leftImg8bit +
# leftDepth uint16 PNGs, depth = px/256 m) under KITTI_ROOT, e.g. via
# torch_uncertainty.datasets.KITTIDepth(root, download=True, remove_unused=True).
_KITTI_DEPTH_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"


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
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = (
        "BIWI_ROOT set but unusable; synthetic head-pose proxy"
        if biwi_root
        else "BIWI_ROOT unset; synthetic head-pose proxy"
    )
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


def _load_utkface_age(
    n_samples: int = 2000,
    img_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """UTKFace age regression (scalar years) with hermetic fallback.

    Real path: ``UTKFACE_ROOT`` dir of ``<age>_<gender>_<race>_*.jpg``
    (standard UTKFace filename convention). Images are RGB-resized to
    ``img_size`` and scaled to [0, 1]; ``y`` is age in years (float32).

    Fallback: face-like Gaussian blobs on 3-channel images with
    ``y = 60*cx/H + 40*cy/H + N(0, 3)`` clipped to [0, 100], preserving the
    (N,3,H,W)→(N,) contract so age suites run hermetically in CI.
    """
    import os as _os

    root = _os.environ.get("UTKFACE_ROOT", "")
    if root:
        p = Path(root)
        files = sorted(p.rglob("*.jpg")) + sorted(p.rglob("*.png"))
        if files:
            try:
                from PIL import Image

                imgs, ages = [], []
                rng_pick = np.random.default_rng(seed)
                order = rng_pick.permutation(len(files))[:n_samples]
                for idx in order:
                    fp = files[int(idx)]
                    try:
                        age = int(fp.name.split("_")[0])
                    except (ValueError, IndexError):
                        continue
                    img = Image.open(fp).convert("RGB").resize((img_size, img_size))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    imgs.append(arr.transpose(2, 0, 1))
                    ages.append(float(age))
                    if len(imgs) >= n_samples:
                        break
                if imgs:
                    X = np.stack(imgs).astype(np.float32)
                    y = np.array(ages, dtype=np.float32)
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
                        "_skip_y_standardization": True,
                    }
            except Exception:
                pass  # fall through to synthetic proxy

    # Synthetic fallback: 3-channel blobs, age in [0, 100].
    rng = np.random.default_rng(seed)
    n = n_samples
    cx = rng.uniform(0, img_size, n).astype(np.float32)
    cy = rng.uniform(0, img_size, n).astype(np.float32)
    y = (60.0 * cx / img_size + 40.0 * cy / img_size + rng.normal(0, 3, n)).astype(np.float32)
    y = np.clip(y, 0, 100).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    X = np.zeros((n, 3, img_size, img_size), dtype=np.float32)
    for i in range(n):
        sigma = img_size / 8
        blob = np.exp(-((xs - cx[i]) ** 2 + (ys - cy[i]) ** 2) / (2 * sigma**2))
        for c in range(3):
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
        "_n_channels": 3,
        "_skip_y_standardization": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = (
        "UTKFACE_ROOT set but unusable; synthetic age proxy"
        if root
        else "UTKFACE_ROOT unset; synthetic age proxy"
    )
    return splits


def _load_boneage(
    n_samples: int = 1000,
    img_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """Bone-age maturity regression (months) with hermetic fallback.

    Real path: ``BONEAGE_ROOT`` dir with the Kaggle RSNA layout —
    ``boneage-training-dataset.csv`` (columns ``id``, ``boneage`` months,
    optional ``male`` bool) plus PNG/JPG hand X-rays (e.g. extracted
    ``boneage-training-dataset.zip``; get it via
    ``kaggle datasets download -d kmader/rsna-bone-age``). Matching is by
    image stem == ``id`` (also tries ``id.png``-style numeric stems).
    Grayscale-resized to ``img_size``; ``y`` is bone age in months.
    The ``male`` sidecar, when present, is exposed as ``male_train/test``.

    Fallback: single-channel bone-like gradient + blob with
    ``y = 228*brightness + N(0, 6)`` clipped to [0, 228] months (0–19y),
    preserving the (N,1,H,W)→(N,) contract for imbalanced/high-stakes tests.
    A set-but-unusable ``BONEAGE_ROOT`` still reports
    ``_synthetic_fallback=True`` (never silently mislabelled).
    """
    import os as _os

    root = _os.environ.get("BONEAGE_ROOT", "")
    used_root = bool(root)
    if root:
        p = Path(root)
        csvs = sorted(p.glob("*.csv"))
        if csvs:
            try:
                import csv as _csv

                from PIL import Image

                # Find id→(boneage, male) mapping in the first csv with 'boneage'.
                mapping: dict[str, tuple[float, float | None]] = {}
                for csv_path in csvs:
                    with open(csv_path, newline="") as fh:
                        reader = _csv.DictReader(fh)
                        if reader.fieldnames and "boneage" in reader.fieldnames:
                            id_key = "id" if "id" in reader.fieldnames else reader.fieldnames[0]
                            male_key = next(
                                (
                                    k
                                    for k in reader.fieldnames
                                    if k.lower() in ("male", "sex", "gender")
                                ),
                                None,
                            )
                            for row in reader:
                                try:
                                    male: float | None = None
                                    if male_key is not None and row[male_key] not in ("", None):
                                        v = str(row[male_key]).strip().lower()
                                        male = 1.0 if v in ("1", "true", "m", "male") else 0.0
                                    mapping[str(row[id_key])] = (float(row["boneage"]), male)
                                except (ValueError, KeyError):
                                    continue
                            break
                if mapping:
                    imgs, ages, males = [], [], []
                    files = (
                        sorted(p.rglob("*.png"))
                        + sorted(p.rglob("*.jpg"))
                        + sorted(p.rglob("*.jpeg"))
                        + sorted(p.rglob("*.PNG"))
                        + sorted(p.rglob("*.JPG"))
                    )
                    rng_pick = np.random.default_rng(seed)
                    order = rng_pick.permutation(len(files))[:n_samples]
                    for idx in order:
                        fp = files[int(idx)]
                        key = fp.stem
                        if key not in mapping:
                            # Kaggle numeric stems (e.g. '1377.png' vs id 1377).
                            key = key.lstrip("0") or "0"
                            if key not in mapping:
                                continue
                        img = Image.open(fp).convert("L").resize((img_size, img_size))
                        arr = np.array(img, dtype=np.float32) / 255.0
                        imgs.append(arr[None, :, :])
                        ages.append(mapping[key][0])
                        males.append(mapping[key][1])
                        if len(imgs) >= n_samples:
                            break
                    if imgs:
                        X = np.stack(imgs).astype(np.float32)
                        y = np.array(ages, dtype=np.float32)
                        idx_all = np.arange(len(imgs))
                        itr, ite = train_test_split(idx_all, test_size=test_size, random_state=seed)
                        out: SplitsDict = {
                            "X_train": X[itr],
                            "X_test": X[ite],
                            "y_train": y[itr],
                            "y_test": y[ite],
                            "_image_data": True,
                            "_image_size": img_size,
                            "_n_channels": 1,
                            "_skip_y_standardization": True,
                        }
                        if any(m is not None for m in males):
                            m_arr = np.array(
                                [m if m is not None else np.nan for m in males],
                                dtype=np.float32,
                            )
                            out["male_train"] = m_arr[itr]
                            out["male_test"] = m_arr[ite]
                        return out
            except Exception:
                pass  # fall through to synthetic proxy

    # Synthetic fallback: gradient + blob, maturity in [0, 228] months.
    rng = np.random.default_rng(seed)
    n = n_samples
    brightness = rng.uniform(0.2, 1.0, n).astype(np.float32)
    y = np.clip(228.0 * brightness + rng.normal(0, 6, n), 0, 228).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    X = np.zeros((n, 1, img_size, img_size), dtype=np.float32)
    for i in range(n):
        grad = (xs / img_size * 0.5 + 0.5) * brightness[i]
        cx = img_size / 2 + rng.normal(0, img_size / 10)
        cy = img_size / 2 + rng.normal(0, img_size / 10)
        blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * (img_size / 6) ** 2))
        X[i, 0] = (0.7 * grad + 0.3 * blob * brightness[i]).astype(np.float32)
    X += 0.03 * rng.standard_normal(X.shape).astype(np.float32)
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
        "_n_channels": 1,
        "_skip_y_standardization": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = (
        "BONEAGE_ROOT set but unusable; synthetic bone-age proxy"
        if used_root
        else "BONEAGE_ROOT unset; synthetic bone-age proxy"
    )
    return splits


def _parse_bbbc_count(stem: str) -> int | None:
    """Extract the cell count from a BBBC005 filename stem.

    Canonical rule (deepchem): third underscore-part, first char stripped,
    e.g. ``..._C12_...`` → 12. Falls back to the first ``[A-Za-z](digits)``
    group, then to the largest integer in the name.
    """
    import re as _re

    parts = stem.split("_")
    if len(parts) >= 3:
        try:
            return int(parts[2][1:])
        except (ValueError, IndexError):
            pass
    m = _re.search(r"[A-Za-z](\d+)", stem)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    ints = [int(x) for x in _re.findall(r"\d+", stem)]
    return max(ints) if ints else None


def _read_bbbc_images(
    n_samples: int, img_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read (images, counts) from the BBBC005 source, or None if unavailable.

    Source resolution: ``BBBC_ROOT`` (zip or dir) else the cached
    ``BBBC005_v1_images.zip`` (downloaded on demand). Returns
    ``X:(N,1,H,W)`` float32 in [0,1] and ``y:(N,)`` filename counts.
    """
    import os as _os
    import zipfile as _zipfile

    cache_dir = _image_cache_dir()
    zip_path = cache_dir / "BBBC005_v1_images.zip"
    root = _os.environ.get("BBBC_ROOT", "")

    names: list[str] = []
    open_zip: _zipfile.ZipFile | None = None
    tmp_dir: Path | None = None
    if root:
        p = Path(root)
        if p.is_file() and _zipfile.is_zipfile(p):
            zip_path = p
        elif p.is_dir():
            names = sorted([q.name for q in p.rglob("*.TIF")] + [q.name for q in p.rglob("*.tif")])
            tmp_dir = p
    if not names and open_zip is None:
        if not (zip_path.exists() and zip_path.stat().st_size > 0):
            _download_with_progress(_BBBC005_URL, zip_path, desc="BBBC005 images")
        try:
            open_zip = _zipfile.ZipFile(zip_path)
            names = sorted(n for n in open_zip.namelist() if n.lower().endswith((".tif", ".tiff")))
        except Exception:
            names = []
    if not names:
        return None
    try:
        import io as _io

        from PIL import Image as _Image

        rng_pick = np.random.default_rng(seed)
        order = rng_pick.permutation(len(names))
        imgs, counts = [], []
        for idx in order:
            if len(imgs) >= n_samples:
                break
            nm = names[int(idx)]
            cnt = _parse_bbbc_count(Path(nm).stem)
            if cnt is None:
                continue
            if open_zip is not None:
                with open_zip.open(nm) as fh:
                    img = _Image.open(_io.BytesIO(fh.read())).convert("L")
            else:
                assert tmp_dir is not None
                img = _Image.open(tmp_dir / nm).convert("L")
            img = img.resize((img_size, img_size))
            arr = np.array(img, dtype=np.float32) / 255.0
            imgs.append(arr[None, :, :])
            counts.append(float(cnt))
        if open_zip is not None:
            open_zip.close()
        if not imgs:
            return None
        return np.stack(imgs).astype(np.float32), np.array(counts, dtype=np.float32)
    except Exception:
        try:
            if open_zip is not None:
                open_zip.close()
        except Exception:
            pass
        return None


def _load_bbbc_cellcount(
    n_samples: int = 2000,
    img_size: int = 128,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """BBBC005 fluorescence-microscopy cell counting (scalar counts).

    Real path: downloads ``BBBC005_v1_images.zip`` (~1.8 GB, Broad mirror,
    CC-BY) into the harness image cache, or reuses ``BBBC_ROOT`` (zip file
    or extracted dir). Grayscale 696×520 TIFs resized to ``img_size``;
    ``y`` is the filename-encoded cell count (1–100, float32).

    Poisson-native data: variance grows with the count, so this is the
    canonical image testbed for count losses (Poisson/NegBin/ZIP) and the
    Poisson-Gaussian imaging-noise loss.

    Fallback: synthetic cell-blob fields with Poisson-drawn counts.
    Counts are never standardized (``_skip_y_standardization``) — count
    losses require raw non-negative targets.
    """
    got = _read_bbbc_images(n_samples, img_size, seed)
    if got is not None:
        X, y = got
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
        return {
            "X_train": Xtr,
            "X_test": Xte,
            "y_train": ytr,
            "y_test": yte,
            "_image_data": True,
            "_image_size": img_size,
            "_n_channels": 1,
            "_skip_y_standardization": True,
            "_count_data": True,
        }

    # Synthetic fallback: Poisson counts + cell-blob fields.
    rng = np.random.default_rng(seed)
    lam = rng.uniform(5, 60, n_samples)
    y = rng.poisson(lam).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    X = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    for i in range(n_samples):
        canvas = np.zeros((img_size, img_size), dtype=np.float32)
        k = min(int(y[i]), 120)
        if k > 0:
            px = rng.uniform(0, img_size, k)
            py = rng.uniform(0, img_size, k)
            for j in range(k):
                canvas += np.exp(
                    -((xs - px[j]) ** 2 + (ys - py[j]) ** 2) / (2 * (img_size / 40) ** 2)
                )
        X[i, 0] = np.clip(canvas, 0, 1).astype(np.float32)
    X += 0.02 * rng.standard_normal(X.shape).astype(np.float32)
    X = np.clip(X, 0, 1).astype(np.float32)
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
        "_n_channels": 1,
        "_skip_y_standardization": True,
        "_count_data": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = "BBBC005 unavailable; synthetic cell-count proxy"
    return splits


def _load_bbbc_censored(
    n_samples: int = 1000,
    img_size: int = 64,
    censor_rate: float = 0.3,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """BBBC005 cell counts with detection-limit right-censoring.

    Real path: same BBBC005 source as :func:`_load_bbbc_cellcount`
    (cached zip or ``BBBC_ROOT``). Counts above the ``(1-censor_rate)``
    quantile are right-censored: ``y_obs = min(count, threshold)``,
    ``censoring = (count > threshold)``. Mirrors the tabular
    ``synthetic_censored`` semantics on real pixels — the detection-limit
    censoring of real microscopy.

    Fallback: synthetic censored cell-blob fields (censoring path exercised).
    """
    got = _read_bbbc_images(max(n_samples, 200), img_size, seed)
    if got is not None:
        X, y_true = got
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(X))[:n_samples]
        X, y_true = X[order], y_true[order]
        threshold = float(np.quantile(y_true, 1.0 - censor_rate))
        y_obs = np.minimum(y_true, threshold).astype(np.float32)
        codes = (y_true > threshold).astype(np.float32)
        Xtr, Xte, ytr, yte, ctr, cte = train_test_split(
            X, y_obs, codes, test_size=test_size, random_state=seed
        )
        return {
            "X_train": Xtr,
            "X_test": Xte,
            "y_train": ytr,
            "y_test": yte,
            "censoring_train": ctr,
            "censoring_test": cte,
            "_image_data": True,
            "_image_size": img_size,
            "_n_channels": 1,
            "_skip_y_standardization": True,
            "_count_data": True,
        }
    # Synthetic fallback: censored cell-blob fields.
    rng = np.random.default_rng(seed)
    lam = rng.uniform(5, 60, n_samples)
    y_true = rng.poisson(lam).astype(np.float32)
    threshold = float(np.quantile(y_true, 1.0 - censor_rate))
    y_obs = np.minimum(y_true, threshold).astype(np.float32)
    codes = (y_true > threshold).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    X = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    for i in range(n_samples):
        canvas = np.zeros((img_size, img_size), dtype=np.float32)
        k = min(int(y_true[i]), 120)
        if k > 0:
            px = rng.uniform(0, img_size, k)
            py = rng.uniform(0, img_size, k)
            for j in range(k):
                canvas += np.exp(
                    -((xs - px[j]) ** 2 + (ys - py[j]) ** 2) / (2 * (img_size / 40) ** 2)
                )
        X[i, 0] = np.clip(canvas, 0, 1).astype(np.float32)
    X += 0.02 * rng.standard_normal(X.shape).astype(np.float32)
    X = np.clip(X, 0, 1).astype(np.float32)
    Xtr, Xte, ytr, yte, ctr, cte = train_test_split(
        X, y_obs, codes, test_size=test_size, random_state=seed
    )
    splits: SplitsDict = {
        "X_train": Xtr.astype(np.float32),
        "X_test": Xte.astype(np.float32),
        "y_train": ytr.astype(np.float32),
        "y_test": yte.astype(np.float32),
        "censoring_train": ctr.astype(np.float32),
        "censoring_test": cte.astype(np.float32),
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": 1,
        "_skip_y_standardization": True,
        "_count_data": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = "BBBC005 unavailable; synthetic censored-cell proxy"
    return splits


_GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
_GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
_GDC_TILE_META = "https://portal.gdc.cancer.gov/auth/api/v0/tile/metadata"
_GDC_TILE = "https://portal.gdc.cancer.gov/auth/api/v0/tile"


def _prefer_ipv4() -> Any:
    """Patch socket.getaddrinfo to prefer IPv4 (returning a restore closure).

    Some egress networks blackhole IPv6 TLS handshakes (observed: GDC tile
    host hangs past all requests timeouts). Preferring AF_INET keeps data
    loaders functional there; restore afterwards.
    """
    import socket as _socket

    orig = _socket.getaddrinfo

    def _patched(host: Any, *args: Any, **kwargs: Any) -> Any:
        res = orig(host, *args, **kwargs)
        res.sort(key=lambda r: 0 if r[0] == _socket.AF_INET else 1)
        return res

    _socket.getaddrinfo = _patched  # type: ignore[assignment]

    def _restore() -> None:
        _socket.getaddrinfo = orig  # type: ignore[assignment]

    return _restore


def _fetch_bytes(
    session: Any, url: str, cap_mb: float = 8.0, budget_s: float = 20.0
) -> bytes | None:
    """GET with hard caps: total bytes + wall-clock budget.

    Uses a ``curl`` subprocess primary: OS-enforced ``--max-time`` kills
    reliably, unlike in-process socket timeouts (observed: IPv6 TLS
    blackholes stall ``requests`` past all timeout settings on some
    egress networks). Falls back to streamed ``requests`` when curl is
    absent. Returns None on any failure.
    """
    import shutil as _shutil
    import subprocess as _sp

    cap = int(cap_mb * 1024 * 1024)
    if _shutil.which("curl") is not None:
        try:
            pr = _sp.run(
                ["curl", "-sS", "--max-time", str(int(budget_s)), "--compressed", url],
                capture_output=True,
                timeout=budget_s + 10,
            )
            if pr.returncode == 0 and 0 < len(pr.stdout) <= cap:
                return pr.stdout
            return None
        except Exception:
            pass
    import time as _time

    try:
        import requests as _rq

        t0 = _time.monotonic()
        with _rq.get(url, timeout=(5, 15), stream=True) as r:
            if r.status_code != 200:
                return None
            buf = bytearray()
            for chunk in r.iter_content(65536):
                buf.extend(chunk)
                if len(buf) > cap or _time.monotonic() - t0 > budget_s:
                    return None
            return bytes(buf)
    except Exception:
        return None


def _load_tcga_survival(
    n_samples: int = 200,
    img_size: int = 128,
    projects: tuple[str, ...] = ("TCGA-BRCA", "TCGA-LUAD"),
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """TCGA diagnostic-slide thumbnails + overall-survival (right-censored).

    Real path (parallel T2B track): GDC API (open, no key) — ``cases``
    query over ``projects`` for cases with diagnostic slides and complete
    survival (``days_to_death``/``days_to_last_follow_up`` + vital_status);
    slide thumbnails (≤512px) via the GDC tile server (metadata → level →
    tile stitch, requests+PIL only, no WSI storage, no OpenSlide).
    ``y`` is overall-survival days; ``censoring`` follows the harness
    convention (1 = right-censored/alive, 0 = observed/dead).

    Staining/site heterogeneity across projects doubles as an OOD axis.

    Fallback: synthetic lesion fields with Weibull survival times.
    """
    import os as _os

    if _os.environ.get("TCGA_OFFLINE", "") == "1":
        pass  # skip straight to synthetic proxy
    else:
        _restore_dns = _prefer_ipv4()
        try:
            return _load_tcga_survival_real(
                n_samples=n_samples,
                img_size=img_size,
                projects=projects,
                test_size=test_size,
                seed=seed,
            )
        except Exception:
            pass  # fall through to synthetic proxy
        finally:
            _restore_dns()

    # Synthetic fallback: lesion fields + Weibull survival.
    rng = np.random.default_rng(seed)
    n = n_samples
    burden = rng.uniform(0.05, 0.6, n).astype(np.float32)
    scale = 2000.0 * np.exp(-3.0 * burden)
    shape = 1.5
    t_event = (scale * rng.weibull(shape, n)).astype(np.float32)
    t_cens = rng.uniform(500, 4000, n).astype(np.float32)
    observed = np.minimum(t_event, t_cens).astype(np.float32)
    codes = (t_event > t_cens).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    X = np.zeros((n, 3, img_size, img_size), dtype=np.float32)
    for i in range(n):
        tissue = 0.6 + 0.1 * rng.standard_normal((img_size, img_size))
        nl = min(int(1 + burden[i] * 12), 20)
        lx = rng.uniform(0, img_size, nl)
        ly = rng.uniform(0, img_size, nl)
        lesion = np.zeros((img_size, img_size), dtype=np.float32)
        for j in range(nl):
            lesion += np.exp(-((xs - lx[j]) ** 2 + (ys - ly[j]) ** 2) / (2 * (img_size / 25) ** 2))
        frame = np.clip(tissue + np.clip(lesion, 0, 1) * 0.3, 0, 1).astype(np.float32)
        for c in range(3):
            X[i, c] = frame
    X += 0.02 * rng.standard_normal(X.shape).astype(np.float32)
    X = np.clip(X, 0, 1).astype(np.float32)
    Xtr, Xte, ytr, yte, ctr, cte = train_test_split(
        X, observed, codes, test_size=test_size, random_state=seed
    )
    splits: SplitsDict = {
        "X_train": Xtr.astype(np.float32),
        "X_test": Xte.astype(np.float32),
        "y_train": ytr.astype(np.float32),
        "y_test": yte.astype(np.float32),
        "censoring_train": ctr.astype(np.float32),
        "censoring_test": cte.astype(np.float32),
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": 3,
        "_skip_y_standardization": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = "GDC unreachable; synthetic survival proxy"
    return splits


def _load_tcga_survival_real(
    n_samples: int,
    img_size: int,
    projects: tuple[str, ...],
    test_size: float,
    seed: int,
) -> SplitsDict:
    """GDC real path for :func:`_load_tcga_survival` (may raise)."""
    import io as _io
    import math as _math
    import os as _os
    import time as _time

    import requests as _requests
    from PIL import Image as _Image

    t_start = _time.monotonic()
    budget_s = 90.0 + 45.0 * n_samples
    _dbg0 = __import__("os").environ.get("TCGA_DEBUG", "") == "1"

    def _gdc_post(url: str, payload: dict, timeout: int = 30) -> dict:
        import json as _json
        import shutil as _shutil
        import subprocess as _sp

        # curl-first: OS-enforced timeouts (see _fetch_bytes note).
        if _shutil.which("curl") is not None:
            try:
                pr = _sp.run(
                    [
                        "curl",
                        "-sS",
                        "--max-time",
                        str(timeout),
                        "-X",
                        "POST",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        _json.dumps(payload),
                        url,
                    ],
                    capture_output=True,
                    timeout=timeout + 10,
                )
                if pr.returncode == 0 and pr.stdout:
                    return _json.loads(pr.stdout)
            except Exception:
                pass
        last: Exception | None = None
        for _ in range(3):
            try:
                r = _requests.post(url, json=payload, timeout=(10, timeout))
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # noqa: BLE001 — transient egress flakes
                last = exc
        raise RuntimeError(f"GDC POST failed 3x: {url}") from last

    # 1) Cases with diagnostic slides in the projects (pages).
    case_ids: list[str] = []
    fetched = 0
    while len(case_ids) < max(n_samples * 4, 200) and fetched < 3000:
        resp = _gdc_post(
            _GDC_CASES_URL,
            {
                "filters": {
                    "op": "and",
                    "content": [
                        {
                            "op": "in",
                            "content": {
                                "field": "cases.project.project_id",
                                "value": list(projects),
                            },
                        },
                        {
                            "op": "=",
                            "content": {
                                "field": "files.experimental_strategy",
                                "value": "Diagnostic Slide",
                            },
                        },
                    ],
                },
                "fields": "case_id",
                "format": "JSON",
                "size": 500,
                "from": fetched,
            },
        )
        hits = resp.get("data", {}).get("hits", [])
        if not hits:
            break
        case_ids.extend(h["case_id"] for h in hits if h.get("case_id"))
        fetched += len(hits)
    if _dbg0:
        print(f"[tcga] staged cases={len(case_ids)}")
    if not case_ids:
        raise RuntimeError("no slide cases")
    # 2) Survival join via /analysis/survival (time + censored flag).
    surv: dict[str, tuple[float, float]] = {}  # case_id -> (days, event)
    for i in range(0, len(case_ids), 500):
        chunk = case_ids[i : i + 500]
        try:
            sresp = _gdc_post(
                "https://api.gdc.cancer.gov/analysis/survival",
                {"caseSet": {"case_ids": chunk}},
                timeout=60,
            )
            for donor in sresp.get("results", [{}])[0].get("donors", []):
                try:
                    t = float(donor["time"])
                except (TypeError, ValueError, KeyError):
                    continue
                if not np.isfinite(t) or t <= 0:
                    continue
                did = donor.get("id")
                if did:
                    surv[did] = (t, 0.0 if donor.get("censored") else 1.0)
        except Exception:
            continue
    if _dbg0:
        print(f"[tcga] survival donors={len(surv)}")
    if not surv:
        raise RuntimeError("empty survival join")
    # 2) Per case: survival target + one diagnostic-slide file id.
    rng_pick = np.random.default_rng(seed)
    cids = [c for c in case_ids if c in surv]
    order = rng_pick.permutation(len(cids))
    imgs, times, events = [], [], []
    _debug = _os.environ.get("TCGA_DEBUG", "") == "1"
    _dbg_shown = 0
    if _debug:
        print(f"[tcga] cases={len(case_ids)} surv={len(surv)} cids={len(cids)}")
    for idx in order:
        if len(imgs) >= n_samples:
            break
        if _time.monotonic() - t_start > budget_s:
            break  # partial real data beats proxy when sufficient (below)
        cid = cids[int(idx)]
        t, e = surv[cid]
        # Slide file id for this case.
        try:
            fresp = _gdc_post(
                _GDC_FILES_URL,
                {
                    "filters": {
                        "op": "and",
                        "content": [
                            {
                                "op": "=",
                                "content": {
                                    "field": "files.experimental_strategy",
                                    "value": "Diagnostic Slide",
                                },
                            },
                            {"op": "=", "content": {"field": "cases.case_id", "value": cid}},
                        ],
                    },
                    "fields": "file_id",
                    "format": "JSON",
                    "size": 5,
                },
                timeout=30,
            )
            fhits = fresp.get("data", {}).get("hits", [])
            if not fhits:
                if _debug and _dbg_shown < 3:
                    print(f"[tcga] no slide files for case {cid}")
                    _dbg_shown += 1
                continue
            fid = fhits[0]["file_id"]
        except Exception as exc:
            if _debug and _dbg_shown < 3:
                print(f"[tcga] files query failed for case {cid}: {type(exc).__name__}: {exc}")
                _dbg_shown += 1
            continue
        # 3) Thumbnail via tile server (≤512px: fewer tiles, resized
        # later to img_size; keeps per-slide latency in check).
        # NOTE: plain session.get can hang on slow-trickling tiles
        # (timeout= only bounds connect/first-byte gaps), so tiles
        # go through the capped _fetch_bytes helper below.
        try:
            import json as _json

            mraw = _fetch_bytes(None, f"{_GDC_TILE_META}/{fid}", cap_mb=1, budget_s=25)
            if mraw is None:
                if _debug and _dbg_shown < 3:
                    print(f"[tcga] meta fetch failed for {fid}")
                    _dbg_shown += 1
                continue
            md = _json.loads(mraw)
            W, H = int(md["Width"]), int(md["Height"])
            ts = int(md.get("TileSize", 512))
            ov = int(md.get("Overlap", 0))
            step = max(ts - ov, 1)
            level = max(0, int(_math.ceil(_math.log2(max(W, H) / 512.0))))
            tw = -(-W // (2**level))  # ceil-div: thumbnail width at level
            th = -(-H // (2**level))
            nx = max(1, -(-tw // step))
            ny = max(1, -(-th // step))
            if nx * ny > 25:
                if _debug and _dbg_shown < 3:
                    print(f"[tcga] geometry skip {fid}: {W}x{H} level={level} nx={nx} ny={ny}")
                    _dbg_shown += 1
                continue  # skip pathological slide geometries
            canvas = _Image.new("RGB", (nx * step, ny * step), (0, 0, 0))
            max_x = max_y = 0
            for ty in range(ny):
                for tx in range(nx):
                    raw = _fetch_bytes(
                        None,
                        f"{_GDC_TILE}/{fid}?level={level}&x={tx}&y={ty}",
                        cap_mb=8,
                        budget_s=20,
                    )
                    if raw is None:
                        continue  # tolerate missing edge tiles; keep partial
                    try:
                        tile = _Image.open(_io.BytesIO(raw)).convert("RGB")
                    except Exception:
                        continue  # tolerate undecodable edge tiles
                    canvas.paste(tile, (tx * step, ty * step))
                    max_x = max(max_x, tx * step + tile.size[0])
                    max_y = max(max_y, ty * step + tile.size[1])
            if max_x == 0 or max_y == 0:
                if _debug and _dbg_shown < 3:
                    print(f"[tcga] stitch failed {fid}: max=({max_x},{max_y})")
                    _dbg_shown += 1
                continue
            canvas = canvas.crop((0, 0, max_x, max_y)).resize((img_size, img_size))
        except Exception as exc:
            if _debug and _dbg_shown < 3:
                print(f"[tcga] tile block failed {fid}: {type(exc).__name__}: {exc}")
                _dbg_shown += 1
            continue
        arr = np.array(canvas, dtype=np.float32) / 255.0
        imgs.append(arr.transpose(2, 0, 1))
        times.append(t)
        events.append(e)
    # Partial acceptance: flaky egress may cap the harvest; use what we
    # have when sufficient for a split, else raise → synthetic fallback.
    # Threshold scales with the request (tiny smokes ask for few slides).
    if len(imgs) >= min(n_samples, 8) and len(imgs) >= 2:
        X = np.stack(imgs).astype(np.float32)
        T = np.array(times, dtype=np.float32)
        E = np.array(events, dtype=np.float32)
        C = (1.0 - E).astype(np.float32)  # harness: 1 = censored
        Xtr, Xte, Ytr, Yte, Ctr, Cte = train_test_split(
            X, T, C, test_size=test_size, random_state=seed
        )
        return {
            "X_train": Xtr,
            "X_test": Xte,
            "y_train": Ytr,
            "y_test": Yte,
            "censoring_train": Ctr,
            "censoring_test": Cte,
            "_image_data": True,
            "_image_size": img_size,
            "_n_channels": 3,
            "_skip_y_standardization": True,
            "_n_real_slides": len(imgs),
        }
    raise RuntimeError(f"only {len(imgs)} slides harvested")


def _load_appa_age(
    n_samples: int = 4000,
    img_size: int = 64,
    seed: int = 42,
    **kwargs: Any,
) -> SplitsDict:
    """APPA-REAL apparent age with observed rater disagreement.

    Real path: downloads ``appa-real-release.zip`` (~845 MB, ChaLearn LAP,
    direct link, no login) into the harness image cache, or reuses
    ``APPA_ROOT`` (zip or extracted dir). Prefers cropped ``*_face.jpg``
    faces; ``y`` is the apparent-age mean and ``y_std_*`` the per-image
    rater std from ``gt_avg_*.csv`` (official train+valid → train pool,
    test → test). Cap with ``n_samples`` (stratified per split).

    The observed std is the aleatoric gold standard: methods that accept
    ``target_variance`` (e.g. NoisyTargetGaussianNLL) train on it; others
    train on the mean. ``_observed_std`` flags the metadata.

    Fallback: synthetic faces with heteroscedastic rater noise.
    """
    import csv as _csv
    import os as _os
    import zipfile as _zipfile

    cache_dir = _image_cache_dir()
    zip_path = cache_dir / "appa-real-release.zip"
    root = _os.environ.get("APPA_ROOT", "")

    def _read_split(
        base: Path, split: str, from_zip: _zipfile.ZipFile | None
    ) -> tuple[list, list, list] | None:
        # gt_avg_<split>.csv summaries: n_ratings, apparent mean/std, real age.
        avg_name = f"gt_avg_{'train' if split == 'train' else split}.csv"
        if split == "valid":
            avg_name = "gt_avg_valid.csv"
        rows: list[dict[str, str]] = []
        try:
            if from_zip is not None:
                candidates = [n for n in from_zip.namelist() if n.endswith(avg_name)]
                if not candidates:
                    return None
                import io as _io

                with from_zip.open(candidates[0]) as fh:
                    text = _io.TextIOWrapper(fh).read()
                rows = list(_csv.DictReader(text.splitlines()))
            else:
                cands = sorted(base.rglob(avg_name))
                if not cands:
                    return None
                with open(cands[0], newline="") as fh:
                    rows = list(_csv.DictReader(fh))
        except Exception:
            return None
        if not rows:
            return None
        fieldnames = list(rows[0].keys())
        lower = {f.lower(): f for f in fieldnames}

        def _pick(*names: str) -> str | None:
            for nm in names:
                if nm in lower:
                    return lower[nm]
            for f in fieldnames:
                fl = f.lower()
                if any(nm in fl for nm in names):
                    return f
            return None

        f_file = _pick("file", "image", "name")
        f_mean = _pick("apparent", "average", "mean")
        f_std = _pick("std", "deviation")
        if f_file is None or f_mean is None:
            return None
        try:
            import io as _io

            from PIL import Image as _Image

            imgs, means, stds = [], [], []
            for row in rows:
                fn = row[f_file].strip()
                cand_names = (
                    [fn + "_face.jpg", fn]
                    if not fn.endswith(".jpg")
                    else [
                        fn.replace(".jpg", "_face.jpg"),
                        fn,
                    ]
                )
                data: bytes | None = None
                if from_zip is not None:
                    for cn in cand_names:
                        hits = [n for n in from_zip.namelist() if n.endswith(cn)]
                        if hits:
                            with from_zip.open(hits[0]) as fh:
                                data = fh.read()
                            break
                else:
                    for cn in cand_names:
                        hits = sorted(base.rglob(cn))
                        if hits:
                            data = hits[0].read_bytes()
                            break
                if data is None:
                    continue
                try:
                    mu = float(row[f_mean])
                except ValueError:
                    continue
                try:
                    sd = float(row[f_std]) if f_std else 0.0
                except ValueError:
                    sd = 0.0
                img = _Image.open(_io.BytesIO(data)).convert("RGB").resize((img_size, img_size))
                arr = np.array(img, dtype=np.float32) / 255.0
                imgs.append(arr.transpose(2, 0, 1))
                means.append(mu)
                stds.append(max(sd, 0.0))
            if not imgs:
                return None
            return imgs, means, stds
        except Exception:
            return None

    Xtr_all: list = []
    ytr_all: list = []
    str_all: list = []
    Xte: list = []
    yte: list = []
    ste: list = []
    used_real = False
    # Resolve source: APPA_ROOT dir/zip, else cached/downloaded zip.
    src_dir: Path | None = None
    src_zip: _zipfile.ZipFile | None = None
    if root:
        p = Path(root)
        if p.is_file() and _zipfile.is_zipfile(p):
            try:
                src_zip = _zipfile.ZipFile(p)
            except Exception:
                src_zip = None
        elif p.is_dir():
            src_dir = p
    if src_dir is None and src_zip is None:
        if not (zip_path.exists() and zip_path.stat().st_size > 0):
            try:
                _download_with_progress(_APPA_URL, zip_path, desc="APPA-REAL")
            except Exception:
                zip_path = cache_dir / "appa-real-release.zip"  # partial cleanup best-effort
        if zip_path.exists() and zip_path.stat().st_size > 0:
            try:
                src_zip = _zipfile.ZipFile(zip_path)
            except Exception:
                src_zip = None
    base_arg: Any = src_dir if src_dir is not None else Path(".")
    for split, is_test in (("train", False), ("valid", False), ("test", True)):
        got = _read_split(base_arg, split, src_zip)
        if got is None:
            continue
        used_real = True
        imgs, means, stds = got
        if is_test:
            Xte.extend(imgs)
            yte.extend(means)
            ste.extend(stds)
        else:
            Xtr_all.extend(imgs)
            ytr_all.extend(means)
            str_all.extend(stds)
    try:
        if src_zip is not None:
            src_zip.close()
    except Exception:
        pass
    if used_real and Xtr_all and Xte:
        rng = np.random.default_rng(seed)
        # Cap per split deterministically.
        n_tr = min(len(Xtr_all), int(n_samples * 0.8))
        n_te = min(len(Xte), n_samples - n_tr)
        idx_tr = rng.choice(len(Xtr_all), n_tr, replace=False)
        idx_te = rng.choice(len(Xte), n_te, replace=False)
        X_train = np.stack([Xtr_all[i] for i in idx_tr]).astype(np.float32)
        y_train = np.array([ytr_all[i] for i in idx_tr], dtype=np.float32)
        y_std_train = np.array([str_all[i] for i in idx_tr], dtype=np.float32)
        X_test = np.stack([Xte[i] for i in idx_te]).astype(np.float32)
        y_test = np.array([yte[i] for i in idx_te], dtype=np.float32)
        y_std_test = np.array([ste[i] for i in idx_te], dtype=np.float32)
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_std_train": y_std_train,
            "y_std_test": y_std_test,
            "_image_data": True,
            "_image_size": img_size,
            "_n_channels": 3,
            "_skip_y_standardization": True,
            "_observed_std": True,
        }

    # Synthetic fallback: heteroscedastic apparent-age judgements.
    rng = np.random.default_rng(seed)
    n = n_samples
    age = rng.uniform(5, 85, n).astype(np.float32)
    rater_std = (1.5 + 0.08 * np.abs(age - 35)).astype(np.float32)
    y = (age + rng.normal(0, 1, n)).astype(np.float32)
    y_std = np.abs(rater_std + rng.normal(0, 0.5, n)).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    X = np.zeros((n, 3, img_size, img_size), dtype=np.float32)
    for i in range(n):
        cx = img_size / 2 + (y[i] - 45) / 80 * img_size * 0.3
        sigma = img_size / 8
        blob = np.exp(-((xs - cx) ** 2 + (ys - img_size / 2) ** 2) / (2 * sigma**2))
        for c in range(3):
            X[i, c] = (blob * (0.6 + 0.1 * c)).astype(np.float32)
    X += 0.05 * rng.standard_normal(X.shape).astype(np.float32)
    X_train, X_test, y_train, y_test, y_std_train, y_std_test = train_test_split(
        X, y, y_std, test_size=0.2, random_state=seed
    )
    splits: SplitsDict = {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "y_std_train": y_std_train.astype(np.float32),
        "y_std_test": y_std_test.astype(np.float32),
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": 3,
        "_skip_y_standardization": True,
        "_observed_std": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = "APPA-REAL unavailable; synthetic apparent-age proxy"
    return splits


def _density_from_points(points: np.ndarray, h: int, w: int, sigma: float = 4.0) -> np.ndarray:
    """Render a crowd density map from head points (count = map sum)."""
    from scipy.ndimage import gaussian_filter

    dens = np.zeros((h, w), dtype=np.float32)
    if len(points):
        pts = np.asarray(points, dtype=np.float32)
        xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
        np.add.at(dens, (ys, xs), 1.0)
        if sigma > 0:
            dens = gaussian_filter(dens, sigma=sigma).astype(np.float32)
    return dens


def _load_shanghai_crowd(
    n_samples: int = 400,
    img_size: int = 128,
    part: str = "B",
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """ShanghaiTech crowd counting (density maps + counts).

    Real path: downloads the dataset zip (~1 GB, Dropbox mirror) into the
    harness image cache, or reuses ``SHANGHAI_ROOT`` (zip or extracted dir).
    Part B (busy Shanghai streets, 400 train / 316 test) is the default;
    Part A (congested web images) via ``part='A'``. Head-point ``.mat``
    annotations become Gaussian density maps (sigma=4 at native scale);
    images are resized to ``img_size`` with points rescaled accordingly.
    ``y`` is the density map (N,H,W); ``count_*`` carries the head counts.

    Fallback: synthetic crowd fields with Poisson head counts.
    """
    import os as _os
    import zipfile as _zipfile

    cache_dir = _image_cache_dir()
    zip_path = cache_dir / "ShanghaiTech_Crowd_Counting_Dataset.zip"
    root = _os.environ.get("SHANGHAI_ROOT", "")

    def _collect(base: Path) -> tuple[list[Path], list[Path]] | None:
        part_dir = base / f"part_{part}_final"
        if not part_dir.is_dir():
            # Maybe rooted one level deeper (extracted inner folder).
            subs = [d for d in base.glob(f"part_{part}_final") if d.is_dir()]
            if not subs:
                return None
            part_dir = subs[0]
        imgs: list[Path] = []
        for split in ("train_data", "test_data"):
            d = part_dir / split / "images"
            if d.is_dir():
                imgs.extend(sorted(d.glob("*.jpg")))
        if not imgs:
            return None
        return imgs, [part_dir] * len(imgs)

    src_dir: Path | None = None
    src_zip: _zipfile.ZipFile | None = None
    if root:
        p = Path(root)
        if p.is_file() and _zipfile.is_zipfile(p):
            try:
                src_zip = _zipfile.ZipFile(p)
            except Exception:
                src_zip = None
        elif p.is_dir():
            src_dir = p
    if src_dir is None and src_zip is None:
        if not (zip_path.exists() and zip_path.stat().st_size > 0):
            try:
                _download_with_progress(_SHANGHAI_URL, zip_path, desc="ShanghaiTech")
            except Exception:
                pass
        if zip_path.exists() and zip_path.stat().st_size > 0:
            try:
                src_zip = _zipfile.ZipFile(zip_path)
            except Exception:
                src_zip = None
    # Materialise zip members under cache/shanghai when using a zip source.
    mat_dir: Path | None = src_dir
    if src_zip is not None:
        mat_dir = cache_dir / "shanghai_extract"
        try:
            # Extract only Part-B (or requested part) members to bound disk use.
            wanted = [n for n in src_zip.namelist() if f"part_{part}_final" in n]
            existing = sum(1 for _ in mat_dir.rglob("*.jpg")) if mat_dir.exists() else 0
            if existing < 10 and wanted:
                mat_dir.mkdir(parents=True, exist_ok=True)
                for n in wanted:
                    src_zip.extract(n, mat_dir)
        except Exception:
            pass
        try:
            src_zip.close()
        except Exception:
            pass
    pairs = _collect(mat_dir) if mat_dir is not None else None
    if pairs is not None:
        try:
            import scipy.io as _sio
            from PIL import Image as _Image

            imgs_all, counts_all, dens_all = [], [], []
            rng_pick = np.random.default_rng(seed)
            order = rng_pick.permutation(len(pairs[0]))[:n_samples]
            for idx in order:
                jp = pairs[0][int(idx)]
                gt = jp.parent.parent / "ground_truth" / f"GT_{jp.stem}.mat"
                if not gt.is_file():
                    continue
                mat = _sio.loadmat(str(gt))
                try:
                    pts = np.asarray(mat["image_info"][0, 0][0, 0][0], dtype=np.float32)
                except Exception:
                    continue
                if pts.ndim == 1:
                    pts = pts.reshape(-1, 2)
                img = _Image.open(jp).convert("RGB")
                w0, h0 = img.size
                img = img.resize((img_size, img_size))
                sx, sy = img_size / max(w0, 1), img_size / max(h0, 1)
                pts_s = pts * np.array([sx, sy], dtype=np.float32)
                dens = _density_from_points(pts_s, img_size, img_size, sigma=4.0)
                arr = np.array(img, dtype=np.float32) / 255.0
                imgs_all.append(arr.transpose(2, 0, 1))
                dens_all.append(dens)
                counts_all.append(float(len(pts)))
            if imgs_all:
                X = np.stack(imgs_all).astype(np.float32)
                Y = np.stack(dens_all).astype(np.float32)
                C = np.array(counts_all, dtype=np.float32)
                Xtr, Xte, Ytr, Yte, Ctr, Cte = train_test_split(
                    X, Y, C, test_size=test_size, random_state=seed
                )
                return {
                    "X_train": Xtr,
                    "X_test": Xte,
                    "y_train": Ytr,
                    "y_test": Yte,
                    "count_train": Ctr,
                    "count_test": Cte,
                    "_image_data": True,
                    "_image_size": img_size,
                    "_n_channels": 3,
                    "_dense_output": True,
                    "_skip_y_standardization": True,
                }
        except Exception:
            pass  # fall through to synthetic proxy

    # Synthetic fallback: Poisson crowd counts + blob fields.
    rng = np.random.default_rng(seed)
    lam = rng.uniform(10, 120, n_samples)
    counts = rng.poisson(lam).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    X = np.zeros((n_samples, 3, img_size, img_size), dtype=np.float32)
    Y = np.zeros((n_samples, img_size, img_size), dtype=np.float32)
    for i in range(n_samples):
        k = min(int(counts[i]), 200)
        canvas = np.zeros((img_size, img_size), dtype=np.float32)
        if k > 0:
            px = rng.uniform(0, img_size, k)
            py = rng.uniform(0, img_size, k)
            for j in range(k):
                canvas += np.exp(
                    -((xs - px[j]) ** 2 + (ys - py[j]) ** 2) / (2 * (img_size / 32) ** 2)
                )
        Y[i] = canvas.astype(np.float32)
        shade = 0.4 + 0.4 * rng.random()
        for c in range(3):
            X[i, c] = np.clip(canvas * shade + 0.05 * rng.standard_normal(canvas.shape), 0, 1)
    X = X.astype(np.float32)
    idx = np.arange(n_samples)
    Xtr, Xte, Ytr, Yte, Ctr, Cte = train_test_split(
        X, Y, counts, test_size=test_size, random_state=seed
    )
    splits: SplitsDict = {
        "X_train": Xtr.astype(np.float32),
        "X_test": Xte.astype(np.float32),
        "y_train": Ytr.astype(np.float32),
        "y_test": Yte.astype(np.float32),
        "count_train": Ctr.astype(np.float32),
        "count_test": Cte.astype(np.float32),
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": 3,
        "_dense_output": True,
        "_skip_y_standardization": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = "ShanghaiTech unavailable; synthetic crowd proxy"
    return splits


def _load_ixi_brainage(
    n_samples: int = 400,
    img_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """IXI T1 brain-age regression (scalar years, 20–82).

    Real path: ``IXI_ROOT`` containing ``IXI-T1.tar`` (or an extracted dir
    of ``*.nii.gz``) plus ``IXI.xls`` demographics. Uses the mid-axial
    slice of each T1 volume, resized to ``img_size``; ``y`` is age in years.
    Requires ``nibabel`` + ``openpyxl``/``xlrd`` (optional harness deps —
    install with ``pip install nibabel openpyxl``; a friendly ImportError
    otherwise). The IC download server blocks automated clients, so no
    headless auto-download is attempted — stage the files manually.

    Long-tail ages (few 70+ subjects) make this the canonical imbalanced
    image testbed (LDS/DensityWeighted/BalancedMSE).

    Fallback: synthetic atrophy brains — ventricle size grows with age
    (mechanistic ground truth: y = age, ventricle radius ∝ age).
    """
    import os as _os

    root = _os.environ.get("IXI_ROOT", "")
    if root:
        try:
            import nibabel as _nib  # ty: ignore[unresolved-import]  # optional dep for IXI_ROOT
        except ImportError as exc:
            raise ImportError(
                "nibabel is required for IXI brain age "
                "(pip install nibabel). Set IXI_ROOT or unset it for the proxy."
            ) from exc
        try:
            p = Path(root)
            tar = p / "IXI-T1.tar" if p.is_dir() else (p if p.suffix == ".tar" else None)
            workdir = p
            if tar is not None and tar.is_file():
                import tarfile as _tarfile

                workdir = p / "IXI-T1-extract" if p.is_dir() else p.parent / "IXI-T1-extract"
                if not any(workdir.glob("*.nii.gz")):
                    workdir.mkdir(parents=True, exist_ok=True)
                    with _tarfile.open(tar) as tf:
                        tf.extractall(workdir)
            niis = sorted(workdir.glob("*.nii.gz")) or sorted(workdir.rglob("*.nii.gz"))
            # IXI.xls → subject id → age.
            ages: dict[str, float] = {}
            xls = p / "IXI.xls" if p.is_dir() else p.parent / "IXI.xls"
            if xls.is_file():
                try:
                    import pandas as _pd

                    df = _pd.read_excel(xls, engine=None)
                    cols = {str(c).upper(): c for c in df.columns}
                    id_col = next(
                        (cols[k] for k in cols if "IXI" in k or "ID" in k or "SUBJECT" in k),
                        df.columns[0],
                    )
                    age_col = next((cols[k] for k in cols if "AGE" in k), df.columns[1])
                    for _, row in df.iterrows():
                        try:
                            ages[str(int(row[id_col])).zfill(3)] = float(row[age_col])
                        except (ValueError, TypeError):
                            continue
                except Exception:
                    pass
            if niis:
                try:
                    from PIL import Image as _Image

                    rng_pick = np.random.default_rng(seed)
                    order = rng_pick.permutation(len(niis))[:n_samples]
                    imgs, y = [], []
                    for idx in order:
                        fp = niis[int(idx)]
                        sid = "".join(ch for ch in fp.stem.split(".")[0] if ch.isdigit())[-3:]
                        if sid not in ages:
                            continue
                        vol = np.asarray(_nib.load(str(fp)).get_fdata(), dtype=np.float32)
                        z = vol.shape[2] // 2  # mid-axial slice
                        sl = vol[:, :, z]
                        lo, hi = float(np.percentile(sl, 1)), float(np.percentile(sl, 99))
                        sl = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
                        img = _Image.fromarray((sl * 255).astype(np.uint8)).resize(
                            (img_size, img_size)
                        )
                        arr = np.array(img, dtype=np.float32) / 255.0
                        imgs.append(arr[None, :, :])
                        y.append(ages[sid])
                        if len(imgs) >= n_samples:
                            break
                    if imgs:
                        X = np.stack(imgs).astype(np.float32)
                        yy = np.array(y, dtype=np.float32)
                        Xtr, Xte, ytr, yte = train_test_split(
                            X, yy, test_size=test_size, random_state=seed
                        )
                        return {
                            "X_train": Xtr,
                            "X_test": Xte,
                            "y_train": ytr,
                            "y_test": yte,
                            "_image_data": True,
                            "_image_size": img_size,
                            "_n_channels": 1,
                            "_skip_y_standardization": True,
                        }
                except Exception:
                    pass  # fall through to synthetic proxy
        except Exception:
            pass  # fall through to synthetic proxy (tar/xls/nibabel issues)

    # Synthetic fallback: atrophy brains (ventricles grow with age).
    rng = np.random.default_rng(seed)
    n = n_samples
    age = rng.uniform(20, 82, n).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    cx = img_size / 2
    X = np.zeros((n, 1, img_size, img_size), dtype=np.float32)
    for i in range(n):
        skull_r = img_size * 0.42
        skull = np.exp(-((np.sqrt((xs - cx) ** 2 + (ys - cx) ** 2) - skull_r) ** 2) / 8.0)
        vent_r = img_size * (0.04 + 0.10 * (age[i] - 20) / 62)  # atrophy
        vent = np.exp(-((xs - cx) ** 2 + (ys - cx) ** 2) / (2 * vent_r**2))
        brain = np.clip(skull * 0.5 + 0.5 - vent * 0.6, 0, 1)
        X[i, 0] = (brain + 0.03 * rng.standard_normal(brain.shape)).astype(np.float32)
    X = np.clip(X, 0, 1).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, age, test_size=test_size, random_state=seed
    )
    splits: SplitsDict = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": 1,
        "_skip_y_standardization": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = "IXI_ROOT unset; synthetic atrophy-brain proxy"
    return splits


def _load_udacity_steering(
    n_samples: int = 2000,
    img_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """Udacity dashboard-cam steering-angle regression (radians, ~[-1, 1]).

    Real path: ``UDACITY_ROOT`` with extracted frames — either
    ``center/*.jpg`` + ``steering.csv``/``interpolated.csv`` (rwightman
    reader layout; steering column matched case-insensitively, image column
    matched against frame/filename/image), or SullyChen-style ``data.txt``
    lines of ``<relpath> <angle>``. Images are RGB-resized to ``img_size``.

    Straight-driving dominates (angle ≈ 0 long tail) — canonical imbalanced
    scalar testbed alongside brain age.

    Fallback: synthetic perspective roads whose lane curvature sets the
    steering angle (mechanistic ground truth + matching imbalance).
    """
    import csv as _csv
    import os as _os

    root = _os.environ.get("UDACITY_ROOT", "")
    if root:
        try:
            from PIL import Image as _Image

            base = Path(root)
            pairs: list[tuple[Path, float]] = []
            # Layout A: center/*.jpg + steering csv.
            centers = sorted((base / "center").glob("*.jpg")) if (base / "center").is_dir() else []
            csvs = sorted(base.glob("*.csv"))
            if centers and csvs:
                for csv_path in csvs:
                    try:
                        with open(csv_path, newline="") as fh:
                            reader = _csv.DictReader(fh)
                            if not reader.fieldnames:
                                continue
                        fn = {str(c).lower(): c for c in reader.fieldnames}
                        f_ang = next((fn[k] for k in fn if "steer" in k or "angle" in k), None)
                        f_img = next(
                            (
                                fn[k]
                                for k in fn
                                if "frame" in k or "file" in k or "image" in k or "name" in k
                            ),
                            None,
                        )
                        if f_ang is None:
                            continue
                        with open(csv_path, newline="") as fh:
                            for row in _csv.DictReader(fh):
                                try:
                                    ang = float(row[f_ang])
                                except (ValueError, TypeError, KeyError):
                                    continue
                                if f_img is not None:
                                    jp = base / "center" / Path(str(row[f_img])).name
                                    if not jp.is_file():
                                        jp = base / str(row[f_img])
                                else:
                                    continue
                                if jp.is_file():
                                    pairs.append((jp, ang))
                        if pairs:
                            break
                    except Exception:
                        continue
            # Layout B: data.txt lines "<relpath> <angle>".
            if not pairs:
                for txt in ("data.txt", "labels.txt"):
                    tp = base / txt
                    if not tp.is_file():
                        continue
                    try:
                        for line in tp.read_text().splitlines():
                            parts = line.split()
                            if len(parts) < 2:
                                continue
                            try:
                                ang = float(parts[-1])
                            except ValueError:
                                continue
                            jp = base / parts[0]
                            if jp.is_file():
                                pairs.append((jp, ang))
                    except Exception:
                        continue
                    if pairs:
                        break
            if pairs:
                rng_pick = np.random.default_rng(seed)
                order = rng_pick.permutation(len(pairs))[:n_samples]
                imgs, angs = [], []
                for idx in order:
                    jp, ang = pairs[int(idx)]
                    try:
                        img = _Image.open(jp).convert("RGB").resize((img_size, img_size))
                    except Exception:
                        continue
                    arr = np.array(img, dtype=np.float32) / 255.0
                    imgs.append(arr.transpose(2, 0, 1))
                    angs.append(float(np.clip(ang, -1.5, 1.5)))
                    if len(imgs) >= n_samples:
                        break
                if imgs:
                    X = np.stack(imgs).astype(np.float32)
                    y = np.array(angs, dtype=np.float32)
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
                        "_skip_y_standardization": True,
                    }
        except Exception:
            pass  # fall through to synthetic proxy

    # Synthetic fallback: perspective roads, curvature → steering.
    rng = np.random.default_rng(seed)
    n = n_samples
    # Imbalanced: mostly straight (|angle| < 0.1), rare sharp turns.
    straight = rng.random(n) < 0.8
    angle = np.where(
        straight,
        rng.normal(0, 0.03, n),
        rng.uniform(-1, 1, n),
    ).astype(np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    X = np.zeros((n, 3, img_size, img_size), dtype=np.float32)
    for i in range(n):
        horizon = img_size * 0.4
        curve = angle[i] * img_size * 0.5
        t = np.clip((ys - horizon) / max(img_size - horizon, 1), 0, 1)
        lane_c = img_size / 2 + curve * t**2
        road = np.exp(-((xs - lane_c) ** 2) / (2 * (img_size * 0.08) ** 2))
        road = np.where(ys > horizon, road, 0.05)
        sky = np.where(ys <= horizon, 0.5, 0.0)
        frame = np.clip(road * 0.8 + sky, 0, 1).astype(np.float32)
        for c in range(3):
            X[i, c] = frame
    X += 0.03 * rng.standard_normal(X.shape).astype(np.float32)
    X = np.clip(X, 0, 1).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, angle, test_size=test_size, random_state=seed
    )
    splits: SplitsDict = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": 3,
        "_skip_y_standardization": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = "UDACITY_ROOT unset; synthetic steering proxy"
    return splits


def _load_kitti_depth(
    n_samples: int = 1000,
    img_size: int = 128,
    max_depth: float = 80.0,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """KITTI outdoor driving depth (sparse LiDAR, metres).

    Real path: ``KITTI_ROOT`` with torch-uncertainty KITTIDepth layout —
    ``train/{leftImg8bit,leftDepth}`` + ``val/...`` of paired ``*.png``
    (depth uint16, metres = px/256). RGB resized to ``img_size``; depth
    resized nearest; invalid pixels (≤0 or >max_depth) are 0 with
    ``mask_*`` float maps (1 = valid). ``_sparse_depth`` flags the
    masked-dense contract for suites (masked losses, valid-only metrics).

    No auto-download: RGB requires per-drive raw zips (~175 GB total).
    Stage once via ``torch_uncertainty.datasets.KITTIDepth(root,
    download=True, remove_unused=True)`` or mirror the layout manually.

    Fallback: synthetic street scenes with LiDAR-style sparse sampling
    (valid-pixel mask included, so the masked path is exercised in CI).
    """
    import os as _os

    root = _os.environ.get("KITTI_ROOT", "")
    if root:
        try:
            from PIL import Image as _Image

            base = Path(root)
            # Accept both <root>/{train,val} and <root>/KITTIDepth/{train,val}.
            cand = base if (base / "train").is_dir() else base / "KITTIDepth"
            pairs: list[tuple[Path, Path]] = []
            for split in ("train", "val"):
                img_d = cand / split / "leftImg8bit"
                dep_d = cand / split / "leftDepth"
                if not (img_d.is_dir() and dep_d.is_dir()):
                    continue
                for jp in sorted(img_d.glob("*.png")):
                    dp = dep_d / (jp.stem + ".png")
                    if dp.is_file():
                        pairs.append((jp, dp))
            if pairs:
                rng_pick = np.random.default_rng(seed)
                order = rng_pick.permutation(len(pairs))[:n_samples]
                Xs, Ys, Ms = [], [], []
                for idx in order:
                    jp, dp = pairs[int(idx)]
                    try:
                        img = _Image.open(jp).convert("RGB").resize((img_size, img_size))
                        dep = _Image.open(dp).resize(
                            (img_size, img_size), _Image.Resampling.NEAREST
                        )
                    except Exception:
                        continue
                    arr = np.array(img, dtype=np.float32) / 255.0
                    d = np.array(dep, dtype=np.float32) / 256.0
                    m = ((d > 0) & (d <= max_depth)).astype(np.float32)
                    d = np.where(m > 0, d, 0.0).astype(np.float32)
                    if m.sum() < 100:
                        continue
                    Xs.append(arr.transpose(2, 0, 1))
                    Ys.append(d)
                    Ms.append(m)
                    if len(Xs) >= n_samples:
                        break
                if Xs:
                    X = np.stack(Xs).astype(np.float32)
                    Y = np.stack(Ys).astype(np.float32)
                    M = np.stack(Ms).astype(np.float32)
                    Xtr, Xte, Ytr, Yte, Mtr, Mte = train_test_split(
                        X, Y, M, test_size=test_size, random_state=seed
                    )
                    return {
                        "X_train": Xtr,
                        "X_test": Xte,
                        "y_train": Ytr,
                        "y_test": Yte,
                        "mask_train": Mtr,
                        "mask_test": Mte,
                        "_image_data": True,
                        "_image_size": img_size,
                        "_n_channels": 3,
                        "_dense_output": True,
                        "_sparse_depth": True,
                        "_skip_y_standardization": True,
                    }
        except Exception:
            pass  # fall through to synthetic proxy

    # Synthetic fallback: street gradient + boxes, sparse LiDAR mask.
    rng = np.random.default_rng(seed)
    n = n_samples
    X = np.zeros((n, 3, img_size, img_size), dtype=np.float32)
    Y = np.zeros((n, img_size, img_size), dtype=np.float32)
    M = np.zeros((n, img_size, img_size), dtype=np.float32)
    ys, xs = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    for i in range(n):
        horizon = img_size * rng.uniform(0.3, 0.5)
        depth = np.where(
            ys > horizon,
            5 + 60 * (ys - horizon) / max(img_size - horizon, 1),
            80.0,
        ).astype(np.float32)
        # Foreground boxes (cars) at random depths.
        for _ in range(int(rng.integers(1, 4))):
            x0 = rng.uniform(0, img_size * 0.7)
            y0 = rng.uniform(horizon, img_size * 0.8)
            bw, bh = rng.uniform(8, 24), rng.uniform(6, 18)
            box = (np.abs(xs - x0) < bw) & (np.abs(ys - y0) < bh)
            depth = np.where(box, rng.uniform(5, 30), depth)
        shade = np.clip(1 - depth / 90, 0.1, 1).astype(np.float32)
        for c in range(3):
            X[i, c] = np.clip(shade + 0.03 * rng.standard_normal(shade.shape), 0, 1)
        Y[i] = depth
        # LiDAR-style sparsity: keep ~15% of pixels (rows-biased scanlines).
        keep = (rng.random((img_size, img_size)) < 0.15).astype(np.float32)
        keep[::4, :] = np.maximum(
            keep[::4, :],
            (rng.random((img_size // 4 + 1, img_size)) < 0.5).astype(np.float32)[
                : keep[::4, :].shape[0]
            ],
        )
        M[i] = keep
    Y = np.where(M > 0, Y, 0).astype(np.float32)
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
        X, Y, M, test_size=test_size, random_state=seed
    )
    splits: SplitsDict = {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "mask_train": mask_train.astype(np.float32),
        "mask_test": mask_test.astype(np.float32),
        "_image_data": True,
        "_image_size": img_size,
        "_n_channels": 3,
        "_dense_output": True,
        "_sparse_depth": True,
        "_skip_y_standardization": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = "KITTI_ROOT unset; synthetic sparse-depth proxy"
    return splits


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
    "survset_hdfail": DatasetInfo(
        name="survset_hdfail",
        category="tabular",
        n_features=41,
        n_samples=52422,
        description=(
            "Backblaze hard-drive failure survival (SurvSet hdfail): real "
            "right-censored days-to-failure + drive telemetry codes."
        ),
        loader=_registry_loader(_load_survset_hdfail, n_samples=20000),
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
    "folktables_income": DatasetInfo(
        name="folktables_income",
        category="tabular",
        n_features=70,
        n_samples=150000,
        description=(
            "US Census income regression (Folktables ACS 2018 CA): "
            "continuous PINCP on the published ACSIncome cohort "
            "(Ding et al., NeurIPS 2021). Modern large-scale benchmark."
        ),
        loader=_registry_loader(_load_folktables_income, n_samples=20000),
    ),
    "folktables_decile": DatasetInfo(
        name="folktables_decile",
        category="tabular",
        n_features=70,
        n_samples=150000,
        description=(
            "US Census income-BRACKET ordinal task (Folktables ACS 2018 CA): "
            "income decile 0-9, ordered, naturally imbalanced."
        ),
        loader=_registry_loader(_load_folktables_decile, n_samples=20000),
    ),
    "energy_efficiency": DatasetInfo(
        name="energy_efficiency",
        category="tabular",
        n_features=8,
        n_samples=768,
        description=(
            "UCI Energy Efficiency (Tsanas & Xifara 2012): heating/cooling "
            "loads. Standard real-data MULTI-TARGET regression benchmark."
        ),
        loader=_registry_loader(_load_energy_efficiency),
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
    "biwi_head_pose_tiny": DatasetInfo(
        name="biwi_head_pose_tiny",
        category="image",
        n_features=-1,
        n_samples=200,
        description=(
            "BIWI tiny smoke variant: 200 samples at 32×32 for fast CPU debug. "
            "Same loader as biwi_head_pose."
        ),
        loader=_registry_loader(_load_biwi_head_pose, n_samples=200, img_size=32),
    ),
    "utkface_age": DatasetInfo(
        name="utkface_age",
        category="image",
        n_features=-1,
        n_samples=2000,
        description=(
            "UTKFace age regression (scalar years, 0–100). Real data via "
            "UTKFACE_ROOT env (<age>_<gender>_<race>_*.jpg); synthetic "
            "face-blob proxy otherwise (flagged). Small-real P0 default."
        ),
        loader=_registry_loader(_load_utkface_age),
    ),
    "utkface_age_tiny": DatasetInfo(
        name="utkface_age_tiny",
        category="image",
        n_features=-1,
        n_samples=200,
        description=("UTKFace tiny smoke variant: 200 samples at 32×32 for fast CPU debug."),
        loader=_registry_loader(_load_utkface_age, n_samples=200, img_size=32),
    ),
    "boneage": DatasetInfo(
        name="boneage",
        category="image",
        n_features=-1,
        n_samples=1000,
        description=(
            "Bone-age maturity (months, 0–228). Real RSNA/Kaggle data via "
            "BONEAGE_ROOT env (csv with id+boneage[+male] + hand X-rays); "
            "synthetic proxy otherwise (flagged). Imbalanced/high-stakes P0 axis."
        ),
        loader=_registry_loader(_load_boneage),
    ),
    "boneage_tiny": DatasetInfo(
        name="boneage_tiny",
        category="image",
        n_features=-1,
        n_samples=200,
        description=("Bone-age tiny smoke variant: 200 samples at 32×32 for fast CPU debug."),
        loader=_registry_loader(_load_boneage, n_samples=200, img_size=32),
    ),
    "bbbc_cellcount": DatasetInfo(
        name="bbbc_cellcount",
        category="image",
        n_features=-1,
        n_samples=2000,
        description=(
            "BBBC005 fluorescence-microscopy cell counts (1–100). Auto-"
            "downloads the Broad zip (~1.8 GB) or reuses BBBC_ROOT; "
            "synthetic Poisson-blob proxy otherwise (flagged). Counts are "
            "never standardized — canonical count-loss testbed."
        ),
        loader=_registry_loader(_load_bbbc_cellcount),
    ),
    "bbbc_cellcount_tiny": DatasetInfo(
        name="bbbc_cellcount_tiny",
        category="image",
        n_features=-1,
        n_samples=200,
        description=("BBBC005 tiny smoke variant: 200 samples at 64×64 for fast CPU debug."),
        loader=_registry_loader(_load_bbbc_cellcount, n_samples=200, img_size=64),
    ),
    "bbbc_censored": DatasetInfo(
        name="bbbc_censored",
        category="image",
        n_features=-1,
        n_samples=1000,
        description=(
            "BBBC005 counts with detection-limit right-censoring "
            "(censoring_train/test codes). Real pixels via the BBBC005 "
            "source; synthetic censored proxy otherwise (flagged)."
        ),
        loader=_registry_loader(_load_bbbc_censored),
    ),
    "bbbc_censored_tiny": DatasetInfo(
        name="bbbc_censored_tiny",
        category="image",
        n_features=-1,
        n_samples=200,
        description=("BBBC-censored tiny smoke variant: 200 samples at 32×32."),
        loader=_registry_loader(_load_bbbc_censored, n_samples=200, img_size=32),
    ),
    "tcga_survival": DatasetInfo(
        name="tcga_survival",
        category="image",
        n_features=-1,
        n_samples=200,
        description=(
            "TCGA diagnostic-slide thumbnails + overall survival "
            "(right-censored) via the open GDC API + tile server "
            "(BRCA/LUAD); synthetic Weibull-lesion proxy otherwise "
            "(flagged). Set TCGA_OFFLINE=1 to force the proxy."
        ),
        loader=_registry_loader(_load_tcga_survival),
    ),
    "tcga_survival_tiny": DatasetInfo(
        name="tcga_survival_tiny",
        category="image",
        n_features=-1,
        n_samples=40,
        description=("TCGA tiny smoke variant: 40 samples at 64×64 (proxy-first)."),
        loader=_registry_loader(_load_tcga_survival, n_samples=40, img_size=64),
    ),
    "appa_age": DatasetInfo(
        name="appa_age",
        category="image",
        n_features=-1,
        n_samples=4000,
        description=(
            "APPA-REAL apparent age (mean + observed rater std, 7.5k faces). "
            "Auto-downloads the ChaLearn zip (~845 MB, no login) or reuses "
            "APPA_ROOT; synthetic heteroscedastic proxy otherwise (flagged)."
        ),
        loader=_registry_loader(_load_appa_age),
    ),
    "appa_age_tiny": DatasetInfo(
        name="appa_age_tiny",
        category="image",
        n_features=-1,
        n_samples=200,
        description=("APPA-REAL tiny smoke variant: 200 samples at 32×32 for fast CPU debug."),
        loader=_registry_loader(_load_appa_age, n_samples=200, img_size=32),
    ),
    "shanghai_crowd": DatasetInfo(
        name="shanghai_crowd",
        category="image",
        n_features=-1,
        n_samples=400,
        description=(
            "ShanghaiTech Part-B crowd density maps + head counts. Auto-"
            "downloads the dataset zip (~1 GB Dropbox mirror) or reuses "
            "SHANGHAI_ROOT; synthetic crowd proxy otherwise (flagged)."
        ),
        loader=_registry_loader(_load_shanghai_crowd),
    ),
    "shanghai_crowd_tiny": DatasetInfo(
        name="shanghai_crowd_tiny",
        category="image",
        n_features=-1,
        n_samples=60,
        description=("ShanghaiTech tiny smoke variant: 60 samples at 64×64 for fast CPU debug."),
        loader=_registry_loader(_load_shanghai_crowd, n_samples=60, img_size=64),
    ),
    "ixi_brainage": DatasetInfo(
        name="ixi_brainage",
        category="image",
        n_features=-1,
        n_samples=400,
        description=(
            "IXI T1 brain age (20–82y, mid-axial slices). Stage IXI-T1.tar + "
            "IXI.xls under IXI_ROOT (no auto-download: server blocks bots); "
            "synthetic atrophy-brain proxy otherwise (flagged). Long-tail "
            "ages — canonical imbalanced image testbed."
        ),
        loader=_registry_loader(_load_ixi_brainage),
    ),
    "ixi_brainage_tiny": DatasetInfo(
        name="ixi_brainage_tiny",
        category="image",
        n_features=-1,
        n_samples=60,
        description=("IXI tiny smoke variant: 60 samples at 32×32 for fast CPU debug."),
        loader=_registry_loader(_load_ixi_brainage, n_samples=60, img_size=32),
    ),
    "udacity_steering": DatasetInfo(
        name="udacity_steering",
        category="image",
        n_features=-1,
        n_samples=2000,
        description=(
            "Udacity dashboard-cam steering angle (~[-1, 1] rad). Place "
            "extracted frames + steering.csv (or data.txt) under "
            "UDACITY_ROOT (bags need external ROS extraction); synthetic "
            "curved-road proxy otherwise (flagged). Straight-driving bias."
        ),
        loader=_registry_loader(_load_udacity_steering),
    ),
    "udacity_steering_tiny": DatasetInfo(
        name="udacity_steering_tiny",
        category="image",
        n_features=-1,
        n_samples=200,
        description=("Steering tiny smoke variant: 200 samples at 32×32 for fast CPU debug."),
        loader=_registry_loader(_load_udacity_steering, n_samples=200, img_size=32),
    ),
    "kitti_depth": DatasetInfo(
        name="kitti_depth",
        category="image",
        n_features=-1,
        n_samples=1000,
        description=(
            "KITTI outdoor driving depth (sparse LiDAR, 0–80m, masks). Stage "
            "torch-uncertainty KITTIDepth layout under KITTI_ROOT (no "
            "auto-download: raw RGB is ~175 GB); synthetic sparse-street "
            "proxy otherwise (flagged, mask path exercised)."
        ),
        loader=_registry_loader(_load_kitti_depth),
    ),
    "kitti_depth_tiny": DatasetInfo(
        name="kitti_depth_tiny",
        category="image",
        n_features=-1,
        n_samples=60,
        description=("KITTI tiny smoke variant: 60 samples at 64×64 for fast CPU debug."),
        loader=_registry_loader(_load_kitti_depth, n_samples=60, img_size=64),
    ),
    "nyu_depth_v2_tiny": DatasetInfo(
        name="nyu_depth_v2_tiny",
        category="image",
        n_features=-1,
        n_samples=50,
        description=(
            "NYU Depth V2 tiny smoke variant: 50 samples at 64×64 for fast "
            "CPU debug. Same loader as nyu_depth_v2."
        ),
        loader=_registry_loader(_load_nyu_depth_v2, n_samples=50, img_size=64),
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
