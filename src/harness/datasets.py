"""Dataset registry for the harness.

Provides a unified API for loading known regression datasets — both real
tabular benchmarks and synthetic stress tests, plus image regression datasets.
Every entry returns ``(X_train, X_test, y_train, y_test)`` as numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    loader: Callable[..., dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# Real tabular datasets
# ---------------------------------------------------------------------------


def _load_california_housing(
    n_samples: int = 5000, test_size: float = 0.3, seed: int = 42
) -> dict[str, np.ndarray]:
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
    n_samples: int = 442, test_size: float = 0.3, seed: int = 42,
) -> dict[str, np.ndarray]:
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
) -> dict[str, np.ndarray]:
    """Load OpenML YearPredictionMSD dataset.

    Falls back to synthetic data if OpenML fetch fails.
    """
    try:
        from sklearn.datasets import fetch_openml

        data = fetch_openml(data_id=data_id or 44031, parser="pandas", as_frame=True)
        df = data.frame
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col]).values.astype(np.float64)
        y = df[target_col].values.astype(np.float64)
    except Exception:
        # Fallback: synthetic data with similar dimensions
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((515345, 90)).astype(np.float64)
        y = 0.7 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.3, 515345).astype(np.float64)

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


# ---------------------------------------------------------------------------
# Synthetic tabular datasets
# ---------------------------------------------------------------------------


def _load_synthetic_heteroscedastic(
    n_samples: int = 2000, n_features: int = 8, test_size: float = 0.3, seed: int = 42,
) -> dict[str, np.ndarray]:
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
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


def _load_synthetic_outliers(
    n_samples: int = 2000, n_features: int = 8,
    outlier_fraction: float = 0.15, test_size: float = 0.3, seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic data with outliers."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (
        2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] ** 2
        + 0.3 * rng.standard_normal(n_samples)
    ).astype(np.float32)

    n_outliers = int(n_samples * outlier_fraction)
    outlier_idx = rng.choice(n_samples, n_outliers, replace=False)
    y[outlier_idx] += rng.uniform(-5, 5, n_outliers).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


def _load_synthetic_tweedie(
    n_samples: int = 2000, n_features: int = 6,
    p_power: float = 1.5, phi: float = 0.6, test_size: float = 0.3, seed: int = 42,
) -> dict[str, np.ndarray]:
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
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


def _load_synthetic_censored(
    n_samples: int = 2000, n_features: int = 8,
    censoring_rate: float = 0.3, test_size: float = 0.3, seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic data with censored targets."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y_true = (
        1.0 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2]
        + 0.2 * rng.standard_normal(n_samples)
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
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "censoring_train": codes_train, "censoring_test": codes_test,
        "_skip_y_standardization": True,  # preserve censoring threshold semantics
    }


def _load_synthetic_eiv(
    n_samples: int = 2000, n_features: int = 6,
    sigma_x: float = 0.2, sigma_y: float = 0.1, test_size: float = 0.3, seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic data with errors-in-variables (noisy inputs)."""
    rng = np.random.default_rng(seed)
    X_true = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    X_obs = (X_true + sigma_x * rng.standard_normal((n_samples, n_features))).astype(np.float32)
    y = (
        0.8 * X_true[:, 0] - 0.6 * X_true[:, 1] + 0.4 * X_true[:, 2]
        + sigma_y * rng.standard_normal(n_samples)
    ).astype(np.float32)

    X_train_obs, X_test_obs, y_train, y_test = train_test_split(
        X_obs, y, test_size=test_size, random_state=seed
    )
    X_train_true, X_test_true, _, _ = train_test_split(
        X_true, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train_obs, "X_test": X_test_obs,
        "y_train": y_train, "y_test": y_test,
        "X_train_true": X_train_true, "X_test_true": X_test_true,
    }


# ---------------------------------------------------------------------------
# Upstream canonical datasets (from external library documentation)
# ---------------------------------------------------------------------------


def _load_insurance_claims(
    n_samples: int = 20000, test_size: float = 0.3, seed: int = 42,
) -> dict[str, np.ndarray]:
    """Load the French Motor Insurance Claims dataset (OpenML ID 41214).

    This is the canonical sklearn Tweedie regression dataset — used in
    ``sklearn.linear_model.TweedieRegressor`` documentation and examples.
    Falls back to synthetic Tweedie data if OpenML fetch fails.

    Canonical source:
    https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
    """
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
            c for c in feature_cols
            if data[c].dtype == object or data[c].dtype.name == "category"
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
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "exposure_train": exp_train.astype(np.float32),
        "exposure_test": exp_test.astype(np.float32),
        "_skip_y_standardization": True,  # Tweedie claim counts — preserve zero-inflation
    }


def _load_house_sales(
    n_samples: int = 20000, test_size: float = 0.3, seed: int = 42,
) -> dict[str, np.ndarray]:
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
                "house_sales target column not found — "
                "falling back to California Housing"
            )

        numeric_df = df.select_dtypes(include=[np.number])
        X = numeric_df.drop(columns=[target_col]).values.astype(np.float64)
        y = numeric_df[target_col].values.astype(np.float64)
    except Exception:
        # Fallback: California Housing
        return _load_california_housing(n_samples=n_samples, test_size=test_size, seed=seed)

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
    n_samples: int = 5000, n_features: int = 10,
    n_informative: int = 6, noise: float = 15.0,
    test_size: float = 0.3, seed: int = 42,
) -> dict[str, np.ndarray]:
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


def _load_synthetic_sinusoidal(
    n_samples: int = 2000, n_features: int = 4,
    test_size: float = 0.3, seed: int = 42,
) -> dict[str, np.ndarray]:
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
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
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
            "crepes quick-start: RandomForestRegressor with "
            "DifficultyEstimator on house_sales."
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
            "sklearn TweedieRegressor example: French Motor Insurance "
            "Claims with p=1.5 log-link."
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
) -> dict[str, np.ndarray]:
    """Generic TabReD loader: try local data, fall back to synthetic."""
    try:
        from pathlib import Path

        csv_path = Path(data_dir) / f"{dataset_id}.csv"
        if csv_path.exists():
            import pandas as pd

            df = pd.read_csv(csv_path)
            target_col = df.columns[-1]
            X = df.drop(columns=[target_col]).values.astype(np.float64)
            y = df[target_col].values.astype(np.float64)
        else:
            raise FileNotFoundError(f"{csv_path} not found — falling back to synthetic")
    except Exception:
        # Fallback: synthetic data with TabReD dimensions
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_rows, n_features)).astype(np.float64)
        # First 4 features are informative (if they exist); rest are noise
        feat_idx = [0, 1, 2, min(3, n_features - 1)] if n_features >= 4 else list(range(n_features))
        coefs = [0.6, -0.4, 0.3, 0.2][:len(feat_idx)]
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
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _registry_loader(
    fn: Callable[..., dict[str, np.ndarray]], **defaults: Any,
) -> Callable[..., dict[str, np.ndarray]]:
    """Wrap a loader so extra kwargs can override defaults."""
    def _loader(**kwargs: Any) -> dict[str, np.ndarray]:
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
    "synthetic_eiv": DatasetInfo(
        name="synthetic_eiv",
        category="synthetic",
        n_features=6,
        n_samples=2000,
        description="Synthetic errors-in-variables with noisy input measurements.",
        loader=_registry_loader(_load_synthetic_eiv),
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
            n_samples=50000,
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
            n_samples=50000,
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
            n_samples=50000,
        ),
    ),
}


def load_dataset(name: str, **kwargs: Any) -> dict[str, np.ndarray]:
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
        **kwargs: Override default loader parameters (e.g., ``n_samples=500``).

    Returns:
        Dictionary with ``X_train``, ``X_test``, ``y_train``, ``y_test``
        as numpy arrays (y arrays are standardised unless opt-out).
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset: {name!r}. Available: {sorted(DATASET_REGISTRY.keys())}"
        )
    splits = DATASET_REGISTRY[name].loader(**kwargs)

    # Standardise targets unless the dataset opts out (Tweedie, censored)
    if not splits.get("_skip_y_standardization"):
        y_scaler = StandardScaler()
        y_scaler.fit(splits["y_train"].reshape(-1, 1))
        splits["y_train"] = (
            y_scaler.transform(splits["y_train"].reshape(-1, 1))
            .ravel().astype(np.float32)
        )
        splits["y_test"] = (
            y_scaler.transform(splits["y_test"].reshape(-1, 1))
            .ravel().astype(np.float32)
        )
        splits["_y_scaler"] = y_scaler

    return splits


def list_datasets(category: str | None = None) -> list[DatasetInfo]:
    """List all registered datasets, optionally filtered by category."""
    datasets = list(DATASET_REGISTRY.values())
    if category is not None:
        datasets = [d for d in datasets if d.category == category]
    return sorted(datasets, key=lambda d: d.name)
