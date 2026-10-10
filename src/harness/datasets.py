"""Dataset registry for the harness.

Provides a unified API for loading known regression datasets — both real
tabular benchmarks and synthetic stress tests, plus image regression datasets.
Every entry returns ``(X_train, X_test, y_train, y_test)`` as numpy arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict, cast

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
    # ``sigma_x`` is the scalar input-noise level for the classical EIV
    # loaders, but the photo-z loaders return a per-band vector.
    sigma_x: float | np.ndarray
    sigma_x_train: np.ndarray
    sigma_x_test: np.ndarray
    sigma_image_train: np.ndarray
    sigma_image_test: np.ndarray
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
    reference: list[np.ndarray]
    prior_dist: Any
    sbc_theta: np.ndarray
    sbc_x: np.ndarray
    # SBI tasks carry a vector-valued target.
    n_targets: int
    prior_low: float
    prior_high: float
    # Causal benchmarks (IHDP / semi-synthetic): treatment arm plus known
    # ground truth (potential outcomes and the true CATE/ATE).
    treatment_train: np.ndarray
    treatment_test: np.ndarray
    cate_true_train: np.ndarray
    cate_true_test: np.ndarray
    pot0_test: np.ndarray
    pot1_test: np.ndarray
    ate_true: float


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

    data = fetch_california_housing(data_home=str(_image_cache_dir() / "sklearn"))
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

        data = fetch_openml(
            data_id=data_id or 44031,
            parser="pandas",
            as_frame=True,
            data_home=str(_image_cache_dir() / "sklearn"),
        )
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


SBIBM_CACHE_VERSION = 1


def _sbibm_cache_path(
    task_name: str, n_samples: int, n_observations: int, n_sbc_simulations: int, seed: int
) -> Path:
    """Frozen copy of one sbibm task draw (training sims, observations, references)."""
    name = (
        f"{task_name}_n{n_samples}_o{n_observations}_sbc{n_sbc_simulations}_seed{seed}"
        f"_v{SBIBM_CACHE_VERSION}.pt"
    )
    return _image_cache_dir() / "sbibm" / name


def _sbibm_splits_from_arrays(arrays: dict[str, Any]) -> SplitsDict:
    ref = [np.asarray(r, dtype=np.float32) for r in arrays["reference"]]
    y = np.asarray(arrays["theta"], dtype=np.float32)
    out: SplitsDict = {
        "X_train": np.asarray(arrays["x"], dtype=np.float32),
        "y_train": y,
        "X_test": np.asarray(arrays["observations"], dtype=np.float32),
        # Point target = reference posterior mean (diagnostic only).
        "y_test": np.stack([r.mean(axis=0) for r in ref], axis=0).astype(np.float32),
        "reference": ref,
        "sbc_theta": np.asarray(arrays["sbc_theta"], dtype=np.float32),
        "sbc_x": np.asarray(arrays["sbc_x"], dtype=np.float32),
        "n_targets": y.shape[1],
        "prior_low": float(y.min()),
        "prior_high": float(y.max()),
        "_skip_y_standardization": True,  # simulator + reference in native units
    }
    if arrays.get("prior_dist") is not None:
        out["prior_dist"] = arrays["prior_dist"]
    return out


def _load_sbibm_task(
    task_name: str,
    n_samples: int = 10000,
    n_observations: int = 3,
    n_sbc_simulations: int = 100,
    seed: int = 42,
) -> SplitsDict:
    """Shared sbibm task loader (Lueckmann et al., NeurIPS 2021 SBI bench).

    Complex posteriors p(theta | x) with REFERENCE posterior samples for
    C2ST comparison. Train = prior simulations; test = benchmark
    observations. ``sbc_theta``/``sbc_x`` are a separate held-out set of
    prior/simulator draws used only for posterior-calibration diagnostics.
    Seeded via torch (+numpy, unused by most simulators).

    sbibm has had no release since 2023 and pins ``sbi<0.22`` (it is installed
    ``--no-deps``).  The first successful draw is therefore frozen under
    ``<cache>/sbibm/`` (see :func:`_sbibm_cache_path`) and later runs read the
    frozen copy without importing sbibm, so a broken sbibm install cannot
    change or break the suite.  ``tools/freeze_sbibm.py`` builds the copies
    ahead of an offline run.
    """
    import torch

    cache = _sbibm_cache_path(task_name, n_samples, n_observations, n_sbc_simulations, seed)
    if cache.exists():
        try:
            arrays = torch.load(cache, weights_only=False)
            splits = _sbibm_splits_from_arrays(arrays)
            splits["_source_url"] = f"frozen sbibm {task_name} ({cache.name})"
            return splits
        except Exception:
            pass  # unreadable copy: rebuild from sbibm below

    try:
        import sbibm  # ty: ignore[unresolved-import]  # optional SBI-benchmark dep

        torch.manual_seed(seed)
        task = sbibm.get_task(task_name)
        prior = task.get_prior()
        simulator = task.get_simulator()
        theta = torch.as_tensor(np.asarray(prior(num_samples=n_samples)), dtype=torch.float32)
        x = torch.as_tensor(np.asarray(simulator(theta)), dtype=torch.float32)
        obs, ref = [], []
        for k in range(1, n_observations + 1):
            o = np.asarray(task.get_observation(num_observation=k), dtype=np.float32)
            r = np.asarray(
                task.get_reference_posterior_samples(num_observation=k),
                dtype=np.float32,
            )
            obs.append(o.reshape(-1))
            ref.append(r)
        # Diagnostic simulations are independently seeded and never included in
        # the amortized training pairs above.
        torch.manual_seed(seed + 1)
        sbc_theta_t = torch.as_tensor(
            np.asarray(prior(num_samples=n_sbc_simulations)), dtype=torch.float32
        )
        sbc_x_t = torch.as_tensor(np.asarray(simulator(sbc_theta_t)), dtype=torch.float32)
        arrays: dict[str, Any] = {
            "task": task_name,
            "theta": theta.numpy().astype(np.float32),
            "x": x.numpy().astype(np.float32),
            "observations": np.stack(obs, axis=0).astype(np.float32),
            "reference": ref,
            "sbc_theta": sbc_theta_t.numpy().astype(np.float32),
            "sbc_x": sbc_x_t.numpy().astype(np.float32),
            "prior_dist": task.get_prior_dist(),
        }
    except Exception:
        synth = _load_synthetic_cond_nongaussian(n_samples=n_samples, seed=seed)
        synth["_synthetic_fallback"] = True
        synth["_fallback_reason"] = f"sbibm {task_name} unavailable and no frozen copy"
        return synth

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        torch.save(arrays, tmp)
        tmp.replace(cache)
    except Exception:
        pass  # read-only cache: still return the fresh draw
    splits = _sbibm_splits_from_arrays(arrays)
    splits["_source_url"] = f"sbibm {task_name}"
    return splits


def _load_ihdp(
    n_samples: int = 747,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """IHDP causal benchmark (Hill 2011).

    Infant Health and Development Program: 25 covariates with treatment
    indicator, factual/counterfactual outcomes and noiseless response
    surfaces (mu0/mu1). True CATE = mu1 - mu0. THE classic causal-ML
    benchmark (747 rows). Sources: local harness cache (e.g. Kaggle
    mirror), else the canonical CEVAE GitHub mirror (DoWhy-endorsed,
    no auth — session-compatible).
    """
    splits: SplitMetadata = {}
    df = None
    try:
        import pandas as pd

        path = _image_cache_dir() / "ihdp" / "ihdp_data.csv"
        if path.exists():
            df = pd.read_csv(path)
    except Exception:
        df = None
    if df is None:
        try:
            import urllib.request

            import pandas as pd

            url = (
                "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/"
                "master/datasets/IHDP/csv/ihdp_npci_1.csv"
            )
            cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [
                f"x{i}" for i in range(1, 26)
            ]
            dest = _image_cache_dir() / "ihdp" / "ihdp_npci_1.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)
            df = pd.read_csv(dest, header=None)
            df.columns = cols  # type: ignore[assignment]
        except Exception:
            raise RuntimeError("IHDP data unavailable (cache + CEVAE mirror failed)")
    assert df is not None
    try:
        x_cols = [c for c in df.columns if str(c).startswith("x")]
        X = df[x_cols].to_numpy(dtype=np.float64)
        t_raw = df["treatment"].to_numpy()
        if t_raw.dtype == bool:
            t = t_raw.astype(np.float64)
        else:
            t = np.isin(t_raw.astype(str), ["True", "true", "1", "1.0"]).astype(np.float64)
        y = df["y_factual"].to_numpy(dtype=np.float64)
        cate_true = (df["mu1"] - df["mu0"]).to_numpy(dtype=np.float64)
        mu0 = df["mu0"].to_numpy(dtype=np.float64)
        mu1 = df["mu1"].to_numpy(dtype=np.float64)
    except Exception:
        raise RuntimeError("IHDP columns unreadable")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_te = max(32, int(len(X) * test_size))
    te, tr = idx[:n_te], idx[n_te:]
    return {
        **splits,
        "X_train": X[tr].astype(np.float32),
        "X_test": X[te].astype(np.float32),
        "y_train": y[tr].astype(np.float32),
        "y_test": y[te].astype(np.float32),
        "treatment_train": t[tr].astype(np.float32),
        "treatment_test": t[te].astype(np.float32),
        "cate_true_train": cate_true[tr].astype(np.float64),
        "cate_true_test": cate_true[te].astype(np.float64),
        "pot0_test": mu0[te].astype(np.float64),
        "pot1_test": mu1[te].astype(np.float64),
        "ate_true": float(cate_true.mean()),
    }


def _load_folktables_causal(
    n_samples: int = 5000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Semi-synthetic causal benchmark on Folktables ACS features.

    Real Census covariates (X) + simulated treatment/outcome with KNOWN
    truth (ACIC-style): propensity logistic in (age, education, hours),
    CATE linear + age-interaction, nonlinear baseline + noise. Scales to
    any n (IHDP has 747 rows). Falls back to California Housing covariates
    if the ACS fetch fails (same simulation).
    """
    splits: SplitMetadata = {}
    try:
        X_full, _ = _fetch_folktables_xy(seed)
    except Exception:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing(data_home=str(_image_cache_dir() / "sklearn"))
        X_full = data.data.astype(np.float64)
        splits["_synthetic_fallback"] = True
        splits["_fallback_reason"] = "Folktables ACS fetch failed"
    rng = np.random.default_rng(seed)
    if n_samples < len(X_full):
        X_full = X_full[rng.choice(len(X_full), n_samples, replace=False)]
    Xs = (X_full - X_full.mean(axis=0)) / (X_full.std(axis=0) + 1e-12)
    d = Xs.shape[1]
    b_prop = rng.normal(0, 0.5, d)
    b_prop[0] += 0.8  # age loads propensity (confounding by construction)
    e = 1.0 / (1.0 + np.exp(-(Xs @ b_prop - 0.5)))
    e = np.clip(e, 0.05, 0.95)
    t = (rng.random(len(Xs)) < e).astype(np.float64)
    tau = 1.0 + 0.5 * Xs[:, 0] + 0.3 * Xs[:, min(1, d - 1)] * Xs[:, min(7, d - 1)]
    mu = 2.0 * np.sin(Xs[:, 0]) + Xs[:, min(2, d - 1)] ** 2 + 0.5 * Xs.sum(axis=1) / d
    y = mu + tau * t + rng.normal(0, 1.0, len(Xs))
    cate_true = tau
    idx = rng.permutation(len(Xs))
    n_te = max(32, int(len(Xs) * test_size))
    te, tr = idx[:n_te], idx[n_te:]
    return {
        **splits,
        "X_train": X_full[tr].astype(np.float32),
        "X_test": X_full[te].astype(np.float32),
        "y_train": y[tr].astype(np.float32),
        "y_test": y[te].astype(np.float32),
        "treatment_train": t[tr].astype(np.float32),
        "treatment_test": t[te].astype(np.float32),
        "cate_true_train": cate_true[tr].astype(np.float64),
        "cate_true_test": cate_true[te].astype(np.float64),
        "pot0_test": mu[te].astype(np.float64),
        "pot1_test": (mu + tau)[te].astype(np.float64),
        "ate_true": float(cate_true.mean()),
    }


def _load_sbibm_twomoons(
    n_samples: int = 10000,
    n_observations: int = 3,
    n_sbc_simulations: int = 100,
    seed: int = 42,
) -> SplitsDict:
    """sbibm Two Moons (bimodal 2-D posterior + reference samples)."""
    return _load_sbibm_task("two_moons", n_samples, n_observations, n_sbc_simulations, seed)


def _load_sbibm_slcp(
    n_samples: int = 10000,
    n_observations: int = 3,
    n_sbc_simulations: int = 100,
    seed: int = 42,
) -> SplitsDict:
    """sbibm SLCP (5-D params, 8-D obs, complex posterior + references)."""
    return _load_sbibm_task("slcp", n_samples, n_observations, n_sbc_simulations, seed)


def _load_sbibm_gaussian_linear(
    n_samples: int = 10000,
    n_observations: int = 3,
    n_sbc_simulations: int = 100,
    seed: int = 42,
) -> SplitsDict:
    """sbibm Gaussian Linear (10-D analytic Gaussian posterior)."""
    return _load_sbibm_task("gaussian_linear", n_samples, n_observations, n_sbc_simulations, seed)


def _load_sbibm_gaussian_mixture(
    n_samples: int = 10000,
    n_observations: int = 3,
    n_sbc_simulations: int = 100,
    seed: int = 42,
) -> SplitsDict:
    """sbibm Gaussian Mixture (2-D bimodal posterior)."""
    return _load_sbibm_task("gaussian_mixture", n_samples, n_observations, n_sbc_simulations, seed)


def _load_sbibm_bernoulli_glm(
    n_samples: int = 10000,
    n_observations: int = 3,
    n_sbc_simulations: int = 100,
    seed: int = 42,
) -> SplitsDict:
    """sbibm Bernoulli GLM with sufficient-statistic observations."""
    return _load_sbibm_task("bernoulli_glm", n_samples, n_observations, n_sbc_simulations, seed)


def _load_galaxiesml_eiv(
    n_samples: int = 20000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """GalaxiesML EIV (Do et al. 2024, HSC PDR2, Zenodo).

    REAL heteroscedastic EIV: grizy cmodel magnitudes with MEASURED
    per-object per-band uncertainties ({band}_cmodel_magsigma) and
    spectroscopic redshifts. The honest flagship: heteroscedastic,
    per-band, measured — not assumed. Reads tabular columns from the 64px
    test HDF5 (chunked, images untouched). Falls back to photoz_eiv if
    the HDF5 is unavailable.
    """
    splits: SplitMetadata = {}
    try:
        import urllib.request

        import h5py  # ty: ignore[unresolved-import]  # optional big-corpus dep
        from sklearn.preprocessing import StandardScaler

        bands = ["g", "r", "i", "z", "y"]
        path = _image_cache_dir() / "galaxiesml" / "test.hdf5"
        if not (path.exists() and path.stat().st_size > 3_000_000_000):
            path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                "https://zenodo.org/api/records/11117528/files/"
                "5x64x64_testing_with_morphology.hdf5/content"
            )
            urllib.request.urlretrieve(url, path)
        with h5py.File(path, "r") as f:
            mags = np.stack([np.asarray(f[f"{b}_cmodel_mag"][:]) for b in bands], axis=1).astype(
                np.float64
            )
            sigs = np.stack(
                [np.asarray(f[f"{b}_cmodel_magsigma"][:]) for b in bands], axis=1
            ).astype(np.float64)
            z = np.asarray(f["specz_redshift"][:]).astype(np.float64).reshape(-1)
        keep = (
            np.isfinite(mags).all(axis=1)
            & np.isfinite(sigs).all(axis=1)
            & np.isfinite(z)
            & (sigs > 0).all(axis=1)
            & (sigs < 1.0).all(axis=1)
            & (z > 0)
            & (z < 6)
        )
        mags, sigs, z = mags[keep], sigs[keep], z[keep]
        if n_samples < len(mags):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(mags), n_samples, replace=False)
            mags, sigs, z = mags[idx], sigs[idx], z[idx]
        # Per-band sigma in standardized units (band stds from train split
        # below — computed post-split to avoid leakage).
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            mags, z, sigs, test_size=test_size, random_state=seed
        )
        band_std = X_train.std(axis=0)
        band_std[band_std < 1e-12] = 1.0
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Per-OBSERVATION sigma [N, 5] in standardized units (heteroscedastic;
        # rows stay aligned with X_train/X_test). Per-band mean kept as the
        # fallback for construction-time-only methods (SIMEX/RC/LatentNN).
        sigma_tr = (s_train / band_std).astype(np.float32)
        sigma_te = (s_test / band_std).astype(np.float32)
        sigma_vec = sigma_tr.mean(axis=0).astype(np.float32)
    except Exception:
        fb = _load_photoz_eiv(n_samples=n_samples, test_size=test_size, seed=seed)
        fb["_synthetic_fallback"] = True
        fb["_fallback_reason"] = "GalaxiesML HDF5 unavailable"
        return fb

    return {
        **splits,
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "sigma_x": sigma_vec,
        "sigma_x_train": sigma_tr,
        "sigma_x_test": sigma_te,
    }


def _load_photoz_eiv(
    n_samples: int = 20000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Photo-z errors-in-variables (pzflow galaxies + photometric errors).

    REAL EIV: ugrizy magnitudes are noisy measurements with KNOWN nominal
    photometric error (sigma_x=0.1 mag, the pzflow-tutorial convention);
    target is spectroscopic redshift. No clean features exist — the honest
    real-data EIV setting. Falls back to synthetic EIV if pzflow is
    unavailable.
    """
    splits: SplitMetadata = {}
    try:
        from pzflow.examples import get_galaxy_data  # ty: ignore[unresolved-import]
        from sklearn.preprocessing import StandardScaler

        data = get_galaxy_data()
        bands = ["u", "g", "r", "i", "z", "y"]
        X = data[bands].to_numpy(dtype=np.float64)
        y = data["redshift"].to_numpy(dtype=np.float64)
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[keep], y[keep]
        if n_samples < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n_samples, replace=False)
            X, y = X[idx], y[idx]
    except Exception:
        synth = _load_synthetic_eiv(n_samples=n_samples, seed=seed)
        synth["_synthetic_fallback"] = True
        synth["_fallback_reason"] = "pzflow galaxy data unavailable"
        return synth

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    # Nominal 0.1-mag photometric error expressed in standardized units
    # (per-band stds are near-identical for magnitudes; median is exact).
    band_std = float(np.median(X_train.std(axis=0)))
    sigma_std = 0.1 / max(band_std, 1e-12)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return {
        **splits,
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "sigma_x": float(sigma_std),
    }


def _load_galaxiesml_cutouts(
    n_samples: int = 4000,
    test_size: float = 0.3,
    seed: int = 42,
    img_size: int = 64,
) -> SplitsDict:
    """GalaxiesML image cutouts for photo-z (Do et al. 2024, HSC PDR2).

    5-band 64×64 coadds + spectroscopic redshifts from the Zenodo HDF5s
    (training/testing splits as published). Returns (N,5,H,W) float32
    images standardized per-channel from train stats; y = spec-z.
    Falls back to synthetic image regression if the HDF5s are unavailable.
    """
    splits: SplitMetadata = {}
    try:
        import h5py  # ty: ignore[unresolved-import]  # optional big-corpus dep

        cache = _image_cache_dir() / "galaxiesml"
        tr_path = cache / "training.hdf5"
        te_path = cache / "test.hdf5"
        if not (tr_path.exists() and tr_path.stat().st_size > 1_000_000_000):
            raise RuntimeError("training HDF5 incomplete")
        if not (te_path.exists() and te_path.stat().st_size > 1_000_000_000):
            raise RuntimeError("test HDF5 incomplete")
        rng = np.random.default_rng(seed)
        with h5py.File(tr_path, "r") as f:
            n_tr = f["image"].shape[0]
            n_tr_take = min(n_tr, max(64, int(n_samples * 0.7)))
            idx_tr = np.sort(rng.choice(n_tr, n_tr_take, replace=False))
            X_train = np.asarray(f["image"][idx_tr]).astype(np.float32)
            y_train = np.asarray(f["specz_redshift"][idx_tr]).astype(np.float64)
        with h5py.File(te_path, "r") as f:
            n_te = f["image"].shape[0]
            n_te_take = min(n_te, max(32, n_samples - n_tr_take))
            idx_te = np.sort(rng.choice(n_te, n_te_take, replace=False))
            X_test = np.asarray(f["image"][idx_te]).astype(np.float32)
            y_test = np.asarray(f["specz_redshift"][idx_te]).astype(np.float64)
        keep_tr = np.isfinite(y_train) & (y_train > 0) & (y_train < 6)
        keep_te = np.isfinite(y_test) & (y_test > 0) & (y_test < 6)
        X_train, y_train = X_train[keep_tr], y_train[keep_tr]
        X_test, y_test = X_test[keep_te], y_test[keep_te]

        def _border_sigma(imgs: np.ndarray) -> np.ndarray:
            # Sky-noise floor per image: robust std (MAD) of the outer
            # 8px frame across bands → median over bands. MEASURED from
            # the cutouts (empty-sky pixels), not assumed.
            edge = np.r_[0:8, -8:0]
            rows = imgs[:, :, edge[:, None], :].reshape(imgs.shape[0], 5, -1)
            cols = imgs[:, :, :, edge].reshape(imgs.shape[0], 5, -1)
            f = np.concatenate([rows, cols], axis=2)  # (N, 5, P)
            mad = np.median(np.abs(f - np.median(f, axis=2, keepdims=True)), axis=2)  # (N, 5)
            return (1.4826 * np.median(mad, axis=1)).astype(np.float64)  # (N,)

        sig_tr_raw = _border_sigma(X_train)
        sig_te_raw = _border_sigma(X_test)
        # Per-channel standardization from train (images are raw coadds).
        mu = X_train.mean(axis=(0, 2, 3), keepdims=True)
        sd = X_train.std(axis=(0, 2, 3), keepdims=True)
        sd[sd < 1e-12] = 1.0
        X_train = ((X_train - mu) / sd).astype(np.float32)
        X_test = ((X_test - mu) / sd).astype(np.float32)
        # Noise in standardized units (median channel scale; bands similar).
        ch_scale = float(np.median(sd))
        sig_tr = (sig_tr_raw / ch_scale).astype(np.float32)
        sig_te = (sig_te_raw / ch_scale).astype(np.float32)
    except Exception:
        synth = _load_synthetic_image_regression(n_samples=n_samples, seed=seed)
        synth["_synthetic_fallback"] = True
        synth["_fallback_reason"] = "GalaxiesML HDF5s unavailable"
        return synth

    return {
        **splits,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "sigma_image_train": sig_tr,
        "sigma_image_test": sig_te,
    }


def _load_pzflow_galaxies(
    n_samples: int = 20000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Photo-z galaxy photometry (pzflow bundled LSST-DESC example sample).

    100k galaxies with ugrizy magnitudes + spectroscopic redshift. Features
    are the standard photo-z set (5 adjacent colors + i-band magnitude);
    target is redshift (multimodal posteriors — the flexible-PDF showcase).
    Ships inside the pzflow package (no download/auth). Falls back to the
    synthetic non-Gaussian conditional if pzflow is unavailable.
    """
    splits: SplitMetadata = {}
    try:
        from pzflow.examples import get_galaxy_data  # ty: ignore[unresolved-import]
        from sklearn.preprocessing import StandardScaler

        data = get_galaxy_data()
        bands = ["u", "g", "r", "i", "z", "y"]
        mags = data[bands].to_numpy(dtype=np.float64)
        colors = np.stack([mags[:, i] - mags[:, i + 1] for i in range(5)], axis=1)
        X = np.hstack([colors, mags[:, 3:4]])  # 5 colors + i mag
        y = data["redshift"].to_numpy(dtype=np.float64)
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[keep], y[keep]
        if n_samples < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n_samples, replace=False)
            X, y = X[idx], y[idx]
    except Exception:
        synth = _load_synthetic_cond_nongaussian(n_samples=n_samples, seed=seed)
        synth["_synthetic_fallback"] = True
        synth["_fallback_reason"] = "pzflow galaxy data unavailable"
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
        from SurvSet.data import SurvLoader  # ty: ignore[unresolved-import]

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

        df = fetch_openml(
            data_id=41214,
            parser="auto",
            as_frame=True,
            data_home=str(_image_cache_dir() / "sklearn"),
        )
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


_US_STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "PR",
]


def _folktables_root(states: list[str] | tuple[str, ...]) -> str:
    """ACS root holding every requested state's 2018 person file, else the cache.

    The 50-state pull (``folktables_eiv`` and friends) is several GB from
    census.gov and took hours per CANFAR session, so it is staged once under
    ``staged/folktables`` (``scripts/skaha_stage_data.sh``); folktables skips
    the download when ``<root>/2018/1-Year/psam_p<fips>.csv`` already exists.
    """
    from folktables.load_acs import _STATE_CODES  # ty: ignore[unresolved-import]

    roots, _ = _staged_roots(os.environ.get("FOLKTABLES_ROOT", ""), "folktables")
    for cand in roots:
        base = cand / "2018" / "1-Year"
        if all((base / f"psam_p{_STATE_CODES[st]}.csv").is_file() for st in states):
            return str(cand)
    return str(_image_cache_dir() / "folktables")


def _fetch_folktables_xy(
    seed: int = 42,
    *,
    published_acs_income_filter: bool = False,
    states: list[str] | tuple[str, ...] = ("CA",),
) -> tuple[np.ndarray, np.ndarray]:
    """Shared ACS 2018 fetch + encoding → (X features, continuous PINCP).

    ``states`` selects the PUMS states (default CA-only, preserving the
    existing benchmarks). Passing all US states yields ~3M person rows
    (~1.5M after the employment filter) — the >1M-row likelihood scale.
    """
    import folktables  # ty: ignore[unresolved-import]  # optional ACS dep
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder

    root = _folktables_root(states)
    ds = folktables.ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person", root_dir=root
    )
    acs = ds.get_data(states=list(states), download=True)
    df = acs.copy()
    filt = (df["AGEP"] > 16) & (df["WKHP"] > 0) & (df["PINCP"] >= 100)
    if published_acs_income_filter:
        # Folktables ACSIncome's adult_filter additionally excludes unpaid
        # family workers (COW == 5).
        filt &= df["COW"] != 5
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

        data = fetch_california_housing(data_home=str(_image_cache_dir() / "sklearn"))
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


def _load_folktables_acs_income_binary(
    n_samples: int = 20000,
    test_size: float = 0.3,
    seed: int = 42,
) -> SplitsDict:
    """Published Folktables ACSIncome task with the canonical $50k label.

    Features and population filtering match :func:`_load_folktables_income`.
    The binary outcome is ``PINCP > $50,000``, which is the target used by the
    Folktables ACSIncome benchmark.  A synthetic binary classification fallback
    keeps offline harness smoke runs executable, and is explicitly flagged.
    """
    splits: SplitMetadata = {}
    try:
        from sklearn.preprocessing import StandardScaler

        X, income = _fetch_folktables_xy(seed, published_acs_income_filter=True)
        y = (income > 50_000.0).astype(np.int64)
        if n_samples < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n_samples, replace=False)
            X, y = X[idx], y[idx]
    except Exception:
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler

        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=8,
            n_redundant=4,
            weights=[0.75, 0.25],
            class_sep=1.2,
            flip_y=0.02,
            random_state=seed,
        )
        splits["_synthetic_fallback"] = True
        splits["_fallback_reason"] = "Folktables ACS fetch failed"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
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
        "n_classes": 2,
        "_skip_y_standardization": True,
    }


def _load_folktables_all(
    n_samples: int | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """US Census income regression on ALL states (Folktables ACS 2018).

    Same cohort/encoding as :func:`_load_folktables_income` but across all
    50 states + PR: ~3M person rows (~1.5M after the employment filter) —
    the >1M-row likelihood scale. ``n_samples=None`` (default) keeps the
    full filtered set; pass an int for a seeded subsample (smoke/debug).
    Falls back to California Housing if the fetch fails.
    """
    splits: SplitMetadata = {}
    try:
        from sklearn.preprocessing import StandardScaler

        X, y = _fetch_folktables_xy(seed, states=_US_STATES)
        if n_samples is not None and n_samples < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n_samples, replace=False)
            X, y = X[idx], y[idx]
    except Exception:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing(data_home=str(_image_cache_dir() / "sklearn"))
        X, y = data.data.astype(np.float64), data.target.astype(np.float64)
        splits["_synthetic_fallback"] = True
        splits["_fallback_reason"] = "Folktables all-states fetch failed"

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


_TLC_MONTH = "2024-01"
_TLC_URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{_TLC_MONTH}.parquet"


def _load_tlc_fare(
    n_samples: int | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """NYC yellow-taxi fare regression (TLC trip records, pinned month).

    ~3M trips for 2024-01; ``fare_amount`` is heavily right-skewed —
    the second >1M-row likelihood domain next to Folktables income.
    Features: passenger_count, trip_distance, PU/DO location IDs
    (ordinal codes), payment_type, pickup hour + day-of-week, and
    trip_duration_seconds from the timestamps. Filters: fare in
    (0, 500], distance in (0, 100] miles, duration in [60s, 3h].
    Targets stay RAW (``_skip_y_standardization``) so likelihood rows
    see the true skew. Parquet is cached under the shared ``staged/tlc``
    mirror (scratch, or the ARC project allocation on CANFAR); an
    explicitly set but unusable ``TLC_ROOT`` raises instead of silently
    substituting a proxy.
    """
    import os as _os

    import pandas as _pd

    root = _os.environ.get("TLC_ROOT", "")
    staged, explicit = _staged_roots(root, "tlc")
    parquet: Path | None = None
    for cand in staged:
        if not cand.is_dir():
            continue
        hits = sorted(cand.glob("yellow_tripdata_*.parquet"))
        if hits:
            # Prefer the pinned month, else the newest available file.
            pinned = [h for h in hits if _TLC_MONTH in h.name]
            parquet = pinned[0] if pinned else hits[-1]
            break
    if parquet is None:
        target_dir = staged[0]
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            dest = target_dir / f"yellow_tripdata_{_TLC_MONTH}.parquet"
            if not (dest.exists() and dest.stat().st_size > 0):
                _download_with_progress(_TLC_URL, dest, desc="NYC TLC yellow")
            parquet = dest
        except Exception:
            parquet = None
    if parquet is None or not parquet.is_file():
        if explicit:
            raise RuntimeError(
                f"TLC_ROOT={root!r} yielded no trip parquet; refusing synthetic fallback."
            )
        return _synthetic_tlc_proxy(n_samples or 20000, test_size, seed)

    df = _pd.read_parquet(parquet)
    needed = {
        "passenger_count",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "fare_amount",
    }
    if not needed.issubset(set(df.columns)):
        if explicit:
            raise RuntimeError(f"TLC parquet {parquet} lacks expected TLC columns.")
        return _synthetic_tlc_proxy(n_samples or 20000, test_size, seed)
    df = df.dropna(subset=list(needed)).copy()
    df["pickup_ts"] = _pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["dropoff_ts"] = _pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_ts", "dropoff_ts"])
    df["duration_s"] = (df["dropoff_ts"] - df["pickup_ts"]).dt.total_seconds()
    df["hour"] = df["pickup_ts"].dt.hour.astype(np.float64)
    df["dow"] = df["pickup_ts"].dt.dayofweek.astype(np.float64)
    filt = (
        (df["fare_amount"] > 0)
        & (df["fare_amount"] <= 500)
        & (df["trip_distance"] > 0)
        & (df["trip_distance"] <= 100)
        & (df["duration_s"] >= 60)
        & (df["duration_s"] <= 10800)
        & (df["passenger_count"] >= 1)
    )
    df = df.loc[filt].reset_index(drop=True)
    if len(df) < 1000:
        if explicit:
            raise RuntimeError(f"TLC parquet {parquet} yielded too few trips after filters.")
        return _synthetic_tlc_proxy(n_samples or 20000, test_size, seed)
    y = df["fare_amount"].to_numpy(dtype=np.float64)
    X = np.column_stack(
        [
            df["passenger_count"].to_numpy(dtype=np.float64),
            df["trip_distance"].to_numpy(dtype=np.float64),
            df["PULocationID"].to_numpy(dtype=np.float64),
            df["DOLocationID"].to_numpy(dtype=np.float64),
            df["payment_type"].to_numpy(dtype=np.float64),
            df["hour"].to_numpy(dtype=np.float64),
            df["dow"].to_numpy(dtype=np.float64),
            df["duration_s"].to_numpy(dtype=np.float64),
        ]
    )
    if n_samples is not None and n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "_skip_y_standardization": True,
    }


def _synthetic_tlc_proxy(n_samples: int, test_size: float, seed: int) -> SplitsDict:
    """Seeded skewed-fare proxy (no I/O) — explicitly flagged fallback."""
    rng = np.random.default_rng(seed)
    n = n_samples
    dist = rng.uniform(0.5, 20, n)
    dur = dist * rng.uniform(120, 300, n) + rng.normal(0, 60, n)
    fare = 2.5 + 2.5 * dist + 0.5 * dur / 60 + np.abs(rng.normal(0, 2, n))
    X = np.column_stack(
        [
            rng.integers(1, 5, n).astype(np.float64),
            dist,
            rng.integers(1, 264, n).astype(np.float64),
            rng.integers(1, 264, n).astype(np.float64),
            rng.integers(1, 3, n).astype(np.float64),
            rng.integers(0, 24, n).astype(np.float64),
            rng.integers(0, 7, n).astype(np.float64),
            np.clip(dur, 60, 10800),
        ]
    )
    y = fare.astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    splits: SplitsDict = {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "_skip_y_standardization": True,
        "_synthetic_fallback": True,
    }
    splits["_fallback_reason"] = "TLC parquet unavailable; synthetic fare proxy"
    return splits


def _load_folktables_eiv(
    n_samples: int | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    noise_frac: float = 0.15,
    hetero_strength: float = 0.5,
) -> SplitsDict:
    """Folktables all-states with CALIBRATED heteroscedastic input noise.

    Real ACS covariates + continuous income truth, but feature measurements
    are corrupted by seed-fixed Gaussian noise whose per-row scale is a
    documented fraction of each feature's standardized scale, modulated by
    ``(1 + hetero_strength * |x|)`` row-wise — i.e. noisier measurements
    for extreme rows. This is synthetic noise on real data (the standard
    EIV benchmark construction, same as ``synthetic_eiv``), explicitly
    labeled as injected: it tests EIV methods at >1M rows where no
    measured-noise corpus exists. Returns noisy ``X_*``, clean
    ``X_*_true``, scalar ``sigma_x`` fallback, and per-row
    ``sigma_x_test`` in standardized units (matching the standardized
    ``X``), so per-observation EIV paths are exercised at scale.
    """
    from sklearn.preprocessing import StandardScaler

    X, y = _fetch_folktables_xy(seed, states=_US_STATES)
    if n_samples is not None and n_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rng = np.random.default_rng(seed + 7)
    row_mod = 1.0 + hetero_strength * np.abs(Xs).mean(axis=1, keepdims=True)
    sigma = (noise_frac * row_mod).astype(np.float64)
    sigma_full = np.repeat(sigma, Xs.shape[1], axis=1)
    X_obs = (Xs + sigma_full * rng.standard_normal(Xs.shape)).astype(np.float32)
    X_true = Xs.astype(np.float32)
    # Single joint split: noisy X, clean X, targets, and per-row sigma
    # stay row-aligned by construction (never reconstruct indices by hand).
    Xtr_o, Xte_o, Xtr_t, Xte_t, ytr, yte, sig_tr, sig_te = train_test_split(
        X_obs, X_true, y, sigma_full, test_size=test_size, random_state=seed
    )
    return {
        "X_train": Xtr_o,
        "X_test": Xte_o,
        "y_train": ytr.astype(np.float32),
        "y_test": yte.astype(np.float32),
        "X_train_true": Xtr_t,
        "X_test_true": Xte_t,
        "sigma_x": float(noise_frac),
        "sigma_x_test": sig_te.astype(np.float32),
        "sigma_y": 0.0,
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


# OpenML copies of the MULAN multi-target regression collection (Spyromitros-
# Xioufis et al., 2016). Targets are
# the dataset's declared target attributes; continuous only (wq/oes dropped:
# integer abundances / not hosted; OpenML's SARCOS keeps one torque only).
_OPENML_MULTITARGET: dict[str, tuple[int, tuple[str, ...]]] = {
    "mt_rf1": (
        41483,
        (
            "CHSI2_48H__0",
            "NASI2_48H__0",
            "EADM7_48H__0",
            "SCLM7_48H__0",
            "CLKM7_48H__0",
            "VALI2_48H__0",
            "NAPM7_48H__0",
            "DLDI4_48H__0",
        ),
    ),
    "mt_scm20d": (
        41486,
        ("LBL",) + tuple(f"MTLp{k}A" for k in range(2, 17)),
    ),
    "mt_enb": (41478, ("Y1", "Y2")),
    "mt_jura": (41479, ("Cd", "Co", "Cu")),
}


def _load_openml_multitarget(
    name: str,
    n_samples: int | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """Real multi-target regression set from OpenML (no synthetic fallback).

    Features are standardised on the training split; targets stay in native
    units with shape ``(N, D)`` (``_skip_y_standardization``) so density suites
    can standardise per target themselves. Rows with missing values are
    dropped; ``n_samples`` subsamples (seeded) before the split.
    """
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler

    data_id, targets = _OPENML_MULTITARGET[name]
    frame = fetch_openml(
        data_id=data_id,
        parser="pandas",
        as_frame=True,
        data_home=str(_image_cache_dir() / "sklearn"),
    ).frame
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    y = frame[list(targets)].to_numpy(dtype=np.float64)
    X = frame.drop(columns=list(targets)).to_numpy(dtype=np.float64)
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    if n_samples is not None and len(X) > n_samples:
        idx = np.random.default_rng(seed).choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler().fit(X_train)
    return {
        "X_train": scaler.transform(X_train).astype(np.float32),
        "X_test": scaler.transform(X_test).astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "n_targets": y.shape[1],
        "_multi_output": True,
        "_skip_y_standardization": True,
        "_source_url": f"https://www.openml.org/d/{data_id}",
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

        data = fetch_openml(
            name="house_sales",
            version=3,
            parser="auto",
            as_frame=True,
            data_home=str(_image_cache_dir() / "sklearn"),
        )
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
    """Return the harness dataset cache directory.

    Policy: datasets live on /scratch (never home). Override with
    TORCHREGRESS_HARNESS_CACHE. Datasets >100GB go to
    /arc/projects/ots/torchregress (set the env var accordingly).
    """
    cache = Path(
        os.environ.get(
            "TORCHREGRESS_HARNESS_CACHE",
            "/scratch/torchregress-harness-cache",
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
        import requests  # optional dep, imported lazily with a friendly ImportError
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

    _socket.getaddrinfo = _patched  # ty: ignore[invalid-assignment]

    def _restore() -> None:
        _socket.getaddrinfo = orig

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


def _tcga_cache_name(projects: tuple[str, ...], n_samples: int, img_size: int) -> str:
    return f"tcga_{'-'.join(sorted(projects))}_n{n_samples}_s{img_size}.npz"


def _tcga_split(
    X: np.ndarray, T: np.ndarray, E: np.ndarray, img_size: int, test_size: float, seed: int
) -> SplitsDict:
    C = (1.0 - E).astype(np.float32)  # harness: 1 = censored
    Xtr, Xte, Ytr, Yte, Ctr, Cte = train_test_split(X, T, C, test_size=test_size, random_state=seed)
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
        "_n_real_slides": len(X),
    }


def _load_tcga_survival_real(
    n_samples: int,
    img_size: int,
    projects: tuple[str, ...],
    test_size: float,
    seed: int,
) -> SplitsDict:
    """GDC real path for :func:`_load_tcga_survival` (may raise).

    The harvest (one GDC tile request per ~1 s, up to 25 tiles per slide)
    takes 35-90 min for 200 slides, so it is cached once as
    ``staged/tcga/<name>.npz`` (seed-independent; the split is per seed).
    """
    import io as _io
    import math as _math
    import os as _os
    import time as _time

    cache_name = _tcga_cache_name(projects, n_samples, img_size)
    staged, _ = _staged_roots(_os.environ.get("TCGA_ROOT", ""), "tcga")
    for cand in staged:
        f = cand / cache_name
        if f.is_file():
            z = np.load(f)
            return _tcga_split(z["X"], z["T"], z["E"], img_size, test_size, seed)

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
        for cand in staged:  # first writable mirror wins; caching is best-effort
            try:
                cand.mkdir(parents=True, exist_ok=True)
                tmp = cand / (cache_name + ".tmp.npz")
                np.savez_compressed(tmp, X=X, T=T, E=E)
                tmp.replace(cand / cache_name)
                break
            except OSError:
                continue
        return _tcga_split(X, T, E, img_size, test_size, seed)
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


def _qnrf_has_pairs(candidate: Path) -> bool:
    """True when a directory holds UCF-QNRF Train/Test jpg + _ann.mat pairs."""
    for split in ("Train", "Test"):
        d = candidate / split
        if not d.is_dir():
            continue
        for jp in sorted(d.glob("*.jpg")):
            if (d / f"{jp.stem}_ann.mat").is_file():
                return True
    return False


def _load_ucf_qnrf(
    n_samples: int = 400,
    img_size: int = 128,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitsDict:
    """UCF-QNRF crowd counting (Idrees et al., ECCV 2018, via Kaggle).

    1535 high-resolution images (1201 train / 334 test) with 1.25M head
    annotations — the SOTA dense-crowd benchmark. Per-image ``*_ann.mat``
    files carry ``annPoints`` (N,2); density maps via the shared Gaussian
    renderer (sigma=4). Images resized to ``img_size`` with rescaled points.
    Cached under the harness cache (Kaggle download). Falls back to the
    synthetic crowd proxy if unavailable.
    """
    import os as _os

    cache_dir = _image_cache_dir()
    root = _os.environ.get("QNRF_ROOT", "")

    base = Path(root) if root else cache_dir / "qnrf" / "UCF-QNRF_ECCV18"
    if not _qnrf_has_pairs(base):
        # Shared staged mirrors (scratch + the CANFAR-visible ARC project
        # allocation): the Kaggle layout nests UCF-QNRF_ECCV18/ one level
        # down, so probe both depths.
        staged, _ = _staged_roots(root, "qnrf")
        for cand in staged:
            for depth in (cand, cand / "UCF-QNRF_ECCV18"):
                if _qnrf_has_pairs(depth):
                    base = depth
                    break
            if _qnrf_has_pairs(base):
                break
    if not _qnrf_has_pairs(base):
        # Try a Kaggle fetch when credentials are available (sessions).
        ku, kk = _os.environ.get("KAGGLE_USERNAME", ""), _os.environ.get("KAGGLE_KEY", "")
        if ku and kk:
            try:
                import subprocess as _sp

                dest = cache_dir / "qnrf"
                dest.mkdir(parents=True, exist_ok=True)
                env = dict(
                    _os.environ,
                    KAGGLE_USERNAME=ku,
                    KAGGLE_KEY=kk,
                    KAGGLE_CONFIG_DIR=str(dest / ".kaggle"),
                )
                (dest / ".kaggle").mkdir(parents=True, exist_ok=True)
                with open(dest / ".kaggle" / "kaggle.json", "w") as fh:
                    fh.write(f'{{"username": "{ku}", "key": "{kk}"}}')
                _sp.run(
                    [
                        "kaggle",
                        "datasets",
                        "download",
                        "-d",
                        "faihajalamtopu/ucf-qnrf",
                        "-p",
                        str(dest),
                        "--unzip",
                    ],
                    env=env,
                    timeout=3600,
                    check=True,
                    capture_output=True,
                )
            except Exception:
                pass
    pairs: list[tuple[Path, Path]] | None = None
    if base.is_dir():
        for split in ("Train", "Test"):
            d = base / split
            if not d.is_dir():
                continue
            for jp in sorted(d.glob("*.jpg")):
                mp = d / f"{jp.stem}_ann.mat"
                if mp.is_file():
                    (pairs := pairs or []).append((jp, mp))
    if pairs is not None:
        try:
            import scipy.io as _sio
            from PIL import Image as _Image

            imgs_all, counts_all, dens_all = [], [], []
            rng_pick = np.random.default_rng(seed)
            order = rng_pick.permutation(len(pairs))[:n_samples]
            for idx in order:
                jp, mp = pairs[int(idx)]
                mat = _sio.loadmat(str(mp))
                pts = np.asarray(mat["annPoints"], dtype=np.float32)
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

    # Synthetic fallback: pure seeded generator (must NOT route through
    # the Shanghai loader — its real-data path would silently substitute a
    # different real corpus while the artifact claims a synthetic proxy).
    synth = _synthetic_crowd_splits(n_samples, img_size, test_size, seed)
    synth["_fallback_reason"] = "UCF-QNRF data unavailable; synthetic crowd proxy"
    return synth


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

    # Synthetic fallback: Poisson crowd counts + blob fields. Purely
    # seeded (no network, no other dataset) so fallback artifacts can never
    # masquerade as a different real corpus.
    splits = _synthetic_crowd_splits(n_samples, img_size, test_size, seed)
    splits["_fallback_reason"] = "ShanghaiTech unavailable; synthetic crowd proxy"
    return splits


def _synthetic_crowd_splits(
    n_samples: int,
    img_size: int,
    test_size: float,
    seed: int,
) -> SplitsDict:
    """Seeded synthetic crowd fields (no I/O): Poisson head counts + blobs."""
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
    return splits


def _staged_roots(explicit: str, dirname: str) -> tuple[list[Path], bool]:
    """Candidate roots for a manually staged corpus, in priority order.

    Returns ``(roots, explicit_set)``: ``roots[0]`` is the explicit env-var
    root when set, followed by the shared ``staged/<dirname>`` mirrors on
    scratch and on the ARC project allocation (CANFAR-visible). Callers try
    each root in order; when ``explicit_set`` is true and no root yields
    usable pairs, the loader must raise instead of silently substituting a
    synthetic proxy.
    """
    roots: list[Path] = []
    if explicit:
        roots.append(Path(explicit))
    cache = Path(
        os.environ.get(
            "TORCHREGRESS_HARNESS_CACHE",
            "/scratch/torchregress-harness-cache",
        )
    )
    roots.append(cache / "staged" / dirname)
    # Override the CANFAR-visible project mirror (tests, alternate mounts).
    arc_base = Path(
        os.environ.get(
            "TORCHREGRESS_HARNESS_STAGED_ROOT",
            "/arc/projects/ots/torchregress-harness/datasets/staged",
        )
    )
    roots.append(arc_base / dirname)
    return roots, bool(explicit)


def _read_ixi_staged_subjects(base: Path) -> list[tuple[Path, float]]:
    """(volume path, age) pairs from a staged IXI mirror (``subjects.csv``).

    Expected layout (e.g. the Kaggle preprocessed-IXI mirror): a
    ``subjects.csv`` with ``age`` and ``registered_mni`` columns, the latter
    holding paths to MNI-registered T1 volumes (``.nii`` or ``.nii.gz``)
    relative to the mirror root. Returns [] when the layout is absent.
    """
    try:
        import pandas as _pd
    except ImportError:
        return []
    csv_paths = [
        base / "subjects.csv",
        base / "T1w_Processed_IXI_with_csv" / "IXI" / "subjects.csv",
    ]
    nested_ixi = base / "T1w_Processed_IXI_with_csv" / "IXI"
    pairs: list[tuple[Path, float]] = []
    seen: set[str] = set()
    for csv_path in csv_paths:
        if not csv_path.is_file():
            continue
        try:
            df = _pd.read_csv(csv_path)
        except Exception:
            continue
        cols = {str(c).lower(): c for c in df.columns}
        age_col = next((cols[k] for k in cols if k == "age"), None)
        path_col = next(
            (cols[k] for k in cols if "mni" in k or "path" in k or "file" in k or "volume" in k),
            None,
        )
        if age_col is None or path_col is None:
            continue
        anchor = csv_path.parent  # staged mirrors nest volumes under the CSV dir
        for _, row in df.iterrows():
            try:
                age = float(row[age_col])
            except (ValueError, TypeError):
                continue
            # The staged mirror marks unknown ages with -1.0 — drop them.
            if not np.isfinite(age) or not 0.0 < age < 120.0:
                continue
            rel = str(row[path_col])
            cand: Path | None = None
            spellings = [rel]
            # Staged mirrors may store uncompressed .nii where the manifest
            # lists .nii.gz (and vice versa) — probe both spellings.
            if rel.endswith(".nii.gz"):
                spellings.append(rel[: -len(".gz")])
            elif rel.endswith(".nii"):
                spellings.append(rel + ".gz")
            for root in (anchor, nested_ixi, base):
                for spelling in spellings:
                    probe = root / spelling
                    if probe.is_file():
                        cand = probe
                        break
                if cand is not None:
                    break
            if cand is not None and str(cand) not in seen:
                seen.add(str(cand))
                pairs.append((cand, age))
    return pairs


def _ixi_volumes_to_splits(
    pairs: list[tuple[Path, float]],
    n_samples: int,
    img_size: int,
    test_size: float,
    seed: int,
) -> SplitsDict | None:
    """Mid-axial slices of staged T1 volumes → image splits (None if empty)."""
    if not pairs:
        return None
    try:
        import nibabel as _nib  # ty: ignore[unresolved-import]  # optional dep for staged IXI
    except ImportError as exc:
        raise ImportError("nibabel is required for IXI brain age (pip install nibabel).") from exc
    from PIL import Image as _Image

    rng_pick = np.random.default_rng(seed)
    order = rng_pick.permutation(len(pairs))[:n_samples]
    imgs, y = [], []
    for idx in order:
        fp, age = pairs[int(idx)]
        try:
            vol = np.asarray(_nib.load(str(fp)).get_fdata(), dtype=np.float32)
        except Exception:
            continue
        if vol.ndim != 3 or min(vol.shape) < 8:
            continue
        z = vol.shape[2] // 2  # mid-axial slice
        sl = vol[:, :, z]
        lo, hi = float(np.percentile(sl, 1)), float(np.percentile(sl, 99))
        sl = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
        img = _Image.fromarray((sl * 255).astype(np.uint8)).resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        imgs.append(arr[None, :, :])
        y.append(float(age))
        if len(imgs) >= n_samples:
            break
    if len(imgs) < 8:
        return None
    X = np.stack(imgs).astype(np.float32)
    yy = np.array(y, dtype=np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=test_size, random_state=seed)
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

    Staged-mirror path: ``subjects.csv`` + MNI-registered volumes under the
    shared ``staged/ixi`` cache mirror (scratch, or the ARC project
    allocation on CANFAR). An explicitly set but unusable ``IXI_ROOT``
    raises instead of silently falling back to the synthetic proxy.

    Long-tail ages (few 70+ subjects) make this the canonical imbalanced
    image testbed (LDS/DensityWeighted/BalancedMSE).

    Fallback: synthetic atrophy brains — ventricle size grows with age
    (mechanistic ground truth: y = age, ventricle radius ∝ age).
    """
    import os as _os

    root = _os.environ.get("IXI_ROOT", "")
    staged, explicit = _staged_roots(root, "ixi")
    for base in staged:
        splits = _ixi_volumes_to_splits(
            _read_ixi_staged_subjects(base), n_samples, img_size, test_size, seed
        )
        if splits is not None:
            return splits
    if explicit:
        raise RuntimeError(
            f"IXI_ROOT={root!r} yielded no usable T1 volumes; refusing synthetic fallback."
        )
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


def _resolve_udacity_frame(base: Path, name: str) -> Path | None:
    """Resolve a logged frame name against known Udacity frame directories.

    Simulator logs store Windows paths (``C:\\...\\IMG\\center_x.jpg``);
    split on both separators so the leaf resolves on POSIX too.
    """
    leaf = str(name).replace("\\", "/").split("/")[-1]
    for cand in (base / "IMG" / leaf, base / "center" / leaf, base / name):
        if cand.is_file():
            return cand
    return None


def _udacity_base_pairs(base: Path) -> list[tuple[Path, float]]:
    """(frame, steering) pairs from one Udacity dataset directory."""
    import csv as _csv

    pairs: list[tuple[Path, float]] = []
    # Layout A: center/*.jpg + steering csv (headered).
    img_dirs = [d for d in (base / "IMG", base / "center") if d.is_dir()]
    csvs = sorted(base.glob("*.csv"))
    for csv_path in csvs:
        try:
            with open(csv_path, newline="") as fh:
                rows = [row for row in _csv.reader(fh) if row]
        except Exception:
            continue
        if not rows:
            continue
        header = [str(c).lower() for c in rows[0]]
        if any("steer" in c or "angle" in c for c in header):
            # Headered log: match steering + image columns by name.
            fn = {str(c).lower(): c for c in rows[0]}
            f_ang = next((fn[k] for k in fn if "steer" in k or "angle" in k), None)
            f_img = next(
                (fn[k] for k in fn if "frame" in k or "file" in k or "image" in k or "name" in k),
                None,
            )
            if f_ang is None or f_img is None:
                continue
            idx_ang = rows[0].index(f_ang)
            idx_img = rows[0].index(f_img)
            for row in rows[1:]:
                if max(idx_ang, idx_img) >= len(row):
                    continue
                try:
                    ang = float(row[idx_ang])
                except (ValueError, TypeError):
                    continue
                jp = _resolve_udacity_frame(base, str(row[idx_img]))
                if jp is not None:
                    pairs.append((jp, ang))
        else:
            # Layout C: headerless driving_log.csv —
            # [center, left, right, steering, throttle, brake, speed].
            # Detect the steering column as the first column that parses
            # as a small float on nearly every row.
            n_cols = max(len(row) for row in rows)
            idx_ang: int | None = None
            for j in range(1, n_cols):
                vals = []
                for row in rows:
                    if j >= len(row):
                        break
                    try:
                        vals.append(float(row[j]))
                    except (ValueError, TypeError):
                        break
                else:
                    if len(vals) == len(rows) and np.median(np.abs(vals)) < 2.0:
                        idx_ang = j
                        break
                continue
            if idx_ang is None:
                continue
            for row in rows:
                if idx_ang >= len(row):
                    continue
                try:
                    ang = float(row[idx_ang])
                except (ValueError, TypeError):
                    continue
                jp = _resolve_udacity_frame(base, str(row[0]))
                if jp is not None:
                    pairs.append((jp, ang))
        if pairs:
            return pairs
    if pairs or not img_dirs:
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
    return pairs


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
    matched against frame/filename/image), headerless ``driving_log.csv``
    (``IMG/`` + ``[center, left, right, steering, ...]`` simulator layout),
    or SullyChen-style ``data.txt`` lines of ``<relpath> <angle>``. Images
    are RGB-resized to ``img_size``. Staged mirrors under the shared
    ``staged/udacity`` cache (scratch, or the ARC project allocation on
    CANFAR) are preferred; a mirror may hold several dataset dirs, all of
    which are scanned. An explicitly set but unusable ``UDACITY_ROOT``
    raises instead of silently falling back to the synthetic proxy.

    Straight-driving dominates (angle ≈ 0 long tail) — canonical imbalanced
    scalar testbed alongside brain age.

    Fallback: synthetic perspective roads whose lane curvature sets the
    steering angle (mechanistic ground truth + matching imbalance).
    """
    import os as _os

    root = _os.environ.get("UDACITY_ROOT", "")
    staged, explicit = _staged_roots(root, "udacity")
    bases: list[Path] = []
    for cand in staged:
        if not cand.is_dir():
            continue
        bases.append(cand)
        try:
            bases.extend([d for d in sorted(cand.iterdir()) if d.is_dir()])
        except OSError:
            continue
    if bases:
        try:
            from PIL import Image as _Image

            pairs: list[tuple[Path, float]] = []
            for base in bases:
                try:
                    pairs.extend(_udacity_base_pairs(base))
                except Exception:
                    continue
            if not pairs and explicit:
                raise RuntimeError(
                    f"UDACITY_ROOT={root!r} yielded no (frame, steering) pairs; "
                    "refusing synthetic fallback."
                )
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
        except RuntimeError:
            raise  # explicit-root failure stays loud (no synthetic fallback)
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


def _resolve_kitti_depth(base: Path, rel: str) -> Path | None:
    """Resolve a listed Eigen ground-truth depth path to an existing PNG.

    Manifests reference ``.../proj_depth/groundtruth/image_02/data/<f>.jpg``;
    mirrors store ``.../proj_depth/groundtruth/image_02/<stem>.png`` (uint16,
    metres = px/256). Both spellings are probed.
    """
    rel_path = Path(rel)
    stem = rel_path.stem
    cands = [
        base / rel_path.parent / (stem + ".png"),
        base / rel_path.parent.parent / rel_path.parent.name / (stem + ".png"),
        base / rel_path.parent.parent / (stem + ".png"),
        base / str(rel_path.with_suffix(".png")),
    ]
    for cand in cands:
        if cand.is_file():
            return cand
    return None


def _read_kitti_eigen_pairs(
    base: Path,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """(train, test) RGB/depth pairs from Eigen-split location manifests.

    Reads ``gt_locations.txt`` (train pool) and ``test_locations.txt`` (test
    pool): whitespace-separated ``<rgb> <depth>`` paths relative to the
    mirror root. Returns ([], []) when the manifests are absent.
    """
    pools: list[list[tuple[Path, Path]]] = []
    for name in ("gt_locations.txt", "test_locations.txt"):
        tp = base / name
        pairs: list[tuple[Path, Path]] = []
        if tp.is_file():
            try:
                lines = tp.read_text().splitlines()
            except OSError:
                lines = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    rgb = base / parts[0]
                    if not rgb.is_file():
                        continue
                    depth = _resolve_kitti_depth(base, parts[1])
                    if depth is None:
                        continue
                    pairs.append((rgb, depth))
                elif len(parts) == 1 and "proj_depth" in parts[0]:
                    # gt manifest: single-column projected-depth refs;
                    # the RGB frame mirrors the same drive/stem path.
                    depth = _resolve_kitti_depth(base, parts[0])
                    if depth is None:
                        continue
                    rgb_rel = parts[0].replace(
                        "proj_depth/groundtruth/image_02/data/", "image_02/data/"
                    )
                    rgb = base / rgb_rel
                    if not rgb.is_file():
                        continue
                    pairs.append((rgb, depth))
        pools.append(pairs)
    return pools[0], pools[1]


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
    (depth uint16, metres = px/256) — or an Eigen-split mirror with
    ``gt_locations.txt``/``test_locations.txt`` manifests (RGB ``*.jpg`` +
    projected-depth ``*.png``). RGB resized to ``img_size``; depth
    resized nearest; invalid pixels (≤0 or >max_depth) are 0 with
    ``mask_*`` float maps (1 = valid). ``_sparse_depth`` flags the
    masked-dense contract for suites (masked losses, valid-only metrics).
    Staged mirrors under the shared ``staged/kitti`` cache (scratch, or the
    ARC project allocation on CANFAR) are preferred. An explicitly set but
    unusable ``KITTI_ROOT`` raises instead of silently falling back.

    No auto-download: RGB requires per-drive raw zips (~175 GB total).
    Stage once via ``torch_uncertainty.datasets.KITTIDepth(root,
    download=True, remove_unused=True)`` or mirror the layout manually.

    Fallback: synthetic street scenes with LiDAR-style sparse sampling
    (valid-pixel mask included, so the masked path is exercised in CI).
    """
    import os as _os

    root = _os.environ.get("KITTI_ROOT", "")
    staged, explicit = _staged_roots(root, "kitti")
    eigen_train: list[tuple[Path, Path]] = []
    eigen_test: list[tuple[Path, Path]] = []
    for cand in staged:
        if not cand.is_dir():
            continue
        try:
            tr, te = _read_kitti_eigen_pairs(cand)
        except Exception:
            continue
        eigen_train.extend(tr)
        eigen_test.extend(te)
    if eigen_train or eigen_test:
        from PIL import Image as _Image

        rng_pick = np.random.default_rng(seed)
        n_test = max(1, int(round(n_samples * test_size)))
        n_train = max(1, n_samples - n_test)

        def _take(
            pool: list[tuple[Path, Path]], k: int
        ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
            Xs, Ys, Ms = [], [], []
            if not pool:
                return Xs, Ys, Ms
            order = rng_pick.permutation(len(pool))
            for idx in order:
                if len(Xs) >= k:
                    break
                jp, dp = pool[int(idx)]
                try:
                    img = _Image.open(jp).convert("RGB").resize((img_size, img_size))
                    dep = _Image.open(dp).resize((img_size, img_size), _Image.Resampling.NEAREST)
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
            return Xs, Ys, Ms

        Xtr_l, Ytr_l, Mtr_l = _take(eigen_train or eigen_test, n_train)
        Xte_l, Yte_l, Mte_l = _take(eigen_test or eigen_train, n_test)
        if Xtr_l and Xte_l:
            return {
                "X_train": np.stack(Xtr_l).astype(np.float32),
                "X_test": np.stack(Xte_l).astype(np.float32),
                "y_train": np.stack(Ytr_l).astype(np.float32),
                "y_test": np.stack(Yte_l).astype(np.float32),
                "mask_train": np.stack(Mtr_l).astype(np.float32),
                "mask_test": np.stack(Mte_l).astype(np.float32),
                "_image_data": True,
                "_image_size": img_size,
                "_n_channels": 3,
                "_dense_output": True,
                "_sparse_depth": True,
                "_skip_y_standardization": True,
            }
        if explicit:
            raise RuntimeError(
                f"KITTI_ROOT={root!r} yielded no usable Eigen RGB/depth pairs; "
                "refusing synthetic fallback."
            )
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

    if explicit:
        raise RuntimeError(
            f"KITTI_ROOT={root!r} yielded no usable RGB/depth pairs; refusing synthetic fallback."
        )

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
        return cast("SplitsDict", load_stress_dataset(name, n=n))

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
    "sbibm_slcp": DatasetInfo(
        name="sbibm_slcp",
        category="synthetic",
        n_features=8,
        n_samples=10000,
        description=(
            "sbibm SLCP (NeurIPS 2021 SBI benchmark): 5-D params, 8-D obs, "
            "complex posterior with reference samples for C2ST."
        ),
        loader=_registry_loader(_load_sbibm_slcp, n_samples=10000),
    ),
    "sbibm_gaussian_linear": DatasetInfo(
        name="sbibm_gaussian_linear",
        category="synthetic",
        n_features=10,
        n_samples=10000,
        description=(
            "sbibm Gaussian Linear (NeurIPS 2021 SBI benchmark): 10-D analytic "
            "Gaussian posterior with reference samples and held-out SBC simulations."
        ),
        loader=_registry_loader(_load_sbibm_gaussian_linear, n_samples=10000),
    ),
    "sbibm_gaussian_mixture": DatasetInfo(
        name="sbibm_gaussian_mixture",
        category="synthetic",
        n_features=2,
        n_samples=10000,
        description=(
            "sbibm Gaussian Mixture (NeurIPS 2021 SBI benchmark): 2-D multimodal "
            "posterior with reference samples and held-out SBC simulations."
        ),
        loader=_registry_loader(_load_sbibm_gaussian_mixture, n_samples=10000),
    ),
    "sbibm_bernoulli_glm": DatasetInfo(
        name="sbibm_bernoulli_glm",
        category="synthetic",
        n_features=10,
        n_samples=10000,
        description=(
            "sbibm Bernoulli GLM (NeurIPS 2021 SBI benchmark): 10-D sufficient "
            "statistics with reference samples and held-out SBC simulations."
        ),
        loader=_registry_loader(_load_sbibm_bernoulli_glm, n_samples=10000),
    ),
    "ihdp": DatasetInfo(
        name="ihdp",
        category="tabular",
        n_features=25,
        n_samples=747,
        description=(
            "IHDP causal benchmark (Hill 2011, CEVAE mirror): 25 covariates "
            "with treatment, factual/counterfactual outcomes, true CATE."
        ),
        loader=_registry_loader(_load_ihdp),
    ),
    "folktables_causal": DatasetInfo(
        name="folktables_causal",
        category="tabular",
        n_features=70,
        n_samples=150000,
        description=(
            "Semi-synthetic causal benchmark on Folktables ACS features: "
            "simulated propensity/treatment/outcome with KNOWN ATE/CATE."
        ),
        loader=_registry_loader(_load_folktables_causal, n_samples=5000),
    ),
    "sbibm_twomoons": DatasetInfo(
        name="sbibm_twomoons",
        category="synthetic",
        n_features=2,
        n_samples=10000,
        description=(
            "sbibm Two Moons (NeurIPS 2021 SBI benchmark): bimodal 2-D "
            "posterior with reference samples for C2ST."
        ),
        loader=_registry_loader(_load_sbibm_twomoons, n_samples=10000),
    ),
    "galaxiesml_eiv": DatasetInfo(
        name="galaxiesml_eiv",
        category="tabular",
        n_features=5,
        n_samples=40000,
        description=(
            "GalaxiesML HSC EIV (Do et al. 2024, Zenodo): grizy mags with "
            "MEASURED per-band heteroscedastic errors + spec-z. Flagship."
        ),
        loader=_registry_loader(_load_galaxiesml_eiv, n_samples=20000),
    ),
    "photoz_eiv": DatasetInfo(
        name="photoz_eiv",
        category="tabular",
        n_features=6,
        n_samples=100000,
        description=(
            "Photo-z errors-in-variables (pzflow galaxies): noisy ugrizy "
            "magnitudes with KNOWN nominal photometric error + spec-z. "
            "Real EIV — no clean features exist."
        ),
        loader=_registry_loader(_load_photoz_eiv, n_samples=20000),
    ),
    "galaxiesml_cutouts": DatasetInfo(
        name="galaxiesml_cutouts",
        category="image",
        n_features=5,
        n_samples=286401,
        description=(
            "GalaxiesML HSC cutouts (Do et al. 2024, Zenodo): 5-band 64px "
            "coadds + spec-z. Real image photo-z benchmark."
        ),
        loader=_registry_loader(_load_galaxiesml_cutouts, n_samples=4000),
    ),
    "pzflow_galaxies": DatasetInfo(
        name="pzflow_galaxies",
        category="tabular",
        n_features=6,
        n_samples=100000,
        description=(
            "Photo-z galaxy photometry (pzflow bundled LSST-DESC sample): "
            "redshift from ugrizy (5 colors + i mag). Real flexible-PDF "
            "benchmark — redshift posteriors are multimodal."
        ),
        loader=_registry_loader(_load_pzflow_galaxies, n_samples=20000),
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
    "folktables_all": DatasetInfo(
        name="folktables_all",
        category="tabular",
        n_features=70,
        n_samples=1500000,
        description=(
            "US Census income regression on ALL states (Folktables ACS 2018): "
            "~3M person rows (~1.5M after the employment filter). "
            ">1M-row likelihood scale; full filtered set by default."
        ),
        loader=_registry_loader(_load_folktables_all),
    ),
    "tlc_fare": DatasetInfo(
        name="tlc_fare",
        category="tabular",
        n_features=8,
        n_samples=2500000,
        description=(
            "NYC yellow-taxi fare regression (TLC 2024-01, ~3M trips): "
            "right-skewed fare_amount with raw (unstandardized) targets. "
            ">1M-row likelihood scale; pinned month, validated schema."
        ),
        loader=_registry_loader(_load_tlc_fare),
    ),
    "folktables_eiv": DatasetInfo(
        name="folktables_eiv",
        category="tabular",
        n_features=70,
        n_samples=1500000,
        description=(
            "Folktables all-states with calibrated heteroscedastic input "
            "noise (seed-fixed injection on real covariates; noisy X, clean "
            "X_true, per-row sigma_x_test). EIV at >1M rows; explicitly "
            "labeled as injected noise (cf. measured GalaxiesML sigmas)."
        ),
        loader=_registry_loader(_load_folktables_eiv),
    ),
    "folktables_acs_income": DatasetInfo(
        name="folktables_acs_income",
        category="tabular",
        n_features=70,
        n_samples=150000,
        description=(
            "Folktables ACSIncome binary classification (ACS 2018 CA): "
            "personal income above $50k on the published adult worker cohort."
        ),
        loader=_registry_loader(_load_folktables_acs_income_binary, n_samples=20000),
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
    "mt_rf1": DatasetInfo(
        name="mt_rf1",
        category="tabular",
        n_features=64,
        n_samples=9125,
        description="MULAN rf1: river-flow sensor readings, 8 targets (48 h ahead). Real multi-target data (OpenML).",
        loader=_registry_loader(_load_openml_multitarget, name="mt_rf1"),
    ),
    "mt_scm20d": DatasetInfo(
        name="mt_scm20d",
        category="tabular",
        n_features=61,
        n_samples=8966,
        description="MULAN scm20d: supply-chain management, 16 price targets. Real multi-target data (OpenML).",
        loader=_registry_loader(_load_openml_multitarget, name="mt_scm20d"),
    ),
    "mt_enb": DatasetInfo(
        name="mt_enb",
        category="tabular",
        n_features=8,
        n_samples=768,
        description="MULAN enb (UCI Energy Efficiency): heating/cooling load, 2 targets. Real multi-target data (OpenML).",
        loader=_registry_loader(_load_openml_multitarget, name="mt_enb"),
    ),
    "mt_jura": DatasetInfo(
        name="mt_jura",
        category="tabular",
        n_features=15,
        n_samples=359,
        description="MULAN jura: Swiss Jura soil, Cd/Co/Cu concentrations, 3 targets. Real multi-target data (OpenML).",
        loader=_registry_loader(_load_openml_multitarget, name="mt_jura"),
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
    "ucf_qnrf": DatasetInfo(
        name="ucf_qnrf",
        category="image",
        n_features=-1,
        n_samples=1535,
        description=(
            "UCF-QNRF dense crowd (Idrees et al., ECCV 2018, via Kaggle): "
            "1535 high-res images, 1.25M annotations. SOTA crowd benchmark."
        ),
        loader=_registry_loader(_load_ucf_qnrf, n_samples=400),
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


STRICT_DATA_ENV = "TORCHREGRESS_HARNESS_STRICT_DATA"


class SyntheticFallbackError(RuntimeError):
    """A real dataset could not be loaded and strict data mode forbids a synthetic proxy."""


def strict_data_enabled() -> bool:
    """True when ``TORCHREGRESS_HARNESS_STRICT_DATA`` is set to a truthy value.

    Paper and release runs set it so a missing download or staged mirror stops
    the run instead of silently benchmarking a synthetic stand-in.
    """
    return os.environ.get(STRICT_DATA_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _enforce_strict_data(name: str, splits: SplitsDict) -> None:
    if not strict_data_enabled() or not splits.get("_synthetic_fallback"):
        return
    if DATASET_REGISTRY[name].category == "synthetic":
        return
    reason = splits.get("_fallback_reason", "loader returned a synthetic proxy")
    raise SyntheticFallbackError(
        f"{name}: real data unavailable ({reason}); {STRICT_DATA_ENV} forbids synthetic fallback. "
        "Stage the dataset or drop it from the run."
    )


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
    _enforce_strict_data(name, splits)

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
