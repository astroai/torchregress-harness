"""Distribution-shift uncertainty suite: GaussianNLL / BetaNLL / MDN vs MSE.

Motivated by Gustafsson, Danelljan & Schön (TMLR 2023), "How Reliable is Your
Regression Model's Uncertainty Under Real-World Distribution Shifts?"
(arXiv:2302.03679).  That benchmark evaluates regression uncertainty under
distribution shifts and reports that methods which look calibrated
in-distribution become strongly overconfident under many shifts.  This suite
is a compact tabular analogue: one shared MLP backbone, four methods with the
same capacity and training budget, three deterministic evaluation views per
dataset, and the harness's standard proper-score / interval / PIT metrics.

Protocol
--------
For every supported dataset the suite builds three deterministic views from a
fixed seed (all shifts are defined here; ``harness/datasets.py`` is untouched):

``id``
    In-distribution test set.  For ``stress_ood`` it is a 20% carve of
    ``X_train`` (that dataset's native test split is the shifted set).  For
    the other datasets it is the native random test split.
``covariate`` (``stress_ood`` only)
    The native test split: train has ``x0 < 0``, test ``x0 >= 0``.  This is
    extrapolation / covariate shift, not a random split.
``mean_shift`` (all other datasets)
    ``x0 <- x0 + 1.5 * std_train(x0)`` on the id test features.  A synthetic
    location shift of the test covariates; labels are unchanged.
``noise_shift``
    Test-time heteroscedastic label noise:
    ``y <- y + sigma(x) * eps``, ``eps ~ N(0, 1)``,
    ``sigma(x) = 0.35 * std(y_fit) * (0.5 + |z0|)`` and
    ``z0 = (x0 - mean_fit(x0)) / std_fit(x0)``.  The shifted noise law is
    never observed during training: this is a robustness stress test, not an
    adaptation benchmark.

Each method is fitted **once per dataset** (on the same rows, with the same
epochs / batch size / lr); all views are then scored against that fit.  Shift
rows carry ``OOD_gap_<metric> = shift - id``.  A positive gap is an increase;
for ``Coverage68`` / ``Coverage95`` a negative gap therefore means
undercoverage.  ``train_s`` is reported once, on the ``id`` row; shift rows say
"fit shared with the id row" in ``Notes``.

Methods (``_MLP(in_dim, _hidden_size(in_dim), out_dim)`` for all four; only
the output head width differs where the parameterisation requires it):

* ``MSE`` — deterministic point MLP + a global in-sample residual std
  (homoscedastic plug-in).  Deliberately the weakest uncertainty baseline:
  it has no input-dependent variance and is included for coverage / width
  comparison, not as a competitive density model.
* ``GaussianNLL`` — ``[mean, log_var]`` head trained with ``GaussianNLLLoss``
  (aleatoric uncertainty only; single model, no ensemble).
* ``BetaNLL`` — the same head trained with ``BetaNLLLoss(beta=0.5)``, which
  down-weights the variance gradient with a detached ``var**beta`` coefficient
  (Seitzer et al., ICLR 2022).
* ``MDN_K3`` — a 3-component Gaussian mixture head (``MixtureDensityLoss``,
  9 outputs), sampled 256 times per test point.  Coverage / width / PIT use
  those samples; NLL and CRPS are the harness's moment-Gaussian scores, the
  same contract used for every sample-based method.

Metrics (all from :func:`harness.metrics.compute_all_metrics`, plus the
explicit coverage columns):

* point: RMSE, MAE (and the other standard point metrics);
* proper scores: NLL, CRPS, Sharpness;
* intervals: the harness 90% ``Coverage`` / ``Width`` / ``IntervalScore``,
  and explicit ``Coverage68`` / ``Width68``, ``Coverage90`` / ``Width90``,
  ``Coverage95`` / ``Width95`` in the original target units;
* calibration: ECE, MACE, RMSCE, PIT_Mean, PIT_Std, PIT_KS, PIT_CvM;
* per-shift breakdown (the ``Shift`` / ``ShiftSpec`` columns) and
  ``OOD_gap_NLL`` / ``CRPS`` / ``RMSE`` / ``ECE`` / ``Coverage68`` /
  ``Coverage95`` / ``Width95``.

Caveats
-------
* The noise shift perturbs test labels with a law unseen in training, so all
  learned variance heads are misspecified by construction.  That is the
  point: the suite measures how far calibration degrades, not how quickly a
  method could adapt with shifted labels.
* ``MDN_K3``'s NLL / CRPS are Gaussian approximations to the mixture (the
  harness reduces samples to mean/std); they are comparable across rows but
  are not native mixture-density scores.
* The ``MSE`` baseline's residual std is estimated in-sample, so its coverage
  is slightly optimistic even in-distribution.
* ``stress_ood``'s id carve costs that dataset 20% of its training rows;
  every method on that dataset pays the same cost.
* Smoke runs (2 epochs, ≤200 fit / ≤50 evaluation rows) validate wiring only;
  they are not evidence about method quality.
* This is a synthetic tabular analogue, not a reproduction of the image
  benchmark in arXiv:2302.03679.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchregress.losses import BetaNLLLoss, GaussianNLLLoss, MixtureDensityLoss

from harness.core import set_harness_seed
from harness.datasets import SplitsDict
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _gaussian_nll_predict, _hidden_size, _train_one

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

_NORMAL = NormalDist()
_Z68 = _NORMAL.inv_cdf(0.84)  # two-sided 68% Gaussian interval
_Z90 = _NORMAL.inv_cdf(0.95)  # two-sided 90% Gaussian interval
_Z95 = _NORMAL.inv_cdf(0.975)  # two-sided 95% Gaussian interval

_MDN_COMPONENTS = 3
_MDN_SAMPLES = 256
_CARVE_FRAC = 0.20  # stress_ood only: id test carved from the training split
_MEAN_SHIFT_STDS = 1.5  # x0 += 1.5 * std_train(x0)
_NOISE_FRAC = 0.35  # noise_shift sigma relative to std(y_fit)
_SMOKE_FIT = 200
_SMOKE_EVAL = 50

# Metrics whose shift-minus-id difference is reported on every row.
_GAP_METRICS = ("NLL", "CRPS", "RMSE", "ECE", "Coverage68", "Coverage95", "Width95")

_METHODS = ("MSE", "GaussianNLL", "BetaNLL", "MDN_K3")


# ---------------------------------------------------------------------------
# Deterministic shift construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ShiftView:
    """One deterministic evaluation view of a dataset."""

    name: str
    X: np.ndarray
    y: np.ndarray
    spec: str


def _safe_std(values: np.ndarray) -> float:
    """Standard deviation with a floor, for degenerate constant features."""
    std = float(np.std(np.asarray(values, dtype=np.float64).reshape(-1)))
    return std if np.isfinite(std) and std > 1e-8 else 1.0


def _noise_spec(seed: int) -> str:
    return (
        "test-time heteroscedastic label noise: y += N(0, sigma(x)^2), "
        f"sigma(x) = {_NOISE_FRAC:.2f} * std(y_fit) * (0.5 + |z0|), "
        f"z0 = (x0 - mean_fit(x0)) / std_fit(x0); seed={seed}"
    )


def _add_noise(
    X: np.ndarray,
    y: np.ndarray,
    sigma0: float,
    mu0: float,
    sd0: float,
    seed: int,
) -> np.ndarray:
    """Add heteroscedastic Gaussian label noise, scaled by |standardised x0|."""
    rng = np.random.default_rng(seed)
    z0 = (np.asarray(X[:, 0], dtype=np.float64) - mu0) / sd0
    sigma = sigma0 * (0.5 + np.abs(z0))
    noise = sigma * rng.standard_normal(len(z0))
    return np.asarray(y, dtype=np.float64).reshape(-1) + noise


def _build_views(
    dataset_name: str,
    splits: SplitsDict,
    seed: int,
    *,
    smoke: bool,
) -> tuple[np.ndarray, np.ndarray, list[_ShiftView]]:
    """Build the fit set and the deterministic id / shift evaluation views.

    Returns ``(X_fit, y_fit, views)``.  The views are derived only from the
    supplied splits plus fixed seeds — no dataset loader is modified.
    """
    X_train = np.asarray(splits["X_train"])
    y_train = np.asarray(splits["y_train"]).reshape(-1)
    X_test = np.asarray(splits["X_test"])
    y_test = np.asarray(splits["y_test"]).reshape(-1)

    if dataset_name == "stress_ood":
        # The native test split is the shifted one (train x0 < 0, test x0 >= 0),
        # so an in-distribution test set must be carved from the training data.
        n = len(X_train)
        n_id = max(1, int(round(_CARVE_FRAC * n)))
        if n - n_id < 10:
            raise ValueError(
                f"stress_ood split too small for a {_CARVE_FRAC:.0%} id carve: "
                f"{n} training rows (need >= 10 fit rows)"
            )
        perm = np.random.default_rng(seed).permutation(n)
        id_idx, fit_idx = perm[:n_id], perm[n_id:]
        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
        X_id, y_id = X_train[id_idx], y_train[id_idx]

        sigma0 = _NOISE_FRAC * _safe_std(y_fit)
        mu0 = float(np.mean(np.asarray(X_fit[:, 0], dtype=np.float64)))
        sd0 = _safe_std(X_fit[:, 0])

        views = [
            _ShiftView(
                "id",
                X_id,
                y_id,
                f"held-out {_CARVE_FRAC:.0%} carve of the training split (x0 < 0); seed={seed}",
            ),
            _ShiftView(
                "covariate",
                X_test,
                y_test,
                "native test split (train x0 < 0, test x0 >= 0): covariate shift / extrapolation",
            ),
            _ShiftView(
                "noise_shift",
                X_id,
                _add_noise(X_id, y_id, sigma0, mu0, sd0, seed + 1),
                f"{_noise_spec(seed + 1)}; applied to the id view rows (x0 < 0)",
            ),
        ]
    else:
        # Native train/test is a random split, so the test split is the id view
        # and the two shifts are paired perturbations of exactly those rows.
        X_fit, y_fit = X_train, y_train
        sigma0 = _NOISE_FRAC * _safe_std(y_fit)
        mu0 = float(np.mean(np.asarray(X_fit[:, 0], dtype=np.float64)))
        sd0 = _safe_std(X_fit[:, 0])
        delta = _MEAN_SHIFT_STDS * _safe_std(X_fit[:, 0])
        X_mean = np.array(X_test, copy=True)
        X_mean[:, 0] = X_mean[:, 0] + delta

        views = [
            _ShiftView("id", X_test, y_test, "native random test split (in-distribution)"),
            _ShiftView(
                "mean_shift",
                X_mean,
                y_test,
                f"x0 += {delta:.4f} ({_MEAN_SHIFT_STDS} * std_train(x0)); labels unchanged",
            ),
            _ShiftView(
                "noise_shift",
                X_test,
                _add_noise(X_test, y_test, sigma0, mu0, sd0, seed + 1),
                _noise_spec(seed + 1),
            ),
        ]

    if smoke:
        # Tiny prefix subsets for CI; the views keep their row correspondence.
        X_fit = X_fit[:_SMOKE_FIT]
        y_fit = y_fit[:_SMOKE_FIT]
        views = [_ShiftView(v.name, v.X[:_SMOKE_EVAL], v.y[:_SMOKE_EVAL], v.spec) for v in views]

    return X_fit, y_fit, views


# ---------------------------------------------------------------------------
# Prediction packing
# ---------------------------------------------------------------------------


def _pack_gaussian(mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    """Gaussian predictive → harness keys + explicit 68/90/95% intervals."""
    mean_np = np.asarray(mean, dtype=np.float64).reshape(-1)
    std_np = np.clip(np.asarray(std, dtype=np.float64).reshape(-1), 1e-8, None)
    lower90 = mean_np - _Z90 * std_np
    upper90 = mean_np + _Z90 * std_np
    return {
        "mean": mean_np,
        "std": std_np,
        "y_pred": mean_np,
        "lower": lower90,
        "upper": upper90,
        "q68_lower": mean_np - _Z68 * std_np,
        "q68_upper": mean_np + _Z68 * std_np,
        "q90_lower": lower90,
        "q90_upper": upper90,
        "q95_lower": mean_np - _Z95 * std_np,
        "q95_upper": mean_np + _Z95 * std_np,
    }


def _pack_samples(samples: np.ndarray) -> dict[str, np.ndarray]:
    """Monte-Carlo predictive samples → harness keys + explicit intervals."""
    sample_np = np.asarray(samples, dtype=np.float64)
    if sample_np.ndim != 2:
        raise ValueError(f"predictive samples must be 2-D (N, S), got {sample_np.shape}")
    mean = sample_np.mean(axis=1)
    std = np.clip(sample_np.std(axis=1), 1e-8, None)
    return {
        "mean": mean,
        "std": std,
        "y_pred": mean,
        "lower": np.quantile(sample_np, 0.05, axis=1),
        "upper": np.quantile(sample_np, 0.95, axis=1),
        "q68_lower": np.quantile(sample_np, 0.16, axis=1),
        "q68_upper": np.quantile(sample_np, 0.84, axis=1),
        "q90_lower": np.quantile(sample_np, 0.05, axis=1),
        "q90_upper": np.quantile(sample_np, 0.95, axis=1),
        "q95_lower": np.quantile(sample_np, 0.025, axis=1),
        "q95_upper": np.quantile(sample_np, 0.975, axis=1),
        # Consumed by the harness PIT path (empirical sample CDF).  NLL / CRPS
        # are still the harness moment-Gaussian scores computed from mean/std.
        "samples": sample_np,
    }


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class UQShiftSuite(HarnessRunner):
    """Uncertainty under covariate / mean / heteroscedastic-noise shifts.

    Four torchregress methods share one MLP backbone and training budget and
    are scored on an in-distribution view and two deterministic shift views
    (see the module docstring for the exact protocol).
    """

    @property
    def name(self) -> str:
        return "uq_shift"

    @property
    def description(self) -> str:
        return (
            "Uncertainty under covariate / mean / heteroscedastic-noise shifts: "
            "GaussianNLL, BetaNLL, MDN vs MSE"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "stress_ood",
            "synthetic_eiv",
            "synthetic_hetero",
        ]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def library_name(self, method: str) -> str:
        return "torchregress"

    def library_version(self, method: str) -> str:
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        notes = {
            "MSE": (
                "deterministic MLP + global in-sample residual std (homoscedastic "
                "plug-in; no input-dependent uncertainty)"
            ),
            "GaussianNLL": (
                "MLP [mean, log_var] head trained with GaussianNLLLoss; aleatoric "
                "uncertainty only (single model, no ensemble)"
            ),
            "BetaNLL": (
                "same MLP [mean, log_var] head trained with BetaNLLLoss(beta=0.5): "
                "detached-variance weighting (Seitzer et al., ICLR 2022)"
            ),
            "MDN_K3": (
                f"{_MDN_COMPONENTS}-component Gaussian mixture head (MixtureDensityLoss, "
                f"{_MDN_COMPONENTS * 3} outputs), {_MDN_SAMPLES} predictive samples for "
                "quantiles/PIT; NLL/CRPS are the harness moment-Gaussian scores"
            ),
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: SplitsDict,
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_np = np.asarray(splits["X_train"])
        y_np = np.asarray(splits["y_train"]).reshape(-1)
        in_dim = X_np.shape[1]
        hidden = _hidden_size(in_dim)
        device = current_device()
        X = torch.from_numpy(X_np).float().to(device)
        y = torch.from_numpy(y_np).float().to(device).unsqueeze(1)

        if method == "MSE":
            model = _MLP(in_dim, hidden, out_dim=1)
            _train_one(model, nn.MSELoss(), X, y, config)
            model.eval()
            with torch.no_grad():
                residuals = (y - model(X)).detach().cpu().numpy().astype(np.float64).reshape(-1)
            sigma = float(np.sqrt(np.mean(np.square(residuals))))
            if not np.isfinite(sigma) or sigma <= 0.0:
                sigma = 1e-6
            return {"type": "mse", "model": model, "sigma": sigma}

        if method == "GaussianNLL":
            model = _MLP(in_dim, hidden, out_dim=2)
            _train_one(model, GaussianNLLLoss(), X, y, config)
            model.eval()
            return {"type": "gaussian", "model": model}

        if method == "BetaNLL":
            model = _MLP(in_dim, hidden, out_dim=2)
            _train_one(model, BetaNLLLoss(beta=0.5), X, y, config)
            model.eval()
            return {"type": "gaussian", "model": model}

        if method == "MDN_K3":
            model = _MLP(in_dim, hidden, out_dim=_MDN_COMPONENTS * 3)
            loss_fn = MixtureDensityLoss(n_components=_MDN_COMPONENTS, n_features=1)
            _train_one(model, loss_fn, X, y, config)
            model.eval()
            return {"type": "mdn", "model": model, "loss_fn": loss_fn}

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        model: nn.Module = model_state["model"]
        device = next(model.parameters()).device
        X_t = torch.from_numpy(np.asarray(X_test, dtype=np.float32)).float().to(device)
        n_test = len(X_test)

        if model_state["type"] == "mse":
            with torch.no_grad():
                mean = model(X_t)[:, 0].detach().cpu().numpy().reshape(-1)
            std = np.full(n_test, float(model_state["sigma"]), dtype=np.float64)
            return _pack_gaussian(mean, std)

        if model_state["type"] == "gaussian":
            mean, std = _gaussian_nll_predict(model, np.asarray(X_test, dtype=np.float32))
            return _pack_gaussian(mean, std)

        if model_state["type"] == "mdn":
            # Sampling is stochastic; pin it so a fitted model's shift rows are
            # reproducible across repeated runs with the same config seed.
            set_harness_seed(config.seed + 19)
            with torch.no_grad():
                raw = model_state["loss_fn"].sample(model(X_t), n_samples=_MDN_SAMPLES)
            samples = raw.squeeze(-1).detach().cpu().numpy().astype(np.float64)
            if samples.shape == (_MDN_SAMPLES, n_test):
                samples = samples.T
            elif samples.shape != (n_test, _MDN_SAMPLES):
                raise RuntimeError(
                    "MDN predictive samples have unexpected shape "
                    f"{samples.shape}; expected ({_MDN_SAMPLES}, {n_test}) or "
                    f"({n_test}, {_MDN_SAMPLES})"
                )
            return _pack_samples(samples)

        raise ValueError(f"Unknown model type: {model_state['type']}")

    # ------------------------------------------------------------------
    # evaluate — standard harness metrics + explicit 68/90/95% coverage
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, Any],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        metrics = super().evaluate(method, predictions, y_test, config)
        y = np.asarray(y_test, dtype=np.float64).reshape(-1)

        # Predictions and y_test are still on the standardized target scale
        # here (coverage is affine-invariant); widths are reported in the
        # original units, matching the harness ``Width`` column.
        y_scaler = predictions.get("_y_scaler")
        scale = 1.0
        if y_scaler is not None:
            try:
                scale = float(np.ravel(y_scaler.scale_)[0])
            except (AttributeError, TypeError, ValueError):
                scale = 1.0

        for level in (68, 90, 95):
            lower = predictions.get(f"q{level}_lower")
            upper = predictions.get(f"q{level}_upper")
            if lower is None or upper is None:
                continue
            lower_np = np.asarray(lower, dtype=np.float64).reshape(-1)
            upper_np = np.asarray(upper, dtype=np.float64).reshape(-1)
            if lower_np.shape != y.shape or upper_np.shape != y.shape:
                continue
            metrics[f"Coverage{level}"] = float(np.mean((y >= lower_np) & (y <= upper_np)))
            metrics[f"Width{level}"] = float(np.mean(upper_np - lower_np)) * scale

        return metrics

    # ------------------------------------------------------------------
    # run — fit once, score id + shift views, attach OOD gaps
    # ------------------------------------------------------------------

    def _row(self, dataset_name: str, method: str, splits: SplitsDict) -> dict[str, Any]:
        return {
            "Method": f"{self.name}/{method}",
            "Library": self.library_name(method),
            "LibraryVersion": self.library_version(method),
            "Dataset": dataset_name,
            "Suite": self.name,
            "SourceURL": splits.get("_source_url", ""),
            "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
            "Device": str(current_device()),
        }

    def _failure_rows(
        self,
        dataset_name: str,
        method: str,
        splits: SplitsDict,
        views: list[_ShiftView],
        note: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for view in views:
            row = self._row(dataset_name, method, splits)
            row.update(
                {
                    "Shift": view.name,
                    "ShiftSpec": view.spec,
                    "RMSE": None,
                    "MAE": None,
                    "train_s": None,
                    "eval_s": None,
                    "Notes": note,
                }
            )
            rows.append(row)
        return rows

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: SplitsDict | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        cfg = config or HarnessConfig()
        dataset_name, splits = self._prepare_scenario_run(
            scenario_name,
            dataset_name,
            splits,
            standardize_targets=cfg.standardize_targets,
        )
        if splits is None:
            if not dataset_name:
                raise ValueError("uq_shift.run requires a dataset name or pre-loaded splits")
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name, standardize_targets=cfg.standardize_targets)

        X_fit, y_fit, views = _build_views(dataset_name, splits, cfg.seed, smoke=cfg.smoke)
        fit_splits: SplitsDict = {**splits, "X_train": X_fit, "y_train": y_fit}

        rows: list[dict[str, Any]] = []
        for method in self.methods():
            set_harness_seed(cfg.seed)
            try:
                t0 = perf_counter()
                model_state = self.fit(method, fit_splits, cfg)
                fit_s = perf_counter() - t0
            except ImportError as exc:
                rows.extend(
                    self._failure_rows(
                        dataset_name, method, splits, views, f"skipped: {type(exc).__name__}: {exc}"
                    )
                )
                continue
            except Exception as exc:
                rows.extend(
                    self._failure_rows(
                        dataset_name, method, splits, views, f"failed: {type(exc).__name__}: {exc}"
                    )
                )
                continue

            id_metrics: dict[str, float] | None = None
            for view in views:
                row = self._row(dataset_name, method, splits)
                row.update({"Shift": view.name, "ShiftSpec": view.spec})
                try:
                    t0 = perf_counter()
                    predictions = self.predict(method, model_state, view.X, cfg)
                    eval_s = perf_counter() - t0
                    if "_y_scaler" in splits:
                        predictions["_y_scaler"] = splits["_y_scaler"]
                    metrics = self.evaluate(method, predictions, view.y, cfg)
                except Exception as exc:
                    row.update(
                        {
                            "RMSE": None,
                            "MAE": None,
                            "train_s": fit_s if view.name == "id" else 0.0,
                            "eval_s": None,
                            "Notes": f"failed: {type(exc).__name__}: {exc}",
                        }
                    )
                    rows.append(row)
                    continue

                gaps: dict[str, float] = {}
                if view.name == "id":
                    id_metrics = metrics
                    gaps = {f"OOD_gap_{key}": 0.0 for key in _GAP_METRICS if key in metrics}
                else:
                    for key in _GAP_METRICS:
                        if id_metrics is None or key not in metrics or key not in id_metrics:
                            continue
                        shifted = float(metrics[key])
                        in_dist = float(id_metrics[key])
                        gaps[f"OOD_gap_{key}"] = (
                            shifted - in_dist
                            if np.isfinite(shifted) and np.isfinite(in_dist)
                            else float("nan")
                        )

                notes = self.method_notes(method)
                if view.name != "id":
                    notes = f"{notes}; fit shared with the id row"
                row.update(metrics)
                row.update(gaps)
                row.update(
                    {
                        "train_s": fit_s if view.name == "id" else 0.0,
                        "eval_s": eval_s,
                        "Notes": notes,
                    }
                )
                rows.append(row)

        return rows
