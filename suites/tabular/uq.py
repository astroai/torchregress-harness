"""Uncertainty quantification suite: torchregress vs sklearn GP + conformal.

Compares three torchregress UQ strategies (single GaussianNLL, MC-dropout
with 20 forward passes, and a 10-member deep ensemble) against sklearn's
``GaussianProcessRegressor`` and a quantile gradient-boosting conformal
baseline on small-to-medium tabular datasets.

Canonical scenarios
-------------------
* ``sklearn_diabetes_reference``
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
* ``sklearn_synthetic_reference``
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

Architecture
------------ * ``GaussianNLL`` → single MLP (mean, log_var) + BetaNLLLoss → ``{"mean", "std"}``
* ``MC_Dropout`` → MLP with 50% dropout, 20 stochastic forward passes at
  test time → empirical mean & std → ``{"mean", "std"}``
* ``DeepEnsemble_K10`` → 10 independently-trained MLPs → mixture mean &
  total variance → ``{"mean", "std"}``

* ``DeepEnsemble_K5`` / ``+VTS`` / ``+CQR`` (Phase 2b recipe rows) → a
  5-member ensemble trained on the training data minus a 20% validation
  split; ``+VTS`` rescales its variance with ``VarianceTemperatureScaler``
  and ``+CQR`` conformalizes its 90% Gaussian interval with ``CQR``, both
  fitted on that validation split → ``{"mean", "std"}`` (+ interval for CQR).
  ECE uses the same Gaussian PIT as ``sklearn_GP``.

Sklearn baselines produce ``{"mean", "std"}`` (GP) or ``{"y_pred", "lower", "upper"}`` (quantile GBR).
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Sklearn — hard deps
# ---------------------------------------------------------------------------
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# ---------------------------------------------------------------------------
# Torchregress losses
# ---------------------------------------------------------------------------
from torchregress.losses import BetaNLLLoss

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import (
    _MLP,
    _gaussian_nll_predict,
    _hidden_size,
    _mixture_gaussian_stats,
    _train_one,
)
from suites._tabular_mlp import N_MEMBERS, fit_tabmlp, predict_gaussian

# ---------------------------------------------------------------------------
# MC-dropout forward pass
# ---------------------------------------------------------------------------

_MC_SAMPLES = 20  # number of stochastic forward passes


def _mc_predict(
    model: _MLP,
    X: torch.Tensor,
    n_samples: int = _MC_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Run n_samples stochastic forward passes, return empirical mean & std.

    Temporarily enables training mode so dropout is active, then restores
    the original mode before returning.
    """
    was_training = model.training
    model.train()  # enable dropout for stochastic forward passes
    try:
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = model(X)
                preds.append(out[:, 0].detach().cpu().numpy().reshape(-1))
        stacked = np.stack(preds, axis=0)  # (n_samples, n_test)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=0).clip(1e-8)
        return mean, std
    finally:
        if not was_training:
            model.eval()


# ---------------------------------------------------------------------------
# Deep ensemble size (larger than ensembles suite's K=5)
# ---------------------------------------------------------------------------

_N_ENSEMBLE = 10
_N_RECIPE = 5  # K for the recalibrated-ensemble recipe rows
_RECIPE_ALPHA = 0.1  # CQR miscoverage of the conformalized-ensemble row


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class UQSuite(HarnessRunner):
    """Uncertainty quantification benchmark suite.

    Compares torchregress UQ methods (GaussianNLL, MC-dropout,
    deep ensemble K=10) against sklearn GaussianProcessRegressor
    and quantile gradient boosting.
    """

    # GP doesn't scale to large N — cap training samples
    _GP_MAX_SAMPLES = 2000

    @property
    def name(self) -> str:
        return "uq"

    @property
    def description(self) -> str:
        return (
            "UQ methods: GaussianNLL / MC-dropout / DeepEnsemble_K10 vs sklearn GP & quantile GBR"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "diabetes",
            "make_regression_synthetic",
            "synthetic_hetero",
            "synthetic_sinusoidal",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """sklearn canonical regression benchmarks."""
        return ["sklearn_diabetes_reference", "sklearn_synthetic_reference"]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    # Note: BetaNLL is not yet available as a standalone loss in torchregress.
    # When torchregress exposes a BetaNLL loss (e.g., torchregress.losses.BetaNLLLoss),
    # add a "BetaNLL" method here that trains _MLP(in_dim, hidden, 2) with it.

    _TR_METHODS: list[str] = [
        "GaussianNLL",
        "MC_Dropout",
        "DeepEnsemble_K10",
        # Phase 2b recipe rows: a K=5 ensemble fitted on the training data
        # minus a validation split, then recalibrated on that split.
        "DeepEnsemble_K5",
        "DeepEnsemble_K5+VTS",
        "DeepEnsemble_K5+CQR",
        # torchregress.models strong-default TabularMLP with a Gaussian head.
        "TabularMLP_GaussianNLL",
        "TabularMLP_Ens5_GaussianNLL",
        # Same 5-member ensemble fitted on the training data minus the 20% carve
        # (control), then recalibrated with VTS on that carve.
        "TabularMLP_Ens5_GaussianNLL_carve",
        "TabularMLP_Ens5_GaussianNLL+VTS",
    ]

    # Fraction of the training data held out to fit the post-hoc layer.
    _RECIPE_VAL_FRAC = 0.2

    def __init__(self) -> None:
        # The three K5 recipe rows share one ensemble per (data, seed, budget).
        self._k5_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        # The two carved TabularMLP rows share one ensemble per (data, seed, budget).
        self._tab_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    _SKLEARN_METHODS: list[str] = [
        "sklearn_GP",
        "sklearn_QuantileGBR",
    ]

    def methods(self) -> list[str]:
        return self._TR_METHODS + self._SKLEARN_METHODS

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn_"):
            return "sklearn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        notes = {
            "GaussianNLL": "single MLP with BetaNLLLoss(beta=0.5) → aleatoric uncertainty",
            "MC_Dropout": f"MLP with 50% dropout, {_MC_SAMPLES} stochastic passes → empirical mean & std",
            "DeepEnsemble_K10": f"{_N_ENSEMBLE}× MLP + GaussianNLL → mixture mean & total variance",
            "DeepEnsemble_K5": (
                f"{_N_RECIPE}× MLP + BetaNLL on the training data minus a "
                f"{self._RECIPE_VAL_FRAC:.0%} validation split, uncalibrated (control for the +VTS/+CQR rows)"
            ),
            "DeepEnsemble_K5+VTS": (
                "DeepEnsemble_K5 + torchregress VarianceTemperatureScaler fitted on the validation "
                "split (std × sqrt(T)); ECE from the same Gaussian PIT as sklearn_GP; "
                "train_s excludes the ensemble shared with DeepEnsemble_K5"
            ),
            "DeepEnsemble_K5+CQR": (
                "DeepEnsemble_K5 + torchregress CQR on the Gaussian 5%/95% quantiles, calibrated on "
                "the validation split; distribution = mean, std + q/z (the conformal widening as "
                "extra std) so NLL/CRPS/ECE are comparable; train_s excludes the shared ensemble"
            ),
            "TabularMLP_GaussianNLL": (
                "torchregress.models TabularMLP (Gaussian head) + fit_tabular, library defaults, "
                "epochs = min(100, config.epochs); 15% internal early-stopping split"
            ),
            "TabularMLP_Ens5_GaussianNLL": (
                "5 TabularMLP Gaussian members (fit_tabular_ensemble, library defaults) combined as "
                "a Gaussian mixture (mixture mean and total variance)"
            ),
            "TabularMLP_Ens5_GaussianNLL_carve": (
                f"TabularMLP_Ens5_GaussianNLL fitted on the training data minus a "
                f"{self._RECIPE_VAL_FRAC:.0%} carve, uncalibrated (control for the +VTS row)"
            ),
            "TabularMLP_Ens5_GaussianNLL+VTS": (
                "TabularMLP_Ens5_GaussianNLL_carve + torchregress VarianceTemperatureScaler fitted "
                "on the carve (std x sqrt(T)); train_s excludes the ensemble shared with the control"
            ),
            "sklearn_GP": "GaussianProcessRegressor(RBF+WhiteKernel) → posterior mean & std",
            "sklearn_QuantileGBR": "GradientBoostingRegressor(q=0.05,0.5,0.95) → median & 90% intervals",
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # K5 recipe helpers
    # ------------------------------------------------------------------

    def _k5_ensemble(
        self, X_train_np: np.ndarray, y_train_np: np.ndarray, config: HarnessConfig
    ) -> dict[str, Any]:
        """K=5 BetaNLL ensemble on the training data minus a validation split (cached)."""
        key = (
            X_train_np.shape,
            hash(np.ascontiguousarray(X_train_np).tobytes()),
            hash(np.ascontiguousarray(y_train_np).tobytes()),
            config.seed,
            config.epochs,
            config.batch_size,
            config.lr,
        )
        if key in self._k5_cache:
            return self._k5_cache[key]
        n = len(X_train_np)
        n_val = max(20, int(round(self._RECIPE_VAL_FRAC * n)))
        if n - n_val < 10:
            raise ValueError(f"training split too small for a validation carve: n={n}")
        perm = np.random.default_rng(config.seed).permutation(n)
        fit_idx, val_idx = perm[n_val:], perm[:n_val]
        in_dim = X_train_np.shape[1]
        hidden = _hidden_size(in_dim)
        X_t = torch.from_numpy(X_train_np[fit_idx]).float().to(current_device())
        y_t = torch.from_numpy(y_train_np[fit_idx]).float().to(current_device()).unsqueeze(1)
        models: list[nn.Module] = []
        for i in range(_N_RECIPE):
            set_harness_seed(config.seed + i * 1000)
            model = _MLP(in_dim, hidden, 2)
            _train_one(model, BetaNLLLoss(beta=0.5), X_t, y_t, config)
            model.eval()
            models.append(model)
        out = {
            "models": models,
            "X_val": X_train_np[val_idx],
            "y_val": np.asarray(y_train_np[val_idx], dtype=np.float64).reshape(-1),
        }
        self._k5_cache = {key: out}  # keep only the current dataset's ensemble
        return out

    def _tab_ensemble(
        self, X_train_np: np.ndarray, y_train_np: np.ndarray, config: HarnessConfig
    ) -> dict[str, Any]:
        """5-member TabularMLP Gaussian ensemble on the training data minus a carve (cached)."""
        key = (
            X_train_np.shape,
            hash(np.ascontiguousarray(X_train_np).tobytes()),
            hash(np.ascontiguousarray(y_train_np).tobytes()),
            config.seed,
            config.epochs,
        )
        if key in self._tab_cache:
            return self._tab_cache[key]
        n = len(X_train_np)
        n_val = max(20, int(round(self._RECIPE_VAL_FRAC * n)))
        if n - n_val < 10:
            raise ValueError(f"training split too small for a validation carve: n={n}")
        perm = np.random.default_rng(config.seed).permutation(n)
        fit_idx, val_idx = perm[n_val:], perm[:n_val]
        fit = fit_tabmlp(
            X_train_np[fit_idx],
            y_train_np[fit_idx],
            config,
            kind="gaussian",
            n_members=N_MEMBERS,
        )
        out = {
            "fit": fit,
            "X_val": X_train_np[val_idx],
            "y_val": np.asarray(y_train_np[val_idx], dtype=np.float64).reshape(-1),
        }
        self._tab_cache = {key: out}
        return out

    @staticmethod
    def _ensemble_stats(models: list[nn.Module], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        stats = [_gaussian_nll_predict(m, np.asarray(X, dtype=np.float32)) for m in models]
        mean, std = _mixture_gaussian_stats([a for a, _ in stats], [b for _, b in stats])
        return mean.astype(np.float64), std.astype(np.float64)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"].reshape(-1)
        in_dim = X_train_np.shape[1]
        hidden = _hidden_size(in_dim)

        # --- GaussianNLL: single MLP with mean + log_var output ---
        if method == "GaussianNLL":
            X = torch.from_numpy(X_train_np).float().to(current_device())
            y = torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1)

            model = _MLP(in_dim, hidden, 2)  # mean, log_variance
            loss_fn = BetaNLLLoss(beta=0.5)
            _train_one(model, loss_fn, X, y, config)
            model.eval()

            return {"type": "gaussian_nll", "model": model}

        # --- MC_Dropout: MLP with dropout, train normally ---
        elif method == "MC_Dropout":
            X = torch.from_numpy(X_train_np).float().to(current_device())
            y = torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1)

            model = _MLP(in_dim, hidden, 1, dropout_p=0.5)
            loss_fn = nn.MSELoss()
            _train_one(model, loss_fn, X, y, config)
            # Store in eval mode; _mc_predict() will temporarily enable train mode
            model.eval()

            return {"type": "mc_dropout", "model": model}

        # --- DeepEnsemble_K10: 10 independent MLPs ---
        elif method == "DeepEnsemble_K10":
            models: list[nn.Module] = []

            for i in range(_N_ENSEMBLE):
                member_seed = config.seed + i * 1000
                set_harness_seed(member_seed)

                X_t = torch.from_numpy(X_train_np).float().to(current_device())
                y_t = torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1)

                model = _MLP(in_dim, hidden, 2)
                loss_fn = BetaNLLLoss(beta=0.5)
                _train_one(model, loss_fn, X_t, y_t, config)
                model.eval()
                models.append(model)

            return {"type": "deep_ensemble", "models": models}

        # --- Phase 2b recipe rows: K=5 ensemble + post-hoc layer on a validation split ---
        elif method in ("DeepEnsemble_K5", "DeepEnsemble_K5+VTS", "DeepEnsemble_K5+CQR"):
            ens = self._k5_ensemble(X_train_np, y_train_np, config)
            models = ens["models"]
            if method == "DeepEnsemble_K5":
                return {"type": "deep_ensemble", "models": models}

            mu_val, sd_val = self._ensemble_stats(models, ens["X_val"])
            y_val = ens["y_val"]
            if method == "DeepEnsemble_K5+VTS":
                from torchregress.calibration import VarianceTemperatureScaler

                vts = VarianceTemperatureScaler().fit(
                    torch.as_tensor(mu_val, dtype=torch.float64),
                    torch.as_tensor(sd_val**2, dtype=torch.float64),
                    torch.as_tensor(y_val, dtype=torch.float64),
                )
                return {"type": "deep_ensemble_vts", "models": models, "scaler": vts}

            from torchregress.losses.conformal import CQR

            z = NormalDist().inv_cdf(1.0 - _RECIPE_ALPHA / 2)
            q_val = np.stack([mu_val - z * sd_val, mu_val + z * sd_val], axis=1)
            cqr = CQR(alpha=_RECIPE_ALPHA)
            cqr.calibrate(
                torch.as_tensor(q_val, dtype=torch.float64),
                torch.as_tensor(y_val, dtype=torch.float64),
            )
            return {"type": "deep_ensemble_cqr", "models": models, "cqr": cqr, "z": z}

        # --- TabularMLP (library defaults) with a Gaussian head ---
        elif method in ("TabularMLP_GaussianNLL", "TabularMLP_Ens5_GaussianNLL"):
            fit = fit_tabmlp(
                X_train_np,
                y_train_np,
                config,
                kind="gaussian",
                n_members=N_MEMBERS if "Ens5" in method else 1,
            )
            return {"type": "tabular_gaussian", "fit": fit}

        elif method in ("TabularMLP_Ens5_GaussianNLL_carve", "TabularMLP_Ens5_GaussianNLL+VTS"):
            ens = self._tab_ensemble(X_train_np, y_train_np, config)
            if method.endswith("_carve"):
                return {"type": "tabular_gaussian", "fit": ens["fit"]}
            from torchregress.calibration import VarianceTemperatureScaler

            mu_val, sd_val = predict_gaussian(ens["fit"], ens["X_val"])
            vts = VarianceTemperatureScaler().fit(
                torch.as_tensor(mu_val, dtype=torch.float64),
                torch.as_tensor(sd_val**2, dtype=torch.float64),
                torch.as_tensor(ens["y_val"], dtype=torch.float64),
            )
            return {"type": "tabular_gaussian_vts", "fit": ens["fit"], "scaler": vts}

        # --- sklearn GP: subsample if dataset is large ---
        elif method == "sklearn_GP":
            n_train = len(X_train_np)
            if n_train > self._GP_MAX_SAMPLES:
                rng = np.random.default_rng(config.seed)
                idx = rng.choice(n_train, self._GP_MAX_SAMPLES, replace=False)
                X_gp = X_train_np[idx]
                y_gp = y_train_np[idx]
            else:
                X_gp = X_train_np
                y_gp = y_train_np

            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=config.seed,
            )
            gp.fit(X_gp, y_gp)

            return {
                "type": "gp",
                "model": gp,
                "n_train": n_train,
                "n_used": len(X_gp),
            }

        # --- sklearn quantile GBR: three separate models per quantile ---
        elif method == "sklearn_QuantileGBR":
            quantiles = [0.05, 0.50, 0.95]
            models = {}
            for q in quantiles:
                gbr = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=config.seed,
                )
                gbr.fit(X_train_np, y_train_np)
                models[q] = gbr

            return {"type": "quantile_gbr", "models": models, "quantiles": quantiles}

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
        model_type = model_state["type"]

        # --- GaussianNLL → mean & std ---
        if model_type == "gaussian_nll":
            mean, std = _gaussian_nll_predict(model_state["model"], X_test)
            return {"mean": mean, "std": std}

        # --- MC_Dropout → empirical mean & std ---
        elif model_type == "mc_dropout":
            m = model_state["model"]
            mean, std = _mc_predict(m, torch.from_numpy(X_test).float().to(current_device()))
            return {"mean": mean, "std": std}

        # --- DeepEnsemble_K10 → mixture mean & total variance ---
        elif model_type == "deep_ensemble":
            means = []
            stds = []
            for m in model_state["models"]:
                mu, std = _gaussian_nll_predict(m, X_test)
                means.append(mu)
                stds.append(std)
            mixture_mean, mixture_std = _mixture_gaussian_stats(means, stds)
            return {"mean": mixture_mean, "std": mixture_std}

        elif model_type == "tabular_gaussian":
            mean, std = predict_gaussian(model_state["fit"], X_test)
            return {"mean": mean, "std": std}

        elif model_type == "tabular_gaussian_vts":
            mean, std = predict_gaussian(model_state["fit"], X_test)
            var = model_state["scaler"].transform(torch.as_tensor(std**2, dtype=torch.float64))
            return {"mean": mean, "std": np.sqrt(var.numpy()).reshape(-1)}

        # --- K5 + VarianceTemperatureScaler → mean & recalibrated std ---
        elif model_type == "deep_ensemble_vts":
            mean, std = self._ensemble_stats(model_state["models"], X_test)
            var = model_state["scaler"].transform(torch.as_tensor(std**2, dtype=torch.float64))
            return {"mean": mean, "std": np.sqrt(var.numpy()).reshape(-1)}

        # --- K5 + CQR → conformalized 90% interval (+ widened Gaussian) ---
        elif model_type == "deep_ensemble_cqr":
            mean, std = self._ensemble_stats(model_state["models"], X_test)
            z = model_state["z"]
            q = np.stack([mean - z * std, mean + z * std], axis=1)
            lo, hi = model_state["cqr"].predict_interval(torch.as_tensor(q, dtype=torch.float64))
            lo_np = lo.detach().cpu().numpy().reshape(-1)
            hi_np = hi.detach().cpu().numpy().reshape(-1)
            # Symmetric CQR adds q_hat to both Gaussian quantiles: the same
            # Gaussian with std + q_hat / z.  An infinite q_hat (tiny
            # validation split) gives an infinite std, which the harness
            # metrics report via InfiniteIntervalFrac / NonFiniteStdFrac.
            std_c = np.maximum((hi_np - lo_np) / (2.0 * z), 1e-8)
            return {"mean": mean, "std": std_c, "lower": lo_np, "upper": hi_np}

        # --- sklearn GP → posterior mean & std ---
        elif model_type == "gp":
            gp = model_state["model"]
            mean, std = gp.predict(X_test, return_std=True)
            return {"mean": mean.reshape(-1), "std": std.clip(1e-8)}

        # --- sklearn quantile GBR → median + 90% intervals ---
        elif model_type == "quantile_gbr":
            lo = model_state["models"][0.05].predict(X_test).reshape(-1)
            med = model_state["models"][0.50].predict(X_test).reshape(-1)
            hi = model_state["models"][0.95].predict(X_test).reshape(-1)
            return {"y_pred": med, "lower": lo, "upper": hi}

        raise ValueError(f"Unknown model type: {model_type}")
