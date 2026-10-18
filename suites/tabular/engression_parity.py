"""Engression parity: official ``engression`` package vs torchregress baselines.

Engression (Shen & Meinshausen, JRSS-B 2024, arXiv:2307.00835) is a neural
distributional regression method trained with an energy score.  Its selling
point is extrapolation for *pre-additive-noise* models

    Y = f(X) + eps,   eps independent of X,

where a conditional mean fit can fail outside the training support while the
stochastic engression network still recovers the noise law.  This suite puts
the official ``engression`` package (v1.1.x, torch-based) next to plain and
distributional torchregress baselines on the harness's existing stress,
synthetic-heteroscedastic and real tabular datasets.  It does not reproduce
the paper's artificial extrapolation benchmark; it measures the same methods
on shared, seeded splits and budgets.

Methods
-------
* ``Engression`` -- official ``engression.engression(...)`` stochastic network
  trained with the energy loss (external package; skipped cleanly when not
  installed).
* ``MSE_MLP`` -- plain MLP + MSE point-prediction baseline.
* ``GaussianNLL_MLP`` -- MLP with a mean/log-variance head trained with the
  Gaussian NLL (heteroscedastic aleatoric baseline).
* ``Quantile_MLP`` -- MLP with seven quantile outputs trained with the
  torchregress ``MultiQuantileLoss`` (pinball); 5%/95% intervals come from its
  own quantiles.
* ``NormalizingFlow_NLL`` -- conditional zuko NSF flow + MLP context encoder
  trained by exact NLL (torchregress ``NormalizingFlowLoss``).

Fairness
--------
All neural methods share the same hidden width (``_hidden_size``), depth (two
hidden layers), optimiser (Adam), learning rate, batch size and epoch budget
from :class:`HarnessConfig`.  ``Engression`` additionally has the package's
100 stochastic noise channels (its default for continuous responses) and
computes two stochastic forward passes per energy-loss step; those are
method-inherent and noted in the result rows.  Batch normalisation is disabled
for Engression to match the plain-MLP baselines (the package default enables
it for continuous responses).

Predictions / metrics
---------------------
Engression draws ``n_samples`` response samples per test point and reports the
conditional mean, standard deviation, 5%/95% quantiles, quantile grid and the
raw samples.  The harness therefore reports point error, Gaussian NLL/CRPS and
sharpness (from the sample moments), 90% coverage/width/interval score, PIT
calibration and mean pinball.  The flow baseline uses the same sample-based
reporting; the quantile baseline reports its own pinball and interval metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import pip_git_commit, set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _gaussian_nll_predict, _hidden_size, _train_one

# ---------------------------------------------------------------------------
# Optional external dependency: the official engression package.
#
# The suite must import (and expose its metadata) even in environments where
# the benchmark extra is not installed; ``fit("Engression")`` then raises an
# ImportError, which HarnessRunner.run records as a clean "skipped" row.
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment dependent
    from engression import engression as _fit_engression

    _ENGRESSION_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    _fit_engression = None
    _ENGRESSION_AVAILABLE = False


_METHODS = ["Engression", "MSE_MLP", "GaussianNLL_MLP", "Quantile_MLP", "NormalizingFlow_NLL"]

# Quantile levels exposed by the engression row and the quantile-MLP baseline.
_QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
_MEDIAN_INDEX = _QUANTILE_LEVELS.index(0.50)
_LOWER_INDEX = _QUANTILE_LEVELS.index(0.05)
_UPPER_INDEX = _QUANTILE_LEVELS.index(0.95)

_N_SAMPLES = 256
_N_SAMPLES_SMOKE = 128

# 90% prediction intervals (nominal), matching harness interval metrics.
_Z_90 = 1.6448536269514722

# Engression architecture knobs held fixed across datasets: ``num_layer=3``
# gives two hidden layers + a linear output, the same linear depth as ``_MLP``.
_ENGRESSION_NUM_LAYER = 3
_ENGRESSION_NOISE_DIM = 100


def _engression_version() -> str:
    """Installed ``engression`` distribution version (``"unknown"`` if absent)."""
    try:
        from importlib.metadata import version

        return version("engression")
    except Exception:
        return "unknown"


def _engression_provenance() -> str:
    """Version string for the external row, including a git SHA when available."""
    ver = _engression_version()
    commit = pip_git_commit("engression")
    if commit:
        return f"engression=={ver} (git {commit[:12]})"
    return f"engression=={ver}"


def _require_engression() -> None:
    """Raise a self-describing ImportError when the optional package is absent."""
    if not _ENGRESSION_AVAILABLE:
        raise ImportError(
            "engression is not installed; run `pip install engression` "
            "(optional extra 'engression') to enable the Engression row"
        )


def _sample_stats(samples: np.ndarray) -> dict[str, np.ndarray]:
    """Summarise ``(n_test, n_samples)`` draws for compute_all_metrics.

    Returns mean/std plus the 5%/95% interval and the shared quantile grid.
    ``std`` is clipped away from zero so the Gaussian NLL/CRPS used by the
    harness stay finite.
    """
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"expected (n_test, n_samples) samples, got shape {s.shape}")
    mean = s.mean(axis=1)
    std = np.maximum(s.std(axis=1), 1e-8)
    quantile_values = np.quantile(s, _QUANTILE_LEVELS, axis=1).T  # (n_test, n_levels)
    return {
        "y_pred": mean,
        "mean": mean,
        "std": std,
        "lower": quantile_values[:, _LOWER_INDEX],
        "upper": quantile_values[:, _UPPER_INDEX],
        "samples": s,
        "quantile_values": quantile_values,
        "quantile_levels": np.asarray(_QUANTILE_LEVELS, dtype=np.float64),
    }


class EngressionParitySuite(HarnessRunner):
    """Official Engression vs torchregress point/distributional baselines."""

    @property
    def name(self) -> str:
        return "engression_parity"

    @property
    def description(self) -> str:
        return (
            "Engression (official package) vs torchregress MSE/GaussianNLL/quantile/flow "
            "baselines on pre-additive-noise and extrapolation tabular data"
        )

    @property
    def supported_datasets(self) -> list[str]:
        # Existing registry datasets only: an EIV stress set, a synthetic
        # heteroscedastic set, and a real tabular benchmark.  No new loaders.
        return ["stress_noisy_inputs", "synthetic_hetero", "california_housing"]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def library_name(self, method: str) -> str:
        return "engression" if method == "Engression" else "torchregress"

    def library_version(self, method: str) -> str:
        if method == "Engression":
            return _engression_provenance()
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        if method == "Engression":
            return (
                "official `engression` package "
                f"v{_engression_version()} — stochastic neural net trained with the "
                "energy loss; internal input/output standardisation; two hidden layers "
                "matched to the torchregress MLPs (num_layer=3, hidden=_hidden_size), "
                f"noise_dim={_ENGRESSION_NOISE_DIM} (package default), batch-norm off to "
                "match the plain-MLP baselines; Adam lr/batch/epochs from HarnessConfig. "
                "Conditional mean/std/5–95% quantiles from model samples; designed for "
                "extrapolation under pre-additive-noise targets Y=f(X)+eps. "
                "Reference: Shen & Meinshausen (2024), JRSS-B, arXiv:2307.00835."
            )
        notes = {
            "MSE_MLP": "plain MLP (two hidden layers) trained with MSE — point-prediction baseline",
            "GaussianNLL_MLP": (
                "MLP mean/log-variance head trained with Gaussian NLL — heteroscedastic "
                "aleatoric baseline; 90% interval = mean ± z·std"
            ),
            "Quantile_MLP": (
                f"MLP with {len(_QUANTILE_LEVELS)} quantile outputs trained with "
                "torchregress MultiQuantileLoss (pinball); 5%/95% interval from its own "
                "quantiles, plus mean pinball over the grid"
            ),
            "NormalizingFlow_NLL": (
                "conditional zuko NSF flow + MLP context encoder trained with exact NLL "
                "(torchregress NormalizingFlowLoss); sample-based mean/std/quantiles"
            ),
        }
        return notes.get(method, "")

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
        device = current_device()
        X_np = np.asarray(splits["X_train"], dtype=np.float32)
        y_np = np.asarray(splits["y_train"], dtype=np.float32).reshape(-1, 1)
        X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
        in_dim = int(X.shape[1])
        hidden = _hidden_size(in_dim)

        if method == "Engression":
            _require_engression()
            return self._fit_engression(X_np, y_np, config)

        if method == "MSE_MLP":
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=1)
            _train_one(model, nn.MSELoss(), X, y, config)
            model.eval()
            return {"type": "mse", "model": model}

        if method == "GaussianNLL_MLP":
            from torchregress.losses import GaussianNLLLoss

            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=2)
            _train_one(model, GaussianNLLLoss(), X, y, config)
            model.eval()
            return {"type": "gaussian_nll", "model": model}

        if method == "Quantile_MLP":
            from torchregress.losses import MultiQuantileLoss

            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=len(_QUANTILE_LEVELS))
            loss_fn = MultiQuantileLoss(quantiles=list(_QUANTILE_LEVELS))
            _train_one(model, loss_fn, X, y, config)
            model.eval()
            return {"type": "quantile", "model": model}

        if method == "NormalizingFlow_NLL":
            from torchregress.losses import NormalizingFlowLoss, create_flow_model

            # Small flow for smoke runs; the documented [64, 64] / 3-transform
            # configuration is used for full runs (see NormalizingFlowLoss docs).
            flow_hidden = 32 if config.smoke else 64
            flow = create_flow_model(
                n_features=1,
                context_dim=hidden,
                flow_type="nsf",
                n_transforms=2 if config.smoke else 3,
                hidden_features=(flow_hidden, flow_hidden),
                bins=8 if config.smoke else 16,
            ).to(device)
            encoder = _MLP(in_dim=in_dim, hidden=hidden, out_dim=hidden)
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_one(
                encoder,
                loss_fn,
                X,
                y,
                config,
                loss_params=list(flow.parameters()),
            )
            encoder.eval()
            loss_fn.eval()
            return {"type": "nflow", "encoder": encoder, "loss_fn": loss_fn}

        raise ValueError(f"Unknown method: {method}")

    def _fit_engression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit the official Engressor with the harness's shared training budget."""
        assert _fit_engression is not None  # guarded by _require_engression()
        model = _fit_engression(
            x=X,
            y=y,
            num_layer=_ENGRESSION_NUM_LAYER,
            hidden_dim=_hidden_size(int(X.shape[1])),
            noise_dim=_ENGRESSION_NOISE_DIM,
            # Match the plain-MLP baselines (official default is BN for continuous y).
            add_bn=False,
            resblock=False,
            beta=1.0,
            lr=config.lr,
            num_epochs=config.epochs,
            batch_size=config.batch_size,
            device=str(current_device()),
            standardize=True,
            verbose=False,
        )
        return {"type": "engression", "model": model}

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
        device = current_device()
        X_np = np.asarray(X_test, dtype=np.float32)
        X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        kind = model_state["type"]
        n_samples = _N_SAMPLES_SMOKE if config.smoke else _N_SAMPLES

        if kind == "engression":
            model = model_state["model"]
            samples = model.sample(X_np, sample_size=n_samples, expand_dim=True)
            # Official layout: (n_test, response_dim, n_samples).
            samples_np = samples.detach().cpu().numpy()[:, 0, :]
            return _sample_stats(samples_np)

        if kind == "mse":
            model = model_state["model"]
            model.eval()
            with torch.no_grad():
                y_pred = model(X).detach().cpu().numpy().reshape(-1)
            return {"y_pred": y_pred}

        if kind == "gaussian_nll":
            mean, std = _gaussian_nll_predict(model_state["model"], X_np)
            return {
                "y_pred": mean,
                "mean": mean,
                "std": std,
                "lower": mean - _Z_90 * std,
                "upper": mean + _Z_90 * std,
            }

        if kind == "quantile":
            model = model_state["model"]
            model.eval()
            with torch.no_grad():
                raw = model(X).detach().cpu().numpy()
            # Quantile heads can cross under a short smoke budget; sorting is
            # the standard rearrangement fix and keeps intervals valid.
            quantile_values = np.sort(raw, axis=1).astype(np.float64)
            return {
                "y_pred": quantile_values[:, _MEDIAN_INDEX],
                "lower": quantile_values[:, _LOWER_INDEX],
                "upper": quantile_values[:, _UPPER_INDEX],
                "quantile_values": quantile_values,
                "quantile_levels": np.asarray(_QUANTILE_LEVELS, dtype=np.float64),
            }

        if kind == "nflow":
            encoder = model_state["encoder"]
            loss_fn = model_state["loss_fn"]
            encoder.eval()
            loss_fn.eval()
            with torch.no_grad():
                context = encoder(X)
                samples = loss_fn.sample(context, n_samples=n_samples)
            samples_np = samples.detach().cpu().numpy().reshape(len(X_np), n_samples)
            return _sample_stats(samples_np)

        raise ValueError(f"Unknown model type: {kind}")

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        del method, config
        from harness.metrics import compute_all_metrics
        from harness.runners import _inverse_transform_predictions, _inverse_transform_y

        y_scaler = predictions.get("_y_scaler")
        quantile_values = predictions.get("quantile_values")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_test = _inverse_transform_y(y_test, y_scaler)
            if quantile_values is not None:
                # The shared inverse-transform helper covers the standard keys
                # but not the quantile grid; transform it here for pinball.
                scale = float(np.asarray(y_scaler.scale_).reshape(-1)[0])
                mean = float(np.asarray(y_scaler.mean_).reshape(-1)[0])
                quantile_values = np.asarray(quantile_values, dtype=np.float64) * scale + mean

        metrics = compute_all_metrics(predictions, y_test)

        # Mean pinball over the predicted quantile grid (proper scoring rule;
        # comparable between Engression and the quantile baseline).
        if quantile_values is not None and "quantile_levels" in predictions:
            try:
                from torchregress.metrics import pinball_metric

                levels = np.asarray(predictions["quantile_levels"], dtype=np.float64).reshape(-1)
                qv = np.asarray(quantile_values, dtype=np.float64)
                qdict = {float(levels[j]): qv[:, j] for j in range(len(levels))}
                metrics["Pinball"] = float(pinball_metric(qdict, np.asarray(y_test).reshape(-1)))
            except Exception:
                pass

        return metrics


__all__ = ["EngressionParitySuite"]
