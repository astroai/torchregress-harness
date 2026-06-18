"""Bayesian linear regression suite: torchregress vs sklearn vs BoTorch.

Ported from ``torchregress/examples/external_comparison_bayesian_linear_vs_botorch.py``.

Compares torchregress's exact-conjugate ``BayesianLinearHead`` against
sklearn's ``BayesianRidge`` (parity reference) and BoTorch's ``SingleTaskGP``
on low-shot linear regression tasks.

Canonical scenarios
-------------------
* ``botorch_reference``
  https://botorch.org/tutorials/fit_model_with_torch_optimizer

Key features
------------
- ``BLR_Linear`` vs ``sklearn_BayesianRidge`` — **exact parity**: same
  conjugate Gaussian-Gaussian formulation with matched hyperparameters.
  Predictive means and standard deviations should be bit-identical.
- ``BayesianLinearHead`` — RBF(20) kernel for nonlinear capacity.
- ``BoTorch_SingleTaskGP`` — GP reference (if botorch installed).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import BayesianRidge

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner

try:
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


class BayesianLinearSuite(HarnessRunner):
    """Low-shot Bayesian linear regression benchmark suite."""

    # ------------------------------------------------------------------
    # Suite metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "bayesian_linear"

    @property
    def description(self) -> str:
        return "Bayesian linear regression: torchregress vs sklearn BayesianRidge (parity) vs BoTorch"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "diabetes",
            "california_housing",
            "synthetic_sinusoidal",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """BoTorch canonical GP regression example."""
        return ["botorch_reference"]

    def methods(self) -> list[str]:
        methods = [
            "BLR_Linear",           # torchregress — parity against sklearn
            "sklearn_BayesianRidge", # sklearn — ground truth
            "BayesianLinearHead",    # torchregress — RBF kernel
        ]
        if _BOTORCH_AVAILABLE:
            methods.append("BoTorch_SingleTaskGP")
        return methods

    def library_name(self, method: str) -> str:
        if method.startswith("BoTorch"):
            return "BoTorch"
        if method.startswith("sklearn"):
            return "sklearn"
        return "torchregress"

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(
        self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        # Enforce low-shot regime for meaningful Bayesian comparison
        X_full = splits["X_train"]
        y_full = splits["y_train"]
        n_train = min(50, len(X_full))

        rng = np.random.default_rng(config.seed)
        idx = rng.choice(len(X_full), n_train, replace=False)
        X = X_full[idx]
        y = y_full[idx]
        in_dim = X.shape[1]

        if method == "BLR_Linear" or method == "BayesianLinearHead":
            from torchregress.test_time import BayesianLinearHead

            X_t = torch.from_numpy(X).float()
            y_t = torch.from_numpy(y).float().unsqueeze(-1)

            if method == "BLR_Linear":
                # Parity: match sklearn BayesianRidge exactly.
                # sklearn centres X and y internally before fitting,
                # then adds the intercept back.  We replicate that here
                # so the prior is applied to centred features only.
                sk_br = BayesianRidge(fit_intercept=True)
                sk_br.fit(X, y)
                # sklearn: alpha_ = noise precision (1/σ²), lambda_ = weight precision (τ)
                prior_prec = float(sk_br.lambda_)       # weight precision
                noise_var = 1.0 / float(sk_br.alpha_)   # noise variance = 1/noise_precision
                rbf = None
                auto = False
                # Center data (same as sklearn internals)
                X_mean = X.mean(axis=0)
                y_mean = float(y.mean())
                X_c = X - X_mean
                y_c = y - y_mean
                X_t = torch.from_numpy(X_c).float()
                y_t = torch.from_numpy(y_c).float().unsqueeze(-1)
            else:
                X_mean = None
                y_mean = None
                prior_prec = 10.0
                noise_var = 1.0
                rbf = 20
                auto = True

            head = BayesianLinearHead(
                in_features=in_dim,
                fit_intercept=False if method == "BLR_Linear" else True,
                prior_precision=prior_prec,
                noise_variance=noise_var,
                auto_noise=auto,
                rbf_centers=rbf,
            ).fit(X_t, y_t)

            result: dict[str, Any] = {
                "type": "blr",
                "head": head,
                "X_test_np": splits["X_test"],
            }
            if method == "BLR_Linear":
                result["X_mean"] = X_mean
                result["y_mean"] = y_mean
            return result

        elif method == "sklearn_BayesianRidge":
            sk_br = BayesianRidge(fit_intercept=True)
            sk_br.fit(X, y)

            return {
                "type": "sklearn_br",
                "est": sk_br,
            }

        elif method == "BoTorch_SingleTaskGP":
            # BoTorch requires double precision
            X_t = torch.from_numpy(X).double()
            y_t = torch.from_numpy(y).double().unsqueeze(-1)

            gp = SingleTaskGP(X_t, y_t)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll, max_attempts=2)
            gp.eval()
            gp.likelihood.eval()

            return {
                "type": "botorch",
                "gp": gp,
                "X_test_np": splits["X_test"],
            }

        raise ValueError(f"Unknown method: {method}")

    def predict(
        self, method: str, model: dict[str, Any], X_test: np.ndarray, config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        if method in ("BLR_Linear", "BayesianLinearHead"):
            head = model["head"]
            X_t = torch.from_numpy(X_test).float()

            # BLR_Linear: un-centre the predictions (trained on centred data)
            if method == "BLR_Linear" and "X_mean" in model:
                X_t = X_t - torch.from_numpy(model["X_mean"]).float()

            pred = head.predict(X_t, return_std=True, include_noise=True)
            mean = pred["mean"].detach().numpy().reshape(-1)
            std = pred["std"].detach().numpy().reshape(-1).clip(1e-8)

            # BLR_Linear: add back y_mean
            if method == "BLR_Linear" and "y_mean" in model:
                mean = mean + model["y_mean"]

            return {"mean": mean, "std": std}

        elif method == "sklearn_BayesianRidge":
            est = model["est"]
            mean, std = est.predict(X_test, return_std=True)
            return {"mean": mean, "std": std}

        elif method == "BoTorch_SingleTaskGP":
            gp = model["gp"]
            X_t = torch.from_numpy(X_test).double()

            with torch.no_grad():
                posterior = gp.posterior(X_t)
                mean = posterior.mean.squeeze(-1).numpy().reshape(-1)
                std = posterior.variance.squeeze(-1).clamp_min(1e-8).sqrt().numpy().reshape(-1)

            return {"mean": mean, "std": std}

        raise ValueError(f"Unknown method: {method}")

    def method_notes(self, method: str) -> str:
        if method == "BLR_Linear":
            return "BLR (no RBF), hyperparams matched to sklearn BayesianRidge"
        elif method == "sklearn_BayesianRidge":
            return "conjugate Bayesian linear regression (ARD priors)"
        elif method == "BayesianLinearHead":
            return "BLR with RBF(20) kernel, γ auto-tuned, auto_noise"
        elif method == "BoTorch_SingleTaskGP":
            return "GP hyperparameter marginal-likelihood fit; capacity not matched"
        return ""

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        """Override to add Cov95 (empirical 95% coverage) alongside standard metrics."""
        # Parent handles inverse-transform and computes RMSE/MAE/R2/NLL/CRPS/etc.
        metrics = super().evaluate(method, predictions, y_test, config)

        # Compute Cov95 on inverse-transformed values (manual — parent only
        # returns aggregate metrics, not the transformed arrays).
        from harness.runners import _inverse_transform_y

        mean = np.asarray(predictions["mean"]).reshape(-1)
        std = np.asarray(predictions["std"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)

        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            mean = _inverse_transform_y(mean, y_scaler)
            scale = float(y_scaler.scale_.item())
            std = std * scale
            y_true = _inverse_transform_y(y_true, y_scaler)

        cov95 = float(np.mean((y_true >= mean - 1.96 * std) & (y_true <= mean + 1.96 * std)))
        metrics["Cov95"] = cov95

        return metrics
