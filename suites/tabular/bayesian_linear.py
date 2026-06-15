"""Bayesian linear regression suite: torchregress vs BoTorch.

Ported from ``torchregress/examples/external_comparison_bayesian_linear_vs_botorch.py``.

Compares torchregress's exact-conjugate ``BayesianLinearHead`` against
BoTorch's ``SingleTaskGP`` on low-shot linear regression tasks. Uses real
tabular datasets by subsampling to low-shot regime.

Canonical scenarios
-------------------
* ``botorch_reference``
  https://botorch.org/tutorials/fit_model_with_torch_optimizer

Key differences from the original script:
- Data comes from harness dataset registry (real tabular), not synthetic generation
- Low-shot regime is enforced by subsampling to n_train ~ 30-50 samples
- BoTorch models use double precision as required by gpytorch
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

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
        return "Low-shot Bayesian linear regression: torchregress vs BoTorch"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "diabetes",
            "california_housing",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """BoTorch canonical GP regression example."""
        return ["botorch_reference"]

    def methods(self) -> list[str]:
        methods = ["BayesianLinearHead"]
        if _BOTORCH_AVAILABLE:
            methods.append("BoTorch_SingleTaskGP")
        return methods

    def library_name(self, method: str) -> str:
        if method.startswith("BoTorch"):
            return "BoTorch"
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

        # Estimate noise from data
        noise = max(float(np.std(y)) * 0.2, 0.01)

        if method == "BayesianLinearHead":
            from torchregress.test_time import BayesianLinearHead

            X_t = torch.from_numpy(X).float()
            y_t = torch.from_numpy(y).float().unsqueeze(-1)

            head = BayesianLinearHead(
                in_features=in_dim,
                fit_intercept=False,
                prior_precision=0.01,
                noise_variance=noise**2,
            ).fit(X_t, y_t)

            return {
                "type": "blr",
                "head": head,
                "X_test_np": splits["X_test"],
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
        if method == "BayesianLinearHead":
            head = model["head"]
            X_t = torch.from_numpy(X_test).float()

            pred = head.predict(X_t, return_std=True, include_noise=True)
            mean = pred["mean"].detach().numpy().reshape(-1)
            std = pred["std"].detach().numpy().reshape(-1).clip(1e-8)

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
        if method == "BayesianLinearHead":
            return "exact-conjugate posterior in closed form"
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
        mean = np.asarray(predictions["mean"]).reshape(-1)
        std = np.asarray(predictions["std"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)

        # Standard point + distribution metrics
        from harness.metrics import (
            compute_distribution_metrics,
            compute_point_metrics,
        )

        metrics = {}
        metrics.update(compute_point_metrics(mean, y_true))
        metrics.update(compute_distribution_metrics(mean, std.clip(1e-8), y_true))

        # Empirical 95% interval coverage (matching original script)
        cov95 = float(np.mean((y_true >= mean - 1.96 * std) & (y_true <= mean + 1.96 * std)))
        metrics["Cov95"] = cov95

        return metrics
