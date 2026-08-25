"""Foundation model suite: torchregress vs TabPFN vs TabICL.

Compares torchregress probabilistic models against TabPFN and TabICL
tabular foundation models on real and synthetic benchmarks.

Canonical scenarios
-------------------
* ``sklearn_diabetes_reference``
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
* ``sklearn_synthetic_reference``
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Torchregress losses
# ---------------------------------------------------------------------------
from torchregress.losses import BetaNLLLoss, MixtureDensityLoss

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _gaussian_nll_predict, _hidden_size, _train_one

# ---------------------------------------------------------------------------
# Optional foundation model imports
# ---------------------------------------------------------------------------

try:
    import tabpfn  # noqa: F401

    _TABPFN_AVAILABLE = True
    _TABPFN_ERROR: str | None = None
except ImportError:
    _TABPFN_AVAILABLE = False
    _TABPFN_ERROR = "TabPFN not installed"

try:
    import tabicl  # noqa: F401

    _TABICL_AVAILABLE = True
    _TABICL_ERROR: str | None = None
except ImportError:
    _TABICL_AVAILABLE = False
    _TABICL_ERROR = "TabICL not installed"


# ---------------------------------------------------------------------------


class FoundationModelSuite(HarnessRunner):
    """Tabular foundation model benchmark suite.

    Compares torchregress GaussianNLL and MDN against TabPFN and TabICL
    foundation models on tabular regression benchmarks.
    """

    _MDN_COMPONENTS = 3  # number of Gaussian mixture components

    @property
    def name(self) -> str:
        return "foundation_model"

    @property
    def description(self) -> str:
        return "Foundation model regression: torchregress (GaussianNLL, MDN) vs TabPFN vs TabICL"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "diabetes",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """sklearn regression examples relevant to foundation model comparison."""
        return ["sklearn_diabetes_reference", "sklearn_synthetic_reference"]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        methods = ["GaussianNLL", "MDN"]
        if _TABPFN_AVAILABLE:
            methods.append("TabPFN")
        if _TABICL_AVAILABLE:
            methods.append("TabICL")
        return methods

    def library_name(self, method: str) -> str:
        if method == "TabPFN":
            return "tabpfn"
        if method == "TabICL":
            return "tabicl"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        notes = {
            "GaussianNLL": "2-head MLP + BetaNLLLoss(beta=0.5) → mean & std",
            "MDN": (
                f"{self._MDN_COMPONENTS}-component MixtureDensityLoss "
                "→ mean & std via predict_mean_std()"
            ),
            "TabPFN": (
                "TabPFNRegressor (pre-trained Transformer) → point predictions.  "
                "Requires: tabpfn --accept-license (one-time)"
            ),
            "TabICL": "TabICLRegressor (in-context learning Transformer) → point predictions",
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

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"].reshape(-1)
        in_dim = X_train_np.shape[1]
        hidden = _hidden_size(in_dim)

        # --- torchregress: GaussianNLL ---
        if method == "GaussianNLL":
            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            model = _MLP(in_dim, hidden, 2)  # mean, log_variance
            loss_fn = BetaNLLLoss(beta=0.5)
            _train_one(model, loss_fn, X, y, config)
            model.eval()

            return {"type": "gaussian_nll", "model": model}

        # --- torchregress: MDN ---
        elif method == "MDN":
            X = torch.from_numpy(X_train_np).float()
            y = torch.from_numpy(y_train_np).float().unsqueeze(1)

            n_components = self._MDN_COMPONENTS
            n_features = 1  # scalar regression

            loss_fn = MixtureDensityLoss(
                n_components=n_components,
                n_features=n_features,
                covariance_type="diagonal",
            )
            out_dim = loss_fn.expected_output_size
            model = _MLP(in_dim, hidden, out_dim)
            _train_one(model, loss_fn, X, y, config)
            model.eval()

            return {
                "type": "mdn",
                "model": model,
                "loss_fn": loss_fn,
                "n_components": n_components,
            }

        # --- TabPFN ---
        elif method == "TabPFN":
            from tabpfn import TabPFNRegressor

            est = TabPFNRegressor(device=str(current_device()))
            est.fit(X_train_np, y_train_np)
            return {"type": "tabpfn", "model": est}

        # --- TabICL ---
        elif method == "TabICL":
            from tabicl import TabICLRegressor

            est = TabICLRegressor(device=str(current_device()))
            est.fit(X_train_np, y_train_np)
            return {"type": "tabicl", "model": est}

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

        # --- torchregress: GaussianNLL → mean & std ---
        if model_type == "gaussian_nll":
            mean, std = _gaussian_nll_predict(model_state["model"], X_test)
            return {"mean": mean, "std": std}

        # --- torchregress: MDN → mean & std via loss.predict_mean_std() ---
        elif model_type == "mdn":
            model = model_state["model"]
            loss_fn = model_state["loss_fn"]
            with torch.no_grad():
                out = model(torch.from_numpy(X_test).float())
                mean, std = loss_fn.predict_mean_std(out)
            return {
                "mean": mean.squeeze(-1).numpy(),
                "std": std.squeeze(-1).numpy(),
            }

        # --- TabPFN: point predictions only ---
        elif model_type == "tabpfn":
            est = model_state["model"]
            y_pred = np.asarray(est.predict(X_test), dtype=np.float64).reshape(-1)
            return {"y_pred": y_pred}

        # --- TabICL: point predictions only ---
        elif model_type == "tabicl":
            est = model_state["model"]
            y_pred = np.asarray(est.predict(X_test), dtype=np.float64).reshape(-1)
            return {"y_pred": y_pred}

        raise ValueError(f"Unknown model type: {model_type}")
