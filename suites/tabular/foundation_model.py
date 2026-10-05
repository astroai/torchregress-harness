"""Foundation model suite: torchregress vs tabular foundation models.

Compares torchregress probabilistic models against pretrained tabular
foundation models used zero-shot (in-context learning, no training):

* TabPFN (Prior Labs, ``tabpfn``) and TabICL (``tabicl``) via their own
  scikit-learn wrappers;
* NVIDIA's Structured Data Models (``sdm``, Apache-2.0 code), which ships
  KumoTabular (NVIDIA, 2026-09), TabICLv2 (ICML 2026) and Google's TabFM
  (2026-06) behind one interface.  TabFM's default weights are under the
  non-commercial ``tabfm-non-commercial-v1.0`` licence, so the row passes
  ``accept_license=True``: fine for benchmarking, not for production use.

Every foundation model returns a grid of predictive quantiles, scored as an
empirical predictive distribution (point metrics from its mean, CRPS/PIT from
the grid, 90% interval from the 5% and 95% quantiles).

Canonical scenarios
-------------------
* ``sklearn_diabetes_reference``
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
* ``sklearn_synthetic_reference``
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
"""

from __future__ import annotations

import os
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

try:
    import sdm  # noqa: F401

    _SDM_AVAILABLE = True
except ImportError:
    _SDM_AVAILABLE = False

# Quantile grid for TabPFN/TabICL: midpoints of 99 equal-mass bins, so the
# quantiles act as an equally weighted sample of the predictive distribution.
_QUANTILE_LEVELS = [(k + 0.5) / 99 for k in range(99)]
# sdm models (KumoTabular, TabICLv2, TabFM) and their constructor options.
_SDM_MODELS: dict[str, tuple[str, dict[str, Any]]] = {
    "KumoTabular": ("KumoTabular", {"size": "large"}),
    "TabICLv2": ("TabICLv2", {}),
    "TabFM": ("TabFM", {"accept_license": True}),
}


def _quantile_predictions(q: np.ndarray, levels: list[float]) -> dict[str, np.ndarray]:
    """Score a [N, Q] quantile grid as samples plus a 90% interval."""
    q = np.sort(np.asarray(q, dtype=np.float64), axis=1)  # repair rare crossings
    lv = np.asarray(levels)
    lower = np.array([np.interp(0.05, lv, row) for row in q])
    upper = np.array([np.interp(0.95, lv, row) for row in q])
    return {"samples": q, "lower": lower, "upper": upper}


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
            "house_sales",
            "openml_year",
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
        # TabPFN is paused for the 0.3.0 release run (gated weights); opt back in with
        # TORCHREGRESS_HARNESS_WITH_TABPFN=1.
        if _TABPFN_AVAILABLE and os.environ.get("TORCHREGRESS_HARNESS_WITH_TABPFN") == "1":
            methods.append("TabPFN")
        if _TABICL_AVAILABLE:
            methods.append("TabICL")
        if _SDM_AVAILABLE:
            methods += list(_SDM_MODELS)
        return methods

    def library_name(self, method: str) -> str:
        if method == "TabPFN":
            return "tabpfn"
        if method == "TabICL":
            return "tabicl"
        if method in _SDM_MODELS:
            return "structured-data-models"
        return "torchregress"

    def library_version(self, method: str) -> str:
        from importlib.metadata import PackageNotFoundError, distribution

        try:
            dist = distribution(self.library_name(method))
        except PackageNotFoundError:
            return "not installed"
        url = dist.read_text("direct_url.json") or ""
        if '"commit_id"' in url:
            import json

            commit = json.loads(url).get("vcs_info", {}).get("commit_id", "")
            return f"{dist.version} (git {commit[:10]})"
        return dist.version

    def method_notes(self, method: str) -> str:
        notes = {
            "GaussianNLL": "2-head MLP + BetaNLLLoss(beta=0.5) → mean & std",
            "MDN": (
                f"{self._MDN_COMPONENTS}-component MixtureDensityLoss "
                "→ mean & std via predict_mean_std()"
            ),
            "TabPFN": (
                "TabPFNRegressor, 99 predictive quantiles.  "
                "TabPFN 9 weights are gated: needs HF_TOKEN with the licence accepted"
            ),
            "TabICL": "TabICLRegressor, 99 predictive quantiles",
            "KumoTabular": "sdm KumoTabular (large, NVIDIA 2026), 999 predictive quantiles",
            "TabICLv2": "sdm TabICLv2 (ICML 2026), 999 predictive quantiles",
            "TabFM": (
                "sdm TabFM (Google 2026), 999 predictive quantiles; "
                "weights are non-commercial (accept_license=True)"
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

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"].reshape(-1)
        in_dim = X_train_np.shape[1]
        hidden = _hidden_size(in_dim)

        # --- torchregress: GaussianNLL ---
        if method == "GaussianNLL":
            X = torch.from_numpy(X_train_np).float().to(current_device())
            y = torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1)

            model = _MLP(in_dim, hidden, 2)  # mean, log_variance
            loss_fn = BetaNLLLoss(beta=0.5)
            _train_one(model, loss_fn, X, y, config)
            model.eval()

            return {"type": "gaussian_nll", "model": model}

        # --- torchregress: MDN ---
        elif method == "MDN":
            X = torch.from_numpy(X_train_np).float().to(current_device())
            y = torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1)

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

        # --- sdm: KumoTabular / TabICLv2 / TabFM (in-context, no training) ---
        elif method in _SDM_MODELS:
            import sdm.models
            from sdm import TableTensor

            cls_name, kwargs = _SDM_MODELS[method]
            device = current_device()
            model = getattr(sdm.models, cls_name)(task="regression", device=device, **kwargs)
            model.eval()
            x = TableTensor.from_tensor(torch.as_tensor(X_train_np, dtype=torch.float32))
            y = TableTensor.from_tensor(
                torch.as_tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
            )
            model.fit(
                x.to(device), y.to(device), generator=torch.Generator().manual_seed(config.seed)
            )
            return {"type": "sdm", "model": model}

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
                out = model(torch.from_numpy(X_test).float().to(current_device()))
                mean, std = loss_fn.predict_mean_std(out)
            return {
                "mean": mean.squeeze(-1).detach().cpu().numpy(),
                "std": std.squeeze(-1).detach().cpu().numpy(),
            }

        # --- TabPFN: predictive quantiles ---
        elif model_type == "tabpfn":
            est = model_state["model"]
            q = est.predict(X_test, output_type="quantiles", quantiles=_QUANTILE_LEVELS)
            return _quantile_predictions(
                np.stack([np.asarray(c) for c in q], axis=1), _QUANTILE_LEVELS
            )

        # --- TabICL: predictive quantiles ---
        elif model_type == "tabicl":
            est = model_state["model"]
            q = np.asarray(est.predict(X_test, output_type="quantiles", alphas=_QUANTILE_LEVELS))
            return _quantile_predictions(q.reshape(len(X_test), -1), _QUANTILE_LEVELS)

        # --- sdm: 999 quantiles q001..q999 ---
        elif model_type == "sdm":
            from sdm import TableTensor

            model = model_state["model"]
            x = TableTensor.from_tensor(torch.as_tensor(X_test, dtype=torch.float32))
            with torch.no_grad():
                out = model.predict(x.to(current_device()))
            q = out.numerical.detach().float().cpu().numpy().reshape(len(X_test), -1)
            levels = [(k + 1) / (q.shape[1] + 1) for k in range(q.shape[1])]
            return _quantile_predictions(q, levels)

        raise ValueError(f"Unknown model type: {model_type}")
