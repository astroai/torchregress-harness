"""Ensembles suite: torchregress deep ensembles vs sklearn ensemble regressors.

Compares independently-trained deep ensembles (5 MLPs per method) against
sklearn's ``VotingRegressor``, ``StackingRegressor``, and ``BaggingRegressor``
on standard tabular datasets.  Deep ensembles provide both point predictions
and uncertainty estimates (via predictive variance across members).

Canonical scenarios
-------------------
* ``sklearn_diabetes_reference``
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
* ``sklearn_synthetic_reference``
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

Architecture
------------
Each torchregress deep ensemble trains *n_ensemble* identical-architecture
MLPs independently (different random seeds), then combines their outputs:

* ``DeepEnsemble_MSE`` → mean of point predictions → ``{"y_pred"}``
* ``DeepEnsemble_GaussianNLL`` → mixture mean & total variance → ``{"mean", "std"}``
* ``DeepEnsemble_MultiQuantile`` → mean of quantile outputs → ``{"y_pred", "lower", "upper"}``

Sklearn ensembles use the native ``VotingRegressor`` / ``StackingRegressor`` /
``BaggingRegressor`` APIs with default hyperparameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Sklearn ensembles — hard dependency (sklearn is core)
# ---------------------------------------------------------------------------
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# ---------------------------------------------------------------------------
# Torchregress losses
# ---------------------------------------------------------------------------
from torchregress.losses import MultiQuantileLoss

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import (
    _MLP,
    _hidden_size,
    _train_one,
)

# ---------------------------------------------------------------------------
# Ensemble size
# ---------------------------------------------------------------------------

_N_ENSEMBLE = 5  # number of independent MLPs per deep ensemble


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class EnsemblesSuite(HarnessRunner):
    """Deep ensemble benchmark suite.

    Compares torchregress deep ensembles (5 MLPs each) against sklearn's
    VotingRegressor, StackingRegressor, and BaggingRegressor.
    """

    @property
    def name(self) -> str:
        return "ensembles"

    @property
    def description(self) -> str:
        return "Deep ensembles (torchregress) vs Voting / Stacking / Bagging (sklearn)"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "diabetes",
            "make_regression_synthetic",
            "synthetic_hetero",
            "synthetic_outliers",
            "tabred_sberbank",
            "tabred_cooking",
            "tabred_delivery",
            "tabred_maps",
            "tabred_weather",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """sklearn canonical regression benchmarks."""
        return ["sklearn_diabetes_reference", "sklearn_synthetic_reference"]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    _TR_METHODS: list[str] = [
        "DeepEnsemble_MSE",
        "DeepEnsemble_GaussianNLL",
        "DeepEnsemble_MultiQuantile",
    ]

    _SKLEARN_METHODS: list[str] = [
        "sklearn_VotingRegressor",
        "sklearn_StackingRegressor",
        "sklearn_BaggingRegressor",
    ]

    def methods(self) -> list[str]:
        return self._TR_METHODS + self._SKLEARN_METHODS

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn_"):
            return "sklearn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        notes = {
            "DeepEnsemble_MSE": f"{_N_ENSEMBLE}× MLP + MSE → mean of point predictions",
            "DeepEnsemble_GaussianNLL": f"{_N_ENSEMBLE}× MLP + BetaNLL → mixture mean & std",
            "DeepEnsemble_MultiQuantile": f"{_N_ENSEMBLE}× MLP + MultiQuantile → mean of quantiles",
            "sklearn_VotingRegressor": "Voting(LinearRegression, RandomForest, SVR, KNeighbors)",
            "sklearn_StackingRegressor": "Stacking(LinearRegression, RandomForest, SVR, KNeighbors) → RidgeCV",
            "sklearn_BaggingRegressor": "Bagging(DecisionTreeRegressor, n=100)",
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # fit — train n_ensemble models independently
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

        # --- torchregress deep ensembles ---
        if method in self._TR_METHODS:
            from harness.wrappers import DeepEnsembleRegressor, HeteroscedasticEnsembleRegressor

            # Identify active device
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            if method == "DeepEnsemble_MSE":
                reg = DeepEnsembleRegressor(
                    ensemble_size=_N_ENSEMBLE,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    seed=config.seed,
                ).to(device)
                reg.fit(X_train_np, y_train_np)
                return {"type": "wrapper", "model": reg}

            elif method == "DeepEnsemble_GaussianNLL":
                reg = HeteroscedasticEnsembleRegressor(
                    ensemble_size=_N_ENSEMBLE,
                    beta=0.5,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    seed=config.seed,
                ).to(device)
                reg.fit(X_train_np, y_train_np)
                return {"type": "wrapper", "model": reg}

            elif method == "DeepEnsemble_MultiQuantile":
                models: list[nn.Module] = []
                for i in range(_N_ENSEMBLE):
                    member_seed = config.seed + i * 1000
                    set_harness_seed(member_seed)
                    X_t = torch.from_numpy(X_train_np).float()
                    y_t = torch.from_numpy(y_train_np).float().unsqueeze(1)
                    quantiles = [0.1, 0.5, 0.9]
                    model = _MLP(in_dim, hidden, len(quantiles))
                    loss_fn = MultiQuantileLoss(quantiles=quantiles)
                    _train_one(model, loss_fn, X_t, y_t, config)
                    model.eval()
                    models.append(model)
                return {"type": "multiquantile", "models": models, "quantiles": [0.1, 0.5, 0.9]}

            else:
                raise ValueError(f"Unknown method: {method}")

        # --- sklearn ensembles ---
        elif method == "sklearn_VotingRegressor":
            est = VotingRegressor(
                [
                    ("lr", LinearRegression()),
                    ("rf", RandomForestRegressor(n_estimators=50, random_state=config.seed)),
                    ("svr", SVR()),
                    ("knn", KNeighborsRegressor(n_neighbors=5)),
                ]
            )
            est.fit(X_train_np, y_train_np)
            return {"type": "sklearn", "model": est}

        elif method == "sklearn_StackingRegressor":
            est = StackingRegressor(
                estimators=[
                    ("lr", LinearRegression()),
                    ("rf", RandomForestRegressor(n_estimators=50, random_state=config.seed)),
                    ("svr", SVR()),
                    ("knn", KNeighborsRegressor(n_neighbors=5)),
                ],
                final_estimator=RidgeCV(),
                cv=3,
            )
            est.fit(X_train_np, y_train_np)
            return {"type": "sklearn", "model": est}

        elif method == "sklearn_BaggingRegressor":
            from sklearn.tree import DecisionTreeRegressor

            est = BaggingRegressor(
                estimator=DecisionTreeRegressor(random_state=config.seed),
                n_estimators=100,
                random_state=config.seed,
                n_jobs=1,
            )
            est.fit(X_train_np, y_train_np)
            return {"type": "sklearn", "model": est}

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict — aggregate ensemble members
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        model_type = model_state["type"]

        # --- wrapper ensembles (DeepEnsemble_MSE / DeepEnsemble_GaussianNLL) ---
        if model_type == "wrapper":
            reg = model_state["model"]
            from harness.wrappers import HeteroscedasticEnsembleRegressor

            if isinstance(reg, HeteroscedasticEnsembleRegressor):
                diag = reg._predict_diagnostics(X_test)
                return {"mean": diag["mean"], "std": diag["std"]}
            else:
                mean = reg.predict(X_test)
                return {"y_pred": mean}

        # --- torchregress: MultiQuantile → mean of quantiles ---
        elif model_type == "multiquantile":
            los = []
            meds = []
            his = []
            for m in model_state["models"]:
                with torch.no_grad():
                    out = m(torch.from_numpy(X_test).float())
                    los.append(out[:, 0].numpy().reshape(-1))
                    meds.append(out[:, 1].numpy().reshape(-1))
                    his.append(out[:, 2].numpy().reshape(-1))
            lo = np.stack(los, axis=0).mean(axis=0)
            med = np.stack(meds, axis=0).mean(axis=0)
            hi = np.stack(his, axis=0).mean(axis=0)
            return {"y_pred": med, "lower": lo, "upper": hi}

        # --- sklearn ensembles ---
        elif model_type == "sklearn":
            est = model_state["model"]
            y_pred = est.predict(X_test).reshape(-1)
            return {"y_pred": y_pred}

        raise ValueError(f"Unknown model type: {model_type}")
