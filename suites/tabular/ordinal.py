"""Ordinal regression suite.

Compares torchregress ordinal losses (OrdinalCrossEntropyLoss,
CumulativeLinkLoss) against a plain MSE baseline on synthetic ordinal data
where targets are discrete class indices 0..K-1.

Methods
------
- MSE                    : MLP + WeightedMSELoss (treats classes as continuous)
- OrdinalCrossEntropy    : MLP + OrdinalCrossEntropyLoss (K logits, softmax CE)
- CumulativeLink         : MLP + CumulativeLinkLoss (K-1 cumulative logits, BCE)
- CORAL                  : MLP + CORALLoss (torchregress, CORAL-style cumulative ordinal)
- CORN                   : MLP + corn_loss from coral-pytorch (K-1 logits)
- mord_LogisticAT        : mord.LogisticAT (all-threshold ordinal logistic)
- statsmodels_OrderedModel : statsmodels OrderedModel(distr='logit')
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _train_one
from torchregress.losses import (
    CORALLoss,
    CumulativeLinkLoss,
    OrdinalCrossEntropyLoss,
    WeightedMSELoss,
)

# Optional external ordinal regression packages
try:
    from coral_pytorch.losses import corn_loss
    _CORAL_PYTORCH_AVAILABLE = True
except ImportError:
    _CORAL_PYTORCH_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

try:
    import mord
    _MORD_AVAILABLE = True
except ImportError:
    _MORD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Thin wrapper for corn_loss so it works with _train_one (which calls
# criterion(out, y) without extra kwargs).
# ---------------------------------------------------------------------------

if _CORAL_PYTORCH_AVAILABLE:

    class _CornModule(torch.nn.Module):
        """nn.Module wrapper that bakes num_classes into corn_loss."""

        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.num_classes = num_classes

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return corn_loss(logits, targets, num_classes=self.num_classes)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class OrdinalSuite(HarnessRunner):
    """Ordinal regression benchmark suite."""

    @property
    def name(self) -> str:
        return "ordinal"

    @property
    def description(self) -> str:
        return (
            "Ordinal regression: torchregress ordinal losses vs MSE baseline "
            "on synthetic discretised targets"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["synthetic_ordinal"]

    @property
    def canonical_scenarios(self) -> list[str]:
        return []

    def methods(self) -> list[str]:
        methods = ["MSE", "OrdinalCrossEntropy", "CumulativeLink", "CORAL"]
        if _CORAL_PYTORCH_AVAILABLE:
            methods.append("CORN")
        if _MORD_AVAILABLE:
            methods.append("mord_LogisticAT")
        if _STATSMODELS_AVAILABLE:
            methods.append("statsmodels_OrderedModel")
        return methods

    def library_name(self, method: str) -> str:
        if method == "CORN":
            return "coral-pytorch"
        if method.startswith("mord"):
            return "mord"
        if method.startswith("statsmodels"):
            return "statsmodels"
        if method == "CORAL":
            return "torchregress"
        return "torchregress"

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
        y_train_np = splits["y_train"]
        n_classes = int(splits.get("n_classes", 5))

        X = torch.from_numpy(X_train_np).float()
        y = torch.from_numpy(y_train_np).long()

        in_dim = X.shape[1]
        hidden = _hidden_size(in_dim)

        # Tuned hyperparameters for neural ordinal methods:
        # wider layers (64 vs default 24) and more epochs (200 vs default 60)
        # to give them a fair shot against external linear baselines
        _ordinal_hidden = 64
        _ordinal_epochs = 200

        if method == "MSE":
            model = _MLP(in_dim, hidden, out_dim=1)
            _train_one(model, WeightedMSELoss(), X, y.float().unsqueeze(1), config)
            return {"type": "torch", "model": model}

        elif method == "OrdinalCrossEntropy":
            model = _MLP(in_dim, _ordinal_hidden, out_dim=n_classes)
            _train_one(model, OrdinalCrossEntropyLoss(), X, y, config, epochs=_ordinal_epochs)
            return {"type": "torch", "model": model, "n_classes": n_classes}

        elif method == "CumulativeLink":
            # CumulativeLinkLoss expects K-1 logits (binary classifiers for P(y > k))
            model = _MLP(in_dim, _ordinal_hidden, out_dim=n_classes - 1)
            _train_one(model, CumulativeLinkLoss(), X, y, config, epochs=_ordinal_epochs)
            return {"type": "torch", "model": model, "n_classes": n_classes}

        elif method == "CORAL":
            # CORAL: K-1 logits + CORALLoss (torchregress native)
            model = _MLP(in_dim, _ordinal_hidden, out_dim=n_classes - 1)
            _train_one(model, CORALLoss(), X, y, config, epochs=_ordinal_epochs)
            return {"type": "torch", "model": model, "n_classes": n_classes}

        elif method == "CORN":
            # CORN: K-1 logits + corn_loss (coral-pytorch)
            model = _MLP(in_dim, _ordinal_hidden, out_dim=n_classes - 1)
            _train_one(model, _CornModule(n_classes), X, y, config, epochs=_ordinal_epochs)
            return {"type": "torch", "model": model, "n_classes": n_classes}

        elif method == "mord_LogisticAT":
            model = mord.LogisticAT(alpha=1.0)
            model.fit(X_train_np, y_train_np)
            return {"type": "sklearn", "model": model}

        elif method == "statsmodels_OrderedModel":
            # OrderedModel needs no intercept — it estimates cutpoints
            ord_model = OrderedModel(y_train_np, X_train_np, distr='logit')
            result = ord_model.fit(method='bfgs', disp=False, maxiter=200)
            return {"type": "statsmodels", "model": result, "n_classes": n_classes}

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
        del config
        m = model_state["model"]
        model_type = model_state.get("type", "torch")
        n_classes = model_state.get("n_classes", 2)

        if model_type == "torch":
            with torch.no_grad():
                X = torch.from_numpy(X_test).float()
                logits = m(X)
                if logits.shape[-1] == 1:
                    y_pred = logits.squeeze(-1).round().clamp(0, n_classes - 1).long().numpy()
                elif logits.shape[-1] == n_classes - 1:
                    # CumulativeLink / CORAL / CORN: K-1 logits
                    if method == "CORAL":
                        from torchregress.utils.ordinal import ordinal_predict
                        y_pred = ordinal_predict(
                            logits, encoding="cumulative_logits", strategy="threshold"
                        ).long().numpy()
                    elif method == "CORN":
                        from coral_pytorch.dataset import corn_label_from_logits
                        y_pred = corn_label_from_logits(logits).long().numpy()
                    else:
                        probs = torch.sigmoid(logits)
                        y_pred = probs.gt(0.5).sum(dim=-1).long().numpy()
                else:
                    y_pred = logits.argmax(dim=-1).long().numpy()

        elif model_type == "sklearn":
            y_pred = np.asarray(m.predict(X_test), dtype=np.int64)

        elif model_type == "statsmodels":
            # predict() returns class probabilities; argmax gives predicted class
            probs = np.asarray(m.predict(X_test))
            y_pred = probs.argmax(axis=1).astype(np.int64)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return {"y_pred": y_pred}

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

        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)

        acc = float(np.mean(y_pred == y_true))
        mae = float(np.mean(np.abs(y_pred.astype(float) - y_true.astype(float))))
        rmse = float(np.sqrt(np.mean((y_pred.astype(float) - y_true.astype(float)) ** 2)))

        return {"Accuracy": acc, "MAE": mae, "RMSE": rmse}

    # ------------------------------------------------------------------
    # notes
    # ------------------------------------------------------------------

    def method_notes(self, method: str) -> str:
        notes: dict[str, str] = {
            "MSE": "MLP + WeightedMSELoss — treats ordinal classes as continuous (baseline)",
            "OrdinalCrossEntropy": (
                "MLP + OrdinalCrossEntropyLoss — K-class softmax cross-entropy"
            ),
            "CumulativeLink": (
                "MLP + CumulativeLinkLoss — K-1 binary classifiers on cumulative logits"
            ),
            "CORAL": (
                "MLP + CORALLoss — CORAL-style cumulative ordinal objective (torchregress)"
            ),
            "CORN": (
                "MLP + corn_loss (coral-pytorch) — conditional ordinal regression, K-1 outputs"
            ),
            "mord_LogisticAT": (
                "mord.LogisticAT(alpha=1.0) — all-threshold ordinal logistic (sklearn API)"
            ),
            "statsmodels_OrderedModel": (
                "statsmodels OrderedModel(distr='logit') — cumulative-link ordinal logistic"
            ),
        }
        return notes.get(method, "")
