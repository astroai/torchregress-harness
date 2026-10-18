"""Representation-level imbalanced regression: PRIME, ConR and continuous group DRO.

Complements the existing ``imbalanced``/``distributional_bins`` coverage with a
controlled long-tailed benchmark on the harness ``stress_imbalanced`` dataset
(exponential target density).  All methods share the same MLP capacity and
training budget; PRIME and ConR add feature-level regularizers, group DRO
replaces the average risk with the worst target-regime risk.

Methods
-------
- ``MSE``: plain MLP + MSE (bulk-dominated baseline).
- ``MSE_GroupDRO``: ``GroupDROTrainer`` over four equal-width target regimes.
- ``MSE_PRIME``: MLP encoder/head + ``PrimeLoss`` proxy regularizer.
- ``MSE_ConR``: MLP encoder/head + ``ConR`` collapse regularizer.

Metrics are the standard harness point/interval metrics plus ``WorstGroupMAE``:
the largest per-target-regime MAE over the fitted regimes (tail robustness).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torchregress.algorithms import ContinuousGroupDRO, GroupDROTrainer
from torchregress.losses import ConR, PrimeLoss, conr_loss, uniform_proxy_targets

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.metrics import compute_all_metrics
from harness.runners import (
    HarnessConfig,
    HarnessRunner,
    _inverse_transform_predictions,
    _inverse_transform_y,
)
from suites._backbones import _MLP, _hidden_size

_METHODS = ["MSE", "MSE_GroupDRO", "MSE_PRIME", "MSE_ConR"]
_N_GROUPS = 4
_FEATURE_DIM = 16


def _make_encoder(in_dim: int, hidden: int) -> tuple[nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, _FEATURE_DIM),
        nn.SiLU(),
    )
    head = nn.Linear(_FEATURE_DIM, 1)
    return encoder, head


class ImbalancedRepresentationSuite(HarnessRunner):
    """PRIME/ConR/group-DRO for deep long-tailed regression."""

    @property
    def name(self) -> str:
        return "imbalanced_representation"

    @property
    def description(self) -> str:
        return (
            "PRIME / ConR / continuous group DRO vs MSE on a long-tailed target "
            "(worst-regime MAE, tail metrics)"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["stress_imbalanced", "california_housing"]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def method_notes(self, method: str) -> str:
        notes = {
            "MSE": "plain MLP + MSE (bulk-dominated baseline)",
            "MSE_GroupDRO": "GroupDROTrainer over 4 equal-width target regimes",
            "MSE_PRIME": "MLP encoder/head + PRIME proxy alignment (class-balanced)",
            "MSE_ConR": "MLP encoder/head + ConR inverse-density contrastive push",
        }
        return notes.get(method, "")

    def _epochs(self, config: HarnessConfig) -> int:
        return min(int(config.epochs), 5) if config.smoke else int(config.epochs)

    def _edges(self, y: torch.Tensor) -> torch.Tensor:
        # Uniform proxy targets are the same interior edges PRIME uses; with
        # G groups they supply the G - 1 interior bin edges.  Build them on CPU
        # (MPS has crashed on tiny cat/linspace combinations under the CLI's
        # default-device setting; the edges are metadata, not model inputs).
        y_cpu = y.detach().cpu().reshape(-1)
        low, high = float(y_cpu.min()), float(y_cpu.max())
        proxies = uniform_proxy_targets(low, high, _N_GROUPS - 1).detach().cpu()
        return torch.cat(
            [
                torch.tensor([low], dtype=torch.float32),
                proxies.to(torch.float32),
                torch.tensor([high], dtype=torch.float32),
            ]
        )

    def _fit_plain(self, X: torch.Tensor, y: torch.Tensor, config: HarnessConfig) -> nn.Module:
        model = _MLP(in_dim=int(X.shape[1]), hidden=_hidden_size(int(X.shape[1])), out_dim=1).to(
            current_device()
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        generator = torch.Generator().manual_seed(int(config.seed) + 3)
        batch_size = min(int(config.batch_size), len(X))
        for _ in range(self._epochs(config)):
            order = torch.randperm(len(X), generator=generator).to(X.device)
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                nn.functional.mse_loss(model(X[idx]), y[idx].unsqueeze(-1)).backward()
                optimizer.step()
        model.eval()
        return model

    def _fit_regularized(
        self,
        method: str,
        X: torch.Tensor,
        y: torch.Tensor,
        config: HarnessConfig,
    ) -> dict[str, Any]:
        device = current_device()
        encoder, head = _make_encoder(int(X.shape[1]), _hidden_size(int(X.shape[1])))
        encoder, head = encoder.to(device), head.to(device)
        if method == "MSE_PRIME":
            regularizer = (
                PrimeLoss(
                    feature_dim=_FEATURE_DIM,
                    n_proxies=32,
                    label_range=(float(y.min()), float(y.max())),
                    class_balanced=True,
                    seed=int(config.seed),
                )
                .fit(y)
                .to(device)
            )
        else:
            regularizer = ConR(label_threshold=0.2, push_scale=0.2).fit(y)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()) + list(regularizer.parameters()),
            lr=config.lr,
        )
        generator = torch.Generator().manual_seed(int(config.seed) + 5)
        batch_size = min(int(config.batch_size), len(X))
        for _ in range(self._epochs(config)):
            order = torch.randperm(len(X), generator=generator).to(X.device)
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                features = encoder(X[idx])
                prediction = head(features).squeeze(-1)
                mse = nn.functional.mse_loss(prediction, y[idx])
                if method == "MSE_PRIME":
                    reg = regularizer(features, y[idx])
                    loss = mse + 0.5 * reg
                else:
                    reg = regularizer(features, y[idx], prediction=prediction)
                    loss = mse + 2.0 * reg
                loss.backward()
                optimizer.step()
        encoder.eval()
        head.eval()
        return {"encoder": encoder, "head": head, "edges": self._edges(y.detach())}

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        device = current_device()
        X = torch.as_tensor(np.asarray(splits["X_train"]), dtype=torch.float32, device=device)
        y = torch.as_tensor(
            np.asarray(splits["y_train"]), dtype=torch.float32, device=device
        ).reshape(-1)
        if method == "MSE":
            return {
                "type": "plain",
                "model": self._fit_plain(X, y, config),
                "edges": self._edges(y.detach().cpu()),
            }
        if method == "MSE_GroupDRO":
            trainer = GroupDROTrainer(
                model_factory=lambda: _MLP(
                    in_dim=int(X.shape[1]),
                    hidden=_hidden_size(int(X.shape[1])),
                    out_dim=1,
                ),
                group_edges=_N_GROUPS,
                base_loss="mse",
                epochs=self._epochs(config),
                lr=config.lr,
                batch_size=min(int(config.batch_size), len(X)),
                seed=int(config.seed),
            )
            trainer.fit(X.detach().cpu(), y.detach().cpu())
            edges = self._edges(y.detach().cpu())
            return {"type": "groupdro", "trainer": trainer, "edges": edges}
        if method in {"MSE_PRIME", "MSE_ConR"}:
            state = self._fit_regularized(method, X, y, config)
            return {"type": "regularized", **state}
        raise ValueError(f"Unknown method: {method}")

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        device = current_device()
        kind = model_state["type"]
        if kind == "plain":
            model = model_state["model"]
            with torch.no_grad():
                prediction = model(
                    torch.as_tensor(np.asarray(X_test), dtype=torch.float32, device=device)
                )
            predictions: dict[str, Any] = {
                "y_pred": np.asarray(prediction.detach().cpu()).reshape(-1)
            }
        elif kind == "groupdro":
            prediction = model_state["trainer"].predict(
                torch.as_tensor(np.asarray(X_test), dtype=torch.float32)
            )
            predictions = {"y_pred": np.asarray(prediction).reshape(-1)}
        else:
            encoder = model_state["encoder"]
            head = model_state["head"]
            with torch.no_grad():
                features = encoder(
                    torch.as_tensor(np.asarray(X_test), dtype=torch.float32, device=device)
                )
                prediction = head(features).squeeze(-1)
            predictions = {"y_pred": np.asarray(prediction.detach().cpu()).reshape(-1)}
        predictions["_group_edges"] = np.asarray(model_state["edges"], dtype=np.float64)
        return predictions

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        del method, config
        edges_standardized = np.asarray(predictions.pop("_group_edges"), dtype=np.float64)
        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_true = np.asarray(_inverse_transform_y(y_test, y_scaler)).reshape(-1)
            scale = float(np.asarray(y_scaler.scale_).reshape(-1)[0])
            shift = float(np.asarray(y_scaler.mean_).reshape(-1)[0])
            edges = edges_standardized * scale + shift
        else:
            y_true = np.asarray(y_test).reshape(-1)
            edges = edges_standardized
        result = compute_all_metrics(predictions, y_true)
        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        group_mae = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (y_true >= lo) & (y_true < hi)
            if mask.sum() >= 5:
                group_mae.append(float(np.mean(np.abs(y_pred[mask] - y_true[mask]))))
        result["WorstGroupMAE"] = float(max(group_mae)) if group_mae else float("nan")
        return result


__all__ = ["ImbalancedRepresentationSuite"]
