"""Label-shift suite — importance weighting under target shift.

Protocol: an MLP point model trains on the source (train split). The TARGET
is a label-shifted resample of test (P(select) logistic in standardized y,
known true weights). Rows: Unweighted (biased baseline), BBSE (binned
label-shift weight estimates from source labels + target predictions), Oracle
(true weights, upper bound). Metrics: weight-recovery RMSE plus weighted MAE
on the shifted target.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchregress.losses import WeightedMSELoss

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _train_one

_SHIFT_STRENGTH = 1.5


def _shift_target(
    y_test: np.ndarray, seed: int, strength: float = _SHIFT_STRENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Label-shifted target resample + normalized true weights."""
    rng = np.random.default_rng(seed)
    ys = (y_test - y_test.mean()) / (y_test.std() + 1e-12)
    sel_prob = 1.0 / (1.0 + np.exp(-strength * ys))
    sel_prob = sel_prob / sel_prob.sum()
    n_tgt = len(y_test)
    tgt_idx = rng.choice(n_tgt, size=n_tgt, replace=True, p=sel_prob)
    # True importance weight for a selected point: p_target/p_source.
    w_true = sel_prob[tgt_idx] / (1.0 / n_tgt)
    w_true = w_true / w_true.mean()
    return tgt_idx, w_true.astype(np.float64)


class LabelShiftSuite(HarnessRunner):
    """Unweighted vs BBSE vs oracle weighting under label shift."""

    @property
    def name(self) -> str:
        return "label_shift"

    @property
    def description(self) -> str:
        return "Label shift: unweighted vs BBSE vs oracle importance weights"

    @property
    def supported_datasets(self) -> list[str]:
        return ["diabetes", "california_housing"]

    def methods(self) -> list[str]:
        return ["Unweighted", "BBSE", "Oracle"]

    def library_name(self, method: str) -> str:
        return "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "Unweighted": "no correction — biased target-risk baseline",
            "BBSE": "BinnedLabelShiftEstimator (EM) weights from source + target preds",
            "Oracle": "TRUE importance weights (upper bound)",
        }.get(method, "")

    # ------------------------------------------------------------------
    # fit — shared MLP point model (+ stashed source for BBSE)
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        del method
        set_harness_seed(config.seed)
        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"]
        in_dim = X_train_np.shape[1]
        hidden = _hidden_size(in_dim)
        model = _MLP(in_dim, hidden, out_dim=1)
        _train_one(
            model,
            WeightedMSELoss(),
            torch.from_numpy(X_train_np).float().to(current_device()),
            torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1),
            config,
        )
        model.eval()
        with torch.no_grad():
            p_src = (
                model(torch.from_numpy(X_train_np).float().to(current_device()))
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
        return {
            "type": "point",
            "model": model,
            "y_train": np.asarray(y_train_np).reshape(-1),
            "p_src": p_src,
        }

    # ------------------------------------------------------------------
    # predict — per-row preds + source passthrough. All shift logic lives
    # in evaluate(), which sees the same (possibly smoke-subsampled) y_test
    # rows predict() saw — indices always align.
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        del method, config
        m = model_state["model"]
        with torch.no_grad():
            pred = (
                m(torch.from_numpy(np.asarray(X_test)).float().to(current_device()))
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
        return {
            "y_pred": pred,
            "y_train": np.asarray(model_state["y_train"]).reshape(-1),
            "p_src": np.asarray(model_state["p_src"]).reshape(-1),
        }

    # ------------------------------------------------------------------
    # evaluate — seeded shift target + per-method weights + metrics
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        from torchregress.calibration.shift import BinnedLabelShiftEstimator

        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)
        # Native data scale (runner standardizes targets; custom evaluates
        # must inverse-transform — default-evaluate precedent).
        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        tgt_idx, w_true = _shift_target(y_true, config.seed)
        err = np.abs(y_pred[tgt_idx] - y_true[tgt_idx])
        if method == "Unweighted":
            w_est = np.ones_like(w_true)
        elif method == "Oracle":
            w_est = w_true
        elif method == "BBSE":
            y_train = np.asarray(predictions["y_train"]).reshape(-1)
            p_src = np.asarray(predictions["p_src"]).reshape(-1)
            est = BinnedLabelShiftEstimator(n_bins=10, method="em")
            est.fit(y_train, p_src, y_pred[tgt_idx])
            w_est = np.asarray(est.sample_weights(y_true[tgt_idx]), dtype=np.float64).reshape(-1)
            w_est = w_est / max(w_est.mean(), 1e-12)
        else:
            raise ValueError(f"Unknown method: {method}")
        w_mae = float(np.sum(w_est * err) / max(w_est.sum(), 1e-12))
        return {
            "WeightRMSE": float(np.sqrt(np.mean((w_est - w_true) ** 2))),
            "TargetMAE": w_mae,
            "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        }
