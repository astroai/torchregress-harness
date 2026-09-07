"""Prediction-powered inference (PPI) suite.

Compares classical CIs (labeled data only) against PPI rectified CIs that
combine a small labeled set with model predictions on unlabeled data
(Angelopoulos et al., Science 2023). Protocol: an MLP point model trains on
the train split; a seeded labeled slice of TEST plus predictions on the rest
of test feed ppi_mean_ci / ppi_quantile_ci. Metrics: model RMSE plus CI
width and coverage of the true test mean/median.
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

_N_LABELED = 64


class PPISuite(HarnessRunner):
    """Classical vs prediction-powered CIs for population moments."""

    @property
    def name(self) -> str:
        return "ppi"

    @property
    def description(self) -> str:
        return "Classical vs PPI rectified CIs (mean + median)"

    @property
    def supported_datasets(self) -> list[str]:
        return ["diabetes", "california_housing"]

    def methods(self) -> list[str]:
        return ["Classical", "PPI_Mean", "PPI_Quantile"]

    def library_name(self, method: str) -> str:
        return "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "Classical": "labeled-only mean/median CIs (no model, no PPI)",
            "PPI_Mean": "ppi_mean_ci: rectified mean + bootstrap CI",
            "PPI_Quantile": "ppi_quantile_ci (q=0.5): rectified median + CI",
        }.get(method, "")

    # ------------------------------------------------------------------
    # fit — shared MLP point model (Classical needs none, kept for parity)
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
        return {"type": "point", "model": model}

    # ------------------------------------------------------------------
    # predict — per-row model predictions (RMSE comparability)
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
        return {"y_pred": pred}

    # ------------------------------------------------------------------
    # evaluate — RMSE + CI width/coverage of the true test moment
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        from torchregress.inference.ppi import (
            PPIConfig,
            ppi_mean_ci,
            ppi_quantile_ci,
        )

        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)
        # Native data scale (custom evaluates must inverse-transform).
        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

        rng = np.random.default_rng(config.seed)
        n_lab = min(_N_LABELED, len(y_true) // 2)
        lab_idx = rng.choice(len(y_true), size=n_lab, replace=False)
        unl_mask = np.ones(len(y_true), dtype=bool)
        unl_mask[lab_idx] = False
        y_l = torch.from_numpy(y_true[lab_idx].astype(np.float64))
        p_l = torch.from_numpy(y_pred[lab_idx].astype(np.float64))
        p_u = torch.from_numpy(y_pred[unl_mask].astype(np.float64))
        cfg = PPIConfig(alpha=0.1, n_boot=500, seed=config.seed)

        out: dict[str, float] = {"RMSE": rmse}
        if method in ("Classical", "PPI_Mean"):
            true_mean = float(y_true.mean())
            if method == "Classical":
                se = float(y_l.double().std() / np.sqrt(n_lab))
                lo = float(y_l.mean()) - 1.645 * se
                hi = float(y_l.mean()) + 1.645 * se
                est = float(y_l.mean())
            else:
                res = ppi_mean_ci(y_l, p_l, p_u, config=cfg)
                est, lo, hi = res["estimate"], res["ci_lower"], res["ci_upper"]
            out.update(
                {
                    "Estimate": float(est),
                    "CIWidth": float(hi - lo),
                    "Covers": float(lo <= true_mean <= hi),
                }
            )
        if method in ("Classical", "PPI_Quantile"):
            true_med = float(np.median(y_true))
            if method == "Classical":
                qs = np.quantile(y_l.numpy(), [0.05, 0.5, 0.95])
                est, lo, hi = float(qs[1]), float(qs[0]), float(qs[2])
            else:
                res = ppi_quantile_ci(y_l, p_l, p_u, q=0.5, config=cfg)
                est, lo, hi = res["estimate"], res["ci_lower"], res["ci_upper"]
            out.update(
                {
                    "Estimate": float(est),
                    "CIWidth": float(hi - lo),
                    "Covers": float(lo <= true_med <= hi),
                }
            )
        return out
