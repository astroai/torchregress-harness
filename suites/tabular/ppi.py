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

try:
    import ppi_py as _ppi_py

    _PPI_PY_AVAILABLE = True
except ImportError:
    _PPI_PY_AVAILABLE = False

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
        return ["diabetes", "california_housing", "folktables_income"]

    def methods(self) -> list[str]:
        methods = [
            "Classical",
            "PPI_Mean",
            "PPI_Quantile",
            "PPI_CalibratedMean",
            "PPI_PPMean",
            "PPI_OLS",
        ]
        if _PPI_PY_AVAILABLE:
            methods += ["PPI_py_Mean", "PPI_py_Quantile"]
        return methods

    def library_name(self, method: str) -> str:
        if method.startswith("PPI_py_"):
            return "ppi_py"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "Classical": "labeled-only mean/median CIs (no model, no PPI)",
            "PPI_Mean": "ppi_mean_ci: rectified mean + bootstrap CI",
            "PPI_Quantile": "ppi_quantile_ci (q=0.5): rectified median + CI",
            "PPI_CalibratedMean": "ppi_calibrated_mean_ci: affine-calibrated rectifier",
            "PPI_PPMean": "ppi_pp_mean_ci: PPI++ power-tuned mean CI",
            "PPI_OLS": "ppi_ols_ci: prediction-powered OLS coefficients",
            "PPI_py_Mean": "OFFICIAL ppi_py ppi_mean_ci (PPI++ power tuning, external)",
            "PPI_py_Quantile": "OFFICIAL ppi_py ppi_quantile_ci (external)",
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
        return {"y_pred": pred, "X_test": np.asarray(X_test, dtype=np.float64)}

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
            ppi_calibrated_mean_ci,
            ppi_diagnostics,
            ppi_mean_ci,
            ppi_ols_ci,
            ppi_pp_mean_ci,
            ppi_quantile_ci,
        )

        if method.startswith("PPI_py_") and not _PPI_PY_AVAILABLE:
            raise ImportError("ppi-python not installed")

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
        # Official-package arrays for the PPI_py_* rows (same split).
        Y_official = y_true[lab_idx]
        Ph_official = y_pred[lab_idx]
        Pu_official = y_pred[unl_mask]

        out: dict[str, float] = {"RMSE": rmse}
        # Validity diagnostics ride every PPI row (same triple; cheap).
        if method.startswith("PPI_"):
            try:
                diag = ppi_diagnostics(y_l, p_l, p_u)
                out["Diag_Corr"] = float(diag["prediction_label_correlation"])
                out["Diag_ResidRMSE"] = float(diag["residual_rmse_labeled"])
                out["Diag_Shift"] = float(diag["prediction_mean_shift_unlabeled_vs_labeled"])
            except Exception:
                pass
        if method in ("Classical", "PPI_Mean", "PPI_py_Mean", "PPI_CalibratedMean", "PPI_PPMean"):
            true_mean = float(y_true.mean())
            if method == "Classical":
                se = float(y_l.double().std() / np.sqrt(n_lab))
                lo = float(y_l.mean()) - 1.645 * se
                hi = float(y_l.mean()) + 1.645 * se
                est = float(y_l.mean())
            elif method == "PPI_py_Mean":
                lo_a, hi_a = _ppi_py.ppi_mean_ci(Y_official, Ph_official, Pu_official, alpha=0.1)
                lo, hi = float(np.ravel(lo_a)[0]), float(np.ravel(hi_a)[0])
                est = float((lo + hi) / 2)
            elif method == "PPI_CalibratedMean":
                res = ppi_calibrated_mean_ci(y_l, p_l, p_u, config=cfg)
                est, lo, hi = res["estimate"], res["ci_lower"], res["ci_upper"]
            elif method == "PPI_PPMean":
                res = ppi_pp_mean_ci(y_l, p_l, p_u, alpha=0.1)
                est, lo, hi = res["estimate"], res["ci_lower"], res["ci_upper"]
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
        if method in ("Classical", "PPI_Quantile", "PPI_py_Quantile"):
            true_med = float(np.median(y_true))
            if method == "Classical":
                qs = np.quantile(y_l.numpy(), [0.05, 0.5, 0.95])
                est, lo, hi = float(qs[1]), float(qs[0]), float(qs[2])
            elif method == "PPI_py_Quantile":
                ci = _ppi_py.ppi_quantile_ci(Y_official, Ph_official, Pu_official, 0.5, alpha=0.1)
                lo, hi = float(np.ravel(ci)[0]), float(np.ravel(ci)[-1])
                est = float((lo + hi) / 2)
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
        if method == "PPI_OLS":
            # Prediction-powered OLS coefficients. Truth = full-test OLS.
            X_te = np.asarray(predictions["X_test"], dtype=np.float64)
            X_lab = X_te[lab_idx]
            X_unl = X_te[unl_mask]
            x_l = torch.from_numpy(X_lab)
            x_u = torch.from_numpy(X_unl)
            res = ppi_ols_ci(x_l, y_l, x_u, p_l, p_u, config=cfg)
            coef = np.asarray(res["coef"], dtype=np.float64).reshape(-1)
            se = np.asarray(res["se"], dtype=np.float64).reshape(-1)
            lo_o = np.asarray(res["ci_lower"], dtype=np.float64).reshape(-1)
            hi_o = np.asarray(res["ci_upper"], dtype=np.float64).reshape(-1)
            Xd = np.concatenate([X_te, np.ones((len(X_te), 1))], axis=1)
            beta_true, *_ = np.linalg.lstsq(Xd, y_true, rcond=None)
            out.update(
                {
                    "Estimate": float(coef.mean()),
                    "CIWidth": float(np.mean(hi_o - lo_o)),
                    "Covers": float(np.mean((lo_o <= beta_true) & (beta_true <= hi_o))),
                }
            )
        return out
