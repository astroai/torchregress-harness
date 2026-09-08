"""Label-shift + covariate-shift suite — importance weighting under shift.

LABEL-shift protocol: an MLP point model trains on the source (train split).
The TARGET is a label-shifted resample of test (P(select) logistic in
standardized y, known true weights). Rows: Unweighted, BBSE, Oracle,
DomainClassifier. Metrics: weight-recovery RMSE + weighted MAE.

COVARIATE-shift protocol (WeightedCP rows): the target is resampled by
P(select) logistic in X (known shift on features). A split-CP baseline vs
the WeightedConformalRegressionAdapter (classifier density-ratio weighted
calibration). Metrics: coverage/width on the shifted target.
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
_COV_SHIFT_STRENGTH = 1.0


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


def _covariate_shift_target(
    X_test: np.ndarray, seed: int, strength: float = _COV_SHIFT_STRENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Covariate-shifted target resample + normalized true weights."""
    rng = np.random.default_rng(seed + 999)
    Xs = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-12)
    w = rng.normal(0, 1, Xs.shape[1])
    w /= np.linalg.norm(w) + 1e-12
    sel_prob = 1.0 / (1.0 + np.exp(-strength * (Xs @ w)))
    sel_prob = sel_prob / sel_prob.sum()
    n_tgt = len(X_test)
    tgt_idx = rng.choice(n_tgt, size=n_tgt, replace=True, p=sel_prob)
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
        return "Label/covariate shift: importance weights + weighted conformal"

    @property
    def supported_datasets(self) -> list[str]:
        return ["diabetes", "california_housing", "folktables_income"]

    def methods(self) -> list[str]:
        return [
            "Unweighted",
            "BBSE",
            "Oracle",
            "DomainClassifier",
            "SplitCP",
            "WeightedCP",
        ]

    def library_name(self, method: str) -> str:
        if method == "DomainClassifier":
            return "sklearn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "Unweighted": "no correction — biased target-risk baseline",
            "BBSE": "BinnedLabelShiftEstimator (EM) weights from source + target preds",
            "Oracle": "TRUE importance weights (upper bound)",
            "DomainClassifier": "sklearn domain classifier density-ratio weights (external)",
            "SplitCP": "split conformal on source cal — undercovers shifted target",
            "WeightedCP": "WeightedConformalRegressionAdapter (density-ratio CP)",
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
        state: dict[str, Any] = {
            "type": "point",
            "model": model,
            "y_train": np.asarray(y_train_np).reshape(-1),
            "p_src": p_src,
            "X_train": np.asarray(X_train_np, dtype=np.float64),
        }
        # Stash target-scale stats so CP intervals can be inverse-transformed
        # to native units in predict() (same scale discipline as BBSE above).
        _ysc = splits.get("_y_scaler")
        if _ysc is not None:
            state["y_smean"] = float(np.asarray(_ysc.mean_).reshape(-1)[0])
            state["y_sscale"] = float(np.asarray(_ysc.scale_).reshape(-1)[0])
        else:
            state["y_smean"] = 0.0
            state["y_sscale"] = 1.0
        if method in ("SplitCP", "WeightedCP"):
            # Calibration split = last 20% of train (residuals + features).
            n_cal = max(16, len(X_train_np) // 5)
            X_cal = np.asarray(X_train_np)[-n_cal:]
            y_cal = np.asarray(y_train_np).reshape(-1)[-n_cal:]
            with torch.no_grad():
                p_cal = (
                    model(torch.from_numpy(X_cal).float().to(current_device()))
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            state["X_cal"] = X_cal.astype(np.float64)
            state["y_cal"] = y_cal.astype(np.float64)
            state["p_cal"] = p_cal.astype(np.float64)
            state["resid_cal"] = np.abs(y_cal - p_cal).astype(np.float64)
        return state

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
        m = model_state["model"]
        with torch.no_grad():
            pred = (
                m(torch.from_numpy(np.asarray(X_test)).float().to(current_device()))
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
        out: dict[str, np.ndarray] = {
            "y_pred": pred,
            "y_train": np.asarray(model_state["y_train"]).reshape(-1),
            "p_src": np.asarray(model_state["p_src"]).reshape(-1),
        }
        if method in ("SplitCP", "WeightedCP"):
            # Covariate-shifted target (seeded — identical across methods).
            X_te = np.asarray(X_test, dtype=np.float64)
            cov_idx, _ = _covariate_shift_target(X_te, config.seed)
            out["cov_tgt_idx"] = cov_idx.astype(np.float64)
            smean = float(model_state.get("y_smean", 0.0))
            sscale = float(model_state.get("y_sscale", 1.0))
            if method == "SplitCP":
                q = float(np.quantile(model_state["resid_cal"], 0.9))
                out["lower"] = (pred - q).astype(np.float64) * sscale + smean
                out["upper"] = (pred + q).astype(np.float64) * sscale + smean
            else:
                from torchregress.test_time.ot_conformal import (
                    WeightedConformalRegressionAdapter,
                )

                adapter = WeightedConformalRegressionAdapter(alpha=0.1)
                adapter.calibrate(
                    torch.from_numpy(np.asarray(model_state["p_cal"]).astype(np.float32)),
                    torch.from_numpy(np.asarray(model_state["y_cal"]).astype(np.float32)),
                    torch.from_numpy(np.asarray(model_state["X_cal"]).astype(np.float32)),
                    torch.from_numpy(X_te.astype(np.float32)),
                )
                lo_t, hi_t = adapter.predict_interval(
                    torch.from_numpy(pred.astype(np.float32)),
                    torch.from_numpy(X_te.astype(np.float32)),
                )
                out["lower"] = np.asarray(lo_t, dtype=np.float64).reshape(-1) * sscale + smean
                out["upper"] = np.asarray(hi_t, dtype=np.float64).reshape(-1) * sscale + smean
        if method == "DomainClassifier":
            # Discriminative density-ratio (Bickel et al. 2009): P(target|x)
            # odds from a source-vs-target domain classifier (external).
            from sklearn.linear_model import LogisticRegression

            X_te = np.asarray(X_test, dtype=np.float64)
            X_tr = np.asarray(model_state["X_train"], dtype=np.float64)
            n_tr = min(len(X_tr), 2000)
            n_te = min(len(X_te), 2000)
            Xd = np.vstack([X_tr[:n_tr], X_te[:n_te]])
            yd = np.concatenate([np.zeros(n_tr), np.ones(n_te)]).astype(int)
            clf = LogisticRegression(max_iter=1000).fit(Xd, yd)
            p = clf.predict_proba(X_te)[:, 1]
            w_dom = (p / np.clip(1.0 - p, 1e-3, None)).astype(np.float64)
            out["w_dom"] = w_dom / max(w_dom.mean(), 1e-12)
        return out

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
            if y_scaler is not None:
                # Source quantities must live in the same (native) space as
                # the inverse-transformed target quantities above — mixing
                # standardized source with native target silently degenerates
                # EM to uniform weights.
                y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
                p_src = y_scaler.inverse_transform(p_src.reshape(-1, 1)).ravel()
            est = BinnedLabelShiftEstimator(n_bins=10, method="em")
            est.fit(y_train, p_src, y_pred[tgt_idx])
            w_est = np.asarray(est.sample_weights(y_true[tgt_idx]), dtype=np.float64).reshape(-1)
            w_est = w_est / max(w_est.mean(), 1e-12)
        elif method == "DomainClassifier":
            w_all = np.asarray(predictions["w_dom"], dtype=np.float64).reshape(-1)
            w_est = w_all[tgt_idx]
            w_est = w_est / max(w_est.mean(), 1e-12)
        elif method in ("SplitCP", "WeightedCP"):
            cov_idx = np.asarray(predictions["cov_tgt_idx"]).reshape(-1).astype(int)
            lo = np.asarray(predictions["lower"]).reshape(-1)[cov_idx]
            hi = np.asarray(predictions["upper"]).reshape(-1)[cov_idx]
            yt = y_true[cov_idx]
            return {
                "ShiftCoverage": float(np.mean((yt >= lo) & (yt <= hi))),
                "ShiftWidth": float(np.mean(hi - lo)),
                "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
            }
        else:
            raise ValueError(f"Unknown method: {method}")
        w_mae = float(np.sum(w_est * err) / max(w_est.sum(), 1e-12))
        return {
            "WeightRMSE": float(np.sqrt(np.mean((w_est - w_true) ** 2))),
            "TargetMAE": w_mae,
            "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        }
