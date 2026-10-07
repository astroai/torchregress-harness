"""Label-shift + covariate-shift suite — importance weighting under shift.

LABEL-shift protocol: an MLP point model trains on the source (train split).
The TARGET is a label-shifted resample of test (P(select) logistic in
standardized y, known true weights). Rows: Unweighted, BBSE, Oracle,
DomainClassifier. Metrics: weight-recovery RMSE + weighted MAE.

``torchregress.test_time`` label-shift rows bin the target (quantile bins of
the source targets, ``gaussian_bin_edges_from_targets``), turn the Gaussian
predictions into bin probabilities (``gaussian_bin_probabilities``) and
estimate the target bin prior: ``TT_BBSE`` with
``estimate_target_prior_bbse`` (confusion-matrix inversion, Lipton et al.
2018), ``TT_EM`` with ``estimate_target_prior_em`` (Saerens et al. EM).
Importance weights are prior ratios; ``PriorTV`` is the total-variation
distance to the empirical target bin prior (the oracle). The
``ShiftFactoredTransport`` rows transport the Gaussian predictive to the
unlabeled shifted target (``ShiftFactoredPredictiveTransport``; the
``+Conformal`` variant calibrates on the last 20% of the training rows) and
are scored on the shifted target rows (TargetMAE, NLL, 90% coverage/width).

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
# Target bins for the test_time BBSE/EM rows (BBSE needs a well-conditioned confusion).
_N_SHIFT_BINS = 5
_Z90 = 1.6448536269514722
_TRANSPORT_METHODS = ("ShiftFactoredTransport", "ShiftFactoredTransport+Conformal")
_CAL_METHODS = ("SplitCP", "WeightedCP", "ShiftFactoredTransport+Conformal")


def _finite_interval_stats(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Coverage over all rows; width over finite intervals + infinite-interval fraction.

    Conformal calibration with too few points returns infinite bounds (the
    finite-sample quantile does not exist): those rows cover, but would make
    the mean width infinite.
    """
    finite = np.isfinite(lo) & np.isfinite(hi)
    return {
        "ShiftCoverage": float(np.mean((y >= lo) & (y <= hi))),
        "ShiftWidth": float(np.mean(hi[finite] - lo[finite])) if finite.any() else float("nan"),
        "InfiniteIntervalFrac": float(np.mean(~finite)),
    }


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin of each value for ``edges`` (values beyond the outer edges go to the end bins)."""
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def _bin_prior(values: np.ndarray, edges: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    counts = np.bincount(_bin_index(values, edges), minlength=len(edges) - 1).astype(np.float64)
    counts = np.clip(counts, eps, None)
    return counts / counts.sum()


def prior_ratio_weights(
    method: str,
    *,
    y_train: np.ndarray,
    p_src: np.ndarray,
    p_tgt: np.ndarray,
    y_tgt: np.ndarray,
    n_bins: int = _N_SHIFT_BINS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-row importance weights (mean 1) from a test_time target-prior estimate.

    All arrays in the same (native) target units.  ``y_tgt`` only indexes
    the estimated prior ratio at each target row and builds the oracle prior
    for ``PriorTV``; the estimators see source labels and target predictions.
    """
    from torchregress.test_time import (
        LabelShiftEMConfig,
        estimate_target_prior_bbse,
        estimate_target_prior_em,
        gaussian_bin_edges_from_targets,
        gaussian_bin_probabilities,
    )

    edges = gaussian_bin_edges_from_targets(y_train, n_bins)
    sigma = np.full_like(p_src, max(float(np.std(y_train - p_src)), 1e-6))
    probs_src = gaussian_bin_probabilities(p_src, sigma, edges)
    probs_tgt = gaussian_bin_probabilities(p_tgt, np.full_like(p_tgt, sigma[0]), edges)
    source_prior = _bin_prior(y_train, edges)
    extra: dict[str, float] = {}
    if method == "TT_BBSE":
        q = estimate_target_prior_bbse(probs_src, _bin_index(y_train, edges), probs_tgt)
        q = np.clip(q, 0.0, None)  # BBSE is an unconstrained moment estimate
        q = q / max(q.sum(), 1e-12)
    elif method == "TT_EM":
        est = estimate_target_prior_em(
            probs_tgt, source_prior=source_prior, config=LabelShiftEMConfig()
        )
        q = est.target_prior
        extra = {"EMIterations": float(est.iterations), "EMConverged": float(est.converged)}
    else:
        raise ValueError(f"Unknown test_time prior method: {method}")
    true_prior = _bin_prior(y_tgt, edges)
    extra["PriorTV"] = float(0.5 * np.abs(q - true_prior).sum())
    w = q[_bin_index(y_tgt, edges)] / source_prior[_bin_index(y_tgt, edges)]
    return w / max(float(w.mean()), 1e-12), extra


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
            "LibDomainClassifier",
            "GaussShiftCorrect",
            "SplitCP",
            "WeightedCP",
            "TT_BBSE",
            "TT_EM",
            *_TRANSPORT_METHODS,
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
            "LibDomainClassifier": "torchregress MLP discriminator density-ratio weights",
            "GaussShiftCorrect": "Gaussian predictions corrected for label shift (EM priors)",
            "SplitCP": "split conformal on source cal — undercovers shifted target",
            "WeightedCP": "WeightedConformalRegressionAdapter (density-ratio CP)",
            "TT_BBSE": "test_time estimate_target_prior_bbse on 5 target bins (prior-ratio weights)",
            "TT_EM": "test_time estimate_target_prior_em on 5 target bins (prior-ratio weights)",
            "ShiftFactoredTransport": "ShiftFactoredPredictiveTransport of the Gaussian predictive",
            "ShiftFactoredTransport+Conformal": "transport + conformal on the last 20% of train rows",
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
        if method in _CAL_METHODS:
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
            out["resid_cal"] = np.asarray(model_state["resid_cal"], dtype=np.float64).reshape(-1)
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
        if method in _TRANSPORT_METHODS:
            # The label-shifted target is drawn from y_test, which predict()
            # does not see: transport runs in evaluate() on these features.
            out["_X_test"] = np.asarray(X_test, dtype=np.float64)
            out["_X_train"] = np.asarray(model_state["X_train"], dtype=np.float64)
            if method.endswith("+Conformal"):
                out["_y_cal"] = np.asarray(model_state["y_cal"], dtype=np.float64)
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
        elif method == "LibDomainClassifier":
            from torchregress.test_time.shift_weights import (
                DomainClassifierRatioEstimator,
            )

            X_te = np.asarray(X_test, dtype=np.float32)
            X_tr = np.asarray(model_state["X_train"], dtype=np.float32)
            # Keep the full-batch discriminator bounded in smoke and full runs.
            n_tr = min(len(X_tr), 2_000)
            n_te = min(len(X_te), 2_000)
            estimator = DomainClassifierRatioEstimator(
                hidden=(32,), epochs=50, lr=1e-3, clip=20.0
            ).fit(torch.from_numpy(X_tr[:n_tr]), torch.from_numpy(X_te[:n_te]))
            w_dom = estimator.weights_for(torch.from_numpy(X_te)).cpu().numpy()
            out["w_lib_dom"] = w_dom / max(w_dom.mean(), 1e-12)
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
        elif method == "LibDomainClassifier":
            w_all = np.asarray(predictions["w_lib_dom"], dtype=np.float64).reshape(-1)
            w_est = w_all[tgt_idx]
            w_est = w_est / max(w_est.mean(), 1e-12)
        elif method == "GaussShiftCorrect":
            from torchregress.test_time.label_shift import (
                correct_gaussian_predictions_for_label_shift,
            )

            y_train = np.asarray(predictions["y_train"]).reshape(-1)
            p_src = np.asarray(predictions["p_src"]).reshape(-1)
            if y_scaler is not None:
                y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
                p_src = y_scaler.inverse_transform(p_src.reshape(-1, 1)).ravel()
            source_std = max(float(np.std(y_train - p_src)), 1e-6)
            corrected_mean, corrected_std, metadata = correct_gaussian_predictions_for_label_shift(
                mean=y_pred,
                std=np.full_like(y_pred, source_std),
                source_targets=y_train,
            )
            y_shift = y_true[tgt_idx]
            mean_shift = corrected_mean[tgt_idx]
            std_shift = corrected_std[tgt_idx]
            z = 1.6448536269514722
            return {
                "TargetMAE": float(np.mean(np.abs(mean_shift - y_shift))),
                "RMSE": float(np.sqrt(np.mean((mean_shift - y_shift) ** 2))),
                "ShiftCoverage": float(np.mean(np.abs(y_shift - mean_shift) <= z * std_shift)),
                "TargetStd": float(np.mean(std_shift)),
                "EMIterations": float(metadata["estimate_iterations"]),
            }
        elif method in ("SplitCP", "WeightedCP"):
            from torchregress.test_time.ot_conformal import (
                OptimalTransportCoverageGap,
            )

            cov_idx = np.asarray(predictions["cov_tgt_idx"]).reshape(-1).astype(int)
            lo = np.asarray(predictions["lower"]).reshape(-1)[cov_idx]
            hi = np.asarray(predictions["upper"]).reshape(-1)[cov_idx]
            yt = y_true[cov_idx]
            # OT coverage-gap diagnostic: CDF mismatch between calibration
            # residuals and shifted-target residuals (what WeightedCP fixes).
            # predictions["y_pred"] is still standardized here (only locals
            # were inverse-transformed above); standardize y_true to match,
            # or compare native when no scaler exists (skip-flag datasets).
            gap: dict[str, float] = {}
            try:
                ps = np.asarray(predictions["y_pred"]).reshape(-1)[cov_idx]
                if y_scaler is not None:
                    _mu = float(np.asarray(y_scaler.mean_).reshape(-1)[0])
                    _sc = float(np.asarray(y_scaler.scale_).reshape(-1)[0])
                    resid_tgt = np.abs(ps - (yt - _mu) / max(_sc, 1e-12))
                else:
                    resid_tgt = np.abs(ps - yt)
                ot = OptimalTransportCoverageGap().estimate(
                    calibration_scores=torch.from_numpy(
                        np.asarray(predictions["resid_cal"]).astype(np.float32)
                    ),
                    target_score_summary=torch.from_numpy(resid_tgt.astype(np.float32)),
                )
                gap = {
                    "OTL2Gap": float(ot["l2_cdf_gap"]),
                    "OTKS": float(ot["ks_max_abs"]),
                }
            except Exception:
                pass
            return {
                **_finite_interval_stats(lo, hi, yt),
                "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
                **gap,
            }
        elif method in ("TT_BBSE", "TT_EM"):
            y_train = np.asarray(predictions["y_train"]).reshape(-1)
            p_src = np.asarray(predictions["p_src"]).reshape(-1)
            if y_scaler is not None:
                y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
                p_src = y_scaler.inverse_transform(p_src.reshape(-1, 1)).ravel()
            w_est, extra = prior_ratio_weights(
                method,
                y_train=y_train,
                p_src=p_src,
                p_tgt=y_pred[tgt_idx],
                y_tgt=y_true[tgt_idx],
            )
            w_mae = float(np.sum(w_est * err) / max(w_est.sum(), 1e-12))
            return {
                "WeightRMSE": float(np.sqrt(np.mean((w_est - w_true) ** 2))),
                "TargetMAE": w_mae,
                "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
                **extra,
            }
        elif method in _TRANSPORT_METHODS:
            return self._evaluate_transport(method, predictions, y_pred, y_true, tgt_idx)
        else:
            raise ValueError(f"Unknown method: {method}")
        w_mae = float(np.sum(w_est * err) / max(w_est.sum(), 1e-12))
        return {
            "WeightRMSE": float(np.sqrt(np.mean((w_est - w_true) ** 2))),
            "TargetMAE": w_mae,
            "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        }

    # ------------------------------------------------------------------
    # ShiftFactoredPredictiveTransport on the label-shifted target
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_transport(
        method: str,
        predictions: dict[str, np.ndarray],
        y_pred: np.ndarray,
        y_true: np.ndarray,
        tgt_idx: np.ndarray,
    ) -> dict[str, float]:
        """Transport the Gaussian predictive to the unlabeled shifted target (native units)."""
        from torchregress.prediction import PredictiveBatch
        from torchregress.test_time import (
            ShiftFactoredPredictiveTransport,
            ShiftFactoredTransportConfig,
        )

        from harness.metrics import compute_distribution_metrics

        y_scaler = predictions.get("_y_scaler")

        def native(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            if y_scaler is None:
                return v
            return y_scaler.inverse_transform(v.reshape(-1, 1)).ravel()

        y_train = native(predictions["y_train"])
        p_src = native(predictions["p_src"])
        sigma = max(float(np.std(y_train - p_src)), 1e-6)

        def batch(mean: np.ndarray) -> PredictiveBatch:
            m = torch.from_numpy(np.asarray(mean, dtype=np.float32))
            return PredictiveBatch(point=m, mean=m, std=torch.full_like(m, sigma))

        X_tr = torch.from_numpy(np.asarray(predictions["_X_train"], dtype=np.float32))
        X_tgt = torch.from_numpy(np.asarray(predictions["_X_test"], dtype=np.float32)[tgt_idx])
        # alpha = 0.1: the harness's 90% interval level (also the library default).
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(alpha=0.1)
        ).fit_source(
            batch(p_src),
            torch.from_numpy(y_train.astype(np.float32)),
            source_inputs=X_tr,
        )
        conformal = method.endswith("+Conformal")
        if conformal:
            # The calibration rows are the last n_cal training rows, so their
            # predictive is the tail of the source predictions.
            n_cal = len(np.asarray(predictions["_y_cal"]).reshape(-1))
            transport.calibrate_target(
                batch(p_src[-n_cal:]),
                torch.from_numpy(native(predictions["_y_cal"]).astype(np.float32)),
            )
        out = transport.predict(
            target_predictions=batch(y_pred[tgt_idx]),
            target_inputs=X_tgt,
            apply_conformal=conformal,
        )
        mean = np.asarray(out.mean, dtype=np.float64).reshape(-1)
        std = np.asarray(out.std, dtype=np.float64).reshape(-1)
        meta = out.extra or {}
        if conformal and "interval_lower" in meta:
            lo = np.asarray(meta["interval_lower"], dtype=np.float64)
            hi = np.asarray(meta["interval_upper"], dtype=np.float64)
        else:
            lo, hi = mean - _Z90 * std, mean + _Z90 * std
        y_shift = y_true[tgt_idx]
        dist = compute_distribution_metrics(mean, std, y_shift)
        return {
            "TargetMAE": float(np.mean(np.abs(mean - y_shift))),
            "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
            "NLL": dist["NLL"],
            "CRPS": dist["CRPS"],
            **_finite_interval_stats(lo, hi, y_shift),
            "PriorTV": float(meta.get("prior_source_target_tv", float("nan"))),
            "TransportApplied": float(bool(meta.get("transport_applied", False))),
            "EMConverged": float(bool(meta.get("estimate_converged", False))),
        }
