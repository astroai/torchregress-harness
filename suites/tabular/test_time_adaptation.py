"""Unlabeled test-time feature and predictive adaptation under covariate shift.

Protocol: the test pool is deterministically resampled into a
covariate-shifted evaluation set (logistic selection in a seeded random
feature direction, mirroring the label-shift suite's covariate protocol).
Unlabeled methods adapt using ONLY source labels plus the shifted target
features; evaluation scores ONLY the shifted rows. Calibration-needing
methods train on the first 80% of source rows and reserve the last 20% as a
disjoint calibration split. No target labels are consumed by any method.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchregress.losses import GaussianNLLLoss
from torchregress.prediction import PredictiveBatch

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _train_one

_COV_SHIFT_STRENGTH = 1.0

_CALIBRATION_METHODS = frozenset(
    {
        "ShiftFactoredTransport+Conformal",
        "JointTTA",
        "JointTTA_OTScore",
        "JointTTA_LabelShiftEM",
        "SemiConformal",
    }
)


def _covariate_shift_target_idx(
    X_test: np.ndarray, seed: int, strength: float = _COV_SHIFT_STRENGTH
) -> np.ndarray:
    """Deterministic covariate-shifted resample indices for the test pool."""
    X = np.asarray(X_test, dtype=np.float64)
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    rng = np.random.default_rng(seed + 999)
    w = rng.normal(0, 1, Xs.shape[1])
    w /= np.linalg.norm(w) + 1e-12
    sel_prob = 1.0 / (1.0 + np.exp(-strength * (Xs @ w)))
    sel_prob = sel_prob / sel_prob.sum()
    n = len(X)
    return rng.choice(n, size=n, replace=True, p=sel_prob).astype(np.int64)


class TestTimeAdaptationSuite(HarnessRunner):
    """Test-time methods evaluated on the covariate-shifted test distribution."""

    @property
    def name(self) -> str:
        return "test_time_adaptation"

    @property
    def description(self) -> str:
        return "Unlabeled feature/predictive TTA under synthetic covariate shift"

    @property
    def supported_datasets(self) -> list[str]:
        return ["diabetes", "california_housing", "folktables_income"]

    def methods(self) -> list[str]:
        return [
            "Static",
            "FeatureNorm",
            "SubspaceAlign",
            "ShiftInflator",
            "ShiftFactoredTransport",
            "ShiftFactoredTransport+Conformal",
            "JointTTA",
            "JointTTA_OTScore",
            "JointTTA_LabelShiftEM",
            "SemiConformal",
        ]

    def method_notes(self, method: str) -> str:
        return {
            "Static": "Gaussian MLP without test-time adaptation (shifted-target ablation)",
            "FeatureNorm": "FeatureStatNormalizer source-to-shifted-target moment alignment",
            "SubspaceAlign": "WeightedSubspaceMomentAligner using source target relevance",
            "ShiftInflator": "RepresentationShiftInflator scales predictive std by target novelty",
            "ShiftFactoredTransport": "unlabeled predictive prior transport + shift inflation",
            "ShiftFactoredTransport+Conformal": "shift-factored transport + source-calibrated conformal",
            "JointTTA": "weighted joint conformal TTA (domain-clf weights); pseudo-label rounds disabled",
            "JointTTA_OTScore": "weighted joint conformal TTA (OT score weights); pseudo-label rounds disabled",
            "JointTTA_LabelShiftEM": "weighted joint conformal TTA (label-shift EM weights); pseudo-label rounds disabled",
            "SemiConformal": (
                "SemiConformalCalibrator: weighted split conformal on normalized residuals "
                "|y - mu| / sigma of the static model, w(x) from a logistic domain classifier "
                "(source vs unlabeled shifted target features, p clipped to [0.01, 0.99]); "
                "Coverage_Unweighted is the same calibrator without weights"
            ),
        }.get(method, "")

    def fit(
        self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        X = np.asarray(splits["X_train"], dtype=np.float32)
        y = np.asarray(splits["y_train"], dtype=np.float32).reshape(-1)
        state: dict[str, Any] = {}
        if method in _CALIBRATION_METHODS:
            # Disjoint calibration split: train on the first 80%, calibrate
            # on the last 20% so weighted-conformal coverage is honest.
            n_cal = max(16, len(X) // 5)
            X_tr, y_tr = X[:-n_cal], y[:-n_cal]
            state["X_cal"] = X[-n_cal:]
            state["y_cal"] = y[-n_cal:]
        else:
            X_tr, y_tr = X, y
        model = _MLP(X_tr.shape[1], _hidden_size(X_tr.shape[1]), out_dim=2)
        _train_one(
            model,
            GaussianNLLLoss(),
            torch.from_numpy(X_tr).to(current_device()),
            torch.from_numpy(y_tr).to(current_device()).unsqueeze(1),
            config,
            transform_output=lambda raw: (raw[:, :1], raw[:, 1:]),
        )
        state.update({"model": model, "X_train": X, "y_train": y})
        return state

    @staticmethod
    def _batch(model: torch.nn.Module, X: np.ndarray) -> PredictiveBatch:
        with torch.no_grad():
            raw = model(torch.from_numpy(np.asarray(X, dtype=np.float32)).to(current_device()))
        mean = raw[:, 0].detach().cpu()
        std = torch.exp(0.5 * raw[:, 1]).clamp_min(1e-6).detach().cpu()
        return PredictiveBatch(point=mean, mean=mean, std=std)

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        X_source = np.asarray(model_state["X_train"], dtype=np.float64)
        X_pool = np.asarray(X_test, dtype=np.float64)
        # Shifted evaluation set, fixed by seed so every method sees the
        # identical target distribution. Unlabeled adaptation consumes ONLY
        # these shifted features; predictions are returned on the same rows
        # and evaluate() subsets y_test with the stashed indices.
        tgt_idx = _covariate_shift_target_idx(X_pool, config.seed)
        X_target = X_pool[tgt_idx]
        model = model_state["model"]
        extra: dict[str, float] = {}
        out_idx = tgt_idx.astype(np.float64)

        def _shift_severity() -> dict[str, float]:
            try:
                from torchregress.metrics import ood_metrics_report as _ood_report

                rep = _ood_report(
                    x_test=X_target,
                    x_reference=X_source[:2000],
                    mean=X_source.mean(axis=0),
                    cov=np.cov(X_source.T) + 1e-6 * np.eye(X_source.shape[1]),
                )
                sev: dict[str, float] = {}
                for _k, _v in rep.items():
                    try:
                        sev[f"Shift_{_k}"] = float(np.asarray(_v, dtype=np.float64).mean())
                    except Exception:
                        continue
                return sev
            except Exception:
                return {}

        if method == "FeatureNorm":
            from torchregress.test_time import FeatureStatNormalizer

            normalizer = FeatureStatNormalizer().fit(X_source)
            X_eval = normalizer.transform(X_target)
            batch = self._batch(model, X_eval)
            mean = np.asarray(batch.mean, dtype=np.float64).reshape(-1)
            std = np.asarray(batch.std, dtype=np.float64).reshape(-1)
            return self._prediction_dict(mean, std, {**extra, **_shift_severity()}, tgt_idx=out_idx)
        elif method == "SubspaceAlign":
            from torchregress.test_time import WeightedSubspaceMomentAligner

            X_eval = (
                WeightedSubspaceMomentAligner()
                .fit(X_source, np.asarray(model_state["y_train"]))
                .transform(X_target)
            )
            batch = self._batch(model, X_eval)
            mean = np.asarray(batch.mean, dtype=np.float64).reshape(-1)
            std = np.asarray(batch.std, dtype=np.float64).reshape(-1)
            return self._prediction_dict(mean, std, {**extra, **_shift_severity()}, tgt_idx=out_idx)
        elif method in ("ShiftFactoredTransport", "ShiftFactoredTransport+Conformal"):
            from torchregress.test_time import ShiftFactoredPredictiveTransport

            source_batch = self._batch(model, X_source)
            target_batch = self._batch(model, X_target)
            transport = ShiftFactoredPredictiveTransport().fit_source(
                source_batch,
                torch.from_numpy(np.asarray(model_state["y_train"], dtype=np.float32)),
                source_inputs=torch.from_numpy(X_source.astype(np.float32)),
            )
            if method.endswith("+Conformal"):
                cal_batch = self._batch(model, np.asarray(model_state["X_cal"], dtype=np.float64))
                transport.calibrate_target(
                    cal_batch,
                    torch.from_numpy(np.asarray(model_state["y_cal"], dtype=np.float32)),
                )
                batch = transport.predict(
                    target_predictions=target_batch,
                    target_inputs=torch.from_numpy(X_target.astype(np.float32)),
                    apply_conformal=True,
                )
            else:
                batch = transport.predict(
                    target_predictions=target_batch,
                    target_inputs=torch.from_numpy(X_target.astype(np.float32)),
                    apply_conformal=False,
                )
            mean = np.asarray(batch.mean, dtype=np.float64).reshape(-1)
            std = np.asarray(batch.std, dtype=np.float64).reshape(-1)
            meta = batch.extra or {}
            extra = {
                "PriorTV": float(meta.get("prior_source_target_tv", float("nan"))),
                "SelectedFraction": float(meta.get("prior_selection_fraction", float("nan"))),
            }
            return self._prediction_dict(mean, std, {**extra, **_shift_severity()}, tgt_idx=out_idx)
        elif method == "ShiftInflator":
            from torchregress.test_time import RepresentationShiftInflator

            batch = self._batch(model, X_target)
            mean = np.asarray(batch.mean, dtype=np.float64).reshape(-1)
            std = np.asarray(
                RepresentationShiftInflator().fit(X_source).calibrate_std(batch.std, X_target),
                dtype=np.float64,
            ).reshape(-1)
            return self._prediction_dict(mean, std, {**_shift_severity()}, tgt_idx=out_idx)
        elif method in ("JointTTA", "JointTTA_OTScore", "JointTTA_LabelShiftEM"):
            from torchregress.test_time import JointDistributionalTTA

            estimator = {
                "JointTTA": "domain_clf",
                "JointTTA_OTScore": "ot",
                "JointTTA_LabelShiftEM": "label_shift_em",
            }[method]
            X_cal = torch.from_numpy(np.asarray(model_state["X_cal"], dtype=np.float32))
            y_cal = torch.from_numpy(np.asarray(model_state["y_cal"], dtype=np.float32)).unsqueeze(
                1
            )
            joint = JointDistributionalTTA(
                weight_estimator=estimator,  # type: ignore[arg-type]
                align_features=False,
                pseudo_label_rounds=0,
            )
            result = joint.adapt_and_calibrate(
                model,
                X_cal,
                y_cal,
                torch.from_numpy(X_target.astype(np.float32)),
            )
            interval = joint.predict_intervals(
                result, torch.from_numpy(X_target.astype(np.float32))
            )

            def _to_numpy(value: Any) -> np.ndarray:
                if torch.is_tensor(value):
                    value = value.detach().cpu()
                return np.asarray(value, dtype=np.float64).reshape(-1)

            mean = _to_numpy(interval["mean"])
            lower = _to_numpy(interval["lower"])
            upper = _to_numpy(interval["upper"])
            bounds = result.diagnostics.get("coverage_bounds", (float("nan"), float("nan")))
            return {
                "y_pred": mean,
                "mean": mean,
                "std": (upper - lower) / (2.0 * 1.6448536269514722),
                "lower": lower,
                "upper": upper,
                "WeightESS": float(result.diagnostics["weight_ess"]),
                "WeightMax": float(result.diagnostics.get("weight_max", float("nan"))),
                "CoverageLo": float(bounds[0]),
                "CoverageHi": float(bounds[1]),
                "tta_tgt_idx": out_idx,
            }
        elif method == "SemiConformal":
            return self._semi_conformal(model_state, X_source, X_target, out_idx)
        else:
            batch = self._batch(model, X_target)
            mean = np.asarray(batch.mean, dtype=np.float64).reshape(-1)
            std = np.asarray(batch.std, dtype=np.float64).reshape(-1)
            return self._prediction_dict(mean, std, {**_shift_severity()}, tgt_idx=out_idx)

    def _semi_conformal(
        self,
        model_state: dict[str, Any],
        X_source: np.ndarray,
        X_target: np.ndarray,
        out_idx: np.ndarray,
    ) -> dict[str, Any]:
        """Weighted split conformal under covariate shift (Tibshirani et al. 2019).

        Shift weights ``w(x) = p_target(x) / p_source(x)`` come from a logistic
        domain classifier fitted on the source training features vs the
        unlabeled shifted target features (no target labels).
        """
        from sklearn.linear_model import LogisticRegression
        from torchregress.calibration import SemiConformalCalibrator

        model = model_state["model"]
        X_cal = np.asarray(model_state["X_cal"], dtype=np.float64)
        y_cal = np.asarray(model_state["y_cal"], dtype=np.float64).reshape(-1)
        domain = LogisticRegression(max_iter=1000).fit(
            np.vstack([X_source, X_target]),
            np.r_[np.zeros(len(X_source)), np.ones(len(X_target))],
        )

        def _weights(X: np.ndarray) -> np.ndarray:
            p = np.clip(domain.predict_proba(X)[:, 1], 0.01, 0.99)
            return p / (1.0 - p) * (len(X_source) / len(X_target))

        w_cal, w_tgt = _weights(X_cal), _weights(X_target)
        cal = self._batch(model, X_cal)
        tgt = self._batch(model, X_target)
        mu_cal = np.asarray(cal.mean, dtype=np.float64).reshape(-1)
        sd_cal = np.asarray(cal.std, dtype=np.float64).reshape(-1)
        mean = np.asarray(tgt.mean, dtype=np.float64).reshape(-1)
        std = np.asarray(tgt.std, dtype=np.float64).reshape(-1)
        scores = torch.from_numpy(np.abs(y_cal - mu_cal) / sd_cal)

        weighted = SemiConformalCalibrator().fit(scores, torch.from_numpy(w_cal))
        # Per-target-point thresholds on the normalized score (+inf where the
        # weighted calibration mass cannot reach the level).
        q_w = weighted.compute_thresholds(torch.from_numpy(w_tgt), alpha=0.1)
        q_w_np = np.asarray(q_w, dtype=np.float64).reshape(-1)
        q_u = SemiConformalCalibrator().fit(scores).compute_thresholds(0.0, alpha=0.1)
        finite = np.isfinite(q_w_np)
        # Same diagnostics as the JointTTA rows: ESS and max of the normalized
        # calibration weights.
        w_norm = w_cal / w_cal.sum()
        return {
            "y_pred": mean,
            "mean": mean,
            "lower": mean - q_w_np * std,
            "upper": mean + q_w_np * std,
            "lower_unweighted": mean - float(q_u) * std,
            "upper_unweighted": mean + float(q_u) * std,
            "WeightESS": float(1.0 / np.sum(w_norm**2)),
            "WeightMax": float(w_norm.max()),
            "InfiniteFraction": float(1.0 - finite.mean()),
            "ThresholdRatio": float(np.median(q_w_np[finite]) / float(q_u))
            if finite.any()
            else float("inf"),
            "tta_tgt_idx": out_idx,
        }

    @staticmethod
    def _prediction_dict(
        mean: np.ndarray,
        std: np.ndarray,
        extra: dict[str, float] | None = None,
        tgt_idx: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | float]:
        z = 1.6448536269514722
        out: dict[str, np.ndarray | float] = {
            "y_pred": mean,
            "mean": mean,
            "std": std,
            "lower": mean - z * std,
            "upper": mean + z * std,
            **(extra or {}),
        }
        if tgt_idx is not None:
            out["tta_tgt_idx"] = np.asarray(tgt_idx).reshape(-1)
        return out

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        # Predictions are already on the shifted evaluation rows (row i is
        # X_pool[tgt_idx[i]]), so only the targets are subset with the stashed
        # indices; indexing the predictions again would score row i against
        # the wrong target.
        y_full = np.asarray(y_test).reshape(-1)
        idx = np.asarray(predictions.get("tta_tgt_idx", np.arange(len(y_full)))).reshape(-1)
        idx = np.clip(idx.astype(np.int64), 0, max(len(y_full) - 1, 0))
        y_tgt = y_full[idx]
        metrics = super().evaluate(method, predictions, y_tgt, config)
        if "lower_unweighted" in predictions:
            lo_u = np.asarray(predictions["lower_unweighted"], dtype=np.float64).reshape(-1)
            hi_u = np.asarray(predictions["upper_unweighted"], dtype=np.float64).reshape(-1)
            metrics["Coverage_Unweighted"] = float(np.mean((y_tgt >= lo_u) & (y_tgt <= hi_u)))
        for key in (
            "PriorTV",
            "SelectedFraction",
            "WeightESS",
            "WeightMax",
            "CoverageLo",
            "CoverageHi",
            "InfiniteFraction",
            "ThresholdRatio",
        ):
            if key in predictions:
                try:
                    metrics[key] = float(np.asarray(predictions[key]).reshape(-1)[0])
                except Exception:
                    continue
        for key, value in predictions.items():
            if key.startswith("Shift_"):
                try:
                    metrics[key] = float(np.asarray(value).reshape(-1)[0])
                except Exception:
                    continue
        metrics["ShiftedRows"] = float(len(idx))
        return metrics
