"""Unlabeled test-time feature and predictive adaptation under covariate shift."""

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
            "JointTTA",
        ]

    def method_notes(self, method: str) -> str:
        return {
            "Static": "Gaussian MLP without test-time adaptation",
            "FeatureNorm": "FeatureStatNormalizer source-to-target moment alignment",
            "SubspaceAlign": "WeightedSubspaceMomentAligner using source target relevance",
            "ShiftInflator": "RepresentationShiftInflator scales predictive std by target novelty",
            "ShiftFactoredTransport": "unlabeled predictive prior transport + shift inflation",
            "JointTTA": "weighted joint conformal TTA; pseudo-label rounds disabled",
        }.get(method, "")

    def fit(
        self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        X = np.asarray(splits["X_train"], dtype=np.float32)
        y = np.asarray(splits["y_train"], dtype=np.float32).reshape(-1)
        model = _MLP(X.shape[1], _hidden_size(X.shape[1]), out_dim=2)
        _train_one(
            model,
            GaussianNLLLoss(),
            torch.from_numpy(X).to(current_device()),
            torch.from_numpy(y).to(current_device()).unsqueeze(1),
            config,
            transform_output=lambda raw: (raw[:, :1], raw[:, 1:]),
        )
        return {"model": model, "X_train": X, "y_train": y}

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
        del config
        X_source = np.asarray(model_state["X_train"], dtype=np.float64)
        X_target = np.asarray(X_test, dtype=np.float64)
        model = model_state["model"]
        extra: dict[str, float] = {}
        if method == "FeatureNorm":
            from torchregress.test_time import FeatureStatNormalizer

            X_eval = FeatureStatNormalizer().fit(X_source).transform(X_target)
        elif method == "SubspaceAlign":
            from torchregress.test_time import WeightedSubspaceMomentAligner

            X_eval = (
                WeightedSubspaceMomentAligner()
                .fit(X_source, np.asarray(model_state["y_train"]))
                .transform(X_target)
            )
        elif method == "ShiftFactoredTransport":
            from torchregress.test_time import ShiftFactoredPredictiveTransport

            source_batch = self._batch(model, X_source)
            target_batch = self._batch(model, X_target)
            transport = ShiftFactoredPredictiveTransport().fit_source(
                source_batch,
                torch.from_numpy(np.asarray(model_state["y_train"], dtype=np.float32)),
                source_inputs=torch.from_numpy(X_source.astype(np.float32)),
            )
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
            return self._prediction_dict(mean, std, extra)
        elif method == "ShiftInflator":
            from torchregress.test_time import RepresentationShiftInflator

            batch = self._batch(model, X_target)
            mean = np.asarray(batch.mean, dtype=np.float64).reshape(-1)
            std = np.asarray(
                RepresentationShiftInflator().fit(X_source).calibrate_std(batch.std, X_target),
                dtype=np.float64,
            ).reshape(-1)
            return self._prediction_dict(mean, std)
        elif method == "JointTTA":
            from torchregress.test_time import JointDistributionalTTA

            n_cal = max(16, len(X_source) // 5)
            X_cal = torch.from_numpy(X_source[-n_cal:].astype(np.float32))
            y_cal = torch.from_numpy(
                np.asarray(model_state["y_train"])[-n_cal:].astype(np.float32)
            ).unsqueeze(1)
            joint = JointDistributionalTTA(
                weight_estimator="domain_clf",
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
            mean = np.asarray(interval["mean"], dtype=np.float64).reshape(-1)
            lower = np.asarray(interval["lower"], dtype=np.float64).reshape(-1)
            upper = np.asarray(interval["upper"], dtype=np.float64).reshape(-1)
            return {
                "y_pred": mean,
                "mean": mean,
                "std": (upper - lower) / (2.0 * 1.6448536269514722),
                "lower": lower,
                "upper": upper,
                "WeightESS": float(result.diagnostics["weight_ess"]),
            }
        else:
            X_eval = X_target
        batch = self._batch(model, X_eval)
        mean = np.asarray(batch.mean, dtype=np.float64).reshape(-1)
        std = np.asarray(batch.std, dtype=np.float64).reshape(-1)
        return self._prediction_dict(mean, std, extra)

    @staticmethod
    def _prediction_dict(
        mean: np.ndarray, std: np.ndarray, extra: dict[str, float] | None = None
    ) -> dict[str, np.ndarray | float]:
        z = 1.6448536269514722
        return {
            "y_pred": mean,
            "mean": mean,
            "std": std,
            "lower": mean - z * std,
            "upper": mean + z * std,
            **(extra or {}),
        }

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        metrics = super().evaluate(method, predictions, y_test, config)
        for key in ("PriorTV", "SelectedFraction", "WeightESS"):
            if key in predictions:
                metrics[key] = float(predictions[key])
        return metrics
