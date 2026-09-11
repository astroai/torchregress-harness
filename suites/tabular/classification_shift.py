"""Binary classification label-shift coverage on Folktables ACS Income.

The target is an exact class-conditional bootstrap of the held-out source
test set.  This preserves ``p(x | y)`` while changing the positive prior to
0.70, providing an observable target prior for estimator error metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from harness.core import set_harness_seed
from harness.datasets import SplitsDict
from harness.runners import HarnessConfig, HarnessRunner

_TARGET_POSITIVE_PRIOR = 0.70
_DOMAIN_ROWS = 2_000


def _prior(labels: np.ndarray) -> np.ndarray:
    """Return a binary empirical class prior with both classes represented."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("label-shift evaluation requires both binary classes")
    return counts / counts.sum()


def _target_bootstrap(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample exact class counts so the target prior is known by construction."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes = [np.flatnonzero(y == cls) for cls in range(2)]
    if any(len(indices) == 0 for indices in classes):
        raise ValueError("label-shift evaluation requires both binary classes")
    n_positive = int(round(len(y) * _TARGET_POSITIVE_PRIOR))
    n_positive = min(max(1, n_positive), len(y) - 1)
    rng = np.random.default_rng(seed)
    target_idx = np.concatenate(
        [
            rng.choice(classes[0], size=len(y) - n_positive, replace=True),
            rng.choice(classes[1], size=n_positive, replace=True),
        ]
    )
    rng.shuffle(target_idx)
    target_prior = np.array([1.0 - n_positive / len(y), n_positive / len(y)], dtype=np.float64)
    return target_idx.astype(np.int64), target_prior


class ClassificationShiftSuite(HarnessRunner):
    """Supervised binary label-shift API coverage without pseudo-label training."""

    @property
    def name(self) -> str:
        return "classification_shift"

    @property
    def description(self) -> str:
        return "Binary ACSIncome label shift: EM, BBSE, posterior correction, and ratios"

    @property
    def supported_datasets(self) -> list[str]:
        return ["folktables_acs_income"]

    def methods(self) -> list[str]:
        return [
            "Uncorrected",
            "EM",
            "BBSE",
            "PosteriorAdapter",
            "EstimatedWeights",
            "DomainClassifierRatio",
            "Oracle",
        ]

    def method_notes(self, method: str) -> str:
        return {
            "Uncorrected": "source posterior without label-shift correction",
            "EM": "estimate_target_prior_em plus apply_label_shift_correction",
            "BBSE": "estimate_target_prior_bbse plus posterior correction",
            "PosteriorAdapter": "PosteriorLabelShiftAdapter estimate and transform",
            "EstimatedWeights": "estimate_label_shift_weights and its EM estimate",
            "DomainClassifierRatio": "DomainClassifierRatioEstimator feature density ratio",
            "Oracle": "known target prior correction; not an unlabeled deployment method",
        }[method]

    def fit(self, method: str, splits: SplitsDict, config: HarnessConfig) -> dict[str, Any]:
        del method
        set_harness_seed(config.seed)
        X = np.asarray(splits["X_train"], dtype=np.float64)
        y = np.asarray(splits["y_train"], dtype=np.int64).reshape(-1)
        X_fit, X_source, y_fit, y_source = train_test_split(
            X, y, test_size=0.25, random_state=config.seed, stratify=y
        )
        classifier = LogisticRegression(max_iter=500, random_state=config.seed).fit(X_fit, y_fit)
        return {
            "classifier": classifier,
            "X_source": X_source.astype(np.float32),
            "y_source": y_source,
            "source_probabilities": classifier.predict_proba(X_source),
            "source_prior": _prior(y_source),
        }

    def predict(
        self,
        method: str,
        model: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray | float]:
        del method, config
        X = np.asarray(X_test, dtype=np.float32)
        source_prior = np.asarray(model["source_prior"], dtype=np.float64)
        probabilities = np.asarray(model["classifier"].predict_proba(X), dtype=np.float64)
        return {
            "probabilities": probabilities,
            "source_probabilities": np.asarray(model["source_probabilities"], dtype=np.float64),
            "source_labels": np.asarray(model["y_source"], dtype=np.int64),
            "source_inputs": np.asarray(model["X_source"], dtype=np.float32),
            "target_inputs": X,
            "source_prior": source_prior,
        }

    def evaluate(
        self,
        method: str,
        predictions: dict[str, Any],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        from torchregress.test_time import (
            DomainClassifierRatioEstimator,
            PosteriorLabelShiftAdapter,
            apply_label_shift_correction,
            estimate_label_shift_weights,
            estimate_target_prior_bbse,
            estimate_target_prior_em,
        )

        y_source_test = np.asarray(y_test, dtype=np.int64).reshape(-1)
        target_idx, target_prior = _target_bootstrap(y_source_test, config.seed)
        y_target = y_source_test[target_idx]
        source_prior = np.asarray(predictions["source_prior"], dtype=np.float64)
        target_probabilities = np.asarray(predictions["probabilities"], dtype=np.float64)[
            target_idx
        ]
        source_probabilities = np.asarray(predictions["source_probabilities"], dtype=np.float64)
        source_labels = np.asarray(predictions["source_labels"], dtype=np.int64)

        estimated_prior = source_prior.copy()
        if method == "Uncorrected":
            corrected = target_probabilities
        elif method == "EM":
            estimate = estimate_target_prior_em(target_probabilities, source_prior=source_prior)
            estimated_prior = estimate.target_prior
            corrected = apply_label_shift_correction(
                target_probabilities, source_prior=source_prior, target_prior=estimated_prior
            )
        elif method == "BBSE":
            estimated_prior = estimate_target_prior_bbse(
                source_probabilities, source_labels, target_probabilities
            )
            corrected = apply_label_shift_correction(
                target_probabilities, source_prior=source_prior, target_prior=estimated_prior
            )
        elif method == "PosteriorAdapter":
            adapter = PosteriorLabelShiftAdapter(
                source_prior=source_prior, random_state=config.seed
            )
            estimate = adapter.estimate(target_probabilities)
            estimated_prior = estimate.target_prior
            corrected = adapter.transform(target_probabilities)
        elif method == "EstimatedWeights":
            _, estimate = estimate_label_shift_weights(source_probabilities, target_probabilities)
            estimated_prior = estimate.target_prior
            corrected = apply_label_shift_correction(
                target_probabilities,
                source_prior=estimate.source_prior,
                target_prior=estimated_prior,
            )
        elif method == "DomainClassifierRatio":
            n_source = min(_DOMAIN_ROWS, len(source_probabilities))
            n_target = min(_DOMAIN_ROWS, len(target_probabilities))
            estimator = DomainClassifierRatioEstimator(hidden=(32,), epochs=50, lr=1e-3, clip=20.0)
            source_inputs = np.asarray(predictions["source_inputs"], dtype=np.float32)
            target_inputs = np.asarray(predictions["target_inputs"], dtype=np.float32)[target_idx]
            estimator.fit(
                torch.from_numpy(source_inputs[:n_source]),
                torch.from_numpy(target_inputs[:n_target]),
            )
            estimator.weights_for(torch.from_numpy(target_inputs))
            corrected = target_probabilities
            source_weights = estimator.weights_for(torch.from_numpy(source_inputs)).cpu().numpy()
            estimated_prior = np.bincount(
                source_labels, weights=source_weights, minlength=2
            ).astype(np.float64)
            estimated_prior /= estimated_prior.sum()
        elif method == "Oracle":
            estimated_prior = target_prior
            corrected = apply_label_shift_correction(
                target_probabilities, source_prior=source_prior, target_prior=target_prior
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        predicted_labels = corrected.argmax(axis=1)
        class_accuracy = [np.mean(predicted_labels[y_target == cls] == cls) for cls in range(2)]
        prior_ratios = estimated_prior / source_prior
        true_prior_ratios = target_prior / source_prior
        return {
            "Accuracy": float(np.mean(predicted_labels == y_target)),
            "BalancedAccuracy": float(np.mean(class_accuracy)),
            "LogLoss": float(
                -np.mean(np.log(np.clip(corrected[np.arange(len(y_target)), y_target], 1e-12, 1.0)))
            ),
            "Brier": float(np.mean((corrected[:, 1] - y_target) ** 2)),
            "PredictedPositiveRate": float(np.mean(predicted_labels)),
            "TargetPositivePrior": float(target_prior[1]),
            "TargetPriorL1": float(np.abs(estimated_prior - target_prior).sum()),
            "PriorRatioRMSE": float(np.sqrt(np.mean((prior_ratios - true_prior_ratios) ** 2))),
        }
