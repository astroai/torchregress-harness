"""Calibration suite: torchregress post-hoc calibrators vs sklearn.

Compares torchregress calibration transforms (VarianceTemperatureScaler,
IsotonicMeanCalibrator, PITCalibrator) against sklearn's IsotonicRegression
on real tabular + synthetic datasets.

Canonical scenarios
-------------------
* ``sklearn_diabetes_reference``
  https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
* ``sklearn_synthetic_reference``
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

Pattern
-------
All methods share a **base model** (Gaussian NLL MLP) trained once on the
training split. The calibration split is carved from the training data
(similar to the conformal suite's ``run()`` override). Each calibrator is
fitted on the calibration set and evaluated on the test set.

Metrics (via the default ``compute_all_metrics`` with ``mean``/``std``):
- RMSE, MAE, R2 (point)
- NLL, CRPS (distribution)
- ECE, PIT_KS (calibration quality)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# sklearn calibrator (hard dependency — sklearn is a core harness dep)
# ---------------------------------------------------------------------------
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------------------
# Torchregress calibrators and base loss
# ---------------------------------------------------------------------------
from torchregress.calibration import (
    IsotonicMeanCalibrator,
    PITCalibrator,
    VarianceTemperatureScaler,
)
from torchregress.losses import BetaNLLLoss

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _carve_calibration_split, _hidden_size, _train_one

# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class CalibrationSuite(HarnessRunner):
    """Post-hoc calibration benchmark suite.

    Overrides ``run()`` to carve a calibration split from training data.
    A shared base model is trained once, then each calibrator is fitted
    and evaluated independently.
    """

    CALIBRATION_FRAC = 0.25

    @property
    def name(self) -> str:
        return "calibration"

    @property
    def description(self) -> str:
        return "Post-hoc calibration transforms: torchregress vs sklearn IsotonicRegression"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "diabetes",
            "synthetic_hetero",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        """sklearn calibration-relevant examples."""
        return ["sklearn_diabetes_reference", "sklearn_synthetic_reference"]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        return [
            "Uncalibrated",
            "VarianceTemperatureScaler",
            "IsotonicMeanCalibrator",
            "sklearn_IsotonicRegression",
        ]

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn"):
            return "sklearn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        notes = {
            "Uncalibrated": "raw Gaussian NLL model; no calibration",
            "VarianceTemperatureScaler": "scalar temperature on predictive variance (NLL optim)",
            "IsotonicMeanCalibrator": "PAVA isotonic regression on predicted means",
            "sklearn_IsotonicRegression": "sklearn IsotonicRegression on predicted means",
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # fit — uses calibration split from augmented_splits
    # ------------------------------------------------------------------

    def fit(
        self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit base model + calibrator.

        ``splits`` must contain ``X_train``, ``y_train``, ``X_cal``, ``y_cal``
        (added by ``run()`` override).
        """
        set_harness_seed(config.seed)

        X_train = torch.from_numpy(splits["X_train"]).float()
        y_train = torch.from_numpy(splits["y_train"]).float().unsqueeze(1)
        X_cal = torch.from_numpy(splits["X_cal"]).float()
        y_cal = torch.from_numpy(splits["y_cal"]).float()

        # Train base model once (shared across all methods)
        in_dim = X_train.shape[1]
        hidden = _hidden_size(in_dim)
        base_model = _MLP(in_dim, hidden, out_dim=2)
        _train_one(base_model, BetaNLLLoss(beta=0.5), X_train, y_train, config)
        base_model.eval()

        # Get base predictions on calibration set
        with torch.no_grad():
            base_out_cal = base_model(X_cal)
            cal_mean = base_out_cal[:, 0].clone()
            cal_log_var = base_out_cal[:, 1].clone()
            cal_var = torch.exp(cal_log_var).clamp_min(1e-8)
            cal_std = torch.sqrt(cal_var)

        # --- Fit calibrator ---
        calibrator = None

        if method == "Uncalibrated":
            pass  # no calibrator

        elif method == "VarianceTemperatureScaler":
            calibrator = VarianceTemperatureScaler()
            calibrator.fit(cal_mean.unsqueeze(1), cal_var.unsqueeze(1), y_cal.unsqueeze(1))

        elif method == "IsotonicMeanCalibrator":
            calibrator = IsotonicMeanCalibrator()
            calibrator.fit(cal_mean, y_cal.unsqueeze(1))

        elif method == "PITCalibrator":
            pit_cal = PITCalibrator.pit_from_gaussian(
                cal_mean, cal_std, y_cal.unsqueeze(1)
            )
            calibrator = PITCalibrator()
            calibrator.fit(pit_cal)

        elif method == "sklearn_IsotonicRegression":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(
                cal_mean.numpy().reshape(-1),
                y_cal.numpy().reshape(-1),
            )

        return {
            "base_model": base_model,
            "calibrator": calibrator,
        }

    # ------------------------------------------------------------------
    # predict — apply calibrator to base model outputs
    # ------------------------------------------------------------------

    def predict(
        self, method: str, model_state: dict[str, Any], X_test: np.ndarray, config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        base_model = model_state["base_model"]
        calibrator = model_state["calibrator"]

        with torch.no_grad():
            out = base_model(torch.from_numpy(X_test).float())
            mean = out[:, 0].clone()
            log_var = out[:, 1].clone()
            var = torch.exp(log_var).clamp_min(1e-8)
            std = torch.sqrt(var)

        # --- Apply calibration ---
        if method == "Uncalibrated":
            calibrated_mean = mean.numpy()
            calibrated_std = std.numpy()

        elif method == "VarianceTemperatureScaler":
            calibrated_mean = mean.numpy()
            calibrated_var = calibrator.transform(var.unsqueeze(1)).numpy()
            calibrated_std = np.sqrt(calibrated_var.clip(1e-8))

        elif method == "IsotonicMeanCalibrator":
            calibrated_mean = calibrator.transform(mean).numpy()
            calibrated_std = std.numpy()

        elif method == "PITCalibrator":
            # PIT calibrator maps PIT values; we apply it to test PITs
            # but the standard approach is: calibrate PIT → map to
            # calibrated probability → invert via Gaussian CDF
            # For simplicity, return original mean/std (ECE is the metric)
            calibrated_mean = mean.numpy()
            calibrated_std = std.numpy()

        elif method == "sklearn_IsotonicRegression":
            calibrated_mean = calibrator.predict(mean.numpy().reshape(-1))
            calibrated_std = std.numpy()

        else:
            calibrated_mean = mean.numpy()
            calibrated_std = std.numpy()

        return {"mean": calibrated_mean.reshape(-1), "std": calibrated_std.reshape(-1)}

    # ------------------------------------------------------------------
    # Override run() to create calibration split
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the calibration suite, training a shared base model once."""
        from time import perf_counter

        cfg = config or HarnessConfig()

        # --- Resolve canonical scenario (reusable helper) ---
        dataset_name, splits = self._prepare_scenario_run(
            scenario_name, dataset_name, splits,
        )

        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name)

        # Read scenario-driven parameters (with suite defaults as fallback)
        cal_frac = float(splits.get("_calibration_fraction", self.CALIBRATION_FRAC))

        # Carve calibration split from training data
        aug = _carve_calibration_split(
            splits,
            cal_frac=cal_frac,
            seed=cfg.seed,
            smoke=cfg.smoke,
            suite_name="calibration",
        )

        X_fit = aug["X_train"]
        y_fit = aug["y_train"]
        X_cal = aug["X_cal"]
        y_cal = aug["y_cal"]
        X_test = aug["X_test"]
        y_test = aug["y_test"]

        # Train base model ONCE (shared across all calibrators)
        set_harness_seed(cfg.seed)
        X_train_t = torch.from_numpy(X_fit).float()
        y_train_t = torch.from_numpy(y_fit).float().unsqueeze(1)

        in_dim = X_train_t.shape[1]
        hidden = _hidden_size(in_dim)
        base_model = _MLP(in_dim, hidden, out_dim=2)
        _train_one(base_model, BetaNLLLoss(beta=0.5), X_train_t, y_train_t, cfg)
        base_model.eval()

        rows: list[dict[str, Any]] = []

        for method in self.methods():
            set_harness_seed(cfg.seed)

            try:
                t0 = perf_counter()
                calibrator = self._fit_calibrator(
                    method, base_model, X_cal, y_cal,
                )
                fit_time = perf_counter() - t0

                t0 = perf_counter()
                predictions = self._predict_calibrated(
                    method, base_model, calibrator, X_test,
                )
                pred_time = perf_counter() - t0

                if "_y_scaler" in aug:
                    predictions["_y_scaler"] = aug["_y_scaler"]

                metrics = self.evaluate(method, predictions, y_test, cfg)

                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    **metrics,
                    "train_s": fit_time,
                    "eval_s": pred_time,
                    "SourceURL": splits.get("_source_url", ""),
                    "Notes": self.method_notes(method),
                }

            except ImportError as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "RMSE": None,
                    "MAE": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "Notes": f"skipped: {type(exc).__name__}: {exc}",
                }

            except Exception as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "RMSE": None,
                    "MAE": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }

            rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Calibrator fitting (base model already trained)
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_calibrator(
        method: str,
        base_model: nn.Module,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> Any:
        """Fit a calibrator on top of a pre-trained base model."""
        X_cal_t = torch.from_numpy(X_cal).float()
        y_cal_t = torch.from_numpy(y_cal).float()

        with torch.no_grad():
            out = base_model(X_cal_t)
            cal_mean = out[:, 0]
            log_var = out[:, 1]
            cal_var = torch.exp(log_var).clamp_min(1e-8)

        if method == "Uncalibrated":
            return None

        elif method == "VarianceTemperatureScaler":
            scaler = VarianceTemperatureScaler()
            scaler.fit(cal_mean.unsqueeze(1), cal_var.unsqueeze(1), y_cal_t.unsqueeze(1))
            return scaler

        elif method == "IsotonicMeanCalibrator":
            calibrator = IsotonicMeanCalibrator()
            calibrator.fit(cal_mean, y_cal_t.unsqueeze(1))
            return calibrator

        elif method == "sklearn_IsotonicRegression":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(
                cal_mean.numpy().reshape(-1),
                y_cal.reshape(-1),
            )
            return calibrator

        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _predict_calibrated(
        method: str,
        base_model: nn.Module,
        calibrator: Any,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Apply calibrator to base model test predictions."""
        with torch.no_grad():
            out = base_model(torch.from_numpy(X_test).float())
            mean = out[:, 0].numpy()
            std = torch.sqrt(torch.exp(out[:, 1]).clamp_min(1e-8)).numpy()

        if method == "Uncalibrated":
            calibrated_mean = mean
            calibrated_std = std

        elif method == "VarianceTemperatureScaler":
            calibrated_mean = mean
            with torch.no_grad():
                var_t = torch.exp(out[:, 1]).clamp_min(1e-8)
                calibrated_var = calibrator.transform(var_t.unsqueeze(1)).numpy()
                calibrated_std = np.sqrt(calibrated_var.clip(1e-8))

        elif method == "IsotonicMeanCalibrator":
            calibrated_mean = calibrator.transform(
                torch.from_numpy(mean).float()
            ).numpy().reshape(-1)
            calibrated_std = std

        elif method == "sklearn_IsotonicRegression":
            calibrated_mean = calibrator.predict(mean.reshape(-1)).reshape(-1)
            calibrated_std = std

        else:
            calibrated_mean = mean
            calibrated_std = std

        return {"mean": calibrated_mean.reshape(-1), "std": calibrated_std.reshape(-1)}
