"""Image calibration suite — post-hoc calibration on frozen CNN predictors.

Mirrors ``tabular.calibration``: Uncalibrated vs VarianceTemperatureScaler vs
IsotonicMeanCalibrator on top of a frozen CNN GaussianNLL predictor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from harness.core import set_harness_seed
from harness.runners import HarnessConfig

from ._base import (
    ImageHarnessRunner,
    _carve_image_calibration_split,
    _image_to_dataloader,
    _train_image_model,
)


class ImageCalibrationSuite(ImageHarnessRunner):
    """Post-hoc calibration for image regression."""

    Z_SCORE = 1.645

    @property
    def name(self) -> str:
        return "image_calibration"

    @property
    def description(self) -> str:
        return "Post-hoc calibration on frozen CNN predictors (temp/isotonic)"

    @property
    def supported_datasets(self) -> list[str]:
        # Scalar targets only (multitarget → keypoint_regression).
        return ["synthetic_image_regression", "imdb_wiki"]

    def methods(self) -> list[str]:
        return ["Uncalibrated", "VarianceTemperatureScaler", "IsotonicMeanCalibrator"]

    def library_name(self, method: str) -> str:
        return "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "Uncalibrated": "Frozen CNN GaussianNLL, no post-hoc calibration",
            "VarianceTemperatureScaler": "Single temperature on predictive variance (cal split)",
            "IsotonicMeanCalibrator": "Isotonic regression on predicted means (cal split)",
        }.get(method, "")

    def fit(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        set_harness_seed(config.seed)
        from torchregress.losses import GaussianNLLLoss

        in_channels, _, _ = self._image_shape(splits)
        carved = _carve_image_calibration_split(splits, cal_fraction=0.2, seed=config.seed)
        fit_splits = {
            "X_train": carved["X_train"],
            "y_train": carved["y_train"],
            "X_test": carved["X_train"][:0],
            "y_test": carved["y_train"][:0],
        }
        loader = self._make_train_loader(fit_splits, batch_size=config.batch_size)
        model = self._build_cnn(in_channels=in_channels, out_dim=2)
        _train_image_model(model, GaussianNLLLoss(), loader, config)
        model.eval()
        mean_cal, std_cal = self._forward(model, np.asarray(carved["X_cal"]))
        y_cal = np.asarray(carved["y_cal"]).reshape(-1)
        if method == "Uncalibrated":
            return {"type": "gaussian", "model": model, "temp": 1.0, "iso": None}
        if method == "VarianceTemperatureScaler":
            # Closed-form-ish 1-D search over log-temp grid on NLL.
            best_t, best_nll = 1.0, float("inf")
            for t in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
                s = np.clip(std_cal * t, 1e-6, None)
                nll = float(
                    np.mean(
                        0.5 * np.log(2 * np.pi) + np.log(s) + 0.5 * ((y_cal - mean_cal) / s) ** 2
                    )
                )
                if nll < best_nll:
                    best_nll, best_t = nll, t
            return {"type": "gaussian", "model": model, "temp": best_t, "iso": None}
        if method == "IsotonicMeanCalibrator":
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(mean_cal, y_cal)
            return {"type": "gaussian", "model": model, "temp": 1.0, "iso": iso}
        raise ValueError(f"Unknown method: {method}")

    def _forward(self, model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X, y=None, batch_size=64, shuffle=False)
        means, stds = [], []
        with torch.no_grad():
            for (xb,) in loader:
                out = model(xb.to(device))
                means.append(out[:, 0].cpu().numpy())
                stds.append(np.sqrt(np.exp(out[:, 1]).clamp_min(1e-8).cpu().numpy()))
        return np.concatenate(means), np.concatenate(stds)

    def predict(self, method: str, state: Any, X_test: np.ndarray, config: HarnessConfig) -> Any:
        mean, std = self._forward(state["model"], X_test)
        std = np.clip(std * float(state.get("temp", 1.0)), 1e-6, None)
        if state.get("iso") is not None:
            mean = np.asarray(state["iso"].predict(mean)).reshape(-1)
        return {
            "lower": mean - self.Z_SCORE * std,
            "upper": mean + self.Z_SCORE * std,
            "y_pred": mean,
            "mean": mean,
            "std": std,
        }
