"""Photo-z imaging suite — redshift from galaxy cutouts (GalaxiesML).

Subclasses the age-regression engine (CNN/ResNet backbones + UQ heads) on
HSC grizy cutouts, with photo-z-native metrics layered on: bias, NMAD and
catastrophic-outlier rate in Δz/(1+z), alongside the standard
RMSE/coverage. Compares torchregress distributional heads against
sklearn pixel baselines on real survey imaging.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from harness.runners import HarnessConfig
from suites.image.age_regression import AgeRegressionSuite


class PhotozImagingSuite(AgeRegressionSuite):
    """Redshift regression on GalaxiesML cutouts + photo-z metrics."""

    @property
    def name(self) -> str:
        return "photoz_imaging"

    @property
    def description(self) -> str:
        return "Photo-z from HSC cutouts (GalaxiesML): CNN UQ heads + outlier metrics"

    @property
    def supported_datasets(self) -> list[str]:
        return ["galaxiesml_cutouts"]

    def methods(self) -> list[str]:
        # Compact flagship set: point/distributional/flow/MC/external.
        return [
            "CNN+GaussianNLL",
            "CNN+MDN_K3",
            "CNN+MultiQuantile",
            "CNN+NormalizingFlow",
            "CNN+MC_Dropout",
            "sklearn_RF",
        ]

    def method_notes(self, method: str) -> str:
        base = super().method_notes(method)
        tag = {
            "CNN+GaussianNLL": " (photo-z baseline head)",
            "CNN+MDN_K3": " (multimodal posteriors for color degeneracies)",
        }.get(method, "")
        return f"{base}{tag}"

    # ------------------------------------------------------------------
    # evaluate — base metrics + photo-z native diagnostics
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        metrics = super().evaluate(method, predictions, y_test, config)
        try:
            y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
            y_true = np.asarray(y_test).reshape(-1)
            # Native redshift units (runner standardizes targets).
            y_scaler = predictions.get("_y_scaler")
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
            dz = (y_pred - y_true) / (1.0 + np.abs(y_true))
            metrics["Bias"] = float(np.mean(dz))
            metrics["NMAD"] = float(1.4826 * np.median(np.abs(dz - np.median(dz))))
            metrics["OutlierFrac"] = float(np.mean(np.abs(dz) > 0.15))
        except Exception:
            pass
        return metrics
