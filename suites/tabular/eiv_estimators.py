"""Forward-inversion and attenuation-free EIV objectives: harness parity.

The existing ``eiv`` suite compares the *discriminative* EIV losses.  This
suite covers the newer estimators that attack attenuation from a different
angle:

* ``SplitMeasurement`` — paired independent measurement views, no noise model.
* ``CorrectedScore`` — Stefanski-Nakamura corrected score (subtracts the
  measurement-error curvature instead of adding a TLS gradient penalty).
* ``ForwardInversion_GN`` / ``ForwardInversion_LBFGS`` — generative ``y -> x``
  surrogate (``ForwardInversionRegressor``) inverted per object by
  Gauss-Newton or L-BFGS; also reports the Fisher-covariance coverage.

Baselines
---------
- ``MSE``: MLP trained on the noisy features (attenuated reference).
- ``sklearn_OLS``: external ordinary least squares on the noisy features
  (classical attenuation baseline).

The synthetic EIV datasets carry clean features (``X_train_true``) and a known
``sigma_x``; the split-measurement second view is generated from the clean
features with independent noise, matching the data-generating process.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn
from torchregress.algorithms import ForwardInversionRegressor, build_decoder_mlp
from torchregress.losses import (
    CorrectedScoreEIVLoss,
    SplitMeasurementLoss,
    corrected_score_eiv_loss,
    split_measurement_loss,
)

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.metrics import compute_all_metrics
from harness.runners import (
    HarnessConfig,
    HarnessRunner,
    _inverse_transform_predictions,
    _inverse_transform_y,
)
from suites._backbones import _MLP, _hidden_size

_METHODS = [
    "MSE",
    "sklearn_OLS",
    "SplitMeasurement",
    "CorrectedScore",
    "ForwardInversion_GN",
    "ForwardInversion_LBFGS",
]


def _epochs(config: HarnessConfig, cap: int = 5) -> int:
    return min(int(config.epochs), cap) if config.smoke else int(config.epochs)


class EIVEstimatorsSuite(HarnessRunner):
    """Forward inversion and attenuation-free EIV losses on synthetic EIV data."""

    @property
    def name(self) -> str:
        return "eiv_estimators"

    @property
    def description(self) -> str:
        return (
            "Forward generative inversion, split-measurement and corrected-score "
            "EIV vs MSE/sklearn OLS attenuation baselines"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "synthetic_eiv",
            "synthetic_eiv_highnoise",
            "synthetic_eiv_nonlinear",
        ]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def library_name(self, method: str) -> str:
        return "sklearn" if method.startswith("sklearn") else "torchregress"

    def library_version(self, method: str) -> str:
        if method.startswith("sklearn"):
            import sklearn

            return str(sklearn.__version__)
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        notes = {
            "MSE": "MLP on noisy features (attenuated baseline)",
            "sklearn_OLS": "ordinary least squares on noisy features (external)",
            "SplitMeasurement": "paired independent views, cross-product loss (no sigma_x)",
            "CorrectedScore": "Stefanski-Nakamura corrected score with known sigma_x",
            "ForwardInversion_GN": "generative y->x surrogate, Gauss-Newton MLE inversion",
            "ForwardInversion_LBFGS": "generative y->x surrogate, L-BFGS MLE inversion",
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _second_view(X_true: np.ndarray, sigma_x: float, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(int(seed) + 101)
        noise = torch.randn(X_true.shape, generator=generator) * float(sigma_x)
        return torch.as_tensor(X_true, dtype=torch.float32) + noise

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        device = current_device()
        X = torch.as_tensor(np.asarray(splits["X_train"]), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(splits["y_train"]), dtype=torch.float32, device=device)
        y = y.reshape(-1, 1)
        sigma_x = float(splits.get("sigma_x", 0.0))

        if method == "sklearn_OLS":
            model = LinearRegression()
            model.fit(np.asarray(splits["X_train"]), np.asarray(splits["y_train"]))
            return {"type": "sklearn_ols", "model": model}

        if method.startswith("ForwardInversion"):
            hidden = (16,) if config.smoke else (64, 64)
            inversion = ForwardInversionRegressor(
                decoder=build_decoder_mlp(
                    n_labels=1,
                    n_observables=int(X.shape[1]),
                    hidden_dims=hidden,
                ),
                sigma_x=sigma_x,
                learn_noise=True,
                epochs=_epochs(config),
                batch_size=min(int(config.batch_size), len(X)),
                lr=config.lr,
                seed=int(config.seed),
                device=str(device),
            )
            inversion.fit(X, y)
            return {"type": "forward_inversion", "model": inversion}

        hidden = _hidden_size(int(X.shape[1]))
        model = _MLP(in_dim=int(X.shape[1]), hidden=hidden, out_dim=1).to(device)
        loss: nn.Module
        x_second: torch.Tensor | None = None
        if method == "MSE":
            loss = nn.MSELoss()
        elif method == "SplitMeasurement":
            from torchregress.losses import SplitMeasurementLoss

            loss = SplitMeasurementLoss(model)
            X_true = np.asarray(splits.get("X_train_true", splits["X_train"]))
            x_second = self._second_view(X_true, sigma_x, int(config.seed)).to(device)
        elif method == "CorrectedScore":
            from torchregress.losses import CorrectedScoreEIVLoss

            loss = CorrectedScoreEIVLoss(model, sigma_x=sigma_x)
        else:
            raise ValueError(f"Unknown method: {method}")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        generator = torch.Generator().manual_seed(int(config.seed) + 3)
        batch_size = min(int(config.batch_size), len(X))
        for _ in range(_epochs(config)):
            order = torch.randperm(len(X), generator=generator).to(device)
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                if method == "MSE":
                    batch_loss = loss(model(X[idx]), y[idx])
                elif x_second is not None:
                    batch_loss = loss(X[idx], y[idx], y_pred_second=x_second[idx].to(device))
                else:
                    batch_loss = loss(X[idx], y[idx])
                batch_loss.backward()
                optimizer.step()
        model.eval()
        return {"type": "torchregress_eiv", "model": model}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        kind = model_state["type"]
        if kind == "sklearn_ols":
            prediction = np.asarray(model_state["model"].predict(np.asarray(X_test)))
            return {"y_pred": prediction.reshape(-1)}
        if kind == "forward_inversion":
            inversion = model_state["model"]
            device = current_device()
            X_t = torch.as_tensor(np.asarray(X_test), dtype=torch.float32, device=device)
            inversion_method = "gauss_newton" if method.endswith("_GN") else "lbfgs"
            mean, std = inversion.predict(
                X_t,
                method=inversion_method,
                max_iter=40,
                return_std=True,
            )
            mean_np = np.asarray(mean.detach().cpu()).reshape(-1)
            std_np = np.asarray(std.detach().cpu()).reshape(-1)
            return {"y_pred": mean_np, "mean": mean_np, "std": std_np}
        model = model_state["model"]
        with torch.no_grad():
            out = model(
                torch.as_tensor(np.asarray(X_test), dtype=torch.float32, device=current_device())
            )
        return {"y_pred": np.asarray(out.detach().cpu()).reshape(-1)}

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        del method, config
        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_true = np.asarray(_inverse_transform_y(y_test, y_scaler)).reshape(-1)
        else:
            y_true = np.asarray(y_test).reshape(-1)
        result = compute_all_metrics(predictions, y_true)
        try:
            from torchregress.metrics import attenuation_factor

            y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
            n = min(len(y_pred), len(y_true))
            result["Attenuation"] = float(attenuation_factor(y_pred[:n], y_true[:n]))
        except Exception:
            pass
        return result


__all__ = ["EIVEstimatorsSuite"]
