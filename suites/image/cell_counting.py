"""Cell-counting suite — microscopy count regression (BBBC005).

Targets are non-negative integer counts (never standardized). Compares
torchregress count losses (Poisson deviance, zero-inflated Poisson,
negative-binomial) and the Poisson-Gaussian imaging-noise loss against
MSE/GaussianNLL point/distribution baselines on fluorescence-microscopy
images where variance grows with the count (Poisson-native data).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import set_harness_seed
from harness.runners import HarnessConfig

from ._base import (
    ImageHarnessRunner,
    _image_to_dataloader,
    _train_image_model,
)


class CellCountingSuite(ImageHarnessRunner):
    """Microscopy cell-count regression with count losses."""

    @property
    def name(self) -> str:
        return "cell_counting"

    @property
    def description(self) -> str:
        return (
            "Microscopy cell counts: Poisson/ZIP/NegBin + PoissonGaussian "
            "vs MSE/GaussianNLL baselines"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["bbbc_cellcount", "bbbc_cellcount_tiny"]

    def methods(self) -> list[str]:
        return [
            "CNN+MSE",
            "CNN+Poisson",
            "CNN+PoissonLR",
            "CNN+ZIP",
            "CNN+NegBin",
            "CNN+GaussianNLL",
            "CNN+PoissonGaussian",
            "CNN+EnhancedPG_fixed",
            "CNN+EnhancedPG_learned",
            "CNN+PoissonGaussianLR",
            "sklearn_RF",
        ]

    def library_name(self, method: str) -> str:
        return "scikit-learn" if method.startswith("sklearn_") else "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "CNN+MSE": "CNN + MSE on raw counts (point baseline, misspecified variance)",
            "CNN+Poisson": "CNN log-rate + Poisson deviance (var=mean)",
            "CNN+PoissonLR": "CNN log-rate + Poisson likelihood-ratio (goodness-of-fit form)",
            "CNN+ZIP": "CNN log-rate + zero-inflation logit (excess-zero counts)",
            "CNN+NegBin": "CNN softplus-mean + learned dispersion (overdispersed counts)",
            "CNN+GaussianNLL": "CNN + diagonal Gaussian NLL (2-head comparison row)",
            "CNN+PoissonGaussian": "CNN log-lambda + shot+readout imaging-noise NLL",
            "CNN+EnhancedPG_fixed": "CNN + calibrated PG (gain=1, read=2.0, shot=1.0 fixed)",
            "CNN+EnhancedPG_learned": "CNN + PG with learned gain/shot noise",
            "CNN+PoissonGaussianLR": "CNN log-lambda + PG likelihood-ratio (diagnostic form)",
            "sklearn_RF": "RandomForestRegressor on flattened pixels — point prediction only",
        }.get(method, "")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        in_channels, _, _ = self._image_shape(splits)
        train_loader = self._make_train_loader(splits, batch_size=config.batch_size)

        if method.startswith("sklearn_"):
            return self._fit_sklearn(splits, config)
        return self._fit_cnn(method, in_channels, train_loader, config)

    def _fit_sklearn(self, splits: Any, config: HarnessConfig) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        Xf = np.asarray(splits["X_train"]).reshape(len(splits["X_train"]), -1)
        est = RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=config.seed, n_jobs=-1
        )
        est.fit(Xf, np.asarray(splits["y_train"]).reshape(-1))
        return {"type": "sklearn", "estimator": est}

    def _fit_cnn(
        self,
        method: str,
        in_channels: int,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
    ) -> dict[str, Any]:
        if method == "CNN+MSE":
            m = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(m, nn.MSELoss(), train_loader, config)
            return {"type": "point", "model": m}
        if method == "CNN+Poisson":
            from torchregress.losses import PoissonDevianceLoss

            m = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(m, PoissonDevianceLoss(log_input=True), train_loader, config)
            return {"type": "log_rate", "model": m}
        if method == "CNN+PoissonLR":
            from torchregress.losses import PoissonLikelihoodRatioLoss

            m = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(m, PoissonLikelihoodRatioLoss(log_input=True), train_loader, config)
            return {"type": "log_rate", "model": m}
        if method == "CNN+ZIP":
            from torchregress.losses import ZeroInflatedPoissonNLLLoss

            m = self._build_cnn(in_channels=in_channels, out_dim=2)
            _train_image_model(
                m, _ZIPWrapper(ZeroInflatedPoissonNLLLoss(log_input=True)), train_loader, config
            )
            return {"type": "zip", "model": m}
        if method == "CNN+NegBin":
            from torchregress.losses import NegativeBinomialNLLLoss

            m = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(
                m,
                _SoftplusMeanWrapper(NegativeBinomialNLLLoss(learn_theta=True)),
                train_loader,
                config,
            )
            return {"type": "negbin", "model": m}
        if method == "CNN+GaussianNLL":
            from torchregress.losses import GaussianNLLLoss

            m = self._build_cnn(in_channels=in_channels, out_dim=2)
            _train_image_model(m, GaussianNLLLoss(), train_loader, config)
            return {"type": "gaussian_nll", "model": m}
        if method == "CNN+PoissonGaussian":
            from torchregress.losses.poisson_gaussian import PoissonGaussianMixtureLoss

            m = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(m, PoissonGaussianMixtureLoss(log_input=True), train_loader, config)
            return {"type": "log_rate", "model": m}
        if method == "CNN+EnhancedPG_fixed":
            from torchregress.losses.poisson_gaussian import EnhancedPoissonGaussianMixtureLoss

            m = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(
                m,
                EnhancedPoissonGaussianMixtureLoss(
                    log_input=True, gain=1.0, offset=0.0, read_noise=2.0, shot_noise=1.0
                ),
                train_loader,
                config,
            )
            return {"type": "log_rate", "model": m}
        if method == "CNN+EnhancedPG_learned":
            from torchregress.losses.poisson_gaussian import EnhancedPoissonGaussianMixtureLoss

            m = self._build_cnn(in_channels=in_channels, out_dim=1)
            loss_fn = EnhancedPoissonGaussianMixtureLoss(
                log_input=True, gain="learn", offset=0.0, read_noise=2.0, shot_noise="learn"
            )
            _train_image_model(
                m, loss_fn, train_loader, config, loss_params=list(loss_fn.parameters())
            )
            return {"type": "log_rate", "model": m}
        if method == "CNN+PoissonGaussianLR":
            from torchregress.losses.poisson_gaussian import PoissonGaussianLikelihoodRatioLoss

            m = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(
                m, PoissonGaussianLikelihoodRatioLoss(log_input=True), train_loader, config
            )
            return {"type": "log_rate", "model": m}
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        t = model_state["type"]
        if t == "sklearn":
            Xf = np.asarray(X_test).reshape(len(np.asarray(X_test)), -1)
            pred = model_state["estimator"].predict(Xf).reshape(-1).astype(np.float32)
            return {"y_pred": np.clip(pred, 0, None)}
        model = model_state["model"]
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)
        if t == "gaussian_nll":
            means, stds = [], []
            with torch.no_grad():
                for (xb,) in loader:
                    out = model(xb.to(device))
                    means.append(out[:, 0].cpu().numpy())
                    stds.append(np.sqrt(np.exp(out[:, 1]).clamp_min(1e-8).cpu().numpy()))
            mean = np.concatenate(means).reshape(-1)
            std = np.concatenate(stds).reshape(-1)
            z = 1.645
            return {
                "lower": np.clip(mean - z * std, 0, None),
                "upper": np.clip(mean + z * std, 0, None),
                "y_pred": np.clip(mean, 0, None),
                "mean": mean,
                "std": std,
            }
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                out = model(xb.to(device))
                if t == "log_rate":
                    outs.append(torch.exp(out.reshape(-1)).clamp_min(0).cpu().numpy())
                elif t == "zip":
                    outs.append(torch.exp(out[:, 0]).clamp_min(0).cpu().numpy())
                elif t == "negbin":
                    outs.append(torch.nn.functional.softplus(out.reshape(-1)).cpu().numpy())
                else:  # point
                    outs.append(out.reshape(-1).cpu().numpy())
        y_pred = np.concatenate(outs).reshape(-1).astype(np.float32)
        return {"y_pred": np.clip(y_pred, 0, None)}


class _ZIPWrapper(nn.Module):
    """Split (log_rate, pi_logit) head output for ZeroInflatedPoissonNLLLoss."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.base(y_pred[:, 0:1], target, pi_logits=y_pred[:, 1:2])


class _SoftplusMeanWrapper(nn.Module):
    """Positive-mean wrapper: softplus(CNN out) → count-mean loss."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return self.base(F.softplus(y_pred), target)
