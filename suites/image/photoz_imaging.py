"""Photo-z imaging suite — redshift from galaxy cutouts (GalaxiesML).

Subclasses the age-regression engine (CNN backbones + UQ heads) on HSC
grizy cutouts, with photo-z-native metrics layered on: bias, NMAD and
catastrophic-outlier rate in Δz/(1+z), alongside the standard
RMSE/coverage. Rows: distributional CNN heads, Laplace-posthoc BNN,
measured-noise EIV training. No pixel-RF (uninformative at 20k dims).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig
from suites.image._base import _image_to_dataloader
from suites.image.age_regression import _TIMM_AVAILABLE, AgeRegressionSuite


class PhotozImagingSuite(AgeRegressionSuite):
    """Redshift regression on GalaxiesML cutouts + photo-z metrics."""

    @property
    def name(self) -> str:
        return "photoz_imaging"

    @property
    def description(self) -> str:
        return "Photo-z from HSC cutouts (GalaxiesML): CNN UQ + Laplace BNN + noise-aware EIV"

    @property
    def supported_datasets(self) -> list[str]:
        return ["galaxiesml_cutouts"]

    def methods(self) -> list[str]:
        # Compact flagship set: distributional heads, measured-noise EIV,
        # MC-Dropout (the image BNN row), SLS level sets, R2C binned
        # conformal, last-layer Laplace BNN (conv-safe path). No sklearn_RF
        # (uninformative on pixels). The timm ResNet-18 comparator exercises
        # the inherited age-regression engine (fit + predict) on the same
        # cutouts, splits, and photo-z metrics — listed only when timm is
        # installed (Skaha image sessions; absent locally).
        methods = [
            "CNN+GaussianNLL",
            "CNN+MDN_K3",
            "CNN+MultiQuantile",
            "CNN+NormalizingFlow",
            "CNN+MC_Dropout",
            "CNN+NoiseAware",
            "CNN+SLS",
            "CNN+R2C",
            "CNN+Laplace",
        ]
        if _TIMM_AVAILABLE:
            methods.append("timm_resnet18+GaussianNLL")
        return methods

    def method_notes(self, method: str) -> str:
        base = super().method_notes(method)
        tag = {
            "CNN+GaussianNLL": " (photo-z baseline head)",
            "CNN+MDN_K3": " (multimodal posteriors for color degeneracies)",
            "CNN+NoiseAware": " (measured sky-sigma input augmentation)",
            "CNN+SLS": " (min-volume level-set photo-z regions)",
            "CNN+R2C": " (imbalanced binned head + R2C conformal)",
            "CNN+Laplace": " (last-layer Laplace BNN, conv-safe Fisher)",
            "timm_resnet18+GaussianNLL": " (external vision comparator: unpretrained ResNet-18 + Gaussian NLL)",
        }.get(method, "")
        return f"{base}{tag}"

    # ------------------------------------------------------------------
    # fit — new rows here, everything else delegates to the age engine
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        if method == "CNN+NoiseAware":
            return self._fit_noiseaware(splits, config)
        if method == "CNN+Laplace":
            return self._fit_laplace(splits, config)
        if method == "CNN+R2C":
            return self._fit_r2c(splits, config)
        return super().fit(method, splits, config)

    def _fit_laplace(self, splits: dict[str, np.ndarray], config: HarnessConfig) -> dict[str, Any]:
        """MAP CNN + last-layer Laplace post-hoc (conv-safe Fisher)."""
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from torchregress.ensemble import FullNetworkLaplace

        set_harness_seed(config.seed)
        in_channels, _, _ = self._image_shape(splits)
        model = self._build_cnn(in_channels=in_channels, out_dim=1)
        dev = current_device()
        model.to(dev)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.asarray(splits["X_train"])).float(),
                torch.from_numpy(np.asarray(splits["y_train"])).float().unsqueeze(1),
            ),
            batch_size=config.batch_size,
            shuffle=True,
        )
        opt = optim.Adam(model.parameters(), lr=config.lr)
        model.train()
        for _ in range(config.epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(dev), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                nn.MSELoss()(model(Xb), yb).backward()
                opt.step()
        model.eval()
        lap = FullNetworkLaplace(model.to(dev), last_layer_only=True)
        lap.fit(loader, nn.MSELoss(reduction="none"))
        return {"type": "laplace", "model": lap, "base": model}

    def _fit_noiseaware(
        self, splits: dict[str, np.ndarray], config: HarnessConfig
    ) -> dict[str, Any]:
        """EIV training: per-image measured sky-sigma input augmentation."""
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        set_harness_seed(config.seed)
        in_channels, _, _ = self._image_shape(splits)
        model = self._build_cnn(in_channels=in_channels, out_dim=1)
        dev = current_device()
        model.to(dev)
        sig = np.asarray(
            splits.get("sigma_image_train", np.full(len(splits["X_train"]), 0.01)),
            dtype=np.float32,
        ).reshape(-1)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.asarray(splits["X_train"])).float(),
                torch.from_numpy(np.asarray(splits["y_train"])).float().unsqueeze(1),
                torch.from_numpy(sig).float(),
            ),
            batch_size=config.batch_size,
            shuffle=True,
        )
        opt = optim.Adam(model.parameters(), lr=config.lr)
        model.train()
        for _ in range(config.epochs):
            for Xb, yb, sb in loader:
                Xb, yb, sb = Xb.to(dev), yb.to(dev), sb.to(dev)
                opt.zero_grad(set_to_none=True)
                noisy = Xb + sb.view(-1, 1, 1, 1) * torch.randn_like(Xb)
                nn.MSELoss()(model(noisy), yb).backward()
                opt.step()
        model.eval()
        return {"type": "point", "model": model}

    def _fit_r2c(self, splits: dict[str, np.ndarray], config: HarnessConfig) -> dict[str, Any]:
        """Imbalanced binned head (inverse-freq WeightedCE) + R2C conformal.

        Uniform z-bins have imbalanced occupancy (realistic); the R2C
        predictor turns the classifier into calibrated intervals —
        imbalance × R2C composition.
        """
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from torchregress.losses import WeightedCrossEntropyLoss
        from torchregress.losses.conformal import R2CConformal

        set_harness_seed(config.seed)
        in_channels, _, _ = self._image_shape(splits)
        y_tr = np.asarray(splits["y_train"]).reshape(-1)
        n_bins = 20
        lo, hi = float(y_tr.min()), float(y_tr.max())
        edges = np.linspace(lo - 1e-6, hi + 1e-6, n_bins + 1).astype(np.float32)
        idx_tr = np.clip(np.digitize(y_tr, edges) - 1, 0, n_bins - 1)
        counts = np.bincount(idx_tr, minlength=n_bins).astype(np.float64)
        weights = len(idx_tr) / np.clip(counts * n_bins, 1.0, None)
        model = self._build_cnn(in_channels=in_channels, out_dim=n_bins)
        dev = current_device()
        model.to(dev)
        loss_fn = WeightedCrossEntropyLoss(
            weight=torch.from_numpy(weights.astype(np.float32)).to(dev)
        )
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.asarray(splits["X_train"])).float(),
                torch.from_numpy(idx_tr).long(),
            ),
            batch_size=config.batch_size,
            shuffle=True,
        )
        opt = optim.Adam(model.parameters(), lr=config.lr)
        model.train()
        for _ in range(config.epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(dev), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                loss_fn(model(Xb), yb).backward()
                opt.step()
        model.eval()
        # Calibrate R2C on the last 20% of train.
        X_tr = np.asarray(splits["X_train"])
        n_cal = max(16, len(X_tr) // 5)
        X_cal = torch.from_numpy(X_tr[-n_cal:]).float().to(dev)
        with torch.no_grad():
            probs_cal = torch.softmax(model(X_cal), dim=-1)
        r2c = R2CConformal(alpha=0.1, bin_edges=torch.from_numpy(edges).float().to(dev))
        r2c.calibrate(
            probs_cal,
            torch.from_numpy(y_tr[-n_cal:].astype(np.float32)).to(dev),
        )
        centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
        scaler = splits.get("_y_scaler")
        if scaler is not None:
            smean = float(np.asarray(scaler.mean_).reshape(-1)[0])
            sscale = float(np.asarray(scaler.scale_).reshape(-1)[0])
        else:
            smean, sscale = 0.0, 1.0
        return {
            "type": "r2c",
            "model": model,
            "r2c": r2c,
            "centers": centers,
            "y_smean": smean,
            "y_sscale": sscale,
        }

    # ------------------------------------------------------------------
    # predict — laplace/r2c branches here; everything else delegates
    # ------------------------------------------------------------------

    def _predict_laplace(
        self, lap: Any, base: nn.Module, X_test: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Last-layer posterior MC mean/var + intervals (batched)."""
        from suites.image._base import _image_to_dataloader

        device = next(base.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)
        means, stds = [], []
        with torch.no_grad():
            for (Xb,) in loader:
                Xb = Xb.to(device)
                mean_b = base(Xb).reshape(-1)
                _, var_b = lap.predict_with_uncertainty(Xb)
                means.append(mean_b.detach().cpu().numpy())
                std_b = np.sqrt(var_b.reshape(-1).detach().cpu().numpy().clip(min=1e-12))
                stds.append(std_b)
        mean_np = np.concatenate(means).reshape(-1)
        std_np = np.concatenate(stds).reshape(-1)
        z = 1.645
        return {
            "lower": mean_np - z * std_np,
            "upper": mean_np + z * std_np,
            "y_pred": mean_np,
            "mean": mean_np,
            "std": std_np,
        }

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        if model_state.get("type") == "laplace":
            return self._predict_laplace(model_state["model"], model_state["base"], X_test)
        if model_state.get("type") == "r2c":
            model = model_state["model"]
            r2c = model_state["r2c"]
            centers = np.asarray(model_state["centers"], dtype=np.float64)
            smean = float(model_state.get("y_smean", 0.0))
            sscale = float(model_state.get("y_sscale", 1.0))
            model.eval()
            device = next(model.parameters()).device
            loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)
            probs_list, means = [], []
            with torch.no_grad():
                for (Xb,) in loader:
                    p = torch.softmax(model(Xb.to(device)), dim=-1)
                    probs_list.append(p.detach().cpu())
                    means.append(
                        (p @ torch.from_numpy(centers.astype(np.float32)).to(device))
                        .reshape(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            probs = torch.cat(probs_list, dim=0).to(device)
            lo_t, hi_t = r2c.predict_interval(probs)
            lo = np.asarray(lo_t.detach().cpu()).reshape(-1) * sscale + smean
            hi = np.asarray(hi_t.detach().cpu()).reshape(-1) * sscale + smean
            y_pred = np.concatenate(means).reshape(-1) * sscale + smean
            return {"lower": lo, "upper": hi, "y_pred": y_pred}
        return super().predict(method, model_state, X_test, config)

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
        import torch as _torch
        from torchregress.metrics import (
            NormalizedMedianAbsoluteDeviation,
            OutlierFraction,
            interval_score,
        )

        metrics = super().evaluate(method, predictions, y_test, config)
        try:
            y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
            y_true = np.asarray(y_test).reshape(-1)
            # Native redshift units (runner standardizes targets).
            y_scaler = predictions.get("_y_scaler")
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
            yp_t = _torch.from_numpy(y_pred.astype(np.float64))
            yt_t = _torch.from_numpy(y_true.astype(np.float64))
            # Library photo-z metrics (relative mode = Δz/(1+z) standard).
            of = OutlierFraction(threshold=0.15, mode="relative")
            of.update(yp_t, yt_t)
            metrics["OutlierFrac"] = float(of.compute().item())
            nmad = NormalizedMedianAbsoluteDeviation(normalization="relative")
            nmad.update(yp_t, yt_t)
            metrics["NMAD"] = float(nmad.compute().item())
            metrics["Bias"] = float(((yp_t - yt_t) / (1.0 + yt_t.abs())).mean().item())
            # Library analytic scores for Gaussian rows (native units).
            if "mean" in predictions and "std" in predictions:
                from torchregress.metrics import crps_gaussian, dss_score

                try:
                    m = np.asarray(predictions["mean"]).reshape(-1)
                    s = np.asarray(predictions["std"]).reshape(-1)
                    if y_scaler is not None:
                        sc = float(np.asarray(y_scaler.scale_).reshape(-1)[0])
                        mu = float(np.asarray(y_scaler.mean_).reshape(-1)[0])
                        m = m * sc + mu
                        s = s * sc
                    metrics["CRPS_Gauss"] = float(crps_gaussian(m, y_true, s))
                    metrics["DSS"] = float(dss_score(m, s, y_true))
                except Exception:
                    pass
            # Library interval score where intervals exist.
            if "lower" in predictions and "upper" in predictions:
                lo = np.asarray(predictions["lower"]).reshape(-1)
                hi = np.asarray(predictions["upper"]).reshape(-1)
                if y_scaler is not None:
                    lo = y_scaler.inverse_transform(lo.reshape(-1, 1)).ravel()
                    hi = y_scaler.inverse_transform(hi.reshape(-1, 1)).ravel()
                try:
                    metrics["IntervalScore"] = float(interval_score(lo, hi, y_true, alpha=0.1))
                except Exception:
                    pass
            # Library CRPS where predictive samples exist (N, S), native units.
            if "samples" in predictions:
                from torchregress.metrics import crps_from_samples

                try:
                    samp = np.asarray(predictions["samples"])
                    if samp.ndim != 2 or samp.shape[0] != len(y_true):
                        samp = np.asarray(samp).reshape(len(y_true), -1)
                    if y_scaler is not None:
                        sc = float(np.asarray(y_scaler.scale_).reshape(-1)[0])
                        mu = float(np.asarray(y_scaler.mean_).reshape(-1)[0])
                        samp = samp * sc + mu
                    metrics["CRPS"] = float(crps_from_samples(samp, y_true.reshape(-1, 1)))
                except Exception:
                    pass
        except Exception:
            pass
        return metrics
