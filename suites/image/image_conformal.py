"""Image conformal suite — Split/CQR on CNN backbones.

Mirrors ``tabular.conformal`` on images: fit a point/quantile CNN, carve a
calibration split (``_carve_image_calibration_split``), apply finite-sample
conformal correction. Externals (MAPIE/crepes/torchcp) run post-hoc on
residuals so they stay backbone-agnostic; missing libs → skipped rows.
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
    _carve_image_calibration_split,
    _conformal_quantile,
    _image_to_dataloader,
    _train_image_model,
)

try:
    import mapie  # noqa: F401

    _MAPIE_AVAILABLE = True
except ImportError:
    _MAPIE_AVAILABLE = False

try:
    import crepes  # noqa: F401

    _CREPES_AVAILABLE = True
except ImportError:
    _CREPES_AVAILABLE = False

try:
    import torchcp  # noqa: F401

    _TORCHCP_AVAILABLE = True
except ImportError:
    _TORCHCP_AVAILABLE = False


class ImageConformalSuite(ImageHarnessRunner):
    """Split/CQR conformal prediction on image-to-scalar regression."""

    ALPHA = 0.1

    @property
    def name(self) -> str:
        return "image_conformal"

    @property
    def description(self) -> str:
        return "Conformal prediction on images: Split/CQR + MAPIE/crepes/torchcp"

    @property
    def supported_datasets(self) -> list[str]:
        # Scalar targets only (multitarget → keypoint_regression).
        return [
            "synthetic_image_regression",
            "imdb_wiki",
            "utkface_age",
            "utkface_age_tiny",
            "boneage",
        ]

    @property
    def canonical_scenarios(self) -> list[str]:
        return []

    def methods(self) -> list[str]:
        m = ["Split+CNN", "CQR+CNN"]
        # torchregress predictor-zoo rows (post-hoc classes on CNN bases).
        # Zoo_Local+CNN parked: representation-space LVD strands far test
        # points at tiny-smoke cal sizes → unbounded/inf intervals by design
        # (tabular Zoo_Local passes; re-enable for full-scale GPU runs).
        m.extend(
            [
                "Zoo_Split+CNN",
                "Zoo_Distributional+CNN",
                "Zoo_Density+CNN",
                "Zoo_Prevalence+CNN",
                "Zoo_UACQR+CNN",
            ]
        )
        if _MAPIE_AVAILABLE:
            m.append("MAPIE_Split+CNN")
        if _CREPES_AVAILABLE:
            m.append("crepes_Split+CNN")
        if _TORCHCP_AVAILABLE:
            m.append("torchcp_Split+CNN")
        return m

    def library_name(self, method: str) -> str:
        if method.startswith("MAPIE_"):
            return "MAPIE"
        if method.startswith("crepes_"):
            return "crepes"
        if method.startswith("torchcp_"):
            return "torchcp"
        return "torchregress"

    def library_version(self, method: str) -> str:
        if method.startswith("MAPIE_"):
            try:
                from importlib.metadata import version as _v

                from harness.core import pip_git_commit

                ver = _v("mapie")
                c = pip_git_commit("mapie")
                return f"{ver} (git {c})" if c else ver
            except Exception:
                return ""
        return ""

    def method_notes(self, method: str) -> str:
        return {
            "Split+CNN": "CNN point predictor + split-conformal residual correction (α=0.1)",
            "CQR+CNN": "CNN 3-quantile head + CQR calibration on cal split",
            "Zoo_Split+CNN": "torchregress SplitConformal class on CNN point base",
            "Zoo_Distributional+CNN": "torchregress DistributionalConformal (PIT-band) on CNN Gaussian base",
            "Zoo_Density+CNN": "torchregress DensityConformal on CNN point base",
            "Zoo_Prevalence+CNN": "torchregress PrevalenceAdjustedCP on CNN point base",
            "Zoo_Local+CNN": "torchregress LocalConformal (median-heuristic band) on CNN point base",
            "Zoo_UACQR+CNN": "torchregress UACQR on CNN quantile base",
            "MAPIE_Split+CNN": "MAPIE SplitConformalRegressor post-hoc on CNN residuals",
            "crepes_Split+CNN": "crepes conformal regressor post-hoc on CNN residuals",
            "torchcp_Split+CNN": "torchcp split predictor post-hoc on CNN residuals",
        }.get(method, "")

    def fit(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        set_harness_seed(config.seed)
        if method.startswith(("MAPIE_", "crepes_", "torchcp_")):
            base = self._fit_base("Split+CNN", splits, config)
            return {"type": "external_posthoc", "base": base, "method": method}
        if method.startswith("Zoo_"):
            return self._fit_zoo(method, splits, config)
        return self._fit_base(method, splits, config)

    def _fit_zoo(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        """Fit a torchregress predictor-zoo row on a CNN base (mirrors tabular)."""
        import torch

        from harness.devices import current_device

        kind = method.split("+")[0]  # Zoo_Split, Zoo_Distributional, ...
        device = current_device()
        if kind == "Zoo_UACQR":
            from torchregress.losses.conformal import UACQR

            base = self._fit_base("CQR+CNN", splits, config)
            model, quantiles = base["model"], base["quantiles"]
            X_cal = np.asarray(base["X_cal"])
            y_cal = torch.from_numpy(np.asarray(base["y_cal"]).reshape(-1, 1)).float().to(device)
            loader = _image_to_dataloader(X_cal, y=None, batch_size=64, shuffle=False)
            outs = []
            with torch.no_grad():
                for (xb,) in loader:
                    o = model(xb.to(device)).cpu()
                    outs.append(o[:, [0, -1]])  # outer quantiles only (UACQR band)
            p_cal = torch.cat(outs, dim=0).to(device)
            predictor = UACQR(alpha=self.ALPHA)
            predictor.calibrate(p_cal, y_cal)
            return {"type": "zoo", "model": model, "predictor": predictor, "kind": kind}
        if kind == "Zoo_Distributional":
            from torchregress.losses import GaussianNLLLoss
            from torchregress.losses.conformal import DistributionalConformal

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
            X_cal = np.asarray(carved["X_cal"])
            y_cal = torch.from_numpy(np.asarray(carved["y_cal"]).reshape(-1, 1)).float().to(device)
            cloader = _image_to_dataloader(X_cal, y=None, batch_size=64, shuffle=False)
            outs = []
            with torch.no_grad():
                for (xb,) in cloader:
                    outs.append(model(xb.to(device)))
            out_cal = torch.cat(outs, dim=0)
            mean_c = out_cal[:, 0:1]
            std_c = torch.exp(out_cal[:, 1:2]).sqrt().clamp_min(1e-6)
            from torch.distributions import Normal as _Normal

            f_cal = _Normal(mean_c, std_c).cdf(y_cal)
            predictor = DistributionalConformal(alpha=self.ALPHA)
            predictor.calibrate(f_cal.detach(), y_cal)
            return {
                "type": "zoo_distributional",
                "model": model,
                "predictor": predictor,
                "kind": kind,
            }
        # Point-base zoo rows (Split / Density / Prevalence / Local).
        # Local uses CNN representation-space features (penultimate layer),
        # not raw pixels: pixel-space kernels strand test points (curse of
        # dimensionality) → unbounded intervals by design.
        from torchregress.losses.conformal import (
            DensityConformal,
            LocalConformal,
            PrevalenceAdjustedCP,
            SplitConformal,
        )

        base = self._fit_base("Split+CNN", splits, config)
        model = base["model"]
        X_cal = np.asarray(base["X_cal"])
        y_cal = torch.from_numpy(np.asarray(base["y_cal"]).reshape(-1, 1)).float().to(device)
        p_cal = (
            torch.from_numpy(self._forward_point(model, X_cal)).float().reshape(-1, 1).to(device)
        )
        if kind == "Zoo_Local":
            feats_cal = self._cnn_features(model, X_cal).to(device)
            mu = feats_cal.mean(dim=0, keepdim=True)
            sd = feats_cal.std(dim=0, keepdim=True).clamp_min(1e-8)
            zc = (feats_cal - mu) / sd
            with torch.no_grad():
                d = torch.cdist(zc[:256], zc[:256])
                # p90 (not median): representation-space neighborhoods must
                # cover far test points or LVD returns unbounded/inf by design.
                bw = float(torch.quantile(d, 0.9).clamp_min(0.5))
            predictor = LocalConformal(alpha=self.ALPHA, bandwidth=bw)
            predictor.calibrate(p_cal, y_cal, x=zc)
            return {
                "type": "zoo",
                "model": model,
                "predictor": predictor,
                "kind": kind,
                "kernel_mu": mu,
                "kernel_sd": sd,
            }
        predictor = {
            "Zoo_Split": SplitConformal,
            "Zoo_Density": DensityConformal,
            "Zoo_Prevalence": PrevalenceAdjustedCP,
        }[kind](alpha=self.ALPHA)
        predictor.calibrate(p_cal, y_cal)
        return {"type": "zoo", "model": model, "predictor": predictor, "kind": kind}

    @staticmethod
    def _cnn_features(model: nn.Module, X: np.ndarray) -> torch.Tensor:
        """Penultimate CNN features (AdaptiveAvgPool output, 64-d)."""
        from ._base import _image_to_dataloader

        device = next(model.parameters()).device
        loader = _image_to_dataloader(X, y=None, batch_size=64, shuffle=False)
        feats = []
        with torch.no_grad():
            for (xb,) in loader:
                feats.append(model.features(xb.to(device)).flatten(1).cpu())
        return torch.cat(feats, dim=0)

    def _fit_base(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        in_channels, _, _ = self._image_shape(splits)
        carved = _carve_image_calibration_split(splits, cal_fraction=0.2, seed=config.seed)
        fit_splits = {
            "X_train": carved["X_train"],
            "y_train": carved["y_train"],
            "X_test": carved["X_train"][:0],
            "y_test": carved["y_train"][:0],
        }
        train_loader = self._make_train_loader(fit_splits, batch_size=config.batch_size)
        if method == "Split+CNN":
            model = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(model, nn.MSELoss(), train_loader, config)
            model.eval()
            q = self._cal_residual_quantile(model, carved)
            return {
                "type": "split",
                "model": model,
                "qhat": q,
                "X_cal": np.asarray(carved["X_cal"]),
                "y_cal": np.asarray(carved["y_cal"]).reshape(-1),
            }
        if method == "CQR+CNN":
            from torchregress.losses import MultiQuantileLoss

            quantiles = [self.ALPHA / 2, 0.5, 1.0 - self.ALPHA / 2]
            model = self._build_cnn(in_channels=in_channels, out_dim=3)
            _train_image_model(model, MultiQuantileLoss(quantiles=quantiles), train_loader, config)
            model.eval()
            corr = self._cal_cqr_correction(model, carved)
            return {
                "type": "cqr",
                "model": model,
                "quantiles": quantiles,
                "corr": corr,
                "X_cal": np.asarray(carved["X_cal"]),
                "y_cal": np.asarray(carved["y_cal"]).reshape(-1),
            }
        raise ValueError(f"Unknown method: {method}")

    def _forward_point(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X, y=None, batch_size=64, shuffle=False)
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                outs.append(model(xb.to(device)).cpu().numpy().reshape(-1))
        return np.concatenate(outs)

    def _cal_residual_quantile(self, model: nn.Module, carved: Any) -> float:
        pred = self._forward_point(model, np.asarray(carved["X_cal"]))
        resid = np.abs(np.asarray(carved["y_cal"]).reshape(-1) - pred.reshape(-1))
        return _conformal_quantile(resid, alpha=self.ALPHA)

    def _cal_cqr_correction(self, model: nn.Module, carved: Any) -> float:
        device = next(model.parameters()).device
        loader = _image_to_dataloader(
            np.asarray(carved["X_cal"]), y=None, batch_size=64, shuffle=False
        )
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                outs.append(model(xb.to(device)).cpu().numpy())
        arr = np.concatenate(outs, axis=0)
        y = np.asarray(carved["y_cal"]).reshape(-1)
        scores = np.maximum(arr[:, 0] - y, y - arr[:, -1])
        return _conformal_quantile(scores, alpha=self.ALPHA)

    def predict(
        self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig
    ) -> Any:
        if model_state.get("type") == "external_posthoc":
            return self._predict_posthoc(model_state, X_test)
        if model_state.get("type") in ("zoo", "zoo_distributional"):
            return self._predict_zoo(model_state, X_test)
        t = model_state["type"]
        if t == "split":
            pred = self._forward_point(model_state["model"], X_test)
            q = float(model_state["qhat"])
            return {"lower": pred - q, "upper": pred + q, "y_pred": pred}
        if t == "cqr":
            device = next(model_state["model"].parameters()).device
            loader = _image_to_dataloader(X_test, y=None, batch_size=64, shuffle=False)
            outs = []
            with torch.no_grad():
                for (xb,) in loader:
                    outs.append(model_state["model"](xb.to(device)).cpu().numpy())
            arr = np.concatenate(outs, axis=0)
            c = float(model_state["corr"])
            return {
                "lower": arr[:, 0] - c,
                "upper": arr[:, -1] + c,
                "y_pred": arr[:, 1],
            }
        raise ValueError(f"Unknown model type: {t}")

    def _predict_zoo(self, state: Any, X_test: np.ndarray) -> Any:
        """Predict intervals via a fitted torchregress predictor (CNN base)."""
        import numpy as _np

        model = state["model"]
        predictor = state["predictor"]
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=64, shuffle=False)
        if state["type"] == "zoo_distributional":
            from torch.distributions import Normal as _Normal

            outs = []
            with torch.no_grad():
                for (xb,) in loader:
                    outs.append(model(xb.to(device)))
            out = torch.cat(outs, dim=0)
            mean_t = out[:, 0:1]
            std_t = torch.exp(out[:, 1:2]).sqrt().clamp_min(1e-6)

            def _icdf(levels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                d = _Normal(mean_t, std_t)
                return torch.stack(
                    [d.icdf(levels[0]).reshape(-1), d.icdf(levels[1]).reshape(-1)], dim=1
                )

            Xt = torch.from_numpy(np.asarray(X_test)).float().to(device)
            lo, hi = predictor.predict_intervals_from_cdf(_icdf, Xt)
            pred = mean_t.reshape(-1).detach().cpu().numpy()
            return {
                "lower": _np.asarray(lo.detach().cpu()).reshape(-1),
                "upper": _np.asarray(hi.detach().cpu()).reshape(-1),
                "y_pred": pred,
                "mean": pred,
            }
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                outs.append(model(xb.to(device)))
        out = torch.cat(outs, dim=0)
        if state.get("kind") == "Zoo_UACQR":
            p_test = out[:, [0, -1]]  # outer-quantile band (matches calibrate)
            pred = out[:, 1].reshape(-1).detach().cpu().numpy()  # median point
        elif out.dim() == 1:
            p_test = out.reshape(-1, 1)
            pred = None
        else:
            p_test = out[:, 0:1]
            pred = None
        if pred is None:
            pred = self._forward_point(model, np.asarray(X_test))
        if state.get("kind") == "Zoo_Local":
            feats = self._cnn_features(state["model"], np.asarray(X_test)).to(device)
            mu = state["kernel_mu"].to(device)
            sd = state["kernel_sd"].to(device)
            lo, hi = predictor.predict_interval(p_test, x=(feats - mu) / sd)
        else:
            lo, hi = predictor.predict_interval(p_test)
        return {
            "lower": _np.asarray(lo.detach().cpu()).reshape(-1),
            "upper": _np.asarray(hi.detach().cpu()).reshape(-1),
            "y_pred": pred,
        }

    def _predict_posthoc(self, state: Any, X_test: np.ndarray) -> Any:
        """Real external-lib post-hoc calls on the fitted CNN base.

        Each branch imports and calls its library (no silent qhat reuse):
        - MAPIE: SplitConformalRegressor with a prefit CNN wrapper.
        - crepes: WrapRegressor around the CNN wrapper, calibrated.
        - torchcp: SplitPredictor on torch loaders with the CNN module.
        Any API mismatch raises (fail-fast) instead of silently emulating.
        """
        from ._base import _image_to_dataloader

        base = state["base"]
        method = state["method"]
        model = base["model"]
        model.eval()

        def _cnn_point(X: np.ndarray) -> np.ndarray:
            return self._forward_point(model, X)

        # Need carved cal split: stored on base at fit time.
        X_cal = np.asarray(base.get("X_cal")) if base.get("X_cal") is not None else None
        y_cal = np.asarray(base.get("y_cal")).reshape(-1) if base.get("y_cal") is not None else None

        if method.startswith("MAPIE_"):
            if not _MAPIE_AVAILABLE:
                raise ImportError("mapie not installed")
            return self._predict_mapie(_cnn_point, X_cal, y_cal, X_test)
        if method.startswith("crepes_"):
            if not _CREPES_AVAILABLE:
                raise ImportError("crepes not installed")
            return self._predict_crepes(_cnn_point, X_cal, y_cal, X_test)
        if method.startswith("torchcp_"):
            if not _TORCHCP_AVAILABLE:
                raise ImportError("torchcp not installed")
            return self._predict_torchcp(model, X_cal, y_cal, X_test)
        # Native fallback (should not happen): reuse qhat honestly labeled.
        pred = _cnn_point(X_test)
        q = float(base["qhat"])
        return {"lower": pred - q, "upper": pred + q, "y_pred": pred}

    # -- real external implementations ------------------------------------

    def _predict_mapie(self, cnn_point: Any, X_cal: Any, y_cal: Any, X_test: Any) -> Any:
        """MAPIE SplitConformalRegressor with prefit CNN wrapper."""
        import numpy as np
        from mapie.regression import SplitConformalRegressor

        class _CNNWrapper:
            def fit(self, X: Any, y: Any) -> "_CNNWrapper":
                return self

            def predict(self, X: Any) -> np.ndarray:
                return np.asarray(cnn_point(np.asarray(X))).reshape(-1)

        # MAPIE needs 2D X; wrap images as (N,-1) but wrapper reshapes back.
        # Store original shapes via closure: cnn_point expects (N,C,H,W).
        Xc = np.asarray(X_cal)
        Xt = np.asarray(X_test)
        # Keep a decode path: flatten for MAPIE, unflatten in wrapper.
        C, H, W = Xt.shape[1], Xt.shape[2], Xt.shape[3]

        class _FlatCNNWrapper(_CNNWrapper):
            def predict(self, Xf: Any) -> np.ndarray:
                Xf = np.asarray(Xf)
                X = Xf.reshape(len(Xf), C, H, W).astype(np.float32)
                return super().predict(X)

        Xc_f = Xc.reshape(len(Xc), -1)
        Xt_f = Xt.reshape(len(Xt), -1)
        # Prefit path (MAPIE>=0.9): estimator already fitted (CNN).
        # conformity_score="absolute" must be explicit: the pinned MAPIE
        # default resolves inconsistently in prefit mode (tabular precedent).
        try:
            reg = SplitConformalRegressor(
                estimator=_FlatCNNWrapper(),
                confidence_level=0.9,
                conformity_score="absolute",
                prefit=True,
            )
            reg.conformalize(Xc_f, np.asarray(y_cal))
            _, intervals = reg.predict_interval(Xt_f)
        except TypeError:
            # Older MAPIE: fit_conformalize / predict API.
            reg = SplitConformalRegressor(
                estimator=_FlatCNNWrapper(),
                confidence_level=0.9,
                conformity_score="absolute",
                prefit=True,
            )
            reg.fit_conformalize(Xc_f, np.asarray(y_cal))
            _, intervals = reg.predict(Xt_f, alpha=0.1)
        lower = (
            np.asarray(intervals)[:, 0, 0]
            if np.asarray(intervals).ndim == 3
            else np.asarray(intervals)[:, 0]
        )
        upper = (
            np.asarray(intervals)[:, 1, 0]
            if np.asarray(intervals).ndim == 3
            else np.asarray(intervals)[:, 1]
        )
        pred = _FlatCNNWrapper().predict(Xt_f)
        return {"lower": lower, "upper": upper, "y_pred": pred}

    def _predict_crepes(self, cnn_point: Any, X_cal: Any, y_cal: Any, X_test: Any) -> Any:
        import numpy as np
        from crepes import WrapRegressor

        Xc = np.asarray(X_cal)
        Xt = np.asarray(X_test)
        C, H, W = Xt.shape[1], Xt.shape[2], Xt.shape[3]

        class _FlatCNN:
            def fit(self, X: Any, y: Any) -> "_FlatCNN":
                return self

            def predict(self, Xf: Any) -> np.ndarray:
                X = np.asarray(Xf).reshape(len(np.asarray(Xf)), C, H, W).astype(np.float32)
                return np.asarray(cnn_point(X)).reshape(-1)

        wr = WrapRegressor(_FlatCNN())
        # Fit is a no-op (CNN already trained); calibrate on residuals.
        wr.fit(Xc.reshape(len(Xc), -1), np.asarray(y_cal).reshape(-1))
        wr.calibrate(Xc.reshape(len(Xc), -1), np.asarray(y_cal).reshape(-1))
        intervals = wr.predict_int(Xt.reshape(len(Xt), -1), confidence=0.9)
        intervals = np.asarray(intervals)
        pred = _FlatCNN().predict(Xt.reshape(len(Xt), -1))
        return {"lower": intervals[:, 0], "upper": intervals[:, 1], "y_pred": pred}

    def _predict_torchcp(self, model: Any, X_cal: Any, y_cal: Any, X_test: Any) -> Any:
        import numpy as np
        import torch
        from torchcp.regression.predictors import SplitPredictor

        device = next(model.parameters()).device
        model.eval()
        # Build torch cal/test loaders in image layout.
        from ._base import _image_to_dataloader

        cal_loader = _image_to_dataloader(X_cal, np.asarray(y_cal), batch_size=64, shuffle=False)
        test_loader = _image_to_dataloader(X_test, y=None, batch_size=64, shuffle=False)
        predictor = SplitPredictor(model=model, device=device)
        predictor.calibrate(cal_loader, alpha=self.ALPHA)
        intervals = predictor.predict(test_loader)
        intervals = np.asarray(intervals)
        # torchcp returns (N,2) [lower, upper]; point = CNN forward.
        pred = self._forward_point(model, np.asarray(X_test))
        if intervals.ndim == 2 and intervals.shape[1] == 2:
            return {"lower": intervals[:, 0], "upper": intervals[:, 1], "y_pred": pred}
        # Fallback layout (N,1,2).
        intervals = intervals.reshape(len(pred), -1)
        return {"lower": intervals[:, 0], "upper": intervals[:, -1], "y_pred": pred}
