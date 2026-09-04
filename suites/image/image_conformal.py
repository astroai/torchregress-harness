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
            "MAPIE_Split+CNN": "MAPIE SplitConformalRegressor post-hoc on CNN residuals",
            "crepes_Split+CNN": "crepes conformal regressor post-hoc on CNN residuals",
            "torchcp_Split+CNN": "torchcp split predictor post-hoc on CNN residuals",
        }.get(method, "")

    def fit(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        set_harness_seed(config.seed)
        if method.startswith(("MAPIE_", "crepes_", "torchcp_")):
            base = self._fit_base("Split+CNN", splits, config)
            return {"type": "external_posthoc", "base": base, "method": method}
        return self._fit_base(method, splits, config)

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
            return {"type": "cqr", "model": model, "quantiles": quantiles, "corr": corr}
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
        try:
            reg = SplitConformalRegressor(
                estimator=_FlatCNNWrapper(), confidence_level=0.9, prefit=True
            )
            reg.conformalize(Xc_f, np.asarray(y_cal))
            _, intervals = reg.predict_interval(Xt_f)
        except TypeError:
            # Older MAPIE: fit_conformalize / predict API.
            reg = SplitConformalRegressor(
                estimator=_FlatCNNWrapper(), confidence_level=0.9, prefit=True
            )
            reg.fit_conformalize(Xc_f, np.asarray(y_cal))
            _, intervals = reg.predict(Xt_f, alpha=0.1)
        lower = np.asarray(intervals)[:, 0, 0] if np.asarray(intervals).ndim == 3 else np.asarray(intervals)[:, 0]
        upper = np.asarray(intervals)[:, 1, 0] if np.asarray(intervals).ndim == 3 else np.asarray(intervals)[:, 1]
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
