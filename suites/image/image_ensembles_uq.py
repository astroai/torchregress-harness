"""Image ensembles/UQ suite — DeepEnsemble/MC-Dropout/SWAG + LUQ-Box.

CNN backbones carry the torchregress UQ rows; LUQ-Box comparators run on
flattened pixels (documented model-family gap, mirroring the tabular
stress-matrix LUQ rows) so the suite works without a LUQ-CNN integration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import set_harness_seed
from harness.runners import HarnessConfig

from ._base import ImageHarnessRunner, _image_to_dataloader, _train_image_model

try:
    import lightning_uq_box  # noqa: F401

    _LUQBOX_AVAILABLE = True
except ImportError:
    _LUQBOX_AVAILABLE = False

_LUQBOX_PIN = "v0.3.0 (git df6366e02055618873f2cc8711b2dbc22e131a12)"


class ImageEnsemblesUQSuite(ImageHarnessRunner):
    """Ensembles + MC-Dropout + SWAG on images, with LUQ-Box comparators."""

    Z_SCORE = 1.645

    @property
    def name(self) -> str:
        return "image_ensembles_uq"

    @property
    def description(self) -> str:
        return "Image UQ: DeepEnsemble/MC-Dropout/SWAG vs LUQ-Box comparators"

    @property
    def supported_datasets(self) -> list[str]:
        # Scalar targets only (multitarget → keypoint_regression).
        return ["synthetic_image_regression", "imdb_wiki"]

    def methods(self) -> list[str]:
        m = ["DeepEnsemble_K5", "MC_Dropout", "SWAG"]
        if _LUQBOX_AVAILABLE:
            m.extend(["luqbox_MCDropout", "luqbox_GaussianNLL"])
        return m

    def library_name(self, method: str) -> str:
        return "lightning-uq-box" if method.startswith("luqbox_") else "torchregress"

    def library_version(self, method: str) -> str:
        if method.startswith("luqbox_"):
            try:
                from importlib.metadata import version as _v

                return f"{_v('lightning-uq-box')} (pin {_LUQBOX_PIN})"
            except Exception:
                return _LUQBOX_PIN
        return ""

    def method_notes(self, method: str) -> str:
        return {
            "DeepEnsemble_K5": "5× CNN + MSE; mean/var across members",
            "MC_Dropout": "dropout CNN (p=0.2), 20 MC passes at predict",
            "SWAG": "SWA-Gaussian approx over CNN weights (last-30% snapshots)",
            "luqbox_MCDropout": "LUQ-Box MCDropoutRegression on flattened pixels (dropout_p=0.5) — model-family gap vs CNN rows",
            "luqbox_GaussianNLL": "LUQ-Box DeterministicRegression+NLL on flattened pixels",
        }.get(method, "")

    def fit(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        set_harness_seed(config.seed)
        if method.startswith("luqbox_"):
            return self._fit_luqbox(method, splits, config)
        in_channels, _, _ = self._image_shape(splits)
        loader = self._make_train_loader(splits, batch_size=config.batch_size)
        if method == "DeepEnsemble_K5":
            members = []
            for i in range(5):
                set_harness_seed(config.seed + i)
                m = self._build_cnn(in_channels=in_channels, out_dim=1)
                _train_image_model(m, nn.MSELoss(), loader, config)
                m.eval()
                members.append(m)
            return {"type": "ensemble", "members": members}
        if method == "MC_Dropout":
            m = self._build_cnn_dropout(in_channels=in_channels, out_dim=1, dropout_p=0.2)
            _train_image_model(m, nn.MSELoss(), loader, config)
            return {"type": "mc_dropout", "model": m, "n_passes": 20}
        if method == "SWAG":
            from torch.optim.swa_utils import SWALR, AveragedModel

            m = self._build_cnn(in_channels=in_channels, out_dim=1)
            opt = torch.optim.Adam(m.parameters(), lr=config.lr)
            device = next(m.parameters()).device
            m.train()
            swa = AveragedModel(m)
            sched = SWALR(opt, swa_lr=config.lr)
            n_epochs = max(1, config.epochs)
            swa_start = max(1, int(0.7 * n_epochs))
            for ep in range(n_epochs):
                for batch in loader:
                    xb, yb = batch[0].to(device), batch[1].to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(m(xb).reshape(-1), yb.reshape(-1))
                    loss.backward()
                    opt.step()
                if ep >= swa_start:
                    swa.update_parameters(m)
                    sched.step()
            return {"type": "swag", "model": swa.module}
        raise ValueError(f"Unknown method: {method}")

    def _fit_luqbox(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        try:
            from lightning_uq_box.models import MLP as LUQMLP
            from lightning_uq_box.uq_methods import DeterministicRegression, MCDropoutRegression
            from lightning_uq_box.uq_methods.loss_functions import NLL as LUQNLL
        except ImportError as exc:
            raise ImportError("lightning-uq-box not installed") from exc
        import torch.nn as _nn

        Xtr = np.asarray(splits["X_train"]).reshape(len(splits["X_train"]), -1)
        ytr = np.asarray(splits["y_train"]).reshape(-1, 1).astype(np.float32)
        in_dim = Xtr.shape[1]
        Xt = torch.from_numpy(Xtr.astype(np.float32))
        yt = torch.from_numpy(ytr)
        ds = torch.utils.data.TensorDataset(Xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=True)
        dropout_p = 0.5 if method == "luqbox_MCDropout" else 0.0
        import lightning as pl

        base = LUQMLP(
            input_dim=in_dim,
            n_hidden=2,
            hidden_dim=64,
            dropout_p=dropout_p,
            activation_fn=_nn.ReLU(),
        )
        mod = (
            MCDropoutRegression(model=base, num_mc_samples=20, loss_fn=LUQNLL())
            if method == "luqbox_MCDropout"
            else DeterministicRegression(model=base, loss_fn=LUQNLL())
        )
        trainer = pl.Trainer(
            max_epochs=min(config.epochs, 10),
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(mod, train_dataloaders=dl)
        return {"type": "luqbox", "module": mod, "method": method}

    def predict(self, method: str, state: Any, X_test: np.ndarray, config: HarnessConfig) -> Any:
        t = state["type"]
        if t == "ensemble":
            return self._predict_ensemble(state["members"], X_test)
        if t == "mc_dropout":
            return self._predict_mc(state["model"], X_test, state.get("n_passes", 20))
        if t == "swag":
            return self._predict_point(state["model"], X_test)
        if t == "luqbox":
            return self._predict_luqbox(state, X_test)
        raise ValueError(f"Unknown model type: {t}")

    def _forward_all(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X, y=None, batch_size=64, shuffle=False)
        outs = []
        with torch.no_grad():
            for (xb,) in loader:
                outs.append(model(xb.to(device)).cpu().numpy().reshape(-1))
        return np.concatenate(outs)

    def _predict_point(self, model: nn.Module, X: np.ndarray) -> Any:
        pred = self._forward_all(model, X)
        return {"y_pred": pred}

    def _predict_ensemble(self, members: list[nn.Module], X: np.ndarray) -> Any:
        stack = np.stack([self._forward_all(m, X) for m in members], axis=1)
        mean, std = stack.mean(1), stack.std(1).clip(min=1e-6)
        return {
            "lower": mean - self.Z_SCORE * std,
            "upper": mean + self.Z_SCORE * std,
            "y_pred": mean,
            "mean": mean,
            "std": std,
            "samples": stack,
        }

    def _predict_mc(self, model: nn.Module, X: np.ndarray, n_passes: int) -> Any:
        model.train()
        stack = np.stack([self._forward_all(model, X) for _ in range(n_passes)], axis=1)
        model.eval()
        mean, std = stack.mean(1), stack.std(1).clip(min=1e-6)
        return {
            "lower": mean - self.Z_SCORE * std,
            "upper": mean + self.Z_SCORE * std,
            "y_pred": mean,
            "mean": mean,
            "std": std,
            "samples": stack,
        }

    def _predict_luqbox(self, state: Any, X_test: np.ndarray) -> Any:
        from lightning_uq_box.uq_methods import MCDropoutRegression

        mod = state["module"]
        Xf = torch.from_numpy(np.asarray(X_test).reshape(len(X_test), -1).astype(np.float32))
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xf), batch_size=64, shuffle=False
        )
        mod.eval()
        means, stds = [], []
        with torch.no_grad():
            for (xb,) in dl:
                if isinstance(mod, MCDropoutRegression):
                    mod.model.train()  # activate_dropout semantics
                    preds = torch.stack([mod.model(xb)[0] for _ in range(20)], dim=0)
                    mod.model.eval()
                    means.append(preds.mean(0).reshape(-1).cpu().numpy())
                    stds.append(preds.std(0).reshape(-1).cpu().numpy().clip(min=1e-6))
                else:
                    out = mod.model(xb)
                    loc = out[0] if isinstance(out, (tuple, list)) else out
                    means.append(np.asarray(loc.cpu()).reshape(-1))
                    stds.append(np.full_like(means[-1], 0.1))
        mean = np.concatenate(means)
        std = np.concatenate(stds)
        return {
            "lower": mean - self.Z_SCORE * std,
            "upper": mean + self.Z_SCORE * std,
            "y_pred": mean,
            "mean": mean,
            "std": std,
        }
