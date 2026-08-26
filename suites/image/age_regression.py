"""Age regression suite — image-to-scalar regression.

Trains and compares multiple regression methods on image datasets where
the target is a scalar value (e.g., age prediction from face images,
Gaussian blob centre coordinates from synthetic images).

Methods
-------
* ``CNN+GaussianNLL`` — lightweight CNN + diagonal Gaussian NLL
* ``CNN+MDN_K3`` — CNN + 3-component mixture density network
* ``CNN+NormalizingFlow`` — CNN context + zuko NSF flow
* ``CNN+Evidential`` — CNN + NIG evidential regression
* ``ResNet+GaussianNLL`` — pretrained ResNet18 + Gaussian NLL (if torchvision)
* ``ResNet+NormalizingFlow`` — ResNet context + zuko NSF flow (if torchvision)
* ``sklearn_RF`` — RandomForestRegressor on flattened pixels (sklearn baseline)
* ``sklearn_GBR`` — GradientBoostingRegressor on flattened pixels (sklearn baseline)
* ``torch_uncertainty`` — torch-uncertainty RegressionRoutine (if available)

Evaluation
----------
Standard point + interval + distribution metrics via ``compute_all_metrics``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import pip_git_commit, set_harness_seed
from harness.runners import HarnessConfig

from ._base import (
    _RESNET_AVAILABLE,
    ImageHarnessRunner,
    _image_to_dataloader,
    _make_resnet_backbone,
    _train_image_model,
)

# ---------------------------------------------------------------------------
# torch-uncertainty optional import
# ---------------------------------------------------------------------------

_TORCH_UNCERTAINTY_AVAILABLE: bool
try:
    import torch_uncertainty  # noqa: F401

    _TORCH_UNCERTAINTY_AVAILABLE = True
except ImportError:
    _TORCH_UNCERTAINTY_AVAILABLE = False

# torch-uncertainty >= 0.13 (v0.13.0, 2026-07-22) reworked the regression
# routine's loss/metric contracts (breaking change): the routine now calls
# ``loss(distribution, targets)`` with a torch ``Distribution`` and a
# probabilistic model must return a dict of distribution parameters.  Older
# versions (v0.12.x) are NOT supported by this suite.  Pin from git — see
# ``EXTERNAL_PINS.md`` for the exact commit.
_TORCH_UNCERTAINTY_MIN_VERSION: tuple[int, int] = (0, 13)


def _require_torch_uncertainty_version() -> None:
    """Raise a clear ``ImportError`` when torch-uncertainty is missing or < 0.13."""
    import re

    try:
        import torch_uncertainty

        raw = getattr(torch_uncertainty, "__version__", None)
    except ImportError as exc:
        raise ImportError(
            "torch-uncertainty not installed — install with: "
            "pip install lightning 'torch-uncertainty>=0.13' "
            "(latest git pin: see EXTERNAL_PINS.md)"
        ) from exc
    if raw is None:
        return
    # Parse the leading major.minor (robust to "v" prefixes and PEP 440
    # suffixes such as "0.13.0.dev0" / "0.13.0+git...").  Unparseable
    # versions fail loudly rather than silently skipping the guard.
    m = re.match(r"\s*v?(\d+)\.(\d+)", raw)
    if m is None:
        raise ImportError(
            f"cannot parse torch-uncertainty version {raw!r} — cannot verify the "
            f">= {'.'.join(map(str, _TORCH_UNCERTAINTY_MIN_VERSION))}.0 requirement. "
            "Install from git main per EXTERNAL_PINS.md."
        )
    parts = (int(m.group(1)), int(m.group(2)))
    if parts < _TORCH_UNCERTAINTY_MIN_VERSION:
        raise ImportError(
            f"torch-uncertainty {raw} is too old for the image regression suite "
            f"(needs >= {'.'.join(map(str, _TORCH_UNCERTAINTY_MIN_VERSION))}.0 for the reworked "
            "DistributionNLLLoss / distribution-parameter API).  Install from git main: "
            "pip install 'git+https://github.com/torch-uncertainty/torch-uncertainty.git@94e511a737'"
        )


def _torch_uncertainty_version() -> str:
    """Installed torch-uncertainty version, annotated with the git commit if a pip git install."""
    try:
        import torch_uncertainty

        ver = getattr(torch_uncertainty, "__version__", "")
    except ImportError:
        return ""
    if not ver:
        try:
            from importlib.metadata import version as _md_version

            ver = _md_version("torch-uncertainty")
        except Exception:
            return "unknown"
    commit = pip_git_commit("torch-uncertainty")
    return f"{ver} (git {commit})" if commit else ver


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class AgeRegressionSuite(ImageHarnessRunner):
    """Image-to-scalar regression suite.

    Compares torchregress methods (GaussianNLL, MDN, NormalizingFlow,
    Evidential) with CNN and ResNet backbones against sklearn baselines
    on flattened pixels and torch-uncertainty (if installed).
    """

    Z_SCORE = 1.645  # 90% prediction interval

    @property
    def name(self) -> str:
        return "age_regression"

    @property
    def description(self) -> str:
        return (
            "Image-to-scalar regression: CNN/ResNet backbones with "
            "GaussianNLL, MDN, NF, Evidential vs sklearn baselines"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "synthetic_image_regression",
            "imdb_wiki",
        ]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        methods = [
            "CNN+GaussianNLL",
            "CNN+MDN_K3",
            "CNN+NormalizingFlow",
            "CNN+Evidential",
            "sklearn_RF",
            "sklearn_GBR",
        ]
        if _RESNET_AVAILABLE:
            methods.extend(["ResNet+GaussianNLL", "ResNet+NormalizingFlow"])
        if _TORCH_UNCERTAINTY_AVAILABLE:
            methods.append("torch_uncertainty")
        return methods

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn_"):
            return "scikit-learn"
        if method == "torch_uncertainty":
            return "torch-uncertainty"
        return "torchregress"

    def library_version(self, method: str) -> str:
        """Version of the external library backing a method row (torch-uncertainty)."""
        if method == "torch_uncertainty":
            return _torch_uncertainty_version()
        return ""

    def method_notes(self, method: str) -> str:
        notes = {
            "CNN+GaussianNLL": "4-layer CNN backbone + diagonal Gaussian NLL loss",
            "CNN+MDN_K3": "4-layer CNN backbone + 3-component mixture density network",
            "CNN+NormalizingFlow": "4-layer CNN context + zuko NSF normalizing flow",
            "CNN+Evidential": "4-layer CNN backbone + NIG evidential regression",
            "ResNet+GaussianNLL": "ResNet18 backbone + diagonal Gaussian NLL loss",
            "ResNet+NormalizingFlow": "ResNet18 context + zuko NSF normalizing flow",
            "sklearn_RF": "RandomForestRegressor on flattened pixels (100 trees) — point prediction only",
            "sklearn_GBR": "GradientBoostingRegressor on flattened pixels — point prediction only",
            "torch_uncertainty": "torch-uncertainty RegressionRoutine (>=0.13) with distributional output — git pin in EXTERNAL_PINS.md",
        }
        return notes.get(method, "")

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

        in_channels, img_h, img_w = self._image_shape(splits)
        train_loader = self._make_train_loader(splits, batch_size=config.batch_size)

        # --- CNN-based methods ---
        if method in ("CNN+GaussianNLL", "CNN+MDN_K3", "CNN+NormalizingFlow", "CNN+Evidential"):
            return self._fit_cnn_method(method, in_channels, train_loader, config)

        # --- ResNet-based methods ---
        if method.startswith("ResNet+"):
            return self._fit_resnet_method(method, in_channels, train_loader, config)

        # --- sklearn baselines on flattened pixels ---
        if method.startswith("sklearn_"):
            return self._fit_sklearn_method(method, splits, config)

        # --- torch-uncertainty ---
        if method == "torch_uncertainty":
            return self._fit_torch_uncertainty(method, in_channels, splits, config)

        raise ValueError(f"Unknown method: {method}")

    def _fit_cnn_method(
        self,
        method: str,
        in_channels: int,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit a CNN-based method."""
        # --- GaussianNLL ---
        if method == "CNN+GaussianNLL":
            from torchregress.losses import GaussianNLLLoss

            model = self._build_cnn(in_channels=in_channels, out_dim=2)
            _train_image_model(model, GaussianNLLLoss(), train_loader, config)
            return {"type": "gaussian_nll", "model": model}

        # --- MDN ---
        elif method == "CNN+MDN_K3":
            from torchregress.losses import MixtureDensityLoss

            loss_fn = MixtureDensityLoss(n_components=3, n_features=1)
            model = self._build_cnn(in_channels=in_channels, out_dim=loss_fn.expected_output_size)
            _train_image_model(model, loss_fn, train_loader, config)
            return {"type": "mdn", "model": model, "loss_fn": loss_fn}

        # --- NormalizingFlow ---
        elif method == "CNN+NormalizingFlow":
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            context_dim = 64
            flow = create_flow_model(
                n_features=1,
                context_dim=context_dim,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            model = self._build_cnn(in_channels=in_channels, out_dim=context_dim, base_filters=32)
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_image_model(
                model,
                loss_fn,
                train_loader,
                config,
                loss_params=list(flow.parameters()),
            )
            return {"type": "nflow", "model": model, "flow": loss_fn}

        # --- Evidential ---
        elif method == "CNN+Evidential":
            from torchregress.losses import EvidentialRegressionLoss

            model = self._build_cnn(in_channels=in_channels, out_dim=4)
            loss_fn = EvidentialRegressionLoss(coeff_nig=0.01)
            self._train_evidential_cnn(model, loss_fn, train_loader, config)
            return {"type": "evidential", "model": model, "loss_fn": loss_fn}

        raise ValueError(f"Unknown CNN method: {method}")

    def _fit_resnet_method(
        self,
        method: str,
        in_channels: int,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit a ResNet-based method."""
        resnet = _make_resnet_backbone(in_channels=in_channels, out_dim=1, pretrained=False)
        if resnet is None:
            raise ImportError("torchvision not available — cannot use ResNet methods")

        if method == "ResNet+GaussianNLL":
            from torchregress.losses import GaussianNLLLoss

            resnet.fc = nn.Linear(resnet.fc.in_features, 2)  # mean + log-var
            from harness.devices import current_device

            resnet.to(current_device())
            _train_image_model(resnet, GaussianNLLLoss(), train_loader, config)
            return {"type": "gaussian_nll", "model": resnet}

        elif method == "ResNet+NormalizingFlow":
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            context_dim = 64
            resnet.fc = nn.Linear(resnet.fc.in_features, context_dim)
            from harness.devices import current_device

            resnet.to(current_device())
            flow = create_flow_model(
                n_features=1,
                context_dim=context_dim,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_image_model(
                resnet,
                loss_fn,
                train_loader,
                config,
                loss_params=list(flow.parameters()),
            )
            return {"type": "nflow", "model": resnet, "flow": loss_fn}

        raise ValueError(f"Unknown ResNet method: {method}")

    def _fit_sklearn_method(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit an sklearn baseline on flattened pixel data."""
        X_train_flat = splits["X_train"].reshape(len(splits["X_train"]), -1)
        y_train = splits["y_train"].ravel()

        if method == "sklearn_RF":
            from sklearn.ensemble import RandomForestRegressor

            est = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=config.seed,
                n_jobs=-1,
            )
        elif method == "sklearn_GBR":
            from sklearn.ensemble import GradientBoostingRegressor

            est = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=config.seed,
            )
        else:
            raise ValueError(f"Unknown sklearn method: {method}")

        est.fit(X_train_flat, y_train)
        return {"type": "sklearn", "estimator": est}

    def _fit_torch_uncertainty(
        self,
        method: str,
        in_channels: int,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit a torch-uncertainty RegressionRoutine (v0.13.x API, git main)."""
        _require_torch_uncertainty_version()
        try:
            import lightning as pl
            from torch_uncertainty.losses import DistributionNLLLoss
            from torch_uncertainty.routines.regression import RegressionRoutine
        except ImportError as exc:
            raise ImportError(
                "torch-uncertainty or lightning not installed — install with: "
                "pip install lightning 'torch-uncertainty>=0.13'"
            ) from exc

        # v0.13 expects a probabilistic model to return a dict of distribution
        # parameters for dist_family="normal" — i.e. {"loc": mean, "scale": std}.
        # Our CNN outputs (B, 2) with [:, 0] = mean, [:, 1] = log-variance.
        class _TUWrapper(nn.Module):
            def __init__(self, backbone: nn.Module) -> None:
                super().__init__()
                self.backbone = backbone

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                out = self.backbone(x)
                mean = out[:, 0]
                std = torch.exp(0.5 * out[:, 1]).clamp_min(1e-6)
                return {"loc": mean, "scale": std}

        wrapped = _TUWrapper(self._build_cnn(in_channels=in_channels, out_dim=2))

        # v0.13's routine calls ``loss(dist, targets)`` with a torch Distribution,
        # so the loss must be torch-uncertainty's DistributionNLLLoss — a plain
        # ``nn.GaussianNLLLoss`` would receive a Distribution object and fail.
        # optim_recipe must call .parameters() on the model it receives.
        routine = RegressionRoutine(
            model=wrapped,
            output_dim=1,
            loss=DistributionNLLLoss(),
            dist_family="normal",
            optim_recipe=lambda m: torch.optim.Adam(m.parameters(), lr=config.lr),
        )

        train_loader = self._make_train_loader(splits, batch_size=config.batch_size)
        trainer = pl.Trainer(
            max_epochs=config.epochs,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(routine, train_dataloaders=train_loader)

        return {"type": "torch_uncertainty", "routine": routine}

    # ------------------------------------------------------------------
    # Custom evidential training for CNN
    # ------------------------------------------------------------------

    @staticmethod
    def _train_evidential_cnn(
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
    ) -> None:
        """Train evidential CNN with softplus constraints on nu, alpha, beta."""
        import torch.nn.functional as F
        import torch.optim as optim

        params = list(model.parameters())
        for p in loss_fn.parameters():
            if p.requires_grad:
                params.append(p)

        opt = optim.Adam(params, lr=config.lr)
        device = next(model.parameters()).device
        loss_fn.to(device)
        model.train()

        for _ in range(config.epochs):
            for batch in train_loader:
                X_batch = batch[0].to(device)
                y_batch = batch[1].to(device)
                opt.zero_grad(set_to_none=True)
                raw = model(X_batch)
                gamma = raw[:, 0:1]
                nu = F.softplus(raw[:, 1:2]) + 0.01
                alpha = F.softplus(raw[:, 2:3]) + 1.01
                beta = F.softplus(raw[:, 3:4]) + 0.01
                constrained = torch.cat([gamma, nu, alpha, beta], dim=1)
                loss = loss_fn(constrained, y_batch)
                loss.backward()
                opt.step()

        model.eval()

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
        model_type = model_state["type"]

        # --- GaussianNLL (CNN or ResNet) ---
        if model_type == "gaussian_nll":
            return self._predict_gaussian_nll(model_state["model"], X_test)

        # --- MDN ---
        elif model_type == "mdn":
            return self._predict_mdn(model_state["model"], model_state["loss_fn"], X_test)

        # --- NormalizingFlow (CNN or ResNet) ---
        elif model_type == "nflow":
            return self._predict_nflow(model_state["model"], model_state["flow"], X_test)

        # --- Evidential ---
        elif model_type == "evidential":
            return self._predict_evidential(model_state["model"], model_state["loss_fn"], X_test)

        # --- sklearn ---
        elif model_type == "sklearn":
            return self._predict_sklearn(model_state["estimator"], X_test)

        # --- torch-uncertainty ---
        elif model_type == "torch_uncertainty":
            return self._predict_torch_uncertainty(model_state["routine"], X_test)

        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_gaussian_nll(
        self,
        model: nn.Module,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """GaussianNLL: mean ± z*std intervals."""
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)

        means, stds = [], []
        with torch.no_grad():
            for (X_batch,) in loader:
                out = model(X_batch.to(device))
                means.append(out[:, 0].cpu().numpy())
                var = torch.exp(out[:, 1]).clamp_min(1e-8).cpu().numpy()
                stds.append(np.sqrt(var))

        mean = np.concatenate(means).reshape(-1)
        std = np.concatenate(stds).reshape(-1)
        z = self.Z_SCORE
        return {
            "lower": mean - z * std,
            "upper": mean + z * std,
            "y_pred": mean,
            "mean": mean,
            "std": std,
        }

    def _predict_mdn(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """MDN: sample-based quantile intervals."""
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)

        all_samples = []
        with torch.no_grad():
            for (X_batch,) in loader:
                out = model(X_batch.to(device))
                s = loss_fn.sample(out, n_samples=256).squeeze(-1)
                # sample() may return (n_samples, batch) or (batch, n_samples)
                s_np = s.cpu().numpy()
                if s_np.ndim == 2 and s_np.shape[0] == 256:
                    s_np = s_np.T  # force (batch, n_samples)
                all_samples.append(s_np)

        samples_np = np.concatenate(all_samples, axis=0)  # (N_test, 256)

        lo = np.quantile(samples_np, 0.05, axis=1)
        hi = np.quantile(samples_np, 0.95, axis=1)
        return {
            "lower": lo,
            "upper": hi,
            "y_pred": samples_np.mean(axis=1),
            "samples": samples_np,
        }

    def _predict_nflow(
        self,
        model: nn.Module,
        flow_loss: nn.Module,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """NormalizingFlow: sample-based quantile intervals."""
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)

        all_samples = []
        with torch.no_grad():
            for (X_batch,) in loader:
                context = model(X_batch.to(device))
                s = flow_loss.sample(context, n_samples=256).squeeze(-1)
                # sample() may return (n_samples, batch) or (batch, n_samples)
                s_np = s.cpu().numpy()
                if s_np.ndim == 2 and s_np.shape[0] == 256:
                    s_np = s_np.T  # force (batch, n_samples)
                all_samples.append(s_np)

        samples_np = np.concatenate(all_samples, axis=0)  # (N_test, 256)

        lo = np.quantile(samples_np, 0.05, axis=1)
        hi = np.quantile(samples_np, 0.95, axis=1)
        return {
            "lower": lo,
            "upper": hi,
            "y_pred": samples_np.mean(axis=1),
            "samples": samples_np,
        }

    def _predict_evidential(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Evidential: Student-t predictive intervals via NIG parameters."""
        import torch.nn.functional as F

        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)

        gammas, nus, alphas, betas = [], [], [], []
        with torch.no_grad():
            for (X_batch,) in loader:
                raw = model(X_batch.to(device))
                gammas.append(raw[:, 0].cpu().numpy())
                nus.append((F.softplus(raw[:, 1:2]) + 0.01).cpu().numpy().reshape(-1))
                alphas.append((F.softplus(raw[:, 2:3]) + 1.01).cpu().numpy().reshape(-1))
                betas.append((F.softplus(raw[:, 3:4]) + 0.01).cpu().numpy().reshape(-1))

        gamma = np.concatenate(gammas).reshape(-1)
        nu = np.concatenate(nus).reshape(-1)
        alpha = np.concatenate(alphas).reshape(-1)
        beta = np.concatenate(betas).reshape(-1)

        # Student-t: df=2α, loc=γ, scale²=β(1+ν)/(να)
        from scipy.stats import t as student_t

        df = 2.0 * alpha
        scale = np.sqrt(beta * (1.0 + nu) / (nu * alpha))
        lo = student_t.ppf(0.05, df=df, loc=gamma, scale=scale)
        hi = student_t.ppf(0.95, df=df, loc=gamma, scale=scale)

        return {
            "lower": lo,
            "upper": hi,
            "y_pred": gamma,
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
        }

    def _predict_sklearn(
        self,
        estimator: Any,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """sklearn: point prediction only (no uncertainty)."""
        X_flat = X_test.reshape(len(X_test), -1)
        y_pred = estimator.predict(X_flat).reshape(-1).astype(np.float32)
        # Point-only prediction — no uncertainty (Coverage/Width/NLL absent)
        return {"y_pred": y_pred}

    def _predict_torch_uncertainty(
        self,
        routine: Any,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """torch-uncertainty: extract mean/std from Lightning routine."""
        import lightning as pl
        from torch.utils.data import DataLoader, Dataset

        # Lightning's default predict_step calls ``self(batch)`` with the raw
        # batch, so the loader must yield bare input tensors (no label tuples)
        # for the routine's forward() to receive a tensor.
        class _InputsOnly(Dataset):
            def __init__(self, arr: np.ndarray) -> None:
                self.data = torch.from_numpy(arr).float()

            def __len__(self) -> int:
                return len(self.data)

            def __getitem__(self, idx: int) -> torch.Tensor:
                return self.data[idx]

        loader = DataLoader(_InputsOnly(X_test), batch_size=32, shuffle=False)
        trainer = pl.Trainer(
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        preds = trainer.predict(routine, dataloaders=loader)

        # v0.13's default predict path returns the routine's forward() output —
        # for a probabilistic normal routine that is a dict of distribution
        # parameters, e.g. {"loc": Tensor(N,), "scale": Tensor(N,)}.  (Older
        # dict keys 'mean'/'std' and raw tensors are accepted as fallbacks.)
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for p in preds:
            if isinstance(p, dict):
                loc = p.get("loc", p.get("mean"))
                scale = p.get("scale", p.get("std"))
                if loc is None:
                    raise RuntimeError(
                        "torch-uncertainty predict dict has no 'loc'/'mean' key; "
                        f"got keys={sorted(p)}"
                    )
                means.append(np.asarray(loc.detach().cpu()).reshape(-1))
                if scale is not None:
                    stds.append(np.maximum(np.asarray(scale.detach().cpu()).reshape(-1), 0.0))
            elif isinstance(p, torch.Tensor):
                arr = p.cpu().numpy()
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    means.append(arr[:, 0])
                    stds.append(np.exp(arr[:, 1] * 0.5))
                else:
                    means.append(arr.reshape(-1))

        mean = np.concatenate(means).reshape(-1)
        if stds:
            std = np.concatenate(stds).reshape(-1)
            z = self.Z_SCORE
            return {
                "lower": mean - z * std,
                "upper": mean + z * std,
                "y_pred": mean,
                "mean": mean,
                "std": std,
            }
        return {"y_pred": mean}
