"""Age regression suite — image-to-scalar regression.

Trains and compares multiple regression methods on image datasets where
the target is a scalar value (e.g., age prediction from face images,
Gaussian blob centre coordinates from synthetic images).

Methods
-------
* ``CNN+GaussianNLL`` — lightweight CNN + diagonal Gaussian NLL
* ``CNN+MDN_K{2,3,5}`` — CNN + K-component mixture density network
* ``CNN+NormalizingFlow`` — CNN context + zuko NSF flow
* ``CNN+NF_Ensemble_K2`` — 2 independent NSF flows, pooled samples
* ``CNN+Evidential`` — CNN + NIG evidential regression
* ``CNN+MultiQuantile`` — CNN + 3-quantile head (0.05/0.5/0.95)
* ``CNN+StudentT`` — CNN + Student-t NLL (heavy-tailed)
* ``CNN+Huber`` / ``CNN+Barron`` — robust losses on CNN features
* ``CNN+MC_Dropout`` — dropout CNN + MSE, 20 MC passes
* ``ResNet+GaussianNLL/MDN_K3/NormalizingFlow/Evidential/MultiQuantile``
* ``timm_resnet18+GaussianNLL`` — timm backbone (if timm installed)
* ``sklearn_RF`` / ``sklearn_GBR`` — flattened-pixel baselines
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
    _TIMM_AVAILABLE,
    ImageHarnessRunner,
    _image_to_dataloader,
    _make_resnet_backbone,
    _make_timm_backbone,
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
        # Scalar targets only — multitarget (biwi_head_pose) lives in
        # keypoint_regression (scalar CNN heads cannot fit (N, D) targets).
        return [
            "synthetic_image_regression",
            "imdb_wiki",
            "imdb_wiki_fullres",
            "utkface_age",
            "utkface_age_tiny",
            "appa_age",
            "appa_age_tiny",
            "boneage",
        ]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    MDN_K_VARIANTS = ("CNN+MDN_K2", "CNN+MDN_K3", "CNN+MDN_K5")

    def methods(self) -> list[str]:
        methods = [
            "CNN+GaussianNLL",
            "CNN+BetaNLL",
            "CNN+FaithfulGaussian",
            "CNN+PoissonGaussian",
            "CNN+ObservedStdNLL",
            "CNN+SLS",
            "CNN+MDN_K2",
            "CNN+MDN_K3",
            "CNN+MDN_K5",
            "CNN+NormalizingFlow",
            "CNN+NF_Ensemble_K2",
            "CNN+Evidential",
            "CNN+MultiQuantile",
            "CNN+StudentT",
            "CNN+Huber",
            "CNN+Barron",
            "CNN+MC_Dropout",
            "sklearn_RF",
            "sklearn_GBR",
        ]
        if _RESNET_AVAILABLE:
            methods.extend(
                [
                    "ResNet+GaussianNLL",
                    "ResNet+MDN_K3",
                    "ResNet+NormalizingFlow",
                    "ResNet+Evidential",
                    "ResNet+MultiQuantile",
                ]
            )
        if _TIMM_AVAILABLE:
            methods.append("timm_resnet18+GaussianNLL")
        if _TORCH_UNCERTAINTY_AVAILABLE:
            methods.append("torch_uncertainty")
        return methods

    def library_name(self, method: str) -> str:
        if method.startswith("sklearn_"):
            return "scikit-learn"
        if method == "torch_uncertainty":
            return "torch-uncertainty"
        if method.startswith("timm_"):
            return "timm"
        return "torchregress"

    def library_version(self, method: str) -> str:
        """Version of the external library backing a method row (torch-uncertainty)."""
        if method == "torch_uncertainty":
            return _torch_uncertainty_version()
        if method.startswith("timm_"):
            try:
                from importlib.metadata import version as _md_version

                ver = _md_version("timm")
            except Exception:
                return ""
            commit = pip_git_commit("timm")
            return f"{ver} (git {commit})" if commit else ver
        return ""

    def method_notes(self, method: str) -> str:
        notes = {
            "CNN+GaussianNLL": "4-layer CNN backbone + diagonal Gaussian NLL loss",
            "CNN+BetaNLL": "4-layer CNN + BetaNLL(beta=0.5) anti-collapse variant (same 2-head)",
            "CNN+FaithfulGaussian": "4-layer CNN + Faithful MSE/NLL-decoupled Gaussian (same 2-head)",
            "CNN+PoissonGaussian": "4-layer CNN + Poisson-Gaussian imaging-noise NLL (log-lambda head, point)",
            "CNN+ObservedStdNLL": "4-layer CNN + NoisyTarget NLL on observed rater std (APPA-REAL aleatoric gold)",
            "CNN+SLS": "4-layer CNN context + SLS min-volume level-set regions (tau=0.9)",
            "CNN+MDN_K2": "4-layer CNN + 2-component MDN (K-sweep minimum)",
            "CNN+MDN_K3": "4-layer CNN backbone + 3-component mixture density network",
            "CNN+MDN_K5": "4-layer CNN + 5-component MDN (K-sweep mid)",
            "CNN+NormalizingFlow": "4-layer CNN context + zuko NSF normalizing flow",
            "CNN+NF_Ensemble_K2": "2 independent NSF flows, pooled samples (K-sweep minimum)",
            "CNN+Evidential": "4-layer CNN backbone + NIG evidential regression",
            "CNN+MultiQuantile": "4-layer CNN + 3-quantile head (0.05/0.5/0.95) → median & intervals",
            "CNN+StudentT": "4-layer CNN + Student-t NLL (heavy-tailed)",
            "CNN+Huber": "4-layer CNN + PseudoHuber robust loss (point prediction)",
            "CNN+Barron": "4-layer CNN + Barron adaptive robust loss (point prediction)",
            "CNN+MC_Dropout": "dropout CNN + MSE, 20 MC passes (train mode at predict)",
            "ResNet+GaussianNLL": "ResNet18 backbone + diagonal Gaussian NLL loss",
            "ResNet+MDN_K3": "ResNet18 backbone + 3-component MDN",
            "ResNet+NormalizingFlow": "ResNet18 context + zuko NSF normalizing flow",
            "ResNet+Evidential": "ResNet18 backbone + NIG evidential regression",
            "ResNet+MultiQuantile": "ResNet18 + 3-quantile head (0.05/0.5/0.95)",
            "timm_resnet18+GaussianNLL": "timm resnet18 backbone + Gaussian NLL (unpretrained)",
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
        if method.startswith("CNN+"):
            return self._fit_cnn_method(method, in_channels, train_loader, config)

        # --- ResNet-based methods ---
        if method.startswith("ResNet+"):
            return self._fit_resnet_method(method, in_channels, train_loader, config)

        # --- timm methods ---
        if method.startswith("timm_"):
            return self._fit_timm_method(method, in_channels, train_loader, config)

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

        # --- BetaNLL / FaithfulGaussian (drop-in 2-head anti-collapse variants) ---
        elif method == "CNN+BetaNLL":
            from torchregress.losses import BetaNLLLoss

            model = self._build_cnn(in_channels=in_channels, out_dim=2)
            _train_image_model(model, BetaNLLLoss(beta=0.5), train_loader, config)
            return {"type": "gaussian_nll", "model": model}
        elif method == "CNN+FaithfulGaussian":
            from torchregress.losses import FaithfulGaussianLoss

            model = self._build_cnn(in_channels=in_channels, out_dim=2)
            _train_image_model(model, FaithfulGaussianLoss(), train_loader, config)
            return {"type": "gaussian_nll", "model": model}

        # --- PoissonGaussian (imaging shot+readout noise; log-lambda head) ---
        elif method == "CNN+PoissonGaussian":
            from torchregress.losses.poisson_gaussian import PoissonGaussianMixtureLoss

            model = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(
                model, PoissonGaussianMixtureLoss(log_input=True), train_loader, config
            )
            return {"type": "poisson_gaussian", "model": model}

        # --- ObservedStdNLL (APPA-REAL rater std as target variance) ---
        elif method == "CNN+ObservedStdNLL":
            return self._fit_observed_std(splits, in_channels, config)

        # --- SLS (min-volume level sets on CNN context) ---
        elif method == "CNN+SLS":
            from torchregress.losses.sls import SLSLoss

            context_dim = 32
            model = self._build_cnn(
                in_channels=in_channels, out_dim=context_dim, base_filters=32
            )
            loss_fn = SLSLoss(
                d=1,
                context_dim=context_dim,
                K=1,
                mode="full",
                hidden_dim=64,
                n_transforms=4,
                tau=0.9,
                warmup_steps=50,
            )
            _train_image_model(
                model, loss_fn, train_loader, config, loss_params=list(loss_fn.parameters())
            )
            model.eval()
            return {"type": "sls", "model": model, "loss_fn": loss_fn}

        # --- MDN K-sweep ---
        elif method in ("CNN+MDN_K2", "CNN+MDN_K3", "CNN+MDN_K5"):
            from torchregress.losses import MixtureDensityLoss

            k = int(method.rsplit("_K", 1)[1])
            loss_fn = MixtureDensityLoss(n_components=k, n_features=1)
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

        # --- NF ensemble K2 (2 independent flows, pooled samples) ---
        elif method == "CNN+NF_Ensemble_K2":
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            context_dim = 64
            members = []
            for _ in range(2):
                flow = create_flow_model(
                    n_features=1,
                    context_dim=context_dim,
                    flow_type="nsf",
                    n_transforms=3,
                    hidden_features=[64, 64],
                )
                model = self._build_cnn(
                    in_channels=in_channels, out_dim=context_dim, base_filters=32
                )
                loss_fn = NormalizingFlowLoss(flow=flow)
                _train_image_model(
                    model,
                    loss_fn,
                    train_loader,
                    config,
                    loss_params=list(flow.parameters()),
                )
            model.eval()
            members.append((model, loss_fn))
            return {"type": "nflow_ensemble", "members": members}

    def _fit_observed_std(
        self,
        splits: dict[str, np.ndarray],
        in_channels: int,
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Train NoisyTargetGaussianNLL on observed rater std (APPA-REAL).

        Falls back to plain GaussianNLL when the dataset carries no
        ``y_std_train`` (all non-APPA sets): same 2-head layout, so the
        predict path is shared.
        """
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        from harness.devices import current_device

        X = np.asarray(splits["X_train"])
        y = np.asarray(splits["y_train"]).reshape(-1, 1).astype(np.float32)
        if splits.get("y_std_train") is not None:
            std = np.asarray(splits["y_std_train"]).reshape(-1, 1).astype(np.float32)
            observed = True
        else:
            std = np.zeros_like(y)
            observed = False
        device = current_device()
        ds = TensorDataset(
            torch.from_numpy(X).float(), torch.from_numpy(y), torch.from_numpy(std)
        )
        loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)
        # NOTE: batches stay on CPU here; moved per-batch (see _base memory note).
        from torchregress.losses.uncertain_gt import NoisyTargetGaussianNLL

        model = self._build_cnn(in_channels=in_channels, out_dim=2)
        loss_fn = NoisyTargetGaussianNLL()
        loss_fn.to(device)
        model.train()
        opt = optim.Adam(model.parameters(), lr=config.lr)
        for _ in range(config.epochs):
            for Xb, yb, sb in loader:
                Xb, yb, sb = Xb.to(device), yb.to(device), sb.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(Xb)
                tvar = (sb**2) if observed else None
                loss = loss_fn(out, yb, target_variance=tvar)
                loss.backward()
                opt.step()
        model.eval()
        return {"type": "gaussian_nll", "model": model}

        # --- Evidential ---
        elif method == "CNN+Evidential":
            from torchregress.losses import EvidentialRegressionLoss

            model = self._build_cnn(in_channels=in_channels, out_dim=4)
            loss_fn = EvidentialRegressionLoss(coeff_nig=0.01)
            self._train_evidential_cnn(model, loss_fn, train_loader, config)
            return {"type": "evidential", "model": model, "loss_fn": loss_fn}

        # --- MultiQuantile ---
        elif method == "CNN+MultiQuantile":
            from torchregress.losses import MultiQuantileLoss

            quantiles = [0.05, 0.5, 0.95]
            model = self._build_cnn(in_channels=in_channels, out_dim=len(quantiles))
            _train_image_model(model, MultiQuantileLoss(quantiles=quantiles), train_loader, config)
            return {"type": "multi_quantile", "model": model, "quantiles": quantiles}

        # --- StudentT ---
        elif method == "CNN+StudentT":
            from torchregress.losses import StudentTLoss

            model = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(model, StudentTLoss(), train_loader, config)
            return {"type": "point", "model": model}

        # --- Robust (point) ---
        elif method == "CNN+Huber":
            from torchregress.losses import PseudoHuberLoss

            model = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(model, PseudoHuberLoss(), train_loader, config)
            return {"type": "point", "model": model}
        elif method == "CNN+Barron":
            from torchregress.losses import BarronLoss

            model = self._build_cnn(in_channels=in_channels, out_dim=1)
            _train_image_model(model, BarronLoss(), train_loader, config)
            return {"type": "point", "model": model}

        # --- MC Dropout ---
        elif method == "CNN+MC_Dropout":
            model = self._build_cnn_dropout(in_channels=in_channels, out_dim=1, dropout_p=0.2)
            _train_image_model(model, nn.MSELoss(), train_loader, config)
            return {"type": "mc_dropout", "model": model, "n_passes": 20}

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

        elif method == "ResNet+MDN_K3":
            from torchregress.losses import MixtureDensityLoss

            loss_fn = MixtureDensityLoss(n_components=3, n_features=1)
            resnet.fc = nn.Linear(resnet.fc.in_features, loss_fn.expected_output_size)
            from harness.devices import current_device

            resnet.to(current_device())
            _train_image_model(resnet, loss_fn, train_loader, config)
            return {"type": "mdn", "model": resnet, "loss_fn": loss_fn}

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

        elif method == "ResNet+Evidential":
            from torchregress.losses import EvidentialRegressionLoss

            resnet.fc = nn.Linear(resnet.fc.in_features, 4)
            from harness.devices import current_device

            resnet.to(current_device())
            loss_fn = EvidentialRegressionLoss(coeff_nig=0.01)
            self._train_evidential_cnn(resnet, loss_fn, train_loader, config)
            return {"type": "evidential", "model": resnet, "loss_fn": loss_fn}

        elif method == "ResNet+MultiQuantile":
            from torchregress.losses import MultiQuantileLoss

            quantiles = [0.05, 0.5, 0.95]
            resnet.fc = nn.Linear(resnet.fc.in_features, len(quantiles))
            from harness.devices import current_device

            resnet.to(current_device())
            _train_image_model(resnet, MultiQuantileLoss(quantiles=quantiles), train_loader, config)
            return {"type": "multi_quantile", "model": resnet, "quantiles": quantiles}

        raise ValueError(f"Unknown ResNet method: {method}")

    def _fit_timm_method(
        self,
        method: str,
        in_channels: int,
        train_loader: torch.utils.data.DataLoader,
        config: HarnessConfig,
    ) -> dict[str, Any]:
        """Fit a timm-backbone method (currently GaussianNLL on resnet18)."""
        if method != "timm_resnet18+GaussianNLL":
            raise ValueError(f"Unknown timm method: {method}")
        from torchregress.losses import GaussianNLLLoss

        model = _make_timm_backbone(
            model_name="resnet18", in_channels=in_channels, out_dim=2, pretrained=False
        )
        if model is None:
            raise ImportError("timm not available — cannot use timm methods")
        _train_image_model(model, GaussianNLLLoss(), train_loader, config)
        return {"type": "gaussian_nll", "model": model}

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

        # --- GaussianNLL (CNN or ResNet; also BetaNLL/Faithful 2-head rows) ---
        if model_type == "gaussian_nll":
            return self._predict_gaussian_nll(model_state["model"], X_test)

        # --- PoissonGaussian (log-lambda head → point rate) ---
        elif model_type == "poisson_gaussian":
            return self._predict_poisson_gaussian(model_state["model"], X_test)

        # --- SLS (boundary-sampled intervals, D=1) ---
        elif model_type == "sls":
            return self._predict_sls(model_state["model"], model_state["loss_fn"], X_test)

        # --- MDN ---
        elif model_type == "mdn":
            return self._predict_mdn(model_state["model"], model_state["loss_fn"], X_test)

        # --- NormalizingFlow (CNN or ResNet) ---
        elif model_type == "nflow":
            return self._predict_nflow(model_state["model"], model_state["flow"], X_test)

        # --- NF ensemble (pooled samples across members) ---
        elif model_type == "nflow_ensemble":
            return self._predict_nflow_ensemble(model_state["members"], X_test)

        # --- MultiQuantile ---
        elif model_type == "multi_quantile":
            return self._predict_multi_quantile(
                model_state["model"], model_state["quantiles"], X_test
            )

        # --- Point (StudentT/Huber/Barron heads: mean-only) ---
        elif model_type == "point":
            return self._predict_point(model_state["model"], X_test)

        # --- MC Dropout ---
        elif model_type == "mc_dropout":
            return self._predict_mc_dropout(
                model_state["model"], X_test, n_passes=int(model_state.get("n_passes", 20))
            )

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

    def _predict_poisson_gaussian(
        self,
        model: nn.Module,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """PoissonGaussian: exp(log-lambda) rate as point prediction.

        Point-only (no calibrated std): the loss models count noise, not a
        Gaussian predictive, so emitting a Gaussian std would mislabel NLL.
        """
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)
        preds = []
        with torch.no_grad():
            for (X_batch,) in loader:
                lam = torch.exp(model(X_batch.to(device)).reshape(-1)).clamp_min(1e-8)
                preds.append(lam.cpu().numpy())
        y_pred = np.concatenate(preds).reshape(-1)
        return {"y_pred": y_pred}

    def _predict_sls(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        X_test: np.ndarray,
        n_samples: int = 500,
    ) -> dict[str, np.ndarray]:
        """SLS D=1: boundary-sampled level-set intervals (scalar)."""
        import torch.nn.functional as F

        frontier = loss_fn.frontier
        quantile_net = loss_fn.quantile_net
        flow = frontier.flow
        model.eval()
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)
        lowers, uppers, means = [], [], []
        with torch.no_grad():
            for (Xb,) in loader:
                context = model(Xb.to(device))  # (B, ctx)
                q = quantile_net(context)[:, 1]  # (B,) tau-quantile
                mu, L_params = frontier._get_params(context)  # (B,1),(B,num_L)
                L_full, _ = frontier._get_L_matrix_and_logdet(L_params)  # (B,1,1)
                r = torch.randn(n_samples, context.shape[0], 1, device=device)
                r = F.normalize(r, dim=-1)
                r = r * torch.sqrt(q).view(1, -1, 1)
                z = torch.linalg.solve_triangular(
                    L_full.unsqueeze(0), r.unsqueeze(-1), upper=False
                ).squeeze(-1)
                z_flow = z + mu.unsqueeze(0)
                ctx_rep = (
                    context.unsqueeze(0).expand(n_samples, -1, -1).reshape(-1, context.shape[-1])
                )
                y_s = flow.inverse(z_flow.reshape(-1, 1), ctx_rep).reshape(n_samples, -1)
                y_np = y_s.cpu().numpy()
                lowers.append(y_np.min(axis=0))
                uppers.append(y_np.max(axis=0))
                means.append(mu.reshape(-1).cpu().numpy())
        lower = np.concatenate(lowers).reshape(-1)
        upper = np.concatenate(uppers).reshape(-1)
        mean = np.concatenate(means).reshape(-1)
        return {"lower": lower, "upper": upper, "y_pred": mean}

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

    def _predict_nflow_ensemble(
        self,
        members: list[tuple[nn.Module, nn.Module]],
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """NF ensemble: pool 128 samples per member → (N, 256)."""
        pools: list[np.ndarray] = []
        for model, flow_loss in members:
            device = next(model.parameters()).device
            loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)
            part = []
            with torch.no_grad():
                for (X_batch,) in loader:
                    context = model(X_batch.to(device))
                    s = flow_loss.sample(context, n_samples=128).squeeze(-1)
                    s_np = s.cpu().numpy()
                    if s_np.ndim == 2 and s_np.shape[0] == 128:
                        s_np = s_np.T
                    part.append(s_np)
            pools.append(np.concatenate(part, axis=0))
        samples_np = np.concatenate(pools, axis=1)
        return {
            "lower": np.quantile(samples_np, 0.05, axis=1),
            "upper": np.quantile(samples_np, 0.95, axis=1),
            "y_pred": samples_np.mean(axis=1),
            "samples": samples_np,
        }

    def _predict_multi_quantile(
        self, model: nn.Module, quantiles: list[float], X_test: np.ndarray
    ) -> dict[str, np.ndarray]:
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)
        outs = []
        with torch.no_grad():
            for (X_batch,) in loader:
                outs.append(model(X_batch.to(device)).cpu().numpy())
        arr = np.concatenate(outs, axis=0)
        return {
            "lower": arr[:, 0].reshape(-1),
            "upper": arr[:, -1].reshape(-1),
            "y_pred": arr[:, len(quantiles) // 2].reshape(-1),
        }

    def _predict_point(self, model: nn.Module, X_test: np.ndarray) -> dict[str, np.ndarray]:
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)
        outs = []
        with torch.no_grad():
            for (X_batch,) in loader:
                outs.append(model(X_batch.to(device)).cpu().numpy().reshape(-1))
        return {"y_pred": np.concatenate(outs).reshape(-1)}

    def _predict_mc_dropout(
        self, model: nn.Module, X_test: np.ndarray, n_passes: int = 20
    ) -> dict[str, np.ndarray]:
        device = next(model.parameters()).device
        loader = _image_to_dataloader(X_test, y=None, batch_size=32, shuffle=False)
        model.train()  # keep dropout active
        passes = []
        with torch.no_grad():
            for _ in range(n_passes):
                outs = []
                for (X_batch,) in loader:
                    outs.append(model(X_batch.to(device)).cpu().numpy().reshape(-1))
                passes.append(np.concatenate(outs))
        model.eval()
        samples = np.stack(passes, axis=1)
        mean = samples.mean(axis=1)
        std = samples.std(axis=1).clip(min=1e-6)
        z = self.Z_SCORE
        return {
            "lower": mean - z * std,
            "upper": mean + z * std,
            "y_pred": mean,
            "mean": mean,
            "std": std,
            "samples": samples,
        }

    def _predict_torch_uncertainty(
        self,
        routine: Any,
        X_test: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """torch-uncertainty: extract mean/std from Lightning routine."""
        import lightning as pl
        from torch.utils.data import DataLoader, Dataset

        from harness.devices import current_device

        # Lightning's default predict_step calls ``self(batch)`` with the raw
        # batch, so the loader must yield bare input tensors (no label tuples)
        # for the routine's forward() to receive a tensor.
        class _InputsOnly(Dataset):
            def __init__(self, arr: np.ndarray) -> None:
                self.data = torch.from_numpy(arr).float().to(current_device())

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
