"""LSST DESC Photometric Redshift Data Challenge benchmark.

All four official task sets and both simulations/scenarios are supported via
held-out splits of the official training files (the official test labels are
withheld).  Nine-band LSST+Roman magnitudes/errors are augmented with colours
and detection flags, then fed to torchregress point, distributional, EIV,
imbalanced and conformal heads plus TabM baselines.

Metrics cover the challenge's point diagnostics (RMSE, MAE, bias,
sigma_NMAD, catastrophic-outlier fraction) and predictive diagnostics
(NLL, CRPS, interval coverage, PIT).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.metrics import compute_all_metrics
from harness.runners import (
    HarnessConfig,
    HarnessRunner,
    _inverse_transform_predictions,
    _inverse_transform_y,
)
from suites._backbones import (
    _MLP,
    _carve_calibration_split,
    _gaussian_nll_predict,
    _hidden_size,
    _train_one,
)

_BANDS = 9
_N_BASE = 2 * _BANDS
_N_FEATURES = _N_BASE + _BANDS + 8 + 2
_METHODS = [
    "MSE_MLP",
    "GaussianNLL_MLP",
    "MDN_K3",
    "Quantile_MLP",
    "NormalizingFlow",
    "TabM_MSE",
    "TabM_GaussianNLL",
    "TabM_SplitCP",
    "SplitCP_MLP",
    "CQR_MLP",
    "MondrianCP_MLP",
    "CVPlus_MLP",
    "LocalCP_MLP",
    "NormalizedCP_MLP",
    "Evidential",
    "BetaNLL_MLP",
    "InputNoiseAug_MLP",
    "DensityWeighted_MLP",
    "BMC_MLP",
    "PZRecipe",
    "PZRecipe_TabM",
]
_QUANTILES = (0.05, 0.16, 0.5, 0.84, 0.95)
_ALPHA = 0.10


def _split_train_cal(n: int, seed: int, frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_cal = max(50, int(round(frac * n)))
    return np.sort(idx[n_cal:]), np.sort(idx[:n_cal])


def _i_band(raw: np.ndarray) -> np.ndarray:
    i = np.asarray(raw, dtype=np.float64)[:, 3]
    return np.where(np.isfinite(i), i, 30.0)


def _feature_noise(raw: np.ndarray) -> np.ndarray:
    """Per-feature Gaussian noise scale propagated from band magnitude errors."""
    raw = np.asarray(raw, dtype=np.float64)
    errs = raw[:, _BANDS:]
    band = np.nanmedian(np.where(np.isfinite(errs) & (errs > 0), errs, np.nan), axis=0)
    band = np.where(np.isfinite(band), band, 0.05)
    sigma = np.empty(_N_FEATURES, dtype=np.float64)
    sigma[:9] = band
    sigma[9:18] = 0.1 * band
    sigma[18:27] = 0.01
    sigma[27:35] = 2.0 * np.mean(band)
    sigma[35] = band[3]
    sigma[36] = 0.01
    return np.clip(sigma, 0.01, 0.3).astype(np.float32)


def _build_features(raw: np.ndarray) -> np.ndarray:
    """Expand 9 magnitudes + 9 errors into a finite colour/flag representation."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != _N_BASE:
        raise ValueError(f"expected [N, {_N_BASE}] raw PZ features, got {raw.shape}")
    mags = raw[:, :_BANDS]
    errs = raw[:, _BANDS:]
    det = np.isfinite(mags) & np.isfinite(errs) & (errs > 0)
    mag_f = np.where(det, mags, 30.0)
    err_f = np.where(det, errs, 1.0)
    colours = []
    # LSST adjacent colours (u-g ... z-y), Roman adjacent (Y-J, J-H),
    # and one LSST/Roman cross-colour.
    for i in range(5):
        colours.append(mag_f[:, i] - mag_f[:, i + 1])
    colours.append(mag_f[:, 6] - mag_f[:, 7])
    colours.append(mag_f[:, 7] - mag_f[:, 8])
    colours.append(mag_f[:, 5] - mag_f[:, 6])
    colours_arr = np.stack(colours, axis=1)
    # Missing-colour entries are set to zero (no colour information).
    for j, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (5, 6)]):
        colours_arr[:, j] = np.where(det[:, a] & det[:, b], colours_arr[:, j], 0.0)
    depth = mag_f[:, 3:4]  # i-band depth
    n_det = det.sum(axis=1, keepdims=True).astype(np.float64)
    out = np.concatenate([mag_f, err_f, det.astype(np.float64), colours_arr, depth, n_det], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class _FeatureScaler:
    def __init__(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


def _train_tabm(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    config: HarnessConfig,
    *,
    nll: bool = False,
) -> None:
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    n = len(X)
    bs = min(max(1024, int(config.batch_size)), n)
    for _ in range(int(config.epochs)):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            opt.zero_grad(set_to_none=True)
            out = model(X[idx])
            if nll:
                # [B, k, 2] -> independent Gaussian NLL per member.
                b, k, _ = out.shape
                mean = out[..., 0].reshape(-1)
                logvar = out[..., 1].reshape(-1).clamp(-20.0, 20.0)
                target = y[idx].reshape(-1, 1).expand(-1, k).reshape(-1)
                loss = 0.5 * (logvar + (target - mean).square() / logvar.exp()).mean()
            else:
                target = y[idx].reshape(-1, 1, 1)
                loss = (out - target).square().mean()
            loss.backward()
            opt.step()


def _tabm_model(n_features: int, *, d_out: int = 1) -> nn.Module:
    from tabm import TabM

    return TabM.make(
        n_num_features=n_features,
        d_out=d_out,
        k=8,
        n_blocks=2,
        d_block=256,
        dropout=0.0,
    )


class PZChallengeSuite(HarnessRunner):
    @property
    def name(self) -> str:
        return "pz_challenge"

    @property
    def description(self) -> str:
        return (
            "LSST DESC PZ Data Challenge task set 1: point and p(z) regression "
            "on 9-band LSST+Roman magnitudes/errors (held-out split)."
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "pz_challenge_cardinal_10yr",
            "pz_challenge_flagship_10yr",
            "pz_challenge_cardinal_1yr",
            "pz_challenge_flagship_1yr",
            "pz_challenge_taskset2_cardinal_10yr",
            "pz_challenge_taskset2_flagship_10yr",
            "pz_challenge_taskset3_cardinal_10yr",
            "pz_challenge_taskset3_flagship_10yr",
            "pz_challenge_taskset4_cardinal_10yr",
            "pz_challenge_taskset4_flagship_10yr",
            "pz_challenge_taskset2_cardinal_1yr",
            "pz_challenge_taskset2_flagship_1yr",
            "pz_challenge_taskset3_cardinal_1yr",
            "pz_challenge_taskset3_flagship_1yr",
            "pz_challenge_taskset4_cardinal_1yr",
            "pz_challenge_taskset4_flagship_1yr",
        ]

    def methods(self) -> list[str]:
        # Targeted reruns can restrict expensive methods without changing the
        # release-wide job list (used by CANFAR single-method validation jobs).
        requested = os.environ.get("TORCHREGRESS_PZ_METHODS", "").strip()
        if not requested:
            return list(_METHODS)
        wanted = {part.strip() for part in requested.split(",") if part.strip()}
        return [method for method in _METHODS if method in wanted]

    def library_name(self, method: str) -> str:
        if method.startswith("TabM"):
            return "tabm"
        return "torchregress"

    def library_version(self, method: str) -> str:
        if method.startswith("TabM"):
            import tabm

            return str(getattr(tabm, "__version__", "unknown"))
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        notes = {
            "MSE_MLP": "plain MLP on magnitude/error/colour/flag features; point MSE",
            "GaussianNLL_MLP": "MLP mean+logvar head; Gaussian NLL",
            "MDN_K3": "3-component mixture density head; samples/mean/std",
            "Quantile_MLP": "5-level multi-quantile head (5/16/50/84/95%); pinball loss",
            "NormalizingFlow": "zuko cNSF (neural spline flow) conditional p(z|x)",
            "NormalizingFlow_cINN": "cINN-style conditional RealNVP coupling flow (nflow-z proxy)",
            "TabM_MSE": "official TabM (k=8, 2 blocks, width 256), point MSE",
            "TabM_GaussianNLL": "TabM with mean/logvar heads, Gaussian NLL",
            "TabM_SplitCP": "TabM point predictions + split-conformal residual intervals",
            "SplitCP_MLP": "point MLP + split-conformal absolute-residual intervals",
            "CQR_MLP": "multi-quantile MLP + CQR calibration",
            "MondrianCP_MLP": "split CP with Mondrian groups on i-band magnitude",
            "CVPlus_MLP": "3-fold CV+ (Barber et al. 2021) point MLP intervals",
            "LocalCP_MLP": "locally weighted split CP (LVD) on MLP residuals",
            "NormalizedCP_MLP": "split CP normalized by Gaussian predictive std",
            "Evidential": "NIG evidential MLP; Student-t predictive intervals",
            "BetaNLL_MLP": "beta-NLL weighted Gaussian MLP",
            "InputNoiseAug_MLP": "EIV-style input-noise augmentation from per-band magnitude errors",
            "DensityWeighted_MLP": "label-density weighted MSE (imbalanced targets)",
            "BMC_MLP": "noise-adaptive Balanced MSE (BMC) for imbalanced targets",
            "PZRecipe": "two-stage FT-Transformer: CE+CDF+RnC-N quantile-bin p(z); no stage-2 TabM",
            "PZRecipe_TabM": "PZRecipe stage 1 + paper-style TabM stage-2 classifier (200 bins)",
        }
        return notes.get(method, "")

    def _fit_pz_recipe(
        self,
        method: str,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        in_dim: int,
        config: HarnessConfig,
        *,
        tabm_stage2: bool,
    ) -> dict[str, Any]:
        from torchregress.losses import PZRecipeLoss
        from torchregress.models import PZRecipe

        device = current_device()
        y_np = y_t.detach().cpu().numpy().reshape(-1)
        n_bins = 100
        q = np.linspace(0.0, 1.0, n_bins + 2)[1:-1]
        centers_np = np.quantile(y_np, q)
        if len(np.unique(centers_np)) < n_bins:
            centers_np = np.linspace(float(y_np.min()), float(y_np.max()), n_bins)
        centers = torch.as_tensor(centers_np, dtype=torch.float32, device=device)
        model = PZRecipe(
            in_dim,
            n_bins=n_bins,
            d_token=64,
            n_layers=2,
            n_heads=4,
            stage2_bins=200 if tabm_stage2 else n_bins,
        ).to(device)
        loss_fn = PZRecipeLoss(centers, ce_weight=0.5, cdf_weight=0.5, rnc_weight=0.1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=min(config.lr, 5e-4), weight_decay=1e-5)
        n = len(X_t)
        bs = min(max(1024, int(config.batch_size)), n)
        for _ in range(int(config.epochs)):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                opt.zero_grad(set_to_none=True)
                logits = model(X_t[idx], stage=1)
                features = model.encode(X_t[idx])
                loss = loss_fn(logits, y_t[idx].reshape(-1), features=features)
                loss.backward()
                opt.step()
        model.eval()
        if not tabm_stage2:
            return {"type": "pz_recipe", "model": model, "centers": centers}

        # Paper-style stage 2: freeze the FT-Transformer, train a TabM
        # classifier on its features and 200 redshift-quantile bins.
        from tabm import TabM

        stage2_bins = 200
        q2 = np.linspace(0.0, 1.0, stage2_bins + 2)[1:-1]
        centers2_np = np.quantile(y_np, q2)
        if len(np.unique(centers2_np)) < stage2_bins:
            centers2_np = np.linspace(float(y_np.min()), float(y_np.max()), stage2_bins)
        centers2 = torch.as_tensor(centers2_np, dtype=torch.float32, device=device)
        tabm = TabM.make(
            n_num_features=64,
            d_out=stage2_bins,
            k=8,
            n_blocks=2,
            d_block=256,
            dropout=0.0,
        ).to(device)
        opt2 = torch.optim.AdamW(tabm.parameters(), lr=min(config.lr, 5e-4), weight_decay=1e-4)
        stage2_epochs = int(config.epochs)
        for _ in range(stage2_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                with torch.no_grad():
                    features = model.encode(X_t[idx])
                    labels = (
                        torch.searchsorted(centers2, y_t[idx].reshape(-1)).clamp(1, stage2_bins) - 1
                    )
                opt2.zero_grad(set_to_none=True)
                out = tabm(features)  # [B, k, n_bins]
                loss = F.cross_entropy(
                    out.reshape(-1, stage2_bins), labels.repeat_interleave(out.shape[1])
                )
                loss.backward()
                opt2.step()
        tabm.eval()
        return {"type": "pz_recipe_tabm", "model": model, "stage2": tabm, "centers": centers2}

    def fit(self, method: str, splits: dict[str, Any], config: HarnessConfig) -> dict[str, Any]:
        set_harness_seed(config.seed)
        feats = _build_features(splits["X_train"])
        scaler = _FeatureScaler(feats)
        X = scaler(feats)
        self._scaler = scaler
        y = np.asarray(splits["y_train"], dtype=np.float32).reshape(-1)
        X_t = torch.from_numpy(X).float().to(current_device())
        y_t = torch.from_numpy(y).float().to(current_device()).reshape(-1, 1)
        in_dim = X_t.shape[1]
        hidden = _hidden_size(in_dim)

        if method in {"PZRecipe", "PZRecipe_TabM"}:
            return self._fit_pz_recipe(
                method, X_t, y_t, in_dim, config, tabm_stage2=method == "PZRecipe_TabM"
            )
        if method == "MSE_MLP":
            model = _MLP(in_dim, hidden, 1).to(current_device())
            _train_one(
                model, nn.MSELoss(), X_t, y_t, config, batch_size=max(1024, int(config.batch_size))
            )
            model.eval()
            return {"type": "point", "model": model}
        if method == "GaussianNLL_MLP":
            from torchregress.losses import GaussianNLLLoss

            model = _MLP(in_dim, hidden, 2).to(current_device())
            _train_one(
                model,
                GaussianNLLLoss(),
                X_t,
                y_t,
                config,
                batch_size=max(1024, int(config.batch_size)),
            )
            model.eval()
            return {"type": "gaussian", "model": model}
        if method == "MDN_K3":
            from torchregress.losses import MixtureDensityLoss

            model = _MLP(in_dim, hidden, 9).to(current_device())
            loss_fn = MixtureDensityLoss(n_components=3, n_features=1)
            _train_one(
                model, loss_fn, X_t, y_t, config, batch_size=max(1024, int(config.batch_size))
            )
            model.eval()
            return {"type": "mdn", "model": model, "loss_fn": loss_fn, "k": 3}
        if method == "Quantile_MLP":
            from torchregress.losses import MultiQuantileLoss

            model = _MLP(in_dim, hidden, len(_QUANTILES)).to(current_device())
            _train_one(
                model,
                MultiQuantileLoss(quantiles=_QUANTILES),
                X_t,
                y_t,
                config,
                batch_size=max(1024, int(config.batch_size)),
            )
            model.eval()
            return {"type": "quantile", "model": model, "levels": np.asarray(_QUANTILES)}
        if method in {"NormalizingFlow", "NormalizingFlow_cINN"}:
            from torchregress.losses import NormalizingFlowLoss, create_flow_model

            context_dim = hidden
            is_cinn = method == "NormalizingFlow_cINN"
            flow = create_flow_model(
                n_features=1,
                context_dim=context_dim,
                flow_type="realnvp" if is_cinn else "nsf",
                n_transforms=4 if is_cinn else 3,
                hidden_features=[64, 64],
                **({} if is_cinn else {"bound": 8.0}),
            )
            model = _MLP(in_dim, hidden, context_dim).to(current_device())
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_one(model, loss_fn, X_t, y_t, config, loss_params=list(flow.parameters()))
            model.eval()
            return {"type": "flow", "model": model, "flow": loss_fn}
        if method in {"SplitCP_MLP", "CQR_MLP", "MondrianCP_MLP"}:
            from torchregress.losses import CQR as CQRCalibrator
            from torchregress.losses import SplitConformal

            tr, cal = _split_train_cal(len(X), int(config.seed))
            X_tr, y_tr = X_t[tr], y_t[tr]
            if method == "SplitCP_MLP":
                model = _MLP(in_dim, hidden, 1).to(current_device())
                _train_one(
                    model,
                    nn.MSELoss(),
                    X_tr,
                    y_tr,
                    config,
                    batch_size=max(1024, int(config.batch_size)),
                )
                model.eval()
                with torch.no_grad():
                    pred_cal = model(X_t[cal]).reshape(-1)
                cp = SplitConformal(alpha=_ALPHA)
                cp.calibrate(pred_cal, y_t[cal].reshape(-1))
                return {"type": "split_cp_mlp", "model": model, "cp": cp}
            if method == "CQR_MLP":
                from torchregress.losses import MultiQuantileLoss

                model = _MLP(in_dim, hidden, len(_QUANTILES)).to(current_device())
                _train_one(
                    model,
                    MultiQuantileLoss(quantiles=_QUANTILES),
                    X_tr,
                    y_tr,
                    config,
                    batch_size=max(1024, int(config.batch_size)),
                )
                model.eval()
                with torch.no_grad():
                    q_cal = np.sort(model(X_t[cal]).cpu().numpy(), axis=1)
                q_cal_t = torch.as_tensor(q_cal[:, [0, -1]], dtype=torch.float32).to(
                    current_device()
                )
                cqr = CQRCalibrator(alpha=_ALPHA)
                cqr.calibrate(q_cal_t, y_t[cal].reshape(-1))
                return {"type": "cqr", "model": model, "cqr": cqr}
            # Mondrian split CP: groups are i-band magnitude quartiles.
            model = _MLP(in_dim, hidden, 1).to(current_device())
            _train_one(
                model,
                nn.MSELoss(),
                X_tr,
                y_tr,
                config,
                batch_size=max(1024, int(config.batch_size)),
            )
            model.eval()
            raw_train = np.asarray(splits["X_train"], dtype=np.float64)
            raw_cal = _i_band(raw_train[cal])
            bins = np.quantile(_i_band(raw_train), [0.25, 0.5, 0.75])
            groups_cal = np.digitize(raw_cal, bins)
            with torch.no_grad():
                pred_cal = model(X_t[cal]).reshape(-1)
            cp = SplitConformal(alpha=_ALPHA)
            cp.calibrate(
                pred_cal,
                y_t[cal].reshape(-1),
                groups=torch.as_tensor(groups_cal, device=current_device()),
            )
            return {"type": "mondrian", "model": model, "cp": cp, "bins": bins}
        if method == "CVPlus_MLP":
            from torchregress.losses import CVPlus

            rng = np.random.default_rng(int(config.seed))
            order = rng.permutation(len(X))
            folds = np.array_split(order, 3)
            models, oof = [], np.zeros(len(X), dtype=np.float64)
            fold_ids = np.zeros(len(X), dtype=np.int64)
            for fold, held in enumerate(folds):
                keep_idx = np.setdiff1d(order, held, assume_unique=False)
                model = _MLP(in_dim, hidden, 1).to(current_device())
                _train_one(
                    model,
                    nn.MSELoss(),
                    X_t[keep_idx],
                    y_t[keep_idx],
                    config,
                    batch_size=max(1024, int(config.batch_size)),
                )
                model.eval()
                with torch.no_grad():
                    oof[held] = model(X_t[held]).reshape(-1).cpu().numpy()
                fold_ids[held] = fold
                models.append(model)
            cp = CVPlus(alpha=_ALPHA)
            cp.calibrate_ensemble(
                torch.as_tensor(oof, dtype=torch.float32).reshape(-1, 1),
                y_t.detach().cpu().reshape(-1, 1),
                torch.as_tensor(fold_ids),
            )
            return {"type": "cvplus", "models": models, "cp": cp}
        if method in {"LocalCP_MLP", "NormalizedCP_MLP"}:
            from torchregress.losses import LocalConformal, SplitConformal

            tr, cal = _split_train_cal(len(X), int(config.seed))
            if method == "LocalCP_MLP":
                model = _MLP(in_dim, hidden, 1).to(current_device())
                _train_one(
                    model,
                    nn.MSELoss(),
                    X_t[tr],
                    y_t[tr],
                    config,
                    batch_size=max(1024, int(config.batch_size)),
                )
                model.eval()
                with torch.no_grad():
                    pred_cal = model(X_t[cal]).reshape(-1)
                cp = LocalConformal(alpha=_ALPHA, bandwidth=2.0)
                cp.calibrate(pred_cal, y_t[cal].reshape(-1), x=X_t[cal])
                return {"type": "localcp", "model": model, "cp": cp}
            from torchregress.losses import GaussianNLLLoss

            model = _MLP(in_dim, hidden, 2).to(current_device())
            _train_one(
                model,
                GaussianNLLLoss(),
                X_t[tr],
                y_t[tr],
                config,
                batch_size=max(1024, int(config.batch_size)),
            )
            model.eval()
            mean, std = _gaussian_nll_predict(model, X[cal])
            scores = np.abs(y_t[cal].reshape(-1).cpu().numpy() - mean) / np.maximum(std, 1e-6)
            q = float(np.quantile(scores, 1.0 - _ALPHA, method="higher"))
            return {"type": "normalizedcp", "model": model, "q": q}
        if method == "Evidential":
            from torchregress.losses import EvidentialRegressionLoss

            loss_fn = EvidentialRegressionLoss()
            model = _MLP(in_dim, hidden, 4).to(current_device())
            _train_one(
                model,
                loss_fn,
                X_t,
                y_t,
                config,
                batch_size=max(1024, int(config.batch_size)),
            )
            model.eval()
            return {"type": "evidential", "model": model, "loss_fn": loss_fn}
        if method == "BetaNLL_MLP":
            from torchregress.losses import BetaNLLLoss

            model = _MLP(in_dim, hidden, 2).to(current_device())
            _train_one(
                model,
                BetaNLLLoss(beta=0.5),
                X_t,
                y_t,
                config,
                batch_size=max(1024, int(config.batch_size)),
            )
            model.eval()
            return {"type": "gaussian", "model": model}
        if method == "InputNoiseAug_MLP":
            from torchregress.losses import InputNoiseAugmentationLoss, WeightedMSELoss

            model = _MLP(in_dim, hidden, 1).to(current_device())
            sigma = _feature_noise(np.asarray(splits["X_train"], dtype=np.float64))
            sigma = sigma / np.maximum(np.asarray(scaler.std).reshape(-1), 1e-6)
            loss_fn = InputNoiseAugmentationLoss(
                model,
                WeightedMSELoss(),
                sigma_x=torch.as_tensor(sigma, device=current_device()),
                n_samples=4,
            )
            opt = torch.optim.Adam(model.parameters(), lr=config.lr)
            bs = min(max(1024, int(config.batch_size)), len(X_t))
            for _ in range(int(config.epochs)):
                perm = torch.randperm(len(X_t), device=current_device())
                for start in range(0, len(X_t), bs):
                    idx = perm[start : start + bs]
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(X_t[idx], y_t[idx])
                    loss.backward()
                    opt.step()
            model.eval()
            return {"type": "point", "model": model}
        if method == "DensityWeighted_MLP":
            from torchregress.losses import DensityWeightedLoss

            loss_fn = DensityWeightedLoss(base_loss="mse")
            loss_fn.fit_density(y_t.detach().cpu())
            model = _MLP(in_dim, hidden, 1).to(current_device())
            _train_one(
                model,
                loss_fn,
                X_t,
                y_t,
                config,
                batch_size=max(1024, int(config.batch_size)),
            )
            model.eval()
            return {"type": "point", "model": model}
        if method == "BMC_MLP":
            from torchregress.losses import BalancedMSEBMC

            sigma_bmc = max(0.05, 0.15 * float(y_t.detach().std().clamp_min(1e-6)))
            loss_fn = BalancedMSEBMC(sigma=sigma_bmc).fit(y_t.detach().cpu())
            model = _MLP(in_dim, hidden, 1).to(current_device())
            _train_one(
                model,
                loss_fn,
                X_t,
                y_t,
                config,
                batch_size=max(1024, int(config.batch_size)),
            )
            model.eval()
            return {"type": "point", "model": model}
        if method in {"TabM_MSE", "TabM_GaussianNLL", "TabM_SplitCP"}:
            nll = method == "TabM_GaussianNLL"
            model = _tabm_model(in_dim, d_out=2 if nll else 1).to(current_device())
            _train_tabm(model, X_t, y_t, config, nll=nll)
            model.eval()
            out: dict[str, Any] = {
                "type": "gaussian_tabm" if nll else "tabm",
                "model": model,
                "nll": nll,
            }
            if method == "TabM_SplitCP":
                rng = np.random.default_rng(config.seed)
                idx = rng.permutation(len(X))
                n_cal = max(50, int(0.15 * len(X)))
                cal, tr = idx[:n_cal], idx[n_cal:]
                # Refit the point TabM on the non-calibration portion for an
                # honest residual calibration.
                model = _tabm_model(in_dim, d_out=1).to(current_device())
                _train_tabm(model, X_t[tr], y_t[tr], config)
                model.eval()
                with torch.no_grad():
                    pred_cal = model(X_t[cal]).mean(dim=1).squeeze(-1).cpu().numpy()
                resid = np.abs(y[cal] - pred_cal)
                q = float(np.quantile(resid, 0.90, method="higher"))
                out.update({"type": "split_cp", "q": q})
            return out
        raise ValueError(f"unknown method {method!r}")

    def predict(
        self, method: str, model_state: dict[str, Any], X_test: np.ndarray, config: HarnessConfig
    ) -> dict[str, np.ndarray]:
        # WFD n(z) tests have 1M rows; one GPU forward would OOM even for TabM.
        n = len(X_test)
        chunk = 50_000
        if n <= chunk:
            out = self._predict_impl(method, model_state, X_test, config)
            out["_raw_features"] = X_test
            return out
        parts = [
            self._predict_impl(method, model_state, X_test[start : start + chunk], config)
            for start in range(0, n, chunk)
        ]
        merged: dict[str, np.ndarray] = {}
        for key in parts[0]:
            vals = [p[key] for p in parts if key in p]
            if not vals:
                continue
            arr = np.concatenate(vals, axis=0)
            merged[key] = arr
        merged["_raw_features"] = X_test
        return merged

    def _predict_impl(
        self, method: str, model_state: dict[str, Any], X_test: np.ndarray, config: HarnessConfig
    ) -> dict[str, np.ndarray]:
        feats = _build_features(X_test)
        X_test_raw = X_test
        # Feature scaler is fitted per method in fit(); recompute from the
        # training split is impossible here, so use the robust global column
        # normalization stored on the state.
        scaler = model_state.get("scaler")
        if scaler is None:
            # fallback: train-file statistics are not available in predict;
            # the suite stores them in fit via _prepare.
            scaler = getattr(self, "_scaler", None)
        X = scaler(feats) if scaler is not None else feats
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(current_device())
        model = model_state.get("model")
        kind = model_state["type"]
        if kind == "pz_recipe_tabm":
            with torch.no_grad():
                features = model.encode(X_t)
                logits = model_state["stage2"](features)  # [N, k, K]
                proba = torch.softmax(logits, dim=-1).mean(dim=1)
                centers = model_state["centers"]
                mean = (proba * centers).sum(dim=-1)
                n_samples = 64 if len(X_test) > 100_000 else 256
                bin_idx = torch.distributions.Categorical(probs=proba).sample((n_samples,)).T
                edges = torch.empty(centers.numel() + 1, device=centers.device)
                edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
                edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
                edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
                low = edges[:-1][bin_idx]
                high = edges[1:][bin_idx]
                samples = (low + (high - low) * torch.rand_like(low)).cpu().numpy()
            return {
                "y_pred": mean.cpu().numpy(),
                "mean": mean.cpu().numpy(),
                "std": samples.std(axis=1),
                "lower": np.quantile(samples, 0.05, axis=1),
                "upper": np.quantile(samples, 0.95, axis=1),
                "samples": samples,
            }
        if kind == "pz_recipe":
            with torch.no_grad():
                logits = model(X_t, stage=1)
                proba = torch.softmax(logits, dim=-1)
                centers = model_state["centers"]
                mean = (proba * centers).sum(dim=-1)
                n_samples = 64 if len(X_test) > 100_000 else 256
                bin_idx = torch.distributions.Categorical(probs=proba).sample((n_samples,)).T
                edges = torch.empty(centers.numel() + 1, device=centers.device)
                edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
                edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
                edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
                low = edges[:-1][bin_idx]
                high = edges[1:][bin_idx]
                samples = (low + (high - low) * torch.rand_like(low)).cpu().numpy()
            return {
                "y_pred": mean.cpu().numpy(),
                "mean": mean.cpu().numpy(),
                "std": samples.std(axis=1),
                "lower": np.quantile(samples, 0.05, axis=1),
                "upper": np.quantile(samples, 0.95, axis=1),
                "samples": samples,
            }
        if kind == "point":
            with torch.no_grad():
                pred = model(X_t).reshape(-1).cpu().numpy()
            return {"y_pred": pred}
        if kind == "gaussian":
            mean, std = _gaussian_nll_predict(model, np.asarray(X, dtype=np.float32))
            z90 = 1.6448536269514722
            return {
                "y_pred": mean,
                "mean": mean,
                "std": std,
                "lower": mean - z90 * std,
                "upper": mean + z90 * std,
            }
        if kind == "gaussian_tabm":
            with torch.no_grad():
                out = model(X_t)  # [N,k,2]
            mean = out[..., 0].mean(dim=1).cpu().numpy()
            var = out[..., 1].clamp(-20.0, 20.0).exp().mean(dim=1).cpu().numpy()
            std = np.sqrt(np.maximum(var, 1e-12))
            z90 = 1.6448536269514722
            return {
                "y_pred": mean,
                "mean": mean,
                "std": std,
                "lower": mean - z90 * std,
                "upper": mean + z90 * std,
            }
        if kind == "tabm":
            with torch.no_grad():
                pred = model(X_t).mean(dim=1).reshape(-1).cpu().numpy()
            return {"y_pred": pred}
        if kind == "split_cp":
            with torch.no_grad():
                pred = model(X_t).mean(dim=1).reshape(-1).cpu().numpy()
            q = float(model_state["q"])
            return {"y_pred": pred, "lower": pred - q, "upper": pred + q}
        if kind == "split_cp_mlp":
            with torch.no_grad():
                pred = model(X_t).reshape(-1)
            lo, hi = model_state["cp"].predict_interval(pred)
            return {
                "y_pred": pred.cpu().numpy(),
                "lower": lo.cpu().numpy(),
                "upper": hi.cpu().numpy(),
            }
        if kind == "cqr":
            with torch.no_grad():
                q_np = np.sort(model(X_t).cpu().numpy(), axis=1)
                q_t = torch.as_tensor(q_np[:, [0, -1]], dtype=torch.float32).to(current_device())
            lo, hi = model_state["cqr"].predict_interval(q_t)
            return {
                "y_pred": q_np[:, 2],
                "lower": lo.cpu().numpy(),
                "upper": hi.cpu().numpy(),
            }
        if kind == "mondrian":
            with torch.no_grad():
                pred = model(X_t).reshape(-1)
            raw_test = np.asarray(X_test_raw)
            groups = np.digitize(_i_band(raw_test), model_state["bins"])
            lo, hi = model_state["cp"].predict_interval(
                pred, groups=torch.as_tensor(groups, device=current_device())
            )
            return {
                "y_pred": pred.cpu().numpy(),
                "lower": lo.cpu().numpy(),
                "upper": hi.cpu().numpy(),
            }
        if kind == "evidential":
            loss_fn = model_state["loss_fn"]
            with torch.no_grad():
                raw = model(X_t)
                gamma = raw[:, 0:1]
                nu = F.softplus(raw[:, 1:2]) + 0.01
                alpha = F.softplus(raw[:, 2:3]) + 1.01
                beta = F.softplus(raw[:, 3:4]) + 0.01
                params = torch.cat([gamma, nu, alpha, beta], dim=1)
                lo, hi = loss_fn.predict_interval(params, confidence=0.90)
                mean = gamma.reshape(-1).cpu().numpy()
                std = torch.sqrt(beta.reshape(-1) / (alpha.reshape(-1) - 1.0))
            return {
                "y_pred": mean,
                "mean": mean,
                "std": std.cpu().numpy(),
                "lower": lo.reshape(-1).cpu().numpy(),
                "upper": hi.reshape(-1).cpu().numpy(),
            }
        if kind == "cvplus":
            members = []
            with torch.no_grad():
                for model in model_state["models"]:
                    members.append(model(X_t).reshape(-1))
            stack = torch.stack(members, dim=0)
            lo, hi = model_state["cp"].predict_interval(stack.unsqueeze(-1))
            return {
                "y_pred": stack.mean(dim=0).cpu().numpy(),
                "lower": lo.reshape(-1).cpu().numpy(),
                "upper": hi.reshape(-1).cpu().numpy(),
            }
        if kind == "localcp":
            with torch.no_grad():
                pred = model(X_t).reshape(-1)
            lo, hi = model_state["cp"].predict_interval(pred, x=X_t)
            return {
                "y_pred": pred.cpu().numpy(),
                "lower": lo.cpu().numpy(),
                "upper": hi.cpu().numpy(),
            }
        if kind == "normalizedcp":
            mean, std = _gaussian_nll_predict(model, np.asarray(X, dtype=np.float32))
            q = float(model_state["q"])
            return {
                "y_pred": mean,
                "mean": mean,
                "std": std,
                "lower": mean - q * std,
                "upper": mean + q * std,
            }
        if kind == "quantile":
            with torch.no_grad():
                pred = model(X_t).cpu().numpy()
            # Independent quantile heads can cross on some rows; enforce the
            # monotone quantile function before forming the 5/95% interval.
            pred = np.sort(pred, axis=1)
            levels = model_state["levels"]
            mid = int(np.argmin(np.abs(levels - 0.5)))
            return {
                "y_pred": pred[:, mid],
                "lower": pred[:, 0],
                "upper": pred[:, -1],
                "quantile_values": pred,
                "quantile_levels": levels,
            }
        if kind == "mdn":
            loss_fn = model_state["loss_fn"]
            with torch.no_grad():
                context = model(X_t)
                samples = loss_fn.sample(context, n_samples=256).squeeze(-1).T
            samples = samples.cpu().numpy()
            return {
                "y_pred": samples.mean(axis=1),
                "mean": samples.mean(axis=1),
                "std": samples.std(axis=1),
                "lower": np.quantile(samples, 0.05, axis=1),
                "upper": np.quantile(samples, 0.95, axis=1),
                "samples": samples,
            }
        if kind == "flow":
            flow = model_state["flow"]
            with torch.no_grad():
                context = model(X_t)
                samples = flow.sample(context, n_samples=256).squeeze(-1).cpu().numpy()
            return {
                "y_pred": samples.mean(axis=1),
                "mean": samples.mean(axis=1),
                "std": samples.std(axis=1),
                "lower": np.quantile(samples, 0.05, axis=1),
                "upper": np.quantile(samples, 0.95, axis=1),
                "samples": samples,
            }
        raise ValueError(f"unknown state type {kind!r}")

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        scaler = predictions.get("_y_scaler")
        pred = dict(predictions)
        if scaler is not None:
            pred = _inverse_transform_predictions(pred, scaler)
            y_true = _inverse_transform_y(np.asarray(y_test), scaler)
        else:
            y_true = np.asarray(y_test, dtype=np.float64).reshape(-1)
        metrics = compute_all_metrics(pred, y_true)
        y_pred = np.asarray(pred.get("y_pred", pred.get("mean")), dtype=np.float64).reshape(-1)
        z = np.asarray(y_true, dtype=np.float64).reshape(-1)
        denom = 1.0 + z
        delta = (y_pred - z) / denom
        med = np.median(delta)
        metrics["SigmaNMAD"] = float(1.4826 * np.median(np.abs(delta - med)))
        metrics["OutlierFraction15"] = float(np.mean(np.abs(delta) > 0.15))
        metrics["BiasZ"] = float(med)
        # Paper diagnostics: PIT CvM, local photometric PIT-W1, central-68% width.
        try:
            from torchregress.metrics import pdf_width_68, pit_cvm, pit_local_w1

            pit = metrics.get("PIT")
            if pit is None and "samples" in pred:
                sm = np.asarray(pred["samples"], dtype=np.float64)
                pit = np.mean(sm <= z[:, None], axis=1)
            if pit is None and "mean" in pred and "std" in pred:
                from scipy.stats import norm

                pit = norm.cdf(
                    z,
                    loc=np.asarray(pred["mean"], dtype=np.float64),
                    scale=np.maximum(np.asarray(pred["std"], dtype=np.float64), 1e-9),
                )
            if pit is not None:
                pit = np.asarray(pit, dtype=np.float64).reshape(-1)
                metrics["PITCVM"] = float(pit_cvm(pit))
            if "samples" in pred:
                metrics["W68"] = float(pdf_width_68(pred["samples"], z))
            elif "mean" in pred and "std" in pred:
                metrics["W68"] = float(
                    np.median(2.0 * 0.994457883209753 * np.asarray(pred["std"]) / (1.0 + z))
                )
            raw = pred.get("_raw_features")
            if raw is not None and pit is not None:
                metrics["LocalW1"] = float(
                    pit_local_w1(
                        _build_features(np.asarray(raw)),
                        pit,
                        k=min(100, max(2, len(pit) - 1)),
                    )
                )
        except Exception:
            pass
        if "quantile_values" in pred and "quantile_levels" in pred:
            qv = np.asarray(pred["quantile_values"], dtype=np.float64)
            levels = np.asarray(pred["quantile_levels"], dtype=np.float64)
            i16 = int(np.argmin(np.abs(levels - 0.16)))
            i84 = int(np.argmin(np.abs(levels - 0.84)))
            lo68, hi68 = qv[:, i16], qv[:, i84]
            metrics["Coverage68"] = float(np.mean((z >= lo68) & (z <= hi68)))
            metrics["Width68"] = float(np.mean(hi68 - lo68))
        elif "samples" in pred:
            sm = np.asarray(pred["samples"], dtype=np.float64)
            lo68, hi68 = np.quantile(sm, 0.16, axis=1), np.quantile(sm, 0.84, axis=1)
            metrics["Coverage68"] = float(np.mean((z >= lo68) & (z <= hi68)))
            metrics["Width68"] = float(np.mean(hi68 - lo68))
        elif "mean" in pred and "std" in pred:
            m = np.asarray(pred["mean"], dtype=np.float64)
            sd = np.asarray(pred["std"], dtype=np.float64)
            z68 = 0.994457883209753
            metrics["Coverage68"] = float(np.mean(np.abs(z - m) <= z68 * sd))
            metrics["Width68"] = float(np.mean(2.0 * z68 * sd))
        return metrics


class PZCPExtendedSuite(PZChallengeSuite):
    """All torchregress conformal variants on every PZ challenge dataset."""

    @property
    def name(self) -> str:
        return "pz_cp"

    @property
    def description(self) -> str:
        return (
            "PZ challenge conformal-prediction comparison: split, CQR, Mondrian, "
            "CV+, local (LVD) and normalized CP across all challenge datasets."
        )

    def methods(self) -> list[str]:
        return [
            "SplitCP_MLP",
            "CQR_MLP",
            "MondrianCP_MLP",
            "CVPlus_MLP",
            "LocalCP_MLP",
            "NormalizedCP_MLP",
        ]


class PZCoreSuite(PZChallengeSuite):
    """Original point/distributional/TabM baseline across all PZ datasets."""

    @property
    def name(self) -> str:
        return "pz_core"

    @property
    def description(self) -> str:
        return (
            "PZ challenge core comparison (MSE, Gaussian NLL, MDN, quantile, NSF, "
            "TabM point/NLL/split-CP) on all task sets and scenarios."
        )

    def methods(self) -> list[str]:
        return [
            "MSE_MLP",
            "GaussianNLL_MLP",
            "MDN_K3",
            "Quantile_MLP",
            "NormalizingFlow",
            "TabM_MSE",
            "TabM_GaussianNLL",
            "TabM_SplitCP",
        ]


class PZExtendedSuite(PZChallengeSuite):
    """Evidential, beta-NLL, EIV input-noise, density-weighted and BMC heads."""

    @property
    def name(self) -> str:
        return "pz_extended"

    @property
    def description(self) -> str:
        return (
            "PZ challenge extended-head comparison: evidential NIG, beta-NLL, "
            "error-aware input-noise augmentation, density-weighted and BMC."
        )

    def methods(self) -> list[str]:
        return [
            "Evidential",
            "BetaNLL_MLP",
            "InputNoiseAug_MLP",
            "DensityWeighted_MLP",
            "BMC_MLP",
        ]


class PZFlowSuite(PZChallengeSuite):
    """cNSF vs cINN-style flow comparison (Ren et al. 2026, ApJ 997, 65)."""

    @property
    def name(self) -> str:
        return "pz_flow"

    @property
    def description(self) -> str:
        return (
            "PZ challenge normalizing-flow comparison: zuko cNSF vs a "
            "conditional RealNVP (cINN-style) flow, plus MDN and TabM reference."
        )

    def methods(self) -> list[str]:
        return ["NormalizingFlow", "NormalizingFlow_cINN", "MDN_K3", "TabM_MSE"]
