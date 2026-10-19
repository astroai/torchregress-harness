"""LSST DESC Photometric Redshift Data Challenge benchmark.

Task set 1 (representative training) is used with a held-out split of the
official training file.  Nine-band LSST+Roman magnitudes/errors are augmented
with colours and detection flags, then fed to torchregress point and
distributional heads plus a TabM point/NLL baseline.

Metrics cover the challenge's point diagnostics (RMSE, MAE, bias,
sigma_NMAD, catastrophic-outlier fraction) and predictive diagnostics
(NLL, CRPS, interval coverage, PIT).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
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
]
_QUANTILES = (0.05, 0.16, 0.5, 0.84, 0.95)


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
        ]

    def methods(self) -> list[str]:
        return list(_METHODS)

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
            "NormalizingFlow": "zuko NSF conditional p(z|x), exact NLL and samples",
            "TabM_MSE": "official TabM (k=8, 2 blocks, width 256), point MSE",
            "TabM_GaussianNLL": "TabM with mean/logvar heads, Gaussian NLL",
            "TabM_SplitCP": "TabM point predictions + split-conformal residual intervals",
        }
        return notes.get(method, "")

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
        if method == "NormalizingFlow":
            from torchregress.losses import NormalizingFlowLoss, create_flow_model

            context_dim = hidden
            flow = create_flow_model(
                n_features=1,
                context_dim=context_dim,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
                bound=8.0,
            )
            model = _MLP(in_dim, hidden, context_dim).to(current_device())
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_one(model, loss_fn, X_t, y_t, config, loss_params=list(flow.parameters()))
            model.eval()
            return {"type": "flow", "model": model, "flow": loss_fn}
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
        feats = _build_features(X_test)
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
        model = model_state["model"]
        kind = model_state["type"]
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
        if kind == "quantile":
            with torch.no_grad():
                pred = model(X_t).cpu().numpy()
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
        return metrics
