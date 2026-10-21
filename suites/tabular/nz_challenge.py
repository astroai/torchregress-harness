"""LSST DESC n(z) challenge benchmark.

Task sets 1 and 2 provide DDF fields with reference redshifts for training and a
wide-field (WFD) catalog without public labels.  We train per-object p(z) heads on
all DDF fields, predict the WFD p(z), and aggregate to n(z) and tomographic-bin
fractions.  WFD truth comes from the post-challenge ``reserved`` archive, matched
by object id, so this is an internal benchmark rather than a challenge submission.

The suite also runs prediction-powered inference (PPI): a small labeled WFD slice
plus model predictions on the rest of WFD are used to estimate tomographic-bin
fractions and the mean redshift, with nominal 90% CIs compared against the full
reserved truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from harness.runners import HarnessConfig
from suites.tabular.pz_challenge import PZChallengeSuite

_NZ_BINS = np.linspace(0.0, 3.0, 76)
_NZ_BIN_EDGES = np.linspace(0.0, 3.0, 5)  # 4 tomographic bins for evaluation


def _histogram(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    h, _ = np.histogram(values, bins=_NZ_BINS, weights=weights)
    total = h.sum()
    return h / total if total > 0 else h


class NZChallengeSuite(PZChallengeSuite):
    """n(z) and tomographic-bin inference from per-object p(z)."""

    def __init__(self) -> None:
        super().__init__()
        self._methods = ["MSE_MLP", "GaussianNLL_MLP", "MDN_K3", "TabM_GaussianNLL"]
        self._train_y_native: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "nz_challenge"

    @property
    def description(self) -> str:
        return (
            "LSST DESC n(z) challenge: DDF training -> WFD n(z), tomographic bins "
            "and PPI CIs, evaluated against reserved WFD truth."
        )

    @property
    def supported_datasets(self) -> list[str]:
        out = []
        for t in (1, 2):
            for sim in ("cardinal", "flagship"):
                for scen in ("1yr", "4yr"):
                    out.append(f"nz_challenge_t{t}_{sim}_{scen}")
        return out

    def methods(self) -> list[str]:
        return list(self._methods)

    def fit(
        self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig
    ) -> dict[str, Any]:
        scaler = splits.get("_y_scaler")
        y = np.asarray(splits["y_train"]).reshape(-1)
        self._train_y_native = (
            scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
            if scaler is not None
            else y.copy()
        )
        return super().fit(method, splits, config)

    def _object_histograms(self, method: str, pred: dict[str, np.ndarray]) -> np.ndarray | None:
        """Per-object p(z) histograms on _NZ_BINS (N x bins)."""
        z_grid = 0.5 * (_NZ_BINS[:-1] + _NZ_BINS[1:])
        if "samples" in pred:
            samples = np.asarray(pred["samples"], dtype=np.float64)
            out = np.zeros((samples.shape[0], len(z_grid)), dtype=np.float64)
            for i in range(samples.shape[0]):
                out[i] = _histogram(samples[i])
            return out
        if "mean" in pred and "std" in pred:
            mean = np.asarray(pred["mean"], dtype=np.float64)
            std = np.maximum(np.asarray(pred["std"], dtype=np.float64), 1e-6)
            from scipy.stats import norm

            cdf = norm.cdf(_NZ_BINS[None, :], loc=mean[:, None], scale=std[:, None])
            probs = np.diff(cdf, axis=1)
            centers = 0.5 * (_NZ_BINS[:-1] + _NZ_BINS[1:])
            out = np.zeros_like(probs)
            out[:, 0 : len(centers)] = probs
            return out
        if "quantile_values" in pred:
            # Build a coarse histogram from the quantile CDF by interpolation.
            qv = np.asarray(pred["quantile_values"], dtype=np.float64)
            levels = np.asarray(pred["quantile_levels"], dtype=np.float64)
            out = np.zeros((qv.shape[0], len(z_grid)), dtype=np.float64)
            for i in range(qv.shape[0]):
                cdf = np.interp(z_grid, qv[i], levels, left=0.0, right=1.0)
                out[i] = np.diff(np.concatenate([[0.0], cdf, [1.0]]))[: len(z_grid)]
            s = out.sum(axis=1, keepdims=True)
            return out / np.where(s > 0, s, 1.0)
        if "y_pred" in pred:
            z = np.asarray(pred["y_pred"], dtype=np.float64)
            out = np.zeros((len(z), len(z_grid)), dtype=np.float64)
            idx = np.clip(np.searchsorted(z_grid, z) - 1, 0, len(z_grid) - 1)
            out[np.arange(len(z)), idx] = 1.0
            return out
        return None

    @staticmethod
    def _nz_metrics(pred_hist: np.ndarray, true_z: np.ndarray) -> dict[str, float]:
        true_hist = _histogram(true_z)
        pred_nz = pred_hist.mean(axis=0)
        pred_nz = pred_nz / max(pred_nz.sum(), 1e-12)
        cdf_p = np.cumsum(pred_nz)
        cdf_t = np.cumsum(true_hist)
        zc = 0.5 * (_NZ_BINS[:-1] + _NZ_BINS[1:])
        pred_mean = float(np.sum(zc * pred_nz))
        pred_med = float(zc[np.argmin(np.abs(cdf_p - 0.5))])
        return {
            "NzL1": float(np.abs(pred_nz - true_hist).sum()),
            "NzL2": float(np.sqrt(np.mean((pred_nz - true_hist) ** 2))),
            "NzKS": float(np.max(np.abs(cdf_p - cdf_t))),
            "NzMeanBias": float(pred_mean - true_z.mean()),
            "NzMedianBias": float(pred_med - np.median(true_z)),
            "NzMean": float(pred_mean),
            "NzTrueMean": float(true_z.mean()),
        }

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
            from harness.runners import _inverse_transform_predictions, _inverse_transform_y

            pred = _inverse_transform_predictions(pred, scaler)
            y_true = _inverse_transform_y(np.asarray(y_test), scaler)
        else:
            y_true = np.asarray(y_test, dtype=np.float64).reshape(-1)
        z_pred = np.asarray(pred.get("y_pred", pred.get("mean")), dtype=np.float64).reshape(-1)
        rmse = float(np.sqrt(np.mean((z_pred - y_true) ** 2)))
        metrics: dict[str, float] = {"RMSE": rmse, "MAE": float(np.mean(np.abs(z_pred - y_true)))}
        hist = self._object_histograms(method, pred)
        if hist is not None:
            metrics.update(self._nz_metrics(hist, y_true))
            p_cdf = hist.cumsum(axis=1)
            true_bin = np.digitize(y_true, _NZ_BIN_EDGES) - 1
            true_bin = np.clip(true_bin, 0, len(_NZ_BIN_EDGES) - 2)
            # Per-object tomographic-bin probabilities.
            bin_probs = np.stack(
                [
                    hist[
                        :,
                        (_NZ_BINS[:-1] >= _NZ_BIN_EDGES[i])
                        & (_NZ_BINS[:-1] < _NZ_BIN_EDGES[i + 1]),
                    ].sum(axis=1)
                    for i in range(len(_NZ_BIN_EDGES) - 1)
                ],
                axis=1,
            )
            pred_bin = np.argmax(bin_probs, axis=1)
            metrics["BinAccuracy"] = float(np.mean(pred_bin == true_bin))
            # Purity/efficiency per bin, averaged over predicted bins with support.
            pur, eff = [], []
            for b in range(bin_probs.shape[1]):
                sel = pred_bin == b
                if sel.any():
                    pur.append(float(np.mean(true_bin[sel] == b)))
                    eff.append(float(np.mean(pred_bin[true_bin == b] == b)))
            metrics["BinPurity"] = float(np.mean(pur)) if pur else float("nan")
            metrics["BinEfficiency"] = float(np.mean(eff)) if eff else float("nan")
            metrics.update(self._ppi_metrics(pred, hist, bin_probs, y_true, config))
        return metrics

    def _ppi_metrics(
        self,
        pred: dict[str, np.ndarray],
        hist: np.ndarray,
        bin_probs: np.ndarray,
        y_true: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        """PPI CIs for mean redshift and tomographic-bin fractions."""
        from torchregress.inference import PPIConfig, ppi_mean_ci, ppi_quantile_ci

        rng = np.random.default_rng(int(config.seed))
        n = len(y_true)
        n_lab = min(n, 5000, max(20, n // 20))
        lab = rng.choice(n, n_lab, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[lab] = False
        ppi_cfg = PPIConfig(alpha=0.1, n_boot=200, seed=int(config.seed))
        y_mean = np.asarray(pred.get("y_pred", pred.get("mean")), dtype=np.float64).reshape(-1)
        out: dict[str, float] = {}
        try:
            res = ppi_mean_ci(y_true[lab], y_mean[lab], y_mean[mask], config=ppi_cfg)
            out["PPI_MeanZ_Estimate"] = float(res["estimate"])
            out["PPI_MeanZ_Coverage"] = float(res["ci_lower"] <= y_true.mean() <= res["ci_upper"])
            out["PPI_MeanZ_CIWidth"] = float(res["ci_upper"] - res["ci_lower"])
        except Exception:
            pass
        try:
            res_q = ppi_quantile_ci(y_true[lab], y_mean[lab], y_mean[mask], q=0.5, config=ppi_cfg)
            out["PPI_MedianZ_Estimate"] = float(res_q["estimate"])
            out["PPI_MedianZ_Coverage"] = float(
                res_q["ci_lower"] <= np.median(y_true) <= res_q["ci_upper"]
            )
            out["PPI_MedianZ_CIWidth"] = float(res_q["ci_upper"] - res_q["ci_lower"])
        except Exception:
            pass
        cov, widths = [], []
        true_frac = []
        for b in range(bin_probs.shape[1]):
            yb = (y_true >= _NZ_BIN_EDGES[b]) & (y_true < _NZ_BIN_EDGES[b + 1])
            true_frac.append(float(yb.mean()))
            try:
                res = ppi_mean_ci(
                    yb[lab].astype(np.float64),
                    bin_probs[lab, b],
                    bin_probs[mask, b],
                    config=ppi_cfg,
                )
                cov.append(float(res["ci_lower"] <= true_frac[-1] <= res["ci_upper"]))
                widths.append(float(res["ci_upper"] - res["ci_lower"]))
            except Exception:
                continue
        if cov:
            out["PPI_BinFraction_Coverage"] = float(np.mean(cov))
            out["PPI_BinFraction_CIWidth"] = float(np.mean(widths))
            out["PPI_BinFractionError"] = float(
                np.mean(
                    np.abs(
                        np.array([bin_probs[:, b].mean() for b in range(bin_probs.shape[1])])
                        - np.array(true_frac)
                    )
                )
            )
        return out
