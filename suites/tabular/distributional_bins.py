"""Distributional bin-comparison suite: CP vs NF vs MDN vs Evidential.

Trains a family of distributional methods on scalar regression targets,
produces prediction intervals for each, then bins test points by predicted
interval width (using Split CP as the reference binning) and reports coverage
per bin.

Methods
-------
* ``Split_CP`` — post-hoc split conformal (baseline, also used for binning)
* ``GaussianNLL`` — diagonal Gaussian NLL → mean ± z·σ intervals
* ``NormalizingFlow`` — zuko NSF flow → sample-based quantile intervals
* ``NF_Ensemble_K{2,5,10,20}`` — K seeded NormalizingFlows with pooled
  samples for per-row quantile intervals (Proposal #10 + K-sweep validation §14).
  Architecture (hidden layers + transforms + per-member sample count) is
  identical across K; only the ensemble size varies.
* ``MDN_K3`` — mixture density network → sample-based quantile intervals
* ``Evidential`` — NIG evidential regression → Student-t predictive intervals

Evaluation
----------
Beyond standard interval metrics, the suite computes **coverage per bin**:
samples are binned into 10 equal-width buckets by Split CP interval width,
and coverage is reported for each method in each bin.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import (
    _MLP,
    NF_ENSEMBLE_K_VARIANTS,
    _carve_calibration_split,
    _gaussian_nll_predict,
    _hidden_size,
    _train_one,
)

# ---------------------------------------------------------------------------
# Skew/shape family spec: method → (loss class, head width, notes)
# Widths verified by forward probe; location is column 0 in all layouts.
# ---------------------------------------------------------------------------

_SKEW_METHODS: dict[str, dict[str, Any]] = {
    "SkewNormal": {"loss": "SkewNormalNLLLoss", "width": 3, "note": "[ξ,ω,α] skew-normal MLE"},
    "SkewT": {"loss": "SkewTLoss", "width": 4, "note": "[ξ,ω,α,ν] skew-t MLE"},
    "BetaRegression": {
        "loss": "BetaRegressionNLLLoss",
        "width": 2,
        "note": "beta MLE on min-max-normalized targets",
        "normalize01": True,
    },
    "JohnsonSU": {"loss": "JohnsonSUNLLLoss", "width": 4, "note": "[ξ,λ,γ,δ] Johnson-SU MLE"},
    "SinhArcsinh": {"loss": "SinhArcsinhNLLLoss", "width": 4, "note": "sinh-arcsinh MLE"},
    "GEV": {"loss": "GEVNLLLoss", "width": 3, "note": "[μ,σ,ξ] extreme-value MLE"},
    "AsymmetricLaplace": {
        "loss": "AsymmetricLaplaceNLLLoss",
        "width": 3,
        "note": "asymmetric-Laplace (quantile-like) MLE",
    },
}


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

_Z_SCORE = 1.645  # 90% prediction interval


def _split_cp_intervals(
    model: nn.Module,
    X_cal: torch.Tensor,
    y_cal: torch.Tensor,
    X_test: torch.Tensor,
    *,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-hoc split conformal: absolute residuals → quantile calibration."""
    model.eval()
    with torch.no_grad():
        pred_cal = model(X_cal).squeeze(-1)
        pred_test = model(X_test).squeeze(-1)
    residuals = (y_cal - pred_cal).abs()
    q = torch.quantile(residuals, 1.0 - alpha).item()
    return (pred_test - q).cpu().numpy(), (pred_test + q).cpu().numpy()


def _skew_cp_intervals(
    model: nn.Module,
    X_cal: torch.Tensor,
    y_cal: torch.Tensor,
    X_test: torch.Tensor,
    *,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split-conformal on a distribution head's location column (col 0)."""
    model.eval()
    with torch.no_grad():
        loc_cal = model(X_cal)[:, 0]
        loc_test = model(X_test)[:, 0]
    residuals = (y_cal.reshape(-1) - loc_cal.reshape(-1)).abs()
    q = torch.quantile(residuals, 1.0 - alpha).item()
    loc = loc_test.reshape(-1).detach().cpu().numpy()
    return loc - q, loc + q, loc


def _bin_metrics(
    bin_edges: np.ndarray,
    bin_assignments: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: np.ndarray,
    *,
    target_coverage: float = 0.9,
) -> list[dict[str, float]]:
    """Compute per-bin coverage, mean width, and deviation from target.

    CoverageDeviation = coverage - target_coverage.
    Positive → over-covers in this bin.  Negative → under-covers.
    """
    results: list[dict[str, float]] = []
    for i in range(len(bin_edges) - 1):
        mask = bin_assignments == i
        n_bin = int(mask.sum())
        if n_bin == 0:
            results.append(
                {
                    "BinCenter": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                    "BinSize": 0,
                    "BinCoverage": np.nan,
                    "BinMeanWidth": np.nan,
                    "CoverageDeviation": np.nan,
                }
            )
            continue
        in_interval = (y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])
        coverage = float(np.mean(in_interval))
        width = float(np.mean(upper[mask] - lower[mask]))
        results.append(
            {
                "BinCenter": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "BinSize": n_bin,
                "BinCoverage": coverage,
                "BinMeanWidth": width,
                "CoverageDeviation": coverage - target_coverage,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class DistributionalBinsSuite(HarnessRunner):
    """Distributional prediction bin-comparison suite.

    Trains several distributional methods on scalar targets, then bins
    test points by Split CP interval width and compares coverage.
    """

    CALIBRATION_FRAC = 0.25
    N_BINS = 10
    TARGET_COVERAGE = 0.9  # α=0.1 → 90% prediction intervals

    @property
    def name(self) -> str:
        return "distributional_bins"

    @property
    def description(self) -> str:
        return "Distributional bin comparison: CP vs NF vs MDN vs Evidential"

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "diabetes",
            "synthetic_hetero",
        ]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        return [
            "Split_CP",
            "GaussianNLL",
            "NormalizingFlow",
            *NF_ENSEMBLE_K_VARIANTS,
            "MDN_K3",
            "Evidential",
            "MC_CP",
            "SLS_CP",
            "CTI",
            "SkewNormal",
            "SkewT",
            "BetaRegression",
            "JohnsonSU",
            "SinhArcsinh",
            "GEV",
            "AsymmetricLaplace",
            "SQR",
        ]

    def library_name(self, method: str) -> str:
        return "torchregress"

    def method_notes(self, method: str) -> str:
        # Templated K-sweep notes so the four K variants stay consistent —
        # editing `k_sweep_contexts` updates all four at once (refactor of
        # the §14 follow-up; previously four separately-edited strings,
        # easy to drift when only one was updated).
        k_sweep_contexts = {
            "NF_Ensemble_K2": ("K-sweep minimum", "\u00a714 follow-up"),
            "NF_Ensemble_K5": (
                "K-sweep \u00a713 baseline",
                "superseded by K=10 (new production default)",
            ),
            "NF_Ensemble_K10": (
                "code-level production default per \u00a714 IS-knee",
                "validated in \u00a714/\u00a715",
            ),
            "NF_Ensemble_K20": ("K-sweep saturation test", "\u00a714 follow-up"),
        }
        if method in k_sweep_contexts:
            ctx_text, section_text = k_sweep_contexts[method]
            K = int(method.rsplit("_K", 1)[-1])
            return (
                f"{K} independent NormalizingFlows (seed offsets) with pooled "
                f"samples ({K}\u00d7n_samples) for quantile intervals \u2014 "
                f"{ctx_text} (Proposal #10 {section_text})"
            )
        notes = {
            "Split_CP": "post-hoc split conformal — baseline, also defines bins",
            "GaussianNLL": "diagonal Gaussian NLL → mean ± z·σ intervals",
            "NormalizingFlow": "zuko NSF normalizing flow → sample-based quantile intervals",
            "MDN_K3": "3-component mixture density network → sample-based quantile intervals",
            "Evidential": "NIG evidential regression → Student-t predictive intervals",
            "MC_CP": "NSF flow samples + MonteCarloConformal (mean center)",
            "SLS_CP": "SLS frontier (d=1) + SLSConformal grid predictor",
            "CTI": "NSF flow density + Conformal Thresholded Intervals (level sets)",
            "SQR": "32-level sorted quantiles → own 5/95% intervals (Duan et al. 2020)",
        }
        if method in _SKEW_METHODS:
            spec = _SKEW_METHODS[method]
            return f"{spec['note']} → location ± conformal intervals"
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # Flow-ensemble internals (Proposal #10)
    #
    # `_NF_ENSEMBLE_K_REFERENCE` is the code-level production default in
    # `distributional_bins` (used by the smoke-test cross-suite parity lock
    # + the §13 PROPOSAL #10 original headline validation).  K=10 is both
    # the doc-paper general-purpose default per §14 IS-knee evidence AND
    # the code-level production default here (bumped from K=5 in line
    # with the §14/§15 IS-knee findings — K=5 stays as the §13 baseline
    # reference, K=10 takes the production slot).  K=5 remains a
    # registered method in the K-sweep family for parity runs against
    # PARITY_REPORT.md §13 baselines.  The actual K used at fit time is
    # parsed from the method name suffix (`NF_Ensemble_K{2,5,10,20}`)
    # — see §14 below.  All other architectural parameters (HIDDEN,
    # TRANSFORMS, N_SAMPLES) stay locked across K so the §14
    # calibration-vs-cost sweep isolates K's effect.
    # Per-dataset caveat (§14): on california_housing-style data,
    # DevAbsMean regresses at K=10 (0.013) vs K=5's 0.008 — K=10 wins
    # IS but loses calibration on clean large data.  Production default
    # is IS-priority; calibration-priority callers may prefer K=5.
    # ------------------------------------------------------------------

    _NF_ENSEMBLE_K_REFERENCE = 10
    _NF_ENSEMBLE_HIDDEN = [64, 64]
    _NF_ENSEMBLE_TRANSFORMS = 3
    _NF_ENSEMBLE_N_SAMPLES = 256

    # ------------------------------------------------------------------
    # fit — carve cal split, train per method
    #
    # Note: `Split_CP` requires pre-carved ``X_cal``/``y_cal`` keys in the
    # ``splits`` dict; ``run()`` carves them via ``_carve_calibration_split``
    # automatically.  All other branches (GaussianNLL, NF, ensembles, MDN,
    # Evidential) only need ``X_train``/``y_train`` so they remain decoupled
    # from the cal-split bookkeeping.
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train = torch.from_numpy(splits["X_train"]).float().to(current_device())
        y_train = torch.from_numpy(splits["y_train"]).float().to(current_device()).unsqueeze(1)
        in_dim = X_train.shape[1]
        hidden = _hidden_size(in_dim)

        # --- Split CP: point model + post-hoc calibration ---
        # Note: X_cal / y_cal are extracted here ONLY for this branch; other
        # branches (GaussianNLL, NF, Ensemble, MDN, Evidential) do not
        # require them so they remain decoupled from the cal split.
        if method == "Split_CP":
            X_cal = torch.from_numpy(splits["X_cal"]).float().to(current_device())
            y_cal = torch.from_numpy(splits["y_cal"]).float().to(current_device())
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=1)
            _train_one(model, nn.MSELoss(), X_train, y_train, config)
            model.eval()
            return {
                "type": "split_cp",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
            }

        # --- GaussianNLL: mean + log_var ---
        elif method == "GaussianNLL":
            from torchregress.losses import GaussianNLLLoss

            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=2)
            _train_one(model, GaussianNLLLoss(), X_train, y_train, config)
            model.eval()
            return {"type": "gaussian_nll", "model": model}

        # --- NormalizingFlow ---
        elif method == "NormalizingFlow":
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            context_dim = hidden
            # 64-unit hidden layers for coupling networks — 32-unit layers
            # (previous default) produce overconfident densities (too-peaked)
            # that lead to severe under-coverage (~0.59 on diabetes).  More
            # transforms backfire by overfitting on small tabular datasets;
            # stick to 3.  Architecture sweep confirms T3_H64 is the sweet
            # spot (0.75 coverage vs 0.65 with H32, 0.62 with T5_H64).
            flow = create_flow_model(
                n_features=1,
                context_dim=context_dim,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=context_dim)
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_one(
                model, loss_fn, X_train, y_train, config, loss_params=list(flow.parameters())
            )
            model.eval()
            return {"type": "nflow", "model": model, "flow": loss_fn}

        # --- NF_Ensemble_K{2,5,10,20}: K-sweep NormalizingFlows (Proposal #10 + §14) ---
        elif method in ("NF_Ensemble_K2", "NF_Ensemble_K5", "NF_Ensemble_K10", "NF_Ensemble_K20"):
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            # Parse K from method name suffix (NF_Ensemble_K<N> -> N).
            K = int(method.rsplit("_K", 1)[-1])
            context_dim = hidden
            # Reuse the same NSF architecture as NormalizingFlow for fair
            # comparison — the ensemble's purpose is reducing epistemic
            # uncertainty, not changing the flow's expressivity.  K is the
            # only sweep dimension; HIDDEN/TRANSFORMS/N_SAMPLES stay locked.
            members: list[dict[str, Any]] = []
            for i in range(K):
                set_harness_seed(config.seed + i)
                flow_i = create_flow_model(
                    n_features=1,
                    context_dim=context_dim,
                    flow_type="nsf",
                    n_transforms=self._NF_ENSEMBLE_TRANSFORMS,
                    hidden_features=self._NF_ENSEMBLE_HIDDEN,
                )
                model_i = _MLP(in_dim=in_dim, hidden=hidden, out_dim=context_dim)
                loss_i = NormalizingFlowLoss(flow=flow_i)
                _train_one(
                    model_i,
                    loss_i,
                    X_train,
                    y_train,
                    config,
                    loss_params=list(flow_i.parameters()),
                )
                model_i.eval()
                members.append(
                    {
                        "model": model_i,
                        "flow": loss_i,
                        "seed": config.seed + i,
                    }
                )
            return {"type": "nflow_ensemble", "members": members, "K": K}

        # --- MDN (3 components) ---
        elif method == "MDN_K3":
            from torchregress.losses import MixtureDensityLoss

            model = _MLP(
                in_dim=in_dim,
                hidden=hidden,
                out_dim=3 * 3,  # 3 comps × (pi, mu, sigma)
            )
            loss_fn = MixtureDensityLoss(n_components=3, n_features=1)
            _train_one(model, loss_fn, X_train, y_train, config)
            model.eval()
            return {"type": "mdn", "model": model, "loss_fn": loss_fn}

        # --- Evidential ---
        elif method == "Evidential":
            from torchregress.losses import EvidentialRegressionLoss

            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=4)
            loss_fn = EvidentialRegressionLoss(coeff_nig=0.01)

            # Custom training with softplus constraints on nu, alpha, beta
            _train_evidential(model, loss_fn, X_train, y_train, config)
            model.eval()
            return {"type": "evidential", "model": model, "loss_fn": loss_fn}

        # --- MonteCarloCP: NF samples + MonteCarloConformal (mean center) ---
        elif method == "MC_CP":
            from torchregress.losses.conformal import MonteCarloConformal
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            context_dim = hidden
            flow = create_flow_model(
                n_features=1,
                context_dim=context_dim,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=context_dim)
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_one(
                model, loss_fn, X_train, y_train, config, loss_params=list(flow.parameters())
            )
            model.eval()
            X_cal = torch.from_numpy(splits["X_cal"]).float().to(current_device())
            y_cal = torch.from_numpy(splits["y_cal"]).float().to(current_device())
            with torch.no_grad():
                cal_samples = loss_fn.sample(model(X_cal), n_samples=256).squeeze(-1)
            predictor = MonteCarloConformal(alpha=0.1)
            # mc_samples layout is [n_samples, batch, ...] → transpose (N, S).
            predictor.calibrate(cal_samples.detach().T, y_cal.reshape(-1))
            return {"type": "mc_cp", "model": model, "flow": loss_fn, "predictor": predictor}

        # --- SLS_CP: SLS frontier (d=1) + SLSConformal grid predictor ---
        elif method == "SLS_CP":
            from torchregress.losses.conformal import SLSConformal
            from torchregress.losses.sls import SLSLoss

            context_dim = hidden
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=context_dim)
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
            _train_one(
                model, loss_fn, X_train, y_train, config, loss_params=list(loss_fn.parameters())
            )
            model.eval()
            X_cal = torch.from_numpy(splits["X_cal"]).float().to(current_device())
            y_cal = torch.from_numpy(splits["y_cal"]).float().to(current_device())
            with torch.no_grad():
                ctx_cal = model(X_cal)
            predictor = SLSConformal(loss_fn, alpha=0.1)
            predictor.calibrate(ctx_cal.detach(), y_cal.reshape(-1, 1))
            y_min = float(y_train.min())
            y_max = float(y_train.max())
            pad = max(y_max - y_min, 1e-3) * 0.1
            return {
                "type": "sls_cp",
                "model": model,
                "predictor": predictor,
                "y_min": y_min - pad,
                "y_max": y_max + pad,
            }

        # --- CTI: NF density + Conformal Thresholded Intervals ---
        elif method == "CTI":
            from torchregress.losses.conformal import CTI
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            context_dim = hidden
            flow = create_flow_model(
                n_features=1,
                context_dim=context_dim,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=context_dim)
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_one(
                model, loss_fn, X_train, y_train, config, loss_params=list(flow.parameters())
            )
            model.eval()
            X_cal = torch.from_numpy(splits["X_cal"]).float().to(current_device())
            y_cal = torch.from_numpy(splits["y_cal"]).float().to(current_device())
            with torch.no_grad():
                log_dens_cal = loss_fn.log_prob(model(X_cal), y_cal)
            predictor = CTI(alpha=0.1)
            predictor.calibrate(log_dens_cal.detach(), y_cal.reshape(-1))
            y_min = float(y_train.min())
            y_max = float(y_train.max())
            pad = max(y_max - y_min, 1e-3) * 0.1
            return {
                "type": "cti",
                "model": model,
                "flow": loss_fn,
                "predictor": predictor,
                "y_min": y_min - pad,
                "y_max": y_max + pad,
            }

        # --- Skew/shape NLLs: distribution-MLE head + split-conformal ---
        # No samplers exist in families.py, so intervals come from conformal
        # residuals on the location column (honest: likelihood trains the
        # point, conformal covers the shape misspecification).
        elif method in _SKEW_METHODS:
            from torchregress.losses import families as _fam

            spec = _SKEW_METHODS[method]
            X_cal = torch.from_numpy(splits["X_cal"]).float().to(current_device())
            y_cal = torch.from_numpy(splits["y_cal"]).float().to(current_device())
            y_min = y_max = None
            y_tr = y_train
            if spec.get("normalize01"):
                y_min = float(y_train.min())
                y_max = float(y_train.max())
                span = max(y_max - y_min, 1e-8)
                # Map into (0, 1) strictly: beta NLL rejects exact 0/1.
                y_tr = 0.01 + 0.98 * (y_train - y_min) / span
                y_cal = 0.01 + 0.98 * (y_cal - y_min) / span
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=spec["width"])
            loss_fn = getattr(_fam, spec["loss"])()
            _train_one(model, loss_fn, X_train, y_tr, config)
            model.eval()
            return {
                "type": "skew_cp",
                "model": model,
                "X_cal": X_cal,
                "y_cal": y_cal,
                "y_min": y_min,
                "y_max": y_max,
            }

        # --- SQR: 32-level sorted-quantile head (own intervals) ---
        elif method == "SQR":
            from torchregress.losses import SQRLoss

            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=32)
            _train_one(model, SQRLoss(), X_train, y_train, config)
            model.eval()
            return {"type": "sqr", "model": model}

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
        model_type = model_state["type"]
        X_t = torch.from_numpy(X_test).float().to(current_device())
        model = model_state.get("model")

        # --- Split CP → post-hoc calibration ---
        if model_type == "split_cp":
            lo, hi = _split_cp_intervals(
                model,
                model_state["X_cal"],
                model_state["y_cal"],
                X_t,
            )
            return {"lower": lo, "upper": hi}

        # --- MonteCarloCP → sample-based conformal intervals ---
        elif model_type == "mc_cp":
            model = model_state["model"]
            flow_loss = model_state["flow"]
            predictor = model_state["predictor"]
            model.eval()
            with torch.no_grad():
                samples = flow_loss.sample(model(X_t), n_samples=256).squeeze(-1)
                samples_np = samples.detach().cpu().numpy()
            lo, hi = predictor.predict_interval(samples.T)
            import numpy as _np

            return {
                "lower": _np.asarray(lo.detach().cpu()).reshape(-1),
                "upper": _np.asarray(hi.detach().cpu()).reshape(-1),
                "y_pred": samples_np.mean(axis=1).reshape(-1),
                "samples": samples_np,
            }

        # --- SLS_CP → calibrated grid level-set intervals (d=1) ---
        elif model_type == "sls_cp":
            model = model_state["model"]
            predictor = model_state["predictor"]
            model.eval()
            with torch.no_grad():
                ctx = model(X_t)
            lo, hi = predictor.predict_interval_from_grid(
                ctx, model_state["y_min"], model_state["y_max"]
            )
            import numpy as _np

            lo_np = _np.asarray(lo.detach().cpu()).reshape(-1)
            hi_np = _np.asarray(hi.detach().cpu()).reshape(-1)
            return {
                "lower": lo_np,
                "upper": hi_np,
                "y_pred": ((lo_np + hi_np) / 2.0).reshape(-1),
            }

        # --- CTI → density level-set intervals from the NF ---
        elif model_type == "cti":
            model = model_state["model"]
            flow_loss = model_state["flow"]
            predictor = model_state["predictor"]
            model.eval()

            def _density_fn(y_grid: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
                # (G,) grid × (B, f) inputs → (B, G) log-densities.
                with torch.no_grad():
                    ctx = model(xb)
                    B = ctx.shape[0]
                    G = y_grid.numel()
                    ctx_rep = ctx.unsqueeze(1).expand(-1, G, -1).reshape(B * G, -1)
                    y_rep = y_grid.reshape(1, -1, 1).expand(B, -1, -1).reshape(B * G, 1)
                    return flow_loss.log_prob(ctx_rep, y_rep).reshape(B, G)

            lo, hi = predictor.predict_intervals_from_density(
                _density_fn, X_t, model_state["y_min"], model_state["y_max"]
            )
            import numpy as _np

            lo_np = _np.asarray(lo.detach().cpu()).reshape(-1)
            hi_np = _np.asarray(hi.detach().cpu()).reshape(-1)
            with torch.no_grad():
                mean_np = (
                    flow_loss.sample(model(X_t), n_samples=64)
                    .squeeze(-1)
                    .mean(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
            return {"lower": lo_np, "upper": hi_np, "y_pred": mean_np}

        # --- Skew/shape NLL + split-conformal on the location column ---
        elif model_type == "skew_cp":
            y_min, y_max = model_state.get("y_min"), model_state.get("y_max")
            lo, hi, loc = _skew_cp_intervals(
                model,
                model_state["X_cal"],
                model_state["y_cal"],
                X_t,
            )
            if y_min is not None:
                span = max(y_max - y_min, 1e-8)

                def _inv(v: np.ndarray) -> np.ndarray:
                    return (v - 0.01) / 0.98 * span + y_min  # type: ignore[no-any-return]

                lo, hi, loc = _inv(lo), _inv(hi), _inv(loc)
            return {"lower": lo, "upper": hi, "y_pred": loc}

        # --- SQR → sorted-level intervals (own quantiles, no conformal) ---
        elif model_type == "sqr":
            model.eval()
            with torch.no_grad():
                arr = model(X_t).detach().cpu().numpy()  # (N, 32)
            # Sort levels per row (SQR trains toward monotonicity but smoke
            # budgets can leave crossings — same treatment as MultiExpectile).
            arr = np.sort(arr, axis=1)

            def _q(p: float) -> np.ndarray:
                return arr[:, min(31, int(p * 32))].reshape(-1)

            return {"lower": _q(0.05), "upper": _q(0.95), "y_pred": _q(0.5)}

        # --- GaussianNLL → z-score intervals ---
        elif model_type == "gaussian_nll":
            mean, std = _gaussian_nll_predict(model, X_test)
            return {
                "lower": mean - _Z_SCORE * std,
                "upper": mean + _Z_SCORE * std,
                "y_pred": mean,
                # Expose mean/std so compute_pit_values dispatches to the
                # Gaussian-PIT branch (norm.cdf).
                "mean": mean,
                "std": std,
            }

        # --- NormalizingFlow → sample quantile intervals ---
        elif model_type == "nflow":
            flow_loss = model_state["flow"]
            model.eval()
            with torch.no_grad():
                context = model(X_t)
                samples = flow_loss.sample(context, n_samples=256).squeeze(-1)  # (N, 256)
                samples_np = samples.detach().cpu().numpy()
            lo = np.quantile(samples_np, 0.05, axis=1)
            hi = np.quantile(samples_np, 0.95, axis=1)
            # Return MC samples so compute_pit_values can compute PIT diagnostics
            # (Proposal #9) — these drive the overconfidence detection that
            # motivates the Flow Ensemble (Proposal #10).
            return {
                "lower": lo,
                "upper": hi,
                "y_pred": samples_np.mean(axis=1),
                "samples": samples_np,
            }

        # --- NF_Ensemble_K5 → pooled sample quantile intervals ---
        elif model_type == "nflow_ensemble":
            members = model_state["members"]
            per_member: list[np.ndarray] = []
            n_test = X_t.shape[0]
            for m in members:
                m["model"].eval()
                with torch.no_grad():
                    ctx = m["model"](X_t)
                    s = (
                        m["flow"]
                        .sample(
                            ctx,
                            n_samples=self._NF_ENSEMBLE_N_SAMPLES,
                        )
                        .squeeze(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )  # expected shape (N, n_samples)
                # Accept both (N, S) post-squeeze and (S, N) layouts; fail
                # loudly for ambiguous shapes rather than silently
                # reshaping (which would corrupt the data).
                if s.ndim == 2 and s.shape == (n_test, self._NF_ENSEMBLE_N_SAMPLES):
                    pass  # (N, S) — expected
                elif s.ndim == 2 and s.shape == (self._NF_ENSEMBLE_N_SAMPLES, n_test):
                    s = s.T  # (S, N) — transpose to (N, S)
                else:
                    raise RuntimeError(
                        f"NF ensemble member produced unexpected shape "
                        f"{s.shape} (expected ({n_test}, "
                        f"{self._NF_ENSEMBLE_N_SAMPLES}))"
                    )
                per_member.append(s)
            # Pool all members' samples → (N, K*n_samples) for quantile estimation
            pooled = np.concatenate(per_member, axis=1)
            lo = np.quantile(pooled, 0.05, axis=1)
            hi = np.quantile(pooled, 0.95, axis=1)
            # Return the pooled samples too so compute_all_metrics can compute
            # PIT diagnostics on the full distribution.
            return {
                "lower": lo,
                "upper": hi,
                "y_pred": pooled.mean(axis=1),
                "samples": pooled,
            }

        # --- MDN → sample-based quantile intervals ---
        elif model_type == "mdn":
            loss_fn = model_state["loss_fn"]
            model.eval()
            with torch.no_grad():
                context = model(X_t)
                samples = loss_fn.sample(context, n_samples=256).squeeze(-1).T  # (N, 256)
                samples_np = samples.detach().cpu().numpy()
            lo = np.quantile(samples_np, 0.05, axis=1)
            hi = np.quantile(samples_np, 0.95, axis=1)
            # Return MC samples for PIT diagnostics.
            return {
                "lower": lo,
                "upper": hi,
                "samples": samples_np,
            }

        # --- Evidential → apply constraints + Student-t predictive intervals ---
        elif model_type == "evidential":
            import torch.nn.functional as F

            loss_fn = model_state["loss_fn"]
            model.eval()
            with torch.no_grad():
                raw = model(X_t)
                gamma = raw[:, 0:1]
                nu = F.softplus(raw[:, 1:2]) + 0.01
                alpha = F.softplus(raw[:, 2:3]) + 1.01
                beta = F.softplus(raw[:, 3:4]) + 0.01
                params = torch.cat([gamma, nu, alpha, beta], dim=1)
                lo, hi = loss_fn.predict_interval(params, confidence=0.90)
                lo_np = lo.detach().cpu().numpy().reshape(-1)
                hi_np = hi.detach().cpu().numpy().reshape(-1)
            # Return NIG parameters so compute_pit_values can compute
            # Student-t PIT (Proposal #9): df=2*α, loc=γ, scale²=β(1+ν)/(να).
            return {
                "lower": lo_np,
                "upper": hi_np,
                "gamma": gamma.detach().cpu().numpy().reshape(-1),
                "nu": nu.detach().cpu().numpy().reshape(-1),
                "alpha": alpha.detach().cpu().numpy().reshape(-1),
                "beta": beta.detach().cpu().numpy().reshape(-1),
            }

        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # evaluate — standard interval metrics + PIT diagnostics
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        from harness.metrics import (
            compute_interval_metrics,
            compute_pit_values,
            pit_uniformity_tests,
        )

        lo = np.asarray(predictions["lower"]).reshape(-1)
        hi = np.asarray(predictions["upper"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)

        metrics = compute_interval_metrics(lo, hi, y_true)

        # --- PIT calibration diagnostics (Proposal #9) ---
        # Dispatcher auto-detects: samples (NF/MDN/Ensemble), mean+std
        # (GaussianNLL), gamma/nu/alpha/beta (Evidential).  Split_CP
        # falls back to uniform placeholder (it is calibrated by-design).
        pit = compute_pit_values(predictions, y_true)
        if pit.size >= 3:
            metrics.update(pit_uniformity_tests(pit))

        # Store raw predictions for post-hoc binning (handled by suite.run override)
        predictions["_lo_raw"] = lo
        predictions["_hi_raw"] = hi
        predictions["_y_true_raw"] = y_true

        return metrics

    # ------------------------------------------------------------------
    # Override run() — carve cal split + add per-bin evaluation
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        from time import perf_counter

        cfg = config or HarnessConfig()

        dataset_name, splits = self._prepare_scenario_run(
            scenario_name,
            dataset_name,
            splits,
            standardize_targets=cfg.standardize_targets,
        )

        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name, standardize_targets=cfg.standardize_targets)

        # Carve calibration split for Split_CP method
        augmented = _carve_calibration_split(
            splits,
            cal_frac=self.CALIBRATION_FRAC,
            seed=cfg.seed,
            smoke=cfg.smoke,
            suite_name="distributional_bins",
        )

        X_test_np = augmented["X_test"]
        y_test_np = augmented["y_test"]

        rows: list[dict[str, Any]] = []
        all_predictions: dict[str, dict[str, np.ndarray]] = {}

        # --- Phase 1: fit + predict all methods ---
        for method in self.methods():
            set_harness_seed(cfg.seed)
            try:
                t0 = perf_counter()
                model_state = self.fit(method, augmented, cfg)
                fit_time = perf_counter() - t0

                t0 = perf_counter()
                predictions = self.predict(method, model_state, X_test_np, cfg)
                pred_time = perf_counter() - t0

                if "_y_scaler" in augmented:
                    predictions["_y_scaler"] = augmented["_y_scaler"]

                metrics = self.evaluate(method, predictions, y_test_np, cfg)

                all_predictions[method] = predictions

                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    **metrics,
                    "train_s": fit_time,
                    "eval_s": pred_time,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": self.method_notes(method),
                }
            except (
                ValueError,
                RuntimeError,
                ImportError,
                KeyError,
                TypeError,
                OSError,
                NameError,
            ) as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "Coverage": None,
                    "Width": None,
                    "IntervalScore": None,
                    "train_s": None,
                    "eval_s": None,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }
            rows.append(row)

        # --- Phase 2: per-bin comparison ---
        if "Split_CP" in all_predictions:
            cp_preds = all_predictions["Split_CP"]
            cp_width = np.asarray(cp_preds["_hi_raw"]) - np.asarray(cp_preds["_lo_raw"])
            y_true = np.asarray(cp_preds["_y_true_raw"])

            width_min = float(cp_width.min())
            width_max = float(cp_width.max())
            if width_max - width_min < 1e-10:
                # Degenerate: all widths identical — single bin
                bin_edges = np.array([width_min - 0.1, width_max + 0.1])
                bin_assignments = np.zeros(len(cp_width), dtype=int)
            else:
                bin_edges = np.linspace(width_min, width_max, self.N_BINS + 1)
                bin_assignments = np.digitize(cp_width, bin_edges[1:-1])

            for method, preds in all_predictions.items():
                lo_m = np.asarray(preds["_lo_raw"])
                hi_m = np.asarray(preds["_hi_raw"])
                bin_results = _bin_metrics(
                    bin_edges,
                    bin_assignments,
                    lo_m,
                    hi_m,
                    y_true,
                    target_coverage=self.TARGET_COVERAGE,
                )

                # Per-bin rows with coverage deviation
                deviations: list[float] = []
                for b_idx, br in enumerate(bin_results):
                    dev = br["CoverageDeviation"]
                    if not np.isnan(dev):
                        deviations.append(dev)
                    rows.append(
                        {
                            "Method": f"{self.name}/{method}/bin_{b_idx}",
                            "Library": self.library_name(method),
                            "Dataset": dataset_name,
                            "Suite": self.name,
                            "Coverage": br["BinCoverage"],
                            "Width": br["BinMeanWidth"],
                            "BinCenter": br["BinCenter"],
                            "BinSize": br["BinSize"],
                            "CoverageDeviation": br["CoverageDeviation"],
                            "IntervalScore": None,
                            "train_s": None,
                            "eval_s": None,
                            "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                            "Device": str(current_device()),
                            "Notes": f"Bin {b_idx} (center={br['BinCenter']:.4f}), n={br['BinSize']}",
                        }
                    )

                # Aggregate deviation summary row per method (size-weighted)
                if deviations:
                    devs = np.array(deviations)
                    sizes = np.array(
                        [
                            br["BinSize"]
                            for br in bin_results
                            if not np.isnan(br.get("CoverageDeviation", np.nan))
                        ]
                    )
                    valid_coverages = np.array(
                        [
                            br["BinCoverage"]
                            for br in bin_results
                            if not np.isnan(br.get("CoverageDeviation", np.nan))
                        ]
                    )
                    valid_widths = np.array(
                        [
                            br["BinMeanWidth"]
                            for br in bin_results
                            if not np.isnan(br.get("CoverageDeviation", np.nan))
                        ]
                    )
                    w_mean_dev = (
                        float(np.average(devs, weights=sizes))
                        if sizes.sum() > 0
                        else float(np.mean(devs))
                    )
                    w_mean_cov = (
                        float(np.average(valid_coverages, weights=sizes))
                        if sizes.sum() > 0
                        else float(np.mean(valid_coverages))
                    )
                    w_mean_wid = (
                        float(np.average(valid_widths, weights=sizes))
                        if sizes.sum() > 0
                        else float(np.mean(valid_widths))
                    )
                    w_rmse = (
                        float(np.sqrt(np.average(devs**2, weights=sizes)))
                        if sizes.sum() > 0
                        else float(np.sqrt(np.mean(devs**2)))
                    )
                    rows.append(
                        {
                            "Method": f"{self.name}/{method}/dev_summary",
                            "Library": self.library_name(method),
                            "Dataset": dataset_name,
                            "Suite": self.name,
                            "AvgBinCoverage": w_mean_cov,
                            "AvgBinWidth": w_mean_wid,
                            "CoverageDeviationMean": w_mean_dev,
                            "CoverageDeviationAbsMean": float(
                                np.average(np.abs(devs), weights=sizes)
                                if sizes.sum() > 0
                                else np.mean(np.abs(devs))
                            ),
                            "CoverageDeviationRMSE": w_rmse,
                            "IntervalScore": None,
                            "train_s": None,
                            "eval_s": None,
                            "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                            "Device": str(current_device()),
                            "Notes": f"Aggregate deviation across {len(deviations)} bins (size-weighted)",
                        }
                    )

        return rows


# ---------------------------------------------------------------------------
# Custom evidential training with softplus constraints
# ---------------------------------------------------------------------------


def _train_evidential(
    model: nn.Module,
    loss_fn: Any,
    X: torch.Tensor,
    y: torch.Tensor,
    config: HarnessConfig,
) -> None:
    """Train evidential model, applying softplus constraints to raw outputs."""
    import torch.nn.functional as F
    import torch.optim as optim

    cfg = config
    _epochs = cfg.epochs
    _batch = cfg.batch_size
    _lr = cfg.lr

    params = list(model.parameters())
    for p in loss_fn.parameters():
        if p.requires_grad:
            params.append(p)
    opt = optim.Adam(params, lr=_lr)
    n = X.shape[0]

    for _ in range(_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, _batch):
            idx = perm[i : i + _batch]
            opt.zero_grad(set_to_none=True)
            raw = model(X[idx])
            # Apply evidential constraints: gamma (mean), nu>0, alpha>1, beta>0
            gamma = raw[:, 0:1]
            nu = F.softplus(raw[:, 1:2]) + 0.01
            alpha = F.softplus(raw[:, 2:3]) + 1.01
            beta = F.softplus(raw[:, 3:4]) + 0.01
            constrained = torch.cat([gamma, nu, alpha, beta], dim=1)
            loss = loss_fn(constrained, y[idx])
            loss.backward()
            opt.step()

    # After training, apply constraints for prediction
    # The model now outputs raw logits; predict() will apply softplus
    model.eval()
