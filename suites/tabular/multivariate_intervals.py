"""Multivariate prediction-interval suite: Gaussian vs NF vs NF-ensemble vs SLS.

Compares five distributional methods for producing prediction intervals on
multi-dimensional regression targets.  Each method learns a joint distribution
over the target vector and produces marginal intervals (per output dimension)
as well as a joint coverage metric.

Methods
-------
* ``GaussianNLL`` — diagonal Gaussian NLL → marginal μ ± z·σ intervals
* ``MultivariateGaussian`` — full-covariance Gaussian NLL → marginal intervals
  from Cholesky-parameterised covariance
* ``NormalizingFlow`` — zuko NSF flow → sample-based quantile intervals
* ``NF_Ensemble_K{2,5,10,20}`` — K independent NSF flows with pooled samples
  for per-dim quantile intervals (Proposal #10 extended to multivariate
  targets; K-sweep family per §15)
* ``SLS`` — Super-Level-Set regression → boundary-sampled marginal intervals
  from minimum-volume level sets
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import NF_ENSEMBLE_K_VARIANTS, _hidden_size, _train_one

# ---------------------------------------------------------------------------
# Multivariate-capable backbones
# ---------------------------------------------------------------------------


class _MultiOutputMLP(nn.Module):
    """MLP backbone for multi-dimensional target regression.

    Outputs ``out_dim`` values — the raw head logits for whichever loss
    function the suite uses.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# GaussianNLL (diagonal) helper — produce marginal intervals
# ---------------------------------------------------------------------------


def _gaussian_nll_multi_predict(
    model: nn.Module,
    X: np.ndarray,
    n_features: int,
    *,
    z_score: float = 1.645,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract marginal mean & std from a 2*D output diagonal Gaussian model."""
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X).float())  # (N, 2*D)
        mean = out[:, :n_features].numpy()
        log_var = out[:, n_features:].numpy()
        std = np.sqrt(np.exp(np.clip(log_var, -20, 20)) + 1e-8)
    return mean, std


# ---------------------------------------------------------------------------
# MultivariateGaussian (full covariance) helpers
# ---------------------------------------------------------------------------


def _covariance_from_cholesky_params(L_params: torch.Tensor, D: int) -> torch.Tensor:
    """Build a batch of PSD covariance matrices from raw Cholesky parameters.

    Diagonal entries are passed through softplus for positivity, producing
    a valid lower-triangular Cholesky factor L.  Covariance is Σ = L @ Lᵀ.

    L_params: (N, D*(D+1)//2) — lower-triangular entries, row-major.
    Returns:  (N, D, D) covariance matrices.
    """
    N = L_params.shape[0]
    device = L_params.device
    dtype = L_params.dtype
    L = torch.zeros(N, D, D, device=device, dtype=dtype)
    idx = 0
    for i in range(D):
        L[:, i, i] = torch.nn.functional.softplus(L_params[:, idx]) + 1e-4
        idx += 1
        for j in range(i):
            L[:, i, j] = L_params[:, idx]
            idx += 1
    return L @ L.transpose(-1, -2)


def _multivariate_gaussian_predict(
    model: nn.Module,
    X: np.ndarray,
    D: int,
    *,
    z_score: float = 1.645,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract marginal mean & std from a full-covariance Gaussian model.

    Model outputs (N, D + D*(D+1)//2): mean (D) + Cholesky params.
    """
    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(X).float())
        mean = raw[:, :D]
        L_params = raw[:, D:]
        cov = _covariance_from_cholesky_params(L_params, D)
        # Marginal std = sqrt(diagonal of covariance)
        var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8)
        std = torch.sqrt(var)
        return mean.numpy(), std.numpy()


class _CholeskyGaussianLossWrapper(nn.Module):
    """Wraps MultivariateGaussianLoss for use with _train_one.

    Splits raw model output into mean + Cholesky params, builds covariance,
    then delegates to the underlying loss.
    """

    def __init__(self, D: int, loss_fn: nn.Module) -> None:
        super().__init__()
        self.D = D
        self.loss_fn = loss_fn

    def forward(self, raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = raw[:, : self.D]
        L_params = raw[:, self.D :]
        cov = _covariance_from_cholesky_params(L_params, self.D)
        return self.loss_fn(mean, target, covariance_matrices=cov)


# ---------------------------------------------------------------------------
# SLS (Super-Level-Set) helpers — boundary sampling + inverse flow
# ---------------------------------------------------------------------------


def _sls_predict_intervals(
    model: nn.Module,
    loss_fn: nn.Module,
    X: np.ndarray,
    D: int,
    *,
    n_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract marginal intervals from SLS level sets via boundary sampling.

    The SLS level set in flow-space is the ellipsoid
        {z : ||L @ (z - mu)||² <= q}
    where L is the frontier's Cholesky factor, mu is the learned centre,
    and q is the quantile threshold from ``quantile_net``.

    We sample points on the *boundary* of this ellipsoid by solving
    ``L @ z = w`` for random unit directions ``w`` scaled to radius ``√q``,
    then inverse-flow to map back to target space.  Per-dimension min/max
    of the mapped samples defines the marginal prediction intervals.
    """
    import torch.nn.functional as F

    frontier = loss_fn.frontier
    quantile_net = loss_fn.quantile_net
    flow = frontier.flow

    model.eval()
    X_t = torch.from_numpy(X).float()
    N = X_t.shape[0]

    with torch.no_grad():
        context = model(X_t)  # (N, context_dim)
        q_vals = quantile_net(context)  # (N, 3)
        q = q_vals[:, 1]  # (N,) — the tau-quantile threshold
        mu, L_params = frontier._get_params(context)  # (N, D), (N, num_L)
        L_full, _ = frontier._get_L_matrix_and_logdet(L_params)  # (N, D, D)

        # Sample random unit directions on the sphere, scale to boundary
        r = torch.randn(n_samples, N, D)
        r = F.normalize(r, dim=-1)
        r = r * torch.sqrt(q).view(1, -1, 1)  # (S, N, D) — boundary of ellipsoid

        # Solve L @ z = r  →  z = L^{-1} @ r  (broadcasting: (1,N,D,D) vs (S,N,D,1))
        z = torch.linalg.solve_triangular(
            L_full.unsqueeze(0),
            r.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)  # (S, N, D)

        # Shift to ellipsoid centre in flow-space
        z_flow = z + mu.unsqueeze(0)  # (S, N, D)

        # Inverse flow: map from flow-space back to target y-space
        ctx_repeated = context.unsqueeze(0).expand(n_samples, -1, -1).reshape(-1, context.shape[-1])
        y_samples = flow.inverse(z_flow.reshape(-1, D), ctx_repeated)
        y_samples = y_samples.reshape(n_samples, N, D)

        y_np = y_samples.numpy()
        lower = y_np.min(axis=0)  # (N, D)
        upper = y_np.max(axis=0)  # (N, D)
        mean = mu.numpy()  # (N, D) — learned centre

    return mean, lower, upper


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class MultivariateIntervalsSuite(HarnessRunner):
    """Multivariate prediction-interval comparison suite."""

    _N_FEATURES = 3  # number of output dimensions

    _NF_ENSEMBLE_K_REFERENCE = 10
    _NF_ENSEMBLE_HIDDEN = [64, 64]
    _NF_ENSEMBLE_TRANSFORMS = 3
    _NF_ENSEMBLE_N_SAMPLES = 512

    @property
    def name(self) -> str:
        return "multivariate_intervals"

    @property
    def description(self) -> str:
        return (
            "Multivariate intervals: GaussianNLL vs MultivariateGaussian "
            "vs NormalizingFlow vs NF-Ensemble K\u2208{2,5,10,20} vs SLS"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "synthetic_multivariate",
            "synthetic_mv_gaussian",
            "synthetic_mv_nongaussian",
            "synthetic_mv_bimodal",
        ]

    # ------------------------------------------------------------------
    # Method registry
    #    # NF_Ensemble_K-sweep family (Proposal #10 §15): all four K variants
    # (_K2 / _K5 / _K10 / _K20) share identical architecture — only K
    # itself varies (parsed from the method name at fit time).  This
    # mirrors distributional_bins.py and lets us sweep K on the 4 MV
    # datasets to confirm or refute the §11b inference that MV
    # JointCoverage does not benefit from larger K.  The shared tuple
    # `NF_ENSEMBLE_K_VARIANTS` is imported from `suites._backbones`
    # (single source of truth across scalar / MV suites).
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        return [
            "GaussianNLL",
            "MultivariateGaussian",
            "NormalizingFlow",
            *NF_ENSEMBLE_K_VARIANTS,
            "SLS",
        ]

    def library_name(self, method: str) -> str:
        return "torchregress"

    def method_notes(self, method: str) -> str:
        # K-sweep context (per-distributional_bins.py pattern) — single
        # template body, K parsed from the method name, then dispatched
        # through k_sweep_contexts for the per-K annotation.
        if method in NF_ENSEMBLE_K_VARIANTS:
            K = int(method.rsplit("_K", 1)[-1])
            k_sweep_contexts: dict[str, tuple[str, str]] = {
                "NF_Ensemble_K2": ("K-sweep minimum", "validated in \u00a715"),
                "NF_Ensemble_K5": (
                    "K-sweep \u00a711b baseline",
                    "superseded by K=10 (new production default)",
                ),
                "NF_Ensemble_K10": (
                    "code-level production default per \u00a715 IS-knee",
                    "validated in \u00a711b/\u00a715",
                ),
                "NF_Ensemble_K20": ("K-sweep ceiling", "validated in \u00a715"),
            }
            ctx_text, section_text = k_sweep_contexts[method]
            return (
                f"{K} independent NSF NormalizingFlows (seed offsets, "
                f"{self._NF_ENSEMBLE_HIDDEN} hidden, {self._NF_ENSEMBLE_TRANSFORMS} transforms) "
                f"with pooled samples ({K}\u00d7{self._NF_ENSEMBLE_N_SAMPLES} samples/test point) "
                f"for per-dim 5%/95% quantile intervals \u2014 multivariate extension "
                f"of Proposal #10 ensemble remedy ({ctx_text}; {section_text})"
            )
        notes = {
            "GaussianNLL": "diagonal Gaussian NLL \u2192 marginal \u03bc \u00b1 z\u00b7\u03c3 intervals per output dim",
            "MultivariateGaussian": "full-covariance Gaussian NLL via Cholesky factor \u2192 marginal std from cov diagonal",
            "NormalizingFlow": "zuko NSF flow \u2192 sample-based quantile intervals per output dim",
            "SLS": "Super-Level-Set regression \u2192 boundary sampling + inverse flow \u2192 marginal intervals per output dim",
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

        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"]  # may be (N*D,) after harness standardization
        in_dim = X_train_np.shape[1]
        hidden = _hidden_size(in_dim)
        D = self._N_FEATURES

        # Reshape flattened targets back to (N, D)
        if y_train_np.ndim == 1:
            y_train_np = y_train_np.reshape(-1, D)

        X = torch.from_numpy(X_train_np).float()
        y = torch.from_numpy(y_train_np).float()

        # --- GaussianNLL: diagonal ---
        if method == "GaussianNLL":
            from torchregress.losses import GaussianNLLLoss

            model = _MultiOutputMLP(in_dim, hidden, out_dim=2 * D)
            loss_fn = GaussianNLLLoss()
            _train_one(model, loss_fn, X, y, config)
            model.eval()
            return {"type": "gaussian_nll", "model": model, "n_features": D}

        # --- MultivariateGaussian: full covariance via Cholesky ---
        elif method == "MultivariateGaussian":
            from torchregress.losses import MultivariateGaussianLoss

            n_params = D + D * (D + 1) // 2  # mean(D) + lower-tri Cholesky params
            model = _MultiOutputMLP(in_dim, hidden, out_dim=n_params)
            mv_loss = MultivariateGaussianLoss(n_features=D)
            loss_fn = _CholeskyGaussianLossWrapper(D, mv_loss)
            _train_one(model, loss_fn, X, y, config)
            model.eval()
            return {"type": "multivariate_gaussian", "model": model, "n_features": D}

        # --- NormalizingFlow ---
        elif method == "NormalizingFlow":
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            context_dim = hidden
            # 64-unit hidden layers (matched to distributional_bins fix).
            # 32-unit layers produce overconfident, too-peaked densities
            # that under-cover on multivariate targets as well.
            flow = create_flow_model(
                n_features=D,
                context_dim=context_dim,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            model = _MultiOutputMLP(in_dim, hidden, out_dim=context_dim)
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_one(model, loss_fn, X, y, config, loss_params=list(flow.parameters()))
            model.eval()
            return {"type": "nflow", "model": model, "flow": loss_fn, "n_features": D}

        # --- NF_Ensemble_K{2,5,10,20}: K independent NSF flows (Proposal #10 extended) ---
        # K parsed from method name \u2014 architecture locked across the family
        # (`_NF_ENSEMBLE_HIDDEN` / `_NF_ENSEMBLE_TRANSFORMS`) so the \u00a715
        # calibration-vs-cost sweep isolates K's effect.
        elif method in NF_ENSEMBLE_K_VARIANTS:
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            K = int(method.rsplit("_K", 1)[-1])
            context_dim = hidden
            # Mirror scalar-distributional_bins ensemble config: same
            # hidden/transforms as single NF, but K seed-offset flows.
            members: list[dict[str, Any]] = []
            for i in range(K):
                set_harness_seed(config.seed + i)
                flow_i = create_flow_model(
                    n_features=D,
                    context_dim=context_dim,
                    flow_type="nsf",
                    n_transforms=self._NF_ENSEMBLE_TRANSFORMS,
                    hidden_features=self._NF_ENSEMBLE_HIDDEN,
                )
                model_i = _MultiOutputMLP(in_dim, hidden, out_dim=context_dim)
                loss_i = NormalizingFlowLoss(flow=flow_i)
                _train_one(
                    model_i,
                    loss_i,
                    X,
                    y,
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
            return {"type": "nflow_ensemble", "members": members, "n_features": D, "K": K}

        # --- SLS: Super-Level-Set regression ---
        elif method == "SLS":
            from torchregress.losses.sls import SLSLoss

            context_dim = hidden
            model = _MultiOutputMLP(in_dim, hidden, out_dim=context_dim)
            # hidden_dim=64, n_transforms=4 match SLSLoss defaults.
            # 32/2 (previous) under-expresses the flow — JointCoverage
            # drops from 0.92 to 0.88 (−4.5%), IS from 1.21 to 1.24.
            # warmup_steps=50 kept (vs library default 500): sweep
            # shows no gain above 50 on these datasets.
            loss_fn = SLSLoss(
                d=D,
                context_dim=context_dim,
                K=1,
                mode="full",
                hidden_dim=64,
                n_transforms=4,
                tau=0.9,
                warmup_steps=50,
            )
            _train_one(
                model,
                loss_fn,
                X,
                y,
                config,
                loss_params=list(loss_fn.parameters()),
            )
            model.eval()
            return {"type": "sls", "model": model, "loss_fn": loss_fn, "n_features": D}

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
        D = model_state["n_features"]
        model_type = model_state["type"]
        z_score = 1.645  # 90% prediction interval (5% each tail)

        # --- GaussianNLL → marginal intervals ---
        if model_type == "gaussian_nll":
            mean, std = _gaussian_nll_multi_predict(
                model_state["model"],
                X_test,
                D,
                z_score=z_score,
            )
            lower = mean - z_score * std
            upper = mean + z_score * std
            # Flatten for harness interval metrics (per-dimension marginal)
            return {
                "lower": lower.reshape(-1),
                "upper": upper.reshape(-1),
                "y_pred": mean.reshape(-1),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- MultivariateGaussian → marginal intervals from cov diagonal ---
        elif model_type == "multivariate_gaussian":
            mean, std = _multivariate_gaussian_predict(
                model_state["model"],
                X_test,
                D,
                z_score=z_score,
            )
            lower = mean - z_score * std
            upper = mean + z_score * std
            return {
                "lower": lower.reshape(-1),
                "upper": upper.reshape(-1),
                "y_pred": mean.reshape(-1),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- NormalizingFlow → sample-based quantile intervals ---
        elif model_type == "nflow":
            model = model_state["model"]
            flow_loss = model_state["flow"]
            model.eval()

            X_t = torch.from_numpy(X_test).float()
            with torch.no_grad():
                context = model(X_t)
                samples = flow_loss.sample(context, n_samples=512)  # (N, 512, D)
                samples_np = samples.numpy()

            lo_q = 0.05
            hi_q = 0.95
            lower = np.quantile(samples_np, lo_q, axis=1)  # (N, D)
            upper = np.quantile(samples_np, hi_q, axis=1)
            mean = samples_np.mean(axis=1)

            return {
                "lower": lower.reshape(-1),
                "upper": upper.reshape(-1),
                "y_pred": mean.reshape(-1),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- NF_Ensemble_K5 → pooled per-dim quantile intervals ---
        elif model_type == "nflow_ensemble":
            members = model_state["members"]
            n_test = X_test.shape[0]
            X_t = torch.from_numpy(X_test).float()
            per_member: list[np.ndarray] = []
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
                        .numpy()
                    )  # expected (N, n_samples, D)
                if s.ndim != 3 or s.shape[0] != n_test or s.shape[-1] != D:
                    raise RuntimeError(
                        f"NF ensemble member produced unexpected shape "
                        f"{s.shape} (expected ({n_test}, *, {D}))"
                    )
                per_member.append(s)
            # Pool K members, take per-dim 5%/95% quantiles (Proposal #10).
            pooled = np.concatenate(per_member, axis=1)  # (N, K*n_samples, D)
            lower = np.quantile(pooled, 0.05, axis=1)  # (N, D)
            upper = np.quantile(pooled, 0.95, axis=1)
            mean = pooled.mean(axis=1)
            return {
                "lower": lower.reshape(-1),
                "upper": upper.reshape(-1),
                "y_pred": mean.reshape(-1),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- SLS → boundary-sampled marginal intervals ---
        elif model_type == "sls":
            mean, lower, upper = _sls_predict_intervals(
                model_state["model"],
                model_state["loss_fn"],
                X_test,
                D,
                n_samples=500,
            )
            return {
                "lower": lower.reshape(-1),
                "upper": upper.reshape(-1),
                "y_pred": mean.reshape(-1),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # evaluate — add joint-coverage metric for multivariate targets
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        from harness.metrics import compute_interval_metrics

        is_multi = predictions.get("_multi_output", False)

        if is_multi:
            D = predictions["_n_features"]
            N = predictions["_y_test_shape"][0]
            # Reshape flat lower/upper back to (N, D)
            lower = np.asarray(predictions["lower"]).reshape(N, D)
            upper = np.asarray(predictions["upper"]).reshape(N, D)
            y_true = np.asarray(y_test).reshape(N, D)

            # Marginal metrics (flattened — harness standard)
            metrics = compute_interval_metrics(
                lower.reshape(-1),
                upper.reshape(-1),
                y_true.reshape(-1),
            )

            # Joint coverage: fraction of points where ALL dims are in interval
            in_interval = (y_true >= lower) & (y_true <= upper)
            joint_coverage = float(np.mean(in_interval.all(axis=1)))
            metrics["JointCoverage"] = joint_coverage

            # Per-dimension coverage
            for d in range(D):
                metrics[f"Coverage_D{d}"] = float(np.mean(in_interval[:, d]))

            return metrics

        # Scalar fallback
        return compute_interval_metrics(
            np.asarray(predictions.get("lower", predictions.get("y_pred", np.zeros(1)))),
            np.asarray(predictions.get("upper", predictions.get("y_pred", np.zeros(1)))),
            np.asarray(y_test).reshape(-1),
        )
