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
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, NF_ENSEMBLE_K_VARIANTS, _hidden_size, _train_one

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


class _ReshapeHead(nn.Module):
    """Reshape a flat head to [B, Q, D] for joint multi-quantile losses."""

    def __init__(self, base: nn.Module, n_quantiles: int, n_features: int) -> None:
        super().__init__()
        self.base = base
        self.n_quantiles = n_quantiles
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x).reshape(-1, self.n_quantiles, self.n_features)


# ---------------------------------------------------------------------------
# GaussianNLL (diagonal) helper — produce marginal intervals
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Predictive sampling helpers (energy / variogram scoring)
# ---------------------------------------------------------------------------

_N_SCORING_SAMPLES = 256


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _diag_gaussian_samples(
    mean: np.ndarray, std: np.ndarray, seed: int, n: int = _N_SCORING_SAMPLES
) -> np.ndarray:
    """Draw [S, N, D] samples from a diagonal Gaussian (seeded)."""
    rng = _seeded_rng(seed)
    z = rng.standard_normal((n,) + mean.shape).astype(np.float64)
    return mean.astype(np.float64)[None] + std.astype(np.float64)[None] * z


def _full_gaussian_samples(
    mean: np.ndarray, cov: np.ndarray, seed: int, n: int = _N_SCORING_SAMPLES
) -> np.ndarray:
    """Draw [S, N, D] samples from full-covariance Gaussians (seeded)."""
    rng = _seeded_rng(seed)
    N, D = mean.shape
    out = np.empty((n, N, D), dtype=np.float64)
    for i in range(N):
        try:
            L = np.linalg.cholesky(cov[i] + 1e-8 * np.eye(D))
        except np.linalg.LinAlgError:
            L = np.sqrt(np.diag(cov[i]).clip(min=1e-8)) * np.eye(D)
        out[:, i, :] = mean[i] + rng.standard_normal((n, D)) @ L.T
    return out


def _to_scoring_samples(
    arr: np.ndarray, n_points: int, seed: int, n: int = _N_SCORING_SAMPLES
) -> np.ndarray:
    """Normalize flow samples to [S, N, D] with a seeded S-subsample."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] == n_points:
        a = np.swapaxes(a, 0, 1)  # (N, S, D) → (S, N, D)
    if a.shape[0] > n:
        a = a[_seeded_rng(seed).choice(a.shape[0], size=n, replace=False)]
    return a


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
        out = model(torch.from_numpy(X).float().to(current_device()))  # (N, 2*D)
        mean = out[:, :n_features].detach().cpu().numpy()
        log_var = out[:, n_features:].detach().cpu().numpy()
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
        raw = model(torch.from_numpy(X).float().to(current_device()))
        mean = raw[:, :D]
        L_params = raw[:, D:]
        cov = _covariance_from_cholesky_params(L_params, D)
        # Marginal std = sqrt(diagonal of covariance)
        var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8)
        std = torch.sqrt(var)
        return mean.detach().cpu().numpy(), std.detach().cpu().numpy()


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


class _LowRankGaussianLossWrapper(nn.Module):
    """Wraps LowRankGaussianLoss for use with _train_one.

    Splits raw model output into mean + factor + diag (mirrors the
    keypoint-regression image adapter).
    """

    def __init__(self, D: int, rank: int, loss_fn: nn.Module) -> None:
        super().__init__()
        self.D = D
        self.rank = rank
        self.loss_fn = loss_fn

    def forward(self, raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        D, r = self.D, self.rank
        mean = raw[:, :D]
        factor = raw[:, D : D + D * r].reshape(-1, D, r)
        diag = F.softplus(raw[:, D + D * r :]) + 1e-4
        return self.loss_fn(mean, target, cov_factor=factor, cov_diag=diag)


def _mv_full_params(model: nn.Module, X: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw (mean, full covariance) from a Cholesky-parameterized MV model."""
    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(X).float().to(current_device()))
        mean = raw[:, :D].detach().cpu().numpy().astype(np.float64)
        cov = (
            _covariance_from_cholesky_params(raw[:, D:], D)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    return mean, cov


def _lowrank_gaussian_predict(
    model: nn.Module, X: np.ndarray, D: int, rank: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean/std + factor-model samples from a low-rank Gaussian head."""
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(X).float().to(current_device()))
        mean_t = raw[:, :D]
        factor = raw[:, D : D + D * rank].reshape(-1, D, rank)
        diag = F.softplus(raw[:, D + D * rank :]) + 1e-4
        var = (factor.pow(2).sum(-1) + diag).clamp_min(1e-8)
        mean = mean_t.detach().cpu().numpy().astype(np.float64)
        std = torch.sqrt(var).detach().cpu().numpy().astype(np.float64)
        # Factor sampling: y = mean + F @ z_r + sqrt(diag) * z_d.
        rng = np.random.default_rng(seed)
        S = _N_SCORING_SAMPLES
        F_np = factor.detach().cpu().numpy().astype(np.float64)
        d_np = diag.detach().cpu().numpy().astype(np.float64)
        N = mean.shape[0]
        z_r = rng.standard_normal((S, N, rank))
        z_d = rng.standard_normal((S, N, D))
        samples = mean[None] + (F_np[None] * z_r[:, :, None, :]).sum(-1) + np.sqrt(d_np)[None] * z_d
    return mean, std, samples


def _mdn_full_samples(
    log_weights: np.ndarray,
    means: np.ndarray,
    chol: np.ndarray,
    seed: int,
    n: int = _N_SCORING_SAMPLES,
) -> np.ndarray:
    """Seeded [S, N, D] draws from a full-covariance Gaussian mixture.

    log_weights/mean/chol are [N, K], [N, K, D], [N, K, D, D] arrays.
    """
    rng = np.random.default_rng(seed)
    N, K = log_weights.shape
    D = means.shape[-1]
    w = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    comp = np.stack([rng.choice(K, size=n, p=w[i]) for i in range(N)], axis=1)  # [S, N]
    z = rng.standard_normal((n, N, D))
    out = np.empty((n, N, D), dtype=np.float64)
    for s in range(n):
        ks = comp[s]
        out[s] = means[np.arange(N), ks] + np.einsum("nij,nj->ni", chol[np.arange(N), ks], z[s])
    return out


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
    X_t = torch.from_numpy(X).float().to(current_device())
    N = X_t.shape[0]

    with torch.no_grad():
        context = model(X_t)  # (N, context_dim)
        q_vals = quantile_net(context)  # (N, 3)
        q = q_vals[:, 1]  # (N,) — the tau-quantile threshold
        mu, L_params = frontier._get_params(context)  # (N, D), (N, num_L)
        if getattr(frontier, "mode", "full") == "low_rank":
            # Low-rank frontier returns (diag, factor): materialize the dense
            # covariance and Cholesky it — exact, trivial cost at small D —
            # then run the identical triangular-solve path below.
            (D_diag, V), _ = frontier._get_L_matrix_and_logdet(L_params)
            cov = torch.diag_embed(D_diag) + V @ V.transpose(-1, -2)
            jitter = 1e-6 * torch.eye(cov.shape[-1], device=cov.device)
            L_full = torch.linalg.cholesky(cov + jitter)
        else:
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

        y_np = y_samples.detach().cpu().numpy()
        lower = y_np.min(axis=0)  # (N, D)
        upper = y_np.max(axis=0)  # (N, D)
        mean = mu.detach().cpu().numpy()  # (N, D) — learned centre

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
            "energy_efficiency",
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
            "LowRankGaussian",
            "WassersteinBound",
            "MultiTargetCP",
            "MVScore",
            "NormalizingFlow",
            *NF_ENSEMBLE_K_VARIANTS,
            "SLS",
            "SLS_lowrank",
            "MDN",
            "MultiQuantile",
            "HeteroEnsembleCov",
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
            "GaussianNLL": "diagonal Gaussian NLL → marginal μ ± z·σ intervals per output dim",
            "MultivariateGaussian": "full-covariance Gaussian NLL via Cholesky factor → marginal std from cov diagonal",
            "LowRankGaussian": "low-rank + diag Gaussian NLL (factor samples) → scalable MV",
            "MultiTargetCP": "per-dimension conformal thresholds on a point MLP",
            "MVScore": "Mahalanobis-score conformal regions (bounding-box intervals)",
            "WassersteinBound": "supervised mean+var (pseudo-labeled cov) via Wasserstein surrogate",
            "NormalizingFlow": "zuko NSF flow → sample-based quantile intervals per output dim",
            "SLS": "Super-Level-Set regression → boundary sampling + inverse flow → marginal intervals per output dim",
            "SLS_lowrank": "SLS with low-rank frontier (dense-Cholesky boundary sampling) → scalable SLS",
            "MDN": "multivariate mixture (3 full-cov Gaussians) → sampled intervals + full scoring",
            "MultiQuantile": "marginal 10/50/90% quantiles per dim → own intervals (no joint samples)",
            "HeteroEnsembleCov": "heteroscedastic ensemble → full epi+alea covariance → marginals + samples",
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

        # Reshape flattened targets back to (N, D). D comes from the
        # target shape when 2D, else the n_targets split tag (energy=2),
        # else the synthetic default (3).
        if y_train_np.ndim == 1:
            D = int(splits.get("n_targets", self._N_FEATURES))
            y_train_np = y_train_np.reshape(-1, D)
        else:
            D = int(y_train_np.shape[1])

        X = torch.from_numpy(X_train_np).float().to(current_device())
        y = torch.from_numpy(y_train_np).float().to(current_device())

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

        # --- LowRankGaussian: factor + diag covariance (scalable MV) ---
        elif method == "LowRankGaussian":
            from torchregress.losses import LowRankGaussianLoss

            rank = min(2, D)
            model = _MultiOutputMLP(in_dim, hidden, out_dim=D + D * rank + D)
            lr_loss = LowRankGaussianLoss()
            loss_fn = _LowRankGaussianLossWrapper(D, rank, lr_loss)
            _train_one(model, loss_fn, X, y, config)
            model.eval()
            return {
                "type": "lowrank_gaussian",
                "model": model,
                "n_features": D,
                "rank": rank,
            }

        # --- MVScore: Mahalanobis-score conformal regions on an MV base ---
        # Region {y : (y-mu)^T Sigma^-1 (y-mu) <= r}; reported as the
        # bounding box mu ± sqrt(r * diag(Sigma)) per dim (conservative).
        elif method == "MVScore":
            from torchregress.losses import MultivariateGaussianLoss
            from torchregress.losses.conformal import MultivariateScoreConformal

            n_params = D + D * (D + 1) // 2
            model = _MultiOutputMLP(in_dim, hidden, out_dim=n_params)
            mv_loss = MultivariateGaussianLoss(n_features=D)
            loss_fn = _CholeskyGaussianLossWrapper(D, mv_loss)
            _train_one(model, loss_fn, X, y, config)
            model.eval()
            n = X.shape[0]
            n_cal = max(10, n // 5)
            gen = torch.Generator(device="cpu").manual_seed(config.seed)
            perm = torch.randperm(n, generator=gen)
            Xc, yc = X[perm[:n_cal]], y[perm[:n_cal]]
            with torch.no_grad():
                out_c = model(Xc)
                mu_c = out_c[:, :D]
                cov_c = _covariance_from_cholesky_params(out_c[:, D:], D)
            predictor = MultivariateScoreConformal(alpha=0.1)
            predictor.calibrate(mu_c.detach(), cov_c.detach(), yc)
            return {"type": "mvscore", "model": model, "predictor": predictor, "n_features": D}

        # --- MultiTargetCP: per-dimension conformal on a point MLP ---
        elif method == "MultiTargetCP":
            from torchregress.losses.conformal import MultiTargetConformal

            model = _MultiOutputMLP(in_dim, hidden, out_dim=D)
            _train_one(model, nn.MSELoss(), X, y, config)
            model.eval()
            n = X.shape[0]
            n_cal = max(10, n // 5)
            gen = torch.Generator(device="cpu").manual_seed(config.seed)
            perm = torch.randperm(n, generator=gen)
            Xc, yc = X[perm[:n_cal]], y[perm[:n_cal]]
            with torch.no_grad():
                p_cal = model(Xc)
            predictor = MultiTargetConformal(alpha=0.1)
            predictor.calibrate(p_cal.detach(), yc)
            return {
                "type": "multitarget_cp",
                "model": model,
                "predictor": predictor,
                "n_features": D,
            }

        # --- WassersteinBound: supervised mean + diagonal variance ---
        # Target variances come from neighborhood covariance pseudo-labels
        # (empirical-Bayes supervision for the covariance head).
        elif method == "WassersteinBound":
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            from torchregress.algorithms.covariance_pseudo_labels import (
                NeighborhoodCovariancePseudoLabeler,
            )
            from torchregress.losses.gaussian_wasserstein import GaussianWassersteinBoundLoss

            with torch.no_grad():
                pseudo = NeighborhoodCovariancePseudoLabeler().fit_predict(X, y)
            # (N, D, D) or (N, D) → diagonal variances.
            tgt_var = (
                pseudo.diagonal(dim1=-2, dim2=-1) if pseudo.dim() == 3 else pseudo
            ).clamp_min(1e-6)
            model = _MultiOutputMLP(in_dim, hidden, out_dim=2 * D)
            loss_fn = GaussianWassersteinBoundLoss(covariance_parameterization="diagonal")
            loader = DataLoader(
                TensorDataset(X, y, tgt_var), batch_size=config.batch_size, shuffle=True
            )
            opt = optim.Adam(model.parameters(), lr=config.lr)
            model.train()
            for _ in range(config.epochs):
                for Xb, yb, vb in loader:
                    opt.zero_grad(set_to_none=True)
                    out = model(Xb)
                    import torch.nn.functional as F

                    pred_var = F.softplus(out[:, D:]).clamp_min(1e-6)
                    loss = loss_fn(out[:, :D], yb, pred_var, vb)
                    loss.backward()
                    opt.step()
            model.eval()
            return {"type": "wasserstein", "model": model, "n_features": D}

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

        # --- SLS_lowrank: low-rank frontier (scalable SLS) ---
        elif method == "SLS_lowrank":
            from torchregress.losses.sls import SLSLoss

            context_dim = hidden
            model = _MultiOutputMLP(in_dim, hidden, out_dim=context_dim)
            loss_fn = SLSLoss(
                d=D,
                context_dim=context_dim,
                K=1,
                mode="low_rank",
                rank=min(2, D),
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

        # --- MDN: multivariate mixture (full-cov components) ---
        elif method == "MDN":
            from torch.utils.data import DataLoader, TensorDataset
            from torchregress.losses import MixtureDensityLoss

            n_comp = 3
            mdn_loss = MixtureDensityLoss(n_components=n_comp, n_features=D, covariance_type="full")
            model = _MultiOutputMLP(in_dim, hidden, out_dim=mdn_loss.expected_output_size)
            _train_one(model, mdn_loss, X, y, config)
            model.eval()
            return {
                "type": "mdn",
                "model": model,
                "loss_fn": mdn_loss,
                "n_features": D,
            }

        # --- MultiQuantile: marginal quantiles per dim ---
        elif method == "MultiQuantile":
            from torchregress.losses import MultiQuantileLoss

            q_levels = [0.1, 0.5, 0.9]
            n_q = len(q_levels)
            base = _MultiOutputMLP(in_dim, hidden, out_dim=n_q * D)
            model = _ReshapeHead(base, n_q, D)
            loss_fn = MultiQuantileLoss(quantiles=q_levels)
            _train_one(model, loss_fn, X, y, config)
            model.eval()
            return {
                "type": "multiquantile",
                "model": model,
                "n_features": D,
                "quantiles": q_levels,
            }

        # --- HeteroEnsembleCov: full epi+alea covariance from ensembles ---
        elif method == "HeteroEnsembleCov":
            from torch.utils.data import DataLoader, TensorDataset
            from torchregress.ensemble import HeteroscedasticEnsembleModel
            from torchregress.losses import GaussianNLLLoss

            ens = HeteroscedasticEnsembleModel(
                _MLP,
                ensemble_size=3,
                device=str(current_device()),
                in_dim=in_dim,
                hidden=hidden,
                out_dim=2 * D,
            )
            loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(X_train_np).float(),
                    torch.from_numpy(y_train_np).float(),
                ),
                batch_size=config.batch_size,
                shuffle=True,
            )
            ens.fit(
                loader,
                GaussianNLLLoss(),
                epochs=config.epochs,
                verbose=False,
                device=str(current_device()),
            )
            ens.eval()
            return {"type": "heteroenscov", "model": ens, "n_features": D}

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
                "samples": _diag_gaussian_samples(mean, std, config.seed),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- WassersteinBound → marginal intervals from var head ---
        elif model_type == "wasserstein":
            import torch.nn.functional as F

            model = model_state["model"]
            model.eval()
            with torch.no_grad():
                out = model(torch.from_numpy(X_test).float().to(current_device()))
                mean = out[:, :D].detach().cpu().numpy()
                std = np.sqrt(F.softplus(out[:, D:]).clamp_min(1e-8).detach().cpu().numpy())
            lower = mean - z_score * std
            upper = mean + z_score * std
            return {
                "lower": lower.reshape(-1),
                "upper": upper.reshape(-1),
                "y_pred": mean.reshape(-1),
                "mean": mean.reshape(-1),
                "std": std.reshape(-1),
                "samples": _diag_gaussian_samples(mean, std, config.seed),
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
                "samples": _full_gaussian_samples(
                    *_mv_full_params(model_state["model"], X_test, D),
                    config.seed,
                ),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- LowRankGaussian → marginal intervals + factor samples ---
        elif model_type == "lowrank_gaussian":
            mean, std, samples = _lowrank_gaussian_predict(
                model_state["model"],
                X_test,
                D,
                model_state["rank"],
                config.seed,
            )
            lower = mean - z_score * std
            upper = mean + z_score * std
            return {
                "lower": lower.reshape(-1),
                "upper": upper.reshape(-1),
                "y_pred": mean.reshape(-1),
                "samples": samples,
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- MDN → mixture-sampled intervals (multimodal MV) ---
        elif model_type == "mdn":
            model = model_state["model"]
            loss_fn = model_state["loss_fn"]
            model.eval()
            with torch.no_grad():
                params = model(torch.from_numpy(X_test).float().to(current_device()))
                lw, means, chol = loss_fn._extract_distribution_parameters(params)
            samples = _mdn_full_samples(
                lw.detach().cpu().numpy().astype(np.float64),
                means.detach().cpu().numpy().astype(np.float64),
                chol.detach().cpu().numpy().astype(np.float64),
                config.seed,
            )
            mean = samples.mean(axis=0)
            lower = np.quantile(samples, 0.05, axis=0)
            upper = np.quantile(samples, 0.95, axis=0)
            return {
                "lower": lower.reshape(-1),
                "upper": upper.reshape(-1),
                "y_pred": mean.reshape(-1),
                "samples": samples,
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- MultiQuantile → marginal quantile intervals per dim ---
        elif model_type == "multiquantile":
            model = model_state["model"]
            qs = model_state["quantiles"]
            model.eval()
            with torch.no_grad():
                out = model(torch.from_numpy(X_test).float().to(current_device()))
            q = out.detach().cpu().numpy().astype(np.float64)  # (N, Q, D)
            lo = q[:, 0, :]
            med = q[:, len(qs) // 2, :]
            hi = q[:, -1, :]
            return {
                "lower": lo.reshape(-1),
                "upper": hi.reshape(-1),
                "y_pred": med.reshape(-1),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- HeteroEnsembleCov → full-covariance ensemble intervals ---
        elif model_type == "heteroenscov":
            ens = model_state["model"]
            ens.eval()
            dev = next(ens.parameters()).device
            with torch.no_grad():
                d = ens.predict_full_covariance(torch.from_numpy(X_test).float().to(dev))
            mean_np = d["mean"].reshape(-1, D).detach().cpu().numpy().astype(np.float64)
            cov_np = d["total_covariance"].detach().cpu().numpy().astype(np.float64)
            std_np = np.sqrt(np.diagonal(cov_np, axis1=-2, axis2=-1).clip(min=1e-12))
            lower = mean_np - z_score * std_np
            upper = mean_np + z_score * std_np
            return {
                "lower": lower.reshape(-1),
                "upper": upper.reshape(-1),
                "y_pred": mean_np.reshape(-1),
                "samples": _full_gaussian_samples(mean_np, cov_np, config.seed),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- MultiTargetCP → per-dimension conformal intervals ---
        elif model_type == "multitarget_cp":
            model = model_state["model"]
            predictor = model_state["predictor"]
            model.eval()
            with torch.no_grad():
                out = model(torch.from_numpy(X_test).float().to(current_device()))
            lo, hi = predictor.predict_interval(out)
            import numpy as _np

            lo_np = _np.asarray(lo.detach().cpu())
            hi_np = _np.asarray(hi.detach().cpu())
            mean_np = _np.asarray(out.detach().cpu())
            N = len(X_test)
            return {
                "lower": lo_np.reshape(-1),
                "upper": hi_np.reshape(-1),
                "y_pred": mean_np.reshape(-1),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (N, D),
            }

        # --- MVScore → ellipsoid bounding-box intervals ---
        elif model_type == "mvscore":
            model = model_state["model"]
            predictor = model_state["predictor"]
            model.eval()
            with torch.no_grad():
                out = model(torch.from_numpy(X_test).float().to(current_device()))
                mean = out[:, :D]
                cov = _covariance_from_cholesky_params(out[:, D:], D)
            r = predictor.region_radius()
            r_val = float(r.detach().cpu()) if isinstance(r, torch.Tensor) else float(r)
            var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8)
            half = np.sqrt(r_val * var.detach().cpu().numpy())
            mean_np = mean.detach().cpu().numpy()
            N = len(X_test)
            return {
                "lower": (mean_np - half).reshape(-1),
                "upper": (mean_np + half).reshape(-1),
                "y_pred": mean_np.reshape(-1),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (N, D),
            }

        # --- NormalizingFlow → sample-based quantile intervals ---
        elif model_type == "nflow":
            model = model_state["model"]
            flow_loss = model_state["flow"]
            model.eval()

            X_t = torch.from_numpy(X_test).float().to(current_device())
            with torch.no_grad():
                context = model(X_t)
                samples = flow_loss.sample(context, n_samples=512)  # (N, 512, D)
                samples_np = samples.detach().cpu().numpy()

            lo_q = 0.05
            hi_q = 0.95
            lower = np.quantile(samples_np, lo_q, axis=1)  # (N, D)
            upper = np.quantile(samples_np, hi_q, axis=1)
            mean = samples_np.mean(axis=1)

            return {
                "lower": lower.reshape(-1),
                "upper": upper.reshape(-1),
                "y_pred": mean.reshape(-1),
                "samples": _to_scoring_samples(samples_np, len(X_test), config.seed),
                "_multi_output": True,
                "_n_features": D,
                "_y_test_shape": (len(X_test), D),
            }

        # --- NF_Ensemble_K5 → pooled per-dim quantile intervals ---
        elif model_type == "nflow_ensemble":
            members = model_state["members"]
            n_test = X_test.shape[0]
            X_t = torch.from_numpy(X_test).float().to(current_device())
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
                        .detach()
                        .cpu()
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
                "samples": _to_scoring_samples(pooled, len(X_test), config.seed),
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

            # Proper scoring rules for distribution-valued forecasts
            # (Gneiting & Raftery 2007; Scheuerer & Hamill 2015). Only rows
            # emitting predictive "samples" [S, N, D] qualify — conformal/SLS
            # rows output regions, not distributions, and honestly skip these.
            if "samples" in predictions:
                from torchregress.metrics import energy_score, variogram_score

                samp = np.asarray(predictions["samples"], dtype=np.float64)
                if samp.ndim == 3 and samp.shape[0] == N and samp.shape[1] != N:
                    samp = np.swapaxes(samp, 0, 1)
                try:
                    metrics["EnergyScore"] = float(energy_score(samp, y_true.astype(np.float64)))
                    metrics["VariogramScore"] = float(
                        variogram_score(samp, y_true.astype(np.float64))
                    )
                except Exception:
                    pass

            return metrics

        # Scalar fallback
        return compute_interval_metrics(
            np.asarray(predictions.get("lower", predictions.get("y_pred", np.zeros(1)))),
            np.asarray(predictions.get("upper", predictions.get("y_pred", np.zeros(1)))),
            np.asarray(y_test).reshape(-1),
        )
