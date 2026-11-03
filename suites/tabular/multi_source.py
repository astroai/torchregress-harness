"""Multi-survey shared-latent likelihoods, survey calibration heads and forward quadrature.

A synthetic multi-survey catalogue with a known ground truth: every object has
a shared three-dimensional latent ``theta`` (think Teff / logg / [Fe/H]) and
eight noisy features ``x``.  Four surveys (A, B, C, D) and a sparse anchor
survey (E) measure subsets of ``theta`` with

* partial, per-survey missingness (NaN values, NaN reported errors),
* heteroscedastic reported errors (per object and survey),
* a full measurement covariance: within-survey correlations between latent
  dimensions *and* cross-survey correlations (A and B share a reduction
  pipeline, so their errors correlate),
* on ``multisurvey_biased``: per-survey additive offsets, scale
  miscalibration (bias linear in ``theta``) and a quality-conditioned bias,
  the anchor survey being unbiased,
* on ``multisurvey_outliers``: heavy-tailed contamination of whole
  measurement blocks (the A+B pipeline block, C, D).

Every row predicts a Gaussian latent ``N(mu(x), L(x) L(x)^T)`` with the same
MLP and budget and is scored against the *true* latent of held-out objects
(RMSE, NLL, marginal and joint 90% coverage, PIT Cramer-von Mises) and by the
held-out NLL of the observed survey measurements under the row's own
observation model.

``multisurvey_forward`` is a scalar latent seen through a nonlinear map
(flux = exp(z) plus Gaussian noise, negative fluxes kept) for
``GaussianLatentForwardObservationNLLLoss`` vs a linearized baseline.  The
held-out observation NLL uses a dense-grid reference integral (checked against
``scipy.integrate.quad``) and reports the error of Gauss-Hermite orders
8..256 on the same predictions (the quadrature convergence check the docs ask
for).

``multisurvey_big`` (1M training objects, mini-batched from host memory, no
full-dataset device tensors) times the library likelihood against the
harness's batched reference implementation of the same likelihood and an
inverse-variance baseline, with peak CUDA memory when run on CUDA.

Exactness checks (library vs ``torch.distributions.MultivariateNormal`` closed
form for the linear-Gaussian case, direct vs information factorizations, the
harness batched reference; forward quadrature on a linear map vs the analytic
Gaussian is reported beside them but not gated, being documented as a
finite-order approximation; B-spline basis and derivative vs ``scipy.interpolate.BSpline``) are
rows of their own.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy import stats
from torch import Tensor, nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP

_D = 3  # latent dimension
_P = 8  # feature dimension
# Surveys: (name, measured latent dims, observation probability, base error sd).
_SURVEYS: tuple[tuple[str, tuple[int, ...], float, tuple[float, ...]], ...] = (
    ("A", (0, 1, 2), 0.70, (0.15, 0.25, 0.10)),
    ("B", (0, 1, 2), 0.50, (0.20, 0.20, 0.15)),
    ("C", (0, 2), 0.40, (0.30, 0.25)),
    ("D", (0,), 0.50, (0.10,)),
    ("E", (0, 1, 2), 0.05, (0.05, 0.05, 0.05)),  # sparse unbiased anchor
)
_S = len(_SURVEYS)
_ANCHOR = 4
_SRC = np.array([s for s, (_, dims, _, _) in enumerate(_SURVEYS) for _ in dims])
_DIM = np.array([d for (_, dims, _, _) in _SURVEYS for d in dims])
_K = len(_SRC)
# Robust-likelihood blocks: A and B share a pipeline (cross-correlated) -> one block.
_BLOCK_OF_SURVEY = np.array([0, 0, 1, 2, 3])
_BLOCK = _BLOCK_OF_SURVEY[_SRC]
_THETA_COV = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, -0.3], [0.2, -0.3, 1.0]])
# Biases on multisurvey_biased: a + c * theta_d + gamma * q_s (anchor E unbiased).
_BIAS_A = np.array(
    [[0.10, -0.15, 0.05], [-0.20, 0.10, 0.12], [0.25, 0.0, -0.10], [0.15, 0.0, 0.0], [0, 0, 0]]
)
_BIAS_C = np.array(
    [[0.05, 0.0, 0.0], [-0.08, 0.06, 0.0], [0.10, 0.0, 0.05], [0.0, 0.0, 0.0], [0, 0, 0]]
)
_BIAS_G = np.array(
    [[0.04, 0.03, 0.0], [0.0, 0.05, 0.05], [0.08, 0.0, 0.0], [0.06, 0.0, 0.0], [0, 0, 0]]
)
_OUTLIER_P = 0.08
_KNOTS = [-3.0] * 4 + [-1.5, 0.0, 1.5] + [3.0] * 4
_HIDDEN = 64

_SIZES = {  # (n_train, n_test) full / smoke
    "multisurvey": ((10_000, 4_000), (256, 128)),
    "multisurvey_biased": ((10_000, 4_000), (256, 128)),
    "multisurvey_outliers": ((6_000, 3_000), (192, 96)),
    "multisurvey_forward": ((10_000, 4_000), (256, 128)),
    "multisurvey_big": ((1_000_000, 100_000), (4_000, 1_000)),
}
_OBS_EVAL_MAX = 20_000  # per-star held-out observation NLL is evaluated on at most this many

_METHODS_BY_DATASET: dict[str, tuple[str, ...]] = {
    "multisurvey": (
        "Exactness_LinearGaussian",
        "MS_Marginal",
        "MS_BatchedRef",
        "Joint_DiagR",
        "PerSource_Product",
        "IVW_Deconv",
        "IVW_GaussNLL",
        "ZeroFillMissing",
        "Oracle_TrueLatent",
    ),
    "multisurvey_biased": (
        "SplineBasis_ScipyParity",
        "MS_NoCalib",
        "MS_RelSpline",
        "MS_RelSpline_Common",
        "MS_FullCalib",
        "IVW_Deconv",
    ),
    "multisurvey_outliers": ("MS_Marginal", "MS_Robust", "IVW_Deconv", "Oracle_TrueLatent"),
    "multisurvey_forward": ("Fwd_Quad_Q32", "Fwd_Quad_Q4", "Fwd_Linearized", "Fwd_Oracle"),
    "multisurvey_big": ("MS_Marginal", "MS_BatchedRef", "IVW_Deconv"),
}

_NOTES = {
    "Exactness_LinearGaussian": (
        "float64 linear-Gaussian case: library per-star NLL vs MultivariateNormal closed form, "
        "direct vs information, harness batched reference (values and mean gradients), "
        "forward quadrature with a linear map vs the analytic Gaussian"
    ),
    "SplineBasis_ScipyParity": (
        "bspline_basis_with_derivative vs scipy.interpolate.BSpline (values, derivatives, "
        "linear tails), constant-boundary policy, OOD flags and autograd derivative"
    ),
    "MS_Marginal": "MarginalizedMultiSourceGaussianNLLLoss, full measurement covariance, masked",
    "MS_BatchedRef": (
        "harness batched (padded, vectorized) reimplementation of the same marginal likelihood; "
        "timing reference for the library's per-star loop"
    ),
    "Joint_DiagR": "library joint likelihood with within/cross-survey error correlations dropped",
    "PerSource_Product": (
        "product of per-measurement Gaussian marginals (latent integrated separately per "
        "measurement; ignores shared latent and correlations)"
    ),
    "IVW_Deconv": (
        "per-dimension inverse-variance average of the surveys (diagonal errors), then "
        "Gaussian NLL with the IVW variance added (deconvolved, correlations ignored)"
    ),
    "IVW_GaussNLL": "Gaussian NLL on inverse-variance averaged labels; measurement noise ignored",
    "ZeroFillMissing": (
        "naive baseline: missing measurements filled with 0 at the survey's base error and "
        "treated as observed (what masking prevents)"
    ),
    "Oracle_TrueLatent": "Gaussian NLL on the true latent (unobservable reference)",
    "MS_NoCalib": "marginal likelihood, survey biases not modelled",
    "MS_RelSpline": "marginal likelihood + RelativeSurveySplineHead (zero-sum relative offsets)",
    "MS_RelSpline_Common": ("+ CommonScaleSplineHead identified by the unbiased anchor survey E"),
    "MS_FullCalib": "+ ConditionBiasHead (quality-conditioned bias, zero at reference quality)",
    "MS_Robust": (
        "RobustMultiSourceGaussianNLLLoss, blocks {A+B pipeline, C, D, E}, "
        f"p={_OUTLIER_P}, scale=10; contamination is Student-t, not the model's Gaussian inflation"
    ),
    "Fwd_Quad_Q32": "GaussianLatentForwardObservationNLLLoss, 32 Gauss-Hermite nodes",
    "Fwd_Quad_Q4": "GaussianLatentForwardObservationNLLLoss, 4 nodes (under-resolved)",
    "Fwd_Linearized": "delta-method Gaussian: N(y; exp(mu), sigma^2 + exp(2 mu) s^2)",
    "Fwd_Oracle": "Gaussian NLL on the true latent (unobservable reference)",
}
_LIBRARY = {
    "MS_BatchedRef": "harness",
    "PerSource_Product": "harness",
    "IVW_Deconv": "harness",
    "IVW_GaussNLL": "harness",
    "Oracle_TrueLatent": "harness",
    "Fwd_Linearized": "harness",
    "Fwd_Oracle": "harness",
}


# --------------------------------------------------------------------- data
def _correlation() -> np.ndarray:
    c = np.eye(_K)
    pos = {(s, d): k for k, (s, d) in enumerate(zip(_SRC, _DIM, strict=True))}

    def put(a: tuple[int, int], b: tuple[int, int], rho: float) -> None:
        c[pos[a], pos[b]] = c[pos[b], pos[a]] = rho

    put((0, 0), (0, 1), 0.4)  # A: dims 0-1 correlated
    put((1, 0), (1, 2), -0.3)  # B: dims 0-2 anticorrelated
    put((2, 0), (2, 2), 0.2)  # C
    for d in range(_D):  # A and B share a reduction pipeline
        put((0, d), (1, d), 0.5)
    put((0, 0), (1, 1), 0.2)
    put((0, 1), (1, 0), 0.2)
    return c


_CORR = _correlation()


def _true_bias(theta: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """``[N,K]`` bias of each packed measurement at the true latent."""
    a = _BIAS_A[_SRC, _DIM]
    c = _BIAS_C[_SRC, _DIM]
    g = _BIAS_G[_SRC, _DIM]
    return a + c * theta[:, _DIM] + g * quality[:, _SRC]


def _generate(name: str, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    design = np.random.default_rng(20261103)  # same feature map for every seed
    w1 = design.normal(size=(_D, _P))
    w2 = design.normal(size=(_D, _P)) * 0.5
    theta = rng.multivariate_normal(np.zeros(_D), _THETA_COV, size=n)
    x = np.tanh(theta @ w1) + theta @ w2 + 0.25 * rng.normal(size=(n, _P))
    observed_survey = rng.random((n, _S)) < np.array([p for _, _, p, _ in _SURVEYS])
    mask = observed_survey[:, _SRC]
    base = np.concatenate([np.asarray(b) for _, _, _, b in _SURVEYS])
    log_m = rng.uniform(-0.4, 0.8, size=(n, _S))  # per (object, survey) error multiplier
    sigma = base * np.exp(log_m[:, _SRC])
    quality = -(log_m - 0.2) / 0.35  # SNR-like condition, reference 0
    quality[:, _ANCHOR] = 0.0  # the anchor is the reference quality by construction
    noise = sigma * (rng.normal(size=(n, _K)) @ np.linalg.cholesky(_CORR).T)
    values = theta[:, _DIM] + noise
    bias = np.zeros((n, _K))
    if name == "multisurvey_biased":
        bias = _true_bias(theta, quality)
        values = values + bias
    if name == "multisurvey_outliers":
        hit = rng.random((n, 3)) < _OUTLIER_P  # blocks A+B, C, D (anchor clean)
        heavy = sigma * 6.0 * rng.standard_t(df=2, size=(n, _K))
        block_hit = np.zeros((n, _K), dtype=bool)
        for b in range(3):
            block_hit[:, _BLOCK == b] = hit[:, [b]]
        values = values + np.where(block_hit, heavy, 0.0)
    values = np.where(mask, values, np.nan)
    sigma = np.where(mask, sigma, np.nan)
    return {
        "x": x.astype(np.float32),
        "theta": theta.astype(np.float32),
        "values": values.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "mask": mask,
        "quality": quality.astype(np.float32),
        "bias": bias.astype(np.float32),
    }


def _generate_forward(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    x = rng.normal(size=(n, 4))
    scatter = 0.2 + 0.2 * np.abs(x[:, 0])
    z = (
        0.8 * np.sin(x[:, 0])
        + 0.5 * x[:, 1]
        - 0.25 * x[:, 2] * x[:, 3]
        + scatter * rng.normal(size=n)
    )
    err = np.full(n, 0.15)
    y = np.exp(z) + err * rng.normal(size=n)
    mask = rng.random(n) < 0.9
    return {
        "x": x.astype(np.float32),
        "theta": z.astype(np.float32),
        "values": np.where(mask, y, np.nan).astype(np.float32),
        "sigma": err.astype(np.float32),
        "mask": mask,
    }


def make_data(name: str, cfg: HarnessConfig) -> dict[str, dict[str, np.ndarray]]:
    """Train/test catalogues for one dataset (seeded by ``cfg.seed``)."""
    if name not in _SIZES:
        raise KeyError(f"multi_source: unknown dataset {name!r}")
    full, smoke = _SIZES[name]
    n_train, n_test = smoke if cfg.smoke else full
    rng = np.random.default_rng(int(cfg.seed))
    if name == "multisurvey_forward":
        return {"train": _generate_forward(n_train, rng), "test": _generate_forward(n_test, rng)}
    return {"train": _generate(name, n_train, rng), "test": _generate(name, n_test, rng)}


def _ivw(values: np.ndarray, sigma: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension inverse-variance average of observed measurements (diagonal errors)."""
    n = len(values)
    w = np.where(mask, 1.0 / np.where(mask, sigma, 1.0) ** 2, 0.0)
    y = np.where(mask, values, 0.0)
    num, den = np.zeros((n, _D)), np.zeros((n, _D))
    for k in range(_K):
        num[:, _DIM[k]] += w[:, k] * y[:, k]
        den[:, _DIM[k]] += w[:, k]
    has = den > 0
    avg = np.where(has, num / np.where(has, den, 1.0), np.nan)
    var = np.where(has, 1.0 / np.where(has, den, 1.0), np.nan)
    return avg.astype(np.float32), var.astype(np.float32)


# --------------------------------------------------------------- likelihoods
_SELECT = np.zeros((_K, _D), dtype=np.float32)
_SELECT[np.arange(_K), _DIM] = 1.0


def _measurement_cov(sigma: Tensor, corr: Tensor) -> Tensor:
    return sigma[:, :, None] * corr * sigma[:, None, :]


def batched_marginal_nll(
    mean: Tensor,
    scale_tril: Tensor,
    values: Tensor,
    mapped: Tensor,
    jacobian: Tensor,
    cov: Tensor,
    mask: Tensor,
) -> Tensor:
    """Harness reference: the same per-object marginal NLL, vectorized over the batch.

    Unobserved entries are replaced by a decoupled unit-variance coordinate with
    zero residual and zero Jacobian, which adds nothing to the quadratic form or
    log-determinant; the ``log(2 pi)`` constant counts observed entries only.
    """
    m = mask.to(mean.dtype)
    eye = torch.eye(mask.shape[1], dtype=mean.dtype, device=mean.device)
    pair = mask[:, :, None] & mask[:, None, :]
    r = torch.where(pair, cov, torch.zeros((), dtype=mean.dtype, device=mean.device))
    r = r + torch.diag_embed(1.0 - m)
    h = jacobian * m[:, :, None]
    hl = h @ scale_tril
    total = r + hl @ hl.transpose(-1, -2)
    resid = torch.where(mask, values - mapped, torch.zeros_like(mapped))
    chol = torch.linalg.cholesky(total)
    white = torch.linalg.solve_triangular(chol, resid[:, :, None], upper=False)[..., 0]
    logdet = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)
    return 0.5 * (m.sum(-1) * math.log(2 * math.pi) + white.square().sum(-1) + logdet)


def _masked_mean(per: Tensor, active: Tensor) -> Tensor:
    w = active.to(per.dtype)
    return (per * w).sum() / w.sum().clamp_min(1.0)


# ------------------------------------------------------------------ models
class _LatentNet(nn.Module):
    """MLP -> (mean [B,D], lower-triangular scale [B,D,D] with positive diagonal)."""

    rows: Tensor
    cols: Tensor

    def __init__(self, in_dim: int, dims: int) -> None:
        super().__init__()
        self.dims = dims
        rows, cols = torch.tril_indices(dims, dims)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.body = _MLP(in_dim, _HIDDEN, dims + len(rows))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.body(x)
        mean, raw = out[:, : self.dims], out[:, self.dims :]
        lower = out.new_zeros((len(x), self.dims, self.dims))
        lower[:, self.rows, self.cols] = raw
        diag = torch.diagonal(lower, dim1=-2, dim2=-1)
        lower = (
            lower - torch.diag_embed(diag) + torch.diag_embed(nn.functional.softplus(diag) + 1e-3)
        )
        return mean, lower


class _ScalarNet(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.body = _MLP(in_dim, _HIDDEN, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.body(x)
        return out[:, 0], nn.functional.softplus(out[:, 1]) + 1e-3


class _SurveyCalibration(nn.Module):
    """Survey bias model built from the torchregress calibration heads.

    Returns the packed ``h(theta)`` and Jacobian for every measurement slot.
    """

    src: Tensor
    dim: Tensor
    biased_slot: Tensor

    def __init__(self, relative: bool, common: bool, condition: bool) -> None:
        super().__init__()
        from torchregress.calibration import (
            CommonScaleSplineHead,
            ConditionBiasHead,
            RelativeSurveySplineHead,
        )

        knots = [torch.tensor(_KNOTS)] * _D
        weights = torch.zeros(_S, _D)
        for s, (_, dims, _, _) in enumerate(_SURVEYS):
            if s != _ANCHOR:
                weights[s, list(dims)] = 1.0
        self.relative = RelativeSurveySplineHead(knots, weights) if relative else None
        self.common = CommonScaleSplineHead(knots) if common else None
        self.condition = ConditionBiasHead(knots, torch.zeros(_S, 1)) if condition else None
        self.register_buffer("src", torch.as_tensor(_SRC))
        self.register_buffer("dim", torch.as_tensor(_DIM))
        self.register_buffer("biased_slot", torch.as_tensor(_SRC != _ANCHOR, dtype=torch.float32))

    def bias(self, theta: Tensor, quality: Tensor) -> tuple[Tensor, Tensor]:
        """Packed predicted bias ``[B,K]`` and its Jacobian ``[B,K,D]`` at ``theta``."""
        values = theta.new_zeros((len(theta), _K))
        jac = theta.new_zeros((len(theta), _K, _D))
        if self.relative is not None:
            v, j = self.relative(theta)
            values = values + v[:, self.src, self.dim]
            jac = jac + j[:, self.src, self.dim]
        if self.common is not None:
            v, j = self.common(theta)
            values = values + v[:, self.dim] * self.biased_slot
            jac = jac + j[:, self.dim] * self.biased_slot[:, None]
        if self.condition is not None:
            v, j = self.condition(theta, quality[:, :, None])
            values = values + v[:, self.src, self.dim]
            jac = jac + j[:, self.src, self.dim]
        return values, jac

    def regularization(self) -> Tensor:
        total = self.biased_slot.new_zeros(())
        for head in (self.relative, self.common, self.condition):
            if head is not None:
                total = total + head.regularization()
        return total


@dataclass
class _Batch:
    x: Tensor
    theta: Tensor
    values: Tensor
    sigma: Tensor
    mask: Tensor
    quality: Tensor | None = None
    ivw: Tensor | None = None
    ivw_var: Tensor | None = None


def _to_batch(data: dict[str, np.ndarray], idx: np.ndarray, dev: torch.device) -> _Batch:
    def t(key: str) -> Tensor | None:
        if key not in data:
            return None
        return torch.from_numpy(np.ascontiguousarray(data[key][idx])).to(dev)

    def req(key: str) -> Tensor:
        out = t(key)
        if out is None:
            raise KeyError(key)
        return out

    return _Batch(
        x=req("x"),
        theta=req("theta"),
        values=req("values"),
        sigma=req("sigma"),
        mask=req("mask"),
        quality=t("quality"),
        ivw=t("ivw"),
        ivw_var=t("ivw_var"),
    )


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@dataclass
class _Spec:
    """What a row trains: networks, its loss, and its held-out observation model."""

    net: nn.Module
    extra: list[nn.Module]
    loss: Callable[[_Batch], Tensor]
    obs_nll: Callable[[_Batch], tuple[Tensor, Tensor]] | None = None  # (per-object, active)
    calibration: _SurveyCalibration | None = None


def _gaussian_obs_nll(
    net: nn.Module, calib: _SurveyCalibration | None, corr: Tensor
) -> Callable[[_Batch], tuple[Tensor, Tensor]]:
    """Held-out NLL of observed survey values under the row's predicted latent (library)."""
    from torchregress.losses import MarginalizedMultiSourceGaussianNLLLoss

    loss = MarginalizedMultiSourceGaussianNLLLoss(reduction="none")

    def fn(b: _Batch) -> tuple[Tensor, Tensor]:
        mean, lower = net(b.x)
        obs = _packed(mean, b, corr, calib)
        return loss.per_star(mean, lower, obs), b.mask.any(-1)

    return fn


def _packed(mean: Tensor, b: _Batch, corr: Tensor, calib: _SurveyCalibration | None) -> Any:
    from torchregress.losses import PackedObservations

    select = torch.as_tensor(_SELECT, device=mean.device, dtype=mean.dtype)
    mapped = mean[:, torch.as_tensor(_DIM, device=mean.device)]
    jac = select.expand(len(mean), -1, -1)
    if calib is not None:
        if b.quality is None:
            raise ValueError("calibration rows need survey quality conditions")
        bias, bias_jac = calib.bias(mean, b.quality)
        mapped, jac = mapped + bias, jac + bias_jac
    return PackedObservations(
        values=b.values,
        mapped_mean=mapped,
        jacobian=jac,
        measurement_covariance=_measurement_cov(b.sigma, corr),
        mask=b.mask,
        block_ids=torch.as_tensor(_BLOCK, device=mean.device).expand(len(mean), -1),
    )


def _build(method: str, n_features: int, dev: torch.device) -> _Spec:
    from torchregress.losses import (
        MarginalizedMultiSourceGaussianNLLLoss,
        PackedObservations,
        RobustMultiSourceGaussianNLLLoss,
    )

    corr = torch.as_tensor(_CORR, dtype=torch.float32, device=dev)
    select = torch.as_tensor(_SELECT, device=dev)
    dim_idx = torch.as_tensor(_DIM, device=dev)
    eye_d = torch.eye(_D, device=dev)
    net = _LatentNet(n_features, _D).to(dev)
    library = MarginalizedMultiSourceGaussianNLLLoss(reduction="mean")
    obs_nll = _gaussian_obs_nll(net, None, corr)

    if method in ("MS_Marginal", "MS_NoCalib"):

        def loss(b: _Batch) -> Tensor:
            mean, lower = net(b.x)
            return library(mean, lower, _packed(mean, b, corr, None))

        return _Spec(net, [], loss, obs_nll)

    if method in ("MS_RelSpline", "MS_RelSpline_Common", "MS_FullCalib"):
        calib = _SurveyCalibration(
            relative=True,
            common=method != "MS_RelSpline",
            condition=method == "MS_FullCalib",
        ).to(dev)

        def loss(b: _Batch) -> Tensor:
            mean, lower = net(b.x)
            nll = library(mean, lower, _packed(mean, b, corr, calib))
            return nll + 1e-4 * calib.regularization()

        return _Spec(net, [calib], loss, _gaussian_obs_nll(net, calib, corr), calib)

    if method == "MS_Robust":
        robust = RobustMultiSourceGaussianNLLLoss(
            outlier_probability=_OUTLIER_P, outlier_scale=10.0, max_blocks=4, max_assignments=16
        )
        robust_none = RobustMultiSourceGaussianNLLLoss(
            reduction="none",
            outlier_probability=_OUTLIER_P,
            outlier_scale=10.0,
            max_blocks=4,
            max_assignments=16,
        )

        def loss(b: _Batch) -> Tensor:
            mean, lower = net(b.x)
            return robust(mean, lower, _packed(mean, b, corr, None))

        def robust_obs(b: _Batch) -> tuple[Tensor, Tensor]:
            mean, lower = net(b.x)
            return robust_none.per_star(mean, lower, _packed(mean, b, corr, None)), b.mask.any(-1)

        return _Spec(net, [], loss, robust_obs)

    if method == "MS_BatchedRef":

        def loss(b: _Batch) -> Tensor:
            mean, lower = net(b.x)
            per = batched_marginal_nll(
                mean,
                lower,
                b.values,
                mean[:, dim_idx],
                select.expand(len(mean), -1, -1),
                _measurement_cov(b.sigma, corr),
                b.mask,
            )
            return _masked_mean(per, b.mask.any(-1))

        return _Spec(net, [], loss, obs_nll)

    if method == "Joint_DiagR":

        def loss(b: _Batch) -> Tensor:
            mean, lower = net(b.x)
            obs = _packed(mean, b, corr, None)
            diag = PackedObservations(
                obs.values,
                obs.mapped_mean,
                obs.jacobian,
                torch.diag_embed(b.sigma.square()),
                obs.mask,
            )
            return library(mean, lower, diag)

        return _Spec(net, [], loss, obs_nll)

    if method == "ZeroFillMissing":
        base = torch.as_tensor(
            np.concatenate([np.asarray(b) for _, _, _, b in _SURVEYS]),
            dtype=torch.float32,
            device=dev,
        )

        def loss(b: _Batch) -> Tensor:
            mean, lower = net(b.x)
            filled = _Batch(
                x=b.x,
                theta=b.theta,
                values=torch.where(b.mask, b.values, torch.zeros_like(b.values)),
                sigma=torch.where(b.mask, b.sigma, base.expand_as(b.sigma)),
                mask=torch.ones_like(b.mask),
            )
            return library(mean, lower, _packed(mean, filled, corr, None))

        return _Spec(net, [], loss, obs_nll)

    if method == "PerSource_Product":

        def loss(b: _Batch) -> Tensor:
            mean, lower = net(b.x)
            latent_var = (lower @ lower.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)[:, dim_idx]
            var = torch.where(b.mask, b.sigma.square(), torch.ones_like(b.sigma)) + latent_var
            resid = torch.where(b.mask, b.values - mean[:, dim_idx], torch.zeros_like(b.values))
            per = 0.5 * (math.log(2 * math.pi) + var.log() + resid.square() / var)
            return _masked_mean((per * b.mask).sum(-1), b.mask.any(-1))

        return _Spec(net, [], loss, obs_nll)

    if method in ("IVW_Deconv", "IVW_GaussNLL"):
        deconv = method == "IVW_Deconv"

        def loss(b: _Batch) -> Tensor:
            if b.ivw is None or b.ivw_var is None:
                raise ValueError("IVW labels missing")
            mean, lower = net(b.x)
            has = torch.isfinite(b.ivw)
            var = torch.where(has, b.ivw_var, torch.ones_like(b.ivw_var))
            if not deconv:
                var = torch.full_like(var, 1e-6)
            per = batched_marginal_nll(
                mean,
                lower,
                b.ivw,
                mean,
                eye_d.expand(len(mean), -1, -1),
                torch.diag_embed(var),
                has,
            )
            return _masked_mean(per, has.any(-1))

        return _Spec(net, [], loss, obs_nll)

    if method == "Oracle_TrueLatent":

        def loss(b: _Batch) -> Tensor:
            mean, lower = net(b.x)
            per = batched_marginal_nll(
                mean,
                lower,
                b.theta,
                mean,
                eye_d.expand(len(mean), -1, -1),
                torch.diag_embed(torch.full_like(b.theta, 1e-6)),
                torch.ones_like(b.theta, dtype=torch.bool),
            )
            return per.mean()

        return _Spec(net, [], loss, obs_nll)

    raise ValueError(f"unknown multi-source method {method!r}")


# ---------------------------------------------------------------- training
def _epochs(cfg: HarnessConfig, big: bool) -> int:
    if cfg.smoke:
        return 1 if big else min(int(cfg.epochs), 2)
    # 1M objects: the library's per-object loop dominates (ms per object on a GPU);
    # two passes bound the run inside one job timeout.
    return min(int(cfg.epochs), 2) if big else int(cfg.epochs)


def _train(
    spec: _Spec,
    data: dict[str, np.ndarray],
    cfg: HarnessConfig,
    epochs: int,
    batch_size: int,
    dev: torch.device,
) -> int:
    modules = [spec.net, *spec.extra]
    params = [p for m in modules for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    n = len(data["x"])
    rng = np.random.default_rng(int(cfg.seed) + 1)
    steps = 0
    for _ in range(epochs):
        for m in modules:
            m.train()
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = _to_batch(data, perm[start : start + batch_size], dev)
            opt.zero_grad(set_to_none=True)
            loss = spec.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {steps}")
            loss.backward()
            opt.step()
            steps += 1
    for m in modules:
        m.eval()
    return steps


# -------------------------------------------------------------- evaluation
def _latent_metrics(mean: np.ndarray, cov: np.ndarray, theta: np.ndarray) -> dict[str, float]:
    """Latent recovery and calibration of ``N(mean, cov)`` against the true latent."""
    if mean.ndim == 1:
        mean, cov, theta = mean[:, None], cov[:, None, None], theta[:, None]
    d = mean.shape[1]
    resid = theta - mean
    sd = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    z = resid / sd
    chol = np.linalg.cholesky(cov)
    white = np.linalg.solve(chol, resid[:, :, None])[..., 0]
    maha = (white**2).sum(-1)
    logdet = 2 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(-1)
    nll = 0.5 * (d * math.log(2 * math.pi) + logdet + maha)
    from torchregress.metrics import pit_cvm

    pit = stats.norm.cdf(z)
    return {
        "RMSE": float(np.sqrt(np.mean(resid**2))),
        "MAE": float(np.mean(np.abs(resid))),
        "NLL": float(np.mean(nll)),
        "Coverage90": float(np.mean(np.abs(z) <= stats.norm.ppf(0.95))),
        "JointCoverage90": float(np.mean(maha <= stats.chi2.ppf(0.9, d))),
        "MeanStd": float(np.mean(sd)),
        "PITCVM": float(np.mean([pit_cvm(pit[:, j]) for j in range(d)])),
    }


@torch.no_grad()
def _predict_latent(
    net: nn.Module, x: np.ndarray, dev: torch.device, chunk: int = 65536
) -> tuple[np.ndarray, np.ndarray]:
    means, covs = [], []
    for start in range(0, len(x), chunk):
        mean, scale = net(torch.from_numpy(x[start : start + chunk]).to(dev))
        if scale.ndim == 1:  # scalar latent: std
            cov = scale.square()
        else:
            cov = scale @ scale.transpose(-1, -2)
        means.append(mean.cpu().double().numpy())
        covs.append(cov.cpu().double().numpy())
    return np.concatenate(means), np.concatenate(covs)


def _obs_nll(spec: _Spec, data: dict[str, np.ndarray], dev: torch.device) -> float:
    obs_fn = spec.obs_nll
    if obs_fn is None:
        raise ValueError("row has no observation model")
    n = min(len(data["x"]), _OBS_EVAL_MAX)
    total, count = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, n, 2048):
            b = _to_batch(data, np.arange(start, min(start + 2048, n)), dev)
            per, active = obs_fn(b)
            total += float((per * active).sum())
            count += float(active.sum())
    return total / max(count, 1.0)


def _bias_errors(
    calib: _SurveyCalibration | None, test: dict[str, np.ndarray], dev: torch.device
) -> dict[str, float]:
    """Total and gauge-invariant (relative) bias recovery on observed slots, at the true latent."""
    theta = test["theta"]
    truth = _true_bias(theta.astype(np.float64), test["quality"].astype(np.float64))
    if calib is None:
        pred = np.zeros_like(truth)
    else:
        with torch.no_grad():
            p, _ = calib.bias(
                torch.from_numpy(theta).to(dev), torch.from_numpy(test["quality"]).to(dev)
            )
        pred = p.cpu().double().numpy()
    mask = test["mask"] & (_SRC != _ANCHOR)[None, :]

    def relative(b: np.ndarray) -> np.ndarray:
        # Subtract, per object and latent dim, the mean over the biased surveys measuring it.
        out = b.copy()
        for d in range(_D):
            slots = np.flatnonzero((_DIM == d) & (_SRC != _ANCHOR))
            out[:, slots] -= b[:, slots].mean(axis=1, keepdims=True)
        return out

    err = (pred - truth)[mask]
    rel_err = (relative(pred) - relative(truth))[mask]
    return {
        "BiasRMSE": float(np.sqrt(np.mean(err**2))),
        "RelBiasRMSE": float(np.sqrt(np.mean(rel_err**2))),
        "TrueBiasRMS": float(np.sqrt(np.mean(truth[mask] ** 2))),
    }


# ----------------------------------------------------------- check rows
def _exactness(test: dict[str, np.ndarray], dev: torch.device) -> dict[str, Any]:
    from torchregress.losses import (
        GaussianLatentForwardObservationNLLLoss,
        MarginalizedMultiSourceGaussianNLLLoss,
        PackedObservations,
    )

    dtype = torch.float32 if dev.type == "mps" else torch.float64
    n = min(256, len(test["x"]))
    gen = torch.Generator().manual_seed(7)
    theta = torch.from_numpy(test["theta"][:n]).to(dtype)
    mean = (theta + 0.3 * torch.randn(n, _D, generator=gen, dtype=dtype)).to(dev)
    lower = torch.linalg.cholesky(torch.as_tensor(_THETA_COV, dtype=dtype) * 0.3).to(dev)
    lower = lower.expand(n, -1, -1).clone()
    jac = torch.as_tensor(_SELECT, dtype=dtype) + 0.1 * torch.randn(
        n, _K, _D, generator=gen, dtype=dtype
    )
    jac = jac.to(dev)
    offset = 0.05 * torch.randn(n, _K, generator=gen, dtype=dtype).to(dev)
    mask = torch.from_numpy(test["mask"][:n]).to(dev)
    values = torch.from_numpy(test["values"][:n]).to(dev, dtype)
    sigma = torch.from_numpy(test["sigma"][:n]).to(dev, dtype)
    cov = _measurement_cov(sigma, torch.as_tensor(_CORR, dtype=dtype, device=dev))
    mean.requires_grad_(True)
    mapped = torch.einsum("bkd,bd->bk", jac, mean) + offset
    obs = PackedObservations(values, mapped, jac, cov, mask)
    direct = MarginalizedMultiSourceGaussianNLLLoss("none", "direct").per_star(mean, lower, obs)
    info = MarginalizedMultiSourceGaussianNLLLoss("none", "information").per_star(mean, lower, obs)
    ref = batched_marginal_nll(mean, lower, values, mapped, jac, cov, mask)
    g_lib = torch.autograd.grad(info.sum(), mean, retain_graph=True)[0]
    g_ref = torch.autograd.grad(ref.sum(), mean, retain_graph=True)[0]
    closed = torch.zeros(n, dtype=dtype, device=dev)
    with torch.no_grad():
        for i in range(n):
            idx = mask[i].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            h = jac[i, idx]
            total = cov[i][idx][:, idx] + h @ lower[i] @ lower[i].T @ h.T
            law = torch.distributions.MultivariateNormal(mapped[i, idx], covariance_matrix=total)
            closed[i] = -law.log_prob(values[i, idx])
    diffs = {
        "MaxAbsDiff_ClosedForm": float((direct - closed).detach().abs().max()),
        "MaxAbsDiff_DirectVsInformation": float((direct - info).detach().abs().max()),
        "MaxAbsDiff_BatchedRef": float((info - ref).detach().abs().max()),
        "MaxAbsGradDiff_BatchedRef": float((g_lib - g_ref).abs().max()),
    }
    # Gate on differences relative to the NLL / gradient magnitude, so the same
    # criterion means the same thing in float64 (cpu, cuda) and float32 (mps).
    nll_scale = max(1.0, float(closed.abs().max()))
    grad_scale = max(1.0, float(g_ref.abs().max()))
    rel = max(
        diffs["MaxAbsDiff_ClosedForm"] / nll_scale,
        diffs["MaxAbsDiff_DirectVsInformation"] / nll_scale,
        diffs["MaxAbsDiff_BatchedRef"] / nll_scale,
        diffs["MaxAbsGradDiff_BatchedRef"] / grad_scale,
    )
    tol = 1e-5 if dtype == torch.float32 else 1e-11
    # Scalar forward quadrature through a linear map has an analytic Gaussian
    # marginal.  The loss is documented as a finite-order approximation, so it
    # is reported here but not part of the exactness gate: once in a broad-noise
    # regime (observation sd 1.0) and once at the survey's own reported errors.
    z_mean = mean[:, 0].detach()
    z_std = lower[:, 0, 0]
    obs0 = mask[:, 0]
    y = torch.where(obs0, values[:, 0], torch.zeros_like(values[:, 0]))
    regimes = {
        "Broad": torch.ones_like(y),
        "Survey": torch.where(obs0, sigma[:, 0], torch.ones_like(y)),
    }
    fwd: dict[str, float] = {}
    for label, err in regimes.items():
        analytic = -torch.distributions.Normal(
            1.3 * z_mean - 0.2, torch.sqrt(err.square() + (1.3 * z_std).square())
        ).log_prob(y)
        for q in (64, 256):
            approx = GaussianLatentForwardObservationNLLLoss(q, "none")(
                z_mean, z_std, y, err, lambda z: 1.3 * z - 0.2, obs0
            )
            fwd[f"FwdQuadLinear{label}_MaxAbsDiff_Q{q}"] = float(
                (approx - analytic)[obs0].abs().max()
            )
    return {
        **diffs,
        "ExactnessMaxRelDiff": rel,
        "ExactnessRelTol": tol,
        "ExactnessPass": bool(rel <= tol),
        **fwd,
        "NChecked": n,
        "NEmptyObjects": int((~mask.any(-1)).sum()),
        "Dtype": str(dtype).removeprefix("torch."),
    }


def _spline_parity(dev: torch.device) -> dict[str, Any]:
    from scipy.interpolate import BSpline
    from torchregress.calibration import bspline_basis_with_derivative

    dtype = torch.float32 if dev.type == "mps" else torch.float64
    knots = np.asarray(_KNOTS)
    grid = np.linspace(-4.0, 4.0, 801)
    x = torch.tensor(grid, dtype=dtype, device=dev, requires_grad=True)
    k = torch.tensor(knots, dtype=dtype, device=dev)
    basis, deriv, ood = bspline_basis_with_derivative(x, k, 3, boundary="linear")
    spline = BSpline(knots, np.eye(len(knots) - 4), 3, extrapolate=False)
    clipped = grid.clip(knots[0], knots[-1])
    expected = spline(clipped) + (grid - clipped)[:, None] * spline.derivative()(clipped)
    probe = torch.linspace(-1.0, 1.0, basis.shape[1], dtype=dtype, device=dev)
    grad = torch.autograd.grad((basis * probe).sum(), x)[0]
    const_b, const_d, _ = bspline_basis_with_derivative(x.detach(), k, 3, boundary="constant")
    outside = (grid < knots[0]) | (grid > knots[-1])
    b_np, d_np = basis.detach().cpu().numpy(), deriv.detach().cpu().numpy()
    const_expected = spline(clipped)
    return {
        "MaxAbsErr_Basis": float(np.abs(b_np - expected).max()),
        "MaxAbsErr_Derivative": float(np.abs(d_np - spline.derivative()(clipped)).max()),
        "MaxAbsErr_AutogradDerivative": float(
            (grad - (deriv * probe).sum(-1)).detach().abs().max().cpu()
        ),
        "MaxAbsErr_ConstantBoundary": float(np.abs(const_b.cpu().numpy() - const_expected).max()),
        "MaxAbs_ConstantBoundaryDerivativeOutside": float(
            np.abs(const_d.cpu().numpy()[outside]).max()
        ),
        "OODFlagMismatches": int((ood.cpu().numpy() != outside).sum()),
        "NGrid": len(grid),
        "Dtype": str(dtype).removeprefix("torch."),
    }


# ---------------------------------------------------------- forward rows
_DENSE_T = 20_001  # nodes on [-10, 10] latent standard deviations (spacing 1e-3)


def _dense_forward_nll(mu: Tensor, sd: Tensor, y: Tensor, err: Tensor) -> Tensor:
    """Reference ``-log E_z N(y; exp(z), err)`` by a dense trapezoid grid in z-score."""
    t = torch.linspace(-10.0, 10.0, _DENSE_T, dtype=mu.dtype, device=mu.device)
    log_dt = math.log(20.0 / (_DENSE_T - 1))
    log_w = torch.full_like(t, log_dt)
    log_w[0] = log_w[-1] = log_dt - math.log(2.0)
    log_prior = -0.5 * t.square() - 0.5 * math.log(2 * math.pi) + log_w
    out = []
    for start in range(0, len(mu), 256):
        sl = slice(start, start + 256)
        mapped = torch.exp(mu[sl, None] + sd[sl, None] * t)
        resid = (y[sl, None] - mapped) / err[sl, None]
        log_lik = -0.5 * resid.square() - err[sl, None].log() - 0.5 * math.log(2 * math.pi)
        out.append(-torch.logsumexp(log_lik + log_prior, dim=-1))
    return torch.cat(out)


def _scipy_check(args: list[Tensor], dense: np.ndarray, n_check: int) -> float:
    """Max |dense - scipy.integrate.quad| over the first ``n_check`` objects."""
    from scipy import integrate

    worst = 0.0
    for i in range(min(n_check, len(dense))):
        mu_i, sd_i, y_i, e_i = (float(a[i]) for a in args)

        def integrand(
            t: float, mu_i: float = mu_i, sd_i: float = sd_i, y_i: float = y_i, e_i: float = e_i
        ) -> float:
            return float(
                stats.norm.pdf(t) * stats.norm.pdf(y_i, loc=math.exp(mu_i + sd_i * t), scale=e_i)
            )

        # Point the adaptive rule at the likelihood peak, which can be far
        # narrower than the latent prior.
        peak = (math.log(y_i) - mu_i) / sd_i if y_i > 0 and sd_i > 0 else 0.0
        points = [peak] if -12 < peak < 12 else None
        value, _ = integrate.quad(
            integrand, -12, 12, epsabs=0.0, epsrel=1e-11, limit=800, points=points
        )
        worst = max(worst, abs(-math.log(value) - float(dense[i])))
    return worst


def _forward_row(
    method: str, train: dict[str, np.ndarray], test: dict[str, np.ndarray], cfg: HarnessConfig
) -> dict[str, Any]:
    from torchregress.losses import GaussianLatentForwardObservationNLLLoss

    dev = current_device()
    net = _ScalarNet(train["x"].shape[1]).to(dev)
    if method.startswith("Fwd_Quad_Q"):
        quad = GaussianLatentForwardObservationNLLLoss(int(method.removeprefix("Fwd_Quad_Q")))

        def loss(b: _Batch) -> Tensor:
            mu, sd = net(b.x)
            return quad(mu, sd, b.values, b.sigma, torch.exp, b.mask)

    elif method == "Fwd_Linearized":

        def loss(b: _Batch) -> Tensor:
            mu, sd = net(b.x)
            pred = torch.exp(mu)
            var = b.sigma.square() + (pred * sd).square()
            y = torch.where(b.mask, b.values, pred)
            per = 0.5 * (math.log(2 * math.pi) + var.log() + (y - pred).square() / var)
            return _masked_mean(per, b.mask)

    elif method == "Fwd_Oracle":

        def loss(b: _Batch) -> Tensor:
            mu, sd = net(b.x)
            return (sd.log() + 0.5 * ((b.theta - mu) / sd).square()).mean()

    else:
        raise ValueError(method)

    spec = _Spec(net, [], loss)
    epochs = _epochs(cfg, big=False)
    batch = 64 if cfg.smoke else 256
    _sync(dev)
    t0 = perf_counter()
    steps = _train(spec, train, cfg, epochs, batch, dev)
    _sync(dev)
    train_s = perf_counter() - t0
    t1 = perf_counter()
    mean, var = _predict_latent(net, test["x"], dev)
    metrics = _latent_metrics(mean, var, test["theta"].astype(np.float64))
    # Held-out observation NLL of every row's latent prediction under the true
    # nonlinear map: a dense-grid reference (validated against scipy adaptive
    # integration), and the error of the library's Gauss-Hermite orders on the
    # same predictions.
    m = test["mask"]
    work = torch.float64 if dev.type != "mps" else torch.float32
    args = [
        torch.as_tensor(a[m], dtype=work, device=dev)
        for a in (
            mean,
            np.sqrt(var),
            test["values"].astype(np.float64),
            test["sigma"].astype(np.float64),
        )
    ]
    with torch.no_grad():
        dense = _dense_forward_nll(*args).cpu().numpy()
        per = {
            q: GaussianLatentForwardObservationNLLLoss(q, "none")(*args, torch.exp).cpu().numpy()
            for q in (8, 16, 32, 64, 256)
        }
    metrics["ObsNLL"] = float(dense.mean())
    for q in (8, 16, 32, 64, 256):
        metrics[f"QuadErr_Q{q}"] = float(np.abs(per[q] - dense).mean())
        metrics[f"QuadMaxErr_Q{q}"] = float(np.abs(per[q] - dense).max())
    metrics["DenseRefVsScipyMaxAbs"] = _scipy_check(args, dense, n_check=16)
    _sync(dev)
    return {
        **metrics,
        "train_s": train_s,
        "eval_s": perf_counter() - t1,
        "Epochs": epochs,
        "Steps": steps,
        "NTrain": len(train["x"]),
        "NTargets": 1,
        "ThroughputObjPerS": len(train["x"]) * epochs / train_s,
    }


# -------------------------------------------------------------------- suite
class MultiSourceSuite(HarnessRunner):
    """Shared-latent multi-survey likelihoods vs naive baselines on known-truth catalogues."""

    @property
    def name(self) -> str:
        return "multi_source"

    @property
    def description(self) -> str:
        return (
            "Multi-survey shared-latent Gaussian NLL (masked, full cross-survey covariance) vs "
            "IVW / per-source / zero-fill baselines; robust block mixture; survey calibration "
            "spline heads; Gauss-Hermite forward NLL vs linearization; 1M-object scaling"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return list(_METHODS_BY_DATASET)

    def methods(self) -> list[str]:
        return list(dict.fromkeys(m for ms in _METHODS_BY_DATASET.values() for m in ms))

    def library_name(self, method: str) -> str:
        return _LIBRARY.get(method, "torchregress")

    def library_version(self, method: str) -> str:
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        return _NOTES[method]

    def fit(self, method: str, splits: Any, config: HarnessConfig) -> Any:
        raise NotImplementedError("multi_source uses a custom run()")

    def predict(self, method: str, model: Any, X_test: np.ndarray, config: HarnessConfig) -> Any:
        raise NotImplementedError("multi_source uses a custom run()")

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: Any = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate the catalogue (``splits`` is ignored: designs are suite-generated)."""
        del splits, scenario_name
        cfg = config or HarnessConfig()
        data = make_data(dataset_name, cfg)
        train, test = data["train"], data["test"]
        if dataset_name != "multisurvey_forward":
            for part in (train, test):
                part["ivw"], part["ivw_var"] = _ivw(part["values"], part["sigma"], part["mask"])
        rows: list[dict[str, Any]] = []
        for method in _METHODS_BY_DATASET[dataset_name]:
            set_harness_seed(cfg.seed)
            dev = current_device()
            base = {
                "Method": f"{self.name}/{method}",
                "Library": self.library_name(method),
                "LibraryVersion": self.library_version(method),
                "Dataset": dataset_name,
                "Suite": self.name,
                "SourceURL": "synthetic multi-survey design in suites/tabular/multi_source.py",
                "SyntheticFallback": False,
                "Device": str(dev),
            }
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            try:
                metrics = self._run_method(method, dataset_name, train, test, cfg)
                if dev.type == "cuda":
                    metrics["PeakGPUMemMB"] = torch.cuda.max_memory_allocated(dev) / 2**20
                rows.append({**base, **metrics, "Notes": self.method_notes(method)})
            except Exception as exc:
                rows.append(
                    {
                        **base,
                        "RMSE": None,
                        "MAE": None,
                        "NLL": None,
                        "Coverage90": None,
                        "train_s": None,
                        "eval_s": None,
                        "Notes": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
        return rows

    def _run_method(
        self,
        method: str,
        dataset: str,
        train: dict[str, np.ndarray],
        test: dict[str, np.ndarray],
        cfg: HarnessConfig,
    ) -> dict[str, Any]:
        dev = current_device()
        if method == "Exactness_LinearGaussian":
            t0 = perf_counter()
            out = _exactness(test, dev)
            return {**out, "train_s": 0.0, "eval_s": perf_counter() - t0}
        if method == "SplineBasis_ScipyParity":
            t0 = perf_counter()
            out = _spline_parity(dev)
            return {**out, "train_s": 0.0, "eval_s": perf_counter() - t0}
        if dataset == "multisurvey_forward":
            return _forward_row(method, train, test, cfg)
        big = dataset == "multisurvey_big"
        spec = _build(method, train["x"].shape[1], dev)
        epochs = _epochs(cfg, big)
        batch = 4096 if big and not cfg.smoke else (64 if cfg.smoke else 128)
        _sync(dev)
        t0 = perf_counter()
        steps = _train(spec, train, cfg, epochs, batch, dev)
        _sync(dev)
        train_s = perf_counter() - t0
        t1 = perf_counter()
        mean, cov = _predict_latent(spec.net, test["x"], dev)
        metrics: dict[str, Any] = _latent_metrics(mean, cov, test["theta"].astype(np.float64))
        _sync(dev)
        predict_s = perf_counter() - t1
        t2 = perf_counter()
        metrics["ObsNLL"] = _obs_nll(spec, test, dev)
        _sync(dev)
        metrics["ObsNLL_s"] = perf_counter() - t2
        if dataset == "multisurvey_biased":
            metrics.update(_bias_errors(spec.calibration, test, dev))
        return {
            **metrics,
            "train_s": train_s,
            "eval_s": predict_s,
            "Epochs": epochs,
            "Steps": steps,
            "BatchSize": batch,
            "NTrain": len(train["x"]),
            "NTest": len(test["x"]),
            "NTargets": _D,
            "ThroughputObjPerS": len(train["x"]) * epochs / train_s,
        }
