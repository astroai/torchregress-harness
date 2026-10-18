"""Astronomical errors-in-variables regression with a flow-based empirical-Bayes baseline.

This suite benchmarks linear errors-in-variables (EIV) regression under a
synthetic design that mimics the difficult parts of astronomical regression:
a non-Gaussian latent distribution, heteroscedastic measurement errors whose
size correlates with the latent value, an errors-in-``y`` baseline, and an
extrapolation region outside the training support.

Methods
-------
- ``OLS``: ordinary least squares on the noisy ``x_obs`` (attenuation baseline).
- ``WLS``: weighted least squares with known per-point ``sigma_y``; it accounts
  for errors in ``y`` only and still ignores errors in ``x``.  Its model-based
  intervals are therefore overconfident whenever the x-error scatter dominates
  (a feature of this baseline, not a numerical artefact).
- ``ForwardInversion``: torchregress ``ForwardInversionRegressor``.  It learns
  ``x_obs = g(y) + N(0, sigma_x^2)`` and recovers ``y`` by maximum-likelihood
  inversion at prediction time (Krutchkoff-style inverse regression).  Because
  the synthetic mean relation is linear, its decoder is a linear layer
  initialised from the inverse OLS fit; this isolates the EIV correction from
  neural-network capacity.
- ``FlowEB``: simplified ML empirical-Bayes EIV (paper-inspired).  A zuko 1-D
  normalizing flow models the intrinsic ``p(x_true)``; ``a, b`` and the flow
  are trained jointly by maximizing the *observed* likelihood, with the latent
  ``x_true`` integrated out by Gauss-Hermite quadrature.

Model and estimators
--------------------
For one object,

    x_obs = x_true + N(0, sigma_x^2),          sigma_x = sigma_x(x_true)
    y_obs = a * x_true + b + N(0, sigma_y^2),  sigma_y = sigma_y(x_true)

with ``x_true ~ p_intrinsic``.  ``FlowEB`` maximizes

    log p(x_obs, y_obs) = log int N(x_obs | x, sigma_x^2)
                                N(y_obs | a x + b, sigma_y^2)
                                p_flow(x) dx.

The integral is evaluated with ``K`` Gauss-Hermite nodes centred on each
object's observed ``x``:

    int N(x_obs | x, sigma_x^2) f(x) dx
        ~= (1 / sqrt(pi)) sum_j w_j f(x_obs + sqrt(2) sigma_x t_j).

For prediction, the same quadrature gives the posterior weights
``p(x | x_obs) ~ N(x_obs | x, sigma_x^2) p_flow(x)``; the conditional mean and
variance of ``y_obs`` are obtained from that discretized posterior, and
predictive samples are drawn by self-normalized importance sampling from
``N(x_obs, sigma_x^2)`` (the exact conditional posterior up to the flow prior).
Intervals are the 5-95% quantiles of those samples.  An exact mixture NLL
(``NLL_Mixture``) is also reported for ``FlowEB``.

Data design
-----------
The generator lives in this suite (no ``datasets.py`` edit):

* ``x_true`` is a two-component Gaussian mixture (bimodal, skewed,
  non-Gaussian), not a Gaussian;
* ``sigma_x`` grows smoothly with ``x_true`` (low-S/N objects have larger
  errors), and ``sigma_y`` shrinks with ``x_true``;
* the training and in-distribution (ID) test sets share the latent law;
* the extrapolation (OOD) test set draws ``x_true`` strictly above the
  training maximum, so it probes both distribution shift and extrapolation;
* targets are generated in native O(1) units, so the suite deliberately does
  not apply the harness's target standardisation; callers that pass their own
  ``splits`` must provide native-scale ``y`` (a ``_y_scaler`` is rejected);
* the true slope is ``1.25`` and the true intercept is ``0.40``.

Deviations from Jing & Li (2025, AJ, arXiv:2411.08747)
------------------------------------------------------
The paper's ``raddest`` implementation is not public, so this is a deliberately
simplified *ML* variant, not a reproduction:

* one-dimensional linear regression (the paper also derives the multidimensional
  and log-linear cases);
* no second normalizing flow for ``p(sigma_x | x_true)``: the per-object
  ``sigma_x`` is known and conditioned on, so the paper's learned error-intrinsic
  correlation is replaced by a known, correlated error model;
* no flow for ``p(sigma_y | y_true)``; ``sigma_y`` is known;
* uniform priors (pure ML), no MAP priors and no KS-test based objective;
* no 2-D KS goodness-of-fit diagnostic for the fitted flow;
* no target standardisation: the synthetic targets are already O(1);
* the flow is a zuko MAF with 1-D spline/affine transforms, not necessarily the
  paper's architecture.

These choices isolate the deconvolution/empirical-Bayes step while keeping the
comparison reproducible with the harness's fixed training budget.

Metrics
-------
Harness ``compute_all_metrics`` supplies RMSE/MAE/R2/NLL/CRPS/Coverage/Width
and PIT diagnostics.  In addition:

* ``Slope``: for ID rows, the method's fitted slope parameter (for
  ``ForwardInversion``, the slope of its fitted inverse relation); for OOD rows,
  the *effective* slope of ``y_pred`` regressed on the known ``x_true``.
* ``SlopeBias`` / ``SlopeRelBias`` / ``Attenuation``: relative to the true 1.25.
* ``Slope_Effective``: slope of ``y_pred`` on ``x_true`` (noisy-input
  predictions; always reported, and the OOD slope).
* ``Slope_Fit``: the global fitted slope, reported on OOD rows for context.

All OOD metrics are emitted with an ``_OOD`` suffix in the same result row.  The
suite reports one row per method, so ID and extrapolation behaviour are directly
comparable without changing the harness row schema.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torchregress.algorithms import ForwardInversionRegressor

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner, SkipMethod

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

DATASET_NAME = "synthetic_flow_eiv"
PAPER_URL = "https://arxiv.org/abs/2411.08747"
TRUE_SLOPE = 1.25
TRUE_INTERCEPT = 0.40

# Two-component Gaussian mixture for the intrinsic x distribution.
_MIXTURE_WEIGHT = 0.68
_MIXTURE_MEAN_LOW = -0.30
_MIXTURE_SD_LOW = 0.50
_MIXTURE_MEAN_HIGH = 1.05
_MIXTURE_SD_HIGH = 0.38

# Measurement-error laws (both correlated with the latent x).
_SIGMA_X_BASE = 0.30
_SIGMA_X_AMPLITUDE = 0.38
_SIGMA_X_SCALE = 0.50
_SIGMA_Y_BASE = 0.10
_SIGMA_Y_AMPLITUDE = 0.10
_SIGMA_Y_SCALE = 0.50

# Training / test sizes (smoke sizes keep CI fast).
_N_TRAIN = 1_500
_N_ID = 500
_N_OOD = 300
_N_TRAIN_SMOKE = 240
_N_ID_SMOKE = 80
_N_OOD_SMOKE = 60

# Flow / quadrature / sampling.
_GH_NODES = 32
_N_SAMPLES = 256
_N_SAMPLES_SMOKE = 128
_SQRT2 = math.sqrt(2.0)
_LOG_2PI = math.log(2.0 * math.pi)
_LOG_PI = math.log(math.pi)
_Z90 = 1.6448536269514722  # standard-normal 95th percentile
_FLOW_LOG_FLOOR = -1.0e4  # numerical guard for a broken flow tail
_SLOPE_EPS = 1.0e-12

_METHODS = ["OLS", "WLS", "ForwardInversion", "FlowEB"]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def _sample_intrinsic_x(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw the bimodal, non-Gaussian latent ``x_true``."""
    n_low = int(rng.binomial(n, _MIXTURE_WEIGHT))
    x = np.empty(n, dtype=np.float64)
    x[:n_low] = rng.normal(_MIXTURE_MEAN_LOW, _MIXTURE_SD_LOW, n_low)
    x[n_low:] = rng.normal(_MIXTURE_MEAN_HIGH, _MIXTURE_SD_HIGH, n - n_low)
    rng.shuffle(x)
    return x


def _measure(
    rng: np.random.Generator,
    x_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the correlated heteroscedastic error model and the linear relation."""
    sigma_x = _SIGMA_X_BASE + _SIGMA_X_AMPLITUDE * _sigmoid((x_true - 0.2) / _SIGMA_X_SCALE)
    sigma_y = _SIGMA_Y_BASE + _SIGMA_Y_AMPLITUDE * _sigmoid(-(x_true - 0.4) / _SIGMA_Y_SCALE)
    x_obs = x_true + sigma_x * rng.standard_normal(x_true.size)
    y_obs = TRUE_SLOPE * x_true + TRUE_INTERCEPT + sigma_y * rng.standard_normal(x_true.size)
    return x_obs, y_obs, sigma_x, sigma_y


def _generate_splits(config: HarnessConfig) -> dict[str, Any]:
    """Generate train / in-distribution / extrapolation splits.

    The OOD region is defined from the *training* latent values only:
    ``x_ood`` starts above the largest training ``x_true`` (and above its 99th
    percentile plus a margin), so no OOD object can be an interpolation point.
    """
    rng = np.random.default_rng(int(config.seed))
    n_train, n_id, n_ood = (
        (_N_TRAIN_SMOKE, _N_ID_SMOKE, _N_OOD_SMOKE) if config.smoke else (_N_TRAIN, _N_ID, _N_OOD)
    )

    x_train = _sample_intrinsic_x(rng, n_train)
    x_id = _sample_intrinsic_x(rng, n_id)

    q25, q75, q99 = np.quantile(x_train, [0.25, 0.75, 0.99])
    iqr = max(float(q75 - q25), 1.0e-3)
    ood_low = max(float(q99 + 0.35 * iqr), float(x_train.max() + 0.15 * iqr))
    ood_high = ood_low + 1.5 * iqr
    x_ood = rng.uniform(ood_low, ood_high, n_ood)

    x_train_obs, y_train, sx_train, sy_train = _measure(rng, x_train)
    x_id_obs, y_id, sx_id, sy_id = _measure(rng, x_id)
    x_ood_obs, y_ood, sx_ood, sy_ood = _measure(rng, x_ood)

    return {
        # Training rows.
        "X_train": x_train_obs[:, None].astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "X_train_true": x_train[:, None].astype(np.float32),
        "sigma_x_train": sx_train[:, None].astype(np.float32),
        "sigma_y_train": sy_train.astype(np.float32),
        # In-distribution test rows.
        "X_test": x_id_obs[:, None].astype(np.float32),
        "y_test": y_id.astype(np.float32),
        "X_test_true": x_id[:, None].astype(np.float32),
        "sigma_x_test": sx_id[:, None].astype(np.float32),
        "sigma_y_test": sy_id.astype(np.float32),
        # Extrapolation test rows.
        "X_test_ood": x_ood_obs[:, None].astype(np.float32),
        "y_test_ood": y_ood.astype(np.float32),
        "X_test_ood_true": x_ood[:, None].astype(np.float32),
        "sigma_x_test_ood": sx_ood[:, None].astype(np.float32),
        "sigma_y_test_ood": sy_ood.astype(np.float32),
        # Provenance / metadata.
        "true_slope": TRUE_SLOPE,
        "true_intercept": TRUE_INTERCEPT,
        "ood_x_min": float(x_ood.min()),
        "ood_x_max": float(x_ood.max()),
        "_source_url": (f"{PAPER_URL} (synthetic design defined in suites/tabular/flow_eiv.py)"),
    }


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def _as_sigma(
    value: Any,
    n: int,
    default: float,
    *,
    name: str,
) -> np.ndarray:
    """Coerce a scalar / per-feature / per-row sigma specification to ``(n,)``."""
    if value is None:
        return np.full(n, float(default), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, 0]
    arr = arr.reshape(-1)
    if arr.size == 1:
        arr = np.full(n, float(arr[0]), dtype=np.float64)
    if arr.size != n:
        raise ValueError(f"{name} has {arr.size} values for {n} rows")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.clip(arr, 1.0e-6, None)


def _median_sigma(value: Any, default: float) -> float:
    """Median of a sigma specification, used as a fallback at predict time."""
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float(default)
    med = float(np.median(arr))
    return med if np.isfinite(med) and med > 0 else float(default)


def _gauss_hermite(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights for ``int exp(-t^2) f(t) dt``."""
    nodes, weights = np.polynomial.hermite.hermgauss(int(n_nodes))
    return nodes.astype(np.float64), weights.astype(np.float64)


def _safe_flow_log_prob(flow: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Evaluate a zuko flow without letting a broken tail poison the loss."""
    log_prob = flow().log_prob(x)
    return torch.nan_to_num(
        log_prob,
        nan=_FLOW_LOG_FLOOR,
        posinf=-_FLOW_LOG_FLOOR,
        neginf=_FLOW_LOG_FLOOR,
    )


def _effective_slope(x_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Least-squares slope of predictions regressed on the known latent x."""
    x = np.asarray(x_true, dtype=np.float64).reshape(-1)
    y = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    var_x = float(np.var(x))
    if not np.isfinite(var_x) or var_x <= _SLOPE_EPS:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / var_x)


def _linear_design(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.column_stack([np.ones_like(x), x])


def _fit_linear(
    x_obs: np.ndarray,
    y: np.ndarray,
    sigma_y: np.ndarray,
    *,
    weighted: bool,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Closed-form OLS / WLS fit with a frequentist coefficient covariance."""
    design = _linear_design(x_obs)
    if weighted:
        w = 1.0 / np.maximum(sigma_y, 1.0e-6) ** 2
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        cov = np.linalg.inv(design.T @ (w[:, None] * design))
        return coef, cov, None

    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef
    dof = max(len(y) - 2, 1)
    residual_var = float(np.sum(residuals**2) / dof)
    cov = residual_var * np.linalg.inv(design.T @ design)
    return coef, cov, residual_var


def _predictive_interval(
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)
    return mean - _Z90 * std, mean + _Z90 * std


# ---------------------------------------------------------------------------
# Flow empirical-Bayes fit / predict
# ---------------------------------------------------------------------------


def _make_flow(config: HarnessConfig) -> nn.Module:
    """Build the intrinsic-density flow ``p_flow(x_true)``."""
    import zuko

    transforms = 2 if config.smoke else 3
    hidden = (32, 32) if config.smoke else (64, 64)
    return zuko.flows.MAF(features=1, transforms=transforms, hidden_features=hidden)


def _fit_flow_eb(
    x_obs: np.ndarray,
    y: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    config: HarnessConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Joint ML fit of the flow and the linear relation ``a, b``."""
    x_mean = float(np.mean(x_obs))
    x_std = float(np.std(x_obs))
    if not np.isfinite(x_std) or x_std < 1.0e-6:
        x_std = 1.0
    z_obs = (x_obs - x_mean) / x_std
    sigma_z = sigma_x / x_std

    # Initialise the linear relation from OLS on the standardised noisy x.
    b_init, a_init = np.linalg.lstsq(_linear_design(z_obs), y, rcond=None)[0]

    flow = _make_flow(config).to(device)
    a_param = nn.Parameter(torch.tensor(float(a_init), dtype=torch.float32, device=device))
    b_param = nn.Parameter(torch.tensor(float(b_init), dtype=torch.float32, device=device))
    params = [*flow.parameters(), a_param, b_param]
    optimizer = torch.optim.Adam(params, lr=float(config.lr))

    nodes_np, weights_np = _gauss_hermite(_GH_NODES)
    t_nodes = torch.as_tensor(nodes_np, dtype=torch.float32, device=device)
    log_weights = torch.as_tensor(np.log(weights_np), dtype=torch.float32, device=device)

    z_t = torch.as_tensor(z_obs, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    sx_t = torch.as_tensor(sigma_z, dtype=torch.float32, device=device)
    sy_t = torch.as_tensor(sigma_y, dtype=torch.float32, device=device).clamp_min(1.0e-6)
    n = int(z_t.shape[0])
    batch_size = max(1, min(int(config.batch_size), n))
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed) + 11)

    flow.train()
    final_loss = float("nan")
    for _ in range(max(1, int(config.epochs))):
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            z_b = z_t[idx]
            y_b = y_t[idx]
            sx_b = sx_t[idx]
            sy_b = sy_t[idx]

            # x_true nodes from the object's measurement Gaussian.
            x_q = z_b[:, None] + _SQRT2 * sx_b[:, None] * t_nodes[None, :]
            log_p_flow = _safe_flow_log_prob(flow, x_q.unsqueeze(-1))
            mean_y = a_param * x_q + b_param
            log_p_y = (
                -0.5 * ((y_b[:, None] - mean_y) / sy_b[:, None]) ** 2
                - torch.log(sy_b)[:, None]
                - 0.5 * _LOG_2PI
            )
            log_integral = torch.logsumexp(log_weights[None, :] + log_p_flow + log_p_y, dim=1)
            loss = -(log_integral - 0.5 * _LOG_PI).mean()

            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            grad_norm = clip_grad_norm_(params, 10.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
            final_loss = float(loss.detach())

    flow.eval()
    a_std = float(a_param.detach().cpu())
    b_std = float(b_param.detach().cpu())
    a_raw = a_std / x_std
    b_raw = b_std - a_raw * x_mean

    t_nodes = t_nodes.detach()
    log_weights = log_weights.detach()
    return {
        "type": "flow_eb",
        "flow": flow,
        "a_param": float(a_std),
        "b_param": float(b_std),
        "a_fit": float(a_raw),
        "b_fit": float(b_raw),
        "x_mean": x_mean,
        "x_std": x_std,
        "t_nodes": t_nodes,
        "log_weights": log_weights,
        "sigma_x_median": float(np.median(sigma_x)),
        "sigma_y_median": float(np.median(sigma_y)),
        "y_train_median": float(np.median(y)),
        "train_nll": final_loss,
        "device": device,
    }


def _flow_posterior(
    state: dict[str, Any],
    x_obs: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return posterior nodes, weights, and y mean/std for each test object."""
    device = state["device"]
    x_mean = state["x_mean"]
    x_std = state["x_std"]
    z = torch.as_tensor((x_obs - x_mean) / x_std, dtype=torch.float32, device=device)
    sigma_z = torch.as_tensor(sigma_x / x_std, dtype=torch.float32, device=device).clamp_min(1.0e-8)
    sy = torch.as_tensor(sigma_y, dtype=torch.float32, device=device).clamp_min(1.0e-6)
    t_nodes = state["t_nodes"]
    log_weights = state["log_weights"]
    flow = state["flow"]
    a = float(state["a_param"])
    b = float(state["b_param"])

    x_q = z[:, None] + _SQRT2 * sigma_z[:, None] * t_nodes[None, :]
    with torch.no_grad():
        log_p_flow = _safe_flow_log_prob(flow, x_q.unsqueeze(-1))
        log_post = log_weights[None, :] + log_p_flow
        post_w = torch.softmax(log_post, dim=1)
        z_mean = (post_w * x_q).sum(dim=1)
        z_var = (post_w * (x_q - z_mean[:, None]) ** 2).sum(dim=1)
        y_mean = a * z_mean + b
        y_var = a**2 * z_var + sy**2
        y_std = torch.sqrt(y_var.clamp_min(1.0e-12))
    return x_q, post_w, y_mean, y_std


def _flow_predict(
    state: dict[str, Any],
    x_obs: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    config: HarnessConfig,
) -> dict[str, np.ndarray]:
    """Posterior mean/std, predictive samples and 90% intervals for FlowEB."""
    device = state["device"]
    flow = state["flow"]
    x_mean = state["x_mean"]
    x_std = state["x_std"]
    a = float(state["a_param"])
    b = float(state["b_param"])
    z = torch.as_tensor((x_obs - x_mean) / x_std, dtype=torch.float32, device=device)
    sigma_z = torch.as_tensor(sigma_x / x_std, dtype=torch.float32, device=device).clamp_min(1.0e-8)
    sy = torch.as_tensor(sigma_y, dtype=torch.float32, device=device).clamp_min(1.0e-6)

    _, _, y_mean, y_std = _flow_posterior(state, x_obs, sigma_x, sigma_y)
    n = int(z.shape[0])
    n_samples = _N_SAMPLES_SMOKE if config.smoke else _N_SAMPLES
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed) + 7919)

    # Self-normalised importance sampling: proposal N(z_obs, sigma_z^2) is the
    # measurement kernel, so the importance weight is p_flow(z).
    noise = torch.randn(n, n_samples, generator=generator).to(device)
    z_prop = z[:, None] + sigma_z[:, None] * noise
    with torch.no_grad():
        log_p_prop = _safe_flow_log_prob(flow, z_prop.unsqueeze(-1)).squeeze(-1)
        weights = torch.softmax(log_p_prop, dim=1)
        cdf = torch.cumsum(weights, dim=1)
        uniforms = torch.rand(n, n_samples, generator=generator).to(device)
        indices = torch.searchsorted(cdf.contiguous(), uniforms.contiguous()).clamp_max(
            n_samples - 1
        )
        z_samples = z_prop.gather(1, indices)
        y_noise = torch.randn(n, n_samples, generator=generator).to(device)
        y_samples = a * z_samples + b + sy[:, None] * y_noise

    samples = y_samples.detach().cpu().numpy()
    lower = np.quantile(samples, 0.05, axis=1)
    upper = np.quantile(samples, 0.95, axis=1)
    # Guard a degenerate interval (constant samples) for the interval metrics.
    lower = lower - 1.0e-9
    upper = upper + 1.0e-9

    mean = y_mean.detach().cpu().numpy().reshape(-1)
    std = y_std.detach().cpu().numpy().reshape(-1)
    return {
        "y_pred": mean,
        "mean": mean,
        "std": std,
        "lower": lower,
        "upper": upper,
        "samples": samples,
    }


def _flow_mixture_nll(
    state: dict[str, Any],
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
) -> float:
    """Exact conditional NLL under the Gauss-Hermite posterior discretisation."""
    x_q, post_w, _, _ = _flow_posterior(state, x_obs, sigma_x, sigma_y)
    device = state["device"]
    y = torch.as_tensor(y_obs, dtype=torch.float32, device=device).reshape(-1, 1)
    sy = torch.as_tensor(sigma_y, dtype=torch.float32, device=device).clamp_min(1.0e-6)
    a = float(state["a_param"])
    b = float(state["b_param"])
    mean_y = a * x_q + b
    log_p_y = -0.5 * ((y - mean_y) / sy[:, None]) ** 2 - torch.log(sy)[:, None] - 0.5 * _LOG_2PI
    log_weights = torch.log(post_w.clamp_min(1.0e-30))
    log_numerator = torch.logsumexp(log_weights + log_p_y, dim=1)
    nll = -(log_numerator).mean()
    return float(nll.detach().cpu())


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class FlowEIVSuite(HarnessRunner):
    """Linear EIV regression with a simplified flow-based empirical-Bayes method."""

    @property
    def name(self) -> str:
        return "flow_eiv"

    @property
    def description(self) -> str:
        return (
            "Astronomical errors-in-variables regression: OLS/WLS/forward-inversion "
            "vs a paper-inspired flow empirical-Bayes ML estimator under a "
            "non-Gaussian latent law, correlated heteroscedastic errors and an "
            "extrapolation split"
        )

    @property
    def supported_datasets(self) -> list[str]:
        # The suite generates this design internally; the registry driver treats
        # an unknown dataset name as suite-generated (``split=None``).
        return [DATASET_NAME]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def library_name(self, method: str) -> str:
        if method in {"OLS", "WLS"}:
            return "numpy"
        if method == "FlowEB":
            return "harness (zuko)"
        return "torchregress"

    def library_version(self, method: str) -> str:
        if method in {"OLS", "WLS"}:
            return str(np.__version__)
        if method == "FlowEB":
            try:
                import zuko
            except ImportError:
                return "unavailable"
            return str(zuko.__version__)
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        notes = {
            "OLS": (
                "closed-form OLS on noisy x_obs; ignores x measurement noise (attenuation baseline)"
            ),
            "WLS": (
                "closed-form WLS with known per-point sigma_y; y-noise only, "
                "still ignores x measurement noise"
            ),
            "ForwardInversion": (
                "torchregress ForwardInversionRegressor: linear x_obs=g(y)+N(0,sigma_x^2) "
                "decoder initialised from inverse OLS, recovers y by ML inversion; "
                "sigma_y added to the predictive variance"
            ),
            "FlowEB": (
                "paper-inspired simplified ML-EB: zuko MAF p(x_true), Gauss-Hermite "
                "observed likelihood, joint Adam on flow+a,b; fixed known sigma_x/sigma_y, "
                "no p(sigma_x|x) flow, no KS variant; generative prior can shrink OOD "
                "extrapolation"
            ),
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # Fit / predict / evaluate
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, Any],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        device = current_device()
        x_obs = np.asarray(splits["X_train"], dtype=np.float64).reshape(-1)
        y = np.asarray(splits["y_train"], dtype=np.float64).reshape(-1)
        n = int(x_obs.size)
        if n < 3:
            raise ValueError(f"need at least 3 training rows, got {n}")
        if not (np.isfinite(x_obs).all() and np.isfinite(y).all()):
            raise ValueError("training data contains non-finite values")

        sigma_x = _as_sigma(
            splits.get("sigma_x_train", splits.get("sigma_x")),
            n,
            _median_sigma(splits.get("sigma_x"), 0.45),
            name="sigma_x_train",
        )
        sigma_y = _as_sigma(
            splits.get("sigma_y_train", splits.get("sigma_y")),
            n,
            _median_sigma(splits.get("sigma_y"), 0.15),
            name="sigma_y_train",
        )

        if method in {"OLS", "WLS"}:
            weighted = method == "WLS"
            coef, cov, residual_var = _fit_linear(x_obs, y, sigma_y, weighted=weighted)
            return {
                "type": "wls" if weighted else "ols",
                "coef": coef,
                "cov": cov,
                "residual_var": residual_var,
                "a_fit": float(coef[1]),
                "b_fit": float(coef[0]),
                "y_train_median": float(np.median(y)),
                "sigma_x_median": float(np.median(sigma_x)),
                "sigma_y_median": float(np.median(sigma_y)),
            }

        if method == "ForwardInversion":
            sigma_x_t = torch.as_tensor(sigma_x[:, None], dtype=torch.float32, device=device)
            # The synthetic mean relation is linear.  Use a linear decoder (the
            # correctly specified forward model), initialised from the inverse
            # OLS relation, so the comparison isolates the EIV correction from
            # neural-network capacity and stays stable in the 2-epoch smoke run.
            inv_coef = np.linalg.lstsq(np.column_stack([np.ones_like(y), y]), x_obs, rcond=None)[0]
            decoder = nn.Linear(1, 1)
            with torch.no_grad():
                decoder.weight.copy_(
                    torch.tensor([[inv_coef[1]]], dtype=torch.float32, device=device)
                )
                decoder.bias.copy_(torch.tensor([inv_coef[0]], dtype=torch.float32, device=device))
            model = ForwardInversionRegressor(
                decoder=decoder,
                sigma_x=sigma_x_t,
                learn_noise=True,
                epochs=max(1, int(config.epochs)),
                batch_size=min(max(1, int(config.batch_size)), n),
                lr=float(config.lr),
                seed=int(config.seed),
                device=str(device),
            )
            model.fit(
                torch.as_tensor(x_obs[:, None], dtype=torch.float32, device=device),
                torch.as_tensor(y[:, None], dtype=torch.float32, device=device),
            )
            a_fit = self._forward_inversion_slope(model, y, x_obs)
            return {
                "type": "forward_inversion",
                "model": model,
                "a_fit": a_fit,
                "b_fit": float("nan"),
                "y_train_median": float(np.median(y)),
                "sigma_x_median": float(np.median(sigma_x)),
                "sigma_y_median": float(np.median(sigma_y)),
            }

        if method == "FlowEB":
            return _fit_flow_eb(x_obs, y, sigma_x, sigma_y, config, device)

        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _forward_inversion_slope(
        model: ForwardInversionRegressor,
        y_train: np.ndarray,
        x_obs_train: np.ndarray,
    ) -> float:
        """Slope of the fitted inverse relation ``y -> x_obs`` over the y range."""
        decoder = getattr(model, "decoder_", None)
        if decoder is not None:
            try:
                device = next(decoder.parameters()).device
                y_grid = torch.linspace(
                    float(np.min(y_train)),
                    float(np.max(y_train)),
                    256,
                    dtype=torch.float32,
                    device=device,
                ).reshape(-1, 1)
                with torch.no_grad():
                    x_grid = decoder(y_grid).detach().cpu().numpy().reshape(-1)
                y_grid_np = y_grid.detach().cpu().numpy().reshape(-1)
                if np.var(x_grid) > _SLOPE_EPS:
                    return float(np.polyfit(x_grid, y_grid_np, 1)[0])
            except Exception:
                pass
        # Fallback diagnostic: OLS slope on the training data (still attenuated,
        # but finite and explicitly a fallback).
        coef = np.linalg.lstsq(_linear_design(x_obs_train), y_train, rcond=None)[0]
        return float(coef[1])

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
        *,
        sigma_x_test: Any = None,
        sigma_y_test: Any = None,
    ) -> dict[str, np.ndarray]:
        x = np.asarray(X_test, dtype=np.float64).reshape(-1)
        n = int(x.size)
        sigma_x = _as_sigma(
            sigma_x_test if sigma_x_test is not None else model_state.get("_sigma_x_test"),
            n,
            float(model_state.get("sigma_x_median", 0.45)),
            name="sigma_x_test",
        )
        sigma_y = _as_sigma(
            sigma_y_test if sigma_y_test is not None else model_state.get("_sigma_y_test"),
            n,
            float(model_state.get("sigma_y_median", 0.15)),
            name="sigma_y_test",
        )

        kind = model_state["type"]
        if kind == "flow_eb":
            return _flow_predict(model_state, x, sigma_x, sigma_y, config)

        if kind in {"ols", "wls"}:
            design = _linear_design(x)
            mean = design @ np.asarray(model_state["coef"], dtype=np.float64)
            cov = np.asarray(model_state["cov"], dtype=np.float64)
            var_fit = np.einsum("ij,jk,ik->i", design, cov, design)
            if kind == "wls":
                var_pred = sigma_y**2 + var_fit
            else:
                var_pred = float(model_state["residual_var"]) + var_fit
            std = np.sqrt(np.maximum(var_pred, 1.0e-12))
            lower, upper = _predictive_interval(mean, std)
            return {"y_pred": mean, "mean": mean, "std": std, "lower": lower, "upper": upper}

        if kind == "forward_inversion":
            model = model_state["model"]
            device = current_device()
            x_t = torch.as_tensor(x[:, None], dtype=torch.float32, device=device)
            sx_t = torch.as_tensor(sigma_x[:, None], dtype=torch.float32, device=device)
            mean_t, std_t = model.predict(
                x_t,
                sigma_x=sx_t,
                method="gauss_newton",
                max_iter=40,
                return_std=True,
            )
            mean = np.asarray(mean_t.detach().cpu(), dtype=np.float64).reshape(-1)
            inversion_std = np.asarray(std_t.detach().cpu(), dtype=np.float64).reshape(-1)
            # The forward model treats y as exact labels; add the known target
            # measurement noise to obtain a predictive distribution for y_obs.
            inversion_std = np.nan_to_num(
                inversion_std,
                nan=float(np.median(sigma_y)),
                posinf=10.0 * float(np.median(sigma_y)),
                neginf=0.0,
            )
            std = np.sqrt(np.maximum(inversion_std**2 + sigma_y**2, 1.0e-12))
            mean = np.nan_to_num(mean, nan=float(model_state.get("y_train_median", 0.0)))
            lower, upper = _predictive_interval(mean, std)
            return {"y_pred": mean, "mean": mean, "std": std, "lower": lower, "upper": upper}

        raise ValueError(f"Unknown model type: {kind}")

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        return super().evaluate(method, predictions, y_test, config)

    # ------------------------------------------------------------------
    # Run (ID + extrapolation metrics in one row per method)
    # ------------------------------------------------------------------

    def _evaluate_region(
        self,
        method: str,
        model_state: dict[str, Any],
        splits: dict[str, Any],
        config: HarnessConfig,
        *,
        region: str,
    ) -> dict[str, float]:
        suffix = "" if region == "ID" else "_ood"
        x_key = "X_test" if region == "ID" else "X_test_ood"
        y_key = "y_test" if region == "ID" else "y_test_ood"
        true_key = "X_test_true" if region == "ID" else "X_test_ood_true"
        sx_key = "sigma_x_test" if region == "ID" else "sigma_x_test_ood"
        sy_key = "sigma_y_test" if region == "ID" else "sigma_y_test_ood"

        if x_key not in splits or y_key not in splits:
            return {}

        x_test = np.asarray(splits[x_key])
        y_test = np.asarray(splits[y_key]).reshape(-1)
        model_state["_sigma_x_test"] = splits.get(sx_key)
        model_state["_sigma_y_test"] = splits.get(sy_key)

        predictions = self.predict(method, model_state, x_test, config)
        metrics = self.evaluate(method, predictions, y_test, config)

        if model_state["type"] == "flow_eb":
            sigma_x = _as_sigma(
                splits.get(sx_key),
                len(x_test),
                float(model_state.get("sigma_x_median", 0.45)),
                name=f"sigma_x{suffix}",
            )
            sigma_y = _as_sigma(
                splits.get(sy_key),
                len(y_test),
                float(model_state.get("sigma_y_median", 0.15)),
                name=f"sigma_y{suffix}",
            )
            metrics["NLL_Mixture"] = _flow_mixture_nll(
                model_state,
                np.asarray(x_test, dtype=np.float64).reshape(-1),
                y_test,
                sigma_x,
                sigma_y,
            )

        x_true = splits.get(true_key)
        slope_eff = (
            _effective_slope(np.asarray(x_true), predictions["y_pred"])
            if x_true is not None
            else float("nan")
        )
        a_fit = model_state.get("a_fit")
        if region == "ID" and a_fit is not None and np.isfinite(a_fit):
            slope = float(a_fit)
        else:
            slope = slope_eff

        metrics["Slope"] = slope
        metrics["SlopeBias"] = slope - TRUE_SLOPE
        metrics["SlopeRelBias"] = (slope - TRUE_SLOPE) / TRUE_SLOPE
        metrics["Attenuation"] = slope / TRUE_SLOPE
        metrics["Slope_Effective"] = slope_eff
        if region != "ID":
            metrics["Slope_Fit"] = float(a_fit) if a_fit is not None else float("nan")
        return metrics

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, Any] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        cfg = config or HarnessConfig()
        if scenario_name is not None:
            raise ValueError(f"Suite '{self.name}' has no canonical scenarios")

        dataset = dataset_name or DATASET_NAME
        if splits is None:
            splits = _generate_splits(cfg)
        if "X_train" not in splits or "y_train" not in splits:
            raise ValueError("splits must contain X_train and y_train")
        if "_y_scaler" in splits:
            raise ValueError(
                "flow_eiv expects native-scale targets; do not pass target-standardised splits"
            )

        rows: list[dict[str, Any]] = []
        for method in self.methods():
            set_harness_seed(cfg.seed)
            try:
                t0 = perf_counter()
                model_state = self.fit(method, splits, cfg)
                train_s = perf_counter() - t0

                t0 = perf_counter()
                id_metrics = self._evaluate_region(method, model_state, splits, cfg, region="ID")
                ood_metrics = self._evaluate_region(method, model_state, splits, cfg, region="OOD")
                eval_s = perf_counter() - t0

                notes = self.method_notes(method)
                notes = f"{notes}; _OOD suffix = extrapolation split"
                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset,
                    "Suite": self.name,
                    **id_metrics,
                    **{f"{key}_OOD": value for key, value in ood_metrics.items()},
                    "N_Train": int(np.asarray(splits["X_train"]).shape[0]),
                    "N_ID": int(np.asarray(splits.get("X_test", [])).shape[0]),
                    "N_OOD": int(np.asarray(splits.get("X_test_ood", [])).shape[0]),
                    "TrueSlope": TRUE_SLOPE,
                    "TrueIntercept": TRUE_INTERCEPT,
                    "train_s": train_s,
                    "eval_s": eval_s,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": notes,
                }
            except (ImportError, SkipMethod) as exc:
                reason = str(exc) if isinstance(exc, SkipMethod) else f"{type(exc).__name__}: {exc}"
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset,
                    "Suite": self.name,
                    "RMSE": None,
                    "MAE": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": f"skipped: {reason}",
                }
            except Exception as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset,
                    "Suite": self.name,
                    "RMSE": None,
                    "MAE": None,
                    "train_s": None,
                    "eval_s": None,
                    "SourceURL": splits.get("_source_url", ""),
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }
            rows.append(row)
        return rows


__all__ = ["FlowEIVSuite", "DATASET_NAME", "TRUE_INTERCEPT", "TRUE_SLOPE"]
