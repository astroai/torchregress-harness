"""Harness-side protocol augmentations for torchregress distribution losses.

PROPOSAL #15 defines a minimal ``DistributionLoss`` protocol::

    Required: log_prob(), sample(), predict_interval()
    Optional: quantile(), cdf(), posterior()

The base ``torchregress`` library classes have inconsistent coverage of
this protocol (see ``PROPOSALS.md`` §15):

* :class:`torchregress.losses.MixtureDensityLoss` — implements
  ``predict_interval()``, ``sample()``, ``predict_mean_std()``, but
  only has **private** ``_log_prob_diagonal`` / ``_log_prob_full``.
  The public ``log_prob()`` is missing.
* :class:`torchregress.losses.NormalizingFlowLoss` — implements
  ``log_prob()``, ``sample()``, ``quantile()``, ``cdf()``, **but
  ``predict_interval()`` is missing** — must be reconstructed from
  ``quantile()``.

This module **monkey-patches the missing public protocol methods
onto the installed library classes at import time** rather than
modifying the library source directly.  The rationale is:

* **Reproducibility**: ``uv pip install -e .`` re-installs do not erase
  the implementation; the monkey-patch re-applies on each Python
  process via the harness import.
* **Cleanliness**: no vendoring, no fork of ``torchregress``.
* **Idempotency**: importing this module multiple times is safe; the
  patch installs a sentinel attribute and skips reapplication.

The harness imports this module from ``src/harness/__init__.py`` so
``import harness`` triggers the protocol augmentation transparently.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

# Sentinel attribute installed on each augmented class to ensure the
# patch is applied exactly once per Python process.  Checking this
# attribute is more robust than scanning ``dir(cls)`` for ``log_prob``.
_PATCH_SENTINEL = "_harness_protocol_v1"

_LOGGER = logging.getLogger(__name__)

# Torchregress loss classes — resolved lazily because the module must
# import cleanly even when torchregress is not installed (harness-side
# unit tests).  The first call resolves and caches the references.
_MDN_CLS: type | None = None
_NF_CLS: type | None = None


def _resolve_loss_classes() -> tuple[type | None, type | None]:
    """Resolve ``MixtureDensityLoss`` and ``NormalizingFlowLoss`` (cached)."""
    global _MDN_CLS, _NF_CLS
    if _MDN_CLS is not None or _NF_CLS is not None:
        return _MDN_CLS, _NF_CLS
    try:
        import torchregress.losses as _losses_mod
    except ImportError:
        return None, None
    _MDN_CLS = getattr(_losses_mod, "MixtureDensityLoss", None)
    _NF_CLS = getattr(_losses_mod, "NormalizingFlowLoss", None)
    return _MDN_CLS, _NF_CLS


def _mixturedensity_log_prob(self, y_pred: Tensor, target: Tensor) -> Tensor:
    """Public ``log_prob`` for ``MixtureDensityLoss``.

    Returns per-sample log probability of ``target`` under the mixture
    implied by ``y_pred``.  Shape: ``[batch_size]``.

    Internal implementation: delegates to the private per-covariance
    helpers (``_log_prob_diagonal`` or ``_log_prob_full``), then mixes
    via ``logsumexp`` weighted by the mixture weights — identical to
    the ``_calculate_nll`` NLL calculation, but returned with a
    sign-flip and **before** the ``reduction`` parameter is applied.
    """
    # Extract mixture weights, component means, and (stds | Cholesky Ls).
    log_weights, means, stds_or_L = self._extract_distribution_parameters(y_pred)

    # Per-component log probabilities
    if self.covariance_type == "diagonal":
        log_probs_comp = self._log_prob_diagonal(target, means, stds_or_L)
    else:  # 'full'
        log_probs_comp = self._log_prob_full(target, means, stds_or_L)

    # log P(y|X) = logsumexp_k [ log w_k + log p_k(y|X) ]
    # _extract_distribution_parameters now returns log_weights (log_softmax) directly
    return torch.logsumexp(log_weights + log_probs_comp, dim=-1)

def _normalizingflow_predict_interval(
    self,
    y_pred: Tensor,
    confidence: float = 0.95,
    n_samples: int = 10000,
) -> tuple[Tensor, Tensor]:
    """Public ``predict_interval`` for ``NormalizingFlowLoss``.

    Thin wrapper around ``self.quantile()`` with the standard
    [α/2, 1-α/2] two-sided interval semantics.  ``confidence`` is the
    target coverage (1 − α).

    Parameters
    ----------
    y_pred : Tensor
        Context vector from model.  Shape: ``[batch_size, context_dim]``.
    confidence : float
        Target coverage level (default 0.95 → 95% interval).
    n_samples : int
        Number of MC samples passed to ``quantile()``.

    Returns
    -------
    (lower, upper) : tuple[Tensor, Tensor]
        Per-dimension interval bounds.  Each shape:
        ``[batch_size, n_features]``.

    Raises
    ------
    ValueError
        If ``confidence`` is not strictly between 0 and 1, or if the
        flow is multi-target (current ``quantile()`` implementation
        is scalar-only).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")

    # Multi-target rejection: torchregress.quantile() raises
    # NotImplementedError for n_features > 1 — this wrapper normalises to
    # ValueError so callers can use a single exception type.
    flow = getattr(self, "flow", None)
    flow_n_features = getattr(flow, "n_features", None) if flow is not None else None
    if flow_n_features is not None and int(flow_n_features) > 1:
        raise ValueError(
            f"NormalizingFlowLoss.predict_interval only supports scalar "
            f"targets (flow n_features={flow_n_features}); use sample-based "
            "PIT via compute_pit_from_loss_fn for D > 1."
        )

    alpha = 1.0 - float(confidence)
    levels = [alpha / 2.0, 1.0 - alpha / 2.0]
    try:
        quantiles = self.quantile(y_pred, levels, n_samples=n_samples)
    except NotImplementedError as exc:
        # Normalise to ValueError so callers can rely on a single
        # exception type from the harness protocol surface.
        raise ValueError(str(exc)) from exc

    # quantile() returns shape [batch_size, len(levels)].
    lower = quantiles[:, 0]
    upper = quantiles[:, 1]
    if lower.dim() == 1:
        lower = lower.unsqueeze(-1)
        upper = upper.unsqueeze(-1)
    return lower, upper


def compute_pit_from_loss_fn(
    loss_fn: Any,
    *,
    context: Tensor,
    target: Tensor,
    n_samples: int = 512,
    seed: int | None = None,
) -> np.ndarray:
    """Compute PIT values from a fitted ``DistributionLoss`` and target contexts.

    Implements the PROPOSAL #15 dispatch the prompt required: ``compute_pit_values``
    in :mod:`harness.metrics` is extended with an optional ``loss_fn + context +
    target`` kwarg path that delegates here.  This helper is the torch-aware
    engine that draws MC samples from the fitted distribution loss and computes
    the empirical CDF at each ``target`` value.

    Implemented strategies
    ----------------------
    * **NormalizingFlowLoss** — draws ``n_samples`` predictive samples per test
      point via ``loss_fn.sample(context, n_samples)``, computes the empirical
      CDF at ``target`` as ``mean(samples <= target)``.  The patched
      ``log_prob()`` is available on the class for callers who want a closed-
      form diagnostic; this helper deliberately uses MC because the
      closed-form PIT for flows requires CDF inversion that zuko exposes via
      ``cdf`` but for high-dim targets is more fragile than MC.

    * **MixtureDensityLoss** — closed-form ``log_prob`` is available via the
      patched public method, but CDFs require integrating that density.  We
      use MC sampling instead for parity with the flow path and to keep
      numerical cost bounded in ``n_features``.

    Note on multi-dim targets (joint marginal PIT)
    ---------------------------------------------
    For ``n_features > 1``, returns the **per-feature PIT averaged across
    output dimensions** — a *joint marginal PIT* heuristic — rather than the
    joint CDF integral.  This is documented behaviour; downstream code that
    needs a *strict* joint PIT (Bernardo & Smith 1994) should integrate the
    joint density via the patched ``loss_fn.log_prob()`` directly.

    Parameters
    ----------
    loss_fn : ``MixtureDensityLoss | NormalizingFlowLoss``
        Fitted torchregress distribution loss with the harness protocol
        patches applied (``install_loss_protocol_augments()`` ensures
        ``log_prob`` / ``predict_interval`` exist).
    context : Tensor
        [batch, context_dim] context for conditional flows.
    target : Tensor
        [batch, n_features] true values to compute PIT at.
    n_samples : int
        Number of MC samples per test point.
    seed : int | None
        Optional RNG seed for ``torch.manual_seed`` reproducibility.

    Returns
    -------
    np.ndarray
        ``[batch]`` PIT values; well-calibrated predictive distributions
        produce values uniformly distributed on ``[0, 1]``.
    """
    if seed is not None:
        torch.manual_seed(int(seed))

    _NormalizingFlowLoss, _MixtureDensityLoss = _resolve_loss_classes()

    if _NormalizingFlowLoss is None and _MixtureDensityLoss is None:
        raise TypeError(
            "compute_pit_from_loss_fn requires torchregress to be installed "
            "but it could not be imported."
        )

    batch_size = int(context.shape[0])

    if _NormalizingFlowLoss is not None and isinstance(loss_fn, _NormalizingFlowLoss):
        # NF.sample() returns [batch, n_samples, n_features].  Pin the
        # assumption defensively — a future library layout change would
        # silently mis-broadcast (samples <= target) and corrupt PIT.
        samples = loss_fn.sample(context, n_samples=n_samples)
        if samples.shape[0] != batch_size:
            raise AssertionError(
                f"NormalizingFlowLoss.sample() returned batch dim "
                f"{samples.shape[0]} (expected {batch_size}) — layout changed?"
            )
        tgt = target.unsqueeze(-1) if target.dim() == 1 else target
        per_feature = (samples <= tgt.unsqueeze(1)).to(samples.dtype).mean(dim=1)
        pit_tensor = per_feature.mean(dim=-1)
        return pit_tensor.detach().cpu().numpy()

    if _MixtureDensityLoss is not None and isinstance(loss_fn, _MixtureDensityLoss):
        # MDN.sample() returns [batch, n_samples, n_features] — same
        # layout as NormalizingFlowLoss.sample().  This was empirically
        # validated against torchregress v0.14.1 (see test failure
        # log: assertion fires on [n_samples, batch, n_features] layout).
        samples = loss_fn.sample(context, n_samples=n_samples)
        if samples.shape[0] != batch_size:
            raise AssertionError(
                f"MixtureDensityLoss.sample() returned batch dim at "
                f"index 0 = {samples.shape[0]} (expected {batch_size}) — "
                "layout changed?"
            )
        tgt = target.unsqueeze(-1) if target.dim() == 1 else target
        per_feature = (samples <= tgt.unsqueeze(1)).to(samples.dtype).mean(dim=1)
        pit_tensor = per_feature.mean(dim=-1)
        return pit_tensor.detach().cpu().numpy()

    raise TypeError(
        f"compute_pit_from_loss_fn: unsupported loss type "
        f"{type(loss_fn).__name__!r} (expected NormalizingFlowLoss "
        "or MixtureDensityLoss)."
    )


__all__ = [
    "compute_pit_from_loss_fn",
    "install_loss_protocol_augments",
    "_PATCH_SENTINEL",
]


def install_loss_protocol_augments() -> None:
    """Apply PROPOSAL #15 protocol patches to installed ``torchregress`` losses.

    Idempotent: inspects a sentinel attribute on each class and skips
    the patch if already installed.  Safe to call multiple times.

    Imports ``torchregress.losses`` lazily so this module can be
    imported even if the user hasn't installed torchregress yet
    (e.g. during isolated unit testing of the harness utilities).
    """
    MixtureDensityLoss, NormalizingFlowLoss = _resolve_loss_classes()

    if MixtureDensityLoss is None and NormalizingFlowLoss is None:
        _LOGGER.debug("torchregress not installed; protocol patch skipped")
        return

    if MixtureDensityLoss is not None:
        if not getattr(MixtureDensityLoss, _PATCH_SENTINEL, False):
            MixtureDensityLoss.log_prob = _mixturedensity_log_prob  # ty: ignore[unresolved-attribute]  # protocol patch adds log_prob to the external class dynamically
            setattr(MixtureDensityLoss, _PATCH_SENTINEL, True)
            _LOGGER.debug("[harness.loss_augments] installed log_prob on MixtureDensityLoss")

    if NormalizingFlowLoss is not None:
        if not getattr(NormalizingFlowLoss, _PATCH_SENTINEL, False):
            NormalizingFlowLoss.predict_interval = (  # ty: ignore[unresolved-attribute]  # protocol patch adds predict_interval to the external class dynamically
                _normalizingflow_predict_interval
            )
            setattr(NormalizingFlowLoss, _PATCH_SENTINEL, True)
            _LOGGER.debug(
                "[harness.loss_augments] installed predict_interval on NormalizingFlowLoss"
            )


# Apply at import time — this runs whenever ``harness.loss_augments`` is
# imported (e.g. via ``src/harness/__init__.py``).
install_loss_protocol_augments()
