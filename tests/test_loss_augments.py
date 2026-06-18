"""Tests for PROPOSAL #15 protocol augmentations.

Verifies that the harness :mod:`harness.loss_augments` module
completes the ``DistributionLoss`` protocol on torchregress loss
classes:

* ``MixtureDensityLoss.log_prob()`` — previously private
  (``_log_prob_diagonal`` / ``_log_prob_full``) is now public.
* ``NormalizingFlowLoss.predict_interval()`` — was missing;
  now wraps the existing ``self.quantile()`` method.

The patches must be **idempotent** (re-importing must not corrupt
the class), **shape-correct**, and **math-equivalent** to the
existing forward pass.

PROPOSAL #15 dispatch contract
------------------------------
``harness.metrics.compute_pit_values`` must accept an optional
``loss_fn=`` path that draws MC samples directly from a fitted
distribution loss — callers should NOT have to flatten the loss
output into a dict before calling it.
"""

from __future__ import annotations

import pytest
import torch

import harness  # noqa: F401  — triggers install_loss_protocol_augments()
from harness.loss_augments import (
    _PATCH_SENTINEL,
    install_loss_protocol_augments,
)


def _build_test_flow(*, n_features: int, context_dim: int) -> torch.nn.Module:
    """Build a tiny conditional NSF flow for unit tests.

    Uses ``[16, 16]`` hidden features and 2 transforms so the test
    runs in milliseconds.  Production code uses ``[64, 64]`` for
    better calibration.
    """
    from torchregress.losses.nflows import create_flow_model

    return create_flow_model(
        n_features=n_features,
        context_dim=context_dim,
        flow_type="nsf",
        n_transforms=2,
        hidden_features=[16, 16],
    )


# ===========================================================================
# Idempotency
# ===========================================================================


def test_patch_sentinel_installed() -> None:
    """After ``import harness``, both classes must carry the sentinel attr."""
    from torchregress.losses import MixtureDensityLoss, NormalizingFlowLoss

    assert getattr(MixtureDensityLoss, _PATCH_SENTINEL, False) is True
    assert getattr(NormalizingFlowLoss, _PATCH_SENTINEL, False) is True


def test_install_idempotent() -> None:
    """Re-installing must leave the bound methods intact (idempotent)."""
    from torchregress.losses import MixtureDensityLoss, NormalizingFlowLoss

    # Both classes should already have the sentinel from `import harness`
    # at the top of this module.
    assert getattr(MixtureDensityLoss, _PATCH_SENTINEL, False) is True
    assert getattr(NormalizingFlowLoss, _PATCH_SENTINEL, False) is True

    # Behavioural check: the second install must not raise.
    install_loss_protocol_augments()
    install_loss_protocol_augments()

    # Sentinel survives the second call.
    assert getattr(MixtureDensityLoss, _PATCH_SENTINEL, False) is True
    assert getattr(NormalizingFlowLoss, _PATCH_SENTINEL, False) is True

    # The methods are still callable (not stubbed/corrupted).
    assert callable(MixtureDensityLoss.log_prob)
    assert callable(NormalizingFlowLoss.predict_interval)


# ===========================================================================
# MixtureDensityLoss.log_prob()
# ===========================================================================


def test_mdn_log_prob_returns_per_sample_logp() -> None:
    """``MixtureDensityLoss.log_prob(y, t)`` returns shape ``[batch]``."""
    from torchregress.losses import MixtureDensityLoss

    torch.manual_seed(42)
    n_components, n_features = 3, 1
    loss_fn = MixtureDensityLoss(
        n_components=n_components,
        n_features=n_features,
        covariance_type="diagonal",
        reduction="none",
    )
    batch_size = 8
    out_size = n_components + 2 * n_components * n_features
    y_pred = torch.randn(batch_size, out_size)
    target = torch.randn(batch_size, n_features)

    lp = loss_fn.log_prob(y_pred, target)
    assert lp.shape == (batch_size,)
    assert torch.all(lp <= 0.0)
    assert torch.all(torch.isfinite(lp))


def test_mdn_log_prob_matches_negative_forward_sum() -> None:
    """``-log_prob(y, t)`` must equal ``forward(y, t)`` (per-element).

    Mathematical parity check.
    """
    from torchregress.losses import MixtureDensityLoss

    torch.manual_seed(7)
    n_components, n_features = 4, 1
    loss_fn = MixtureDensityLoss(
        n_components=n_components,
        n_features=n_features,
        covariance_type="diagonal",
        reduction="none",
    )
    batch_size = 16
    out_size = n_components + 2 * n_components * n_features
    y_pred = torch.randn(batch_size, out_size)
    target = torch.randn(batch_size, n_features)

    lp = loss_fn.log_prob(y_pred, target)
    nll_per_sample = -lp
    nll_per_sample_from_forward = loss_fn.forward(y_pred, target)

    assert nll_per_sample.shape == nll_per_sample_from_forward.shape
    assert torch.allclose(
        nll_per_sample,
        nll_per_sample_from_forward,
        atol=1e-5,
    )


def test_mdn_log_prob_full_covariance() -> None:
    """``log_prob`` works for full-covariance MDN (different code path)."""
    from torchregress.losses import MixtureDensityLoss

    torch.manual_seed(13)
    n_components, n_features = 2, 2
    loss_fn = MixtureDensityLoss(
        n_components=n_components,
        n_features=n_features,
        covariance_type="full",
        reduction="none",
    )
    batch_size = 4
    n_tril = n_features * (n_features + 1) // 2
    out_size = n_components + n_components * n_features + n_components * n_tril
    y_pred = torch.randn(batch_size, out_size)
    target = torch.randn(batch_size, n_features)

    lp = loss_fn.log_prob(y_pred, target)
    assert lp.shape == (batch_size,)
    assert torch.all(torch.isfinite(lp))


# ===========================================================================
# NormalizingFlowLoss.predict_interval()
# ===========================================================================


def test_nflow_predict_interval_shape_and_bounds() -> None:
    """``predict_interval(y, confidence=0.9)`` returns (lo, hi)."""
    from torchregress.losses import NormalizingFlowLoss

    flow = _build_test_flow(n_features=1, context_dim=8)
    loss_fn = NormalizingFlowLoss(flow=flow, reduction="none")

    batch_size = 32
    context = torch.randn(batch_size, 8)
    lower, upper = loss_fn.predict_interval(
        context,
        confidence=0.9,
        n_samples=256,
    )

    assert lower.shape == (batch_size, 1)
    assert upper.shape == (batch_size, 1)
    assert torch.all(lower <= upper)


def test_nflow_predict_interval_higher_confidence_wider() -> None:
    """Higher confidence widens intervals."""
    from torchregress.losses import NormalizingFlowLoss

    flow = _build_test_flow(n_features=1, context_dim=4)
    loss_fn = NormalizingFlowLoss(flow=flow, reduction="none")
    context = torch.randn(16, 4)

    lo_50, hi_50 = loss_fn.predict_interval(context, confidence=0.50, n_samples=512)
    lo_99, hi_99 = loss_fn.predict_interval(context, confidence=0.99, n_samples=512)

    assert torch.all(lo_99 <= lo_50)
    assert torch.all(hi_50 <= hi_99)


def test_nflow_predict_interval_rejects_bad_confidence() -> None:
    """``confidence`` outside (0, 1) must raise ``ValueError``."""
    from torchregress.losses import NormalizingFlowLoss

    flow = _build_test_flow(n_features=1, context_dim=4)
    loss_fn = NormalizingFlowLoss(flow=flow, reduction="none")

    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            loss_fn.predict_interval(torch.randn(4, 4), confidence=bad)


def test_nflow_predict_interval_rejects_n_features_gt_1() -> None:
    """Multi-target predict_interval must raise ``ValueError`` (not NotImplementedError).

    Normalising the exception type to ``ValueError`` lets callers rely on a
    single ``except ValueError`` for any predict_interval misuse — the
    underlying ``self.quantile`` raises ``NotImplementedError`` which this
    wrapper converts to ``ValueError``.
    """
    from torchregress.losses import NormalizingFlowLoss

    flow = _build_test_flow(n_features=2, context_dim=4)
    loss_fn = NormalizingFlowLoss(flow=flow, reduction="none")

    with pytest.raises(ValueError):
        loss_fn.predict_interval(torch.randn(4, 4), confidence=0.9)


# ===========================================================================
# PROPOSAL #15 dispatch contract: compute_pit_values dispatches on loss_fn
# ===========================================================================


def test_compute_pit_values_dispatches_on_loss_fn() -> None:
    """``compute_pit_values`` accepts ``loss_fn/context/target`` kwargs.

    The legacy (predictions dict) path must continue to work; the loss_fn
    path must return a 1-D ``[batch]`` numpy array of PIT values that
    fall within ``[0, 1]``.
    """
    import numpy as np
    from torchregress.losses import NormalizingFlowLoss

    from harness.metrics import compute_pit_values, pit_uniformity_tests

    flow = _build_test_flow(n_features=1, context_dim=4)
    loss_fn = NormalizingFlowLoss(flow=flow, reduction="none")
    batch_size = 64
    context = torch.randn(batch_size, 4)
    target = torch.randn(batch_size, 1)

    pit = compute_pit_values(
        predictions={},  # empty dict, would normally return empty
        y_true=target.detach().cpu().numpy(),
        loss_fn=loss_fn,
        context=context,
        target=target,
        n_samples=128,
    )
    assert pit.shape == (batch_size,)
    assert np.all((pit >= 0.0) & (pit <= 1.0))
    # pit_uniformity_tests must accept the output unchanged.
    metrics = pit_uniformity_tests(pit)
    assert set(metrics) >= {"PIT_Mean", "PIT_Std", "PIT_KS", "PIT_CvM"}


def test_compute_pit_values_legacy_predictions_dict_still_works() -> None:
    """Backwards compatibility: predictions-dict path unchanged."""
    import numpy as np

    from harness.metrics import compute_pit_values

    rng = np.random.default_rng(0)
    samples = rng.normal(size=(50, 256))  # (N, S) layout
    y_true = rng.normal(size=50)

    pit = compute_pit_values(predictions={"samples": samples}, y_true=y_true)
    assert pit.shape == (50,)
    assert np.all((pit >= 0.0) & (pit <= 1.0))


def test_compute_pit_values_loss_fn_overrides_predictions() -> None:
    """If both ``predictions`` and ``loss_fn`` are passed, ``loss_fn`` wins.

    Documented in the metrics.py docstring: PROPOSAL #15 dispatch path
    takes precedence.  The empty predictions dict is ignored.
    """
    import numpy as np
    import torch
    from torchregress.losses import NormalizingFlowLoss

    from harness.metrics import compute_pit_values

    flow = _build_test_flow(n_features=1, context_dim=4)
    loss_fn = NormalizingFlowLoss(flow=flow, reduction="none")
    context = torch.randn(32, 4)
    target = torch.randn(32, 1)

    pit_dict = compute_pit_values(
        predictions={"samples": np.zeros((32, 256))},  # would yield PIT=0
        y_true=target.detach().cpu().numpy(),
        loss_fn=loss_fn,
        context=context,
        target=target,
        n_samples=64,
    )
    # The dict path would have produced all zeros (samples uniform, all y > all
    # samples → mean=0).  The loss_fn path produces a uniform-ish empirical
    # draw.  At least one PIT value must be > 0.05.
    assert pit_dict.max() > 0.05


def test_compute_pit_values_rejects_unsupported_loss_type() -> None:
    """Passing an unsupported object as ``loss_fn`` raises ``TypeError``."""
    import torch

    from harness.metrics import compute_pit_values

    with pytest.raises(TypeError):
        compute_pit_values(
            predictions={},
            y_true=torch.zeros(4).numpy(),
            loss_fn=torch.nn.Linear(4, 4),  # arbitrary torch module
            context=torch.randn(4, 4),
            target=torch.randn(4, 1),
        )


def test_compute_pit_values_loss_fn_path_bridges_numpy_target() -> None:
    """Headline entry point: caller passes numpy y_true + torch context only.

    Regression test for a previously-fixed bug where the loss_fn dispatch path
    left ``target`` as a NumPy array if the caller didn't supply it, causing
    ``target.unsqueeze(-1)`` to fail with ``AttributeError``.  The fix bridges
    NumPy → torch tensor at the dispatch boundary so callers can use the
    headline new entry point:

        compute_pit_values(
            y_true=np.asarray(y_test),  # numpy
            loss_fn=fitted_loss,         # torchregress DistributionLoss
            context=X_test_tensor,        # torch.Tensor
        )

    without manually constructing a torch ``target`` argument.
    """
    import numpy as np
    import torch
    from torchregress.losses import NormalizingFlowLoss

    from harness.metrics import compute_pit_values

    flow = _build_test_flow(n_features=1, context_dim=4)
    loss_fn = NormalizingFlowLoss(flow=flow, reduction="none")
    context = torch.randn(32, 4)
    target = torch.randn(32, 1)

    # No ``target`` kwarg — the caller's headline new entry point.
    pit = compute_pit_values(
        predictions={},
        y_true=target.detach().cpu().numpy(),  # numpy array as headline call
        loss_fn=loss_fn,
        context=context,  # torch tensor
        n_samples=64,
        # target=None implicitly
    )
    assert pit.shape == (32,)
    assert np.all((pit >= 0.0) & (pit <= 1.0))
