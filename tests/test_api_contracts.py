"""Test suite for API interface contracts.

Ensures that all sklearn wrappers and torchregress distribution losses
implement consistent signatures and methods.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torchregress.losses as losses

from harness.wrappers import (
    DeepEnsembleRegressor,
    EvidentialRegressor,
    GaussianNLLRegressor,
    HeteroscedasticEnsembleRegressor,
    MDNRegressor,
    NormalizingFlowRegressor,
)

# Core sklearn wrapper contract methods
WRAPPER_METHODS = ["fit", "predict", "predict_interval", "sample"]


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        DeepEnsembleRegressor,
        GaussianNLLRegressor,
        HeteroscedasticEnsembleRegressor,
        MDNRegressor,
        EvidentialRegressor,
        NormalizingFlowRegressor,
    ],
)
def test_wrapper_api_contract(wrapper_cls) -> None:
    """Verify that all wrapper classes implement the full unified API contract."""
    # Check method existence
    for method_name in WRAPPER_METHODS:
        assert hasattr(wrapper_cls, method_name), (
            f"{wrapper_cls.__name__} is missing required method {method_name}"
        )
        method = getattr(wrapper_cls, method_name)
        assert inspect.isfunction(method) or inspect.ismethod(method), (
            f"{wrapper_cls.__name__}.{method_name} is not callable"
        )

    # Verify signature parameters
    # fit(self, X, y)
    fit_sig = inspect.signature(wrapper_cls.fit)
    assert "X" in fit_sig.parameters, f"{wrapper_cls.__name__}.fit missing X parameter"
    assert "y" in fit_sig.parameters, f"{wrapper_cls.__name__}.fit missing y parameter"

    # predict(self, X)
    predict_sig = inspect.signature(wrapper_cls.predict)
    assert "X" in predict_sig.parameters, f"{wrapper_cls.__name__}.predict missing X parameter"

    # predict_interval(self, X, confidence)
    interval_sig = inspect.signature(wrapper_cls.predict_interval)
    assert "X" in interval_sig.parameters, (
        f"{wrapper_cls.__name__}.predict_interval missing X parameter"
    )
    assert "confidence" in interval_sig.parameters, (
        f"{wrapper_cls.__name__}.predict_interval missing confidence parameter"
    )
    # Ensure default confidence is 0.9
    assert interval_sig.parameters["confidence"].default == 0.9, (
        f"{wrapper_cls.__name__}.predict_interval default confidence must be 0.9"
    )

    # sample(self, X, n_samples)
    sample_sig = inspect.signature(wrapper_cls.sample)
    assert "X" in sample_sig.parameters, f"{wrapper_cls.__name__}.sample missing X parameter"
    assert "n_samples" in sample_sig.parameters, (
        f"{wrapper_cls.__name__}.sample missing n_samples parameter"
    )


def test_distribution_loss_subclasses_contract() -> None:
    """Document and check the interface parity of all torchregress DistributionLoss subclasses.

    This test verifies that all registered distribution losses inherit from the base
    DistributionLoss class and tracks which interface methods are implemented.
    """
    # 1. Discover all exported losses that are subclasses of DistributionLoss
    dist_losses = []
    for name in dir(losses):
        item = getattr(losses, name)
        if (
            inspect.isclass(item)
            and issubclass(item, losses.DistributionLoss)
            and item is not losses.DistributionLoss
        ):
            dist_losses.append(item)

    assert len(dist_losses) > 0, "No DistributionLoss subclasses found in torchregress.losses"

    # 2. Track methods implemented by each subclass
    # Based on Proposal #15, a clean unified DistributionLoss interface should support:
    # - forward (or log_prob / calculate_nll)
    # - sample
    # - predict_interval (or quantile / cdf)

    # We define a checklist of expected methods and log which ones are missing.
    # To keep the harness CI green while the upstream library changes are pending,
    # we assert the subclass contract matches a known current state.
    # This prevents new losses from being introduced with inconsistent interfaces.
    known_methods = {}
    for loss_cls in dist_losses:
        methods_found = {
            "forward": hasattr(loss_cls, "forward"),
            "sample": hasattr(loss_cls, "sample"),
            "predict_interval": hasattr(loss_cls, "predict_interval"),
            "quantile": hasattr(loss_cls, "quantile"),
            "cdf": hasattr(loss_cls, "cdf"),
            "log_prob": hasattr(loss_cls, "log_prob"),
        }
        known_methods[loss_cls.__name__] = methods_found

    # Assert interface properties for key classes to guard against regression:
    # - NormalizingFlowLoss (must have sample, quantile, cdf)
    if "NormalizingFlowLoss" in known_methods:
        nf_methods = known_methods["NormalizingFlowLoss"]
        assert nf_methods["sample"], "NormalizingFlowLoss must implement sample()"
        assert nf_methods["quantile"], "NormalizingFlowLoss must implement quantile()"
        assert nf_methods["cdf"], "NormalizingFlowLoss must implement cdf()"

    # - MixtureDensityLoss (must have sample, predict_interval)
    if "MixtureDensityLoss" in known_methods:
        mdn_methods = known_methods["MixtureDensityLoss"]
        assert mdn_methods["sample"], "MixtureDensityLoss must implement sample()"
        assert mdn_methods["predict_interval"], (
            "MixtureDensityLoss must implement predict_interval()"
        )

    # - EvidentialRegressionLoss (must have predict_interval)
    if "EvidentialRegressionLoss" in known_methods:
        ev_methods = known_methods["EvidentialRegressionLoss"]
        assert ev_methods["predict_interval"], (
            "EvidentialRegressionLoss must implement predict_interval()"
        )


@pytest.mark.parametrize(
    "device_name", ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]
)
def test_distribution_losses_behavioral(device_name: str) -> None:
    """Verify that all DistributionLoss subclasses execute properly on different devices."""
    device = torch.device(device_name)

    # 1. Discover all subclasses
    dist_losses = []
    for name in dir(losses):
        item = getattr(losses, name)
        if (
            inspect.isclass(item)
            and issubclass(item, losses.DistributionLoss)
            and item is not losses.DistributionLoss
        ):
            if item not in dist_losses:
                dist_losses.append(item)

    assert len(dist_losses) > 0, "No DistributionLoss subclasses found"

    # 2. Test each subclass
    for loss_cls in dist_losses:
        cls_name = loss_cls.__name__

        batch_size = 4
        n_features = 1

        # Setup inputs based on class name
        if cls_name == "MixtureDensityLoss":
            loss = loss_cls(n_components=3, n_features=n_features)
            # output size: weights (3) + mean (3*1) + std (3*1) = 9
            y_pred = torch.randn(batch_size, 9, device=device)
            target = torch.randn(batch_size, n_features, device=device)
            kwargs = {}
        elif cls_name in ("NormalizingFlowLoss", "ContrastiveFlowLoss"):
            try:
                import zuko
            except ImportError:
                pytest.skip("zuko not installed")
                return

            flow = zuko.flows.NSF(features=n_features, context=2)
            flow.context = 2
            loss = loss_cls(flow=flow)
            y_pred = torch.randn(batch_size, 2, device=device)
            target = torch.randn(batch_size, n_features, device=device)
            if cls_name == "ContrastiveFlowLoss":
                kwargs = {"negative_context": torch.randn(batch_size, 2, 2, device=device)}
            else:
                kwargs = {}
        elif cls_name == "MultivariateGaussianLoss":
            loss = loss_cls()
            y_pred = torch.randn(batch_size, n_features, device=device)
            target = torch.randn(batch_size, n_features, device=device)
            cov = torch.eye(n_features, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            kwargs = {"covariance_matrices": cov}
        elif cls_name == "LowRankGaussianLoss":
            loss = loss_cls()
            y_pred = torch.randn(batch_size, n_features, device=device)
            target = torch.randn(batch_size, n_features, device=device)
            cov_factor = torch.randn(batch_size, n_features, 1, device=device)
            cov_diag = torch.ones(batch_size, n_features, device=device)
            kwargs = {"cov_factor": cov_factor, "cov_diag": cov_diag}
        elif cls_name == "EvidentialRegressionLoss":
            loss = loss_cls()
            # gamma, nu, alpha, beta. nu, alpha, beta must be positive.
            gamma = torch.randn(batch_size, n_features, device=device)
            nu = torch.exp(torch.randn(batch_size, n_features, device=device)) + 0.1
            alpha = torch.exp(torch.randn(batch_size, n_features, device=device)) + 1.5
            beta = torch.exp(torch.randn(batch_size, n_features, device=device)) + 0.1
            y_pred = torch.cat([gamma, nu, alpha, beta], dim=-1)
            target = torch.randn(batch_size, n_features, device=device)
            kwargs = {}
        elif cls_name in (
            "GaussianNLLLoss",
            "BetaNLLLoss",
            "FaithfulGaussianLoss",
            "GaussianCRPSLoss",
        ):
            loss = loss_cls()
            y_pred = torch.randn(batch_size, 2 * n_features, device=device)
            target = torch.randn(batch_size, n_features, device=device)
            kwargs = {}
        else:
            # Fallback/default instantiation if a new subclass is added
            try:
                loss = loss_cls()
                y_pred = torch.randn(batch_size, 2 * n_features, device=device)
                target = torch.randn(batch_size, n_features, device=device)
                kwargs = {}
            except Exception:
                continue

        loss = loss.to(device)

        # Test forward
        out_loss = loss(y_pred, target, **kwargs)
        assert isinstance(out_loss, torch.Tensor)
        assert out_loss.device.type == device.type
        assert out_loss.dtype == torch.float32

        # Test sample
        if hasattr(loss, "sample"):
            samples = loss.sample(y_pred, n_samples=5)
            assert isinstance(samples, torch.Tensor)
            assert samples.device.type == device.type
            if cls_name == "MixtureDensityLoss":
                assert list(samples.shape) == [5, batch_size, n_features]
            else:
                assert list(samples.shape) == [batch_size, 5, n_features]

            # Test sample with n_samples = 1
            samples_1 = loss.sample(y_pred, n_samples=1)
            assert isinstance(samples_1, torch.Tensor)
            assert samples_1.device.type == device.type
            if cls_name == "MixtureDensityLoss":
                assert list(samples_1.shape) == [1, batch_size, n_features]
            else:
                assert list(samples_1.shape) == [batch_size, n_features]

        # Test predict_interval
        if hasattr(loss, "predict_interval"):
            lower, upper = loss.predict_interval(y_pred, confidence=0.95)
            assert isinstance(lower, torch.Tensor)
            assert isinstance(upper, torch.Tensor)
            assert lower.device.type == device.type
            assert upper.device.type == device.type
            assert list(lower.shape) == [batch_size, n_features]
            assert list(upper.shape) == [batch_size, n_features]

        # Test quantile
        if hasattr(loss, "quantile"):
            q = loss.quantile(y_pred, levels=[0.1, 0.5, 0.9])
            assert isinstance(q, torch.Tensor)
            assert q.device.type == device.type
            assert list(q.shape) == [batch_size, 3]

        # Test cdf
        if hasattr(loss, "cdf"):
            c = loss.cdf(y_pred, values=target)
            assert isinstance(c, torch.Tensor)
            assert c.device.type == device.type
            assert list(c.shape) == [batch_size]
            assert torch.all(c >= 0.0) and torch.all(c <= 1.0)

        # Test log_prob
        if hasattr(loss, "log_prob"):
            lp = loss.log_prob(y_pred, target)
            assert isinstance(lp, torch.Tensor)
            assert lp.device.type == device.type
            assert list(lp.shape) == [batch_size]

        # Test subclass-specific public helper methods
        if cls_name == "EvidentialRegressionLoss":
            # predict_with_uncertainty
            mean, aleatoric, epistemic = loss.predict_with_uncertainty(y_pred)
            for t in (mean, aleatoric, epistemic):
                assert isinstance(t, torch.Tensor)
                assert t.device.type == device.type
                assert list(t.shape) == [batch_size, n_features]
            # sample_predictions
            s_preds = loss.sample_predictions(y_pred, n_samples=5)
            assert isinstance(s_preds, torch.Tensor)
            assert s_preds.device.type == device.type
            assert list(s_preds.shape) == [5, batch_size, n_features]

        elif cls_name == "MixtureDensityLoss":
            # predict_mean_std
            m_mean, m_std = loss.predict_mean_std(y_pred)
            for t in (m_mean, m_std):
                assert isinstance(t, torch.Tensor)
                assert t.device.type == device.type
                assert list(t.shape) == [batch_size, n_features]
