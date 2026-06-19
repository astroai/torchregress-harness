"""Unit tests for train_with_loss() and train_baseline() external opt parameter.

Verifies the refactoring that added optional `opt` parameter to these
training functions, now located in the shared diagnostic_utils module.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

# ── Import from shared diagnostic utilities ───────────────────────────
_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from diagnostic_utils import make_mlp, train_with_loss, train_baseline

# Smoke-test that imports resolve correctly
assert callable(make_mlp)
assert callable(train_with_loss)
assert callable(train_baseline)


def test_imports_resolve() -> None:
    """Verify critical imports from diagnostic scripts actually resolve."""
    assert callable(train_with_loss)
    assert callable(train_baseline)
    assert callable(make_mlp)
    # make_mlp constructs a valid module
    m = make_mlp(in_dim=4)
    assert isinstance(m, nn.Module)
    assert callable(m.forward)


# ── Synthetic test data ───────────────────────────────────────────────


def _make_synthetic_data(
    n_samples: int = 200, n_features: int = 4, seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Tiny synthetic dataset for fast unit tests."""
    rng = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, n_features, generator=rng)
    beta = torch.randn(n_features, 1, generator=rng)
    y = X @ beta + 0.1 * torch.randn(n_samples, 1, generator=rng)

    n_train = n_samples // 2
    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test_true": X[n_train:],
        "y_test": y[n_train:],
    }


# ── Tests: train_with_loss (from diagnose_hybrid_mc.py) ──────────────


def test_train_with_loss_mse_no_opt_parameter() -> None:
    """train_with_loss works without opt (backward compat — creates internal Adam)."""
    data = _make_synthetic_data()
    model = make_mlp(in_dim=4, hidden=16)
    result = train_with_loss(
        model, data["X_train"], data["y_train"],
        data["X_test_true"], data["y_test"],
        loss_type="mse", sigma_x=0.5, sigma_y=0.1, epochs=5, lr=1e-2, batch_size=32,
    )
    assert result["loss_type"] == "mse"
    assert isinstance(result["rmse_clean"], float)
    assert result["rmse_clean"] > 0.0
    assert isinstance(result["history"], list)
    assert len(result["history"]) >= 1


def test_train_with_loss_external_opt_momentum_carries() -> None:
    """External optimizer's state (momentum) carries over after training."""
    data = _make_synthetic_data()
    model = make_mlp(in_dim=4, hidden=16)

    opt = Adam(model.parameters(), lr=1e-2)
    # Snapshot optimizer state before training
    state_before = {
        name: {
            "step": opt.state[p].get("step", 0) if p in opt.state else 0,
            "exp_avg": opt.state[p].get("exp_avg", None) if p in opt.state else None,
        }
        for name, p in model.named_parameters()
    }

    result = train_with_loss(
        model, data["X_train"], data["y_train"],
        data["X_test_true"], data["y_test"],
        loss_type="mse", sigma_x=0.5, sigma_y=0.1, epochs=5, lr=1e-2, batch_size=32,
        opt=opt,
    )

    # Optimizer state should have been updated (step count increased, exp_avg populated)
    for name, p in model.named_parameters():
        before = state_before[name]
        assert p in opt.state, f"Parameter {name} not in optimizer state"
        after_step = opt.state[p].get("step", 0)
        assert after_step > before["step"], (
            f"Optimizer step count didn't increase for {name}"
        )
        after_exp_avg = opt.state[p].get("exp_avg", None)
        assert after_exp_avg is not None, f"exp_avg not set for {name}"
        if before["exp_avg"] is not None:
            assert not torch.equal(before["exp_avg"], after_exp_avg), (
                f"exp_avg unchanged for {name} — optimizer didn't update"
            )

    assert result["loss_type"] == "mse"


def test_train_with_loss_external_opt_respects_lr() -> None:
    """External optimizer with different LR overrides the internal default."""
    data = _make_synthetic_data()
    model = make_mlp(in_dim=4, hidden=16)
    opt = Adam(model.parameters(), lr=0.123)
    # The `lr` argument to train_with_loss should be ignored when opt is provided
    train_with_loss(
        model, data["X_train"], data["y_train"],
        data["X_test_true"], data["y_test"],
        loss_type="mse", sigma_x=0.5, sigma_y=0.1, epochs=2,
        lr=0.999,  # This should be ignored — external opt has LR=0.123
        batch_size=32, opt=opt,
    )
    # Verify the optimizer's LR was not changed to 0.999
    for pg in opt.param_groups:
        assert pg["lr"] == 0.123


def test_train_with_loss_all_loss_types_no_opt() -> None:
    """All four loss types work without explicit opt (backward compat)."""
    data = _make_synthetic_data()
    sigma_x = 0.5

    for loss_type in ("mse", "feiv", "mc", "hybrid"):
        model = make_mlp(in_dim=4, hidden=16)
        result = train_with_loss(
            model, data["X_train"], data["y_train"],
            data["X_test_true"], data["y_test"],
            loss_type=loss_type, sigma_x=sigma_x, sigma_y=0.1,
            epochs=3, lr=1e-2, batch_size=32,
        )
        assert result["loss_type"] == loss_type
        assert isinstance(result["rmse_clean"], float)
        assert not np.isnan(result["rmse_clean"]), f"{loss_type} returned NaN"


def test_train_with_loss_zero_epochs_still_evaluates() -> None:
    """train_with_loss with epochs=0 still returns valid result (final eval)."""
    data = _make_synthetic_data()
    model = make_mlp(in_dim=4, hidden=16)
    result = train_with_loss(
        model, data["X_train"], data["y_train"],
        data["X_test_true"], data["y_test"],
        loss_type="mse", sigma_x=0.5, sigma_y=0.1, epochs=0, lr=1e-2, batch_size=32,
    )
    assert isinstance(result["rmse_clean"], float)
    assert not np.isnan(result["rmse_clean"])


def test_train_with_loss_external_opt_zero_epochs_no_error() -> None:
    """External opt with epochs=0 works (edge case — no training, just eval)."""
    data = _make_synthetic_data()
    model = make_mlp(in_dim=4, hidden=16)
    opt = Adam(model.parameters(), lr=1e-2)
    result = train_with_loss(
        model, data["X_train"], data["y_train"],
        data["X_test_true"], data["y_test"],
        loss_type="mse", sigma_x=0.5, sigma_y=0.1, epochs=0, lr=1e-2, batch_size=32,
        opt=opt,
    )
    assert isinstance(result["rmse_clean"], float)


def test_train_with_loss_mismatched_opt_silently_ignored() -> None:
    """Mismatched optimizer (from a different model) is a caller error.

    train_with_loss doesn't validate the opt/model pairing — it trusts
    the caller. Depending on PyTorch version, this either raises or
    silently trains the wrong parameters. We verify it doesn't crash
    in an unexpected way.
    """
    data = _make_synthetic_data()
    model_a = make_mlp(in_dim=4, hidden=16)
    model_b = make_mlp(in_dim=4, hidden=16)
    opt_b = Adam(model_b.parameters(), lr=1e-2)

    # Using model_a with opt_b (optimizer attached to model_b).
    # This is a misuse pattern — train_with_loss doesn't guard against it.
    # We just verify it doesn't crash with an unexpected exception type.
    result = train_with_loss(
        model_a, data["X_train"], data["y_train"],
        data["X_test_true"], data["y_test"],
        loss_type="mse", sigma_x=0.5, sigma_y=0.1,
        epochs=1, lr=1e-2, batch_size=32, opt=opt_b,
    )
    # The result should still be structurally valid even if training
    # was a no-op (optimizer stepped wrong model's params)
    assert isinstance(result["rmse_clean"], float)
    assert not np.isnan(result["rmse_clean"])


# ── Tests: train_baseline (from diagnose_mc_warmup.py) ────────────────


def test_train_baseline_mse_no_opt_parameter() -> None:
    """train_baseline works without opt (backward compat)."""
    data = _make_synthetic_data()
    model = make_mlp(in_dim=4, hidden=16)
    result = train_baseline(
        model, data["X_train"], data["y_train"],
        data["X_test_true"], data["y_test"],
        label="MSE", sigma_x=0.5, total_epochs=5, lr=1e-2, batch_size=32,
    )
    assert result["label"] == "MSE"
    assert isinstance(result["rmse_clean"], float)
    assert result["rmse_clean"] > 0.0


def test_train_baseline_external_opt_momentum_carries() -> None:
    """External optimizer passed to train_baseline has its state updated."""
    data = _make_synthetic_data()
    model = make_mlp(in_dim=4, hidden=16)
    opt = Adam(model.parameters(), lr=1e-2)

    # Snapshot before
    state_before = {
        name: opt.state[p].get("step", 0) if p in opt.state else 0
        for name, p in model.named_parameters()
    }

    result = train_baseline(
        model, data["X_train"], data["y_train"],
        data["X_test_true"], data["y_test"],
        label="MSE", sigma_x=0.5, total_epochs=5, lr=1e-2, batch_size=32,
        opt=opt,
    )

    for name, p in model.named_parameters():
        assert p in opt.state
        assert opt.state[p].get("step", 0) > state_before[name]

    assert result["label"] == "MSE"


def test_train_baseline_external_opt_and_no_opt_produce_similar_results() -> None:
    """With same seed, external-opt and internal-opt runs should be similar.

    Both paths use Adam with same LR and same initialization, so results
    should be within numerical tolerance of each other.
    """
    data = _make_synthetic_data(seed=123)

    # Run 1: internal optimizer
    torch.manual_seed(42)
    model_a = make_mlp(in_dim=4, hidden=16)
    result_a = train_baseline(
        model_a, data["X_train"], data["y_train"],
        data["X_test_true"], data["y_test"],
        label="MSE", sigma_x=0.5, total_epochs=10, lr=1e-2, batch_size=32,
    )

    # Run 2: external optimizer (same seed)
    torch.manual_seed(42)
    model_b = make_mlp(in_dim=4, hidden=16)
    opt = Adam(model_b.parameters(), lr=1e-2)
    result_b = train_baseline(
        model_b, data["X_train"], data["y_train"],
        data["X_test_true"], data["y_test"],
        label="MSE", sigma_x=0.5, total_epochs=10, lr=1e-2, batch_size=32,
        opt=opt,
    )

    assert result_a["rmse_clean"] == pytest.approx(result_b["rmse_clean"], rel=1e-5)
