"""Contracts for the ``*+Conformal`` rows of ``multivariate_intervals``.

``SLS+SLSConformal`` / ``SLS_lowrank+SLSConformal`` calibrate the SLS level set with
``SLSConformal``; ``MultivariateGaussian+Conformal`` is the matched
``MultivariateScoreConformal`` baseline.  All three train on the training data
minus a calibration carve.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from suites.tabular.multivariate_intervals import (
    MultivariateIntervalsSuite,
    _sls_region_box,
)

from harness.runners import HarnessConfig

CFG = HarnessConfig(seed=0, epochs=2, batch_size=64, smoke=True, standardize_targets=True)
ROWS = ["SLS+SLSConformal", "SLS_lowrank+SLSConformal", "MultivariateGaussian+Conformal"]
NEW_ROWS = [*ROWS, "SLS_earlystop"]


def test_rows_registered_and_noted() -> None:
    suite = MultivariateIntervalsSuite()
    for m in ["SLS", "SLS_lowrank", "MVScore", *NEW_ROWS]:
        assert m in suite.methods()
    for m in ROWS:
        assert "calibration carve" in suite.method_notes(m)


@pytest.fixture(scope="module")
def rows() -> dict[str, dict]:
    suite = MultivariateIntervalsSuite()
    suite.methods = lambda: ["SLS", *NEW_ROWS]  # type: ignore[method-assign]
    out = suite.run("synthetic_mv_gaussian", CFG)
    return {r["Method"].split("/", 1)[1]: r for r in out}


@pytest.mark.parametrize("method", NEW_ROWS)
def test_conformal_rows_report_coverage_and_width(rows: dict[str, dict], method: str) -> None:
    r = rows[method]
    assert r.get("Status", "ok") != "failed", r
    for key in ("JointCoverage", "Coverage_D0", "Coverage_D1", "Coverage_D2", "MeanWidth"):
        assert key in r or key.replace("MeanWidth", "Width") in r, (key, sorted(r))
    assert 0.0 <= r["JointCoverage"] <= 1.0
    width = r.get("MeanWidth", r.get("Width"))
    assert np.isfinite(width) and width > 0


@pytest.mark.parametrize("mode", ["SLS+SLSConformal", "SLS_lowrank+SLSConformal"])
def test_region_box_boundary_lies_on_calibrated_level_set(mode: str) -> None:
    """Boundary samples satisfy G(x, y) = scale * q_tau(x) (full and low-rank)."""
    suite = MultivariateIntervalsSuite()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, :3] + 0.3 * rng.normal(size=(120, 3))).reshape(-1)
    cfg = HarnessConfig(seed=0, epochs=1, batch_size=64, smoke=True, standardize_targets=False)
    state = suite.fit(mode, {"X_train": X, "y_train": y, "X_test": X[:5], "y_test": y[:15]}, cfg)
    assert state["n_cal"] >= 30
    loss_fn, model, D = state["loss_fn"], state["model"], state["n_features"]
    scale = 1.7
    _, lo, hi = _sls_region_box(model, loss_fn, X[:5], D, scale=scale, n_samples=64)
    _, lo2, hi2 = _sls_region_box(model, loss_fn, X[:5], D, scale=2 * scale, n_samples=64)
    assert np.all(hi >= lo) and np.all(hi2 - lo2 > hi - lo)
    # Record the boundary points the helper maps back and evaluate G on them.
    flow = loss_fn.frontier.flow
    captured: list[torch.Tensor] = []
    orig = flow.inverse

    def spy(z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        out = orig(z, c)
        captured.append(out)
        return out

    flow.inverse = spy  # type: ignore[method-assign]
    try:
        _sls_region_box(model, loss_fn, X[:5], D, scale=scale, n_samples=16)
    finally:
        del flow.inverse
    y_b = captured[0].reshape(16, 5, D)
    with torch.no_grad():
        ctx = model(torch.from_numpy(X[:5]).float())
        q = loss_fn.quantile_net(ctx)[:, 1] * scale
        g, _ = loss_fn.evaluate_frontier(
            y_b.reshape(-1, D), ctx.unsqueeze(0).expand(16, -1, -1).reshape(-1, ctx.shape[-1])
        )
    ratio = g.reshape(16, 5) / q.unsqueeze(0)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=2e-3), ratio


def test_sls_earlystop_restores_best_epoch_on_validation_carve() -> None:
    """Early stopping runs past warmup, keeps the best held-out-coverage epoch, restores it."""
    from suites.tabular.multivariate_intervals import (
        _ES_VAL_FRAC,
        _MultiOutputMLP,
        _train_sls_early_stop,
    )
    from torchregress.losses.sls import SLSLoss

    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.normal(size=(120, 4))).float()
    y = X[:, :3] + 0.3 * torch.from_numpy(rng.normal(size=(120, 3))).float()
    torch.manual_seed(0)
    model = _MultiOutputMLP(4, 16, out_dim=16)
    loss_fn = SLSLoss(
        d=3,
        context_dim=16,
        K=1,
        mode="full",
        hidden_dim=16,
        n_transforms=2,
        tau=0.9,
        warmup_steps=10,
    )
    cfg = HarnessConfig(seed=0, epochs=30, batch_size=16, smoke=True)
    info = _train_sls_early_stop(model, loss_fn, X, y, cfg)
    assert info["n_val"] == round(_ES_VAL_FRAC * 120)
    assert 0 <= info["best_epoch"] < info["epochs_run"] <= 30
    assert not model.training
    # Restored weights reproduce the recorded best validation coverage.
    gen = torch.Generator().manual_seed(0)
    va = torch.randperm(120, generator=gen)[: info["n_val"]]
    with torch.no_grad():
        ctx = model(X[va])
        g, _ = loss_fn.evaluate_frontier(y[va], ctx)
        cov = float((g <= loss_fn.quantile_net(ctx)[:, 1]).float().mean())
    assert cov == pytest.approx(info["val_region_coverage"])
