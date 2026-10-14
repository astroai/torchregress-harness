"""sklearn_baselines MultiQuantile: crossed quantile heads are rearranged."""

from __future__ import annotations

import numpy as np
import torch
from suites.tabular.sklearn_baselines import SklearnBaselinesSuite

from harness.metrics import compute_pit_values
from harness.runners import HarnessConfig


class _Crossed(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (0.9, 0.5, 0.1) quantiles in the (0.1, 0.5, 0.9) slots: fully crossed
        z = x[:, :1]
        return torch.cat([z + 1.0, z, z - 1.0], dim=1)


def test_crossed_quantiles_are_sorted() -> None:
    X = np.random.default_rng(0).normal(size=(20, 2)).astype(np.float32)
    out = SklearnBaselinesSuite().predict(
        "MultiQuantile", {"type": "multi_quantile", "model": _Crossed()}, X, HarnessConfig()
    )
    assert np.all(out["lower"] < out["y_pred"]) and np.all(out["y_pred"] < out["upper"])
    np.testing.assert_allclose(out["y_pred"], X[:, 0], rtol=1e-6)
    compute_pit_values(out, X[:, 0])  # no "upper > lower" failure
