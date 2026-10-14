"""Foundation-model rows: quantile grids scored as predictive distributions."""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular import foundation_model as fm

from harness.runners import HarnessConfig


def test_quantile_predictions_interval_and_sorting() -> None:
    levels = [0.05, 0.5, 0.95]
    q = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0]])  # first row crosses
    out = fm._quantile_predictions(q, levels)
    assert np.all(np.diff(out["samples"], axis=1) >= 0)
    assert out["lower"].tolist() == [0.0, 0.0]
    assert out["upper"].tolist() == [2.0, 2.0]


class _FakeTabPFN:
    def predict(self, X, *, output_type="mean", quantiles=None):
        assert output_type == "quantiles"
        return [np.full(len(X), q) for q in quantiles]


class _FakeTabICL:
    def predict(self, X, output_type="mean", alphas=None):
        assert output_type == "quantiles"
        return np.tile(np.asarray(alphas), (len(X), 1))


@pytest.mark.parametrize(("kind", "est"), [("tabpfn", _FakeTabPFN()), ("tabicl", _FakeTabICL())])
def test_tabpfn_tabicl_quantile_layout(kind: str, est) -> None:
    suite = fm.FoundationModelSuite()
    out = suite.predict("x", {"type": kind, "model": est}, np.zeros((5, 3)), HarnessConfig())
    assert out["samples"].shape == (5, len(fm._QUANTILE_LEVELS))
    assert np.allclose(out["lower"], 0.05, atol=0.011)
    assert np.allclose(out["upper"], 0.95, atol=0.011)


def test_sdm_rows_end_to_end_with_untrained_weights(monkeypatch) -> None:
    """The sdm path runs fit -> predict -> metrics (random weights, no download)."""
    sdm_models = pytest.importorskip("sdm.models")
    real = sdm_models.KumoTabular

    def small_untrained(**kwargs):
        kwargs.update(size="small", pretrained=False)
        return real(**kwargs)

    monkeypatch.setattr(sdm_models, "KumoTabular", small_untrained)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.normal(size=80)).astype(np.float32)
    splits = {"X_train": X[:60], "y_train": y[:60], "X_test": X[60:], "y_test": y[60:]}
    suite = fm.FoundationModelSuite()
    monkeypatch.setattr(fm.FoundationModelSuite, "methods", lambda self: ["KumoTabular"])
    (row,) = suite.run("unit", config=HarnessConfig(seed=0, smoke=True), splits=splits)
    assert "failed" not in str(row["Notes"]), row["Notes"]
    assert row["Library"] == "structured-data-models"
    for key in ("RMSE", "CRPS", "Coverage"):
        assert key in row and np.isfinite(row[key]), key
