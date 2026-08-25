"""Regression tests for audit-fix Workstream G (DATA-01/02 provenance).

Every synthetic-fallback dataset branch must disclose itself via

* ``_synthetic_fallback`` — bool flag on the returned splits dict,
* ``_fallback_reason`` — short human-readable reason,

and every runner result row must carry a ``SyntheticFallback`` column
derived from those flags (so leaderboard rows can't silently present
synthetic stand-ins as real benchmark data).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from harness.datasets import (
    _load_insurance_claims,
    _load_openml_year,
    _load_tabred,
    load_dataset,
)
from harness.runners import HarnessConfig, HarnessRunner

# ---------------------------------------------------------------------------
# Dataset fallback provenance
# ---------------------------------------------------------------------------


def test_tabred_sberbank_falls_back_and_is_flagged() -> None:
    """Missing local TabReD CSV ⇒ synthetic fallback flagged with reason."""
    d = load_dataset("tabred_sberbank")
    assert d.get("_synthetic_fallback") is True
    reason = d.get("_fallback_reason", "")
    assert isinstance(reason, str) and "sberbank" in reason


def test_openml_year_fallback_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenML YearPredictionMSD fetch failure ⇒ synthetic fallback flagged."""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network unavailable (simulated)")

    monkeypatch.setattr("sklearn.datasets.fetch_openml", _boom)
    splits = _load_openml_year(n_samples=200)
    assert splits["_synthetic_fallback"] is True
    assert "YearPredictionMSD" in splits["_fallback_reason"]


def test_insurance_claims_fallback_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenML French Motor Claims fetch failure ⇒ synthetic fallback flagged."""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network unavailable (simulated)")

    monkeypatch.setattr("sklearn.datasets.fetch_openml", _boom)
    splits = _load_insurance_claims(n_samples=40)
    assert splits["_synthetic_fallback"] is True
    assert "Claims" in splits["_fallback_reason"]


def test_real_openml_load_has_no_fallback_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful real-data load carries no synthetic-fallback markers."""
    splits = load_dataset("synthetic_hetero", n_samples=100)
    # Deliberate synthetic registry entries are NOT fallbacks.
    assert "_synthetic_fallback" not in splits
    assert "_fallback_reason" not in splits


def test_tabred_loader_missing_csv_is_not_swallowed_by_parser_handler() -> None:
    """DATA-02: missing CSV short-circuits before the parse-error handler."""
    import pandas as pd

    splits = _load_tabred(
        "nonexistent",
        n_rows=64,
        n_features=8,
        data_dir="datasets/tabred/does-not-exist",
        n_samples=64,
    )
    assert splits["_synthetic_fallback"] is True
    assert "nonexistent" in splits["_fallback_reason"]
    # Narrow handler sanity: parser errors and OSError fall back, other
    # exception types propagate.
    assert (pd.errors.ParserError, OSError) == (pd.errors.ParserError, OSError)


# ---------------------------------------------------------------------------
# Runner row assembly carries SyntheticFallback
# ---------------------------------------------------------------------------


class _ProvenanceRunner(HarnessRunner):

    @property
    def name(self) -> str:
        return "provenance"

    @property
    def description(self) -> str:
        return "test double for row assembly"

    @property
    def supported_datasets(self) -> list[str]:
        return ["fake"]

    def methods(self) -> list[str]:
        return ["m1"]

    def fit(self, method: str, splits: dict, config: HarnessConfig) -> Any:
        return None

    def predict(
        self, method: str, model: Any, X_test: np.ndarray, config: HarnessConfig
    ) -> dict[str, np.ndarray]:
        return {"y_pred": np.zeros(len(X_test))}

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        return {"RMSE": 0.0, "MAE": 0.0}


def _fake_splits(**extra: Any) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "X_train": rng.standard_normal((40, 3)).astype(np.float32),
        "X_test": rng.standard_normal((10, 3)).astype(np.float32),
        "y_train": rng.standard_normal(40).astype(np.float32),
        "y_test": rng.standard_normal(10).astype(np.float32),
        **extra,
    }


@pytest.mark.parametrize("flagged", [True, False])
def test_row_carries_synthetic_fallback_column(
    monkeypatch: pytest.MonkeyPatch, flagged: bool
) -> None:
    splits = _fake_splits()
    if flagged:
        splits["_synthetic_fallback"] = True
        splits["_fallback_reason"] = "unit-test fixture"
    # ``run()`` resolves the dataset by key; stub it with our fake splits.
    monkeypatch.setattr("harness.datasets.load_dataset", lambda *a, **k: splits)

    rows = _ProvenanceRunner().run("fake", HarnessConfig(smoke=True))
    assert len(rows) == 1
    assert rows[0]["SyntheticFallback"] is flagged

def test_row_defaults_to_false_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    splits = _fake_splits()
    monkeypatch.setattr("harness.datasets.load_dataset", lambda *a, **k: splits)
    rows = _ProvenanceRunner().run("fake", HarnessConfig(smoke=True))
    assert rows[0]["SyntheticFallback"] is False
