"""Contracts for the orthogonal (DML) inference suite."""

from __future__ import annotations

import numpy as np
import pytest
from suites.tabular.orthogonal_inference import (
    DESIGNS,
    Estimate,
    OrthogonalInferenceSuite,
    _per_covariate_powers,
    summarize,
)
from tools._suite_registry import ALL_SUITES

from harness.runners import HarnessConfig


def test_registered() -> None:
    assert ALL_SUITES["orthogonal_inference"] is OrthogonalInferenceSuite


def test_summarize_known_values() -> None:
    est = [Estimate(0.4, 0.1, 0.2, 0.6), Estimate(0.6, 0.1, 0.55, 0.65)]
    out = summarize(est, theta=0.5)
    assert out["Bias"] == pytest.approx(0.0)
    assert out["RMSE"] == pytest.approx(0.1)
    assert out["Coverage95"] == pytest.approx(0.5)
    assert out["CIWidth"] == pytest.approx(0.25)
    # empirical sd of {0.4, 0.6} is sqrt(0.02); mean se is 0.1
    assert out["SEratio"] == pytest.approx(0.1 / np.sqrt(0.02))


def test_power_basis_matches_torchregress_poly_without_intercept() -> None:
    import torch
    from torchregress.inference.orthogonal import _poly_features

    x = np.random.default_rng(0).normal(size=(6, 3))
    ours = _per_covariate_powers(x)
    theirs = _poly_features(torch.as_tensor(x), 3).numpy()[:, 1:]
    assert np.allclose(ours, theirs)


@pytest.mark.parametrize("name", list(DESIGNS))
def test_designs_are_confounded(name: str) -> None:
    """The naive OLS slope must be off, otherwise the benchmark proves nothing."""
    design = DESIGNS[name]
    y, d, x = design.draw(np.random.default_rng(0), 20_000, design.p, design.theta)
    naive = np.polyfit(d, y, 1)[0]
    assert abs(naive - design.theta) > 0.05


def test_torchregress_rows_recover_theta() -> None:
    suite = OrthogonalInferenceSuite()
    suite.methods = lambda: ["tr_Orthogonal_Poly3", "tr_Naive"]  # type: ignore[method-assign]
    rows = {
        r["Method"].split("/")[1]: r
        for r in suite.run("plr_ccddhnr", config=HarnessConfig(seed=0, smoke=True))
    }
    assert abs(rows["tr_Orthogonal_Poly3"]["Bias"]) < 0.06
    assert rows["tr_Naive"]["Bias"] > 0.1
    assert rows["tr_Orthogonal_Poly3"]["SyntheticFallback"] is False


def test_unknown_design_gives_failed_rows() -> None:
    rows = OrthogonalInferenceSuite().run("diabetes", config=HarnessConfig(smoke=True))
    assert rows and all(str(r["Notes"]).startswith("failed") for r in rows)
