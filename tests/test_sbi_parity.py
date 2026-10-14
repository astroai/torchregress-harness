"""Focused contracts for SBI task expansion and calibration diagnostics."""

import numpy as np
from suites.tabular.sbi_parity import SBIParitySuite

from harness.datasets import DATASET_REGISTRY
from harness.runners import HarnessConfig


def test_sbi_parity_registers_stage_three_tasks() -> None:
    expected = {
        "sbibm_gaussian_linear",
        "sbibm_gaussian_mixture",
        "sbibm_bernoulli_glm",
    }
    suite = SBIParitySuite()
    assert expected <= set(DATASET_REGISTRY)
    assert expected <= set(suite.supported_datasets)


def test_sbc_diagnostics_use_held_out_truth_and_posterior_draws(monkeypatch) -> None:
    suite = SBIParitySuite()
    draws = np.tile(np.array([[0.0, 0.0], [0.4, 0.4], [0.6, 0.6], [1.0, 1.0]]), (50, 1))
    monkeypatch.setattr(suite, "_posterior_samples", lambda *args, **kwargs: draws)

    diagnostics = suite._sbc_diagnostics(
        {},
        {
            "sbc_x": np.zeros((2, 2), dtype=np.float32),
            "sbc_theta": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        },
        HarnessConfig(smoke=True),
    )

    assert diagnostics["SBC_Coverage95"] == 1.0
    assert diagnostics["SBC_RankMean"] == 0.5
    assert 0.0 <= diagnostics["SBC_RankKS"] <= 1.0


def test_paired_methods_share_estimator_family() -> None:
    suite = SBIParitySuite()
    methods = set(suite.methods())
    for tr, sbi in [
        ("torchregress_NSF", "sbi_NPE"),
        ("torchregress_MAF", "sbi_NPE_MAF"),
        ("torchregress_MDN", "sbi_NPE_MDN"),
    ]:
        assert {tr, sbi} <= methods
        assert sbi in suite.method_notes(tr)
    assert {"sbi_FMPE", "sbi_NPSE"} <= methods
    assert suite.library_name("sbi_FMPE") == "sbi"
    assert suite.library_name("torchregress_MDN") == "torchregress"


def test_fixed_draws_posterior_replays_and_guards() -> None:
    import pytest
    import torch
    from suites.tabular.sbi_parity import _FixedDrawsPosterior

    xs = np.zeros((3, 2), dtype=np.float32)
    draws = np.random.default_rng(0).normal(size=(10, 3, 1))
    post = _FixedDrawsPosterior(xs, draws)
    out = post.sample_batched((10,), x=torch.zeros(3, 2))
    assert out.shape == (10, 3, 1)
    with pytest.raises(ValueError, match="observations"):
        post.sample_batched((10,), x=torch.ones(3, 2))
    with pytest.raises(ValueError, match="draws"):
        post.sample_batched((11,), x=torch.zeros(3, 2))


def test_sbi_diagnostics_agree_with_harness_on_calibrated_draws() -> None:
    """Draws from the exact posterior: both implementations must call it calibrated.

    Model: theta ~ N(0, 1), x = theta + N(0, 1), so p(theta | x) = N(x/2, 1/2).
    """
    import pytest

    pytest.importorskip("sbi")
    rng = np.random.default_rng(1)
    n, s = 200, 300
    theta = rng.normal(size=(n, 1))
    x = theta + rng.normal(size=(n, 1))
    draws = x[None] / 2 + np.sqrt(0.5) * rng.normal(size=(s, n, 1))
    suite = SBIParitySuite()
    packed = (theta, x.astype(np.float32), draws)
    ours = suite._sbc_diagnostics({}, {}, HarnessConfig(), draws=packed)
    theirs = suite._sbi_diagnostics(packed, rng.normal(size=(n, 1)))
    assert abs(ours["SBC_Coverage95"] - 0.95) < 0.05
    assert ours["SBC_RankKS"] < 0.15
    assert theirs["SBI_SBC_KSpMin"] > 0.01
    assert abs(theirs["TARP_ATC"]) < 0.15


def test_sbi_diagnostics_flag_overconfident_draws() -> None:
    import pytest

    pytest.importorskip("sbi")
    rng = np.random.default_rng(2)
    n, s = 200, 300
    theta = rng.normal(size=(n, 1))
    x = theta + rng.normal(size=(n, 1))
    draws = x[None] / 2 + 0.1 * rng.normal(size=(s, n, 1))  # far too narrow
    suite = SBIParitySuite()
    packed = (theta, x.astype(np.float32), draws)
    ours = suite._sbc_diagnostics({}, {}, HarnessConfig(), draws=packed)
    theirs = suite._sbi_diagnostics(packed, rng.normal(size=(n, 1)))
    assert ours["SBC_Coverage95"] < 0.5
    assert theirs["SBI_SBC_KSpMin"] < 1e-3


def test_sbibm_frozen_copy_is_read_without_sbibm(tmp_path, monkeypatch) -> None:
    import builtins

    import torch

    from harness import datasets

    monkeypatch.setenv("TORCHREGRESS_HARNESS_CACHE", str(tmp_path))
    path = datasets._sbibm_cache_path("two_moons", 50, 1, 4, 42)
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "task": "two_moons",
            "theta": np.zeros((50, 2), np.float32),
            "x": np.ones((50, 2), np.float32),
            "observations": np.ones((1, 2), np.float32),
            "reference": [np.full((7, 2), 0.5, np.float32)],
            "sbc_theta": np.zeros((4, 2), np.float32),
            "sbc_x": np.zeros((4, 2), np.float32),
            "prior_dist": None,
        },
        path,
    )
    real_import = builtins.__import__

    def no_sbibm(name, *args, **kwargs):
        if name == "sbibm":
            raise ImportError("sbibm must not be imported when a frozen copy exists")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sbibm)
    splits = datasets._load_sbibm_task(
        "two_moons", n_samples=50, n_observations=1, n_sbc_simulations=4
    )
    assert "frozen" in splits["_source_url"]
    assert not splits.get("_synthetic_fallback")
    assert splits["y_test"].tolist() == [[0.5, 0.5]]
    assert "prior_dist" not in splits


def _wide_prior_splits(n: int = 2000, seed: int = 0) -> dict[str, np.ndarray]:
    """gaussian_mixture-like task: theta ~ U(-10, 10)^2, x = theta + noise."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-10.0, 10.0, size=(n, 2)).astype(np.float32)
    x = (theta + 0.3 * rng.standard_normal((n, 2))).astype(np.float32)
    return {"X_train": x, "y_train": theta}


def test_torchregress_nsf_reaches_wide_prior_beyond_spline_bound() -> None:
    """Regression: raw theta in U(-10, 10) put |theta| > 5 outside the NSF
    spline domain and posteriors collapsed onto the bound edge (sbibm
    gaussian_mixture C2ST 0.997). The suite now derives the tail bound from
    the training theta range."""
    import warnings

    suite = SBIParitySuite()
    splits = _wide_prior_splits()
    cfg = HarnessConfig(epochs=8, batch_size=128, lr=3e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # spline-range warning must not fire
        state = suite.fit("torchregress_NSF", splits, cfg)
    x_obs = np.array([-9.0, 8.0], dtype=np.float32)
    draws = suite._posterior_samples(state, x_obs, cfg, n_draws=1000)
    assert draws.shape == (1000, 2)
    # Draws centred near the observation, not piled at the +/-5 bound edge.
    np.testing.assert_allclose(draws.mean(0), x_obs, atol=1.0)
    assert float((np.abs(draws) > 6.0).mean()) > 0.5
    assert np.all(draws.std(0) < 2.0)


def test_torchregress_nsf_tail_bound_follows_theta_range() -> None:
    """Wide priors widen the spline domain; narrow ones keep zuko's default 5
    (so twomoons/slcp/gaussian_linear rows are unchanged)."""
    suite = SBIParitySuite()
    cfg = HarnessConfig(epochs=1, batch_size=256)
    wide = suite.fit("torchregress_NSF", _wide_prior_splits(n=512), cfg)
    assert wide["flow"].flow.tail_bound >= 12.0
    rng = np.random.default_rng(0)
    theta = rng.uniform(-1.0, 1.0, size=(512, 2)).astype(np.float32)
    narrow = suite.fit("torchregress_NSF", {"X_train": theta, "y_train": theta}, cfg)
    assert getattr(narrow["flow"].flow, "tail_bound", 5.0) == 5.0
