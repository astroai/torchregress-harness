"""Contracts for the release-run job lists (tools/release_jobs.py)."""

from __future__ import annotations

import pytest
from tools._suite_registry import ALL_SUITES
from tools.release_jobs import EXCLUDED_DATASETS, GROUPS, jobs


def test_groups_partition_every_suite() -> None:
    covered = {s for g in ("tabular", *GROUPS) for s, _ in jobs(g)}
    # Suites whose datasets are all excluded (e.g. photoz_imaging: GalaxiesML not staged).
    runnable = {
        name
        for name, cls in ALL_SUITES.items()
        if set(cls().supported_datasets) - EXCLUDED_DATASETS
    }
    assert covered == runnable


def test_groups_do_not_overlap() -> None:
    seen: set[tuple[str, str]] = set()
    for group in ("tabular", *GROUPS):
        pairs = set(jobs(group))
        assert not pairs & seen, group
        seen |= pairs


def test_gone_mirrors_are_excluded() -> None:
    datasets = {d for g in ("tabular", *GROUPS) for _, d in jobs(g)}
    assert not datasets & EXCLUDED_DATASETS


def test_new_suites_and_models_are_scheduled() -> None:
    tab = {s for s, _ in jobs("tabular")}
    assert {"orthogonal_inference", "constrained_outputs", "calibration"} <= tab
    assert "semi_supervised" in tab
    tab_pairs = set(jobs("tabular"))
    assert {("semi_supervised", "diabetes"), ("semi_supervised", "ssl_friedman")} <= tab_pairs
    families = ("skewnormal", "skewt", "johnsonsu", "sinharcsinh", "gev", "beta")
    assert {("likelihoods", f"lik_{f}") for f in families} <= tab_pairs
    assert {d for _, d in jobs("sbi")} == set(ALL_SUITES["sbi_parity"]().supported_datasets)
    assert jobs("fm") and {s for s, _ in jobs("fm")} == {
        "foundation_model",
        "fm_uq_scenarios",
        "fm_multitarget",
    }
    assert ("density_parity", "mt_rf1") in tab_pairs
    assert ("image_density_parity", "udacity_steering") in set(jobs("image"))


def test_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown group"):
        jobs("nope")


def test_group_job_counts() -> None:
    """Job counts quoted in RELEASE_0.3_STATUS.md (update both together)."""
    counts = {g: len(jobs(g)) for g in ("tabular", *GROUPS)}
    assert len(jobs("tabular_big")) == 3
    assert counts == {"tabular": 170, "sbi": 5, "fm": 14, "image": 31}
