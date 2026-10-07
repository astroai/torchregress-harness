"""Contracts for the release-run job lists (tools/release_jobs.py)."""

from __future__ import annotations

import pytest
from tools._suite_registry import ALL_SUITES
from tools.release_jobs import EXCLUDED_DATASETS, GROUPS, jobs


def test_groups_partition_every_suite() -> None:
    covered = {s for g in ("tabular", *GROUPS) for s, _ in jobs(g)}
    assert covered == set(ALL_SUITES)


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
    assert jobs("fm") and all(s == "foundation_model" for s, _ in jobs("fm"))


def test_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown group"):
        jobs("nope")


def test_group_job_counts() -> None:
    """Job counts quoted in RELEASE_0.3_STATUS.md (update both together)."""
    counts = {g: len(jobs(g)) for g in ("tabular", *GROUPS)}
    assert counts == {"tabular": 133, "sbi": 5, "fm": 4, "image": 49}
