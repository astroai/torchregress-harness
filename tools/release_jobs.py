#!/usr/bin/env python
"""List the (suite, dataset) jobs of one release-run group, one "suite dataset" per line.

Used by ``scripts/skaha_release.sh`` to shard a group across headless replicas::

    python tools/release_jobs.py tabular      # CPU tabular suites
    python tools/release_jobs.py sbi          # sbi_parity tasks
    python tools/release_jobs.py fm           # foundation models (GPU)
    python tools/release_jobs.py image        # image suites (GPU)

Jobs come from each suite's ``supported_datasets``; datasets whose upstream
source is known to be gone are skipped (see ``EXCLUDED_DATASETS``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Runnable as a script from the repo root (same convention as tools/run_all.py).
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO), str(_REPO / "src")]

from tools._suite_registry import ALL_SUITES  # noqa: E402

GROUPS: dict[str, tuple[str, ...]] = {
    "sbi": ("sbi_parity",),
    "fm": ("foundation_model", "fm_uq_scenarios", "fm_multitarget"),
    "image": (
        "age_regression",
        "cell_counting",
        "censored_imaging",
        "crowd_counting",
        "depth_estimation",
        "image_calibration",
        "image_conformal",
        "image_density_parity",
        "image_ensembles_uq",
        "image_robust",
        "keypoint_regression",
        "photoz_imaging",
    ),
}
# Upstream mirror gone (ETH wiki_meta.tar 404s since 2026-08) or full-resolution
# variants that exceed one session; tiny variants of the same data still run.
# Not staged on /arc (licence agreement or Kaggle login needed; see
# RELEASE_0.3_STATUS.md) -- strict mode would only log a skip for each.
_NOT_STAGED = (
    "utkface_age",
    "utkface_age_tiny",
    "boneage",
    "boneage_tiny",
    "biwi_head_pose",
    "biwi_head_pose_tiny",
    "appa_age",
    "appa_age_tiny",
    "galaxiesml_cutouts",
)
# Datasets of >= 1M rows: their tabular-suite jobs form the ``tabular_big``
# group (GPU, 128 GB) instead of sharing 32 GB CPU sessions with small jobs.
BIG_DATASETS = frozenset({"tlc_fare", "folktables_eiv", "folktables_all"})
EXCLUDED_DATASETS = frozenset(
    {"imdb_wiki", "imdb_wiki_fullres", "nyu_depth_v2_fullres", *_NOT_STAGED}
)


def jobs(group: str) -> list[tuple[str, str]]:
    """``(suite, dataset)`` pairs for a group, in a stable order."""
    if group in ("tabular", "tabular_big"):
        taken = {s for g in GROUPS.values() for s in g}
        suites = tuple(s for s in ALL_SUITES if s not in taken)
    elif group in GROUPS:
        suites = GROUPS[group]
    else:
        raise ValueError(
            f"unknown group {group!r}; expected tabular, tabular_big, {', '.join(GROUPS)}"
        )
    out: list[tuple[str, str]] = []
    for suite in suites:
        for dataset in ALL_SUITES[suite]().supported_datasets:
            if dataset in EXCLUDED_DATASETS:
                continue
            if group in ("tabular", "tabular_big") and (
                (dataset in BIG_DATASETS) != (group == "tabular_big")
            ):
                continue
            out.append((suite, dataset))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("group", choices=["tabular", "tabular_big", *GROUPS])
    args = parser.parse_args(argv)
    for suite, dataset in jobs(args.group):
        print(suite, dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
