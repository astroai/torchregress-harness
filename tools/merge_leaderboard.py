"""Merge phase-wise leaderboard reruns into the canonical artifacts.

H3 regeneration produced:
  - leaderboard_results.json/md : full 17-suite run (all rows EXCEPT the
    conformal block, which was wiped by a since-fixed NameError, and with five
    inline-row suites lacking the SyntheticFallback provenance column)
    results/latest/_rerun_suites6.json : single consolidated rerun of
    conformal/eiv/calibration/pzflow_parity/r2c_conformal/distributional_bins
    on fully fixed code (crepes `confidence` kwarg, RC fit/transform, RC
    branches computing X_cal_test, provenance column present)

Supersedes those suites' rows with their reruns, re-aggregates with the
leaderboard module's own functions, rewrites both artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/src/torchregress-harness")
sys.path.insert(0, "/scratch/src/torchregress-harness/src")

from tools.leaderboard_regression import (  # noqa: E402
    aggregate_per_method,
    render_leaderboard_markdown,
)

RERUN_SUITES = {
    "conformal",
    "eiv",
    "calibration",
    "pzflow_parity",
    "r2c_conformal",
    "distributional_bins",
}
HN = Path("/scratch/src/torchregress-harness")


def suite_of(row: dict) -> str:
    return str(row.get("Method", "?")).split("/", 1)[0]


def main() -> None:
    old = json.loads((HN / "leaderboard_results.json").read_text())
    rr = json.loads((HN / "results/latest/_rerun_suites6.json").read_text())

    kept = [r for r in old["rows"] if suite_of(r) not in RERUN_SUITES]
    rows = kept + [r for r in rr["rows"] if suite_of(r) in RERUN_SUITES]

    missing = [r["Method"] for r in rows if "SyntheticFallback" not in r]
    assert not missing, f"{len(missing)} rows lack provenance: {missing[:5]}"

    target_coverage = old["metadata"].get("target_coverage", 0.9)
    summary = aggregate_per_method(rows, target_coverage=target_coverage)
    md = render_leaderboard_markdown(summary, rows, target_coverage=target_coverage)

    metadata = dict(old["metadata"])
    metadata.update(
        {
            "n_raw_rows": len(rows),
            "n_methods_total": len(summary),
            "regeneration_note": (
                "Phase-wise against pinned externals (MAPIE git 6478f16c9a, "
                "torch-uncertainty git 94e511a737): full 17-suite pass, then "
                "targeted reruns of conformal+eiv (post NameError/kwarg fixes) "
                "and four inline-row suites (provenance column)."
            ),
        }
    )
    (HN / "leaderboard_results.json").write_text(
        json.dumps(
            {"metadata": metadata, "rows": rows, "leaderboard": summary}, indent=2, default=float
        )
    )
    (HN / "leaderboard_results.md").write_text(md)

    sf_true = sum(1 for r in rows if r.get("SyntheticFallback"))
    print(f"rows={len(rows)} methods={len(summary)} synthetic_fallback_true={sf_true}")
    print("MERGE_OK")


if __name__ == "__main__":
    main()
