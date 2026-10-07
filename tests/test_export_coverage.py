"""Every public torchregress export is benchmarked by a suite or allowlisted with a reason."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from tools.export_coverage import (
    MODULES,
    _identifiers,
    build_coverage,
    main,
    public_exports,
    render_markdown,
    scan_suites,
)
from tools.export_coverage_allowlist import ALLOWLIST


@pytest.fixture(scope="module")
def report() -> dict:
    return build_coverage()


def test_every_export_is_covered_or_allowlisted(report: dict) -> None:
    uncovered = [
        f"torchregress.{mod}.{name}"
        for mod, entries in report["modules"].items()
        for name, entry in entries.items()
        if entry["status"] == "uncovered"
    ]
    assert not uncovered, (
        "exports neither referenced by a suite nor in tools/export_coverage_allowlist.py: "
        + ", ".join(uncovered)
    )


def test_allowlist_has_no_stale_entries(report: dict) -> None:
    assert not report["stale_allowlist"], (
        "allowlisted names that a suite now covers (or torchregress no longer exports); "
        f"delete them from the allowlist: {report['stale_allowlist']}"
    )


@pytest.mark.parametrize("name", sorted(ALLOWLIST))
def test_allowlist_reasons_are_one_line(name: str) -> None:
    reason = ALLOWLIST[name]
    assert isinstance(reason, str) and len(reason.strip()) >= 15
    assert "\n" not in reason


def test_release_coverage_targets_are_covered(report: dict) -> None:
    """The 0.3.0 coverage gaps this harness closed stay closed."""
    must = {
        "semi_supervised": {"TeacherStudentTrainer", "SAGERegLoss", "disagreement_to_weight"},
        "losses": {
            "JohnsonSUNLLLoss",
            "SinhArcsinhNLLLoss",
            "SkewNormalNLLLoss",
            "SkewTLoss",
            "GEVNLLLoss",
            "BetaRegressionNLLLoss",
            "RankNContrastLoss",
        },
        "test_time": {
            "estimate_target_prior_bbse",
            "estimate_target_prior_em",
            "ShiftFactoredPredictiveTransport",
        },
        "ensemble": {"BatchEnsembleRegressor", "BatchEnsembleLinear", "VariationalLinear"},
    }
    for mod, names in must.items():
        for name in names:
            assert report["modules"][mod][name]["status"] == "covered", (mod, name)
    assert (
        "semi_supervised" in report["modules"]["semi_supervised"]["TeacherStudentTrainer"]["suites"]
    )
    assert "likelihoods" in report["modules"]["losses"]["GEVNLLLoss"]["suites"]
    assert "age_regression" in report["modules"]["losses"]["RankNContrastLoss"]["suites"]


def test_public_exports_cover_all_modules() -> None:
    exports = public_exports()
    assert tuple(exports) == MODULES
    assert all(exports[m] for m in MODULES)


def test_docstrings_and_comments_do_not_count() -> None:
    tree = ast.parse(
        '"""Mentions GEVNLLLoss."""\n'
        "# SkewTLoss in a comment\n"
        "from torchregress.losses import JohnsonSUNLLLoss\n"
        "x = getattr(mod, 'SinhArcsinhNLLLoss')\n"
        "y = 'not an identifier!'\n"
    )
    names = _identifiers(tree)
    assert {"JohnsonSUNLLLoss", "SinhArcsinhNLLLoss"} <= names
    assert "GEVNLLLoss" not in names and "SkewTLoss" not in names


def test_helper_modules_are_attributed_to_importing_suite() -> None:
    refs = scan_suites({"suites.tabular.likelihoods": "likelihoods"})
    # beta_regression_nll is named only in suites/tabular/_likelihood_designs.py.
    assert refs["beta_regression_nll"] == {"likelihoods"}


def test_cli_writes_markdown_and_json(tmp_path: Path, report: dict) -> None:
    md, js = tmp_path / "cov.md", tmp_path / "cov.json"
    assert main(["--markdown", str(md), "--json", str(js)]) == 0
    data = json.loads(js.read_text())
    assert data["summary"] == report["summary"]
    text = md.read_text()
    assert text.startswith("# torchregress export coverage")
    assert render_markdown(report).splitlines()[0] in text
    assert main(["--check"]) == 0
