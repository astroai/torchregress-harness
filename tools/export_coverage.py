#!/usr/bin/env python
"""Which torchregress public exports does the harness benchmark?

Imports torchregress, enumerates ``__all__`` of the public modules in
``MODULES`` and maps each export to the registered harness suites whose
source references it (static scan of ``suites/``; no suite is run).

A suite references a name when its module, or a ``suites`` helper module it
imports (``suites/_backbones.py``, ``suites/tabular/_likelihood_designs.py``,
...), mentions it in code: an import, an attribute, a bare name, or a string
constant equal to the name (``getattr(losses, "SkewTLoss")`` registries).
Comments and docstrings do not count.

Every export must be covered by a suite or listed, with a one-line reason, in
``tools/export_coverage_allowlist.py`` (enforced by
``tests/test_export_coverage.py``).

Usage::

    python tools/export_coverage.py                 # write EXPORT_COVERAGE.md + export_coverage.json
    python tools/export_coverage.py --check         # exit 1 on an export neither covered nor allowlisted
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path[:0] = [str(_REPO), str(_REPO / "src")]

MODULES: tuple[str, ...] = (
    "losses",
    "metrics",
    "calibration",
    "ensemble",
    "algorithms",
    "test_time",
    "inference",
    "causal",
    "semi_supervised",
    "constraints",
)
SUITES_DIR = _REPO / "suites"
DEFAULT_MD = _REPO / "EXPORT_COVERAGE.md"
DEFAULT_JSON = _REPO / "export_coverage.json"


def public_exports(modules: tuple[str, ...] = MODULES) -> dict[str, list[str]]:
    """``{module: sorted __all__}`` for each ``torchregress.<module>``."""
    out: dict[str, list[str]] = {}
    for mod in modules:
        m = importlib.import_module(f"torchregress.{mod}")
        names = getattr(m, "__all__", None)
        if names is None:
            raise RuntimeError(f"torchregress.{mod} has no __all__")
        out[mod] = sorted(dict.fromkeys(str(n) for n in names))
    return out


def _module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _identifiers(tree: ast.AST) -> set[str]:
    """Names a module's code mentions (docstrings and comments excluded)."""
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            found.add(node.name.rsplit(".", 1)[-1])
            if node.asname:
                found.add(node.asname)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.isidentifier()
            and id(node) not in docstrings
        ):
            found.add(node.value)
    return found


def _imported_suite_modules(tree: ast.AST, module: str) -> set[str]:
    """``suites.*`` modules a module imports (absolute or relative)."""
    package = module.rsplit(".", 1)[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                target = ".".join([*base, node.module] if node.module else base)
            else:
                target = node.module or ""
            if target.startswith("suites"):
                out.add(target)
                out.update(f"{target}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith("suites"))
    return out


def scan_suites(suite_modules: dict[str, str], root: Path = SUITES_DIR) -> dict[str, set[str]]:
    """``{identifier: {suite names}}`` over the registered suites and their helpers.

    ``suite_modules`` maps a suite's module (``suites.tabular.likelihoods``)
    to its registry name (``likelihoods``).
    """
    idents: dict[str, set[str]] = {}
    imports: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        mod = _module_name(path, root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        idents[mod] = _identifiers(tree)
        imports[mod] = {m for m in _imported_suite_modules(tree, mod)}

    def closure(mod: str) -> set[str]:
        seen: set[str] = set()
        stack = [mod]
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in idents:
                continue
            seen.add(cur)
            stack.extend(imports.get(cur, ()))
        return seen

    refs: dict[str, set[str]] = defaultdict(set)
    for mod, suite in suite_modules.items():
        for dep in closure(mod):
            for name in idents[dep]:
                refs[name].add(suite)
    return dict(refs)


def registered_suite_modules() -> dict[str, str]:
    from tools._suite_registry import ALL_SUITES

    return {cls.__module__: name for name, cls in ALL_SUITES.items()}


def build_coverage(allowlist: dict[str, str] | None = None) -> dict[str, Any]:
    """Coverage report: per module, per export -> suites / allowlist status."""
    if allowlist is None:
        from tools.export_coverage_allowlist import ALLOWLIST

        allowlist = ALLOWLIST
    exports = public_exports()
    refs = scan_suites(registered_suite_modules())
    modules: dict[str, dict[str, dict[str, Any]]] = {}
    n_total = n_cov = n_allow = n_unc = 0
    for mod, names in exports.items():
        modules[mod] = {}
        for name in names:
            suites = sorted(refs.get(name, ()))
            if suites:
                status, reason = "covered", ""
                n_cov += 1
            elif name in allowlist:
                status, reason = "allowlisted", allowlist[name]
                n_allow += 1
            else:
                status, reason = "uncovered", ""
                n_unc += 1
            n_total += 1
            modules[mod][name] = {"suites": suites, "status": status, "reason": reason}
    covered_names = {n for names in exports.values() for n in names if refs.get(n)}
    all_names = {n for names in exports.values() for n in names}
    import torchregress

    return {
        "torchregress_version": getattr(torchregress, "__version__", "unknown"),
        "torchregress_commit": _library_commit(torchregress.__file__),
        "modules": modules,
        "summary": {
            "exports": n_total,
            "covered": n_cov,
            "allowlisted": n_allow,
            "uncovered": n_unc,
            "unique_names": len(all_names),
            "unique_covered": len(covered_names),
        },
        # Allowlist entries that are now covered or no longer exported.
        "stale_allowlist": sorted(n for n in allowlist if n in covered_names or n not in all_names),
    }


def _library_commit(package_file: str) -> str | None:
    """Short git commit of an editable torchregress checkout (None for a wheel)."""
    import subprocess

    for parent in Path(package_file).resolve().parents:
        if (parent / ".git").exists():
            try:
                out = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            return out.stdout.strip() or None
    return None


def render_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# torchregress export coverage",
        "",
        f"Generated by `tools/export_coverage.py` against torchregress "
        f"`{report['torchregress_version']}`"
        + (
            f" (commit `{report['torchregress_commit']}`)"
            if report.get("torchregress_commit")
            else ""
        )
        + ". Static scan of the registered suites "
        "(and the `suites` helper modules they import) for each public export of "
        f"{', '.join(f'`{m}`' for m in MODULES)}.",
        "",
        f"- Exports (module, name): **{s['exports']}**",
        f"- Covered by at least one suite: **{s['covered']}**",
        f"- Allowlisted with a reason (`tools/export_coverage_allowlist.py`): **{s['allowlisted']}**",
        f"- Uncovered: **{s['uncovered']}**",
        "",
        "An export counts once per module that lists it in `__all__` (e.g. "
        "`ExpectedCalibrationError` in both `metrics` and `calibration`).",
        "",
    ]
    for mod, entries in report["modules"].items():
        n_cov = sum(1 for e in entries.values() if e["status"] == "covered")
        lines += [
            f"## `torchregress.{mod}` ({n_cov}/{len(entries)} covered)",
            "",
            "| Export | Suites | Status |",
            "|:---|:---|:---|",
        ]
        for name, e in entries.items():
            suites = ", ".join(f"`{x}`" for x in e["suites"]) or "-"
            status = e["status"] if e["status"] != "allowlisted" else f"allowlisted: {e['reason']}"
            lines.append(f"| `{name}` | {suites} | {status} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--check", action="store_true", help="only check; exit 1 on uncovered exports"
    )
    args = parser.parse_args(argv)
    report = build_coverage()
    s = report["summary"]
    print(
        f"{s['exports']} exports: {s['covered']} covered, {s['allowlisted']} allowlisted, "
        f"{s['uncovered']} uncovered"
    )
    uncovered = [
        f"{mod}.{name}"
        for mod, entries in report["modules"].items()
        for name, e in entries.items()
        if e["status"] == "uncovered"
    ]
    for item in uncovered:
        print(f"  UNCOVERED {item}")
    for name in report["stale_allowlist"]:
        print(f"  STALE allowlist entry {name}")
    if args.check:
        return 1 if uncovered or report["stale_allowlist"] else 0
    args.markdown.write_text(render_markdown(report) + "\n", encoding="utf-8")
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.markdown} and {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
