"""Tests for tools/build_parity_report.py (PARITY_REPORT.md generator)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tools.build_parity_report import (
    Options,
    build_report,
    evaluate_pairs,
    index_rows,
    load_run_dir,
    main,
    rel_diff,
    verdict,
)


def _row(suite: str, method: str, library: str, dataset: str, **kw: Any) -> dict[str, Any]:
    row = {
        "Method": f"{suite}/{method}",
        "Library": library,
        "Dataset": dataset,
        "Suite": suite,
        "SyntheticFallback": False,
        "Device": "cpu",
        "Notes": "",
        "train_s": 0.5,
        "eval_s": 0.01,
    }
    row.update(kw)
    return row


def _conformal_rows(dataset: str, seed: int) -> list[dict[str, Any]]:
    s = "conformal"
    width = 2.0 + 0.1 * seed
    return [
        # tr post-hoc and MAPIE identical -> exact
        _row(
            s,
            "Split+Linear(post-hoc)",
            "torchregress",
            dataset,
            Coverage=0.9,
            Width=width,
            IntervalScore=3.0 + seed,
            Notes="linear + post-hoc",
        ),
        _row(
            s,
            "MAPIE_Split+Linear",
            "MAPIE",
            dataset,
            Coverage=0.9,
            Width=width,
            IntervalScore=3.0 + seed,
            Notes="mapie split",
        ),
        # tr Split+MLP vs MAPIE_Split+MLP: 1% apart -> close
        _row(
            s,
            "Split+MLP",
            "torchregress",
            dataset,
            Coverage=0.9,
            Width=width,
            IntervalScore=2.0,
            Notes="mlp split",
        ),
        _row(
            s,
            "MAPIE_Split+MLP",
            "MAPIE",
            dataset,
            Coverage=0.9,
            Width=width,
            IntervalScore=2.02,
            Notes="mapie mlp",
        ),
        _row(
            s,
            "Zoo_Local",
            "torchregress",
            dataset,
            Coverage=None,
            Width=None,
            IntervalScore=None,
            Notes="failed: RuntimeError boom",
        ),
        _row(
            s,
            "Zoo_Prevalence",
            "torchregress",
            dataset,
            Coverage=None,
            Width=None,
            IntervalScore=None,
            Notes="skipped: needs labels",
        ),
        _row(
            s,
            "CQR+MLP",
            "torchregress",
            dataset,
            Coverage=0.91,
            Width=width,
            IntervalScore=1.5,
            Notes="cqr mlp; this row ran on cpu (cpu retry)",
        ),
    ]


def _robust_rows(dataset: str, seed: int, synthetic_fallback: bool = False) -> list[dict[str, Any]]:
    s = "robust_regression"
    return [
        _row(
            s,
            "Huber_Linear",
            "torchregress",
            dataset,
            RMSE=1.1,
            MAE=0.8,
            R2=0.5,
            SyntheticFallback=synthetic_fallback,
            Notes="huber linear",
        ),
        _row(s, "statsmodels_RLM", "statsmodels", dataset, RMSE=1.0, MAE=0.8, R2=0.6, Notes="rlm"),
        _row(
            s,
            "MSE",
            "torchregress",
            dataset,
            RMSE=1.3 + 0.01 * seed,
            MAE=0.9,
            R2=0.7,
            Notes="mlp mse",
        ),
    ]


def _write(
    group_dir: Path, suite: str, dataset: str, seed: int, rows: list[dict[str, Any]]
) -> None:
    payload = {
        "artifact": "harness_run",
        "timestamp": f"2026-10-13T19:0{seed}:00+00:00",
        "environment": {"python": "3.13.16", "torch": "2.13.0", "cuda_available": False},
        "config": {"seed": seed, "epochs": 60, "batch_size": 64, "lr": 0.001, "device": "cpu"},
        "rows": rows,
    }
    path = group_dir / f"{suite}__{dataset}__seed{seed}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    g = root / "tabular"
    g.mkdir(parents=True)
    for dataset in ("diabetes", "synthetic_hetero"):
        for seed in (0, 1):
            _write(g, "conformal", dataset, seed, _conformal_rows(dataset, seed))
            fallback = dataset == "diabetes" and seed == 1
            _write(g, "robust_regression", dataset, seed, _robust_rows(dataset, seed, fallback))
    (g / "versions_replica_0.txt").write_text(
        "python 3.13.16 | torch 2.13.0 | cuda False\n"
        'mapie\t1.5.1\t{"vcs_info": {"commit_id": "929740af07f5"}}\n'
        "crepes\t0.9.1\t\n",
        encoding="utf-8",
    )
    (g / "versions_replica_1.txt").write_text(
        "python 3.13.16 | torch 2.13.0 | cuda False\nmapie\t1.5.2\t\ncrepes\t0.9.1\t\n",
        encoding="utf-8",
    )
    return root


def _opts(**kw: Any) -> Options:
    base = {
        "run_tag": "test_run",
        "harness_ref": "abc1234",
        "lib_ref": "def5678",
        "date": "2026-10-15",
    }
    base.update(kw)
    return Options(**base)


def test_output_is_deterministic(run_dir: Path, tmp_path: Path) -> None:
    out1, out2 = tmp_path / "a.md", tmp_path / "b.md"
    args = ["--run-dir", str(run_dir), "--run-tag", "t", "--date", "2026-10-15"]
    assert main([*args, "--out", str(out1)]) == 0
    assert main([*args, "--out", str(out2)]) == 0
    assert out1.read_text() == out2.read_text()
    text = out1.read_text()
    assert text.startswith("# torchregress Parity Report")
    assert "Generated 2026-10-15" in text


def test_pairing_verdicts(run_dir: Path) -> None:
    run = load_run_dir(run_dir)
    pairs = {(p.suite, p.dataset, p.tr, p.comp): p for p in evaluate_pairs(index_rows(run.rows))}
    exact = pairs[("conformal", "diabetes", "Split+Linear(post-hoc)", "MAPIE_Split+Linear")]
    assert exact.verdict == "exact"
    assert exact.kind == "deterministic"
    close = pairs[("conformal", "diabetes", "Split+MLP", "MAPIE_Split+MLP")]
    assert close.verdict == "close"
    assert close.n_seeds == 2
    differs = pairs[("robust_regression", "synthetic_hetero", "Huber_Linear", "statsmodels_RLM")]
    assert differs.verdict == "differs"
    # comparators are never paired with each other or with unrelated methods
    assert not any(p.tr.startswith("MAPIE_") for p in pairs.values())
    assert not any(p.tr in {"MSE", "CQR+MLP"} for p in pairs.values())


def test_verdict_thresholds_and_log_scale() -> None:
    assert verdict(0.0) == "exact"
    assert verdict(1e-7) == "exact"
    assert verdict(0.015) == "close"
    assert verdict(0.05) == "differs"
    # NLL near zero: absolute nats over max(|ref|, 1), not a blown-up ratio
    assert rel_diff("NLL", 0.01, 0.02) == pytest.approx(0.01)
    assert rel_diff("RMSE", 1.0, 2.0) == pytest.approx(0.5)


def test_bold_best_respects_direction(run_dir: Path) -> None:
    text = build_report(load_run_dir(run_dir), _opts())
    section = text.split("##### robust_regression / synthetic_hetero")[1].split("\n#")[0]
    lines = {ln.split("|")[1].strip(): ln for ln in section.splitlines() if ln.startswith("| `")}
    # RMSE: lower is better -> statsmodels_RLM 1.0 is bold, MSE is not
    assert "**1.0000 ± 0.0000**" in lines["`statsmodels_RLM`"]
    # R2: higher is better -> MSE (0.7) bold, not Huber_Linear (0.5)
    assert "**0.7000 ± 0.0000**" in lines["`MSE`"]
    assert "**0.5000" not in lines["`Huber_Linear`"]
    # Coverage is target-valued: never bold
    conf = text.split("##### conformal / diabetes")[1].split("\n#")[0]
    assert "**0.9000" not in conf and "**0.9100" not in conf
    # IntervalScore lower is better: CQR+MLP 1.5
    assert "**1.5000 ± 0.0000**" in conf


def test_failure_ledger_and_cpu_mark(run_dir: Path) -> None:
    text = build_report(load_run_dir(run_dir), _opts())
    conf = text.split("##### conformal / diabetes")[1].split("\n#")[0]
    assert "`Zoo_Local` (failed, seed 0, 1): failed: RuntimeError boom" in conf
    assert "`Zoo_Prevalence` (skipped, seed 0, 1): skipped: needs labels" in conf
    assert "`CQR+MLP` ‡" in conf
    ledger = text.split("## Appendix A: failure and skip ledger")[1].split("## Appendix B")[0]
    assert "4 failed and 4 skipped rows" in ledger
    assert (
        "| failed | `conformal` | `Zoo_Local` | diabetes (0,1); synthetic_hetero (0,1) | 4 |"
        in ledger
    )
    assert "(n=2)" in conf  # fewer than 3 seeds is shown


def test_synthetic_fallback_alarm(run_dir: Path) -> None:
    text = build_report(load_run_dir(run_dir), _opts())
    assert "SYNTHETIC FALLBACK ON REAL DATA: 1 row(s)" in text
    assert "`robust_regression/Huber_Linear` on `diabetes` seed 1" in text
    # synthetic-by-design datasets do not trigger the alarm
    assert "on `synthetic_hetero`" not in text.split("## Summary")[0]


def test_versions_union_flags_differences(run_dir: Path) -> None:
    text = build_report(load_run_dir(run_dir), _opts(notes=("reruns pending",)))
    mapie = next(ln for ln in text.splitlines() if ln.startswith("| `mapie`"))
    assert "**differs**" in mapie and "`929740af0`" in mapie
    crepes = next(ln for ln in text.splitlines() if ln.startswith("| `crepes`"))
    assert "differs" not in crepes
    assert "- reruns pending" in text


def test_preamble_inserted_after_header(run_dir: Path, tmp_path: Path) -> None:
    pre = tmp_path / "pre.md"
    pre.write_text("## Key findings\n\n- something\n", encoding="utf-8")
    out = tmp_path / "r.md"
    assert main(["--run-dir", str(run_dir), "--out", str(out), "--preamble", str(pre)]) == 0
    text = out.read_text()
    assert text.index("## Key findings") < text.index("## Summary")
