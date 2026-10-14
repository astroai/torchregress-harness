"""tools/cpu_retry.py: only CUDA-OOM rows are replaced by the cpu rerun."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import cpu_retry  # noqa: E402

OOM = 'failed: RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at "x"'


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"rows": rows}))
    return path


def test_merge_replaces_only_oom_rows(tmp_path: Path) -> None:
    gpu = _write(
        tmp_path / "gpu.json",
        [
            {"Method": "s/A", "RMSE": 1.0, "Notes": ""},
            {"Method": "s/B", "RMSE": None, "Notes": OOM},
            {"Method": "s/C", "RMSE": None, "Notes": "failed: ValueError: other"},
        ],
    )
    cpu = _write(
        tmp_path / "cpu.json",
        [
            {"Method": "s/A", "RMSE": 9.0, "Notes": ""},
            {"Method": "s/B", "RMSE": 2.0, "Notes": "n"},
            {"Method": "s/C", "RMSE": 3.0, "Notes": ""},
        ],
    )
    assert cpu_retry.needs_retry(gpu)
    assert cpu_retry.merge(gpu, cpu) == 1
    rows = {r["Method"]: r for r in json.loads(gpu.read_text())["rows"]}
    assert rows["s/A"]["RMSE"] == 1.0
    assert rows["s/B"]["RMSE"] == 2.0 and "ran on cpu" in rows["s/B"]["Notes"]
    assert rows["s/C"]["RMSE"] is None
    assert not cpu_retry.needs_retry(gpu)


def test_needs_retry_false_without_oom(tmp_path: Path) -> None:
    gpu = _write(tmp_path / "g.json", [{"Method": "s/A", "Notes": "failed: ValueError: x"}])
    assert not cpu_retry.needs_retry(gpu)
    assert not cpu_retry.needs_retry(tmp_path / "missing.json")
