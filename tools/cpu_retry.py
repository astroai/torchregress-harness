#!/usr/bin/env python
"""Merge a CPU rerun into a GPU job artifact for rows that hit a CUDA OOM.

The CANFAR GPU sessions get an H100 slice of ~11 GiB with NVML queries
denied, and the conda torch build then reports an out-of-memory as
``NVML_SUCCESS == r INTERNAL ASSERT FAILED`` instead of
``OutOfMemoryError`` (release run #6: 30 fm rows and 6 likelihood NSF rows,
all on the largest datasets). ``skaha_release.sh`` reruns such a job with
``--device cpu`` and calls this tool, which replaces only the OOM rows::

    python tools/cpu_retry.py needs GPU.json          # exit 0 if any OOM row
    python tools/cpu_retry.py merge GPU.json CPU.json  # in place into GPU.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OOM_MARKERS = ("NVML_SUCCESS == r INTERNAL ASSERT", "out of memory", "OutOfMemoryError")


def is_oom_row(row: dict) -> bool:
    note = str(row.get("Notes", ""))
    return note.startswith("failed") and any(m in note for m in OOM_MARKERS)


def needs_retry(path: Path) -> bool:
    try:
        rows = json.loads(path.read_text()).get("rows", [])
    except (OSError, ValueError):
        return False
    return any(is_oom_row(r) for r in rows)


def merge(gpu_path: Path, cpu_path: Path) -> int:
    gpu = json.loads(gpu_path.read_text())
    cpu_rows = {r["Method"]: r for r in json.loads(cpu_path.read_text()).get("rows", [])}
    replaced = 0
    for i, row in enumerate(gpu.get("rows", [])):
        if not is_oom_row(row):
            continue
        new = cpu_rows.get(row["Method"])
        if new is None:
            continue
        new = dict(new)
        note = "CUDA out of memory on the GPU slice; this row ran on cpu"
        new["Notes"] = f"{new['Notes']}; {note}" if new.get("Notes") else note
        gpu["rows"][i] = new
        replaced += 1
    gpu.setdefault("notes", []).append(f"cpu_retry: {replaced} CUDA-OOM row(s) rerun on cpu")
    gpu_path.write_text(json.dumps(gpu, indent=2) + "\n")
    return replaced


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "needs":
        return 0 if needs_retry(Path(argv[1])) else 1
    if len(argv) == 3 and argv[0] == "merge":
        print(f"cpu_retry: replaced {merge(Path(argv[1]), Path(argv[2]))} row(s)")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
