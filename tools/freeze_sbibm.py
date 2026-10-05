#!/usr/bin/env python
"""Freeze the sbibm task draws used by ``sbi_parity`` into the harness cache.

Run once where sbibm imports (``pip install --no-deps sbibm==1.1.0``), then copy
``$TORCHREGRESS_HARNESS_CACHE/sbibm/`` to offline nodes.  Later runs read the
frozen copies and never import sbibm.

    python tools/freeze_sbibm.py            # all registered sbibm tasks
    python tools/freeze_sbibm.py sbibm_slcp
"""

from __future__ import annotations

import argparse
import sys

from harness.datasets import DATASET_REGISTRY


def main(argv: list[str] | None = None) -> int:
    tasks = sorted(n for n in DATASET_REGISTRY if n.startswith("sbibm_"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*", default=tasks, help=f"default: {' '.join(tasks)}")
    args = parser.parse_args(argv)

    failed = []
    for name in args.tasks:
        splits = DATASET_REGISTRY[name].loader()
        if splits.get("_synthetic_fallback"):
            failed.append(name)
            print(f"FAIL {name}: {splits.get('_fallback_reason')}", file=sys.stderr)
        else:
            print(f"OK   {name}: {splits.get('_source_url')}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
