#!/usr/bin/env python
"""Per-stage CUDA memory of the conformal_density suite under a GPU-slice cap.

Runs ``ConformalDensitySuite.run`` on one dataset with the process limited to
``--cap-gib`` of the device (CANFAR GPU sessions get ~10.75 GiB H100 slices),
records ``memory_allocated`` / ``max_memory_allocated`` at every suite stage
and after every flow ``sample`` call, prints the first exception's full
traceback (the suite folds exceptions into failed rows), and writes the stage
log plus the rows as JSON::

    python tools/diagnose_conformal_density_memory.py \\
        --dataset synthetic_multitarget_big --cap-gib 10.75 --output mem.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

LOG: list[dict[str, Any]] = []
_T0 = perf_counter()
_FIRST_ERROR: list[str] = []


def _mib(x: int) -> float:
    return round(x / 2**20, 1)


def _mark(stage: str, **extra: Any) -> None:
    rec: dict[str, Any] = {"stage": stage, "t": round(perf_counter() - _T0, 2), **extra}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        rec["alloc_MiB"] = _mib(torch.cuda.memory_allocated())
        rec["peak_MiB"] = _mib(torch.cuda.max_memory_allocated())
        rec["reserved_MiB"] = _mib(torch.cuda.memory_reserved())
    LOG.append(rec)
    print(json.dumps(rec), flush=True)


def _wrap(owner: Any, name: str, label: str, every: int = 0) -> None:
    # Look in the class __dict__: getattr unwraps staticmethods.
    fn = vars(owner)[name] if isinstance(owner, type) else getattr(owner, name)
    raw = fn.__func__ if isinstance(fn, staticmethod) else fn
    calls = {"n": 0}

    @functools.wraps(raw)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        n = calls["n"]
        quiet = every and n % every
        if not quiet:
            _mark(f"{label}:enter", call=n)
        try:
            out = raw(*args, **kwargs)
        except Exception:
            _FIRST_ERROR.append(traceback.format_exc())
            _mark(f"{label}:error", call=n)
            print(_FIRST_ERROR[-1], flush=True)
            raise
        if not quiet:
            _mark(f"{label}:exit", call=n)
        return out

    setattr(owner, name, staticmethod(wrapper) if isinstance(fn, staticmethod) else wrapper)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="synthetic_multitarget_big")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cap-gib", type=float, default=10.75)
    ap.add_argument("--sample-log-every", type=int, default=5)
    ap.add_argument("--output", type=Path, default=Path("conformal_density_memory.json"))
    args = ap.parse_args()

    # As tools/run_all.py: repo root + src importable, NVML denied on CANFAR GPU slices.
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]
    from harness.devices import set_default_device
    from harness.runners import HarnessConfig

    set_default_device("cuda")
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1.0, args.cap_gib * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    print(
        f"device {torch.cuda.get_device_name(device)} total {_mib(total)} MiB -> cap "
        f"{args.cap_gib} GiB (fraction {fraction:.4f})",
        flush=True,
    )

    import suites.tabular.conformal_density as cd
    from torchregress.losses import NormalizingFlowLoss

    for name in ("_streamed_train", "_encode", "_raw_metrics"):
        _wrap(cd, name, name)
    for name in ("_level_set", "_waldo"):
        _wrap(cd.ConformalDensitySuite, name, name)
    _wrap(NormalizingFlowLoss, "sample", "sample", every=args.sample_log_every)

    _mark("start")
    cfg = HarnessConfig(seed=args.seed)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    rows = cd.ConformalDensitySuite().run(args.dataset, cfg)
    _mark("done")
    keep = (
        "Method",
        "Device",
        "PeakGPUMemMB",
        "Coverage90",
        "Width90",
        "NLL",
        "CRPS",
        "train_s",
        "eval_s",
        "Notes",
    )
    summary = [{k: r.get(k) for k in keep} for r in rows]
    for r in summary:
        print(json.dumps(r), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "cap_gib": args.cap_gib,
                "fraction": fraction,
                "stages": LOG,
                "first_error": _FIRST_ERROR[0] if _FIRST_ERROR else None,
                "rows": summary,
            },
            indent=1,
        )
    )
    failed = [r["Method"] for r in summary if str(r.get("Notes", "")).startswith("failed")]
    print(f"rows {len(summary)} failed {len(failed)}: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
