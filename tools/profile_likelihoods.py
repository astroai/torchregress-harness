"""Per-row wall-clock profile of the ``likelihoods`` suite on one dataset.

Runs each method's ``fit`` / ``predict`` / test NLL / ``evaluate`` separately
(the same calls ``LikelihoodsSuite.run`` makes) and prints the wall time and
peak device memory of every phase, with a heartbeat while a phase runs, so a
job that would otherwise print only at the end can be attributed row by row.

    pixi run -e gpu python tools/profile_likelihoods.py --dataset tlc_fare \
        --epochs 2 --mem-gib 10.75 [--methods GaussianNLL,NSF]

``--epochs`` lower than the release budget gives the per-epoch fit cost
(fit time is linear in epochs for every torch row; prediction and scoring do
not depend on it).  ``--mem-gib`` caps the CUDA caching allocator to mimic a
MIG slice of that size on a full GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from harness.datasets import load_dataset  # noqa: E402
from harness.devices import set_default_device  # noqa: E402
from harness.runners import HarnessConfig  # noqa: E402
from suites.tabular.likelihoods import LikelihoodsSuite  # noqa: E402


class _Heartbeat:
    def __init__(self, every: float) -> None:
        self.every, self.label, self.t0 = every, "", time.perf_counter()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def phase(self, label: str) -> None:
        self.label, self.t0 = label, time.perf_counter()

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            if self.label:
                print(f"  ... {self.label}: {time.perf_counter() - self.t0:.0f}s", flush=True)


def _peak_gib() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", default="")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mem-gib", type=float, default=0.0)
    ap.add_argument("--heartbeat", type=float, default=60.0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    dev = set_default_device(args.device)
    info = {"device": str(dev), "cpus": os.cpu_count(), "torch": torch.__version__}
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info |= {"gpu": props.name, "gpu_mem_gib": round(props.total_memory / 2**30, 2)}
        if args.mem_gib > 0:
            frac = min(1.0, args.mem_gib * 2**30 / props.total_memory)
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            info["mem_cap_gib"] = args.mem_gib
    print("ENV", json.dumps(info), flush=True)

    hb = _Heartbeat(args.heartbeat)
    t0 = time.perf_counter()
    splits = load_dataset(args.dataset, standardize_targets=True)
    print(
        f"LOAD {args.dataset}: {time.perf_counter() - t0:.1f}s "
        f"train={np.shape(splits['X_train'])} test={np.shape(splits['X_test'])}",
        flush=True,
    )
    suite = LikelihoodsSuite()
    cfg = HarnessConfig(seed=args.seed, epochs=args.epochs, device=str(dev))
    methods = [m for m in args.methods.split(",") if m] or suite.methods()
    X_test = np.asarray(splits["X_test"])
    y_test = np.asarray(splits["y_test"])
    results = []
    for method in methods:
        rec: dict[str, object] = {"method": method, "epochs": args.epochs}
        try:
            for phase in ("fit", "predict", "nll", "evaluate"):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                hb.phase(f"{method}/{phase}")
                t = time.perf_counter()
                if phase == "fit":
                    state = suite.fit(method, splits, cfg)
                elif phase == "predict":
                    pred = suite.predict(method, state, X_test, cfg)
                elif phase == "nll":
                    pred["_nll"] = np.full(1, suite._test_nll(method, state, pred, X_test, y_test))
                else:
                    metrics = suite.evaluate(method, pred, y_test, cfg)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                rec[f"{phase}_s"] = round(time.perf_counter() - t, 2)
                rec[f"{phase}_peak_gib"] = round(_peak_gib(), 3)
                print(f"  {method} {phase}: {rec[f'{phase}_s']}s peak={rec[f'{phase}_peak_gib']}GiB", flush=True)
            rec["metrics"] = {k: metrics.get(k) for k in ("NLL", "CRPS", "Coverage", "RMSE")}
            ft = state.get("ft") if isinstance(state, dict) else None
            if ft is not None:
                rec["ft"] = {
                    "epochs_run": ft.epochs_run,
                    "epochs_budget": ft.epochs_budget,
                    "capped": ft.capped_by_token_budget,
                    "n_tokens": ft.n_tokens,
                }
        except Exception as exc:  # noqa: BLE001 - report and keep profiling
            rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
        hb.phase("")
        state = pred = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("ROW", json.dumps(rec), flush=True)
        results.append(rec)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"env": info, "dataset": args.dataset, "rows": results}, indent=1))


if __name__ == "__main__":
    main()
