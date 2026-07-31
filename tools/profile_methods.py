#!/usr/bin/env python
"""Performance and memory profiler for torchregress wrappers.

Measures fit time, prediction latency, sampling latency, and peak CPU/device memory usage.
Supports outputting profile results as JSON and comparing them against a baseline profile.
"""

from __future__ import annotations

import argparse
import datetime
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.wrappers import (
    DeepEnsembleRegressor,
    EvidentialRegressor,
    GaussianNLLRegressor,
    HeteroscedasticEnsembleRegressor,
    MDNRegressor,
    NormalizingFlowRegressor,
)


def get_peak_cpu_memory_mb() -> float:
    """Return peak CPU Resident Set Size (RSS) in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    maxrss = usage.ru_maxrss
    if sys.platform == "darwin":
        # macOS returns maxrss in bytes
        return maxrss / (1024.0 * 1024.0)
    else:
        # Linux returns maxrss in kilobytes
        return maxrss / 1024.0


def profile_wrapper(
    wrapper_cls: type,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Profile training and inference operations for a specific regressor wrapper."""
    # Reset seeds for consistency
    torch.manual_seed(0)
    np.random.seed(0)

    # Initialize model
    if wrapper_cls == MDNRegressor:
        reg = wrapper_cls(n_components=3, epochs=20)
    else:
        reg = wrapper_cls(epochs=20)
    reg = reg.to(device)

    results = {}

    # CPU/Device memory baseline
    cpu_start = get_peak_cpu_memory_mb()

    device_start = 0.0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        device_start = torch.cuda.memory_allocated(device)
    elif device.type == "mps":
        torch.mps.empty_cache()
        device_start = torch.mps.current_allocated_memory()

    # --- FIT PHASE ---
    t0 = time.perf_counter()
    reg.fit(X, y)
    t1 = time.perf_counter()
    results["fit_time_s"] = t1 - t0

    # --- PREDICT PHASE ---
    t0 = time.perf_counter()
    reg.predict(X)
    t1 = time.perf_counter()
    results["predict_time_s"] = t1 - t0

    # --- PREDICT INTERVAL PHASE ---
    t0 = time.perf_counter()
    reg.predict_interval(X, confidence=0.95)
    t1 = time.perf_counter()
    results["predict_interval_time_s"] = t1 - t0

    # --- SAMPLE PHASE ---
    t0 = time.perf_counter()
    reg.sample(X, n_samples=100)
    t1 = time.perf_counter()
    results["sample_time_s"] = t1 - t0

    # CPU/Device memory final peak difference
    cpu_end = get_peak_cpu_memory_mb()
    results["peak_cpu_memory_mb"] = max(0.0, cpu_end - cpu_start)

    device_peak = 0.0
    if device.type == "cuda":
        device_peak = torch.cuda.max_memory_allocated(device) - device_start
    elif device.type == "mps":
        device_peak = torch.mps.current_allocated_memory() - device_start
    results["peak_device_memory_mb"] = max(0.0, device_peak / (1024.0 * 1024.0))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile torchregress wrapper performance.")
    parser.add_argument("--output", type=Path, help="Path to write the profile JSON output")
    parser.add_argument(
        "--compare", type=Path, help="Path to reference baseline profile JSON for regression check"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Regression threshold ratio (default: 0.15 for 15%)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if regression exceeds threshold",
    )
    args = parser.parse_args()

    # Set active device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Starting profiling run on device: {device}")

    # Generate synthetic tabular dataset
    np.random.seed(0)
    X = np.random.randn(1000, 10).astype(np.float32)
    y = (X[:, 0] * 2.0 + X[:, 1] * 0.5 + np.random.randn(1000) * 0.1).astype(np.float32)

    wrappers = {
        "GaussianNLLRegressor": GaussianNLLRegressor,
        "MDNRegressor": MDNRegressor,
        "EvidentialRegressor": EvidentialRegressor,
        "NormalizingFlowRegressor": NormalizingFlowRegressor,
        "DeepEnsembleRegressor": DeepEnsembleRegressor,
        "HeteroscedasticEnsembleRegressor": HeteroscedasticEnsembleRegressor,
    }

    current_profile: dict[str, Any] = {
        "device": device.type,
        "system": sys.platform,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "profiles": {},
    }

    # Profile each wrapper
    for name, wrapper_cls in wrappers.items():
        print(f"Profiling {name}...")
        try:
            current_profile["profiles"][name] = profile_wrapper(wrapper_cls, X, y, device)
            p = current_profile["profiles"][name]
            print(
                f"  Fit: {p['fit_time_s']:.3f}s | Predict: {p['predict_time_s']:.3f}s | "
                f"Sample: {p['sample_time_s']:.3f}s | Peak Device Mem: {p['peak_device_memory_mb']:.1f}MB"
            )
        except Exception as e:
            print(f"  Failed to profile {name}: {e}")

    # Write output if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(current_profile, f, indent=2)
        print(f"Saved current profile to {args.output}")

    # Compare with reference baseline if provided
    if args.compare:
        if not args.compare.exists():
            print(f"Error: reference baseline file {args.compare} does not exist.")
            sys.exit(1)

        with open(args.compare) as f:
            baseline = json.load(f)

        print()
        print(f"Comparing current profile vs baseline: {args.compare}")
        print("-" * 80)

        regression_detected = False
        threshold = args.threshold
        metrics_to_check = [
            "fit_time_s",
            "predict_time_s",
            "predict_interval_time_s",
            "sample_time_s",
        ]

        for name, current_p in current_profile["profiles"].items():
            if name not in baseline["profiles"]:
                print(f"  Method {name} not found in baseline profile. Skipping check.")
                continue

            baseline_p = baseline["profiles"][name]
            for metric in metrics_to_check:
                curr_val = current_p.get(metric, 0.0)
                base_val = baseline_p.get(metric, 0.0)

                if base_val > 0.0:
                    ratio = (curr_val - base_val) / base_val
                    if ratio > threshold:
                        print(
                            f"  [REGRESSION] {name}.{metric}: {base_val:.4f}s -> {curr_val:.4f}s (+{ratio:.1%}, threshold is +{threshold:.1%})"
                        )
                        regression_detected = True
                    else:
                        print(
                            f"  [OK] {name}.{metric}: {base_val:.4f}s -> {curr_val:.4f}s ({ratio:+.1%})"
                        )

        print("-" * 80)
        if regression_detected:
            if args.fail_on_regression:
                print("Error: performance regressions detected. Exiting with status 1.")
                sys.exit(1)
            else:
                print("Warning: performance regressions detected.")
        else:
            print("Performance check passed. No regressions detected.")


if __name__ == "__main__":
    main()
