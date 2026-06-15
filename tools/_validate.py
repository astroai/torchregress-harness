"""Shared validation helpers for pinpointing torchregress/harness issues.

Used by ``run_all.py --validate`` and ``run_canonical.py --validate``
to check core imports, key losses, calibrators, and suite instantiation
before running benchmarks.
"""

from __future__ import annotations


def validate_torchregress() -> tuple[bool, list[str]]:
    """Run basic sanity checks on torchregress to pinpoint import/API issues.

    Returns (ok, messages) where ok is True if all checks pass.
    """
    ok = True
    msgs: list[str] = []

    # 1. Core import
    try:
        import torchregress  # noqa: F401
        msgs.append("  ✓ torchregress imported")
    except ImportError as exc:
        ok = False
        msgs.append(f"  ✗ torchregress import failed: {exc}")
        return ok, msgs  # can't continue without core

    # 2. Key losses
    loss_checks = [
        ("GaussianNLLLoss", "torchregress.losses"),
        ("MixtureDensityLoss", "torchregress.losses"),
        ("MultiQuantileLoss", "torchregress.losses"),
        ("TweedieLoss", "torchregress.losses"),
        ("ConformalLoss", "torchregress.losses.conformal"),
        ("BetaNLLLoss", "torchregress.losses"),
    ]
    for name, module_path in loss_checks:
        try:
            mod = __import__(module_path, fromlist=[name])
            getattr(mod, name)
            msgs.append(f"  ✓ {module_path}.{name}")
        except Exception as exc:
            msgs.append(f"  ✗ {module_path}.{name}: {type(exc).__name__}: {exc}")

    # 3. Calibration
    cal_checks = [
        ("VarianceTemperatureScaler", "torchregress.calibration"),
        ("IsotonicMeanCalibrator", "torchregress.calibration"),
    ]
    for name, module_path in cal_checks:
        try:
            mod = __import__(module_path, fromlist=[name])
            getattr(mod, name)
            msgs.append(f"  ✓ {module_path}.{name}")
        except Exception as exc:
            msgs.append(f"  ✗ {module_path}.{name}: {type(exc).__name__}: {exc}")

    # 4. Core metrics
    try:
        import numpy as np

        from harness.metrics import compute_all_metrics

        result = compute_all_metrics(
            {"y_pred": np.array([1.0, 2.0])},
            np.array([1.1, 1.9]),
        )
        msgs.append(f"  ✓ compute_all_metrics smoke: {sorted(result.keys())}")
    except Exception as exc:
        msgs.append(f"  ✗ compute_all_metrics: {type(exc).__name__}: {exc}")

    return ok, msgs


def validate_suites(all_suites: dict) -> tuple[bool, list[str]]:
    """Check that all registered suites can be instantiated and queried.

    Args:
        all_suites: Dict mapping suite name → suite class (from ALL_SUITES).

    Returns (ok, messages) where ok is True if all checks pass.
    """
    ok = True
    msgs: list[str] = []

    for name, cls in all_suites.items():
        try:
            suite = cls()
            methods = suite.methods()
            datasets = suite.supported_datasets
            msgs.append(
                f"  ✓ {name}: {len(methods)} methods, "
                f"{len(datasets)} datasets"
            )
        except Exception as exc:
            ok = False
            msgs.append(f"  ✗ {name}: {type(exc).__name__}: {exc}")

    return ok, msgs
