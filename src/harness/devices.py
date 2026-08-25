"""Device resolution — single source of truth for harness benchmarks.

Suites should rarely hardcode ``"cpu"``.  Resolve once per process via
:func:`set_default_device` (called by ``tools/run_all.py`` /
``tools/leaderboard_regression.py`` from ``--device``), then either

* construct models normally — ``suites._backbones`` places new modules on the
  default device automatically and ``_train_one`` follows the model's device,
  or
* call :func:`current_device` at sites that need an explicit device string
  (external adapters such as TabICL / torchcp / TabPFN).
"""

from __future__ import annotations

import torch

_DEFAULT: torch.device = torch.device("cpu")


def resolve_device(device: str | None) -> torch.device:
    """Resolve a device request with graceful availability fallback.

    ``None``/``""``/``"auto"`` pick cuda > mps > cpu.  A requested backend
    that is not available falls back to cpu instead of raising, so a
    benchmark never dies on device selection.
    """
    if device is None or device == "" or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    d = torch.device(device)
    if d.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if d.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            return torch.device("cpu")
    return d


def set_default_device(device: str | torch.device | None) -> torch.device:
    """Set the process-wide default device; returns the resolved device."""
    global _DEFAULT
    _DEFAULT = resolve_device(str(device) if device else None)
    return _DEFAULT


def current_device() -> torch.device:
    """The process-wide default device (cpu unless overridden)."""
    return _DEFAULT
