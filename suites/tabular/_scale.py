"""Large-data helpers shared by the multi-target density and conformal suites.

Training data stay on the host: every mini-batch is gathered on the CPU and
copied to :func:`harness.devices.current_device` on its own, so no suite
materializes the full dataset (or any ``N x N`` tensor) on the GPU.  Rows on
>= ``BIG_ROWS`` training rows switch to the large-batch / bounded-epoch
schedule below and report peak CUDA memory, so CPU and GPU scaling runs are
directly comparable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import torch

from harness.devices import current_device

# Training-set size from which a run counts as "big data".
BIG_ROWS = 200_000
# Large-data schedule: mini-batch rows, epoch cap and early-stopping patience.
BIG_BATCH = 2048
BIG_MAX_EPOCHS = 20
BIG_PATIENCE = 3
# Rows per forward pass when scoring (validation / calibration / test).
EVAL_CHUNK = 8192


def is_big(n_rows: int) -> bool:
    """True when a training split of ``n_rows`` uses the large-data schedule."""
    return int(n_rows) >= BIG_ROWS


def to_device(array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Copy one host array (a batch or a chunk, never a full big split) to the device."""
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype).to(current_device())


def iter_chunks(n_rows: int, chunk: int = EVAL_CHUNK) -> Iterator[slice]:
    """Contiguous row slices of at most ``chunk`` rows."""
    for start in range(0, int(n_rows), int(chunk)):
        yield slice(start, min(start + int(chunk), int(n_rows)))


@torch.no_grad()
def chunked_rows(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    X: np.ndarray,
    Y: np.ndarray,
    chunk: int = EVAL_CHUNK,
) -> np.ndarray:
    """Evaluate a per-row function ``fn(x_batch, y_batch) -> [B]`` in device-sized chunks."""
    out = [
        fn(to_device(X[s]), to_device(Y[s])).reshape(-1).cpu().numpy()
        for s in iter_chunks(len(X), chunk)
    ]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def reset_peak_memory() -> None:
    """Start a fresh peak-memory window on CUDA (no-op on other devices)."""
    dev = current_device()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)


def peak_memory_mb() -> float | None:
    """Peak CUDA memory allocated since :func:`reset_peak_memory`, in MiB (None off CUDA)."""
    dev = current_device()
    if dev.type != "cuda":
        return None
    torch.cuda.synchronize(dev)
    return float(torch.cuda.max_memory_allocated(dev)) / 2**20
