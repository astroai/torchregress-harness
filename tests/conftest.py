"""Pytest configuration for torchregress-harness.

Adds the repo root to sys.path so that `suites` and `harness` packages
are importable from any working directory.
"""
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
