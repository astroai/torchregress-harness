"""Experiment config validation for suites/experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str
    type: Literal["tabular", "image"]
    n_cap: int = 5000
    target_dim: int = 2
    rho: float = 0.6


@dataclass
class MethodConfig:
    loss: str
    loss_kwargs: dict = field(default_factory=dict)
    backbone: str = "mlp"


@dataclass
class ExperimentConfig:
    datasets: List[DatasetConfig]
    methods: List[MethodConfig]
    seeds: List[int] = field(default_factory=lambda: [42])
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    device: str = "auto"
    n_grid: int = 200

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text())
        datasets = [DatasetConfig(**d) for d in data.get("datasets", [])]
        methods = [MethodConfig(**m) for m in data.get("methods", [])]
        return cls(
            datasets=datasets,
            methods=methods,
            seeds=data.get("seeds", [42]),
            epochs=int(data.get("epochs", 20)),
            batch_size=int(data.get("batch_size", 256)),
            lr=float(data.get("lr", 1e-3)),
            device=str(data.get("device", "auto")),
            n_grid=int(data.get("n_grid", 200)),
        )
