"""Constrained-outputs suite: built-in constraints vs post-hoc fixes and external models.

Benchmarks ``torchregress.constraints`` (``NonNegativeHead``, ``BoundedHead``,
``SimplexHead``, ``NonCrossingSort``, ``SpectralNormWrapper``) on synthetic
designs where the target obeys a known constraint.  For each design the same
MLP backbone is trained unconstrained, with a post-hoc repair (clip,
renormalise, sort after training), and with the torchregress constraint inside
the model; tree models that satisfy the constraint by construction are the
external reference.

Metrics: ``RMSE`` (or pinball loss and 80% coverage for quantiles) and
``ViolationRate`` - the fraction of test predictions that break the
constraint (negative values, outside [0, 1], off the simplex, crossing
quantiles).  Built-in constraints must reach a zero violation rate at no
accuracy cost; post-hoc repairs are expected to reach zero violations too, so
the comparison is about accuracy.

Designs (generated in-suite, synthetic by design):

* ``con_nonneg`` - right-skewed non-negative target with many values near 0.
* ``con_bounded`` - target in [0, 1] with mass near both bounds.
* ``con_simplex`` - 4-part compositions (rows sum to 1).
* ``con_quantiles`` - heteroscedastic target, nine quantile levels.
* ``con_lipschitz`` - smooth target, trained on |x| < 1 and tested on
  1 < |x| < 2: does spectral normalisation tame extrapolation?
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _train_one

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_TRAIN, N_TEST, N_FEATURES = 2000, 1000, 5
HIDDEN = 64


@dataclass(frozen=True)
class Design:
    kind: str  # "nonneg" | "bounded" | "simplex" | "quantiles" | "lipschitz"
    draw: Callable[[np.random.Generator], dict[str, np.ndarray]]


def _features(rng: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
    return rng.uniform(-scale, scale, size=(n, N_FEATURES)).astype(np.float32)


def _draw_nonneg(rng: np.random.Generator) -> dict[str, np.ndarray]:
    x = _features(rng, N_TRAIN + N_TEST)
    mean = np.exp(1.5 * x[:, 0] - 1.0) * (1.0 + 0.5 * x[:, 1] ** 2)
    y = rng.gamma(shape=0.7, scale=mean / 0.7).astype(np.float32)
    return _split(x, y[:, None])


def _draw_bounded(rng: np.random.Generator) -> dict[str, np.ndarray]:
    x = _features(rng, N_TRAIN + N_TEST)
    mu = 1.0 / (1.0 + np.exp(-3.0 * x[:, 0] + 1.5 * x[:, 1]))
    phi = 8.0
    y = rng.beta(mu * phi + 1e-3, (1.0 - mu) * phi + 1e-3).astype(np.float32)
    return _split(x, y[:, None])


def _draw_simplex(rng: np.random.Generator) -> dict[str, np.ndarray]:
    x = _features(rng, N_TRAIN + N_TEST)
    w = np.array([[1.5, -1.0, 0.5, 0.0], [0.0, 1.0, -1.5, 0.5], [0.5, 0.5, 0.0, -1.0]])
    logits = x[:, :3] @ w
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.stack([rng.dirichlet(30.0 * pi + 1e-3) for pi in p]).astype(np.float32)
    return _split(x, y)


def _draw_quantiles(rng: np.random.Generator) -> dict[str, np.ndarray]:
    x = _features(rng, N_TRAIN + N_TEST)
    scale = 0.2 + 1.5 * np.abs(x[:, 1])
    y = (np.sin(2.0 * x[:, 0]) + scale * rng.standard_t(df=4, size=len(x))).astype(np.float32)
    return _split(x, y[:, None])


def _draw_lipschitz(rng: np.random.Generator) -> dict[str, np.ndarray]:
    def target(x: np.ndarray) -> np.ndarray:
        return (np.sin(x[:, 0]) + 0.5 * x[:, 1]).astype(np.float32)

    x_train = _features(rng, N_TRAIN, scale=1.0)
    x_test = _features(rng, N_TEST, scale=1.0)
    x_test[:, :2] = np.sign(x_test[:, :2]) * (1.0 + np.abs(x_test[:, :2]))  # 1 < |x| < 2
    y_train = target(x_train) + 0.1 * rng.normal(size=N_TRAIN).astype(np.float32)
    return {
        "X_train": x_train,
        "y_train": y_train[:, None],
        "X_test": x_test,
        "y_test": target(x_test)[:, None],
    }


def _split(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "X_train": x[:N_TRAIN],
        "y_train": y[:N_TRAIN],
        "X_test": x[N_TRAIN:],
        "y_test": y[N_TRAIN:],
    }


DESIGNS: dict[str, Design] = {
    "con_nonneg": Design("nonneg", _draw_nonneg),
    "con_bounded": Design("bounded", _draw_bounded),
    "con_simplex": Design("simplex", _draw_simplex),
    "con_quantiles": Design("quantiles", _draw_quantiles),
    "con_lipschitz": Design("lipschitz", _draw_lipschitz),
}

METHODS: dict[str, tuple[str, ...]] = {
    "nonneg": ("MLP", "MLP+clip", "NonNegativeHead", "HGB_Poisson"),
    "bounded": ("MLP", "MLP+clip", "BoundedHead", "RandomForest"),
    "simplex": ("MLP", "MLP+renormalize", "SimplexHead", "RandomForest"),
    "quantiles": ("MLP_MQ", "MLP_MQ+sort", "NonCrossingSort", "LightGBM_quantile"),
    "lipschitz": ("MLP", "SpectralNormWrapper"),
}


def violation_rate(kind: str, pred: np.ndarray, tol: float = 1e-6) -> float:
    """Fraction of rows whose prediction breaks the design's constraint."""
    if kind == "nonneg":
        bad = (pred < -tol).any(axis=1)
    elif kind == "bounded":
        bad = ((pred < -tol) | (pred > 1.0 + tol)).any(axis=1)
    elif kind == "simplex":
        bad = (pred < -tol).any(axis=1) | (np.abs(pred.sum(axis=1) - 1.0) > 1e-4)
    elif kind == "quantiles":
        bad = (np.diff(pred, axis=1) < -tol).any(axis=1)
    else:
        return 0.0
    return float(bad.mean())


def pinball(pred: np.ndarray, y: np.ndarray, levels: tuple[float, ...] = QUANTILES) -> float:
    q = np.asarray(levels)[None, :]
    diff = y - pred  # [N, Q] (y broadcast from [N, 1])
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


class _Spectral(nn.Module):
    """The harness MLP with every linear layer spectrally normalised."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        from torchregress.constraints import SpectralNormWrapper

        self.net = nn.Sequential(
            SpectralNormWrapper(nn.Linear(in_dim, hidden)),
            nn.ReLU(),
            SpectralNormWrapper(nn.Linear(hidden, hidden)),
            nn.ReLU(),
            SpectralNormWrapper(nn.Linear(hidden, out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConstrainedOutputsSuite(HarnessRunner):
    """torchregress output constraints vs post-hoc repair and tree baselines."""

    @property
    def name(self) -> str:
        return "constrained_outputs"

    @property
    def description(self) -> str:
        return "Output constraints (non-negative, bounded, simplex, non-crossing, Lipschitz)"

    @property
    def supported_datasets(self) -> list[str]:
        return list(DESIGNS)

    def methods(self) -> list[str]:
        return sorted({m for ms in METHODS.values() for m in ms})

    def library_name(self, method: str) -> str:
        if method == "HGB_Poisson" or method == "RandomForest":
            return "scikit-learn"
        if method == "LightGBM_quantile":
            return "lightgbm"
        return "torchregress"

    def library_version(self, method: str) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(self.library_name(method))
        except PackageNotFoundError:
            return "not installed"

    def method_notes(self, method: str) -> str:
        return {
            "MLP": "unconstrained MLP (MSE)",
            "MLP+clip": "unconstrained MLP, predictions clipped to the feasible set after training",
            "MLP+renormalize": "unconstrained MLP, negatives clipped then rows renormalised",
            "NonNegativeHead": "NonNegativeHead (softplus) inside the model",
            "BoundedHead": "BoundedHead (scaled sigmoid) inside the model",
            "SimplexHead": "SimplexHead (softmax) inside the model",
            "MLP_MQ": "MultiQuantileLoss head, no ordering constraint",
            "MLP_MQ+sort": "MultiQuantileLoss head, quantiles sorted after training",
            "NonCrossingSort": "NonCrossingSort inside the model (sorted during training)",
            "SpectralNormWrapper": "SpectralNormWrapper on every linear layer",
            "HGB_Poisson": "HistGradientBoostingRegressor(loss='poisson'), positive by construction",
            "RandomForest": "RandomForestRegressor: averages of feasible targets stay feasible",
            "LightGBM_quantile": "one LightGBM model per quantile level (can cross)",
        }.get(method, "")

    # ------------------------------------------------------------------
    # train + predict one method on one design
    # ------------------------------------------------------------------

    def _fit_predict(
        self, kind: str, method: str, data: dict[str, np.ndarray], cfg: HarnessConfig
    ) -> np.ndarray:
        X, y, X_test = data["X_train"], data["y_train"], data["X_test"]
        if method in ("HGB_Poisson", "RandomForest", "LightGBM_quantile"):
            return self._fit_predict_external(method, X, y, X_test, cfg)

        from torchregress.constraints import (
            BoundedHead,
            NonCrossingSort,
            NonNegativeHead,
            SimplexHead,
        )
        from torchregress.losses import MultiQuantileLoss, WeightedMSELoss

        device = current_device()
        out_dim = len(QUANTILES) if kind == "quantiles" else y.shape[1]
        if method == "SpectralNormWrapper":
            model: nn.Module = _Spectral(X.shape[1], HIDDEN, out_dim)
        else:
            model = _MLP(X.shape[1], HIDDEN, out_dim)
        if method == "NonNegativeHead":
            model = NonNegativeHead(model)
        elif method == "BoundedHead":
            model = BoundedHead(model, low=0.0, high=1.0)
        elif method == "SimplexHead":
            model = SimplexHead(model)
        elif method == "NonCrossingSort":
            model = nn.Sequential(model, NonCrossingSort())
        model = model.to(device)
        loss: nn.Module = (
            MultiQuantileLoss(quantiles=list(QUANTILES))
            if kind == "quantiles"
            else WeightedMSELoss()
        )
        _train_one(model, loss, torch.from_numpy(X), torch.from_numpy(y), cfg)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X_test).to(device)).detach().cpu().numpy()
        if method == "MLP+clip":
            pred = np.clip(pred, 0.0, 1.0 if kind == "bounded" else None)
        elif method == "MLP+renormalize":
            pred = np.clip(pred, 0.0, None)
            pred = pred / np.clip(pred.sum(axis=1, keepdims=True), 1e-12, None)
        elif method == "MLP_MQ+sort":
            pred = np.sort(pred, axis=1)
        return pred

    @staticmethod
    def _fit_predict_external(
        method: str, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, cfg: HarnessConfig
    ) -> np.ndarray:
        if method == "HGB_Poisson":
            from sklearn.ensemble import HistGradientBoostingRegressor

            m = HistGradientBoostingRegressor(loss="poisson", random_state=cfg.seed)
            return m.fit(X, y[:, 0]).predict(X_test)[:, None]
        if method == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=cfg.seed)
            pred = rf.fit(X, y if y.shape[1] > 1 else y[:, 0]).predict(X_test)
            return pred if pred.ndim == 2 else pred[:, None]
        if method == "LightGBM_quantile":
            import lightgbm as lgb

            cols = [
                lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=cfg.seed, verbose=-1)
                .fit(X, y[:, 0])
                .predict(X_test)
                for q in QUANTILES
            ]
            return np.stack(cols, axis=1)
        raise ValueError(f"Unknown external method {method}")

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        del splits, scenario_name
        cfg = config or HarnessConfig()
        design = DESIGNS.get(dataset_name)
        methods = METHODS[design.kind] if design is not None else tuple(self.methods())
        data = design.draw(np.random.default_rng(cfg.seed)) if design is not None else {}
        rows: list[dict[str, Any]] = []
        for method in methods:
            base = {
                "Method": f"{self.name}/{method}",
                "Library": self.library_name(method),
                "LibraryVersion": self.library_version(method),
                "Dataset": dataset_name,
                "Suite": self.name,
                "SourceURL": "",
                "SyntheticFallback": False,  # synthetic by design
                "Device": str(current_device()),
            }
            if design is None:
                rows.append({**base, "Notes": f"failed: unknown design {dataset_name!r}"})
                continue
            try:
                set_harness_seed(cfg.seed)
                t0 = perf_counter()
                pred = self._fit_predict(design.kind, method, data, cfg)
                elapsed = perf_counter() - t0
                rows.append(
                    {
                        **base,
                        **self._metrics(design.kind, pred, data["y_test"]),
                        "train_s": elapsed,
                        "eval_s": 0.0,
                        "Notes": self.method_notes(method),
                    }
                )
            except Exception as exc:  # noqa: BLE001 — one method failing must not drop the rest
                rows.append({**base, "Notes": f"failed: {type(exc).__name__}: {exc}"})
        return rows

    @staticmethod
    def _metrics(kind: str, pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
        # The Lipschitz design has no output constraint to violate.
        out = {} if kind == "lipschitz" else {"ViolationRate": violation_rate(kind, pred)}
        if kind == "quantiles":
            lo, hi = pred[:, 0], pred[:, -1]
            out["Pinball"] = pinball(pred, y)
            out["Coverage80"] = float(np.mean((lo <= y[:, 0]) & (y[:, 0] <= hi)))
            out["RMSE"] = float(np.sqrt(np.mean((pred[:, len(QUANTILES) // 2] - y[:, 0]) ** 2)))
        else:
            out["RMSE"] = float(np.sqrt(np.mean((pred - y) ** 2)))
        return out

    # HarnessRunner abstract surface (unused: custom run drives everything).
    def fit(self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig) -> Any:
        raise NotImplementedError("constrained_outputs uses custom run()")

    def predict(
        self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("constrained_outputs uses custom run()")

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        raise NotImplementedError("constrained_outputs uses custom run()")
