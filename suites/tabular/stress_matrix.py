"""Stress-matrix suite: torchregress methods under data stress conditions.

The point is **diversity of use cases**, not leaderboard winning: every row
answers "does torchregress have a first-class tool for this stress axis, and
does it hold up against an external comparator on standard metrics?"

Stress axes (one synthetic dataset each, seeded, deterministic):
* ``stress_imbalanced``     — long-tailed target density (exponential law).
* ``stress_noisy_inputs``   — additive observation noise on train features
  (errors-in-variables); test features are clean.
* ``stress_noisy_labels``   — heteroscedastic Gaussian label noise with
  per-sample known variance.
* ``stress_sparse_labels``  — only a fraction of train targets observed;
  pseudo-targets with known variance cover the rest.
* ``stress_ood``            — covariate shift (train/test marginals split
  along x0 rather than randomly).
* ``stress_semi_sup``       — small labeled pool + large unlabeled pool
  (the SAGE regime).

Methods are grouped per axis; comparators use the harness's existing
backbones and (when importable) Lightning-UQ-Box:
* imbalanced:    DensityWeightedLoss / LDSLoss / BinReweightedMSELoss vs MSE.
* noisy_inputs:  InputNoiseAugmentationLoss vs plain MSE.
* noisy_labels:  NoisyTargetGaussianNLL / BetaNLL vs GaussianNLL.
* sparse_labels: PseudoLabelNLL vs GaussianNLL (labeled-only).
* ood:           GaussianNLL vs MC_Dropout vs DeepEnsemble vs LUQ-box rows.
* semi_sup:      SAGE (SAGERegLoss + EMA teacher) vs supervised-only.

Lightning-UQ-Box rows (``luqbox_*``) degrade to skip rows when the package
is not importable (matching how MAPIE/torch-uncertainty rows behave offline).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchregress.losses import (
    BetaNLLLoss,
    BinReweightedMSELoss,
    DensityWeightedLoss,
    GaussianNLLLoss,
    InputNoiseAugmentationLoss,
    LDSLoss,
    NoisyTargetGaussianNLL,
    PseudoLabelNLL,
    WeightedMSELoss,
)
from torchregress.semi_supervised import (
    PredictiveBatch,
    SAGERegLoss,
)
from torchregress.utils import update_ema_teacher_

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import (
    _MLP,
    _gaussian_nll_predict,
    _hidden_size,
    _mixture_gaussian_stats,
    _train_one,
)

# ---------------------------------------------------------------------------
# Optional deps (never fatal — skip rows instead)
# ---------------------------------------------------------------------------

try:  # Lightning-UQ-Box
    import lightning_uq_box  # noqa: F401

    _LUQBOX_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _LUQBOX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Stress dataset generators (all seeded, deterministic)
# ---------------------------------------------------------------------------

_STRESS_SEED = 42
_D = 12  # features per stress dataset


def _base_function(X: np.ndarray) -> np.ndarray:
    """Shared non-linear ground truth so stress axes are comparable."""
    z = X[:, 0] + 0.5 * np.sin(3.0 * X[:, 1]) + 0.3 * X[:, 2] ** 2
    return z.astype(np.float64)


def _make_imbalanced(n: int, seed: int) -> dict[str, np.ndarray]:
    """Long-tailed target density: heavy right tail, sparse upper head."""
    rng = np.random.default_rng(seed)
    n_train = int(n * 0.8)
    X = rng.normal(size=(n, _D))
    y = _base_function(X) + 0.5 * rng.exponential(scale=1.0, size=n) + 0.05 * rng.normal(size=n)
    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "y_test": y[n_train:],
    }


def _make_noisy_inputs(n: int, seed: int, sigma_x: float = 0.5) -> dict[str, np.ndarray]:
    """Observation noise on train inputs; test inputs are clean (EIV setting)."""
    rng = np.random.default_rng(seed)
    n_train = int(n * 0.8)
    X = rng.normal(size=(n, _D))
    y = _base_function(X) + 0.05 * rng.normal(size=n)
    X_obs = X.copy()
    X_obs[:n_train] += rng.normal(scale=sigma_x, size=(n_train, _D))
    return {
        "X_train": X_obs[:n_train],  # noisy
        "y_train": y[:n_train],
        "X_test": X[n_train:],  # clean
        "y_test": y[n_train:],
        "sigma_x": np.float64(sigma_x),
    }


def _make_noisy_labels(n: int, seed: int) -> dict[str, np.ndarray]:
    """Heteroscedastic label noise with known per-sample variance."""
    rng = np.random.default_rng(seed)
    n_train = int(n * 0.8)
    X = rng.normal(size=(n, _D))
    clean = _base_function(X)
    var = 0.1 + 0.4 * np.abs(X[:, 0])  # known noise variance per sample
    y = clean + rng.normal(size=n) * np.sqrt(var)
    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "y_test": y[n_train:],
        "target_variance_train": var[:n_train].astype(np.float32),
    }


def _make_sparse_labels(n: int, seed: int, labeled_frac: float = 0.2) -> dict[str, np.ndarray]:
    """Sparse supervision: a fraction of train targets observed.

    Exposes a labeled mask plus (deliberately imperfect) pseudo-targets with
    known variance so PseudoLabelNLL can consume variance-weighted supervision.
    """
    rng = np.random.default_rng(seed)
    n_train = int(n * 0.8)
    X = rng.normal(size=(n, _D))
    y = _base_function(X) + 0.05 * rng.normal(size=n)
    labeled = rng.random(n_train) < labeled_frac
    pseudo = y + rng.normal(scale=0.5, size=n)  # overconfident proxy of the truth
    pseudo_var = np.full(n_train, 0.25, dtype=np.float32)
    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "y_test": y[n_train:],
        "labeled_mask_train": labeled.astype(np.float32),
        "pseudo_target_train": pseudo.astype(np.float32),
        "pseudo_variance_train": pseudo_var,
    }


def _make_ood(n: int, seed: int) -> dict[str, np.ndarray]:
    """Covariate shift: train/test split by x0 sign, not by random sampling."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, _D))
    y = _base_function(X) + 0.05 * rng.normal(size=n)
    mask = X[:, 0] >= 0  # genuine covariate shift, not a random split
    return {
        "X_train": X[~mask],
        "y_train": y[~mask],
        "X_test": X[mask],
        "y_test": y[mask],
    }


def _make_semi_sup(n: int, seed: int, n_labeled: int = 200) -> dict[str, np.ndarray]:
    """Small labeled pool + large unlabeled pool (SAGE's regime)."""
    rng = np.random.default_rng(seed)
    n_train = int(n * 0.8)
    X = rng.normal(size=(n, _D))
    y = _base_function(X) + 0.05 * rng.normal(size=n)
    labeled = np.zeros(n_train, dtype=bool)
    labeled[rng.choice(n_train, size=min(n_labeled, n_train), replace=False)] = True
    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "y_test": y[n_train:],
        "labeled_mask_train": labeled.astype(np.float32),
    }


_STRESS_GENERATORS: dict[str, Any] = {
    "stress_imbalanced": _make_imbalanced,
    "stress_noisy_inputs": _make_noisy_inputs,
    "stress_noisy_labels": _make_noisy_labels,
    "stress_sparse_labels": _make_sparse_labels,
    "stress_ood": _make_ood,
    "stress_semi_sup": _make_semi_sup,
}


def load_stress_dataset(name: str, n: int = 4000, seed: int = _STRESS_SEED) -> dict[str, Any]:
    """Load a synthetic stress dataset by registry name (deterministic)."""
    if name not in _STRESS_GENERATORS:
        raise KeyError(f"Unknown stress dataset: {name!r}. Available: {sorted(_STRESS_GENERATORS)}")
    data = _STRESS_GENERATORS[name](n, seed)
    return {"_source_url": "", "_synthetic_fallback": True, **data}


# ---------------------------------------------------------------------------
# Local training helper (losses whose forward needs per-sample extras)
# ---------------------------------------------------------------------------


def _train_special(
    model: nn.Module,
    loss_fn: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    config: HarnessConfig,
    *,
    extra: dict[str, torch.Tensor] | None = None,
    epochs: int | None = None,
) -> None:
    """Mini-batch Adam loop for losses whose forward takes per-sample extras.

    ``extra`` values are indexed per batch (first dim = batch) and forwarded
    as keyword arguments to ``loss_fn(pred, y, **kwargs)``.
    """
    epochs = epochs if epochs is not None else (2 if config.smoke else 30)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    n = X.shape[0]
    bs = min(config.batch_size, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=X.device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            pred = model(X[idx])
            kwargs = {k: v[idx] for k, v in (extra or {}).items()}
            loss = loss_fn(pred, y[idx], **kwargs)
            opt.zero_grad()
            loss.backward()
            opt.step()


# ---------------------------------------------------------------------------
# Lightning-UQ-Box comparator (self-contained; any failure -> ImportError)
# ---------------------------------------------------------------------------

_LUQBOX_PIN = "v0.3.0 (git df6366e02055618873f2cc8711b2dbc22e131a12)"


def _luqbox_run(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    epochs: int,
    seed: int,
    smoke: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Lightning-UQ-Box regression method, return (mean, std) on X_test.

    Raises ImportError for *any* environment/signature problem so the base
    loop records a skip row instead of failing the suite.
    """
    try:
        import lightning
        from lightning_uq_box.models import MLP as LUQMLP
        from lightning_uq_box.uq_methods import DeterministicRegression, MCDropoutRegression
        from lightning_uq_box.uq_methods.loss_functions import NLL as LUQNLL

        set_harness_seed(seed)
        in_dim = X_train.shape[1]
        hidden = _hidden_size(in_dim)

        from torch.utils.data import DataLoader, TensorDataset

        def _loader(X: torch.Tensor, y: torch.Tensor, shuffle: bool) -> DataLoader:
            # LUQ-box batches are dicts with input/target keys (class defaults).
            ds = TensorDataset(X, y)

            def _collate(
                samples: list[tuple[torch.Tensor, torch.Tensor]],
            ) -> dict[str, torch.Tensor]:
                xs = torch.stack([s[0] for s in samples])
                ys = torch.stack([s[1] for s in samples])
                return {"input": xs, "target": ys}

            return DataLoader(ds, batch_size=64, shuffle=shuffle, collate_fn=_collate)

        def _trainer() -> Any:
            return lightning.Trainer(
                max_epochs=epochs,
                accelerator="cpu",
                devices=1,
                enable_progress_bar=False,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
            )

        X_tr = torch.from_numpy(X_train).float()
        y_tr = torch.from_numpy(y_train).float().unsqueeze(1)  # (N, 1) target
        X_te = torch.from_numpy(X_test).float()
        n_mc = 5 if smoke else 20

        if method == "luqbox_MCDropout":
            # LUQ-box v0.3.0 MLP signature: n_hidden (not hidden_dims);
            # activation_fn takes an nn.Module instance.
            base = LUQMLP(
                n_inputs=in_dim, n_outputs=2, n_hidden=[hidden, hidden], activation_fn=nn.ReLU()
            )
            mod = MCDropoutRegression(model=base, num_mc_samples=n_mc, loss_fn=LUQNLL())
            _trainer().fit(mod, train_dataloaders=_loader(X_tr, y_tr, True))
            # MC-dropout predict: dropout active (train mode), no_grad sampling
            # over stacked forwards — same semantics as predict_step.
            base.train()
            with torch.no_grad():
                preds = torch.stack([base(X_te) for _ in range(n_mc)], dim=-1)  # (N, 2, S)
            mean = preds[:, 0, :].mean(dim=-1)
            std = preds[:, 0, :].std(dim=-1, unbiased=False)
            return (
                mean.numpy().reshape(-1),
                np.clip(std.numpy().reshape(-1), 1e-8, None),
            )
        elif method == "luqbox_GaussianNLL":
            base = LUQMLP(
                n_inputs=in_dim, n_outputs=2, n_hidden=[hidden, hidden], activation_fn=nn.ReLU()
            )
            mod = DeterministicRegression(model=base, loss_fn=LUQNLL())
            _trainer().fit(mod, train_dataloaders=_loader(X_tr, y_tr, True))
            base.eval()
            with torch.no_grad():
                out = base(X_te)
            mean = out[:, 0].numpy().reshape(-1)
            std = torch.exp(0.5 * out[:, 1]).numpy().reshape(-1)  # log-var -> std
            return mean, np.clip(std, 1e-8, None)
        else:
            raise ImportError(f"Unknown luqbox method {method!r}")
    except ImportError:
        raise
    except Exception as exc:  # signature drift, lightning version mismatch, ...
        raise ImportError(f"lightning-uq-box comparator unavailable: {exc}") from exc


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class StressMatrixSuite(HarnessRunner):
    """Diversity-of-use-cases matrix: stress axes x methods x metrics.

    torchregress provides a first-class method for every axis; external
    comparators (Lightning-UQ-Box, when installed) provide reference rows.
    """

    _N = 4000  # samples per stress dataset

    @property
    def name(self) -> str:
        return "stress_matrix"

    @property
    def description(self) -> str:
        return (
            "Diversity matrix: imbalanced / noisy-input / noisy-label / sparse-label "
            "/ OOD / semi-supervised axes; torchregress methods vs Lightning-UQ-Box"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return list(_STRESS_GENERATORS)

    @property
    def canonical_scenarios(self) -> list[str]:
        return []

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    _METHODS: list[str] = [
        # --- imbalanced ---
        "stress_imbalanced/MSE_MLP",
        "stress_imbalanced/DensityWeighted_MLP",
        "stress_imbalanced/LDS_MLP",
        "stress_imbalanced/BinReweighted_MLP",
        # --- noisy inputs (EIV) ---
        "stress_noisy_inputs/MSE_MLP",
        "stress_noisy_inputs/InputNoise_MLP",
        # --- noisy labels ---
        "stress_noisy_labels/GaussianNLL_MLP",
        "stress_noisy_labels/NoisyTargetGaussianNLL_MLP",
        "stress_noisy_labels/BetaNLL_MLP",
        # --- sparse labels ---
        "stress_sparse_labels/SupervisedOnly_GaussianNLL",
        "stress_sparse_labels/PseudoLabelNLL_MLP",
        # --- OOD ---
        "stress_ood/GaussianNLL_MLP",
        "stress_ood/MC_Dropout",
        "stress_ood/DeepEnsemble_K10",
        # --- semi-supervised (SAGE) ---
        "stress_semi_sup/SupervisedOnly_MLP",
        "stress_semi_sup/SAGE_MLP",
        # --- external comparators (import-gated) ---
        "stress_ood/luqbox_MCDropout",
        "stress_ood/luqbox_GaussianNLL",
    ]

    def methods(self) -> list[str]:
        methods = list(self._METHODS)
        if not _LUQBOX_AVAILABLE:
            methods = [m for m in methods if not m.endswith(tuple(self._LUQBOX_METHODS))]
        return methods

    _LUQBOX_METHODS = ("luqbox_MCDropout", "luqbox_GaussianNLL")

    def library_name(self, method: str) -> str:
        if method.endswith(self._LUQBOX_METHODS):
            return "lightning-uq-box"
        return "torchregress"

    def library_version(self, method: str) -> str:
        if method.endswith(self._LUQBOX_METHODS):
            try:
                from importlib.metadata import version

                return f"{version('lightning-uq-box')} (pin {_LUQBOX_PIN})"
            except Exception:
                return _LUQBOX_PIN
        return ""

    def method_notes(self, method: str) -> str:
        axis = method.split("/", 1)[0]
        return {
            "stress_imbalanced": "long-tailed target density",
            "stress_noisy_inputs": "EIV: noisy train features, clean test",
            "stress_noisy_labels": "heteroscedastic known label-noise variance",
            "stress_sparse_labels": "20% labeled; variance-weighted pseudo-labels",
            "stress_ood": "covariate shift along x0",
            "stress_semi_sup": "200 labeled + 3000 unlabeled (SAGE)",
        }.get(axis, "")

    # ------------------------------------------------------------------
    # Orchestration (stress datasets live outside the global registry)
    # ------------------------------------------------------------------

    # Axis prefix -> which extra split columns the axis' methods require.
    _AXIS_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
        "stress_sparse_labels": ("labeled_mask_train",),
        "stress_semi_sup": ("labeled_mask_train",),
        "stress_noisy_labels": ("target_variance_train",),
        "stress_noisy_inputs": ("sigma_x",),
    }

    @staticmethod
    def _method_axis(method: str) -> str:
        """stress_imbalanced/MSE_MLP -> stress_imbalanced."""
        return method.split("/", 1)[0]

    def _method_for_axis(self, axis: str) -> list[str]:
        """Methods that belong to a stress axis (their required columns exist only there)."""
        return [m for m in self.methods() if self._method_axis(m) == axis]

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        from time import perf_counter

        cfg = config or HarnessConfig()
        if dataset_name and dataset_name not in _STRESS_GENERATORS:
            raise KeyError(f"Unknown stress dataset: {dataset_name!r}")

        rows: list[dict[str, Any]] = []
        datasets = [dataset_name] if dataset_name else list(_STRESS_GENERATORS)
        for ds in datasets:
            data = (
                splits if (splits is not None and ds == dataset_name) else load_stress_dataset(ds)
            )
            X_test = data["X_test"]
            y_test = data["y_test"]
            if cfg.smoke:
                X_train, y_train = data["X_train"], data["y_train"]
                data = dict(data)
                data["X_train"], data["y_train"] = X_train[:200], y_train[:200]
                X_test, y_test = X_test[:50], y_test[:50]
            # Each method requires its own axis' extra columns (e.g.
            # labeled_mask_train), so only run methods whose axis matches the
            # dataset — cross-axis combinations would KeyError by construction.
            axis_methods = self._method_for_axis(ds)
            for method in axis_methods:
                set_harness_seed(cfg.seed)
                try:
                    t0 = perf_counter()
                    model = self.fit(method, data, cfg)
                    fit_time = perf_counter() - t0
                    t0 = perf_counter()
                    predictions = self.predict(method, model, X_test, cfg)
                    pred_time = perf_counter() - t0
                    metrics = self.evaluate(method, predictions, y_test, cfg)
                    rows.append(
                        {
                            "Method": f"{self.name}/{method}",
                            "Library": self.library_name(method),
                            "LibraryVersion": self.library_version(method),
                            "Dataset": ds,
                            "Suite": self.name,
                            **metrics,
                            "train_s": fit_time,
                            "eval_s": pred_time,
                            "SourceURL": data.get("_source_url", ""),
                            "SyntheticFallback": True,
                            "Device": str(current_device()),
                            "Notes": self.method_notes(method),
                        }
                    )
                except ImportError as exc:
                    rows.append(
                        {
                            "Method": f"{self.name}/{method}",
                            "Library": self.library_name(method),
                            "LibraryVersion": self.library_version(method),
                            "Dataset": ds,
                            "Suite": self.name,
                            "RMSE": None,
                            "MAE": None,
                            "train_s": None,
                            "eval_s": None,
                            "SourceURL": "",
                            "SyntheticFallback": True,
                            "Device": str(current_device()),
                            "Notes": f"skipped: {exc}",
                        }
                    )
                except Exception as exc:  # one bad method must not kill the matrix
                    rows.append(
                        {
                            "Method": f"{self.name}/{method}",
                            "Library": self.library_name(method),
                            "LibraryVersion": self.library_version(method),
                            "Dataset": ds,
                            "Suite": self.name,
                            "RMSE": None,
                            "MAE": None,
                            "train_s": None,
                            "eval_s": None,
                            "SourceURL": "",
                            "SyntheticFallback": True,
                            "Device": str(current_device()),
                            "Notes": f"error: {type(exc).__name__}: {exc}",
                        }
                    )
        return rows

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        axis, name = method.split("/", 1)
        if name in self._LUQBOX_METHODS:
            # LUQ-box trains + predicts in one call; arrays stored for predict.
            return {
                "type": "luqbox",
                "name": name,
                "X": splits["X_train"],
                "y": splits["y_train"].reshape(-1),
            }

        X_np = splits["X_train"]
        y_np = splits["y_train"].reshape(-1)
        in_dim = X_np.shape[1]
        hidden = _hidden_size(in_dim)
        dev = current_device()

        X = torch.from_numpy(X_np).float().to(dev)
        y = torch.from_numpy(y_np).float().to(dev).unsqueeze(1)

        # ---------------- imbalanced ----------------
        if name == "MSE_MLP":
            model = _MLP(in_dim, hidden, out_dim=1)
            _train_one(model, nn.MSELoss(), X, y, config)
            return {"type": "torch", "model": model, "head": "point"}
        elif name == "DensityWeighted_MLP":
            model = _MLP(in_dim, hidden, out_dim=1)
            loss = DensityWeightedLoss(base_loss="mse")
            loss.fit_density(y.reshape(-1))
            _train_one(model, loss, X, y, config)
            return {"type": "torch", "model": model, "head": "point"}
        elif name == "LDS_MLP":
            model = _MLP(in_dim, hidden, out_dim=1)
            loss = LDSLoss()
            loss.fit(y.reshape(-1))  # LDSLoss API: fit(), not fit_density()
            _train_one(model, loss, X, y, config)
            return {"type": "torch", "model": model, "head": "point"}
        elif name == "BinReweighted_MLP":
            model = _MLP(in_dim, hidden, out_dim=1)
            loss = BinReweightedMSELoss(num_bins=20, binning="quantile")
            loss.fit(y.reshape(-1))
            _train_one(model, loss, X, y, config)
            return {"type": "torch", "model": model, "head": "point"}

        # ---------------- noisy inputs (EIV) ----------------
        elif name == "InputNoise_MLP":
            model = _MLP(in_dim, hidden, out_dim=1)
            sigma = float(splits.get("sigma_x", 0.5))
            # WeightedMSELoss: mask/weights-aware MSE (plain nn.MSELoss cannot
            # accept the mask kwarg _train_special forwards).
            loss = InputNoiseAugmentationLoss(
                model=model, base_loss=WeightedMSELoss(), sigma_x=sigma
            )
            # x_obs is the observed (noisy) input batch; y_pred arg is ignored
            # when x_obs is provided.
            _train_special(model, loss, X, y, config, extra={"x_obs": X})
            return {"type": "torch", "model": model, "head": "point"}

        # ---------------- noisy labels ----------------
        elif name == "NoisyTargetGaussianNLL_MLP":
            model = _MLP(in_dim, hidden, out_dim=2)
            loss = NoisyTargetGaussianNLL()
            var = splits.get("target_variance_train")
            if var is None:
                raise ValueError("stress_noisy_labels split lacks target_variance_train")
            extra = {"target_variance": torch.from_numpy(var).float().to(dev).unsqueeze(1)}
            _train_special(model, loss, X, y, config, extra=extra)
            return {"type": "torch", "model": model, "head": "gaussian"}
        elif name == "BetaNLL_MLP":
            model = _MLP(in_dim, hidden, out_dim=2)
            _train_one(model, BetaNLLLoss(), X, y, config)
            return {"type": "torch", "model": model, "head": "gaussian"}
        elif name == "GaussianNLL_MLP":
            model = _MLP(in_dim, hidden, out_dim=2)
            _train_one(model, GaussianNLLLoss(), X, y, config)
            return {"type": "torch", "model": model, "head": "gaussian"}

        # ---------------- sparse labels ----------------
        elif name == "SupervisedOnly_GaussianNLL":
            model = _MLP(in_dim, hidden, out_dim=2)
            mask = torch.from_numpy(splits["labeled_mask_train"].astype(bool)).to(dev).unsqueeze(1)
            _train_special(model, GaussianNLLLoss(), X, y, config, extra={"mask": mask})
            return {"type": "torch", "model": model, "head": "gaussian"}
        elif name == "PseudoLabelNLL_MLP":
            model = _MLP(in_dim, hidden, out_dim=2)
            # bool mask: PseudoLabelNLL applies (~label_mask) — invalid on floats
            mask = torch.from_numpy(splits["labeled_mask_train"].astype(bool)).to(dev).unsqueeze(1)
            pseudo = torch.from_numpy(splits["pseudo_target_train"]).float().to(dev).unsqueeze(1)
            pseudo_var = splits.get("pseudo_variance_train")
            if pseudo_var is not None:
                conf = 1.0 / (
                    1.0 + torch.from_numpy(np.asarray(pseudo_var)).float().to(dev).unsqueeze(1)
                )
            else:
                conf = None
            extra: dict[str, torch.Tensor] = {"pseudo_target": pseudo, "label_mask": mask}
            if conf is not None:
                extra["pseudo_confidence"] = conf
            loss = PseudoLabelNLL(pseudo_weight=0.5)
            _train_special(model, loss, X, y, config, extra=extra)
            return {"type": "torch", "model": model, "head": "gaussian"}

        # ---------------- OOD (same heads as uq suite) ----------------
        elif name == "MC_Dropout":
            model = _MLP(in_dim, hidden, out_dim=1, dropout_p=0.5)
            _train_one(model, nn.MSELoss(), X, y, config)
            return {"type": "torch", "model": model, "head": "point", "mc": True}
        elif name == "DeepEnsemble_K10":
            members = []
            for k in range(10):
                set_harness_seed(config.seed + 1000 + k)
                m = _MLP(in_dim, hidden, out_dim=2)
                _train_one(m, GaussianNLLLoss(), X, y, config)
                members.append(m)
            return {"type": "ensemble", "members": members, "head": "gaussian"}

        # ---------------- semi-supervised (SAGE) ----------------
        elif name == "SupervisedOnly_MLP":
            model = _MLP(in_dim, hidden, out_dim=2)
            mask = torch.from_numpy(splits["labeled_mask_train"].astype(bool)).to(dev).unsqueeze(1)
            _train_special(model, GaussianNLLLoss(), X, y, config, extra={"mask": mask})
            return {"type": "torch", "model": model, "head": "gaussian"}
        elif name == "SAGE_MLP":
            return self._fit_sage(splits, config, in_dim, hidden, dev, y_train_len=y_np.shape[0])

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # SAGE teacher-student training (real SAGERegLoss + EMA teacher)
    # ------------------------------------------------------------------

    def _fit_sage(
        self,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
        in_dim: int,
        hidden: int,
        dev: torch.device,
        *,
        y_train_len: int | None = None,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        X_np = splits["X_train"]
        y_np = splits["y_train"].reshape(-1)
        # The mask is stored at full generator length; under cfg.smoke the train
        # split is truncated to the first 200 rows, so truncate the mask too.
        mask_full = splits["labeled_mask_train"].astype(bool)
        n_train = y_train_len if y_train_len is not None else X_np.shape[0]
        mask_np = mask_full[:n_train]
        X = torch.from_numpy(X_np).float().to(dev)
        y = torch.from_numpy(y_np).float().to(dev).unsqueeze(1)

        student = _MLP(in_dim, hidden, out_dim=2)
        teacher = copy.deepcopy(student)
        for p in teacher.parameters():
            p.requires_grad_(False)

        sage_loss = SAGERegLoss(tau=0.2, agreement_weight=0.5, n_support=64)
        sup_loss = GaussianNLLLoss()

        labeled_idx = torch.where(torch.from_numpy(mask_np))[0]
        unlabeled_idx = torch.where(~torch.from_numpy(mask_np))[0]
        bs = min(config.batch_size, max(1, len(labeled_idx)))
        epochs = 2 if config.smoke else 30
        ema_momentum = 0.99

        opt = torch.optim.Adam(student.parameters(), lr=config.lr)
        for _ in range(epochs):
            perm = torch.randperm(len(labeled_idx), device=dev)
            for i in range(0, len(labeled_idx), bs):
                idx = labeled_idx[perm[i : i + bs]]
                pred = student(X[idx])
                loss = sup_loss(pred, y[idx])
                if len(unlabeled_idx) >= 8:
                    u_idx = unlabeled_idx[torch.randint(0, len(unlabeled_idx), (bs,), device=dev)]
                    with torch.no_grad():
                        t_out = teacher(X[u_idx])
                    t_view = PredictiveBatch(
                        mean=t_out[:, 0].detach(),
                        std=torch.exp(t_out[:, 1]).clamp_min(1e-6).detach(),
                    )
                    s_out = student(X[u_idx])
                    s_view = PredictiveBatch(
                        mean=s_out[:, 0],
                        std=torch.exp(s_out[:, 1]).clamp_min(1e-6),
                    )
                    out = sage_loss(loss, [s_view, t_view])
                    opt.zero_grad()
                    out.total_loss.backward()
                    opt.step()
                else:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                update_ema_teacher_(teacher, student, momentum=ema_momentum)
        return {"type": "torch", "model": teacher, "head": "gaussian"}

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        head = model.get("head", "point")

        if model.get("type") == "luqbox":
            mean, std = _luqbox_run(
                model["name"],
                model["X"],
                model["y"],
                X_test,
                epochs=2 if config.smoke else 15,
                seed=config.seed,
                smoke=config.smoke,
            )
            return {"mean": mean, "std": std}

        if model["type"] == "ensemble":
            means, stds = [], []
            for m in model["members"]:
                mu, sd = _gaussian_nll_predict(m, X_test)
                means.append(mu)
                stds.append(sd)
            mean, std = _mixture_gaussian_stats(means, stds)
            return {"mean": mean, "std": std}

        if head == "gaussian":
            mean, std = _gaussian_nll_predict(model["model"], X_test)
            return {"mean": mean, "std": std}

        m = model["model"]
        device = next(m.parameters()).device
        if model.get("mc"):
            was_training = m.training
            m.train()
            preds = []
            try:
                with torch.no_grad():
                    for _ in range(20):
                        preds.append(
                            m(torch.from_numpy(X_test).float().to(device))[:, 0]
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )
            finally:
                if not was_training:
                    m.eval()
            stacked = np.stack(preds, axis=0)
            return {"mean": stacked.mean(axis=0), "std": stacked.std(axis=0, ddof=0).clip(1e-8)}

        with torch.no_grad():
            pred = m(torch.from_numpy(X_test).float().to(device))[:, 0].cpu().numpy().reshape(-1)
        return {"y_pred": pred}
