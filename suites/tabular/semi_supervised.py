"""Semi-supervised regression suite: few labels plus an unlabeled pool.

Benchmarks ``torchregress.semi_supervised`` (``TeacherStudentTrainer``,
``SAGERegLoss``, ``disagreement_to_weight``, ``predictive_agreement_score``)
against models trained on the labeled subset alone.  Every method sees the
same labeled subset (``N_LABELED`` rows drawn from the training split by the
seed); the semi-supervised methods also see the remaining training features
without their targets.  Scores are on the untouched test split.

Methods
-------
* ``Supervised_GaussianNLL`` - the shared Gaussian MLP trained on the labeled
  rows only, same number of optimizer steps as the semi-supervised rows.
  The baseline every semi-supervised row must beat.
* ``TeacherStudent_SAGEReg`` - ``TeacherStudentTrainer`` with an EMA teacher,
  four input-noise views and the default disagreement trust weights.
* ``TeacherStudent_BatchRelative`` - same trainer with a custom
  ``sample_weight_fn``: ``predictive_agreement_score`` turned into weights by
  ``disagreement_to_weight(batch_relative_mode=True)``.
* ``TeacherStudent_UncertaintyWeight`` - same trainer, trust weights from the
  consensus predictive std (``uncertainty_to_weight``) instead of disagreement.
* ``SAGERegLoss_SelfAgreement`` - no teacher: ``SAGERegLoss`` on four
  input-noise views of the student itself.
* ``Ridge_labeled`` / ``MLP_labeled`` - scikit-learn ``RidgeCV`` and
  ``MLPRegressor`` on the labeled rows (scikit-learn has no
  ``SelfTrainingRegressor``; ``SelfTrainingClassifier`` is classification only).
* ``SelfTraining_RF`` - classic self-training with a scikit-learn random
  forest: pseudo-label the unlabeled rows whose per-tree spread is in the
  lowest quartile, refit, two rounds (harness implementation).
* ``Oracle_FullyLabeled`` - the Gaussian MLP trained with every training
  target: the upper bound, not a competitor.

Datasets
--------
* ``ssl_friedman`` - Friedman #1 (10 features, 5 informative, noise 1),
  2000 training rows of which 100 labeled, 1000 test rows (synthetic by
  design, generated in-suite).
* ``diabetes`` - scikit-learn's bundled diabetes data (no download); 50
  labeled rows.  Loaded through ``harness.datasets.load_dataset``, so strict
  data mode applies and a synthetic stand-in is never used silently (rows
  carry ``SyntheticFallback`` from the loader).
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP

HIDDEN = 64
N_VIEWS = 4
NOISE_STD = 0.1  # input-noise augmentation (features are standardized)
STEPS_PER_EPOCH = 20  # optimizer steps per config epoch, identical for every torch row
LABELED_BATCH = 32
UNLABELED_BATCH = 128
_Z90 = 1.6448536269514722

# name -> (n_labeled, n_labeled in smoke mode)
N_LABELED: dict[str, tuple[int, int]] = {"ssl_friedman": (100, 40), "diabetes": (50, 30)}
SYNTHETIC_DESIGNS = ("ssl_friedman",)

TORCH_METHODS = (
    "Supervised_GaussianNLL",
    "TeacherStudent_SAGEReg",
    "TeacherStudent_BatchRelative",
    "TeacherStudent_UncertaintyWeight",
    "SAGERegLoss_SelfAgreement",
    "Oracle_FullyLabeled",
)
SKLEARN_METHODS = ("Ridge_labeled", "MLP_labeled", "SelfTraining_RF")


def draw_friedman(seed: int, *, smoke: bool) -> dict[str, np.ndarray]:
    """Friedman #1 with standardized features and targets (test split separate)."""
    from sklearn.datasets import make_friedman1

    n_train, n_test = (400, 200) if smoke else (2000, 1000)
    X, y = make_friedman1(n_samples=n_train + n_test, n_features=10, noise=1.0, random_state=seed)
    X = (X - X[:n_train].mean(axis=0)) / X[:n_train].std(axis=0)
    mu, sd = y[:n_train].mean(), y[:n_train].std()
    y = (y - mu) / sd
    return {
        "X_train": X[:n_train].astype(np.float32),
        "y_train": y[:n_train].astype(np.float32),
        "X_test": X[n_train:].astype(np.float32),
        "y_test": y[n_train:].astype(np.float32),
    }


def labeled_split(n_train: int, n_labeled: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint (labeled, unlabeled) index arrays over the training rows."""
    if n_labeled >= n_train:
        raise ValueError(f"need more training rows ({n_train}) than labels ({n_labeled})")
    perm = np.random.default_rng(seed + 7).permutation(n_train)
    return np.sort(perm[:n_labeled]), np.sort(perm[n_labeled:])


def _gaussian_batch(model: nn.Module, x: torch.Tensor) -> Any:
    from torchregress.prediction import PredictiveBatch

    out = model(x)
    return PredictiveBatch(mean=out[:, 0], std=torch.exp(0.5 * out[:, 1]).clamp_min(1e-3))


def _noise(x: torch.Tensor) -> torch.Tensor:
    return x + NOISE_STD * torch.randn_like(x)


class SemiSupervisedSuite(HarnessRunner):
    """torchregress teacher-student / SAGEReg vs labeled-only baselines with few labels."""

    @property
    def name(self) -> str:
        return "semi_supervised"

    @property
    def description(self) -> str:
        return "Semi-supervised regression (TeacherStudent, SAGEReg) vs labeled-only baselines"

    @property
    def supported_datasets(self) -> list[str]:
        return list(N_LABELED)

    def methods(self) -> list[str]:
        return [*TORCH_METHODS, *SKLEARN_METHODS]

    def library_name(self, method: str) -> str:
        return "scikit-learn" if method in SKLEARN_METHODS else "torchregress"

    def library_version(self, method: str) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(self.library_name(method))
        except PackageNotFoundError:
            return "not installed"

    def method_notes(self, method: str) -> str:
        return {
            "Supervised_GaussianNLL": "Gaussian MLP on labeled rows only (same step budget)",
            "TeacherStudent_SAGEReg": "TeacherStudentTrainer, EMA teacher, 4 noise views, default weights",
            "TeacherStudent_BatchRelative": "TeacherStudentTrainer, disagreement_to_weight(batch_relative_mode)",
            "TeacherStudent_UncertaintyWeight": "TeacherStudentTrainer, uncertainty_to_weight(consensus std)",
            "SAGERegLoss_SelfAgreement": "SAGERegLoss on 4 noise views of the student (no teacher)",
            "Oracle_FullyLabeled": "Gaussian MLP with every training target (upper bound)",
            "Ridge_labeled": "RidgeCV on labeled rows only",
            "MLP_labeled": "MLPRegressor(64, 64) on labeled rows only",
            "SelfTraining_RF": "RandomForest self-training, low-spread pseudo-labels, 2 rounds",
        }.get(method, "")

    # ------------------------------------------------------------------
    # torch rows
    # ------------------------------------------------------------------

    def _fit_torch(
        self,
        method: str,
        X_lab: np.ndarray,
        y_lab: np.ndarray,
        X_unl: np.ndarray,
        cfg: HarnessConfig,
    ) -> tuple[nn.Module, dict[str, float]]:
        from torchregress.losses import GaussianNLLLoss

        dev = current_device()
        n_steps = max(1, cfg.epochs * STEPS_PER_EPOCH)
        g = np.random.default_rng(cfg.seed)
        Xl = torch.from_numpy(X_lab).float().to(dev)
        yl = torch.from_numpy(y_lab).float().reshape(-1, 1).to(dev)
        Xu = torch.from_numpy(X_unl).float().to(dev)
        model = _MLP(X_lab.shape[1], HIDDEN, 2).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        nll = GaussianNLLLoss()

        def lab_batches(k: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
            idx = g.integers(0, len(Xl), size=(k, min(LABELED_BATCH, len(Xl))))
            return [(Xl[i], yl[i]) for i in idx]

        def unl_batches(k: int) -> list[torch.Tensor]:
            idx = g.integers(0, len(Xu), size=(k, min(UNLABELED_BATCH, len(Xu))))
            return [Xu[i] for i in idx]

        def supervised_steps(batches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
            model.train()
            for xb, yb in batches:
                opt.zero_grad(set_to_none=True)
                nll(model(xb), yb).backward()
                opt.step()

        extra: dict[str, float] = {}
        if method in ("Supervised_GaussianNLL", "Oracle_FullyLabeled"):
            supervised_steps(lab_batches(n_steps))
        else:
            # Supervised warm-up so the first teacher views are informative;
            # the semi-supervised phase uses the rest of the same step budget.
            n_warm = n_steps // 4
            supervised_steps(lab_batches(n_warm))
            n_ssl = max(1, n_steps - n_warm)
            if method.startswith("TeacherStudent_"):
                extra = self._teacher_student(
                    method, model, opt, nll, lab_batches(n_ssl), unl_batches(n_ssl)
                )
            elif method == "SAGERegLoss_SelfAgreement":
                extra = self._self_agreement(
                    model, opt, nll, lab_batches(n_ssl), unl_batches(n_ssl)
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        model.eval()
        return model, extra

    @staticmethod
    def _teacher_student(
        method: str,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        nll: nn.Module,
        labeled: list[tuple[torch.Tensor, torch.Tensor]],
        unlabeled: list[torch.Tensor],
    ) -> dict[str, float]:
        from torchregress.semi_supervised import (
            TeacherStudentTrainer,
            disagreement_to_weight,
            predictive_agreement_score,
            uncertainty_to_weight,
        )

        kwargs: dict[str, Any] = {}
        if method == "TeacherStudent_BatchRelative":

            def weight_fn(views: Any, consensus: Any) -> torch.Tensor:
                del consensus
                d = predictive_agreement_score(views, reduction="none")
                return disagreement_to_weight(d, tau=1.0, batch_relative_mode=True)

            kwargs["sample_weight_fn"] = weight_fn
        elif method == "TeacherStudent_UncertaintyWeight":
            # tau = 1 on standardized targets: weight exp(-std) of the consensus.
            kwargs["sample_weight_fn"] = lambda views, consensus: uncertainty_to_weight(
                consensus, tau=1.0
            )
        trainer = TeacherStudentTrainer(
            optimizer=opt,
            supervised_loss_fn=lambda m, x, y: nll(m(x), y),
            predictive_batch_fn=_gaussian_batch,
            augment_fn=_noise,
            n_views=N_VIEWS,
            ema_decay=0.99,
            **kwargs,
        )
        hist = trainer.fit(model, labeled, unlabeled, epochs=1)
        return {
            "MeanTrustWeight": float(np.mean(hist["mean_weight"])),
            "MeanDisagreement": float(np.mean(hist["mean_disagreement"])),
        }

    @staticmethod
    def _self_agreement(
        model: nn.Module,
        opt: torch.optim.Optimizer,
        nll: nn.Module,
        labeled: list[tuple[torch.Tensor, torch.Tensor]],
        unlabeled: list[torch.Tensor],
    ) -> dict[str, float]:
        from torchregress.semi_supervised import SAGERegLoss

        sagereg = SAGERegLoss(tau=0.2, agreement_weight=1.0)
        weights: list[float] = []
        model.train()
        for (xb, yb), xu in zip(labeled, unlabeled, strict=True):
            opt.zero_grad(set_to_none=True)
            views = [_gaussian_batch(model, xu)]
            views += [_gaussian_batch(model, _noise(xu)) for _ in range(N_VIEWS - 1)]
            out = sagereg(nll(model(xb), yb), views)
            out.total_loss.backward()
            opt.step()
            weights.append(float(out.agreement.weights.mean().item()))
        return {"MeanTrustWeight": float(np.mean(weights))}

    @staticmethod
    def _predict_torch(model: nn.Module, X: np.ndarray) -> dict[str, np.ndarray]:
        with torch.no_grad():
            out = model(torch.from_numpy(X).float().to(current_device())).cpu().numpy()
        mean = out[:, 0].astype(np.float64)
        std = np.sqrt(np.exp(out[:, 1].astype(np.float64)).clip(1e-12))
        return {
            "y_pred": mean,
            "mean": mean,
            "std": std,
            "lower": mean - _Z90 * std,
            "upper": mean + _Z90 * std,
        }

    # ------------------------------------------------------------------
    # scikit-learn rows
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_predict_sklearn(
        method: str,
        X_lab: np.ndarray,
        y_lab: np.ndarray,
        X_unl: np.ndarray,
        X_test: np.ndarray,
        seed: int,
        smoke: bool,
    ) -> dict[str, np.ndarray]:
        if method == "Ridge_labeled":
            from sklearn.linear_model import RidgeCV

            est = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X_lab, y_lab)
            return {"y_pred": est.predict(X_test)}
        if method == "MLP_labeled":
            from sklearn.neural_network import MLPRegressor

            est = MLPRegressor(
                hidden_layer_sizes=(HIDDEN, HIDDEN),
                max_iter=200 if smoke else 2000,
                random_state=seed,
            ).fit(X_lab, y_lab)
            return {"y_pred": est.predict(X_test)}
        if method == "SelfTraining_RF":
            from sklearn.ensemble import RandomForestRegressor

            n_trees = 50 if smoke else 200
            X_fit, y_fit = X_lab, y_lab
            pool = X_unl
            for _ in range(2):
                rf = RandomForestRegressor(
                    n_estimators=n_trees, min_samples_leaf=2, random_state=seed
                ).fit(X_fit, y_fit)
                if len(pool) == 0:
                    break
                per_tree = np.stack([t.predict(pool) for t in rf.estimators_])
                spread = per_tree.std(axis=0)
                keep = spread <= np.quantile(spread, 0.25)
                X_fit = np.vstack([X_fit, pool[keep]])
                y_fit = np.concatenate([y_fit, per_tree.mean(axis=0)[keep]])
                pool = pool[~keep]
            rf = RandomForestRegressor(
                n_estimators=n_trees, min_samples_leaf=2, random_state=seed
            ).fit(X_fit, y_fit)
            return {"y_pred": rf.predict(X_test)}
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def _load(self, dataset_name: str, splits: dict[str, Any] | None, cfg: HarnessConfig) -> Any:
        if dataset_name in SYNTHETIC_DESIGNS:
            return draw_friedman(cfg.seed, smoke=cfg.smoke)
        if splits is None:
            from harness.datasets import load_dataset

            # Strict data mode is enforced by load_dataset: a real dataset
            # never silently turns into a synthetic stand-in here.
            splits = load_dataset(dataset_name, standardize_targets=cfg.standardize_targets)
        return splits

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, Any] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        del scenario_name
        cfg = config or HarnessConfig()
        rows: list[dict[str, Any]] = []
        base_meta: dict[str, Any] = {
            "Dataset": dataset_name,
            "Suite": self.name,
            "Device": str(current_device()),
        }
        if dataset_name not in N_LABELED:
            for method in self.methods():
                rows.append(
                    {
                        **base_meta,
                        "Method": f"{self.name}/{method}",
                        "Library": self.library_name(method),
                        "Notes": f"failed: unsupported dataset {dataset_name!r}",
                    }
                )
            return rows
        data = self._load(dataset_name, splits, cfg)
        X_tr = np.asarray(data["X_train"], dtype=np.float32)
        y_tr = np.asarray(data["y_train"], dtype=np.float32).reshape(-1)
        X_te = np.asarray(data["X_test"], dtype=np.float32)
        y_te = np.asarray(data["y_test"], dtype=np.float32).reshape(-1)
        n_lab = N_LABELED[dataset_name][1 if cfg.smoke else 0]
        lab, unl = labeled_split(len(X_tr), n_lab, cfg.seed)
        synthetic_fallback = bool(data.get("_synthetic_fallback", False))
        for method in self.methods():
            row: dict[str, Any] = {
                **base_meta,
                "Method": f"{self.name}/{method}",
                "Library": self.library_name(method),
                "LibraryVersion": self.library_version(method),
                "SourceURL": data.get("_source_url", ""),
                "SyntheticFallback": synthetic_fallback,
            }
            try:
                set_harness_seed(cfg.seed)
                t0 = perf_counter()
                extra: dict[str, float] = {}
                if method in TORCH_METHODS:
                    fit_idx = np.arange(len(X_tr)) if method == "Oracle_FullyLabeled" else lab
                    model, extra = self._fit_torch(
                        method, X_tr[fit_idx], y_tr[fit_idx], X_tr[unl], cfg
                    )
                    train_s = perf_counter() - t0
                    t0 = perf_counter()
                    pred: dict[str, Any] = self._predict_torch(model, X_te)
                else:
                    pred = self._fit_predict_sklearn(
                        method, X_tr[lab], y_tr[lab], X_tr[unl], X_te, cfg.seed, cfg.smoke
                    )
                    train_s = perf_counter() - t0
                    t0 = perf_counter()
                eval_s = perf_counter() - t0
                if "_y_scaler" in data:
                    pred["_y_scaler"] = data["_y_scaler"]
                metrics = super().evaluate(method, pred, y_te, cfg)
                rows.append(
                    {
                        **row,
                        **metrics,
                        **extra,
                        "NLabeled": float(len(X_tr) if method == "Oracle_FullyLabeled" else n_lab),
                        "NUnlabeled": float(0 if method == "Oracle_FullyLabeled" else len(unl)),
                        "train_s": train_s,
                        "eval_s": eval_s,
                        "Notes": self.method_notes(method),
                    }
                )
            except Exception as exc:  # noqa: BLE001 — one method failing must not drop the rest
                rows.append({**row, "Notes": f"failed: {type(exc).__name__}: {exc}"})
        return rows

    # HarnessRunner abstract surface (unused: custom run drives everything).
    def fit(self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig) -> Any:
        raise NotImplementedError("semi_supervised uses custom run()")

    def predict(
        self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("semi_supervised uses custom run()")
