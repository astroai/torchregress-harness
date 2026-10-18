"""Contrastive Normalizing Flow (CNF) reproduction.

Reproduces the core mechanism of Elsharkawy & Kahn (2025), "Contrastive
Normalizing Flows for Uncertainty-Aware Parameter Estimation"
(arXiv:2505.08709): train normalizing flows with the contrastive objective

    L_s = -E[log p_s(x_s)] + c E[log p_s(x_b)]

(and the reverse for p_b), clamp the loss at k = -50 to avoid mode collapse,
then feed the learned signal/background discrimination functions
``Phi = p / (1 + p)`` plus the raw features to a DNN classifier.  The paper's
gated-affine/quadratic coupling architecture is approximated with zuko NSF
couplings; that architectural deviation is stated in the method notes.

Tasks:
* ``cnf_toy_gaussian`` -- the paper's 2-D two-Gaussian toy, with rotation
  nuisance tests (accuracy vs rotation angle).
* ``fair_universe_sample`` -- the real simulated HEP starting-kit sample
  (1000 events, 28 features) from the FAIR-Universe HiggsML dataset.  The full
  challenge data is 15.1 GB (Zenodo 15131565); this suite reproduces the
  method on the public sample, not the challenge score.

Methods: ``DNN`` (raw-feature classifier), ``NF_NLL`` (two standard flows
trained by NLL -> log-density ratio), and ``CNF`` (paper method).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _hidden_size

_METHODS = ["DNN", "NF_NLL", "CNF"]
_BATCH = 256
_ROTATIONS = (15.0, 30.0, 45.0)


def _make_flow(dim: int, config: HarnessConfig) -> nn.Module:
    import zuko

    hidden = 32 if config.smoke else 64
    return zuko.flows.NSF(
        dim,
        transforms=2 if config.smoke else 4,
        hidden_features=(hidden, hidden),
        bins=8 if config.smoke else 16,
    )


def _train_flow(
    flow: nn.Module,
    pos: torch.Tensor,
    neg: torch.Tensor | None,
    *,
    contrastive_c: float | None,
    steps: int,
    lr: float,
    seed: int,
) -> None:
    """Train a flow by NLL (``contrastive_c is None``) or the CNF objective."""
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = len(pos)
    for _ in range(steps):
        pi = torch.randint(n, (_BATCH,), generator=generator).to(pos.device)
        dist = flow()
        loss = -dist.log_prob(pos[pi]).mean()
        if contrastive_c is not None and neg is not None:
            ni = torch.randint(len(neg), (_BATCH,), generator=generator).to(pos.device)
            loss = loss + float(contrastive_c) * dist.log_prob(neg[ni]).mean()
            # Paper's mode-collapse guard (Sec. 4 / App. A).
            loss = torch.clamp(loss, min=-50.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    flow.eval()


def _train_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int,
    lr: float,
    seed: int,
) -> nn.Module:
    hidden = 64
    model = nn.Sequential(
        nn.Linear(int(features.shape[1]), hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, 1),
    ).to(features.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(features)
    for _ in range(steps):
        idx = torch.randint(n, (_BATCH,), generator=generator).to(features.device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features[idx]).squeeze(-1), labels[idx])
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    n = min(len(pos), len(neg))
    if n == 0:
        raise ValueError("CNF reproduction needs both signal and background events")
    chosen = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    rng.shuffle(chosen)
    return chosen


def _phi(flow: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        log_p = flow().log_prob(x).clamp(-30.0, 30.0).unsqueeze(-1)
    p = torch.exp(log_p)
    return p / (1.0 + p)


class CNFParitySuite(HarnessRunner):
    """CNF (paper method) vs raw DNN and NLL likelihood-ratio flows."""

    @property
    def name(self) -> str:
        return "cnf_parity"

    @property
    def description(self) -> str:
        return "Contrastive Normalizing Flow reproduction vs DNN and NLL flow baselines"

    @property
    def supported_datasets(self) -> list[str]:
        return ["cnf_toy_gaussian", "fair_universe_sample"]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def library_name(self, method: str) -> str:
        return "harness (zuko)"

    def library_version(self, method: str) -> str:
        try:
            import zuko

            return str(zuko.__version__)
        except Exception:
            return "unknown"

    def method_notes(self, method: str) -> str:
        notes = {
            "DNN": "MLP classifier on raw features (3 hidden layers, GELU)",
            "NF_NLL": "two zuko NSF flows trained by class-conditional NLL; log-density-ratio score",
            "CNF": "contrastive flows (c=2 for p_s, c=0.5 for p_b, clamp -50) + Phi-probability DNN; "
            "zuko NSF coupling instead of the paper's gated-affine/quadratic layers",
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _steps(self, n: int, config: HarnessConfig) -> int:
        per_epoch = max(1, int(np.ceil(n / _BATCH)))
        steps = int(config.epochs) * per_epoch
        return min(steps, 60) if config.smoke else steps

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        device = current_device()
        X_train = np.asarray(splits["X_train"], dtype=np.float32)
        y_train = np.asarray(splits["y_train"], dtype=np.float32).reshape(-1)
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-6
        X_std = (X_train - mean) / std
        balanced = _balanced_indices(y_train, int(config.seed))
        X_bal = torch.as_tensor(X_std[balanced], dtype=torch.float32, device=device)
        y_bal = torch.as_tensor(y_train[balanced], dtype=torch.float32, device=device)
        pos = X_bal[y_bal == 1]
        neg = X_bal[y_bal == 0]
        steps = self._steps(len(X_bal), config)

        state: dict[str, Any] = {
            "method": method,
            "mean": mean,
            "std": std,
            "n_features": X_std.shape[1],
        }
        if method == "DNN":
            state["clf"] = _train_classifier(
                X_bal, y_bal, steps=steps, lr=config.lr, seed=int(config.seed) + 1
            )
            return state
        if method == "NF_NLL":
            f_s = _make_flow(X_std.shape[1], config).to(device)
            f_b = _make_flow(X_std.shape[1], config).to(device)
            _train_flow(
                f_s,
                pos,
                None,
                contrastive_c=None,
                steps=steps,
                lr=config.lr,
                seed=int(config.seed) + 2,
            )
            _train_flow(
                f_b,
                neg,
                None,
                contrastive_c=None,
                steps=steps,
                lr=config.lr,
                seed=int(config.seed) + 3,
            )
            state.update({"f_s": f_s, "f_b": f_b})
            return state
        if method == "CNF":
            f_s = _make_flow(X_std.shape[1], config).to(device)
            f_b = _make_flow(X_std.shape[1], config).to(device)
            _train_flow(
                f_s,
                pos,
                neg,
                contrastive_c=2.0,
                steps=steps,
                lr=config.lr,
                seed=int(config.seed) + 4,
            )
            _train_flow(
                f_b,
                neg,
                pos,
                contrastive_c=0.5,
                steps=steps,
                lr=config.lr,
                seed=int(config.seed) + 5,
            )
            phi_s = _phi(f_s, X_bal)
            phi_b = _phi(f_b, X_bal)
            features = torch.cat([X_bal, phi_s, phi_b], dim=1)
            state["clf"] = _train_classifier(
                features, y_bal, steps=steps, lr=config.lr, seed=int(config.seed) + 6
            )
            state.update({"f_s": f_s, "f_b": f_b})
            return state
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Prediction / evaluation
    # ------------------------------------------------------------------

    def _features(self, state: dict[str, Any], X: np.ndarray) -> torch.Tensor:
        device = current_device()
        x = (np.asarray(X, dtype=np.float32) - state["mean"]) / state["std"]
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        method = state["method"]
        if method == "DNN":
            return x_t
        if method == "NF_NLL":
            with torch.no_grad():
                return (state["f_s"]().log_prob(x_t) - state["f_b"]().log_prob(x_t)).unsqueeze(-1)
        if method == "CNF":
            return torch.cat([x_t, _phi(state["f_s"], x_t), _phi(state["f_b"], x_t)], dim=1)
        raise ValueError(f"Unknown state method: {method}")

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        del method, config
        features = self._features(model_state, X_test)
        if model_state["method"] == "NF_NLL":
            score = features.squeeze(-1)
        else:
            with torch.no_grad():
                score = model_state["clf"](features).squeeze(-1)
        return {
            "y_pred": torch.sigmoid(score).detach().cpu().numpy(),
            "score": score.detach().cpu().numpy(),
        }

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        from sklearn.metrics import log_loss, roc_auc_score

        del method, config
        y = np.asarray(y_test).reshape(-1).astype(int)
        prob = np.asarray(predictions["y_pred"], dtype=np.float64).reshape(-1)
        return {
            "Accuracy": float(np.mean((prob >= 0.5) == y)),
            "AUC": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
            "LogLoss": float(log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7))),
        }

    # ------------------------------------------------------------------
    # Custom run: clean metrics + toy rotation nuisance metrics
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        from time import perf_counter

        from sklearn.metrics import log_loss, roc_auc_score

        cfg = config or HarnessConfig()
        dataset_name, splits = self._prepare_scenario_run(
            scenario_name, dataset_name, splits, standardize_targets=False
        )
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name, standardize_targets=False)
        X_test = np.asarray(splits["X_test"], dtype=np.float32)
        y_test = np.asarray(splits["y_test"], dtype=np.float32).reshape(-1)

        rows: list[dict[str, Any]] = []
        for method in self.methods():
            set_harness_seed(cfg.seed)
            base: dict[str, Any] = {
                "Method": f"{self.name}/{method}",
                "Library": self.library_name(method),
                "LibraryVersion": self.library_version(method),
                "Dataset": dataset_name,
                "Suite": self.name,
                "SourceURL": splits.get("_source_url", ""),
                "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                "Device": str(current_device()),
                "Notes": self.method_notes(method),
            }
            try:
                t0 = perf_counter()
                state = self.fit(method, splits, cfg)
                train_s = perf_counter() - t0
                t0 = perf_counter()
                pred = self.predict(method, state, X_test, cfg)
                metrics = self.evaluate(method, pred, y_test, cfg)
                metrics["n_train"] = int(len(splits["y_train"]))
                metrics["n_test"] = int(len(y_test))
                if dataset_name == "cnf_toy_gaussian":
                    for angle in _ROTATIONS:
                        theta = np.deg2rad(angle)
                        rot = np.array(
                            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                            dtype=np.float32,
                        )
                        X_rot = X_test @ rot.T
                        pred_rot = self.predict(method, state, X_rot, cfg)
                        p = np.asarray(pred_rot["y_pred"])
                        metrics[f"Rotation{int(angle)}_Accuracy"] = float(
                            np.mean((p >= 0.5) == y_test.astype(int))
                        )
                        metrics[f"Rotation{int(angle)}_AUC"] = float(roc_auc_score(y_test, p))
                eval_s = perf_counter() - t0
                rows.append({**base, **metrics, "train_s": train_s, "eval_s": eval_s})
            except ImportError as exc:
                rows.append(
                    {
                        **base,
                        "Accuracy": None,
                        "AUC": None,
                        "train_s": None,
                        "eval_s": None,
                        "Notes": f"skipped: {type(exc).__name__}: {exc}",
                    }
                )
            except Exception as exc:  # keep one failed method from aborting the suite
                rows.append(
                    {
                        **base,
                        "Accuracy": None,
                        "AUC": None,
                        "train_s": None,
                        "eval_s": None,
                        "Notes": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
        return rows


__all__ = ["CNFParitySuite"]
