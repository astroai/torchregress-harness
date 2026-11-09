"""Representation-level imbalanced regression: PRIME, ConR and continuous group DRO.

Complements the existing ``imbalanced``/``distributional_bins`` coverage with a
controlled long-tailed benchmark on the harness ``stress_imbalanced`` dataset
(exponential target density).  All methods share the same MLP capacity and
training budget; PRIME and ConR add feature-level regularizers, group DRO
replaces the average risk with the worst target-regime risk.

Methods
-------
- ``MSE``: plain MLP + MSE (bulk-dominated baseline).
- ``MSE_GroupDRO``: ``GroupDROTrainer`` over four equal-width target regimes.
- ``MSE_PRIME``: MLP encoder/head + ``PrimeLoss`` proxy regularizer.
- ``MSE_ConR``: MLP encoder/head + ``ConR`` collapse regularizer.
- ``MSE_UAConR``: same as ``MSE_ConR`` with ``UncertaintyAwareConR`` (Gaussian
  label-uncertainty-softened pair memberships, label sd ``0.1 * std(y)``).
- ``MSE_BMC``: MLP + true noise-adaptive Balanced MSE (BMC; Ren et al. 2022).
- ``MSE_RankSim``: MLP encoder/head + task MSE + RankSim ranking-similarity regularizer.
- ``VIR``: Variational Imbalanced Regression (NIG evidential head, label-density smoothing).

Metrics are the standard harness point/interval metrics plus ``WorstGroupMAE``:
the largest per-target-regime MAE over the fitted regimes (tail robustness).

Representation geometry
-----------------------
Every row also carries ``Rep_*`` columns from
``torchregress.metrics.representation_geometry_report`` on the learned test
features (the encoder output for encoder/head rows, the last hidden layer of
the MLP rows, the VIR encoder): neighbour target gap (k=10), feature/target
distance ordering (Spearman) and frozen OLS / k-NN readouts, in target units.
A ``rep_geometry/raw_inputs`` row gives the same report on the raw test
inputs as the no-learning reference.  The O(N^2) diagnostics use at most
``REP_MAX_POINTS`` seeded test points.

Paired comparisons
------------------
``paired_abs_error/<A>-vs-<B>`` rows give ``mean(|e_A| - |e_B|)`` over the
shared test points with a 95% paired percentile bootstrap interval
(``paired_bootstrap_ci``; test-sample uncertainty only, no retraining variation).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torchregress.algorithms import ContinuousGroupDRO, GroupDROTrainer
from torchregress.algorithms.vir import VIRLoss, VIRRegressor  # noqa: F401 - VIRLoss evidence
from torchregress.losses import (  # noqa: F401 - functional helpers kept for export coverage
    BalancedMSEBMC,
    ConR,
    PrimeLoss,
    RankSimLoss,
    UncertaintyAwareConR,
    balanced_mse_bmc_loss,
    conr_loss,
    rank_sim_regularizer,
    uniform_proxy_targets,
)
from torchregress.metrics import paired_bootstrap_ci, representation_geometry_report

from harness.core import set_harness_seed
from harness.datasets import SplitsDict
from harness.devices import current_device
from harness.metrics import compute_all_metrics
from harness.runners import (
    HarnessConfig,
    HarnessRunner,
    _inverse_transform_predictions,
    _inverse_transform_y,
)
from suites._backbones import _MLP, _hidden_size

_METHODS = [
    "MSE",
    "MSE_GroupDRO",
    "MSE_PRIME",
    "MSE_ConR",
    "MSE_UAConR",
    "MSE_BMC",
    "MSE_RankSim",
    "VIR",
]
_N_GROUPS = 4
_FEATURE_DIM = 16
# Assumed label standard deviation for UncertaintyAwareConR, in units of std(y_train):
# the benchmark datasets carry no per-sample label errors.
_UACONR_LABEL_SD = 0.1
# Paired (A, B) comparisons reported as mean(|e_A| - |e_B|) with a bootstrap CI.
_PAIRS = (("MSE_ConR", "MSE"), ("MSE_UAConR", "MSE_ConR"), ("MSE_PRIME", "MSE"))
REP_MAX_POINTS = 2000
_REP_K = 10


def _geometry_columns(features: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    """``Rep_*`` columns: representation_geometry_report on <= REP_MAX_POINTS seeded points."""
    features = np.asarray(features, dtype=np.float64).reshape(len(y), -1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) > REP_MAX_POINTS:
        keep = np.sort(np.random.default_rng(seed).choice(len(y), REP_MAX_POINTS, replace=False))
        features, y = features[keep], y[keep]
    k = min(_REP_K, len(y) - 1)
    report = representation_geometry_report(features, y, k=k, readouts=True, seed=seed)
    out = {f"Rep_{key}": float(value) for key, value in report.items()}
    out["Rep_points"] = float(len(y))
    return out


def _make_encoder(in_dim: int, hidden: int) -> tuple[nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, _FEATURE_DIM),
        nn.SiLU(),
    )
    head = nn.Linear(_FEATURE_DIM, 1)
    return encoder, head


class ImbalancedRepresentationSuite(HarnessRunner):
    """PRIME/ConR/group-DRO for deep long-tailed regression."""

    def __init__(self) -> None:
        self._abs_errors: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "imbalanced_representation"

    @property
    def description(self) -> str:
        return (
            "PRIME / ConR / continuous group DRO vs MSE on a long-tailed target "
            "(worst-regime MAE, tail metrics)"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["stress_imbalanced", "california_housing"]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def method_notes(self, method: str) -> str:
        notes = {
            "MSE": "plain MLP + MSE (bulk-dominated baseline)",
            "MSE_GroupDRO": "GroupDROTrainer over 4 equal-width target regimes",
            "MSE_PRIME": "MLP encoder/head + PRIME proxy alignment (class-balanced)",
            "MSE_ConR": "MLP encoder/head + ConR inverse-density contrastive push",
            "MSE_UAConR": (
                "MLP encoder/head + UncertaintyAwareConR (ConR with Gaussian label-noise-softened "
                f"pair memberships, assumed label sd = {_UACONR_LABEL_SD} * std(y_train))"
            ),
            "MSE_BMC": "MLP + BMC (Ren et al. 2022), sigma=0.15*std(y)",
            "MSE_RankSim": "MLP encoder/head + task MSE plus RankSim ranking-similarity regularizer",
            "VIR": "Variational Imbalanced Regression: NIG evidential head + label-density smoothing",
        }
        return notes.get(method, "")

    def _epochs(self, config: HarnessConfig) -> int:
        return min(int(config.epochs), 5) if config.smoke else int(config.epochs)

    def _edges(self, y: torch.Tensor) -> torch.Tensor:
        # Uniform proxy targets are the same interior edges PRIME uses; with
        # G groups they supply the G - 1 interior bin edges.  Build them on CPU
        # (MPS has crashed on tiny cat/linspace combinations under the CLI's
        # default-device setting; the edges are metadata, not model inputs).
        y_cpu = y.detach().cpu().reshape(-1)
        low, high = float(y_cpu.min()), float(y_cpu.max())
        proxies = uniform_proxy_targets(low, high, _N_GROUPS - 1).detach().cpu()
        return torch.cat(
            [
                torch.tensor([low], dtype=torch.float32),
                proxies.to(torch.float32),
                torch.tensor([high], dtype=torch.float32),
            ]
        )

    def _fit_plain(self, X: torch.Tensor, y: torch.Tensor, config: HarnessConfig) -> nn.Module:
        model = _MLP(in_dim=int(X.shape[1]), hidden=_hidden_size(int(X.shape[1])), out_dim=1).to(
            current_device()
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        generator = torch.Generator().manual_seed(int(config.seed) + 3)
        batch_size = min(int(config.batch_size), len(X))
        for _ in range(self._epochs(config)):
            order = torch.randperm(len(X), generator=generator).to(X.device)
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                nn.functional.mse_loss(model(X[idx]), y[idx].unsqueeze(-1)).backward()
                optimizer.step()
        model.eval()
        return model

    def _fit_regularized(
        self,
        method: str,
        X: torch.Tensor,
        y: torch.Tensor,
        config: HarnessConfig,
    ) -> dict[str, Any]:
        device = current_device()
        encoder, head = _make_encoder(int(X.shape[1]), _hidden_size(int(X.shape[1])))
        encoder, head = encoder.to(device), head.to(device)
        if method == "MSE_PRIME":
            regularizer = (
                PrimeLoss(
                    feature_dim=_FEATURE_DIM,
                    n_proxies=32,
                    label_range=(float(y.min()), float(y.max())),
                    class_balanced=True,
                    seed=int(config.seed),
                )
                .fit(y)
                .to(device)
            )
        elif method == "MSE_UAConR":
            label_sd = _UACONR_LABEL_SD * float(y.detach().std().clamp_min(1e-6))
            regularizer = UncertaintyAwareConR(
                label_sigma=label_sd, label_threshold=0.2, push_scale=0.2
            ).fit(y)
        else:
            regularizer = ConR(label_threshold=0.2, push_scale=0.2).fit(y)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()) + list(regularizer.parameters()),
            lr=config.lr,
        )
        generator = torch.Generator().manual_seed(int(config.seed) + 5)
        batch_size = min(int(config.batch_size), len(X))
        for _ in range(self._epochs(config)):
            order = torch.randperm(len(X), generator=generator).to(X.device)
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                features = encoder(X[idx])
                prediction = head(features).squeeze(-1)
                mse = nn.functional.mse_loss(prediction, y[idx])
                if method == "MSE_PRIME":
                    reg = regularizer(features, y[idx])
                    loss = mse + 0.5 * reg
                else:
                    reg = regularizer(features, y[idx], prediction=prediction)
                    loss = mse + 2.0 * reg
                loss.backward()
                optimizer.step()
        encoder.eval()
        head.eval()
        return {"encoder": encoder, "head": head, "edges": self._edges(y.detach())}

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        device = current_device()
        X = torch.as_tensor(np.asarray(splits["X_train"]), dtype=torch.float32, device=device)
        y = torch.as_tensor(
            np.asarray(splits["y_train"]), dtype=torch.float32, device=device
        ).reshape(-1)
        if method == "MSE":
            return {
                "type": "plain",
                "model": self._fit_plain(X, y, config),
                "edges": self._edges(y.detach().cpu()),
            }
        if method == "MSE_GroupDRO":
            trainer = GroupDROTrainer(
                model_factory=lambda: _MLP(
                    in_dim=int(X.shape[1]),
                    hidden=_hidden_size(int(X.shape[1])),
                    out_dim=1,
                ),
                group_edges=_N_GROUPS,
                base_loss="mse",
                epochs=self._epochs(config),
                lr=config.lr,
                batch_size=min(int(config.batch_size), len(X)),
                seed=int(config.seed),
            )
            trainer.fit(X.detach().cpu(), y.detach().cpu())
            edges = self._edges(y.detach().cpu())
            return {"type": "groupdro", "trainer": trainer, "edges": edges}
        if method in {"MSE_PRIME", "MSE_ConR", "MSE_UAConR"}:
            state = self._fit_regularized(method, X, y, config)
            return {"type": "regularized", **state}
        if method == "MSE_BMC":
            model = _MLP(
                in_dim=int(X.shape[1]), hidden=_hidden_size(int(X.shape[1])), out_dim=1
            ).to(device)
            # BMC needs a target-noise scale.  The paper learns it jointly;
            # this benchmark uses a fixed normalized scale (0.25 * std(y)) so
            # the row is comparable across datasets.  A target-scale proxy or
            # a linear-residual MAD is too large here and over-smooths the
            # batch softmax (measured degradation documented in the results).
            sigma = max(0.05, 0.15 * float(y.detach().std().clamp_min(1e-6)))
            loss_fn = BalancedMSEBMC(sigma=sigma).fit(y.detach().cpu())
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            generator = torch.Generator().manual_seed(int(config.seed) + 7)
            batch_size = min(int(config.batch_size), len(X))
            for _ in range(self._epochs(config)):
                order = torch.randperm(len(X), generator=generator).to(device)
                for start in range(0, len(X), batch_size):
                    idx = order[start : start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(X[idx]).squeeze(-1), y[idx])
                    loss.backward()
                    optimizer.step()
            model.eval()
            return {"type": "plain", "model": model, "edges": self._edges(y.detach().cpu())}
        if method == "MSE_RankSim":
            encoder, head = _make_encoder(int(X.shape[1]), _hidden_size(int(X.shape[1])))
            encoder, head = encoder.to(device), head.to(device)
            regularizer = RankSimLoss(gamma=1.0).to(device)
            optimizer = torch.optim.Adam(
                list(encoder.parameters()) + list(head.parameters()), lr=config.lr
            )
            generator = torch.Generator().manual_seed(int(config.seed) + 8)
            batch_size = min(int(config.batch_size), len(X))
            for _ in range(self._epochs(config)):
                order = torch.randperm(len(X), generator=generator).to(device)
                for start in range(0, len(X), batch_size):
                    idx = order[start : start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    features = encoder(X[idx])
                    prediction = head(features).squeeze(-1)
                    loss = torch.nn.functional.mse_loss(prediction, y[idx])
                    loss = loss + regularizer(features, y[idx])
                    loss.backward()
                    optimizer.step()
            encoder.eval()
            head.eval()
            return {
                "type": "regularized",
                "encoder": encoder,
                "head": head,
                "edges": self._edges(y.detach().cpu()),
            }
        if method == "VIR":
            encoder = nn.Sequential(
                nn.Linear(int(X.shape[1]), _hidden_size(int(X.shape[1]))),
                nn.SiLU(),
                nn.Linear(_hidden_size(int(X.shape[1])), _FEATURE_DIM),
                nn.SiLU(),
            )
            regressor = VIRRegressor(
                encoder,
                n_bins=12,
                epochs=self._epochs(config),
                batch_size=min(int(config.batch_size), len(X)),
                lr=config.lr,
                seed=int(config.seed),
                device=str(device),
            )
            regressor.fit(X, y)
            return {
                "type": "vir",
                "regressor": regressor,
                "edges": self._edges(y.detach().cpu()),
            }
        raise ValueError(f"Unknown method: {method}")

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        device = current_device()
        kind = model_state["type"]
        X_t = torch.as_tensor(np.asarray(X_test), dtype=torch.float32, device=device)
        if kind == "plain":
            model = model_state["model"]
            with torch.no_grad():
                prediction = model(X_t)
                features = model.net[:-1](X_t)
            predictions: dict[str, Any] = {
                "y_pred": np.asarray(prediction.detach().cpu()).reshape(-1)
            }
        elif kind == "vir":
            regressor = model_state["regressor"]
            mean, std = regressor.predict(X_t, return_std=True)
            with torch.no_grad():
                encoder = regressor.model_.encoder  # the trained copy (fit deep-copies)
                features = encoder(X_t.to(next(encoder.parameters()).device))
            predictions = {
                "y_pred": np.asarray(mean.detach().cpu()).reshape(-1),
                "mean": np.asarray(mean.detach().cpu()).reshape(-1),
                "std": np.asarray(std.detach().cpu()).reshape(-1),
            }
        elif kind == "groupdro":
            trainer = model_state["trainer"]
            prediction = trainer.predict(torch.as_tensor(np.asarray(X_test), dtype=torch.float32))
            predictions = {"y_pred": np.asarray(prediction).reshape(-1)}
            net = trainer.model
            with torch.no_grad():
                features = net.net[:-1](X_t.to(next(net.parameters()).device))
        else:
            encoder = model_state["encoder"]
            head = model_state["head"]
            with torch.no_grad():
                features = encoder(X_t)
                prediction = head(features).squeeze(-1)
            predictions = {"y_pred": np.asarray(prediction.detach().cpu()).reshape(-1)}
        predictions["_group_edges"] = np.asarray(model_state["edges"], dtype=np.float64)
        predictions["_features"] = features.detach().cpu().double().numpy()
        return predictions

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        edges_standardized = np.asarray(predictions.pop("_group_edges"), dtype=np.float64)
        features = predictions.pop("_features")
        y_scaler = predictions.get("_y_scaler")
        if y_scaler is not None:
            predictions = _inverse_transform_predictions(predictions, y_scaler)
            y_true = np.asarray(_inverse_transform_y(y_test, y_scaler)).reshape(-1)
            scale = float(np.asarray(y_scaler.scale_).reshape(-1)[0])
            shift = float(np.asarray(y_scaler.mean_).reshape(-1)[0])
            edges = edges_standardized * scale + shift
        else:
            y_true = np.asarray(y_test).reshape(-1)
            edges = edges_standardized
        result = compute_all_metrics(predictions, y_true)
        y_pred = np.asarray(predictions["y_pred"]).reshape(-1)
        group_mae = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (y_true >= lo) & (y_true < hi)
            if mask.sum() >= 5:
                group_mae.append(float(np.mean(np.abs(y_pred[mask] - y_true[mask]))))
        result["WorstGroupMAE"] = float(max(group_mae)) if group_mae else float("nan")
        result.update(_geometry_columns(features, y_true, int(config.seed)))
        self._abs_errors[method] = np.abs(y_pred - y_true)
        return result

    # ------------------------------------------------------------------
    # run: method rows + raw-input geometry reference + paired comparisons
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: SplitsDict | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        cfg = config or HarnessConfig()
        dataset_name, splits = self._prepare_scenario_run(
            scenario_name, dataset_name, splits, standardize_targets=cfg.standardize_targets
        )
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name, standardize_targets=cfg.standardize_targets)
        assert splits is not None
        self._abs_errors = {}
        rows = super().run(dataset_name, cfg, splits=splits)
        rows.append(self._raw_input_row(dataset_name, splits, cfg))
        rows.extend(self._paired_rows(dataset_name, cfg))
        return rows

    def _base_row(self, method: str, dataset_name: str) -> dict[str, Any]:
        return {
            "Method": f"{self.name}/{method}",
            "Library": "torchregress",
            "LibraryVersion": "",
            "Dataset": dataset_name,
            "Suite": self.name,
            "train_s": None,
            "eval_s": None,
            "Device": str(current_device()),
        }

    def _raw_input_row(
        self, dataset_name: str, splits: SplitsDict, config: HarnessConfig
    ) -> dict[str, Any]:
        """representation_geometry_report on the raw test inputs (no-learning reference)."""
        X_test = np.asarray(splits["X_test"], dtype=np.float64)
        y_test = np.asarray(splits["y_test"], dtype=np.float64).reshape(-1)
        if config.smoke:  # same test subset as HarnessRunner.run
            X_test, y_test = X_test[: min(50, len(X_test))], y_test[: min(50, len(y_test))]
        y_scaler = splits.get("_y_scaler")
        if y_scaler is not None:
            y_test = np.asarray(_inverse_transform_y(y_test, y_scaler)).reshape(-1)
        row = self._base_row("rep_geometry/raw_inputs", dataset_name)
        try:
            row.update(_geometry_columns(X_test, y_test, int(config.seed)))
            row["Notes"] = "representation_geometry_report on the raw test inputs (reference)"
        except Exception as exc:  # noqa: BLE001 - reported as a failed row, never dropped
            row["Notes"] = f"failed: {type(exc).__name__}: {exc}"
        return row

    def _paired_rows(self, dataset_name: str, config: HarnessConfig) -> list[dict[str, Any]]:
        rows = []
        for a, b in _PAIRS:
            if a not in self.methods() or b not in self.methods():
                continue
            row = self._base_row(f"paired_abs_error/{a}-vs-{b}", dataset_name)
            missing = [m for m in (a, b) if m not in self._abs_errors]
            if missing:
                row["Notes"] = f"failed: no test predictions for {', '.join(missing)} (row failed)"
                rows.append(row)
                continue
            diff, lo, hi = paired_bootstrap_ci(
                self._abs_errors[a],
                self._abs_errors[b],
                n_boot=2000,
                alpha=0.05,
                seed=int(config.seed),
            )
            row.update({"MAE_Diff": diff, "MAE_Diff_Lo": lo, "MAE_Diff_Hi": hi})
            row["Notes"] = (
                f"mean(|e_{a}| - |e_{b}|) on {len(self._abs_errors[a])} shared test points; "
                "95% paired percentile bootstrap (paired_bootstrap_ci, 2000 resamples); "
                "test-sample uncertainty only, no retraining variation"
            )
            rows.append(row)
        return rows


__all__ = ["ImbalancedRepresentationSuite"]
