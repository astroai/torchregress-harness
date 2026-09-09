"""Streaming test-time adaptation suite (COSA delayed labels).

Protocol: an MLP point model (+ train-residual std) serves test rows in
streaming order. Static predicts i.i.d.; COSAAdapted wraps the model in
DelayedLabelResidualAdapter, revealing labels with a one-chunk delay and
partial_fit-ing the residual/variance EMAs. Metrics: RMSE + Gaussian
interval coverage/width — adaptation should track drift the static model
cannot see.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchregress.losses import WeightedMSELoss

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _train_one

_CHUNK = 32


class _BatchBase:
    """Callable base exposing predict_distribution for the adapter."""

    def __init__(self, model: torch.nn.Module, std: float) -> None:
        from torchregress.prediction import PredictiveBatch

        self._PB = PredictiveBatch
        self.model = model
        self.std = std

    def predict_distribution(self, X: Any) -> Any:
        if not torch.is_tensor(X):
            X = torch.from_numpy(np.asarray(X)).float()
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            mean = self.model(X.to(dev)).reshape(-1).detach().cpu()
        n = mean.shape[0]
        return self._PB(
            point=mean.clone(),
            mean=mean.clone(),
            std=torch.full((n,), self.std),
        )


class StreamingTTASuite(HarnessRunner):
    """Static vs COSA delayed-label adaptation under streaming."""

    @property
    def name(self) -> str:
        return "streaming_tta"

    @property
    def description(self) -> str:
        return "Streaming TTA: static vs delayed-label residual adaptation"

    @property
    def supported_datasets(self) -> list[str]:
        return ["diabetes", "california_housing", "folktables_income"]

    def methods(self) -> list[str]:
        return [
            "Static",
            "COSAAdapted",
            "CausalHarnessCOSA",
            "CausalHarnessCOSA_Unlabeled",
            "CausalHarnessCOSA_Delay2",
        ]

    def library_name(self, method: str) -> str:
        return "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "Static": "MLP point + train-residual std, no adaptation (ablation)",
            "COSAAdapted": "DelayedLabelResidualAdapter, 1-chunk label delay",
            "CausalHarnessCOSA": "CausalTTAHarness + DelayedLabelResidualAdapter, 1-chunk delay",
            "CausalHarnessCOSA_Unlabeled": "CausalTTAHarness + adapter, no labels ever (delay=-1)",
            "CausalHarnessCOSA_Delay2": "CausalTTAHarness + adapter, 2-chunk label delay",
        }.get(method, "")

    # ------------------------------------------------------------------
    # fit — shared base model (+ adapter for COSAAdapted)
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        X_train_np = splits["X_train"]
        y_train_np = splits["y_train"]
        in_dim = X_train_np.shape[1]
        hidden = _hidden_size(in_dim)
        model = _MLP(in_dim, hidden, out_dim=1)
        _train_one(
            model,
            WeightedMSELoss(),
            torch.from_numpy(X_train_np).float().to(current_device()),
            torch.from_numpy(y_train_np).float().to(current_device()).unsqueeze(1),
            config,
        )
        model.eval()
        with torch.no_grad():
            p_tr = (
                model(torch.from_numpy(X_train_np).float().to(current_device()))
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
        resid_std = float(np.std(np.asarray(y_train_np).reshape(-1) - p_tr)) + 1e-6
        base = _BatchBase(model, resid_std)
        state: dict[str, Any] = {"type": "point", "model": model, "base": base}
        if method in (
            "COSAAdapted",
            "CausalHarnessCOSA",
            "CausalHarnessCOSA_Unlabeled",
            "CausalHarnessCOSA_Delay2",
        ):
            from torchregress.test_time.cosa import DelayedLabelResidualAdapter

            state["adapter"] = DelayedLabelResidualAdapter(base)
        elif method != "Static":
            raise ValueError(f"Unknown method: {method}")
        return state

    # ------------------------------------------------------------------
    # predict — chunked streaming; COSA reveals prior-chunk labels
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        del config
        X = np.asarray(X_test)
        n = len(X)
        means, stds = [], []
        if method == "Static":
            base = model_state["base"]
            with torch.no_grad():
                for s in range(0, n, _CHUNK):
                    pb = base.predict_distribution(X[s : s + _CHUNK])
                    means.append(np.asarray(pb.mean).reshape(-1))
                    stds.append(np.asarray(pb.std).reshape(-1))
        elif method == "COSAAdapted":
            from torchregress.test_time.cosa import DelayedLabelResidualAdapter

            adapter: DelayedLabelResidualAdapter = model_state["adapter"]
            # NOTE: true labels are unavailable in predict(); the runner
            # passes y_test only to evaluate(). Streaming with delay needs
            # labels mid-predict, so y_test rides the model_state channel
            # (fit stashes nothing; run() merges splits into state below).
            # See evaluate(): the actual delayed update happens there on
            # identical chunking; predict() emits static-equivalent output
            # only if labels are absent.
            y_full = model_state.get("y_test_full")
            with torch.no_grad():
                for t, s in enumerate(range(0, n, _CHUNK)):
                    pb = adapter.predict_distribution(X[s : s + _CHUNK])
                    means.append(np.asarray(pb.mean).reshape(-1))
                    stds.append(np.asarray(pb.std).reshape(-1))
                    if y_full is not None and t > 0:
                        ps = max(0, s - _CHUNK)
                        adapter.partial_fit(X[ps:s], np.asarray(y_full).reshape(-1)[ps:s])
        else:
            raise ValueError(f"Unknown method: {method}")
        mean = np.concatenate(means).reshape(-1)
        std = np.concatenate(stds).reshape(-1).clip(min=1e-8)
        z = 1.645
        return {
            "y_pred": mean,
            "mean": mean,
            "std": std,
            "lower": mean - z * std,
            "upper": mean + z * std,
        }

    # ------------------------------------------------------------------
    # Custom run(): merge y_test into state so streaming sees labels
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

        from harness.metrics import compute_interval_metrics

        cfg = config or HarnessConfig()
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name)
        assert splits is not None
        y_test = np.asarray(splits["y_test"]).reshape(-1)
        # Smoke subset (mirror base-runner semantics).
        if cfg.smoke:
            n_te = min(50, len(y_test))
            X_te = np.asarray(splits["X_test"])[:n_te]
            y_te = y_test[:n_te]
        else:
            X_te, y_te = np.asarray(splits["X_test"]), y_test

        rows: list[dict[str, Any]] = []
        for method in self.methods():
            set_harness_seed(cfg.seed)
            try:
                t0 = perf_counter()
                state = self.fit(method, splits, cfg)
                fit_time = perf_counter() - t0
                state["y_test_full"] = y_te
                t0 = perf_counter()
                if method.startswith("CausalHarnessCOSA"):
                    from torchregress.test_time import CausalTTAHarness

                    delay = {"CausalHarnessCOSA": 1}.get(method, 1)
                    if method.endswith("_Unlabeled"):
                        delay = -1
                    elif method.endswith("_Delay2"):
                        delay = 2
                    causal = CausalTTAHarness(delay=delay, alpha=0.1)
                    causal_metrics = causal.evaluate(
                        state["adapter"],
                        torch.from_numpy(X_te.astype(np.float32)),
                        torch.from_numpy(y_te.astype(np.float32)),
                        batch_size=_CHUNK,
                    )
                    pred = None
                else:
                    pred = self.predict(method, state, X_te, cfg)
                pred_time = perf_counter() - t0
                if pred is None:
                    metrics = {
                        "RMSE": float(causal_metrics["RMSE"]),
                        "MAE": float(causal_metrics["MAE"]),
                        "Coverage": float(causal_metrics["coverage"]),
                        "Width": float(causal_metrics["mean_width"]),
                        "IntervalScore": float(causal_metrics["winkler_score"]),
                        "NLL": float(causal_metrics.get("NLL", float("nan"))),
                        "Delay": float(delay),
                    }
                else:
                    metrics = compute_interval_metrics(
                        np.asarray(pred["lower"]).reshape(-1),
                        np.asarray(pred["upper"]).reshape(-1),
                        y_te.reshape(-1),
                    )
                    # Point accuracy on the same stream (enables AdaptGain
                    # vs the static baseline below).
                    yp = np.asarray(pred["y_pred"]).reshape(-1)
                    yt = y_te.reshape(-1)
                    metrics["RMSE"] = float(np.sqrt(np.mean((yp - yt) ** 2)))
                    metrics["MAE"] = float(np.mean(np.abs(yp - yt)))
                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    **metrics,
                    "train_s": fit_time,
                    "eval_s": pred_time,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": self.method_notes(method),
                }
            except Exception as exc:  # noqa: BLE001 — row-level resilience
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "Coverage": None,
                    "train_s": None,
                    "eval_s": None,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }
            rows.append(row)
        # Adaptation gain vs the static baseline at the same seed: positive
        # means the delayed-label adapter beat no adaptation on RMSE.
        static_rmse: float | None = None
        for row in rows:
            if row.get("Method") == f"{self.name}/Static" and row.get("RMSE") is not None:
                try:
                    static_rmse = float(row["RMSE"])
                except (TypeError, ValueError):
                    static_rmse = None
                break
        if static_rmse is not None:
            for row in rows:
                if row.get("RMSE") is None or row.get("Method") == f"{self.name}/Static":
                    continue
                try:
                    row["AdaptGain"] = float(static_rmse - float(row["RMSE"]))
                except (TypeError, ValueError):
                    continue
        return rows
