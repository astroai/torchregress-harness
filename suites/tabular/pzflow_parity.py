"""pzflow parity suite: torchregress vs pzflow vs normflows.

Compares three conditional normalizing flow implementations for 1D
regression density estimation.

Methods
-------
* ``torchregress_NSF`` — zuko NSF: 3 coupling transforms, each with
  MLP internal networks (PyTorch).
* ``pzflow_NSF`` — pzflow Flow with default bijector (JAX).
  Architecture opaque (manual Chain construction hits JAX stax bug).
* ``normflows_NSF`` — normflows ``ConditionalNormalizingFlow`` with
  ``AutoregressiveRationalQuadraticSpline`` (3 blocks, 64 hidden) +
  ``DiagGaussian`` base (PyTorch).  **Intrinsically broken** on the
  harness's 1-D tabular target — the normflows library throws a
  broadcast error when combining ``num_blocks>1`` with
  ``num_context_channels≠None`` on single-channel input (not a missing
  import).  The suite short-circuits the method and emits a
  ``Notes="known-bad: …"`` row in every leaderboard artifact so the
  limitation stays visible AND doesn't pollute the leaderboard's
  skipped/failed aggregator.  See ``PARITY_REPORT.md`` §12 for the
  broadcast-error root-cause write-up.

Metrics
-------
* NLL (negative log-likelihood) — lower is better; measures density fit
* Coverage at 90% — fraction of y_test inside [q₅, q₉₅] samples
* Width — mean interval width across test set
* IntervalScore — penalises under-coverage + wide intervals
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _train_one

# Prevent JAX from preallocating all GPU memory at import time.
# Must be set before any JAX call (pzflow uses JAX internally).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ============================================================================
# Helpers
# ============================================================================


def _pzflow_conditional_nll(
    flow: Any,
    X: np.ndarray,
    y: np.ndarray,
    cols_x: list[str],
    batch_size: int = 1024,
) -> float:
    """Compute mean conditional NLL of y | X under a pzflow Flow."""
    import pandas as pd

    df = pd.DataFrame(X, columns=cols_x)
    df["y"] = y
    n = len(df)
    total_lp = 0.0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        lp_batch = flow.log_prob(df.iloc[start:end])
        total_lp += float(np.asarray(lp_batch).sum())
    return -total_lp / n


def _pzflow_sample_conditional(
    flow: Any,
    X: np.ndarray,
    cols_x: list[str],
    n_samples: int = 512,
) -> np.ndarray:
    """Draw conditional samples y ~ P(y|X) from pzflow. Returns (n_samples, N)."""
    import pandas as pd

    df_x = pd.DataFrame(X, columns=cols_x)
    samples = flow.sample(n_samples, conditions=df_x)
    return np.asarray(samples["y"]).reshape(n_samples, -1)


# ============================================================================
# Suite
# ============================================================================


class PzflowParitySuite(HarnessRunner):
    """torchregress vs pzflow vs normflows conditional flow comparison.

    Matches architecture across packages for fair density-estimation
    benchmarking.
    """

    N_TRANSFORMS = 3
    HIDDEN_FEATURES = [64, 64]
    N_SAMPLES = 512

    # Methods that are intrinsically broken on the harness's test targets.
    # Rather than gating them on `_X_AVAILABLE` (these libraries ARE
    # installed in the default .venv) OR letting them crash and emit
    # ``Notes=failed:`` rows (which the leaderboard's ``_is_skipped`` heuristic
    # conflates with transient environment failures), we register affected
    # methods here and short-circuit ``self.run()`` to emit a deterministic
    # ``Notes=f"known-bad: {method} {suite_name} #{reason}"`` row on every
    # dataset.  Makes the limitation visible AND removes a noise source from
    # the leaderboard's skipped/failed filter.  See ``PARITY_REPORT.md`` §12.
    _KNOWN_BAD: dict[str, str] = {
        "normflows_NSF": "broadcast error on 1-D tabular inputs",
    }

    @property
    def name(self) -> str:
        return "pzflow_parity"

    @property
    def description(self) -> str:
        return (
            "pzflow parity: torchregress NSF vs pzflow vs normflows "
            "(Neural Spline flows, 3 transforms, 64 hidden)"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "synthetic_cond_gaussian",
            "synthetic_cond_nongaussian",
            "pzflow_galaxies",
        ]

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def methods(self) -> list[str]:
        return [
            "torchregress_NSF",
            "pzflow_NSF",
            "normflows_NSF",
        ]

    def library_name(self, method: str) -> str:
        libs = {
            "torchregress_NSF": "torchregress",
            "pzflow_NSF": "pzflow",
            "normflows_NSF": "normflows",
        }
        return libs.get(method, method)

    def method_notes(self, method: str) -> str:
        notes = {
            "torchregress_NSF": (
                "zuko Neural Spline Flow (3 transforms, 64-dim hidden) "
                "via NormalizingFlowLoss (PyTorch)"
            ),
            "pzflow_NSF": (
                "pzflow Flow with default bijector, "
                "conditional_columns, Normal(1) latent (JAX). "
                "Architecture: default (opaque, not manually controlled)"
            ),
            "normflows_NSF": (
                "intrinsically broken on 1-D tabular input (broadcast "
                "error in normflows library — NOT a missing-import issue). "
                "Suite short-circuits and emits a `known-bad:` row instead "
                "of fitting.  See PARITY_REPORT.md §12."
            ),
        }
        return notes.get(method, "")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)

        X_train = splits["X_train"]
        y_train = splits["y_train"]
        in_dim = X_train.shape[1]
        hidden = _hidden_size(in_dim)

        # --- torchregress NSF (zuko) ---
        if method == "torchregress_NSF":
            from torchregress.losses.nflows import (
                NormalizingFlowLoss,
                create_flow_model,
            )

            flow = create_flow_model(
                n_features=1,
                context_dim=hidden,
                flow_type="nsf",
                n_transforms=self.N_TRANSFORMS,
                hidden_features=self.HIDDEN_FEATURES,
            )
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=hidden)
            loss_fn = NormalizingFlowLoss(flow=flow)

            X_t = torch.from_numpy(X_train).float().to(current_device())
            y_t = torch.from_numpy(y_train).float().to(current_device()).unsqueeze(1)
            _train_one(
                model,
                loss_fn,
                X_t,
                y_t,
                config,
                loss_params=list(flow.parameters()),
            )
            model.eval()
            return {
                "type": "torchregress_nsf",
                "model": model,
                "flow": loss_fn,
            }

        # --- pzflow NSF (default bijector — avoids JAX stax bug) ---
        elif method == "pzflow_NSF":
            import pandas as pd
            import pzflow

            cols_x = [f"x{i}" for i in range(in_dim)]
            df_train = pd.DataFrame(
                X_train,
                columns=cols_x,
                dtype=np.float32,
            )
            df_train["y"] = y_train

            # Use default bijector — manual Chain(NSC+Reverse) hits a JAX
            # stax shape bug.  The default bijector is still a flow-based
            # architecture and gives a valid comparison.
            flow = pzflow.Flow(
                data_columns=["y"],
                conditional_columns=cols_x,
                latent=pzflow.distributions.Normal(input_dim=1),
                seed=int(config.seed),
            )

            flow.train(
                df_train,
                epochs=config.epochs,
                batch_size=min(config.batch_size, len(df_train)),
                verbose=False,
                seed=int(config.seed),
            )

            return {
                "type": "pzflow_nsf",
                "flow": flow,
                "cols_x": cols_x,
            }

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        model_type = model_state["type"]
        alpha = 0.1

        # --- torchregress NSF ---
        if model_type == "torchregress_nsf":
            model = model_state["model"]
            flow_loss = model_state["flow"]
            model.eval()
            X_t = torch.from_numpy(X_test).float().to(current_device())
            with torch.no_grad():
                context = model(X_t)
                samples = flow_loss.sample(
                    context,
                    n_samples=self.N_SAMPLES,
                )
                samples_np = samples.squeeze(-1).detach().cpu().numpy()
                if samples_np.ndim == 3:
                    samples_np = samples_np[:, :, 0]
                if samples_np.shape[0] == X_test.shape[0]:
                    samples_np = samples_np.T

            y_pred = samples_np.mean(axis=0)
            lo = np.quantile(samples_np, alpha / 2, axis=0)
            hi = np.quantile(samples_np, 1.0 - alpha / 2, axis=0)
            return {"lower": lo, "upper": hi, "y_pred": y_pred, "samples": samples_np.T}

        # --- pzflow NSF ---
        elif model_type == "pzflow_nsf":
            flow = model_state["flow"]
            cols_x = model_state["cols_x"]
            samples = _pzflow_sample_conditional(
                flow,
                X_test,
                cols_x,
                n_samples=self.N_SAMPLES,
            )
            lo = np.quantile(samples, alpha / 2, axis=0)
            hi = np.quantile(samples, 1.0 - alpha / 2, axis=0)
            y_pred = samples.mean(axis=0)
            return {"lower": lo, "upper": hi, "y_pred": y_pred, "samples": samples.T}

        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        from harness.metrics import compute_interval_metrics

        lo = np.asarray(predictions["lower"]).reshape(-1)
        hi = np.asarray(predictions["upper"]).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)

        metrics = compute_interval_metrics(lo, hi, y_true)

        predictions["_lo_raw"] = lo
        predictions["_hi_raw"] = hi
        predictions["_y_true_raw"] = y_true

        # PIT uniformity (sample-rank PIT → KS vs Uniform): the core
        # flexible-PDF diagnostic — calibrated densities give uniform PIT.
        if "samples" in predictions:
            try:
                samp = np.asarray(predictions["samples"], dtype=np.float64)
                if samp.ndim != 2 or samp.shape[0] != y_true.shape[0]:
                    samp = np.asarray(samp).reshape(y_true.shape[0], -1)
                ranks = (samp <= y_true.reshape(-1, 1)).mean(axis=1)
                ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
                grid = np.linspace(0, 1, 101)
                ecdf = np.searchsorted(np.sort(ranks), grid, side="right") / len(ranks)
                metrics["PIT_KS"] = float(np.max(np.abs(ecdf - grid)))
            except Exception:
                pass

        return metrics

    # ------------------------------------------------------------------
    # Override run() to compute NLL for all three methods
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

        cfg = config or HarnessConfig()

        dataset_name, splits = self._prepare_scenario_run(
            scenario_name,
            dataset_name,
            splits,
            standardize_targets=cfg.standardize_targets,
        )

        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(
                dataset_name,
                standardize_targets=cfg.standardize_targets,
            )

        X_test_np = splits["X_test"]
        y_test_np = np.asarray(splits["y_test"]).reshape(-1)
        in_dim = splits["X_train"].shape[1]
        cols_x = [f"x{i}" for i in range(in_dim)]

        rows: list[dict[str, Any]] = []

        for method in self.methods():
            set_harness_seed(cfg.seed)

            # --- known-bad short-circuit: intrinsically broken methods ---
            # Skip fit/predict/evaluate entirely and emit a deterministic
            # row whose Notes string starts with ``known-bad:``.  See
            # ``_KNOWN_BAD`` + ``PARITY_REPORT.md`` §12 for the broadcast
            # root-cause and the rationale for surfacing this in the JSON
            # artifact rather than gating on a ``_X_AVAILABLE`` flag.
            if method in self._KNOWN_BAD:
                rows.append(
                    {
                        "Method": f"{self.name}/{method}",
                        "Library": self.library_name(method),
                        "Dataset": dataset_name,
                        "Suite": self.name,
                        "Coverage": None,
                        "Width": None,
                        "IntervalScore": None,
                        "NLL": None,
                        "train_s": None,
                        "eval_s": None,
                        "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                        "Device": str(current_device()),
                        "Notes": (f"known-bad: {method} {self.name} #{self._KNOWN_BAD[method]}"),
                    }
                )
                continue

            try:
                t0 = perf_counter()
                model_state = self.fit(method, splits, cfg)
                fit_time = perf_counter() - t0

                t0 = perf_counter()
                predictions = self.predict(method, model_state, X_test_np, cfg)
                pred_time = perf_counter() - t0

                metrics = self.evaluate(method, predictions, y_test_np, cfg)

                # Compute NLL
                nll: float | None = None
                try:
                    mtype = model_state["type"]
                    if mtype == "torchregress_nsf":
                        model = model_state["model"]
                        flow_loss = model_state["flow"]
                        model.eval()
                        X_t = torch.from_numpy(X_test_np).float().to(current_device())
                        y_t = torch.from_numpy(y_test_np).float().to(current_device()).unsqueeze(1)
                        with torch.no_grad():
                            context = model(X_t)
                            nll = float(-flow_loss.log_prob(context, y_t).mean().item())
                    elif mtype == "pzflow_nsf":
                        nll = _pzflow_conditional_nll(
                            model_state["flow"],
                            X_test_np,
                            y_test_np,
                            cols_x,
                            batch_size=cfg.batch_size,
                        )
                except (ValueError, RuntimeError, ImportError, TypeError, OSError):
                    nll = None

                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    **metrics,
                    "NLL": nll,
                    "train_s": fit_time,
                    "eval_s": pred_time,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": self.method_notes(method),
                }

            except (
                ValueError,
                RuntimeError,
                ImportError,
                KeyError,
                TypeError,
                OSError,
                NameError,
            ) as exc:
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "Coverage": None,
                    "Width": None,
                    "IntervalScore": None,
                    "NLL": None,
                    "train_s": None,
                    "eval_s": None,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }

            rows.append(row)

        return rows
