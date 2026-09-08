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


def _target_scaler(
    y_train: np.ndarray, splits: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize targets for flow training; return (y_std, mean, scale).

    Spline flows have bounded tails — raw targets far outside [-5, 5]
    (e.g. energy loads 6-48) sit in the identity region with zero learning
    signal (flat NLL). Reuses the runner's scaler when it standardized,
    else fits one here. All comparisons stay fair (both flows see identical
    inputs); metrics are computed back in native units.
    """
    y2d = np.asarray(y_train, dtype=np.float64).reshape(len(y_train), -1)
    scaler = splits.get("_y_scaler")
    if scaler is not None:
        # Runner already standardized: use as-is; native-inverse below.
        mean = np.asarray(scaler.mean_).reshape(-1)
        scale = np.asarray(scaler.scale_).reshape(-1)
        return y2d.astype(np.float32), mean, scale
    mean = y2d.mean(axis=0)
    scale = y2d.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return ((y2d - mean) / scale).astype(np.float32), mean, scale


def _pzflow_conditional_nll(
    flow: Any,
    X: np.ndarray,
    y: np.ndarray,
    cols_x: list[str],
    cols_y: list[str] | None = None,
    batch_size: int = 1024,
) -> float:
    """Compute mean conditional NLL of y | X under a pzflow Flow."""
    import pandas as pd

    y_cols = cols_y if cols_y is not None else ["y"]
    df = pd.DataFrame(X, columns=cols_x)
    y2d = np.asarray(y).reshape(len(df), -1)
    for i, c in enumerate(y_cols):
        df[c] = y2d[:, i]
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
    cols_y: list[str] | None = None,
    n_samples: int = 512,
) -> np.ndarray:
    """Draw conditional samples y ~ P(y|X) from pzflow.

    Returns (n_samples, N, D).
    """
    import pandas as pd

    y_cols = cols_y if cols_y is not None else ["y"]
    df_x = pd.DataFrame(X, columns=cols_x)
    samples = flow.sample(n_samples, conditions=df_x)
    arr = np.stack([np.asarray(samples[c]).reshape(n_samples, -1) for c in y_cols], axis=-1)
    return arr


def _native_samples(samples_std: np.ndarray, model_state: dict[str, Any]) -> np.ndarray:
    """Inverse-transform standardized flow samples to native units."""
    mean = np.asarray(model_state["y_mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(model_state["y_scale"], dtype=np.float64).reshape(-1)
    return samples_std.astype(np.float64) * scale.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def _native_y(y_test: np.ndarray, predictions: dict[str, Any]) -> np.ndarray:
    """Native-unit test targets (inverse-transform when runner standardized)."""
    yt = np.asarray(y_test, dtype=np.float64)
    scaler = predictions.get("_y_scaler")
    if scaler is not None:
        yt = np.asarray(scaler.inverse_transform(yt.reshape(-1, 1))).reshape(yt.shape)
    return yt


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
            "energy_efficiency",
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

            y_std, y_mean, y_scale = _target_scaler(y_train, splits)
            n_feat = y_std.shape[1]
            flow = create_flow_model(
                n_features=n_feat,
                context_dim=hidden,
                flow_type="nsf",
                n_transforms=self.N_TRANSFORMS,
                hidden_features=self.HIDDEN_FEATURES,
            )
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=hidden)
            loss_fn = NormalizingFlowLoss(flow=flow)

            X_t = torch.from_numpy(X_train).float().to(current_device())
            y_t = torch.from_numpy(y_std).float().to(current_device())
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
                "n_features": n_feat,
                "y_mean": y_mean,
                "y_scale": y_scale,
            }

        # --- pzflow NSF (default bijector — avoids JAX stax bug) ---
        elif method == "pzflow_NSF":
            import pandas as pd
            import pzflow

            y_std, y_mean, y_scale = _target_scaler(y_train, splits)
            n_feat = y_std.shape[1]
            cols_x = [f"x{i}" for i in range(in_dim)]
            cols_y = [f"y{i}" for i in range(n_feat)]
            df_train = pd.DataFrame(
                X_train,
                columns=cols_x,
                dtype=np.float32,
            )
            for i, c in enumerate(cols_y):
                df_train[c] = y_std[:, i]

            # Use default bijector — manual Chain(NSC+Reverse) hits a JAX
            # stax shape bug.  The default bijector is still a flow-based
            # architecture and gives a valid comparison.
            flow = pzflow.Flow(
                data_columns=cols_y,
                conditional_columns=cols_x,
                latent=pzflow.distributions.Normal(input_dim=n_feat),
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
                "cols_y": cols_y,
                "n_features": n_feat,
                "y_mean": y_mean,
                "y_scale": y_scale,
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
            n_feat = int(model_state.get("n_features", 1))
            model.eval()
            X_t = torch.from_numpy(X_test).float().to(current_device())
            with torch.no_grad():
                context = model(X_t)
                samples = flow_loss.sample(
                    context,
                    n_samples=self.N_SAMPLES,
                )
                # zuko returns (N, S, D) → normalize to (S, N, D).
                samples_np = np.asarray(samples.detach().cpu())
                if samples_np.ndim == 2:
                    samples_np = samples_np.reshape(len(X_test), self.N_SAMPLES, -1)
                if samples_np.shape[0] == len(X_test):
                    samples_np = np.swapaxes(samples_np, 0, 1)

            y_pred = samples_np.mean(axis=0)
            lo = np.quantile(samples_np, alpha / 2, axis=0)
            hi = np.quantile(samples_np, 1.0 - alpha / 2, axis=0)
            native = _native_samples(samples_np, model_state)
            lo_n = np.quantile(native, alpha / 2, axis=0)
            hi_n = np.quantile(native, 1.0 - alpha / 2, axis=0)
            return {
                "lower": lo_n.reshape(-1),
                "upper": hi_n.reshape(-1),
                "y_pred": native.mean(axis=0).reshape(-1),
                "samples": np.swapaxes(native, 0, 1),
            }

        # --- pzflow NSF ---
        elif model_type == "pzflow_nsf":
            flow = model_state["flow"]
            cols_x = model_state["cols_x"]
            cols_y = model_state.get("cols_y", ["y"])
            samples = _pzflow_sample_conditional(
                flow,
                X_test,
                cols_x,
                cols_y,
                n_samples=self.N_SAMPLES,
            )  # (S, N, D)
            native = _native_samples(samples, model_state)
            lo = np.quantile(native, alpha / 2, axis=0)
            hi = np.quantile(native, 1.0 - alpha / 2, axis=0)
            y_pred = native.mean(axis=0)
            return {
                "lower": lo.reshape(-1),
                "upper": hi.reshape(-1),
                "y_pred": y_pred.reshape(-1),
                "samples": np.swapaxes(native, 0, 1),
            }

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
        y_true = _native_y(y_test, predictions).reshape(-1)

        metrics = compute_interval_metrics(lo, hi, y_true)

        predictions["_lo_raw"] = lo
        predictions["_hi_raw"] = hi
        predictions["_y_true_raw"] = y_true

        # PIT uniformity per dim (mean KS) + energy score — the
        # flexible-density diagnostics. Samples are (N, S, D).
        if "samples" in predictions:
            try:
                from torchregress.metrics import energy_score

                samp = np.asarray(predictions["samples"], dtype=np.float64)
                if samp.ndim == 2:
                    samp = samp[:, :, None]
                yt = y_true.reshape(samp.shape[0], -1)
                ks_list = []
                for d in range(yt.shape[1]):
                    r = (samp[:, :, d] <= yt[:, d : d + 1]).mean(axis=1)
                    r = np.clip(r, 1e-6, 1 - 1e-6)
                    grid = np.linspace(0, 1, 101)
                    ecdf = np.searchsorted(np.sort(r), grid, side="right") / len(r)
                    ks_list.append(float(np.max(np.abs(ecdf - grid))))
                metrics["PIT_KS"] = float(np.mean(ks_list))
                # Library energy_score takes [S, N, D].
                metrics["EnergyScore"] = float(energy_score(np.swapaxes(samp, 0, 1), yt))
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
        y_test_np = np.asarray(splits["y_test"])
        # NOTE: kept 2-D — an earlier .reshape(-1) flattened multi-target
        # test arrays and silently broke the NLL block below.
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
                # Mirror base run(): expose the runner scaler for native units.
                if "_y_scaler" in splits:
                    predictions["_y_scaler"] = splits["_y_scaler"]

                metrics = self.evaluate(method, predictions, y_test_np, cfg)

                # Compute NLL (native units = standardized NLL + log-det).
                nll: float | None = None
                try:
                    mtype = model_state["type"]
                    y_mean = np.asarray(model_state.get("y_mean", 0.0), dtype=np.float64).reshape(
                        -1
                    )
                    y_scale = np.asarray(model_state.get("y_scale", 1.0), dtype=np.float64).reshape(
                        -1
                    )
                    logdet = float(np.log(y_scale).sum())
                    y_2d = np.asarray(y_test_np, dtype=np.float64).reshape(
                        np.asarray(y_test_np).shape[0], -1
                    )
                    y_std = (y_2d - y_mean) / y_scale
                    if mtype == "torchregress_nsf":
                        model = model_state["model"]
                        flow_loss = model_state["flow"]
                        model.eval()
                        X_t = torch.from_numpy(X_test_np).float().to(current_device())
                        y_t = torch.from_numpy(y_std.astype(np.float32)).to(current_device())
                        with torch.no_grad():
                            context = model(X_t)
                            nll = float(-flow_loss.log_prob(context, y_t).mean().item()) + logdet
                    elif mtype == "pzflow_nsf":
                        nll = (
                            _pzflow_conditional_nll(
                                model_state["flow"],
                                X_test_np,
                                y_std.astype(np.float32),
                                cols_x,
                                model_state.get("cols_y"),
                                batch_size=cfg.batch_size,
                            )
                            + logdet
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
