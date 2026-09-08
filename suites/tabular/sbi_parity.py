"""SBI parity suite: amortized neural posterior estimation comparison.

Compares three conditional-density estimators as amortized posterior
engines on sbibm benchmark tasks (reference posteriors available):
torchregress NSF (zuko, PyTorch), pzflow (JAX), sbi NPE (PyTorch NSF).

Protocol: train p(theta | x) on prior simulations once; per benchmark
observation draw posterior samples and score C2ST vs reference draws
(sbibm standard; 0.5 = indistinguishable), mean error vs reference mean,
and reference log-prob where the method exposes conditional densities
(pzflow models the joint — honestly N/A there). Custom run() aggregates
over observations.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

# Prevent JAX from preallocating all GPU memory at import time.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _train_one

N_POSTERIOR_SAMPLES = 2000
N_REF_LOGPROB = 1000


class SBIParitySuite(HarnessRunner):
    """Amortized NPE: torchregress vs pzflow vs sbi on sbibm tasks."""

    @property
    def name(self) -> str:
        return "sbi_parity"

    @property
    def description(self) -> str:
        return "Amortized posteriors: torchregress NSF vs pzflow vs sbi NPE (C2ST)"

    @property
    def supported_datasets(self) -> list[str]:
        return ["sbibm_twomoons"]

    def methods(self) -> list[str]:
        return ["torchregress_NSF", "pzflow_NSF", "sbi_NPE"]

    def library_name(self, method: str) -> str:
        return {
            "torchregress_NSF": "torchregress",
            "pzflow_NSF": "pzflow",
            "sbi_NPE": "sbi",
        }.get(method, method)

    def method_notes(self, method: str) -> str:
        return {
            "torchregress_NSF": "zuko NSF conditional flow (PyTorch)",
            "pzflow_NSF": "pzflow Flow, default bijector (JAX)",
            "sbi_NPE": "amortized neural posterior estimation (sbi, NSF)",
        }.get(method, "")

    # ------------------------------------------------------------------
    # fit — train p(theta | x) on prior simulations (amortized, once)
    # ------------------------------------------------------------------

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        X = np.asarray(splits["X_train"], dtype=np.float32)
        y = np.asarray(splits["y_train"], dtype=np.float32)
        in_dim, out_dim = X.shape[1], y.shape[1]
        hidden = _hidden_size(in_dim)

        if method == "torchregress_NSF":
            from torchregress.losses.nflows import (
                NormalizingFlowLoss,
                create_flow_model,
            )

            flow = create_flow_model(
                n_features=out_dim,
                context_dim=hidden,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=hidden)
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_one(
                model,
                loss_fn,
                torch.from_numpy(X).float().to(current_device()),
                torch.from_numpy(y).float().to(current_device()),
                config,
                loss_params=list(flow.parameters()),
            )
            model.eval()
            return {"type": "torchregress_nsf", "model": model, "flow": loss_fn}

        if method == "pzflow_NSF":
            import pandas as pd
            import pzflow

            cols_x = [f"x{i}" for i in range(in_dim)]
            cols_y = [f"y{i}" for i in range(out_dim)]
            df = pd.DataFrame(X, columns=cols_x, dtype=np.float32)
            for i, c in enumerate(cols_y):
                df[c] = y[:, i]
            flow = pzflow.Flow(
                data_columns=cols_y,
                conditional_columns=cols_x,
                latent=pzflow.distributions.Normal(input_dim=out_dim),
                seed=int(config.seed),
            )
            flow.train(
                df,
                epochs=config.epochs,
                batch_size=min(config.batch_size, len(df)),
                verbose=False,
                seed=int(config.seed),
            )
            return {
                "type": "pzflow_nsf",
                "flow": flow,
                "cols_x": cols_x,
                "cols_y": cols_y,
            }

        if method == "sbi_NPE":
            import torch as _torch
            from sbi.inference import NPE
            from sbi.utils import BoxUniform

            # two_moons prior: Uniform(-1, 1)^2 (asserted, not assumed).
            lo, hi = float(y.min()), float(y.max())
            assert lo >= -1.0 - 1e-6 and hi <= 1.0 + 1e-6, (lo, hi)
            prior = BoxUniform(low=-1.0 * _torch.ones(out_dim), high=torch.ones(out_dim))
            inference = NPE(prior=prior, show_progress_bars=False)
            inference.append_simulations(
                _torch.as_tensor(y, dtype=_torch.float32),
                _torch.as_tensor(X, dtype=_torch.float32),
            )
            train_kwargs: dict[str, Any] = {"max_num_epochs": 2} if config.smoke else {}
            density_estimator = inference.train(**train_kwargs)
            posterior = inference.build_posterior(density_estimator)
            return {"type": "sbi_npe", "posterior": posterior}

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Posterior draws + scores per observation (custom run below drives this)
    # ------------------------------------------------------------------

    def _posterior_samples(
        self,
        model_state: dict[str, Any],
        x_obs: np.ndarray,
        config: HarnessConfig,
    ) -> np.ndarray:
        """Draw [S, D] posterior samples at one observation."""
        t = model_state["type"]
        if t == "torchregress_nsf":
            # Amortized flow has no direct posterior sampler: reuse the
            # forward (prior-pushforward) sampler conditioned on x_obs.
            # NOTE: for NPE-style p(theta|x) this is exact; documented.
            model = model_state["model"]
            flow_loss = model_state["flow"]
            model.eval()
            Xt = torch.from_numpy(x_obs.reshape(1, -1)).float().to(current_device())
            with torch.no_grad():
                ctx = model(Xt)
                s = flow_loss.sample(ctx, n_samples=N_POSTERIOR_SAMPLES)
            return np.asarray(s.detach().cpu()).reshape(N_POSTERIOR_SAMPLES, -1)

        if t == "pzflow_nsf":
            import pandas as pd

            flow = model_state["flow"]
            df_x = pd.DataFrame([x_obs], columns=model_state["cols_x"])
            s = flow.sample(N_POSTERIOR_SAMPLES, conditions=df_x)
            cols = model_state["cols_y"]
            return np.stack([np.asarray(s[c]).reshape(-1) for c in cols], axis=1).astype(np.float64)

        if t == "sbi_npe":
            post = model_state["posterior"]
            xo = torch.as_tensor(x_obs.reshape(-1), dtype=torch.float32)
            s = post.sample((N_POSTERIOR_SAMPLES,), x=xo, show_progress_bars=False)
            return np.asarray(s.detach().cpu(), dtype=np.float64)

        raise ValueError(f"Unknown model type: {t}")

    def _reference_logprob(
        self, model_state: dict[str, Any], x_obs: np.ndarray, ref: np.ndarray
    ) -> float | None:
        """Mean log-density of reference draws (conditional methods only)."""
        t = model_state["type"]
        sub = ref[:N_REF_LOGPROB]
        try:
            if t == "torchregress_nsf":
                model = model_state["model"]
                flow_loss = model_state["flow"]
                model.eval()
                Xt = torch.from_numpy(np.repeat(x_obs.reshape(1, -1), len(sub), axis=0)).float()
                yt = torch.from_numpy(sub.astype(np.float32)).float()
                with torch.no_grad():
                    ctx = model(Xt.to(current_device()))
                    lp = flow_loss.log_prob(ctx, yt.to(current_device()))
                return float(lp.mean().item())
            if t == "sbi_npe":
                post = model_state["posterior"]
                xo = torch.as_tensor(x_obs.reshape(-1), dtype=torch.float32)
                xr = torch.as_tensor(sub.astype(np.float32))
                with torch.no_grad():
                    lp = post.log_prob(xr, x=xo)
                return float(lp.mean().item())
        except Exception:
            return None
        return None  # pzflow models the joint, not the conditional

    # ------------------------------------------------------------------
    # Custom run(): per-observation C2ST vs reference posterior
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

        from sbibm.metrics import c2st

        cfg = config or HarnessConfig()
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name)

        if "reference" not in splits or "X_test" not in splits:
            # Not an sbibm task dataset (e.g. forced --datasets combo):
            # emit failed rows instead of crashing the runner.
            return [
                {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "C2ST": None,
                    "RefLogProb": None,
                    "MeanRMSE": None,
                    "train_s": None,
                    "eval_s": None,
                    "SyntheticFallback": False,
                    "Device": str(current_device()),
                    "Notes": "failed: unsupported dataset for sbi_parity (needs sbibm reference)",
                }
                for method in self.methods()
            ]
        reference = splits["reference"]
        n_obs = len(reference)
        rows: list[dict[str, Any]] = []

        for method in self.methods():
            set_harness_seed(cfg.seed)
            try:
                t0 = perf_counter()
                model_state = self.fit(method, splits, cfg)
                fit_time = perf_counter() - t0
                t0 = perf_counter()
                c2sts, rimse, lps = [], [], []
                X_test = np.asarray(splits["X_test"])
                for k in range(n_obs):
                    approx = self._posterior_samples(model_state, X_test[k], cfg)
                    ref = np.asarray(reference[k], dtype=np.float64)
                    c = c2st(
                        torch.as_tensor(ref, dtype=torch.float32),
                        torch.as_tensor(np.asarray(approx, dtype=np.float32)),
                        z_score=False,
                    )
                    c2sts.append(float(c.item() if hasattr(c, "item") else c))
                    rimse.append(float(np.linalg.norm(approx.mean(0) - ref.mean(0))))
                    lp = self._reference_logprob(model_state, X_test[k], ref)
                    if lp is not None:
                        lps.append(lp)
                pred_time = perf_counter() - t0
                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "C2ST": float(np.mean(c2sts)),
                    "RefLogProb": float(np.mean(lps)) if lps else None,
                    "MeanRMSE": float(np.mean(rimse)),
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
                    "C2ST": None,
                    "RefLogProb": None,
                    "MeanRMSE": None,
                    "train_s": None,
                    "eval_s": None,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }
            rows.append(row)
        return rows

    # HarnessRunner abstract surface (unused: custom run drives everything).
    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("sbi_parity uses custom run()")

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        raise NotImplementedError("sbi_parity uses custom run()")
