"""SBI parity suite: amortized neural posterior estimation comparison.

Compares conditional-density estimators as amortized posterior engines on
sbibm benchmark tasks (reference posteriors available).  Paired rows share a
density-estimator family:

* NSF: ``torchregress_NSF`` (zuko) vs ``sbi_NPE`` (sbi NPE, NSF) vs ``pzflow_NSF`` (JAX)
* MAF: ``torchregress_MAF`` vs ``sbi_NPE_MAF``
* MDN: ``torchregress_MDN`` (``MDNLoss``, K=10, full covariance) vs ``sbi_NPE_MDN`` (K=10, full)

``sbi_FMPE`` (flow matching) and ``sbi_NPSE`` (score-based) are reference rows
with no torchregress counterpart.

Protocol: train p(theta | x) on prior simulations once; per benchmark
observation draw posterior samples and score C2ST vs reference draws
(sbibm standard; 0.5 = indistinguishable), mean error vs reference mean,
and reference log-prob where the method exposes conditional densities
(pzflow models the joint — honestly N/A there). Independently simulated,
held-out theta/x pairs provide marginal 95% coverage and SBC rank
diagnostics. The same posteriors are also scored with sbi's own diagnostics
(``run_sbc``/``check_sbc`` and TARP), an independent implementation: if the
harness's SBC and sbi's disagree on a method, one of them is wrong.
Custom run() aggregates over observations.
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
N_SBC_POSTERIOR_SAMPLES = 500

# Matched capacity for paired density estimators (torchregress vs sbi).
FLOW_HIDDEN = 64
FLOW_TRANSFORMS = 3
MDN_COMPONENTS = 10

_SBI_NPE_MODELS = {"sbi_NPE": "nsf", "sbi_NPE_MAF": "maf", "sbi_NPE_MDN": "mdn"}
_TR_FLOWS = {"torchregress_NSF": "nsf", "torchregress_MAF": "maf"}


class _FixedDrawsPosterior:
    """Minimal posterior for sbi diagnostics that replays precomputed draws.

    ``run_sbc``/``run_tarp`` only call ``sample_batched``; replaying the draws
    the harness already scored makes the two diagnostics directly comparable
    and works for every method, including the non-sbi ones.
    """

    def __init__(self, xs: np.ndarray, draws: np.ndarray) -> None:
        self._xs = torch.as_tensor(xs, dtype=torch.float32)
        self._draws = torch.as_tensor(draws, dtype=torch.float32)  # [S, N, D]

    def sample_batched(
        self, sample_shape: tuple[int, ...], x: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        if x.shape != self._xs.shape or not torch.equal(x, self._xs):
            raise ValueError("diagnostics must use the observations the draws were made for")
        n = int(sample_shape[0])
        if n > self._draws.shape[0]:
            raise ValueError(f"asked for {n} draws, only {self._draws.shape[0]} stored")
        return self._draws[:n]


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
        return [
            "sbibm_twomoons",
            "sbibm_slcp",
            "sbibm_gaussian_linear",
            "sbibm_gaussian_mixture",
            "sbibm_bernoulli_glm",
        ]

    def methods(self) -> list[str]:
        return [
            "torchregress_NSF",
            "torchregress_MAF",
            "torchregress_MDN",
            "pzflow_NSF",
            "sbi_NPE",
            "sbi_NPE_MAF",
            "sbi_NPE_MDN",
            "sbi_FMPE",
            "sbi_NPSE",
        ]

    def library_name(self, method: str) -> str:
        if method.startswith("torchregress_"):
            return "torchregress"
        if method.startswith("sbi_"):
            return "sbi"
        return {"pzflow_NSF": "pzflow"}.get(method, method)

    def library_version(self, method: str) -> str:
        import json
        from importlib.metadata import PackageNotFoundError, distribution

        name = {"torchregress": "torchregress", "sbi": "sbi", "pzflow": "pzflow"}.get(
            self.library_name(method), ""
        )
        if not name:
            return ""
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            return "not installed"
        commit = json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info", {})
        commit_id = commit.get("commit_id", "")
        return f"{dist.version} (git {commit_id[:10]})" if commit_id else dist.version

    def method_notes(self, method: str) -> str:
        return {
            "torchregress_NSF": (
                "zuko NSF conditional flow (PyTorch), spline tail bound from "
                "recommended_tail_bound(theta_train); pair: sbi_NPE"
            ),
            "torchregress_MAF": "zuko MAF conditional flow; pair: sbi_NPE_MAF",
            "torchregress_MDN": f"MDNLoss, {MDN_COMPONENTS} diagonal components; pair: sbi_NPE_MDN",
            "pzflow_NSF": "pzflow Flow, default bijector (JAX)",
            "sbi_NPE": "sbi NPE, NSF density estimator (matched capacity)",
            "sbi_NPE_MAF": "sbi NPE, MAF density estimator (matched capacity)",
            "sbi_NPE_MDN": f"sbi NPE, MDN with {MDN_COMPONENTS} components",
            "sbi_FMPE": "sbi flow-matching posterior estimation (reference, no pair)",
            "sbi_NPSE": "sbi score-based posterior estimation, VE SDE (reference, no pair)",
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

        if method in _TR_FLOWS:
            from torchregress.losses.nflows import (
                NormalizingFlowLoss,
                create_flow_model,
                recommended_tail_bound,
            )

            # The NSF spline is the identity outside [-B, B] (zuko B=5), so it
            # cannot place mass at |theta| > 5: on gaussian_mixture
            # (theta ~ U(-10, 10)^2) every posterior collapsed onto the bound
            # edge (C2ST 0.997). Derive B from the training theta range instead
            # (12.5 there; stays 5 for twomoons/slcp/gaussian_linear, whose
            # rows are unchanged). theta is deliberately not z-scored: that
            # also fixes gaussian_mixture but moved the other tasks' rows
            # (gaussian_linear 0.62 -> 0.67 under this fixed-budget loop).
            flow_kwargs: dict[str, Any] = {}
            if _TR_FLOWS[method] == "nsf":
                flow_kwargs["bound"] = recommended_tail_bound(y)
            flow = create_flow_model(
                n_features=out_dim,
                context_dim=hidden,
                flow_type=_TR_FLOWS[method],
                n_transforms=FLOW_TRANSFORMS,
                hidden_features=[FLOW_HIDDEN, FLOW_HIDDEN],
                **flow_kwargs,
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

        if method == "torchregress_MDN":
            from torchregress.losses.mdn import MDNLoss

            # Full covariance, as sbi_NPE_MDN (needs torchregress >= 0.3 sampling).
            mdn = MDNLoss(n_components=MDN_COMPONENTS, n_features=out_dim, covariance_type="full")
            model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=mdn.expected_output_size)
            _train_one(
                model,
                mdn,
                torch.from_numpy(X).float().to(current_device()),
                torch.from_numpy(y).float().to(current_device()),
                config,
            )
            model.eval()
            return {"type": "torchregress_mdn", "model": model, "mdn": mdn}

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

        if method.startswith("sbi_"):
            import torch as _torch
            from sbi.inference import FMPE, NPE, NPSE
            from sbi.neural_nets import posterior_nn
            from sbi.utils import BoxUniform

            # sbibm's exact prior matters for the unbounded Gaussian and GLM
            # tasks. The box fallback preserves compatibility with older splits.
            prior = splits.get("prior_dist")
            if prior is None:
                plo = float(splits.get("prior_low", -1.0))
                phi = float(splits.get("prior_high", 1.0))
                assert plo < phi
                prior = BoxUniform(low=plo * _torch.ones(out_dim), high=phi * _torch.ones(out_dim))
            if method in _SBI_NPE_MODELS:
                kind = _SBI_NPE_MODELS[method]
                # sbi main rejects options a model does not use, so pass only
                # the ones that apply to each density estimator.
                arch: dict[str, int] = (
                    {"num_components": MDN_COMPONENTS}
                    if kind == "mdn"
                    else {"num_transforms": FLOW_TRANSFORMS}
                )
                estimator = posterior_nn(model=kind, hidden_features=FLOW_HIDDEN, **arch)
                inference: Any = NPE(
                    prior=prior, density_estimator=estimator, show_progress_bars=False
                )
            elif method == "sbi_FMPE":
                inference = FMPE(prior=prior, show_progress_bars=False)
            elif method == "sbi_NPSE":
                inference = NPSE(prior=prior, sde_type="ve", show_progress_bars=False)
            else:
                raise ValueError(f"Unknown method: {method}")
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
        n_draws: int = N_POSTERIOR_SAMPLES,
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
                s = flow_loss.sample(ctx, n_samples=n_draws)
            return np.asarray(s.detach().cpu()).reshape(n_draws, -1)

        if t == "torchregress_mdn":
            model = model_state["model"]
            Xt = torch.from_numpy(x_obs.reshape(1, -1)).float().to(current_device())
            with torch.no_grad():
                s = model_state["mdn"].sample(model(Xt), n_samples=n_draws)
            return np.asarray(s.detach().cpu(), dtype=np.float64).reshape(n_draws, -1)

        if t == "pzflow_nsf":
            import pandas as pd

            flow = model_state["flow"]
            df_x = pd.DataFrame([x_obs], columns=model_state["cols_x"])
            s = flow.sample(n_draws, conditions=df_x)
            cols = model_state["cols_y"]
            return np.stack([np.asarray(s[c]).reshape(-1) for c in cols], axis=1).astype(np.float64)

        if t == "sbi_npe":
            post = model_state["posterior"]
            xo = torch.as_tensor(x_obs.reshape(-1), dtype=torch.float32)
            s = post.sample((n_draws,), x=xo, show_progress_bars=False)
            return np.asarray(s.detach().cpu(), dtype=np.float64)

        raise ValueError(f"Unknown model type: {t}")

    def _sbc_draws(
        self,
        model_state: dict[str, Any],
        splits: dict[str, Any],
        config: HarnessConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior draws at the held-out SBC simulations.

        Returns ``(theta [N, D], x [N, F], draws [S, N, D])``, shared by the
        harness's own SBC and by sbi's diagnostics so both score the same draws.
        """
        sbc_x = np.asarray(splits["sbc_x"], dtype=np.float32)
        sbc_theta = np.asarray(splits["sbc_theta"], dtype=np.float64)
        n_cases = min(len(sbc_x), 8 if config.smoke else len(sbc_x))
        n_draws = min(N_SBC_POSTERIOR_SAMPLES, 200) if config.smoke else N_SBC_POSTERIOR_SAMPLES
        draws = np.stack(
            [
                self._posterior_samples(model_state, x, config, n_draws=n_draws)
                for x in sbc_x[:n_cases]
            ],
            axis=1,
        )
        return sbc_theta[:n_cases], sbc_x[:n_cases], draws

    def _sbc_diagnostics(
        self,
        model_state: dict[str, Any],
        splits: dict[str, Any],
        config: HarnessConfig,
        draws: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Compute posterior calibration from held-out prior/simulator pairs.

        Ranks are the number of posterior draws below the generating parameter
        per parameter dimension. They are discrete-uniform for a calibrated
        posterior; no reference posterior or synthetic proxy is substituted.
        """
        theta, _, samples = draws or self._sbc_draws(model_state, splits, config)
        n_draws = samples.shape[0]
        ranks = np.sum(samples < theta[None], axis=0) / n_draws  # [N, D]
        lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
        covered = (lower <= theta) & (theta <= upper)

        rank_values = np.sort(ranks.reshape(-1))
        n_ranks = len(rank_values)
        empirical_upper = np.arange(1, n_ranks + 1) / n_ranks
        empirical_lower = np.arange(n_ranks) / n_ranks
        return {
            "SBC_Coverage95": float(covered.mean()),
            "SBC_RankMean": float(rank_values.mean()),
            "SBC_RankKS": float(
                max(
                    np.max(np.abs(empirical_upper - rank_values)),
                    np.max(np.abs(rank_values - empirical_lower)),
                )
            ),
        }

    def _sbi_diagnostics(
        self,
        draws: tuple[np.ndarray, np.ndarray, np.ndarray],
        prior_samples: np.ndarray,
    ) -> dict[str, float]:
        """Score the same SBC draws with sbi's own SBC and TARP implementations.

        An independent check on :meth:`_sbc_diagnostics`: ``SBI_SBC_KSpMin`` is
        the smallest per-dimension KS p-value for rank uniformity (sbi
        ``check_sbc``), ``TARP_ATC`` the TARP area-to-curve (0 = calibrated) and
        ``TARP_KSp`` its KS p-value (sbi ``check_tarp``).
        """
        import warnings

        from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp

        theta, xs, samples = draws
        posterior = _FixedDrawsPosterior(xs, samples)
        theta_t = torch.as_tensor(theta, dtype=torch.float32)
        xs_t = torch.as_tensor(xs, dtype=torch.float32)
        n_draws = samples.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sbi warns below 100 SBC cases
            ranks, dap = run_sbc(
                theta_t, xs_t, posterior, num_posterior_samples=n_draws, show_progress_bar=False
            )
            checks = check_sbc(
                ranks,
                torch.as_tensor(prior_samples[: len(theta)], dtype=torch.float32),
                dap,
                num_posterior_samples=n_draws,
            )
            ecp, alpha = run_tarp(
                theta_t,
                xs_t,
                posterior,
                num_posterior_samples=n_draws,
                show_progress_bar=False,
                num_bins=max(4, len(theta) // 10),  # sbi's default N//10 is 0 below 10 cases
            )
            atc, ks_p = check_tarp(ecp, alpha)
        return {
            "SBI_SBC_KSpMin": float(checks["ks_pvals"].min()),
            "TARP_ATC": float(atc),
            "TARP_KSp": float(ks_p),
        }

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
            if t == "torchregress_mdn":
                model = model_state["model"]
                Xt = torch.from_numpy(np.repeat(x_obs.reshape(1, -1), len(sub), axis=0)).float()
                yt = torch.from_numpy(sub.astype(np.float32)).float().to(current_device())
                mdn = model_state["mdn"]
                with torch.no_grad():
                    params = mdn._extract_distribution_parameters(model(Xt.to(current_device())))
                    lp = -mdn._calculate_nll(yt, params)
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
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "C2ST": None,
                    "RefLogProb": None,
                    "MeanRMSE": None,
                    "SBC_Coverage95": None,
                    "SBC_RankMean": None,
                    "SBC_RankKS": None,
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
        # Smoke thrift: 1 observation, fewer posterior draws (C2ST dominates).
        if cfg.smoke:
            n_obs = 1
            n_draws = 500
        else:
            n_draws = N_POSTERIOR_SAMPLES
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
                    approx = self._posterior_samples(model_state, X_test[k], cfg, n_draws=n_draws)
                    ref = np.asarray(reference[k], dtype=np.float64)
                    # C2ST needs balanced classes: sbibm ships 10k reference draws
                    # against n_draws=2k here, and a 5:1 split puts the
                    # chance-level floor at 0.833 instead of 0.5.
                    n_bal = min(len(ref), len(approx))
                    pick = np.random.default_rng(cfg.seed + k).permutation(len(ref))[:n_bal]
                    c = c2st(
                        torch.as_tensor(ref[pick], dtype=torch.float32),
                        torch.as_tensor(np.asarray(approx, dtype=np.float32)[:n_bal]),
                        z_score=False,
                    )
                    c2sts.append(float(c.item() if hasattr(c, "item") else c))
                    rimse.append(float(np.linalg.norm(approx.mean(0) - ref.mean(0))))
                    lp = self._reference_logprob(model_state, X_test[k], ref)
                    if lp is not None:
                        lps.append(lp)
                draws = self._sbc_draws(model_state, splits, cfg)
                sbc = self._sbc_diagnostics(model_state, splits, cfg, draws=draws)
                diag_note = ""
                try:
                    sbc.update(self._sbi_diagnostics(draws, np.asarray(splits["y_train"])))
                except Exception as exc:  # noqa: BLE001 — harness SBC is still reported
                    diag_note = f"; sbi diagnostics failed: {type(exc).__name__}: {exc}"
                pred_time = perf_counter() - t0
                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "C2ST": float(np.mean(c2sts)),
                    "RefLogProb": float(np.mean(lps)) if lps else None,
                    "MeanRMSE": float(np.mean(rimse)),
                    **sbc,
                    "train_s": fit_time,
                    "eval_s": pred_time,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                    "Notes": self.method_notes(method) + diag_note,
                }
            except Exception as exc:  # noqa: BLE001 — row-level resilience
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "C2ST": None,
                    "RefLogProb": None,
                    "MeanRMSE": None,
                    "SBC_Coverage95": None,
                    "SBC_RankMean": None,
                    "SBC_RankKS": None,
                    "SBI_SBC_KSpMin": None,
                    "TARP_ATC": None,
                    "TARP_KSp": None,
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
