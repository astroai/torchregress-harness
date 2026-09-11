"""Non-Gaussian likelihood suite — skewed/heavy-tail predictive distributions.

Compares torchregress parametric likelihoods (Gaussian, skew-normal,
skew-t, Johnson-SU, sinh-arcsinh, asymmetric-Laplace, MDN, NSF) against
CatBoost uncertainty + LightGBM point baselines on RIGHT-SKEWED real
targets (Folktables all-states income, NYC TLC fares, insurance claims).

Targets stay RAW (loaders set ``_skip_y_standardization``) so likelihood
rows see the true skew. Scoring is proper throughout: per-row NLL from
the library elementwise functionals, CRPS/PIT from predictive samples,
90% sample-quantile coverage/width, plus rows/sec throughput (efficiency
is part of any SOTA claim at >1M rows).

Design notes
------------
* GEV/BetaRegression/SQR are intentionally excluded: GEV needs an
  extreme-value framing and Beta needs (0, 1) targets.
* NSF standardizes targets internally (z-score from train) because the
  zuko spline bound cannot represent raw dollar-scale mass; samples are
  mapped back and NLL is corrected by ``+ log(std)`` to native units.
* Torch rows train with batch >= 1024 (throughput-motivated; documented).
* CatBoost ``RMSEWithUncertainty`` gives (mean, var); scored as Gaussian.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import torch
from torchregress.losses import GaussianNLLLoss

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites._backbones import _MLP, _hidden_size, _train_one

_N_SAMPLES = 256
_BIG_BATCH = 1024
_Z = 1.6448536269514722


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) + 1e-6


def _family_samples(family: str, raw: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """[n, N] predictive draws for a raw head output (torch-native).

    Shape parameters are clamped to sampling-stable ranges; this affects
    only the predictive draws, never the trained NLL.
    """
    g = torch.Generator(device=raw.device)
    g.manual_seed(seed)
    N = raw.shape[0]

    def _randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=g, device=raw.device, dtype=raw.dtype)

    if family == "gaussian":
        mu, logs = raw[:, 0], raw[:, 1]
        return mu[None] + torch.exp(0.5 * logs)[None] * _randn(n, N)
    if family == "skewnormal":
        xi, sig, alpha = raw[:, 0], _softplus(raw[:, 1]), raw[:, 2].clamp(-10, 10)
        delta = alpha / torch.sqrt(1 + alpha**2)
        z = delta[None] * _randn(n, N).abs() + torch.sqrt(1 - delta**2)[None] * _randn(n, N)
        return xi[None] + sig[None] * z
    if family == "skewt":
        xi, om = raw[:, 0], _softplus(raw[:, 1])
        alpha = raw[:, 2].clamp(-10, 10)
        nu = _softplus(raw[:, 3]).clamp(2.1, 60.0)
        delta = alpha / torch.sqrt(1 + alpha**2)
        # Chi-square via Gamma (device-safe representation).
        gam = torch.distributions.Gamma(
            nu[None].expand(n, N) / 2, torch.ones((), device=raw.device, dtype=raw.dtype)
        ).sample()
        w = torch.sqrt(nu[None] / (2 * gam))
        z = (delta[None] * _randn(n, N).abs() + torch.sqrt(1 - delta**2)[None] * _randn(n, N)) * w
        return xi[None] + om[None] * z
    if family == "johnsonsu":
        xi, lam = raw[:, 0], _softplus(raw[:, 1])
        gamma, delta = raw[:, 2].clamp(-8, 8), _softplus(raw[:, 3]).clamp(0.3, 10.0)
        u = torch.sinh((_randn(n, N) - gamma[None]) / delta[None])
        return xi[None] + lam[None] * u
    if family == "sinharcsinh":
        mu, sig = raw[:, 0], _softplus(raw[:, 1])
        eps_, delta = raw[:, 2].clamp(-8, 8), _softplus(raw[:, 3]).clamp(0.3, 10.0)
        y = mu[None] + sig[None] * torch.sinh(delta[None] * torch.asinh(_randn(n, N)) + eps_[None])
        return y
    if family == "aslaplace":
        mu, sig = raw[:, 0], _softplus(raw[:, 1])
        kappa = _softplus(raw[:, 2]).clamp(0.2, 5.0)
        p = (kappa**2 / (1 + kappa**2))[None]
        e1 = torch.distributions.Exponential(
            torch.ones((), device=raw.device, dtype=raw.dtype)
        ).sample((n, N))
        e2 = torch.distributions.Exponential(
            torch.ones((), device=raw.device, dtype=raw.dtype)
        ).sample((n, N))
        right = mu[None] + e1 * sig[None] * kappa[None] / np.sqrt(2.0)
        left = mu[None] - e2 * sig[None] / (np.sqrt(2.0) * kappa[None])
        u = torch.rand(n, N, generator=g, device=raw.device, dtype=raw.dtype)
        return torch.where(u < p, right, left)
    raise ValueError(f"Unknown family: {family}")


_FAMILY_LOSS = {
    "gaussian": ("torchregress.losses", "GaussianNLLLoss", 2),
    "skewnormal": ("torchregress.losses", "SkewNormalNLLLoss", 3),
    "skewt": ("torchregress.losses", "SkewTLoss", 4),
    "johnsonsu": ("torchregress.losses", "JohnsonSUNLLLoss", 4),
    "sinharcsinh": ("torchregress.losses", "SinhArcsinhNLLLoss", 4),
    "aslaplace": ("torchregress.losses", "AsymmetricLaplaceNLLLoss", 3),
}


def _family_nll(family: str, raw: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean per-row NLL via the library elementwise functionals (native units)."""
    import importlib

    modname, clsname, _ = _FAMILY_LOSS[family]
    mod = importlib.import_module(modname)
    loss_fn = getattr(mod, clsname)()
    yt = y.reshape(-1, 1)
    if family == "gaussian":
        # GaussianNLLLoss takes (mean, logvar) raw head output.
        return loss_fn(raw, yt)
    return loss_fn(raw, yt)


class LikelihoodsSuite(HarnessRunner):
    """Parametric likelihoods on skewed real targets at scale."""

    @property
    def name(self) -> str:
        return "likelihoods"

    @property
    def description(self) -> str:
        return "Non-Gaussian predictive likelihoods on skewed targets (>1M rows)"

    @property
    def supported_datasets(self) -> list[str]:
        return ["folktables_all", "tlc_fare", "insurance_claims"]

    def methods(self) -> list[str]:
        return [
            "GaussianNLL",
            "SkewNormal",
            "SkewT",
            "JohnsonSU",
            "SinhArcsinh",
            "AsymmetricLaplace",
            "MDN_K3",
            "NSF",
            "CatBoost_Uncertainty",
            "LightGBM_Point",
        ]

    def library_name(self, method: str) -> str:
        if method.startswith("CatBoost"):
            return "catboost"
        if method.startswith("LightGBM"):
            return "lightgbm"
        return "torchregress"

    def library_version(self, method: str) -> str:
        if method.startswith("CatBoost"):
            try:
                from importlib.metadata import version as _v

                return _v("catboost")
            except Exception:
                return ""
        if method.startswith("LightGBM"):
            try:
                from importlib.metadata import version as _v

                return _v("lightgbm")
            except Exception:
                return ""
        return ""

    def method_notes(self, method: str) -> str:
        return {
            "GaussianNLL": "diagonal Gaussian NLL baseline (2-head MLP)",
            "SkewNormal": "Azzalini skew-normal NLL (3-head: loc, scale, alpha)",
            "SkewT": "Azzalini skew-t NLL (4-head: + dof)",
            "JohnsonSU": "Johnson-SU NLL (4-head: loc, scale, gamma, delta)",
            "SinhArcsinh": "Jones-Pewsey sinh-arcsinh NLL (4-head)",
            "AsymmetricLaplace": "ALD NLL (3-head: loc, scale, kappa)",
            "MDN_K3": "3-component diagonal mixture density network",
            "NSF": "zuko NSF flow; internal z-score (spline bound) with native-unit NLL correction",
            "CatBoost_Uncertainty": "CatBoost RMSEWithUncertainty (mean+var), scored Gaussian",
            "LightGBM_Point": "LightGBM L2 point baseline (no distribution)",
        }.get(method, "")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig
    ) -> dict[str, Any]:
        set_harness_seed(config.seed)
        X = np.asarray(splits["X_train"], dtype=np.float32)
        y_raw = np.asarray(splits["y_train"], dtype=np.float32).reshape(-1)
        bs = max(config.batch_size, _BIG_BATCH)
        # Suite-level target z-scoring for TORCH rows (affine →
        # shape-preserving; NLL/samples are mapped back with the exact
        # Jacobian correction, so native-unit scores stay comparable).
        # Raw-scale heads otherwise diverge across families (observed:
        # sinh-arcsinh NLL 1107 vs NSF 2.5 at smoke budget on TLC fares).
        y_mean, y_std = float(y_raw.mean()), max(float(y_raw.std()), 1e-6)
        y = ((y_raw - y_mean) / y_std).astype(np.float32)
        if method in (
            "GaussianNLL",
            "SkewNormal",
            "SkewT",
            "JohnsonSU",
            "SinhArcsinh",
            "AsymmetricLaplace",
        ):
            import importlib

            family = {
                "GaussianNLL": "gaussian",
                "SkewNormal": "skewnormal",
                "SkewT": "skewt",
                "JohnsonSU": "johnsonsu",
                "SinhArcsinh": "sinharcsinh",
                "AsymmetricLaplace": "aslaplace",
            }[method]
            modname, clsname, out_dim = _FAMILY_LOSS[family]
            mod = importlib.import_module(modname)
            model = _MLP(X.shape[1], _hidden_size(X.shape[1]), out_dim=out_dim)
            _train_one(
                model,
                getattr(mod, clsname)(),
                torch.from_numpy(X).to(current_device()),
                torch.from_numpy(y).to(current_device()).unsqueeze(1),
                config,
                batch_size=bs,
            )
            model.eval()
            return {
                "type": "family",
                "family": family,
                "model": model,
                "y_mean": y_mean,
                "y_std": y_std,
            }
        if method == "MDN_K3":
            from torchregress.losses import MixtureDensityLoss

            mdn_loss = MixtureDensityLoss(n_components=3, n_features=1)
            model = _MLP(X.shape[1], _hidden_size(X.shape[1]), mdn_loss.expected_output_size)
            _train_one(
                model,
                mdn_loss,
                torch.from_numpy(X).to(current_device()),
                torch.from_numpy(y).to(current_device()).unsqueeze(1),
                config,
                batch_size=bs,
            )
            model.eval()
            return {
                "type": "mdn",
                "model": model,
                "loss_fn": mdn_loss,
                "y_mean": y_mean,
                "y_std": y_std,
            }
        if method == "NSF":
            from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

            mu, sc = y_mean, y_std
            hidden = _hidden_size(X.shape[1])
            flow = create_flow_model(
                n_features=1,
                context_dim=hidden,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            )
            model = _MLP(X.shape[1], hidden, out_dim=hidden)
            loss_fn = NormalizingFlowLoss(flow=flow)
            _train_one(
                model,
                loss_fn,
                torch.from_numpy(X).to(current_device()),
                torch.from_numpy(((y - mu) / sc).astype(np.float32))
                .to(current_device())
                .unsqueeze(1),
                config,
                loss_params=list(flow.parameters()),
                batch_size=bs,
            )
            model.eval()
            return {
                "type": "nsf",
                "model": model,
                "flow": loss_fn,
                "y_mean": mu,
                "y_std": sc,
            }
        if method == "CatBoost_Uncertainty":
            from catboost import CatBoostRegressor, Pool

            iters = 20 if config.smoke else 300
            est = CatBoostRegressor(
                loss_function="RMSEWithUncertainty",
                iterations=iters,
                depth=6,
                learning_rate=0.05,
                thread_count=-1,
                random_seed=config.seed,
                verbose=False,
            )
            est.fit(Pool(X, y_raw))
            return {"type": "catboost", "model": est}
        if method == "LightGBM_Point":
            import lightgbm as lgb

            iters = 20 if config.smoke else 200
            train_set = lgb.Dataset(X, label=y_raw)
            est = lgb.train(
                {"objective": "regression", "verbosity": -1, "seed": config.seed},
                train_set,
                num_boost_round=iters,
            )
            return {"type": "lightgbm", "model": est}
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # predict — mean/std/samples/NLL on RAW targets
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        t = model_state["type"]
        Xt = np.asarray(X_test, dtype=np.float32)
        dev = current_device()
        t0 = perf_counter()
        if t == "family":
            model = model_state["model"]
            model.eval()
            with torch.no_grad():
                raw = model(torch.from_numpy(Xt).to(dev))
            with torch.no_grad():
                samp = _family_samples(model_state["family"], raw, _N_SAMPLES, config.seed)
            mu, sc = float(model_state["y_mean"]), float(model_state["y_std"])
            samples = (
                samp.detach().cpu().numpy().astype(np.float64) * sc + mu
            )  # [S, N] native units
            out: dict[str, Any] = {
                "samples": samples,
                "raw": raw.detach().cpu().numpy().astype(np.float64),
            }
        elif t == "mdn":
            model = model_state["model"]
            loss_fn = model_state["loss_fn"]
            model.eval()
            with torch.no_grad():
                raw = model(torch.from_numpy(Xt).to(dev))
                samp = loss_fn.sample(raw, n_samples=_N_SAMPLES)  # [S, N, 1]
            mu, sc = float(model_state["y_mean"]), float(model_state["y_std"])
            samples = (
                np.asarray(samp.detach().cpu(), dtype=np.float64).reshape(_N_SAMPLES, -1) * sc + mu
            )
            out = {
                "samples": samples,
                "raw": raw.detach().cpu().numpy().astype(np.float64),
            }
        elif t == "nsf":
            model = model_state["model"]
            flow_loss = model_state["flow"]
            mu, sc = model_state["y_mean"], model_state["y_std"]
            model.eval()
            with torch.no_grad():
                ctx = model(torch.from_numpy(Xt).to(dev))
                samp = flow_loss.sample(ctx, n_samples=_N_SAMPLES)
            samples = (
                np.asarray(samp.detach().cpu(), dtype=np.float64).reshape(_N_SAMPLES, -1) * sc + mu
            )
            out = {
                "samples": samples,
                "ctx": ctx.detach().cpu().numpy().astype(np.float64),
                "nsf_mean": np.full(1, mu),
                "nsf_std": np.full(1, sc),
            }
        elif t == "catboost":
            pred = np.asarray(model_state["model"].predict(Xt), dtype=np.float64).reshape(-1, 2)
            mean, var = pred[:, 0], np.clip(pred[:, 1], 1e-12, None)
            std = np.sqrt(var)
            rng = np.random.default_rng(config.seed)
            samples = mean[None] + std[None] * rng.standard_normal((_N_SAMPLES, len(mean)))
            out = {"mean": mean, "std": std, "samples": samples}
        elif t == "lightgbm":
            yp = np.asarray(model_state["model"].predict(Xt), dtype=np.float64).reshape(-1)
            out = {"y_pred": yp}
        else:
            raise ValueError(f"Unknown model type: {t}")
        out["eval_s_per_row"] = np.full(1, 0.0)
        out["_eval_s"] = float(perf_counter() - t0)
        return out

    # ------------------------------------------------------------------
    # evaluate — proper scores on raw targets
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        from torchregress.metrics import crps_from_samples
        from torchregress.metrics import (
            kolmogorov_smirnov_uniform_statistic as _ks,
        )

        y = np.asarray(y_test, dtype=np.float64).reshape(-1)
        metrics: dict[str, float] = {}
        if "samples" in predictions:
            samp = np.asarray(predictions["samples"], dtype=np.float64)
            if samp.ndim == 3:
                samp = samp.reshape(samp.shape[0], -1)
            # samp: [S, N]
            # samp is [S, N]; crps_from_samples takes [n_samples, batch].
            metrics["CRPS"] = float(crps_from_samples(samp, y))
            ranks = (samp <= y[None, :]).mean(axis=0)
            ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
            metrics["PIT_KS"] = float(_ks(ranks.astype(np.float64)).item())
            lo = np.quantile(samp, 0.05, axis=0)
            hi = np.quantile(samp, 0.95, axis=0)
            metrics["Coverage"] = float(np.mean((y >= lo) & (y <= hi)))
            metrics["Width"] = float(np.mean(hi - lo))
            med = np.median(samp, axis=0)
            metrics["RMSE"] = float(np.sqrt(np.mean((med - y) ** 2)))
            metrics["MAE"] = float(np.mean(np.abs(med - y)))
        elif "y_pred" in predictions:
            from harness.metrics import compute_point_metrics

            metrics.update(compute_point_metrics(np.asarray(predictions["y_pred"]).reshape(-1), y))
        eval_s = float(np.asarray(predictions.get("_eval_s", 0.0)).reshape(-1)[0])
        metrics["EvalRowsPerSec"] = float(len(y) / max(eval_s, 1e-9))
        nll = predictions.get("_nll")
        try:
            metrics["NLL"] = float(np.asarray(nll).reshape(-1)[0])
        except Exception:
            metrics["NLL"] = float("nan")
        return metrics

    def _test_nll(
        self,
        method: str,
        model_state: dict[str, Any],
        predictions: dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """Mean per-row NLL in NATIVE units (needs model state + targets)."""
        t = model_state["type"]
        dev = current_device()
        y_arr = np.asarray(y_test, dtype=np.float64).reshape(-1)
        with torch.no_grad():
            if t == "family":
                mu = float(model_state["y_mean"])
                sc = float(model_state["y_std"])
                model = model_state["model"]
                model.eval()
                raw = model(torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(dev))
                yt_std = torch.from_numpy(((y_arr - mu) / sc).astype(np.float32)).reshape(-1, 1)
                return float(
                    _family_nll(model_state["family"], raw.detach().cpu(), yt_std).item()
                    + np.log(sc)
                )
            if t == "mdn":
                from torchregress.losses import MixtureDensityLoss

                mu = float(model_state["y_mean"])
                sc = float(model_state["y_std"])
                model = model_state["model"]
                model.eval()
                raw = model(torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(dev))
                loss_fn = MixtureDensityLoss(n_components=3, n_features=1)
                yt_std = torch.from_numpy(((y_arr - mu) / sc).astype(np.float32)).reshape(-1, 1)
                return float(loss_fn(raw.detach().cpu(), yt_std).item() + np.log(sc))
            if t == "nsf":
                model = model_state["model"]
                flow_loss = model_state["flow"]
                mu, sc = model_state["y_mean"], model_state["y_std"]
                model.eval()
                ctx = model(torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(dev))
                yt_std = torch.from_numpy(((y_arr - mu) / sc).astype(np.float32)).reshape(-1, 1)
                lp = flow_loss.log_prob(ctx, yt_std.to(ctx.device, ctx.dtype))
                return float(-lp.mean().item() + np.log(sc))
            if t == "catboost":
                from torchregress.metrics import gaussian_nll

                mean = np.asarray(predictions["mean"], dtype=np.float64).reshape(-1)
                var = np.asarray(predictions["std"], dtype=np.float64).reshape(-1) ** 2
                out = gaussian_nll(mean, y_test.reshape(-1), var)
                return float(out.item() if hasattr(out, "item") else out)
            if t == "lightgbm":
                return float("nan")  # point-only baseline: no predictive NLL
        raise ValueError(f"No NLL path for type: {t}")

    # ------------------------------------------------------------------
    # Custom run(): thread model state into NLL scoring
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

        from harness.devices import current_device as _dev

        cfg = config or HarnessConfig()
        dataset_name, splits = self._prepare_scenario_run(scenario_name, dataset_name, splits)
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name)
        assert splits is not None
        X_train = np.asarray(splits["X_train"])
        y_train = np.asarray(splits["y_train"]).reshape(-1)
        X_test = np.asarray(splits["X_test"])
        y_test = np.asarray(splits["y_test"]).reshape(-1)
        if cfg.smoke:
            n_tr = min(2000, len(X_train))
            X_train, y_train = X_train[:n_tr], y_train[:n_tr]
            n_te = min(500, len(X_test))
            X_te, y_te = X_test[:n_te], y_test[:n_te]
            splits = dict(splits)
            splits["X_train"], splits["y_train"] = X_train, y_train
            splits["X_test"], splits["y_test"] = X_te, y_te
        rows: list[dict[str, Any]] = []
        for method in self.methods():
            set_harness_seed(cfg.seed)
            try:
                t0 = perf_counter()
                state = self.fit(method, splits, cfg)
                fit_time = perf_counter() - t0
                t0 = perf_counter()
                pred = self.predict(method, state, np.asarray(splits["X_test"]), cfg)
                pred_time = perf_counter() - t0
                nll = self._test_nll(
                    method,
                    state,
                    pred,
                    np.asarray(splits["X_test"]),
                    np.asarray(splits["y_test"]),
                )
                pred["_nll"] = np.full(1, nll)
                metrics = self.evaluate(method, pred, np.asarray(splits["y_test"]), cfg)
                row: dict[str, Any] = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": self.library_version(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    **metrics,
                    "train_s": fit_time,
                    "eval_s": pred_time,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(_dev()),
                    "Notes": self.method_notes(method),
                }
            except Exception as exc:  # noqa: BLE001 — row-level resilience
                row = {
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "Dataset": dataset_name,
                    "Suite": self.name,
                    "NLL": None,
                    "train_s": None,
                    "eval_s": None,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(_dev()),
                    "Notes": f"failed: {type(exc).__name__}: {exc}",
                }
            rows.append(row)
        return rows
