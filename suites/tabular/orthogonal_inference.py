"""Orthogonal (double/debiased ML) inference suite.

Benchmarks ``torchregress.inference.orthogonal_partially_linear`` against
DoubleML and EconML on Monte Carlo replications of partially linear models
with a known treatment effect::

    y = theta * d + g(x) + u,    d = m(x) + v.

Each replication draws a fresh sample, every method estimates ``theta`` with a
95% interval, and the suite reports, over replications: bias, RMSE, interval
coverage, mean interval width, and ``SEratio`` (mean reported standard error /
empirical standard deviation of the estimates; 1 = honest standard errors).

Pairs share the nuisance learner so differences come from the estimator, not
the regression model: ``tr_Orthogonal_Poly3`` vs ``DoubleML_PLR_Poly3Ridge``
(the same per-covariate cubic ridge basis); ``tr_Naive`` is the biased OLS
baseline every orthogonal estimator must beat.

Datasets are generated in-suite (synthetic by design, known ``theta``):

* ``plr_ccddhnr`` - Chernozhukov et al. (2018) design as in DoubleML's
  ``make_plr_CCDDHNR2018``: 20 Toeplitz-correlated covariates, logistic and
  linear nuisances, theta = 0.5, n = 500.
* ``plr_cubic`` - nuisances exactly cubic in 5 independent covariates, so the
  cubic basis is well specified, theta = 1.0, n = 500.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch

from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner

N_REPLICATIONS = 200
N_REPLICATIONS_SMOKE = 10
FOLDS = 5
CONFIDENCE = 0.95


@dataclass(frozen=True)
class PLRDesign:
    """A partially linear data-generating process with known ``theta``."""

    name: str
    theta: float
    n: int
    p: int
    draw: Callable[[np.random.Generator, int, int, float], tuple[np.ndarray, ...]]


def _sigmoid(t: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-t))


def _draw_ccddhnr(
    rng: np.random.Generator, n: int, p: int, theta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(p)
    cov = 0.7 ** np.abs(np.subtract.outer(idx, idx))
    x = rng.multivariate_normal(np.zeros(p), cov, size=n)
    d = x[:, 0] + 0.25 * _sigmoid(x[:, 2]) + rng.normal(size=n)
    y = theta * d + _sigmoid(x[:, 0]) + 0.25 * x[:, 2] + rng.normal(size=n)
    return y, d, x


def _draw_cubic(
    rng: np.random.Generator, n: int, p: int, theta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, p))
    # Shared x0, x1, x2 terms confound d and y (naive OLS slope is off by ~0.8).
    d = x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.3 * x[:, 2] ** 3 + rng.normal(size=n)
    y = (
        theta * d
        + 0.8 * x[:, 0] ** 3
        + x[:, 1] ** 2
        - 0.5 * x[:, 2]
        + 0.5 * x[:, 3]
        + rng.normal(size=n)
    )
    return y, d, x


DESIGNS: dict[str, PLRDesign] = {
    "plr_ccddhnr": PLRDesign("plr_ccddhnr", theta=0.5, n=500, p=20, draw=_draw_ccddhnr),
    "plr_cubic": PLRDesign("plr_cubic", theta=1.0, n=500, p=5, draw=_draw_cubic),
}


@dataclass(frozen=True)
class Estimate:
    theta: float
    se: float
    ci_low: float
    ci_high: float


def _per_covariate_powers(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """x, x^2, ..., x^degree per covariate (no interactions), as in torchregress."""
    return np.concatenate([x**k for k in range(1, degree + 1)], axis=1)


def _rff(x: np.ndarray, *, seed: int, n_features: int = 256) -> np.ndarray:
    """Random Fourier features of an RBF kernel with unit bandwidth per standardized covariate."""
    g = np.random.default_rng(seed)
    w = g.normal(size=(x.shape[1], n_features)) / math.sqrt(x.shape[1])
    b = g.uniform(0.0, 2.0 * math.pi, size=n_features)
    return math.sqrt(2.0 / n_features) * np.cos(x @ w + b)


def summarize(estimates: list[Estimate], theta: float) -> dict[str, float]:
    """Monte Carlo summary of interval estimates of a known ``theta``."""
    est = np.array([e.theta for e in estimates])
    se = np.array([e.se for e in estimates])
    lo = np.array([e.ci_low for e in estimates])
    hi = np.array([e.ci_high for e in estimates])
    sd = float(est.std(ddof=1)) if len(est) > 1 else float("nan")
    return {
        "Bias": float(est.mean() - theta),
        "RMSE": float(np.sqrt(np.mean((est - theta) ** 2))),
        "Coverage95": float(np.mean((lo <= theta) & (theta <= hi))),
        "CIWidth": float(np.mean(hi - lo)),
        "SEratio": float(se.mean() / sd) if sd > 0 else float("nan"),
        "EmpiricalSD": sd,
    }


class OrthogonalInferenceSuite(HarnessRunner):
    """torchregress orthogonal estimator vs DoubleML / EconML on known-effect PLR designs."""

    @property
    def name(self) -> str:
        return "orthogonal_inference"

    @property
    def description(self) -> str:
        return "Partially linear DML: torchregress vs DoubleML/EconML (bias, coverage)"

    @property
    def supported_datasets(self) -> list[str]:
        return list(DESIGNS)

    def methods(self) -> list[str]:
        return [
            "tr_Orthogonal_Poly3",
            "tr_Orthogonal_RFF",
            "tr_Orthogonal_RFF_legacy",
            "tr_Orthogonal_NoCrossFit",
            "tr_Naive",
            "DoubleML_PLR_Poly3Ridge",
            "DoubleML_PLR_Lasso",
            "DoubleML_PLR_RF",
            "EconML_LinearDML_Lasso",
        ]

    def library_name(self, method: str) -> str:
        if method.startswith("tr_"):
            return "torchregress"
        return method.split("_", 1)[0].lower()

    def library_version(self, method: str) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(self.library_name(method))
        except PackageNotFoundError:
            return "not installed"

    def method_notes(self, method: str) -> str:
        return {
            "tr_Orthogonal_Poly3": "orthogonal_partially_linear, cubic per-covariate ridge, 5 folds",
            "tr_Orthogonal_RFF": (
                "orthogonal_partially_linear with inference.random_fourier_features "
                "(256 features, median bandwidth, polynomial_degree=3) and ridge='gcv'"
            ),
            "tr_Orthogonal_RFF_legacy": (
                "pre-d67289a3 hand-rolled recipe for comparison: 256 random Fourier features, "
                "W ~ N(0, 1/K) (unit bandwidth, unstandardised), no linear terms, ridge=1e-2"
            ),
            "tr_Orthogonal_NoCrossFit": "folds=1 (no cross-fitting): diagnostic only",
            "tr_Naive": "naive_linear_estimate: OLS of y on d, ignores x (biased baseline)",
            "DoubleML_PLR_Poly3Ridge": "DoubleMLPLR, same cubic ridge basis; pair: tr_Orthogonal_Poly3",
            "DoubleML_PLR_Lasso": "DoubleMLPLR, LassoCV nuisances on raw covariates",
            "DoubleML_PLR_RF": "DoubleMLPLR, random-forest nuisances (100 trees, depth 5)",
            "EconML_LinearDML_Lasso": "econml LinearDML, LassoCV nuisances, constant effect",
        }.get(method, "")

    # ------------------------------------------------------------------
    # One estimate per (method, replication)
    # ------------------------------------------------------------------

    def _estimate(
        self, method: str, y: np.ndarray, d: np.ndarray, x: np.ndarray, seed: int
    ) -> Estimate:
        if method.startswith("tr_"):
            from torchregress.inference import (
                naive_linear_estimate,
                orthogonal_partially_linear,
            )

            yt, dt = torch.as_tensor(y), torch.as_tensor(d)
            if method == "tr_Naive":
                r = naive_linear_estimate(yt, dt, confidence=CONFIDENCE)
            else:
                kwargs: dict[str, Any] = {"folds": FOLDS, "confidence": CONFIDENCE, "seed": seed}
                if method == "tr_Orthogonal_RFF":
                    from torchregress.inference import random_fourier_features

                    kwargs["nuisance_features"] = lambda z: random_fourier_features(z, seed=seed)
                    kwargs["ridge"] = "gcv"
                elif method == "tr_Orthogonal_RFF_legacy":
                    kwargs["nuisance_features"] = lambda z: torch.as_tensor(
                        _rff(z.numpy(), seed=seed)
                    )
                    kwargs["ridge"] = 1e-2
                elif method == "tr_Orthogonal_NoCrossFit":
                    kwargs["folds"] = 1
                r = orthogonal_partially_linear(yt, dt, torch.as_tensor(x), **kwargs)
            return Estimate(r.theta, r.sigma, r.ci_low, r.ci_high)

        if method.startswith("DoubleML_"):
            from doubleml import DoubleMLData, DoubleMLPLR

            learner = self._sklearn_learner(method, seed)
            dml = DoubleMLPLR(
                DoubleMLData.from_arrays(x, y, d),
                ml_l=learner,
                ml_m=self._sklearn_learner(method, seed),
                n_folds=FOLDS,
            )
            np.random.seed(seed)  # DoubleML draws its folds from the global RNG
            dml.fit()
            lo, hi = dml.confint(level=CONFIDENCE).to_numpy()[0]
            return Estimate(float(dml.coef[0]), float(dml.se[0]), float(lo), float(hi))

        if method == "EconML_LinearDML_Lasso":
            from econml.dml import LinearDML
            from sklearn.linear_model import LassoCV

            est = LinearDML(model_y=LassoCV(), model_t=LassoCV(), cv=FOLDS, random_state=seed)
            est.fit(y, d, X=None, W=x)
            lo, hi = est.intercept__interval(alpha=1.0 - CONFIDENCE)
            se = float(est.intercept__inference().stderr)
            return Estimate(float(est.intercept_), se, float(lo), float(hi))

        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _sklearn_learner(method: str, seed: int) -> Any:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LassoCV, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer

        if method == "DoubleML_PLR_Poly3Ridge":
            # alpha matches torchregress's default ridge; its basis carries an
            # explicit intercept column, sklearn fits the intercept instead.
            return make_pipeline(
                FunctionTransformer(_per_covariate_powers), Ridge(alpha=1e-6, solver="cholesky")
            )
        if method == "DoubleML_PLR_Lasso":
            return LassoCV(random_state=seed)
        if method == "DoubleML_PLR_RF":
            return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=seed)
        raise ValueError(f"No sklearn learner for {method}")

    # ------------------------------------------------------------------
    # Monte Carlo run
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        del splits, scenario_name
        cfg = config or HarnessConfig()
        design = DESIGNS.get(dataset_name)
        n_rep = N_REPLICATIONS_SMOKE if cfg.smoke else N_REPLICATIONS
        rows: list[dict[str, Any]] = []
        for method in self.methods():
            base = {
                "Method": f"{self.name}/{method}",
                "Library": self.library_name(method),
                "LibraryVersion": self.library_version(method),
                "Dataset": dataset_name,
                "Suite": self.name,
                "SourceURL": "",
                "SyntheticFallback": False,  # synthetic by design: known theta
                "Device": str(current_device()),
            }
            if design is None:
                rows.append({**base, "Notes": f"failed: unknown design {dataset_name!r}"})
                continue
            try:
                estimates: list[Estimate] = []
                t0 = perf_counter()
                for rep in range(n_rep):
                    seed = cfg.seed * 100_003 + rep
                    y, d, x = design.draw(
                        np.random.default_rng(seed), design.n, design.p, design.theta
                    )
                    estimates.append(self._estimate(method, y, d, x, seed))
                elapsed = perf_counter() - t0
                rows.append(
                    {
                        **base,
                        **summarize(estimates, design.theta),
                        "Theta": design.theta,
                        "Replications": n_rep,
                        "train_s": elapsed / n_rep,
                        "eval_s": 0.0,
                        "Notes": self.method_notes(method),
                    }
                )
            except Exception as exc:  # noqa: BLE001 — one method failing must not drop the rest
                rows.append({**base, "Notes": f"failed: {type(exc).__name__}: {exc}"})
        return rows

    # HarnessRunner abstract surface (unused: custom run drives everything).
    def fit(self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig) -> Any:
        raise NotImplementedError("orthogonal_inference uses custom run()")

    def predict(
        self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("orthogonal_inference uses custom run()")

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        raise NotImplementedError("orthogonal_inference uses custom run()")
