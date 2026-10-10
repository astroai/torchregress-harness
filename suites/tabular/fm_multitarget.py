"""Multi-target regression with tabular foundation models vs joint density estimators.

Tabular foundation models (TabICL v2, KumoTabular, TabFM, TabPFN) predict one
target at a time.  This suite asks how much that costs on real multi-target
data: each model is fitted once per target, its marginal quantile grids are
joined into a joint predictive distribution, and the result is scored against
torchregress's joint density estimators (NSF flow and full-covariance MDN,
trained with the shared ``density_parity`` loop).

Joint methods per base (row ``<method>+<base>``):
* ``Indep`` -- the model's own marginals, sampled independently (what a
  single-target model gives you out of the box).
* ``GaussCopula`` -- the same marginals joined by a Gaussian copula whose
  correlation is estimated from the normal scores of the calibration PITs
  (shrunk 5% towards the identity).
* ``tr_PITRecal+GaussCopula`` -- marginals first recalibrated with torchregress
  ``PITCalibrator``, then the copula.

Reference rows: ``torchregress_NSF`` / ``torchregress_MDN`` (one joint model,
trained on train + calibration rows).

Scores (standardised targets, ``density_parity._scores`` on 200 samples per test
row): energy score ``ES``, variogram score ``VS``, mean marginal ``CRPS``,
``Cov90_marg``, ``PIT_KS``, ``CorrErr`` (predicted vs empirical residual
correlation), ``NLL`` / ``Cov90_joint`` (density models only), and
``Cov90_ell`` -- joint coverage of the 90% Mahalanobis ellipsoid fitted to each
row's samples, defined the same way for every method.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites.tabular import _fm_grid as G

N_TRAIN_MAX = 6000
N_TEST_MAX = 1000
CAL_FRACTION = 0.2
N_SAMPLES = 200
COPULA_SHRINK = 0.05
JOINT_METHODS = ("Indep", "GaussCopula", "tr_PITRecal+GaussCopula")
DENSITY_REFS = ("torchregress_NSF", "torchregress_MDN")


def copula_corr(pit_cal: np.ndarray) -> np.ndarray:
    """Gaussian-copula correlation from calibration PITs [n, D] (shrunk to I)."""
    z = G.ppf_normal(pit_cal)
    R = np.corrcoef(z.T) if z.shape[1] > 1 else np.ones((1, 1))
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    return (1 - COPULA_SHRINK) * R + COPULA_SHRINK * np.eye(len(R))


def sample_joint(
    q_test: list[np.ndarray], n_s: int, rng: np.random.Generator, R: np.ndarray | None
) -> np.ndarray:
    """Samples [N, S, D] from marginal grids (independent if R is None)."""
    N, D = q_test[0].shape[0], len(q_test)
    if R is None:
        u = rng.uniform(size=(N, n_s, D))
    else:
        L = np.linalg.cholesky(R + 1e-9 * np.eye(D))
        u = G.cdf_normal(rng.standard_normal((N, n_s, D)) @ L.T)
    return np.stack([G.quantile_at(q_test[d], u[:, :, d]) for d in range(D)], axis=-1)


def ellipsoid_coverage(y: np.ndarray, S: np.ndarray, level: float = 0.9) -> float:
    """Fraction of rows whose truth lies in the level-ellipsoid of its own samples."""
    hits = []
    for yi, Si in zip(y, S, strict=True):
        mu = Si.mean(0)
        C = np.atleast_2d(np.cov(Si.T)) + 1e-9 * np.eye(Si.shape[1])
        P = np.linalg.pinv(C)
        d_s = np.einsum("sd,de,se->s", Si - mu, P, Si - mu)
        d_y = float((yi - mu) @ P @ (yi - mu))
        hits.append(d_y <= np.quantile(d_s, level))
    return float(np.mean(hits))


class FMMultiTargetSuite(HarnessRunner):
    """Per-target foundation-model marginals + copula vs joint density models."""

    @property
    def name(self) -> str:
        return "fm_multitarget"

    @property
    def description(self) -> str:
        return (
            "Multi-target: tabular foundation models fitted per target (independent or "
            "Gaussian-copula joint) vs torchregress NSF / MDN joint densities"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["mt_enb", "mt_jura", "mt_scm20d", "mt_rf1"]

    def bases(self) -> list[str]:
        return G.available_bases()

    def methods(self) -> list[str]:
        return [*DENSITY_REFS, *(f"{m}+{b}" for b in self.bases() for m in JOINT_METHODS)]

    def library_name(self, method: str) -> str:
        if method in DENSITY_REFS or method.startswith("tr_"):
            return "torchregress"
        return G.library_of(method.rpartition("+")[2])

    def library_version(self, method: str) -> str:
        return ""

    # -- density references (shared density_parity trainer) --------------------
    def _density_ref(
        self,
        method: str,
        X: np.ndarray,
        Y: np.ndarray,
        Xte: np.ndarray,
        Yte: np.ndarray,
        cfg: HarnessConfig,
        n_s: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        from suites.tabular.density_parity import DensityParitySuite, _fit_shared

        dp = DensityParitySuite()
        dp._image_channels = None
        state = dp._build(method, X.shape[1], Y.shape[1], Y, X)
        epochs = _fit_shared(state["modules"], dp._nll_fn(state), X, Y, cfg)
        lp, S, lp_s = dp._log_prob_and_sample(state, Xte, Yte, n_s)
        return lp, S, lp_s, epochs

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        from suites.tabular.density_parity import _scores

        cfg = config or HarnessConfig()
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name)
        rng = np.random.default_rng(cfg.seed)
        X = np.asarray(splits["X_train"], dtype=np.float32)
        Y = np.asarray(splits["y_train"], dtype=np.float64)
        Xte = np.asarray(splits["X_test"], dtype=np.float32)
        Yte = np.asarray(splits["y_test"], dtype=np.float64)
        if Y.ndim == 1:
            Y, Yte = Y[:, None], Yte[:, None]
        n_tr, n_te = (400, 64) if cfg.smoke else (N_TRAIN_MAX, N_TEST_MAX)
        if len(X) > n_tr:
            idx = rng.choice(len(X), n_tr, replace=False)
            X, Y = X[idx], Y[idx]
        if len(Xte) > n_te:
            idx = rng.choice(len(Xte), n_te, replace=False)
            Xte, Yte = Xte[idx], Yte[idx]
        mu, sd = Y.mean(0), Y.std(0) + 1e-8
        Y, Yte = (Y - mu) / sd, (Yte - mu) / sd
        perm = rng.permutation(len(X))
        n_cal = int(CAL_FRACTION * len(X))
        cal, fit = perm[:n_cal], perm[n_cal:]
        D = Y.shape[1]
        n_s = 50 if cfg.smoke else N_SAMPLES
        common = {
            "Suite": self.name,
            "Dataset": dataset_name,
            "NTargets": D,
            "NTrain": len(fit),
            "NCal": len(cal),
            "NTest": len(Xte),
            "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
            "Device": str(current_device()),
        }
        rows: list[dict[str, Any]] = []

        def _row(method: str, extra: dict[str, Any]) -> None:
            rows.append(
                {
                    **common,
                    "Method": f"{self.name}/{method}",
                    "Library": self.library_name(method),
                    "LibraryVersion": "",
                    **extra,
                }
            )

        for method in DENSITY_REFS:
            set_harness_seed(cfg.seed)
            try:
                t0 = perf_counter()
                lp, S, lp_s, epochs = self._density_ref(
                    method, X, Y.astype(np.float32), Xte, Yte.astype(np.float32), cfg, n_s
                )
                sc = _scores(Yte, S, lp, lp_s)
                sc["Cov90_ell"] = ellipsoid_coverage(Yte, S)
                _row(
                    method,
                    {
                        **sc,
                        "Epochs": epochs,
                        "train_s": perf_counter() - t0,
                        "Notes": "joint density (density_parity trainer)",
                    },
                )
            except Exception as exc:  # noqa: BLE001 — row-level resilience
                _row(method, {"Notes": f"failed: {type(exc).__name__}: {exc}"[:300]})

        for base in self.bases():
            set_harness_seed(cfg.seed)
            try:
                q_cal, q_test, fit_s = [], [], 0.0
                for d in range(D):
                    (qc, qt), s = G.fit_grids(base, X[fit], Y[fit, d], [X[cal], Xte], cfg)
                    q_cal.append(qc)
                    q_test.append(qt)
                    fit_s += s
            except Exception as exc:  # noqa: BLE001 — every row of this base fails
                for m in JOINT_METHODS:
                    _row(
                        f"{m}+{base}",
                        {"Notes": f"failed: base {base}: {type(exc).__name__}: {exc}"[:300]},
                    )
                continue
            pit_cal = np.stack([G.cdf_at(q_cal[d], Y[cal, d]) for d in range(D)], axis=1)
            for m in JOINT_METHODS:
                try:
                    t1 = perf_counter()
                    srng = np.random.default_rng(cfg.seed + 7)
                    if m == "Indep":
                        S = sample_joint(q_test, n_s, srng, None)
                    elif m == "GaussCopula":
                        S = sample_joint(q_test, n_s, srng, copula_corr(pit_cal))
                    else:
                        qt_r = [G.pit_recalibrate(q_cal[d], Y[cal, d], q_test[d]) for d in range(D)]
                        qc_r = [G.pit_recalibrate(q_cal[d], Y[cal, d], q_cal[d]) for d in range(D)]
                        pit_r = np.stack([G.cdf_at(qc_r[d], Y[cal, d]) for d in range(D)], axis=1)
                        S = sample_joint(qt_r, n_s, srng, copula_corr(pit_r))
                    sc = _scores(Yte, S, np.full(len(Yte), np.nan), None)
                    sc["NLL"] = None
                    sc["Cov90_ell"] = ellipsoid_coverage(Yte, S)
                    _row(
                        f"{m}+{base}",
                        {
                            **sc,
                            "Base": base,
                            "train_s": fit_s,
                            "eval_s": perf_counter() - t1,
                            "Notes": f"{m}: {D} per-target {base} fits",
                        },
                    )
                except Exception as exc:  # noqa: BLE001 — row-level resilience
                    _row(
                        f"{m}+{base}",
                        {"Base": base, "Notes": f"failed: {type(exc).__name__}: {exc}"[:300]},
                    )
        return rows

    # HarnessRunner abstract surface (unused: custom run drives everything).
    def fit(self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig) -> Any:
        raise NotImplementedError("fm_multitarget uses custom run()")

    def predict(
        self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("fm_multitarget uses custom run()")

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        raise NotImplementedError("fm_multitarget uses custom run()")
