"""Tabular foundation models + the torchregress UQ layer under stress scenarios.

Question: given a strong pretrained tabular model (TabICL v2, KumoTabular,
TabFM, TabPFN when a Prior Labs key is present), how good is its *own*
predictive distribution on real data -- clean, contaminated, shifted,
extrapolated, with noisy features, imbalanced or heavy-tailed -- and does the
torchregress post-hoc layer repair it?  LightGBM quantile regression and a
TabularMLP Gaussian ensemble are the reference bases.

Every base is fitted once per (dataset, scenario, seed) and reduced to a
predictive quantile grid (``_fm_grid.LEVELS``, 103 levels); every method then
uses the same calibration and test rows.

Scenarios (row ``Dataset`` = ``<name>@<scenario>``), all built from the
pooled real data (at most 6000 rows), 60 / 20 / 20 train / calibration / test
-------------------------------------------------------------------------------
* ``clean`` -- random split.
* ``outliers`` -- 5% of train *and* calibration targets shifted by 5-10 target
  standard deviations (random sign); test targets clean.
* ``shift`` -- covariate shift: test rows drawn with probability proportional to
  ``exp(z)``, ``z`` the standardised first principal component of X.  The
  weighted methods estimate ``w(x) = p(test | x) / p(cal | x)`` with a logistic
  regression (no oracle weights).
* ``extrapolate`` -- test = the 20% of rows with the largest first principal
  component; no train / calibration row lies there (out of support).
* ``noise_test`` -- Gaussian noise (0.5 x feature sd) added to the test
  features only: a deployment-time sensor degradation.
* ``noise_all`` -- the same noise on train, calibration and test features
  (errors-in-variables everywhere: attenuation of the fitted signal).
* ``imbalance`` -- rows with target above the pooled 80th percentile are kept
  with probability 0.1 in train and calibration; the test split is not thinned
  (rare-target regime, scored by ``TailCoverage``).
* ``heavy_tail`` -- targets mapped by ``sinh(1.5 y) / 1.5`` then standardised:
  very long tails on both sides.

Methods (row ``<method>+<base>``)
---------------------------------
Distribution methods (full predictive distribution -> every score):
* ``Raw`` -- the base model's own quantile grid.
* ``tr_PITRecal`` -- quantile recalibration of the whole grid with torchregress
  ``PITCalibrator`` (Kuleshov et al. 2018).
* ``tr_ResidualDist`` -- split conformal predictive distribution: base median
  plus calibration-residual quantiles (homoscedastic baseline; the only
  distribution available for point-only models).

Point-only bases (``PointOnly`` column; TabFM in sdm has a single regression
output) run ``tr_ResidualDist``, ``tr_Split``, ``tr_Mondrian`` (and
``tr_WeightedSplit`` under shift) only.

Interval methods (calibrated at each of the 50 / 80 / 90 / 95% levels):
* ``tr_Split`` -- ``SplitConformal`` on the base median.
* ``tr_Normalized`` -- residual / spread scores (spread = base 90% width).
* ``tr_CQR`` -- ``CQR`` on the base's own quantiles at that level.
* ``tr_Mondrian`` -- ``SplitConformal`` per bin of the predicted median (up to 5
  bins of >= 40 calibration rows).
* ``tr_DCP`` -- ``DistributionalConformal`` (PIT-band scores, CDF inversion).
* ``tr_WeightedSplit`` / ``tr_WeightedCQR`` -- ``shift`` only, weighted
  conformal (Tibshirani et al. 2019) with the estimated density ratio.

Scores (standardised target units)
----------------------------------
``Coverage50/80/90/95``; ``Width`` / ``IntervalScore`` / ``TailCoverage`` (test
rows above the clean training 90th percentile) / ``WorstDecileCoverage`` (over
the ten clean-training target deciles), all at 90%; ``RMSE`` and ``MAE`` of the
median; and, for distribution methods, ``CRPS`` and ``PIT_KS``.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import torch

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites.tabular import _fm_grid as G

LEVELS_SCORED = (0.5, 0.2, 0.1, 0.05)  # miscoverage alphas -> Coverage50/80/90/95
ALPHA = 0.1  # headline interval level for width / interval score / tail metrics
N_MAX = 6000
OUTLIER_FRAC = 0.05
SHIFT_BETA = 1.0
EXTRAPOLATE_FRAC = 0.2
NOISE_SD = 0.5
IMBALANCE_QUANTILE = 0.8
IMBALANCE_KEEP = 0.1
HEAVY_DELTA = 1.5
N_MONDRIAN_BINS = 5
MIN_PER_BIN = 40
SCENARIOS = (
    "clean",
    "outliers",
    "shift",
    "extrapolate",
    "noise_test",
    "noise_all",
    "imbalance",
    "heavy_tail",
)
DIST_METHODS = ("Raw", "tr_PITRecal", "tr_ResidualDist")
# Point-only bases (TabFM in sdm: one regression output) have no own distribution.
POINT_METHODS = ("tr_ResidualDist", "tr_Split", "tr_Mondrian")
POINT_SHIFT_METHODS = ("tr_WeightedSplit",)
INTERVAL_METHODS = ("tr_Split", "tr_Normalized", "tr_CQR", "tr_Mondrian", "tr_DCP")
SHIFT_METHODS = ("tr_WeightedSplit", "tr_WeightedCQR")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _pc1(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc[: min(len(X), 5000)], full_matrices=False)
    z = Xc @ vt[0]
    return (z - z.mean()) / (z.std() + 1e-12)


def scenario_splits(X: np.ndarray, y: np.ndarray, scenario: str, seed: int) -> dict[str, Any]:
    """Train / calibration / test rows for one scenario (targets standardised)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if scenario == "heavy_tail":
        y = np.sinh(HEAVY_DELTA * y) / HEAVY_DELTA
        y = (y - y.mean()) / (y.std() + 1e-12)
    n = len(X)
    n_test = n // 5
    if scenario == "shift":
        p = np.exp(SHIFT_BETA * _pc1(X))
        test = rng.choice(n, n_test, replace=False, p=p / p.sum())
    elif scenario == "extrapolate":
        test = np.argsort(_pc1(X))[-int(EXTRAPOLATE_FRAC * n) :]
    else:
        test = rng.permutation(n)[:n_test]
    rest = rng.permutation(np.setdiff1d(np.arange(n), test))
    if scenario == "imbalance":
        hi = y[rest] > np.quantile(y, IMBALANCE_QUANTILE)
        keep = ~hi | (rng.random(len(rest)) < IMBALANCE_KEEP)
        rest = rest[keep]
    n_cal = len(rest) // 4
    cal, train = rest[:n_cal], rest[n_cal:]
    Xs = {k: X[idx].copy() for k, idx in (("train", train), ("cal", cal), ("test", test))}
    if scenario in ("noise_test", "noise_all"):
        sd = X[train].std(0, keepdims=True) + 1e-12
        for k in ("test",) if scenario == "noise_test" else ("train", "cal", "test"):
            Xs[k] = (Xs[k] + NOISE_SD * sd * rng.standard_normal(Xs[k].shape)).astype(np.float32)
    y_train, y_cal = y[train].copy(), y[cal].copy()
    if scenario == "outliers":
        sd_y = float(np.std(y[train]))
        for arr in (y_train, y_cal):
            k = max(1, int(OUTLIER_FRAC * len(arr)))
            idx = rng.choice(len(arr), k, replace=False)
            arr[idx] += rng.choice([-1.0, 1.0], k) * rng.uniform(5.0, 10.0, k) * sd_y
    return {
        "X_train": Xs["train"],
        "y_train": y_train,
        "X_cal": Xs["cal"],
        "y_cal": y_cal,
        "X_test": Xs["test"],
        "y_test": y[test],
        "y_train_clean": y[train],
    }


def density_ratio(
    X_cal: np.ndarray, X_test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Estimated w(x) = p(test | x) / p(cal | x) on calibration and test rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xall = np.vstack([X_cal, X_test])
    lab = np.r_[np.zeros(len(X_cal)), np.ones(len(X_test))]
    sc = StandardScaler().fit(Xall)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(sc.transform(Xall), lab)
    prior = len(X_cal) / len(X_test)

    def w(Xe: np.ndarray) -> np.ndarray:
        p = clf.predict_proba(sc.transform(Xe))[:, 1]
        return np.clip(prior * p / np.clip(1 - p, 1e-6, None), 1e-2, 1e2)

    return w(X_cal), w(X_test)


# ---------------------------------------------------------------------------
# torchregress interval methods on a quantile grid
# ---------------------------------------------------------------------------


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


def _np(a: torch.Tensor) -> np.ndarray:
    return a.detach().cpu().numpy().astype(np.float64).reshape(-1)


def interval_method(
    method: str,
    q_cal: np.ndarray,
    q_test: np.ndarray,
    y_cal: np.ndarray,
    alpha: float,
    weights: tuple[np.ndarray, np.ndarray] | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) on the test rows for one torchregress conformal method.

    ``info``, when given, receives diagnostics (``unseen_test_rows`` for
    ``tr_Mondrian``: test rows whose bin has no calibration row).
    """
    from torchregress.losses.conformal import CQR, DistributionalConformal, SplitConformal

    med_c, med_t = G.quantile_at(q_cal, 0.5), G.quantile_at(q_test, 0.5)
    yc = _t(y_cal)
    if method in ("tr_Split", "tr_WeightedSplit"):
        cp = SplitConformal(alpha=alpha)
        if method == "tr_WeightedSplit":
            assert weights is not None
            cp.calibrate(_t(med_c), yc, weights=_t(weights[0]))
            lo, hi = cp.predict_interval(_t(med_t), test_weights=_t(weights[1]))
        else:
            cp.calibrate(_t(med_c), yc)
            lo, hi = cp.predict_interval(_t(med_t))
    elif method == "tr_Normalized":
        lo_c, hi_c = G.interval(q_cal, ALPHA)
        lo_t, hi_t = G.interval(q_test, ALPHA)
        floor = max(0.05 * float(np.median(hi_c - lo_c)), 1e-6)
        s_c, s_t = np.maximum(hi_c - lo_c, floor), np.maximum(hi_t - lo_t, floor)
        cp = SplitConformal(alpha=alpha, normalize_fn=lambda _p, spread: spread)
        cp.calibrate(_t(med_c), yc, x=_t(s_c))
        lo, hi = cp.predict_interval(_t(med_t), x=_t(s_t))
    elif method in ("tr_CQR", "tr_WeightedCQR"):
        qc = np.stack(G.interval(q_cal, alpha), axis=1)
        qt = np.stack(G.interval(q_test, alpha), axis=1)
        cq = CQR(alpha=alpha)
        if method == "tr_WeightedCQR":
            assert weights is not None
            cq.calibrate(_t(qc), yc, weights=_t(weights[0]))
            lo, hi = cq.predict_interval(_t(qt), test_weights=_t(weights[1]))
        else:
            cq.calibrate(_t(qc), yc)
            lo, hi = cq.predict_interval(_t(qt))
    elif method == "tr_Mondrian":
        n_bins = max(1, min(N_MONDRIAN_BINS, len(y_cal) // MIN_PER_BIN))
        # unique: tied medians (tree models) would otherwise leave a bin empty
        cuts = np.unique(np.quantile(med_c, np.linspace(0, 1, n_bins + 1)[1:-1]))
        g_c, g_t = np.searchsorted(cuts, med_c), np.searchsorted(cuts, med_t)
        # The library rejects test groups absent from calibration (a gap in the
        # calibration medians, e.g. under imbalance thinning). Those rows take
        # the pooled split-conformal threshold: marginal coverage is kept, group
        # conditionality is not claimed for them.
        seen = np.isin(g_t, np.unique(g_c))
        lo, hi = np.empty(len(med_t)), np.empty(len(med_t))
        if seen.any():
            cp = SplitConformal(alpha=alpha)
            cp.calibrate(_t(med_c), yc, groups=torch.as_tensor(g_c))
            lo_s, hi_s = cp.predict_interval(_t(med_t[seen]), groups=torch.as_tensor(g_t[seen]))
            lo[seen], hi[seen] = _np(lo_s), _np(hi_s)
        if info is not None:
            info["unseen_test_rows"] = int((~seen).sum())
        if (~seen).any():
            pooled_cp = SplitConformal(alpha=alpha)
            pooled_cp.calibrate(_t(med_c), yc)
            lo_p, hi_p = pooled_cp.predict_interval(_t(med_t[~seen]))
            lo[~seen], hi[~seen] = _np(lo_p), _np(hi_p)
        return lo, hi
    elif method == "tr_DCP":
        dcp = DistributionalConformal(alpha=alpha)
        dcp.calibrate(_t(G.cdf_at(q_cal, y_cal)), yc)

        def icdf(levels: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            rows = q_test[idx.long().cpu().numpy().reshape(-1)]
            lv = levels.detach().cpu().numpy()
            return torch.as_tensor(np.stack([G.quantile_at(rows, float(t)) for t in lv], axis=1))

        lo, hi = dcp.predict_intervals_from_cdf(
            icdf, torch.arange(len(q_test), dtype=torch.float64).reshape(-1, 1)
        )
    else:
        raise ValueError(method)
    return _np(lo), _np(hi)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def _interval_scores(lo: np.ndarray, hi: np.ndarray, s: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(s["y_test"], dtype=np.float64)
    inside = (y >= lo) & (y <= hi)
    finite = np.isfinite(lo) & np.isfinite(hi)
    width = np.where(finite, hi - lo, np.nan)
    isc = width + (2 / ALPHA) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))
    y_tr = np.asarray(s["y_train_clean"], dtype=np.float64)
    tail = y >= np.quantile(y_tr, 0.9)
    dec = np.searchsorted(np.quantile(y_tr, np.linspace(0, 1, 11)[1:-1]), y)
    dec_cov = [inside[dec == k].mean() for k in range(10) if (dec == k).sum() >= 5]
    return {
        "Width": float(np.nanmean(width)) if finite.any() else float("inf"),
        "IntervalScore": float(np.nanmean(isc)) if finite.any() else float("inf"),
        "InfiniteIntervalFrac": float(1 - finite.mean()),
        "TailCoverage": float(inside[tail].mean()) if tail.any() else None,
        "WorstDecileCoverage": float(min(dec_cov)) if dec_cov else None,
    }


def _point_scores(q_test: np.ndarray, y: np.ndarray) -> dict[str, float]:
    med = G.quantile_at(q_test, 0.5)
    return {
        "RMSE": float(np.sqrt(np.mean((med - y) ** 2))),
        "MAE": float(np.mean(np.abs(med - y))),
    }


def score_distribution(q_test: np.ndarray, s: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(s["y_test"], dtype=np.float64)
    out: dict[str, Any] = {}
    for a in LEVELS_SCORED:
        lo, hi = G.interval(q_test, a)
        out[f"Coverage{round(100 * (1 - a))}"] = float(np.mean((y >= lo) & (y <= hi)))
    out.update(_interval_scores(*G.interval(q_test, ALPHA), s))
    out.update(_point_scores(q_test, y))
    out["CRPS"] = float(np.mean(G.crps(q_test, y)))
    out["PIT_KS"] = G.pit_ks(G.cdf_at(q_test, y))
    return out


def score_interval_method(
    method: str,
    q_cal: np.ndarray,
    q_test: np.ndarray,
    s: dict[str, Any],
    weights: tuple[np.ndarray, np.ndarray] | None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y = np.asarray(s["y_test"], dtype=np.float64)
    out: dict[str, Any] = {}
    for a in LEVELS_SCORED:
        lo, hi = interval_method(method, q_cal, q_test, s["y_cal"], a, weights, info)
        out[f"Coverage{round(100 * (1 - a))}"] = float(np.mean((y >= lo) & (y <= hi)))
        if a == ALPHA:
            out.update(_interval_scores(lo, hi, s))
    out.update(_point_scores(q_test, y))
    out["CRPS"] = None
    out["PIT_KS"] = None
    return out


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


def pooled(splits: dict[str, Any], n_max: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([np.asarray(splits["X_train"]), np.asarray(splits["X_test"])]).astype(np.float32)
    y = np.concatenate(
        [np.asarray(splits["y_train"]).reshape(-1), np.asarray(splits["y_test"]).reshape(-1)]
    ).astype(np.float64)
    if len(X) > n_max:
        idx = np.random.default_rng(seed).choice(len(X), n_max, replace=False)
        X, y = X[idx], y[idx]
    return X, (y - y.mean()) / (y.std() + 1e-12)


class FMUQScenariosSuite(HarnessRunner):
    """Foundation-model bases x torchregress UQ layer x stress scenarios (custom run)."""

    @property
    def name(self) -> str:
        return "fm_uq_scenarios"

    @property
    def description(self) -> str:
        return (
            "Tabular foundation models (TabICL v2 / KumoTabular / TabFM / TabPFN) with and "
            "without the torchregress UQ layer: distribution quality, OOD shift and "
            "extrapolation, feature noise, outliers, imbalance and heavy tails"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return [
            "california_housing",
            "house_sales",
            "insurance_claims",
            "diabetes",
            "openml_year",
            "zurich_delays",
        ]

    def bases(self) -> list[str]:
        return G.available_bases()

    def methods(self) -> list[str]:
        ms = (*DIST_METHODS, *INTERVAL_METHODS, *SHIFT_METHODS)
        return [f"{m}+{b}" for b in self.bases() for m in ms]

    def library_name(self, method: str) -> str:
        uq, _, base = method.rpartition("+")
        return "torchregress" if uq.startswith("tr_") else G.library_of(base)

    def library_version(self, method: str) -> str:
        return ""

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        cfg = config or HarnessConfig()
        if splits is None:
            from harness.datasets import load_dataset

            splits = load_dataset(dataset_name)
        X, y = pooled(splits, 600 if cfg.smoke else N_MAX, cfg.seed)
        scenarios = (scenario_name,) if scenario_name else SCENARIOS
        rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            s = scenario_splits(X, y, scenario, cfg.seed)
            w = density_ratio(s["X_cal"], s["X_test"], cfg.seed) if scenario == "shift" else None
            methods = (*DIST_METHODS, *INTERVAL_METHODS, *(SHIFT_METHODS if w is not None else ()))
            for base in self.bases():
                set_harness_seed(cfg.seed)
                common = {
                    "Suite": self.name,
                    "Dataset": f"{dataset_name}@{scenario}",
                    "Scenario": scenario,
                    "Base": base,
                    "NTrain": len(s["y_train"]),
                    "NCal": len(s["y_cal"]),
                    "NTest": len(s["y_test"]),
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                }
                try:
                    (q_cal, q_test), fit_s = G.fit_grids(
                        base, s["X_train"], s["y_train"], [s["X_cal"], s["X_test"]], cfg
                    )
                    point_only = G.is_point(q_cal)
                    common["PointOnly"] = point_only
                    if point_only:
                        base_methods = (
                            *POINT_METHODS,
                            *(POINT_SHIFT_METHODS if w is not None else ()),
                        )
                    else:
                        base_methods = methods
                except Exception as exc:  # noqa: BLE001 — every row of this base fails
                    note = f"failed: base {base}: {type(exc).__name__}: {exc}"[:300]
                    for m in methods:
                        rows.append(
                            {
                                **common,
                                "Method": f"{self.name}/{m}+{base}",
                                "Library": self.library_name(f"{m}+{base}"),
                                "LibraryVersion": "",
                                "Notes": note,
                            }
                        )
                    continue
                for m in base_methods:
                    row = {
                        **common,
                        "Method": f"{self.name}/{m}+{base}",
                        "Library": self.library_name(f"{m}+{base}"),
                        "LibraryVersion": "",
                    }
                    try:
                        t1 = perf_counter()
                        if m == "Raw":
                            row.update(score_distribution(q_test, s))
                        elif m == "tr_ResidualDist":
                            pc, pt = G.quantile_at(q_cal, 0.5), G.quantile_at(q_test, 0.5)
                            row.update(score_distribution(G.residual_grid(pc, s["y_cal"], pt), s))
                        elif m == "tr_PITRecal":
                            row.update(
                                score_distribution(G.pit_recalibrate(q_cal, s["y_cal"], q_test), s)
                            )
                        else:
                            info: dict[str, Any] = {}
                            row.update(score_interval_method(m, q_cal, q_test, s, w, info))
                        note = f"{m} on {base} ({scenario})"
                        n_unseen = info.get("unseen_test_rows", 0) if m == "tr_Mondrian" else 0
                        if n_unseen:
                            note += (
                                f"; {n_unseen} test rows in bins empty in calibration "
                                "use the pooled split-conformal threshold"
                            )
                        row.update(
                            {
                                "train_s": fit_s,
                                "eval_s": perf_counter() - t1,
                                "Notes": note,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 — row-level resilience
                        row["Notes"] = f"failed: {type(exc).__name__}: {exc}"[:300]
                    rows.append(row)
        return rows

    # HarnessRunner abstract surface (unused: custom run drives everything).
    def fit(self, method: str, splits: dict[str, np.ndarray], config: HarnessConfig) -> Any:
        raise NotImplementedError("fm_uq_scenarios uses custom run()")

    def predict(
        self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("fm_uq_scenarios uses custom run()")

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        raise NotImplementedError("fm_uq_scenarios uses custom run()")
