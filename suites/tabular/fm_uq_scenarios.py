"""Tabular foundation models + the torchregress UQ layer under stress scenarios.

Question: given a strong pretrained tabular model (TabICL, TabICLv2, KumoTabular,
TabFM, optionally TabPFN), does the torchregress uncertainty layer improve its
own predictive intervals on real data, also when the data are contaminated,
shifted or long-tailed?  ``uq_layer`` answers this on clean splits against
MAPIE / crepes; this suite fixes the UQ methods to torchregress and varies the
*scenario*.

Every base model is fitted once per (dataset, scenario, seed) on the same
train split; every UQ method then uses the same calibration and test rows.

Scenarios (built from each real dataset; row ``Dataset`` = ``<name>@<scenario>``)
------------------------------------------------------------------------------
* ``clean`` -- random train / calibration / test split.
* ``outliers`` -- 5% of training *and* calibration targets replaced by gross
  errors (shifted by 5-10 target standard deviations, random sign); the test
  targets stay clean.  Robustness of both the base fit and the calibration.
* ``shift`` -- covariate shift: test rows drawn with probability proportional to
  ``exp(beta * z)``, ``z`` the standardised first principal component of X
  (beta = 1), train / calibration rows uniformly from the rest.  The weighted
  methods estimate ``w(x) = p(test | x) / p(cal | x)`` with a logistic
  regression on X (no oracle weights).

Imbalance is scored in every scenario: ``TailCoverage`` is the coverage on test
rows whose target is above the training 90th percentile, ``WorstDecileCoverage``
the lowest coverage over the ten target deciles (training cut points).  The
heavy-tailed real targets (``insurance_claims``, ``house_sales``) make these the
rare-target view.

UQ methods (row ``<method>+<base>``)
------------------------------------
* ``Raw`` -- the base model's own 90% interval (its alpha/2, 1-alpha/2 quantiles).
* ``tr_Split`` -- absolute-residual split conformal (``SplitConformal``).
* ``tr_Normalized`` -- residual / spread scores, spread = base quantile width.
* ``tr_CQR`` -- conformalized quantile regression on the base quantiles.
* ``tr_Mondrian`` -- split conformal with Mondrian groups = up to 5 bins (>= 40
  calibration rows each) of the predicted value (calibration quantile cut points): class-conditional
  coverage for low / high predicted targets.
* ``tr_WeightedSplit`` / ``tr_WeightedCQR`` -- ``shift`` only: weighted
  conformal (Tibshirani et al., 2019) with the estimated density ratio.

Targets are standardised by ``load_dataset``; interval metrics are reported in
those units (comparable across bases).
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import numpy as np
import torch

from harness.core import set_harness_seed
from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner
from suites.tabular.uq_layer import (
    _GBM_BASE,
    _BasePreds,
    _fit_gbm,
    _fit_tabicl,
    _fit_tabular_mlp,
    _from_quantiles,
    _np,
    _t,
)

ALPHA = 0.1
N_MAX = 6000  # rows pooled per dataset (in-context models scale with the context)
OUTLIER_FRAC = 0.05
SHIFT_BETA = 1.0
N_MONDRIAN_BINS = 5
MIN_PER_BIN = 40
SCENARIOS = ("clean", "outliers", "shift")
_SDM_MODELS = {
    "TabICLv2": ("TabICLv2", {}),
    "KumoTabular": ("KumoTabular", {}),
    "TabFM": ("TabFM", {}),
}
_BASE_METHODS = ("Raw", "tr_Split", "tr_Normalized", "tr_CQR", "tr_Mondrian")
_SHIFT_METHODS = ("tr_WeightedSplit", "tr_WeightedCQR")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _scenario_splits(
    X: np.ndarray, y: np.ndarray, scenario: str, seed: int
) -> dict[str, np.ndarray]:
    """Train / calibration / test rows (60 / 20 / 20) for one scenario."""
    rng = np.random.default_rng(seed)
    n = len(X)
    n_test = n // 5
    if scenario == "shift":
        Xc = X - X.mean(0)
        _, _, vt = np.linalg.svd(Xc[: min(n, 5000)], full_matrices=False)
        z = Xc @ vt[0]
        z = (z - z.mean()) / (z.std() + 1e-12)
        p = np.exp(SHIFT_BETA * z)
        test = rng.choice(n, n_test, replace=False, p=p / p.sum())
        rest = np.setdiff1d(np.arange(n), test)
        rest = rng.permutation(rest)
    else:
        perm = rng.permutation(n)
        test, rest = perm[:n_test], perm[n_test:]
    n_cal = len(rest) // 4
    cal, train = rest[:n_cal], rest[n_cal:]
    y_train, y_cal = y[train].copy(), y[cal].copy()
    if scenario == "outliers":
        sd = float(np.std(y[train]))
        for arr in (y_train, y_cal):
            k = max(1, int(OUTLIER_FRAC * len(arr)))
            idx = rng.choice(len(arr), k, replace=False)
            arr[idx] += rng.choice([-1.0, 1.0], k) * rng.uniform(5.0, 10.0, k) * sd
    return {
        "X_train": X[train],
        "y_train": y_train,
        "X_cal": X[cal],
        "y_cal": y_cal,
        "X_test": X[test],
        "y_test": y[test],
        "y_train_clean": y[train],
    }


def _density_ratio(
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
# Foundation-model bases (sdm: TabICLv2 / KumoTabular / TabFM; TabPFN)
# ---------------------------------------------------------------------------


def _fit_sdm(name: str, s: dict[str, Any], *, alpha: float, seed: int) -> _BasePreds:
    import sdm.models
    from sdm import TableTensor

    cls_name, kwargs = _SDM_MODELS[name]
    device = current_device()
    model = getattr(sdm.models, cls_name)(task="regression", device=device, **kwargs)
    model.eval()
    x = TableTensor.from_tensor(torch.as_tensor(s["X_train"], dtype=torch.float32))
    y = TableTensor.from_tensor(torch.as_tensor(s["y_train"], dtype=torch.float32).reshape(-1, 1))
    model.fit(
        x.to(device), y.to(device), generator=torch.Generator(device=device).manual_seed(seed)
    )

    def _quant(Xe: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model.predict(
                TableTensor.from_tensor(torch.as_tensor(Xe, dtype=torch.float32)).to(device)
            )
        q = out.numerical.detach().float().cpu().numpy().reshape(len(Xe), -1)
        q = np.sort(q, axis=1)
        lv = (np.arange(q.shape[1]) + 1) / (q.shape[1] + 1)
        want = [alpha / 2, 0.5, 1 - alpha / 2]
        return np.stack([np.array([np.interp(t, lv, row) for row in q]) for t in want], axis=1)

    qc, qt = _quant(s["X_cal"]), _quant(s["X_test"])
    return _BasePreds(**_from_quantiles(qc[:, 1], qt[:, 1], qc, qt, alpha=alpha), library="sdm")


def _fit_tabpfn(s: dict[str, Any], *, alpha: float) -> _BasePreds:
    from tabpfn import TabPFNRegressor

    est = TabPFNRegressor(device=str(current_device()))
    est.fit(s["X_train"], s["y_train"].reshape(-1))
    levels = [alpha / 2, 0.5, 1 - alpha / 2]

    def _q(Xe: np.ndarray) -> np.ndarray:
        q = est.predict(Xe, output_type="quantiles", quantiles=levels)
        return np.stack([np.asarray(c) for c in q], axis=1)

    qc, qt = _q(s["X_cal"]), _q(s["X_test"])
    return _BasePreds(
        **_from_quantiles(est.predict(s["X_cal"]), est.predict(s["X_test"]), qc, qt, alpha=alpha),
        library="tabpfn",
    )


# ---------------------------------------------------------------------------
# UQ methods and scores
# ---------------------------------------------------------------------------


def _uq(
    method: str, b: _BasePreds, s: dict[str, Any], w: tuple[np.ndarray, np.ndarray] | None
) -> tuple[np.ndarray, np.ndarray]:
    from torchregress.losses.conformal import CQR, SplitConformal

    y_cal = np.asarray(s["y_cal"], dtype=np.float64)
    if method == "Raw":
        return b.q_test[:, 0], b.q_test[:, 2]
    if method == "tr_Split":
        cp = SplitConformal(alpha=ALPHA)
        cp.calibrate(_t(b.point_cal), _t(y_cal))
        lo, hi = cp.predict_interval(_t(b.point_test))
    elif method == "tr_Normalized":
        cp = SplitConformal(alpha=ALPHA, normalize_fn=lambda _p, spread: spread)
        cp.calibrate(_t(b.point_cal), _t(y_cal), x=_t(b.spread_cal))
        lo, hi = cp.predict_interval(_t(b.point_test), x=_t(b.spread_test))
    elif method == "tr_CQR":
        cq = CQR(alpha=ALPHA)
        cq.calibrate(_t(b.q_cal[:, [0, 2]]), _t(y_cal))
        lo, hi = cq.predict_interval(_t(b.q_test[:, [0, 2]]))
    elif method == "tr_Mondrian":
        # >= MIN_PER_BIN rows per bin: below ceil((n+1)(1-alpha)) > n the bin's interval is infinite.
        n_bins = max(1, min(N_MONDRIAN_BINS, len(y_cal) // MIN_PER_BIN))
        cuts = np.quantile(b.point_cal, np.linspace(0, 1, n_bins + 1)[1:-1])
        g_cal = torch.as_tensor(np.searchsorted(cuts, b.point_cal))
        g_test = torch.as_tensor(np.searchsorted(cuts, b.point_test))
        cp = SplitConformal(alpha=ALPHA)
        cp.calibrate(_t(b.point_cal), _t(y_cal), groups=g_cal)
        lo, hi = cp.predict_interval(_t(b.point_test), groups=g_test)
    elif method in _SHIFT_METHODS:
        assert w is not None
        w_cal, w_test = _t(w[0]), _t(w[1])
        if method == "tr_WeightedSplit":
            cp = SplitConformal(alpha=ALPHA)
            cp.calibrate(_t(b.point_cal), _t(y_cal), weights=w_cal)
            lo, hi = cp.predict_interval(_t(b.point_test), test_weights=w_test)
        else:
            cq = CQR(alpha=ALPHA)
            cq.calibrate(_t(b.q_cal[:, [0, 2]]), _t(y_cal), weights=w_cal)
            lo, hi = cq.predict_interval(_t(b.q_test[:, [0, 2]]), test_weights=w_test)
    else:
        raise ValueError(method)
    return _np(lo).reshape(-1), _np(hi).reshape(-1)


def _scores(lo: np.ndarray, hi: np.ndarray, point: np.ndarray, s: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(s["y_test"], dtype=np.float64)
    inside = (y >= lo) & (y <= hi)
    finite = np.isfinite(lo) & np.isfinite(hi)
    width = np.where(finite, hi - lo, np.nan)
    isc = width + (2 / ALPHA) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))
    y_tr = np.asarray(s["y_train_clean"], dtype=np.float64)
    tail = y >= np.quantile(y_tr, 0.9)
    cuts = np.quantile(y_tr, np.linspace(0, 1, 11)[1:-1])
    dec = np.searchsorted(cuts, y)
    dec_cov = [inside[dec == k].mean() for k in range(10) if (dec == k).sum() >= 5]
    return {
        "Coverage": float(inside.mean()),
        "Width": float(np.nanmean(width)) if finite.any() else float("inf"),
        "IntervalScore": float(np.nanmean(isc)) if finite.any() else float("inf"),
        "InfiniteIntervalFrac": float(1 - finite.mean()),
        "TailCoverage": float(inside[tail].mean()) if tail.any() else None,
        "WorstDecileCoverage": float(min(dec_cov)) if dec_cov else None,
        "RMSE": float(np.sqrt(np.mean((point - y) ** 2))),
    }


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class FMUQScenariosSuite(HarnessRunner):
    """Foundation-model bases x torchregress UQ layer x stress scenarios (custom run)."""

    @property
    def name(self) -> str:
        return "fm_uq_scenarios"

    @property
    def description(self) -> str:
        return (
            "Tabular foundation models (TabICL/TabICLv2/KumoTabular/TabFM/TabPFN) with and "
            "without the torchregress UQ layer, under clean / outlier / covariate-shift "
            "scenarios with rare-target coverage"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["california_housing", "house_sales", "insurance_claims", "diabetes", "openml_year"]

    def bases(self) -> list[str]:
        out = [_GBM_BASE, "TabularMLP", "TabICL", *_SDM_MODELS]
        if os.environ.get("TORCHREGRESS_HARNESS_WITH_TABPFN") == "1":
            out.append("TabPFN")
        return out

    def methods(self) -> list[str]:
        return [f"{m}+{b}" for b in self.bases() for m in (*_BASE_METHODS, *_SHIFT_METHODS)]

    def library_name(self, method: str) -> str:
        uq, _, base = method.rpartition("+")
        if uq.startswith("tr_"):
            return "torchregress"
        return {"TabICL": "tabicl", "TabPFN": "tabpfn", "TabularMLP": "torchregress"}.get(
            base, "sdm" if base in _SDM_MODELS else "lightgbm"
        )

    def library_version(self, method: str) -> str:
        return ""

    def _fit_base(self, base: str, s: dict[str, Any], cfg: HarnessConfig) -> _BasePreds:
        if base == _GBM_BASE:
            return _fit_gbm(s, alpha=ALPHA, seed=cfg.seed, smoke=cfg.smoke)
        if base == "TabularMLP":
            return _fit_tabular_mlp(s, alpha=ALPHA, config=cfg)
        if base == "TabICL":
            return _fit_tabicl(s, alpha=ALPHA, seed=cfg.seed, smoke=cfg.smoke)
        if base == "TabPFN":
            return _fit_tabpfn(s, alpha=ALPHA)
        return _fit_sdm(base, s, alpha=ALPHA, seed=cfg.seed)

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
        X = np.vstack([np.asarray(splits["X_train"]), np.asarray(splits["X_test"])]).astype(
            np.float32
        )
        y = np.concatenate(
            [np.asarray(splits["y_train"]).reshape(-1), np.asarray(splits["y_test"]).reshape(-1)]
        ).astype(np.float64)
        n_max = 600 if cfg.smoke else N_MAX
        if len(X) > n_max:
            idx = np.random.default_rng(cfg.seed).choice(len(X), n_max, replace=False)
            X, y = X[idx], y[idx]
        rows: list[dict[str, Any]] = []
        for scenario in SCENARIOS:
            s = _scenario_splits(X, y, scenario, cfg.seed)
            w = _density_ratio(s["X_cal"], s["X_test"], cfg.seed) if scenario == "shift" else None
            uq_methods = (*_BASE_METHODS, *_SHIFT_METHODS) if scenario == "shift" else _BASE_METHODS
            for base in self.bases():
                set_harness_seed(cfg.seed)
                common = {
                    "Suite": self.name,
                    "Dataset": f"{dataset_name}@{scenario}",
                    "Scenario": scenario,
                    "Base": base,
                    "SyntheticFallback": bool(splits.get("_synthetic_fallback", False)),
                    "Device": str(current_device()),
                }
                try:
                    t0 = perf_counter()
                    b = self._fit_base(base, s, cfg)
                    fit_s = perf_counter() - t0
                except Exception as exc:  # noqa: BLE001 — every row of this base fails with the reason
                    for m in uq_methods:
                        rows.append(
                            {
                                **common,
                                "Method": f"{self.name}/{m}+{base}",
                                "Library": self.library_name(f"{m}+{base}"),
                                "LibraryVersion": "",
                                "Notes": f"failed: base {base}: {type(exc).__name__}: {exc}"[:300],
                            }
                        )
                    continue
                for m in uq_methods:
                    method = f"{m}+{base}"
                    row = {
                        **common,
                        "Method": f"{self.name}/{method}",
                        "Library": self.library_name(method),
                        "LibraryVersion": "",
                    }
                    try:
                        t1 = perf_counter()
                        lo, hi = _uq(m, b, s, w)
                        row.update(_scores(lo, hi, b.point_test, s))
                        row.update(
                            {
                                "train_s": fit_s,
                                "eval_s": perf_counter() - t1,
                                "Notes": f"{m} on {base} ({scenario})",
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
