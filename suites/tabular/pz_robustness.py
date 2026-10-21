"""PZ challenge robustness experiments: spectroscopic removal and selection/blend shift.

The removal suite mirrors the paper's paired design: fit on the full training set
and on a reduced training set with one population removed, then evaluate on the
*same* held-out test galaxies.  The shift suite trains on task set 1 and applies
the same model to task sets 2-4 (spectroscopic selection / narrowband / blends).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner
from suites.tabular.pz_challenge import PZChallengeSuite


def _i_band(raw: np.ndarray) -> np.ndarray:
    i = np.asarray(raw, dtype=np.float64)[:, 3]
    return np.where(np.isfinite(i), i, 30.0)


def _gr(raw: np.ndarray) -> np.ndarray:
    g = np.asarray(raw, dtype=np.float64)[:, 1]
    r = np.asarray(raw, dtype=np.float64)[:, 2]
    out = g - r
    return np.where(np.isfinite(out), out, 0.0)


class PZRemovalSuite(PZChallengeSuite):
    """Paired full-vs-reduced-training evaluation on identical test galaxies."""

    def __init__(self) -> None:
        super().__init__()
        self._methods = ["MSE_MLP", "GaussianNLL_MLP", "MDN_K3", "TabM_MSE", "SplitCP_MLP"]

    @property
    def name(self) -> str:
        return "pz_removal"

    @property
    def description(self) -> str:
        return (
            "PZ challenge paired training-removal test: full training vs omission "
            "by magnitude, colour or sky quadrant, same held-out test galaxies."
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["pz_challenge_cardinal_10yr", "pz_challenge_taskset4_cardinal_10yr"]

    def methods(self) -> list[str]:
        return list(self._methods)

    def _scenarios(self, raw: np.ndarray, ra: np.ndarray, dec: np.ndarray) -> dict[str, np.ndarray]:
        n = len(raw)
        keep = {
            "full": np.ones(n, dtype=bool),
            "omit_faint": _i_band(raw) <= np.nanquantile(_i_band(raw), 2.0 / 3.0),
            "omit_red": _gr(raw) <= np.nanquantile(_gr(raw), 0.80),
        }
        # A sky quadrant is the closest analogue of a survey field.
        ra_med = np.nanmedian(ra) if np.isfinite(ra).any() else 0.0
        dec_med = np.nanmedian(dec) if np.isfinite(dec).any() else 0.0
        field = ~((ra < ra_med) & (dec < dec_med))
        keep["omit_field"] = field
        return keep

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, Any] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        from harness.datasets import load_dataset

        cfg = config or HarnessConfig()
        dataset = dataset_name or "pz_challenge_cardinal_10yr"
        if splits is None:
            splits = load_dataset(dataset, standardize_targets=True)
        scaler = splits.get("_y_scaler")
        X_train = np.asarray(splits["X_train"], dtype=np.float32)
        y_train = np.asarray(splits["y_train"], dtype=np.float32).reshape(-1)
        X_test = np.asarray(splits["X_test"], dtype=np.float32)
        y_test = np.asarray(splits["y_test"], dtype=np.float32).reshape(-1)
        ra = np.asarray(splits.get("ra_train", np.zeros(len(X_train))), dtype=np.float64)
        dec = np.asarray(splits.get("dec_train", np.zeros(len(X_train))), dtype=np.float64)
        keep = self._scenarios(X_train, ra, dec)
        rows: list[dict[str, Any]] = []
        for scenario, subset in keep.items():
            X_sub, y_sub = X_train[subset], y_train[subset]
            for method in self.methods():
                set_harness_seed(cfg.seed)
                child = PZChallengeSuite()
                t0 = __import__("time").perf_counter()
                state = child.fit(method, {"X_train": X_sub, "y_train": y_sub}, cfg)
                fit_s = __import__("time").perf_counter() - t0
                pred = child.predict(method, state, X_test, cfg)
                if scaler is not None:
                    pred["_y_scaler"] = scaler
                metrics = child.evaluate(method, pred, y_test, cfg)
                rows.append(
                    {
                        "Method": f"{self.name}/{method}",
                        "Library": child.library_name(method),
                        "LibraryVersion": child.library_version(method),
                        "Dataset": dataset,
                        "Suite": self.name,
                        "Scenario": scenario,
                        "N_Train": int(subset.sum()),
                        "RemovedFraction": float(1.0 - subset.mean()),
                        "train_s": fit_s,
                        "eval_s": 0.0,
                        "Device": "cuda/cpu",
                        "Notes": child.method_notes(method),
                        **metrics,
                    }
                )
        return rows


class PZOODSuite(PZChallengeSuite):
    """Train on representative task set 1; evaluate on task sets 1/2/3/4."""

    def __init__(self) -> None:
        super().__init__()
        self._methods = ["MSE_MLP", "GaussianNLL_MLP", "MDN_K3", "TabM_MSE", "SplitCP_MLP"]

    @property
    def name(self) -> str:
        return "pz_ood"

    @property
    def description(self) -> str:
        return (
            "PZ challenge selection/blend shift: one model trained on task set 1, "
            "evaluated on the same distribution and on task sets 2-4."
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["pz_challenge_cardinal_10yr"]

    def methods(self) -> list[str]:
        return list(self._methods)

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, Any] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        from harness.datasets import load_dataset

        cfg = config or HarnessConfig()
        id_splits = splits or load_dataset("pz_challenge_cardinal_10yr", standardize_targets=True)
        scaler = id_splits.get("_y_scaler")
        ood_names = {
            "OOD_task2": "pz_challenge_taskset2_cardinal_10yr",
            "OOD_task3": "pz_challenge_taskset3_cardinal_10yr",
            "OOD_task4": "pz_challenge_taskset4_cardinal_10yr",
        }
        ood = {k: load_dataset(v, standardize_targets=False) for k, v in ood_names.items()}
        for view in ood.values():
            if scaler is not None:
                view["y_test"] = (
                    scaler.transform(np.asarray(view["y_test"], dtype=np.float64).reshape(-1, 1))
                    .ravel()
                    .astype(np.float32)
                )
        X_train = np.asarray(id_splits["X_train"], dtype=np.float32)
        y_train = np.asarray(id_splits["y_train"], dtype=np.float32).reshape(-1)
        rows: list[dict[str, Any]] = []
        for method in self.methods():
            set_harness_seed(cfg.seed)
            child = PZChallengeSuite()
            state = child.fit(method, {"X_train": X_train, "y_train": y_train}, cfg)
            views = {"ID_task1": id_splits, **ood}
            for scenario, view in views.items():
                pred = child.predict(method, state, np.asarray(view["X_test"]), cfg)
                if scaler is not None:
                    pred["_y_scaler"] = scaler
                metrics = child.evaluate(
                    method, pred, np.asarray(view["y_test"], dtype=np.float32), cfg
                )
                rows.append(
                    {
                        "Method": f"{self.name}/{method}",
                        "Library": child.library_name(method),
                        "LibraryVersion": child.library_version(method),
                        "Dataset": scenario,
                        "Suite": self.name,
                        "Scenario": scenario,
                        "train_s": 0.0,
                        "eval_s": 0.0,
                        "Device": "cuda/cpu",
                        "Notes": child.method_notes(method),
                        **metrics,
                    }
                )
        return rows
