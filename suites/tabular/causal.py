"""Causal suite — doubly-robust ATE/CATE estimation.

Benchmarks with KNOWN truth: IHDP (real covariates + counterfactuals) and
Folktables semi-synthetic (real Census X, simulated T/Y). Rows: torchregress
dr_ate/dr_cate (cross-fitted DR), econml DRLearner (external SOTA), naive
difference-in-means + T-learner baselines. Metrics: ATE error, PEHE
(CATE RMSE), CI coverage of the true ATE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from harness.core import set_harness_seed
from harness.runners import HarnessConfig, HarnessRunner


class CausalSuite(HarnessRunner):
    """DR/Qini-style causal estimation vs econml on known-truth benchmarks."""

    @property
    def name(self) -> str:
        return "causal"

    @property
    def description(self) -> str:
        return "Doubly-robust ATE/CATE vs econml on IHDP + Folktables truth"

    @property
    def supported_datasets(self) -> list[str]:
        return ["ihdp", "folktables_causal"]

    def methods(self) -> list[str]:
        return [
            "DR",
            "EconML_DRLearner",
            "NaiveDiM",
            "TLearner",
            "Policy_DR",
            "Policy_EconML",
            "Policy_AllTreat",
        ]

    def library_name(self, method: str) -> str:
        if method.startswith("EconML"):
            return "econml"
        if method in ("NaiveDiM", "TLearner"):
            return "sklearn"
        return "torchregress"

    def method_notes(self, method: str) -> str:
        return {
            "DR": "torchregress dr_ate/dr_cate (cross-fitted DR, RF nuisances)",
            "EconML_DRLearner": "econml DRLearner (external SOTA)",
            "NaiveDiM": "difference-in-means (confounded baseline)",
            "TLearner": "sklearn RF T-learner (no debiasing)",
            "Policy_DR": "AIPW value of treat-if-CATE>0 (DR CATE) vs truth",
            "Policy_EconML": "AIPW value of treat-if-CATE>0 (econml CATE) vs truth",
            "Policy_AllTreat": "AIPW value of treat-everyone vs truth (baseline)",
        }.get(method, "")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _cate_dr(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        t_tr: np.ndarray,
        X_te: np.ndarray,
        config: HarnessConfig,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        """DR CATE: returns (ate_dict, cate_test, cate_train_hat)."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from torchregress.causal.dr import dr_ate, dr_cate

        x_t = torch.from_numpy(X_tr).float()
        t_t = torch.from_numpy(t_tr).float()
        y_t = torch.from_numpy(y_tr).float()

        def out_factory() -> RandomForestRegressor:
            return RandomForestRegressor(n_estimators=100, random_state=config.seed, n_jobs=-1)

        def prop_factory() -> LogisticRegression:
            return LogisticRegression(max_iter=1000)

        ate_res = dr_ate(
            x_t,
            t_t,
            y_t,
            outcome_model=out_factory,
            propensity_model=prop_factory,
            folds=2,
            seed=config.seed,
        )
        cate_res = dr_cate(
            x_t,
            t_t,
            y_t,
            cate_model=Ridge(),
            outcome_model=out_factory,
            propensity_model=prop_factory,
            folds=2,
            seed=config.seed,
        )
        # dr_cate returns IN-SAMPLE cate_hat (train rows). Second-stage
        # regression of the DR pseudo-outcomes on X gives test CATE —
        # the standard DR-learner recipe for new points.
        stage2 = Ridge().fit(X_tr, np.asarray(cate_res["pseudo_outcome"], dtype=np.float64))
        cate_te = stage2.predict(X_te).astype(np.float64)
        cate_tr = np.asarray(cate_res["cate_hat"], dtype=np.float64).reshape(-1)
        return dict(ate_res), cate_te, cate_tr

    def _cate_econml(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        t_tr: np.ndarray,
        X_te: np.ndarray,
        config: HarnessConfig,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        """EconML DRLearner CATE: returns (ate_dict, cate_test, cate_train)."""
        from econml.dr import DRLearner
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, Ridge

        est = DRLearner(
            model_regression=RandomForestRegressor(
                n_estimators=100, random_state=config.seed, n_jobs=-1
            ),
            model_propensity=LogisticRegression(max_iter=1000),
            model_final=Ridge(),
            cv=2,
        )
        est.fit(y_tr, t_tr, X=X_tr)
        cate_te = np.asarray(est.effect(X_te), dtype=np.float64).reshape(-1)
        cate_tr = np.asarray(est.effect(X_tr), dtype=np.float64).reshape(-1)
        try:
            ate_lb, ate_ub = est.ate_inference(X_te).conf_int(alpha=0.1)
            ate = {
                "estimate": float(np.mean(est.ate(X_te))),
                "ci_low": float(np.ravel(ate_lb)[0]),
                "ci_high": float(np.ravel(ate_ub)[0]),
            }
        except Exception:
            m = float(cate_te.mean())
            ate = {"estimate": m, "ci_low": m, "ci_high": m}
        return ate, cate_te, cate_tr

    def fit(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
    ) -> dict[str, Any]:
        from sklearn.ensemble import RandomForestRegressor

        set_harness_seed(config.seed)
        X_tr = np.asarray(splits["X_train"], dtype=np.float64)
        y_tr = np.asarray(splits["y_train"], dtype=np.float64).reshape(-1)
        t_tr = np.asarray(splits["treatment_train"], dtype=np.float64).reshape(-1)
        X_te = np.asarray(splits["X_test"], dtype=np.float64)
        ate_true = float(splits["ate_true"])
        cate_te = np.asarray(splits["cate_true_test"], dtype=np.float64).reshape(-1)

        state: dict[str, Any] = {
            "type": "causal",
            "ate_true": ate_true,
            "cate_true": cate_te,
            "X_test": X_te,
        }
        if method == "DR":
            ate_res, cate_te, _ = self._cate_dr(X_tr, y_tr, t_tr, X_te, config)
            state.update(
                {
                    "ate_est": float(ate_res["estimate"]),
                    "ate_lo": float(ate_res["ci_low"]),
                    "ate_hi": float(ate_res["ci_high"]),
                    "cate_est": cate_te,
                }
            )
        elif method == "EconML_DRLearner":
            ate_res, cate_est, _ = self._cate_econml(X_tr, y_tr, t_tr, X_te, config)
            state.update(
                {
                    "ate_est": float(ate_res["estimate"]),
                    "ate_lo": float(ate_res["ci_low"]),
                    "ate_hi": float(ate_res["ci_high"]),
                    "cate_est": cate_est,
                }
            )
        elif method in ("Policy_DR", "Policy_EconML", "Policy_AllTreat"):
            if method == "Policy_DR":
                _, cate_te, cate_tr = self._cate_dr(X_tr, y_tr, t_tr, X_te, config)
            elif method == "Policy_EconML":
                _, cate_te, cate_tr = self._cate_econml(X_tr, y_tr, t_tr, X_te, config)
            else:
                cate_te = np.full_like(cate_te, np.inf)
                cate_tr = np.full_like(y_tr, np.inf)
            state.update({"cate_est": cate_te, "cate_tr": cate_tr})
            self._fit_policy(method, splits, config, state)
        elif method == "NaiveDiM":
            m1, m0 = y_tr[t_tr > 0.5].mean(), y_tr[t_tr <= 0.5].mean()
            ate_est = float(m1 - m0)
            state.update(
                {
                    "ate_est": ate_est,
                    "ate_lo": ate_est,
                    "ate_hi": ate_est,
                    "cate_est": np.full_like(cate_te, ate_est),
                }
            )
        elif method == "TLearner":
            m1 = RandomForestRegressor(n_estimators=100, random_state=config.seed, n_jobs=-1).fit(
                X_tr[t_tr > 0.5], y_tr[t_tr > 0.5]
            )
            m0 = RandomForestRegressor(n_estimators=101, random_state=config.seed, n_jobs=-1).fit(
                X_tr[t_tr <= 0.5], y_tr[t_tr <= 0.5]
            )
            cate_est = (m1.predict(X_te) - m0.predict(X_te)).astype(np.float64)
            ate_est = float(cate_est.mean())
            state.update(
                {
                    "ate_est": ate_est,
                    "ate_lo": ate_est,
                    "ate_hi": ate_est,
                    "cate_est": cate_est,
                }
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        if method.startswith("Policy_"):
            self._fit_policy(method, splits, config, state)
        return state

    def _fit_policy(
        self,
        method: str,
        splits: dict[str, np.ndarray],
        config: HarnessConfig,
        state: dict[str, Any],
    ) -> None:
        """AIPW policy value + true value + overlap diagnostics.

        Policies: treat-if-CATE>0 from the DR / econml CATE fit, or
        treat-everyone. AIPW is estimated on train rows; the truth comes
        from test potential outcomes. Overlap report from a plain
        propensity fit (dataset diagnostic, method-independent).
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        from torchregress.causal.diagnostics import causal_overlap_report
        from torchregress.causal.dr import dr_policy_value

        X_tr = np.asarray(splits["X_train"], dtype=np.float64)
        y_tr = np.asarray(splits["y_train"], dtype=np.float64).reshape(-1)
        t_tr = np.asarray(splits["treatment_train"], dtype=np.float64).reshape(-1)
        pot0 = np.asarray(splits["pot0_test"], dtype=np.float64).reshape(-1)
        pot1 = np.asarray(splits["pot1_test"], dtype=np.float64).reshape(-1)
        cate_te = np.asarray(state["cate_est"], dtype=np.float64).reshape(-1)

        if method == "Policy_AllTreat":
            pol_tr = np.ones_like(y_tr)
        else:
            cate_tr = np.asarray(state["cate_tr"], dtype=np.float64).reshape(-1)
            pol_tr = (cate_tr[: y_tr.shape[0]] > 0).astype(np.float64)

        aipw = dr_policy_value(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(t_tr).float(),
            torch.from_numpy(y_tr).float(),
            policy=torch.from_numpy(pol_tr).float(),
            outcome_model=lambda: RandomForestRegressor(
                n_estimators=100, random_state=config.seed, n_jobs=-1
            ),
            propensity_model=lambda: LogisticRegression(max_iter=1000),
            folds=2,
            seed=config.seed,
        )
        pol_te = (
            (cate_te > 0).astype(np.float64) if method != "Policy_AllTreat" else np.ones_like(pot0)
        )
        true_value = float(np.mean(np.where(pol_te > 0.5, pot1, pot0)))

        prop = LogisticRegression(max_iter=1000).fit(X_tr, (t_tr > 0.5).astype(int))
        overlap = causal_overlap_report(
            torch.from_numpy(prop.predict_proba(X_tr)[:, 1].astype(np.float32)),
            torch.from_numpy(t_tr.astype(np.float32)),
        )
        state.update(
            {
                "pol_aipw": float(aipw["estimate"]),
                "pol_se": float(aipw["se"]),
                "pol_true": true_value,
                "overlap_rate": float(overlap["overlap_rate"]),
                "overlap_trimmed": float(overlap["n_trimmed"]),
            }
        )

    # ------------------------------------------------------------------
    # predict / evaluate on causal estimands
    # ------------------------------------------------------------------

    def predict(
        self,
        method: str,
        model_state: dict[str, Any],
        X_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, np.ndarray]:
        del config
        # Point channel carries the CATE estimate; truth rides along for
        # evaluate() (predictions-dict passthrough precedent). Truncate
        # truth to the presented rows: smoke subsets X_test/y_test by prefix.
        n_rows = len(np.asarray(X_test))
        out: dict[str, np.ndarray] = {
            "y_pred": np.asarray(model_state["cate_est"], dtype=np.float64).reshape(-1)[:n_rows],
            "cate_true": np.asarray(model_state["cate_true"], dtype=np.float64).reshape(-1)[
                :n_rows
            ],
            "ate_true": np.asarray([model_state["ate_true"]], dtype=np.float64),
        }
        if method.startswith("Policy_"):
            out["pol_aipw"] = np.asarray([model_state["pol_aipw"]], dtype=np.float64)
            out["pol_se"] = np.asarray([model_state["pol_se"]], dtype=np.float64)
            out["pol_true"] = np.asarray([model_state["pol_true"]], dtype=np.float64)
            out["overlap_rate"] = np.asarray([model_state["overlap_rate"]], dtype=np.float64)
            out["overlap_trimmed"] = np.asarray([model_state["overlap_trimmed"]], dtype=np.float64)
        else:
            out["ate_lo"] = np.asarray([model_state["ate_lo"]], dtype=np.float64)
            out["ate_hi"] = np.asarray([model_state["ate_hi"]], dtype=np.float64)
        return out

    def evaluate(
        self,
        method: str,
        predictions: dict[str, np.ndarray],
        y_test: np.ndarray,
        config: HarnessConfig,
    ) -> dict[str, float]:
        del y_test, config
        if method.startswith("Policy_"):
            pol_aipw = float(np.asarray(predictions["pol_aipw"]).reshape(-1)[0])
            pol_se = float(np.asarray(predictions["pol_se"]).reshape(-1)[0])
            pol_true = float(np.asarray(predictions["pol_true"]).reshape(-1)[0])
            return {
                "PolicyValue": pol_aipw,
                "PolicySE": pol_se,
                "PolicyTrue": pol_true,
                "PolicyErr": float(abs(pol_aipw - pol_true)),
                "OverlapRate": float(np.asarray(predictions["overlap_rate"]).reshape(-1)[0]),
                "OverlapTrimmed": float(np.asarray(predictions["overlap_trimmed"]).reshape(-1)[0]),
                "RMSE": float(abs(pol_aipw - pol_true)),
            }
        cate_est = np.asarray(predictions["y_pred"]).reshape(-1)
        cate_true = np.asarray(predictions["cate_true"]).reshape(-1)
        ate_true = float(np.asarray(predictions["ate_true"]).reshape(-1)[0])
        ate_lo = float(np.asarray(predictions["ate_lo"]).reshape(-1)[0])
        ate_hi = float(np.asarray(predictions["ate_hi"]).reshape(-1)[0])
        ate_est = float(cate_est.mean())
        return {
            "ATEErr": float(abs(ate_est - ate_true)),
            "PEHE": float(np.sqrt(np.mean((cate_est - cate_true) ** 2))),
            "Covers": float(ate_lo <= ate_true <= ate_hi),
            "RMSE": float(np.sqrt(np.mean((cate_est - cate_true) ** 2))),
        }
