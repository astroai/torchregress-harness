## Key findings

Curated summary for the torchregress 0.3.0 release (run #6, `release_20261013_1917`).
Everything else in this report is generated. Pair counts and verdicts are defined
in [Parity](#parity-torchregress-vs-comparators) and Appendix C. Run narrative:
[`ANALYSIS.md`](results/release_20261013_1917/ANALYSIS.md).

**Parity with external implementations**

- **Deterministic pairs (same algorithm on the same fitted model or data):** 280 pairs.
  203 are exact (≤ 1e-6 relative), 63 close (≤ 2%), 4 differ and 10 cannot be scored
  because both rows are skipped (Jackknife+ refits above 500 models on 9 `uq_layer`
  datasets; torchcp 1.2.1 has no `torchcp.regression.predictors` module for the image
  CNN row).
- **Conformal wrappers match MAPIE and crepes exactly.** All 181 scoreable `uq_layer`
  pairs (`tr_{Split,Normalized,CQR,CVPlus}+Base` vs `MAPIE_*` / `crepes_*` on DeepEnsK5,
  LightGBM, RF and TabularMLP bases, 10 datasets) are exact. The same holds for
  `BLR_Linear` vs sklearn `BayesianRidge` (3/3) and `IsotonicMeanCalibrator` vs sklearn
  `IsotonicRegression` (4/4).
- **Linear-model parity is close, not exact.** The cause is L-BFGS in float32 against
  closed-form or IRLS solvers. `Split+Linear(post-hoc)` vs MAPIE / crepes / torchcp is
  close on all 10 conformal datasets (30 pairs; at most 0.55% in width, on
  `tabred_delivery`). `Tweedie_Linear` / `CompoundPoisson_Linear` vs the statsmodels GLM
  agree to within 0.01% in deviance (R² differs by 1.24% on `insurance_claims`, where
  R² ≈ 0.007). `tr_Linear` vs the scipy MLE matches NLL to within 0.006% on all 6
  likelihood families. `Huber_Linear` vs statsmodels RLM and `Quantile_Linear` vs
  sklearn `QuantileRegressor` are within 1.9%.
- **New deterministic differences: `sklearn_baselines / OLS`.** This path uses
  `torch.linalg.lstsq` and does not match sklearn `LinearRegression` on 4 of 17
  datasets. On `tabred_delivery`, `tabred_maps` and `tabred_weather` its RMSE is
  10–14% higher, and on `insurance_claims` its MAE differs by 27%. The L-BFGS
  `GaussianNLL_Linear` row stays within 0.3% of sklearn on every dataset, so the
  numerical issue is in the lstsq path. The earlier hand-curated report listed OLS as
  identical. This is an open item.
- **Independently trained flows and posteriors (matched pairs) do not agree within
  2%.** 77 pairs: 11 close, 65 differ, 1 unscored. 46 of them agree within seed noise
  (z ≤ 2 on every key metric). On `sbi_parity`, sbi NPE has the lower C2ST than
  `torchregress_NSF` on 4 of 5 sbibm tasks: bernoulli_glm 0.749 vs 0.675,
  gaussian_linear 0.628 vs 0.534, gaussian_mixture 0.624 vs 0.513 and twomoons
  0.588 vs 0.498. torchregress is better on slcp (0.805 vs 0.851). gaussian_mixture
  is fixed relative to the earlier 0.997 ([ANALYSIS §5](results/release_20261013_1917/ANALYSIS.md#5-reruns-after-the-follow-up-fixes-2026-10-15))
  but still trails.
- **MDN density parity is mixed.** In `density_parity`, `torchregress_MDN` has a higher
  NLL than `sbi_MDN` on energy_efficiency (−0.76 vs −1.33), mt_enb (−0.85 vs −1.24) and
  mt_scm20d (0.33 vs −0.01). It has a lower NLL on mt_rf1 and pzflow_galaxies. The
  precision-parameterisation library item stays open
  ([ANALYSIS §2–3](results/release_20261013_1917/ANALYSIS.md#3-open-harness--library-items-from-this-run)).
- **torchregress_NSF has much tighter calibrated intervals than pzflow's default
  flow.** In `pzflow_parity` its interval score is 73–96% lower on all 4 datasets
  (e.g. 0.082 vs 2.03 on pzflow_galaxies), and its NLL is lower on 3 of 4. The
  exception is energy_efficiency (2.61 vs 2.49). `normflows_NSF` is known-bad (a
  broadcast error on 1-D inputs) on all 12 rows.
- **DoubleML parity holds.** `tr_Orthogonal_Poly3` vs `DoubleML_PLR_Poly3Ridge` is
  within 2% and within seed noise on both PLR designs.
- **Method pairs, where the model or recipe differs by design: 79 pairs, 6 close.**
  These are worth reading before making claims:
  - torchcp's R2C collapses on house_sales (coverage 0.00) and openml_year (0.60),
    while `R2C+Softmax` covers 0.947–0.995.
  - The library `IRLS` row (Huber weights, `max_iter=10`) has 21–52% higher RMSE than
    statsmodels RLM. It is not tuned to match; `Huber_Linear` is the parity row.

**Run health**

- Provenance: no real dataset fell back to synthetic data. 54 synthetic-fallback rows
  are all on `stress_*` datasets, which are synthetic by design. 30 rows ran on CPU
  after CUDA failures (marked ‡): likelihoods NSF, TabFM, KumoTabular and the
  multitarget NSF.
- 39 rows failed and 99 were skipped (Appendix A). Of the failures:
  - 15 are `age_regression / CNN+RankNContrast` and 6 are keypoint GaussianNLL.
    Both are device-handling bugs, fixed at harness `1435827`; reruns are pending.
  - 3 are `image_density_parity / sbi_zuko_NSF` (NVML assert).
  - 3 are `label_shift / TT_BBSE` on folktables_income (ill-conditioned confusion
    matrix; raises by design).
  - 12 are the known-bad `normflows_NSF` rows.
- `tweedie` MLP rows still fail on `insurance_claims` (R² −20.8 for Tweedie /
  CompoundPoisson against 0.007 for the GLM). The rerun at harness `9e8c72b` (GLM
  intercept start, clipping, early stopping) is pending
  ([ANALYSIS §3](results/release_20261013_1917/ANALYSIS.md#3-open-harness--library-items-from-this-run)).
- Comparator drift:
  - `MAPIE_CrossCQR+GBR` moved to nominal coverage because the MAPIE pin moved to
    `929740a` ([ANALYSIS §2](results/release_20261013_1917/ANALYSIS.md#2-against-run-5-same-jobs-2-364-common-cells)).
  - In the `fm` group, `structured-data-models` was recorded at two commits
    (`837fc07`, `70ca27e`), and `tabpfn` is installed in only 1 of 6 replicas (see
    Comparator versions).
  - The pip metadata for `torchregress` reads 0.2.0 in every replica. The commit
    above is the authoritative reference.
- Seed spread: 1 847 table cells have a relative std above 5%. 1 298 of them are in
  `fm_uq_scenarios`, mostly the perturbed scenarios. Rank those cells with error bars
  only.
