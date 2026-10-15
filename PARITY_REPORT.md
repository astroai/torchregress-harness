# torchregress Parity Report

Generated 2026-10-15 by `tools/build_parity_report.py` from the release run artifacts. Do not edit by hand: regenerate.

| Field | Value |
| --- | --- |
| Run tag | `release_20261013_1917` |
| torchregress commit | `6ae7272` |
| torchregress-harness commit | `9e8c72b` |
| Job groups | `fm`, `image`, `sbi`, `tabular`, `tabular_big` |
| Job artifacts | 579 |
| Result rows | 15616 (1512 per-bin detail rows not tabulated) |

**Notes:**

- Pending reruns: image `age_regression`, `keypoint_regression` and `image_density_parity` (device-handling fixes at harness `1435827`) and `tweedie / insurance_claims` (MLP log-mean fix at `9e8c72b`). This report will be regenerated when they land.

## Contents

- [Comparator versions](#comparator-versions)
- [Environment and configuration](#environment-and-configuration)
- [Provenance](#provenance)
- [Summary](#summary)
- [Parity: torchregress vs comparators](#parity-torchregress-vs-comparators)
- [Results by suite](#results-by-suite)
- [Seed spread](#seed-spread)
- [Appendix A: failure and skip ledger](#appendix-a-failure-and-skip-ledger)
- [Appendix B: method descriptions](#appendix-b-method-descriptions)
- [Appendix C: methodology](#appendix-c-methodology)

## Comparator versions

Union of the `versions_replica_*.txt` files (replicas per group: fm: 6, image: 10, sbi: 5, tabular: 24, tabular_big: 3). Each version is followed by the groups that recorded it. **differs**: more than one installed version across replicas (1 packages); partial: installed in some replicas only.

| Package | Version (groups) | Flag |
| --- | --- | --- |
| `botorch` | 0.18.1 (fm, image, sbi, tabular, tabular_big) |  |
| `catboost` | 1.2.10 (fm, image, sbi, tabular, tabular_big) |  |
| `crepes` | 0.9.1 (fm, image, sbi, tabular, tabular_big) |  |
| `doubleml` | 0.11.4 (fm, image, sbi, tabular, tabular_big) |  |
| `econml` | 0.17.0 (fm, image, sbi, tabular, tabular_big) |  |
| `laplace-torch` | 0.3 (fm, image, sbi, tabular, tabular_big) |  |
| `lightgbm` | 4.7.0 (fm, image, sbi, tabular, tabular_big) |  |
| `lightning-uq-box` | 0.5.0.dev0 @ `d1b7f6a49` (fm, image, sbi, tabular, tabular_big) |  |
| `mapie` | 1.5.1.dev13+g929740af0 @ `929740af0` (fm, image, sbi, tabular, tabular_big) |  |
| `ngboost` | 0.5.11 (fm, image, sbi, tabular, tabular_big) |  |
| `numpy` | 2.5.4 (fm, image, sbi, tabular, tabular_big) |  |
| `pzflow` | 4.0.0 @ `bf9c70d6d` (sbi, tabular, tabular_big); not installed (fm, image) | partial |
| `sbi` | 0.27.0 @ `29af13750` (image, sbi, tabular, tabular_big); not installed (fm) | partial |
| `sbibm` | 1.1.0 (sbi); not installed (fm, image, tabular, tabular_big) | partial |
| `scikit-learn` | 1.9.1 (fm, image, sbi, tabular, tabular_big) |  |
| `scipy` | 1.18.1 (fm, image, sbi, tabular, tabular_big) |  |
| `scoringrules` | 0.11.0 (fm, image, sbi, tabular, tabular_big) |  |
| `structured-data-models` | 0.2.0rc2.dev34+g837fc07b8 @ `837fc07b8` (fm); 0.2.0rc2.dev49+g70ca27ed4 @ `70ca27ed4` (fm); not installed (image, sbi, tabular, tabular_big) | **differs** |
| `tabicl` | 2.2.0 @ `c91f00df1` (fm); not installed (image, sbi, tabular, tabular_big) | partial |
| `tabpfn` | 9.2.0 @ `ffd53715d` (fm); not installed (fm, image, sbi, tabular, tabular_big) | partial |
| `torch-uncertainty` | 0.13.0 @ `3f82fe5d1` (fm, image, sbi, tabular, tabular_big) |  |
| `torchcp` | 1.2.1 (fm, image, sbi, tabular, tabular_big) |  |
| `torchregress` | 0.2.0 (fm, image, sbi, tabular, tabular_big) |  |
| `torchregress-harness` | 0.1.0 (fm, image, sbi, tabular, tabular_big) |  |
| `xgboost` | 3.4.1 (fm, image, sbi, tabular, tabular_big) |  |

## Environment and configuration

| Group | Jobs | Python | torch | CUDA | Device | Seeds | Epochs | Batch | LR | Warmup | Smoke | Replica env line(s) |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fm` | 42 | 3.13.16 | 2.13.0 | True | cuda | 0, 1, 2 | 60 | 64 | 0.001 | 20 | False | python 3.13.16 \| torch 2.13.0 \| cuda True |
| `image` | 93 | 3.13.16 | 2.13.0 | True | cuda | 0, 1, 2 | 60 | 64 | 0.001 | 20 | False | python 3.13.16 \| torch 2.13.0 \| cuda True |
| `sbi` | 15 | 3.13.16 | 2.13.0 | False | cpu | 0, 1, 2 | 60 | 64 | 0.001 | 20 | False | python 3.13.16 \| torch 2.13.0 \| cuda False |
| `tabular` | 420 | 3.13.16 | 2.13.0 | False | cpu | 0, 1, 2 | 60 | 64 | 0.001 | 20 | False | python 3.13.16 \| torch 2.13.0 \| cuda False |
| `tabular_big` | 9 | 3.13.16 | 2.13.0 | True | cuda | 0, 1, 2 | 60 | 64 | 0.001 | 20 | False | python 3.13.16 \| torch 2.13.0 \| cuda True |

Job timestamps span 2026-10-13T19:21:11.114299+00:00 to 2026-10-15T11:49:09.223295+00:00.

## Provenance

- Tabulated rows: 14104 (ok 13966, failed 39, skipped 99); per-bin detail rows: 1512.
- Rows that ran on CPU after a CUDA failure (`‡` in the tables): 30.
- Rows with `SyntheticFallback` true: 54, all on synthetic-by-design datasets unless listed below.

Synthetic-fallback check: **OK** (no real dataset fell back to synthetic data).

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

## Summary

| Group | Suite | Jobs | Datasets | Methods | Rows ok | Failed | Skipped |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fm` | [fm_multitarget](#fm_multitarget-fm) | 12 | 4 | 19 | 210 | 0 | 0 |
| `fm` | [fm_uq_scenarios](#fm_uq_scenarios-fm) | 18 | 6 | 289 | 5192 | 0 | 0 |
| `fm` | [foundation_model](#foundation_model-fm) | 12 | 4 | 6 | 65 | 0 | 0 |
| `image` | [age_regression](#age_regression-image) | 15 | 5 | 28 | 405 | 15 | 0 |
| `image` | [cell_counting](#cell_counting-image) | 6 | 2 | 11 | 66 | 0 | 0 |
| `image` | [censored_imaging](#censored_imaging-image) | 12 | 4 | 5 | 60 | 0 | 0 |
| `image` | [crowd_counting](#crowd_counting-image) | 9 | 3 | 5 | 45 | 0 | 0 |
| `image` | [depth_estimation](#depth_estimation-image) | 15 | 5 | 12 | 180 | 0 | 0 |
| `image` | [image_calibration](#image_calibration-image) | 3 | 1 | 3 | 9 | 0 | 0 |
| `image` | [image_conformal](#image_conformal-image) | 3 | 1 | 10 | 27 | 0 | 3 |
| `image` | [image_density_parity](#image_density_parity-image) | 9 | 3 | 8 | 69 | 3 | 0 |
| `image` | [image_ensembles_uq](#image_ensembles_uq-image) | 3 | 1 | 5 | 15 | 0 | 0 |
| `image` | [image_robust](#image_robust-image) | 15 | 5 | 12 | 180 | 0 | 0 |
| `image` | [keypoint_regression](#keypoint_regression-image) | 3 | 1 | 9 | 21 | 6 | 0 |
| `sbi` | [sbi_parity](#sbi_parity-sbi) | 15 | 5 | 9 | 135 | 0 | 0 |
| `tabular` | [bayesian_linear](#bayesian_linear-tabular) | 9 | 3 | 6 | 54 | 0 | 0 |
| `tabular` | [calibration](#calibration-tabular) | 12 | 4 | 6 | 72 | 0 | 0 |
| `tabular` | [causal](#causal-tabular) | 6 | 2 | 7 | 42 | 0 | 0 |
| `tabular` | [censored](#censored-tabular) | 6 | 2 | 5 | 30 | 0 | 0 |
| `tabular` | [classification_shift](#classification_shift-tabular) | 3 | 1 | 7 | 21 | 0 | 0 |
| `tabular` | [conformal](#conformal-tabular) | 30 | 10 | 22 | 660 | 0 | 0 |
| `tabular` | [constrained_outputs](#constrained_outputs-tabular) | 15 | 5 | 13 | 54 | 0 | 0 |
| `tabular` | [density_parity](#density_parity-tabular) | 21 | 7 | 10 | 210 | 0 | 0 |
| `tabular` | [distributional_bins](#distributional_bins-tabular) | 9 | 3 | 48 | 432 | 0 | 0 |
| `tabular` | [eiv](#eiv-tabular) | 24 | 8 | 25 | 582 | 0 | 18 |
| `tabular` | [ensembles](#ensembles-tabular) | 30 | 10 | 23 | 690 | 0 | 0 |
| `tabular` | [label_shift](#label_shift-tabular) | 9 | 3 | 12 | 105 | 3 | 0 |
| `tabular` | [likelihoods](#likelihoods-tabular) | 21 | 7 | 16 | 138 | 0 | 0 |
| `tabular` | [missing_data](#missing_data-tabular) | 9 | 3 | 4 | 36 | 0 | 0 |
| `tabular` | [multivariate_intervals](#multivariate_intervals-tabular) | 15 | 5 | 22 | 330 | 0 | 0 |
| `tabular` | [ordinal](#ordinal-tabular) | 6 | 2 | 7 | 42 | 0 | 0 |
| `tabular` | [orthogonal_inference](#orthogonal_inference-tabular) | 6 | 2 | 9 | 54 | 0 | 0 |
| `tabular` | [ppi](#ppi-tabular) | 9 | 3 | 8 | 72 | 0 | 0 |
| `tabular` | [pzflow_parity](#pzflow_parity-tabular) | 12 | 4 | 4 | 36 | 12 | 0 |
| `tabular` | [r2c_conformal](#r2c_conformal-tabular) | 15 | 5 | 6 | 90 | 0 | 0 |
| `tabular` | [robust_regression](#robust_regression-tabular) | 9 | 3 | 41 | 369 | 0 | 0 |
| `tabular` | [semi_supervised](#semi_supervised-tabular) | 6 | 2 | 9 | 54 | 0 | 0 |
| `tabular` | [sklearn_baselines](#sklearn_baselines-tabular) | 51 | 17 | 22 | 1122 | 0 | 0 |
| `tabular` | [streaming_tta](#streaming_tta-tabular) | 9 | 3 | 5 | 45 | 0 | 0 |
| `tabular` | [stress_matrix](#stress_matrix-tabular) | 18 | 6 | 18 | 54 | 0 | 0 |
| `tabular` | [test_time_adaptation](#test_time_adaptation-tabular) | 9 | 3 | 10 | 90 | 0 | 0 |
| `tabular` | [tweedie](#tweedie-tabular) | 6 | 2 | 11 | 48 | 0 | 18 |
| `tabular` | [uq](#uq-tabular) | 15 | 5 | 12 | 180 | 0 | 0 |
| `tabular` | [uq_layer](#uq_layer-tabular) | 30 | 10 | 50 | 1446 | 0 | 54 |
| `tabular_big` | [eiv](#eiv-tabular_big) | 3 | 1 | 25 | 69 | 0 | 6 |
| `tabular_big` | [likelihoods](#likelihoods-tabular_big) | 6 | 2 | 10 | 60 | 0 | 0 |
| **total** |  | **579** | **193** | **904** | **13966** | **39** | **99** |

## Parity: torchregress vs comparators

Pairs come from the explicit rules in Appendix C, evaluated per dataset on the seeds where both rows are ok. Verdict on the largest relative difference of the seed means over the pair's key metrics: exact ≤ 1e-06, close ≤ 2%, otherwise differs. *Within noise*: every key metric has |Δmean| ≤ 2 standard errors (z column). Pair kinds: *deterministic* pairs apply the same algorithm to the same fitted model or data and should agree to numerical precision; *matched* pairs train the same architecture / capacity independently, so they should agree within seed noise; *method* pairs share the method but not the model or training recipe, so a difference is not a defect by itself.

### Verdict counts

| Kind | Suite | Exact | Close | Differs | n/a | Within noise | Pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic | `bayesian_linear` | 3 | 0 | 0 | 0 | 0 | 3 |
| deterministic | `calibration` | 4 | 0 | 0 | 0 | 0 | 4 |
| deterministic | `conformal` | 0 | 30 | 0 | 0 | 0 | 30 |
| deterministic | `image_conformal` | 0 | 2 | 0 | 1 | 0 | 3 |
| deterministic | `likelihoods` | 0 | 6 | 0 | 0 | 0 | 6 |
| deterministic | `robust_regression` | 0 | 6 | 0 | 0 | 0 | 6 |
| deterministic | `sklearn_baselines` | 15 | 15 | 4 | 0 | 0 | 34 |
| deterministic | `tweedie` | 0 | 4 | 0 | 0 | 0 | 4 |
| deterministic | `uq_layer` | 181 | 0 | 0 | 9 | 0 | 190 |
| matched | `conformal` | 0 | 4 | 6 | 0 | 7 | 10 |
| matched | `density_parity` | 0 | 2 | 33 | 0 | 22 | 35 |
| matched | `image_density_parity` | 0 | 3 | 11 | 1 | 9 | 15 |
| matched | `orthogonal_inference` | 0 | 2 | 0 | 0 | 2 | 2 |
| matched | `sbi_parity` | 0 | 0 | 15 | 0 | 6 | 15 |
| method | `age_regression` | 0 | 0 | 10 | 0 | 4 | 10 |
| method | `causal` | 0 | 0 | 2 | 0 | 2 | 2 |
| method | `censored` | 0 | 0 | 2 | 0 | 0 | 2 |
| method | `censored_imaging` | 0 | 0 | 4 | 0 | 0 | 4 |
| method | `density_parity` | 0 | 0 | 14 | 0 | 2 | 14 |
| method | `ensembles` | 0 | 2 | 8 | 0 | 3 | 10 |
| method | `image_ensembles_uq` | 0 | 0 | 1 | 0 | 0 | 1 |
| method | `label_shift` | 0 | 0 | 3 | 0 | 0 | 3 |
| method | `missing_data` | 0 | 1 | 2 | 0 | 2 | 3 |
| method | `ordinal` | 0 | 0 | 2 | 0 | 0 | 2 |
| method | `ppi` | 0 | 3 | 3 | 0 | 5 | 6 |
| method | `pzflow_parity` | 0 | 0 | 4 | 4 | 0 | 8 |
| method | `r2c_conformal` | 0 | 0 | 5 | 0 | 0 | 5 |
| method | `robust_regression` | 0 | 0 | 3 | 0 | 0 | 3 |
| method | `sbi_parity` | 0 | 0 | 5 | 0 | 0 | 5 |
| method | `stress_matrix` | 0 | 0 | 1 | 0 | 0 | 1 |
| **deterministic total** |  | **203** | **63** | **4** | **10** | **0** | **280** |
| **matched total** |  | **0** | **11** | **65** | **1** | **46** | **77** |
| **method total** |  | **0** | **6** | **69** | **4** | **18** | **79** |

### Deterministic pairs

| Suite | Dataset | torchregress | Comparator | Seeds | Key metrics: tr vs comp (rel. diff, z) | Within noise | Verdict |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| `bayesian_linear` | california_housing | `BLR_Linear` | `sklearn_BayesianRidge` | 3 | RMSE 1.0393 vs 1.0393 (2.0e-05%, z=0.0)<br>NLL 1.2377 vs 1.2377 (1.1e-05%, z=0.0)<br>CRPS 0.4605 vs 0.4605 (1.5e-05%, z=0.0) | - | exact |
| `bayesian_linear` | diabetes | `BLR_Linear` | `sklearn_BayesianRidge` | 3 | RMSE 56.4870 vs 56.4869 (3.1e-06%, z=0.0)<br>NLL 5.4670 vs 5.4670 (6.3e-07%, z=0.0)<br>CRPS 32.1853 vs 32.1853 (3.6e-06%, z=0.0) | - | exact |
| `bayesian_linear` | synthetic_sinusoidal | `BLR_Linear` | `sklearn_BayesianRidge` | 3 | RMSE 0.7638 vs 0.7638 (2.4e-07%, z=0.0)<br>NLL 1.1522 vs 1.1522 (1.8e-07%, z=0.0)<br>CRPS 0.4430 vs 0.4430 (2.1e-07%, z=0.0) | - | exact |
| `calibration` | california_housing | `IsotonicMeanCalibrator` | `sklearn_IsotonicRegression` | 3 | RMSE 0.6001 vs 0.6001 (2.2e-08%, z=0.0)<br>MAE 0.4022 vs 0.4022 (2.1e-07%, z=0.0) | - | exact |
| `calibration` | diabetes | `IsotonicMeanCalibrator` | `sklearn_IsotonicRegression` | 3 | RMSE 58.4691 vs 58.4691 (4.9e-09%, z=0.0)<br>MAE 46.0007 vs 46.0007 (1.4e-08%, z=0.0) | - | exact |
| `calibration` | synthetic_hetero | `IsotonicMeanCalibrator` | `sklearn_IsotonicRegression` | 3 | RMSE 0.3972 vs 0.3972 (1.2e-07%, z=0.0)<br>MAE 0.3070 vs 0.3070 (2.0e-07%, z=0.0) | - | exact |
| `calibration` | synthetic_outliers | `IsotonicMeanCalibrator` | `sklearn_IsotonicRegression` | 3 | RMSE 1.3095 vs 1.3095 (7.1e-08%, z=0.0)<br>MAE 0.7736 vs 0.7736 (7.2e-08%, z=0.0) | - | exact |
| `conformal` | california_housing | `Split+Linear(post-hoc)` | `MAPIE_Split+Linear` | 3 | IntervalScore 3.5668 vs 3.5667 (7.7e-04%, z=0.0)<br>Coverage 0.9051 vs 0.9051 (0, z=0.0)<br>Width 2.1499 vs 2.1499 (1.0e-03%, z=0.0) | - | close |
| `conformal` | california_housing | `Split+Linear(post-hoc)` | `crepes_Split+Linear` | 3 | IntervalScore 3.5668 vs 3.5667 (7.7e-04%, z=0.0)<br>Coverage 0.9051 vs 0.9051 (0, z=0.0)<br>Width 2.1499 vs 2.1499 (1.1e-03%, z=0.0) | - | close |
| `conformal` | california_housing | `Split+Linear(post-hoc)` | `torchcp_Split+Linear` | 3 | IntervalScore 3.5668 vs 3.5667 (7.7e-04%, z=0.0)<br>Coverage 0.9051 vs 0.9051 (0, z=0.0)<br>Width 2.1499 vs 2.1499 (1.1e-03%, z=0.0) | - | close |
| `conformal` | diabetes | `Split+Linear(post-hoc)` | `MAPIE_Split+Linear` | 3 | IntervalScore 222.5 vs 222.5 (2.0e-03%, z=0.0)<br>Coverage 0.9424 vs 0.9424 (0, z=0.0)<br>Width 198.2 vs 198.2 (4.2e-03%, z=0.0) | - | close |
| `conformal` | diabetes | `Split+Linear(post-hoc)` | `crepes_Split+Linear` | 3 | IntervalScore 222.5 vs 222.5 (2.0e-03%, z=0.0)<br>Coverage 0.9424 vs 0.9424 (0, z=0.0)<br>Width 198.2 vs 198.2 (4.1e-03%, z=0.0) | - | close |
| `conformal` | diabetes | `Split+Linear(post-hoc)` | `torchcp_Split+Linear` | 3 | IntervalScore 222.5 vs 222.5 (2.0e-03%, z=0.0)<br>Coverage 0.9424 vs 0.9424 (0, z=0.0)<br>Width 198.2 vs 198.2 (4.1e-03%, z=0.0) | - | close |
| `conformal` | house_sales | `Split+Linear(post-hoc)` | `MAPIE_Split+Linear` | 3 | IntervalScore 9.322e+05 vs 9.322e+05 (4.9e-04%, z=0.0)<br>Coverage 0.8913 vs 0.8913 (0, z=0.0)<br>Width 5.086e+05 vs 5.086e+05 (1.7e-03%, z=0.0) | - | close |
| `conformal` | house_sales | `Split+Linear(post-hoc)` | `crepes_Split+Linear` | 3 | IntervalScore 9.322e+05 vs 9.322e+05 (4.8e-04%, z=0.0)<br>Coverage 0.8913 vs 0.8913 (0, z=0.0)<br>Width 5.086e+05 vs 5.086e+05 (1.7e-03%, z=0.0) | - | close |
| `conformal` | house_sales | `Split+Linear(post-hoc)` | `torchcp_Split+Linear` | 3 | IntervalScore 9.322e+05 vs 9.322e+05 (4.8e-04%, z=0.0)<br>Coverage 0.8913 vs 0.8913 (0, z=0.0)<br>Width 5.086e+05 vs 5.086e+05 (1.7e-03%, z=0.0) | - | close |
| `conformal` | insurance_claims | `Split+Linear(post-hoc)` | `MAPIE_Split+Linear` | 3 | IntervalScore 1.1127 vs 1.1127 (7.2e-04%, z=0.0)<br>Coverage 0.8937 vs 0.8938 (0.01%, z=0.1)<br>Width 0.1819 vs 0.1820 (0.01%, z=0.0) | - | close |
| `conformal` | insurance_claims | `Split+Linear(post-hoc)` | `crepes_Split+Linear` | 3 | IntervalScore 1.1127 vs 1.1128 (9.6e-03%, z=0.1)<br>Coverage 0.8937 vs 0.8940 (0.04%, z=0.1)<br>Width 0.1819 vs 0.1821 (0.11%, z=0.0) | - | close |
| `conformal` | insurance_claims | `Split+Linear(post-hoc)` | `torchcp_Split+Linear` | 3 | IntervalScore 1.1127 vs 1.1128 (9.6e-03%, z=0.1)<br>Coverage 0.8937 vs 0.8940 (0.04%, z=0.1)<br>Width 0.1819 vs 0.1821 (0.11%, z=0.0) | - | close |
| `conformal` | openml_year | `Split+Linear(post-hoc)` | `MAPIE_Split+Linear` | 3 | IntervalScore 0.9256 vs 0.9256 (5.6e-04%, z=0.0)<br>Coverage 0.9046 vs 0.9046 (0, z=0.0)<br>Width 0.6830 vs 0.6829 (3.1e-03%, z=0.0) | - | close |
| `conformal` | openml_year | `Split+Linear(post-hoc)` | `crepes_Split+Linear` | 3 | IntervalScore 0.9256 vs 0.9256 (5.6e-04%, z=0.0)<br>Coverage 0.9046 vs 0.9046 (0, z=0.0)<br>Width 0.6830 vs 0.6829 (3.1e-03%, z=0.0) | - | close |
| `conformal` | openml_year | `Split+Linear(post-hoc)` | `torchcp_Split+Linear` | 3 | IntervalScore 0.9256 vs 0.9256 (5.6e-04%, z=0.0)<br>Coverage 0.9046 vs 0.9046 (0, z=0.0)<br>Width 0.6830 vs 0.6829 (3.1e-03%, z=0.0) | - | close |
| `conformal` | synthetic_hetero | `Split+Linear(post-hoc)` | `MAPIE_Split+Linear` | 3 | IntervalScore 1.6882 vs 1.6882 (2.9e-05%, z=0.0)<br>Coverage 0.9156 vs 0.9156 (0, z=0.0)<br>Width 1.3772 vs 1.3773 (1.8e-03%, z=0.0) | - | close |
| `conformal` | synthetic_hetero | `Split+Linear(post-hoc)` | `crepes_Split+Linear` | 3 | IntervalScore 1.6882 vs 1.6882 (3.1e-05%, z=0.0)<br>Coverage 0.9156 vs 0.9156 (0, z=0.0)<br>Width 1.3772 vs 1.3773 (1.9e-03%, z=0.0) | - | close |
| `conformal` | synthetic_hetero | `Split+Linear(post-hoc)` | `torchcp_Split+Linear` | 3 | IntervalScore 1.6882 vs 1.6882 (3.1e-05%, z=0.0)<br>Coverage 0.9156 vs 0.9156 (0, z=0.0)<br>Width 1.3772 vs 1.3773 (1.9e-03%, z=0.0) | - | close |
| `conformal` | tabred_cooking | `Split+Linear(post-hoc)` | `MAPIE_Split+Linear` | 3 | IntervalScore 2.0653 vs 2.0637 (0.08%, z=0.4)<br>Coverage 0.8979 vs 0.8979 (0, z=0.0)<br>Width 1.5745 vs 1.5725 (0.13%, z=0.1) | - | close |
| `conformal` | tabred_cooking | `Split+Linear(post-hoc)` | `crepes_Split+Linear` | 3 | IntervalScore 2.0653 vs 2.0637 (0.08%, z=0.4)<br>Coverage 0.8979 vs 0.8979 (0, z=0.0)<br>Width 1.5745 vs 1.5725 (0.13%, z=0.1) | - | close |
| `conformal` | tabred_cooking | `Split+Linear(post-hoc)` | `torchcp_Split+Linear` | 3 | IntervalScore 2.0653 vs 2.0637 (0.08%, z=0.4)<br>Coverage 0.8979 vs 0.8979 (0, z=0.0)<br>Width 1.5745 vs 1.5725 (0.13%, z=0.1) | - | close |
| `conformal` | tabred_delivery | `Split+Linear(post-hoc)` | `MAPIE_Split+Linear` | 3 | IntervalScore 2.3412 vs 2.3359 (0.23%, z=0.7)<br>Coverage 0.8910 vs 0.8879 (0.35%, z=1.7)<br>Width 1.7909 vs 1.7811 (0.55%, z=1.0) | - | close |
| `conformal` | tabred_delivery | `Split+Linear(post-hoc)` | `crepes_Split+Linear` | 3 | IntervalScore 2.3412 vs 2.3359 (0.23%, z=0.7)<br>Coverage 0.8910 vs 0.8879 (0.35%, z=1.7)<br>Width 1.7909 vs 1.7811 (0.55%, z=1.0) | - | close |
| `conformal` | tabred_delivery | `Split+Linear(post-hoc)` | `torchcp_Split+Linear` | 3 | IntervalScore 2.3412 vs 2.3359 (0.23%, z=0.7)<br>Coverage 0.8910 vs 0.8879 (0.35%, z=1.7)<br>Width 1.7909 vs 1.7811 (0.55%, z=1.0) | - | close |
| `conformal` | tabred_weather | `Split+Linear(post-hoc)` | `MAPIE_Split+Linear` | 3 | IntervalScore 7.8308 vs 7.8437 (0.16%, z=1.7)<br>Coverage 0.9069 vs 0.9047 (0.25%, z=0.3)<br>Width 5.6862 vs 5.6613 (0.44%, z=0.2) | - | close |
| `conformal` | tabred_weather | `Split+Linear(post-hoc)` | `crepes_Split+Linear` | 3 | IntervalScore 7.8308 vs 7.8437 (0.16%, z=1.7)<br>Coverage 0.9069 vs 0.9047 (0.25%, z=0.3)<br>Width 5.6862 vs 5.6612 (0.44%, z=0.2) | - | close |
| `conformal` | tabred_weather | `Split+Linear(post-hoc)` | `torchcp_Split+Linear` | 3 | IntervalScore 7.8308 vs 7.8437 (0.16%, z=1.7)<br>Coverage 0.9069 vs 0.9047 (0.25%, z=0.3)<br>Width 5.6862 vs 5.6612 (0.44%, z=0.2) | - | close |
| `conformal` | zurich_delays | `Split+Linear(post-hoc)` | `MAPIE_Split+Linear` | 3 | IntervalScore 476.4 vs 476.4 (8.3e-05%, z=0.0)<br>Coverage 0.9026 vs 0.9029 (0.03%, z=0.3)<br>Width 237.7 vs 237.9 (0.09%, z=0.2) | - | close |
| `conformal` | zurich_delays | `Split+Linear(post-hoc)` | `crepes_Split+Linear` | 3 | IntervalScore 476.4 vs 476.4 (2.1e-03%, z=0.0)<br>Coverage 0.9026 vs 0.9029 (0.03%, z=0.3)<br>Width 237.7 vs 237.8 (0.07%, z=0.2) | - | close |
| `conformal` | zurich_delays | `Split+Linear(post-hoc)` | `torchcp_Split+Linear` | 3 | IntervalScore 476.4 vs 476.4 (2.1e-03%, z=0.0)<br>Coverage 0.9026 vs 0.9029 (0.03%, z=0.3)<br>Width 237.7 vs 237.8 (0.07%, z=0.2) | - | close |
| `image_conformal` | synthetic_image_regression | `Split+CNN` | `MAPIE_Split+CNN` | 3 | IntervalScore 0.2483 vs 0.2483 (0.01%, z=0.0)<br>Coverage 0.8889 vs 0.8889 (0, z=0.0)<br>Width 0.1877 vs 0.1874 (0.13%, z=0.0) | - | close |
| `image_conformal` | synthetic_image_regression | `Split+CNN` | `crepes_Split+CNN` | 3 | IntervalScore 0.2483 vs 0.2483 (0.01%, z=0.0)<br>Coverage 0.8889 vs 0.8889 (0, z=0.0)<br>Width 0.1877 vs 0.1874 (0.13%, z=0.0) | - | close |
| `image_conformal` | synthetic_image_regression | `Split+CNN` | `torchcp_Split+CNN` | 0 | no common ok seeds | - | n/a |
| `likelihoods` | lik_beta | `tr_Linear` | `scipy_MLE_Linear` | 3 | NLL -0.3354 vs -0.3354 (2.5e-04%, z=0.0)<br>LocRMSE 0.0070 vs 0.0070 (0.03%, z=0.0)<br>ScaleRMSE 0.1290 vs 0.1291 (0.08%, z=0.0) | - | close |
| `likelihoods` | lik_gev | `tr_Linear` | `scipy_MLE_Linear` | 3 | NLL 1.0364 vs 1.0364 (1.0e-04%, z=0.0)<br>LocRMSE 0.0264 vs 0.0264 (0.01%, z=0.0)<br>ScaleRMSE 0.0203 vs 0.0203 (0.03%, z=0.0) | - | close |
| `likelihoods` | lik_johnsonsu | `tr_Linear` | `scipy_MLE_Linear` | 3 | NLL 0.7621 vs 0.7621 (3.8e-04%, z=0.0)<br>LocRMSE 0.0843 vs 0.0843 (0.06%, z=0.0)<br>ScaleRMSE 0.0620 vs 0.0620 (0.05%, z=0.0) | - | close |
| `likelihoods` | lik_sinharcsinh | `tr_Linear` | `scipy_MLE_Linear` | 3 | NLL 0.5759 vs 0.5759 (1.6e-04%, z=0.0)<br>LocRMSE 0.0438 vs 0.0437 (0.29%, z=0.0)<br>ScaleRMSE 0.0661 vs 0.0659 (0.21%, z=0.0) | - | close |
| `likelihoods` | lik_skewnormal | `tr_Linear` | `scipy_MLE_Linear` | 3 | NLL 0.2947 vs 0.2947 (1.8e-03%, z=0.0)<br>LocRMSE 0.0157 vs 0.0158 (0.40%, z=0.0)<br>ScaleRMSE 0.0193 vs 0.0193 (0.32%, z=0.0) | - | close |
| `likelihoods` | lik_skewt | `tr_Linear` | `scipy_MLE_Linear` | 3 | NLL 0.5651 vs 0.5651 (1.2e-03%, z=0.0)<br>LocRMSE 0.0296 vs 0.0297 (0.12%, z=0.0)<br>ScaleRMSE 0.0414 vs 0.0414 (0.14%, z=0.0) | - | close |
| `robust_regression` | california_housing | `Huber_Linear` | `statsmodels_RLM` | 3 | RMSE 0.8248 vs 0.8260 (0.14%, z=50.4)<br>MAE 0.5176 vs 0.5127 (0.94%, z=3589.4) | - | close |
| `robust_regression` | california_housing | `Quantile_Linear` | `sklearn_QuantileRegressor` | 3 | RMSE 0.8249 vs 0.8243 (0.08%, z=1.5)<br>MAE 0.5084 vs 0.5084 (5.1e-03%, z=2.2) | - | close |
| `robust_regression` | synthetic_outliers | `Huber_Linear` | `statsmodels_RLM` | 3 | RMSE 1.3855 vs 1.3873 (0.13%, z=2028.3)<br>MAE 0.8519 vs 0.8415 (1.22%, z=11813.9) | - | close |
| `robust_regression` | synthetic_outliers | `Quantile_Linear` | `sklearn_QuantileRegressor` | 3 | RMSE 1.3974 vs 1.3975 (9.8e-03%, z=3.9)<br>MAE 0.8439 vs 0.8440 (0.02%, z=2.8) | - | close |
| `robust_regression` | zurich_delays | `Huber_Linear` | `statsmodels_RLM` | 3 | RMSE 118.5 vs 119.7 (1.00%, z=2892.7)<br>MAE 60.5592 vs 59.4249 (1.87%, z=896.9) | - | close |
| `robust_regression` | zurich_delays | `Quantile_Linear` | `sklearn_QuantileRegressor` | 3 | RMSE 121.1 vs 121.1 (9.2e-03%, z=1.7)<br>MAE 59.0513 vs 59.0482 (5.3e-03%, z=2.0) | - | close |
| `sklearn_baselines` | california_housing | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 0.7984 vs 0.7984 (4.5e-04%, z=0.6)<br>MAE 0.5357 vs 0.5357 (2.0e-04%, z=2.1) | - | close |
| `sklearn_baselines` | california_housing | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 0.7984 vs 0.7984 (3.9e-05%, z=33.2)<br>MAE 0.5357 vs 0.5357 (1.3e-05%, z=102.2) | - | exact |
| `sklearn_baselines` | diabetes | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 53.1200 vs 53.1202 (2.1e-04%, z=0.5)<br>MAE 41.9194 vs 41.9194 (5.0e-05%, z=0.2) | - | close |
| `sklearn_baselines` | diabetes | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 53.1202 vs 53.1202 (2.2e-06%, z=93.8)<br>MAE 41.9194 vs 41.9194 (1.4e-06%, z=10.8) | - | exact |
| `sklearn_baselines` | house_sales | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 2.075e+05 vs 2.075e+05 (6.0e-05%, z=1.3)<br>MAE 1.282e+05 vs 1.282e+05 (2.3e-04%, z=1.1) | - | close |
| `sklearn_baselines` | house_sales | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 2.075e+05 vs 2.075e+05 (2.7e-06%, z=2.1)<br>MAE 1.282e+05 vs 1.282e+05 (3.1e-05%, z=7.6) | - | exact |
| `sklearn_baselines` | insurance_claims | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 0.2362 vs 0.2362 (9.7e-03%, z=78.1)<br>MAE 0.0983 vs 0.0983 (0.03%, z=883.4) | - | close |
| `sklearn_baselines` | insurance_claims | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 0.2407 vs 0.2362 (1.85%, z=2.0)<br>MAE 0.0714 vs 0.0983 (27.39%, z=2.0) | - | **differs** |
| `sklearn_baselines` | make_regression_synthetic | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 14.9186 vs 14.9186 (7.0e-05%, z=0.6)<br>MAE 11.9373 vs 11.9373 (7.1e-05%, z=0.5) | - | exact |
| `sklearn_baselines` | make_regression_synthetic | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 14.9186 vs 14.9186 (2.4e-05%, z=89.8)<br>MAE 11.9373 vs 11.9373 (3.3e-05%, z=43.6) | - | exact |
| `sklearn_baselines` | openml_year | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 0.2156 vs 0.2156 (5.0e-05%, z=0.9)<br>MAE 0.1670 vs 0.1670 (5.4e-05%, z=0.8) | - | exact |
| `sklearn_baselines` | openml_year | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 0.2156 vs 0.2156 (6.8e-07%, z=5.5)<br>MAE 0.1670 vs 0.1670 (9.6e-06%, z=4.8) | - | exact |
| `sklearn_baselines` | synthetic_censored | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 0.3787 vs 0.3787 (1.6e-04%, z=0.8)<br>MAE 0.2998 vs 0.2998 (1.6e-04%, z=0.8) | - | close |
| `sklearn_baselines` | synthetic_censored | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 0.3787 vs 0.3787 (5.2e-06%)<br>MAE 0.2998 vs 0.2998 (0, z=0.0) | - | exact |
| `sklearn_baselines` | synthetic_eiv | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 0.2322 vs 0.2322 (9.7e-05%, z=0.3)<br>MAE 0.1850 vs 0.1850 (2.9e-04%, z=0.5) | - | close |
| `sklearn_baselines` | synthetic_eiv | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 0.2322 vs 0.2322 (1.5e-06%, z=0.6)<br>MAE 0.1850 vs 0.1850 (1.1e-05%, z=2.2) | - | exact |
| `sklearn_baselines` | synthetic_hetero | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 0.3850 vs 0.3850 (3.8e-06%, z=0.3)<br>MAE 0.2959 vs 0.2959 (1.5e-07%, z=0.0) | - | exact |
| `sklearn_baselines` | synthetic_hetero | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 0.3850 vs 0.3850 (3.7e-06%)<br>MAE 0.2959 vs 0.2959 (3.1e-08%) | - | exact |
| `sklearn_baselines` | synthetic_outliers | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 1.3885 vs 1.3885 (2.0e-04%, z=1.0)<br>MAE 0.8542 vs 0.8542 (4.1e-04%, z=1.0) | - | close |
| `sklearn_baselines` | synthetic_outliers | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 1.3885 vs 1.3885 (2.0e-07%, z=0.2)<br>MAE 0.8542 vs 0.8542 (1.4e-06%, z=0.4) | - | exact |
| `sklearn_baselines` | synthetic_sinusoidal | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 0.7437 vs 0.7437 (3.1e-06%, z=0.2)<br>MAE 0.6393 vs 0.6393 (2.8e-05%, z=0.2) | - | exact |
| `sklearn_baselines` | synthetic_sinusoidal | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 0.7437 vs 0.7437 (3.6e-07%, z=5.0)<br>MAE 0.6393 vs 0.6393 (5.3e-07%, z=4.9) | - | exact |
| `sklearn_baselines` | synthetic_tweedie | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 1.2697 vs 1.2697 (6.8e-05%, z=0.5)<br>MAE 0.8529 vs 0.8529 (1.2e-04%, z=1.6) | - | close |
| `sklearn_baselines` | synthetic_tweedie | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 1.2697 vs 1.2697 (3.7e-06%, z=1.7)<br>MAE 0.8529 vs 0.8529 (2.3e-06%, z=1.0) | - | exact |
| `sklearn_baselines` | tabred_cooking | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 0.4766 vs 0.4763 (0.07%, z=194.5)<br>MAE 0.3711 vs 0.3709 (0.06%, z=102.9) | - | close |
| `sklearn_baselines` | tabred_cooking | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 0.4824 vs 0.4763 (1.25%, z=2.1)<br>MAE 0.3758 vs 0.3709 (1.31%, z=2.0) | - | close |
| `sklearn_baselines` | tabred_delivery | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 0.5624 vs 0.5626 (0.05%, z=9.6)<br>MAE 0.4494 vs 0.4500 (0.13%, z=23.7) | - | close |
| `sklearn_baselines` | tabred_delivery | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 0.6570 vs 0.5626 (14.36%, z=6.1)<br>MAE 0.5188 vs 0.4500 (13.25%, z=3.9) | - | **differs** |
| `sklearn_baselines` | tabred_maps | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 0.1649 vs 0.1652 (0.19%, z=146.4)<br>MAE 0.1247 vs 0.1249 (0.19%, z=129.4) | - | close |
| `sklearn_baselines` | tabred_maps | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 0.1830 vs 0.1652 (9.73%, z=65.2)<br>MAE 0.1400 vs 0.1249 (10.76%, z=62.1) | - | **differs** |
| `sklearn_baselines` | tabred_weather | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 1.7631 vs 1.7666 (0.20%, z=183.6)<br>MAE 1.3120 vs 1.3157 (0.28%, z=392.9) | - | close |
| `sklearn_baselines` | tabred_weather | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 2.0389 vs 1.7666 (13.36%, z=2.7)<br>MAE 1.5367 vs 1.3157 (14.38%, z=3.0) | - | **differs** |
| `sklearn_baselines` | zurich_delays | `GaussianNLL_Linear` | `sklearn_LinearRegression` | 3 | RMSE 117.6 vs 117.6 (1.5e-04%, z=0.8)<br>MAE 63.3154 vs 63.3105 (7.9e-03%, z=8.4) | - | close |
| `sklearn_baselines` | zurich_delays | `OLS` | `sklearn_LinearRegression` | 3 | RMSE 118.8 vs 117.6 (1.02%, z=1.0)<br>MAE 64.2471 vs 63.3105 (1.46%, z=1.0) | - | close |
| `tweedie` | insurance_claims | `CompoundPoisson_Linear` | `statsmodels_GLM` | 3 | TweedieDeviance 1.4131 vs 1.4133 (0.01%, z=38.2)<br>RMSE 0.2361 vs 0.2361 (4.5e-03%, z=54.3)<br>R2 0.0072 vs 0.0071 (1.24%, z=53.5) | - | close |
| `tweedie` | insurance_claims | `Tweedie_Linear` | `statsmodels_GLM` | 3 | TweedieDeviance 1.4131 vs 1.4133 (0.01%, z=38.2)<br>RMSE 0.2361 vs 0.2361 (4.5e-03%, z=54.3)<br>R2 0.0072 vs 0.0071 (1.24%, z=53.5) | - | close |
| `tweedie` | synthetic_tweedie | `CompoundPoisson_Linear` | `statsmodels_GLM` | 3 | TweedieDeviance 0.7174 vs 0.7174 (1.6e-05%, z=0.9)<br>RMSE 1.1060 vs 1.1060 (7.6e-05%, z=0.9)<br>R2 0.5489 vs 0.5489 (1.3e-04%, z=0.9) | - | close |
| `tweedie` | synthetic_tweedie | `Tweedie_Linear` | `statsmodels_GLM` | 3 | TweedieDeviance 0.7174 vs 0.7174 (1.6e-05%, z=0.9)<br>RMSE 1.1060 vs 1.1060 (7.6e-05%, z=0.9)<br>R2 0.5489 vs 0.5489 (1.3e-04%, z=0.9) | - | close |
| `uq_layer` | california_housing | `tr_CQR+DeepEnsK5` | `MAPIE_CQR+DeepEnsK5` | 3 | IntervalScore 2.4059 vs 2.4059 (0, z=0.0)<br>Coverage 0.9124 vs 0.9124 (0, z=0.0)<br>Width 1.6535 vs 1.6535 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_CQR+LightGBM` | `MAPIE_CQR+LightGBM` | 3 | IntervalScore 2.4918 vs 2.4918 (0, z=0.0)<br>Coverage 0.9049 vs 0.9049 (0, z=0.0)<br>Width 1.7488 vs 1.7488 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_CQR+RF` | `MAPIE_CQR+RF` | 3 | IntervalScore 2.3999 vs 2.3999 (0, z=0.0)<br>Coverage 0.9098 vs 0.9098 (0, z=0.0)<br>Width 1.6373 vs 1.6373 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_CQR+TabularMLP` | `MAPIE_CQR+TabularMLP` | 3 | IntervalScore 2.3323 vs 2.3323 (0, z=0.0)<br>Coverage 0.9098 vs 0.9098 (0, z=0.0)<br>Width 1.5819 vs 1.5819 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_CVPlus+LightGBM` | `MAPIE_CVPlus+LightGBM` | 3 | IntervalScore 2.4956 vs 2.4956 (0, z=0.0)<br>Coverage 0.9111 vs 0.9111 (0, z=0.0)<br>Width 1.5881 vs 1.5881 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_CVPlus+RF` | `MAPIE_CVPlus+RF` | 3 | IntervalScore 2.7675 vs 2.7675 (0, z=0.0)<br>Coverage 0.9104 vs 0.9104 (0, z=0.0)<br>Width 1.8197 vs 1.8197 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_JackknifePlus+LightGBM` | `MAPIE_JackknifePlus+LightGBM` | 0 | no common ok seeds | - | n/a |
| `uq_layer` | california_housing | `tr_Normalized+DeepEnsK5` | `crepes_Normalized+DeepEnsK5` | 3 | IntervalScore 2.4076 vs 2.4076 (0, z=0.0)<br>Coverage 0.9122 vs 0.9122 (0, z=0.0)<br>Width 1.6432 vs 1.6432 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Normalized+LightGBM` | `crepes_Normalized+LightGBM` | 3 | IntervalScore 2.3359 vs 2.3359 (0, z=0.0)<br>Coverage 0.9102 vs 0.9102 (0, z=0.0)<br>Width 1.5377 vs 1.5377 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Normalized+RF` | `crepes_Normalized+RF` | 3 | IntervalScore 2.4500 vs 2.4500 (0, z=0.0)<br>Coverage 0.9053 vs 0.9053 (0, z=0.0)<br>Width 1.5950 vs 1.5950 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Normalized+TabularMLP` | `crepes_Normalized+TabularMLP` | 3 | IntervalScore 2.3355 vs 2.3355 (0, z=0.0)<br>Coverage 0.9122 vs 0.9122 (0, z=0.0)<br>Width 1.5789 vs 1.5789 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Split+DeepEnsK5` | `MAPIE_Split+DeepEnsK5` | 3 | IntervalScore 2.8208 vs 2.8208 (0, z=0.0)<br>Coverage 0.9138 vs 0.9138 (0, z=0.0)<br>Width 1.8909 vs 1.8909 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Split+DeepEnsK5` | `crepes_Split+DeepEnsK5` | 3 | IntervalScore 2.8208 vs 2.8208 (0, z=0.0)<br>Coverage 0.9138 vs 0.9138 (0, z=0.0)<br>Width 1.8909 vs 1.8909 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Split+LightGBM` | `MAPIE_Split+LightGBM` | 3 | IntervalScore 2.5854 vs 2.5854 (0, z=0.0)<br>Coverage 0.9036 vs 0.9036 (0, z=0.0)<br>Width 1.6116 vs 1.6116 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Split+LightGBM` | `crepes_Split+LightGBM` | 3 | IntervalScore 2.5854 vs 2.5854 (0, z=0.0)<br>Coverage 0.9036 vs 0.9036 (0, z=0.0)<br>Width 1.6116 vs 1.6116 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Split+RF` | `MAPIE_Split+RF` | 3 | IntervalScore 2.8344 vs 2.8344 (0, z=0.0)<br>Coverage 0.9093 vs 0.9093 (0, z=0.0)<br>Width 1.8372 vs 1.8372 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Split+RF` | `crepes_Split+RF` | 3 | IntervalScore 2.8344 vs 2.8344 (0, z=0.0)<br>Coverage 0.9093 vs 0.9093 (0, z=0.0)<br>Width 1.8372 vs 1.8372 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Split+TabularMLP` | `MAPIE_Split+TabularMLP` | 3 | IntervalScore 2.8026 vs 2.8026 (0, z=0.0)<br>Coverage 0.9029 vs 0.9029 (0, z=0.0)<br>Width 1.7687 vs 1.7687 (0, z=0.0) | - | exact |
| `uq_layer` | california_housing | `tr_Split+TabularMLP` | `crepes_Split+TabularMLP` | 3 | IntervalScore 2.8026 vs 2.8026 (0, z=0.0)<br>Coverage 0.9029 vs 0.9029 (0, z=0.0)<br>Width 1.7687 vs 1.7687 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_CQR+DeepEnsK5` | `MAPIE_CQR+DeepEnsK5` | 3 | IntervalScore 211.1 vs 211.1 (0, z=0.0)<br>Coverage 0.9524 vs 0.9524 (0, z=0.0)<br>Width 200.3 vs 200.3 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_CQR+LightGBM` | `MAPIE_CQR+LightGBM` | 3 | IntervalScore 220.4 vs 220.4 (0, z=0.0)<br>Coverage 0.9424 vs 0.9424 (0, z=0.0)<br>Width 200.8 vs 200.8 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_CQR+RF` | `MAPIE_CQR+RF` | 3 | IntervalScore 215.9 vs 215.9 (0, z=0.0)<br>Coverage 0.9524 vs 0.9524 (0, z=0.0)<br>Width 198.9 vs 198.9 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_CQR+TabularMLP` | `MAPIE_CQR+TabularMLP` | 3 | IntervalScore 215.6 vs 215.6 (0, z=0.0)<br>Coverage 0.9449 vs 0.9449 (0, z=0.0)<br>Width 192.6 vs 192.6 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_CVPlus+LightGBM` | `MAPIE_CVPlus+LightGBM` | 3 | IntervalScore 250.3 vs 250.3 (0, z=0.0)<br>Coverage 0.9499 vs 0.9499 (0, z=0.0)<br>Width 222.3 vs 222.3 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_CVPlus+RF` | `MAPIE_CVPlus+RF` | 3 | IntervalScore 219.6 vs 219.6 (0, z=0.0)<br>Coverage 0.9524 vs 0.9524 (0, z=0.0)<br>Width 195.1 vs 195.1 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_JackknifePlus+LightGBM` | `MAPIE_JackknifePlus+LightGBM` | 3 | IntervalScore 251.9 vs 251.9 (0, z=0.0)<br>Coverage 0.9398 vs 0.9398 (0, z=0.0)<br>Width 216.1 vs 216.1 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Normalized+DeepEnsK5` | `crepes_Normalized+DeepEnsK5` | 3 | IntervalScore 210.1 vs 210.1 (0, z=0.0)<br>Coverage 0.9474 vs 0.9474 (0, z=0.0)<br>Width 198.8 vs 198.8 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Normalized+LightGBM` | `crepes_Normalized+LightGBM` | 3 | IntervalScore 270.1 vs 270.1 (0, z=0.0)<br>Coverage 0.9749 vs 0.9749 (0, z=0.0)<br>Width 259.4 vs 259.4 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Normalized+RF` | `crepes_Normalized+RF` | 3 | IntervalScore 220.8 vs 220.8 (0, z=0.0)<br>Coverage 0.9323 vs 0.9323 (0, z=0.0)<br>Width 192.7 vs 192.7 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Normalized+TabularMLP` | `crepes_Normalized+TabularMLP` | 3 | IntervalScore 214.9 vs 214.9 (0, z=0.0)<br>Coverage 0.9449 vs 0.9449 (0, z=0.0)<br>Width 191.8 vs 191.8 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Split+DeepEnsK5` | `MAPIE_Split+DeepEnsK5` | 3 | IntervalScore 227.4 vs 227.4 (0, z=0.0)<br>Coverage 0.9549 vs 0.9549 (0, z=0.0)<br>Width 208.4 vs 208.4 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Split+DeepEnsK5` | `crepes_Split+DeepEnsK5` | 3 | IntervalScore 227.4 vs 227.4 (0, z=0.0)<br>Coverage 0.9549 vs 0.9549 (0, z=0.0)<br>Width 208.4 vs 208.4 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Split+LightGBM` | `MAPIE_Split+LightGBM` | 3 | IntervalScore 259.5 vs 259.5 (0, z=0.0)<br>Coverage 0.9549 vs 0.9549 (0, z=0.0)<br>Width 234.4 vs 234.4 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Split+LightGBM` | `crepes_Split+LightGBM` | 3 | IntervalScore 259.5 vs 259.5 (0, z=0.0)<br>Coverage 0.9549 vs 0.9549 (0, z=0.0)<br>Width 234.4 vs 234.4 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Split+RF` | `MAPIE_Split+RF` | 3 | IntervalScore 233.2 vs 233.2 (0, z=0.0)<br>Coverage 0.9524 vs 0.9524 (0, z=0.0)<br>Width 213.0 vs 213.0 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Split+RF` | `crepes_Split+RF` | 3 | IntervalScore 233.2 vs 233.2 (0, z=0.0)<br>Coverage 0.9524 vs 0.9524 (0, z=0.0)<br>Width 213.0 vs 213.0 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Split+TabularMLP` | `MAPIE_Split+TabularMLP` | 3 | IntervalScore 229.6 vs 229.6 (0, z=0.0)<br>Coverage 0.9624 vs 0.9624 (0, z=0.0)<br>Width 210.3 vs 210.3 (0, z=0.0) | - | exact |
| `uq_layer` | diabetes | `tr_Split+TabularMLP` | `crepes_Split+TabularMLP` | 3 | IntervalScore 229.6 vs 229.6 (0, z=0.0)<br>Coverage 0.9624 vs 0.9624 (0, z=0.0)<br>Width 210.3 vs 210.3 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_CQR+DeepEnsK5` | `MAPIE_CQR+DeepEnsK5` | 3 | IntervalScore 3.945e+05 vs 3.945e+05 (0, z=0.0)<br>Coverage 0.9024 vs 0.9024 (0, z=0.0)<br>Width 2.679e+05 vs 2.679e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_CQR+LightGBM` | `MAPIE_CQR+LightGBM` | 3 | IntervalScore 4.291e+05 vs 4.291e+05 (0, z=0.0)<br>Coverage 0.8984 vs 0.8984 (0, z=0.0)<br>Width 3.022e+05 vs 3.022e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_CQR+RF` | `MAPIE_CQR+RF` | 3 | IntervalScore 4.306e+05 vs 4.306e+05 (0, z=0.0)<br>Coverage 0.8963 vs 0.8963 (0, z=0.0)<br>Width 3.066e+05 vs 3.066e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_CQR+TabularMLP` | `MAPIE_CQR+TabularMLP` | 3 | IntervalScore 3.691e+05 vs 3.691e+05 (0, z=0.0)<br>Coverage 0.8975 vs 0.8975 (0, z=0.0)<br>Width 2.648e+05 vs 2.648e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_CVPlus+LightGBM` | `MAPIE_CVPlus+LightGBM` | 3 | IntervalScore 5.235e+05 vs 5.235e+05 (0, z=0.0)<br>Coverage 0.9108 vs 0.9108 (0, z=0.0)<br>Width 3.072e+05 vs 3.072e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_CVPlus+RF` | `MAPIE_CVPlus+RF` | 3 | IntervalScore 5.968e+05 vs 5.968e+05 (0, z=0.0)<br>Coverage 0.9129 vs 0.9129 (0, z=0.0)<br>Width 3.293e+05 vs 3.293e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_JackknifePlus+LightGBM` | `MAPIE_JackknifePlus+LightGBM` | 0 | no common ok seeds | - | n/a |
| `uq_layer` | house_sales | `tr_Normalized+DeepEnsK5` | `crepes_Normalized+DeepEnsK5` | 3 | IntervalScore 3.947e+05 vs 3.947e+05 (0, z=0.0)<br>Coverage 0.9025 vs 0.9025 (0, z=0.0)<br>Width 2.677e+05 vs 2.677e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Normalized+LightGBM` | `crepes_Normalized+LightGBM` | 3 | IntervalScore 3.957e+05 vs 3.957e+05 (0, z=0.0)<br>Coverage 0.9041 vs 0.9041 (0, z=0.0)<br>Width 2.985e+05 vs 2.985e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Normalized+RF` | `crepes_Normalized+RF` | 3 | IntervalScore 4.323e+05 vs 4.323e+05 (0, z=0.0)<br>Coverage 0.8973 vs 0.8973 (0, z=0.0)<br>Width 3.158e+05 vs 3.158e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Normalized+TabularMLP` | `crepes_Normalized+TabularMLP` | 3 | IntervalScore 3.692e+05 vs 3.692e+05 (0, z=0.0)<br>Coverage 0.8976 vs 0.8976 (0, z=0.0)<br>Width 2.610e+05 vs 2.610e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Split+DeepEnsK5` | `MAPIE_Split+DeepEnsK5` | 3 | IntervalScore 5.624e+05 vs 5.624e+05 (0, z=0.0)<br>Coverage 0.8982 vs 0.8982 (0, z=0.0)<br>Width 2.923e+05 vs 2.923e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Split+DeepEnsK5` | `crepes_Split+DeepEnsK5` | 3 | IntervalScore 5.624e+05 vs 5.624e+05 (0, z=0.0)<br>Coverage 0.8982 vs 0.8982 (0, z=0.0)<br>Width 2.923e+05 vs 2.923e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Split+LightGBM` | `MAPIE_Split+LightGBM` | 3 | IntervalScore 5.693e+05 vs 5.693e+05 (0, z=0.0)<br>Coverage 0.9003 vs 0.9003 (0, z=0.0)<br>Width 2.983e+05 vs 2.983e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Split+LightGBM` | `crepes_Split+LightGBM` | 3 | IntervalScore 5.693e+05 vs 5.693e+05 (0, z=0.0)<br>Coverage 0.9003 vs 0.9003 (0, z=0.0)<br>Width 2.983e+05 vs 2.983e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Split+RF` | `MAPIE_Split+RF` | 3 | IntervalScore 6.415e+05 vs 6.415e+05 (0, z=0.0)<br>Coverage 0.9005 vs 0.9005 (0, z=0.0)<br>Width 3.139e+05 vs 3.139e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Split+RF` | `crepes_Split+RF` | 3 | IntervalScore 6.415e+05 vs 6.415e+05 (0, z=0.0)<br>Coverage 0.9005 vs 0.9005 (0, z=0.0)<br>Width 3.139e+05 vs 3.139e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Split+TabularMLP` | `MAPIE_Split+TabularMLP` | 3 | IntervalScore 5.375e+05 vs 5.375e+05 (0, z=0.0)<br>Coverage 0.8979 vs 0.8979 (0, z=0.0)<br>Width 2.777e+05 vs 2.777e+05 (0, z=0.0) | - | exact |
| `uq_layer` | house_sales | `tr_Split+TabularMLP` | `crepes_Split+TabularMLP` | 3 | IntervalScore 5.375e+05 vs 5.375e+05 (0, z=0.0)<br>Coverage 0.8979 vs 0.8979 (0, z=0.0)<br>Width 2.777e+05 vs 2.777e+05 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_CQR+DeepEnsK5` | `MAPIE_CQR+DeepEnsK5` | 3 | IntervalScore 1.2177 vs 1.2177 (0, z=0.0)<br>Coverage 0.8995 vs 0.8995 (0, z=0.0)<br>Width 0.4412 vs 0.4412 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_CQR+LightGBM` | `MAPIE_CQR+LightGBM` | 3 | IntervalScore 0.9492 vs 0.9492 (0, z=0.0)<br>Coverage 0.9494 vs 0.9494 (0, z=0.0)<br>Width 0.2410 vs 0.2410 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_CQR+RF` | `MAPIE_CQR+RF` | 3 | IntervalScore 0.9917 vs 0.9917 (0, z=0.0)<br>Coverage 0.9498 vs 0.9498 (0, z=0.0)<br>Width 0.3096 vs 0.3096 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_CQR+TabularMLP` | `MAPIE_CQR+TabularMLP` | 3 | IntervalScore 1.0941 vs 1.0941 (0, z=0.0)<br>Coverage 0.8986 vs 0.8986 (0, z=0.0)<br>Width 0.1799 vs 0.1799 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_CVPlus+LightGBM` | `MAPIE_CVPlus+LightGBM` | 3 | IntervalScore 1.1480 vs 1.1480 (0, z=0.0)<br>Coverage 0.9302 vs 0.9302 (0, z=0.0)<br>Width 0.2803 vs 0.2803 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_CVPlus+RF` | `MAPIE_CVPlus+RF` | 3 | IntervalScore 1.1561 vs 1.1561 (0, z=0.0)<br>Coverage 0.9219 vs 0.9219 (0, z=0.0)<br>Width 0.2992 vs 0.2992 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_JackknifePlus+LightGBM` | `MAPIE_JackknifePlus+LightGBM` | 0 | no common ok seeds | - | n/a |
| `uq_layer` | insurance_claims | `tr_Normalized+DeepEnsK5` | `crepes_Normalized+DeepEnsK5` | 3 | IntervalScore 1.1005 vs 1.1005 (0, z=0.0)<br>Coverage 0.9002 vs 0.9002 (0, z=0.0)<br>Width 0.1428 vs 0.1428 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Normalized+LightGBM` | `crepes_Normalized+LightGBM` | 3 | IntervalScore 35976.8 vs 35976.8 (0, z=0.0)<br>Coverage 0.9033 vs 0.9033 (0, z=0.0)<br>Width 35976.2 vs 35976.2 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Normalized+RF` | `crepes_Normalized+RF` | 3 | IntervalScore 1.0930 vs 1.0930 (0, z=0.0)<br>Coverage 0.8927 vs 0.8927 (0, z=0.0)<br>Width 0.1619 vs 0.1619 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Normalized+TabularMLP` | `crepes_Normalized+TabularMLP` | 3 | IntervalScore 1.0718 vs 1.0718 (0, z=0.0)<br>Coverage 0.8958 vs 0.8958 (0, z=0.0)<br>Width 0.1180 vs 0.1180 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Split+DeepEnsK5` | `MAPIE_Split+DeepEnsK5` | 3 | IntervalScore 1.1902 vs 1.1902 (0, z=0.0)<br>Coverage 0.8985 vs 0.8985 (0, z=0.0)<br>Width 0.1855 vs 0.1855 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Split+DeepEnsK5` | `crepes_Split+DeepEnsK5` | 3 | IntervalScore 1.1902 vs 1.1902 (0, z=0.0)<br>Coverage 0.8985 vs 0.8985 (0, z=0.0)<br>Width 0.1855 vs 0.1855 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Split+LightGBM` | `MAPIE_Split+LightGBM` | 3 | IntervalScore 1.2004 vs 1.2004 (0, z=0.0)<br>Coverage 0.8977 vs 0.8977 (0, z=0.0)<br>Width 0.2646 vs 0.2646 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Split+LightGBM` | `crepes_Split+LightGBM` | 3 | IntervalScore 1.2004 vs 1.2004 (0, z=0.0)<br>Coverage 0.8977 vs 0.8977 (0, z=0.0)<br>Width 0.2646 vs 0.2646 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Split+RF` | `MAPIE_Split+RF` | 3 | IntervalScore 1.1936 vs 1.1936 (0, z=0.0)<br>Coverage 0.8932 vs 0.8932 (0, z=0.0)<br>Width 0.2828 vs 0.2828 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Split+RF` | `crepes_Split+RF` | 3 | IntervalScore 1.1936 vs 1.1936 (0, z=0.0)<br>Coverage 0.8932 vs 0.8932 (0, z=0.0)<br>Width 0.2828 vs 0.2828 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Split+TabularMLP` | `MAPIE_Split+TabularMLP` | 3 | IntervalScore 1.0870 vs 1.0870 (0, z=0.0)<br>Coverage 0.8963 vs 0.8963 (0, z=0.0)<br>Width 0.1370 vs 0.1370 (0, z=0.0) | - | exact |
| `uq_layer` | insurance_claims | `tr_Split+TabularMLP` | `crepes_Split+TabularMLP` | 3 | IntervalScore 1.0870 vs 1.0870 (0, z=0.0)<br>Coverage 0.8963 vs 0.8963 (0, z=0.0)<br>Width 0.1370 vs 0.1370 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_CQR+DeepEnsK5` | `MAPIE_CQR+DeepEnsK5` | 3 | IntervalScore 0.6754 vs 0.6754 (0, z=0.0)<br>Coverage 0.9074 vs 0.9074 (0, z=0.0)<br>Width 0.4963 vs 0.4963 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_CQR+LightGBM` | `MAPIE_CQR+LightGBM` | 3 | IntervalScore 0.6902 vs 0.6902 (0, z=0.0)<br>Coverage 0.9124 vs 0.9124 (0, z=0.0)<br>Width 0.5563 vs 0.5563 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_CQR+RF` | `MAPIE_CQR+RF` | 3 | IntervalScore 0.6651 vs 0.6651 (0, z=0.0)<br>Coverage 0.9134 vs 0.9134 (0, z=0.0)<br>Width 0.5000 vs 0.5000 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_CQR+TabularMLP` | `MAPIE_CQR+TabularMLP` | 3 | IntervalScore 0.6451 vs 0.6451 (0, z=0.0)<br>Coverage 0.9036 vs 0.9036 (0, z=0.0)<br>Width 0.4742 vs 0.4742 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_CVPlus+LightGBM` | `MAPIE_CVPlus+LightGBM` | 3 | IntervalScore 0.6467 vs 0.6467 (0, z=0.0)<br>Coverage 0.9133 vs 0.9133 (0, z=0.0)<br>Width 0.4608 vs 0.4608 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_CVPlus+RF` | `MAPIE_CVPlus+RF` | 3 | IntervalScore 0.7159 vs 0.7159 (0, z=0.0)<br>Coverage 0.9158 vs 0.9158 (0, z=0.0)<br>Width 0.5139 vs 0.5139 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_JackknifePlus+LightGBM` | `MAPIE_JackknifePlus+LightGBM` | 0 | no common ok seeds | - | n/a |
| `uq_layer` | openml_year | `tr_Normalized+DeepEnsK5` | `crepes_Normalized+DeepEnsK5` | 3 | IntervalScore 0.6750 vs 0.6750 (0, z=0.0)<br>Coverage 0.9082 vs 0.9082 (0, z=0.0)<br>Width 0.4936 vs 0.4936 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Normalized+LightGBM` | `crepes_Normalized+LightGBM` | 3 | IntervalScore 0.6425 vs 0.6425 (0, z=0.0)<br>Coverage 0.9020 vs 0.9020 (0, z=0.0)<br>Width 0.4612 vs 0.4612 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Normalized+RF` | `crepes_Normalized+RF` | 3 | IntervalScore 0.6641 vs 0.6641 (0, z=0.0)<br>Coverage 0.9120 vs 0.9120 (0, z=0.0)<br>Width 0.4810 vs 0.4810 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Normalized+TabularMLP` | `crepes_Normalized+TabularMLP` | 3 | IntervalScore 0.6447 vs 0.6447 (0, z=0.0)<br>Coverage 0.9028 vs 0.9028 (0, z=0.0)<br>Width 0.4718 vs 0.4718 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Split+DeepEnsK5` | `MAPIE_Split+DeepEnsK5` | 3 | IntervalScore 0.7320 vs 0.7320 (0, z=0.0)<br>Coverage 0.9064 vs 0.9064 (0, z=0.0)<br>Width 0.5143 vs 0.5143 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Split+DeepEnsK5` | `crepes_Split+DeepEnsK5` | 3 | IntervalScore 0.7320 vs 0.7320 (0, z=0.0)<br>Coverage 0.9064 vs 0.9064 (0, z=0.0)<br>Width 0.5143 vs 0.5143 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Split+LightGBM` | `MAPIE_Split+LightGBM` | 3 | IntervalScore 0.6693 vs 0.6693 (0, z=0.0)<br>Coverage 0.9044 vs 0.9044 (0, z=0.0)<br>Width 0.4565 vs 0.4565 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Split+LightGBM` | `crepes_Split+LightGBM` | 3 | IntervalScore 0.6693 vs 0.6693 (0, z=0.0)<br>Coverage 0.9044 vs 0.9044 (0, z=0.0)<br>Width 0.4565 vs 0.4565 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Split+RF` | `MAPIE_Split+RF` | 3 | IntervalScore 0.7343 vs 0.7343 (0, z=0.0)<br>Coverage 0.9132 vs 0.9132 (0, z=0.0)<br>Width 0.5213 vs 0.5213 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Split+RF` | `crepes_Split+RF` | 3 | IntervalScore 0.7343 vs 0.7343 (0, z=0.0)<br>Coverage 0.9132 vs 0.9132 (0, z=0.0)<br>Width 0.5213 vs 0.5213 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Split+TabularMLP` | `MAPIE_Split+TabularMLP` | 3 | IntervalScore 0.7202 vs 0.7202 (0, z=0.0)<br>Coverage 0.9088 vs 0.9088 (0, z=0.0)<br>Width 0.5042 vs 0.5042 (0, z=0.0) | - | exact |
| `uq_layer` | openml_year | `tr_Split+TabularMLP` | `crepes_Split+TabularMLP` | 3 | IntervalScore 0.7202 vs 0.7202 (0, z=0.0)<br>Coverage 0.9088 vs 0.9088 (0, z=0.0)<br>Width 0.5042 vs 0.5042 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_CQR+DeepEnsK5` | `MAPIE_CQR+DeepEnsK5` | 3 | IntervalScore 1.5803 vs 1.5803 (0, z=0.0)<br>Coverage 0.9222 vs 0.9222 (0, z=0.0)<br>Width 1.2960 vs 1.2960 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_CQR+LightGBM` | `MAPIE_CQR+LightGBM` | 3 | IntervalScore 1.7357 vs 1.7357 (0, z=0.0)<br>Coverage 0.9067 vs 0.9067 (0, z=0.0)<br>Width 1.4217 vs 1.4217 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_CQR+RF` | `MAPIE_CQR+RF` | 3 | IntervalScore 1.7054 vs 1.7054 (0, z=0.0)<br>Coverage 0.9167 vs 0.9167 (0, z=0.0)<br>Width 1.4047 vs 1.4047 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_CQR+TabularMLP` | `MAPIE_CQR+TabularMLP` | 3 | IntervalScore 1.4205 vs 1.4205 (0, z=0.0)<br>Coverage 0.9267 vs 0.9267 (0, z=0.0)<br>Width 1.2139 vs 1.2139 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_CVPlus+LightGBM` | `MAPIE_CVPlus+LightGBM` | 3 | IntervalScore 1.6963 vs 1.6963 (0, z=0.0)<br>Coverage 0.9256 vs 0.9256 (0, z=0.0)<br>Width 1.3964 vs 1.3964 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_CVPlus+RF` | `MAPIE_CVPlus+RF` | 3 | IntervalScore 1.7566 vs 1.7566 (0, z=0.0)<br>Coverage 0.9294 vs 0.9294 (0, z=0.0)<br>Width 1.4189 vs 1.4189 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_JackknifePlus+LightGBM` | `MAPIE_JackknifePlus+LightGBM` | 0 | no common ok seeds | - | n/a |
| `uq_layer` | synthetic_hetero | `tr_Normalized+DeepEnsK5` | `crepes_Normalized+DeepEnsK5` | 3 | IntervalScore 1.5817 vs 1.5817 (0, z=0.0)<br>Coverage 0.9189 vs 0.9189 (0, z=0.0)<br>Width 1.2932 vs 1.2932 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Normalized+LightGBM` | `crepes_Normalized+LightGBM` | 3 | IntervalScore 1.7236 vs 1.7236 (0, z=0.0)<br>Coverage 0.8933 vs 0.8933 (0, z=0.0)<br>Width 1.3770 vs 1.3770 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Normalized+RF` | `crepes_Normalized+RF` | 3 | IntervalScore 1.7063 vs 1.7063 (0, z=0.0)<br>Coverage 0.9183 vs 0.9183 (0, z=0.0)<br>Width 1.4095 vs 1.4095 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Normalized+TabularMLP` | `crepes_Normalized+TabularMLP` | 3 | IntervalScore 1.4250 vs 1.4250 (0, z=0.0)<br>Coverage 0.9244 vs 0.9244 (0, z=0.0)<br>Width 1.2217 vs 1.2217 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Split+DeepEnsK5` | `MAPIE_Split+DeepEnsK5` | 3 | IntervalScore 1.6622 vs 1.6622 (0, z=0.0)<br>Coverage 0.9206 vs 0.9206 (0, z=0.0)<br>Width 1.3324 vs 1.3324 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Split+DeepEnsK5` | `crepes_Split+DeepEnsK5` | 3 | IntervalScore 1.6622 vs 1.6622 (0, z=0.0)<br>Coverage 0.9206 vs 0.9206 (0, z=0.0)<br>Width 1.3324 vs 1.3324 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Split+LightGBM` | `MAPIE_Split+LightGBM` | 3 | IntervalScore 1.7697 vs 1.7697 (0, z=0.0)<br>Coverage 0.9039 vs 0.9039 (0, z=0.0)<br>Width 1.3281 vs 1.3281 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Split+LightGBM` | `crepes_Split+LightGBM` | 3 | IntervalScore 1.7697 vs 1.7697 (0, z=0.0)<br>Coverage 0.9039 vs 0.9039 (0, z=0.0)<br>Width 1.3281 vs 1.3281 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Split+RF` | `MAPIE_Split+RF` | 3 | IntervalScore 1.8105 vs 1.8105 (0, z=0.0)<br>Coverage 0.9233 vs 0.9233 (0, z=0.0)<br>Width 1.4446 vs 1.4446 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Split+RF` | `crepes_Split+RF` | 3 | IntervalScore 1.8105 vs 1.8105 (0, z=0.0)<br>Coverage 0.9233 vs 0.9233 (0, z=0.0)<br>Width 1.4446 vs 1.4446 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Split+TabularMLP` | `MAPIE_Split+TabularMLP` | 3 | IntervalScore 1.6243 vs 1.6243 (0, z=0.0)<br>Coverage 0.9283 vs 0.9283 (0, z=0.0)<br>Width 1.3021 vs 1.3021 (0, z=0.0) | - | exact |
| `uq_layer` | synthetic_hetero | `tr_Split+TabularMLP` | `crepes_Split+TabularMLP` | 3 | IntervalScore 1.6243 vs 1.6243 (0, z=0.0)<br>Coverage 0.9283 vs 0.9283 (0, z=0.0)<br>Width 1.3021 vs 1.3021 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_CQR+DeepEnsK5` | `MAPIE_CQR+DeepEnsK5` | 3 | IntervalScore 2.2589 vs 2.2589 (0, z=0.0)<br>Coverage 0.8961 vs 0.8961 (0, z=0.0)<br>Width 1.7427 vs 1.7427 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_CQR+LightGBM` | `MAPIE_CQR+LightGBM` | 3 | IntervalScore 1.9922 vs 1.9922 (0, z=0.0)<br>Coverage 0.8947 vs 0.8947 (0, z=0.0)<br>Width 1.5566 vs 1.5566 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_CQR+RF` | `MAPIE_CQR+RF` | 3 | IntervalScore 1.9659 vs 1.9659 (0, z=0.0)<br>Coverage 0.9004 vs 0.9004 (0, z=0.0)<br>Width 1.5481 vs 1.5481 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_CQR+TabularMLP` | `MAPIE_CQR+TabularMLP` | 3 | IntervalScore 1.9871 vs 1.9871 (0, z=0.0)<br>Coverage 0.8922 vs 0.8922 (0, z=0.0)<br>Width 1.5189 vs 1.5189 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_CVPlus+LightGBM` | `MAPIE_CVPlus+LightGBM` | 3 | IntervalScore 2.0325 vs 2.0325 (0, z=0.0)<br>Coverage 0.8979 vs 0.8979 (0, z=0.0)<br>Width 1.5519 vs 1.5519 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_CVPlus+RF` | `MAPIE_CVPlus+RF` | 3 | IntervalScore 2.0462 vs 2.0462 (0, z=0.0)<br>Coverage 0.8994 vs 0.8994 (0, z=0.0)<br>Width 1.5541 vs 1.5541 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_JackknifePlus+LightGBM` | `MAPIE_JackknifePlus+LightGBM` | 0 | no common ok seeds | - | n/a |
| `uq_layer` | tabred_cooking | `tr_Normalized+DeepEnsK5` | `crepes_Normalized+DeepEnsK5` | 3 | IntervalScore 2.3223 vs 2.3223 (0, z=0.0)<br>Coverage 0.8969 vs 0.8969 (0, z=0.0)<br>Width 1.8128 vs 1.8128 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Normalized+LightGBM` | `crepes_Normalized+LightGBM` | 3 | IntervalScore 1.9821 vs 1.9821 (0, z=0.0)<br>Coverage 0.8922 vs 0.8922 (0, z=0.0)<br>Width 1.5334 vs 1.5334 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Normalized+RF` | `crepes_Normalized+RF` | 3 | IntervalScore 1.9732 vs 1.9732 (0, z=0.0)<br>Coverage 0.8969 vs 0.8969 (0, z=0.0)<br>Width 1.5304 vs 1.5304 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Normalized+TabularMLP` | `crepes_Normalized+TabularMLP` | 3 | IntervalScore 1.9874 vs 1.9874 (0, z=0.0)<br>Coverage 0.8932 vs 0.8932 (0, z=0.0)<br>Width 1.5200 vs 1.5200 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Split+DeepEnsK5` | `MAPIE_Split+DeepEnsK5` | 3 | IntervalScore 2.2758 vs 2.2758 (0, z=0.0)<br>Coverage 0.8939 vs 0.8939 (0, z=0.0)<br>Width 1.7177 vs 1.7177 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Split+DeepEnsK5` | `crepes_Split+DeepEnsK5` | 3 | IntervalScore 2.2758 vs 2.2758 (0, z=0.0)<br>Coverage 0.8939 vs 0.8939 (0, z=0.0)<br>Width 1.7177 vs 1.7177 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Split+LightGBM` | `MAPIE_Split+LightGBM` | 3 | IntervalScore 2.0585 vs 2.0585 (0, z=0.0)<br>Coverage 0.8950 vs 0.8950 (0, z=0.0)<br>Width 1.5623 vs 1.5623 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Split+LightGBM` | `crepes_Split+LightGBM` | 3 | IntervalScore 2.0585 vs 2.0585 (0, z=0.0)<br>Coverage 0.8950 vs 0.8950 (0, z=0.0)<br>Width 1.5623 vs 1.5623 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Split+RF` | `MAPIE_Split+RF` | 3 | IntervalScore 2.0642 vs 2.0642 (0, z=0.0)<br>Coverage 0.9034 vs 0.9034 (0, z=0.0)<br>Width 1.5862 vs 1.5862 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Split+RF` | `crepes_Split+RF` | 3 | IntervalScore 2.0642 vs 2.0642 (0, z=0.0)<br>Coverage 0.9034 vs 0.9034 (0, z=0.0)<br>Width 1.5862 vs 1.5862 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Split+TabularMLP` | `MAPIE_Split+TabularMLP` | 3 | IntervalScore 2.0662 vs 2.0662 (0, z=0.0)<br>Coverage 0.8963 vs 0.8963 (0, z=0.0)<br>Width 1.5750 vs 1.5750 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_cooking | `tr_Split+TabularMLP` | `crepes_Split+TabularMLP` | 3 | IntervalScore 2.0662 vs 2.0662 (0, z=0.0)<br>Coverage 0.8963 vs 0.8963 (0, z=0.0)<br>Width 1.5750 vs 1.5750 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_CQR+DeepEnsK5` | `MAPIE_CQR+DeepEnsK5` | 3 | IntervalScore 2.6244 vs 2.6244 (0, z=0.0)<br>Coverage 0.8897 vs 0.8897 (0, z=0.0)<br>Width 2.0021 vs 2.0021 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_CQR+LightGBM` | `MAPIE_CQR+LightGBM` | 3 | IntervalScore 2.2659 vs 2.2659 (0, z=0.0)<br>Coverage 0.9079 vs 0.9079 (0, z=0.0)<br>Width 1.9063 vs 1.9063 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_CQR+RF` | `MAPIE_CQR+RF` | 3 | IntervalScore 2.2637 vs 2.2637 (0, z=0.0)<br>Coverage 0.8881 vs 0.8881 (0, z=0.0)<br>Width 1.7619 vs 1.7619 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_CQR+TabularMLP` | `MAPIE_CQR+TabularMLP` | 3 | IntervalScore 2.2563 vs 2.2563 (0, z=0.0)<br>Coverage 0.8951 vs 0.8951 (0, z=0.0)<br>Width 1.7612 vs 1.7612 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_CVPlus+LightGBM` | `MAPIE_CVPlus+LightGBM` | 3 | IntervalScore 2.2827 vs 2.2827 (0, z=0.0)<br>Coverage 0.8974 vs 0.8974 (0, z=0.0)<br>Width 1.7879 vs 1.7879 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_CVPlus+RF` | `MAPIE_CVPlus+RF` | 3 | IntervalScore 2.3205 vs 2.3205 (0, z=0.0)<br>Coverage 0.8946 vs 0.8946 (0, z=0.0)<br>Width 1.7820 vs 1.7820 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_JackknifePlus+LightGBM` | `MAPIE_JackknifePlus+LightGBM` | 0 | no common ok seeds | - | n/a |
| `uq_layer` | tabred_delivery | `tr_Normalized+DeepEnsK5` | `crepes_Normalized+DeepEnsK5` | 3 | IntervalScore 2.7527 vs 2.7527 (0, z=0.0)<br>Coverage 0.8899 vs 0.8899 (0, z=0.0)<br>Width 2.1034 vs 2.1034 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Normalized+LightGBM` | `crepes_Normalized+LightGBM` | 3 | IntervalScore 2.2773 vs 2.2773 (0, z=0.0)<br>Coverage 0.9009 vs 0.9009 (0, z=0.0)<br>Width 1.8198 vs 1.8198 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Normalized+RF` | `crepes_Normalized+RF` | 3 | IntervalScore 2.3120 vs 2.3120 (0, z=0.0)<br>Coverage 0.8937 vs 0.8937 (0, z=0.0)<br>Width 1.7632 vs 1.7632 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Normalized+TabularMLP` | `crepes_Normalized+TabularMLP` | 3 | IntervalScore 2.2544 vs 2.2544 (0, z=0.0)<br>Coverage 0.8946 vs 0.8946 (0, z=0.0)<br>Width 1.7581 vs 1.7581 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Split+DeepEnsK5` | `MAPIE_Split+DeepEnsK5` | 3 | IntervalScore 2.5846 vs 2.5846 (0, z=0.0)<br>Coverage 0.8961 vs 0.8961 (0, z=0.0)<br>Width 2.0031 vs 2.0031 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Split+DeepEnsK5` | `crepes_Split+DeepEnsK5` | 3 | IntervalScore 2.5846 vs 2.5846 (0, z=0.0)<br>Coverage 0.8961 vs 0.8961 (0, z=0.0)<br>Width 2.0031 vs 2.0031 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Split+LightGBM` | `MAPIE_Split+LightGBM` | 3 | IntervalScore 2.3189 vs 2.3189 (0, z=0.0)<br>Coverage 0.8882 vs 0.8882 (0, z=0.0)<br>Width 1.7753 vs 1.7753 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Split+LightGBM` | `crepes_Split+LightGBM` | 3 | IntervalScore 2.3189 vs 2.3189 (0, z=0.0)<br>Coverage 0.8882 vs 0.8882 (0, z=0.0)<br>Width 1.7753 vs 1.7753 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Split+RF` | `MAPIE_Split+RF` | 3 | IntervalScore 2.3388 vs 2.3388 (0, z=0.0)<br>Coverage 0.8907 vs 0.8907 (0, z=0.0)<br>Width 1.7715 vs 1.7715 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Split+RF` | `crepes_Split+RF` | 3 | IntervalScore 2.3388 vs 2.3388 (0, z=0.0)<br>Coverage 0.8907 vs 0.8907 (0, z=0.0)<br>Width 1.7715 vs 1.7715 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Split+TabularMLP` | `MAPIE_Split+TabularMLP` | 3 | IntervalScore 2.2917 vs 2.2917 (0, z=0.0)<br>Coverage 0.8891 vs 0.8891 (0, z=0.0)<br>Width 1.7686 vs 1.7686 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_delivery | `tr_Split+TabularMLP` | `crepes_Split+TabularMLP` | 3 | IntervalScore 2.2917 vs 2.2917 (0, z=0.0)<br>Coverage 0.8891 vs 0.8891 (0, z=0.0)<br>Width 1.7686 vs 1.7686 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_CQR+DeepEnsK5` | `MAPIE_CQR+DeepEnsK5` | 3 | IntervalScore 7.5074 vs 7.5074 (0, z=0.0)<br>Coverage 0.9089 vs 0.9089 (0, z=0.0)<br>Width 5.6204 vs 5.6204 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_CQR+LightGBM` | `MAPIE_CQR+LightGBM` | 3 | IntervalScore 7.5487 vs 7.5487 (0, z=0.0)<br>Coverage 0.9092 vs 0.9092 (0, z=0.0)<br>Width 5.8276 vs 5.8276 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_CQR+RF` | `MAPIE_CQR+RF` | 3 | IntervalScore 7.3896 vs 7.3896 (0, z=0.0)<br>Coverage 0.9028 vs 0.9028 (0, z=0.0)<br>Width 5.5364 vs 5.5364 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_CQR+TabularMLP` | `MAPIE_CQR+TabularMLP` | 3 | IntervalScore 7.1083 vs 7.1083 (0, z=0.0)<br>Coverage 0.9082 vs 0.9082 (0, z=0.0)<br>Width 5.4131 vs 5.4131 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_CVPlus+LightGBM` | `MAPIE_CVPlus+LightGBM` | 3 | IntervalScore 7.3757 vs 7.3757 (0, z=0.0)<br>Coverage 0.9127 vs 0.9127 (0, z=0.0)<br>Width 5.4030 vs 5.4030 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_CVPlus+RF` | `MAPIE_CVPlus+RF` | 3 | IntervalScore 7.6916 vs 7.6916 (0, z=0.0)<br>Coverage 0.9042 vs 0.9042 (0, z=0.0)<br>Width 5.5701 vs 5.5701 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_JackknifePlus+LightGBM` | `MAPIE_JackknifePlus+LightGBM` | 0 | no common ok seeds | - | n/a |
| `uq_layer` | tabred_weather | `tr_Normalized+DeepEnsK5` | `crepes_Normalized+DeepEnsK5` | 3 | IntervalScore 7.5652 vs 7.5652 (0, z=0.0)<br>Coverage 0.9059 vs 0.9059 (0, z=0.0)<br>Width 5.6958 vs 5.6958 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Normalized+LightGBM` | `crepes_Normalized+LightGBM` | 3 | IntervalScore 7.2129 vs 7.2129 (0, z=0.0)<br>Coverage 0.9099 vs 0.9099 (0, z=0.0)<br>Width 5.4530 vs 5.4530 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Normalized+RF` | `crepes_Normalized+RF` | 3 | IntervalScore 7.3362 vs 7.3362 (0, z=0.0)<br>Coverage 0.9070 vs 0.9070 (0, z=0.0)<br>Width 5.5231 vs 5.5231 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Normalized+TabularMLP` | `crepes_Normalized+TabularMLP` | 3 | IntervalScore 7.1016 vs 7.1016 (0, z=0.0)<br>Coverage 0.9091 vs 0.9091 (0, z=0.0)<br>Width 5.3846 vs 5.3846 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Split+DeepEnsK5` | `MAPIE_Split+DeepEnsK5` | 3 | IntervalScore 7.7778 vs 7.7778 (0, z=0.0)<br>Coverage 0.9078 vs 0.9078 (0, z=0.0)<br>Width 5.6137 vs 5.6137 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Split+DeepEnsK5` | `crepes_Split+DeepEnsK5` | 3 | IntervalScore 7.7778 vs 7.7778 (0, z=0.0)<br>Coverage 0.9078 vs 0.9078 (0, z=0.0)<br>Width 5.6137 vs 5.6137 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Split+LightGBM` | `MAPIE_Split+LightGBM` | 3 | IntervalScore 7.5120 vs 7.5120 (0, z=0.0)<br>Coverage 0.9072 vs 0.9072 (0, z=0.0)<br>Width 5.3710 vs 5.3710 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Split+LightGBM` | `crepes_Split+LightGBM` | 3 | IntervalScore 7.5120 vs 7.5120 (0, z=0.0)<br>Coverage 0.9072 vs 0.9072 (0, z=0.0)<br>Width 5.3710 vs 5.3710 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Split+RF` | `MAPIE_Split+RF` | 3 | IntervalScore 7.7544 vs 7.7544 (0, z=0.0)<br>Coverage 0.9040 vs 0.9040 (0, z=0.0)<br>Width 5.5929 vs 5.5929 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Split+RF` | `crepes_Split+RF` | 3 | IntervalScore 7.7544 vs 7.7544 (0, z=0.0)<br>Coverage 0.9040 vs 0.9040 (0, z=0.0)<br>Width 5.5929 vs 5.5929 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Split+TabularMLP` | `MAPIE_Split+TabularMLP` | 3 | IntervalScore 7.5293 vs 7.5293 (0, z=0.0)<br>Coverage 0.9071 vs 0.9071 (0, z=0.0)<br>Width 5.4590 vs 5.4590 (0, z=0.0) | - | exact |
| `uq_layer` | tabred_weather | `tr_Split+TabularMLP` | `crepes_Split+TabularMLP` | 3 | IntervalScore 7.5293 vs 7.5293 (0, z=0.0)<br>Coverage 0.9071 vs 0.9071 (0, z=0.0)<br>Width 5.4590 vs 5.4590 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_CQR+DeepEnsK5` | `MAPIE_CQR+DeepEnsK5` | 3 | IntervalScore 412.6 vs 412.6 (0, z=0.0)<br>Coverage 0.8992 vs 0.8992 (0, z=0.0)<br>Width 265.0 vs 265.0 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_CQR+LightGBM` | `MAPIE_CQR+LightGBM` | 3 | IntervalScore 365.3 vs 365.3 (0, z=0.0)<br>Coverage 0.9008 vs 0.9008 (0, z=0.0)<br>Width 231.7 vs 231.7 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_CQR+RF` | `MAPIE_CQR+RF` | 3 | IntervalScore 359.0 vs 359.0 (0, z=0.0)<br>Coverage 0.9008 vs 0.9008 (0, z=0.0)<br>Width 226.4 vs 226.4 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_CQR+TabularMLP` | `MAPIE_CQR+TabularMLP` | 3 | IntervalScore 387.6 vs 387.6 (0, z=0.0)<br>Coverage 0.8983 vs 0.8983 (0, z=0.0)<br>Width 258.0 vs 258.0 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_CVPlus+LightGBM` | `MAPIE_CVPlus+LightGBM` | 3 | IntervalScore 436.7 vs 436.7 (0, z=0.0)<br>Coverage 0.9080 vs 0.9080 (0, z=0.0)<br>Width 237.6 vs 237.6 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_CVPlus+RF` | `MAPIE_CVPlus+RF` | 3 | IntervalScore 427.0 vs 427.0 (0, z=0.0)<br>Coverage 0.9100 vs 0.9100 (0, z=0.0)<br>Width 240.8 vs 240.8 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_JackknifePlus+LightGBM` | `MAPIE_JackknifePlus+LightGBM` | 0 | no common ok seeds | - | n/a |
| `uq_layer` | zurich_delays | `tr_Normalized+DeepEnsK5` | `crepes_Normalized+DeepEnsK5` | 3 | IntervalScore 407.0 vs 407.0 (0, z=0.0)<br>Coverage 0.8996 vs 0.8996 (0, z=0.0)<br>Width 253.4 vs 253.4 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Normalized+LightGBM` | `crepes_Normalized+LightGBM` | 3 | IntervalScore 394.1 vs 394.1 (0, z=0.0)<br>Coverage 0.9005 vs 0.9005 (0, z=0.0)<br>Width 224.3 vs 224.3 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Normalized+RF` | `crepes_Normalized+RF` | 3 | IntervalScore 376.5 vs 376.5 (0, z=0.0)<br>Coverage 0.9003 vs 0.9003 (0, z=0.0)<br>Width 241.2 vs 241.2 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Normalized+TabularMLP` | `crepes_Normalized+TabularMLP` | 3 | IntervalScore 382.2 vs 382.2 (0, z=0.0)<br>Coverage 0.8996 vs 0.8996 (0, z=0.0)<br>Width 245.3 vs 245.3 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Split+DeepEnsK5` | `MAPIE_Split+DeepEnsK5` | 3 | IntervalScore 456.2 vs 456.2 (0, z=0.0)<br>Coverage 0.9014 vs 0.9014 (0, z=0.0)<br>Width 233.1 vs 233.1 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Split+DeepEnsK5` | `crepes_Split+DeepEnsK5` | 3 | IntervalScore 456.2 vs 456.2 (0, z=0.0)<br>Coverage 0.9014 vs 0.9014 (0, z=0.0)<br>Width 233.1 vs 233.1 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Split+LightGBM` | `MAPIE_Split+LightGBM` | 3 | IntervalScore 457.3 vs 457.3 (0, z=0.0)<br>Coverage 0.9008 vs 0.9008 (0, z=0.0)<br>Width 232.9 vs 232.9 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Split+LightGBM` | `crepes_Split+LightGBM` | 3 | IntervalScore 457.3 vs 457.3 (0, z=0.0)<br>Coverage 0.9008 vs 0.9008 (0, z=0.0)<br>Width 232.9 vs 232.9 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Split+RF` | `MAPIE_Split+RF` | 3 | IntervalScore 450.5 vs 450.5 (0, z=0.0)<br>Coverage 0.9018 vs 0.9018 (0, z=0.0)<br>Width 237.4 vs 237.4 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Split+RF` | `crepes_Split+RF` | 3 | IntervalScore 450.5 vs 450.5 (0, z=0.0)<br>Coverage 0.9018 vs 0.9018 (0, z=0.0)<br>Width 237.4 vs 237.4 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Split+TabularMLP` | `MAPIE_Split+TabularMLP` | 3 | IntervalScore 443.7 vs 443.7 (0, z=0.0)<br>Coverage 0.9028 vs 0.9028 (0, z=0.0)<br>Width 232.1 vs 232.1 (0, z=0.0) | - | exact |
| `uq_layer` | zurich_delays | `tr_Split+TabularMLP` | `crepes_Split+TabularMLP` | 3 | IntervalScore 443.7 vs 443.7 (0, z=0.0)<br>Coverage 0.9028 vs 0.9028 (0, z=0.0)<br>Width 232.1 vs 232.1 (0, z=0.0) | - | exact |

### Matched pairs

| Suite | Dataset | torchregress | Comparator | Seeds | Key metrics: tr vs comp (rel. diff, z) | Within noise | Verdict |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| `conformal` | california_housing | `Split+MLP` | `MAPIE_Split+MLP` | 3 | IntervalScore 2.8665 vs 2.8677 (0.04%, z=0.0)<br>Coverage 0.9104 vs 0.9124 (0.22%, z=0.5)<br>Width 1.8583 vs 1.8759 (0.94%, z=0.4) | yes | close |
| `conformal` | diabetes | `Split+MLP` | `MAPIE_Split+MLP` | 3 | IntervalScore 234.8 vs 236.3 (0.61%, z=0.2)<br>Coverage 0.9524 vs 0.9549 (0.26%, z=0.2)<br>Width 213.6 vs 218.1 (2.09%, z=0.7) | yes | **differs** |
| `conformal` | house_sales | `Split+MLP` | `MAPIE_Split+MLP` | 3 | IntervalScore 6.137e+05 vs 6.410e+05 (4.26%, z=3.9)<br>Coverage 0.8964 vs 0.8987 (0.25%, z=0.4)<br>Width 3.179e+05 vs 3.407e+05 (6.71%, z=2.9) | no | **differs** |
| `conformal` | insurance_claims | `Split+MLP` | `MAPIE_Split+MLP` | 3 | IntervalScore 1.4887 vs 1.3978 (6.11%, z=8.4)<br>Coverage 0.8964 vs 0.8997 (0.36%, z=0.5)<br>Width 0.5489 vs 0.5432 (1.03%, z=0.2) | no | **differs** |
| `conformal` | openml_year | `Split+MLP` | `MAPIE_Split+MLP` | 3 | IntervalScore 0.7409 vs 0.7552 (1.89%, z=1.7)<br>Coverage 0.9089 vs 0.8974 (1.26%, z=1.9)<br>Width 0.5163 vs 0.5032 (2.54%, z=0.8) | yes | **differs** |
| `conformal` | synthetic_hetero | `Split+MLP` | `MAPIE_Split+MLP` | 3 | IntervalScore 1.7036 vs 1.7648 (3.47%, z=1.8)<br>Coverage 0.9211 vs 0.9156 (0.60%, z=0.5)<br>Width 1.3725 vs 1.4041 (2.25%, z=0.6) | yes | **differs** |
| `conformal` | tabred_cooking | `Split+MLP` | `MAPIE_Split+MLP` | 3 | IntervalScore 2.6804 vs 2.6848 (0.16%, z=0.2)<br>Coverage 0.9023 vs 0.9017 (0.07%, z=0.1)<br>Width 2.0541 vs 2.0816 (1.32%, z=1.4) | yes | close |
| `conformal` | tabred_delivery | `Split+MLP` | `MAPIE_Split+MLP` | 3 | IntervalScore 3.1124 vs 3.1283 (0.51%, z=0.4)<br>Coverage 0.9006 vs 0.8989 (0.19%, z=0.3)<br>Width 2.4113 vs 2.4362 (1.02%, z=1.4) | yes | close |
| `conformal` | tabred_weather | `Split+MLP` | `MAPIE_Split+MLP` | 3 | IntervalScore 9.0419 vs 9.5814 (5.63%, z=3.9)<br>Coverage 0.9050 vs 0.8959 (1.01%, z=6.7)<br>Width 6.6293 vs 6.9041 (3.98%, z=1.8) | no | **differs** |
| `conformal` | zurich_delays | `Split+MLP` | `MAPIE_Split+MLP` | 3 | IntervalScore 485.3 vs 480.5 (1.00%, z=0.8)<br>Coverage 0.9031 vs 0.9035 (0.05%, z=0.3)<br>Width 249.9 vs 251.0 (0.45%, z=0.6) | yes | close |
| `density_parity` | california_housing | `torchregress_MAF` | `sbi_MAF` | 3 | NLL 1.5171 vs 0.6135 (90.36%, z=1.1)<br>CRPS 0.2489 vs 0.2513 (0.95%, z=0.3)<br>Cov90_marg 0.9037 vs 0.8933 (1.14%, z=0.8) | yes | **differs** |
| `density_parity` | california_housing | `torchregress_MAF` | `sbi_zuko_MAF` | 3 | NLL 1.5171 vs 1.8541 (18.17%, z=0.2)<br>CRPS 0.2489 vs 0.2459 (1.19%, z=0.5)<br>Cov90_marg 0.9037 vs 0.8750 (3.17%, z=1.5) | yes | **differs** |
| `density_parity` | california_housing | `torchregress_MDN` | `sbi_MDN` | 3 | NLL 0.3228 vs 0.3103 (1.25%, z=0.7)<br>CRPS 0.2351 vs 0.2356 (0.25%, z=0.1)<br>Cov90_marg 0.8800 vs 0.8837 (0.41%, z=0.8) | yes | close |
| `density_parity` | california_housing | `torchregress_NSF` | `sbi_NSF` | 3 | NLL 0.2378 vs 0.1581 (7.97%, z=2.5)<br>CRPS 0.2369 vs 0.2358 (0.45%, z=0.1)<br>Cov90_marg 0.8767 vs 0.8800 (0.38%, z=0.4) | no | **differs** |
| `density_parity` | california_housing | `torchregress_NSF` | `sbi_zuko_NSF` | 3 | NLL 0.2378 vs 0.2313 (0.65%, z=0.2)<br>CRPS 0.2369 vs 0.2400 (1.32%, z=0.6)<br>Cov90_marg 0.8767 vs 0.8803 (0.42%, z=0.4) | yes | close |
| `density_parity` | energy_efficiency | `torchregress_MAF` | `sbi_MAF` | 3 | NLL -1.4311 vs -1.3077 (9.43%, z=1.0)<br>CRPS 0.1141 vs 0.1181 (3.41%, z=0.3)<br>Cov90_marg 0.8478 vs 0.9156 (7.41%, z=3.6) | no | **differs** |
| `density_parity` | energy_efficiency | `torchregress_MAF` | `sbi_zuko_MAF` | 3 | NLL -1.4311 vs -1.4867 (3.74%, z=0.6)<br>CRPS 0.1141 vs 0.1048 (8.14%, z=1.0)<br>Cov90_marg 0.8478 vs 0.8636 (1.84%, z=0.7) | yes | **differs** |
| `density_parity` | energy_efficiency | `torchregress_MDN` | `sbi_MDN` | 3 | NLL -0.7622 vs -1.3326 (42.81%, z=4.6)<br>CRPS 0.1543 vs 0.1531 (0.78%, z=0.1)<br>Cov90_marg 0.8997 vs 0.9300 (3.26%, z=2.0) | no | **differs** |
| `density_parity` | energy_efficiency | `torchregress_NSF` | `sbi_NSF` | 3 | NLL -1.3509 vs -1.3605 (0.70%, z=0.0)<br>CRPS 0.1397 vs 0.1528 (8.60%, z=1.4)<br>Cov90_marg 0.8853 vs 0.8838 (0.16%, z=0.1) | yes | **differs** |
| `density_parity` | energy_efficiency | `torchregress_NSF` | `sbi_zuko_NSF` | 3 | NLL -1.3509 vs -1.3614 (0.77%, z=0.0)<br>CRPS 0.1397 vs 0.1452 (3.80%, z=0.6)<br>Cov90_marg 0.8853 vs 0.8911 (0.65%, z=0.4) | yes | **differs** |
| `density_parity` | mt_enb | `torchregress_MAF` | `sbi_MAF` | 3 | NLL -1.7258 vs -1.2897 (33.81%, z=4.1)<br>CRPS 0.1027 vs 0.1242 (17.28%, z=2.4)<br>Cov90_marg 0.8788 vs 0.9026 (2.64%, z=1.2) | no | **differs** |
| `density_parity` | mt_enb | `torchregress_MAF` | `sbi_zuko_MAF` | 3 | NLL -1.7258 vs -1.7314 (0.33%, z=0.0)<br>CRPS 0.1027 vs 0.0959 (6.64%, z=1.0)<br>Cov90_marg 0.8788 vs 0.8647 (1.60%, z=0.6) | yes | **differs** |
| `density_parity` | mt_enb | `torchregress_MDN` | `sbi_MDN` | 3 | NLL -0.8535 vs -1.2402 (31.18%, z=1.1)<br>CRPS 0.1539 vs 0.1538 (0.04%, z=0.0)<br>Cov90_marg 0.9026 vs 0.9048 (0.24%, z=0.4) | yes | **differs** |
| `density_parity` | mt_enb | `torchregress_NSF` | `sbi_NSF` | 3 | NLL -1.7311 vs -1.5248 (13.53%, z=1.0)<br>CRPS 0.1224 vs 0.1501 (18.47%, z=3.3)<br>Cov90_marg 0.9069 vs 0.9058 (0.12%, z=0.1) | no | **differs** |
| `density_parity` | mt_enb | `torchregress_NSF` | `sbi_zuko_NSF` | 3 | NLL -1.7311 vs -1.4338 (20.73%, z=1.6)<br>CRPS 0.1224 vs 0.1447 (15.42%, z=3.5)<br>Cov90_marg 0.9069 vs 0.8929 (1.55%, z=0.6) | no | **differs** |
| `density_parity` | mt_jura | `torchregress_MAF` | `sbi_MAF` | 3 | NLL 2.0979 vs 2.0941 (0.18%, z=0.0)<br>CRPS 0.2860 vs 0.3190 (10.35%, z=1.1)<br>Cov90_marg 0.8750 vs 0.8951 (2.24%, z=0.9) | yes | **differs** |
| `density_parity` | mt_jura | `torchregress_MAF` | `sbi_zuko_MAF` | 3 | NLL 2.0979 vs 1.9892 (5.47%, z=0.6)<br>CRPS 0.2860 vs 0.2874 (0.48%, z=0.1)<br>Cov90_marg 0.8750 vs 0.8873 (1.39%, z=0.7) | yes | **differs** |
| `density_parity` | mt_jura | `torchregress_MDN` | `sbi_MDN` | 3 | NLL 1.9480 vs 1.9151 (1.72%, z=0.2)<br>CRPS 0.2917 vs 0.2802 (3.95%, z=0.9)<br>Cov90_marg 0.9059 vs 0.8611 (4.94%, z=2.9) | no | **differs** |
| `density_parity` | mt_jura | `torchregress_NSF` | `sbi_NSF` | 3 | NLL 1.6085 vs 1.7478 (7.97%, z=0.7)<br>CRPS 0.2670 vs 0.2802 (4.72%, z=0.7)<br>Cov90_marg 0.8519 vs 0.8580 (0.72%, z=0.5) | yes | **differs** |
| `density_parity` | mt_jura | `torchregress_NSF` | `sbi_zuko_NSF` | 3 | NLL 1.6085 vs 1.6576 (2.96%, z=0.4)<br>CRPS 0.2670 vs 0.2701 (1.15%, z=0.3)<br>Cov90_marg 0.8519 vs 0.8565 (0.54%, z=0.6) | yes | **differs** |
| `density_parity` | mt_rf1 | `torchregress_MAF` | `sbi_MAF` | 3 | NLL 35486.0 vs -7.1889 (493719.04%, z=1.0)<br>CRPS 0.1163 vs 0.1292 (9.99%, z=0.1)<br>Cov90_marg 0.8910 vs 0.9080 (1.87%, z=2.7) | no | **differs** |
| `density_parity` | mt_rf1 | `torchregress_MAF` | `sbi_zuko_MAF` | 3 | NLL 35486.0 vs 11116.2 (219.23%, z=0.7)<br>CRPS 0.1163 vs 0.1146 (1.42%, z=0.0)<br>Cov90_marg 0.8910 vs 0.9083 (1.90%, z=3.2) | no | **differs** |
| `density_parity` | mt_rf1 | `torchregress_MDN` | `sbi_MDN` | 3 | NLL 33.6979 vs 43.8466 (23.15%, z=0.3)<br>CRPS 0.1117 vs 0.1494 (25.22%, z=0.4)<br>Cov90_marg 0.9152 vs 0.9189 (0.40%, z=0.4) | yes | **differs** |
| `density_parity` | mt_rf1 | `torchregress_NSF` | `sbi_NSF` | 3 | NLL -15.0640 vs -13.8070 (9.10%, z=0.4)<br>CRPS 0.0385 vs 0.0522 (26.22%, z=0.5)<br>Cov90_marg 0.9013 vs 0.8859 (1.71%, z=1.7) | yes | **differs** |
| `density_parity` | mt_rf1 | `torchregress_NSF` | `sbi_zuko_NSF` | 3 | NLL -15.0640 vs -15.1936 (0.85%, z=0.1)<br>CRPS 0.0385 vs 0.0354 (8.04%, z=0.3)<br>Cov90_marg 0.9013 vs 0.9078 (0.72%, z=0.8) | yes | **differs** |
| `density_parity` | mt_scm20d | `torchregress_MAF` | `sbi_MAF` | 3 | NLL -1.4803 vs -1.5619 (5.23%, z=0.5)<br>CRPS 20043.6 vs 0.1909 (100.00%, z=1.7)<br>Cov90_marg 0.8690 vs 0.8581 (1.25%, z=2.4) | no | **differs** |
| `density_parity` | mt_scm20d | `torchregress_MAF` | `sbi_zuko_MAF` | 3 | NLL -1.4803 vs -0.9904 (48.99%, z=1.1)<br>CRPS 20043.6 vs 0.2018 (100.00%, z=1.7)<br>Cov90_marg 0.8690 vs 0.8645 (0.52%, z=0.6) | yes | **differs** |
| `density_parity` | mt_scm20d | `torchregress_MDN` | `sbi_MDN` | 3 | NLL 0.3321 vs -0.0138 (34.60%, z=1.1)<br>CRPS 0.2885 vs 0.2801 (2.92%, z=0.6)<br>Cov90_marg 0.8921 vs 0.8866 (0.62%, z=0.9) | yes | **differs** |
| `density_parity` | mt_scm20d | `torchregress_NSF` | `sbi_NSF` | 3 | NLL -2.5702 vs -2.2358 (14.96%, z=3.7)<br>CRPS 0.1925 vs 0.1797 (6.64%, z=3.3)<br>Cov90_marg 0.8669 vs 0.8459 (2.42%, z=2.3) | no | **differs** |
| `density_parity` | mt_scm20d | `torchregress_NSF` | `sbi_zuko_NSF` | 3 | NLL -2.5702 vs -2.2784 (12.81%, z=1.9)<br>CRPS 0.1925 vs 0.1877 (2.48%, z=1.0)<br>Cov90_marg 0.8669 vs 0.8590 (0.91%, z=1.6) | yes | **differs** |
| `density_parity` | pzflow_galaxies | `torchregress_MAF` | `sbi_MAF` | 3 | NLL -2.3702 vs -2.0366 (16.38%, z=2.7)<br>CRPS 0.0225 vs 0.0311 (27.62%, z=2.9)<br>Cov90_marg 0.9080 vs 0.9023 (0.62%, z=0.7) | no | **differs** |
| `density_parity` | pzflow_galaxies | `torchregress_MAF` | `sbi_zuko_MAF` | 3 | NLL -2.3702 vs -2.3600 (0.44%, z=0.1)<br>CRPS 0.0225 vs 0.0211 (6.22%, z=0.5)<br>Cov90_marg 0.9080 vs 0.9020 (0.66%, z=0.4) | yes | **differs** |
| `density_parity` | pzflow_galaxies | `torchregress_MDN` | `sbi_MDN` | 3 | NLL -2.4001 vs -2.1658 (10.82%, z=1.2)<br>CRPS 0.0237 vs 0.0312 (24.15%, z=1.1)<br>Cov90_marg 0.9323 vs 0.9323 (0, z=0.0) | yes | **differs** |
| `density_parity` | pzflow_galaxies | `torchregress_NSF` | `sbi_NSF` | 3 | NLL -2.6724 vs -2.7175 (1.66%, z=0.4)<br>CRPS 0.0167 vs 0.0235 (29.11%, z=3.1)<br>Cov90_marg 0.9090 vs 0.9097 (0.07%, z=0.1) | no | **differs** |
| `density_parity` | pzflow_galaxies | `torchregress_NSF` | `sbi_zuko_NSF` | 3 | NLL -2.6724 vs -2.6498 (0.85%, z=0.2)<br>CRPS 0.0167 vs 0.0190 (12.45%, z=0.8)<br>Cov90_marg 0.9090 vs 0.9233 (1.55%, z=1.8) | yes | **differs** |
| `image_density_parity` | ixi_brainage | `torchregress_MAF` | `sbi_MAF` | 3 | NLL 1.5087 vs 1.5196 (0.71%, z=1.0)<br>CRPS 0.6415 vs 0.6471 (0.87%, z=1.1)<br>Cov90_marg 0.9042 vs 0.8750 (3.23%, z=1.0) | yes | **differs** |
| `image_density_parity` | ixi_brainage | `torchregress_MAF` | `sbi_zuko_MAF` | 3 | NLL 1.5087 vs 1.4069 (7.24%, z=1.0)<br>CRPS 0.6415 vs 0.5787 (9.79%, z=0.9)<br>Cov90_marg 0.9042 vs 0.8750 (3.23%, z=1.0) | yes | **differs** |
| `image_density_parity` | ixi_brainage | `torchregress_MDN` | `sbi_MDN` | 3 | NLL 1.3932 vs 1.5082 (7.63%, z=18.8)<br>CRPS 0.6355 vs 0.6489 (2.06%, z=2.3)<br>Cov90_marg 0.8292 vs 0.9167 (9.55%, z=3.1) | no | **differs** |
| `image_density_parity` | ixi_brainage | `torchregress_NSF` | `sbi_NSF` | 3 | NLL 1.2656 vs 1.3963 (9.36%, z=0.8)<br>CRPS 0.5642 vs 0.6303 (10.48%, z=1.0)<br>Cov90_marg 0.8792 vs 0.9042 (2.76%, z=1.6) | yes | **differs** |
| `image_density_parity` | ixi_brainage | `torchregress_NSF` | `sbi_zuko_NSF` | 3 | NLL 1.2656 vs 1.2435 (1.78%, z=0.1)<br>CRPS 0.5642 vs 0.5480 (2.87%, z=0.2)<br>Cov90_marg 0.8792 vs 0.8750 (0.47%, z=0.1) | yes | **differs** |
| `image_density_parity` | synthetic_image_keypoint | `torchregress_MAF` | `sbi_MAF` | 3 | NLL 11.3877 vs 11.3783 (0.08%, z=1.9)<br>CRPS 0.5894 vs 0.5892 (0.04%, z=0.4)<br>Cov90_marg 0.9201 vs 0.9237 (0.39%, z=0.9) | yes | close |
| `image_density_parity` | synthetic_image_keypoint | `torchregress_MAF` | `sbi_zuko_MAF` | 3 | NLL 11.3877 vs 9.7852 (16.38%, z=1.0)<br>CRPS 0.5894 vs 0.5546 (5.91%, z=1.0)<br>Cov90_marg 0.9201 vs 0.9094 (1.16%, z=2.2) | no | **differs** |
| `image_density_parity` | synthetic_image_keypoint | `torchregress_MDN` | `sbi_MDN` | 3 | NLL 11.4285 vs 11.3806 (0.42%, z=4.6)<br>CRPS 0.5892 vs 0.5898 (0.11%, z=1.2)<br>Cov90_marg 0.9297 vs 0.9227 (0.76%, z=1.2) | no | close |
| `image_density_parity` | synthetic_image_keypoint | `torchregress_NSF` | `sbi_NSF` | 3 | NLL 9.1681 vs 10.5001 (12.69%, z=1.1)<br>CRPS 0.5553 vs 0.5817 (4.54%, z=0.9)<br>Cov90_marg 0.8826 vs 0.8914 (0.99%, z=2.4) | no | **differs** |
| `image_density_parity` | synthetic_image_keypoint | `torchregress_NSF` | `sbi_zuko_NSF` | 0 | no common ok seeds | - | n/a |
| `image_density_parity` | udacity_steering | `torchregress_MAF` | `sbi_MAF` | 3 | NLL 9.4583 vs 1.2557 (653.22%, z=1.0)<br>CRPS 0.4676 vs 0.4899 (4.56%, z=0.8)<br>Cov90_marg 0.8708 vs 0.8733 (0.29%, z=0.1) | yes | **differs** |
| `image_density_parity` | udacity_steering | `torchregress_MAF` | `sbi_zuko_MAF` | 3 | NLL 9.4583 vs 104.0 (90.91%, z=1.8)<br>CRPS 0.4676 vs 0.4582 (2.01%, z=0.3)<br>Cov90_marg 0.8708 vs 0.8467 (2.78%, z=1.2) | yes | **differs** |
| `image_density_parity` | udacity_steering | `torchregress_MDN` | `sbi_MDN` | 3 | NLL 0.1382 vs 6.708e-04 (13.75%, z=0.7)<br>CRPS 0.4871 vs 0.4772 (2.03%, z=1.5)<br>Cov90_marg 0.8950 vs 0.9033 (0.92%, z=1.9) | yes | **differs** |
| `image_density_parity` | udacity_steering | `torchregress_NSF` | `sbi_NSF` | 3 | NLL -0.5144 vs -0.5954 (8.09%, z=0.2)<br>CRPS 0.4955 vs 0.5037 (1.63%, z=1.6)<br>Cov90_marg 0.8758 vs 0.7892 (9.90%, z=3.2) | no | **differs** |
| `image_density_parity` | udacity_steering | `torchregress_NSF` | `sbi_zuko_NSF` | 3 | NLL -0.5144 vs -0.4968 (1.76%, z=0.1)<br>CRPS 0.4955 vs 0.4942 (0.26%, z=0.5)<br>Cov90_marg 0.8758 vs 0.8667 (1.05%, z=0.7) | yes | close |
| `orthogonal_inference` | plr_ccddhnr | `tr_Orthogonal_Poly3` | `DoubleML_PLR_Poly3Ridge` | 3 | RMSE 0.0528 vs 0.0538 (1.91%, z=0.8)<br>Coverage95 0.9183 vs 0.9017 (1.81%, z=1.4)<br>CIWidth 0.1790 vs 0.1792 (0.12%, z=0.2) | yes | close |
| `orthogonal_inference` | plr_cubic | `tr_Orthogonal_Poly3` | `DoubleML_PLR_Poly3Ridge` | 3 | RMSE 0.0461 vs 0.0465 (0.93%, z=0.2)<br>Coverage95 0.9367 vs 0.9367 (0, z=0.0)<br>CIWidth 0.1753 vs 0.1757 (0.19%, z=0.9) | yes | close |
| `sbi_parity` | sbibm_bernoulli_glm | `torchregress_MAF` | `sbi_NPE_MAF` | 3 | C2ST 0.7240 vs 0.6641 (8.27%, z=3.2)<br>MeanRMSE 0.6605 vs 0.4392 (33.51%, z=2.6) | no | **differs** |
| `sbi_parity` | sbibm_bernoulli_glm | `torchregress_MDN` | `sbi_NPE_MDN` | 3 | C2ST 0.7866 vs 0.6992 (11.11%, z=4.3)<br>MeanRMSE 1.1267 vs 0.7588 (32.65%, z=4.8) | no | **differs** |
| `sbi_parity` | sbibm_bernoulli_glm | `torchregress_NSF` | `sbi_NPE` | 3 | C2ST 0.7493 vs 0.6750 (9.92%, z=7.6)<br>MeanRMSE 0.7053 vs 0.4756 (32.57%, z=3.9) | no | **differs** |
| `sbi_parity` | sbibm_gaussian_linear | `torchregress_MAF` | `sbi_NPE_MAF` | 3 | C2ST 0.5739 vs 0.5272 (8.14%, z=3.0)<br>MeanRMSE 0.1307 vs 0.0851 (34.93%, z=3.2) | no | **differs** |
| `sbi_parity` | sbibm_gaussian_linear | `torchregress_MDN` | `sbi_NPE_MDN` | 3 | C2ST 0.5483 vs 0.5568 (1.53%, z=0.5)<br>MeanRMSE 0.1005 vs 0.1140 (11.77%, z=1.0) | yes | **differs** |
| `sbi_parity` | sbibm_gaussian_linear | `torchregress_NSF` | `sbi_NPE` | 3 | C2ST 0.6285 vs 0.5335 (15.12%, z=7.4)<br>MeanRMSE 0.1991 vs 0.0691 (65.32%, z=12.0) | no | **differs** |
| `sbi_parity` | sbibm_gaussian_mixture | `torchregress_MAF` | `sbi_NPE_MAF` | 3 | C2ST 0.6927 vs 0.6519 (5.89%, z=1.2)<br>MeanRMSE 0.2389 vs 0.1861 (22.10%, z=1.3) | yes | **differs** |
| `sbi_parity` | sbibm_gaussian_mixture | `torchregress_MDN` | `sbi_NPE_MDN` | 3 | C2ST 0.5505 vs 0.5405 (1.82%, z=0.5)<br>MeanRMSE 0.1231 vs 0.1705 (27.78%, z=3.2) | no | **differs** |
| `sbi_parity` | sbibm_gaussian_mixture | `torchregress_NSF` | `sbi_NPE` | 3 | C2ST 0.6236 vs 0.5130 (17.74%, z=4.8)<br>MeanRMSE 0.1996 vs 0.0909 (54.45%, z=4.2) | no | **differs** |
| `sbi_parity` | sbibm_slcp | `torchregress_MAF` | `sbi_NPE_MAF` | 3 | C2ST 0.9166 vs 0.9292 (1.35%, z=0.6)<br>MeanRMSE 1.0100 vs 1.0748 (6.03%, z=0.4) | yes | **differs** |
| `sbi_parity` | sbibm_slcp | `torchregress_MDN` | `sbi_NPE_MDN` | 3 | C2ST 0.8971 vs 0.8716 (2.85%, z=0.7)<br>MeanRMSE 1.2191 vs 0.9831 (19.36%, z=0.8) | yes | **differs** |
| `sbi_parity` | sbibm_slcp | `torchregress_NSF` | `sbi_NPE` | 3 | C2ST 0.8047 vs 0.8514 (5.47%, z=2.8)<br>MeanRMSE 0.8948 vs 1.2247 (26.94%, z=2.6) | no | **differs** |
| `sbi_parity` | sbibm_twomoons | `torchregress_MAF` | `sbi_NPE_MAF` | 3 | C2ST 0.5889 vs 0.6285 (6.30%, z=0.7)<br>MeanRMSE 0.0180 vs 0.0241 (25.49%, z=0.9) | yes | **differs** |
| `sbi_parity` | sbibm_twomoons | `torchregress_MDN` | `sbi_NPE_MDN` | 3 | C2ST 0.6861 vs 0.5338 (22.21%, z=1.2)<br>MeanRMSE 0.0269 vs 0.0419 (35.62%, z=0.9) | yes | **differs** |
| `sbi_parity` | sbibm_twomoons | `torchregress_NSF` | `sbi_NPE` | 3 | C2ST 0.5885 vs 0.4982 (15.33%, z=2.6)<br>MeanRMSE 0.0161 vs 0.0187 (14.26%, z=0.9) | no | **differs** |

### Method pairs

| Suite | Dataset | torchregress | Comparator | Seeds | Key metrics: tr vs comp (rel. diff, z) | Within noise | Verdict |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| `age_regression` | ixi_brainage | `CNN+GaussianNLL` | `torch_uncertainty` | 3 | RMSE 16.8909 vs 21.9184 (22.94%, z=1.5)<br>NLL 4.2584 vs 4.5153 (5.69%, z=1.7) | yes | **differs** |
| `age_regression` | ixi_brainage | `ResNet+GaussianNLL` | `timm_resnet18+GaussianNLL` | 3 | RMSE 11.5590 vs 11.9945 (3.63%, z=1.7)<br>NLL 18.1356 vs 14.8417 (22.19%, z=0.4) | yes | **differs** |
| `age_regression` | ixi_brainage_tiny | `CNN+GaussianNLL` | `torch_uncertainty` | 3 | RMSE 37.9748 vs 37.5054 (1.24%, z=0.3)<br>NLL 5.1164 vs 12.0427 (57.51%, z=10.0) | no | **differs** |
| `age_regression` | ixi_brainage_tiny | `ResNet+GaussianNLL` | `timm_resnet18+GaussianNLL` | 3 | RMSE 28.1474 vs 19.3738 (31.17%, z=4.6)<br>NLL 5.8543 vs 6.0104 (2.60%, z=0.1) | no | **differs** |
| `age_regression` | synthetic_image_regression | `CNN+GaussianNLL` | `torch_uncertainty` | 3 | RMSE 0.0572 vs 0.2333 (75.50%, z=85.2)<br>NLL -1.4404 vs -0.0365 (140.39%, z=50.7) | no | **differs** |
| `age_regression` | synthetic_image_regression | `ResNet+GaussianNLL` | `timm_resnet18+GaussianNLL` | 3 | RMSE 0.0779 vs 0.0679 (12.87%, z=0.9)<br>NLL -1.1572 vs -1.0695 (8.20%, z=0.9) | yes | **differs** |
| `age_regression` | udacity_steering | `CNN+GaussianNLL` | `torch_uncertainty` | 3 | RMSE 0.3268 vs 0.3447 (5.21%, z=3.6)<br>NLL 0.0473 vs 0.3505 (30.32%, z=3.1) | no | **differs** |
| `age_regression` | udacity_steering | `ResNet+GaussianNLL` | `timm_resnet18+GaussianNLL` | 3 | RMSE 0.2408 vs 0.2442 (1.39%, z=0.4)<br>NLL 0.3541 vs 2.1897 (83.83%, z=1.6) | yes | **differs** |
| `age_regression` | udacity_steering_tiny | `CNN+GaussianNLL` | `torch_uncertainty` | 3 | RMSE 0.3354 vs 0.3397 (1.27%, z=2.5)<br>NLL 0.5483 vs 0.3433 (20.50%, z=5.6) | no | **differs** |
| `age_regression` | udacity_steering_tiny | `ResNet+GaussianNLL` | `timm_resnet18+GaussianNLL` | 3 | RMSE 0.3319 vs 0.3159 (4.84%, z=0.6)<br>NLL 1.6939 vs 5.8408 (71.00%, z=3.6) | no | **differs** |
| `causal` | folktables_causal | `DR` | `EconML_DRLearner` | 3 | ATEErr 0.5473 vs 0.5349 (2.27%, z=0.4)<br>PEHE 1.1839 vs 1.1256 (4.93%, z=0.2) | yes | **differs** |
| `causal` | ihdp | `DR` | `EconML_DRLearner` | 3 | ATEErr 2.1759 vs 2.0880 (4.04%, z=1.8)<br>PEHE 2.3460 vs 2.2465 (4.24%, z=1.6) | yes | **differs** |
| `censored` | survset_hdfail | `AFT` | `lifelines_LogNormalAFT` | 3 | RMSE 849.9 vs 6353.3 (86.62%, z=17553.9)<br>Concordance 0.8913 vs 0.8943 (0.34%, z=3.6) | no | **differs** |
| `censored` | synthetic_censored | `AFT` | `lifelines_LogNormalAFT` | 3 | RMSE 3.6900 vs 1.1939 (67.64%, z=8.6)<br>Concordance 0.8498 vs 0.9567 (11.18%, z=4.9) | no | **differs** |
| `censored_imaging` | bbbc_censored | `AFT` | `lifelines_LogNormalAFT` | 3 | RMSE 47.0337 vs 101.8 (53.79%, z=102.7) | no | **differs** |
| `censored_imaging` | bbbc_censored_tiny | `AFT` | `lifelines_LogNormalAFT` | 3 | RMSE 49.0376 vs 155.5 (68.47%, z=20.4) | no | **differs** |
| `censored_imaging` | tcga_survival | `AFT` | `lifelines_LogNormalAFT` | 3 | RMSE 796.8 vs 4727.3 (83.14%, z=117.2) | no | **differs** |
| `censored_imaging` | tcga_survival_tiny | `AFT` | `lifelines_LogNormalAFT` | 3 | RMSE 1217.9 vs 2071.3 (41.20%, z=187.0) | no | **differs** |
| `density_parity` | california_housing | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL 0.2378 vs 0.3155 (7.78%, z=2.0)<br>CRPS 0.2369 vs 0.2349 (0.81%, z=0.4)<br>Cov90_marg 0.8767 vs 0.8607 (1.83%, z=0.7) | no | **differs** |
| `density_parity` | california_housing | `torchregress_NSF` | `sbi_NPE_NSF` | 3 | NLL 0.2378 vs 0.2516 (1.39%, z=0.7)<br>CRPS 0.2369 vs 0.2414 (1.90%, z=0.8)<br>Cov90_marg 0.8767 vs 0.8947 (2.01%, z=1.4) | yes | **differs** |
| `density_parity` | energy_efficiency | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL -1.3509 vs -2.1681 (37.69%, z=2.9)<br>CRPS 0.1397 vs 0.1152 (17.55%, z=3.0)<br>Cov90_marg 0.8853 vs 0.8752 (1.14%, z=0.7) | no | **differs** |
| `density_parity` | energy_efficiency | `torchregress_NSF` | `sbi_NPE_NSF` | 3 | NLL -1.3509 vs -1.5914 (15.11%, z=1.1)<br>CRPS 0.1397 vs 0.1455 (4.03%, z=0.8)<br>Cov90_marg 0.8853 vs 0.8997 (1.60%, z=0.9) | yes | **differs** |
| `density_parity` | mt_enb | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL -1.7311 vs -2.2723 (23.82%, z=3.2)<br>CRPS 0.1224 vs 0.1101 (10.07%, z=2.0)<br>Cov90_marg 0.9069 vs 0.8680 (4.30%, z=5.4) | no | **differs** |
| `density_parity` | mt_enb | `torchregress_NSF` | `sbi_NPE_NSF` | 3 | NLL -1.7311 vs -1.6486 (5.01%, z=0.7)<br>CRPS 0.1224 vs 0.1411 (13.24%, z=3.3)<br>Cov90_marg 0.9069 vs 0.9167 (1.06%, z=0.6) | no | **differs** |
| `density_parity` | mt_jura | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL 1.6085 vs 2.6439 (39.16%, z=9.6)<br>CRPS 0.2670 vs 0.2495 (6.55%, z=2.3)<br>Cov90_marg 0.8519 vs 0.7284 (14.49%, z=25.3) | no | **differs** |
| `density_parity` | mt_jura | `torchregress_NSF` | `sbi_NPE_NSF` | 3 | NLL 1.6085 vs 1.8902 (14.91%, z=2.8)<br>CRPS 0.2670 vs 0.2759 (3.23%, z=0.8)<br>Cov90_marg 0.8519 vs 0.8858 (3.83%, z=1.8) | no | **differs** |
| `density_parity` | mt_rf1 | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL -15.0640 vs -3.3006 (356.40%, z=6.0)<br>CRPS 0.0385 vs nan (n/a)<br>Cov90_marg 0.9013 vs 0.8961 (0.58%, z=0.5) | - | **differs** |
| `density_parity` | mt_rf1 | `torchregress_NSF` | `sbi_NPE_NSF` | 3 | NLL -15.0640 vs -10.4751 (43.81%, z=2.6)<br>CRPS 0.0385 vs 0.0873 (55.85%, z=2.7)<br>Cov90_marg 0.9013 vs 0.8978 (0.39%, z=0.3) | no | **differs** |
| `density_parity` | mt_scm20d | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL -2.5702 vs -1.3180 (95.01%, z=8.9)<br>CRPS 0.1925 vs 0.1570 (18.42%, z=11.1)<br>Cov90_marg 0.8669 vs 0.8141 (6.09%, z=8.2) | no | **differs** |
| `density_parity` | mt_scm20d | `torchregress_NSF` | `sbi_NPE_NSF` | 3 | NLL -2.5702 vs -0.3514 (221.88%, z=16.9)<br>CRPS 0.1925 vs 0.2526 (23.81%, z=21.9)<br>Cov90_marg 0.8669 vs 0.8798 (1.47%, z=3.1) | no | **differs** |
| `density_parity` | pzflow_galaxies | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL -2.6724 vs -2.9965 (10.81%, z=8.0)<br>CRPS 0.0167 vs 0.0121 (27.43%, z=9.9)<br>Cov90_marg 0.9090 vs 0.9143 (0.58%, z=0.5) | no | **differs** |
| `density_parity` | pzflow_galaxies | `torchregress_NSF` | `sbi_NPE_NSF` | 3 | NLL -2.6724 vs -2.8234 (5.35%, z=3.9)<br>CRPS 0.0167 vs 0.0282 (40.85%, z=5.5)<br>Cov90_marg 0.9090 vs 0.8890 (2.20%, z=2.0) | no | **differs** |
| `ensembles` | california_housing | `Stacking` | `sklearn_StackingRegressor` | 3 | RMSE 0.5882 vs 0.5695 (3.18%, z=5.4) | no | **differs** |
| `ensembles` | diabetes | `Stacking` | `sklearn_StackingRegressor` | 3 | RMSE 59.7686 vs 51.6778 (13.54%, z=2.6) | no | **differs** |
| `ensembles` | make_regression_synthetic | `Stacking` | `sklearn_StackingRegressor` | 3 | RMSE 15.7502 vs 14.9234 (5.25%, z=7.3) | no | **differs** |
| `ensembles` | synthetic_hetero | `Stacking` | `sklearn_StackingRegressor` | 3 | RMSE 0.3814 vs 0.3736 (2.04%, z=1.3) | yes | **differs** |
| `ensembles` | synthetic_outliers | `Stacking` | `sklearn_StackingRegressor` | 3 | RMSE 1.2522 vs 1.2553 (0.25%, z=1.7) | yes | close |
| `ensembles` | tabred_cooking | `Stacking` | `sklearn_StackingRegressor` | 3 | RMSE 0.5551 vs 0.4743 (14.54%, z=23.1) | no | **differs** |
| `ensembles` | tabred_delivery | `Stacking` | `sklearn_StackingRegressor` | 3 | RMSE 0.6517 vs 0.5545 (14.91%, z=35.0) | no | **differs** |
| `ensembles` | tabred_maps | `Stacking` | `sklearn_StackingRegressor` | 3 | RMSE 0.1831 vs 0.1611 (12.04%, z=14.7) | no | **differs** |
| `ensembles` | tabred_weather | `Stacking` | `sklearn_StackingRegressor` | 3 | RMSE 1.8358 vs 1.6762 (8.69%, z=23.3) | no | **differs** |
| `ensembles` | zurich_delays | `Stacking` | `sklearn_StackingRegressor` | 3 | RMSE 111.5 vs 111.1 (0.30%, z=0.2) | yes | close |
| `image_ensembles_uq` | synthetic_image_regression | `MC_Dropout` | `luqbox_MCDropout` | 3 | RMSE 0.0567 vs 0.0787 (27.98%, z=2.7)<br>NLL -0.1096 vs -0.8788 (76.92%, z=3.6) | no | **differs** |
| `label_shift` | california_housing | `LibDomainClassifier` | `DomainClassifier` | 3 | WeightRMSE 0.4443 vs 0.8611 (48.40%, z=9.9)<br>RMSE 0.5843 vs 0.5843 (0, z=0.0) | no | **differs** |
| `label_shift` | diabetes | `LibDomainClassifier` | `DomainClassifier` | 3 | WeightRMSE 0.4737 vs 0.5654 (16.20%, z=4.6)<br>RMSE 53.1870 vs 53.1870 (0, z=0.0) | no | **differs** |
| `label_shift` | folktables_income | `LibDomainClassifier` | `DomainClassifier` | 3 | WeightRMSE 0.4135 vs 5.9586 (93.06%, z=3.9)<br>RMSE 80208.4 vs 80208.4 (0, z=0.0) | no | **differs** |
| `missing_data` | california_housing | `PropensityWeighted` | `sklearn_IPW` | 3 | RMSE 0.7527 vs 0.8029 (6.25%, z=0.4) | yes | **differs** |
| `missing_data` | diabetes | `PropensityWeighted` | `sklearn_IPW` | 3 | RMSE 54.8692 vs 54.2739 (1.08%, z=1.6) | yes | close |
| `missing_data` | folktables_income | `PropensityWeighted` | `sklearn_IPW` | 3 | RMSE 82628.1 vs 74295.5 (10.08%, z=6.6) | no | **differs** |
| `ordinal` | folktables_decile | `CumulativeLink` | `statsmodels_OrderedModel` | 3 | MAE 1.4714 vs 1.5238 (3.44%, z=19.8)<br>QWK 0.7227 vs 0.7300 (1.01%, z=20.7) | no | **differs** |
| `ordinal` | synthetic_ordinal | `CumulativeLink` | `statsmodels_OrderedModel` | 3 | MAE 0.2642 vs 0.2500 (5.36%, z=2.6)<br>QWK 0.9322 vs 0.9357 (0.38%, z=2.5) | no | **differs** |
| `ppi` | california_housing | `PPI_Mean` | `PPI_py_Mean` | 3 | Estimate 1.9805 vs 1.9806 (7.8e-03%, z=0.0)<br>CIWidth 0.2082 vs 0.2102 (0.94%, z=0.2) | yes | close |
| `ppi` | california_housing | `PPI_Quantile` | `PPI_py_Quantile` | 3 | Estimate 1.7457 vs 1.7574 (0.67%, z=0.2)<br>CIWidth 0.4767 vs 0.4751 (0.35%, z=0.0) | yes | close |
| `ppi` | diabetes | `PPI_Mean` | `PPI_py_Mean` | 3 | Estimate 150.6 vs 150.6 (0.02%, z=0.0)<br>CIWidth 31.3126 vs 27.1887 (13.17%, z=2.3) | no | **differs** |
| `ppi` | diabetes | `PPI_Quantile` | `PPI_py_Quantile` | 3 | Estimate 129.2 vs 142.6 (9.34%, z=0.8)<br>CIWidth 76.2911 vs 76.7340 (0.58%, z=0.0) | yes | **differs** |
| `ppi` | folktables_income | `PPI_Mean` | `PPI_py_Mean` | 3 | Estimate 68391.6 vs 67143.0 (1.83%, z=0.3)<br>CIWidth 30454.0 vs 31298.5 (2.70%, z=0.3) | yes | **differs** |
| `ppi` | folktables_income | `PPI_Quantile` | `PPI_py_Quantile` | 3 | Estimate 39529.2 vs 39977.3 (1.12%, z=0.1)<br>CIWidth 24257.0 vs 23974.6 (1.16%, z=0.0) | yes | close |
| `pzflow_parity` | energy_efficiency | `torchregress_NSF` | `normflows_NSF` | 0 | no common ok seeds | - | n/a |
| `pzflow_parity` | energy_efficiency | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL 2.6066 vs 2.4945 (4.49%, z=0.8)<br>IntervalScore 7.9387 vs 32.5117 (75.58%, z=59.2)<br>Coverage 0.9278 vs 0.8939 (3.65%, z=3.3) | no | **differs** |
| `pzflow_parity` | pzflow_galaxies | `torchregress_NSF` | `normflows_NSF` | 0 | no common ok seeds | - | n/a |
| `pzflow_parity` | pzflow_galaxies | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL 11.3800 vs 21.0829 (46.02%, z=26.9)<br>IntervalScore 0.0818 vs 2.0272 (95.96%, z=793.4)<br>Coverage 0.8887 vs 0.8971 (0.94%, z=0.5) | no | **differs** |
| `pzflow_parity` | synthetic_cond_gaussian | `torchregress_NSF` | `normflows_NSF` | 0 | no common ok seeds | - | n/a |
| `pzflow_parity` | synthetic_cond_gaussian | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL 5.9795 vs 7.1500 (16.37%, z=4.0)<br>IntervalScore 1.4719 vs 11.6726 (87.39%, z=283.8)<br>Coverage 0.8306 vs 0.9000 (7.72%, z=5.9) | no | **differs** |
| `pzflow_parity` | synthetic_cond_nongaussian | `torchregress_NSF` | `normflows_NSF` | 0 | no common ok seeds | - | n/a |
| `pzflow_parity` | synthetic_cond_nongaussian | `torchregress_NSF` | `pzflow_NSF` | 3 | NLL 4.0895 vs 4.7856 (14.55%, z=5.7)<br>IntervalScore 3.4259 vs 12.5392 (72.68%, z=153.8)<br>Coverage 0.8150 vs 0.8928 (8.71%, z=2.4) | no | **differs** |
| `r2c_conformal` | california_housing | `R2C+Softmax` | `torchcp_R2C` | 3 | IntervalScore 2.6816 vs 5.6522 (52.56%, z=49.2)<br>Coverage 0.9582 vs 0.9542 (0.42%, z=0.1)<br>Width 2.1722 vs 5.1647 (57.94%, z=5.6) | no | **differs** |
| `r2c_conformal` | diabetes | `R2C+Softmax` | `torchcp_R2C` | 3 | IntervalScore 288.8 vs 352.9 (18.16%, z=3.5)<br>Coverage 0.9950 vs 1.0000 (0.50%, z=2.0)<br>Width 288.8 vs 352.9 (18.17%, z=3.5) | no | **differs** |
| `r2c_conformal` | house_sales | `R2C+Softmax` | `torchcp_R2C` | 3 | IntervalScore 6.751e+05 vs 4.697e+06 (85.63%, z=806.8)<br>Coverage 0.9917 vs 0.0000 (100.00%, z=2248.9)<br>Width 6.499e+05 vs 0.0000 (100.00%, z=116.0) | no | **differs** |
| `r2c_conformal` | openml_year | `R2C+Softmax` | `torchcp_R2C` | 3 | IntervalScore 0.7881 vs 3.0270 (73.96%, z=3.1)<br>Coverage 0.9471 vs 0.6020 (36.44%, z=2.1)<br>Width 0.6418 vs 0.9639 (33.42%, z=1.5) | no | **differs** |
| `r2c_conformal` | synthetic_hetero | `R2C+Softmax` | `torchcp_R2C` | 3 | IntervalScore 2.0193 vs 8.6797 (76.74%, z=15.2)<br>Coverage 0.9633 vs 0.9917 (2.86%, z=3.3)<br>Width 1.8779 vs 8.4732 (77.84%, z=10.6) | no | **differs** |
| `robust_regression` | california_housing | `IRLS` | `statsmodels_RLM` | 3 | RMSE 1.3286 vs 0.8260 (37.83%, z=3.9)<br>MAE 0.9870 vs 0.5127 (48.06%, z=9.0) | no | **differs** |
| `robust_regression` | synthetic_outliers | `IRLS` | `statsmodels_RLM` | 3 | RMSE 2.8930 vs 1.3873 (52.05%, z=6.0)<br>MAE 2.3157 vs 0.8415 (63.66%, z=7.1) | no | **differs** |
| `robust_regression` | zurich_delays | `IRLS` | `statsmodels_RLM` | 3 | RMSE 152.3 vs 119.7 (21.41%, z=11.1)<br>MAE 98.8364 vs 59.4249 (39.88%, z=12.8) | no | **differs** |
| `sbi_parity` | sbibm_bernoulli_glm | `torchregress_NSF` | `pzflow_NSF` | 3 | C2ST 0.7493 vs 0.8006 (6.41%, z=6.2)<br>MeanRMSE 0.7053 vs 0.5768 (18.22%, z=2.3) | no | **differs** |
| `sbi_parity` | sbibm_gaussian_linear | `torchregress_NSF` | `pzflow_NSF` | 3 | C2ST 0.6285 vs 0.7928 (20.73%, z=11.9)<br>MeanRMSE 0.1991 vs 0.2056 (3.14%, z=0.6) | no | **differs** |
| `sbi_parity` | sbibm_gaussian_mixture | `torchregress_NSF` | `pzflow_NSF` | 3 | C2ST 0.6236 vs 0.5546 (11.06%, z=2.9)<br>MeanRMSE 0.1996 vs 0.1068 (46.48%, z=3.2) | no | **differs** |
| `sbi_parity` | sbibm_slcp | `torchregress_NSF` | `pzflow_NSF` | 3 | C2ST 0.8047 vs 0.8591 (6.32%, z=4.0)<br>MeanRMSE 0.8948 vs 1.0855 (17.57%, z=1.3) | no | **differs** |
| `sbi_parity` | sbibm_twomoons | `torchregress_NSF` | `pzflow_NSF` | 3 | C2ST 0.5885 vs 0.5167 (12.20%, z=2.0)<br>MeanRMSE 0.0161 vs 0.0425 (62.17%, z=2.7) | no | **differs** |
| `stress_matrix` | stress_ood | `stress_ood/MC_Dropout` | `stress_ood/luqbox_MCDropout` | 3 | RMSE 0.6794 vs 0.7663 (11.34%, z=1.5)<br>NLL 1.8226 vs 1.1330 (60.86%, z=6.0) | no | **differs** |

## Results by suite

### Group `fm`

#### fm_multitarget (fm)

Group `fm`; 4 dataset(s). Columns: ES ↓, VS ↓, CRPS ↓, Cov90_marg ·, Cov90_joint ·, CorrErr ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: ES.

##### fm_multitarget / mt_enb

| Method | Library | ES | VS | CRPS | Cov90_marg | Cov90_joint | CorrErr | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussCopula+KumoTabular` | sdm | **0.0358 ± 0.0016** | **0.0057 ± 0.0001** | **0.0223 ± 0.0011** | 0.8810 ± 0.0050 | - | 0.0554 ± 0.0175 | 1.3353 |
| `GaussCopula+LightGBM` | lightgbm | 0.0769 ± 0.0016 | 0.0112 ± 0.0001 | 0.0458 ± 0.0012 | 0.7619 ± 0.0131 | - | 0.0241 ± 0.0196 | 1.1123 |
| `GaussCopula+TabFM` | sdm | 0.0437 ± 0.0016 | 0.0068 ± 0.0009 | 0.0275 ± 0.0010 | 0.8842 ± 0.0291 | - | 0.2167 ± 0.0626 | 8.1012 |
| `GaussCopula+TabICL` | tabicl | 0.0772 ± 0.0012 | 0.0089 ± 0.0002 | 0.0453 ± 0.0008 | 0.9102 ± 0.0153 | - | 0.3948 ± 0.0300 | 2.4513 |
| `GaussCopula+TabularMLP` | torchregress | 0.1017 ± 0.0010 | 0.0169 ± 0.0002 | 0.0622 ± 0.0007 | 0.9740 ± 0.0086 | - | 0.2224 ± 0.0782 | 12.9627 |
| `Indep+KumoTabular` | sdm | 0.0360 ± 0.0014 | 0.0058 ± 0.0002 | 0.0223 ± 0.0009 | 0.8810 ± 0.0167 | - | 0.0397 ± 0.0442 | 1.3353 |
| `Indep+LightGBM` | lightgbm | 0.0777 ± 0.0016 | 0.0114 ± 0.0003 | 0.0462 ± 0.0012 | 0.7511 ± 0.0230 | - | 0.0288 ± 0.0224 | 1.1123 |
| `Indep+TabFM` | sdm | 0.0437 ± 0.0018 | 0.0069 ± 0.0009 | 0.0275 ± 0.0011 | 0.8777 ± 0.0325 | - | 0.2472 ± 0.0557 | 8.1012 |
| `Indep+TabICL` | tabicl | 0.0785 ± 0.0014 | 0.0095 ± 0.0003 | 0.0460 ± 0.0009 | 0.9048 ± 0.0114 | - | 0.5704 ± 0.0136 | 2.4513 |
| `Indep+TabularMLP` | torchregress | 0.1021 ± 0.0007 | 0.0174 ± 0.0004 | 0.0624 ± 0.0006 | 0.9729 ± 0.0050 | - | 0.1733 ± 0.0315 | 12.9627 |
| `torchregress_MDN` | torchregress | 0.2165 ± 0.0263 † | 0.0208 ± 0.0027 | 0.1432 ± 0.0185 | 0.8918 ± 0.0131 | 0.9134 ± 0.0037 | **0.0125 ± 0.0130** | 7.1627 |
| `torchregress_NSF` | torchregress | 0.2026 ± 0.0284 † | 0.0199 ± 0.0040 | 0.1327 ± 0.0199 | 0.8950 ± 0.0150 | 0.9113 ± 0.0198 | 0.0422 ± 0.0054 | 21.5054 |
| `tr_PITRecal+GaussCopula+KumoTabular` | torchregress | 0.0362 ± 0.0018 † | 0.0058 ± 0.0001 | 0.0226 ± 0.0012 | 0.8355 ± 0.0295 | - | 0.0413 ± 0.0196 | 1.3353 |
| `tr_PITRecal+GaussCopula+LightGBM` | torchregress | 0.0898 ± 0.0072 † | 0.0187 ± 0.0054 | 0.0513 ± 0.0035 | 0.8799 ± 0.0162 | - | 0.3366 ± 0.0511 | 1.1123 |
| `tr_PITRecal+GaussCopula+TabICL` | torchregress | 0.0778 ± 0.0009 | 0.0094 ± 0.0008 | 0.0458 ± 0.0006 | 0.8517 ± 0.0123 | - | 0.3995 ± 0.0256 | 2.4513 |
| `tr_PITRecal+GaussCopula+TabularMLP` | torchregress | 0.1007 ± 0.0012 | 0.0180 ± 0.0016 | 0.0610 ± 0.0007 | 0.8506 ± 0.0298 | - | 0.2199 ± 0.0671 | 12.9627 |

##### fm_multitarget / mt_jura

| Method | Library | ES | VS | CRPS | Cov90_marg | Cov90_joint | CorrErr | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussCopula+KumoTabular` | sdm | 0.4726 ± 0.0129 | 0.3041 ± 0.0151 | 0.2194 ± 0.0058 | 0.8843 ± 0.0122 | - | 0.1956 ± 0.0540 | 1.5570 |
| `GaussCopula+LightGBM` | lightgbm | 0.5156 ± 0.0043 | 0.3235 ± 0.0044 | 0.2472 ± 0.0039 | 0.6698 ± 0.0255 | - | 0.1243 ± 0.0550 | 1.6821 |
| `GaussCopula+TabFM` | sdm | 0.5209 ± 0.0278 † | 0.3367 ± 0.0264 | 0.2466 ± 0.0100 | 0.8966 ± 0.0193 | - | 0.1215 ± 0.0290 | 10.3222 |
| `GaussCopula+TabICL` | tabicl | 0.4773 ± 0.0144 | 0.2980 ± 0.0168 | 0.2236 ± 0.0046 | 0.9090 ± 0.0053 | - | 0.1783 ± 0.0253 | 2.6276 |
| `GaussCopula+TabularMLP` | torchregress | 0.5621 ± 0.0122 | 0.3376 ± 0.0087 | 0.2672 ± 0.0049 | 0.9414 ± 0.0117 | - | 0.1563 ± 0.0385 | 6.4403 |
| `Indep+KumoTabular` | sdm | **0.4717 ± 0.0119** | 0.3048 ± 0.0187 | **0.2187 ± 0.0047** | 0.8843 ± 0.0046 | - | 0.1455 ± 0.0312 | 1.5570 |
| `Indep+LightGBM` | lightgbm | 0.5137 ± 0.0037 | 0.3223 ± 0.0043 | 0.2458 ± 0.0022 | 0.6960 ± 0.0149 | - | 0.1476 ± 0.0283 | 1.6821 |
| `Indep+TabFM` | sdm | 0.5164 ± 0.0316 † | 0.3337 ± 0.0293 | 0.2446 ± 0.0116 | 0.8951 ± 0.0117 | - | 0.1431 ± 0.0030 | 10.3222 |
| `Indep+TabICL` | tabicl | 0.4767 ± 0.0165 | **0.2976 ± 0.0211** | 0.2229 ± 0.0062 | 0.9090 ± 0.0027 | - | 0.1590 ± 0.0150 | 2.6276 |
| `Indep+TabularMLP` | torchregress | 0.5617 ± 0.0138 | 0.3358 ± 0.0123 | 0.2665 ± 0.0054 | 0.9444 ± 0.0093 | - | 0.1638 ± 0.0197 | 6.4403 |
| `torchregress_MDN` | torchregress | 0.6282 ± 0.0411 † | 0.3749 ± 0.0146 | 0.2970 ± 0.0190 | 0.8997 ± 0.0107 | 0.8380 ± 0.0350 | 0.1609 ± 0.0768 | 2.9703 |
| `torchregress_NSF` | torchregress | 0.5576 ± 0.0085 | 0.3373 ± 0.0112 | 0.2626 ± 0.0020 | 0.8380 ± 0.0231 | 0.8056 ± 0.0501 | 0.1325 ± 0.0148 | 8.8081 |
| `tr_PITRecal+GaussCopula+KumoTabular` | torchregress | 0.4807 ± 0.0259 † | 0.3134 ± 0.0215 | 0.2242 ± 0.0102 | 0.8549 ± 0.0107 | - | 0.1724 ± 0.0419 | 1.5570 |
| `tr_PITRecal+GaussCopula+LightGBM` | torchregress | 0.5398 ± 0.0135 | 0.3366 ± 0.0132 | 0.2539 ± 0.0025 | 0.8688 ± 0.0354 | - | **0.1197 ± 0.0457** | 1.6821 |
| `tr_PITRecal+GaussCopula+TabICL` | torchregress | 0.4834 ± 0.0270 † | 0.3055 ± 0.0216 | 0.2273 ± 0.0096 | 0.8596 ± 0.0347 | - | 0.1768 ± 0.0159 | 2.6276 |
| `tr_PITRecal+GaussCopula+TabularMLP` | torchregress | 0.5572 ± 0.0077 | 0.3320 ± 0.0103 | 0.2651 ± 0.0035 | 0.8966 ± 0.0027 | - | 0.1427 ± 0.0236 | 6.4403 |

##### fm_multitarget / mt_rf1

| Method | Library | ES | VS | CRPS | Cov90_marg | Cov90_joint | CorrErr | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussCopula+KumoTabular` | sdm | 0.0313 ± 0.0049 † | 0.0174 ± 0.0042 | 0.0072 ± 0.0009 | 0.9092 ± 0.0013 | - | 0.0316 ± 0.0012 | 38.5831 |
| `GaussCopula+LightGBM` | lightgbm | 0.2311 ± 0.0168 † | 0.3693 ± 0.0345 | 0.0451 ± 0.0039 | 0.8811 ± 0.0024 | - | 0.2032 ± 0.0114 | 89.7409 |
| `GaussCopula+TabFM` ‡ | sdm | 0.0458 ± 0.0020 | 0.0303 ± 0.0028 | 0.0113 ± 0.0006 | 0.8923 ± 0.0014 | - | 0.0336 ± 0.0062 | 383.8 |
| `GaussCopula+TabICL` | tabicl | 0.0332 ± 0.0057 † | 0.0191 ± 0.0053 | 0.0077 ± 0.0011 | 0.8742 ± 0.0029 | - | 0.0332 ± 0.0033 | 47.7815 |
| `GaussCopula+TabPFN` | tabpfn | 0.0343 ± 0.0053 † | 0.0192 ± 0.0046 | 0.0080 ± 0.0010 | 0.9377 ± 0.0009 | - | 0.0331 ± 0.0056 | 105.3 |
| `GaussCopula+TabularMLP` | torchregress | 0.1531 ± 0.0353 † | 0.1951 ± 0.0695 | 0.0349 ± 0.0061 | 0.9735 ± 0.0057 | - | 0.1021 ± 0.0132 | 357.0 |
| `Indep+KumoTabular` | sdm | **0.0312 ± 0.0048** † | **0.0174 ± 0.0042** | **0.0072 ± 0.0009** | 0.9095 ± 0.0029 | - | **0.0237 ± 0.0017** | 38.5831 |
| `Indep+LightGBM` | lightgbm | 0.2368 ± 0.0182 † | 0.3783 ± 0.0387 | 0.0449 ± 0.0039 | 0.8818 ± 0.0020 | - | 0.2657 ± 0.0049 | 89.7409 |
| `Indep+TabFM` ‡ | sdm | 0.0457 ± 0.0020 | 0.0302 ± 0.0027 | 0.0113 ± 0.0006 | 0.8922 ± 0.0019 | - | 0.0473 ± 0.0107 | 383.8 |
| `Indep+TabICL` | tabicl | 0.0332 ± 0.0057 † | 0.0190 ± 0.0052 | 0.0077 ± 0.0011 | 0.8735 ± 0.0025 | - | 0.0292 ± 0.0038 | 47.7815 |
| `Indep+TabPFN` | tabpfn | 0.0342 ± 0.0052 † | 0.0191 ± 0.0045 | 0.0080 ± 0.0010 | 0.9388 ± 0.0035 | - | 0.0266 ± 0.0016 | 105.3 |
| `Indep+TabularMLP` | torchregress | 0.1534 ± 0.0351 † | 0.1971 ± 0.0675 | 0.0348 ± 0.0060 | 0.9733 ± 0.0036 | - | 0.1343 ± 0.0114 | 357.0 |
| `torchregress_MDN` | torchregress | 0.3944 ± 0.0803 † | 0.8677 ± 0.1342 | 0.1026 ± 0.0231 | 0.9233 ± 0.0046 | 0.9367 ± 0.0035 | 0.2105 ± 0.2039 | 52.9488 |
| `torchregress_NSF` | torchregress | 0.1873 ± 0.0661 † | 0.2975 ± 0.1550 | 0.0449 ± 0.0145 | 0.8979 ± 0.0090 | 0.9187 ± 0.0146 | 0.0943 ± 0.0256 | 327.4 |
| `tr_PITRecal+GaussCopula+KumoTabular` | torchregress | 0.0314 ± 0.0049 † | 0.0175 ± 0.0042 | 0.0073 ± 0.0009 | 0.8955 ± 0.0029 | - | 0.0317 ± 0.0008 | 38.5831 |
| `tr_PITRecal+GaussCopula+LightGBM` | torchregress | 0.2299 ± 0.0205 † | 0.3694 ± 0.0411 | 0.0451 ± 0.0043 | 0.8920 ± 0.0010 | - | 0.2011 ± 0.0125 | 89.7409 |
| `tr_PITRecal+GaussCopula+TabICL` | torchregress | 0.0334 ± 0.0057 † | 0.0191 ± 0.0052 | 0.0077 ± 0.0011 | 0.8926 ± 0.0045 | - | 0.0348 ± 0.0023 | 47.7815 |
| `tr_PITRecal+GaussCopula+TabPFN` | torchregress | 0.0338 ± 0.0054 † | 0.0191 ± 0.0048 | 0.0078 ± 0.0010 | 0.8934 ± 0.0041 | - | 0.0342 ± 0.0060 | 105.3 |
| `tr_PITRecal+GaussCopula+TabularMLP` | torchregress | 0.1474 ± 0.0309 † | 0.2018 ± 0.0696 | 0.0330 ± 0.0049 | 0.8944 ± 0.0034 | - | 0.1041 ± 0.0158 | 357.0 |

‡ ran on CPU after a CUDA failure (cpu retry).

##### fm_multitarget / mt_scm20d

| Method | Library | ES | VS | CRPS | Cov90_marg | Cov90_joint | CorrErr | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussCopula+KumoTabular` | sdm | **0.6087 ± 0.0116** | **4.1997 ± 0.0833** | 0.1125 ± 0.0016 | 0.8874 ± 0.0015 | - | 0.0607 ± 0.0037 | 79.1516 |
| `GaussCopula+LightGBM` | lightgbm | 1.2704 ± 0.0162 | 9.2062 ± 0.0564 | 0.2466 ± 0.0037 | 0.8322 ± 0.0044 | - | 0.0943 ± 0.0120 | 182.3 |
| `GaussCopula+TabFM` ‡ | sdm | 0.6896 ± 0.0046 | 5.0944 ± 0.0596 | 0.1298 ± 0.0009 | 0.8916 ± 0.0024 | - | **0.0570 ± 0.0031** | 723.7 |
| `GaussCopula+TabICL` | tabicl | 0.6490 ± 0.0057 | 4.5188 ± 0.0694 | 0.1208 ± 0.0005 | 0.8906 ± 0.0024 | - | 0.0603 ± 0.0031 | 91.4949 |
| `GaussCopula+TabPFN` | tabpfn | 0.6263 ± 0.0093 | 4.3423 ± 0.0723 | 0.1162 ± 0.0013 | 0.9003 ± 0.0021 | - | 0.0579 ± 0.0026 | 206.4 |
| `GaussCopula+TabularMLP` | torchregress | 0.9083 ± 0.0180 | 6.6549 ± 0.1748 | 0.1807 ± 0.0039 | 0.9086 ± 0.0038 | - | 0.0586 ± 0.0056 | 538.8 |
| `Indep+KumoTabular` | sdm | 0.6099 ± 0.0118 | 4.2388 ± 0.0861 | **0.1125 ± 0.0016** | 0.8876 ± 0.0010 | - | 0.1037 ± 0.0023 | 79.1516 |
| `Indep+LightGBM` | lightgbm | 1.3351 ± 0.0162 | 10.4246 ± 0.0685 | 0.2464 ± 0.0036 | 0.8316 ± 0.0035 | - | 0.4292 ± 0.0127 | 182.3 |
| `Indep+TabFM` ‡ | sdm | 0.6921 ± 0.0043 | 5.1856 ± 0.0593 | 0.1298 ± 0.0009 | 0.8909 ± 0.0018 | - | 0.1138 ± 0.0064 | 723.7 |
| `Indep+TabICL` | tabicl | 0.6509 ± 0.0062 | 4.5844 ± 0.0731 | 0.1208 ± 0.0005 | 0.8909 ± 0.0034 | - | 0.1207 ± 0.0059 | 91.4949 |
| `Indep+TabPFN` | tabpfn | 0.6283 ± 0.0097 | 4.4118 ± 0.0745 | 0.1161 ± 0.0013 | 0.9009 ± 0.0031 | - | 0.1139 ± 0.0053 | 206.4 |
| `Indep+TabularMLP` | torchregress | 0.9192 ± 0.0183 | 7.1239 ± 0.1952 | 0.1807 ± 0.0039 | 0.9085 ± 0.0037 | - | 0.2849 ± 0.0311 | 538.8 |
| `torchregress_MDN` | torchregress | 1.4872 ± 0.0378 | 9.8181 ± 0.1156 | 0.3135 ± 0.0083 | 0.8932 ± 0.0056 | 0.7727 ± 0.0199 | 0.0656 ± 0.0106 | 50.5097 |
| `torchregress_NSF` ‡ | torchregress | 0.9878 ± 0.0076 | 6.6976 ± 0.0477 | 0.1994 ± 0.0027 | 0.8557 ± 0.0074 | 0.7247 ± 0.0535 | 0.0604 ± 0.0132 | 77.5659 |
| `tr_PITRecal+GaussCopula+KumoTabular` | torchregress | 0.6089 ± 0.0120 | 4.2020 ± 0.0857 | 0.1126 ± 0.0017 | 0.8871 ± 0.0014 | - | 0.0601 ± 0.0034 | 79.1516 |
| `tr_PITRecal+GaussCopula+LightGBM` | torchregress | 1.3311 ± 0.0144 | 10.0594 ± 0.0637 | 0.2518 ± 0.0034 | 0.8887 ± 0.0043 | - | 0.1144 ± 0.0093 | 182.3 |
| `tr_PITRecal+GaussCopula+TabICL` | torchregress | 0.6493 ± 0.0060 | 4.5223 ± 0.0747 | 0.1209 ± 0.0005 | 0.8870 ± 0.0029 | - | 0.0591 ± 0.0027 | 91.4949 |
| `tr_PITRecal+GaussCopula+TabPFN` | torchregress | 0.6248 ± 0.0096 | 4.3204 ± 0.0773 | 0.1161 ± 0.0013 | 0.8879 ± 0.0007 | - | 0.0573 ± 0.0021 | 206.4 |
| `tr_PITRecal+GaussCopula+TabularMLP` | torchregress | 0.9054 ± 0.0184 | 6.5590 ± 0.1565 | 0.1800 ± 0.0041 | 0.8894 ± 0.0027 | - | 0.0584 ± 0.0057 | 538.8 |

‡ ran on CPU after a CUDA failure (cpu retry).

#### fm_uq_scenarios (fm)

Group `fm`; 54 dataset(s). Columns: Coverage90 ·, Width ·, IntervalScore ↓, RMSE ↓, WorstDecileCoverage · (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: IntervalScore.

##### fm_uq_scenarios / california_housing

_No successful rows._

##### fm_uq_scenarios / california_housing@clean

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [clean]` | sdm | 0.8933 ± 0.0127 | 0.8945 ± 0.0339 | 1.2997 ± 0.0721 † | 0.3858 ± 0.0159 | 0.8478 ± 0.0308 | 2.2332 |
| `Raw+LightGBM [clean]` | lightgbm | 0.8300 ± 0.0056 | 1.5006 ± 0.0212 | 2.2182 ± 0.0723 | 0.4881 ± 0.0179 | 0.4543 ± 0.0296 | 2.0354 |
| `Raw+TabICL [clean]` | tabicl | 0.8880 ± 0.0191 | 0.9960 ± 0.0580 | 1.4704 ± 0.0237 | 0.4124 ± 0.0035 | 0.7828 ± 0.0184 | 1.8248 |
| `Raw+TabularMLP [clean]` | torchregress | 0.9140 ± 0.0000 | 1.4305 ± 0.0149 | 2.0198 ± 0.0192 | 0.5022 ± 0.0087 | 0.7176 ± 0.0204 | 24.5051 |
| `tr_CQR+KumoTabular [clean]` | torchregress | 0.9000 ± 0.0200 | 0.9099 ± 0.0279 | 1.3005 ± 0.0736 † | 0.3858 ± 0.0159 | 0.8577 ± 0.0365 | 2.2332 |
| `tr_CQR+LightGBM [clean]` | torchregress | 0.8993 ± 0.0035 | 1.5922 ± 0.0201 | 2.1962 ± 0.0658 | 0.4881 ± 0.0179 | 0.6845 ± 0.0549 | 2.0354 |
| `tr_CQR+TabICL [clean]` | torchregress | 0.8980 ± 0.0217 | 1.0161 ± 0.0400 | 1.4691 ± 0.0265 | 0.4124 ± 0.0035 | 0.8164 ± 0.0260 | 1.8248 |
| `tr_CQR+TabularMLP [clean]` | torchregress | 0.8910 ± 0.0111 | 1.3403 ± 0.0382 | 2.0183 ± 0.0159 | 0.5022 ± 0.0087 | 0.6967 ± 0.0337 | 24.5051 |
| `tr_DCP+KumoTabular [clean]` | torchregress | 0.9000 ± 0.0182 | 0.9210 ± 0.0335 | **1.2992 ± 0.0763** † | 0.3858 ± 0.0159 | 0.8573 ± 0.0318 | 2.2332 |
| `tr_DCP+LightGBM [clean]` | torchregress | 0.8930 ± 0.0078 | 2.9702 ± 0.1511 | 3.1421 ± 0.0949 | 0.4881 ± 0.0179 | 0.5939 ± 0.0122 | 2.0354 |
| `tr_DCP+TabICL [clean]` | torchregress | 0.8960 ± 0.0200 | 1.0236 ± 0.0385 | 1.4667 ± 0.0303 | 0.4124 ± 0.0035 | 0.8020 ± 0.0230 | 1.8248 |
| `tr_DCP+TabularMLP [clean]` | torchregress | 0.8877 ± 0.0075 | 1.3073 ± 0.0349 | 2.0265 ± 0.0115 | 0.5022 ± 0.0087 | 0.6875 ± 0.0311 | 24.5051 |
| `tr_Mondrian+KumoTabular [clean]` | torchregress | 0.9027 ± 0.0102 | 1.0128 ± 0.0057 | 1.7444 ± 0.0868 | 0.3858 ± 0.0159 | 0.7955 ± 0.0156 | 2.2332 |
| `tr_Mondrian+LightGBM [clean]` | torchregress | 0.8907 ± 0.0206 | 1.3074 ± 0.0531 | 2.1756 ± 0.1072 | 0.4881 ± 0.0179 | 0.6801 ± 0.0577 | 2.0354 |
| `tr_Mondrian+TabFM [clean]` | torchregress | 0.8983 ± 0.0111 | 1.0048 ± 0.0116 | 1.6956 ± 0.1056 † | **0.3799 ± 0.0206** | 0.8026 ± 0.0390 | 8.2082 |
| `tr_Mondrian+TabICL [clean]` | torchregress | 0.8963 ± 0.0107 | 1.0874 ± 0.0393 | 1.8774 ± 0.0278 | 0.4124 ± 0.0035 | 0.7802 ± 0.0138 | 1.8248 |
| `tr_Mondrian+TabularMLP [clean]` | torchregress | 0.8897 ± 0.0153 | 1.3911 ± 0.0258 | 2.1637 ± 0.0299 | 0.5022 ± 0.0087 | 0.6835 ± 0.0522 | 24.5051 |
| `tr_Normalized+KumoTabular [clean]` | torchregress | 0.9030 ± 0.0149 | 0.9522 ± 0.0430 | 1.4576 ± 0.0625 | 0.3858 ± 0.0159 | 0.7853 ± 0.0335 | 2.2332 |
| `tr_Normalized+LightGBM [clean]` | torchregress | 0.8950 ± 0.0140 | 1.2706 ± 0.0069 | 2.1006 ± 0.0817 | 0.4881 ± 0.0179 | 0.6805 ± 0.0479 | 2.0354 |
| `tr_Normalized+TabICL [clean]` | torchregress | 0.8977 ± 0.0142 | 1.0001 ± 0.0479 | 1.6178 ± 0.0337 | 0.4124 ± 0.0035 | 0.7769 ± 0.0141 | 1.8248 |
| `tr_Normalized+TabularMLP [clean]` | torchregress | 0.8877 ± 0.0075 | 1.3073 ± 0.0349 | 2.0265 ± 0.0115 | 0.5022 ± 0.0087 | 0.6875 ± 0.0311 | 24.5051 |
| `tr_PITRecal+KumoTabular [clean]` | torchregress | 0.9013 ± 0.0231 | 0.9233 ± 0.0388 | 1.3003 ± 0.0762 † | 0.3862 ± 0.0161 | 0.8579 ± 0.0383 | 2.2332 |
| `tr_PITRecal+LightGBM [clean]` | torchregress | 0.8963 ± 0.0099 | 2.3624 ± 0.2339 | 2.6704 ± 0.1532 † | 0.4863 ± 0.0177 | 0.5310 ± 0.0460 | 2.0354 |
| `tr_PITRecal+TabICL [clean]` | torchregress | 0.8967 ± 0.0199 | 1.0234 ± 0.0400 | 1.4676 ± 0.0291 | 0.4132 ± 0.0057 | 0.8080 ± 0.0266 | 1.8248 |
| `tr_PITRecal+TabularMLP [clean]` | torchregress | 0.8843 ± 0.0074 | 1.3414 ± 0.0556 | 1.9979 ± 0.0187 | 0.5103 ± 0.0116 | 0.7181 ± 0.0319 | 24.5051 |
| `tr_ResidualDist+KumoTabular [clean]` | torchregress | 0.8953 ± 0.0023 | 1.0473 ± 0.0452 | 1.8842 ± 0.0716 | 0.3859 ± 0.0160 | 0.6789 ± 0.0351 | 2.2332 |
| `tr_ResidualDist+LightGBM [clean]` | torchregress | 0.8927 ± 0.0057 | 1.4046 ± 0.0207 | 2.3284 ± 0.1075 | 0.4877 ± 0.0179 | 0.6179 ± 0.0375 | 2.0354 |
| `tr_ResidualDist+TabFM [clean]` | torchregress | 0.8993 ± 0.0110 | 1.0915 ± 0.0802 | 1.8571 ± 0.0991 † | 0.3814 ± 0.0215 | 0.6860 ± 0.0675 | 8.2082 |
| `tr_ResidualDist+TabICL [clean]` | torchregress | 0.9040 ± 0.0087 | 1.1818 ± 0.0391 | 2.0233 ± 0.0254 | 0.4125 ± 0.0041 | 0.6988 ± 0.0595 | 1.8248 |
| `tr_ResidualDist+TabularMLP [clean]` | torchregress | 0.8890 ± 0.0066 | 1.5033 ± 0.0395 | 2.4082 ± 0.0478 | 0.5053 ± 0.0096 | 0.5999 ± 0.0284 | 24.5051 |
| `tr_Split+KumoTabular [clean]` | torchregress | 0.9013 ± 0.0042 | 1.0458 ± 0.0591 | 1.9131 ± 0.0750 | 0.3858 ± 0.0159 | 0.6619 ± 0.0177 | 2.2332 |
| `tr_Split+LightGBM [clean]` | torchregress | 0.8957 ± 0.0116 | 1.4071 ± 0.0258 | 2.4008 ± 0.1264 † | 0.4881 ± 0.0179 | 0.5589 ± 0.0533 | 2.0354 |
| `tr_Split+TabFM [clean]` | torchregress | 0.9010 ± 0.0108 | 1.0827 ± 0.0663 | 1.8602 ± 0.0993 † | **0.3799 ± 0.0206** | 0.6678 ± 0.0508 | 8.2082 |
| `tr_Split+TabICL [clean]` | torchregress | 0.8983 ± 0.0080 | 1.1509 ± 0.0567 | 2.0596 ± 0.0099 | 0.4124 ± 0.0035 | 0.6534 ± 0.0141 | 1.8248 |
| `tr_Split+TabularMLP [clean]` | torchregress | 0.8927 ± 0.0110 | 1.4669 ± 0.0341 | 2.4345 ± 0.0598 | 0.5022 ± 0.0087 | 0.5547 ± 0.0075 | 24.5051 |

##### fm_uq_scenarios / california_housing@extrapolate

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [extrapolate]` | sdm | 0.9360 ± 0.0046 | 1.6564 ± 0.0369 | **1.8464 ± 0.0101** | 0.4809 ± 0.0155 | 0.8697 ± 0.0279 | 1.8082 |
| `Raw+LightGBM [extrapolate]` | lightgbm | 0.7900 ± 0.0114 | 1.6482 ± 0.0560 | 2.5024 ± 0.0670 | 0.5605 ± 0.0132 | 0.3749 ± 0.0095 | 1.8767 |
| `Raw+TabICL [extrapolate]` | tabicl | 0.9407 ± 0.0076 | 1.9352 ± 0.0786 | 2.1301 ± 0.0511 | 0.4958 ± 0.0096 | 0.8465 ± 0.0422 | 1.7219 |
| `Raw+TabularMLP [extrapolate]` | torchregress | 0.9040 ± 0.0166 | 1.7948 ± 0.0893 | 2.4701 ± 0.0116 | 0.5898 ± 0.0234 | 0.7742 ± 0.0264 | 24.9950 |
| `tr_CQR+KumoTabular [extrapolate]` | torchregress | 0.9393 ± 0.0059 | 1.6690 ± 0.0525 | 1.8511 ± 0.0168 | 0.4809 ± 0.0155 | 0.8770 ± 0.0318 | 1.8082 |
| `tr_CQR+LightGBM [extrapolate]` | torchregress | 0.8737 ± 0.0035 | 1.7675 ± 0.0505 | 2.4433 ± 0.0622 | 0.5605 ± 0.0132 | 0.5282 ± 0.0721 | 1.8767 |
| `tr_CQR+TabICL [extrapolate]` | torchregress | 0.9447 ± 0.0059 | 1.9465 ± 0.0784 | 2.1346 ± 0.0543 | 0.4958 ± 0.0096 | 0.8658 ± 0.0423 | 1.7219 |
| `tr_CQR+TabularMLP [extrapolate]` | torchregress | 0.8847 ± 0.0117 | 1.7219 ± 0.0710 | 2.4725 ± 0.0176 | 0.5898 ± 0.0234 | 0.7380 ± 0.0353 | 24.9950 |
| `tr_DCP+KumoTabular [extrapolate]` | torchregress | 0.9407 ± 0.0080 | 1.7043 ± 0.1050 | 1.8756 ± 0.0499 | 0.4809 ± 0.0155 | 0.8749 ± 0.0363 | 1.8082 |
| `tr_DCP+LightGBM [extrapolate]` | torchregress | 0.8523 ± 0.0378 | 2.3416 ± 0.1425 | 2.7515 ± 0.0738 | 0.5605 ± 0.0132 | 0.4654 ± 0.1205 | 1.8767 |
| `tr_DCP+TabICL [extrapolate]` | torchregress | 0.9453 ± 0.0050 | 1.9859 ± 0.1041 | 2.1596 ± 0.0800 | 0.4958 ± 0.0096 | 0.8668 ± 0.0524 | 1.7219 |
| `tr_DCP+TabularMLP [extrapolate]` | torchregress | 0.8780 ± 0.0089 | 1.6559 ± 0.0457 | 2.4602 ± 0.0272 | 0.5898 ± 0.0234 | 0.7304 ± 0.0251 | 24.9950 |
| `tr_Mondrian+KumoTabular [extrapolate]` | torchregress | 0.8210 ± 0.0223 | 1.1834 ± 0.0579 | 2.3763 ± 0.1167 | 0.4809 ± 0.0155 | 0.5137 ± 0.0664 | 1.8082 |
| `tr_Mondrian+LightGBM [extrapolate]` | torchregress | 0.8667 ± 0.0211 | 1.6135 ± 0.0857 | 2.4992 ± 0.0400 | 0.5605 ± 0.0132 | 0.5571 ± 0.0515 | 1.8767 |
| `tr_Mondrian+TabFM [extrapolate]` | torchregress | 0.8360 ± 0.0182 | 1.2241 ± 0.0599 | 2.2366 ± 0.0628 | 0.4690 ± 0.0100 | 0.4230 ± 0.0439 | 8.2300 |
| `tr_Mondrian+TabICL [extrapolate]` | torchregress | 0.8467 ± 0.0295 | 1.3068 ± 0.0784 | 2.3550 ± 0.0814 | 0.4958 ± 0.0096 | 0.4707 ± 0.0874 | 1.7219 |
| `tr_Mondrian+TabularMLP [extrapolate]` | torchregress | 0.8380 ± 0.0207 | 1.7064 ± 0.0170 | 2.7140 ± 0.0880 | 0.5898 ± 0.0234 | 0.4366 ± 0.1425 | 24.9950 |
| `tr_Normalized+KumoTabular [extrapolate]` | torchregress | 0.9337 ± 0.0055 | 1.6968 ± 0.0460 | 2.0898 ± 0.0264 | 0.4809 ± 0.0155 | 0.8784 ± 0.0151 | 1.8082 |
| `tr_Normalized+LightGBM [extrapolate]` | torchregress | 0.8623 ± 0.0160 | 1.4551 ± 0.0557 | 2.4433 ± 0.0520 | 0.5605 ± 0.0132 | 0.7772 ± 0.0159 | 1.8767 |
| `tr_Normalized+TabICL [extrapolate]` | torchregress | 0.9490 ± 0.0056 | 1.9397 ± 0.0512 | 2.3232 ± 0.0056 | 0.4958 ± 0.0096 | 0.8946 ± 0.0105 | 1.7219 |
| `tr_Normalized+TabularMLP [extrapolate]` | torchregress | 0.8780 ± 0.0089 | 1.6559 ± 0.0457 | 2.4602 ± 0.0272 | 0.5898 ± 0.0234 | 0.7304 ± 0.0251 | 24.9950 |
| `tr_PITRecal+KumoTabular [extrapolate]` | torchregress | 0.9430 ± 0.0092 | 1.7107 ± 0.0924 | 1.8798 ± 0.0429 | 0.4817 ± 0.0140 | 0.8768 ± 0.0439 | 1.8082 |
| `tr_PITRecal+LightGBM [extrapolate]` | torchregress | 0.8587 ± 0.0281 | 2.1934 ± 0.0584 | 2.6910 ± 0.0414 | 0.5625 ± 0.0115 | 0.5480 ± 0.0501 | 1.8767 |
| `tr_PITRecal+TabICL [extrapolate]` | torchregress | 0.9433 ± 0.0074 | 1.9915 ± 0.0937 | 2.1663 ± 0.0729 | 0.4946 ± 0.0080 | 0.8525 ± 0.0707 | 1.7219 |
| `tr_PITRecal+TabularMLP [extrapolate]` | torchregress | 0.8863 ± 0.0049 | 1.7225 ± 0.0358 | 2.4630 ± 0.0057 | 0.5717 ± 0.0036 | 0.7785 ± 0.0273 | 24.9950 |
| `tr_ResidualDist+KumoTabular [extrapolate]` | torchregress | 0.7997 ± 0.0286 | 0.9868 ± 0.0290 | 2.4963 ± 0.1480 † | 0.4809 ± 0.0154 | 0.6911 ± 0.0165 | 1.8082 |
| `tr_ResidualDist+LightGBM [extrapolate]` | torchregress | 0.8547 ± 0.0175 | 1.4602 ± 0.0744 | 2.5618 ± 0.0810 | 0.5620 ± 0.0118 | 0.7606 ± 0.0193 | 1.8767 |
| `tr_ResidualDist+TabFM [extrapolate]` | torchregress | 0.8057 ± 0.0230 | 1.0097 ± 0.0359 | 2.3298 ± 0.0914 | **0.4668 ± 0.0104** | 0.6974 ± 0.0407 | 8.2300 |
| `tr_ResidualDist+TabICL [extrapolate]` | torchregress | 0.8010 ± 0.0036 | 1.0933 ± 0.0207 | 2.5641 ± 0.0431 | 0.4951 ± 0.0097 | 0.6347 ± 0.0133 | 1.7219 |
| `tr_ResidualDist+TabularMLP [extrapolate]` | torchregress | 0.8547 ± 0.0272 | 1.5933 ± 0.0740 | 2.7827 ± 0.1085 | 0.5844 ± 0.0161 | 0.7820 ± 0.0458 | 24.9950 |
| `tr_Split+KumoTabular [extrapolate]` | torchregress | 0.8093 ± 0.0163 | 0.9900 ± 0.0165 | 2.4986 ± 0.1262 † | 0.4809 ± 0.0155 | 0.6986 ± 0.0110 | 1.8082 |
| `tr_Split+LightGBM [extrapolate]` | torchregress | 0.8420 ± 0.0120 | 1.4006 ± 0.0633 | 2.6181 ± 0.0685 | 0.5605 ± 0.0132 | 0.7107 ± 0.0150 | 1.8767 |
| `tr_Split+TabFM [extrapolate]` | torchregress | 0.8050 ± 0.0220 | 0.9995 ± 0.0302 | 2.3214 ± 0.0966 | 0.4690 ± 0.0100 | 0.7085 ± 0.0445 | 8.2300 |
| `tr_Split+TabICL [extrapolate]` | torchregress | 0.8020 ± 0.0010 | 1.0366 ± 0.0083 | 2.5336 ± 0.0722 | 0.4958 ± 0.0096 | 0.6224 ± 0.0144 | 1.7219 |
| `tr_Split+TabularMLP [extrapolate]` | torchregress | 0.8497 ± 0.0313 | 1.5069 ± 0.0758 | 2.7431 ± 0.1143 | 0.5898 ± 0.0234 | 0.7622 ± 0.0414 | 24.9950 |

##### fm_uq_scenarios / california_housing@heavy_tail

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [heavy_tail]` | sdm | 0.8887 ± 0.0021 | 0.7449 ± 0.0511 | **1.1516 ± 0.0931** † | 0.5556 ± 0.0173 | 0.8429 ± 0.0099 | 1.7918 |
| `Raw+LightGBM [heavy_tail]` | lightgbm | 0.8410 ± 0.0070 | 1.3028 ± 0.0545 | 2.2804 ± 0.1740 † | 0.6765 ± 0.0391 | 0.4561 ± 0.0383 | 1.8326 |
| `Raw+TabICL [heavy_tail]` | tabicl | 0.8913 ± 0.0186 | 0.8524 ± 0.0583 | 1.4582 ± 0.0833 † | 0.6092 ± 0.0205 | 0.7791 ± 0.0324 | 1.6578 |
| `Raw+TabularMLP [heavy_tail]` | torchregress | 0.9523 ± 0.0021 | 2.2289 ± 0.1048 | 3.1416 ± 0.2232 † | 0.6838 ± 0.0401 | 0.7279 ± 0.0317 | 19.0766 |
| `tr_CQR+KumoTabular [heavy_tail]` | torchregress | 0.8897 ± 0.0186 | 0.7471 ± 0.0556 | 1.1521 ± 0.0928 † | 0.5556 ± 0.0173 | 0.8387 ± 0.0161 | 1.7918 |
| `tr_CQR+LightGBM [heavy_tail]` | torchregress | 0.8903 ± 0.0025 | 1.3322 ± 0.0530 | 2.2732 ± 0.1748 † | 0.6765 ± 0.0391 | 0.6437 ± 0.0136 | 1.8326 |
| `tr_CQR+TabICL [heavy_tail]` | torchregress | 0.8990 ± 0.0101 | 0.8611 ± 0.0468 | 1.4572 ± 0.0839 † | 0.6092 ± 0.0205 | 0.7914 ± 0.0221 | 1.6578 |
| `tr_CQR+TabularMLP [heavy_tail]` | torchregress | 0.9013 ± 0.0078 | 1.9941 ± 0.1486 | 3.0601 ± 0.2326 † | 0.6838 ± 0.0401 | 0.7048 ± 0.0539 | 19.0766 |
| `tr_DCP+KumoTabular [heavy_tail]` | torchregress | 0.8930 ± 0.0161 | 0.7559 ± 0.0706 | 1.1524 ± 0.0875 † | 0.5556 ± 0.0173 | 0.8428 ± 0.0273 | 1.7918 |
| `tr_DCP+LightGBM [heavy_tail]` | torchregress | 0.8870 ± 0.0131 | 3.6948 ± 0.4567 | 3.9099 ± 0.3298 † | 0.6765 ± 0.0391 | 0.5676 ± 0.0259 | 1.8326 |
| `tr_DCP+TabICL [heavy_tail]` | torchregress | 0.8973 ± 0.0121 | 0.8789 ± 0.0274 | 1.4468 ± 0.0919 † | 0.6092 ± 0.0205 | 0.7977 ± 0.0151 | 1.6578 |
| `tr_DCP+TabularMLP [heavy_tail]` | torchregress | 0.8990 ± 0.0137 | 1.4518 ± 0.1918 | 2.8281 ± 0.1949 † | 0.6838 ± 0.0401 | 0.5957 ± 0.0865 | 19.0766 |
| `tr_Mondrian+KumoTabular [heavy_tail]` | torchregress | 0.9030 ± 0.0052 | 1.0910 ± 0.1787 | 2.0416 ± 0.0602 | 0.5556 ± 0.0173 | 0.7736 ± 0.0464 | 1.7918 |
| `tr_Mondrian+LightGBM [heavy_tail]` | torchregress | 0.8913 ± 0.0133 | 1.3732 ± 0.0687 | 2.3658 ± 0.1445 † | 0.6765 ± 0.0391 | 0.6777 ± 0.0524 | 1.8326 |
| `tr_Mondrian+TabFM [heavy_tail]` | torchregress | 0.9047 ± 0.0040 | 0.9881 ± 0.0665 | 1.7405 ± 0.0839 | **0.5256 ± 0.0219** | 0.7673 ± 0.0300 | 8.2662 |
| `tr_Mondrian+TabICL [heavy_tail]` | torchregress | 0.8943 ± 0.0084 | 1.0413 ± 0.1760 | 2.2236 ± 0.0898 | 0.6092 ± 0.0205 | 0.6792 ± 0.0939 | 1.6578 |
| `tr_Mondrian+TabularMLP [heavy_tail]` | torchregress | 0.8937 ± 0.0137 | 1.3599 ± 0.0705 | 2.3774 ± 0.1643 † | 0.6838 ± 0.0401 | 0.6325 ± 0.0523 | 19.0766 |
| `tr_Normalized+KumoTabular [heavy_tail]` | torchregress | 0.8967 ± 0.0084 | 0.8092 ± 0.0579 | 1.6591 ± 0.0936 † | 0.5556 ± 0.0173 | 0.7143 ± 0.0300 | 1.7918 |
| `tr_Normalized+LightGBM [heavy_tail]` | torchregress | 0.8880 ± 0.0205 | 0.7933 ± 0.0779 | 2.8596 ± 0.2875 † | 0.6765 ± 0.0391 | 0.3966 ± 0.0801 | 1.8326 |
| `tr_Normalized+TabICL [heavy_tail]` | torchregress | 0.8987 ± 0.0064 | 0.8615 ± 0.0471 | 1.9810 ± 0.0975 | 0.6092 ± 0.0205 | 0.6608 ± 0.0282 | 1.6578 |
| `tr_Normalized+TabularMLP [heavy_tail]` | torchregress | 0.8990 ± 0.0137 | 1.4518 ± 0.1918 | 2.8281 ± 0.1949 † | 0.6838 ± 0.0401 | 0.5957 ± 0.0865 | 19.0766 |
| `tr_PITRecal+KumoTabular [heavy_tail]` | torchregress | 0.8933 ± 0.0182 | 0.7581 ± 0.0752 | 1.1533 ± 0.0858 † | 0.5554 ± 0.0164 | 0.8428 ± 0.0273 | 1.7918 |
| `tr_PITRecal+LightGBM [heavy_tail]` | torchregress | 0.8907 ± 0.0087 | 1.7828 ± 0.0719 | 2.5317 ± 0.1970 † | 0.6759 ± 0.0377 | 0.5034 ± 0.0331 | 1.8326 |
| `tr_PITRecal+TabICL [heavy_tail]` | torchregress | 0.8957 ± 0.0143 | 0.8738 ± 0.0152 | 1.4476 ± 0.0921 † | 0.6100 ± 0.0161 | 0.7955 ± 0.0156 | 1.6578 |
| `tr_PITRecal+TabularMLP [heavy_tail]` | torchregress | 0.8930 ± 0.0125 | 1.5559 ± 0.0993 | 2.6738 ± 0.2150 † | 0.8607 ± 0.1262 | 0.6749 ± 0.0444 | 19.0766 |
| `tr_ResidualDist+KumoTabular [heavy_tail]` | torchregress | 0.9013 ± 0.0067 | 0.6682 ± 0.0724 | 2.6192 ± 0.1350 † | 0.5556 ± 0.0172 | 0.4747 ± 0.0051 | 1.7918 |
| `tr_ResidualDist+LightGBM [heavy_tail]` | torchregress | 0.8990 ± 0.0104 | 1.3281 ± 0.1591 | 3.3592 ± 0.2241 † | 0.6765 ± 0.0390 | 0.4505 ± 0.0552 | 1.8326 |
| `tr_ResidualDist+TabFM [heavy_tail]` | torchregress | 0.8967 ± 0.0064 | 0.9584 ± 0.1319 | 2.7326 ± 0.1295 | 0.5264 ± 0.0219 | 0.3961 ± 0.0775 | 8.2662 |
| `tr_ResidualDist+TabICL [heavy_tail]` | torchregress | 0.8973 ± 0.0086 | 0.7798 ± 0.0675 | 2.9814 ± 0.1033 | 0.6094 ± 0.0200 | 0.4276 ± 0.0525 | 1.6578 |
| `tr_ResidualDist+TabularMLP [heavy_tail]` | torchregress | 0.8957 ± 0.0110 | 1.8085 ± 0.2020 | 3.4717 ± 0.2242 † | 0.6903 ± 0.0419 | 0.4555 ± 0.0398 | 19.0766 |
| `tr_Split+KumoTabular [heavy_tail]` | torchregress | 0.9013 ± 0.0049 | 0.5987 ± 0.0468 | 2.6576 ± 0.1549 † | 0.5556 ± 0.0173 | 0.4396 ± 0.0163 | 1.7918 |
| `tr_Split+LightGBM [heavy_tail]` | torchregress | 0.8953 ± 0.0140 | 0.9665 ± 0.1749 | 3.5945 ± 0.2427 † | 0.6765 ± 0.0391 | 0.2776 ± 0.0781 | 1.8326 |
| `tr_Split+TabFM [heavy_tail]` | torchregress | 0.8930 ± 0.0100 | 0.9352 ± 0.1364 | 2.7365 ± 0.1310 | **0.5256 ± 0.0219** | 0.3813 ± 0.0686 | 8.2662 |
| `tr_Split+TabICL [heavy_tail]` | torchregress | 0.8950 ± 0.0050 | 0.6723 ± 0.0517 | 3.0713 ± 0.1339 | 0.6092 ± 0.0205 | 0.3497 ± 0.0501 | 1.6578 |
| `tr_Split+TabularMLP [heavy_tail]` | torchregress | 0.8947 ± 0.0085 | 1.3840 ± 0.1313 | 3.6694 ± 0.2576 † | 0.6838 ± 0.0401 | 0.2957 ± 0.0280 | 19.0766 |

##### fm_uq_scenarios / california_housing@imbalance

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [imbalance]` | sdm | 0.8707 ± 0.0015 | 0.8550 ± 0.0409 | 1.6147 ± 0.0756 | 0.4547 ± 0.0280 | 0.7610 ± 0.0268 | 1.5485 |
| `Raw+LightGBM [imbalance]` | lightgbm | 0.7047 ± 0.0061 | 1.1326 ± 0.0079 | 3.4118 ± 0.2780 † | 0.6049 ± 0.0219 | 0.2876 ± 0.0344 | 1.9440 |
| `Raw+TabICL [imbalance]` | tabicl | 0.8417 ± 0.0031 | 0.9057 ± 0.0362 | 1.8223 ± 0.1706 † | 0.4850 ± 0.0323 | 0.6801 ± 0.0428 | 1.4783 |
| `Raw+TabularMLP [imbalance]` | torchregress | 0.8503 ± 0.0150 | 1.2162 ± 0.0032 | 2.4986 ± 0.1483 † | 0.5647 ± 0.0212 | 0.6308 ± 0.0402 | 22.4301 |
| `tr_CQR+KumoTabular [imbalance]` | torchregress | 0.8693 ± 0.0096 | 0.8605 ± 0.0292 | 1.6136 ± 0.0784 | 0.4547 ± 0.0280 | 0.7561 ± 0.0197 | 1.5485 |
| `tr_CQR+LightGBM [imbalance]` | torchregress | 0.7923 ± 0.0258 | 1.2766 ± 0.0472 | 3.2125 ± 0.3132 † | 0.6049 ± 0.0219 | 0.4034 ± 0.0328 | 1.9440 |
| `tr_CQR+TabICL [imbalance]` | torchregress | 0.8357 ± 0.0184 | 0.9049 ± 0.0249 | 1.8243 ± 0.1736 † | 0.4850 ± 0.0323 | 0.6745 ± 0.0541 | 1.4783 |
| `tr_CQR+TabularMLP [imbalance]` | torchregress | 0.8240 ± 0.0096 | 1.1588 ± 0.0189 | 2.5333 ± 0.1571 † | 0.5647 ± 0.0212 | 0.6021 ± 0.0344 | 22.4301 |
| `tr_DCP+KumoTabular [imbalance]` | torchregress | 0.8700 ± 0.0085 | 0.8631 ± 0.0232 | **1.6068 ± 0.0947** † | 0.4547 ± 0.0280 | 0.7611 ± 0.0251 | 1.5485 |
| `tr_DCP+LightGBM [imbalance]` | torchregress | 0.7897 ± 0.0193 | 1.5000 ± 0.0530 | 3.2234 ± 0.3268 † | 0.6049 ± 0.0219 | 0.4015 ± 0.0297 | 1.9440 |
| `tr_DCP+TabICL [imbalance]` | torchregress | 0.8327 ± 0.0257 | 0.8939 ± 0.0359 | 1.8360 ± 0.1821 † | 0.4850 ± 0.0323 | 0.6701 ± 0.0697 | 1.4783 |
| `tr_DCP+TabularMLP [imbalance]` | torchregress | 0.8217 ± 0.0122 | 1.1458 ± 0.0339 | 2.5537 ± 0.1805 † | 0.5647 ± 0.0212 | 0.5897 ± 0.0338 | 22.4301 |
| `tr_Mondrian+KumoTabular [imbalance]` | torchregress | 0.8187 ± 0.0090 | 0.7638 ± 0.0285 | 2.3799 ± 0.1856 † | 0.4547 ± 0.0280 | 0.5603 ± 0.0059 | 1.5485 |
| `tr_Mondrian+LightGBM [imbalance]` | torchregress | 0.8013 ± 0.0162 | 1.0534 ± 0.0512 | 3.2073 ± 0.2385 † | 0.6049 ± 0.0219 | 0.4442 ± 0.0288 | 1.9440 |
| `tr_Mondrian+TabFM [imbalance]` | torchregress | 0.8313 ± 0.0116 | 0.8126 ± 0.0661 | 2.2670 ± 0.1260 † | **0.4477 ± 0.0211** | 0.5874 ± 0.0192 | 7.4770 |
| `tr_Mondrian+TabICL [imbalance]` | torchregress | 0.8287 ± 0.0153 | 0.8730 ± 0.0554 | 2.5042 ± 0.1687 † | 0.4850 ± 0.0323 | 0.5898 ± 0.0164 | 1.4783 |
| `tr_Mondrian+TabularMLP [imbalance]` | torchregress | 0.8123 ± 0.0197 | 1.0765 ± 0.0620 | 2.8423 ± 0.0665 | 0.5647 ± 0.0212 | 0.5435 ± 0.0599 | 22.4301 |
| `tr_Normalized+KumoTabular [imbalance]` | torchregress | 0.8513 ± 0.0038 | 0.8615 ± 0.0414 | 1.8138 ± 0.0960 † | 0.4547 ± 0.0280 | 0.6796 ± 0.0445 | 1.5485 |
| `tr_Normalized+LightGBM [imbalance]` | torchregress | 0.8117 ± 0.0180 | 1.0854 ± 0.0723 | 2.9186 ± 0.2874 † | 0.6049 ± 0.0219 | 0.5075 ± 0.0396 | 1.9440 |
| `tr_Normalized+TabICL [imbalance]` | torchregress | 0.8470 ± 0.0066 | 0.9104 ± 0.0177 | 2.0408 ± 0.1993 † | 0.4850 ± 0.0323 | 0.6556 ± 0.0295 | 1.4783 |
| `tr_Normalized+TabularMLP [imbalance]` | torchregress | 0.8217 ± 0.0122 | 1.1458 ± 0.0339 | 2.5537 ± 0.1805 † | 0.5647 ± 0.0212 | 0.5897 ± 0.0338 | 22.4301 |
| `tr_PITRecal+KumoTabular [imbalance]` | torchregress | 0.8687 ± 0.0107 | 0.8568 ± 0.0251 | 1.6224 ± 0.1006 † | 0.4570 ± 0.0255 | 0.7574 ± 0.0202 | 1.5485 |
| `tr_PITRecal+LightGBM [imbalance]` | torchregress | 0.7947 ± 0.0206 | 1.4997 ± 0.0686 | 3.2309 ± 0.3747 † | 0.6105 ± 0.0196 | 0.4053 ± 0.0606 | 1.9440 |
| `tr_PITRecal+TabICL [imbalance]` | torchregress | 0.8357 ± 0.0229 | 0.9019 ± 0.0279 | 1.8306 ± 0.1989 † | 0.4886 ± 0.0266 | 0.6739 ± 0.0713 | 1.4783 |
| `tr_PITRecal+TabularMLP [imbalance]` | torchregress | 0.8293 ± 0.0170 | 1.1619 ± 0.0391 | 2.4606 ± 0.1895 † | 0.5939 ± 0.0230 | 0.6345 ± 0.0456 | 22.4301 |
| `tr_ResidualDist+KumoTabular [imbalance]` | torchregress | 0.8083 ± 0.0170 | 0.7389 ± 0.0390 | 2.4946 ± 0.2456 † | 0.4553 ± 0.0273 | 0.5139 ± 0.0329 | 1.5485 |
| `tr_ResidualDist+LightGBM [imbalance]` | torchregress | 0.7800 ± 0.0135 | 1.0035 ± 0.0281 | 3.3261 ± 0.2667 † | 0.6061 ± 0.0213 | 0.4022 ± 0.0317 | 1.9440 |
| `tr_ResidualDist+TabFM [imbalance]` | torchregress | 0.8037 ± 0.0067 | 0.7562 ± 0.0377 | 2.4831 ± 0.2112 † | 0.4536 ± 0.0217 | 0.4943 ± 0.0276 | 7.4770 |
| `tr_ResidualDist+TabICL [imbalance]` | torchregress | 0.8147 ± 0.0110 | 0.8241 ± 0.0247 | 2.6295 ± 0.2555 † | 0.4859 ± 0.0305 | 0.5309 ± 0.0353 | 1.4783 |
| `tr_ResidualDist+TabularMLP [imbalance]` | torchregress | 0.7933 ± 0.0070 | 1.0531 ± 0.0054 | 3.0937 ± 0.2386 † | 0.5777 ± 0.0210 | 0.4640 ± 0.0177 | 22.4301 |
| `tr_Split+KumoTabular [imbalance]` | torchregress | 0.7953 ± 0.0060 | 0.7266 ± 0.0323 | 2.5707 ± 0.2339 † | 0.4547 ± 0.0280 | 0.4782 ± 0.0160 | 1.5485 |
| `tr_Split+LightGBM [imbalance]` | torchregress | 0.7723 ± 0.0006 | 0.9919 ± 0.0117 | 3.5404 ± 0.2111 † | 0.6049 ± 0.0219 | 0.3423 ± 0.0401 | 1.9440 |
| `tr_Split+TabFM [imbalance]` | torchregress | 0.8017 ± 0.0046 | 0.7576 ± 0.0323 | 2.5001 ± 0.1670 † | **0.4477 ± 0.0211** | 0.4887 ± 0.0326 | 7.4770 |
| `tr_Split+TabICL [imbalance]` | torchregress | 0.8093 ± 0.0068 | 0.8267 ± 0.0198 | 2.7162 ± 0.2674 † | 0.4850 ± 0.0323 | 0.5028 ± 0.0356 | 1.4783 |
| `tr_Split+TabularMLP [imbalance]` | torchregress | 0.7943 ± 0.0075 | 1.0468 ± 0.0194 | 3.1329 ± 0.1479 | 0.5647 ± 0.0212 | 0.4553 ± 0.0285 | 22.4301 |

##### fm_uq_scenarios / california_housing@noise_all

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_all]` | sdm | 0.8960 ± 0.0096 | 2.1330 ± 0.0100 | **2.8107 ± 0.1755** † | 0.7261 ± 0.0307 | 0.6786 ± 0.0357 | 1.8141 |
| `Raw+LightGBM [noise_all]` | lightgbm | 0.8260 ± 0.0148 | 2.0995 ± 0.0173 | 3.1102 ± 0.1544 | 0.7447 ± 0.0306 | 0.3667 ± 0.0410 | 1.8510 |
| `Raw+TabICL [noise_all]` | tabicl | 0.8980 ± 0.0131 | 2.1733 ± 0.0057 | 2.8321 ± 0.1972 † | 0.7296 ± 0.0349 | 0.6907 ± 0.0423 | 1.6664 |
| `Raw+TabularMLP [noise_all]` | torchregress | 0.9180 ± 0.0123 | 2.3034 ± 0.0310 | 3.1112 ± 0.1424 | 0.7304 ± 0.0304 | 0.6225 ± 0.0198 | 17.4593 |
| `tr_CQR+KumoTabular [noise_all]` | torchregress | 0.8937 ± 0.0211 | 2.1372 ± 0.0306 | 2.8123 ± 0.1734 † | 0.7261 ± 0.0307 | 0.6544 ± 0.0710 | 1.8141 |
| `tr_CQR+LightGBM [noise_all]` | torchregress | 0.8980 ± 0.0281 | 2.3154 ± 0.0794 | 3.0497 ± 0.1208 | 0.7447 ± 0.0306 | 0.5966 ± 0.0742 | 1.8510 |
| `tr_CQR+TabICL [noise_all]` | torchregress | 0.8967 ± 0.0279 | 2.1788 ± 0.0401 | 2.8378 ± 0.1918 † | 0.7296 ± 0.0349 | 0.6752 ± 0.0891 | 1.6664 |
| `tr_CQR+TabularMLP [noise_all]` | torchregress | 0.8963 ± 0.0234 | 2.1688 ± 0.0887 | 3.1043 ± 0.1522 | 0.7304 ± 0.0304 | 0.5967 ± 0.0074 | 17.4593 |
| `tr_DCP+KumoTabular [noise_all]` | torchregress | 0.8957 ± 0.0191 | 2.1168 ± 0.0909 | 2.8170 ± 0.1703 † | 0.7261 ± 0.0307 | 0.6813 ± 0.0426 | 1.8141 |
| `tr_DCP+LightGBM [noise_all]` | torchregress | 0.9007 ± 0.0116 | 3.5113 ± 0.0280 | 3.6461 ± 0.0165 | 0.7447 ± 0.0306 | 0.5722 ± 0.0387 | 1.8510 |
| `tr_DCP+TabICL [noise_all]` | torchregress | 0.8987 ± 0.0300 | 2.1685 ± 0.1192 | 2.8485 ± 0.1841 † | 0.7296 ± 0.0349 | 0.6866 ± 0.0708 | 1.6664 |
| `tr_DCP+TabularMLP [noise_all]` | torchregress | 0.8957 ± 0.0244 | 2.1330 ± 0.1151 | 3.1012 ± 0.1568 † | 0.7304 ± 0.0304 | 0.5911 ± 0.0170 | 17.4593 |
| `tr_Mondrian+KumoTabular [noise_all]` | torchregress | 0.8967 ± 0.0201 | 2.1531 ± 0.0998 | 3.2719 ± 0.2419 † | 0.7261 ± 0.0307 | 0.5781 ± 0.0297 | 1.8141 |
| `tr_Mondrian+LightGBM [noise_all]` | torchregress | 0.9027 ± 0.0287 | 2.2149 ± 0.1676 | 3.3444 ± 0.1949 † | 0.7447 ± 0.0306 | 0.5766 ± 0.0452 | 1.8510 |
| `tr_Mondrian+TabFM [noise_all]` | torchregress | 0.8957 ± 0.0237 | 2.1207 ± 0.1263 | 3.0773 ± 0.2091 † | **0.7174 ± 0.0320** | 0.6137 ± 0.0233 | 8.2253 |
| `tr_Mondrian+TabICL [noise_all]` | torchregress | 0.8933 ± 0.0224 | 2.1540 ± 0.1294 | 3.3073 ± 0.2237 † | 0.7296 ± 0.0349 | 0.5781 ± 0.0297 | 1.6664 |
| `tr_Mondrian+TabularMLP [noise_all]` | torchregress | 0.9003 ± 0.0285 | 2.1455 ± 0.0554 | 3.1230 ± 0.1946 † | 0.7304 ± 0.0304 | 0.6023 ± 0.0225 | 17.4593 |
| `tr_Normalized+KumoTabular [noise_all]` | torchregress | 0.8977 ± 0.0108 | 2.1867 ± 0.0949 | 3.1824 ± 0.1686 † | 0.7261 ± 0.0307 | 0.5927 ± 0.0185 | 1.8141 |
| `tr_Normalized+LightGBM [noise_all]` | torchregress | 0.8980 ± 0.0311 | 2.2474 ± 0.1297 | 3.3177 ± 0.1696 † | 0.7447 ± 0.0306 | 0.5479 ± 0.0481 | 1.8510 |
| `tr_Normalized+TabICL [noise_all]` | torchregress | 0.9027 ± 0.0155 | 2.1898 ± 0.1030 | 3.1933 ± 0.2165 † | 0.7296 ± 0.0349 | 0.6143 ± 0.0123 | 1.6664 |
| `tr_Normalized+TabularMLP [noise_all]` | torchregress | 0.8957 ± 0.0244 | 2.1330 ± 0.1151 | 3.1012 ± 0.1568 † | 0.7304 ± 0.0304 | 0.5911 ± 0.0170 | 17.4593 |
| `tr_PITRecal+KumoTabular [noise_all]` | torchregress | 0.8890 ± 0.0218 | 2.0578 ± 0.0865 | 2.8384 ± 0.2014 † | 0.7257 ± 0.0300 | 0.6970 ± 0.0404 | 1.8141 |
| `tr_PITRecal+LightGBM [noise_all]` | torchregress | 0.8953 ± 0.0110 | 3.0406 ± 0.1699 | 3.3439 ± 0.1059 | 0.7433 ± 0.0329 | 0.4796 ± 0.0593 | 1.8510 |
| `tr_PITRecal+TabICL [noise_all]` | torchregress | 0.8937 ± 0.0315 | 2.1281 ± 0.1255 | 2.8601 ± 0.2035 † | 0.7294 ± 0.0341 | 0.6928 ± 0.0655 | 1.6664 |
| `tr_PITRecal+TabularMLP [noise_all]` | torchregress | 0.8993 ± 0.0143 | 2.2346 ± 0.0910 | 3.0116 ± 0.1425 | 0.7379 ± 0.0317 | 0.7042 ± 0.0252 | 17.4593 |
| `tr_ResidualDist+KumoTabular [noise_all]` | torchregress | 0.9023 ± 0.0067 | 2.3084 ± 0.0765 | 3.2490 ± 0.2036 † | 0.7260 ± 0.0305 | 0.6054 ± 0.0355 | 1.8141 |
| `tr_ResidualDist+LightGBM [noise_all]` | torchregress | 0.8980 ± 0.0151 | 2.3683 ± 0.1198 | 3.2806 ± 0.1825 † | 0.7447 ± 0.0314 | 0.5951 ± 0.0681 | 1.8510 |
| `tr_ResidualDist+TabFM [noise_all]` | torchregress | 0.8947 ± 0.0152 | 2.2662 ± 0.1111 | 3.1966 ± 0.2312 † | 0.7230 ± 0.0329 | 0.5857 ± 0.0540 | 8.2253 |
| `tr_ResidualDist+TabICL [noise_all]` | torchregress | 0.8957 ± 0.0074 | 2.2885 ± 0.0871 | 3.2718 ± 0.2340 † | 0.7294 ± 0.0344 | 0.5997 ± 0.0338 | 1.6664 |
| `tr_ResidualDist+TabularMLP [noise_all]` | torchregress | 0.9010 ± 0.0142 | 2.3339 ± 0.1165 | 3.2220 ± 0.2206 † | 0.7368 ± 0.0319 | 0.5942 ± 0.0500 | 17.4593 |
| `tr_Split+KumoTabular [noise_all]` | torchregress | 0.8960 ± 0.0238 | 2.2600 ± 0.1601 | 3.4186 ± 0.2072 † | 0.7261 ± 0.0307 | 0.4955 ± 0.0154 | 1.8141 |
| `tr_Split+LightGBM [noise_all]` | torchregress | 0.8973 ± 0.0199 | 2.2756 ± 0.1448 | 3.4693 ± 0.2148 † | 0.7447 ± 0.0306 | 0.4581 ± 0.0327 | 1.8510 |
| `tr_Split+TabFM [noise_all]` | torchregress | 0.8887 ± 0.0267 | 2.1621 ± 0.1508 | 3.2682 ± 0.2371 † | **0.7174 ± 0.0320** | 0.5087 ± 0.0335 | 8.2253 |
| `tr_Split+TabICL [noise_all]` | torchregress | 0.8957 ± 0.0231 | 2.2502 ± 0.1447 | 3.4260 ± 0.2407 † | 0.7296 ± 0.0349 | 0.5105 ± 0.0183 | 1.6664 |
| `tr_Split+TabularMLP [noise_all]` | torchregress | 0.8957 ± 0.0211 | 2.2065 ± 0.1271 | 3.2915 ± 0.2209 † | 0.7304 ± 0.0304 | 0.5077 ± 0.0219 | 17.4593 |

##### fm_uq_scenarios / california_housing@noise_test

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_test]` | sdm | 0.8630 ± 0.0422 | 2.7563 ± 0.3692 | 4.0288 ± 0.0299 | 0.9539 ± 0.0114 | 0.7934 ± 0.0574 | 1.7896 |
| `Raw+LightGBM [noise_test]` | lightgbm | 0.7420 ± 0.0140 | 1.7796 ± 0.0735 | 3.3818 ± 0.0717 | 0.8224 ± 0.0175 | 0.3732 ± 0.0188 | 1.8444 |
| `Raw+TabICL [noise_test]` | tabicl | 0.8677 ± 0.0316 | 2.6448 ± 0.1909 | 3.7771 ± 0.1679 | 0.8619 ± 0.0157 | 0.8146 ± 0.0184 | 1.6868 |
| `Raw+TabularMLP [noise_test]` | torchregress | 0.6670 ± 0.0062 | 1.7963 ± 0.0354 | 5.8204 ± 0.1092 | 0.9563 ± 0.0175 | 0.4555 ± 0.0388 | 23.9784 |
| `tr_CQR+KumoTabular [noise_test]` | torchregress | 0.8663 ± 0.0372 | 2.7717 ± 0.3518 | 4.0183 ± 0.0225 | 0.9539 ± 0.0114 | 0.7902 ± 0.0522 | 1.7896 |
| `tr_CQR+LightGBM [noise_test]` | torchregress | 0.7963 ± 0.0114 | 1.8712 ± 0.0644 | 3.2690 ± 0.0536 | 0.8224 ± 0.0175 | 0.5518 ± 0.0262 | 1.8444 |
| `tr_CQR+TabICL [noise_test]` | torchregress | 0.8710 ± 0.0270 | 2.6650 ± 0.1672 | 3.7655 ± 0.1512 | 0.8619 ± 0.0157 | 0.8146 ± 0.0184 | 1.6868 |
| `tr_CQR+TabularMLP [noise_test]` | torchregress | 0.6297 ± 0.0070 | 1.7061 ± 0.0121 | 6.0472 ± 0.0577 | 0.9563 ± 0.0175 | 0.4466 ± 0.0383 | 23.9784 |
| `tr_DCP+KumoTabular [noise_test]` | torchregress | 0.8750 ± 0.0295 | 2.8186 ± 0.2952 | 3.9794 ± 0.0306 | 0.9539 ± 0.0114 | 0.8088 ± 0.0323 | 1.7896 |
| `tr_DCP+LightGBM [noise_test]` | torchregress | 0.8547 ± 0.0029 | 3.0866 ± 0.1543 | 3.4684 ± 0.0830 | 0.8224 ± 0.0175 | 0.5007 ± 0.0304 | 1.8444 |
| `tr_DCP+TabICL [noise_test]` | torchregress | 0.8750 ± 0.0220 | 2.7016 ± 0.1216 | 3.7473 ± 0.1293 | 0.8619 ± 0.0157 | 0.8211 ± 0.0147 | 1.6868 |
| `tr_DCP+TabularMLP [noise_test]` | torchregress | 0.6270 ± 0.0046 | 1.6412 ± 0.0166 | 6.0307 ± 0.0709 | 0.9563 ± 0.0175 | 0.4438 ± 0.0430 | 23.9784 |
| `tr_Mondrian+KumoTabular [noise_test]` | torchregress | 0.4533 ± 0.0297 | 0.9578 ± 0.0419 | 7.6992 ± 0.2256 | 0.9539 ± 0.0114 | 0.3068 ± 0.0584 | 1.7896 |
| `tr_Mondrian+LightGBM [noise_test]` | torchregress | 0.6290 ± 0.0325 | 1.2915 ± 0.0469 | 5.0019 ± 0.2529 † | 0.8224 ± 0.0175 | 0.5428 ± 0.0282 | 1.8444 |
| `tr_Mondrian+TabFM [noise_test]` | torchregress | 0.4557 ± 0.0163 | 1.0227 ± 0.0630 | 6.8739 ± 0.1782 | 0.8959 ± 0.0092 | 0.2744 ± 0.0302 | 8.0608 |
| `tr_Mondrian+TabICL [noise_test]` | torchregress | 0.5203 ± 0.0045 | 1.0384 ± 0.0376 | 6.2620 ± 0.1494 | 0.8619 ± 0.0157 | 0.3376 ± 0.0119 | 1.6868 |
| `tr_Mondrian+TabularMLP [noise_test]` | torchregress | 0.5830 ± 0.0040 | 1.3251 ± 0.0248 | 6.1994 ± 0.0520 | 0.9563 ± 0.0175 | 0.4293 ± 0.0493 | 23.9784 |
| `tr_Normalized+KumoTabular [noise_test]` | torchregress | 0.8297 ± 0.0421 | 2.9276 ± 0.3156 | 4.8907 ± 0.1343 | 0.9539 ± 0.0114 | 0.5887 ± 0.0584 | 1.7896 |
| `tr_Normalized+LightGBM [noise_test]` | torchregress | 0.6837 ± 0.0299 | 1.5067 ± 0.0510 | 4.5364 ± 0.2377 † | 0.8224 ± 0.0175 | 0.5383 ± 0.0209 | 1.8444 |
| `tr_Normalized+TabICL [noise_test]` | torchregress | 0.8297 ± 0.0257 | 2.6535 ± 0.1080 | 4.4316 ± 0.1557 | 0.8619 ± 0.0157 | 0.6423 ± 0.0416 | 1.6868 |
| `tr_Normalized+TabularMLP [noise_test]` | torchregress | 0.6270 ± 0.0046 | 1.6412 ± 0.0166 | 6.0307 ± 0.0709 | 0.9563 ± 0.0175 | 0.4438 ± 0.0430 | 23.9784 |
| `tr_PITRecal+KumoTabular [noise_test]` | torchregress | 0.8763 ± 0.0292 | 2.8233 ± 0.2807 | 3.9622 ± 0.0528 | 0.9560 ± 0.0115 | 0.8116 ± 0.0301 | 1.7896 |
| `tr_PITRecal+LightGBM [noise_test]` | torchregress | 0.8553 ± 0.0032 | 2.5443 ± 0.2331 | **3.1600 ± 0.1194** | 0.8226 ± 0.0181 | 0.4752 ± 0.0327 | 1.8444 |
| `tr_PITRecal+TabICL [noise_test]` | torchregress | 0.8743 ± 0.0230 | 2.7016 ± 0.1237 | 3.7434 ± 0.1297 | 0.8663 ± 0.0101 | 0.8211 ± 0.0147 | 1.6868 |
| `tr_PITRecal+TabularMLP [noise_test]` | torchregress | 0.6117 ± 0.0049 | 1.6835 ± 0.0188 | 5.9485 ± 0.0659 | 0.9345 ± 0.0188 | 0.4667 ± 0.0203 | 23.9784 |
| `tr_ResidualDist+KumoTabular [noise_test]` | torchregress | 0.5140 ± 0.0332 | 1.0473 ± 0.0452 | 7.2981 ± 0.3890 † | 0.9541 ± 0.0110 | 0.2472 ± 0.0486 | 1.7896 |
| `tr_ResidualDist+LightGBM [noise_test]` | torchregress | 0.6583 ± 0.0335 | 1.4046 ± 0.0207 | 4.9721 ± 0.3249 † | **0.8221 ± 0.0178** | 0.4641 ± 0.0310 | 1.8444 |
| `tr_ResidualDist+TabFM [noise_test]` | torchregress | 0.5093 ± 0.0197 | 1.0915 ± 0.0802 | 6.8181 ± 0.2322 | 0.8964 ± 0.0083 | 0.2237 ± 0.0440 | 8.0608 |
| `tr_ResidualDist+TabICL [noise_test]` | torchregress | 0.5930 ± 0.0358 | 1.1818 ± 0.0391 | 5.7727 ± 0.3878 † | 0.8623 ± 0.0147 | 0.2576 ± 0.0183 | 1.6868 |
| `tr_ResidualDist+TabularMLP [noise_test]` | torchregress | 0.6463 ± 0.0115 | 1.5033 ± 0.0395 | 6.0480 ± 0.0935 | 0.9607 ± 0.0165 | 0.3355 ± 0.0765 | 23.9784 |
| `tr_Split+KumoTabular [noise_test]` | torchregress | 0.5123 ± 0.0365 | 1.0458 ± 0.0591 | 7.4437 ± 0.4077 † | 0.9539 ± 0.0114 | 0.2413 ± 0.0526 | 1.7896 |
| `tr_Split+LightGBM [noise_test]` | torchregress | 0.6693 ± 0.0246 | 1.4071 ± 0.0258 | 4.8859 ± 0.2436 | 0.8224 ± 0.0175 | 0.4190 ± 0.0167 | 1.8444 |
| `tr_Split+TabFM [noise_test]` | torchregress | 0.5063 ± 0.0218 | 1.0827 ± 0.0663 | 6.8123 ± 0.2267 | 0.8959 ± 0.0092 | 0.2083 ± 0.0584 | 8.0608 |
| `tr_Split+TabICL [noise_test]` | torchregress | 0.5803 ± 0.0272 | 1.1509 ± 0.0567 | 5.9975 ± 0.3824 † | 0.8619 ± 0.0157 | 0.2228 ± 0.0365 | 1.6868 |
| `tr_Split+TabularMLP [noise_test]` | torchregress | 0.6373 ± 0.0123 | 1.4669 ± 0.0341 | 6.1454 ± 0.1015 | 0.9563 ± 0.0175 | 0.2936 ± 0.0757 | 23.9784 |

##### fm_uq_scenarios / california_housing@outliers

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [outliers]` | sdm | 0.9733 ± 0.0025 | 1.8636 ± 0.0655 | 1.9876 ± 0.1001 † | 0.3939 ± 0.0132 | 0.9477 ± 0.0129 | 1.8096 |
| `Raw+LightGBM [outliers]` | lightgbm | 0.9163 ± 0.0055 | 2.4762 ± 0.0194 | 3.0395 ± 0.1047 | 0.4958 ± 0.0157 | 0.7154 ± 0.0185 | 2.1389 |
| `Raw+TabICL [outliers]` | tabicl | 0.9487 ± 0.0159 | 1.9699 ± 0.0706 | 2.1746 ± 0.0265 | 0.4284 ± 0.0063 | 0.9078 ± 0.0072 | 1.6903 |
| `Raw+TabularMLP [outliers]` | torchregress | 0.9973 ± 0.0029 | 5.9978 ± 0.1824 | 6.0232 ± 0.1542 | 0.6215 ± 0.0158 | 0.9763 ± 0.0255 | 11.9441 |
| `tr_CQR+KumoTabular [outliers]` | torchregress | 0.9510 ± 0.0096 | 1.7742 ± 0.0638 | 1.9304 ± 0.0969 † | 0.3939 ± 0.0132 | 0.8918 ± 0.0290 | 1.8096 |
| `tr_CQR+LightGBM [outliers]` | torchregress | 0.9480 ± 0.0062 | 2.6635 ± 0.0317 | 3.1006 ± 0.1113 | 0.4958 ± 0.0157 | 0.7726 ± 0.0262 | 2.1389 |
| `tr_CQR+TabICL [outliers]` | torchregress | 0.9400 ± 0.0123 | 1.9268 ± 0.0500 | 2.1532 ± 0.0199 | 0.4284 ± 0.0063 | 0.8958 ± 0.0180 | 1.6903 |
| `tr_CQR+TabularMLP [outliers]` | torchregress | 0.9440 ± 0.0115 | 2.5250 ± 0.1774 | 3.0412 ± 0.1436 | 0.6215 ± 0.0158 | 0.6952 ± 0.0482 | 11.9441 |
| `tr_DCP+KumoTabular [outliers]` | torchregress | 0.9447 ± 0.0101 | 1.2062 ± 0.1057 | **1.4538 ± 0.1361** † | 0.3939 ± 0.0132 | 0.8959 ± 0.0341 | 1.8096 |
| `tr_DCP+LightGBM [outliers]` | torchregress | 0.9407 ± 0.0046 | 3.0073 ± 0.0803 | 3.4034 ± 0.1155 | 0.4958 ± 0.0157 | 0.7639 ± 0.0319 | 2.1389 |
| `tr_DCP+TabICL [outliers]` | torchregress | 0.9403 ± 0.0119 | 1.7915 ± 0.0304 | 2.0372 ± 0.0368 | 0.4284 ± 0.0063 | 0.8814 ± 0.0049 | 1.6903 |
| `tr_DCP+TabularMLP [outliers]` | torchregress | 0.9427 ± 0.0121 | 2.4200 ± 0.0872 | 2.9687 ± 0.0979 | 0.6215 ± 0.0158 | 0.6392 ± 0.0370 | 11.9441 |
| `tr_Mondrian+KumoTabular [outliers]` | torchregress | 0.9473 ± 0.0108 | 1.4375 ± 0.0581 | 1.9002 ± 0.0794 | 0.3939 ± 0.0132 | 0.8785 ± 0.0133 | 1.8096 |
| `tr_Mondrian+LightGBM [outliers]` | torchregress | 0.9420 ± 0.0113 | 1.9215 ± 0.2280 | 2.4514 ± 0.1275 † | 0.4958 ± 0.0157 | 0.7673 ± 0.0463 | 2.1389 |
| `tr_Mondrian+TabFM [outliers]` | torchregress | 0.9483 ± 0.0068 | 1.5186 ± 0.0320 | 1.9393 ± 0.0209 | 0.4147 ± 0.0081 | 0.8965 ± 0.0192 | 7.9646 |
| `tr_Mondrian+TabICL [outliers]` | torchregress | 0.9397 ± 0.0087 | 1.4986 ± 0.0605 | 2.0264 ± 0.0202 | 0.4284 ± 0.0063 | 0.8313 ± 0.0141 | 1.6903 |
| `tr_Mondrian+TabularMLP [outliers]` | torchregress | 0.9450 ± 0.0115 | 2.3462 ± 0.1398 | 2.9132 ± 0.0785 | 0.6215 ± 0.0158 | 0.7068 ± 0.0356 | 11.9441 |
| `tr_Normalized+KumoTabular [outliers]` | torchregress | 0.9470 ± 0.0113 | 1.4679 ± 0.0968 | 1.7806 ± 0.0784 | 0.3939 ± 0.0132 | 0.8538 ± 0.1045 | 1.8096 |
| `tr_Normalized+LightGBM [outliers]` | torchregress | 0.9490 ± 0.0078 | 2.2107 ± 0.1160 | 2.7039 ± 0.1231 | 0.4958 ± 0.0157 | 0.8008 ± 0.0425 | 2.1389 |
| `tr_Normalized+TabICL [outliers]` | torchregress | 0.9383 ± 0.0031 | 1.7307 ± 0.0420 | 2.0781 ± 0.0806 | 0.4284 ± 0.0063 | 0.8418 ± 0.0425 | 1.6903 |
| `tr_Normalized+TabularMLP [outliers]` | torchregress | 0.9427 ± 0.0121 | 2.4200 ± 0.0872 | 2.9687 ± 0.0979 | 0.6215 ± 0.0158 | 0.6392 ± 0.0370 | 11.9441 |
| `tr_PITRecal+KumoTabular [outliers]` | torchregress | 0.9487 ± 0.0075 | 1.2281 ± 0.0714 | 1.4700 ± 0.1185 † | **0.3931 ± 0.0136** | 0.9091 ± 0.0203 | 1.8096 |
| `tr_PITRecal+LightGBM [outliers]` | torchregress | 0.9420 ± 0.0044 | 2.9681 ± 0.1574 | 3.3802 ± 0.1362 | 0.4969 ± 0.0152 | 0.7445 ± 0.0099 | 2.1389 |
| `tr_PITRecal+TabICL [outliers]` | torchregress | 0.9397 ± 0.0107 | 1.7906 ± 0.0414 | 2.0525 ± 0.0281 | 0.4307 ± 0.0032 | 0.8760 ± 0.0167 | 1.6903 |
| `tr_PITRecal+TabularMLP [outliers]` | torchregress | 0.9397 ± 0.0087 | 2.3238 ± 0.1077 | 2.7794 ± 0.0474 | 0.6200 ± 0.0095 | 0.7141 ± 0.0516 | 11.9441 |
| `tr_ResidualDist+KumoTabular [outliers]` | torchregress | 0.9473 ± 0.0090 | 1.5748 ± 0.1045 | 2.0595 ± 0.0987 | 0.3937 ± 0.0134 | 0.7792 ± 0.0315 | 1.8096 |
| `tr_ResidualDist+LightGBM [outliers]` | torchregress | 0.9370 ± 0.0066 | 1.9390 ± 0.0362 | 2.4610 ± 0.0455 | 0.4965 ± 0.0154 | 0.7214 ± 0.0145 | 2.1389 |
| `tr_ResidualDist+TabFM [outliers]` | torchregress | 0.9420 ± 0.0095 | 1.6259 ± 0.0886 | 2.1216 ± 0.0721 | 0.4137 ± 0.0069 | 0.7708 ± 0.0584 | 7.9646 |
| `tr_ResidualDist+TabICL [outliers]` | torchregress | 0.9430 ± 0.0026 | 1.6495 ± 0.0242 | 2.1982 ± 0.0380 | 0.4291 ± 0.0047 | 0.7511 ± 0.0332 | 1.6903 |
| `tr_ResidualDist+TabularMLP [outliers]` | torchregress | 0.9457 ± 0.0181 | 2.3866 ± 0.1281 | 2.8396 ± 0.0344 | 0.6200 ± 0.0099 | 0.7299 ± 0.0743 | 11.9441 |
| `tr_Split+KumoTabular [outliers]` | torchregress | 0.9457 ± 0.0025 | 1.5501 ± 0.0693 | 2.0805 ± 0.0795 | 0.3939 ± 0.0132 | 0.7532 ± 0.0185 | 1.8096 |
| `tr_Split+LightGBM [outliers]` | torchregress | 0.9450 ± 0.0069 | 1.9152 ± 0.0366 | 2.5344 ± 0.0664 | 0.4958 ± 0.0157 | 0.6685 ± 0.0351 | 2.1389 |
| `tr_Split+TabFM [outliers]` | torchregress | 0.9430 ± 0.0096 | 1.6274 ± 0.0866 | 2.1136 ± 0.0742 | 0.4147 ± 0.0081 | 0.7643 ± 0.0507 | 7.9646 |
| `tr_Split+TabICL [outliers]` | torchregress | 0.9393 ± 0.0051 | 1.6261 ± 0.0450 | 2.2295 ± 0.0640 | 0.4284 ± 0.0063 | 0.7123 ± 0.0428 | 1.6903 |
| `tr_Split+TabularMLP [outliers]` | torchregress | 0.9413 ± 0.0146 | 2.4353 ± 0.1630 | 3.0177 ± 0.1245 | 0.6215 ± 0.0158 | 0.6320 ± 0.0721 | 11.9441 |

##### fm_uq_scenarios / california_housing@shift

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [shift]` | sdm | 0.9060 ± 0.0118 | 0.9839 ± 0.0380 | **1.3136 ± 0.0370** | **0.3878 ± 0.0247** | 0.8573 ± 0.0070 | 1.7863 |
| `Raw+LightGBM [shift]` | lightgbm | 0.8507 ± 0.0107 | 2.0472 ± 0.8638 | 2.5011 ± 0.6317 † | 0.4756 ± 0.0087 | 0.5683 ± 0.0440 | 1.8032 |
| `Raw+TabICL [shift]` | tabicl | 0.8927 ± 0.0076 | 1.0920 ± 0.0239 | 1.4896 ± 0.0472 | 0.4215 ± 0.0152 | 0.8409 ± 0.0186 | 1.6777 |
| `Raw+TabularMLP [shift]` | torchregress | 0.9260 ± 0.0053 | 1.4858 ± 0.0088 | 1.9775 ± 0.0521 | 0.4919 ± 0.0030 | 0.8417 ± 0.0144 | 26.5717 |
| `tr_CQR+KumoTabular [shift]` | torchregress | 0.9050 ± 0.0257 | 0.9901 ± 0.0573 | 1.3173 ± 0.0313 | **0.3878 ± 0.0247** | 0.8506 ± 0.0355 | 1.7863 |
| `tr_CQR+LightGBM [shift]` | torchregress | 0.8987 ± 0.0121 | 2.0835 ± 0.8452 | 2.4963 ± 0.6354 † | 0.4756 ± 0.0087 | 0.7002 ± 0.0553 | 1.8032 |
| `tr_CQR+TabICL [shift]` | torchregress | 0.8967 ± 0.0125 | 1.0971 ± 0.0213 | 1.4898 ± 0.0481 | 0.4215 ± 0.0152 | 0.8351 ± 0.0171 | 1.6777 |
| `tr_CQR+TabularMLP [shift]` | torchregress | 0.8943 ± 0.0083 | 1.3654 ± 0.0314 | 1.9649 ± 0.0553 | 0.4919 ± 0.0030 | 0.8257 ± 0.0153 | 26.5717 |
| `tr_DCP+KumoTabular [shift]` | torchregress | 0.9023 ± 0.0290 | 0.9876 ± 0.0918 | 1.3229 ± 0.0290 | **0.3878 ± 0.0247** | 0.8473 ± 0.0407 | 1.7863 |
| `tr_DCP+LightGBM [shift]` | torchregress | 0.8897 ± 0.0197 | 2.8571 ± 0.5545 | 3.0407 ± 0.4253 † | 0.4756 ± 0.0087 | 0.6180 ± 0.0239 | 1.8032 |
| `tr_DCP+TabICL [shift]` | torchregress | 0.8960 ± 0.0090 | 1.1037 ± 0.0462 | 1.4905 ± 0.0493 | 0.4215 ± 0.0152 | 0.8390 ± 0.0108 | 1.6777 |
| `tr_DCP+TabularMLP [shift]` | torchregress | 0.8923 ± 0.0125 | 1.3190 ± 0.0540 | 1.9638 ± 0.0550 | 0.4919 ± 0.0030 | 0.7972 ± 0.0232 | 26.5717 |
| `tr_Mondrian+KumoTabular [shift]` | torchregress | 0.8757 ± 0.0208 | 0.9490 ± 0.0352 | 1.8369 ± 0.1085 † | **0.3878 ± 0.0247** | 0.8084 ± 0.0250 | 1.7863 |
| `tr_Mondrian+LightGBM [shift]` | torchregress | 0.8763 ± 0.0116 | 1.2754 ± 0.0610 | 2.1799 ± 0.0724 | 0.4756 ± 0.0087 | 0.8007 ± 0.0304 | 1.8032 |
| `tr_Mondrian+TabFM [shift]` | torchregress | 0.8727 ± 0.0142 | 0.9915 ± 0.0436 | 1.8261 ± 0.1237 † | 0.3901 ± 0.0308 | 0.7957 ± 0.0326 | 8.0697 |
| `tr_Mondrian+TabICL [shift]` | torchregress | 0.8733 ± 0.0140 | 1.0558 ± 0.0416 | 2.0095 ± 0.0744 | 0.4215 ± 0.0152 | 0.8077 ± 0.0280 | 1.6777 |
| `tr_Mondrian+TabularMLP [shift]` | torchregress | 0.8847 ± 0.0110 | 1.3294 ± 0.0376 | 2.1558 ± 0.0200 | 0.4919 ± 0.0030 | 0.8055 ± 0.0331 | 26.5717 |
| `tr_Normalized+KumoTabular [shift]` | torchregress | 0.9013 ± 0.0207 | 0.9929 ± 0.0668 | 1.4748 ± 0.0442 | **0.3878 ± 0.0247** | 0.8193 ± 0.0398 | 1.7863 |
| `tr_Normalized+LightGBM [shift]` | torchregress | 0.8873 ± 0.0112 | 1.2794 ± 0.0883 | 2.1658 ± 0.1480 † | 0.4756 ± 0.0087 | 0.7154 ± 0.0907 | 1.8032 |
| `tr_Normalized+TabICL [shift]` | torchregress | 0.9067 ± 0.0015 | 1.0970 ± 0.0249 | 1.6569 ± 0.0313 | 0.4215 ± 0.0152 | 0.8225 ± 0.0076 | 1.6777 |
| `tr_Normalized+TabularMLP [shift]` | torchregress | 0.8923 ± 0.0125 | 1.3190 ± 0.0540 | 1.9638 ± 0.0550 | 0.4919 ± 0.0030 | 0.7972 ± 0.0232 | 26.5717 |
| `tr_PITRecal+KumoTabular [shift]` | torchregress | 0.9040 ± 0.0298 | 0.9857 ± 0.0772 | 1.3220 ± 0.0258 | 0.3882 ± 0.0251 | 0.8604 ± 0.0216 | 1.7863 |
| `tr_PITRecal+LightGBM [shift]` | torchregress | 0.8940 ± 0.0154 | 2.2461 ± 0.4671 | 2.5499 ± 0.3376 † | 0.4766 ± 0.0078 | 0.5463 ± 0.0708 | 1.8032 |
| `tr_PITRecal+TabICL [shift]` | torchregress | 0.8977 ± 0.0121 | 1.1018 ± 0.0380 | 1.4927 ± 0.0472 | 0.4213 ± 0.0157 | 0.8481 ± 0.0253 | 1.6777 |
| `tr_PITRecal+TabularMLP [shift]` | torchregress | 0.8953 ± 0.0181 | 1.3554 ± 0.0775 | 1.9471 ± 0.0525 | 0.5000 ± 0.0047 | 0.8359 ± 0.0365 | 26.5717 |
| `tr_ResidualDist+KumoTabular [shift]` | torchregress | 0.8803 ± 0.0227 | 0.9831 ± 0.0663 | 1.8988 ± 0.1421 † | 0.3879 ± 0.0247 | 0.7036 ± 0.0279 | 1.7863 |
| `tr_ResidualDist+LightGBM [shift]` | torchregress | 0.8943 ± 0.0050 | 1.3616 ± 0.0255 | 2.2601 ± 0.0784 | 0.4764 ± 0.0086 | 0.7220 ± 0.0094 | 1.8032 |
| `tr_ResidualDist+TabFM [shift]` | torchregress | 0.8813 ± 0.0166 | 1.0248 ± 0.0669 | 1.9209 ± 0.1744 † | 0.3901 ± 0.0296 | 0.7369 ± 0.0645 | 8.0697 |
| `tr_ResidualDist+TabICL [shift]` | torchregress | 0.8790 ± 0.0215 | 1.0818 ± 0.0561 | 2.0747 ± 0.1015 | 0.4216 ± 0.0152 | 0.7245 ± 0.0704 | 1.6777 |
| `tr_ResidualDist+TabularMLP [shift]` | torchregress | 0.8940 ± 0.0137 | 1.4408 ± 0.0766 | 2.3248 ± 0.0337 | 0.4945 ± 0.0028 | 0.7090 ± 0.0285 | 26.5717 |
| `tr_Split+KumoTabular [shift]` | torchregress | 0.8797 ± 0.0112 | 0.9744 ± 0.0514 | 1.9244 ± 0.1365 † | **0.3878 ± 0.0247** | 0.7062 ± 0.0330 | 1.7863 |
| `tr_Split+LightGBM [shift]` | torchregress | 0.8813 ± 0.0059 | 1.2963 ± 0.0133 | 2.3217 ± 0.0552 | 0.4756 ± 0.0087 | 0.6650 ± 0.0174 | 1.8032 |
| `tr_Split+TabFM [shift]` | torchregress | 0.8787 ± 0.0107 | 1.0190 ± 0.0383 | 1.9255 ± 0.1753 † | 0.3901 ± 0.0308 | 0.7215 ± 0.0519 | 8.0697 |
| `tr_Split+TabICL [shift]` | torchregress | 0.8817 ± 0.0199 | 1.0771 ± 0.0661 | 2.0888 ± 0.0918 | 0.4215 ± 0.0152 | 0.7234 ± 0.0551 | 1.6777 |
| `tr_Split+TabularMLP [shift]` | torchregress | 0.8903 ± 0.0095 | 1.3861 ± 0.0452 | 2.3384 ± 0.0285 | 0.4919 ± 0.0030 | 0.6907 ± 0.0258 | 26.5717 |
| `tr_WeightedCQR+KumoTabular [shift]` | torchregress | 0.9030 ± 0.0271 | 1.0172 ± 0.0607 | 1.3309 ± 0.0298 | **0.3878 ± 0.0247** | 0.8448 ± 0.0362 | 1.7863 |
| `tr_WeightedCQR+LightGBM [shift]` | torchregress | 0.9060 ± 0.0145 | 2.1260 ± 0.8423 | 2.5107 ± 0.6270 † | 0.4756 ± 0.0087 | 0.7272 ± 0.0334 | 1.8032 |
| `tr_WeightedCQR+TabICL [shift]` | torchregress | 0.8983 ± 0.0160 | 1.1287 ± 0.0333 | 1.5134 ± 0.0485 | 0.4215 ± 0.0152 | 0.8425 ± 0.0205 | 1.6777 |
| `tr_WeightedCQR+TabularMLP [shift]` | torchregress | 0.8927 ± 0.0161 | 1.4113 ± 0.0633 | 1.9884 ± 0.0512 | 0.4919 ± 0.0030 | 0.8128 ± 0.0185 | 26.5717 |
| `tr_WeightedSplit+KumoTabular [shift]` | torchregress | 0.8943 ± 0.0180 | 1.0557 ± 0.0540 | 1.9243 ± 0.1302 † | **0.3878 ± 0.0247** | 0.7189 ± 0.0509 | 1.7863 |
| `tr_WeightedSplit+LightGBM [shift]` | torchregress | 0.8887 ± 0.0085 | 1.3718 ± 0.0511 | 2.3311 ± 0.0642 | 0.4756 ± 0.0087 | 0.6701 ± 0.0158 | 1.8032 |
| `tr_WeightedSplit+TabFM [shift]` | torchregress | 0.8870 ± 0.0226 | 1.0572 ± 0.0813 | 1.8713 ± 0.1158 † | 0.3901 ± 0.0308 | 0.7212 ± 0.0748 | 8.0697 |
| `tr_WeightedSplit+TabICL [shift]` | torchregress | 0.8937 ± 0.0273 | 1.1776 ± 0.1262 | 2.0993 ± 0.0815 | 0.4215 ± 0.0152 | 0.7336 ± 0.0726 | 1.6777 |
| `tr_WeightedSplit+TabularMLP [shift]` | torchregress | 0.8887 ± 0.0086 | 1.4233 ± 0.0654 | 2.3614 ± 0.0454 | 0.4919 ± 0.0030 | 0.6775 ± 0.0349 | 26.5717 |

##### fm_uq_scenarios / diabetes

_No successful rows._

##### fm_uq_scenarios / diabetes@clean

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [clean]` | sdm | 0.9091 ± 0.0601 | 2.2248 ± 0.0576 | 2.6394 ± 0.4831 † | 0.7399 ± 0.1107 | 0.7044 ± 0.1553 | 0.9858 |
| `Raw+LightGBM [clean]` | lightgbm | 0.6136 ± 0.0521 | 1.4321 ± 0.0809 | 4.1336 ± 0.5521 † | 0.7930 ± 0.1110 | 0.1032 ± 0.0901 | 0.5270 |
| `Raw+TabICL [clean]` | tabicl | 0.9015 ± 0.0722 | 2.2285 ± 0.0422 | **2.6316 ± 0.5079** † | 0.7359 ± 0.1111 | 0.6627 ± 0.0894 | 0.8439 |
| `Raw+TabularMLP [clean]` | torchregress | 0.8674 ± 0.0980 | 2.2339 ± 0.2849 | 3.1516 ± 0.7939 † | 0.7327 ± 0.1213 | 0.6540 ± 0.1029 | 2.3964 |
| `tr_CQR+KumoTabular [clean]` | torchregress | 0.9167 ± 0.0572 | 2.3130 ± 0.1741 | 2.6744 ± 0.4039 † | 0.7399 ± 0.1107 | 0.7127 ± 0.1691 | 0.9858 |
| `tr_CQR+LightGBM [clean]` | torchregress | 0.9205 ± 0.0410 | 2.4437 ± 0.1316 | 2.8607 ± 0.3084 † | 0.7930 ± 0.1110 | 0.7090 ± 0.1629 | 0.5270 |
| `tr_CQR+TabICL [clean]` | torchregress | 0.9129 ± 0.0694 | 2.3738 ± 0.2605 | 2.7137 ± 0.3958 † | 0.7359 ± 0.1111 | 0.7127 ± 0.1691 | 0.8439 |
| `tr_CQR+TabularMLP [clean]` | torchregress | 0.9053 ± 0.0853 | 2.4727 ± 0.1189 | 3.0223 ± 0.5146 † | 0.7327 ± 0.1213 | 0.6929 ± 0.1622 | 2.3964 |
| `tr_DCP+KumoTabular [clean]` | torchregress | 0.9091 ± 0.0601 | 2.2713 ± 0.2006 | 2.6372 ± 0.3971 † | 0.7399 ± 0.1107 | 0.7127 ± 0.1691 | 0.9858 |
| `tr_DCP+LightGBM [clean]` | torchregress | 0.9394 ± 0.0237 | 3.0200 ± 0.3299 | 3.2214 ± 0.1917 † | 0.7930 ± 0.1110 | 0.5985 ± 0.1332 | 0.5270 |
| `tr_DCP+TabICL [clean]` | torchregress | 0.9129 ± 0.0694 | 2.3436 ± 0.2125 | 2.6893 ± 0.3901 † | 0.7359 ± 0.1111 | 0.7127 ± 0.1691 | 0.8439 |
| `tr_DCP+TabularMLP [clean]` | torchregress | 0.9053 ± 0.0853 | 2.4901 ± 0.1255 | 3.0232 ± 0.4723 † | 0.7327 ± 0.1213 | 0.6929 ± 0.1622 | 2.3964 |
| `tr_Mondrian+KumoTabular [clean]` | torchregress | 0.9015 ± 0.0347 | 2.6095 ± 0.2663 | 3.0820 ± 0.4533 † | 0.7399 ± 0.1107 | 0.6765 ± 0.0883 | 0.9858 |
| `tr_Mondrian+LightGBM [clean]` | torchregress | 0.8750 ± 0.0710 | 2.5582 ± 0.2413 | 3.3609 ± 0.4517 † | 0.7930 ± 0.1110 | 0.4973 ± 0.1546 | 0.5270 |
| `tr_Mondrian+TabFM [clean]` | torchregress | 0.9053 ± 0.0806 | 2.5105 ± 0.2673 | 2.9720 ± 0.5112 † | **0.7232 ± 0.1136** | 0.6981 ± 0.2067 | 3.7653 |
| `tr_Mondrian+TabICL [clean]` | torchregress | 0.8977 ± 0.0197 | 2.5421 ± 0.2097 | 3.0734 ± 0.3654 † | 0.7359 ± 0.1111 | 0.6976 ± 0.0721 | 0.8439 |
| `tr_Mondrian+TabularMLP [clean]` | torchregress | 0.8902 ± 0.0911 | 2.4628 ± 0.2184 | 3.0172 ± 0.6233 † | 0.7327 ± 0.1213 | 0.6724 ± 0.2260 | 2.3964 |
| `tr_Normalized+KumoTabular [clean]` | torchregress | 0.9015 ± 0.0473 | 2.4344 ± 0.2997 | 2.9415 ± 0.3602 † | 0.7399 ± 0.1107 | 0.7061 ± 0.1394 | 0.9858 |
| `tr_Normalized+LightGBM [clean]` | torchregress | 0.8636 ± 0.0341 | 2.4812 ± 0.1791 | 3.3710 ± 0.3717 † | 0.7930 ± 0.1110 | 0.6033 ± 0.0854 | 0.5270 |
| `tr_Normalized+TabICL [clean]` | torchregress | 0.9129 ± 0.0666 | 2.4587 ± 0.2914 | 2.9309 ± 0.3728 † | 0.7359 ± 0.1111 | 0.7476 ± 0.1434 | 0.8439 |
| `tr_Normalized+TabularMLP [clean]` | torchregress | 0.9053 ± 0.0853 | 2.4901 ± 0.1255 | 3.0232 ± 0.4723 † | 0.7327 ± 0.1213 | 0.6929 ± 0.1622 | 2.3964 |
| `tr_PITRecal+KumoTabular [clean]` | torchregress | 0.9015 ± 0.0694 | 2.2590 ± 0.1902 | 2.7164 ± 0.5173 † | 0.7448 ± 0.0931 | 0.6567 ± 0.2234 | 0.9858 |
| `tr_PITRecal+LightGBM [clean]` | torchregress | 0.9167 ± 0.0365 | 2.8038 ± 0.1528 | 3.0611 ± 0.0643 | 0.7955 ± 0.0827 | 0.5317 ± 0.1222 | 0.5270 |
| `tr_PITRecal+TabICL [clean]` | torchregress | 0.8939 ± 0.0459 | 2.2519 ± 0.2141 | 2.7913 ± 0.5630 † | 0.7479 ± 0.0871 | 0.6151 ± 0.1668 | 0.8439 |
| `tr_PITRecal+TabularMLP [clean]` | torchregress | 0.8826 ± 0.0953 | 2.3245 ± 0.1598 | 2.9919 ± 0.6660 † | 0.7519 ± 0.1032 | 0.6540 ± 0.1029 | 2.3964 |
| `tr_ResidualDist+KumoTabular [clean]` | torchregress | 0.8939 ± 0.0572 | 2.3912 ± 0.2967 | 2.9994 ± 0.5488 † | 0.7435 ± 0.0920 | 0.6712 ± 0.0911 | 0.9858 |
| `tr_ResidualDist+LightGBM [clean]` | torchregress | 0.8826 ± 0.0739 | 2.4545 ± 0.1980 | 3.2757 ± 0.6524 † | 0.8055 ± 0.0884 | 0.6873 ± 0.1641 | 0.5270 |
| `tr_ResidualDist+TabFM [clean]` | torchregress | 0.9015 ± 0.0572 | 2.3719 ± 0.1660 | 2.9259 ± 0.5834 † | 0.7370 ± 0.0987 | 0.6905 ± 0.1031 | 3.7653 |
| `tr_ResidualDist+TabICL [clean]` | torchregress | 0.8864 ± 0.0495 | 2.2705 ± 0.2285 | 3.0305 ± 0.6559 † | 0.7409 ± 0.0911 | 0.6905 ± 0.1031 | 0.8439 |
| `tr_ResidualDist+TabularMLP [clean]` | torchregress | 0.8902 ± 0.0757 | 2.3973 ± 0.1383 | 3.0138 ± 0.6657 † | 0.7503 ± 0.1019 | 0.6540 ± 0.1029 | 2.3964 |
| `tr_Split+KumoTabular [clean]` | torchregress | 0.9205 ± 0.0633 | 2.6070 ± 0.3000 | 3.0841 ± 0.3997 † | 0.7399 ± 0.1107 | 0.7284 ± 0.1778 | 0.9858 |
| `tr_Split+LightGBM [clean]` | torchregress | 0.8750 ± 0.0795 | 2.5520 ± 0.1655 | 3.3679 ± 0.4976 † | 0.7930 ± 0.1110 | 0.5486 ± 0.1622 | 0.5270 |
| `tr_Split+TabFM [clean]` | torchregress | 0.8977 ± 0.0930 | 2.4704 ± 0.2003 | 2.9687 ± 0.5277 † | **0.7232 ± 0.1136** | 0.7237 ± 0.1958 | 3.7653 |
| `tr_Split+TabICL [clean]` | torchregress | 0.9280 ± 0.0646 | 2.6689 ± 0.3521 | 3.0761 ± 0.3659 † | 0.7359 ± 0.1111 | 0.7809 ± 0.1978 | 0.8439 |
| `tr_Split+TabularMLP [clean]` | torchregress | 0.9053 ± 0.0966 | 2.4409 ± 0.0979 | 3.0464 ± 0.6524 † | 0.7327 ± 0.1213 | 0.6848 ± 0.2399 | 2.3964 |

##### fm_uq_scenarios / diabetes@extrapolate

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [extrapolate]` | sdm | 0.9508 ± 0.0131 | 1.8833 ± 0.0950 | **2.0646 ± 0.0547** | 0.5652 ± 0.0083 | 0.6389 ± 0.0241 | 0.6479 |
| `Raw+LightGBM [extrapolate]` | lightgbm | 0.7462 ± 0.0430 | 1.3432 ± 0.1594 | 2.3867 ± 0.0991 | 0.5469 ± 0.0111 | 0.3256 ± 0.1969 | 0.6009 |
| `Raw+TabICL [extrapolate]` | tabicl | 0.9015 ± 0.0066 | 1.7847 ± 0.1162 | 2.1914 ± 0.0917 | 0.5742 ± 0.0095 | 0.6167 ± 0.0144 | 0.7851 |
| `Raw+TabularMLP [extrapolate]` | torchregress | 0.9318 ± 0.0114 | 2.3127 ± 0.1299 | 2.6032 ± 0.0940 | 0.5645 ± 0.0249 | 0.6083 ± 0.0144 | 2.0368 |
| `tr_CQR+KumoTabular [extrapolate]` | torchregress | 0.9242 ± 0.0537 | 1.8231 ± 0.2617 | 2.0803 ± 0.0567 | 0.5652 ± 0.0083 | 0.6389 ± 0.0241 | 0.6479 |
| `tr_CQR+LightGBM [extrapolate]` | torchregress | 0.9848 ± 0.0131 | 2.4329 ± 0.1328 | 2.4683 ± 0.1050 | 0.5469 ± 0.0111 | 0.8333 ± 0.1443 | 0.6009 |
| `tr_CQR+TabICL [extrapolate]` | torchregress | 0.9053 ± 0.0666 | 1.8172 ± 0.2156 | 2.2568 ± 0.1185 † | 0.5742 ± 0.0095 | 0.6389 ± 0.0241 | 0.7851 |
| `tr_CQR+TabularMLP [extrapolate]` | torchregress | 0.9432 ± 0.0227 | 2.3758 ± 0.0971 | 2.6307 ± 0.0695 | 0.5645 ± 0.0249 | 0.6722 ± 0.0752 | 2.0368 |
| `tr_DCP+KumoTabular [extrapolate]` | torchregress | 0.9280 ± 0.0473 | 1.7819 ± 0.3135 | 2.1043 ± 0.0741 | 0.5652 ± 0.0083 | 0.5972 ± 0.0867 | 0.6479 |
| `tr_DCP+LightGBM [extrapolate]` | torchregress | 0.9356 ± 0.0347 | 2.7814 ± 0.2107 | 2.9045 ± 0.1527 † | 0.5469 ± 0.0111 | 0.7855 ± 0.1072 | 0.6009 |
| `tr_DCP+TabICL [extrapolate]` | torchregress | 0.9091 ± 0.0410 | 1.7689 ± 0.1650 | 2.2256 ± 0.1456 † | 0.5742 ± 0.0095 | 0.6389 ± 0.0241 | 0.7851 |
| `tr_DCP+TabularMLP [extrapolate]` | torchregress | 0.9394 ± 0.0237 | 2.3692 ± 0.1063 | 2.6268 ± 0.0725 | 0.5645 ± 0.0249 | 0.6500 ± 0.0866 | 2.0368 |
| `tr_Mondrian+KumoTabular [extrapolate]` | torchregress | 0.9583 ± 0.0237 | 2.7718 ± 0.2950 | 2.8914 ± 0.1700 † | 0.5652 ± 0.0083 | 0.6111 ± 0.2295 | 0.6479 |
| `tr_Mondrian+LightGBM [extrapolate]` | torchregress | 0.9848 ± 0.0262 | 3.0499 ± 0.4931 | 3.1238 ± 0.3699 † | 0.5469 ± 0.0111 | 0.8750 ± 0.2165 | 0.6009 |
| `tr_Mondrian+TabFM [extrapolate]` | torchregress | 0.9545 ± 0.0301 | 2.5355 ± 0.4460 | 2.7489 ± 0.2199 † | 0.5482 ± 0.0106 | 0.6111 ± 0.2295 | 3.8783 |
| `tr_Mondrian+TabICL [extrapolate]` | torchregress | 0.9621 ± 0.0174 | 2.6562 ± 0.3524 | 2.8588 ± 0.1905 † | 0.5742 ± 0.0095 | 0.6389 ± 0.1273 | 0.7851 |
| `tr_Mondrian+TabularMLP [extrapolate]` | torchregress | 0.9583 ± 0.0237 | 2.4457 ± 0.3024 | 2.6770 ± 0.1550 † | 0.5645 ± 0.0249 | 0.6722 ± 0.0752 | 2.0368 |
| `tr_Normalized+KumoTabular [extrapolate]` | torchregress | 0.9053 ± 0.0262 | 1.9297 ± 0.1656 | 2.5183 ± 0.0642 | 0.5652 ± 0.0083 | 0.4583 ± 0.0722 | 0.6479 |
| `tr_Normalized+LightGBM [extrapolate]` | torchregress | 0.9508 ± 0.0066 | 2.3532 ± 0.0975 | 2.6485 ± 0.1534 † | 0.5469 ± 0.0111 | 0.6250 ± 0.1250 | 0.6009 |
| `tr_Normalized+TabICL [extrapolate]` | torchregress | 0.8864 ± 0.0227 | 1.8862 ± 0.1436 | 2.7808 ± 0.1083 | 0.5742 ± 0.0095 | 0.5083 ± 0.1127 | 0.7851 |
| `tr_Normalized+TabularMLP [extrapolate]` | torchregress | 0.9394 ± 0.0237 | 2.3692 ± 0.1063 | 2.6268 ± 0.0725 | 0.5645 ± 0.0249 | 0.6500 ± 0.0866 | 2.0368 |
| `tr_PITRecal+KumoTabular [extrapolate]` | torchregress | 0.9280 ± 0.0656 | 1.8172 ± 0.3156 | 2.0893 ± 0.0323 | 0.5409 ± 0.0410 | 0.6944 ± 0.1735 | 0.6479 |
| `tr_PITRecal+LightGBM [extrapolate]` | torchregress | 0.9015 ± 0.0328 | 2.8443 ± 0.2037 | 3.0529 ± 0.1954 † | 0.5536 ± 0.0208 | 0.6682 ± 0.0989 | 0.6009 |
| `tr_PITRecal+TabICL [extrapolate]` | torchregress | 0.8902 ± 0.0328 | 1.7409 ± 0.1615 | 2.1858 ± 0.1417 † | 0.5502 ± 0.0431 | 0.6389 ± 0.0241 | 0.7851 |
| `tr_PITRecal+TabularMLP [extrapolate]` | torchregress | 0.9697 ± 0.0174 | 2.2463 ± 0.2253 | 2.3780 ± 0.1780 † | 0.5419 ± 0.0387 | 0.7361 ± 0.1049 | 2.0368 |
| `tr_ResidualDist+KumoTabular [extrapolate]` | torchregress | 0.9621 ± 0.0286 | 2.3348 ± 0.3622 | 2.4413 ± 0.2556 † | 0.5464 ± 0.0391 | 0.6111 ± 0.2295 | 0.6479 |
| `tr_ResidualDist+LightGBM [extrapolate]` | torchregress | 0.9886 ± 0.0197 | 2.5406 ± 0.2878 | 2.5904 ± 0.2188 † | 0.5609 ± 0.0331 | 0.9333 ± 0.1155 | 0.6009 |
| `tr_ResidualDist+TabFM [extrapolate]` | torchregress | 0.9735 ± 0.0174 | 2.3810 ± 0.3459 | 2.4958 ± 0.2271 † | **0.5380 ± 0.0199** | 0.7778 ± 0.1339 | 3.8783 |
| `tr_ResidualDist+TabICL [extrapolate]` | torchregress | 0.9621 ± 0.0174 | 2.2553 ± 0.2768 | 2.4190 ± 0.1703 † | 0.5491 ± 0.0524 | 0.6806 ± 0.0636 | 0.7851 |
| `tr_ResidualDist+TabularMLP [extrapolate]` | torchregress | 0.9811 ± 0.0066 | 2.3236 ± 0.1683 | 2.4272 ± 0.1815 † | 0.5416 ± 0.0378 | 0.7778 ± 0.0481 | 2.0368 |
| `tr_Split+KumoTabular [extrapolate]` | torchregress | 0.9508 ± 0.0174 | 2.5709 ± 0.1408 | 2.7562 ± 0.0479 | 0.5652 ± 0.0083 | 0.5000 ± 0.1250 | 0.6479 |
| `tr_Split+LightGBM [extrapolate]` | torchregress | 0.9545 ± 0.0114 | 2.5543 ± 0.0908 | 2.7031 ± 0.0458 | 0.5469 ± 0.0111 | 0.5833 ± 0.1443 | 0.6009 |
| `tr_Split+TabFM [extrapolate]` | torchregress | 0.9508 ± 0.0262 | 2.5219 ± 0.3920 | 2.7323 ± 0.1916 † | 0.5482 ± 0.0106 | 0.5556 ± 0.1577 | 3.8783 |
| `tr_Split+TabICL [extrapolate]` | torchregress | 0.9583 ± 0.0131 | 2.4793 ± 0.1624 | 2.7373 ± 0.0525 | 0.5742 ± 0.0095 | 0.5972 ± 0.0867 | 0.7851 |
| `tr_Split+TabularMLP [extrapolate]` | torchregress | 0.9697 ± 0.0066 | 2.5691 ± 0.0616 | 2.7240 ± 0.0734 | 0.5645 ± 0.0249 | 0.6806 ± 0.0636 | 2.0368 |

##### fm_uq_scenarios / diabetes@heavy_tail

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [heavy_tail]` | sdm | 0.9053 ± 0.0656 | 2.1392 ± 0.0566 | 2.8142 ± 0.7307 † | 0.7389 ± 0.1261 | 0.6429 ± 0.2034 | 0.6493 |
| `Raw+LightGBM [heavy_tail]` | lightgbm | 0.6288 ± 0.0685 | 1.2388 ± 0.0133 | 4.3406 ± 1.0344 † | 0.7845 ± 0.1052 | 0.1032 ± 0.0901 | 0.5906 |
| `Raw+TabICL [heavy_tail]` | tabicl | 0.9129 ± 0.0525 | 2.1361 ± 0.0099 | **2.7623 ± 0.7476** † | 0.7386 ± 0.1201 | 0.6627 ± 0.0894 | 0.7979 |
| `Raw+TabularMLP [heavy_tail]` | torchregress | 0.9280 ± 0.0561 | 2.3185 ± 0.1207 | 3.1952 ± 0.7094 † | 0.7479 ± 0.1132 | 0.6540 ± 0.1029 | 2.0537 |
| `tr_CQR+KumoTabular [heavy_tail]` | torchregress | 0.9242 ± 0.0626 | 2.3055 ± 0.2914 | 2.8781 ± 0.5966 † | 0.7389 ± 0.1261 | 0.7238 ± 0.1876 | 0.6493 |
| `tr_CQR+LightGBM [heavy_tail]` | torchregress | 0.9242 ± 0.0430 | 2.2812 ± 0.0308 | 3.2915 ± 0.7431 † | 0.7845 ± 0.1052 | 0.5101 ± 0.1518 | 0.5906 |
| `tr_CQR+TabICL [heavy_tail]` | torchregress | 0.9356 ± 0.0572 | 2.2474 ± 0.1524 | 2.7838 ± 0.6830 † | 0.7386 ± 0.1201 | 0.7183 ± 0.1783 | 0.7979 |
| `tr_CQR+TabularMLP [heavy_tail]` | torchregress | 0.9280 ± 0.0583 | 2.3166 ± 0.2325 | 3.2077 ± 0.6103 † | 0.7479 ± 0.1132 | 0.6540 ± 0.2286 | 2.0537 |
| `tr_DCP+KumoTabular [heavy_tail]` | torchregress | 0.9205 ± 0.0521 | 2.3356 ± 0.3418 | 2.8761 ± 0.5773 † | 0.7389 ± 0.1261 | 0.7183 ± 0.1783 | 0.6493 |
| `tr_DCP+LightGBM [heavy_tail]` | torchregress | 0.9394 ± 0.0328 | 3.1155 ± 0.6601 | 3.7711 ± 0.3191 † | 0.7845 ± 0.1052 | 0.6263 ± 0.1808 | 0.5906 |
| `tr_DCP+TabICL [heavy_tail]` | torchregress | 0.9356 ± 0.0512 | 2.3278 ± 0.3007 | 2.8430 ± 0.6104 † | 0.7386 ± 0.1201 | 0.7183 ± 0.1783 | 0.7979 |
| `tr_DCP+TabularMLP [heavy_tail]` | torchregress | 0.9318 ± 0.0601 | 2.3484 ± 0.2988 | 3.2552 ± 0.5541 † | 0.7479 ± 0.1132 | 0.6540 ± 0.2286 | 2.0537 |
| `tr_Mondrian+KumoTabular [heavy_tail]` | torchregress | 0.9394 ± 0.0430 | 2.5166 ± 0.5575 | 3.4245 ± 0.5760 † | 0.7389 ± 0.1261 | 0.7003 ± 0.1742 | 0.6493 |
| `tr_Mondrian+LightGBM [heavy_tail]` | torchregress | 0.9167 ± 0.0561 | 2.3555 ± 0.3825 | 3.5926 ± 0.4994 † | 0.7845 ± 0.1052 | 0.5404 ± 0.1117 | 0.5906 |
| `tr_Mondrian+TabFM [heavy_tail]` | torchregress | 0.9356 ± 0.0237 | 2.2615 ± 0.2064 | 3.1486 ± 0.5537 † | **0.7212 ± 0.1246** | 0.6677 ± 0.1853 | 3.9073 |
| `tr_Mondrian+TabICL [heavy_tail]` | torchregress | 0.9356 ± 0.0473 | 2.4586 ± 0.6306 | 3.4773 ± 0.5624 † | 0.7386 ± 0.1201 | 0.7003 ± 0.1742 | 0.7979 |
| `tr_Mondrian+TabularMLP [heavy_tail]` | torchregress | 0.9205 ± 0.0455 | 2.1745 ± 0.2703 | 3.3592 ± 0.5432 † | 0.7479 ± 0.1132 | 0.6448 ± 0.2126 | 2.0537 |
| `tr_Normalized+KumoTabular [heavy_tail]` | torchregress | 0.9318 ± 0.0568 | 2.3693 ± 0.4157 | 3.1772 ± 0.6152 † | 0.7389 ± 0.1261 | 0.6848 ± 0.2399 | 0.6493 |
| `tr_Normalized+LightGBM [heavy_tail]` | torchregress | 0.9015 ± 0.0459 | 2.7805 ± 0.5548 | 3.7875 ± 0.6665 † | 0.7845 ± 0.1052 | 0.6374 ± 0.1999 | 0.5906 |
| `tr_Normalized+TabICL [heavy_tail]` | torchregress | 0.9356 ± 0.0512 | 2.3248 ± 0.3694 | 3.0978 ± 0.6234 † | 0.7386 ± 0.1201 | 0.6700 ± 0.2172 | 0.7979 |
| `tr_Normalized+TabularMLP [heavy_tail]` | torchregress | 0.9318 ± 0.0601 | 2.3484 ± 0.2988 | 3.2552 ± 0.5541 † | 0.7479 ± 0.1132 | 0.6540 ± 0.2286 | 2.0537 |
| `tr_PITRecal+KumoTabular [heavy_tail]` | torchregress | 0.9015 ± 0.0685 | 2.2831 ± 0.4170 | 2.9585 ± 0.5992 † | 0.7466 ± 0.1030 | 0.6429 ± 0.2034 | 0.6493 |
| `tr_PITRecal+LightGBM [heavy_tail]` | torchregress | 0.9280 ± 0.0286 | 2.7762 ± 0.4282 | 3.5391 ± 0.4004 † | 0.7796 ± 0.0909 | 0.5707 ± 0.0862 | 0.5906 |
| `tr_PITRecal+TabICL [heavy_tail]` | torchregress | 0.9053 ± 0.0694 | 2.1654 ± 0.2807 | 2.8736 ± 0.7240 † | 0.7487 ± 0.1019 | 0.6151 ± 0.1668 | 0.7979 |
| `tr_PITRecal+TabularMLP [heavy_tail]` | torchregress | 0.8977 ± 0.0819 | 2.1837 ± 0.3669 | 3.2001 ± 0.6273 † | 0.7616 ± 0.0968 | 0.5873 ± 0.2160 | 2.0537 |
| `tr_ResidualDist+KumoTabular [heavy_tail]` | torchregress | 0.9280 ± 0.0459 | 2.4018 ± 0.1641 | 3.4891 ± 0.5430 † | 0.7426 ± 0.1073 | 0.6448 ± 0.2126 | 0.6493 |
| `tr_ResidualDist+LightGBM [heavy_tail]` | torchregress | 0.9205 ± 0.0601 | 2.4826 ± 0.3550 | 3.6822 ± 0.5269 † | 0.7904 ± 0.0924 | 0.6145 ± 0.2387 | 0.5906 |
| `tr_ResidualDist+TabFM [heavy_tail]` | torchregress | 0.9205 ± 0.0495 | 2.3814 ± 0.1823 | 3.5272 ± 0.5594 † | 0.7273 ± 0.1147 | 0.6460 ± 0.1944 | 3.9073 |
| `tr_ResidualDist+TabICL [heavy_tail]` | torchregress | 0.9015 ± 0.0430 | 2.2634 ± 0.2133 | 3.4572 ± 0.6176 † | 0.7444 ± 0.1001 | 0.6374 ± 0.1999 | 0.7979 |
| `tr_ResidualDist+TabularMLP [heavy_tail]` | torchregress | 0.8864 ± 0.0301 | 2.2027 ± 0.2025 | 3.5410 ± 0.6394 † | 0.7508 ± 0.1061 | 0.5985 ± 0.1332 | 2.0537 |
| `tr_Split+KumoTabular [heavy_tail]` | torchregress | 0.9242 ± 0.0646 | 2.5028 ± 0.5979 | 3.6963 ± 0.5033 † | 0.7389 ± 0.1261 | 0.5471 ± 0.2111 | 0.6493 |
| `tr_Split+LightGBM [heavy_tail]` | torchregress | 0.9205 ± 0.0601 | 2.4575 ± 0.3366 | 3.8548 ± 0.5734 † | 0.7845 ± 0.1052 | 0.4428 ± 0.1499 | 0.5906 |
| `tr_Split+TabFM [heavy_tail]` | torchregress | 0.9129 ± 0.0722 | 2.3248 ± 0.2827 | 3.5504 ± 0.6024 † | **0.7212 ± 0.1246** | 0.5471 ± 0.2111 | 3.9073 |
| `tr_Split+TabICL [heavy_tail]` | torchregress | 0.9205 ± 0.0710 | 2.4476 ± 0.4619 | 3.6679 ± 0.5017 † | 0.7386 ± 0.1201 | 0.5168 ± 0.2529 | 0.7979 |
| `tr_Split+TabularMLP [heavy_tail]` | torchregress | 0.9205 ± 0.0902 | 2.3629 ± 0.4880 | 3.7114 ± 0.5330 † | 0.7479 ± 0.1132 | 0.5842 ± 0.2726 | 2.0537 |

##### fm_uq_scenarios / diabetes@imbalance

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [imbalance]` | sdm | 0.7803 ± 0.0473 | 1.9286 ± 0.0101 | 3.3752 ± 0.5886 † | 0.8345 ± 0.0804 | 0.4348 ± 0.1130 | 0.6233 |
| `Raw+LightGBM [imbalance]` | lightgbm | 0.4242 ± 0.0572 | 1.1664 ± 0.0312 | 6.6988 ± 0.7617 † | 0.8958 ± 0.0666 | 0.0545 ± 0.0640 | 0.5578 |
| `Raw+TabICL [imbalance]` | tabicl | 0.7689 ± 0.0583 | 1.9191 ± 0.0118 | 3.4192 ± 0.6348 † | 0.8359 ± 0.0788 | 0.4277 ± 0.0389 | 0.7615 |
| `Raw+TabularMLP [imbalance]` | torchregress | 0.7500 ± 0.0521 | 1.9707 ± 0.0779 | 3.8025 ± 0.4446 † | 0.8492 ± 0.0691 | 0.3859 ± 0.0822 | 1.7918 |
| `tr_CQR+KumoTabular [imbalance]` | torchregress | 0.7992 ± 0.0473 | 2.0489 ± 0.2096 | 3.2452 ± 0.4257 † | 0.8345 ± 0.0804 | 0.4708 ± 0.1460 | 0.6233 |
| `tr_CQR+LightGBM [imbalance]` | torchregress | 0.7841 ± 0.0455 | 2.1534 ± 0.2918 | 3.9182 ± 0.6334 † | 0.8958 ± 0.0666 | 0.4307 ± 0.1479 | 0.5578 |
| `tr_CQR+TabICL [imbalance]` | torchregress | 0.7917 ± 0.0430 | 2.0310 ± 0.1931 | 3.2595 ± 0.4783 † | 0.8359 ± 0.0788 | 0.4435 ± 0.1582 | 0.7615 |
| `tr_CQR+TabularMLP [imbalance]` | torchregress | 0.7992 ± 0.0262 | 2.1813 ± 0.2880 | 3.5385 ± 0.2761 † | 0.8492 ± 0.0691 | 0.4556 ± 0.1347 | 1.7918 |
| `tr_DCP+KumoTabular [imbalance]` | torchregress | 0.8182 ± 0.0691 | 2.0878 ± 0.2252 | 3.1520 ± 0.4501 † | 0.8345 ± 0.0804 | 0.4989 ± 0.1940 | 0.6233 |
| `tr_DCP+LightGBM [imbalance]` | torchregress | 0.7614 ± 0.0601 | 2.1845 ± 0.0871 | 4.2330 ± 0.8220 † | 0.8958 ± 0.0666 | 0.3089 ± 0.0663 | 0.5578 |
| `tr_DCP+TabICL [imbalance]` | torchregress | 0.7917 ± 0.0430 | 2.0093 ± 0.1840 | 3.2092 ± 0.4429 † | 0.8359 ± 0.0788 | 0.4459 ± 0.1648 | 0.7615 |
| `tr_DCP+TabularMLP [imbalance]` | torchregress | 0.8030 ± 0.0237 | 2.2100 ± 0.3172 | 3.4878 ± 0.2370 † | 0.8492 ± 0.0691 | 0.4556 ± 0.1347 | 1.7918 |
| `tr_Mondrian+KumoTabular [imbalance]` | torchregress | 0.7652 ± 0.0473 | 2.0968 ± 0.3330 | 3.6828 ± 0.4194 † | 0.8345 ± 0.0804 | 0.3971 ± 0.1180 | 0.6233 |
| `tr_Mondrian+LightGBM [imbalance]` | torchregress | 0.7614 ± 0.0691 | 2.0925 ± 0.3127 | 4.3314 ± 0.4910 † | 0.8958 ± 0.0666 | 0.4058 ± 0.1632 | 0.5578 |
| `tr_Mondrian+TabFM [imbalance]` | torchregress | 0.7955 ± 0.0795 | 2.0904 ± 0.2233 | 3.6075 ± 0.6626 † | **0.8214 ± 0.0823** | 0.4500 ± 0.1373 | 3.7655 |
| `tr_Mondrian+TabICL [imbalance]` | torchregress | 0.7576 ± 0.0626 | 2.0956 ± 0.2738 | 3.7349 ± 0.6303 † | 0.8359 ± 0.0788 | 0.3826 ± 0.1431 | 0.7615 |
| `tr_Mondrian+TabularMLP [imbalance]` | torchregress | 0.7689 ± 0.0430 | 2.0537 ± 0.2976 | 3.8589 ± 0.3942 † | 0.8492 ± 0.0691 | 0.3802 ± 0.1461 | 1.7918 |
| `tr_Normalized+KumoTabular [imbalance]` | torchregress | 0.7955 ± 0.0710 | 2.1468 ± 0.4002 | 3.4254 ± 0.4472 † | 0.8345 ± 0.0804 | 0.5028 ± 0.2117 | 0.6233 |
| `tr_Normalized+LightGBM [imbalance]` | torchregress | 0.7614 ± 0.0227 | 2.2389 ± 0.1780 | 4.4696 ± 0.6820 † | 0.8958 ± 0.0666 | 0.4348 ± 0.1130 | 0.5578 |
| `tr_Normalized+TabICL [imbalance]` | torchregress | 0.7955 ± 0.0795 | 2.1516 ± 0.4246 | 3.4164 ± 0.4434 † | 0.8359 ± 0.0788 | 0.4883 ± 0.2366 | 0.7615 |
| `tr_Normalized+TabularMLP [imbalance]` | torchregress | 0.8030 ± 0.0237 | 2.2100 ± 0.3172 | 3.4878 ± 0.2370 † | 0.8492 ± 0.0691 | 0.4556 ± 0.1347 | 1.7918 |
| `tr_PITRecal+KumoTabular [imbalance]` | torchregress | 0.7992 ± 0.0757 | 2.0136 ± 0.2011 | 3.1380 ± 0.5164 † | 0.8865 ± 0.0325 | 0.4433 ± 0.2161 | 0.6233 |
| `tr_PITRecal+LightGBM [imbalance]` | torchregress | 0.7614 ± 0.0691 | 2.1854 ± 0.1583 | 4.3648 ± 0.7457 † | 0.9567 ± 0.0250 | 0.2985 ± 0.0748 | 0.5578 |
| `tr_PITRecal+TabICL [imbalance]` | torchregress | 0.7955 ± 0.0787 | 1.9563 ± 0.1742 | **3.1351 ± 0.5496** † | 0.8971 ± 0.0402 | 0.4545 ± 0.2750 | 0.7615 |
| `tr_PITRecal+TabularMLP [imbalance]` | torchregress | 0.8068 ± 0.0633 | 2.0051 ± 0.1569 | 3.2163 ± 0.3859 † | 0.9135 ± 0.0620 | 0.4177 ± 0.1718 | 1.7918 |
| `tr_ResidualDist+KumoTabular [imbalance]` | torchregress | 0.8220 ± 0.0656 | 2.0536 ± 0.2688 | 3.1354 ± 0.5480 † | 0.8886 ± 0.0295 | 0.5535 ± 0.1946 | 0.6233 |
| `tr_ResidualDist+LightGBM [imbalance]` | torchregress | 0.7879 ± 0.0774 | 2.1112 ± 0.2685 | 3.8888 ± 0.6631 † | 0.9733 ± 0.0341 | 0.4604 ± 0.1405 | 0.5578 |
| `tr_ResidualDist+TabFM [imbalance]` | torchregress | 0.7879 ± 0.0583 | 1.9663 ± 0.2071 | 3.1785 ± 0.6469 † | 0.8814 ± 0.0418 | 0.4877 ± 0.1342 | 3.7655 |
| `tr_ResidualDist+TabICL [imbalance]` | torchregress | 0.7955 ± 0.0888 | 1.9639 ± 0.2649 | 3.1978 ± 0.5793 † | 0.8907 ± 0.0385 | 0.5245 ± 0.2334 | 0.7615 |
| `tr_ResidualDist+TabularMLP [imbalance]` | torchregress | 0.7765 ± 0.0666 | 1.9797 ± 0.2348 | 3.4554 ± 0.3440 † | 0.9105 ± 0.0577 | 0.4058 ± 0.1632 | 1.7918 |
| `tr_Split+KumoTabular [imbalance]` | torchregress | 0.7652 ± 0.0473 | 2.0968 ± 0.3330 | 3.6828 ± 0.4194 † | 0.8345 ± 0.0804 | 0.3971 ± 0.1180 | 0.6233 |
| `tr_Split+LightGBM [imbalance]` | torchregress | 0.7614 ± 0.0691 | 2.0925 ± 0.3127 | 4.3314 ± 0.4910 † | 0.8958 ± 0.0666 | 0.4058 ± 0.1632 | 0.5578 |
| `tr_Split+TabFM [imbalance]` | torchregress | 0.7955 ± 0.0795 | 2.0904 ± 0.2233 | 3.6075 ± 0.6626 † | **0.8214 ± 0.0823** | 0.4500 ± 0.1373 | 3.7655 |
| `tr_Split+TabICL [imbalance]` | torchregress | 0.7576 ± 0.0626 | 2.0956 ± 0.2738 | 3.7349 ± 0.6303 † | 0.8359 ± 0.0788 | 0.3826 ± 0.1431 | 0.7615 |
| `tr_Split+TabularMLP [imbalance]` | torchregress | 0.7689 ± 0.0430 | 2.0537 ± 0.2976 | 3.8589 ± 0.3942 † | 0.8492 ± 0.0691 | 0.3802 ± 0.1461 | 1.7918 |

##### fm_uq_scenarios / diabetes@noise_all

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_all]` | sdm | 0.8826 ± 0.0757 | 2.3185 ± 0.0092 | 2.8232 ± 0.6530 † | 0.7863 ± 0.1523 | 0.6540 ± 0.1029 | 0.6434 |
| `Raw+LightGBM [noise_all]` | lightgbm | 0.6326 ± 0.0838 | 1.5277 ± 0.0077 | 4.2395 ± 1.0429 † | 0.7977 ± 0.1152 | 0.0278 ± 0.0481 | 0.7714 |
| `Raw+TabICL [noise_all]` | tabicl | 0.9015 ± 0.0722 | 2.3482 ± 0.0556 | **2.7786 ± 0.5253** † | 0.7836 ± 0.1471 | 0.6540 ± 0.1029 | 0.7902 |
| `Raw+TabularMLP [noise_all]` | torchregress | 0.8788 ± 0.0572 | 2.4226 ± 0.2006 | 3.0749 ± 0.5407 † | **0.7574 ± 0.1346** | 0.5960 ± 0.1225 | 1.8367 |
| `tr_CQR+KumoTabular [noise_all]` | torchregress | 0.9318 ± 0.0888 | 2.5199 ± 0.0119 | 2.8321 ± 0.4985 † | 0.7863 ± 0.1523 | 0.7707 ± 0.2014 | 0.6434 |
| `tr_CQR+LightGBM [noise_all]` | torchregress | 0.9356 ± 0.0537 | 2.7122 ± 0.1181 | 3.0145 ± 0.3419 † | 0.7977 ± 0.1152 | 0.6957 ± 0.1667 | 0.7714 |
| `tr_CQR+TabICL [noise_all]` | torchregress | 0.9318 ± 0.0710 | 2.5633 ± 0.0542 | 2.8554 ± 0.4009 † | 0.7836 ± 0.1471 | 0.7714 ± 0.1422 | 0.7902 |
| `tr_CQR+TabularMLP [noise_all]` | torchregress | 0.9205 ± 0.0495 | 2.6717 ± 0.0319 | 3.0933 ± 0.4805 † | **0.7574 ± 0.1346** | 0.6540 ± 0.1029 | 1.8367 |
| `tr_DCP+KumoTabular [noise_all]` | torchregress | 0.9167 ± 0.0774 | 2.5330 ± 0.1128 | 2.8329 ± 0.4411 † | 0.7863 ± 0.1523 | 0.6818 ± 0.1445 | 0.6434 |
| `tr_DCP+LightGBM [noise_all]` | torchregress | 0.9280 ± 0.0365 | 3.0465 ± 0.3917 | 3.2871 ± 0.2468 † | 0.7977 ± 0.1152 | 0.5985 ± 0.1332 | 0.7714 |
| `tr_DCP+TabICL [noise_all]` | torchregress | 0.9318 ± 0.0633 | 2.5708 ± 0.1420 | 2.8642 ± 0.3624 † | 0.7836 ± 0.1471 | 0.7659 ± 0.1327 | 0.7902 |
| `tr_DCP+TabularMLP [noise_all]` | torchregress | 0.9167 ± 0.0473 | 2.6547 ± 0.0223 | 3.0928 ± 0.4814 † | **0.7574 ± 0.1346** | 0.6540 ± 0.1029 | 1.8367 |
| `tr_Mondrian+KumoTabular [noise_all]` | torchregress | 0.9167 ± 0.0860 | 2.9203 ± 0.1086 | 3.4160 ± 0.5980 † | 0.7863 ± 0.1523 | 0.7042 ± 0.2244 | 0.6434 |
| `tr_Mondrian+LightGBM [noise_all]` | torchregress | 0.9091 ± 0.0455 | 2.8008 ± 0.2110 | 3.2683 ± 0.4553 † | 0.7977 ± 0.1152 | 0.6159 ± 0.1616 | 0.7714 |
| `tr_Mondrian+TabFM [noise_all]` | torchregress | 0.9242 ± 0.0731 | 2.7063 ± 0.1145 | 3.1328 ± 0.5524 † | 0.7608 ± 0.1353 | 0.6968 ± 0.2154 | 3.8822 |
| `tr_Mondrian+TabICL [noise_all]` | torchregress | 0.9356 ± 0.0459 | 2.9558 ± 0.1212 | 3.2946 ± 0.4797 † | 0.7836 ± 0.1471 | 0.7951 ± 0.0839 | 0.7902 |
| `tr_Mondrian+TabularMLP [noise_all]` | torchregress | 0.9167 ± 0.0666 | 2.7008 ± 0.0258 | 3.1184 ± 0.5294 † | **0.7574 ± 0.1346** | 0.6237 ± 0.1523 | 1.8367 |
| `tr_Normalized+KumoTabular [noise_all]` | torchregress | 0.9167 ± 0.0583 | 2.7550 ± 0.2281 | 3.2770 ± 0.5943 † | 0.7863 ± 0.1523 | 0.7951 ± 0.0839 | 0.6434 |
| `tr_Normalized+LightGBM [noise_all]` | torchregress | 0.9129 ± 0.0685 | 2.9322 ± 0.1247 | 3.3613 ± 0.4545 † | 0.7977 ± 0.1152 | 0.6861 ± 0.1513 | 0.7714 |
| `tr_Normalized+TabICL [noise_all]` | torchregress | 0.9015 ± 0.0525 | 2.6648 ± 0.1221 | 3.2140 ± 0.6193 † | 0.7836 ± 0.1471 | 0.7581 ± 0.0270 | 0.7902 |
| `tr_Normalized+TabularMLP [noise_all]` | torchregress | 0.9167 ± 0.0473 | 2.6547 ± 0.0223 | 3.0928 ± 0.4814 † | **0.7574 ± 0.1346** | 0.6540 ± 0.1029 | 1.8367 |
| `tr_PITRecal+KumoTabular [noise_all]` | torchregress | 0.9053 ± 0.0666 | 2.4851 ± 0.1069 | 2.8811 ± 0.4743 † | 0.7865 ± 0.1463 | 0.6627 ± 0.0894 | 0.6434 |
| `tr_PITRecal+LightGBM [noise_all]` | torchregress | 0.8977 ± 0.0691 | 2.9014 ± 0.3424 | 3.2100 ± 0.1400 | 0.8082 ± 0.1082 | 0.3889 ± 0.3469 | 0.7714 |
| `tr_PITRecal+TabICL [noise_all]` | torchregress | 0.9129 ± 0.0626 | 2.4859 ± 0.1068 | 2.8877 ± 0.4372 † | 0.7919 ± 0.1333 | 0.6627 ± 0.0894 | 0.7902 |
| `tr_PITRecal+TabularMLP [noise_all]` | torchregress | 0.8977 ± 0.0710 | 2.5524 ± 0.1113 | 3.0075 ± 0.4751 † | 0.7661 ± 0.1238 | 0.6540 ± 0.1029 | 1.8367 |
| `tr_ResidualDist+KumoTabular [noise_all]` | torchregress | 0.9242 ± 0.0572 | 2.7440 ± 0.1864 | 3.1893 ± 0.6488 † | 0.7876 ± 0.1460 | 0.7488 ± 0.0210 | 0.6434 |
| `tr_ResidualDist+LightGBM [noise_all]` | torchregress | 0.9242 ± 0.0583 | 2.7620 ± 0.2236 | 3.2000 ± 0.4630 † | 0.8002 ± 0.1105 | 0.6957 ± 0.1667 | 0.7714 |
| `tr_ResidualDist+TabFM [noise_all]` | torchregress | 0.9280 ± 0.0583 | 2.6475 ± 0.1539 | 3.0631 ± 0.5367 † | 0.7713 ± 0.1339 | 0.7659 ± 0.0611 | 3.8822 |
| `tr_ResidualDist+TabICL [noise_all]` | torchregress | 0.9242 ± 0.0572 | 2.7145 ± 0.1118 | 3.1808 ± 0.6301 † | 0.7930 ± 0.1331 | 0.7488 ± 0.0210 | 0.7902 |
| `tr_ResidualDist+TabularMLP [noise_all]` | torchregress | 0.9167 ± 0.0694 | 2.6522 ± 0.0653 | 3.0795 ± 0.4187 † | 0.7650 ± 0.1233 | 0.6540 ± 0.1029 | 1.8367 |
| `tr_Split+KumoTabular [noise_all]` | torchregress | 0.9091 ± 0.0991 | 2.7806 ± 0.1115 | 3.3838 ± 0.6994 † | 0.7863 ± 0.1523 | 0.7042 ± 0.2244 | 0.6434 |
| `tr_Split+LightGBM [noise_all]` | torchregress | 0.9242 ± 0.0430 | 2.8959 ± 0.0297 | 3.2750 ± 0.4131 † | 0.7977 ± 0.1152 | 0.6929 ± 0.1622 | 0.7714 |
| `tr_Split+TabFM [noise_all]` | torchregress | 0.9053 ± 0.0806 | 2.6782 ± 0.1058 | 3.1471 ± 0.5722 † | 0.7608 ± 0.1353 | 0.7042 ± 0.2244 | 3.8822 |
| `tr_Split+TabICL [noise_all]` | torchregress | 0.9280 ± 0.0583 | 2.9018 ± 0.1137 | 3.3054 ± 0.5100 † | 0.7836 ± 0.1471 | 0.7951 ± 0.0839 | 0.7902 |
| `tr_Split+TabularMLP [noise_all]` | torchregress | 0.9129 ± 0.0722 | 2.6880 ± 0.0727 | 3.1608 ± 0.5089 † | **0.7574 ± 0.1346** | 0.6237 ± 0.1523 | 1.8367 |

##### fm_uq_scenarios / diabetes@noise_test

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_test]` | sdm | 0.9318 ± 0.0455 | 2.4826 ± 0.0160 | 2.7910 ± 0.3723 † | 0.7665 ± 0.0721 | 0.6197 ± 0.2300 | 0.6323 |
| `Raw+LightGBM [noise_test]` | lightgbm | 0.6288 ± 0.0430 | 1.5045 ± 0.0398 | 3.9399 ± 0.3701 † | 0.7833 ± 0.1316 | 0.1465 ± 0.1375 | 0.6002 |
| `Raw+TabICL [noise_test]` | tabicl | 0.9091 ± 0.0301 | 2.3870 ± 0.0186 | **2.7494 ± 0.3755** † | 0.7710 ± 0.0521 | 0.6151 ± 0.1668 | 0.7878 |
| `Raw+TabularMLP [noise_test]` | torchregress | 0.8712 ± 0.0966 | 2.2775 ± 0.2549 | 3.0496 ± 0.6165 † | **0.7390 ± 0.0996** | 0.6151 ± 0.1668 | 1.9799 |
| `tr_CQR+KumoTabular [noise_test]` | torchregress | 0.9356 ± 0.0459 | 2.5708 ± 0.2194 | 2.8558 ± 0.3316 † | 0.7665 ± 0.0721 | 0.7196 ± 0.1667 | 0.6323 |
| `tr_CQR+LightGBM [noise_test]` | torchregress | 0.9091 ± 0.0227 | 2.5161 ± 0.1247 | 2.8912 ± 0.2048 † | 0.7833 ± 0.1316 | 0.6720 ± 0.1033 | 0.6002 |
| `tr_CQR+TabICL [noise_test]` | torchregress | 0.9242 ± 0.0561 | 2.5324 ± 0.2960 | 2.8424 ± 0.3454 † | 0.7710 ± 0.0521 | 0.6243 ± 0.2377 | 0.7878 |
| `tr_CQR+TabularMLP [noise_test]` | torchregress | 0.9205 ± 0.0495 | 2.5163 ± 0.1321 | 2.9509 ± 0.3977 † | **0.7390 ± 0.0996** | 0.6243 ± 0.1784 | 1.9799 |
| `tr_DCP+KumoTabular [noise_test]` | torchregress | 0.9394 ± 0.0537 | 2.5549 ± 0.2996 | 2.8344 ± 0.3334 † | 0.7665 ± 0.0721 | 0.7566 ± 0.2252 | 0.6323 |
| `tr_DCP+LightGBM [noise_test]` | torchregress | 0.9470 ± 0.0399 | 3.0385 ± 0.3367 | 3.2584 ± 0.1517 | 0.7833 ± 0.1316 | 0.7103 ± 0.2084 | 0.6002 |
| `tr_DCP+TabICL [noise_test]` | torchregress | 0.9280 ± 0.0537 | 2.5237 ± 0.2711 | 2.8249 ± 0.3464 † | 0.7710 ± 0.0521 | 0.6614 ± 0.2302 | 0.7878 |
| `tr_DCP+TabularMLP [noise_test]` | torchregress | 0.9129 ± 0.0525 | 2.5424 ± 0.1478 | 2.9722 ± 0.3899 † | **0.7390 ± 0.0996** | 0.6243 ± 0.1784 | 1.9799 |
| `tr_Mondrian+KumoTabular [noise_test]` | torchregress | 0.9091 ± 0.0227 | 2.6095 ± 0.2566 | 3.1775 ± 0.3100 † | 0.7665 ± 0.0721 | 0.7281 ± 0.0444 | 0.6323 |
| `tr_Mondrian+LightGBM [noise_test]` | torchregress | 0.8939 ± 0.0694 | 2.5505 ± 0.2275 | 3.2012 ± 0.7511 † | 0.7833 ± 0.1316 | 0.5724 ± 0.2654 | 0.6002 |
| `tr_Mondrian+TabFM [noise_test]` | torchregress | 0.8939 ± 0.0694 | 2.5127 ± 0.2563 | 3.1196 ± 0.4874 † | 0.7661 ± 0.0790 | 0.6653 ± 0.2244 | 3.8520 |
| `tr_Mondrian+TabICL [noise_test]` | torchregress | 0.8750 ± 0.0114 | 2.5544 ± 0.2105 | 3.1498 ± 0.2593 † | 0.7710 ± 0.0521 | 0.5808 ± 0.0310 | 0.7878 |
| `tr_Mondrian+TabularMLP [noise_test]` | torchregress | 0.8977 ± 0.0455 | 2.4788 ± 0.1993 | 3.0003 ± 0.3571 † | **0.7390 ± 0.0996** | 0.6567 ± 0.2234 | 1.9799 |
| `tr_Normalized+KumoTabular [noise_test]` | torchregress | 0.9356 ± 0.0347 | 2.7224 ± 0.3903 | 3.1092 ± 0.4606 † | 0.7665 ± 0.0721 | 0.7887 ± 0.0875 | 0.6323 |
| `tr_Normalized+LightGBM [noise_test]` | torchregress | 0.8902 ± 0.0459 | 2.6115 ± 0.2379 | 3.3452 ± 0.4371 † | 0.7833 ± 0.1316 | 0.6236 ± 0.1714 | 0.6002 |
| `tr_Normalized+TabICL [noise_test]` | torchregress | 0.9053 ± 0.0430 | 2.6394 ± 0.3705 | 3.1018 ± 0.2658 † | 0.7710 ± 0.0521 | 0.6549 ± 0.1087 | 0.7878 |
| `tr_Normalized+TabularMLP [noise_test]` | torchregress | 0.9129 ± 0.0525 | 2.5424 ± 0.1478 | 2.9722 ± 0.3899 † | **0.7390 ± 0.0996** | 0.6243 ± 0.1784 | 1.9799 |
| `tr_PITRecal+KumoTabular [noise_test]` | torchregress | 0.9091 ± 0.0521 | 2.5634 ± 0.3054 | 2.9202 ± 0.4379 † | 0.7736 ± 0.0666 | 0.5780 ± 0.1619 | 0.6323 |
| `tr_PITRecal+LightGBM [noise_test]` | torchregress | 0.9242 ± 0.0262 | 2.8250 ± 0.1785 | 3.1083 ± 0.0588 | 0.7839 ± 0.1112 | 0.5873 ± 0.2160 | 0.6002 |
| `tr_PITRecal+TabICL [noise_test]` | torchregress | 0.8826 ± 0.0328 | 2.4556 ± 0.2999 | 2.9386 ± 0.5056 † | 0.7870 ± 0.0403 | 0.5503 ± 0.1191 | 0.7878 |
| `tr_PITRecal+TabularMLP [noise_test]` | torchregress | 0.8826 ± 0.0694 | 2.3729 ± 0.1702 | 2.9755 ± 0.6058 † | 0.7547 ± 0.0837 | 0.6151 ± 0.1668 | 1.9799 |
| `tr_ResidualDist+KumoTabular [noise_test]` | torchregress | 0.8788 ± 0.0572 | 2.3912 ± 0.2967 | 3.2328 ± 0.2727 † | 0.7692 ± 0.0601 | 0.6257 ± 0.1080 | 0.6323 |
| `tr_ResidualDist+LightGBM [noise_test]` | torchregress | 0.9053 ± 0.0757 | 2.4545 ± 0.1980 | 3.2193 ± 0.8613 † | 0.7929 ± 0.1156 | 0.6508 ± 0.2147 | 0.6002 |
| `tr_ResidualDist+TabFM [noise_test]` | torchregress | 0.8902 ± 0.0512 | 2.3719 ± 0.1660 | 3.1349 ± 0.4828 † | 0.7770 ± 0.0689 | 0.5780 ± 0.1619 | 3.8520 |
| `tr_ResidualDist+TabICL [noise_test]` | torchregress | 0.8712 ± 0.0459 | 2.2705 ± 0.2285 | 3.1573 ± 0.4893 † | 0.7790 ± 0.0410 | 0.5364 ± 0.0996 | 0.7878 |
| `tr_ResidualDist+TabularMLP [noise_test]` | torchregress | 0.9015 ± 0.0459 | 2.3973 ± 0.1383 | 2.9986 ± 0.4944 † | 0.7536 ± 0.0828 | 0.6151 ± 0.1668 | 1.9799 |
| `tr_Split+KumoTabular [noise_test]` | torchregress | 0.9167 ± 0.0430 | 2.6070 ± 0.3000 | 3.1182 ± 0.2782 † | 0.7665 ± 0.0721 | 0.7652 ± 0.1077 | 0.6323 |
| `tr_Split+LightGBM [noise_test]` | torchregress | 0.8939 ± 0.0860 | 2.5520 ± 0.1655 | 3.2111 ± 0.7062 † | 0.7833 ± 0.1316 | 0.6094 ± 0.3120 | 0.6002 |
| `tr_Split+TabFM [noise_test]` | torchregress | 0.8902 ± 0.0757 | 2.4704 ± 0.2003 | 3.0850 ± 0.5169 † | 0.7661 ± 0.0790 | 0.6567 ± 0.2234 | 3.8520 |
| `tr_Split+TabICL [noise_test]` | torchregress | 0.9129 ± 0.0666 | 2.6689 ± 0.3521 | 3.1033 ± 0.2389 † | 0.7710 ± 0.0521 | 0.7090 ± 0.1629 | 0.7878 |
| `tr_Split+TabularMLP [noise_test]` | torchregress | 0.9015 ± 0.0626 | 2.4409 ± 0.0979 | 2.9515 ± 0.4491 † | **0.7390 ± 0.0996** | 0.6567 ± 0.2234 | 1.9799 |

##### fm_uq_scenarios / diabetes@outliers

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [outliers]` | sdm | 0.9583 ± 0.0459 | 3.4442 ± 0.5266 | 3.5776 ± 0.6277 † | 0.7378 ± 0.1108 | 0.8714 ± 0.1450 | 0.5858 |
| `Raw+LightGBM [outliers]` | lightgbm | 0.7879 ± 0.0399 | 2.9707 ± 0.5479 | 4.4338 ± 1.0614 † | 0.7834 ± 0.1119 | 0.4396 ± 0.0896 | 0.5489 |
| `Raw+TabICL [outliers]` | tabicl | 0.9621 ± 0.0473 | 4.0664 ± 0.1855 | 4.2292 ± 0.4135 † | 0.7541 ± 0.1193 | 0.8714 ± 0.1450 | 0.7636 |
| `Raw+TabularMLP [outliers]` | torchregress | 1.0000 ± 0.0000 | 5.6761 ± 0.3262 | 5.6761 ± 0.3262 † | 0.7404 ± 0.0840 | 1.0000 ± 0.0000 | 1.8731 |
| `tr_CQR+KumoTabular [outliers]` | torchregress | 0.9470 ± 0.0399 | 3.3429 ± 0.7272 | 3.5054 ± 0.7840 † | 0.7378 ± 0.1108 | 0.8056 ± 0.1140 | 0.5858 |
| `tr_CQR+LightGBM [outliers]` | torchregress | 0.9583 ± 0.0525 | 4.1905 ± 0.3580 | 4.4362 ± 0.6885 † | 0.7834 ± 0.1119 | 0.8010 ± 0.1511 | 0.5489 |
| `tr_CQR+TabICL [outliers]` | torchregress | 0.9394 ± 0.0347 | 3.7446 ± 0.5155 | 4.0561 ± 0.6003 † | 0.7541 ± 0.1193 | 0.7689 ± 0.1141 | 0.7636 |
| `tr_CQR+TabularMLP [outliers]` | torchregress | 0.9394 ± 0.0399 | 3.6148 ± 0.2718 | 3.9831 ± 0.6388 † | 0.7404 ± 0.0840 | 0.7182 ± 0.2289 | 1.8731 |
| `tr_DCP+KumoTabular [outliers]` | torchregress | 0.9545 ± 0.0410 | 3.4048 ± 0.9387 | 3.5649 ± 0.9438 † | 0.7378 ± 0.1108 | 0.8270 ± 0.1097 | 0.5858 |
| `tr_DCP+LightGBM [outliers]` | torchregress | 0.9621 ± 0.0459 | 5.2442 ± 0.6495 | 5.5060 ± 0.7521 † | 0.7834 ± 0.1119 | 0.7816 ± 0.1275 | 0.5489 |
| `tr_DCP+TabICL [outliers]` | torchregress | 0.9470 ± 0.0430 | 3.6055 ± 0.6410 | 3.8322 ± 0.7739 † | 0.7541 ± 0.1193 | 0.8159 ± 0.0941 | 0.7636 |
| `tr_DCP+TabularMLP [outliers]` | torchregress | 0.9470 ± 0.0525 | 3.1342 ± 0.1453 | 3.5077 ± 0.3862 † | 0.7404 ± 0.0840 | 0.7256 ± 0.2364 | 1.8731 |
| `tr_Mondrian+KumoTabular [outliers]` | torchregress | 0.9697 ± 0.0286 | 5.0570 ± 3.1105 | 5.1724 ± 3.1359 † | 0.7378 ± 0.1108 | 0.8415 ± 0.2010 | 0.5858 |
| `tr_Mondrian+LightGBM [outliers]` | torchregress | 0.9621 ± 0.0365 | 5.0670 ± 3.1869 | 5.3365 ± 3.1289 † | 0.7834 ± 0.1119 | 0.8415 ± 0.2010 | 0.5489 |
| `tr_Mondrian+TabFM [outliers]` | torchregress | 0.9470 ± 0.0286 | 4.8498 ± 3.1881 | 5.1137 ± 3.1523 † | **0.7363 ± 0.1129** | 0.7489 ± 0.1835 | 3.8529 |
| `tr_Mondrian+TabICL [outliers]` | torchregress | 0.9659 ± 0.0301 | 4.9509 ± 3.0791 | 5.1471 ± 3.1290 † | 0.7541 ± 0.1193 | 0.8162 ± 0.1929 | 0.7636 |
| `tr_Mondrian+TabularMLP [outliers]` | torchregress | 0.9432 ± 0.0410 | 5.0466 ± 3.6169 | 5.3128 ± 3.5595 † | 0.7404 ± 0.0840 | 0.7569 ± 0.1894 | 1.8731 |
| `tr_Normalized+KumoTabular [outliers]` | torchregress | 0.9394 ± 0.0237 | 3.6437 ± 0.8662 | 3.8562 ± 0.8274 † | 0.7378 ± 0.1108 | 0.7621 ± 0.0922 | 0.5858 |
| `tr_Normalized+LightGBM [outliers]` | torchregress | 0.9053 ± 0.0430 | 4.3911 ± 0.7681 | 5.1529 ± 1.0535 † | 0.7834 ± 0.1119 | 0.6839 ± 0.1011 | 0.5489 |
| `tr_Normalized+TabICL [outliers]` | torchregress | 0.9356 ± 0.0459 | 3.9758 ± 0.8454 | 4.3135 ± 0.8149 † | 0.7541 ± 0.1193 | 0.7988 ± 0.0901 | 0.7636 |
| `tr_Normalized+TabularMLP [outliers]` | torchregress | 0.9470 ± 0.0525 | 3.1342 ± 0.1453 | 3.5077 ± 0.3862 † | 0.7404 ± 0.0840 | 0.7256 ± 0.2364 | 1.8731 |
| `tr_PITRecal+KumoTabular [outliers]` | torchregress | 0.9318 ± 0.0301 | 3.1599 ± 0.4255 | 3.4578 ± 0.5821 † | 0.7466 ± 0.0866 | 0.7358 ± 0.1504 | 0.5858 |
| `tr_PITRecal+LightGBM [outliers]` | torchregress | 0.9432 ± 0.0394 | 4.7290 ± 0.5710 | 5.0336 ± 0.8869 † | 0.7995 ± 0.0898 | 0.7460 ± 0.1512 | 0.5489 |
| `tr_PITRecal+TabICL [outliers]` | torchregress | 0.9394 ± 0.0286 | 3.4777 ± 0.4888 | 3.7701 ± 0.6814 † | 0.7764 ± 0.0971 | 0.7945 ± 0.0954 | 0.7636 |
| `tr_PITRecal+TabularMLP [outliers]` | torchregress | 0.9356 ± 0.0525 | 3.0152 ± 0.0283 | 3.4097 ± 0.4255 † | 0.7665 ± 0.0735 | 0.6626 ± 0.2061 | 1.8731 |
| `tr_ResidualDist+KumoTabular [outliers]` | torchregress | 0.9508 ± 0.0473 | 2.9217 ± 0.0798 | 3.1903 ± 0.3007 † | 0.7448 ± 0.0878 | 0.8194 ± 0.1879 | 0.5858 |
| `tr_ResidualDist+LightGBM [outliers]` | torchregress | 0.9280 ± 0.0459 | 2.8408 ± 0.1185 | 3.3915 ± 0.3814 † | 0.8059 ± 0.0738 | 0.7543 ± 0.1627 | 0.5489 |
| `tr_ResidualDist+TabFM [outliers]` | torchregress | 0.9280 ± 0.0537 | 2.8678 ± 0.2657 | 3.3802 ± 0.3963 † | 0.7527 ± 0.0887 | 0.7289 ± 0.2358 | 3.8529 |
| `tr_ResidualDist+TabICL [outliers]` | torchregress | 0.9394 ± 0.0694 | 3.0136 ± 0.1454 | 3.3557 ± 0.3577 † | 0.7656 ± 0.0959 | 0.8016 ± 0.2160 | 0.7636 |
| `tr_ResidualDist+TabularMLP [outliers]` | torchregress | 0.9508 ± 0.0561 | 2.8160 ± 0.2572 | 3.1701 ± 0.1672 † | 0.7645 ± 0.0748 | 0.7090 ± 0.1629 | 1.8731 |
| `tr_Split+KumoTabular [outliers]` | torchregress | 0.9545 ± 0.0410 | 3.0092 ± 0.1796 | 3.2294 ± 0.2366 † | 0.7378 ± 0.1108 | 0.8616 ± 0.0626 | 0.5858 |
| `tr_Split+LightGBM [outliers]` | torchregress | 0.9242 ± 0.0473 | 2.8553 ± 0.2832 | 3.3618 ± 0.3533 † | 0.7834 ± 0.1119 | 0.7284 ± 0.1778 | 0.5489 |
| `tr_Split+TabFM [outliers]` | torchregress | 0.9205 ± 0.0745 | 2.8616 ± 0.3448 | 3.2543 ± 0.3497 † | **0.7363 ± 0.1129** | 0.7289 ± 0.2358 | 3.8529 |
| `tr_Split+TabICL [outliers]` | torchregress | 0.9508 ± 0.0512 | 3.1323 ± 0.3288 | 3.3738 ± 0.3299 † | 0.7541 ± 0.1193 | 0.8492 ± 0.1435 | 0.7636 |
| `tr_Split+TabularMLP [outliers]` | torchregress | 0.9432 ± 0.0691 | 2.7686 ± 0.2343 | **3.1051 ± 0.2064** † | 0.7404 ± 0.0840 | 0.6700 ± 0.2172 | 1.8731 |

##### fm_uq_scenarios / diabetes@shift

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [shift]` | sdm | 0.8788 ± 0.0131 | 2.0350 ± 0.1610 | **2.4583 ± 0.0743** | 0.6449 ± 0.0079 | 0.6939 ± 0.0471 | 0.5760 |
| `Raw+LightGBM [shift]` | lightgbm | 0.6402 ± 0.1317 | 1.3706 ± 0.1489 | 3.5072 ± 0.4895 † | 0.6981 ± 0.0368 | 0.1714 ± 0.1891 | 0.6398 |
| `Raw+TabICL [shift]` | tabicl | 0.8864 ± 0.0114 | 2.0075 ± 0.0915 | 2.4977 ± 0.1076 | **0.6344 ± 0.0109** | 0.7205 ± 0.0065 | 0.7594 |
| `Raw+TabularMLP [shift]` | torchregress | 0.9280 ± 0.0262 | 2.3258 ± 0.1955 | 2.7360 ± 0.0557 | 0.6531 ± 0.0500 | 0.7937 ± 0.0687 | 1.7980 |
| `tr_CQR+KumoTabular [shift]` | torchregress | 0.9318 ± 0.0410 | 2.2413 ± 0.1819 | 2.5199 ± 0.0506 | 0.6449 ± 0.0079 | 0.7872 ± 0.1152 | 0.5760 |
| `tr_CQR+LightGBM [shift]` | torchregress | 0.9432 ± 0.0114 | 2.4415 ± 0.1367 | 2.6748 ± 0.0393 | 0.6981 ± 0.0368 | 0.8114 ± 0.1033 | 0.6398 |
| `tr_CQR+TabICL [shift]` | torchregress | 0.9205 ± 0.0455 | 2.1791 ± 0.2172 | 2.5514 ± 0.0711 | **0.6344 ± 0.0109** | 0.7739 ± 0.0922 | 0.7594 |
| `tr_CQR+TabularMLP [shift]` | torchregress | 0.9356 ± 0.0237 | 2.4708 ± 0.2921 | 2.8040 ± 0.1176 | 0.6531 ± 0.0500 | 0.8114 ± 0.1033 | 1.7980 |
| `tr_DCP+KumoTabular [shift]` | torchregress | 0.9242 ± 0.0328 | 2.2405 ± 0.2161 | 2.5327 ± 0.1097 | 0.6449 ± 0.0079 | 0.7472 ± 0.0462 | 0.5760 |
| `tr_DCP+LightGBM [shift]` | torchregress | 0.9167 ± 0.0262 | 2.9342 ± 0.1657 | 3.1752 ± 0.1213 | 0.6981 ± 0.0368 | 0.6983 ± 0.0888 | 0.6398 |
| `tr_DCP+TabICL [shift]` | torchregress | 0.9205 ± 0.0394 | 2.1500 ± 0.1544 | 2.5332 ± 0.0767 | **0.6344 ± 0.0109** | 0.7472 ± 0.0462 | 0.7594 |
| `tr_DCP+TabularMLP [shift]` | torchregress | 0.9356 ± 0.0237 | 2.4670 ± 0.2890 | 2.8001 ± 0.1035 | 0.6531 ± 0.0500 | 0.8114 ± 0.1033 | 1.7980 |
| `tr_Mondrian+KumoTabular [shift]` | torchregress | 0.9242 ± 0.0919 | 2.5421 ± 0.5460 | 3.1025 ± 0.1514 | 0.6449 ± 0.0079 | 0.8287 ± 0.2228 | 0.5760 |
| `tr_Mondrian+LightGBM [shift]` | torchregress | 0.9318 ± 0.0227 | 2.5291 ± 0.2592 | 3.0032 ± 0.1326 | 0.6981 ± 0.0368 | 0.7349 ± 0.1426 | 0.6398 |
| `tr_Mondrian+TabFM [shift]` | torchregress | 0.9205 ± 0.0495 | 2.4247 ± 0.4162 | 2.9189 ± 0.1313 | 0.6413 ± 0.0227 | 0.7971 ± 0.1957 | 3.8408 |
| `tr_Mondrian+TabICL [shift]` | torchregress | 0.9394 ± 0.0473 | 2.5996 ± 0.4908 | 3.0433 ± 0.1434 | **0.6344 ± 0.0109** | 0.7971 ± 0.1957 | 0.7594 |
| `tr_Mondrian+TabularMLP [shift]` | torchregress | 0.9167 ± 0.0328 | 2.4406 ± 0.3555 | 2.8531 ± 0.1691 † | 0.6531 ± 0.0500 | 0.7638 ± 0.1771 | 1.7980 |
| `tr_Normalized+KumoTabular [shift]` | torchregress | 0.9053 ± 0.0656 | 2.2423 ± 0.3826 | 2.8242 ± 0.2196 † | 0.6449 ± 0.0079 | 0.7349 ± 0.1426 | 0.5760 |
| `tr_Normalized+LightGBM [shift]` | torchregress | 0.9470 ± 0.0365 | 2.7370 ± 0.3185 | 3.0465 ± 0.1083 | 0.6981 ± 0.0368 | 0.8381 ± 0.1466 | 0.6398 |
| `tr_Normalized+TabICL [shift]` | torchregress | 0.9280 ± 0.0583 | 2.3590 ± 0.4183 | 2.8559 ± 0.1497 † | **0.6344 ± 0.0109** | 0.7429 ± 0.1512 | 0.7594 |
| `tr_Normalized+TabularMLP [shift]` | torchregress | 0.9356 ± 0.0237 | 2.4670 ± 0.2890 | 2.8001 ± 0.1035 | 0.6531 ± 0.0500 | 0.8114 ± 0.1033 | 1.7980 |
| `tr_PITRecal+KumoTabular [shift]` | torchregress | 0.8523 ± 0.0710 | 1.9339 ± 0.4605 | 2.8710 ± 0.6495 † | 0.6707 ± 0.0430 | 0.6729 ± 0.0880 | 0.5760 |
| `tr_PITRecal+LightGBM [shift]` | torchregress | 0.8902 ± 0.0066 | 2.5890 ± 0.2675 | 2.9546 ± 0.1508 † | 0.7045 ± 0.0095 | 0.6502 ± 0.0584 | 0.6398 |
| `tr_PITRecal+TabICL [shift]` | torchregress | 0.8712 ± 0.0694 | 1.9430 ± 0.4036 | 2.8564 ± 0.5045 † | 0.6582 ± 0.0346 | 0.6862 ± 0.1008 | 0.7594 |
| `tr_PITRecal+TabularMLP [shift]` | torchregress | 0.9091 ± 0.0301 | 2.3835 ± 0.1823 | 2.8768 ± 0.1585 † | 0.6705 ± 0.0147 | 0.7029 ± 0.2409 | 1.7980 |
| `tr_ResidualDist+KumoTabular [shift]` | torchregress | 0.9205 ± 0.0787 | 2.6252 ± 0.2386 | 3.1865 ± 0.3462 † | 0.6741 ± 0.0479 | 0.8027 ± 0.2003 | 0.5760 |
| `tr_ResidualDist+LightGBM [shift]` | torchregress | 0.9280 ± 0.0066 | 2.6358 ± 0.2572 | 3.1130 ± 0.2317 † | 0.7329 ± 0.0390 | 0.7238 ± 0.1320 | 0.6398 |
| `tr_ResidualDist+TabFM [shift]` | torchregress | 0.9205 ± 0.0301 | 2.5103 ± 0.1383 | 2.9940 ± 0.2275 † | 0.6608 ± 0.0229 | 0.7838 ± 0.1842 | 3.8408 |
| `tr_ResidualDist+TabICL [shift]` | torchregress | 0.9280 ± 0.0473 | 2.6383 ± 0.2192 | 3.1386 ± 0.2892 † | 0.6612 ± 0.0368 | 0.7505 ± 0.1601 | 0.7594 |
| `tr_ResidualDist+TabularMLP [shift]` | torchregress | 0.9280 ± 0.0347 | 2.6238 ± 0.1354 | 3.0347 ± 0.1691 † | 0.6755 ± 0.0083 | 0.7638 ± 0.1771 | 1.7980 |
| `tr_Split+KumoTabular [shift]` | torchregress | 0.9583 ± 0.0131 | 2.7144 ± 0.1903 | 2.9785 ± 0.0507 | 0.6449 ± 0.0079 | 0.8160 ± 0.2129 | 0.5760 |
| `tr_Split+LightGBM [shift]` | torchregress | 0.9394 ± 0.0237 | 2.6426 ± 0.2171 | 2.9984 ± 0.1408 | 0.6981 ± 0.0368 | 0.7638 ± 0.1771 | 0.6398 |
| `tr_Split+TabFM [shift]` | torchregress | 0.9470 ± 0.0131 | 2.5341 ± 0.1655 | 2.8531 ± 0.0732 | 0.6413 ± 0.0227 | 0.8225 ± 0.1033 | 3.8408 |
| `tr_Split+TabICL [shift]` | torchregress | 0.9508 ± 0.0174 | 2.6242 ± 0.3894 | 2.9311 ± 0.1357 | **0.6344 ± 0.0109** | 0.7971 ± 0.1957 | 0.7594 |
| `tr_Split+TabularMLP [shift]` | torchregress | 0.9470 ± 0.0131 | 2.5142 ± 0.2867 | 2.8426 ± 0.1584 † | 0.6531 ± 0.0500 | 0.8225 ± 0.1033 | 1.7980 |
| `tr_WeightedCQR+KumoTabular [shift]` | torchregress | 0.9318 ± 0.0495 | 2.3271 ± 0.1570 | 2.6126 ± 0.0417 | 0.6449 ± 0.0079 | 0.7720 ± 0.1292 | 0.5760 |
| `tr_WeightedCQR+LightGBM [shift]` | torchregress | 0.9242 ± 0.0237 | 2.4216 ± 0.2307 | 2.7446 ± 0.0872 | 0.6981 ± 0.0368 | 0.8114 ± 0.1033 | 0.6398 |
| `tr_WeightedCQR+TabICL [shift]` | torchregress | 0.9356 ± 0.0537 | 2.2904 ± 0.2353 | 2.6814 ± 0.0365 | **0.6344 ± 0.0109** | 0.7872 ± 0.1152 | 0.7594 |
| `tr_WeightedCQR+TabularMLP [shift]` | torchregress | 0.9280 ± 0.0365 | 2.5091 ± 0.2734 | 2.9036 ± 0.1865 † | 0.6531 ± 0.0500 | 0.8114 ± 0.1033 | 1.7980 |
| `tr_WeightedSplit+KumoTabular [shift]` | torchregress | 0.9280 ± 0.0174 | 2.5347 ± 0.1222 | 3.0514 ± 0.1368 | 0.6449 ± 0.0079 | 0.7460 ± 0.1512 | 0.5760 |
| `tr_WeightedSplit+LightGBM [shift]` | torchregress | 0.9205 ± 0.0114 | 2.6264 ± 0.2331 | 3.1063 ± 0.2456 † | 0.6981 ± 0.0368 | 0.7349 ± 0.1426 | 0.6398 |
| `tr_WeightedSplit+TabFM [shift]` | torchregress | 0.9318 ± 0.0227 | 2.4742 ± 0.1624 | 2.9776 ± 0.1507 † | 0.6413 ± 0.0227 | 0.7638 ± 0.1771 | 3.8408 |
| `tr_WeightedSplit+TabICL [shift]` | torchregress | 0.9280 ± 0.0174 | 2.5654 ± 0.2043 | 3.0320 ± 0.1655 † | **0.6344 ± 0.0109** | 0.7460 ± 0.1512 | 0.7594 |
| `tr_WeightedSplit+TabularMLP [shift]` | torchregress | 0.9242 ± 0.0430 | 2.5196 ± 0.3110 | 3.0069 ± 0.1398 | 0.6531 ± 0.0500 | 0.8114 ± 0.1033 | 1.7980 |

##### fm_uq_scenarios / house_sales

_No successful rows._

##### fm_uq_scenarios / house_sales@clean

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [clean]` | sdm | 0.8908 ± 0.0137 | 0.6609 ± 0.0202 | 0.9648 ± 0.0938 † | 0.2888 ± 0.0428 | 0.8054 ± 0.0227 | 2.9847 |
| `Raw+LightGBM [clean]` | lightgbm | 0.8119 ± 0.0222 | 0.7374 ± 0.0176 | 1.4665 ± 0.0999 † | 0.3754 ± 0.0241 | 0.6336 ± 0.0275 | 2.9414 |
| `Raw+TabICL [clean]` | tabicl | 0.8836 ± 0.0229 | 0.7400 ± 0.0259 | 1.0372 ± 0.0937 † | 0.3270 ± 0.0526 | 0.8026 ± 0.0172 | 2.7676 |
| `Raw+TabularMLP [clean]` | torchregress | 0.8967 ± 0.0231 | 0.8003 ± 0.0279 | 1.1345 ± 0.0978 † | 0.3331 ± 0.0257 | 0.8099 ± 0.0401 | 31.4837 |
| `tr_CQR+KumoTabular [clean]` | torchregress | 0.8956 ± 0.0104 | 0.6609 ± 0.0379 | 0.9642 ± 0.0931 † | 0.2888 ± 0.0428 | 0.8102 ± 0.0102 | 2.9847 |
| `tr_CQR+LightGBM [clean]` | torchregress | 0.8911 ± 0.0083 | 0.8784 ± 0.0644 | 1.4030 ± 0.0806 † | 0.3754 ± 0.0241 | 0.7031 ± 0.0209 | 2.9414 |
| `tr_CQR+TabICL [clean]` | torchregress | 0.8867 ± 0.0235 | 0.7526 ± 0.0468 | 1.0356 ± 0.0898 † | 0.3270 ± 0.0526 | 0.8032 ± 0.0255 | 2.7676 |
| `tr_CQR+TabularMLP [clean]` | torchregress | 0.8839 ± 0.0164 | 0.7782 ± 0.0531 | 1.1349 ± 0.0959 † | 0.3331 ± 0.0257 | 0.7955 ± 0.0187 | 31.4837 |
| `tr_DCP+KumoTabular [clean]` | torchregress | 0.8969 ± 0.0115 | 0.6648 ± 0.0476 | **0.9641 ± 0.0928** † | 0.2888 ± 0.0428 | 0.8102 ± 0.0102 | 2.9847 |
| `tr_DCP+LightGBM [clean]` | torchregress | 0.8956 ± 0.0055 | 1.0610 ± 0.0875 | 1.5066 ± 0.1091 † | 0.3754 ± 0.0241 | 0.7217 ± 0.0059 | 2.9414 |
| `tr_DCP+TabICL [clean]` | torchregress | 0.8878 ± 0.0234 | 0.7578 ± 0.0604 | 1.0346 ± 0.0919 † | 0.3270 ± 0.0526 | 0.8058 ± 0.0235 | 2.7676 |
| `tr_DCP+TabularMLP [clean]` | torchregress | 0.8847 ± 0.0151 | 0.7746 ± 0.0613 | 1.1352 ± 0.0966 † | 0.3331 ± 0.0257 | 0.8091 ± 0.0269 | 31.4837 |
| `tr_Mondrian+KumoTabular [clean]` | torchregress | 0.8936 ± 0.0025 | 0.7401 ± 0.0319 | 1.1777 ± 0.1314 † | 0.2888 ± 0.0428 | 0.8124 ± 0.0112 | 2.9847 |
| `tr_Mondrian+LightGBM [clean]` | torchregress | 0.8933 ± 0.0090 | 0.9032 ± 0.0795 | 1.4560 ± 0.1064 † | 0.3754 ± 0.0241 | 0.7838 ± 0.0080 | 2.9414 |
| `tr_Mondrian+TabFM [clean]` | torchregress | 0.8975 ± 0.0093 | 0.7486 ± 0.0378 | 1.1724 ± 0.1327 † | 0.2889 ± 0.0415 | 0.8204 ± 0.0341 | 9.6403 |
| `tr_Mondrian+TabICL [clean]` | torchregress | 0.8919 ± 0.0117 | 0.8020 ± 0.0421 | 1.3180 ± 0.1680 † | 0.3270 ± 0.0526 | 0.7794 ± 0.0295 | 2.7676 |
| `tr_Mondrian+TabularMLP [clean]` | torchregress | 0.8850 ± 0.0156 | 0.8392 ± 0.0415 | 1.3240 ± 0.0871 † | 0.3331 ± 0.0257 | 0.7854 ± 0.0098 | 31.4837 |
| `tr_Normalized+KumoTabular [clean]` | torchregress | 0.8956 ± 0.0064 | 0.6672 ± 0.0456 | 0.9787 ± 0.0988 † | 0.2888 ± 0.0428 | 0.7950 ± 0.0099 | 2.9847 |
| `tr_Normalized+LightGBM [clean]` | torchregress | 0.8906 ± 0.0200 | 0.8188 ± 0.0607 | 1.1887 ± 0.0467 | 0.3754 ± 0.0241 | 0.7870 ± 0.0397 | 2.9414 |
| `tr_Normalized+TabICL [clean]` | torchregress | 0.8958 ± 0.0095 | 0.7726 ± 0.0641 | 1.0542 ± 0.0935 † | 0.3270 ± 0.0526 | 0.8025 ± 0.0031 | 2.7676 |
| `tr_Normalized+TabularMLP [clean]` | torchregress | 0.8847 ± 0.0151 | 0.7746 ± 0.0613 | 1.1352 ± 0.0966 † | 0.3331 ± 0.0257 | 0.8091 ± 0.0269 | 31.4837 |
| `tr_PITRecal+KumoTabular [clean]` | torchregress | 0.8958 ± 0.0098 | 0.6620 ± 0.0438 | 0.9642 ± 0.0923 † | 0.2890 ± 0.0442 | 0.8179 ± 0.0055 | 2.9847 |
| `tr_PITRecal+LightGBM [clean]` | torchregress | 0.8933 ± 0.0022 | 1.0414 ± 0.0893 | 1.5086 ± 0.1098 † | 0.3738 ± 0.0245 | 0.7227 ± 0.0139 | 2.9414 |
| `tr_PITRecal+TabICL [clean]` | torchregress | 0.8906 ± 0.0207 | 0.7583 ± 0.0610 | 1.0341 ± 0.0890 † | 0.3279 ± 0.0551 | 0.8206 ± 0.0116 | 2.7676 |
| `tr_PITRecal+TabularMLP [clean]` | torchregress | 0.8858 ± 0.0169 | 0.7751 ± 0.0558 | 1.1353 ± 0.0992 † | 0.3344 ± 0.0256 | 0.8057 ± 0.0291 | 31.4837 |
| `tr_ResidualDist+KumoTabular [clean]` | torchregress | 0.9003 ± 0.0186 | 0.7211 ± 0.0557 | 1.3658 ± 0.1553 † | **0.2887 ± 0.0432** | 0.5487 ± 0.0266 | 2.9847 |
| `tr_ResidualDist+LightGBM [clean]` | torchregress | 0.8983 ± 0.0131 | 0.8700 ± 0.0852 | 1.7256 ± 0.1250 † | 0.3752 ± 0.0241 | 0.5220 ± 0.0125 | 2.9414 |
| `tr_ResidualDist+TabFM [clean]` | torchregress | 0.9033 ± 0.0195 | 0.7328 ± 0.0591 | 1.3780 ± 0.1764 † | 0.2890 ± 0.0417 | 0.5754 ± 0.0440 | 9.6403 |
| `tr_ResidualDist+TabICL [clean]` | torchregress | 0.9014 ± 0.0192 | 0.7853 ± 0.0258 | 1.5556 ± 0.2375 † | 0.3269 ± 0.0529 | 0.5379 ± 0.0442 | 2.7676 |
| `tr_ResidualDist+TabularMLP [clean]` | torchregress | 0.8939 ± 0.0208 | 0.8222 ± 0.0695 | 1.6044 ± 0.1377 † | 0.3332 ± 0.0257 | 0.5428 ± 0.0240 | 31.4837 |
| `tr_Split+KumoTabular [clean]` | torchregress | 0.8964 ± 0.0138 | 0.6952 ± 0.0384 | 1.3674 ± 0.1486 † | 0.2888 ± 0.0428 | 0.5241 ± 0.0097 | 2.9847 |
| `tr_Split+LightGBM [clean]` | torchregress | 0.8933 ± 0.0090 | 0.8203 ± 0.0672 | 1.7434 ± 0.1089 † | 0.3754 ± 0.0241 | 0.4579 ± 0.0090 | 2.9414 |
| `tr_Split+TabFM [clean]` | torchregress | 0.9050 ± 0.0154 | 0.7298 ± 0.0530 | 1.3733 ± 0.1730 † | 0.2889 ± 0.0415 | 0.5664 ± 0.0358 | 9.6403 |
| `tr_Split+TabICL [clean]` | torchregress | 0.9036 ± 0.0109 | 0.7806 ± 0.0163 | 1.5508 ± 0.2253 † | 0.3270 ± 0.0526 | 0.5224 ± 0.0176 | 2.7676 |
| `tr_Split+TabularMLP [clean]` | torchregress | 0.8889 ± 0.0220 | 0.7864 ± 0.0386 | 1.6022 ± 0.1300 † | 0.3331 ± 0.0257 | 0.5251 ± 0.0290 | 31.4837 |

##### fm_uq_scenarios / house_sales@extrapolate

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [extrapolate]` | sdm | 0.9036 ± 0.0204 | 1.2256 ± 1.1258 | 1.4940 ± 1.1771 † | **0.3949 ± 0.3456** | 0.4407 ± 0.1604 | 2.5757 |
| `Raw+LightGBM [extrapolate]` | lightgbm | 0.7931 ± 0.0575 | 0.8981 ± 0.4379 | 3.1731 ± 3.6601 † | 0.5997 ± 0.6695 | 0.2361 ± 0.2055 | 2.8177 |
| `Raw+TabICL [extrapolate]` | tabicl | 0.9217 ± 0.0131 | 1.1449 ± 0.7643 | 2.0007 ± 1.9299 † | 0.4467 ± 0.4274 | 0.3851 ± 0.2299 | 2.5935 |
| `Raw+TabularMLP [extrapolate]` | torchregress | 0.8808 ± 0.0591 | 0.9940 ± 0.4942 | 1.9402 ± 1.7989 † | 0.4895 ± 0.4815 | 0.3687 ± 0.1177 | 31.2115 |
| `tr_CQR+KumoTabular [extrapolate]` | torchregress | 0.9133 ± 0.0183 | 1.2436 ± 1.1241 | 1.4957 ± 1.1793 † | **0.3949 ± 0.3456** | 0.4407 ± 0.1604 | 2.5757 |
| `tr_CQR+LightGBM [extrapolate]` | torchregress | 0.8422 ± 0.0815 | 1.0076 ± 0.4324 | 3.0874 ± 3.5882 † | 0.5997 ± 0.6695 | 0.3220 ± 0.2096 | 2.8177 |
| `tr_CQR+TabICL [extrapolate]` | torchregress | 0.9283 ± 0.0140 | 1.1649 ± 0.7631 | 2.0058 ± 1.9271 † | 0.4467 ± 0.4274 | 0.4407 ± 0.1604 | 2.5935 |
| `tr_CQR+TabularMLP [extrapolate]` | torchregress | 0.8767 ± 0.0578 | 0.9898 ± 0.4936 | 1.9410 ± 1.8024 † | 0.4895 ± 0.4815 | 0.3687 ± 0.1177 | 31.2115 |
| `tr_DCP+KumoTabular [extrapolate]` | torchregress | 0.9147 ± 0.0192 | 1.2955 ± 1.1973 | 1.5300 ± 1.2360 † | **0.3949 ± 0.3456** | 0.4710 ± 0.1465 | 2.5757 |
| `tr_DCP+LightGBM [extrapolate]` | torchregress | 0.8444 ± 0.0429 | 1.1621 ± 0.5918 | 3.0432 ± 3.3509 † | 0.5997 ± 0.6695 | 0.4192 ± 0.2963 | 2.8177 |
| `tr_DCP+TabICL [extrapolate]` | torchregress | 0.9300 ± 0.0139 | 1.2016 ± 0.7900 | 2.0305 ± 1.9464 † | 0.4467 ± 0.4274 | 0.3851 ± 0.2299 | 2.5935 |
| `tr_DCP+TabularMLP [extrapolate]` | torchregress | 0.8767 ± 0.0600 | 0.9840 ± 0.4801 | 1.9456 ± 1.8103 † | 0.4895 ± 0.4815 | 0.3687 ± 0.1177 | 31.2115 |
| `tr_Mondrian+KumoTabular [extrapolate]` | torchregress | 0.8114 ± 0.0010 | 0.5782 ± 0.2634 | 1.8411 ± 1.5273 † | **0.3949 ± 0.3456** | 0.4144 ± 0.2944 | 2.5757 |
| `tr_Mondrian+LightGBM [extrapolate]` | torchregress | 0.7664 ± 0.0728 | 0.6545 ± 0.2997 | 3.4946 ± 4.2553 † | 0.5997 ± 0.6695 | 0.2803 ± 0.2062 | 2.8177 |
| `tr_Mondrian+TabFM [extrapolate]` | torchregress | 0.8031 ± 0.0124 | 0.5792 ± 0.2641 | 1.9302 ± 1.6815 † | 0.4204 ± 0.3868 | 0.4495 ± 0.2463 | 9.6370 |
| `tr_Mondrian+TabICL [extrapolate]` | torchregress | 0.7631 ± 0.0512 | 0.5692 ± 0.2337 | 2.2610 ± 2.2247 † | 0.4467 ± 0.4274 | 0.3220 ± 0.2096 | 2.5935 |
| `tr_Mondrian+TabularMLP [extrapolate]` | torchregress | 0.7703 ± 0.0400 | 0.6281 ± 0.2779 | 2.4806 ± 2.5336 † | 0.4895 ± 0.4815 | 0.2812 ± 0.3006 | 31.2115 |
| `tr_Normalized+KumoTabular [extrapolate]` | torchregress | 0.9114 ± 0.0180 | 1.2461 ± 1.1194 | 1.5193 ± 1.1965 † | **0.3949 ± 0.3456** | 0.4407 ± 0.1604 | 2.5757 |
| `tr_Normalized+LightGBM [extrapolate]` | torchregress | 0.8708 ± 0.0760 | 0.9729 ± 0.4777 | 2.7588 ± 3.1599 † | 0.5997 ± 0.6695 | 0.4504 ± 0.2475 | 2.8177 |
| `tr_Normalized+TabICL [extrapolate]` | torchregress | 0.9269 ± 0.0049 | 1.1592 ± 0.7572 | 1.7364 ± 1.4389 † | 0.4467 ± 0.4274 | 0.3548 ± 0.2399 | 2.5935 |
| `tr_Normalized+TabularMLP [extrapolate]` | torchregress | 0.8767 ± 0.0600 | 0.9840 ± 0.4801 | 1.9456 ± 1.8103 † | 0.4895 ± 0.4815 | 0.3687 ± 0.1177 | 31.2115 |
| `tr_PITRecal+KumoTabular [extrapolate]` | torchregress | 0.9133 ± 0.0145 | 1.2146 ± 1.0489 | **1.4715 ± 1.1381** † | 0.3956 ± 0.3466 | 0.4710 ± 0.1465 | 2.5757 |
| `tr_PITRecal+LightGBM [extrapolate]` | torchregress | 0.8386 ± 0.0558 | 1.1551 ± 0.5954 | 3.1316 ± 3.5195 † | 0.6051 ± 0.6803 | 0.4125 ± 0.2880 | 2.8177 |
| `tr_PITRecal+TabICL [extrapolate]` | torchregress | 0.9244 ± 0.0220 | 1.1673 ± 0.7384 | 2.0283 ± 1.9446 † | 0.4467 ± 0.4290 | 0.4407 ± 0.1604 | 2.5935 |
| `tr_PITRecal+TabularMLP [extrapolate]` | torchregress | 0.8714 ± 0.0748 | 0.9750 ± 0.4704 | 2.0080 ± 1.9341 † | 0.4966 ± 0.4948 | 0.3990 ± 0.0888 | 31.2115 |
| `tr_ResidualDist+KumoTabular [extrapolate]` | torchregress | 0.8525 ± 0.1573 | 0.6834 ± 0.1228 | 2.0873 ± 1.9467 † | 0.3949 ± 0.3457 | 0.3771 ± 0.2486 | 2.5757 |
| `tr_ResidualDist+LightGBM [extrapolate]` | torchregress | 0.8297 ± 0.2055 | 0.8170 ± 0.1521 | 3.9148 ± 4.9138 † | 0.6004 ± 0.6713 | 0.2942 ± 0.1773 | 2.8177 |
| `tr_ResidualDist+TabFM [extrapolate]` | torchregress | 0.8444 ± 0.1807 | 0.6979 ± 0.1364 | 2.1999 ± 2.1144 † | 0.4200 ± 0.3870 | 0.3966 ± 0.1861 | 9.6370 |
| `tr_ResidualDist+TabICL [extrapolate]` | torchregress | 0.8400 ± 0.1935 | 0.7235 ± 0.1570 | 2.5616 ± 2.7023 † | 0.4463 ± 0.4277 | 0.3506 ± 0.2258 | 2.5935 |
| `tr_ResidualDist+TabularMLP [extrapolate]` | torchregress | 0.8283 ± 0.2107 | 0.8000 ± 0.1647 | 2.8520 ± 3.1118 † | 0.4903 ± 0.4838 | 0.2144 ± 0.1852 | 31.2115 |
| `tr_Split+KumoTabular [extrapolate]` | torchregress | 0.8481 ± 0.1587 | 0.6723 ± 0.1228 | 2.0790 ± 1.9277 † | **0.3949 ± 0.3456** | 0.3332 ± 0.2243 | 2.5757 |
| `tr_Split+LightGBM [extrapolate]` | torchregress | 0.8294 ± 0.2067 | 0.7898 ± 0.1393 | 3.9168 ± 4.9479 † | 0.5997 ± 0.6695 | 0.2516 ± 0.1615 | 2.8177 |
| `tr_Split+TabFM [extrapolate]` | torchregress | 0.8406 ± 0.1845 | 0.6934 ± 0.1394 | 2.2154 ± 2.1362 † | 0.4204 ± 0.3868 | 0.3957 ± 0.1853 | 9.6370 |
| `tr_Split+TabICL [extrapolate]` | torchregress | 0.8375 ± 0.1971 | 0.7193 ± 0.1476 | 2.5686 ± 2.7061 † | 0.4467 ± 0.4274 | 0.3484 ± 0.2242 | 2.5935 |
| `tr_Split+TabularMLP [extrapolate]` | torchregress | 0.8331 ± 0.2047 | 0.7922 ± 0.1647 | 2.8280 ± 3.0724 † | 0.4895 ± 0.4815 | 0.2171 ± 0.1899 | 31.2115 |

##### fm_uq_scenarios / house_sales@heavy_tail

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [heavy_tail]` | sdm | 0.9989 ± 0.0019 | 0.0050 ± 0.0037 | 0.0050 ± 0.0037 † | 8.857e-04 ± 8.294e-04 | 0.9954 ± 0.0080 | 2.5747 |
| `Raw+LightGBM [heavy_tail]` | lightgbm | 0.6017 ± 0.3262 | 0.0745 ± 0.0419 | 0.0745 ± 0.0419 † | 6.023e-04 ± 1.042e-03 | 0.0740 ± 0.0844 | 2.4948 |
| `Raw+TabICL [heavy_tail]` | tabicl | 0.9950 ± 0.0043 | 0.0021 ± 0.0007 | 0.0025 ± 0.0006 † | 8.396e-04 ± 9.035e-04 | 0.9869 ± 0.0073 | 2.6148 |
| `Raw+TabularMLP [heavy_tail]` | torchregress | 0.9997 ± 0.0005 | 0.0505 ± 0.0420 | 0.0507 ± 0.0418 † | 0.0023 ± 0.0023 | 0.9969 ± 0.0054 | 29.6983 |
| `tr_CQR+KumoTabular [heavy_tail]` | torchregress | 0.9025 ± 0.0075 | 0.0038 ± 0.0036 | 0.0041 ± 0.0036 † | 8.857e-04 ± 8.294e-04 | 0.7475 ± 0.1111 | 2.5747 |
| `tr_CQR+LightGBM [heavy_tail]` | torchregress | 0.8958 ± 0.0042 | 0.0745 ± 0.0419 | 0.0745 ± 0.0419 † | 6.023e-04 ± 1.042e-03 | 0.5399 ± 0.1000 | 2.4948 |
| `tr_CQR+TabICL [heavy_tail]` | torchregress | 0.8981 ± 0.0112 | 0.0014 ± 0.0008 | 0.0020 ± 0.0007 † | 8.396e-04 ± 9.035e-04 | 0.8496 ± 0.0110 | 2.6148 |
| `tr_CQR+TabularMLP [heavy_tail]` | torchregress | 0.9142 ± 0.0046 | 0.0138 ± 0.0187 | 0.0157 ± 0.0200 † | 0.0023 ± 0.0023 | 0.8111 ± 0.0062 | 29.6983 |
| `tr_DCP+KumoTabular [heavy_tail]` | torchregress | 0.8992 ± 0.0115 | 0.0012 ± 0.0004 | 0.0017 ± 0.0005 † | 8.857e-04 ± 8.294e-04 | 0.7354 ± 0.0814 | 2.5747 |
| `tr_DCP+LightGBM [heavy_tail]` | torchregress | 0.9019 ± 0.0185 | 0.1724 ± 0.0885 | 0.1724 ± 0.0885 † | 6.023e-04 ± 1.042e-03 | 0.5797 ± 0.2152 | 2.4948 |
| `tr_DCP+TabICL [heavy_tail]` | torchregress | 0.8936 ± 0.0246 | 9.441e-04 ± 2.474e-04 | 0.0016 ± 0.0005 † | 8.396e-04 ± 9.035e-04 | 0.8268 ± 0.0568 | 2.6148 |
| `tr_DCP+TabularMLP [heavy_tail]` | torchregress | 0.9019 ± 0.0035 | 0.0049 ± 0.0075 | 0.0064 ± 0.0089 † | 0.0023 ± 0.0023 | 0.7887 ± 0.0371 | 29.6983 |
| `tr_Mondrian+KumoTabular [heavy_tail]` | torchregress | 0.9083 ± 0.0088 | 8.704e-04 ± 1.871e-04 | 0.0014 ± 0.0006 † | 8.857e-04 ± 8.294e-04 | 0.8503 ± 0.0297 | 2.5747 |
| `tr_Mondrian+LightGBM [heavy_tail]` | torchregress | 0.8925 ± 0.0113 | 9.782e-09 ± 6.092e-09 | **3.485e-04 ± 6.022e-04** † | 6.023e-04 ± 1.042e-03 | 0.6203 ± 0.1921 | 2.4948 |
| `tr_Mondrian+TabFM [heavy_tail]` | torchregress | 0.9044 ± 0.0100 | 0.0174 ± 0.0041 | 0.2098 ± 0.2480 † | 0.2996 ± 0.3866 | 0.7717 ± 0.0527 | 9.6964 |
| `tr_Mondrian+TabICL [heavy_tail]` | torchregress | 0.8944 ± 0.0205 | 6.943e-04 ± 1.203e-04 | 0.0012 ± 0.0006 † | 8.396e-04 ± 9.035e-04 | 0.7989 ± 0.0539 | 2.6148 |
| `tr_Mondrian+TabularMLP [heavy_tail]` | torchregress | 0.9008 ± 0.0044 | 0.0041 ± 0.0063 | 0.0049 ± 0.0067 † | 0.0023 ± 0.0023 | 0.8262 ± 0.0232 | 29.6983 |
| `tr_Normalized+KumoTabular [heavy_tail]` | torchregress | 0.8986 ± 0.0137 | 0.0022 ± 0.0012 | 0.0024 ± 0.0013 † | 8.857e-04 ± 8.294e-04 | 0.7332 ± 0.0837 | 2.5747 |
| `tr_Normalized+LightGBM [heavy_tail]` | torchregress | 0.8897 ± 0.0189 | 9.656e-05 ± 7.373e-05 | 4.198e-04 ± 5.480e-04 † | 6.023e-04 ± 1.042e-03 | 0.3584 ± 0.1898 | 2.4948 |
| `tr_Normalized+TabICL [heavy_tail]` | torchregress | 0.8942 ± 0.0232 | 0.0012 ± 0.0006 | 0.0018 ± 0.0006 † | 8.396e-04 ± 9.035e-04 | 0.8252 ± 0.0508 | 2.6148 |
| `tr_Normalized+TabularMLP [heavy_tail]` | torchregress | 0.9019 ± 0.0035 | 0.0049 ± 0.0075 | 0.0064 ± 0.0089 † | 0.0023 ± 0.0023 | 0.7887 ± 0.0371 | 29.6983 |
| `tr_PITRecal+KumoTabular [heavy_tail]` | torchregress | 0.9094 ± 0.0067 | 6.472e-04 ± 2.081e-04 | 0.0012 ± 0.0002 † | 8.294e-04 ± 7.760e-04 | 0.7918 ± 0.0375 | 2.5747 |
| `tr_PITRecal+LightGBM [heavy_tail]` | torchregress | 0.9033 ± 0.0017 | 0.1086 ± 0.0703 | 0.1086 ± 0.0703 † | 0.4854 ± 0.8391 | 0.5905 ± 0.0218 | 2.4948 |
| `tr_PITRecal+TabICL [heavy_tail]` | torchregress | 0.8925 ± 0.0230 | 9.442e-04 ± 2.542e-04 | 0.0016 ± 0.0006 † | 8.382e-04 ± 9.048e-04 | 0.8160 ± 0.0633 | 2.6148 |
| `tr_PITRecal+TabularMLP [heavy_tail]` | torchregress | 0.9008 ± 0.0079 | 0.0050 ± 0.0080 | 0.0063 ± 0.0091 † | 0.0021 ± 0.0020 | 0.7820 ± 0.0318 | 29.6983 |
| `tr_ResidualDist+KumoTabular [heavy_tail]` | torchregress | 0.9103 ± 0.0043 | 6.093e-04 ± 1.791e-04 | 0.0012 ± 0.0003 † | 7.722e-04 ± 8.729e-04 | 0.8275 ± 0.0396 | 2.5747 |
| `tr_ResidualDist+LightGBM [heavy_tail]` | torchregress | 0.8889 ± 0.0128 | 2.706e-09 ± 2.021e-09 | 3.485e-04 ± 6.022e-04 † | **6.023e-04 ± 1.042e-03** | 0.3807 ± 0.2175 | 2.4948 |
| `tr_ResidualDist+TabFM [heavy_tail]` | torchregress | 0.9056 ± 0.0081 | 0.0209 ± 0.0035 | 0.2196 ± 0.2491 † | 0.2996 ± 0.3866 | 0.6316 ± 0.0513 | 9.6964 |
| `tr_ResidualDist+TabICL [heavy_tail]` | torchregress | 0.8928 ± 0.0237 | 9.530e-04 ± 2.414e-04 | 0.0017 ± 0.0006 † | 8.385e-04 ± 9.045e-04 | 0.7548 ± 0.0339 | 2.6148 |
| `tr_ResidualDist+TabularMLP [heavy_tail]` | torchregress | 0.9067 ± 0.0066 | 0.0055 ± 0.0087 | 0.0069 ± 0.0100 † | 0.0022 ± 0.0022 | 0.7821 ± 0.0086 | 29.6983 |
| `tr_Split+KumoTabular [heavy_tail]` | torchregress | 0.9042 ± 0.0058 | 0.0011 ± 0.0003 | 0.0018 ± 0.0005 † | 8.857e-04 ± 8.294e-04 | 0.7650 ± 0.0491 | 2.5747 |
| `tr_Split+LightGBM [heavy_tail]` | torchregress | 0.8953 ± 0.0173 | 1.548e-09 ± 6.290e-10 | 3.485e-04 ± 6.022e-04 † | 6.023e-04 ± 1.042e-03 | 0.2450 ± 0.1365 | 2.4948 |
| `tr_Split+TabFM [heavy_tail]` | torchregress | 0.9083 ± 0.0123 | 0.0222 ± 0.0059 | 0.2208 ± 0.2485 † | 0.2996 ± 0.3866 | 0.6926 ± 0.1105 | 9.6964 |
| `tr_Split+TabICL [heavy_tail]` | torchregress | 0.8878 ± 0.0279 | 9.420e-04 ± 2.419e-04 | 0.0017 ± 0.0006 † | 8.396e-04 ± 9.035e-04 | 0.7406 ± 0.0417 | 2.6148 |
| `tr_Split+TabularMLP [heavy_tail]` | torchregress | 0.9042 ± 0.0043 | 0.0057 ± 0.0088 | 0.0074 ± 0.0106 † | 0.0023 ± 0.0023 | 0.7660 ± 0.0851 | 29.6983 |

##### fm_uq_scenarios / house_sales@imbalance

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [imbalance]` | sdm | 0.8642 ± 0.0180 | 0.7494 ± 0.0102 | 1.1541 ± 0.0890 † | 0.3651 ± 0.0355 | 0.7813 ± 0.0292 | 2.1491 |
| `Raw+LightGBM [imbalance]` | lightgbm | 0.7281 ± 0.0076 | 0.7127 ± 0.0218 | 2.4031 ± 0.2441 † | 0.6146 ± 0.0712 | 0.4362 ± 0.0421 | 2.6096 |
| `Raw+TabICL [imbalance]` | tabicl | 0.8653 ± 0.0075 | 0.8247 ± 0.0378 | 1.3323 ± 0.0795 † | 0.4449 ± 0.0158 | 0.7795 ± 0.0238 | 2.2689 |
| `Raw+TabularMLP [imbalance]` | torchregress | 0.8678 ± 0.0230 | 1.5262 ± 0.5893 | 2.1013 ± 0.6068 † | 0.5226 ± 0.0137 | 0.7392 ± 0.0474 | 21.8889 |
| `tr_CQR+KumoTabular [imbalance]` | torchregress | 0.8725 ± 0.0145 | 0.7580 ± 0.0358 | 1.1510 ± 0.0805 † | 0.3651 ± 0.0355 | 0.7833 ± 0.0174 | 2.1491 |
| `tr_CQR+LightGBM [imbalance]` | torchregress | 0.8214 ± 0.0171 | 0.8556 ± 0.0368 | 2.2320 ± 0.2314 † | 0.6146 ± 0.0712 | 0.5304 ± 0.0524 | 2.6096 |
| `tr_CQR+TabICL [imbalance]` | torchregress | 0.8800 ± 0.0116 | 0.8517 ± 0.0211 | 1.3250 ± 0.0766 † | 0.4449 ± 0.0158 | 0.7980 ± 0.0038 | 2.2689 |
| `tr_CQR+TabularMLP [imbalance]` | torchregress | 0.8589 ± 0.0161 | 1.5081 ± 0.6029 | 2.1062 ± 0.6040 † | 0.5226 ± 0.0137 | 0.7275 ± 0.0321 | 21.8889 |
| `tr_DCP+KumoTabular [imbalance]` | torchregress | 0.8711 ± 0.0137 | 0.7620 ± 0.0575 | **1.1486 ± 0.0803** † | 0.3651 ± 0.0355 | 0.7854 ± 0.0095 | 2.1491 |
| `tr_DCP+LightGBM [imbalance]` | torchregress | 0.8297 ± 0.0156 | 1.0400 ± 0.0601 | 2.2882 ± 0.2032 † | 0.6146 ± 0.0712 | 0.5508 ± 0.0356 | 2.6096 |
| `tr_DCP+TabICL [imbalance]` | torchregress | 0.8817 ± 0.0117 | 0.8695 ± 0.0199 | 1.3258 ± 0.0749 † | 0.4449 ± 0.0158 | 0.8029 ± 0.0054 | 2.2689 |
| `tr_DCP+TabularMLP [imbalance]` | torchregress | 0.8567 ± 0.0145 | 1.4777 ± 0.6182 | 2.0989 ± 0.6020 † | 0.5226 ± 0.0137 | 0.7264 ± 0.0338 | 21.8889 |
| `tr_Mondrian+KumoTabular [imbalance]` | torchregress | 0.8333 ± 0.0100 | 0.5757 ± 0.0247 | 1.6717 ± 0.0911 † | 0.3651 ± 0.0355 | 0.6074 ± 0.0357 | 2.1491 |
| `tr_Mondrian+LightGBM [imbalance]` | torchregress | 0.8156 ± 0.0211 | 0.6720 ± 0.0539 | 2.7838 ± 0.5179 † | 0.6146 ± 0.0712 | 0.5244 ± 0.0651 | 2.6096 |
| `tr_Mondrian+TabFM [imbalance]` | torchregress | 0.8461 ± 0.0242 | 0.5857 ± 0.0202 | 1.6180 ± 0.1974 † | 0.3637 ± 0.0448 | 0.6594 ± 0.0374 | 8.6570 |
| `tr_Mondrian+TabICL [imbalance]` | torchregress | 0.8342 ± 0.0067 | 0.6290 ± 0.0032 | 2.0200 ± 0.0955 | 0.4449 ± 0.0158 | 0.6150 ± 0.0042 | 2.2689 |
| `tr_Mondrian+TabularMLP [imbalance]` | torchregress | 0.8325 ± 0.0189 | 0.6606 ± 0.0261 | 2.2438 ± 0.1746 † | 0.5226 ± 0.0137 | 0.6158 ± 0.0219 | 21.8889 |
| `tr_Normalized+KumoTabular [imbalance]` | torchregress | 0.8678 ± 0.0188 | 0.7719 ± 0.0531 | 1.1872 ± 0.0790 † | 0.3651 ± 0.0355 | 0.7670 ± 0.0184 | 2.1491 |
| `tr_Normalized+LightGBM [imbalance]` | torchregress | 0.8286 ± 0.0113 | 0.8110 ± 0.0320 | 2.1094 ± 0.2062 † | 0.6146 ± 0.0712 | 0.6050 ± 0.0328 | 2.6096 |
| `tr_Normalized+TabICL [imbalance]` | torchregress | 0.8708 ± 0.0154 | 0.8728 ± 0.0301 | 1.3522 ± 0.0241 | 0.4449 ± 0.0158 | 0.7725 ± 0.0131 | 2.2689 |
| `tr_Normalized+TabularMLP [imbalance]` | torchregress | 0.8567 ± 0.0145 | 1.4777 ± 0.6182 | 2.0989 ± 0.6020 † | 0.5226 ± 0.0137 | 0.7264 ± 0.0338 | 21.8889 |
| `tr_PITRecal+KumoTabular [imbalance]` | torchregress | 0.8694 ± 0.0145 | 0.7616 ± 0.0539 | 1.1558 ± 0.0740 † | 0.3645 ± 0.0370 | 0.7810 ± 0.0135 | 2.1491 |
| `tr_PITRecal+LightGBM [imbalance]` | torchregress | 0.8253 ± 0.0248 | 1.0322 ± 0.0770 | 2.3108 ± 0.2283 † | 0.6061 ± 0.0716 | 0.5380 ± 0.0483 | 2.6096 |
| `tr_PITRecal+TabICL [imbalance]` | torchregress | 0.8722 ± 0.0157 | 0.8481 ± 0.0418 | 1.3418 ± 0.0733 † | 0.4464 ± 0.0208 | 0.7811 ± 0.0263 | 2.2689 |
| `tr_PITRecal+TabularMLP [imbalance]` | torchregress | 0.8539 ± 0.0151 | 1.4800 ± 0.6196 | 2.1153 ± 0.5822 † | 0.5768 ± 0.0863 | 0.7133 ± 0.0487 | 21.8889 |
| `tr_ResidualDist+KumoTabular [imbalance]` | torchregress | 0.7989 ± 0.0112 | 0.5202 ± 0.0180 | 1.8794 ± 0.1279 † | 0.3650 ± 0.0357 | 0.4698 ± 0.0061 | 2.1491 |
| `tr_ResidualDist+LightGBM [imbalance]` | torchregress | 0.7808 ± 0.0101 | 0.5971 ± 0.0212 | 3.0686 ± 0.4366 † | 0.6139 ± 0.0712 | 0.3747 ± 0.0050 | 2.6096 |
| `tr_ResidualDist+TabFM [imbalance]` | torchregress | 0.8167 ± 0.0188 | 0.5297 ± 0.0014 | 1.8129 ± 0.2172 † | **0.3633 ± 0.0451** | 0.5117 ± 0.0428 | 8.6570 |
| `tr_ResidualDist+TabICL [imbalance]` | torchregress | 0.8072 ± 0.0084 | 0.5676 ± 0.0143 | 2.2651 ± 0.0850 | 0.4450 ± 0.0168 | 0.4710 ± 0.0210 | 2.2689 |
| `tr_ResidualDist+TabularMLP [imbalance]` | torchregress | 0.8017 ± 0.0175 | 0.5933 ± 0.0227 | 2.5215 ± 0.1541 † | 0.5239 ± 0.0135 | 0.4597 ± 0.0110 | 21.8889 |
| `tr_Split+KumoTabular [imbalance]` | torchregress | 0.7972 ± 0.0104 | 0.5191 ± 0.0139 | 1.8887 ± 0.1326 † | 0.3651 ± 0.0355 | 0.4617 ± 0.0097 | 2.1491 |
| `tr_Split+LightGBM [imbalance]` | torchregress | 0.7811 ± 0.0096 | 0.5969 ± 0.0181 | 3.1255 ± 0.4324 † | 0.6146 ± 0.0712 | 0.3601 ± 0.0100 | 2.6096 |
| `tr_Split+TabFM [imbalance]` | torchregress | 0.8131 ± 0.0143 | 0.5263 ± 0.0085 | 1.8134 ± 0.2250 † | 0.3637 ± 0.0448 | 0.5027 ± 0.0342 | 8.6570 |
| `tr_Split+TabICL [imbalance]` | torchregress | 0.7997 ± 0.0119 | 0.5568 ± 0.0200 | 2.2896 ± 0.0963 | 0.4449 ± 0.0158 | 0.4525 ± 0.0172 | 2.2689 |
| `tr_Split+TabularMLP [imbalance]` | torchregress | 0.8011 ± 0.0159 | 0.5934 ± 0.0234 | 2.5250 ± 0.1418 † | 0.5226 ± 0.0137 | 0.4563 ± 0.0031 | 21.8889 |

##### fm_uq_scenarios / house_sales@noise_all

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_all]` | sdm | 0.8878 ± 0.0132 | 1.1841 ± 0.0380 | 1.6364 ± 0.0980 † | 0.4660 ± 0.0536 | 0.7568 ± 0.0370 | 2.5621 |
| `Raw+LightGBM [noise_all]` | lightgbm | 0.8083 ± 0.0126 | 1.0711 ± 0.0227 | 2.1359 ± 0.1228 † | 0.5343 ± 0.0258 | 0.5044 ± 0.0432 | 4.5227 |
| `Raw+TabICL [noise_all]` | tabicl | 0.8925 ± 0.0082 | 1.2386 ± 0.0422 | 1.6805 ± 0.0764 | 0.4800 ± 0.0538 | 0.7674 ± 0.0218 | 2.5898 |
| `Raw+TabularMLP [noise_all]` | torchregress | 0.9103 ± 0.0050 | 1.4290 ± 0.1062 | 1.9111 ± 0.1545 † | 0.5063 ± 0.0431 | 0.7884 ± 0.0399 | 18.8751 |
| `tr_CQR+KumoTabular [noise_all]` | torchregress | 0.8931 ± 0.0063 | 1.1965 ± 0.0529 | **1.6343 ± 0.0985** † | 0.4660 ± 0.0536 | 0.7680 ± 0.0554 | 2.5621 |
| `tr_CQR+LightGBM [noise_all]` | torchregress | 0.9028 ± 0.0041 | 1.2974 ± 0.0559 | 2.0485 ± 0.1245 † | 0.5343 ± 0.0258 | 0.6313 ± 0.0377 | 4.5227 |
| `tr_CQR+TabICL [noise_all]` | torchregress | 0.8947 ± 0.0057 | 1.2378 ± 0.0678 | 1.6802 ± 0.0754 | 0.4800 ± 0.0538 | 0.7796 ± 0.0131 | 2.5898 |
| `tr_CQR+TabularMLP [noise_all]` | torchregress | 0.8914 ± 0.0088 | 1.3501 ± 0.1574 | 1.9108 ± 0.1551 † | 0.5063 ± 0.0431 | 0.7750 ± 0.0506 | 18.8751 |
| `tr_DCP+KumoTabular [noise_all]` | torchregress | 0.8922 ± 0.0061 | 1.1991 ± 0.0594 | 1.6391 ± 0.1016 † | 0.4660 ± 0.0536 | 0.7611 ± 0.0394 | 2.5621 |
| `tr_DCP+LightGBM [noise_all]` | torchregress | 0.9017 ± 0.0142 | 1.5064 ± 0.1074 | 2.1274 ± 0.1066 † | 0.5343 ± 0.0258 | 0.6114 ± 0.0684 | 4.5227 |
| `tr_DCP+TabICL [noise_all]` | torchregress | 0.8933 ± 0.0055 | 1.2406 ± 0.0810 | 1.6826 ± 0.0760 | 0.4800 ± 0.0538 | 0.7747 ± 0.0023 | 2.5898 |
| `tr_DCP+TabularMLP [noise_all]` | torchregress | 0.8908 ± 0.0109 | 1.3190 ± 0.1711 | 1.9041 ± 0.1517 † | 0.5063 ± 0.0431 | 0.7711 ± 0.0539 | 18.8751 |
| `tr_Mondrian+KumoTabular [noise_all]` | torchregress | 0.9000 ± 0.0142 | 1.2762 ± 0.1062 | 1.9558 ± 0.2051 † | 0.4660 ± 0.0536 | 0.7600 ± 0.0573 | 2.5621 |
| `tr_Mondrian+LightGBM [noise_all]` | torchregress | 0.8931 ± 0.0082 | 1.3785 ± 0.0549 | 2.1825 ± 0.1481 † | 0.5343 ± 0.0258 | 0.7298 ± 0.0199 | 4.5227 |
| `tr_Mondrian+TabFM [noise_all]` | torchregress | 0.9019 ± 0.0093 | 1.2827 ± 0.0560 | 1.8798 ± 0.1774 † | **0.4607 ± 0.0440** | 0.7582 ± 0.0464 | 9.6900 |
| `tr_Mondrian+TabICL [noise_all]` | torchregress | 0.8953 ± 0.0073 | 1.3163 ± 0.0749 | 2.0139 ± 0.2105 † | 0.4800 ± 0.0538 | 0.7440 ± 0.0313 | 2.5898 |
| `tr_Mondrian+TabularMLP [noise_all]` | torchregress | 0.8983 ± 0.0029 | 1.3348 ± 0.0332 | 2.0275 ± 0.1698 † | 0.5063 ± 0.0431 | 0.7666 ± 0.0167 | 18.8751 |
| `tr_Normalized+KumoTabular [noise_all]` | torchregress | 0.8875 ± 0.0109 | 1.1673 ± 0.0892 | 1.7479 ± 0.1436 † | 0.4660 ± 0.0536 | 0.7579 ± 0.0333 | 2.5621 |
| `tr_Normalized+LightGBM [noise_all]` | torchregress | 0.8933 ± 0.0067 | 1.2551 ± 0.0826 | 1.9532 ± 0.1116 † | 0.5343 ± 0.0258 | 0.7000 ± 0.0145 | 4.5227 |
| `tr_Normalized+TabICL [noise_all]` | torchregress | 0.8908 ± 0.0079 | 1.2108 ± 0.0882 | 1.7882 ± 0.1209 † | 0.4800 ± 0.0538 | 0.7571 ± 0.0355 | 2.5898 |
| `tr_Normalized+TabularMLP [noise_all]` | torchregress | 0.8908 ± 0.0109 | 1.3190 ± 0.1711 | 1.9041 ± 0.1517 † | 0.5063 ± 0.0431 | 0.7711 ± 0.0539 | 18.8751 |
| `tr_PITRecal+KumoTabular [noise_all]` | torchregress | 0.8953 ± 0.0034 | 1.1944 ± 0.0599 | 1.6396 ± 0.1040 † | 0.4668 ± 0.0551 | 0.7842 ± 0.0211 | 2.5621 |
| `tr_PITRecal+LightGBM [noise_all]` | torchregress | 0.9011 ± 0.0107 | 1.4771 ± 0.1055 | 2.1231 ± 0.1116 † | 0.5334 ± 0.0245 | 0.6214 ± 0.0570 | 4.5227 |
| `tr_PITRecal+TabICL [noise_all]` | torchregress | 0.8942 ± 0.0052 | 1.2418 ± 0.0663 | 1.6806 ± 0.0754 | 0.4776 ± 0.0529 | 0.7722 ± 0.0037 | 2.5898 |
| `tr_PITRecal+TabularMLP [noise_all]` | torchregress | 0.8922 ± 0.0113 | 1.3625 ± 0.1629 | 1.8612 ± 0.1583 † | 0.5190 ± 0.0427 | 0.7560 ± 0.0588 | 18.8751 |
| `tr_ResidualDist+KumoTabular [noise_all]` | torchregress | 0.8972 ± 0.0035 | 1.2997 ± 0.0904 | 2.1917 ± 0.2195 † | 0.4663 ± 0.0541 | 0.5390 ± 0.0049 | 2.5621 |
| `tr_ResidualDist+LightGBM [noise_all]` | torchregress | 0.9000 ± 0.0068 | 1.4073 ± 0.1125 | 2.3843 ± 0.1748 † | 0.5342 ± 0.0254 | 0.5321 ± 0.0592 | 4.5227 |
| `tr_ResidualDist+TabFM [noise_all]` | torchregress | 0.9008 ± 0.0022 | 1.3253 ± 0.0990 | 2.2141 ± 0.2040 † | 0.4613 ± 0.0444 | 0.5341 ± 0.0262 | 9.6900 |
| `tr_ResidualDist+TabICL [noise_all]` | torchregress | 0.8997 ± 0.0059 | 1.3207 ± 0.0754 | 2.2396 ± 0.2314 † | 0.4789 ± 0.0536 | 0.5359 ± 0.0050 | 2.5898 |
| `tr_ResidualDist+TabularMLP [noise_all]` | torchregress | 0.8939 ± 0.0055 | 1.3643 ± 0.1132 | 2.3467 ± 0.2324 † | 0.5073 ± 0.0434 | 0.5186 ± 0.0131 | 18.8751 |
| `tr_Split+KumoTabular [noise_all]` | torchregress | 0.9022 ± 0.0049 | 1.2598 ± 0.0735 | 2.2188 ± 0.2215 † | 0.4660 ± 0.0536 | 0.5189 ± 0.0039 | 2.5621 |
| `tr_Split+LightGBM [noise_all]` | torchregress | 0.8981 ± 0.0104 | 1.3056 ± 0.0495 | 2.4548 ± 0.1732 † | 0.5343 ± 0.0258 | 0.4614 ± 0.0705 | 4.5227 |
| `tr_Split+TabFM [noise_all]` | torchregress | 0.8981 ± 0.0021 | 1.2940 ± 0.0711 | 2.2080 ± 0.2054 † | **0.4607 ± 0.0440** | 0.5199 ± 0.0276 | 9.6900 |
| `tr_Split+TabICL [noise_all]` | torchregress | 0.9061 ± 0.0038 | 1.3190 ± 0.0798 | 2.2768 ± 0.2321 † | 0.4800 ± 0.0538 | 0.5096 ± 0.0123 | 2.5898 |
| `tr_Split+TabularMLP [noise_all]` | torchregress | 0.8933 ± 0.0052 | 1.3186 ± 0.0907 | 2.3495 ± 0.2310 † | 0.5063 ± 0.0431 | 0.4721 ± 0.0129 | 18.8751 |

##### fm_uq_scenarios / house_sales@noise_test

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_test]` | sdm | 0.9994 ± 0.0005 | 14.5251 ± 3.4295 | 14.5275 ± 3.4305 † | 0.6021 ± 0.0550 | 0.9946 ± 0.0048 | 2.5717 |
| `Raw+LightGBM [noise_test]` | lightgbm | 0.7506 ± 0.0113 | 1.2751 ± 0.0302 | 2.4643 ± 0.1305 † | 0.7554 ± 0.0232 | 0.5410 ± 0.0273 | 2.8532 |
| `Raw+TabICL [noise_test]` | tabicl | 0.9936 ± 0.0017 | 9.1284 ± 0.2450 | 9.1635 ± 0.2480 | 1.2666 ± 0.1439 | 0.9826 ± 0.0016 | 2.5809 |
| `Raw+TabularMLP [noise_test]` | torchregress | 0.6817 ± 0.0198 | 0.8886 ± 0.0491 | 2.8139 ± 0.4091 † | 0.5598 ± 0.0631 | 0.5823 ± 0.0479 | 31.6715 |
| `tr_CQR+KumoTabular [noise_test]` | torchregress | 0.9994 ± 0.0005 | 14.5251 ± 3.4298 | 14.5274 ± 3.4309 † | 0.6021 ± 0.0550 | 0.9946 ± 0.0048 | 2.5717 |
| `tr_CQR+LightGBM [noise_test]` | torchregress | 0.8267 ± 0.0229 | 1.4162 ± 0.0779 | 2.3119 ± 0.1380 † | 0.7554 ± 0.0232 | 0.6799 ± 0.0608 | 2.8532 |
| `tr_CQR+TabICL [noise_test]` | torchregress | 0.9936 ± 0.0017 | 9.1410 ± 0.2677 | 9.1755 ± 0.2690 | 1.2666 ± 0.1439 | 0.9826 ± 0.0016 | 2.5809 |
| `tr_CQR+TabularMLP [noise_test]` | torchregress | 0.6669 ± 0.0305 | 0.8665 ± 0.0614 | 2.8652 ± 0.4266 † | 0.5598 ± 0.0631 | 0.5792 ± 0.0493 | 31.6715 |
| `tr_DCP+KumoTabular [noise_test]` | torchregress | 0.9994 ± 0.0005 | 14.6400 ± 3.6513 | 14.6415 ± 3.6531 † | 0.6021 ± 0.0550 | 0.9946 ± 0.0048 | 2.5717 |
| `tr_DCP+LightGBM [noise_test]` | torchregress | 0.8594 ± 0.0091 | 1.6135 ± 0.0621 | 2.2840 ± 0.1772 † | 0.7554 ± 0.0232 | 0.6297 ± 0.0283 | 2.8532 |
| `tr_DCP+TabICL [noise_test]` | torchregress | 0.9922 ± 0.0048 | 9.4580 ± 1.0032 | 9.4929 ± 0.9877 † | 1.2666 ± 0.1439 | 0.9803 ± 0.0051 | 2.5809 |
| `tr_DCP+TabularMLP [noise_test]` | torchregress | 0.6686 ± 0.0283 | 0.8595 ± 0.0689 | 2.8658 ± 0.4227 † | 0.5598 ± 0.0631 | 0.5667 ± 0.0599 | 31.6715 |
| `tr_Mondrian+KumoTabular [noise_test]` | torchregress | 0.5942 ± 0.0454 | 0.7599 ± 0.0319 | 3.4145 ± 0.3989 † | 0.6021 ± 0.0550 | 0.4356 ± 0.0866 | 2.5717 |
| `tr_Mondrian+LightGBM [noise_test]` | torchregress | 0.5797 ± 0.0248 | 1.1680 ± 0.1040 | 3.7974 ± 0.3697 † | 0.7554 ± 0.0232 | 0.3457 ± 0.0371 | 2.8532 |
| `tr_Mondrian+TabFM [noise_test]` | torchregress | 0.3036 ± 0.0190 | 1.1624 ± 0.0599 | 7.4763 ± 0.5149 † | 0.9974 ± 0.0074 | 0.0139 ± 0.0180 | 9.7178 |
| `tr_Mondrian+TabICL [noise_test]` | torchregress | 0.3906 ± 0.0461 | 1.1535 ± 0.0426 | 8.6765 ± 1.6127 † | 1.2666 ± 0.1439 | 0.2154 ± 0.0078 | 2.5809 |
| `tr_Mondrian+TabularMLP [noise_test]` | torchregress | 0.6675 ± 0.0160 | 0.8387 ± 0.0394 | 2.9449 ± 0.3932 † | 0.5598 ± 0.0631 | 0.5771 ± 0.0686 | 31.6715 |
| `tr_Normalized+KumoTabular [noise_test]` | torchregress | 1.0000 ± 0.0000 | 14.6189 ± 3.3109 | 14.6189 ± 3.3109 † | 0.6021 ± 0.0550 | 1.0000 ± 0.0000 | 2.5717 |
| `tr_Normalized+LightGBM [noise_test]` | torchregress | 0.7064 ± 0.0459 | 1.4155 ± 0.0954 | 2.8775 ± 0.2335 † | 0.7554 ± 0.0232 | 0.5982 ± 0.0572 | 2.8532 |
| `tr_Normalized+TabICL [noise_test]` | torchregress | 0.9967 ± 0.0017 | 9.5239 ± 0.6180 | 9.5827 ± 0.6149 † | 1.2666 ± 0.1439 | 0.9737 ± 0.0104 | 2.5809 |
| `tr_Normalized+TabularMLP [noise_test]` | torchregress | 0.6686 ± 0.0283 | 0.8595 ± 0.0689 | 2.8658 ± 0.4227 † | 0.5598 ± 0.0631 | 0.5667 ± 0.0599 | 31.6715 |
| `tr_PITRecal+KumoTabular [noise_test]` | torchregress | 0.9997 ± 0.0005 | 14.3466 ± 3.4295 | 14.3477 ± 3.4314 † | 0.6191 ± 0.0314 | 0.9970 ± 0.0052 | 2.5717 |
| `tr_PITRecal+LightGBM [noise_test]` | torchregress | 0.8653 ± 0.0077 | 1.5887 ± 0.0754 | **2.2747 ± 0.1723** † | 0.7595 ± 0.0238 | 0.6424 ± 0.0295 | 2.8532 |
| `tr_PITRecal+TabICL [noise_test]` | torchregress | 0.9947 ± 0.0021 | 9.2434 ± 0.9391 | 9.2648 ± 0.9329 † | 1.3275 ± 0.2268 | 0.9849 ± 0.0056 | 2.5809 |
| `tr_PITRecal+TabularMLP [noise_test]` | torchregress | 0.6703 ± 0.0249 | 0.8602 ± 0.0647 | 2.8677 ± 0.4274 † | **0.5594 ± 0.0630** | 0.5701 ± 0.0564 | 31.6715 |
| `tr_ResidualDist+KumoTabular [noise_test]` | torchregress | 0.6208 ± 0.0834 | 0.7211 ± 0.0557 | 3.8897 ± 0.6163 † | 0.6019 ± 0.0547 | 0.2376 ± 0.0437 | 2.5717 |
| `tr_ResidualDist+LightGBM [noise_test]` | torchregress | 0.4961 ± 0.0272 | 0.8700 ± 0.0852 | 6.2500 ± 0.2183 | 0.7561 ± 0.0235 | 0.2994 ± 0.0303 | 2.8532 |
| `tr_ResidualDist+TabFM [noise_test]` | torchregress | 0.2081 ± 0.0140 | 0.7328 ± 0.0591 | 10.6795 ± 0.3953 | 1.0003 ± 0.0034 | 0.1059 ± 0.0157 | 9.7178 |
| `tr_ResidualDist+TabICL [noise_test]` | torchregress | 0.3325 ± 0.0404 | 0.7853 ± 0.0258 | 12.1730 ± 2.2545 † | 1.2689 ± 0.1465 | 0.1440 ± 0.0293 | 2.5809 |
| `tr_ResidualDist+TabularMLP [noise_test]` | torchregress | 0.7011 ± 0.0392 | 0.8222 ± 0.0695 | 3.2851 ± 0.4971 † | 0.5601 ± 0.0635 | 0.3157 ± 0.0586 | 31.6715 |
| `tr_Split+KumoTabular [noise_test]` | torchregress | 0.6133 ± 0.0753 | 0.6952 ± 0.0384 | 3.9168 ± 0.5517 † | 0.6021 ± 0.0550 | 0.2238 ± 0.0242 | 2.5717 |
| `tr_Split+LightGBM [noise_test]` | torchregress | 0.5194 ± 0.0361 | 0.8203 ± 0.0672 | 5.7937 ± 0.2158 | 0.7554 ± 0.0232 | 0.2622 ± 0.0125 | 2.8532 |
| `tr_Split+TabFM [noise_test]` | torchregress | 0.2178 ± 0.0148 | 0.7298 ± 0.0530 | 10.4595 ± 0.5563 † | 0.9974 ± 0.0074 | 0.1213 ± 0.0389 | 9.7178 |
| `tr_Split+TabICL [noise_test]` | torchregress | 0.3589 ± 0.0131 | 0.7806 ± 0.0163 | 11.6745 ± 1.8123 † | 1.2666 ± 0.1439 | 0.1384 ± 0.0375 | 2.5809 |
| `tr_Split+TabularMLP [noise_test]` | torchregress | 0.6917 ± 0.0317 | 0.7864 ± 0.0386 | 3.3121 ± 0.4446 † | 0.5598 ± 0.0631 | 0.2907 ± 0.0154 | 31.6715 |

##### fm_uq_scenarios / house_sales@outliers

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [outliers]` | sdm | 0.9775 ± 0.0052 | 1.5644 ± 0.2138 | 1.6730 ± 0.1324 † | 0.2981 ± 0.0470 | 0.9107 ± 0.0443 | 2.5633 |
| `Raw+LightGBM [outliers]` | lightgbm | 0.9353 ± 0.0113 | 2.0394 ± 0.0692 | 2.3743 ± 0.0676 | 0.3843 ± 0.0342 | 0.7907 ± 0.0375 | 2.8247 |
| `Raw+TabICL [outliers]` | tabicl | 0.9558 ± 0.0154 | 1.3922 ± 0.0190 | 1.5451 ± 0.0935 † | 0.3374 ± 0.0553 | 0.8937 ± 0.0263 | 2.5967 |
| `Raw+TabularMLP [outliers]` | torchregress | 0.9997 ± 0.0005 | 6.1547 ± 0.1274 | 6.1617 ± 0.1348 | 0.5019 ± 0.0360 | 0.9969 ± 0.0054 | 13.7094 |
| `tr_CQR+KumoTabular [outliers]` | torchregress | 0.9397 ± 0.0050 | 1.3756 ± 0.2476 | 1.5512 ± 0.1562 † | 0.2981 ± 0.0470 | 0.8374 ± 0.0633 | 2.5633 |
| `tr_CQR+LightGBM [outliers]` | torchregress | 0.9456 ± 0.0029 | 2.0782 ± 0.0383 | 2.3888 ± 0.0497 | 0.3843 ± 0.0342 | 0.8048 ± 0.0522 | 2.8247 |
| `tr_CQR+TabICL [outliers]` | torchregress | 0.9467 ± 0.0147 | 1.3623 ± 0.0412 | 1.5284 ± 0.1103 † | 0.3374 ± 0.0553 | 0.8881 ± 0.0248 | 2.5967 |
| `tr_CQR+TabularMLP [outliers]` | torchregress | 0.9494 ± 0.0027 | 1.7856 ± 0.0787 | 2.1319 ± 0.1258 † | 0.5019 ± 0.0360 | 0.8065 ± 0.0233 | 13.7094 |
| `tr_DCP+KumoTabular [outliers]` | torchregress | 0.9458 ± 0.0022 | 0.8601 ± 0.1261 | 1.0568 ± 0.1432 † | 0.2981 ± 0.0470 | 0.8616 ± 0.0398 | 2.5633 |
| `tr_DCP+LightGBM [outliers]` | torchregress | 0.9408 ± 0.0051 | 2.1237 ± 0.0866 | 2.4368 ± 0.0401 | 0.3843 ± 0.0342 | 0.8048 ± 0.0522 | 2.8247 |
| `tr_DCP+TabICL [outliers]` | torchregress | 0.9431 ± 0.0084 | 1.2580 ± 0.1439 | 1.4376 ± 0.1941 † | 0.3374 ± 0.0553 | 0.8796 ± 0.0158 | 2.5967 |
| `tr_DCP+TabularMLP [outliers]` | torchregress | 0.9531 ± 0.0041 | 1.5466 ± 0.0930 | 2.1623 ± 0.0597 | 0.5019 ± 0.0360 | 0.7180 ± 0.0307 | 13.7094 |
| `tr_Mondrian+KumoTabular [outliers]` | torchregress | 0.9475 ± 0.0051 | 0.9940 ± 0.0981 | 1.2807 ± 0.1646 † | 0.2981 ± 0.0470 | 0.8613 ± 0.0267 | 2.5633 |
| `tr_Mondrian+LightGBM [outliers]` | torchregress | 0.9422 ± 0.0043 | 1.2651 ± 0.1597 | 1.5951 ± 0.1827 † | 0.3843 ± 0.0342 | 0.8438 ± 0.0369 | 2.8247 |
| `tr_Mondrian+TabFM [outliers]` | torchregress | 0.9450 ± 0.0052 | 1.0937 ± 0.0397 | 1.3379 ± 0.1252 † | 0.3108 ± 0.0400 | 0.8698 ± 0.0366 | 9.6952 |
| `tr_Mondrian+TabICL [outliers]` | torchregress | 0.9375 ± 0.0065 | 1.0802 ± 0.0770 | 1.3889 ± 0.1494 † | 0.3374 ± 0.0553 | 0.8577 ± 0.0240 | 2.5967 |
| `tr_Mondrian+TabularMLP [outliers]` | torchregress | 0.9439 ± 0.0076 | 1.6161 ± 0.1006 | 2.0398 ± 0.1808 † | 0.5019 ± 0.0360 | 0.8234 ± 0.0479 | 13.7094 |
| `tr_Normalized+KumoTabular [outliers]` | torchregress | 0.9478 ± 0.0041 | 0.9694 ± 0.1047 | 1.2120 ± 0.0933 † | 0.2981 ± 0.0470 | 0.8269 ± 0.0296 | 2.5633 |
| `tr_Normalized+LightGBM [outliers]` | torchregress | 0.9456 ± 0.0038 | 1.4169 ± 0.0613 | 1.7407 ± 0.0385 | 0.3843 ± 0.0342 | 0.7909 ± 0.0395 | 2.8247 |
| `tr_Normalized+TabICL [outliers]` | torchregress | 0.9475 ± 0.0063 | 1.2848 ± 0.0767 | 1.4765 ± 0.1524 † | 0.3374 ± 0.0553 | 0.8814 ± 0.0164 | 2.5967 |
| `tr_Normalized+TabularMLP [outliers]` | torchregress | 0.9531 ± 0.0041 | 1.5466 ± 0.0930 | 2.1623 ± 0.0597 | 0.5019 ± 0.0360 | 0.7180 ± 0.0307 | 13.7094 |
| `tr_PITRecal+KumoTabular [outliers]` | torchregress | 0.9456 ± 0.0027 | 0.8535 ± 0.0887 | **1.0506 ± 0.1242** † | **0.2975 ± 0.0473** | 0.8726 ± 0.0295 | 2.5633 |
| `tr_PITRecal+LightGBM [outliers]` | torchregress | 0.9400 ± 0.0063 | 2.2229 ± 0.1981 | 2.5018 ± 0.1618 † | 0.3838 ± 0.0333 | 0.8167 ± 0.0289 | 2.8247 |
| `tr_PITRecal+TabICL [outliers]` | torchregress | 0.9472 ± 0.0149 | 1.2807 ± 0.1634 | 1.4507 ± 0.2097 † | 0.3374 ± 0.0555 | 0.8904 ± 0.0075 | 2.5967 |
| `tr_PITRecal+TabularMLP [outliers]` | torchregress | 0.9536 ± 0.0089 | 1.7371 ± 0.2233 | 2.2148 ± 0.1691 † | 0.5002 ± 0.0373 | 0.7683 ± 0.0491 | 13.7094 |
| `tr_ResidualDist+KumoTabular [outliers]` | torchregress | 0.9508 ± 0.0080 | 1.1085 ± 0.1167 | 1.5405 ± 0.1834 † | 0.2978 ± 0.0471 | 0.6951 ± 0.0246 | 2.5633 |
| `tr_ResidualDist+LightGBM [outliers]` | torchregress | 0.9428 ± 0.0079 | 1.4253 ± 0.3128 | 1.9481 ± 0.3294 † | 0.3843 ± 0.0341 | 0.7052 ± 0.0388 | 2.8247 |
| `tr_ResidualDist+TabFM [outliers]` | torchregress | 0.9567 ± 0.0144 | 1.2556 ± 0.1484 | 1.6512 ± 0.1919 † | 0.3061 ± 0.0435 | 0.7475 ± 0.0483 | 9.6952 |
| `tr_ResidualDist+TabICL [outliers]` | torchregress | 0.9506 ± 0.0120 | 1.2530 ± 0.1338 | 1.7486 ± 0.2466 † | 0.3375 ± 0.0554 | 0.6878 ± 0.0576 | 2.5967 |
| `tr_ResidualDist+TabularMLP [outliers]` | torchregress | 0.9522 ± 0.0113 | 1.9001 ± 0.3199 | 2.4985 ± 0.3385 † | 0.4982 ± 0.0335 | 0.7506 ± 0.0455 | 13.7094 |
| `tr_Split+KumoTabular [outliers]` | torchregress | 0.9492 ± 0.0104 | 1.0656 ± 0.1135 | 1.5127 ± 0.1418 † | 0.2981 ± 0.0470 | 0.6694 ± 0.0459 | 2.5633 |
| `tr_Split+LightGBM [outliers]` | torchregress | 0.9506 ± 0.0038 | 1.2808 ± 0.1444 | 1.8813 ± 0.1942 † | 0.3843 ± 0.0342 | 0.6376 ± 0.0530 | 2.8247 |
| `tr_Split+TabFM [outliers]` | torchregress | 0.9525 ± 0.0148 | 1.1532 ± 0.0837 | 1.5774 ± 0.1233 † | 0.3108 ± 0.0400 | 0.7079 ± 0.0433 | 9.6952 |
| `tr_Split+TabICL [outliers]` | torchregress | 0.9472 ± 0.0068 | 1.1577 ± 0.0525 | 1.6802 ± 0.1904 † | 0.3374 ± 0.0553 | 0.6500 ± 0.0464 | 2.5967 |
| `tr_Split+TabularMLP [outliers]` | torchregress | 0.9544 ± 0.0046 | 1.7200 ± 0.1387 | 2.4075 ± 0.1423 † | 0.5019 ± 0.0360 | 0.6899 ± 0.0359 | 13.7094 |

##### fm_uq_scenarios / house_sales@shift

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [shift]` | sdm | 0.8900 ± 0.0113 | 0.7485 ± 0.3553 | **1.0500 ± 0.4527** † | 0.3621 ± 0.2593 | 0.7807 ± 0.0415 | 2.5745 |
| `Raw+LightGBM [shift]` | lightgbm | 0.8053 ± 0.0184 | 0.7452 ± 0.2755 | 2.0192 ± 1.5488 † | 0.4906 ± 0.4265 | 0.5022 ± 0.1495 | 2.8585 |
| `Raw+TabICL [shift]` | tabicl | 0.8794 ± 0.0111 | 0.7335 ± 0.2864 | 1.4333 ± 1.0118 † | 0.4099 ± 0.3206 | 0.7572 ± 0.0502 | 2.5956 |
| `Raw+TabularMLP [shift]` | torchregress | 0.8972 ± 0.0213 | 0.7970 ± 0.2224 | 1.4046 ± 0.8855 † | 0.4347 ± 0.3470 | 0.7481 ± 0.0396 | 30.8016 |
| `tr_CQR+KumoTabular [shift]` | torchregress | 0.8900 ± 0.0169 | 0.7505 ± 0.3676 | 1.0510 ± 0.4524 † | 0.3621 ± 0.2593 | 0.7807 ± 0.0415 | 2.5745 |
| `tr_CQR+LightGBM [shift]` | torchregress | 0.8772 ± 0.0261 | 0.8816 ± 0.2650 | 1.9478 ± 1.5281 † | 0.4906 ± 0.4265 | 0.5474 ± 0.1546 | 2.8585 |
| `tr_CQR+TabICL [shift]` | torchregress | 0.8928 ± 0.0099 | 0.7539 ± 0.2837 | 1.4304 ± 1.0129 † | 0.4099 ± 0.3206 | 0.7645 ± 0.0425 | 2.5956 |
| `tr_CQR+TabularMLP [shift]` | torchregress | 0.8914 ± 0.0121 | 0.7779 ± 0.2428 | 1.4033 ± 0.8854 † | 0.4347 ± 0.3470 | 0.7408 ± 0.0380 | 30.8016 |
| `tr_DCP+KumoTabular [shift]` | torchregress | 0.8892 ± 0.0186 | 0.7562 ± 0.3817 | 1.0543 ± 0.4571 † | 0.3621 ± 0.2593 | 0.7807 ± 0.0415 | 2.5745 |
| `tr_DCP+LightGBM [shift]` | torchregress | 0.8783 ± 0.0068 | 1.0096 ± 0.3116 | 2.0101 ± 1.4547 † | 0.4906 ± 0.4265 | 0.5810 ± 0.1311 | 2.8585 |
| `tr_DCP+TabICL [shift]` | torchregress | 0.8908 ± 0.0142 | 0.7627 ± 0.2927 | 1.4347 ± 1.0188 † | 0.4099 ± 0.3206 | 0.7913 ± 0.0194 | 2.5956 |
| `tr_DCP+TabularMLP [shift]` | torchregress | 0.8906 ± 0.0114 | 0.7774 ± 0.2492 | 1.4010 ± 0.8810 † | 0.4347 ± 0.3470 | 0.7285 ± 0.0572 | 30.8016 |
| `tr_Mondrian+KumoTabular [shift]` | torchregress | 0.8825 ± 0.0156 | 0.6377 ± 0.1413 | 1.4455 ± 0.9449 † | 0.3621 ± 0.2593 | 0.7296 ± 0.0570 | 2.5745 |
| `tr_Mondrian+LightGBM [shift]` | torchregress | 0.8767 ± 0.0082 | 0.8082 ± 0.2784 | 1.9729 ± 1.6187 † | 0.4906 ± 0.4265 | 0.7160 ± 0.0913 | 2.8585 |
| `tr_Mondrian+TabFM [shift]` | torchregress | 0.8761 ± 0.0113 | 0.6558 ± 0.1581 | 1.4952 ± 1.0096 † | 0.3659 ± 0.2610 | 0.7379 ± 0.0704 | 9.7494 |
| `tr_Mondrian+TabICL [shift]` | torchregress | 0.8747 ± 0.0188 | 0.7092 ± 0.2151 | 1.6519 ± 1.1845 † | 0.4099 ± 0.3206 | 0.7430 ± 0.0669 | 2.5956 |
| `tr_Mondrian+TabularMLP [shift]` | torchregress | 0.8747 ± 0.0099 | 0.7669 ± 0.2067 | 1.7125 ± 1.2579 † | 0.4347 ± 0.3470 | 0.7675 ± 0.0302 | 30.8016 |
| `tr_Normalized+KumoTabular [shift]` | torchregress | 0.8878 ± 0.0157 | 0.7489 ± 0.3686 | 1.0712 ± 0.4645 † | 0.3621 ± 0.2593 | 0.7400 ± 0.0478 | 2.5745 |
| `tr_Normalized+LightGBM [shift]` | torchregress | 0.8942 ± 0.0118 | 0.8249 ± 0.3084 | 1.6068 ± 1.1527 † | 0.4906 ± 0.4265 | 0.7167 ± 0.0576 | 2.8585 |
| `tr_Normalized+TabICL [shift]` | torchregress | 0.8908 ± 0.0130 | 0.7537 ± 0.2689 | 1.3915 ± 0.9240 † | 0.4099 ± 0.3206 | 0.7725 ± 0.0281 | 2.5956 |
| `tr_Normalized+TabularMLP [shift]` | torchregress | 0.8906 ± 0.0114 | 0.7774 ± 0.2492 | 1.4010 ± 0.8810 † | 0.4347 ± 0.3470 | 0.7285 ± 0.0572 | 30.8016 |
| `tr_PITRecal+KumoTabular [shift]` | torchregress | 0.8892 ± 0.0175 | 0.7527 ± 0.3738 | 1.0535 ± 0.4547 † | **0.3616 ± 0.2584** | 0.7807 ± 0.0415 | 2.5745 |
| `tr_PITRecal+LightGBM [shift]` | torchregress | 0.8792 ± 0.0082 | 0.9926 ± 0.3369 | 2.0070 ± 1.4864 † | 0.4955 ± 0.4339 | 0.5723 ± 0.1741 | 2.8585 |
| `tr_PITRecal+TabICL [shift]` | torchregress | 0.8925 ± 0.0120 | 0.7644 ± 0.2921 | 1.4345 ± 1.0189 † | 0.4099 ± 0.3209 | 0.7913 ± 0.0194 | 2.5956 |
| `tr_PITRecal+TabularMLP [shift]` | torchregress | 0.8911 ± 0.0163 | 0.7749 ± 0.2405 | 1.4051 ± 0.8874 † | 0.4367 ± 0.3486 | 0.7285 ± 0.0572 | 30.8016 |
| `tr_ResidualDist+KumoTabular [shift]` | torchregress | 0.8911 ± 0.0799 | 0.6879 ± 0.0792 | 1.6489 ± 1.0952 † | 0.3621 ± 0.2592 | 0.4345 ± 0.1170 | 2.5745 |
| `tr_ResidualDist+LightGBM [shift]` | torchregress | 0.8825 ± 0.0960 | 0.8424 ± 0.1083 | 2.2988 ± 1.9274 † | 0.4911 ± 0.4272 | 0.4127 ± 0.1222 | 2.8585 |
| `tr_ResidualDist+TabFM [shift]` | torchregress | 0.8883 ± 0.0834 | 0.7141 ± 0.1042 | 1.7255 ± 1.1987 † | 0.3661 ± 0.2613 | 0.4559 ± 0.1120 | 9.7494 |
| `tr_ResidualDist+TabICL [shift]` | torchregress | 0.8892 ± 0.0919 | 0.7701 ± 0.1398 | 1.9424 ± 1.4620 † | 0.4099 ± 0.3206 | 0.4621 ± 0.1642 | 2.5956 |
| `tr_ResidualDist+TabularMLP [shift]` | torchregress | 0.8922 ± 0.0858 | 0.8106 ± 0.1150 | 1.9956 ± 1.4880 † | 0.4350 ± 0.3471 | 0.5608 ± 0.0453 | 30.8016 |
| `tr_Split+KumoTabular [shift]` | torchregress | 0.8886 ± 0.0806 | 0.6753 ± 0.0780 | 1.6569 ± 1.1117 † | 0.3621 ± 0.2593 | 0.4011 ± 0.1413 | 2.5745 |
| `tr_Split+LightGBM [shift]` | torchregress | 0.8825 ± 0.1004 | 0.8102 ± 0.0962 | 2.3291 ± 1.9873 † | 0.4906 ± 0.4265 | 0.3626 ± 0.1026 | 2.8585 |
| `tr_Split+TabFM [shift]` | torchregress | 0.8867 ± 0.0852 | 0.6988 ± 0.1032 | 1.7264 ± 1.2103 † | 0.3659 ± 0.2610 | 0.4551 ± 0.1109 | 9.7494 |
| `tr_Split+TabICL [shift]` | torchregress | 0.8842 ± 0.0889 | 0.7416 ± 0.0946 | 1.9310 ± 1.4675 † | 0.4099 ± 0.3206 | 0.4103 ± 0.2123 | 2.5956 |
| `tr_Split+TabularMLP [shift]` | torchregress | 0.8903 ± 0.0865 | 0.8110 ± 0.1210 | 2.0054 ± 1.5035 † | 0.4347 ± 0.3470 | 0.5460 ± 0.0648 | 30.8016 |
| `tr_WeightedCQR+KumoTabular [shift]` | torchregress | 0.8906 ± 0.0126 | 0.7498 ± 0.3328 | 1.0509 ± 0.4561 † | 0.3621 ± 0.2593 | 0.7661 ± 0.0219 | 2.5745 |
| `tr_WeightedCQR+LightGBM [shift]` | torchregress | 0.8842 ± 0.0068 | 0.9107 ± 0.3269 | 1.8937 ± 1.4345 † | 0.4906 ± 0.4265 | 0.5564 ± 0.1680 | 2.8585 |
| `tr_WeightedCQR+TabICL [shift]` | torchregress | 0.8981 ± 0.0169 | 0.7714 ± 0.2713 | 1.4203 ± 0.9941 † | 0.4099 ± 0.3206 | 0.7790 ± 0.0365 | 2.5956 |
| `tr_WeightedCQR+TabularMLP [shift]` | torchregress | 0.8836 ± 0.0262 | 0.7653 ± 0.2145 | 1.4023 ± 0.8841 † | 0.4347 ± 0.3470 | 0.7239 ± 0.0182 | 30.8016 |
| `tr_WeightedSplit+KumoTabular [shift]` | torchregress | 0.8894 ± 0.0284 | 0.6773 ± 0.1605 | 1.5578 ± 0.9788 † | 0.3621 ± 0.2593 | 0.3787 ± 0.2631 | 2.5745 |
| `tr_WeightedSplit+LightGBM [shift]` | torchregress | 0.9025 ± 0.0088 | 0.9053 ± 0.3844 | 2.0964 ± 1.6574 † | 0.4906 ± 0.4265 | 0.4136 ± 0.2699 | 2.8585 |
| `tr_WeightedSplit+TabFM [shift]` | torchregress | 0.8839 ± 0.0218 | 0.7102 ± 0.1966 | 1.6135 ± 1.0586 † | 0.3659 ± 0.2610 | 0.4190 ± 0.2584 | 9.7494 |
| `tr_WeightedSplit+TabICL [shift]` | torchregress | 0.8867 ± 0.0254 | 0.7770 ± 0.2467 | 1.7873 ± 1.2679 † | 0.4099 ± 0.3206 | 0.3932 ± 0.2752 | 2.5956 |
| `tr_WeightedSplit+TabularMLP [shift]` | torchregress | 0.8903 ± 0.0279 | 0.8170 ± 0.2263 | 1.8672 ± 1.3295 † | 0.4347 ± 0.3470 | 0.5108 ± 0.1482 | 30.8016 |

##### fm_uq_scenarios / insurance_claims

_No successful rows._

##### fm_uq_scenarios / insurance_claims@clean

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [clean]` | sdm | 0.9494 ± 0.0089 | 1.6870 ± 0.4410 | 4.2407 ± 0.4956 † | 1.0616 ± 0.0911 | 0.0131 ± 0.0226 | 3.9166 |
| `Raw+LightGBM [clean]` | lightgbm | 0.3192 ± 0.5528 | 1.5805 ± 0.2552 | 4.2736 ± 0.5068 † | 1.0780 ± 0.0762 | 0.0000 ± 0.0000 | 1.4354 |
| `Raw+TabICL [clean]` | tabicl | 0.9453 ± 0.0092 | 1.0126 ± 0.4167 | 4.3781 ± 0.5178 † | 1.0570 ± 0.0986 | 0.0314 ± 0.0415 | 4.5000 |
| `Raw+TabularMLP [clean]` | torchregress | 0.9489 ± 0.0080 | 3.2366 ± 0.2061 | 6.0182 ± 0.3791 † | **1.0342 ± 0.0886** | 0.0000 ± 0.0000 | 14.0010 |
| `tr_CQR+KumoTabular [clean]` | torchregress | 0.8858 ± 0.0022 | 1.6842 ± 0.4414 | **4.2395 ± 0.4956** † | 1.0616 ± 0.0911 | 0.0131 ± 0.0226 | 3.9166 |
| `tr_CQR+LightGBM [clean]` | torchregress | 0.9489 ± 0.0080 | 1.5805 ± 0.2552 | 4.2736 ± 0.5068 † | 1.0780 ± 0.0762 | 0.0000 ± 0.0000 | 1.4354 |
| `tr_CQR+TabICL [clean]` | torchregress | 0.8936 ± 0.0081 | 1.0110 ± 0.4164 | 4.3776 ± 0.5180 † | 1.0570 ± 0.0986 | 0.0314 ± 0.0415 | 4.5000 |
| `tr_CQR+TabularMLP [clean]` | torchregress | 0.8878 ± 0.0106 | 0.7180 ± 0.2284 | 4.8196 ± 0.5525 † | **1.0342 ± 0.0886** | 0.0000 ± 0.0000 | 14.0010 |
| `tr_DCP+KumoTabular [clean]` | torchregress | 0.8886 ± 0.0184 | 0.0096 ± 0.0146 | 4.6528 ± 0.7657 † | 1.0616 ± 0.0911 | 0.0065 ± 0.0113 | 3.9166 |
| `tr_DCP+LightGBM [clean]` | torchregress | 0.3444 ± 0.4892 | 7.5397 ± 9.0932 | 8.8903 ± 7.6687 † | 1.0780 ± 0.0762 | 0.0062 ± 0.0085 | 1.4354 |
| `tr_DCP+TabICL [clean]` | torchregress | 0.9019 ± 0.0106 | 0.0803 ± 0.0068 | 4.6325 ± 0.7355 † | 1.0570 ± 0.0986 | 0.0065 ± 0.0113 | 4.5000 |
| `tr_DCP+TabularMLP [clean]` | torchregress | 0.8928 ± 0.0047 | 0.4739 ± 0.0687 | 4.6786 ± 0.6550 † | **1.0342 ± 0.0886** | 0.0000 ± 0.0000 | 14.0010 |
| `tr_Mondrian+KumoTabular [clean]` | torchregress | 0.8892 ± 0.0102 | 0.0012 ± 0.0004 | 4.6920 ± 0.6986 † | 1.0616 ± 0.0911 | 0.0000 ± 0.0000 | 3.9166 |
| `tr_Mondrian+LightGBM [clean]` | torchregress | 0.9492 ± 0.0085 | 0.0886 ± 0.0769 | 4.7160 ± 0.7368 † | 1.0780 ± 0.0762 | 0.0065 ± 0.0113 | 1.4354 |
| `tr_Mondrian+TabFM [clean]` | torchregress | 0.8953 ± 0.0063 | 0.5823 ± 0.0675 | 4.7288 ± 0.5423 † | 1.0351 ± 0.0800 | 0.0000 ± 0.0000 | 11.0489 |
| `tr_Mondrian+TabICL [clean]` | torchregress | 0.9033 ± 0.0145 | 0.0038 ± 0.0010 | 4.6671 ± 0.7411 † | 1.0570 ± 0.0986 | 0.0000 ± 0.0000 | 4.5000 |
| `tr_Mondrian+TabularMLP [clean]` | torchregress | 0.8975 ± 0.0177 | 0.4701 ± 0.0714 | 4.6729 ± 0.6561 † | **1.0342 ± 0.0886** | 0.0000 ± 0.0000 | 14.0010 |
| `tr_Normalized+KumoTabular [clean]` | torchregress | 0.8886 ± 0.0077 | 0.0024 ± 0.0005 | 4.6924 ± 0.6985 † | 1.0616 ± 0.0911 | 0.0000 ± 0.0000 | 3.9166 |
| `tr_Normalized+LightGBM [clean]` | torchregress | 0.9022 ± 0.0226 | 6.676e-08 ± 4.172e-08 | 4.9059 ± 0.4474 † | 1.0780 ± 0.0762 | 0.0000 ± 0.0000 | 1.4354 |
| `tr_Normalized+TabICL [clean]` | torchregress | 0.8919 ± 0.0076 | 0.0187 ± 0.0107 | 4.6701 ± 0.7431 † | 1.0570 ± 0.0986 | 0.0000 ± 0.0000 | 4.5000 |
| `tr_Normalized+TabularMLP [clean]` | torchregress | 0.8928 ± 0.0047 | 0.4739 ± 0.0687 | 4.6786 ± 0.6550 † | **1.0342 ± 0.0886** | 0.0000 ± 0.0000 | 14.0010 |
| `tr_PITRecal+KumoTabular [clean]` | torchregress | 0.8961 ± 0.0042 | 1.1627 ± 0.9868 | 4.4298 ± 0.5631 † | 1.0601 ± 0.0937 | 0.0065 ± 0.0113 | 3.9166 |
| `tr_PITRecal+LightGBM [clean]` | torchregress | 0.2856 ± 0.4802 | 1.9605 ± 0.4136 | 4.8156 ± 0.0371 | 1.1370 ± 0.0390 | 0.0048 ± 0.0082 | 1.4354 |
| `tr_PITRecal+TabICL [clean]` | torchregress | 0.9022 ± 0.0092 | 0.5568 ± 0.5287 | 4.4164 ± 0.5415 † | 1.0564 ± 0.0998 | 0.0065 ± 0.0113 | 4.5000 |
| `tr_PITRecal+TabularMLP [clean]` | torchregress | 0.8933 ± 0.0058 | 1.4984 ± 1.2354 | 4.5215 ± 0.5729 † | 1.0589 ± 0.0954 | 0.0000 ± 0.0000 | 14.0010 |
| `tr_ResidualDist+KumoTabular [clean]` | torchregress | 0.8950 ± 0.0030 | 1.5698 ± 1.3597 | 4.5247 ± 0.5472 † | 1.0616 ± 0.0910 | 0.0000 ± 0.0000 | 3.9166 |
| `tr_ResidualDist+LightGBM [clean]` | torchregress | 0.6231 ± 0.5397 | 1.5685 ± 1.3601 | 4.7387 ± 0.3191 † | 1.0780 ± 0.0762 | 0.0000 ± 0.0000 | 1.4354 |
| `tr_ResidualDist+TabFM [clean]` | torchregress | 0.9014 ± 0.0017 | 1.7485 ± 1.2219 | 4.8224 ± 0.4269 † | 1.0543 ± 0.0871 | 0.0000 ± 0.0000 | 11.0489 |
| `tr_ResidualDist+TabICL [clean]` | torchregress | 0.9008 ± 0.0014 | 1.5714 ± 1.3585 | 4.5016 ± 0.5909 † | 1.0571 ± 0.0986 | 0.0000 ± 0.0000 | 4.5000 |
| `tr_ResidualDist+TabularMLP [clean]` | torchregress | 0.9028 ± 0.0043 | 1.6516 ± 1.2784 | 4.5613 ± 0.5366 † | 1.0582 ± 0.0956 | 0.0000 ± 0.0000 | 14.0010 |
| `tr_Split+KumoTabular [clean]` | torchregress | 0.8908 ± 0.0152 | 0.0014 ± 0.0004 | 4.6925 ± 0.6987 † | 1.0616 ± 0.0911 | 0.0000 ± 0.0000 | 3.9166 |
| `tr_Split+LightGBM [clean]` | torchregress | 0.9386 ± 0.0112 | 8.821e-09 ± 4.277e-09 | 4.9059 ± 0.4474 † | 1.0780 ± 0.0762 | 0.0000 ± 0.0000 | 1.4354 |
| `tr_Split+TabFM [clean]` | torchregress | 0.9000 ± 0.0025 | 0.7265 ± 0.0511 | 4.9722 ± 0.4846 † | 1.0351 ± 0.0800 | 0.0000 ± 0.0000 | 11.0489 |
| `tr_Split+TabICL [clean]` | torchregress | 0.9039 ± 0.0046 | 0.0045 ± 0.0006 | 4.6687 ± 0.7414 † | 1.0570 ± 0.0986 | 0.0000 ± 0.0000 | 4.5000 |
| `tr_Split+TabularMLP [clean]` | torchregress | 0.9011 ± 0.0019 | 0.5340 ± 0.1106 | 4.7216 ± 0.6156 † | **1.0342 ± 0.0886** | 0.0000 ± 0.0000 | 14.0010 |

##### fm_uq_scenarios / insurance_claims@extrapolate

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [extrapolate]` | sdm | 0.9536 ± 0.0049 | 2.0428 ± 0.7328 | 4.0753 ± 0.1041 | 0.9583 ± 0.0196 | 0.0192 ± 0.0333 | 3.4438 |
| `Raw+LightGBM [extrapolate]` | lightgbm | 0.3175 ± 0.5492 | 1.7350 ± 0.7761 | 4.0662 ± 0.3606 † | 0.9582 ± 0.0197 | 0.0000 ± 0.0000 | 1.4623 |
| `Raw+TabICL [extrapolate]` | tabicl | 0.9469 ± 0.0121 | 1.0408 ± 0.3743 | 4.0083 ± 0.1427 | 0.9580 ± 0.0197 | 0.0303 ± 0.0290 | 4.2985 |
| `Raw+TabularMLP [extrapolate]` | torchregress | 0.9528 ± 0.0035 | 3.3121 ± 0.1322 | 5.6480 ± 0.0765 | 0.9375 ± 0.0212 | 0.0000 ± 0.0000 | 13.9207 |
| `tr_CQR+KumoTabular [extrapolate]` | torchregress | 0.8928 ± 0.0488 | 2.0394 ± 0.7328 | 4.0737 ± 0.1039 | 0.9583 ± 0.0196 | 0.0192 ± 0.0333 | 3.4438 |
| `tr_CQR+LightGBM [extrapolate]` | torchregress | 0.9531 ± 0.0039 | 1.7350 ± 0.7761 | 4.0662 ± 0.3606 † | 0.9582 ± 0.0197 | 0.0064 ± 0.0111 | 1.4623 |
| `tr_CQR+TabICL [extrapolate]` | torchregress | 0.9119 ± 0.0278 | 1.0391 ± 0.3741 | **4.0078 ± 0.1427** | 0.9580 ± 0.0197 | 0.0303 ± 0.0290 | 4.2985 |
| `tr_CQR+TabularMLP [extrapolate]` | torchregress | 0.8444 ± 0.1351 | 0.7116 ± 0.1278 | 4.3596 ± 0.2194 † | 0.9375 ± 0.0212 | 0.0000 ± 0.0000 | 13.9207 |
| `tr_DCP+KumoTabular [extrapolate]` | torchregress | 0.9328 ± 0.0188 | 0.0033 ± 0.0031 | 4.1128 ± 0.2106 † | 0.9583 ± 0.0196 | 0.0000 ± 0.0000 | 3.4438 |
| `tr_DCP+LightGBM [extrapolate]` | torchregress | 0.3525 ± 0.5020 | 12.2626 ± 10.5311 | 13.6082 ± 8.2049 † | 0.9582 ± 0.0197 | 0.0114 ± 0.0198 | 1.4623 |
| `tr_DCP+TabICL [extrapolate]` | torchregress | 0.9008 ± 0.0246 | 0.0762 ± 0.0316 | 4.0585 ± 0.2645 † | 0.9580 ± 0.0197 | 0.0064 ± 0.0111 | 4.2985 |
| `tr_DCP+TabularMLP [extrapolate]` | torchregress | 0.9325 ± 0.0109 | 0.5345 ± 0.0176 | 4.1897 ± 0.2005 | 0.9375 ± 0.0212 | 0.0000 ± 0.0000 | 13.9207 |
| `tr_Mondrian+KumoTabular [extrapolate]` | torchregress | 0.8964 ± 0.0268 | 0.0015 ± 0.0010 | 4.1201 ± 0.1973 | 0.9583 ± 0.0196 | 0.0000 ± 0.0000 | 3.4438 |
| `tr_Mondrian+LightGBM [extrapolate]` | torchregress | 0.9547 ± 0.0034 | 0.0715 ± 0.0786 | 4.0347 ± 0.2051 † | 0.9582 ± 0.0197 | 0.0412 ± 0.0095 | 1.4623 |
| `tr_Mondrian+TabFM [extrapolate]` | torchregress | 0.8950 ± 0.0055 | 0.6958 ± 0.3349 | 4.2202 ± 0.0421 | **0.9344 ± 0.0174** | 0.0000 ± 0.0000 | 11.0125 |
| `tr_Mondrian+TabICL [extrapolate]` | torchregress | 0.9014 ± 0.0087 | 0.0034 ± 0.0001 | 4.1201 ± 0.1988 | 0.9580 ± 0.0197 | 0.0000 ± 0.0000 | 4.2985 |
| `tr_Mondrian+TabularMLP [extrapolate]` | torchregress | 0.9094 ± 0.0059 | 0.4906 ± 0.0200 | 4.1665 ± 0.1843 | 0.9375 ± 0.0212 | 0.0000 ± 0.0000 | 13.9207 |
| `tr_Normalized+KumoTabular [extrapolate]` | torchregress | 0.9053 ± 0.0744 | 0.0164 ± 0.0233 | 4.1282 ± 0.1853 | 0.9583 ± 0.0196 | 0.0000 ± 0.0000 | 3.4438 |
| `tr_Normalized+LightGBM [extrapolate]` | torchregress | 0.9417 ± 0.0052 | 6.244e-08 ± 7.080e-08 | 4.1677 ± 0.1140 | 0.9582 ± 0.0197 | 0.0000 ± 0.0000 | 1.4623 |
| `tr_Normalized+TabICL [extrapolate]` | torchregress | 0.8869 ± 0.0725 | 0.0136 ± 0.0055 | 4.1250 ± 0.1988 | 0.9580 ± 0.0197 | 0.0000 ± 0.0000 | 4.2985 |
| `tr_Normalized+TabularMLP [extrapolate]` | torchregress | 0.9325 ± 0.0109 | 0.5345 ± 0.0176 | 4.1897 ± 0.2005 | 0.9375 ± 0.0212 | 0.0000 ± 0.0000 | 13.9207 |
| `tr_PITRecal+KumoTabular [extrapolate]` | torchregress | 0.9111 ± 0.0348 | 0.4356 ± 0.7247 | 4.0390 ± 0.2438 † | 0.9582 ± 0.0197 | 0.0000 ± 0.0000 | 3.4438 |
| `tr_PITRecal+LightGBM [extrapolate]` | torchregress | 0.3164 ± 0.5201 | 1.0597 ± 0.9949 | 4.2962 ± 0.4099 † | 1.0124 ± 0.1126 | 0.0000 ± 0.0000 | 1.4623 |
| `tr_PITRecal+TabICL [extrapolate]` | torchregress | 0.9028 ± 0.0270 | 0.3365 ± 0.5069 | 4.0552 ± 0.2361 † | 0.9581 ± 0.0197 | 0.0000 ± 0.0000 | 4.2985 |
| `tr_PITRecal+TabularMLP [extrapolate]` | torchregress | 0.9275 ± 0.0075 | 0.7476 ± 1.1061 | 4.1551 ± 0.2230 † | 0.9619 ± 0.0248 | 0.0000 ± 0.0000 | 13.9207 |
| `tr_ResidualDist+KumoTabular [extrapolate]` | torchregress | 0.8875 ± 0.0038 | 0.7625 ± 1.3179 | 4.1196 ± 0.1981 | 0.9582 ± 0.0197 | 0.0000 ± 0.0000 | 3.4438 |
| `tr_ResidualDist+LightGBM [extrapolate]` | torchregress | 0.6319 ± 0.5473 | 0.7612 ± 1.3184 | 4.1677 ± 0.1140 | 0.9582 ± 0.0197 | 0.0000 ± 0.0000 | 1.4623 |
| `tr_ResidualDist+TabFM [extrapolate]` | torchregress | 0.9258 ± 0.0036 | 1.0577 ± 1.1943 | 4.3695 ± 0.1767 | 0.9515 ± 0.0264 | 0.0000 ± 0.0000 | 11.0125 |
| `tr_ResidualDist+TabICL [extrapolate]` | torchregress | 0.9192 ± 0.0142 | 0.7650 ± 1.3167 | 4.1216 ± 0.1988 | 0.9581 ± 0.0198 | 0.0000 ± 0.0000 | 4.2985 |
| `tr_ResidualDist+TabularMLP [extrapolate]` | torchregress | 0.9275 ± 0.0051 | 0.9107 ± 1.2803 | 4.2180 ± 0.2664 † | 0.9641 ± 0.0295 | 0.0000 ± 0.0000 | 13.9207 |
| `tr_Split+KumoTabular [extrapolate]` | torchregress | 0.8353 ± 0.1670 | 0.0020 ± 0.0005 | 4.1211 ± 0.1965 | 0.9583 ± 0.0196 | 0.0000 ± 0.0000 | 3.4438 |
| `tr_Split+LightGBM [extrapolate]` | torchregress | 0.9492 ± 0.0043 | 8.821e-09 ± 4.277e-09 | 4.1677 ± 0.1140 | 0.9582 ± 0.0197 | 0.0000 ± 0.0000 | 1.4623 |
| `tr_Split+TabFM [extrapolate]` | torchregress | 0.9222 ± 0.0084 | 0.8241 ± 0.0025 | 4.3892 ± 0.1494 | **0.9344 ± 0.0174** | 0.0000 ± 0.0000 | 11.0125 |
| `tr_Split+TabICL [extrapolate]` | torchregress | 0.9169 ± 0.0140 | 0.0054 ± 0.0004 | 4.1216 ± 0.1988 | 0.9580 ± 0.0197 | 0.0000 ± 0.0000 | 4.2985 |
| `tr_Split+TabularMLP [extrapolate]` | torchregress | 0.9281 ± 0.0118 | 0.6252 ± 0.0911 | 4.2416 ± 0.2334 † | 0.9375 ± 0.0212 | 0.0000 ± 0.0000 | 13.9207 |

##### fm_uq_scenarios / insurance_claims@heavy_tail

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [heavy_tail]` | sdm | 0.9628 ± 0.0302 | 0.0062 ± 0.0029 | 1.0941 ± 0.9465 † | 0.8312 ± 0.7203 | 0.3238 ± 0.5347 | 3.4169 |
| `Raw+LightGBM [heavy_tail]` | lightgbm | 0.3183 ± 0.5485 | 0.0593 ± 0.0702 | 1.1439 ± 0.9756 † | 0.8312 ± 0.7203 | 0.0065 ± 0.0113 | 1.3459 |
| `Raw+TabICL [heavy_tail]` | tabicl | 0.9625 ± 0.0291 | 0.0127 ± 0.0111 | 1.0878 ± 0.9432 † | 0.8312 ± 0.7203 | 0.3291 ± 0.5302 | 4.2940 |
| `Raw+TabularMLP [heavy_tail]` | torchregress | 0.9964 ± 0.0010 | 1387.7 ± 1719.5 | 1388.7 ± 1720.0 † | **0.8306 ± 0.7199** | 0.9301 ± 0.0104 | 24.8138 |
| `tr_CQR+KumoTabular [heavy_tail]` | torchregress | 0.8878 ± 0.0034 | 0.0041 ± 0.0025 | 1.0930 ± 0.9467 † | 0.8312 ± 0.7203 | 0.1931 ± 0.3083 | 3.4169 |
| `tr_CQR+LightGBM [heavy_tail]` | torchregress | 0.9472 ± 0.0010 | 0.0593 ± 0.0702 | 1.1439 ± 0.9756 † | 0.8312 ± 0.7203 | 0.0404 ± 0.0393 | 1.3459 |
| `tr_CQR+TabICL [heavy_tail]` | torchregress | 0.8861 ± 0.0038 | 0.0105 ± 0.0098 | **1.0869 ± 0.9429** † | 0.8312 ± 0.7203 | 0.1996 ± 0.3196 | 4.2940 |
| `tr_CQR+TabularMLP [heavy_tail]` | torchregress | 0.8981 ± 0.0131 | 1386.4 ± 1718.4 | 1387.4 ± 1718.9 † | **0.8306 ± 0.7199** | 0.5634 ± 0.4262 | 24.8138 |
| `tr_DCP+KumoTabular [heavy_tail]` | torchregress | 0.8889 ± 0.0164 | 9.756e-04 ± 1.146e-04 | 1.0951 ± 0.9488 † | 0.8312 ± 0.7203 | 0.2092 ± 0.3623 | 3.4169 |
| `tr_DCP+LightGBM [heavy_tail]` | torchregress | 0.3372 ± 0.4881 | 19.6404 ± 32.8902 | 20.2106 ± 32.4247 † | 0.8312 ± 0.7203 | 0.0080 ± 0.0101 | 1.3459 |
| `tr_DCP+TabICL [heavy_tail]` | torchregress | 0.8972 ± 0.0124 | 0.0014 ± 0.0006 | 1.0951 ± 0.9488 † | 0.8312 ± 0.7203 | 0.1699 ± 0.2943 | 4.2940 |
| `tr_DCP+TabularMLP [heavy_tail]` | torchregress | 0.9003 ± 0.0134 | 36.6482 ± 56.5343 | 37.7263 ± 56.9537 † | **0.8306 ± 0.7199** | 0.6171 ± 0.4796 | 24.8138 |
| `tr_Mondrian+KumoTabular [heavy_tail]` | torchregress | 0.9022 ± 0.0077 | 7.359e-04 ± 1.159e-04 | 1.0949 ± 0.9490 † | 0.8312 ± 0.7203 | 0.1503 ± 0.2604 | 3.4169 |
| `tr_Mondrian+LightGBM [heavy_tail]` | torchregress | 0.9486 ± 0.0087 | 1.779e-06 ± 3.077e-06 | 1.0945 ± 0.9492 † | 0.8312 ± 0.7203 | 0.0065 ± 0.0113 | 1.3459 |
| `tr_Mondrian+TabFM [heavy_tail]` | torchregress | 0.8936 ± 0.0099 | 0.0933 ± 0.0414 | 1.1973 ± 0.9426 † | 0.8413 ± 0.6994 | 0.7146 ± 0.1535 | 10.9671 |
| `tr_Mondrian+TabICL [heavy_tail]` | torchregress | 0.8986 ± 0.0104 | 0.0013 ± 0.0009 | 1.0955 ± 0.9494 † | 0.8312 ± 0.7203 | 0.0784 ± 0.1358 | 4.2940 |
| `tr_Mondrian+TabularMLP [heavy_tail]` | torchregress | 0.8903 ± 0.0150 | 0.0306 ± 0.0368 | 1.1217 ± 0.9705 † | **0.8306 ± 0.7199** | 0.5617 ± 0.3220 | 24.8138 |
| `tr_Normalized+KumoTabular [heavy_tail]` | torchregress | 0.8850 ± 0.0146 | 0.0015 ± 0.0006 | 1.0952 ± 0.9489 † | 0.8312 ± 0.7203 | 0.2288 ± 0.3962 | 3.4169 |
| `tr_Normalized+LightGBM [heavy_tail]` | torchregress | 0.9067 ± 0.0188 | 2.828e-07 ± 1.877e-07 | 1.0946 ± 0.9492 † | 0.8312 ± 0.7203 | 0.0000 ± 0.0000 | 1.3459 |
| `tr_Normalized+TabICL [heavy_tail]` | torchregress | 0.8956 ± 0.0086 | 0.0021 ± 0.0016 | 1.0951 ± 0.9488 † | 0.8312 ± 0.7203 | 0.1699 ± 0.2943 | 4.2940 |
| `tr_Normalized+TabularMLP [heavy_tail]` | torchregress | 0.9003 ± 0.0134 | 36.6482 ± 56.5343 | 37.7263 ± 56.9537 † | **0.8306 ± 0.7199** | 0.6171 ± 0.4796 | 24.8138 |
| `tr_PITRecal+KumoTabular [heavy_tail]` | torchregress | 0.8792 ± 0.0110 | 0.0074 ± 0.0069 | 1.0927 ± 0.9469 † | 0.8312 ± 0.7203 | 0.2453 ± 0.3482 | 3.4169 |
| `tr_PITRecal+LightGBM [heavy_tail]` | torchregress | 0.2903 ± 0.4970 | 0.0653 ± 0.0520 | 1.1513 ± 0.9928 † | 0.8458 ± 0.6953 | 0.0077 ± 0.0104 | 1.3459 |
| `tr_PITRecal+TabICL [heavy_tail]` | torchregress | 0.8947 ± 0.0104 | 0.0119 ± 0.0133 | 1.0880 ± 0.9439 † | 0.8311 ± 0.7202 | 0.2496 ± 0.3958 | 4.2940 |
| `tr_PITRecal+TabularMLP [heavy_tail]` | torchregress | 0.9069 ± 0.0135 | 21.6494 ± 31.1558 | 22.7319 ± 31.6274 † | 277.0 ± 416.3 | 0.3315 ± 0.2626 | 24.8138 |
| `tr_ResidualDist+KumoTabular [heavy_tail]` | torchregress | 0.8847 ± 0.0065 | 0.0122 ± 0.0100 | 1.0934 ± 0.9476 † | 0.8312 ± 0.7203 | 0.2288 ± 0.3962 | 3.4169 |
| `tr_ResidualDist+LightGBM [heavy_tail]` | torchregress | 0.9447 ± 0.0027 | 0.0116 ± 0.0102 | 1.0932 ± 0.9482 † | 0.8312 ± 0.7203 | 0.0000 ± 0.0000 | 1.3459 |
| `tr_ResidualDist+TabFM [heavy_tail]` | torchregress | 0.9025 ± 0.0115 | 0.1021 ± 0.0530 | 1.2372 ± 0.9666 † | 0.8400 ± 0.7033 | 0.5701 ± 0.2850 | 10.9671 |
| `tr_ResidualDist+TabICL [heavy_tail]` | torchregress | 0.8906 ± 0.0111 | 0.0123 ± 0.0103 | 1.0941 ± 0.9482 † | 0.8312 ± 0.7203 | 0.2353 ± 0.4075 | 4.2940 |
| `tr_ResidualDist+TabularMLP [heavy_tail]` | torchregress | 0.9053 ± 0.0132 | 0.0286 ± 0.0364 | 1.1292 ± 0.9766 † | 0.8311 ± 0.7203 | 0.3109 ± 0.2866 | 24.8138 |
| `tr_Split+KumoTabular [heavy_tail]` | torchregress | 0.8931 ± 0.0106 | 0.0011 ± 0.0002 | 1.0952 ± 0.9489 † | 0.8312 ± 0.7203 | 0.2353 ± 0.4075 | 3.4169 |
| `tr_Split+LightGBM [heavy_tail]` | torchregress | 0.9447 ± 0.0027 | 1.720e-09 ± 1.389e-09 | 1.0946 ± 0.9492 † | 0.8312 ± 0.7203 | 0.0000 ± 0.0000 | 1.3459 |
| `tr_Split+TabFM [heavy_tail]` | torchregress | 0.9053 ± 0.0055 | 0.1572 ± 0.0806 | 1.3113 ± 0.9928 † | 0.8413 ± 0.6994 | 0.8701 ± 0.0392 | 10.9671 |
| `tr_Split+TabICL [heavy_tail]` | torchregress | 0.8889 ± 0.0122 | 0.0015 ± 0.0006 | 1.0958 ± 0.9494 † | 0.8312 ± 0.7203 | 0.1830 ± 0.3170 | 4.2940 |
| `tr_Split+TabularMLP [heavy_tail]` | torchregress | 0.9008 ± 0.0227 | 0.0445 ± 0.0613 | 1.1420 ± 0.9852 † | **0.8306 ± 0.7199** | 0.6010 ± 0.5072 | 24.8138 |

##### fm_uq_scenarios / insurance_claims@imbalance

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [imbalance]` | sdm | 0.9489 ± 0.0080 | 9.366e-04 ± 2.995e-04 | 4.6675 ± 0.7397 † | 1.0590 ± 0.0955 | 0.0000 ± 0.0000 | 3.3071 |
| `Raw+LightGBM [imbalance]` | lightgbm | 0.0056 ± 0.0096 | 0.0474 ± 0.0150 | 4.6479 ± 0.7654 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 1.2556 |
| `Raw+TabICL [imbalance]` | tabicl | 0.9467 ± 0.0093 | 0.0341 ± 0.0230 | 4.6101 ± 0.7476 † | 1.0591 ± 0.0954 | 0.0000 ± 0.0000 | 4.1779 |
| `Raw+TabularMLP [imbalance]` | torchregress | 0.9489 ± 0.0080 | 1.0117 ± 0.2212 | 5.1393 ± 0.5304 † | 1.0548 ± 0.0955 | 0.0000 ± 0.0000 | 12.9306 |
| `tr_CQR+KumoTabular [imbalance]` | torchregress | 0.8550 ± 0.0217 | 4.038e-04 ± 1.294e-04 | 4.6673 ± 0.7398 † | 1.0590 ± 0.0955 | 0.0000 ± 0.0000 | 3.3071 |
| `tr_CQR+LightGBM [imbalance]` | torchregress | 0.9489 ± 0.0080 | 0.0474 ± 0.0150 | 4.6479 ± 0.7654 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 1.2556 |
| `tr_CQR+TabICL [imbalance]` | torchregress | 0.8806 ± 0.0054 | 0.0337 ± 0.0230 | **4.6100 ± 0.7477** † | 1.0591 ± 0.0954 | 0.0000 ± 0.0000 | 4.1779 |
| `tr_CQR+TabularMLP [imbalance]` | torchregress | 0.8581 ± 0.0246 | 0.1177 ± 0.0805 | 4.6940 ± 0.7512 † | 1.0548 ± 0.0955 | 0.0000 ± 0.0000 | 12.9306 |
| `tr_DCP+KumoTabular [imbalance]` | torchregress | 0.8681 ± 0.0125 | 3.200e-04 ± 1.353e-04 | 4.6674 ± 0.7399 † | 1.0590 ± 0.0955 | 0.0000 ± 0.0000 | 3.3071 |
| `tr_DCP+LightGBM [imbalance]` | torchregress | 0.3314 ± 0.4780 | 9.3761 ± 7.9918 | 10.6137 ± 5.8519 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 1.2556 |
| `tr_DCP+TabICL [imbalance]` | torchregress | 0.8683 ± 0.0150 | 0.0074 ± 0.0065 | 4.6522 ± 0.7578 † | 1.0591 ± 0.0954 | 0.0000 ± 0.0000 | 4.1779 |
| `tr_DCP+TabularMLP [imbalance]` | torchregress | 0.8503 ± 0.0169 | 0.0456 ± 0.0046 | 4.6703 ± 0.7347 † | 1.0548 ± 0.0955 | 0.0000 ± 0.0000 | 12.9306 |
| `tr_Mondrian+KumoTabular [imbalance]` | torchregress | 0.8617 ± 0.0281 | 2.784e-04 ± 1.426e-04 | 4.6674 ± 0.7399 † | 1.0590 ± 0.0955 | 0.0000 ± 0.0000 | 3.3071 |
| `tr_Mondrian+LightGBM [imbalance]` | torchregress | 0.9489 ± 0.0080 | 8.821e-09 ± 4.277e-09 | 4.6671 ± 0.7398 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 1.2556 |
| `tr_Mondrian+TabFM [imbalance]` | torchregress | 0.8625 ± 0.0030 | 0.0540 ± 0.0180 | 4.6817 ± 0.7095 † | **1.0543 ± 0.0954** | 0.0000 ± 0.0000 | 10.6096 |
| `tr_Mondrian+TabICL [imbalance]` | torchregress | 0.8783 ± 0.0051 | 5.348e-04 ± 1.394e-04 | 4.6880 ± 0.7123 † | 1.0591 ± 0.0954 | 0.0000 ± 0.0000 | 4.1779 |
| `tr_Mondrian+TabularMLP [imbalance]` | torchregress | 0.8494 ± 0.0042 | 0.0419 ± 0.0061 | 4.6675 ± 0.7332 † | 1.0548 ± 0.0955 | 0.0000 ± 0.0000 | 12.9306 |
| `tr_Normalized+KumoTabular [imbalance]` | torchregress | 0.8658 ± 0.0194 | 3.554e-04 ± 1.529e-04 | 4.6674 ± 0.7399 † | 1.0590 ± 0.0955 | 0.0000 ± 0.0000 | 3.3071 |
| `tr_Normalized+LightGBM [imbalance]` | torchregress | 0.9489 ± 0.0080 | 4.453e-04 ± 3.400e-04 | 4.6674 ± 0.7400 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 1.2556 |
| `tr_Normalized+TabICL [imbalance]` | torchregress | 0.8656 ± 0.0158 | 0.0155 ± 0.0094 | 4.6644 ± 0.7319 † | 1.0591 ± 0.0954 | 0.0000 ± 0.0000 | 4.1779 |
| `tr_Normalized+TabularMLP [imbalance]` | torchregress | 0.8503 ± 0.0169 | 0.0456 ± 0.0046 | 4.6703 ± 0.7347 † | 1.0548 ± 0.0955 | 0.0000 ± 0.0000 | 12.9306 |
| `tr_PITRecal+KumoTabular [imbalance]` | torchregress | 0.8719 ± 0.0091 | 1.769e-04 ± 7.104e-06 | 4.6672 ± 0.7397 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 3.3071 |
| `tr_PITRecal+LightGBM [imbalance]` | torchregress | 0.3136 ± 0.5432 | 0.0450 ± 0.0780 | 4.9883 ± 0.2029 | 1.1241 ± 0.0198 | 0.0000 ± 0.0000 | 1.2556 |
| `tr_PITRecal+TabICL [imbalance]` | torchregress | 0.8664 ± 0.0089 | 0.0082 ± 0.0061 | 4.6507 ± 0.7528 † | 1.0610 ± 0.0922 | 0.0000 ± 0.0000 | 4.1779 |
| `tr_PITRecal+TabularMLP [imbalance]` | torchregress | 0.8522 ± 0.0196 | 0.0090 ± 0.0067 | 4.6724 ± 0.7424 † | 1.0591 ± 0.0955 | 0.0000 ± 0.0000 | 12.9306 |
| `tr_ResidualDist+KumoTabular [imbalance]` | torchregress | 0.8731 ± 0.0073 | 1.716e-04 ± 1.166e-05 | 4.6672 ± 0.7397 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 3.3071 |
| `tr_ResidualDist+LightGBM [imbalance]` | torchregress | 0.6297 ± 0.5454 | -9.252e-18 ± 1.602e-17 | 4.6671 ± 0.7398 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 1.2556 |
| `tr_ResidualDist+TabFM [imbalance]` | torchregress | 0.8536 ± 0.0125 | 0.0523 ± 0.0403 | 4.7197 ± 0.6804 † | 1.0579 ± 0.0973 | 0.0000 ± 0.0000 | 10.6096 |
| `tr_ResidualDist+TabICL [imbalance]` | torchregress | 0.8586 ± 0.0138 | 4.643e-04 ± 1.395e-04 | 4.6883 ± 0.7121 † | 1.0591 ± 0.0954 | 0.0000 ± 0.0000 | 4.1779 |
| `tr_ResidualDist+TabularMLP [imbalance]` | torchregress | 0.8500 ± 0.0126 | 0.0096 ± 0.0097 | 4.6730 ± 0.7430 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 12.9306 |
| `tr_Split+KumoTabular [imbalance]` | torchregress | 0.8667 ± 0.0123 | 3.556e-04 ± 1.583e-04 | 4.6675 ± 0.7399 † | 1.0590 ± 0.0955 | 0.0000 ± 0.0000 | 3.3071 |
| `tr_Split+LightGBM [imbalance]` | torchregress | 0.9489 ± 0.0080 | 8.821e-09 ± 4.277e-09 | 4.6671 ± 0.7398 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 1.2556 |
| `tr_Split+TabFM [imbalance]` | torchregress | 0.8653 ± 0.0198 | 0.0894 ± 0.0455 | 4.7217 ± 0.6685 † | **1.0543 ± 0.0954** | 0.0000 ± 0.0000 | 10.6096 |
| `tr_Split+TabICL [imbalance]` | torchregress | 0.8597 ± 0.0121 | 4.464e-04 ± 1.016e-04 | 4.6883 ± 0.7121 † | 1.0591 ± 0.0954 | 0.0000 ± 0.0000 | 4.1779 |
| `tr_Split+TabularMLP [imbalance]` | torchregress | 0.8486 ± 0.0178 | 0.0469 ± 0.0013 | 4.6716 ± 0.7360 † | 1.0548 ± 0.0955 | 0.0000 ± 0.0000 | 12.9306 |

##### fm_uq_scenarios / insurance_claims@noise_all

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_all]` | sdm | 0.9497 ± 0.0094 | 1.6750 ± 0.3735 | 4.6033 ± 0.3948 † | 1.0590 ± 0.0955 | 0.0196 ± 0.0340 | 3.4428 |
| `Raw+LightGBM [noise_all]` | lightgbm | 0.3192 ± 0.5528 | 1.9260 ± 1.1345 | 4.6575 ± 0.3503 † | 1.1018 ± 0.0803 | 0.0000 ± 0.0000 | 7.4287 |
| `Raw+TabICL [noise_all]` | tabicl | 0.9139 ± 0.0092 | 1.4443 ± 0.5731 | 4.4997 ± 0.3567 † | 1.0633 ± 0.0879 | 0.0357 ± 0.0397 | 4.3449 |
| `Raw+TabularMLP [noise_all]` | torchregress | 0.9489 ± 0.0080 | 3.2320 ± 0.2477 | 6.0378 ± 0.3546 † | **1.0348 ± 0.0885** | 0.0000 ± 0.0000 | 13.1078 |
| `tr_CQR+KumoTabular [noise_all]` | torchregress | 0.9019 ± 0.0111 | 1.6718 ± 0.3740 | 4.6019 ± 0.3948 † | 1.0590 ± 0.0955 | 0.0196 ± 0.0340 | 3.4428 |
| `tr_CQR+LightGBM [noise_all]` | torchregress | 0.9489 ± 0.0080 | 1.9260 ± 1.1345 | 4.6575 ± 0.3503 † | 1.1018 ± 0.0803 | 0.0000 ± 0.0000 | 7.4287 |
| `tr_CQR+TabICL [noise_all]` | torchregress | 0.8919 ± 0.0025 | 1.4438 ± 0.5731 | 4.4997 ± 0.3567 † | 1.0633 ± 0.0879 | 0.0357 ± 0.0397 | 4.3449 |
| `tr_CQR+TabularMLP [noise_all]` | torchregress | 0.9033 ± 0.0042 | 0.6057 ± 0.2418 | 4.7681 ± 0.5438 † | **1.0348 ± 0.0885** | 0.0000 ± 0.0000 | 13.1078 |
| `tr_DCP+KumoTabular [noise_all]` | torchregress | 0.9036 ± 0.0021 | 0.0060 ± 0.0085 | 4.6492 ± 0.7717 † | 1.0590 ± 0.0955 | 0.0000 ± 0.0000 | 3.4428 |
| `tr_DCP+LightGBM [noise_all]` | torchregress | 0.3578 ± 0.4602 | 7.5917 ± 9.3088 | 8.9946 ± 7.8374 † | 1.1018 ± 0.0803 | 0.0280 ± 0.0332 | 7.4287 |
| `tr_DCP+TabICL [noise_all]` | torchregress | 0.8906 ± 0.0043 | 0.6228 ± 0.3926 | 4.5420 ± 0.5628 † | 1.0633 ± 0.0879 | 0.0065 ± 0.0113 | 4.3449 |
| `tr_DCP+TabularMLP [noise_all]` | torchregress | 0.8961 ± 0.0113 | 0.4760 ± 0.0841 | 4.6884 ± 0.6528 † | **1.0348 ± 0.0885** | 0.0000 ± 0.0000 | 13.1078 |
| `tr_Mondrian+KumoTabular [noise_all]` | torchregress | 0.9064 ± 0.0077 | 9.094e-04 ± 2.519e-04 | 4.6700 ± 0.7360 † | 1.0590 ± 0.0955 | 0.0000 ± 0.0000 | 3.4428 |
| `tr_Mondrian+LightGBM [noise_all]` | torchregress | 0.9483 ± 0.0088 | 0.1477 ± 0.0452 | 4.8148 ± 0.7222 † | 1.1018 ± 0.0803 | 0.0000 ± 0.0000 | 7.4287 |
| `tr_Mondrian+TabFM [noise_all]` | torchregress | 0.8917 ± 0.0008 | 1.0922 ± 0.9082 | 4.8818 ± 0.6065 † | 1.0395 ± 0.0759 | 0.0106 ± 0.0183 | 11.0238 |
| `tr_Mondrian+TabICL [noise_all]` | torchregress | 0.9000 ± 0.0073 | 0.0037 ± 0.0005 | 4.7200 ± 0.6520 † | 1.0633 ± 0.0879 | 0.0000 ± 0.0000 | 4.3449 |
| `tr_Mondrian+TabularMLP [noise_all]` | torchregress | 0.8925 ± 0.0079 | 0.4569 ± 0.0840 | 4.6751 ± 0.6500 † | **1.0348 ± 0.0885** | 0.0000 ± 0.0000 | 13.1078 |
| `tr_Normalized+KumoTabular [noise_all]` | torchregress | 0.9006 ± 0.0051 | 0.0016 ± 0.0007 | 4.6704 ± 0.7356 † | 1.0590 ± 0.0955 | 0.0000 ± 0.0000 | 3.4428 |
| `tr_Normalized+LightGBM [noise_all]` | torchregress | 0.8992 ± 0.0073 | 2.620e-08 ± 1.000e-08 | 5.1831 ± 0.6091 † | 1.1018 ± 0.0803 | 0.0000 ± 0.0000 | 7.4287 |
| `tr_Normalized+TabICL [noise_all]` | torchregress | 0.8958 ± 0.0113 | 0.0093 ± 0.0029 | 4.7216 ± 0.6526 † | 1.0633 ± 0.0879 | 0.0000 ± 0.0000 | 4.3449 |
| `tr_Normalized+TabularMLP [noise_all]` | torchregress | 0.8961 ± 0.0113 | 0.4760 ± 0.0841 | 4.6884 ± 0.6528 † | **1.0348 ± 0.0885** | 0.0000 ± 0.0000 | 13.1078 |
| `tr_PITRecal+KumoTabular [noise_all]` | torchregress | 0.9006 ± 0.0182 | 1.4835 ± 1.2681 | 4.4787 ± 0.6155 † | 1.0590 ± 0.0955 | 0.0065 ± 0.0113 | 3.4428 |
| `tr_PITRecal+LightGBM [noise_all]` | torchregress | 0.3308 ± 0.5064 | 2.0133 ± 0.9503 | 5.4888 ± 1.1248 † | 1.1414 ± 0.0477 | 0.0000 ± 0.0000 | 7.4287 |
| `tr_PITRecal+TabICL [noise_all]` | torchregress | 0.8925 ± 0.0073 | 0.9875 ± 0.8239 | **4.4346 ± 0.5612** † | 1.0589 ± 0.0957 | 0.0065 ± 0.0113 | 4.3449 |
| `tr_PITRecal+TabularMLP [noise_all]` | torchregress | 0.8961 ± 0.0113 | 1.5686 ± 1.2698 | 4.5289 ± 0.5722 † | 1.0588 ± 0.0951 | 0.0000 ± 0.0000 | 13.1078 |
| `tr_ResidualDist+KumoTabular [noise_all]` | torchregress | 0.8975 ± 0.0153 | 1.5695 ± 1.3598 | 4.5026 ± 0.5848 † | 1.0590 ± 0.0954 | 0.0000 ± 0.0000 | 3.4428 |
| `tr_ResidualDist+LightGBM [noise_all]` | torchregress | 0.6189 ± 0.5360 | 1.5685 ± 1.3601 | 5.0159 ± 0.4735 † | 1.1018 ± 0.0803 | 0.0000 ± 0.0000 | 7.4287 |
| `tr_ResidualDist+TabFM [noise_all]` | torchregress | 0.8947 ± 0.0065 | 1.7295 ± 1.2817 | 4.7682 ± 0.3223 † | 1.0594 ± 0.0859 | 0.0000 ± 0.0000 | 11.0238 |
| `tr_ResidualDist+TabICL [noise_all]` | torchregress | 0.8900 ± 0.0036 | 1.5721 ± 1.3586 | 4.5539 ± 0.5002 † | 1.0634 ± 0.0879 | 0.0000 ± 0.0000 | 4.3449 |
| `tr_ResidualDist+TabularMLP [noise_all]` | torchregress | 0.8950 ± 0.0036 | 1.6482 ± 1.2774 | 4.5530 ± 0.5450 † | 1.0588 ± 0.0949 | 0.0000 ± 0.0000 | 13.1078 |
| `tr_Split+KumoTabular [noise_all]` | torchregress | 0.9089 ± 0.0072 | 0.0012 ± 0.0001 | 4.6703 ± 0.7360 † | 1.0590 ± 0.0955 | 0.0000 ± 0.0000 | 3.4428 |
| `tr_Split+LightGBM [noise_all]` | torchregress | 0.9347 ± 0.0112 | 8.821e-09 ± 4.277e-09 | 5.1831 ± 0.6091 † | 1.1018 ± 0.0803 | 0.0000 ± 0.0000 | 7.4287 |
| `tr_Split+TabFM [noise_all]` | torchregress | 0.8947 ± 0.0055 | 0.6611 ± 0.1116 | 4.9278 ± 0.3994 † | 1.0395 ± 0.0759 | 0.0000 ± 0.0000 | 11.0238 |
| `tr_Split+TabICL [noise_all]` | torchregress | 0.8869 ± 0.0059 | 0.0051 ± 0.0003 | 4.7212 ± 0.6518 † | 1.0633 ± 0.0879 | 0.0000 ± 0.0000 | 4.3449 |
| `tr_Split+TabularMLP [noise_all]` | torchregress | 0.8958 ± 0.0052 | 0.5103 ± 0.1149 | 4.7126 ± 0.6276 † | **1.0348 ± 0.0885** | 0.0000 ± 0.0000 | 13.1078 |

##### fm_uq_scenarios / insurance_claims@noise_test

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_test]` | sdm | 1.0000 ± 0.0000 | 19.9212 ± 4.8752 | 19.9212 ± 4.8752 † | 1.2053 ± 0.1680 | 1.0000 ± 0.0000 | 3.4634 |
| `Raw+LightGBM [noise_test]` | lightgbm | 0.3106 ± 0.5357 | 1.6521 ± 0.8964 | 4.6309 ± 0.4576 † | 1.0812 ± 0.0701 | 0.0000 ± 0.0000 | 1.3592 |
| `Raw+TabICL [noise_test]` | tabicl | 0.9994 ± 0.0010 | 9.9557 ± 0.2723 | 9.9604 ± 0.2649 | 1.1951 ± 0.0486 | 0.9894 ± 0.0183 | 4.3102 |
| `Raw+TabularMLP [noise_test]` | torchregress | 0.9489 ± 0.0080 | 3.2176 ± 0.1588 | 6.0107 ± 0.4035 † | **1.0339 ± 0.0892** | 0.0000 ± 0.0000 | 13.6000 |
| `tr_CQR+KumoTabular [noise_test]` | torchregress | 1.0000 ± 0.0000 | 19.9183 ± 4.8757 | 19.9183 ± 4.8757 † | 1.2053 ± 0.1680 | 1.0000 ± 0.0000 | 3.4634 |
| `tr_CQR+LightGBM [noise_test]` | torchregress | 0.9403 ± 0.0098 | 1.6521 ± 0.8964 | 4.6309 ± 0.4576 † | 1.0812 ± 0.0701 | 0.0143 ± 0.0247 | 1.3592 |
| `tr_CQR+TabICL [noise_test]` | torchregress | 0.9994 ± 0.0010 | 9.9540 ± 0.2722 | 9.9587 ± 0.2648 | 1.1951 ± 0.0486 | 0.9894 ± 0.0183 | 4.3102 |
| `tr_CQR+TabularMLP [noise_test]` | torchregress | 0.8914 ± 0.0240 | 0.6991 ± 0.1800 | 4.8111 ± 0.5738 † | **1.0339 ± 0.0892** | 0.0000 ± 0.0000 | 13.6000 |
| `tr_DCP+KumoTabular [noise_test]` | torchregress | 0.8572 ± 0.1513 | 4.5285 ± 5.9940 | 7.2889 ± 3.9589 † | 1.2053 ± 0.1680 | 0.2228 ± 0.3859 | 3.4634 |
| `tr_DCP+LightGBM [noise_test]` | torchregress | 0.3214 ± 0.4018 | 7.5546 ± 8.0327 | 8.8762 ± 6.4577 † | 1.0812 ± 0.0701 | 0.0276 ± 0.0447 | 1.3592 |
| `tr_DCP+TabICL [noise_test]` | torchregress | 0.7586 ± 0.4116 | 6.9769 ± 1.0045 | 7.3226 ± 0.6208 † | 1.1951 ± 0.0486 | 0.7059 ± 0.3914 | 4.3102 |
| `tr_DCP+TabularMLP [noise_test]` | torchregress | 0.8969 ± 0.0229 | 0.4708 ± 0.0614 | 4.6784 ± 0.6563 † | **1.0339 ± 0.0892** | 0.0000 ± 0.0000 | 13.6000 |
| `tr_Mondrian+KumoTabular [noise_test]` | torchregress | 0.0017 ± 0.0029 | 9.145e-04 ± 6.268e-04 | 7.3326 ± 3.1245 † | 1.2053 ± 0.1680 | 0.0000 ± 0.0000 | 3.4634 |
| `tr_Mondrian+LightGBM [noise_test]` | torchregress | 0.9503 ± 0.0092 | 0.1217 ± 0.1203 | 4.6910 ± 0.7568 † | 1.0812 ± 0.0701 | 0.0302 ± 0.0294 | 1.3592 |
| `tr_Mondrian+TabFM [noise_test]` | torchregress | 8.333e-04 ± 1.443e-03 | 1.3244 ± 0.3420 | 16.6730 ± 7.6864 † | 1.5324 ± 0.2454 | 0.0000 ± 0.0000 | 18.4081 |
| `tr_Mondrian+TabICL [noise_test]` | torchregress | 0.0000 ± 0.0000 | 0.0110 ± 0.0055 | 18.8780 ± 0.6342 | 1.1951 ± 0.0486 | 0.0000 ± 0.0000 | 4.3102 |
| `tr_Mondrian+TabularMLP [noise_test]` | torchregress | 0.8983 ± 0.0116 | 0.4622 ± 0.0545 | 4.6685 ± 0.6644 † | **1.0339 ± 0.0892** | 0.0000 ± 0.0000 | 13.6000 |
| `tr_Normalized+KumoTabular [noise_test]` | torchregress | 0.0633 ± 0.0674 | 0.0286 ± 0.0074 | 7.0914 ± 3.0944 † | 1.2053 ± 0.1680 | 0.0000 ± 0.0000 | 3.4634 |
| `tr_Normalized+LightGBM [noise_test]` | torchregress | 0.7036 ± 0.1527 | 6.309e-08 ± 3.007e-08 | 4.9380 ± 0.4218 † | 1.0812 ± 0.0701 | 0.0000 ± 0.0000 | 1.3592 |
| `tr_Normalized+TabICL [noise_test]` | torchregress | 0.0000 ± 0.0000 | 0.2268 ± 0.1663 | 16.9358 ± 1.7394 † | 1.1951 ± 0.0486 | 0.0000 ± 0.0000 | 4.3102 |
| `tr_Normalized+TabularMLP [noise_test]` | torchregress | 0.8969 ± 0.0229 | 0.4708 ± 0.0614 | 4.6784 ± 0.6563 † | **1.0339 ± 0.0892** | 0.0000 ± 0.0000 | 13.6000 |
| `tr_PITRecal+KumoTabular [noise_test]` | torchregress | 0.5781 ± 0.4797 | 8.6183 ± 1.8870 | 8.9649 ± 1.8760 † | 1.0755 ± 0.0543 | 0.5408 ± 0.4892 | 3.4634 |
| `tr_PITRecal+LightGBM [noise_test]` | torchregress | 0.2683 ± 0.4475 | 1.5436 ± 0.7191 | 5.4078 ± 0.5237 † | 1.2420 ± 0.1268 | 0.0012 ± 0.0020 | 1.3592 |
| `tr_PITRecal+TabICL [noise_test]` | torchregress | 0.8667 ± 0.2259 | 7.2769 ± 1.9367 | 7.4508 ± 1.6993 † | 1.1357 ± 0.0507 | 0.8285 ± 0.2051 | 4.3102 |
| `tr_PITRecal+TabularMLP [noise_test]` | torchregress | 0.8956 ± 0.0267 | 1.5019 ± 1.2404 | **4.5247 ± 0.5746** † | 1.0590 ± 0.0949 | 0.0000 ± 0.0000 | 13.6000 |
| `tr_ResidualDist+KumoTabular [noise_test]` | torchregress | 0.0017 ± 0.0029 | 1.5698 ± 1.3597 | 7.1675 ± 3.2670 † | 1.2052 ± 0.1681 | 0.0000 ± 0.0000 | 3.4634 |
| `tr_ResidualDist+LightGBM [noise_test]` | torchregress | 0.6208 ± 0.5377 | 1.5685 ± 1.3601 | 4.7708 ± 0.3023 † | 1.0812 ± 0.0701 | 0.0000 ± 0.0000 | 1.3592 |
| `tr_ResidualDist+TabFM [noise_test]` | torchregress | 0.0017 ± 0.0029 | 1.7485 ± 1.2219 | 21.8048 ± 5.9842 † | 1.4026 ± 0.2448 | 0.0000 ± 0.0000 | 18.4081 |
| `tr_ResidualDist+TabICL [noise_test]` | torchregress | 0.0000 ± 0.0000 | 1.5714 ± 1.3585 | 18.7711 ± 0.4741 | 1.1950 ± 0.0486 | 0.0000 ± 0.0000 | 4.3102 |
| `tr_ResidualDist+TabularMLP [noise_test]` | torchregress | 0.8972 ± 0.0104 | 1.6516 ± 1.2784 | 4.5622 ± 0.5346 † | 1.0586 ± 0.0945 | 0.0000 ± 0.0000 | 13.6000 |
| `tr_Split+KumoTabular [noise_test]` | torchregress | 0.0014 ± 0.0024 | 0.0014 ± 0.0004 | 7.3281 ± 3.1304 † | 1.2053 ± 0.1680 | 0.0000 ± 0.0000 | 3.4634 |
| `tr_Split+LightGBM [noise_test]` | torchregress | 0.9358 ± 0.0092 | 8.821e-09 ± 4.277e-09 | 4.9380 ± 0.4218 † | 1.0812 ± 0.0701 | 0.0000 ± 0.0000 | 1.3592 |
| `tr_Split+TabFM [noise_test]` | torchregress | 0.0000 ± 0.0000 | 0.7265 ± 0.0511 | 22.0537 ± 5.9504 † | 1.5324 ± 0.2454 | 0.0000 ± 0.0000 | 18.4081 |
| `tr_Split+TabICL [noise_test]` | torchregress | 0.0000 ± 0.0000 | 0.0045 ± 0.0006 | 18.9366 ± 0.6025 | 1.1951 ± 0.0486 | 0.0000 ± 0.0000 | 4.3102 |
| `tr_Split+TabularMLP [noise_test]` | torchregress | 0.8975 ± 0.0120 | 0.5340 ± 0.1106 | 4.7223 ± 0.6141 † | **1.0339 ± 0.0892** | 0.0000 ± 0.0000 | 13.6000 |

##### fm_uq_scenarios / insurance_claims@outliers

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [outliers]` | sdm | 0.9767 ± 0.0161 | 4.1166 ± 0.2993 | 4.6963 ± 0.1527 | 1.0590 ± 0.0957 | 0.5315 ± 0.3103 | 3.4440 |
| `Raw+LightGBM [outliers]` | lightgbm | 0.7769 ± 0.3030 | 3.2156 ± 0.1698 | 4.8097 ± 0.3436 † | 1.0918 ± 0.0662 | 0.1426 ± 0.0960 | 1.4713 |
| `Raw+TabICL [outliers]` | tabicl | 0.9672 ± 0.0104 | 3.6012 ± 0.4677 | 5.2009 ± 0.3159 † | 1.0563 ± 0.0977 | 0.3689 ± 0.1299 | 4.3249 |
| `Raw+TabularMLP [outliers]` | torchregress | 0.9506 ± 0.0083 | 6.6737 ± 0.3947 | 7.7006 ± 0.1422 | 1.0373 ± 0.0887 | 0.0339 ± 0.0304 | 14.4452 |
| `tr_CQR+KumoTabular [outliers]` | torchregress | 0.9492 ± 0.0068 | 4.0697 ± 0.3129 | 4.6639 ± 0.1526 | 1.0590 ± 0.0957 | 0.4948 ± 0.3284 | 3.4440 |
| `tr_CQR+LightGBM [outliers]` | torchregress | 0.9522 ± 0.0142 | 3.2156 ± 0.1698 | 4.8097 ± 0.3436 † | 1.0918 ± 0.0662 | 0.1426 ± 0.0960 | 1.4713 |
| `tr_CQR+TabICL [outliers]` | torchregress | 0.9442 ± 0.0072 | 3.5750 ± 0.4595 | 5.1838 ± 0.3209 † | 1.0563 ± 0.0977 | 0.3623 ± 0.1193 | 4.3249 |
| `tr_CQR+TabularMLP [outliers]` | torchregress | 0.9442 ± 0.0033 | 3.0061 ± 3.1272 | 5.7613 ± 1.6988 † | 1.0373 ± 0.0887 | 0.0143 ± 0.0247 | 14.4452 |
| `tr_DCP+KumoTabular [outliers]` | torchregress | 0.9464 ± 0.0059 | 1.4012 ± 2.1685 | **4.4032 ± 0.6441** † | 1.0590 ± 0.0957 | 0.0303 ± 0.0369 | 3.4440 |
| `tr_DCP+LightGBM [outliers]` | torchregress | 0.9389 ± 0.0118 | 2.8500 ± 1.9498 | 4.7810 ± 0.8307 † | 1.0918 ± 0.0662 | 0.2344 ± 0.3420 | 1.4713 |
| `tr_DCP+TabICL [outliers]` | torchregress | 0.9461 ± 0.0027 | 1.0204 ± 0.4723 | 4.4775 ± 0.6841 † | 1.0563 ± 0.0977 | 0.0422 ± 0.0279 | 4.3249 |
| `tr_DCP+TabularMLP [outliers]` | torchregress | 0.9458 ± 0.0022 | 2.7176 ± 3.3689 | 5.6158 ± 1.8122 † | 1.0373 ± 0.0887 | 0.0143 ± 0.0247 | 14.4452 |
| `tr_Mondrian+KumoTabular [outliers]` | torchregress | 0.9428 ± 0.0046 | 3.9436 ± 1.9321 | 6.6180 ± 1.4653 † | 1.0590 ± 0.0957 | 0.0598 ± 0.0033 | 3.4440 |
| `tr_Mondrian+LightGBM [outliers]` | torchregress | 0.9481 ± 0.0067 | 2.8119 ± 4.7090 | 5.8510 ± 2.4357 † | 1.0918 ± 0.0662 | 0.0000 ± 0.0000 | 1.4713 |
| `tr_Mondrian+TabFM [outliers]` | torchregress | 0.9436 ± 0.0047 | 3.9964 ± 1.0747 | 6.5170 ± 1.3313 † | **1.0365 ± 0.0883** | 0.0924 ± 0.0519 | 10.9988 |
| `tr_Mondrian+TabICL [outliers]` | torchregress | 0.9475 ± 0.0073 | 3.5632 ± 0.6984 | 6.0281 ± 0.5030 † | 1.0563 ± 0.0977 | 0.0967 ± 0.0326 | 4.3249 |
| `tr_Mondrian+TabularMLP [outliers]` | torchregress | 0.9472 ± 0.0129 | 3.5108 ± 0.0477 | 6.2193 ± 0.4926 † | 1.0373 ± 0.0887 | 0.1490 ± 0.1439 | 14.4452 |
| `tr_Normalized+KumoTabular [outliers]` | torchregress | 0.9461 ± 0.0051 | 2.7134 ± 4.6904 | 5.7976 ± 2.4768 † | 1.0590 ± 0.0957 | 0.0571 ± 0.0990 | 3.4440 |
| `tr_Normalized+LightGBM [outliers]` | torchregress | 0.9442 ± 0.0014 | 1.7817 ± 3.0860 | 5.7157 ± 1.4069 † | 1.0918 ± 0.0662 | 0.0190 ± 0.0330 | 1.4713 |
| `tr_Normalized+TabICL [outliers]` | torchregress | 0.9469 ± 0.0035 | 1.3554 ± 2.1285 | 5.1983 ± 1.3773 † | 1.0563 ± 0.0977 | 0.0238 ± 0.0412 | 4.3249 |
| `tr_Normalized+TabularMLP [outliers]` | torchregress | 0.9458 ± 0.0022 | 2.7176 ± 3.3689 | 5.6158 ± 1.8122 † | 1.0373 ± 0.0887 | 0.0143 ± 0.0247 | 14.4452 |
| `tr_PITRecal+KumoTabular [outliers]` | torchregress | 0.9422 ± 0.0049 | 4.1205 ± 0.0689 | 4.6138 ± 0.1853 | 1.0590 ± 0.0955 | 0.5284 ± 0.1988 | 3.4440 |
| `tr_PITRecal+LightGBM [outliers]` | torchregress | 0.9506 ± 0.0242 | 4.4378 ± 0.4427 | 5.5325 ± 0.4567 † | 1.2505 ± 0.2024 | 0.4807 ± 0.1912 | 1.4713 |
| `tr_PITRecal+TabICL [outliers]` | torchregress | 0.9500 ± 0.0058 | 3.7933 ± 0.2722 | 4.9233 ± 0.4542 † | 1.0605 ± 0.0926 | 0.4424 ± 0.1083 | 4.3249 |
| `tr_PITRecal+TabularMLP [outliers]` | torchregress | 0.9475 ± 0.0142 | 4.4650 ± 0.1404 | 4.8735 ± 0.2219 | 1.0610 ± 0.0939 | 0.5008 ± 0.2001 | 14.4452 |
| `tr_ResidualDist+KumoTabular [outliers]` | torchregress | 0.9314 ± 0.0071 | 4.2679 ± 0.1273 | 4.5769 ± 0.1704 | 1.0590 ± 0.0955 | 0.5089 ± 0.0192 | 3.4440 |
| `tr_ResidualDist+LightGBM [outliers]` | torchregress | 0.6531 ± 0.5318 | 4.2669 ± 0.1273 | 4.9851 ± 0.2854 † | 1.0918 ± 0.0662 | 0.3069 ± 0.5315 | 1.4713 |
| `tr_ResidualDist+TabFM [outliers]` | torchregress | 0.9469 ± 0.0038 | 4.4663 ± 0.1520 | 4.8412 ± 0.1714 | 1.0593 ± 0.0943 | 0.5451 ± 0.1687 | 10.9988 |
| `tr_ResidualDist+TabICL [outliers]` | torchregress | 0.9481 ± 0.0029 | 4.2848 ± 0.1263 | 4.6365 ± 0.1631 | 1.0556 ± 0.0975 | 0.4874 ± 0.1078 | 4.3249 |
| `tr_ResidualDist+TabularMLP [outliers]` | torchregress | 0.9411 ± 0.0176 | 4.4118 ± 0.0923 | 4.7599 ± 0.1630 | 1.0603 ± 0.0955 | 0.4689 ± 0.1976 | 14.4452 |
| `tr_Split+KumoTabular [outliers]` | torchregress | 0.9411 ± 0.0061 | 2.7714 ± 4.7906 | 5.8223 ± 2.5182 † | 1.0590 ± 0.0957 | 0.0095 ± 0.0165 | 3.4440 |
| `tr_Split+LightGBM [outliers]` | torchregress | 0.9431 ± 0.0017 | 2.7679 ± 4.7942 | 6.1638 ± 2.1723 † | 1.0918 ± 0.0662 | 0.0000 ± 0.0000 | 1.4713 |
| `tr_Split+TabFM [outliers]` | torchregress | 0.9428 ± 0.0017 | 3.1754 ± 3.7726 | 5.8550 ± 1.9287 † | **1.0365 ± 0.0883** | 0.0095 ± 0.0165 | 10.9988 |
| `tr_Split+TabICL [outliers]` | torchregress | 0.9458 ± 0.0008 | 2.8360 ± 4.7208 | 5.8758 ± 2.4735 † | 1.0563 ± 0.0977 | 0.0286 ± 0.0495 | 4.3249 |
| `tr_Split+TabularMLP [outliers]` | torchregress | 0.9453 ± 0.0035 | 3.0312 ± 3.9203 | 5.7618 ± 2.0522 † | 1.0373 ± 0.0887 | 0.0143 ± 0.0247 | 14.4452 |

##### fm_uq_scenarios / insurance_claims@shift

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [shift]` | sdm | 0.9514 ± 0.0013 | 1.8315 ± 0.4020 | 4.0105 ± 0.0403 | 1.0093 ± 0.0624 | 0.0215 ± 0.0241 | 3.4583 |
| `Raw+LightGBM [shift]` | lightgbm | 0.3161 ± 0.5468 | 2.1389 ± 0.9165 | 4.4244 ± 0.3598 † | 1.0092 ± 0.0624 | 0.0000 ± 0.0000 | 1.3325 |
| `Raw+TabICL [shift]` | tabicl | 0.9456 ± 0.0043 | 0.9525 ± 0.3234 | 4.2509 ± 0.2791 † | 1.0091 ± 0.0625 | 0.0268 ± 0.0329 | 4.3188 |
| `Raw+TabularMLP [shift]` | torchregress | 0.9503 ± 0.0025 | 3.1850 ± 0.1425 | 5.7516 ± 0.1594 | 0.9839 ± 0.0598 | 0.0000 ± 0.0000 | 13.9440 |
| `tr_CQR+KumoTabular [shift]` | torchregress | 0.8947 ± 0.0171 | 1.8286 ± 0.4019 | **4.0092 ± 0.0403** | 1.0093 ± 0.0624 | 0.0215 ± 0.0241 | 3.4583 |
| `tr_CQR+LightGBM [shift]` | torchregress | 0.9506 ± 0.0027 | 2.1389 ± 0.9165 | 4.4244 ± 0.3598 † | 1.0092 ± 0.0624 | 0.0056 ± 0.0098 | 1.3325 |
| `tr_CQR+TabICL [shift]` | torchregress | 0.8903 ± 0.0164 | 0.9510 ± 0.3237 | 4.2505 ± 0.2792 † | 1.0091 ± 0.0625 | 0.0268 ± 0.0329 | 4.3188 |
| `tr_CQR+TabularMLP [shift]` | torchregress | 0.9019 ± 0.0154 | 0.7553 ± 0.1114 | 4.5590 ± 0.3571 † | 0.9839 ± 0.0598 | 0.0000 ± 0.0000 | 13.9440 |
| `tr_DCP+KumoTabular [shift]` | torchregress | 0.9053 ± 0.0136 | 0.0013 ± 0.0002 | 4.4098 ± 0.3290 † | 1.0093 ± 0.0624 | 0.0000 ± 0.0000 | 3.4583 |
| `tr_DCP+LightGBM [shift]` | torchregress | 0.3528 ± 0.5028 | 12.3788 ± 10.6168 | 13.9207 ± 7.9503 † | 1.0092 ± 0.0624 | 0.0099 ± 0.0088 | 1.3325 |
| `tr_DCP+TabICL [shift]` | torchregress | 0.8822 ± 0.0158 | 0.0737 ± 0.0148 | 4.3611 ± 0.2648 † | 1.0091 ± 0.0625 | 0.0000 ± 0.0000 | 4.3188 |
| `tr_DCP+TabularMLP [shift]` | torchregress | 0.9125 ± 0.0104 | 0.4779 ± 0.0458 | 4.4212 ± 0.3105 † | 0.9839 ± 0.0598 | 0.0000 ± 0.0000 | 13.9440 |
| `tr_Mondrian+KumoTabular [shift]` | torchregress | 0.9039 ± 0.0024 | 9.613e-04 ± 1.732e-04 | 4.4097 ± 0.3288 † | 1.0093 ± 0.0624 | 0.0000 ± 0.0000 | 3.4583 |
| `tr_Mondrian+LightGBM [shift]` | torchregress | 0.9511 ± 0.0024 | 0.0410 ± 0.0374 | 4.3797 ± 0.3178 † | 1.0092 ± 0.0624 | 0.0166 ± 0.0170 | 1.3325 |
| `tr_Mondrian+TabFM [shift]` | torchregress | 0.9000 ± 0.0068 | 0.5463 ± 0.0568 | 4.4109 ± 0.2758 † | **0.9825 ± 0.0596** | 0.0000 ± 0.0000 | 11.0327 |
| `tr_Mondrian+TabICL [shift]` | torchregress | 0.9158 ± 0.0022 | 0.0034 ± 0.0005 | 4.4107 ± 0.3295 † | 1.0091 ± 0.0625 | 0.0000 ± 0.0000 | 4.3188 |
| `tr_Mondrian+TabularMLP [shift]` | torchregress | 0.9033 ± 0.0060 | 0.4722 ± 0.0384 | 4.4144 ± 0.3078 † | 0.9839 ± 0.0598 | 0.0000 ± 0.0000 | 13.9440 |
| `tr_Normalized+KumoTabular [shift]` | torchregress | 0.9022 ± 0.0027 | 0.0090 ± 0.0123 | 4.4125 ± 0.3289 † | 1.0093 ± 0.0624 | 0.0000 ± 0.0000 | 3.4583 |
| `tr_Normalized+LightGBM [shift]` | torchregress | 0.9197 ± 0.0113 | 5.037e-08 ± 4.297e-08 | 4.4089 ± 0.3289 † | 1.0092 ± 0.0624 | 0.0000 ± 0.0000 | 1.3325 |
| `tr_Normalized+TabICL [shift]` | torchregress | 0.9000 ± 0.0147 | 0.0280 ± 0.0146 | 4.4205 ± 0.3285 † | 1.0091 ± 0.0625 | 0.0000 ± 0.0000 | 4.3188 |
| `tr_Normalized+TabularMLP [shift]` | torchregress | 0.9125 ± 0.0104 | 0.4779 ± 0.0458 | 4.4212 ± 0.3105 † | 0.9839 ± 0.0598 | 0.0000 ± 0.0000 | 13.9440 |
| `tr_PITRecal+KumoTabular [shift]` | torchregress | 0.8994 ± 0.0093 | 1.3840 ± 0.7235 | 4.2404 ± 0.1165 | 1.0092 ± 0.0624 | 0.0000 ± 0.0000 | 3.4583 |
| `tr_PITRecal+LightGBM [shift]` | torchregress | 0.3114 ± 0.5155 | 2.0409 ± 1.0322 | 4.5677 ± 0.4843 † | 1.0725 ± 0.1550 | 0.0000 ± 0.0000 | 1.3325 |
| `tr_PITRecal+TabICL [shift]` | torchregress | 0.8894 ± 0.0081 | 0.9643 ± 0.5319 | 4.2698 ± 0.2687 † | 1.0092 ± 0.0625 | 0.0056 ± 0.0098 | 4.3188 |
| `tr_PITRecal+TabularMLP [shift]` | torchregress | 0.9125 ± 0.0104 | 2.0017 ± 0.2753 | 4.4075 ± 0.2248 † | 1.0085 ± 0.0624 | 0.0000 ± 0.0000 | 13.9440 |
| `tr_ResidualDist+KumoTabular [shift]` | torchregress | 0.8983 ± 0.0129 | 2.3475 ± 0.0699 | 4.4221 ± 0.2143 | 1.0092 ± 0.0624 | 0.0000 ± 0.0000 | 3.4583 |
| `tr_ResidualDist+LightGBM [shift]` | torchregress | 0.6317 ± 0.5470 | 2.3468 ± 0.0700 | 4.4215 ± 0.2143 | 1.0092 ± 0.0624 | 0.0000 ± 0.0000 | 1.3325 |
| `tr_ResidualDist+TabFM [shift]` | torchregress | 0.9169 ± 0.0029 | 2.4707 ± 0.0518 | 4.6536 ± 0.2974 † | 1.0038 ± 0.0601 | 0.0000 ± 0.0000 | 11.0327 |
| `tr_ResidualDist+TabICL [shift]` | torchregress | 0.9058 ± 0.0050 | 2.3488 ± 0.0698 | 4.4246 ± 0.2148 | 1.0092 ± 0.0625 | 0.0000 ± 0.0000 | 4.3188 |
| `tr_ResidualDist+TabularMLP [shift]` | torchregress | 0.9156 ± 0.0042 | 2.4148 ± 0.0712 | 4.4974 ± 0.2678 † | 1.0078 ± 0.0622 | 0.0000 ± 0.0000 | 13.9440 |
| `tr_Split+KumoTabular [shift]` | torchregress | 0.9167 ± 0.0038 | 0.0015 ± 0.0000 | 4.4101 ± 0.3288 † | 1.0093 ± 0.0624 | 0.0000 ± 0.0000 | 3.4583 |
| `tr_Split+LightGBM [shift]` | torchregress | 0.9475 ± 0.0033 | 8.821e-09 ± 4.277e-09 | 4.4089 ± 0.3289 † | 1.0092 ± 0.0624 | 0.0000 ± 0.0000 | 1.3325 |
| `tr_Split+TabFM [shift]` | torchregress | 0.9158 ± 0.0025 | 0.7522 ± 0.1207 | 4.6387 ± 0.3782 † | **0.9825 ± 0.0596** | 0.0000 ± 0.0000 | 11.0327 |
| `tr_Split+TabICL [shift]` | torchregress | 0.9053 ± 0.0067 | 0.0047 ± 0.0006 | 4.4120 ± 0.3293 † | 1.0091 ± 0.0625 | 0.0000 ± 0.0000 | 4.3188 |
| `tr_Split+TabularMLP [shift]` | torchregress | 0.9150 ± 0.0029 | 0.5733 ± 0.0990 | 4.4858 ± 0.3568 † | 0.9839 ± 0.0598 | 0.0000 ± 0.0000 | 13.9440 |
| `tr_WeightedCQR+KumoTabular [shift]` | torchregress | 0.8942 ± 0.0131 | 1.8307 ± 0.4025 | 4.0112 ± 0.0432 | 1.0093 ± 0.0624 | 0.0215 ± 0.0241 | 3.4583 |
| `tr_WeightedCQR+LightGBM [shift]` | torchregress | 0.9506 ± 0.0027 | 2.1447 ± 0.9143 | 4.4301 ± 0.3691 † | 1.0092 ± 0.0624 | 0.0056 ± 0.0098 | 1.3325 |
| `tr_WeightedCQR+TabICL [shift]` | torchregress | 0.8961 ± 0.0025 | 0.9575 ± 0.3197 | 4.2569 ± 0.2856 † | 1.0091 ± 0.0625 | 0.0268 ± 0.0329 | 4.3188 |
| `tr_WeightedCQR+TabularMLP [shift]` | torchregress | 0.9025 ± 0.0044 | 0.7597 ± 0.1483 | 4.5637 ± 0.3648 † | 0.9839 ± 0.0598 | 0.0000 ± 0.0000 | 13.9440 |
| `tr_WeightedSplit+KumoTabular [shift]` | torchregress | 0.8986 ± 0.0034 | 0.0084 ± 0.0122 | 4.4172 ± 0.3395 † | 1.0093 ± 0.0624 | 0.0000 ± 0.0000 | 3.4583 |
| `tr_WeightedSplit+LightGBM [shift]` | torchregress | 0.9475 ± 0.0033 | 0.0071 ± 0.0123 | 4.4160 ± 0.3397 † | 1.0092 ± 0.0624 | 0.0000 ± 0.0000 | 1.3325 |
| `tr_WeightedSplit+TabFM [shift]` | torchregress | 0.9003 ± 0.0035 | 0.6924 ± 0.0944 | 4.6400 ± 0.3840 † | **0.9825 ± 0.0596** | 0.0000 ± 0.0000 | 11.0327 |
| `tr_WeightedSplit+TabICL [shift]` | torchregress | 0.8997 ± 0.0159 | 0.0117 ± 0.0127 | 4.4191 ± 0.3400 † | 1.0091 ± 0.0625 | 0.0000 ± 0.0000 | 4.3188 |
| `tr_WeightedSplit+TabularMLP [shift]` | torchregress | 0.9022 ± 0.0067 | 0.5644 ± 0.0989 | 4.4908 ± 0.3651 † | 0.9839 ± 0.0598 | 0.0000 ± 0.0000 | 13.9440 |

##### fm_uq_scenarios / openml_year

_No successful rows._

##### fm_uq_scenarios / openml_year@clean

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [clean]` | sdm | 0.9114 ± 0.0094 | 0.9129 ± 0.0141 | **1.1862 ± 0.0908** † | 0.3297 ± 0.0288 | 0.8580 ± 0.0298 | 2.7079 |
| `Raw+LightGBM [clean]` | lightgbm | 0.8694 ± 0.0125 | 1.5208 ± 0.0315 | 2.0358 ± 0.1157 † | 0.4265 ± 0.0256 | 0.6463 ± 0.0319 | 2.1524 |
| `Raw+TabICL [clean]` | tabicl | 0.9033 ± 0.0159 | 1.0039 ± 0.0038 | 1.3610 ± 0.1067 † | 0.3618 ± 0.0273 | 0.8412 ± 0.0412 | 2.1491 |
| `Raw+TabularMLP [clean]` | torchregress | 0.9125 ± 0.0166 | 1.3777 ± 0.0169 | 1.8246 ± 0.1318 † | 0.4387 ± 0.0330 | 0.7965 ± 0.0702 | 32.1940 |
| `tr_CQR+KumoTabular [clean]` | torchregress | 0.9128 ± 0.0210 | 0.9208 ± 0.0384 | 1.1898 ± 0.0903 † | 0.3297 ± 0.0288 | 0.8608 ± 0.0485 | 2.7079 |
| `tr_CQR+LightGBM [clean]` | torchregress | 0.9047 ± 0.0049 | 1.6187 ± 0.0413 | 2.0239 ± 0.1094 † | 0.4265 ± 0.0256 | 0.7355 ± 0.0125 | 2.1524 |
| `tr_CQR+TabICL [clean]` | torchregress | 0.9042 ± 0.0164 | 1.0119 ± 0.0188 | 1.3616 ± 0.1054 † | 0.3618 ± 0.0273 | 0.8410 ± 0.0427 | 2.1491 |
| `tr_CQR+TabularMLP [clean]` | torchregress | 0.8969 ± 0.0195 | 1.3120 ± 0.0567 | 1.8226 ± 0.1400 † | 0.4387 ± 0.0330 | 0.7631 ± 0.0684 | 32.1940 |
| `tr_DCP+KumoTabular [clean]` | torchregress | 0.9131 ± 0.0188 | 0.9236 ± 0.0542 | 1.1904 ± 0.0916 † | 0.3297 ± 0.0288 | 0.8672 ± 0.0486 | 2.7079 |
| `tr_DCP+LightGBM [clean]` | torchregress | 0.9044 ± 0.0046 | 1.8968 ± 0.1206 | 2.2244 ± 0.1458 † | 0.4265 ± 0.0256 | 0.6782 ± 0.0041 | 2.1524 |
| `tr_DCP+TabICL [clean]` | torchregress | 0.9069 ± 0.0168 | 1.0179 ± 0.0262 | 1.3621 ± 0.1036 † | 0.3618 ± 0.0273 | 0.8467 ± 0.0473 | 2.1491 |
| `tr_DCP+TabularMLP [clean]` | torchregress | 0.8972 ± 0.0192 | 1.3001 ± 0.0654 | 1.8221 ± 0.1404 † | 0.4387 ± 0.0330 | 0.7714 ± 0.0672 | 32.1940 |
| `tr_Mondrian+KumoTabular [clean]` | torchregress | 0.8992 ± 0.0169 | 0.9727 ± 0.0512 | 1.5671 ± 0.1302 † | 0.3297 ± 0.0288 | 0.8426 ± 0.0233 | 2.7079 |
| `tr_Mondrian+LightGBM [clean]` | torchregress | 0.8981 ± 0.0080 | 1.3309 ± 0.0270 | 1.9758 ± 0.1222 † | 0.4265 ± 0.0256 | 0.7750 ± 0.0232 | 2.1524 |
| `tr_Mondrian+TabFM [clean]` | torchregress | 0.9047 ± 0.0240 | 0.9979 ± 0.0459 | 1.5577 ± 0.1523 † | **0.3291 ± 0.0346** | 0.8262 ± 0.0299 | 9.0642 |
| `tr_Mondrian+TabICL [clean]` | torchregress | 0.8989 ± 0.0178 | 1.0728 ± 0.0668 | 1.7316 ± 0.1356 † | 0.3618 ± 0.0273 | 0.8313 ± 0.0372 | 2.1491 |
| `tr_Mondrian+TabularMLP [clean]` | torchregress | 0.9069 ± 0.0140 | 1.4062 ± 0.0099 | 2.0335 ± 0.1740 † | 0.4387 ± 0.0330 | 0.7858 ± 0.0130 | 32.1940 |
| `tr_Normalized+KumoTabular [clean]` | torchregress | 0.9050 ± 0.0267 | 0.9173 ± 0.0728 | 1.2723 ± 0.1033 † | 0.3297 ± 0.0288 | 0.8509 ± 0.0372 | 2.7079 |
| `tr_Normalized+LightGBM [clean]` | torchregress | 0.9017 ± 0.0036 | 1.2979 ± 0.0304 | 1.8860 ± 0.1214 † | 0.4265 ± 0.0256 | 0.7890 ± 0.0189 | 2.1524 |
| `tr_Normalized+TabICL [clean]` | torchregress | 0.9017 ± 0.0123 | 0.9937 ± 0.0432 | 1.4517 ± 0.1314 † | 0.3618 ± 0.0273 | 0.8124 ± 0.0206 | 2.1491 |
| `tr_Normalized+TabularMLP [clean]` | torchregress | 0.8972 ± 0.0192 | 1.3001 ± 0.0654 | 1.8221 ± 0.1404 † | 0.4387 ± 0.0330 | 0.7714 ± 0.0672 | 32.1940 |
| `tr_PITRecal+KumoTabular [clean]` | torchregress | 0.9119 ± 0.0191 | 0.9149 ± 0.0586 | 1.1907 ± 0.0886 † | 0.3310 ± 0.0279 | 0.8595 ± 0.0386 | 2.7079 |
| `tr_PITRecal+LightGBM [clean]` | torchregress | 0.9044 ± 0.0075 | 1.8060 ± 0.0467 | 2.1809 ± 0.0954 | 0.4267 ± 0.0252 | 0.6717 ± 0.0572 | 2.1524 |
| `tr_PITRecal+TabICL [clean]` | torchregress | 0.9042 ± 0.0172 | 1.0137 ± 0.0248 | 1.3607 ± 0.1080 † | 0.3622 ± 0.0270 | 0.8417 ± 0.0548 | 2.1491 |
| `tr_PITRecal+TabularMLP [clean]` | torchregress | 0.9025 ± 0.0176 | 1.3284 ± 0.0443 | 1.8016 ± 0.1275 † | 0.4414 ± 0.0316 | 0.8089 ± 0.0661 | 32.1940 |
| `tr_ResidualDist+KumoTabular [clean]` | torchregress | 0.8972 ± 0.0217 | 0.9869 ± 0.0523 | 1.5704 ± 0.1591 † | 0.3298 ± 0.0287 | 0.8120 ± 0.0374 | 2.7079 |
| `tr_ResidualDist+LightGBM [clean]` | torchregress | 0.9003 ± 0.0013 | 1.3158 ± 0.0319 | 1.9587 ± 0.1434 † | 0.4271 ± 0.0255 | 0.7889 ± 0.0189 | 2.1524 |
| `tr_ResidualDist+TabFM [clean]` | torchregress | 0.9042 ± 0.0175 | 0.9943 ± 0.0446 | 1.5532 ± 0.1736 † | 0.3294 ± 0.0344 | 0.8284 ± 0.0314 | 9.0642 |
| `tr_ResidualDist+TabICL [clean]` | torchregress | 0.8986 ± 0.0158 | 1.0677 ± 0.0346 | 1.7219 ± 0.1524 † | 0.3619 ± 0.0273 | 0.8121 ± 0.0533 | 2.1491 |
| `tr_ResidualDist+TabularMLP [clean]` | torchregress | 0.9106 ± 0.0079 | 1.4056 ± 0.0210 | 2.0155 ± 0.1731 † | 0.4403 ± 0.0324 | 0.7919 ± 0.0298 | 32.1940 |
| `tr_Split+KumoTabular [clean]` | torchregress | 0.8944 ± 0.0193 | 0.9795 ± 0.0470 | 1.5844 ± 0.1524 † | 0.3297 ± 0.0288 | 0.7899 ± 0.0362 | 2.7079 |
| `tr_Split+LightGBM [clean]` | torchregress | 0.8994 ± 0.0090 | 1.3379 ± 0.0394 | 1.9840 ± 0.1300 † | 0.4265 ± 0.0256 | 0.7469 ± 0.0169 | 2.1524 |
| `tr_Split+TabFM [clean]` | torchregress | 0.8992 ± 0.0154 | 0.9826 ± 0.0462 | 1.5578 ± 0.1711 † | **0.3291 ± 0.0346** | 0.8064 ± 0.0109 | 9.0642 |
| `tr_Split+TabICL [clean]` | torchregress | 0.8903 ± 0.0133 | 1.0571 ± 0.0452 | 1.7410 ± 0.1461 † | 0.3618 ± 0.0273 | 0.7994 ± 0.0450 | 2.1491 |
| `tr_Split+TabularMLP [clean]` | torchregress | 0.9083 ± 0.0106 | 1.4087 ± 0.0222 | 2.0332 ± 0.1674 † | 0.4387 ± 0.0330 | 0.7516 ± 0.0375 | 32.1940 |

##### fm_uq_scenarios / openml_year@extrapolate

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [extrapolate]` | sdm | 0.9369 ± 0.0121 | 1.6982 ± 0.0885 | **1.8862 ± 0.1474** † | 0.4622 ± 0.0358 | 0.8653 ± 0.0521 | 2.2333 |
| `Raw+LightGBM [extrapolate]` | lightgbm | 0.7458 ± 0.0697 | 1.6109 ± 0.1353 | 2.8495 ± 0.5782 † | 0.5481 ± 0.0814 | 0.3375 ± 0.1644 | 1.9986 |
| `Raw+TabICL [extrapolate]` | tabicl | 0.9450 ± 0.0197 | 2.0225 ± 0.0323 | 2.2016 ± 0.1260 † | 0.5078 ± 0.0862 | 0.8571 ± 0.0550 | 1.9726 |
| `Raw+TabularMLP [extrapolate]` | torchregress | 0.8544 ± 0.0778 | 1.7553 ± 0.0868 | 2.6662 ± 0.7955 † | 0.6258 ± 0.1908 | 0.6936 ± 0.2248 | 33.2413 |
| `tr_CQR+KumoTabular [extrapolate]` | torchregress | 0.9414 ± 0.0156 | 1.7157 ± 0.1015 | 1.8958 ± 0.1458 † | 0.4622 ± 0.0358 | 0.8747 ± 0.0461 | 2.2333 |
| `tr_CQR+LightGBM [extrapolate]` | torchregress | 0.8139 ± 0.0689 | 1.7565 ± 0.0902 | 2.6819 ± 0.5067 † | 0.5481 ± 0.0814 | 0.4344 ± 0.1938 | 1.9986 |
| `tr_CQR+TabICL [extrapolate]` | torchregress | 0.9439 ± 0.0222 | 2.0247 ± 0.0199 | 2.2038 ± 0.1225 † | 0.5078 ± 0.0862 | 0.8532 ± 0.0619 | 1.9726 |
| `tr_CQR+TabularMLP [extrapolate]` | torchregress | 0.8456 ± 0.0643 | 1.7184 ± 0.1418 | 2.6556 ± 0.7716 † | 0.6258 ± 0.1908 | 0.6847 ± 0.2092 | 33.2413 |
| `tr_DCP+KumoTabular [extrapolate]` | torchregress | 0.9403 ± 0.0178 | 1.7381 ± 0.1479 | 1.9165 ± 0.1568 † | 0.4622 ± 0.0358 | 0.8677 ± 0.0443 | 2.2333 |
| `tr_DCP+LightGBM [extrapolate]` | torchregress | 0.8186 ± 0.0697 | 2.0675 ± 0.1667 | 2.8913 ± 0.4765 † | 0.5481 ± 0.0814 | 0.4095 ± 0.1792 | 1.9986 |
| `tr_DCP+TabICL [extrapolate]` | torchregress | 0.9472 ± 0.0215 | 2.0344 ± 0.0541 | 2.2107 ± 0.1213 † | 0.5078 ± 0.0862 | 0.8617 ± 0.0595 | 1.9726 |
| `tr_DCP+TabularMLP [extrapolate]` | torchregress | 0.8464 ± 0.0621 | 1.7060 ± 0.1696 | 2.6344 ± 0.7624 † | 0.6258 ± 0.1908 | 0.6878 ± 0.2018 | 33.2413 |
| `tr_Mondrian+KumoTabular [extrapolate]` | torchregress | 0.7692 ± 0.0529 | 1.0085 ± 0.0477 | 2.4335 ± 0.3272 † | 0.4622 ± 0.0358 | 0.5346 ± 0.1056 | 2.2333 |
| `tr_Mondrian+LightGBM [extrapolate]` | torchregress | 0.8022 ± 0.0658 | 1.3685 ± 0.0787 | 2.6771 ± 0.5951 † | 0.5481 ± 0.0814 | 0.4881 ± 0.1557 | 1.9986 |
| `tr_Mondrian+TabFM [extrapolate]` | torchregress | 0.7550 ± 0.0972 | 1.0027 ± 0.0892 | 2.6496 ± 0.7801 † | 0.4869 ± 0.0708 | 0.4135 ± 0.1141 | 9.1394 |
| `tr_Mondrian+TabICL [extrapolate]` | torchregress | 0.7533 ± 0.0883 | 1.0708 ± 0.0850 | 2.7820 ± 0.9531 † | 0.5078 ± 0.0862 | 0.3674 ± 0.0368 | 1.9726 |
| `tr_Mondrian+TabularMLP [extrapolate]` | torchregress | 0.7803 ± 0.1016 | 1.4001 ± 0.0945 | 3.3780 ± 1.6892 † | 0.6258 ± 0.1908 | 0.4033 ± 0.1637 | 33.2413 |
| `tr_Normalized+KumoTabular [extrapolate]` | torchregress | 0.9356 ± 0.0158 | 1.7462 ± 0.1100 | 2.0232 ± 0.0550 | 0.4622 ± 0.0358 | 0.8667 ± 0.0518 | 2.2333 |
| `tr_Normalized+LightGBM [extrapolate]` | torchregress | 0.8358 ± 0.0327 | 1.4510 ± 0.1431 | 2.4287 ± 0.3947 † | 0.5481 ± 0.0814 | 0.6870 ± 0.1482 | 1.9986 |
| `tr_Normalized+TabICL [extrapolate]` | torchregress | 0.9572 ± 0.0106 | 2.0386 ± 0.0473 | 2.2493 ± 0.0293 | 0.5078 ± 0.0862 | 0.9181 ± 0.0134 | 1.9726 |
| `tr_Normalized+TabularMLP [extrapolate]` | torchregress | 0.8464 ± 0.0621 | 1.7060 ± 0.1696 | 2.6344 ± 0.7624 † | 0.6258 ± 0.1908 | 0.6878 ± 0.2018 | 33.2413 |
| `tr_PITRecal+KumoTabular [extrapolate]` | torchregress | 0.9397 ± 0.0206 | 1.7409 ± 0.1455 | 1.9227 ± 0.1469 † | **0.4601 ± 0.0323** | 0.8700 ± 0.0457 | 2.2333 |
| `tr_PITRecal+LightGBM [extrapolate]` | torchregress | 0.8281 ± 0.0484 | 2.0091 ± 0.0962 | 2.7040 ± 0.2833 † | 0.5497 ± 0.0826 | 0.4954 ± 0.1643 | 1.9986 |
| `tr_PITRecal+TabICL [extrapolate]` | torchregress | 0.9447 ± 0.0260 | 2.0418 ± 0.0563 | 2.2205 ± 0.1351 † | 0.5060 ± 0.0847 | 0.8597 ± 0.0813 | 1.9726 |
| `tr_PITRecal+TabularMLP [extrapolate]` | torchregress | 0.8408 ± 0.0754 | 1.7183 ± 0.1538 | 2.7528 ± 0.9756 † | 0.6237 ± 0.1899 | 0.6653 ± 0.2327 | 33.2413 |
| `tr_ResidualDist+KumoTabular [extrapolate]` | torchregress | 0.7625 ± 0.0479 | 0.9842 ± 0.0325 | 2.4514 ± 0.3438 † | 0.4622 ± 0.0356 | 0.6298 ± 0.1080 | 2.2333 |
| `tr_ResidualDist+LightGBM [extrapolate]` | torchregress | 0.7942 ± 0.0884 | 1.3094 ± 0.0323 | 2.7532 ± 0.8358 † | 0.5481 ± 0.0812 | 0.5610 ± 0.1882 | 1.9986 |
| `tr_ResidualDist+TabFM [extrapolate]` | torchregress | 0.7478 ± 0.0830 | 0.9733 ± 0.0421 | 2.6367 ± 0.7003 † | 0.4835 ± 0.0695 | 0.4907 ± 0.1202 | 9.1394 |
| `tr_ResidualDist+TabICL [extrapolate]` | torchregress | 0.7403 ± 0.0881 | 1.0417 ± 0.0588 | 2.8502 ± 0.9561 † | 0.5076 ± 0.0862 | 0.5102 ± 0.1413 | 1.9726 |
| `tr_ResidualDist+TabularMLP [extrapolate]` | torchregress | 0.7833 ± 0.0966 | 1.3688 ± 0.0482 | 3.4527 ± 1.8525 † | 0.6268 ± 0.1888 | 0.5203 ± 0.2046 | 33.2413 |
| `tr_Split+KumoTabular [extrapolate]` | torchregress | 0.7694 ± 0.0554 | 0.9725 ± 0.0245 | 2.4317 ± 0.3723 † | 0.4622 ± 0.0358 | 0.6457 ± 0.1128 | 2.2333 |
| `tr_Split+LightGBM [extrapolate]` | torchregress | 0.8022 ± 0.0722 | 1.3209 ± 0.0473 | 2.6700 ± 0.6618 † | 0.5481 ± 0.0814 | 0.6067 ± 0.1421 | 1.9986 |
| `tr_Split+TabFM [extrapolate]` | torchregress | 0.7500 ± 0.0893 | 0.9596 ± 0.0342 | 2.6438 ± 0.7860 † | 0.4869 ± 0.0708 | 0.5214 ± 0.1659 | 9.1394 |
| `tr_Split+TabICL [extrapolate]` | torchregress | 0.7528 ± 0.1007 | 1.0427 ± 0.0663 | 2.7674 ± 0.9780 † | 0.5078 ± 0.0862 | 0.5537 ± 0.1705 | 1.9726 |
| `tr_Split+TabularMLP [extrapolate]` | torchregress | 0.7922 ± 0.0972 | 1.3885 ± 0.0304 | 3.3381 ± 1.7132 † | 0.6258 ± 0.1908 | 0.5411 ± 0.1858 | 33.2413 |

##### fm_uq_scenarios / openml_year@heavy_tail

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [heavy_tail]` | sdm | 0.9119 ± 0.0067 | 0.9243 ± 0.0151 | **1.2034 ± 0.1415** † | 0.4081 ± 0.0418 | 0.8625 ± 0.0226 | 2.2104 |
| `Raw+LightGBM [heavy_tail]` | lightgbm | 0.8611 ± 0.0039 | 1.4121 ± 0.0301 | 2.1706 ± 0.1839 † | 0.5154 ± 0.0301 | 0.4680 ± 0.0164 | 1.9519 |
| `Raw+TabICL [heavy_tail]` | tabicl | 0.9011 ± 0.0124 | 1.0257 ± 0.0108 | 1.4189 ± 0.1814 † | 0.4399 ± 0.0354 | 0.8196 ± 0.0421 | 1.9801 |
| `Raw+TabularMLP [heavy_tail]` | torchregress | 0.9319 ± 0.0180 | 1.5313 ± 0.0219 | 2.0901 ± 0.1964 † | 0.5195 ± 0.0532 | 0.7554 ± 0.0614 | 28.6843 |
| `tr_CQR+KumoTabular [heavy_tail]` | torchregress | 0.9092 ± 0.0159 | 0.9220 ± 0.0339 | 1.2043 ± 0.1421 † | 0.4081 ± 0.0418 | 0.8620 ± 0.0311 | 2.2104 |
| `tr_CQR+LightGBM [heavy_tail]` | torchregress | 0.9072 ± 0.0098 | 1.4897 ± 0.0300 | 2.1682 ± 0.1783 † | 0.5154 ± 0.0301 | 0.6874 ± 0.0253 | 1.9519 |
| `tr_CQR+TabICL [heavy_tail]` | torchregress | 0.9022 ± 0.0159 | 1.0289 ± 0.0244 | 1.4197 ± 0.1806 † | 0.4399 ± 0.0354 | 0.8136 ± 0.0409 | 1.9801 |
| `tr_CQR+TabularMLP [heavy_tail]` | torchregress | 0.9022 ± 0.0087 | 1.3965 ± 0.0413 | 2.0649 ± 0.2173 † | 0.5195 ± 0.0532 | 0.7416 ± 0.0570 | 28.6843 |
| `tr_DCP+KumoTabular [heavy_tail]` | torchregress | 0.9100 ± 0.0173 | 0.9183 ± 0.0577 | 1.2041 ± 0.1456 † | 0.4081 ± 0.0418 | 0.8679 ± 0.0317 | 2.2104 |
| `tr_DCP+LightGBM [heavy_tail]` | torchregress | 0.9061 ± 0.0010 | 3.2179 ± 0.5013 | 3.5058 ± 0.4952 † | 0.5154 ± 0.0301 | 0.5870 ± 0.0329 | 1.9519 |
| `tr_DCP+TabICL [heavy_tail]` | torchregress | 0.9022 ± 0.0151 | 1.0357 ± 0.0481 | 1.4204 ± 0.1787 † | 0.4399 ± 0.0354 | 0.8165 ± 0.0416 | 1.9801 |
| `tr_DCP+TabularMLP [heavy_tail]` | torchregress | 0.9031 ± 0.0071 | 1.3404 ± 0.0628 | 2.0528 ± 0.2210 † | 0.5195 ± 0.0532 | 0.7166 ± 0.0457 | 28.6843 |
| `tr_Mondrian+KumoTabular [heavy_tail]` | torchregress | 0.9047 ± 0.0099 | 1.0389 ± 0.0602 | 1.7838 ± 0.2016 † | 0.4081 ± 0.0418 | 0.8001 ± 0.0277 | 2.2104 |
| `tr_Mondrian+LightGBM [heavy_tail]` | torchregress | 0.8961 ± 0.0038 | 1.3326 ± 0.0607 | 2.1930 ± 0.2129 † | 0.5154 ± 0.0301 | 0.6875 ± 0.0133 | 1.9519 |
| `tr_Mondrian+TabFM [heavy_tail]` | torchregress | 0.9069 ± 0.0061 | 1.0286 ± 0.0741 | 1.6839 ± 0.1900 † | **0.3961 ± 0.0483** | 0.8127 ± 0.0136 | 9.1768 |
| `tr_Mondrian+TabICL [heavy_tail]` | torchregress | 0.9050 ± 0.0104 | 1.1207 ± 0.0110 | 1.9257 ± 0.1811 † | 0.4399 ± 0.0354 | 0.7651 ± 0.0351 | 1.9801 |
| `tr_Mondrian+TabularMLP [heavy_tail]` | torchregress | 0.9111 ± 0.0126 | 1.3787 ± 0.0473 | 2.1765 ± 0.2768 † | 0.5195 ± 0.0532 | 0.7136 ± 0.0393 | 28.6843 |
| `tr_Normalized+KumoTabular [heavy_tail]` | torchregress | 0.9094 ± 0.0059 | 0.9457 ± 0.0389 | 1.4302 ± 0.1624 † | 0.4081 ± 0.0418 | 0.7485 ± 0.0286 | 2.2104 |
| `tr_Normalized+LightGBM [heavy_tail]` | torchregress | 0.8992 ± 0.0126 | 1.1912 ± 0.0722 | 2.2033 ± 0.1480 † | 0.5154 ± 0.0301 | 0.5640 ± 0.0340 | 1.9519 |
| `tr_Normalized+TabICL [heavy_tail]` | torchregress | 0.9067 ± 0.0008 | 1.0491 ± 0.0145 | 1.6329 ± 0.1716 † | 0.4399 ± 0.0354 | 0.7361 ± 0.0248 | 1.9801 |
| `tr_Normalized+TabularMLP [heavy_tail]` | torchregress | 0.9031 ± 0.0071 | 1.3404 ± 0.0628 | 2.0528 ± 0.2210 † | 0.5195 ± 0.0532 | 0.7166 ± 0.0457 | 28.6843 |
| `tr_PITRecal+KumoTabular [heavy_tail]` | torchregress | 0.9078 ± 0.0178 | 0.9140 ± 0.0674 | 1.2047 ± 0.1461 † | 0.4094 ± 0.0417 | 0.8628 ± 0.0421 | 2.2104 |
| `tr_PITRecal+LightGBM [heavy_tail]` | torchregress | 0.9053 ± 0.0038 | 2.0453 ± 0.1614 | 2.5324 ± 0.1368 † | 0.5160 ± 0.0297 | 0.5207 ± 0.0398 | 1.9519 |
| `tr_PITRecal+TabICL [heavy_tail]` | torchregress | 0.9022 ± 0.0140 | 1.0271 ± 0.0351 | 1.4232 ± 0.1843 † | 0.4403 ± 0.0353 | 0.8092 ± 0.0602 | 1.9801 |
| `tr_PITRecal+TabularMLP [heavy_tail]` | torchregress | 0.9022 ± 0.0099 | 1.3446 ± 0.0484 | 2.0446 ± 0.2119 † | 0.5254 ± 0.0461 | 0.7220 ± 0.0445 | 28.6843 |
| `tr_ResidualDist+KumoTabular [heavy_tail]` | torchregress | 0.9083 ± 0.0066 | 1.0456 ± 0.0385 | 2.0721 ± 0.2017 † | 0.4082 ± 0.0418 | 0.6083 ± 0.0160 | 2.2104 |
| `tr_ResidualDist+LightGBM [heavy_tail]` | torchregress | 0.9033 ± 0.0038 | 1.4714 ± 0.1020 | 2.6332 ± 0.1429 † | 0.5157 ± 0.0300 | 0.5067 ± 0.0324 | 1.9519 |
| `tr_ResidualDist+TabFM [heavy_tail]` | torchregress | 0.9086 ± 0.0079 | 1.1178 ± 0.0441 | 2.0143 ± 0.2231 † | 0.3964 ± 0.0484 | 0.6152 ± 0.0252 | 9.1768 |
| `tr_ResidualDist+TabICL [heavy_tail]` | torchregress | 0.9094 ± 0.0076 | 1.1780 ± 0.0480 | 2.2528 ± 0.1753 † | 0.4400 ± 0.0354 | 0.6032 ± 0.0172 | 1.9801 |
| `tr_ResidualDist+TabularMLP [heavy_tail]` | torchregress | 0.9064 ± 0.0100 | 1.5355 ± 0.0617 | 2.5871 ± 0.2818 † | 0.5204 ± 0.0532 | 0.5334 ± 0.0202 | 28.6843 |
| `tr_Split+KumoTabular [heavy_tail]` | torchregress | 0.9092 ± 0.0071 | 1.0264 ± 0.0475 | 2.0788 ± 0.2089 † | 0.4081 ± 0.0418 | 0.6025 ± 0.0255 | 2.2104 |
| `tr_Split+LightGBM [heavy_tail]` | torchregress | 0.8989 ± 0.0029 | 1.3725 ± 0.1153 | 2.6710 ± 0.1475 † | 0.5154 ± 0.0301 | 0.4666 ± 0.0202 | 1.9519 |
| `tr_Split+TabFM [heavy_tail]` | torchregress | 0.9086 ± 0.0084 | 1.1145 ± 0.0740 | 2.0106 ± 0.2259 † | **0.3961 ± 0.0483** | 0.6092 ± 0.0120 | 9.1768 |
| `tr_Split+TabICL [heavy_tail]` | torchregress | 0.9061 ± 0.0069 | 1.1494 ± 0.0540 | 2.2594 ± 0.1772 † | 0.4399 ± 0.0354 | 0.5787 ± 0.0315 | 1.9801 |
| `tr_Split+TabularMLP [heavy_tail]` | torchregress | 0.9058 ± 0.0083 | 1.5147 ± 0.0660 | 2.5897 ± 0.2894 † | 0.5195 ± 0.0532 | 0.5280 ± 0.0184 | 28.6843 |

##### fm_uq_scenarios / openml_year@imbalance

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [imbalance]` | sdm | 0.8719 ± 0.0137 | 0.8685 ± 0.0149 | 1.4209 ± 0.0334 | **0.3735 ± 0.0071** | 0.7439 ± 0.0445 | 1.8962 |
| `Raw+LightGBM [imbalance]` | lightgbm | 0.7439 ± 0.0161 | 1.3385 ± 0.0526 | 2.8961 ± 0.0968 | 0.4903 ± 0.0061 | 0.3367 ± 0.0737 | 2.0896 |
| `Raw+TabICL [imbalance]` | tabicl | 0.8717 ± 0.0104 | 0.9455 ± 0.0064 | 1.5968 ± 0.0351 | 0.3954 ± 0.0105 | 0.7579 ± 0.0441 | 1.7485 |
| `Raw+TabularMLP [imbalance]` | torchregress | 0.8611 ± 0.0106 | 1.2791 ± 0.0761 | 2.1023 ± 0.0529 | 0.4807 ± 0.0192 | 0.6635 ± 0.0243 | 24.3161 |
| `tr_CQR+KumoTabular [imbalance]` | torchregress | 0.8783 ± 0.0172 | 0.8853 ± 0.0361 | 1.4176 ± 0.0412 | **0.3735 ± 0.0071** | 0.7481 ± 0.0430 | 1.8962 |
| `tr_CQR+LightGBM [imbalance]` | torchregress | 0.8178 ± 0.0038 | 1.4989 ± 0.0289 | 2.7146 ± 0.0688 | 0.4903 ± 0.0061 | 0.4781 ± 0.0308 | 2.0896 |
| `tr_CQR+TabICL [imbalance]` | torchregress | 0.8717 ± 0.0044 | 0.9508 ± 0.0265 | 1.5944 ± 0.0329 | 0.3954 ± 0.0105 | 0.7588 ± 0.0291 | 1.7485 |
| `tr_CQR+TabularMLP [imbalance]` | torchregress | 0.8433 ± 0.0147 | 1.2257 ± 0.0634 | 2.1287 ± 0.0466 | 0.4807 ± 0.0192 | 0.6340 ± 0.0339 | 24.3161 |
| `tr_DCP+KumoTabular [imbalance]` | torchregress | 0.8814 ± 0.0200 | 0.8898 ± 0.0493 | **1.4136 ± 0.0535** | **0.3735 ± 0.0071** | 0.7576 ± 0.0463 | 1.8962 |
| `tr_DCP+LightGBM [imbalance]` | torchregress | 0.8103 ± 0.0042 | 1.7300 ± 0.0591 | 2.8953 ± 0.0549 | 0.4903 ± 0.0061 | 0.4477 ± 0.0182 | 2.0896 |
| `tr_DCP+TabICL [imbalance]` | torchregress | 0.8731 ± 0.0091 | 0.9591 ± 0.0475 | 1.5889 ± 0.0358 | 0.3954 ± 0.0105 | 0.7620 ± 0.0217 | 1.7485 |
| `tr_DCP+TabularMLP [imbalance]` | torchregress | 0.8408 ± 0.0172 | 1.2073 ± 0.0692 | 2.1393 ± 0.0476 | 0.4807 ± 0.0192 | 0.6296 ± 0.0358 | 24.3161 |
| `tr_Mondrian+KumoTabular [imbalance]` | torchregress | 0.8475 ± 0.0145 | 0.9004 ± 0.0418 | 1.8791 ± 0.0254 | **0.3735 ± 0.0071** | 0.6703 ± 0.0557 | 1.8962 |
| `tr_Mondrian+LightGBM [imbalance]` | torchregress | 0.8297 ± 0.0180 | 1.1894 ± 0.0480 | 2.4776 ± 0.0703 | 0.4903 ± 0.0061 | 0.5655 ± 0.0829 | 2.0896 |
| `tr_Mondrian+TabFM [imbalance]` | torchregress | 0.8436 ± 0.0134 | 0.8975 ± 0.0437 | 1.8941 ± 0.0740 | 0.3765 ± 0.0170 | 0.6551 ± 0.0534 | 8.2543 |
| `tr_Mondrian+TabICL [imbalance]` | torchregress | 0.8467 ± 0.0058 | 0.9534 ± 0.0151 | 1.9957 ± 0.0500 | 0.3954 ± 0.0105 | 0.6779 ± 0.0423 | 1.7485 |
| `tr_Mondrian+TabularMLP [imbalance]` | torchregress | 0.8608 ± 0.0164 | 1.2914 ± 0.0471 | 2.2767 ± 0.0956 | 0.4807 ± 0.0192 | 0.6607 ± 0.0278 | 24.3161 |
| `tr_Normalized+KumoTabular [imbalance]` | torchregress | 0.8681 ± 0.0117 | 0.8890 ± 0.0500 | 1.5213 ± 0.0679 | **0.3735 ± 0.0071** | 0.7165 ± 0.0321 | 1.8962 |
| `tr_Normalized+LightGBM [imbalance]` | torchregress | 0.8489 ± 0.0079 | 1.2299 ± 0.0327 | 2.2552 ± 0.0577 | 0.4903 ± 0.0061 | 0.6457 ± 0.0033 | 2.0896 |
| `tr_Normalized+TabICL [imbalance]` | torchregress | 0.8631 ± 0.0189 | 0.9483 ± 0.0699 | 1.6660 ± 0.0594 | 0.3954 ± 0.0105 | 0.7347 ± 0.0160 | 1.7485 |
| `tr_Normalized+TabularMLP [imbalance]` | torchregress | 0.8408 ± 0.0172 | 1.2073 ± 0.0692 | 2.1393 ± 0.0476 | 0.4807 ± 0.0192 | 0.6296 ± 0.0358 | 24.3161 |
| `tr_PITRecal+KumoTabular [imbalance]` | torchregress | 0.8817 ± 0.0184 | 0.8900 ± 0.0483 | 1.4299 ± 0.0637 | 0.3740 ± 0.0090 | 0.7554 ± 0.0444 | 1.8962 |
| `tr_PITRecal+LightGBM [imbalance]` | torchregress | 0.8075 ± 0.0058 | 1.7236 ± 0.0541 | 2.9312 ± 0.0319 | 0.4931 ± 0.0050 | 0.4310 ± 0.0139 | 2.0896 |
| `tr_PITRecal+TabICL [imbalance]` | torchregress | 0.8708 ± 0.0068 | 0.9486 ± 0.0416 | 1.6130 ± 0.0624 | 0.3979 ± 0.0120 | 0.7455 ± 0.0118 | 1.7485 |
| `tr_PITRecal+TabularMLP [imbalance]` | torchregress | 0.8497 ± 0.0096 | 1.2191 ± 0.0641 | 2.0878 ± 0.0951 | 0.4909 ± 0.0155 | 0.6575 ± 0.0085 | 24.3161 |
| `tr_ResidualDist+KumoTabular [imbalance]` | torchregress | 0.8458 ± 0.0043 | 0.8762 ± 0.0270 | 1.8591 ± 0.0704 | 0.3738 ± 0.0081 | 0.6744 ± 0.0263 | 1.8962 |
| `tr_ResidualDist+LightGBM [imbalance]` | torchregress | 0.8317 ± 0.0118 | 1.1680 ± 0.0019 | 2.4074 ± 0.0678 | 0.4920 ± 0.0059 | 0.5778 ± 0.0303 | 2.0896 |
| `tr_ResidualDist+TabFM [imbalance]` | torchregress | 0.8514 ± 0.0092 | 0.8926 ± 0.0333 | 1.8487 ± 0.1396 † | 0.3796 ± 0.0180 | 0.6849 ± 0.0141 | 8.2543 |
| `tr_ResidualDist+TabICL [imbalance]` | torchregress | 0.8561 ± 0.0094 | 0.9519 ± 0.0329 | 1.9384 ± 0.1096 † | 0.3966 ± 0.0115 | 0.7052 ± 0.0097 | 1.7485 |
| `tr_ResidualDist+TabularMLP [imbalance]` | torchregress | 0.8486 ± 0.0111 | 1.2496 ± 0.0314 | 2.3194 ± 0.1361 † | 0.4878 ± 0.0171 | 0.6325 ± 0.0134 | 24.3161 |
| `tr_Split+KumoTabular [imbalance]` | torchregress | 0.8361 ± 0.0118 | 0.8652 ± 0.0353 | 1.9059 ± 0.0374 | **0.3735 ± 0.0071** | 0.6487 ± 0.0479 | 1.8962 |
| `tr_Split+LightGBM [imbalance]` | torchregress | 0.8228 ± 0.0138 | 1.1594 ± 0.0056 | 2.5037 ± 0.0325 | 0.4903 ± 0.0061 | 0.5519 ± 0.0416 | 2.0896 |
| `tr_Split+TabFM [imbalance]` | torchregress | 0.8467 ± 0.0044 | 0.8848 ± 0.0284 | 1.8989 ± 0.1016 † | 0.3765 ± 0.0170 | 0.6679 ± 0.0278 | 8.2543 |
| `tr_Split+TabICL [imbalance]` | torchregress | 0.8508 ± 0.0090 | 0.9540 ± 0.0494 | 1.9897 ± 0.0720 | 0.3954 ± 0.0105 | 0.6870 ± 0.0068 | 1.7485 |
| `tr_Split+TabularMLP [imbalance]` | torchregress | 0.8464 ± 0.0103 | 1.2356 ± 0.0358 | 2.3304 ± 0.1022 | 0.4807 ± 0.0192 | 0.6250 ± 0.0105 | 24.3161 |

##### fm_uq_scenarios / openml_year@noise_all

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_all]` | sdm | 0.9036 ± 0.0071 | 2.3474 ± 0.0251 | 2.8805 ± 0.0668 | 0.7192 ± 0.0131 | 0.6802 ± 0.0421 | 2.2407 |
| `Raw+LightGBM [noise_all]` | lightgbm | 0.8533 ± 0.0096 | 2.2806 ± 0.0195 | 3.0353 ± 0.0969 | 0.7306 ± 0.0113 | 0.4485 ± 0.0246 | 1.9706 |
| `Raw+TabICL [noise_all]` | tabicl | 0.9058 ± 0.0076 | 2.3869 ± 0.0129 | 2.9176 ± 0.0704 | 0.7248 ± 0.0147 | 0.6839 ± 0.0325 | 1.9525 |
| `Raw+TabularMLP [noise_all]` | torchregress | 0.9136 ± 0.0086 | 2.4595 ± 0.0656 | 3.0278 ± 0.0925 | 0.7304 ± 0.0180 | 0.6697 ± 0.0273 | 19.2777 |
| `tr_CQR+KumoTabular [noise_all]` | torchregress | 0.9042 ± 0.0074 | 2.3520 ± 0.0427 | 2.8801 ± 0.0676 | 0.7192 ± 0.0131 | 0.6827 ± 0.0208 | 2.2407 |
| `tr_CQR+LightGBM [noise_all]` | torchregress | 0.9067 ± 0.0073 | 2.5065 ± 0.0740 | 3.0010 ± 0.0834 | 0.7306 ± 0.0113 | 0.6253 ± 0.0273 | 1.9706 |
| `tr_CQR+TabICL [noise_all]` | torchregress | 0.9025 ± 0.0052 | 2.3798 ± 0.0297 | 2.9168 ± 0.0713 | 0.7248 ± 0.0147 | 0.6700 ± 0.0174 | 1.9525 |
| `tr_CQR+TabularMLP [noise_all]` | torchregress | 0.9053 ± 0.0089 | 2.3929 ± 0.0587 | 3.0210 ± 0.0951 | 0.7304 ± 0.0180 | 0.6472 ± 0.0233 | 19.2777 |
| `tr_DCP+KumoTabular [noise_all]` | torchregress | 0.9058 ± 0.0058 | 2.3603 ± 0.0634 | **2.8797 ± 0.0666** | 0.7192 ± 0.0131 | 0.6877 ± 0.0290 | 2.2407 |
| `tr_DCP+LightGBM [noise_all]` | torchregress | 0.9008 ± 0.0094 | 2.7636 ± 0.1881 | 3.1431 ± 0.1405 | 0.7306 ± 0.0113 | 0.5357 ± 0.0724 | 1.9706 |
| `tr_DCP+TabICL [noise_all]` | torchregress | 0.9050 ± 0.0058 | 2.3677 ± 0.0503 | 2.9157 ± 0.0713 | 0.7248 ± 0.0147 | 0.6864 ± 0.0285 | 1.9525 |
| `tr_DCP+TabularMLP [noise_all]` | torchregress | 0.9044 ± 0.0097 | 2.3870 ± 0.0663 | 3.0196 ± 0.0962 | 0.7304 ± 0.0180 | 0.6417 ± 0.0150 | 19.2777 |
| `tr_Mondrian+KumoTabular [noise_all]` | torchregress | 0.9069 ± 0.0088 | 2.4216 ± 0.0514 | 3.1466 ± 0.0939 | 0.7192 ± 0.0131 | 0.6393 ± 0.0184 | 2.2407 |
| `tr_Mondrian+LightGBM [noise_all]` | torchregress | 0.9061 ± 0.0087 | 2.4417 ± 0.0489 | 3.1245 ± 0.0920 | 0.7306 ± 0.0113 | 0.6325 ± 0.0320 | 1.9706 |
| `tr_Mondrian+TabFM [noise_all]` | torchregress | 0.9069 ± 0.0021 | 2.3753 ± 0.0937 | 3.0619 ± 0.0832 | **0.7170 ± 0.0178** | 0.6557 ± 0.0276 | 9.1857 |
| `tr_Mondrian+TabICL [noise_all]` | torchregress | 0.9033 ± 0.0088 | 2.4162 ± 0.0538 | 3.1744 ± 0.1022 | 0.7248 ± 0.0147 | 0.6366 ± 0.0171 | 1.9525 |
| `tr_Mondrian+TabularMLP [noise_all]` | torchregress | 0.9053 ± 0.0119 | 2.4367 ± 0.1071 | 3.0384 ± 0.0882 | 0.7304 ± 0.0180 | 0.6386 ± 0.0207 | 19.2777 |
| `tr_Normalized+KumoTabular [noise_all]` | torchregress | 0.9044 ± 0.0064 | 2.3412 ± 0.0492 | 3.0735 ± 0.0889 | 0.7192 ± 0.0131 | 0.6508 ± 0.0436 | 2.2407 |
| `tr_Normalized+LightGBM [noise_all]` | torchregress | 0.9069 ± 0.0068 | 2.4385 ± 0.0410 | 3.1164 ± 0.1004 | 0.7306 ± 0.0113 | 0.6341 ± 0.0276 | 1.9706 |
| `tr_Normalized+TabICL [noise_all]` | torchregress | 0.9019 ± 0.0076 | 2.3241 ± 0.0696 | 3.1003 ± 0.0919 | 0.7248 ± 0.0147 | 0.6513 ± 0.0484 | 1.9525 |
| `tr_Normalized+TabularMLP [noise_all]` | torchregress | 0.9044 ± 0.0097 | 2.3870 ± 0.0663 | 3.0196 ± 0.0962 | 0.7304 ± 0.0180 | 0.6417 ± 0.0150 | 19.2777 |
| `tr_PITRecal+KumoTabular [noise_all]` | torchregress | 0.9058 ± 0.0043 | 2.3440 ± 0.0339 | 2.8822 ± 0.0719 | 0.7217 ± 0.0131 | 0.6697 ± 0.0318 | 2.2407 |
| `tr_PITRecal+LightGBM [noise_all]` | torchregress | 0.9033 ± 0.0083 | 2.6520 ± 0.1075 | 3.0819 ± 0.1068 | 0.7306 ± 0.0114 | 0.5565 ± 0.0381 | 1.9706 |
| `tr_PITRecal+TabICL [noise_all]` | torchregress | 0.9036 ± 0.0065 | 2.3663 ± 0.0276 | 2.9218 ± 0.0743 | 0.7272 ± 0.0132 | 0.6700 ± 0.0353 | 1.9525 |
| `tr_PITRecal+TabularMLP [noise_all]` | torchregress | 0.9103 ± 0.0054 | 2.4510 ± 0.0154 | 3.0030 ± 0.0832 | 0.7316 ± 0.0156 | 0.7043 ± 0.0478 | 19.2777 |
| `tr_ResidualDist+KumoTabular [noise_all]` | torchregress | 0.9119 ± 0.0069 | 2.4588 ± 0.0606 | 3.0965 ± 0.0612 | 0.7201 ± 0.0127 | 0.7208 ± 0.0360 | 2.2407 |
| `tr_ResidualDist+LightGBM [noise_all]` | torchregress | 0.9133 ± 0.0096 | 2.5154 ± 0.0538 | 3.0966 ± 0.0642 | 0.7311 ± 0.0114 | 0.6845 ± 0.0115 | 1.9706 |
| `tr_ResidualDist+TabFM [noise_all]` | torchregress | 0.9094 ± 0.0105 | 2.4445 ± 0.0816 | 3.0521 ± 0.0690 | 0.7201 ± 0.0172 | 0.6916 ± 0.0207 | 9.1857 |
| `tr_ResidualDist+TabICL [noise_all]` | torchregress | 0.9064 ± 0.0106 | 2.4567 ± 0.0561 | 3.1114 ± 0.0570 | 0.7261 ± 0.0141 | 0.7125 ± 0.0227 | 1.9525 |
| `tr_ResidualDist+TabularMLP [noise_all]` | torchregress | 0.9086 ± 0.0041 | 2.4899 ± 0.0474 | 3.0490 ± 0.0845 | 0.7312 ± 0.0167 | 0.6964 ± 0.0279 | 19.2777 |
| `tr_Split+KumoTabular [noise_all]` | torchregress | 0.9117 ± 0.0094 | 2.4329 ± 0.0869 | 3.1467 ± 0.0854 | 0.7192 ± 0.0131 | 0.6775 ± 0.0253 | 2.2407 |
| `tr_Split+LightGBM [noise_all]` | torchregress | 0.9078 ± 0.0060 | 2.4640 ± 0.0237 | 3.1394 ± 0.0775 | 0.7306 ± 0.0113 | 0.6637 ± 0.0317 | 1.9706 |
| `tr_Split+TabFM [noise_all]` | torchregress | 0.9094 ± 0.0121 | 2.4068 ± 0.0945 | 3.0655 ± 0.0853 | **0.7170 ± 0.0178** | 0.6748 ± 0.0225 | 9.1857 |
| `tr_Split+TabICL [noise_all]` | torchregress | 0.9100 ± 0.0051 | 2.4236 ± 0.0722 | 3.1605 ± 0.0850 | 0.7248 ± 0.0147 | 0.6770 ± 0.0388 | 1.9525 |
| `tr_Split+TabularMLP [noise_all]` | torchregress | 0.9106 ± 0.0059 | 2.4391 ± 0.0482 | 3.0550 ± 0.0886 | 0.7304 ± 0.0180 | 0.6664 ± 0.0155 | 19.2777 |

##### fm_uq_scenarios / openml_year@noise_test

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_test]` | sdm | 0.9447 ± 0.0159 | 3.7198 ± 0.1265 | 4.1741 ± 0.0355 | 1.0118 ± 0.0071 | 0.8809 ± 0.0323 | 2.2291 |
| `Raw+LightGBM [noise_test]` | lightgbm | 0.7928 ± 0.0131 | 2.0962 ± 0.0576 | 3.6132 ± 0.0940 | 0.8828 ± 0.0345 | 0.4302 ± 0.0426 | 1.9769 |
| `Raw+TabICL [noise_test]` | tabicl | 0.9456 ± 0.0077 | 3.4997 ± 0.0843 | 3.9648 ± 0.0837 | 0.9184 ± 0.0095 | 0.8875 ± 0.0223 | 1.9634 |
| `Raw+TabularMLP [noise_test]` | torchregress | 0.6672 ± 0.0342 | 2.4633 ± 0.1802 | 6.6863 ± 0.5693 † | 1.1310 ± 0.0289 | 0.5514 ± 0.0051 | 32.2454 |
| `tr_CQR+KumoTabular [noise_test]` | torchregress | 0.9472 ± 0.0109 | 3.7277 ± 0.0979 | 4.1752 ± 0.0367 | 1.0118 ± 0.0071 | 0.8945 ± 0.0100 | 2.2291 |
| `tr_CQR+LightGBM [noise_test]` | torchregress | 0.8178 ± 0.0149 | 2.1942 ± 0.0700 | 3.5215 ± 0.0931 | 0.8828 ± 0.0345 | 0.4865 ± 0.0460 | 1.9769 |
| `tr_CQR+TabICL [noise_test]` | torchregress | 0.9469 ± 0.0060 | 3.5077 ± 0.0698 | 3.9679 ± 0.0780 | 0.9184 ± 0.0095 | 0.8951 ± 0.0156 | 1.9634 |
| `tr_CQR+TabularMLP [noise_test]` | torchregress | 0.6536 ± 0.0267 | 2.3975 ± 0.1299 | 6.8334 ± 0.5196 † | 1.1310 ± 0.0289 | 0.5347 ± 0.0140 | 32.2454 |
| `tr_DCP+KumoTabular [noise_test]` | torchregress | 0.9481 ± 0.0063 | 3.7558 ± 0.0124 | 4.1847 ± 0.0523 | 1.0118 ± 0.0071 | 0.8918 ± 0.0141 | 2.2291 |
| `tr_DCP+LightGBM [noise_test]` | torchregress | 0.8414 ± 0.0156 | 2.3827 ± 0.1078 | **3.4002 ± 0.0341** | 0.8828 ± 0.0345 | 0.4582 ± 0.0628 | 1.9769 |
| `tr_DCP+TabICL [noise_test]` | torchregress | 0.9503 ± 0.0021 | 3.5340 ± 0.0315 | 3.9779 ± 0.0663 | 0.9184 ± 0.0095 | 0.8949 ± 0.0100 | 1.9634 |
| `tr_DCP+TabularMLP [noise_test]` | torchregress | 0.6461 ± 0.0189 | 2.3193 ± 0.0749 | 6.8657 ± 0.5133 † | 1.1310 ± 0.0289 | 0.5293 ± 0.0148 | 32.2454 |
| `tr_Mondrian+KumoTabular [noise_test]` | torchregress | 0.3947 ± 0.0094 | 0.9551 ± 0.0410 | 9.0880 ± 0.2358 | 1.0118 ± 0.0071 | 0.1985 ± 0.0071 | 2.2291 |
| `tr_Mondrian+LightGBM [noise_test]` | torchregress | 0.5756 ± 0.0104 | 1.3217 ± 0.0288 | 5.8914 ± 0.3251 † | 0.8828 ± 0.0345 | 0.4565 ± 0.0538 | 1.9769 |
| `tr_Mondrian+TabFM [noise_test]` | torchregress | 0.4206 ± 0.0348 | 0.9928 ± 0.0374 | 8.0899 ± 0.4009 | 0.9437 ± 0.0150 | 0.1430 ± 0.1079 | 9.1945 |
| `tr_Mondrian+TabICL [noise_test]` | torchregress | 0.4722 ± 0.0322 | 1.0665 ± 0.0734 | 7.3276 ± 0.4228 † | 0.9184 ± 0.0095 | 0.2609 ± 0.0432 | 1.9634 |
| `tr_Mondrian+TabularMLP [noise_test]` | torchregress | 0.4808 ± 0.0159 | 1.3676 ± 0.0157 | 8.6891 ± 0.3186 | 1.1310 ± 0.0289 | 0.3811 ± 0.0116 | 32.2454 |
| `tr_Normalized+KumoTabular [noise_test]` | torchregress | 0.8961 ± 0.0113 | 3.7295 ± 0.1412 | 4.7545 ± 0.1016 | 1.0118 ± 0.0071 | 0.7104 ± 0.0973 | 2.2291 |
| `tr_Normalized+LightGBM [noise_test]` | torchregress | 0.7011 ± 0.0156 | 1.7891 ± 0.0595 | 4.7507 ± 0.1605 | 0.8828 ± 0.0345 | 0.5839 ± 0.0516 | 1.9769 |
| `tr_Normalized+TabICL [noise_test]` | torchregress | 0.9108 ± 0.0095 | 3.4616 ± 0.0740 | 4.3900 ± 0.0704 | 0.9184 ± 0.0095 | 0.7902 ± 0.0233 | 1.9634 |
| `tr_Normalized+TabularMLP [noise_test]` | torchregress | 0.6461 ± 0.0189 | 2.3193 ± 0.0749 | 6.8657 ± 0.5133 † | 1.1310 ± 0.0289 | 0.5293 ± 0.0148 | 32.2454 |
| `tr_PITRecal+KumoTabular [noise_test]` | torchregress | 0.9464 ± 0.0067 | 3.7784 ± 0.0259 | 4.2450 ± 0.0847 | 1.0198 ± 0.0135 | 0.8775 ± 0.0219 | 2.2291 |
| `tr_PITRecal+LightGBM [noise_test]` | torchregress | 0.8364 ± 0.0207 | 2.3404 ± 0.0567 | 3.4486 ± 0.0887 | **0.8811 ± 0.0327** | 0.5118 ± 0.0526 | 1.9769 |
| `tr_PITRecal+TabICL [noise_test]` | torchregress | 0.9492 ± 0.0014 | 3.5252 ± 0.0314 | 3.9726 ± 0.0451 | 0.9190 ± 0.0113 | 0.9007 ± 0.0191 | 1.9634 |
| `tr_PITRecal+TabularMLP [noise_test]` | torchregress | 0.6503 ± 0.0200 | 2.3714 ± 0.0909 | 6.8624 ± 0.5604 † | 1.1160 ± 0.0299 | 0.5524 ± 0.0151 | 32.2454 |
| `tr_ResidualDist+KumoTabular [noise_test]` | torchregress | 0.4128 ± 0.0182 | 0.9869 ± 0.0523 | 8.8174 ± 0.2987 | 1.0122 ± 0.0071 | 0.1882 ± 0.0122 | 2.2291 |
| `tr_ResidualDist+LightGBM [noise_test]` | torchregress | 0.5753 ± 0.0054 | 1.3158 ± 0.0319 | 5.8909 ± 0.3495 † | 0.8830 ± 0.0339 | 0.4651 ± 0.0326 | 1.9769 |
| `tr_ResidualDist+TabFM [noise_test]` | torchregress | 0.4278 ± 0.0298 | 0.9943 ± 0.0446 | 8.0015 ± 0.4718 † | 0.9451 ± 0.0139 | 0.1467 ± 0.0648 | 9.1945 |
| `tr_ResidualDist+TabICL [noise_test]` | torchregress | 0.4733 ± 0.0250 | 1.0677 ± 0.0346 | 7.2407 ± 0.1981 | 0.9186 ± 0.0096 | 0.2756 ± 0.0409 | 1.9634 |
| `tr_ResidualDist+TabularMLP [noise_test]` | torchregress | 0.4931 ± 0.0125 | 1.4056 ± 0.0210 | 8.5439 ± 0.2492 | 1.1291 ± 0.0285 | 0.4009 ± 0.0167 | 32.2454 |
| `tr_Split+KumoTabular [noise_test]` | torchregress | 0.4083 ± 0.0132 | 0.9795 ± 0.0470 | 8.9484 ± 0.2612 | 1.0118 ± 0.0071 | 0.1800 ± 0.0098 | 2.2291 |
| `tr_Split+LightGBM [noise_test]` | torchregress | 0.5914 ± 0.0126 | 1.3379 ± 0.0394 | 5.8108 ± 0.3004 † | 0.8828 ± 0.0345 | 0.4912 ± 0.0267 | 1.9769 |
| `tr_Split+TabFM [noise_test]` | torchregress | 0.4208 ± 0.0268 | 0.9826 ± 0.0462 | 8.1132 ± 0.4025 | 0.9437 ± 0.0150 | 0.1322 ± 0.0817 | 9.1945 |
| `tr_Split+TabICL [noise_test]` | torchregress | 0.4742 ± 0.0260 | 1.0571 ± 0.0452 | 7.3422 ± 0.3082 | 0.9184 ± 0.0095 | 0.2635 ± 0.0502 | 1.9634 |
| `tr_Split+TabularMLP [noise_test]` | torchregress | 0.4967 ± 0.0181 | 1.4087 ± 0.0222 | 8.4811 ± 0.3564 | 1.1310 ± 0.0289 | 0.3840 ± 0.0245 | 32.2454 |

##### fm_uq_scenarios / openml_year@outliers

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [outliers]` | sdm | 0.9833 ± 0.0025 | 2.0099 ± 0.1377 | 2.0823 ± 0.1837 † | **0.3354 ± 0.0296** | 0.9589 ± 0.0003 | 2.2073 |
| `Raw+LightGBM [outliers]` | lightgbm | 0.9347 ± 0.0038 | 2.5670 ± 0.0782 | 2.8940 ± 0.0854 | 0.4306 ± 0.0234 | 0.7943 ± 0.0314 | 2.3136 |
| `Raw+TabICL [outliers]` | tabicl | 0.9550 ± 0.0036 | 1.9686 ± 0.0952 | 2.1117 ± 0.0456 | 0.3667 ± 0.0284 | 0.8885 ± 0.0120 | 1.9752 |
| `Raw+TabularMLP [outliers]` | torchregress | 0.9997 ± 0.0005 | 6.0412 ± 0.1565 | 6.0429 ± 0.1543 | 0.5741 ± 0.0095 | 0.9977 ± 0.0039 | 14.3338 |
| `tr_CQR+KumoTabular [outliers]` | torchregress | 0.9481 ± 0.0076 | 1.9670 ± 0.1513 | 2.0502 ± 0.1952 † | **0.3354 ± 0.0296** | 0.6682 ± 0.0556 | 2.2073 |
| `tr_CQR+LightGBM [outliers]` | torchregress | 0.9536 ± 0.0034 | 2.7214 ± 0.0561 | 2.9626 ± 0.0712 | 0.4306 ± 0.0234 | 0.8477 ± 0.0318 | 2.3136 |
| `tr_CQR+TabICL [outliers]` | torchregress | 0.9500 ± 0.0030 | 1.9510 ± 0.0856 | 2.1024 ± 0.0369 | 0.3667 ± 0.0284 | 0.8857 ± 0.0076 | 1.9752 |
| `tr_CQR+TabularMLP [outliers]` | torchregress | 0.9497 ± 0.0075 | 2.4503 ± 0.0939 | 2.8284 ± 0.1722 † | 0.5741 ± 0.0095 | 0.7718 ± 0.0018 | 14.3338 |
| `tr_DCP+KumoTabular [outliers]` | torchregress | 0.9617 ± 0.0043 | 1.3047 ± 0.1204 | 1.4480 ± 0.1530 † | **0.3354 ± 0.0296** | 0.9334 ± 0.0164 | 2.2073 |
| `tr_DCP+LightGBM [outliers]` | torchregress | 0.9517 ± 0.0055 | 2.9464 ± 0.0934 | 3.1884 ± 0.0317 | 0.4306 ± 0.0234 | 0.8348 ± 0.0345 | 2.3136 |
| `tr_DCP+TabICL [outliers]` | torchregress | 0.9464 ± 0.0077 | 1.8125 ± 0.1127 | 1.9866 ± 0.0434 | 0.3667 ± 0.0284 | 0.8703 ± 0.0202 | 1.9752 |
| `tr_DCP+TabularMLP [outliers]` | torchregress | 0.9458 ± 0.0088 | 2.2469 ± 0.0954 | 2.6565 ± 0.0380 | 0.5741 ± 0.0095 | 0.7999 ± 0.0304 | 14.3338 |
| `tr_Mondrian+KumoTabular [outliers]` | torchregress | 0.9492 ± 0.0096 | 1.4002 ± 0.0039 | 1.7375 ± 0.1152 † | **0.3354 ± 0.0296** | 0.9149 ± 0.0247 | 2.2073 |
| `tr_Mondrian+LightGBM [outliers]` | torchregress | 0.9461 ± 0.0083 | 1.7370 ± 0.0736 | 2.1110 ± 0.0535 | 0.4306 ± 0.0234 | 0.8647 ± 0.0271 | 2.3136 |
| `tr_Mondrian+TabFM [outliers]` | torchregress | 0.9508 ± 0.0101 | 1.4858 ± 0.0940 | 1.7925 ± 0.1341 † | 0.3523 ± 0.0323 | 0.8915 ± 0.0053 | 9.0859 |
| `tr_Mondrian+TabICL [outliers]` | torchregress | 0.9486 ± 0.0139 | 1.5094 ± 0.0354 | 1.8685 ± 0.1123 † | 0.3667 ± 0.0284 | 0.9136 ± 0.0226 | 1.9752 |
| `tr_Mondrian+TabularMLP [outliers]` | torchregress | 0.9472 ± 0.0106 | 2.2476 ± 0.1480 | 2.6421 ± 0.0454 | 0.5741 ± 0.0095 | 0.8357 ± 0.0372 | 14.3338 |
| `tr_Normalized+KumoTabular [outliers]` | torchregress | 0.9483 ± 0.0117 | 1.4400 ± 0.0882 | 1.6508 ± 0.1515 † | **0.3354 ± 0.0296** | 0.8930 ± 0.0219 | 2.2073 |
| `tr_Normalized+LightGBM [outliers]` | torchregress | 0.9533 ± 0.0096 | 2.0160 ± 0.0765 | 2.3195 ± 0.0600 | 0.4306 ± 0.0234 | 0.8787 ± 0.0177 | 2.3136 |
| `tr_Normalized+TabICL [outliers]` | torchregress | 0.9453 ± 0.0051 | 1.7990 ± 0.1327 | 2.0191 ± 0.0790 | 0.3667 ± 0.0284 | 0.8926 ± 0.0186 | 1.9752 |
| `tr_Normalized+TabularMLP [outliers]` | torchregress | 0.9458 ± 0.0088 | 2.2469 ± 0.0954 | 2.6565 ± 0.0380 | 0.5741 ± 0.0095 | 0.7999 ± 0.0304 | 14.3338 |
| `tr_PITRecal+KumoTabular [outliers]` | torchregress | 0.9633 ± 0.0058 | 1.3088 ± 0.1119 | **1.4477 ± 0.1364** † | 0.3358 ± 0.0293 | 0.9370 ± 0.0167 | 2.2073 |
| `tr_PITRecal+LightGBM [outliers]` | torchregress | 0.9533 ± 0.0060 | 2.9859 ± 0.2199 | 3.2195 ± 0.1280 | 0.4307 ± 0.0233 | 0.8505 ± 0.0213 | 2.3136 |
| `tr_PITRecal+TabICL [outliers]` | torchregress | 0.9486 ± 0.0077 | 1.8795 ± 0.1406 | 2.0380 ± 0.0794 | 0.3675 ± 0.0277 | 0.8757 ± 0.0272 | 1.9752 |
| `tr_PITRecal+TabularMLP [outliers]` | torchregress | 0.9467 ± 0.0046 | 2.2398 ± 0.0492 | 2.6258 ± 0.0702 | 0.5710 ± 0.0115 | 0.8518 ± 0.0427 | 14.3338 |
| `tr_ResidualDist+KumoTabular [outliers]` | torchregress | 0.9517 ± 0.0095 | 1.3958 ± 0.0516 | 1.7239 ± 0.1206 † | 0.3354 ± 0.0295 | 0.9083 ± 0.0097 | 2.2073 |
| `tr_ResidualDist+LightGBM [outliers]` | torchregress | 0.9533 ± 0.0130 | 1.7628 ± 0.0553 | 2.0972 ± 0.1178 † | 0.4309 ± 0.0235 | 0.8897 ± 0.0254 | 2.3136 |
| `tr_ResidualDist+TabFM [outliers]` | torchregress | 0.9539 ± 0.0126 | 1.4241 ± 0.0483 | 1.7510 ± 0.1333 † | 0.3499 ± 0.0310 | 0.8983 ± 0.0224 | 9.0859 |
| `tr_ResidualDist+TabICL [outliers]` | torchregress | 0.9508 ± 0.0121 | 1.4977 ± 0.0399 | 1.8557 ± 0.1254 † | 0.3670 ± 0.0283 | 0.8979 ± 0.0289 | 1.9752 |
| `tr_ResidualDist+TabularMLP [outliers]` | torchregress | 0.9469 ± 0.0068 | 2.2103 ± 0.0601 | 2.6038 ± 0.0451 | 0.5711 ± 0.0118 | 0.8483 ± 0.0437 | 14.3338 |
| `tr_Split+KumoTabular [outliers]` | torchregress | 0.9536 ± 0.0096 | 1.3969 ± 0.0306 | 1.7233 ± 0.1240 † | **0.3354 ± 0.0296** | 0.8992 ± 0.0188 | 2.2073 |
| `tr_Split+LightGBM [outliers]` | torchregress | 0.9486 ± 0.0113 | 1.7239 ± 0.0649 | 2.0967 ± 0.0857 | 0.4306 ± 0.0234 | 0.8485 ± 0.0345 | 2.3136 |
| `tr_Split+TabFM [outliers]` | torchregress | 0.9517 ± 0.0109 | 1.4105 ± 0.0647 | 1.7561 ± 0.1305 † | 0.3523 ± 0.0323 | 0.8914 ± 0.0104 | 9.0859 |
| `tr_Split+TabICL [outliers]` | torchregress | 0.9508 ± 0.0131 | 1.4716 ± 0.0366 | 1.8452 ± 0.1070 † | 0.3667 ± 0.0284 | 0.8853 ± 0.0331 | 1.9752 |
| `tr_Split+TabularMLP [outliers]` | torchregress | 0.9472 ± 0.0077 | 2.2388 ± 0.1131 | 2.6420 ± 0.0384 | 0.5741 ± 0.0095 | 0.8184 ± 0.0371 | 14.3338 |

##### fm_uq_scenarios / openml_year@shift

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [shift]` | sdm | 0.9036 ± 0.0059 | 0.9797 ± 0.0243 | 1.2788 ± 0.0248 | 0.3462 ± 0.0060 | 0.8583 ± 0.0211 | 2.2220 |
| `Raw+LightGBM [shift]` | lightgbm | 0.8511 ± 0.0087 | 1.5147 ± 0.0478 | 2.0594 ± 0.0137 | 0.4313 ± 0.0092 | 0.5659 ± 0.0532 | 1.9650 |
| `Raw+TabICL [shift]` | tabicl | 0.9050 ± 0.0079 | 1.0817 ± 0.0242 | 1.4045 ± 0.0330 | 0.3712 ± 0.0157 | 0.8067 ± 0.0852 | 1.9936 |
| `Raw+TabularMLP [shift]` | torchregress | 0.9053 ± 0.0145 | 1.4179 ± 0.0358 | 1.8557 ± 0.0764 | 0.4490 ± 0.0111 | 0.8323 ± 0.0483 | 33.0733 |
| `tr_CQR+KumoTabular [shift]` | torchregress | 0.8936 ± 0.0068 | 0.9676 ± 0.0205 | 1.2787 ± 0.0246 | 0.3462 ± 0.0060 | 0.8454 ± 0.0099 | 2.2220 |
| `tr_CQR+LightGBM [shift]` | torchregress | 0.8989 ± 0.0179 | 1.6300 ± 0.0313 | 2.0408 ± 0.0090 | 0.4313 ± 0.0092 | 0.6954 ± 0.0570 | 1.9650 |
| `tr_CQR+TabICL [shift]` | torchregress | 0.8983 ± 0.0117 | 1.0726 ± 0.0311 | 1.4047 ± 0.0321 | 0.3712 ± 0.0157 | 0.7803 ± 0.1139 | 1.9936 |
| `tr_CQR+TabularMLP [shift]` | torchregress | 0.9053 ± 0.0231 | 1.4180 ± 0.0375 | 1.8619 ± 0.0695 | 0.4490 ± 0.0111 | 0.8257 ± 0.0700 | 33.0733 |
| `tr_DCP+KumoTabular [shift]` | torchregress | 0.8967 ± 0.0036 | 0.9556 ± 0.0166 | 1.2782 ± 0.0244 | 0.3462 ± 0.0060 | 0.8395 ± 0.0054 | 2.2220 |
| `tr_DCP+LightGBM [shift]` | torchregress | 0.9003 ± 0.0173 | 2.0118 ± 0.1659 | 2.3284 ± 0.1358 † | 0.4313 ± 0.0092 | 0.6250 ± 0.0652 | 1.9650 |
| `tr_DCP+TabICL [shift]` | torchregress | 0.9017 ± 0.0123 | 1.0678 ± 0.0390 | 1.4046 ± 0.0322 | 0.3712 ± 0.0157 | 0.7970 ± 0.0941 | 1.9936 |
| `tr_DCP+TabularMLP [shift]` | torchregress | 0.9053 ± 0.0240 | 1.4162 ± 0.0520 | 1.8610 ± 0.0695 | 0.4490 ± 0.0111 | 0.8224 ± 0.0754 | 33.0733 |
| `tr_Mondrian+KumoTabular [shift]` | torchregress | 0.8867 ± 0.0096 | 0.9775 ± 0.0527 | 1.6444 ± 0.0405 | 0.3462 ± 0.0060 | 0.8098 ± 0.0393 | 2.2220 |
| `tr_Mondrian+LightGBM [shift]` | torchregress | 0.9067 ± 0.0058 | 1.3705 ± 0.0266 | 1.9762 ± 0.0765 | 0.4313 ± 0.0092 | 0.8221 ± 0.0562 | 1.9650 |
| `tr_Mondrian+TabFM [shift]` | torchregress | 0.8822 ± 0.0133 | 0.9666 ± 0.0255 | 1.6384 ± 0.1303 † | **0.3438 ± 0.0255** | 0.7676 ± 0.0374 | 9.0888 |
| `tr_Mondrian+TabICL [shift]` | torchregress | 0.8900 ± 0.0080 | 1.0737 ± 0.0151 | 1.7727 ± 0.0979 † | 0.3712 ± 0.0157 | 0.8211 ± 0.0237 | 1.9936 |
| `tr_Mondrian+TabularMLP [shift]` | torchregress | 0.9061 ± 0.0070 | 1.4352 ± 0.0533 | 2.0492 ± 0.0587 | 0.4490 ± 0.0111 | 0.7946 ± 0.0490 | 33.0733 |
| `tr_Normalized+KumoTabular [shift]` | torchregress | 0.9033 ± 0.0014 | 0.9774 ± 0.0055 | 1.3804 ± 0.0266 | 0.3462 ± 0.0060 | 0.8276 ± 0.0283 | 2.2220 |
| `tr_Normalized+LightGBM [shift]` | torchregress | 0.9036 ± 0.0093 | 1.3388 ± 0.0427 | 1.8703 ± 0.0419 | 0.4313 ± 0.0092 | 0.8207 ± 0.0492 | 1.9650 |
| `tr_Normalized+TabICL [shift]` | torchregress | 0.9017 ± 0.0129 | 1.0908 ± 0.0367 | 1.5071 ± 0.0499 | 0.3712 ± 0.0157 | 0.8121 ± 0.0542 | 1.9936 |
| `tr_Normalized+TabularMLP [shift]` | torchregress | 0.9053 ± 0.0240 | 1.4162 ± 0.0520 | 1.8610 ± 0.0695 | 0.4490 ± 0.0111 | 0.8224 ± 0.0754 | 33.0733 |
| `tr_PITRecal+KumoTabular [shift]` | torchregress | 0.8958 ± 0.0017 | 0.9538 ± 0.0133 | **1.2780 ± 0.0252** | 0.3463 ± 0.0060 | 0.8395 ± 0.0054 | 2.2220 |
| `tr_PITRecal+LightGBM [shift]` | torchregress | 0.8994 ± 0.0143 | 1.8216 ± 0.0431 | 2.1776 ± 0.0190 | 0.4306 ± 0.0087 | 0.6382 ± 0.0779 | 1.9650 |
| `tr_PITRecal+TabICL [shift]` | torchregress | 0.9011 ± 0.0092 | 1.0672 ± 0.0306 | 1.4047 ± 0.0308 | 0.3717 ± 0.0158 | 0.7940 ± 0.0910 | 1.9936 |
| `tr_PITRecal+TabularMLP [shift]` | torchregress | 0.9097 ± 0.0223 | 1.4416 ± 0.0638 | 1.8622 ± 0.0643 | 0.4500 ± 0.0094 | 0.8472 ± 0.0528 | 33.0733 |
| `tr_ResidualDist+KumoTabular [shift]` | torchregress | 0.8958 ± 0.0121 | 0.9930 ± 0.0522 | 1.6301 ± 0.0347 | 0.3462 ± 0.0059 | 0.8018 ± 0.0388 | 2.2220 |
| `tr_ResidualDist+LightGBM [shift]` | torchregress | 0.9128 ± 0.0051 | 1.3833 ± 0.0138 | 1.9499 ± 0.0703 | 0.4310 ± 0.0090 | 0.8297 ± 0.0543 | 1.9650 |
| `tr_ResidualDist+TabFM [shift]` | torchregress | 0.8869 ± 0.0135 | 0.9878 ± 0.0420 | 1.6264 ± 0.1184 † | 0.3444 ± 0.0250 | 0.7933 ± 0.0145 | 9.0888 |
| `tr_ResidualDist+TabICL [shift]` | torchregress | 0.8961 ± 0.0071 | 1.0892 ± 0.0454 | 1.7432 ± 0.0833 | 0.3715 ± 0.0158 | 0.8169 ± 0.0361 | 1.9936 |
| `tr_ResidualDist+TabularMLP [shift]` | torchregress | 0.9092 ± 0.0101 | 1.4496 ± 0.0340 | 2.0425 ± 0.0647 | 0.4495 ± 0.0103 | 0.8122 ± 0.0645 | 33.0733 |
| `tr_Split+KumoTabular [shift]` | torchregress | 0.8894 ± 0.0104 | 0.9767 ± 0.0611 | 1.6437 ± 0.0270 | 0.3462 ± 0.0060 | 0.7717 ± 0.0390 | 2.2220 |
| `tr_Split+LightGBM [shift]` | torchregress | 0.9069 ± 0.0076 | 1.3632 ± 0.0218 | 1.9738 ± 0.0550 | 0.4313 ± 0.0092 | 0.7859 ± 0.0863 | 1.9650 |
| `tr_Split+TabFM [shift]` | torchregress | 0.8872 ± 0.0138 | 0.9855 ± 0.0316 | 1.6239 ± 0.1083 † | **0.3438 ± 0.0255** | 0.7932 ± 0.0350 | 9.0888 |
| `tr_Split+TabICL [shift]` | torchregress | 0.8903 ± 0.0107 | 1.0756 ± 0.0457 | 1.7550 ± 0.0706 | 0.3712 ± 0.0157 | 0.7921 ± 0.0353 | 1.9936 |
| `tr_Split+TabularMLP [shift]` | torchregress | 0.9133 ± 0.0096 | 1.4540 ± 0.0375 | 2.0477 ± 0.0543 | 0.4490 ± 0.0111 | 0.8096 ± 0.0794 | 33.0733 |
| `tr_WeightedCQR+KumoTabular [shift]` | torchregress | 0.8919 ± 0.0064 | 0.9719 ± 0.0266 | 1.2825 ± 0.0276 | 0.3462 ± 0.0060 | 0.8324 ± 0.0317 | 2.2220 |
| `tr_WeightedCQR+LightGBM [shift]` | torchregress | 0.9156 ± 0.0118 | 1.7003 ± 0.0632 | 2.0469 ± 0.0061 | 0.4313 ± 0.0092 | 0.7484 ± 0.0346 | 1.9650 |
| `tr_WeightedCQR+TabICL [shift]` | torchregress | 0.8958 ± 0.0073 | 1.0775 ± 0.0317 | 1.4106 ± 0.0372 | 0.3712 ± 0.0157 | 0.7732 ± 0.1148 | 1.9936 |
| `tr_WeightedCQR+TabularMLP [shift]` | torchregress | 0.9144 ± 0.0208 | 1.4867 ± 0.0529 | 1.8752 ± 0.0514 | 0.4490 ± 0.0111 | 0.8432 ± 0.0552 | 33.0733 |
| `tr_WeightedSplit+KumoTabular [shift]` | torchregress | 0.9094 ± 0.0146 | 1.0695 ± 0.1145 | 1.6344 ± 0.0316 | 0.3462 ± 0.0060 | 0.8040 ± 0.0435 | 2.2220 |
| `tr_WeightedSplit+LightGBM [shift]` | torchregress | 0.9261 ± 0.0126 | 1.4973 ± 0.1014 | 1.9927 ± 0.0514 | 0.4313 ± 0.0092 | 0.8131 ± 0.0527 | 1.9650 |
| `tr_WeightedSplit+TabFM [shift]` | torchregress | 0.9089 ± 0.0247 | 1.0826 ± 0.1149 | 1.6193 ± 0.0948 † | **0.3438 ± 0.0255** | 0.8185 ± 0.0418 | 9.0888 |
| `tr_WeightedSplit+TabICL [shift]` | torchregress | 0.9136 ± 0.0185 | 1.2050 ± 0.1254 | 1.7601 ± 0.0470 | 0.3712 ± 0.0157 | 0.8359 ± 0.0299 | 1.9936 |
| `tr_WeightedSplit+TabularMLP [shift]` | torchregress | 0.9256 ± 0.0090 | 1.5594 ± 0.0667 | 2.0648 ± 0.0450 | 0.4490 ± 0.0111 | 0.8367 ± 0.0630 | 33.0733 |

##### fm_uq_scenarios / zurich_delays

_No successful rows._

##### fm_uq_scenarios / zurich_delays@clean

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [clean]` | sdm | 0.9033 ± 0.0162 | 1.5267 ± 0.6907 | **2.2603 ± 1.1435** † | 0.6484 ± 0.3291 | 0.5612 ± 0.0732 | 5.1230 |
| `Raw+LightGBM [clean]` | lightgbm | 0.8286 ± 0.0113 | 1.2183 ± 0.6006 | 2.4331 ± 1.2211 † | 0.6597 ± 0.3320 | 0.2919 ± 0.0072 | 2.2628 |
| `Raw+TabICL [clean]` | tabicl | 0.8958 ± 0.0109 | 1.4497 ± 0.6945 | 2.2622 ± 1.1605 † | 0.6486 ± 0.3317 | 0.5303 ± 0.0443 | 7.0951 |
| `Raw+TabularMLP [clean]` | torchregress | 0.9472 ± 0.0120 | 1.9081 ± 0.9036 | 2.6707 ± 1.2908 † | 0.6479 ± 0.3320 | 0.5409 ± 0.0561 | 20.8735 |
| `tr_CQR+KumoTabular [clean]` | torchregress | 0.8842 ± 0.0088 | 1.4782 ± 0.6757 | 2.2639 ± 1.1514 † | 0.6484 ± 0.3291 | 0.4656 ± 0.0469 | 5.1230 |
| `tr_CQR+LightGBM [clean]` | torchregress | 0.8983 ± 0.0030 | 1.3966 ± 0.6915 | 2.3729 ± 1.1949 † | 0.6597 ± 0.3320 | 0.4080 ± 0.0177 | 2.2628 |
| `tr_CQR+TabICL [clean]` | torchregress | 0.8883 ± 0.0079 | 1.4301 ± 0.6828 | 2.2641 ± 1.1634 † | 0.6486 ± 0.3317 | 0.4927 ± 0.0054 | 7.0951 |
| `tr_CQR+TabularMLP [clean]` | torchregress | 0.8850 ± 0.0172 | 1.4944 ± 0.7180 | 2.5557 ± 1.2653 † | 0.6479 ± 0.3320 | 0.4058 ± 0.0178 | 20.8735 |
| `tr_DCP+KumoTabular [clean]` | torchregress | 0.8811 ± 0.0096 | 1.4093 ± 0.6747 | 2.2667 ± 1.1484 † | 0.6484 ± 0.3291 | 0.4795 ± 0.0352 | 5.1230 |
| `tr_DCP+LightGBM [clean]` | torchregress | 0.8936 ± 0.0043 | 1.5332 ± 0.7399 | 2.3935 ± 1.1901 † | 0.6597 ± 0.3320 | 0.4936 ± 0.0374 | 2.2628 |
| `tr_DCP+TabICL [clean]` | torchregress | 0.8881 ± 0.0082 | 1.4130 ± 0.6748 | 2.2643 ± 1.1642 † | 0.6486 ± 0.3317 | 0.4975 ± 0.0056 | 7.0951 |
| `tr_DCP+TabularMLP [clean]` | torchregress | 0.8836 ± 0.0171 | 1.3725 ± 0.6654 | 2.5074 ± 1.2317 † | 0.6479 ± 0.3320 | 0.3426 ± 0.0347 | 20.8735 |
| `tr_Mondrian+KumoTabular [clean]` | torchregress | 0.8850 ± 0.0080 | 1.3939 ± 0.6943 | 2.7046 ± 1.3577 † | 0.6484 ± 0.3291 | 0.3307 ± 0.0247 | 5.1230 |
| `tr_Mondrian+LightGBM [clean]` | torchregress | 0.8822 ± 0.0125 | 1.3881 ± 0.7620 | 2.7984 ± 1.3927 † | 0.6597 ± 0.3320 | 0.2174 ± 0.0906 | 2.2628 |
| `tr_Mondrian+TabFM [clean]` | torchregress | 0.8908 ± 0.0066 | 1.4023 ± 0.7428 | 2.5809 ± 1.2313 † | **0.6471 ± 0.3283** | 0.3853 ± 0.0254 | 13.3470 |
| `tr_Mondrian+TabICL [clean]` | torchregress | 0.8786 ± 0.0029 | 1.3818 ± 0.7095 | 2.7436 ± 1.3625 † | 0.6486 ± 0.3317 | 0.2846 ± 0.0163 | 7.0951 |
| `tr_Mondrian+TabularMLP [clean]` | torchregress | 0.8850 ± 0.0092 | 1.3538 ± 0.7139 | 2.5244 ± 1.2466 † | 0.6479 ± 0.3320 | 0.3544 ± 0.0402 | 20.8735 |
| `tr_Normalized+KumoTabular [clean]` | torchregress | 0.8792 ± 0.0014 | 1.3395 ± 0.6552 | 2.5847 ± 1.2530 † | 0.6484 ± 0.3291 | 0.2986 ± 0.0115 | 5.1230 |
| `tr_Normalized+LightGBM [clean]` | torchregress | 0.8886 ± 0.0099 | 1.3930 ± 0.7143 | 2.6862 ± 1.3219 † | 0.6597 ± 0.3320 | 0.2329 ± 0.0375 | 2.2628 |
| `tr_Normalized+TabICL [clean]` | torchregress | 0.8844 ± 0.0034 | 1.4132 ± 0.7256 | 2.6105 ± 1.3029 † | 0.6486 ± 0.3317 | 0.2947 ± 0.0137 | 7.0951 |
| `tr_Normalized+TabularMLP [clean]` | torchregress | 0.8836 ± 0.0171 | 1.3725 ± 0.6654 | 2.5074 ± 1.2317 † | 0.6479 ± 0.3320 | 0.3426 ± 0.0347 | 20.8735 |
| `tr_PITRecal+KumoTabular [clean]` | torchregress | 0.8808 ± 0.0096 | 1.3767 ± 0.6720 | 2.2741 ± 1.1459 † | 0.6548 ± 0.3311 | 0.4710 ± 0.0128 | 5.1230 |
| `tr_PITRecal+LightGBM [clean]` | torchregress | 0.8928 ± 0.0025 | 1.5045 ± 0.7503 | 2.3887 ± 1.1838 † | 0.6647 ± 0.3323 | 0.4839 ± 0.0256 | 2.2628 |
| `tr_PITRecal+TabICL [clean]` | torchregress | 0.8883 ± 0.0079 | 1.4178 ± 0.6900 | 2.2645 ± 1.1644 † | 0.6488 ± 0.3329 | 0.5225 ± 0.0358 | 7.0951 |
| `tr_PITRecal+TabularMLP [clean]` | torchregress | 0.8842 ± 0.0088 | 1.4745 ± 0.7274 | 2.4342 ± 1.2349 † | 0.6626 ± 0.3315 | 0.4866 ± 0.0319 | 20.8735 |
| `tr_ResidualDist+KumoTabular [clean]` | torchregress | 0.8950 ± 0.0074 | 1.4772 ± 0.7653 | 2.5730 ± 1.2879 † | 0.6518 ± 0.3297 | 0.4594 ± 0.0580 | 5.1230 |
| `tr_ResidualDist+LightGBM [clean]` | torchregress | 0.8908 ± 0.0146 | 1.4957 ± 0.7966 | 2.5817 ± 1.2664 † | 0.6626 ± 0.3323 | 0.4231 ± 0.0924 | 2.2628 |
| `tr_ResidualDist+TabFM [clean]` | torchregress | 0.8989 ± 0.0070 | 1.5103 ± 0.7905 | 2.6880 ± 1.3215 † | 0.6529 ± 0.3286 | 0.4400 ± 0.0414 | 13.3470 |
| `tr_ResidualDist+TabICL [clean]` | torchregress | 0.8903 ± 0.0092 | 1.4551 ± 0.7539 | 2.5706 ± 1.2885 † | 0.6485 ± 0.3324 | 0.4309 ± 0.0827 | 7.0951 |
| `tr_ResidualDist+TabularMLP [clean]` | torchregress | 0.8983 ± 0.0106 | 1.5031 ± 0.7710 | 2.6445 ± 1.3176 † | 0.6546 ± 0.3320 | 0.4387 ± 0.0611 | 20.8735 |
| `tr_Split+KumoTabular [clean]` | torchregress | 0.8972 ± 0.0099 | 1.3544 ± 0.6995 | 2.7994 ± 1.3555 † | 0.6484 ± 0.3291 | 0.0858 ± 0.0392 | 5.1230 |
| `tr_Split+LightGBM [clean]` | torchregress | 0.8989 ± 0.0133 | 1.3654 ± 0.7074 | 2.8446 ± 1.3721 † | 0.6597 ± 0.3320 | 0.0622 ± 0.0483 | 2.2628 |
| `tr_Split+TabFM [clean]` | torchregress | 0.8964 ± 0.0117 | 1.4177 ± 0.7421 | 2.7434 ± 1.3282 † | **0.6471 ± 0.3283** | 0.2564 ± 0.0461 | 13.3470 |
| `tr_Split+TabICL [clean]` | torchregress | 0.8975 ± 0.0098 | 1.3977 ± 0.7283 | 2.8373 ± 1.3642 † | 0.6486 ± 0.3317 | 0.0823 ± 0.0308 | 7.0951 |
| `tr_Split+TabularMLP [clean]` | torchregress | 0.9017 ± 0.0108 | 1.4338 ± 0.7355 | 2.6836 ± 1.3231 † | 0.6479 ± 0.3320 | 0.2856 ± 0.0365 | 20.8735 |

##### fm_uq_scenarios / zurich_delays@extrapolate

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [extrapolate]` | sdm | 0.9222 ± 0.0010 | 1.7052 ± 0.8665 | 2.2197 ± 1.1929 † | **0.5834 ± 0.3345** | 0.5230 ± 0.0991 | 4.6529 |
| `Raw+LightGBM [extrapolate]` | lightgbm | 0.8511 ± 0.0210 | 1.3080 ± 0.7096 | 2.2052 ± 1.2652 † | 0.6010 ± 0.3504 | 0.2797 ± 0.0433 | 2.1892 |
| `Raw+TabICL [extrapolate]` | tabicl | 0.9214 ± 0.0034 | 1.5648 ± 0.8171 | **2.0928 ± 1.1223** † | 0.5928 ± 0.3429 | 0.5265 ± 0.0787 | 6.7419 |
| `Raw+TabularMLP [extrapolate]` | torchregress | 0.9583 ± 0.0090 | 2.3396 ± 1.3653 | 2.8863 ± 1.7354 † | 0.5962 ± 0.3398 | 0.5253 ± 0.0718 | 19.4284 |
| `tr_CQR+KumoTabular [extrapolate]` | torchregress | 0.9158 ± 0.0052 | 1.6826 ± 0.8535 | 2.2156 ± 1.1901 † | **0.5834 ± 0.3345** | 0.4949 ± 0.1247 | 4.6529 |
| `tr_CQR+LightGBM [extrapolate]` | torchregress | 0.9200 ± 0.0066 | 1.5094 ± 0.8120 | 2.1884 ± 1.2673 † | 0.6010 ± 0.3504 | 0.4523 ± 0.0511 | 2.1892 |
| `tr_CQR+TabICL [extrapolate]` | torchregress | 0.9256 ± 0.0021 | 1.5763 ± 0.8133 | 2.0953 ± 1.1213 † | 0.5928 ± 0.3429 | 0.5265 ± 0.0787 | 6.7419 |
| `tr_CQR+TabularMLP [extrapolate]` | torchregress | 0.9147 ± 0.0174 | 2.0115 ± 1.2359 | 2.7568 ± 1.6913 † | 0.5962 ± 0.3398 | 0.3780 ± 0.1860 | 19.4284 |
| `tr_DCP+KumoTabular [extrapolate]` | torchregress | 0.9136 ± 0.0063 | 1.6365 ± 0.8284 | 2.1897 ± 1.1831 † | **0.5834 ± 0.3345** | 0.4790 ± 0.1184 | 4.6529 |
| `tr_DCP+LightGBM [extrapolate]` | torchregress | 0.9119 ± 0.0091 | 1.7657 ± 0.9471 | 2.3499 ± 1.3151 † | 0.6010 ± 0.3504 | 0.5112 ± 0.0570 | 2.1892 |
| `tr_DCP+TabICL [extrapolate]` | torchregress | 0.9239 ± 0.0017 | 1.5869 ± 0.8102 | 2.1032 ± 1.1199 † | 0.5928 ± 0.3429 | 0.5265 ± 0.0787 | 6.7419 |
| `tr_DCP+TabularMLP [extrapolate]` | torchregress | 0.9167 ± 0.0213 | 1.8633 ± 1.1302 | 2.6276 ± 1.5879 † | 0.5962 ± 0.3398 | 0.3421 ± 0.2270 | 19.4284 |
| `tr_Mondrian+KumoTabular [extrapolate]` | torchregress | 0.9044 ± 0.0343 | 1.5699 ± 1.1550 | 2.4590 ± 1.4989 † | **0.5834 ± 0.3345** | 0.2763 ± 0.2093 | 4.6529 |
| `tr_Mondrian+LightGBM [extrapolate]` | torchregress | 0.9200 ± 0.0096 | 1.5907 ± 1.0283 | 2.5301 ± 1.5759 † | 0.6010 ± 0.3504 | 0.2742 ± 0.1032 | 2.1892 |
| `tr_Mondrian+TabFM [extrapolate]` | torchregress | 0.8894 ± 0.0529 | 1.5059 ± 0.9498 | 3.6872 ± 1.1832 † | 1.0859 ± 0.5297 | 0.3508 ± 0.1813 | 13.1460 |
| `tr_Mondrian+TabICL [extrapolate]` | torchregress | 0.9197 ± 0.0217 | 1.6220 ± 1.0587 | 2.5023 ± 1.4667 † | 0.5928 ± 0.3429 | 0.2758 ± 0.1765 | 6.7419 |
| `tr_Mondrian+TabularMLP [extrapolate]` | torchregress | 0.9014 ± 0.0252 | 1.5032 ± 1.0000 | 2.3557 ± 1.4298 † | 0.5962 ± 0.3398 | 0.2944 ± 0.1477 | 19.4284 |
| `tr_Normalized+KumoTabular [extrapolate]` | torchregress | 0.9194 ± 0.0193 | 1.6554 ± 0.9367 | 2.4215 ± 1.3780 † | **0.5834 ± 0.3345** | 0.3388 ± 0.2002 | 4.6529 |
| `tr_Normalized+LightGBM [extrapolate]` | torchregress | 0.9319 ± 0.0091 | 1.6276 ± 0.9615 | 2.4462 ± 1.4862 † | 0.6010 ± 0.3504 | 0.3064 ± 0.0951 | 2.1892 |
| `tr_Normalized+TabICL [extrapolate]` | torchregress | 0.9294 ± 0.0147 | 1.6377 ± 0.8959 | 2.3490 ± 1.3160 † | 0.5928 ± 0.3429 | 0.3241 ± 0.1762 | 6.7419 |
| `tr_Normalized+TabularMLP [extrapolate]` | torchregress | 0.9167 ± 0.0213 | 1.8633 ± 1.1302 | 2.6276 ± 1.5879 † | 0.5962 ± 0.3398 | 0.3421 ± 0.2270 | 19.4284 |
| `tr_PITRecal+KumoTabular [extrapolate]` | torchregress | 0.9156 ± 0.0061 | 1.6719 ± 0.8673 | 2.2008 ± 1.1837 † | 0.5852 ± 0.3335 | 0.4408 ± 0.1620 | 4.6529 |
| `tr_PITRecal+LightGBM [extrapolate]` | torchregress | 0.9181 ± 0.0059 | 1.7604 ± 0.9781 | 2.3231 ± 1.2955 † | 0.6007 ± 0.3501 | 0.4806 ± 0.1164 | 2.1892 |
| `tr_PITRecal+TabICL [extrapolate]` | torchregress | 0.9222 ± 0.0050 | 1.6629 ± 0.8821 | 2.1473 ± 1.1570 † | 0.5923 ± 0.3409 | 0.4825 ± 0.1456 | 6.7419 |
| `tr_PITRecal+TabularMLP [extrapolate]` | torchregress | 0.9139 ± 0.0075 | 1.9644 ± 1.1461 | 2.6007 ± 1.5347 † | 0.6819 ± 0.4132 | 0.4575 ± 0.1552 | 19.4284 |
| `tr_ResidualDist+KumoTabular [extrapolate]` | torchregress | 0.9186 ± 0.0097 | 1.5742 ± 0.8145 | 2.3433 ± 1.3220 † | 0.5845 ± 0.3333 | 0.4428 ± 0.0520 | 4.6529 |
| `tr_ResidualDist+LightGBM [extrapolate]` | torchregress | 0.9028 ± 0.0050 | 1.6266 ± 0.8678 | 2.4307 ± 1.3674 † | 0.6008 ± 0.3499 | 0.4764 ± 0.0950 | 2.1892 |
| `tr_ResidualDist+TabFM [extrapolate]` | torchregress | 0.9028 ± 0.0177 | 1.6112 ± 0.8056 | 3.7312 ± 0.9107 † | 1.0861 ± 0.5349 | 0.4933 ± 0.0592 | 13.1460 |
| `tr_ResidualDist+TabICL [extrapolate]` | torchregress | 0.9219 ± 0.0071 | 1.5734 ± 0.8159 | 2.3821 ± 1.3211 † | 0.5905 ± 0.3402 | 0.4522 ± 0.0408 | 6.7419 |
| `tr_ResidualDist+TabularMLP [extrapolate]` | torchregress | 0.9194 ± 0.0119 | 1.6588 ± 0.8757 | 2.4783 ± 1.4064 † | 0.5970 ± 0.3447 | 0.4580 ± 0.0526 | 19.4284 |
| `tr_Split+KumoTabular [extrapolate]` | torchregress | 0.9286 ± 0.0032 | 1.4893 ± 0.7916 | 2.4616 ± 1.3913 † | **0.5834 ± 0.3345** | 0.1300 ± 0.1534 | 4.6529 |
| `tr_Split+LightGBM [extrapolate]` | torchregress | 0.9322 ± 0.0099 | 1.4966 ± 0.7800 | 2.5176 ± 1.4700 † | 0.6010 ± 0.3504 | 0.1427 ± 0.1006 | 2.1892 |
| `tr_Split+TabFM [extrapolate]` | torchregress | 0.9106 ± 0.0280 | 1.5350 ± 0.7712 | 3.7468 ± 1.0050 † | 1.0859 ± 0.5297 | 0.3315 ± 0.1620 | 13.1460 |
| `tr_Split+TabICL [extrapolate]` | torchregress | 0.9283 ± 0.0072 | 1.5192 ± 0.7919 | 2.5624 ± 1.4463 † | 0.5928 ± 0.3429 | 0.0997 ± 0.1084 | 6.7419 |
| `tr_Split+TabularMLP [extrapolate]` | torchregress | 0.9169 ± 0.0125 | 1.4548 ± 0.7321 | 2.4323 ± 1.3840 † | 0.5962 ± 0.3398 | 0.2494 ± 0.1734 | 19.4284 |

##### fm_uq_scenarios / zurich_delays@heavy_tail

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [heavy_tail]` | sdm | 0.9989 ± 0.0019 | 0.0018 ± 0.0015 | 0.4344 ± 0.7478 † | 0.7456 ± 1.2907 | 0.9903 ± 0.0168 | 4.7240 |
| `Raw+LightGBM [heavy_tail]` | lightgbm | 0.8192 (n=1) | 4.266e-04 (n=1) | 1.2983 (n=1) | 2.2360 (n=1) | 0.2905 (n=1) | 2.1956 |
| `Raw+TabICL [heavy_tail]` | tabicl | 0.9947 ± 0.0005 | 0.0013 ± 0.0011 | 0.4339 ± 0.7482 † | 0.7456 ± 1.2908 | 0.9899 ± 0.0081 | 6.9538 |
| `Raw+TabularMLP [heavy_tail]` | torchregress | 0.9992 ± 0.0014 | 0.0392 ± 0.0319 | 0.4717 ± 0.7276 † | 0.7470 ± 1.2896 | 0.9935 ± 0.0112 | 25.9421 |
| `tr_CQR+KumoTabular [heavy_tail]` | torchregress | 0.8958 ± 0.0146 | 9.432e-04 ± 9.513e-04 | 0.4337 ± 0.7484 † | 0.7456 ± 1.2907 | 0.8793 ± 0.0181 | 4.7240 |
| `tr_CQR+LightGBM [heavy_tail]` | torchregress | 0.8833 (n=1) | 4.266e-04 (n=1) | 1.2983 (n=1) | 2.2360 (n=1) | 0.4660 (n=1) | 2.1956 |
| `tr_CQR+TabICL [heavy_tail]` | torchregress | 0.9056 ± 0.0136 | 7.888e-04 ± 6.727e-04 | 0.4335 ± 0.7485 † | 0.7456 ± 1.2908 | 0.8949 ± 0.0319 | 6.9538 |
| `tr_CQR+TabularMLP [heavy_tail]` | torchregress | 0.8969 ± 0.0041 | 0.0077 ± 0.0077 | 0.4417 ± 0.7419 † | 0.7470 ± 1.2896 | 0.8824 ± 0.0211 | 25.9421 |
| `tr_DCP+KumoTabular [heavy_tail]` | torchregress | 0.8992 ± 0.0044 | 8.775e-04 ± 8.749e-04 | 0.4336 ± 0.7485 † | 0.7456 ± 1.2907 | 0.8817 ± 0.0276 | 4.7240 |
| `tr_DCP+LightGBM [heavy_tail]` | torchregress | 0.8867 (n=1) | 0.0014 (n=1) | 1.2993 (n=1) | 2.2360 (n=1) | 0.4932 (n=1) | 2.1956 |
| `tr_DCP+TabICL [heavy_tail]` | torchregress | 0.9042 ± 0.0139 | 7.858e-04 ± 6.715e-04 | 0.4335 ± 0.7485 † | 0.7456 ± 1.2908 | 0.8820 ± 0.0522 | 6.9538 |
| `tr_DCP+TabularMLP [heavy_tail]` | torchregress | 0.8986 ± 0.0173 | 0.0043 ± 0.0071 | 0.4376 ± 0.7451 † | 0.7470 ± 1.2896 | 0.8829 ± 0.0369 | 25.9421 |
| `tr_Mondrian+KumoTabular [heavy_tail]` | torchregress | 0.9014 ± 0.0155 | 5.913e-04 ± 5.704e-04 | 0.4333 ± 0.7488 † | 0.7456 ± 1.2907 | 0.8860 ± 0.0379 | 4.7240 |
| `tr_Mondrian+LightGBM [heavy_tail]` | torchregress | 0.9608 ± 0.0678 | 8.066e-09 ± 1.284e-08 | 0.4326 ± 0.7493 † | 0.7453 ± 1.2910 | 0.7670 ± 0.4036 | 1.2357 |
| `tr_Mondrian+TabFM [heavy_tail]` | torchregress | 0.9083 ± 0.0079 | 0.0277 ± 0.0236 | 0.4777 ± 0.7114 † | 0.7642 ± 1.2746 | 0.8834 ± 0.0499 | 13.3050 |
| `tr_Mondrian+TabICL [heavy_tail]` | torchregress | 0.8997 ± 0.0142 | 5.421e-04 ± 4.629e-04 | 0.4333 ± 0.7488 † | 0.7456 ± 1.2908 | 0.8666 ± 0.0698 | 6.9538 |
| `tr_Mondrian+TabularMLP [heavy_tail]` | torchregress | 0.9072 ± 0.0104 | 0.0033 ± 0.0054 | 0.4362 ± 0.7463 † | 0.7470 ± 1.2896 | 0.8886 ± 0.0219 | 25.9421 |
| `tr_Normalized+KumoTabular [heavy_tail]` | torchregress | 0.8989 ± 0.0025 | 7.641e-04 ± 7.139e-04 | 0.4335 ± 0.7486 † | 0.7456 ± 1.2907 | 0.8882 ± 0.0161 | 4.7240 |
| `tr_Normalized+LightGBM [heavy_tail]` | torchregress | 0.9092 (n=1) | 3.994e-06 (n=1) | 1.2978 (n=1) | 2.2360 (n=1) | 0.1068 (n=1) | 2.1956 |
| `tr_Normalized+TabICL [heavy_tail]` | torchregress | 0.9039 ± 0.0229 | 8.028e-04 ± 6.862e-04 | 0.4335 ± 0.7485 † | 0.7456 ± 1.2908 | 0.8853 ± 0.0550 | 6.9538 |
| `tr_Normalized+TabularMLP [heavy_tail]` | torchregress | 0.8986 ± 0.0173 | 0.0043 ± 0.0071 | 0.4376 ± 0.7451 † | 0.7470 ± 1.2896 | 0.8829 ± 0.0369 | 25.9421 |
| `tr_PITRecal+KumoTabular [heavy_tail]` | torchregress | 0.8947 ± 0.0092 | 3.726e-04 ± 3.190e-04 | 0.4331 ± 0.7489 † | 0.7455 ± 1.2909 | 0.8767 ± 0.0235 | 4.7240 |
| `tr_PITRecal+LightGBM [heavy_tail]` | torchregress | 0.8858 (n=1) | 0.0016 (n=1) | 1.2994 (n=1) | 2.2360 (n=1) | 0.4797 (n=1) | 2.1956 |
| `tr_PITRecal+TabICL [heavy_tail]` | torchregress | 0.8972 ± 0.0126 | 5.520e-04 ± 4.691e-04 | 0.4333 ± 0.7487 † | 0.7455 ± 1.2908 | 0.8783 ± 0.0416 | 6.9538 |
| `tr_PITRecal+TabularMLP [heavy_tail]` | torchregress | 0.8936 ± 0.0083 | 0.0018 ± 0.0028 | 0.4349 ± 0.7474 † | 0.7460 ± 1.2904 | 0.8746 ± 0.0335 | 25.9421 |
| `tr_ResidualDist+KumoTabular [heavy_tail]` | torchregress | 0.9069 ± 0.0039 | 3.750e-04 ± 3.180e-04 | 0.4331 ± 0.7489 † | 0.7455 ± 1.2909 | 0.8922 ± 0.0246 | 4.7240 |
| `tr_ResidualDist+LightGBM [heavy_tail]` | torchregress | 0.9642 ± 0.0621 | 5.453e-09 ± 9.444e-09 | **0.4326 ± 0.7493** † | **0.7453 ± 1.2910** | 0.8378 ± 0.2809 | 1.2357 |
| `tr_ResidualDist+TabFM [heavy_tail]` | torchregress | 0.8981 ± 0.0243 | 0.0162 ± 0.0136 | 0.4716 ± 0.7169 † | 0.7599 ± 1.2784 | 0.8663 ± 0.0490 | 13.3050 |
| `tr_ResidualDist+TabICL [heavy_tail]` | torchregress | 0.8953 ± 0.0059 | 5.419e-04 ± 4.603e-04 | 0.4334 ± 0.7487 † | 0.7455 ± 1.2908 | 0.8778 ± 0.0319 | 6.9538 |
| `tr_ResidualDist+TabularMLP [heavy_tail]` | torchregress | 0.8914 ± 0.0092 | 0.0019 ± 0.0029 | 0.4351 ± 0.7472 † | 0.7460 ± 1.2904 | 0.8735 ± 0.0328 | 25.9421 |
| `tr_Split+KumoTabular [heavy_tail]` | torchregress | 0.8994 ± 0.0092 | 7.644e-04 ± 7.063e-04 | 0.4335 ± 0.7486 † | 0.7456 ± 1.2907 | 0.8767 ± 0.0321 | 4.7240 |
| `tr_Split+LightGBM [heavy_tail]` | torchregress | 0.9708 ± 0.0505 | 3.953e-09 ± 5.713e-09 | 0.4326 ± 0.7493 † | 0.7453 ± 1.2910 | 0.6990 ± 0.5213 | 1.2357 |
| `tr_Split+TabFM [heavy_tail]` | torchregress | 0.9044 ± 0.0216 | 0.0351 ± 0.0301 | 0.4924 ± 0.6991 † | 0.7642 ± 1.2746 | 0.8668 ± 0.0865 | 13.3050 |
| `tr_Split+TabICL [heavy_tail]` | torchregress | 0.8956 ± 0.0117 | 7.646e-04 ± 6.536e-04 | 0.4336 ± 0.7485 † | 0.7456 ± 1.2908 | 0.8750 ± 0.0464 | 6.9538 |
| `tr_Split+TabularMLP [heavy_tail]` | torchregress | 0.8917 ± 0.0090 | 0.0043 ± 0.0071 | 0.4378 ± 0.7449 † | 0.7470 ± 1.2896 | 0.8704 ± 0.0400 | 25.9421 |

##### fm_uq_scenarios / zurich_delays@imbalance

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [imbalance]` | sdm | 0.7469 ± 0.0048 | 0.7957 ± 0.4017 | 3.3606 ± 1.5122 † | 0.6936 ± 0.3420 | 0.1778 ± 0.0231 | 4.0301 |
| `Raw+LightGBM [imbalance]` | lightgbm | 0.6736 ± 0.0143 | 0.6936 ± 0.3382 | 3.6387 ± 1.6838 † | 0.6935 ± 0.3439 | 0.1027 ± 0.0322 | 2.0129 |
| `Raw+TabICL [imbalance]` | tabicl | 0.7636 ± 0.0101 | 0.8544 ± 0.4330 | **3.1975 ± 1.4617** † | 0.6956 ± 0.3445 | 0.2506 ± 0.0139 | 6.0894 |
| `Raw+TabularMLP [imbalance]` | torchregress | 0.8342 ± 0.0405 | 1.3642 ± 0.6846 | 3.3667 ± 1.8395 † | **0.6838 ± 0.3390** | 0.4216 ± 0.1276 | 15.3823 |
| `tr_CQR+KumoTabular [imbalance]` | torchregress | 0.7306 ± 0.0113 | 0.7582 ± 0.3762 | 3.4227 ± 1.5655 † | 0.6936 ± 0.3420 | 0.1449 ± 0.0335 | 4.0301 |
| `tr_CQR+LightGBM [imbalance]` | torchregress | 0.7428 ± 0.0090 | 0.8134 ± 0.3915 | 3.4097 ± 1.6007 † | 0.6935 ± 0.3439 | 0.1740 ± 0.0441 | 2.0129 |
| `tr_CQR+TabICL [imbalance]` | torchregress | 0.7456 ± 0.0104 | 0.8184 ± 0.4221 | 3.2511 ± 1.4958 † | 0.6956 ± 0.3445 | 0.2243 ± 0.0446 | 6.0894 |
| `tr_CQR+TabularMLP [imbalance]` | torchregress | 0.7525 ± 0.0208 | 1.1457 ± 0.6660 | 3.5811 ± 1.8907 † | **0.6838 ± 0.3390** | 0.2426 ± 0.0996 | 15.3823 |
| `tr_DCP+KumoTabular [imbalance]` | torchregress | 0.7308 ± 0.0095 | 0.7519 ± 0.3709 | 3.4453 ± 1.5853 † | 0.6936 ± 0.3420 | 0.1450 ± 0.0280 | 4.0301 |
| `tr_DCP+LightGBM [imbalance]` | torchregress | 0.7428 ± 0.0129 | 0.8589 ± 0.4071 | 3.3436 ± 1.5429 † | 0.6935 ± 0.3439 | 0.1890 ± 0.0556 | 2.0129 |
| `tr_DCP+TabICL [imbalance]` | torchregress | 0.7414 ± 0.0146 | 0.8049 ± 0.4096 | 3.2936 ± 1.5426 † | 0.6956 ± 0.3445 | 0.2022 ± 0.0598 | 6.0894 |
| `tr_DCP+TabularMLP [imbalance]` | torchregress | 0.7447 ± 0.0150 | 1.0497 ± 0.6143 | 3.5494 ± 1.8501 † | **0.6838 ± 0.3390** | 0.2115 ± 0.0874 | 15.3823 |
| `tr_Mondrian+KumoTabular [imbalance]` | torchregress | 0.7256 ± 0.0055 | 0.7367 ± 0.3615 | 3.6262 ± 1.7406 † | 0.6936 ± 0.3420 | 0.1103 ± 0.0247 | 4.0301 |
| `tr_Mondrian+LightGBM [imbalance]` | torchregress | 0.7297 ± 0.0051 | 0.7595 ± 0.3717 | 3.5914 ± 1.7314 † | 0.6935 ± 0.3439 | 0.1211 ± 0.0295 | 2.0129 |
| `tr_Mondrian+TabFM [imbalance]` | torchregress | 0.7356 ± 0.0046 | 0.7739 ± 0.3770 | 3.5769 ± 1.6721 † | 0.7042 ± 0.3326 | 0.1517 ± 0.0196 | 11.8039 |
| `tr_Mondrian+TabICL [imbalance]` | torchregress | 0.7300 ± 0.0088 | 0.7408 ± 0.3578 | 3.6687 ± 1.7905 † | 0.6956 ± 0.3445 | 0.1051 ± 0.0160 | 6.0894 |
| `tr_Mondrian+TabularMLP [imbalance]` | torchregress | 0.7294 ± 0.0112 | 0.7392 ± 0.3528 | 3.5112 ± 1.7116 † | **0.6838 ± 0.3390** | 0.1607 ± 0.0621 | 15.3823 |
| `tr_Normalized+KumoTabular [imbalance]` | torchregress | 0.7267 ± 0.0092 | 0.7513 ± 0.3748 | 3.5389 ± 1.6356 † | 0.6936 ± 0.3420 | 0.1191 ± 0.0202 | 4.0301 |
| `tr_Normalized+LightGBM [imbalance]` | torchregress | 0.7403 ± 0.0079 | 0.8207 ± 0.3964 | 3.5015 ± 1.6618 † | 0.6935 ± 0.3439 | 0.1533 ± 0.0496 | 2.0129 |
| `tr_Normalized+TabICL [imbalance]` | torchregress | 0.7356 ± 0.0115 | 0.8064 ± 0.4074 | 3.4908 ± 1.6470 † | 0.6956 ± 0.3445 | 0.1383 ± 0.0420 | 6.0894 |
| `tr_Normalized+TabularMLP [imbalance]` | torchregress | 0.7447 ± 0.0150 | 1.0497 ± 0.6143 | 3.5494 ± 1.8501 † | **0.6838 ± 0.3390** | 0.2115 ± 0.0874 | 15.3823 |
| `tr_PITRecal+KumoTabular [imbalance]` | torchregress | 0.7286 ± 0.0142 | 0.7489 ± 0.3669 | 3.4784 ± 1.6325 † | 0.7005 ± 0.3469 | 0.1328 ± 0.0429 | 4.0301 |
| `tr_PITRecal+LightGBM [imbalance]` | torchregress | 0.7450 ± 0.0129 | 0.8634 ± 0.4044 | 3.3960 ± 1.5863 † | 0.6977 ± 0.3449 | 0.1781 ± 0.0579 | 2.0129 |
| `tr_PITRecal+TabICL [imbalance]` | torchregress | 0.7406 ± 0.0126 | 0.8006 ± 0.4053 | 3.3649 ± 1.6046 † | 0.6975 ± 0.3466 | 0.1747 ± 0.0736 | 6.0894 |
| `tr_PITRecal+TabularMLP [imbalance]` | torchregress | 0.7447 ± 0.0195 | 1.0514 ± 0.6174 | 3.5788 ± 1.8575 † | 0.7744 ± 0.4356 | 0.1952 ± 0.0901 | 15.3823 |
| `tr_ResidualDist+KumoTabular [imbalance]` | torchregress | 0.7261 ± 0.0071 | 0.7307 ± 0.3573 | 3.6052 ± 1.7384 † | 0.6994 ± 0.3452 | 0.1211 ± 0.0281 | 4.0301 |
| `tr_ResidualDist+LightGBM [imbalance]` | torchregress | 0.7294 ± 0.0085 | 0.7492 ± 0.3691 | 3.5714 ± 1.7364 † | 0.6974 ± 0.3454 | 0.1425 ± 0.0468 | 2.0129 |
| `tr_ResidualDist+TabFM [imbalance]` | torchregress | 0.7367 ± 0.0096 | 0.7669 ± 0.3742 | 3.5627 ± 1.6862 † | 0.7109 ± 0.3377 | 0.1654 ± 0.0382 | 11.8039 |
| `tr_ResidualDist+TabICL [imbalance]` | torchregress | 0.7244 ± 0.0131 | 0.7347 ± 0.3560 | 3.6053 ± 1.7716 † | 0.6971 ± 0.3459 | 0.1286 ± 0.0326 | 6.0894 |
| `tr_ResidualDist+TabularMLP [imbalance]` | torchregress | 0.7289 ± 0.0113 | 0.7380 ± 0.3574 | 3.5711 ± 1.7173 † | 0.6951 ± 0.3426 | 0.1307 ± 0.0376 | 15.3823 |
| `tr_Split+KumoTabular [imbalance]` | torchregress | 0.7256 ± 0.0077 | 0.7306 ± 0.3570 | 3.6312 ± 1.7393 † | 0.6936 ± 0.3420 | 0.1127 ± 0.0139 | 4.0301 |
| `tr_Split+LightGBM [imbalance]` | torchregress | 0.7244 ± 0.0076 | 0.7485 ± 0.3687 | 3.6120 ± 1.7352 † | 0.6935 ± 0.3439 | 0.1101 ± 0.0363 | 2.0129 |
| `tr_Split+TabFM [imbalance]` | torchregress | 0.7383 ± 0.0092 | 0.7677 ± 0.3767 | 3.5808 ± 1.6677 † | 0.7042 ± 0.3326 | 0.1579 ± 0.0395 | 11.8039 |
| `tr_Split+TabICL [imbalance]` | torchregress | 0.7269 ± 0.0094 | 0.7382 ± 0.3597 | 3.6614 ± 1.7778 † | 0.6956 ± 0.3445 | 0.1049 ± 0.0253 | 6.0894 |
| `tr_Split+TabularMLP [imbalance]` | torchregress | 0.7281 ± 0.0114 | 0.7373 ± 0.3543 | 3.5112 ± 1.7075 † | **0.6838 ± 0.3390** | 0.1597 ± 0.0562 | 15.3823 |

##### fm_uq_scenarios / zurich_delays@noise_all

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_all]` | sdm | 0.9014 ± 0.0089 | 1.5908 ± 0.7790 | 2.5043 ± 1.2916 † | 0.6684 ± 0.3364 | 0.5414 ± 0.0359 | 4.7711 |
| `Raw+LightGBM [noise_all]` | lightgbm | 0.8000 ± 0.0167 | 1.1249 ± 0.5435 | 2.7232 ± 1.3166 † | 0.6748 ± 0.3392 | 0.1448 ± 0.0397 | 15.9809 |
| `Raw+TabICL [noise_all]` | tabicl | 0.9033 ± 0.0132 | 1.6584 ± 0.8184 | 2.6095 ± 1.3177 † | 0.6764 ± 0.3369 | 0.5528 ± 0.0665 | 6.4647 |
| `Raw+TabularMLP [noise_all]` | torchregress | 0.9603 ± 0.0128 | 2.2083 ± 0.9910 | 3.0386 ± 1.4160 † | 0.6670 ± 0.3372 | 0.5969 ± 0.1566 | 14.2075 |
| `tr_CQR+KumoTabular [noise_all]` | torchregress | 0.8908 ± 0.0098 | 1.5468 ± 0.7521 | 2.5081 ± 1.2981 † | 0.6684 ± 0.3364 | 0.4878 ± 0.0421 | 4.7711 |
| `tr_CQR+LightGBM [noise_all]` | torchregress | 0.8989 ± 0.0071 | 1.4104 ± 0.7090 | 2.5960 ± 1.2502 † | 0.6748 ± 0.3392 | 0.3139 ± 0.0380 | 15.9809 |
| `tr_CQR+TabICL [noise_all]` | torchregress | 0.8903 ± 0.0156 | 1.6129 ± 0.7876 | 2.6131 ± 1.3236 † | 0.6764 ± 0.3369 | 0.4830 ± 0.0620 | 6.4647 |
| `tr_CQR+TabularMLP [noise_all]` | torchregress | 0.8981 ± 0.0101 | 1.4628 ± 0.6934 | 2.7456 ± 1.3946 † | 0.6670 ± 0.3372 | 0.2961 ± 0.0963 | 14.2075 |
| `tr_DCP+KumoTabular [noise_all]` | torchregress | 0.8925 ± 0.0113 | 1.5063 ± 0.7319 | 2.4973 ± 1.2879 † | 0.6684 ± 0.3364 | 0.5134 ± 0.0469 | 4.7711 |
| `tr_DCP+LightGBM [noise_all]` | torchregress | 0.8950 ± 0.0055 | 1.7360 ± 0.8579 | 2.6978 ± 1.3073 † | 0.6748 ± 0.3392 | 0.4410 ± 0.0681 | 15.9809 |
| `tr_DCP+TabICL [noise_all]` | torchregress | 0.8889 ± 0.0147 | 1.5352 ± 0.7278 | 2.5956 ± 1.3119 † | 0.6764 ± 0.3369 | 0.4883 ± 0.0548 | 6.4647 |
| `tr_DCP+TabularMLP [noise_all]` | torchregress | 0.8958 ± 0.0058 | 1.3763 ± 0.6486 | 2.7117 ± 1.3736 † | 0.6670 ± 0.3372 | 0.2242 ± 0.1033 | 14.2075 |
| `tr_Mondrian+KumoTabular [noise_all]` | torchregress | 0.8858 ± 0.0066 | 1.4295 ± 0.7597 | 2.8381 ± 1.4018 † | 0.6684 ± 0.3364 | 0.2499 ± 0.0796 | 4.7711 |
| `tr_Mondrian+LightGBM [noise_all]` | torchregress | 0.8894 ± 0.0055 | 1.3883 ± 0.7275 | 2.9249 ± 1.4125 † | 0.6748 ± 0.3392 | 0.1329 ± 0.0760 | 15.9809 |
| `tr_Mondrian+TabFM [noise_all]` | torchregress | 0.8919 ± 0.0075 | 1.3354 ± 0.6827 | 2.6300 ± 1.2856 † | **0.6607 ± 0.3381** | 0.2793 ± 0.0477 | 13.3537 |
| `tr_Mondrian+TabICL [noise_all]` | torchregress | 0.8919 ± 0.0125 | 1.4664 ± 0.7959 | 2.9216 ± 1.4147 † | 0.6764 ± 0.3369 | 0.2311 ± 0.0735 | 6.4647 |
| `tr_Mondrian+TabularMLP [noise_all]` | torchregress | 0.8992 ± 0.0058 | 1.3938 ± 0.6682 | 2.6912 ± 1.3646 † | 0.6670 ± 0.3372 | 0.2584 ± 0.0766 | 14.2075 |
| `tr_Normalized+KumoTabular [noise_all]` | torchregress | 0.8897 ± 0.0065 | 1.4325 ± 0.7634 | 2.7895 ± 1.3959 † | 0.6684 ± 0.3364 | 0.2526 ± 0.0765 | 4.7711 |
| `tr_Normalized+LightGBM [noise_all]` | torchregress | 0.8956 ± 0.0098 | 1.3732 ± 0.7044 | 2.8755 ± 1.3695 † | 0.6748 ± 0.3392 | 0.1336 ± 0.0511 | 15.9809 |
| `tr_Normalized+TabICL [noise_all]` | torchregress | 0.8894 ± 0.0090 | 1.4915 ± 0.8054 | 2.9272 ± 1.4375 † | 0.6764 ± 0.3369 | 0.2092 ± 0.0819 | 6.4647 |
| `tr_Normalized+TabularMLP [noise_all]` | torchregress | 0.8958 ± 0.0058 | 1.3763 ± 0.6486 | 2.7117 ± 1.3736 † | 0.6670 ± 0.3372 | 0.2242 ± 0.1033 | 14.2075 |
| `tr_PITRecal+KumoTabular [noise_all]` | torchregress | 0.8897 ± 0.0102 | 1.4731 ± 0.7328 | **2.4947 ± 1.2864** † | 0.6743 ± 0.3382 | 0.5016 ± 0.0410 | 4.7711 |
| `tr_PITRecal+LightGBM [noise_all]` | torchregress | 0.8944 ± 0.0035 | 1.5922 ± 0.7829 | 2.6655 ± 1.2581 † | 0.6804 ± 0.3415 | 0.4621 ± 0.0537 | 15.9809 |
| `tr_PITRecal+TabICL [noise_all]` | torchregress | 0.8861 ± 0.0149 | 1.5219 ± 0.7491 | 2.5957 ± 1.3096 † | 0.6785 ± 0.3393 | 0.4696 ± 0.0451 | 6.4647 |
| `tr_PITRecal+TabularMLP [noise_all]` | torchregress | 0.8972 ± 0.0096 | 1.5691 ± 0.8049 | 2.6263 ± 1.3123 † | 0.6842 ± 0.3433 | 0.4573 ± 0.0254 | 14.2075 |
| `tr_ResidualDist+KumoTabular [noise_all]` | torchregress | 0.8944 ± 0.0060 | 1.5076 ± 0.7857 | 2.6059 ± 1.2862 † | 0.6732 ± 0.3380 | 0.4494 ± 0.0794 | 4.7711 |
| `tr_ResidualDist+LightGBM [noise_all]` | torchregress | 0.8956 ± 0.0027 | 1.5391 ± 0.8148 | 2.6401 ± 1.3121 † | 0.6804 ± 0.3425 | 0.4480 ± 0.0785 | 15.9809 |
| `tr_ResidualDist+TabFM [noise_all]` | torchregress | 0.8925 ± 0.0058 | 1.5354 ± 0.7992 | 2.6587 ± 1.3265 † | 0.6692 ± 0.3393 | 0.4235 ± 0.0523 | 13.3537 |
| `tr_ResidualDist+TabICL [noise_all]` | torchregress | 0.8900 ± 0.0017 | 1.5150 ± 0.7849 | 2.6399 ± 1.2960 † | 0.6783 ± 0.3389 | 0.4487 ± 0.0732 | 6.4647 |
| `tr_ResidualDist+TabularMLP [noise_all]` | torchregress | 0.8919 ± 0.0035 | 1.5375 ± 0.8044 | 2.6387 ± 1.2980 † | 0.6792 ± 0.3419 | 0.4573 ± 0.0724 | 14.2075 |
| `tr_Split+KumoTabular [noise_all]` | torchregress | 0.8983 ± 0.0145 | 1.3807 ± 0.7382 | 2.8958 ± 1.3855 † | 0.6684 ± 0.3364 | 0.0464 ± 0.0269 | 4.7711 |
| `tr_Split+LightGBM [noise_all]` | torchregress | 0.8981 ± 0.0142 | 1.3878 ± 0.7288 | 2.9377 ± 1.4029 † | 0.6748 ± 0.3392 | 0.0369 ± 0.0333 | 15.9809 |
| `tr_Split+TabFM [noise_all]` | torchregress | 0.9019 ± 0.0088 | 1.4568 ± 0.7682 | 2.7186 ± 1.3271 † | **0.6607 ± 0.3381** | 0.2429 ± 0.0695 | 13.3537 |
| `tr_Split+TabICL [noise_all]` | torchregress | 0.8983 ± 0.0134 | 1.4038 ± 0.7420 | 2.9563 ± 1.4012 † | 0.6764 ± 0.3369 | 0.0313 ± 0.0283 | 6.4647 |
| `tr_Split+TabularMLP [noise_all]` | torchregress | 0.8975 ± 0.0118 | 1.3446 ± 0.6476 | 2.7256 ± 1.3554 † | 0.6670 ± 0.3372 | 0.1776 ± 0.0898 | 14.2075 |

##### fm_uq_scenarios / zurich_delays@noise_test

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [noise_test]` | sdm | 1.0000 ± 0.0000 | 99.6755 ± 40.2934 | 99.6755 ± 40.2934 † | 16.9424 ± 22.2262 | 1.0000 ± 0.0000 | 4.7285 |
| `Raw+LightGBM [noise_test]` | lightgbm | 0.8369 ± 0.0263 | 1.7522 ± 0.7176 | 2.9574 ± 1.4393 † | 0.6797 ± 0.3435 | 0.3181 ± 0.0648 | 2.2564 |
| `Raw+TabICL [noise_test]` | tabicl | 0.9994 ± 0.0010 | 22.7602 ± 3.7418 | 22.8068 ± 3.6691 † | 2.0300 ± 0.4837 | 0.9968 ± 0.0056 | 6.9405 |
| `Raw+TabularMLP [noise_test]` | torchregress | 0.9383 ± 0.0145 | 1.7612 ± 0.8641 | 2.7842 ± 1.3562 † | **0.6640 ± 0.3397** | 0.4388 ± 0.0936 | 20.2037 |
| `tr_CQR+KumoTabular [noise_test]` | torchregress | 1.0000 ± 0.0000 | 99.6270 ± 40.3161 | 99.6270 ± 40.3161 † | 16.9424 ± 22.2262 | 1.0000 ± 0.0000 | 4.7285 |
| `tr_CQR+LightGBM [noise_test]` | torchregress | 0.9025 ± 0.0145 | 1.9305 ± 0.8086 | 2.8950 ± 1.3844 † | 0.6797 ± 0.3435 | 0.5673 ± 0.0380 | 2.2564 |
| `tr_CQR+TabICL [noise_test]` | torchregress | 0.9994 ± 0.0010 | 22.7407 ± 3.7597 | 22.7875 ± 3.6866 † | 2.0300 ± 0.4837 | 0.9968 ± 0.0056 | 6.9405 |
| `tr_CQR+TabularMLP [noise_test]` | torchregress | 0.8681 ± 0.0156 | 1.3474 ± 0.6838 | 2.7205 ± 1.3489 † | **0.6640 ± 0.3397** | 0.2575 ± 0.0247 | 20.2037 |
| `tr_DCP+KumoTabular [noise_test]` | torchregress | 1.0000 ± 0.0000 | 91.3840 ± 36.4161 | 91.3840 ± 36.4161 † | 16.9424 ± 22.2262 | 1.0000 ± 0.0000 | 4.7285 |
| `tr_DCP+LightGBM [noise_test]` | torchregress | 0.8947 ± 0.0172 | 2.1658 ± 0.8781 | 3.0648 ± 1.4395 † | 0.6797 ± 0.3435 | 0.4780 ± 0.0955 | 2.2564 |
| `tr_DCP+TabICL [noise_test]` | torchregress | 0.9994 ± 0.0010 | 22.2904 ± 3.9059 | 22.3394 ± 3.8334 † | 2.0300 ± 0.4837 | 0.9968 ± 0.0056 | 6.9405 |
| `tr_DCP+TabularMLP [noise_test]` | torchregress | 0.8683 ± 0.0167 | 1.2675 ± 0.6391 | 2.6846 ± 1.3191 † | **0.6640 ± 0.3397** | 0.2186 ± 0.0137 | 20.2037 |
| `tr_Mondrian+KumoTabular [noise_test]` | torchregress | 0.1097 ± 0.1751 | 0.9918 ± 0.2154 | 322.7 ± 449.7 † | 16.9424 ± 22.2262 | 0.0000 ± 0.0000 | 4.7285 |
| `tr_Mondrian+LightGBM [noise_test]` | torchregress | 0.8944 ± 0.0281 | 1.5542 ± 0.7879 | 3.0298 ± 1.6540 † | 0.6797 ± 0.3435 | 0.2879 ± 0.1323 | 2.2564 |
| `tr_Mondrian+TabFM [noise_test]` | torchregress | 0.3194 ± 0.5533 | 1.6511 ± 1.6315 | 434.0 ± 518.1 † | 22.2934 ± 25.6345 | 0.2362 ± 0.4092 | 13.2524 |
| `tr_Mondrian+TabICL [noise_test]` | torchregress | 0.2272 ± 0.3409 | 1.7131 ± 1.7737 | 24.5290 ± 14.9104 † | 2.0300 ± 0.4837 | 0.0951 ± 0.1576 | 6.9405 |
| `tr_Mondrian+TabularMLP [noise_test]` | torchregress | 0.8767 ± 0.0079 | 1.3008 ± 0.6771 | 2.7276 ± 1.3936 † | **0.6640 ± 0.3397** | 0.2631 ± 0.0343 | 20.2037 |
| `tr_Normalized+KumoTabular [noise_test]` | torchregress | 1.0000 ± 0.0000 | 85.1053 ± 32.3283 | 85.1053 ± 32.3283 † | 16.9424 ± 22.2262 | 1.0000 ± 0.0000 | 4.7285 |
| `tr_Normalized+LightGBM [noise_test]` | torchregress | 0.9158 ± 0.0179 | 1.9992 ± 0.8672 | 3.1646 ± 1.5519 † | 0.6797 ± 0.3435 | 0.4026 ± 0.0713 | 2.2564 |
| `tr_Normalized+TabICL [noise_test]` | torchregress | 0.9994 ± 0.0010 | 21.8023 ± 4.9700 | 21.8576 ± 4.9102 † | 2.0300 ± 0.4837 | 0.9968 ± 0.0056 | 6.9405 |
| `tr_Normalized+TabularMLP [noise_test]` | torchregress | 0.8683 ± 0.0167 | 1.2675 ± 0.6391 | 2.6846 ± 1.3191 † | **0.6640 ± 0.3397** | 0.2186 ± 0.0137 | 20.2037 |
| `tr_PITRecal+KumoTabular [noise_test]` | torchregress | 1.0000 ± 0.0000 | 92.0589 ± 36.3348 | 92.0589 ± 36.3348 † | 18.0712 ± 23.1274 | 1.0000 ± 0.0000 | 4.7285 |
| `tr_PITRecal+LightGBM [noise_test]` | torchregress | 0.8989 ± 0.0173 | 2.1442 ± 0.8874 | 3.0548 ± 1.4438 † | 0.6787 ± 0.3451 | 0.5166 ± 0.0941 | 2.2564 |
| `tr_PITRecal+TabICL [noise_test]` | torchregress | 0.9992 ± 0.0008 | 22.2714 ± 4.0906 | 22.3227 ± 4.0199 † | 1.8798 ± 0.3001 | 0.9941 ± 0.0052 | 6.9405 |
| `tr_PITRecal+TabularMLP [noise_test]` | torchregress | 0.8686 ± 0.0111 | 1.3620 ± 0.6986 | **2.5906 ± 1.3047** † | 0.6791 ± 0.3417 | 0.3643 ± 0.0555 | 20.2037 |
| `tr_ResidualDist+KumoTabular [noise_test]` | torchregress | 0.2581 ± 0.4340 | 1.4772 ± 0.7653 | 316.5 ± 445.2 † | 16.9567 ± 22.2328 | 0.0129 ± 0.0224 | 4.7285 |
| `tr_ResidualDist+LightGBM [noise_test]` | torchregress | 0.8339 ± 0.0384 | 1.4957 ± 0.7966 | 2.7730 ± 1.2827 † | 0.6805 ± 0.3447 | 0.3780 ± 0.0530 | 2.2564 |
| `tr_ResidualDist+TabFM [noise_test]` | torchregress | 0.2458 ± 0.4258 | 1.5103 ± 0.7905 | 429.9 ± 517.4 † | 22.3284 ± 25.6525 | 0.0676 ± 0.1170 | 13.2524 |
| `tr_ResidualDist+TabICL [noise_test]` | torchregress | 0.0767 ± 0.0671 | 1.4551 ± 0.7539 | 24.3054 ± 1.6941 † | 2.0259 ± 0.4812 | 0.0061 ± 0.0106 | 6.9405 |
| `tr_ResidualDist+TabularMLP [noise_test]` | torchregress | 0.9000 ± 0.0093 | 1.5031 ± 0.7710 | 2.6868 ± 1.3356 † | 0.6755 ± 0.3420 | 0.3735 ± 0.0665 | 20.2037 |
| `tr_Split+KumoTabular [noise_test]` | torchregress | 0.1781 ± 0.2962 | 1.3544 ± 0.6995 | 320.4 ± 447.9 † | 16.9424 ± 22.2262 | 0.0026 ± 0.0046 | 4.7285 |
| `tr_Split+LightGBM [noise_test]` | torchregress | 0.8869 ± 0.0113 | 1.3654 ± 0.7074 | 2.8774 ± 1.4458 † | 0.6797 ± 0.3435 | 0.1246 ± 0.0484 | 2.2564 |
| `tr_Split+TabFM [noise_test]` | torchregress | 0.2636 ± 0.4566 | 1.4177 ± 0.7421 | 431.6 ± 518.0 † | 22.2934 ± 25.6345 | 0.1081 ± 0.1872 | 13.2524 |
| `tr_Split+TabICL [noise_test]` | torchregress | 0.0864 ± 0.0880 | 1.3977 ± 0.7283 | 25.1706 ± 7.8307 † | 2.0300 ± 0.4837 | 0.0031 ± 0.0053 | 6.9405 |
| `tr_Split+TabularMLP [noise_test]` | torchregress | 0.9017 ± 0.0060 | 1.4338 ± 0.7355 | 2.7602 ± 1.3527 † | **0.6640 ± 0.3397** | 0.2005 ± 0.0340 | 20.2037 |

##### fm_uq_scenarios / zurich_delays@outliers

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [outliers]` | sdm | 0.9653 ± 0.0065 | 2.5538 ± 1.0837 | 2.9770 ± 1.2988 † | **0.6496 ± 0.3280** | 0.7525 ± 0.0637 | 4.7615 |
| `Raw+LightGBM [outliers]` | lightgbm | 0.9161 ± 0.0176 | 2.6422 ± 0.4121 | 3.4097 ± 0.8694 † | 0.6620 ± 0.3339 | 0.5995 ± 0.0821 | 2.2383 |
| `Raw+TabICL [outliers]` | tabicl | 0.9678 ± 0.0099 | 2.6156 ± 0.8108 | 3.0686 ± 1.1418 † | 0.6558 ± 0.3281 | 0.7744 ± 0.0774 | 6.9728 |
| `Raw+TabularMLP [outliers]` | torchregress | 0.9947 ± 0.0049 | 7.1745 ± 0.7010 | 7.4494 ± 0.4245 † | 0.6659 ± 0.3330 | 0.9500 ± 0.0420 | 13.5614 |
| `tr_CQR+KumoTabular [outliers]` | torchregress | 0.9439 ± 0.0019 | 2.3692 ± 1.0443 | 2.8741 ± 1.2714 † | **0.6496 ± 0.3280** | 0.6797 ± 0.0424 | 4.7615 |
| `tr_CQR+LightGBM [outliers]` | torchregress | 0.9481 ± 0.0075 | 2.8622 ± 0.5730 | 3.4788 ± 0.9178 † | 0.6620 ± 0.3339 | 0.6607 ± 0.0947 | 2.2383 |
| `tr_CQR+TabICL [outliers]` | torchregress | 0.9417 ± 0.0022 | 2.4340 ± 0.8093 | 2.9642 ± 1.1483 † | 0.6558 ± 0.3281 | 0.6984 ± 0.0593 | 6.9728 |
| `tr_CQR+TabularMLP [outliers]` | torchregress | 0.9425 ± 0.0126 | 1.8992 ± 1.0162 | 2.8828 ± 1.4511 † | 0.6659 ± 0.3330 | 0.4310 ± 0.1132 | 13.5614 |
| `tr_DCP+KumoTabular [outliers]` | torchregress | 0.9433 ± 0.0017 | 2.0346 ± 1.0670 | **2.5928 ± 1.2956** † | **0.6496 ± 0.3280** | 0.6606 ± 0.0602 | 4.7615 |
| `tr_DCP+LightGBM [outliers]` | torchregress | 0.9469 ± 0.0070 | 3.2774 ± 0.7954 | 3.8478 ± 1.0434 † | 0.6620 ± 0.3339 | 0.6994 ± 0.0985 | 2.2383 |
| `tr_DCP+TabICL [outliers]` | torchregress | 0.9436 ± 0.0057 | 2.0724 ± 1.0850 | 2.6835 ± 1.3392 † | 0.6558 ± 0.3281 | 0.6543 ± 0.0531 | 6.9728 |
| `tr_DCP+TabularMLP [outliers]` | torchregress | 0.9397 ± 0.0101 | 1.8471 ± 1.0161 | 2.8729 ± 1.4659 † | 0.6659 ± 0.3330 | 0.3940 ± 0.0856 | 13.5614 |
| `tr_Mondrian+KumoTabular [outliers]` | torchregress | 0.9392 ± 0.0082 | 1.9935 ± 0.9642 | 2.8729 ± 1.3898 † | **0.6496 ± 0.3280** | 0.4719 ± 0.0365 | 4.7615 |
| `tr_Mondrian+LightGBM [outliers]` | torchregress | 0.9419 ± 0.0057 | 2.1490 ± 0.9037 | 3.0369 ± 1.3901 † | 0.6620 ± 0.3339 | 0.4844 ± 0.0689 | 2.2383 |
| `tr_Mondrian+TabFM [outliers]` | torchregress | 0.9436 ± 0.0013 | 1.9658 ± 0.9390 | 2.7837 ± 1.2726 † | 0.6583 ± 0.3155 | 0.5139 ± 0.0582 | 13.3136 |
| `tr_Mondrian+TabICL [outliers]` | torchregress | 0.9403 ± 0.0029 | 2.0356 ± 0.9930 | 2.9355 ± 1.3847 † | 0.6558 ± 0.3281 | 0.4700 ± 0.0475 | 6.9728 |
| `tr_Mondrian+TabularMLP [outliers]` | torchregress | 0.9475 ± 0.0017 | 2.2443 ± 1.0639 | 3.0613 ± 1.4489 † | 0.6659 ± 0.3330 | 0.5420 ± 0.0649 | 13.5614 |
| `tr_Normalized+KumoTabular [outliers]` | torchregress | 0.9383 ± 0.0145 | 2.1740 ± 1.1114 | 2.9592 ± 1.4117 † | **0.6496 ± 0.3280** | 0.4994 ± 0.1082 | 4.7615 |
| `tr_Normalized+LightGBM [outliers]` | torchregress | 0.9408 ± 0.0123 | 2.5498 ± 0.9700 | 3.3778 ± 1.3378 † | 0.6620 ± 0.3339 | 0.5423 ± 0.0461 | 2.2383 |
| `tr_Normalized+TabICL [outliers]` | torchregress | 0.9394 ± 0.0055 | 2.1993 ± 0.9839 | 3.0095 ± 1.3431 † | 0.6558 ± 0.3281 | 0.4907 ± 0.0675 | 6.9728 |
| `tr_Normalized+TabularMLP [outliers]` | torchregress | 0.9397 ± 0.0101 | 1.8471 ± 1.0161 | 2.8729 ± 1.4659 † | 0.6659 ± 0.3330 | 0.3940 ± 0.0856 | 13.5614 |
| `tr_PITRecal+KumoTabular [outliers]` | torchregress | 0.9469 ± 0.0021 | 2.2151 ± 1.1604 | 2.7252 ± 1.3580 † | 0.6540 ± 0.3297 | 0.6981 ± 0.0603 | 4.7615 |
| `tr_PITRecal+LightGBM [outliers]` | torchregress | 0.9489 ± 0.0138 | 3.3883 ± 0.8693 | 3.8870 ± 1.0260 † | 0.6674 ± 0.3357 | 0.7827 ± 0.0808 | 2.2383 |
| `tr_PITRecal+TabICL [outliers]` | torchregress | 0.9403 ± 0.0025 | 2.2202 ± 1.0880 | 2.7575 ± 1.3148 † | 0.6587 ± 0.3291 | 0.6912 ± 0.0077 | 6.9728 |
| `tr_PITRecal+TabularMLP [outliers]` | torchregress | 0.9569 ± 0.0025 | 2.1838 ± 1.0621 | 2.8579 ± 1.3777 † | 0.6786 ± 0.3368 | 0.7690 ± 0.0070 | 13.5614 |
| `tr_ResidualDist+KumoTabular [outliers]` | torchregress | 0.9511 ± 0.0017 | 2.1075 ± 1.0214 | 2.8003 ± 1.3719 † | 0.6527 ± 0.3294 | 0.7554 ± 0.0239 | 4.7615 |
| `tr_ResidualDist+LightGBM [outliers]` | torchregress | 0.9514 ± 0.0059 | 2.1739 ± 1.0778 | 2.8411 ± 1.3996 † | 0.6658 ± 0.3353 | 0.7669 ± 0.0132 | 2.2383 |
| `tr_ResidualDist+TabFM [outliers]` | torchregress | 0.9550 ± 0.0071 | 2.2578 ± 0.9985 | 2.9439 ± 1.3388 † | 0.6609 ± 0.3167 | 0.7779 ± 0.0769 | 13.3136 |
| `tr_ResidualDist+TabICL [outliers]` | torchregress | 0.9481 ± 0.0055 | 2.1314 ± 1.0485 | 2.8054 ± 1.3702 † | 0.6578 ± 0.3286 | 0.7493 ± 0.0097 | 6.9728 |
| `tr_ResidualDist+TabularMLP [outliers]` | torchregress | 0.9567 ± 0.0022 | 2.2222 ± 1.0956 | 2.8953 ± 1.4213 † | 0.6777 ± 0.3361 | 0.7584 ± 0.0219 | 13.5614 |
| `tr_Split+KumoTabular [outliers]` | torchregress | 0.9400 ± 0.0112 | 1.9791 ± 1.0440 | 2.9604 ± 1.4570 † | **0.6496 ± 0.3280** | 0.4042 ± 0.0894 | 4.7615 |
| `tr_Split+LightGBM [outliers]` | torchregress | 0.9392 ± 0.0118 | 2.0088 ± 1.0688 | 3.0234 ± 1.5081 † | 0.6620 ± 0.3339 | 0.3797 ± 0.0876 | 2.2383 |
| `tr_Split+TabFM [outliers]` | torchregress | 0.9456 ± 0.0032 | 2.0138 ± 0.9566 | 2.8728 ± 1.3335 † | 0.6583 ± 0.3155 | 0.5540 ± 0.0623 | 13.3136 |
| `tr_Split+TabICL [outliers]` | torchregress | 0.9433 ± 0.0139 | 2.0437 ± 1.1042 | 3.0170 ± 1.4748 † | 0.6558 ± 0.3281 | 0.4235 ± 0.1129 | 6.9728 |
| `tr_Split+TabularMLP [outliers]` | torchregress | 0.9386 ± 0.0098 | 1.8286 ± 1.0028 | 2.8792 ± 1.4740 † | 0.6659 ± 0.3330 | 0.3789 ± 0.0754 | 13.5614 |

##### fm_uq_scenarios / zurich_delays@shift

| Method | Library | Coverage90 | Width | IntervalScore | RMSE | WorstDecileCoverage | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Raw+KumoTabular [shift]` | sdm | 0.9008 ± 0.0033 | 1.5037 ± 0.8203 | **2.7001 ± 0.5977** † | 1.2612 ± 0.7608 | 0.4494 ± 0.1062 | 4.7446 |
| `Raw+LightGBM [shift]` | lightgbm | 0.8397 ± 0.0055 | 1.1950 ± 0.6479 | 2.9624 ± 0.8381 † | 1.2778 ± 0.7474 | 0.2343 ± 0.0942 | 2.2334 |
| `Raw+TabICL [shift]` | tabicl | 0.8950 ± 0.0096 | 1.4604 ± 0.8451 | 2.7095 ± 0.6003 † | 1.2519 ± 0.7690 | 0.4127 ± 0.0197 | 6.9448 |
| `Raw+TabularMLP [shift]` | torchregress | 0.9297 ± 0.0079 | 1.8688 ± 1.0455 | 3.1165 ± 0.9076 † | 1.2538 ± 0.7657 | 0.4727 ± 0.0576 | 19.3335 |
| `tr_CQR+KumoTabular [shift]` | torchregress | 0.9008 ± 0.0046 | 1.5020 ± 0.8128 | 2.7002 ± 0.5972 † | 1.2612 ± 0.7608 | 0.4397 ± 0.1221 | 4.7446 |
| `tr_CQR+LightGBM [shift]` | torchregress | 0.9106 ± 0.0163 | 1.4278 ± 0.7497 | 2.9195 ± 0.8054 † | 1.2778 ± 0.7474 | 0.3928 ± 0.0627 | 2.2334 |
| `tr_CQR+TabICL [shift]` | torchregress | 0.9067 ± 0.0079 | 1.5169 ± 0.8644 | 2.7099 ± 0.6024 † | 1.2519 ± 0.7690 | 0.4556 ± 0.0242 | 6.9448 |
| `tr_CQR+TabularMLP [shift]` | torchregress | 0.8897 ± 0.0284 | 1.6300 ± 0.8866 | 3.1001 ± 0.9258 † | 1.2538 ± 0.7657 | 0.4031 ± 0.0466 | 19.3335 |
| `tr_DCP+KumoTabular [shift]` | torchregress | 0.8992 ± 0.0058 | 1.4923 ± 0.8022 | 2.7049 ± 0.6000 † | 1.2612 ± 0.7608 | 0.4425 ± 0.1338 | 4.7446 |
| `tr_DCP+LightGBM [shift]` | torchregress | 0.9047 ± 0.0108 | 1.5768 ± 0.8350 | 2.9087 ± 0.7880 † | 1.2778 ± 0.7474 | 0.4949 ± 0.0586 | 2.2334 |
| `tr_DCP+TabICL [shift]` | torchregress | 0.9058 ± 0.0074 | 1.5420 ± 0.8807 | 2.7114 ± 0.6025 † | 1.2519 ± 0.7690 | 0.4620 ± 0.0420 | 6.9448 |
| `tr_DCP+TabularMLP [shift]` | torchregress | 0.8967 ± 0.0274 | 1.5442 ± 0.8086 | 3.0535 ± 0.8795 † | 1.2538 ± 0.7657 | 0.3686 ± 0.0471 | 19.3335 |
| `tr_Mondrian+KumoTabular [shift]` | torchregress | 0.9056 ± 0.0079 | 1.7124 ± 1.0500 | 3.2954 ± 1.0822 † | 1.2612 ± 0.7608 | 0.3404 ± 0.1486 | 4.7446 |
| `tr_Mondrian+LightGBM [shift]` | torchregress | 0.9064 ± 0.0096 | 1.6250 ± 0.9130 | 3.3741 ± 1.1151 † | 1.2778 ± 0.7474 | 0.3169 ± 0.1326 | 2.2334 |
| `tr_Mondrian+TabFM [shift]` | torchregress | 0.9067 ± 0.0025 | 1.5372 ± 0.8629 | 3.0229 ± 0.8530 † | **1.2334 ± 0.7834** | 0.3966 ± 0.1018 | 13.2982 |
| `tr_Mondrian+TabICL [shift]` | torchregress | 0.9056 ± 0.0064 | 1.7099 ± 1.0336 | 3.2859 ± 1.0424 † | 1.2519 ± 0.7690 | 0.3374 ± 0.1269 | 6.9448 |
| `tr_Mondrian+TabularMLP [shift]` | torchregress | 0.9042 ± 0.0138 | 1.5165 ± 0.8622 | 3.0888 ± 0.9030 † | 1.2538 ± 0.7657 | 0.3565 ± 0.1044 | 19.3335 |
| `tr_Normalized+KumoTabular [shift]` | torchregress | 0.8992 ± 0.0175 | 1.5235 ± 0.8168 | 3.0962 ± 0.8806 † | 1.2612 ± 0.7608 | 0.3304 ± 0.0636 | 4.7446 |
| `tr_Normalized+LightGBM [shift]` | torchregress | 0.9081 ± 0.0200 | 1.5325 ± 0.7929 | 3.2434 ± 1.0364 † | 1.2778 ± 0.7474 | 0.2893 ± 0.0669 | 2.2334 |
| `tr_Normalized+TabICL [shift]` | torchregress | 0.9047 ± 0.0135 | 1.5668 ± 0.8358 | 3.1007 ± 0.8545 † | 1.2519 ± 0.7690 | 0.3246 ± 0.0310 | 6.9448 |
| `tr_Normalized+TabularMLP [shift]` | torchregress | 0.8967 ± 0.0274 | 1.5442 ± 0.8086 | 3.0535 ± 0.8795 † | 1.2538 ± 0.7657 | 0.3686 ± 0.0471 | 19.3335 |
| `tr_PITRecal+KumoTabular [shift]` | torchregress | 0.8967 ± 0.0073 | 1.4757 ± 0.7591 | 2.7125 ± 0.6195 † | 1.2603 ± 0.7624 | 0.4709 ± 0.0884 | 4.7446 |
| `tr_PITRecal+LightGBM [shift]` | torchregress | 0.9022 ± 0.0082 | 1.5438 ± 0.7799 | 2.9095 ± 0.7925 † | 1.2777 ± 0.7488 | 0.4957 ± 0.0515 | 2.2334 |
| `tr_PITRecal+TabICL [shift]` | torchregress | 0.9033 ± 0.0044 | 1.5668 ± 0.8466 | 2.7126 ± 0.6018 † | 1.2440 ± 0.7777 | 0.4879 ± 0.0722 | 6.9448 |
| `tr_PITRecal+TabularMLP [shift]` | torchregress | 0.8964 ± 0.0092 | 1.6784 ± 0.8905 | 2.9251 ± 0.7449 † | 1.2716 ± 0.7499 | 0.4793 ± 0.0815 | 19.3335 |
| `tr_ResidualDist+KumoTabular [shift]` | torchregress | 0.8994 ± 0.0235 | 1.5196 ± 0.7338 | 3.1130 ± 0.9370 † | 1.2612 ± 0.7612 | 0.4730 ± 0.0299 | 4.7446 |
| `tr_ResidualDist+LightGBM [shift]` | torchregress | 0.9072 ± 0.0214 | 1.5672 ± 0.7498 | 3.1281 ± 0.9221 † | 1.2784 ± 0.7475 | 0.4904 ± 0.0566 | 2.2334 |
| `tr_ResidualDist+TabFM [shift]` | torchregress | 0.9042 ± 0.0325 | 1.5496 ± 0.7127 | 3.1115 ± 0.9127 † | 1.2376 ± 0.7801 | 0.4813 ± 0.0369 | 13.2982 |
| `tr_ResidualDist+TabICL [shift]` | torchregress | 0.9058 ± 0.0219 | 1.5500 ± 0.7415 | 3.0830 ± 0.8944 † | 1.2497 ± 0.7714 | 0.5083 ± 0.0506 | 6.9448 |
| `tr_ResidualDist+TabularMLP [shift]` | torchregress | 0.9075 ± 0.0257 | 1.5691 ± 0.7470 | 3.1314 ± 0.9254 † | 1.2580 ± 0.7620 | 0.4814 ± 0.0465 | 19.3335 |
| `tr_Split+KumoTabular [shift]` | torchregress | 0.9083 ± 0.0191 | 1.4569 ± 0.6904 | 3.3753 ± 1.1096 † | 1.2612 ± 0.7608 | 0.1650 ± 0.0502 | 4.7446 |
| `tr_Split+LightGBM [shift]` | torchregress | 0.9067 ± 0.0177 | 1.4710 ± 0.7285 | 3.4435 ± 1.1647 † | 1.2778 ± 0.7474 | 0.1199 ± 0.0577 | 2.2334 |
| `tr_Split+TabFM [shift]` | torchregress | 0.9075 ± 0.0246 | 1.4847 ± 0.6709 | 3.1547 ± 0.9242 † | **1.2334 ± 0.7834** | 0.3067 ± 0.0642 | 13.2982 |
| `tr_Split+TabICL [shift]` | torchregress | 0.9067 ± 0.0202 | 1.4916 ± 0.7108 | 3.3887 ± 1.0711 † | 1.2519 ± 0.7690 | 0.1327 ± 0.0500 | 6.9448 |
| `tr_Split+TabularMLP [shift]` | torchregress | 0.9075 ± 0.0290 | 1.4580 ± 0.6786 | 3.1967 ± 0.9788 † | 1.2538 ± 0.7657 | 0.2822 ± 0.0232 | 19.3335 |
| `tr_WeightedCQR+KumoTabular [shift]` | torchregress | 0.9108 ± 0.0065 | 1.5377 ± 0.8135 | 2.7050 ± 0.6000 † | 1.2612 ± 0.7608 | 0.4585 ± 0.1289 | 4.7446 |
| `tr_WeightedCQR+LightGBM [shift]` | torchregress | 0.9139 ± 0.0114 | 1.4459 ± 0.7623 | 2.9133 ± 0.7954 † | 1.2778 ± 0.7474 | 0.3996 ± 0.0523 | 2.2334 |
| `tr_WeightedCQR+TabICL [shift]` | torchregress | 0.9122 ± 0.0034 | 1.5354 ± 0.8462 | 2.7140 ± 0.6045 † | 1.2519 ± 0.7690 | 0.4531 ± 0.0047 | 6.9448 |
| `tr_WeightedCQR+TabularMLP [shift]` | torchregress | 0.9050 ± 0.0210 | 1.6900 ± 0.9251 | 3.0882 ± 0.8993 † | 1.2538 ± 0.7657 | 0.4178 ± 0.0454 | 19.3335 |
| `tr_WeightedSplit+KumoTabular [shift]` | torchregress | 0.9072 ± 0.0104 | 1.4822 ± 0.7868 | 3.3619 ± 1.0987 † | 1.2612 ± 0.7608 | 0.1615 ± 0.0945 | 4.7446 |
| `tr_WeightedSplit+LightGBM [shift]` | torchregress | 0.9061 ± 0.0106 | 1.4865 ± 0.8332 | 3.4297 ± 1.1547 † | 1.2778 ± 0.7474 | 0.1358 ± 0.1096 | 2.2334 |
| `tr_WeightedSplit+TabFM [shift]` | torchregress | 0.9078 ± 0.0135 | 1.4874 ± 0.7622 | 3.1368 ± 0.9129 † | **1.2334 ± 0.7834** | 0.3150 ± 0.0967 | 13.2982 |
| `tr_WeightedSplit+TabICL [shift]` | torchregress | 0.9064 ± 0.0089 | 1.4957 ± 0.8153 | 3.3698 ± 1.0583 † | 1.2519 ± 0.7690 | 0.1435 ± 0.1216 | 6.9448 |
| `tr_WeightedSplit+TabularMLP [shift]` | torchregress | 0.9056 ± 0.0139 | 1.4806 ± 0.7824 | 3.1804 ± 0.9648 † | 1.2538 ± 0.7657 | 0.2874 ± 0.0889 | 19.3335 |

#### foundation_model (fm)

Group `fm`; 4 dataset(s). Columns: RMSE ↓, R2 ↑, NLL ↓, CRPS ↓, Coverage ·, IntervalScore ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### foundation_model / california_housing

| Method | Library | RMSE | R2 | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.5980 ± 0.0055 | 0.7315 ± 0.0049 | 4.1694 ± 2.9616 | 0.2889 ± 0.0022 | - | - | 17.6328 |
| `KumoTabular` | structured-data-models | **0.4323 ± 0.0035** | **0.8597 ± 0.0023** | **0.0733 ± 0.0098** | **0.1826 ± 0.0011** | 0.9062 ± 0.0010 | **1.4676 ± 0.0228** | 0.8691 |
| `MDN` | torchregress | 0.5863 ± 0.0078 | 0.7419 ± 0.0069 | 0.7130 ± 0.0192 | 0.2830 ± 0.0018 | - | - | 19.8062 |
| `TabFM` | structured-data-models | 0.4415 ± 0.0005 | 0.8537 ± 0.0003 | - | - | - | - | 6.3676 |
| `TabICL` | tabicl | 0.4740 ± 0.0000 | 0.8313 ± 0.0000 | 0.2154 ± 0.0000 | 0.2043 ± 0.0000 | 0.9000 ± 0.0000 | 1.6478 ± 0.0000 | 0.4703 |

##### foundation_model / diabetes

| Method | Library | RMSE | R2 | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 52.7822 ± 0.7438 | 0.4839 ± 0.0145 | 5.4028 ± 0.0266 | 29.6520 ± 0.4380 | - | - | 1.8793 |
| `KumoTabular` | structured-data-models | 51.3266 ± 0.3092 | 0.5120 ± 0.0059 | **5.3006 ± 0.0040** | **28.4194 ± 0.1517** | 0.9198 ± 0.0043 | **192.0 ± 1.0** | 2.4542 |
| `MDN` | torchregress | 54.0776 ± 0.7190 | 0.4582 ± 0.0144 | 5.4411 ± 0.0460 | 30.3096 ± 0.3288 | - | - | 1.7771 |
| `TabFM` | structured-data-models | **51.2376 ± 0.0328** | **0.5137 ± 0.0006** | - | - | - | - | 10.0075 |
| `TabICL` | tabicl | 51.3603 ± 0.0000 | 0.5113 ± 0.0000 | 5.3083 ± 0.0000 | 28.4414 ± 0.0000 | 0.9173 ± 0.0000 | 195.2 ± 0.0 | 0.4259 |

##### foundation_model / house_sales

| Method | Library | RMSE | R2 | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 1.233e+05 ± 4.004e+03 | 0.8881 ± 0.0073 | 12.6774 ± 0.0117 | 50418.2 ± 474.6 | - | - | 66.5995 |
| `KumoTabular` ‡ | structured-data-models | 1.014e+05 ± 1.089e+03 | 0.9244 ± 0.0016 | 12.3618 ± 0.0007 | 39485.9 ± 43.6 | 0.8949 ± 0.0020 | 3.164e+05 ± 1.268e+03 | 69.1299 |
| `MDN` | torchregress | 1.254e+05 ± 2.162e+03 | 0.8843 ± 0.0040 | 12.6669 ± 0.0127 | 50365.9 ± 97.0 | - | - | 75.6513 |
| `TabFM` ‡ | structured-data-models | **99083.1 ± 291.1** | **0.9278 ± 0.0004** | - | - | - | - | 97.9808 |
| `TabICL` | tabicl | 1.119e+05 ± 0.000e+00 | 0.9080 ± 0.0000 | 12.4301 ± 0.0000 | 42726.7 ± 0.0 | 0.8970 ± 0.0000 | 3.420e+05 ± 0.000e+00 | 0.7049 |
| `TabPFN` | tabpfn | 1.001e+05 ± 0.000e+00 (n=2) | 0.9264 ± 0.0000 (n=2) | **12.3612 ± 0.0000** (n=2) | **39270.9 ± 0.0** (n=2) | 0.9027 ± 0.0000 (n=2) | **3.147e+05 ± 0.000e+00** (n=2) | 0.6123 |

‡ ran on CPU after a CUDA failure (cpu retry).

##### foundation_model / openml_year

| Method | Library | RMSE | R2 | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.1636 ± 0.0028 | 0.7827 ± 0.0075 | -0.4466 ± 0.0162 | 0.0857 ± 0.0016 | - | - | 33.5385 |
| `KumoTabular` | structured-data-models | **0.1113 ± 0.0008** | **0.8994 ± 0.0014** | **-1.1082 ± 0.0027** | **0.0517 ± 0.0003** | 0.9140 ± 0.0009 | **0.4124 ± 0.0022** | 0.5509 |
| `MDN` | torchregress | 0.1596 ± 0.0020 | 0.7933 ± 0.0051 | -0.4696 ± 0.0410 | 0.0824 ± 0.0009 | - | - | 38.9536 |
| `TabFM` ‡ | structured-data-models | 0.1132 ± 0.0000 | 0.8960 ± 0.0000 | - | - | - | - | 32.4427 |
| `TabICL` | tabicl | 0.1221 ± 0.0000 | 0.8791 ± 0.0000 | -0.9538 ± 0.0000 | 0.0583 ± 0.0000 | 0.9070 ± 0.0000 | 0.4724 ± 0.0000 | 0.3643 |
| `TabPFN` | tabpfn | 0.1133 ± 0.0000 | 0.8959 ± 0.0000 | -1.0773 ± 0.0000 | 0.0521 ± 0.0000 | 0.9203 ± 0.0000 | 0.4195 ± 0.0000 | 0.4653 |

‡ ran on CPU after a CUDA failure (cpu retry).

### Group `image`

#### age_regression (image)

Group `image`; 5 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑, NLL ↓, CRPS ↓, Coverage ·, Width ·, IntervalScore ↓, ECE ↓, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### age_regression / ixi_brainage

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CNN+Barron` | torchregress | 16.1743 ± 0.3816 | 13.7731 ± 0.6362 | 0.1949 ± 0.0382 | - | - | - | - | - | - | - | 3.0667 |
| `CNN+BetaNLL` | torchregress | 17.0727 ± 0.1951 | 15.2731 ± 0.1095 | 0.1032 ± 0.0205 | 4.2697 ± 0.0200 | 10.0298 ± 0.1314 | 0.9375 ± 0.0433 | 57.1338 ± 3.4342 | 59.4206 ± 2.0890 | 0.0102 ± 0.0009 | 0.1563 ± 0.0174 | 2.8504 |
| `CNN+CRPS` | torchregress | 16.6921 ± 0.1553 | 14.9153 ± 0.1832 | 0.1428 ± 0.0159 | 4.2611 ± 0.0043 | 9.8143 ± 0.1430 | 0.9292 ± 0.0921 | 58.5143 ± 12.6334 | 62.2556 ± 7.0702 | 0.0109 ± 0.0015 | 0.1467 ± 0.0446 | 3.2339 |
| `CNN+Evidential` | torchregress | 18.2498 ± 0.5020 | 16.3841 ± 0.3179 | -0.0251 ± 0.0563 | - | - | 1.0000 ± 0.0000 | 873.3 ± 16.9 | 873.3 ± 16.9 | 0.0012 ± 0.0002 | 0.4229 ± 0.0005 | 3.4643 |
| `CNN+FaithfulGaussian` | torchregress | 15.6860 ± 0.3147 | 14.0030 ± 0.3344 | 0.2429 ± 0.0304 | 20.7503 ± 1.0164 | 3.588e+08 ± 2.404e+08 | 1.0000 ± 0.0000 | 5.051e+09 ± 3.384e+09 | 5.051e+09 ± 3.384e+09 | 1.036e-08 ± 1.301e-08 | 0.4937 ± 0.0000 | 2.9609 |
| `CNN+GaussianNLL` | torchregress | 16.8909 ± 0.0887 | 15.2255 ± 0.1201 | 0.1223 ± 0.0092 | **4.2584 ± 0.0192** | 9.9283 ± 0.0732 | 0.9500 ± 0.0451 | 57.3337 ± 7.8456 | 59.0314 ± 6.3880 | 0.0103 ± 0.0010 | **0.1302 ± 0.0195** | 3.2645 |
| `CNN+Huber` | torchregress | 15.7968 ± 0.7835 | 13.6747 ± 0.8403 | 0.2311 ± 0.0754 | - | - | - | - | - | - | - | 2.4570 |
| `CNN+MC_Dropout` | torchregress | 17.3061 ± 0.3214 | 15.5390 ± 0.0969 | 0.0784 ± 0.0344 | 5.7308 ± 0.3824 | 11.9128 ± 0.1988 | 0.3500 ± 0.0696 | 25.3348 ± 2.1248 | 121.9 ± 9.0 | 0.0319 ± 0.0062 | 0.3312 ± 0.0250 | 2.4834 |
| `CNN+MDN_K2` | torchregress | 16.1423 ± 0.1011 | 14.3391 ± 0.1726 | 0.1984 ± 0.0100 | - | - | 0.8833 ± 0.0260 | 48.6769 ± 2.6688 | 55.4524 ± 1.0488 | 0.0140 ± 0.0018 | 0.1419 ± 0.0086 | 3.0460 |
| `CNN+MDN_K3` | torchregress | 16.6097 ± 0.2600 | 14.7026 ± 0.0699 | 0.1512 ± 0.0265 | - | - | 0.8750 ± 0.0250 | 47.6161 ± 2.7469 | 56.7661 ± 2.6079 | 0.0115 ± 0.0004 | 0.1880 ± 0.0567 | 3.0853 |
| `CNN+MDN_K5` | torchregress | 16.2482 ± 0.3868 | 14.5317 ± 0.3854 | 0.1875 ± 0.0388 | - | - | 0.8708 ± 0.0289 | 47.4590 ± 1.2634 | 55.4838 ± 1.3962 | 0.0124 ± 0.0031 | 0.1510 ± 0.0309 | 3.1089 |
| `CNN+MultiQuantile` | torchregress | 16.3036 ± 0.3413 | 14.4869 ± 0.3404 | 0.1821 ± 0.0340 | - | - | 0.8583 ± 0.0289 | 44.5163 ± 2.2695 | **53.9359 ± 2.5214** | 0.0109 ± 0.0002 | 0.1359 ± 0.0334 | 2.6111 |
| `CNN+NF_Ensemble_K2` | torchregress | 51.4817 ± 0.2967 | 48.2185 ± 0.3162 | -7.1535 ± 0.0939 | - | - | 0.0000 ± 0.0000 | 3.1211 ± 0.3404 | 937.6 ± 5.6 | **0.0000 ± 0.0000** | 0.9937 ± 0.0000 | 24.0046 |
| `CNN+NormalizingFlow` | torchregress | 51.5351 ± 0.1880 | 48.2769 ± 0.2009 | -7.1703 ± 0.0596 | - | - | 0.0000 ± 0.0000 | 3.0546 ± 0.4557 | 937.8 ± 7.8 | **0.0000 ± 0.0000** | 0.9937 ± 0.0000 | 12.0021 |
| `CNN+ObservedStdNLL` | torchregress | 25.5074 ± 12.1244 † | 22.8560 ± 10.6191 | -1.3030 ± 2.1638 | 4.6141 ± 0.4424 | 15.1149 ± 7.1227 | 0.9167 ± 0.0191 | 81.7154 ± 44.2908 | 87.1950 ± 44.2368 | 0.0077 ± 0.0033 | 0.2911 ± 0.2313 | 2.7869 |
| `CNN+PoissonGaussian` | torchregress | 15.1306 ± 3.1476 † | 12.3690 ± 2.6643 | 0.2754 ± 0.3106 | - | - | - | - | - | - | - | 3.1545 |
| `CNN+SLS` | torchregress | 49.7994 ± 0.2106 | 46.4209 ± 0.2259 | -6.6293 ± 0.0645 | - | - | 0.9417 ± 0.0629 | 146.1 ± 28.1 | 151.5 ± 23.1 | 0.0038 ± 0.0016 | 0.6349 ± 0.0225 | 5.6714 |
| `CNN+StudentT` | torchregress | 18.6545 ± 2.4705 † | 15.6027 ± 1.1409 | -0.0830 ± 0.2855 | - | - | - | - | - | - | - | 2.4756 |
| `ResNet+Evidential` | torchregress | 10.6576 ± 0.5335 † | **8.7741 ± 0.3402** | 0.6500 ± 0.0350 | - | - | 1.0000 ± 0.0000 | 144.8 ± 25.4 | 144.8 ± 25.4 | 0.0063 ± 0.0031 | 0.2578 ± 0.0228 | 8.3239 |
| `ResNet+GaussianNLL` | torchregress | 11.5590 ± 0.1189 | 9.2728 ± 0.1304 | 0.5889 ± 0.0085 | 18.1356 ± 14.0770 | **8.1181 ± 0.3932** | 0.2792 ± 0.0629 | 8.2488 ± 3.1526 | 125.3 ± 20.7 | 0.0149 ± 0.0029 | 0.4944 ± 0.0922 | 7.8533 |
| `ResNet+MDN_K3` | torchregress | **10.3245 ± 0.3889** | 8.7968 ± 0.3854 | **0.6718 ± 0.0246** | - | - | 0.1708 ± 0.0505 | 7.2764 ± 2.7019 | 119.3 ± 11.5 | 0.0257 ± 0.0136 | 0.4208 ± 0.0753 | 8.0988 |
| `ResNet+MultiQuantile` | torchregress | 11.9684 ± 1.4284 † | 9.6154 ± 1.1106 | 0.5552 ± 0.1088 | - | - | 0.3375 ± 0.0650 | 9.8478 ± 2.1725 | 121.4 ± 7.4 | 0.0481 ± 0.0268 | 0.4894 ± 0.0685 | 7.4817 |
| `ResNet+NormalizingFlow` | torchregress | 51.4462 ± 0.3021 | 48.1804 ± 0.3178 | -7.1423 ± 0.0955 | - | - | 0.0000 ± 0.0000 | 3.2758 ± 0.2448 | 936.1 ± 7.7 | **0.0000 ± 0.0000** | 0.9937 ± 0.0000 | 14.8496 |
| `sklearn_GBR` | scikit-learn | 11.2159 ± 0.0667 | 8.8822 ± 0.0203 | 0.6130 ± 0.0046 | - | - | - | - | - | - | - | 11.3635 |
| `sklearn_RF` | scikit-learn | 11.8732 ± 0.1130 | 9.7340 ± 0.1168 | 0.5663 ± 0.0082 | - | - | - | - | - | - | - | 2.0396 |
| `timm_resnet18+GaussianNLL` | timm | 11.9945 ± 0.4188 | 9.5675 ± 0.2319 | 0.5571 ± 0.0310 | 14.8417 ± 2.6449 | 8.3968 ± 0.2304 | 0.2542 ± 0.0832 | 7.7618 ± 1.0902 | 130.8 ± 6.6 | 0.0192 ± 0.0060 | 0.4649 ± 0.0248 | 7.7255 |
| `torch_uncertainty` | torch-uncertainty | 21.9184 ± 5.9441 † | 18.8448 ± 3.6834 | -0.5504 ± 0.8643 | 4.5153 ± 0.2597 | 12.9405 ± 3.3356 | 0.9000 ± 0.0331 | 68.9046 ± 22.9918 | 78.0991 ± 26.5875 | 0.0078 ± 0.0037 | 0.2422 ± 0.1491 | 8.2521 |

Failed / skipped:

- `CNN+RankNContrast` (failed, seed 0, 1, 2): failed: RuntimeError: Expected all tensors to be on the same device, but got b is on cuda:0, different from other tenso…

##### age_regression / ixi_brainage_tiny

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CNN+Barron` | torchregress | 14.9707 ± 0.2459 | 13.1097 ± 0.1100 | 0.0492 ± 0.0314 | - | - | - | - | - | - | - | 0.6050 |
| `CNN+BetaNLL` | torchregress | 38.9250 ± 0.9751 | 35.8137 ± 1.0560 | -5.4291 ± 0.3220 | 5.0895 ± 0.0804 | 22.8398 ± 0.6371 | 1.0000 ± 0.0000 | 142.4 ± 35.9 | 142.4 ± 35.9 | non-finite (3) | 0.6010 ± 0.0209 | 0.4928 |
| `CNN+CRPS` | torchregress | 15.0358 ± 0.1435 | 13.2202 ± 0.0941 | 0.0411 ± 0.0183 | **4.1726 ± 0.0484** | **8.8711 ± 0.0913** | 1.0000 ± 0.0000 | 60.8092 ± 9.3295 | 60.8092 ± 9.3295 | non-finite (3) | **0.1334 ± 0.0072** | 0.6268 |
| `CNN+Evidential` | torchregress | 15.5933 ± 0.6261 | 13.7720 ± 0.4260 | -0.0324 ± 0.0835 | - | - | 1.0000 ± 0.0000 | 665.1 ± 30.1 | 665.1 ± 30.1 | non-finite (3) | 0.4061 ± 0.0113 | 0.7234 |
| `CNN+FaithfulGaussian` | torchregress | 15.2554 ± 0.2712 | 13.3544 ± 0.0894 | 0.0127 ± 0.0350 | 24.3849 ± 1.1466 | 5.828e+09 ± 5.310e+09 | 1.0000 ± 0.0000 | 8.205e+10 ± 7.476e+10 | 8.205e+10 ± 7.476e+10 | non-finite (3) | 0.4583 ± 0.0000 | 0.5624 |
| `CNN+GaussianNLL` | torchregress | 37.9748 ± 1.1347 | 34.7818 ± 1.2344 | -5.1201 ± 0.3654 | 5.1164 ± 0.0443 | 22.2952 ± 0.6794 | 1.0000 ± 0.0000 | 168.8 ± 13.8 | 168.8 ± 13.8 | non-finite (3) | 0.5688 ± 0.0143 | 1.1021 |
| `CNN+Huber` | torchregress | 14.9705 ± 0.2461 | 13.1094 ± 0.1101 | 0.0493 ± 0.0314 | - | - | - | - | - | - | - | 0.4875 |
| `CNN+MC_Dropout` | torchregress | 14.8684 ± 0.8270 † | 12.9842 ± 0.7897 | 0.0604 ± 0.1060 | 4.3523 ± 0.1433 | 9.2908 ± 0.6748 | 0.5556 ± 0.0481 | 30.9955 ± 1.1020 | 64.8087 ± 10.5303 | non-finite (3) | 0.2528 ± 0.0674 | 0.5109 |
| `CNN+MDN_K2` | torchregress | 14.7180 ± 0.3865 | 13.0142 ± 0.4462 | 0.0808 ± 0.0486 | - | - | 1.0000 ± 0.0000 | 49.5728 ± 6.8186 | 49.5728 ± 6.8186 | non-finite (3) | 0.1610 ± 0.0246 | 0.6194 |
| `CNN+MDN_K3` | torchregress | 14.9648 ± 0.3253 | 13.2622 ± 0.2661 | 0.0499 ± 0.0415 | - | - | 0.9722 ± 0.0481 | 54.9855 ± 16.8572 | 55.0796 ± 16.7542 | non-finite (3) | 0.1793 ± 0.0223 | 0.5848 |
| `CNN+MDN_K5` | torchregress | 14.7241 ± 0.5577 | 13.0763 ± 0.4903 | 0.0796 ± 0.0703 | - | - | 1.0000 ± 0.0000 | 54.8033 ± 4.9413 | 54.8033 ± 4.9413 | non-finite (3) | 0.1385 ± 0.0380 | 0.6109 |
| `CNN+MultiQuantile` | torchregress | 16.2141 ± 0.8928 † | 14.0697 ± 0.5516 | -0.1173 ± 0.1215 | - | - | 0.9444 ± 0.0962 | 48.4556 ± 1.2598 | **49.5034 ± 1.1469** | non-finite (3) | 0.1680 ± 0.0555 | 0.5166 |
| `CNN+NF_Ensemble_K2` | torchregress | 48.2945 ± 0.3004 | 45.7858 ± 0.3144 | -8.8927 ± 0.1229 | - | - | 0.0000 ± 0.0000 | 3.1027 ± 0.3203 | 889.2 ± 5.4 | non-finite (3) | 0.9583 ± 0.0000 | 4.8101 |
| `CNN+NormalizingFlow` | torchregress | 48.3506 ± 0.1862 | 45.8497 ± 0.1927 | -8.9156 ± 0.0764 | - | - | 0.0000 ± 0.0000 | 3.0354 ± 0.4160 | 889.4 ± 7.2 | non-finite (3) | 0.9583 ± 0.0000 | 2.4876 |
| `CNN+ObservedStdNLL` | torchregress | 38.9662 ± 2.2890 † | 35.8395 ± 2.4859 | -5.4548 ± 0.7486 | 12.1381 ± 0.8898 | 22102.6 ± 14328.4 | 1.0000 ± 0.0000 | 3.112e+05 ± 2.017e+05 | 3.112e+05 ± 2.017e+05 | non-finite (3) | 0.4584 ± 0.0001 | 0.5535 |
| `CNN+PoissonGaussian` | torchregress | 14.2163 ± 0.7762 † | 12.3386 ± 0.5666 | 0.1411 ± 0.0946 | - | - | - | - | - | - | - | 0.6674 |
| `CNN+SLS` | torchregress | 46.7031 ± 0.1774 | 44.1067 ± 0.1879 | -8.2513 ± 0.0703 | - | - | 0.9444 ± 0.0962 | 152.7 ± 45.4 | 155.5 ± 42.7 | non-finite (3) | 0.6201 ± 0.0433 | 1.5386 |
| `CNN+StudentT` | torchregress | 15.0353 ± 0.3037 | 13.2068 ± 0.1587 | 0.0409 ± 0.0390 | - | - | - | - | - | - | - | 0.4931 |
| `ResNet+Evidential` | torchregress | 25.0034 ± 3.4208 † | 18.3474 ± 2.2907 | -1.6847 ± 0.7519 | - | - | 1.0000 ± 0.0000 | 194.0 ± 67.0 | 194.0 ± 67.0 | non-finite (3) | 0.2382 ± 0.0692 | 1.7024 |
| `ResNet+GaussianNLL` | torchregress | 28.1474 ± 2.0627 † | 21.9773 ± 1.7854 | -2.3724 ± 0.4992 | 5.8543 ± 2.1097 | 15.6497 ± 0.8003 | 0.6667 ± 0.1667 | 62.6353 ± 17.1314 | 135.7 ± 21.7 | non-finite (3) | 0.4243 ± 0.0860 | 1.6526 |
| `ResNet+MDN_K3` | torchregress | 22.1403 ± 11.3290 † | 18.7263 ± 9.9004 | -1.4420 ± 1.9166 | - | - | 0.4444 ± 0.4811 | 32.2124 ± 28.5705 | 263.7 ± 174.0 | non-finite (3) | 0.5239 ± 0.3144 | 1.6495 |
| `ResNet+MultiQuantile` | torchregress | 24.9998 ± 2.9957 † | 20.7151 ± 1.8655 | -1.6762 ± 0.6237 | - | - | 0.0833 ± 0.0833 | 4.0322 ± 0.2880 | 381.3 ± 36.5 | non-finite (3) | 0.5972 ± 0.1273 | 1.5729 |
| `ResNet+NormalizingFlow` | torchregress | 48.3044 ± 0.3314 | 45.7858 ± 0.3627 | -8.8968 ± 0.1357 | - | - | 0.0000 ± 0.0000 | 3.2314 ± 0.3131 | 888.5 ± 9.3 | non-finite (3) | 0.9583 ± 0.0000 | 3.5160 |
| `sklearn_GBR` | scikit-learn | **11.9908 ± 0.1539** | **8.8381 ± 0.2005** | **0.3901 ± 0.0157** | - | - | - | - | - | - | - | 0.4972 |
| `sklearn_RF` | scikit-learn | 12.7916 ± 0.5027 | 10.7386 ± 0.4842 | 0.3053 ± 0.0551 | - | - | - | - | - | - | - | 0.1833 |
| `timm_resnet18+GaussianNLL` | timm | 19.3738 ± 2.5944 † | 16.8319 ± 1.2601 | -0.6110 ± 0.4424 | 6.0104 ± 0.6572 | 52.1871 ± 34.7206 | 1.0000 ± 0.0000 | 720.5 ± 496.0 | 720.5 ± 496.0 | non-finite (3) | 0.4102 ± 0.0323 | 1.6365 |
| `torch_uncertainty` | torch-uncertainty | 37.5054 ± 2.1668 † | 34.2524 ± 2.3570 | -4.9795 ± 0.7004 | 12.0427 ± 1.1930 | 22933.5 ± 16382.7 | 1.0000 ± 0.0000 | 3.229e+05 ± 2.306e+05 | 3.229e+05 ± 2.306e+05 | non-finite (3) | 0.4584 ± 0.0001 | 1.4951 |

Failed / skipped:

- `CNN+RankNContrast` (failed, seed 0, 1, 2): failed: RuntimeError: Expected all tensors to be on the same device, but got b is on cuda:0, different from other tenso…

##### age_regression / synthetic_image_regression

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CNN+Barron` | torchregress | 0.0535 ± 0.0004 | 0.0427 ± 0.0004 | 0.9475 ± 0.0008 | - | - | - | - | - | - | - | 6.5960 |
| `CNN+BetaNLL` | torchregress | 0.0541 ± 0.0016 | 0.0433 ± 0.0010 | 0.9463 ± 0.0031 | **-1.4949 ± 0.0310** | **0.0305 ± 0.0009** | 0.8889 ± 0.0291 | 0.1764 ± 0.0074 | **0.2242 ± 0.0057** | 0.0046 ± 0.0004 | 0.0888 ± 0.0350 | 6.2538 |
| `CNN+CRPS` | torchregress | 0.0551 ± 0.0021 | 0.0439 ± 0.0018 | 0.9442 ± 0.0043 | -1.4808 ± 0.0343 | 0.0310 ± 0.0012 | 0.8978 ± 0.0192 | 0.1789 ± 0.0068 | 0.2309 ± 0.0065 | 0.0033 ± 0.0004 | 0.0577 ± 0.0195 | 6.7227 |
| `CNN+Evidential` | torchregress | 0.0537 ± 0.0008 | 0.0427 ± 0.0006 | 0.9470 ± 0.0016 | - | - | 0.1644 ± 0.0135 | 0.0226 ± 0.0005 | 0.6691 ± 0.0085 | 0.0187 ± 0.0144 | 0.7993 ± 0.0241 | 7.6739 |
| `CNN+FaithfulGaussian` | torchregress | 0.0554 ± 0.0019 | 0.0440 ± 0.0015 | 0.9437 ± 0.0038 | -1.4732 ± 0.0308 | 0.0312 ± 0.0011 | 0.8956 ± 0.0241 | 0.1782 ± 0.0166 | 0.2296 ± 0.0026 | 0.0032 ± 0.0004 | 0.1076 ± 0.0561 | 6.0589 |
| `CNN+GaussianNLL` | torchregress | 0.0572 ± 0.0036 † | 0.0457 ± 0.0025 | 0.9399 ± 0.0076 | -1.4404 ± 0.0479 | 0.0323 ± 0.0019 | 0.8878 ± 0.0448 | 0.1759 ± 0.0249 | 0.2352 ± 0.0109 | 0.0042 ± 0.0011 | 0.0773 ± 0.0097 | 6.4006 |
| `CNN+Huber` | torchregress | 0.0535 ± 0.0004 | 0.0428 ± 0.0005 | 0.9474 ± 0.0008 | - | - | - | - | - | - | - | 5.2707 |
| `CNN+MC_Dropout` | torchregress | 0.0569 ± 0.0025 | 0.0447 ± 0.0023 | 0.9405 ± 0.0054 | -0.0226 ± 0.3730 | 0.0339 ± 0.0019 | 0.6311 ± 0.0375 | 0.1022 ± 0.0063 | 0.3489 ± 0.0322 | 0.0221 ± 0.0047 | 0.2117 ± 0.0379 | 5.1772 |
| `CNN+MDN_K2` | torchregress | **0.0533 ± 0.0018** | **0.0426 ± 0.0013** | **0.9479 ± 0.0035** | - | - | 0.8778 ± 0.0214 | 0.1638 ± 0.0146 | 0.2277 ± 0.0081 | 0.0049 ± 0.0006 | 0.0984 ± 0.0688 | 6.6436 |
| `CNN+MDN_K3` | torchregress | 0.0559 ± 0.0010 | 0.0446 ± 0.0014 | 0.9427 ± 0.0020 | - | - | 0.8622 ± 0.0252 | 0.1628 ± 0.0128 | 0.2424 ± 0.0086 | 0.0056 ± 0.0015 | 0.0894 ± 0.0611 | 6.4349 |
| `CNN+MDN_K5` | torchregress | 0.0565 ± 0.0012 | 0.0448 ± 0.0007 | 0.9414 ± 0.0025 | - | - | 0.8356 ± 0.0168 | 0.1616 ± 0.0044 | 0.2519 ± 0.0130 | 0.0039 ± 0.0008 | 0.1619 ± 0.0461 | 6.3888 |
| `CNN+MultiQuantile` | torchregress | 0.0550 ± 0.0010 | 0.0435 ± 0.0006 | 0.9445 ± 0.0020 | - | - | 0.8744 ± 0.0340 | 0.1737 ± 0.0044 | 0.2394 ± 0.0089 | 0.0056 ± 0.0012 | 0.1084 ± 0.0126 | 5.5968 |
| `CNN+NF_Ensemble_K2` | torchregress | 0.0554 ± 0.0041 † | 0.0446 ± 0.0033 | 0.9436 ± 0.0084 | - | - | 0.8622 ± 0.0269 | 0.1662 ± 0.0121 | 0.2346 ± 0.0200 | 0.0061 ± 0.0020 | 0.1319 ± 0.0911 | 52.6007 |
| `CNN+NormalizingFlow` | torchregress | 0.0600 ± 0.0068 † | 0.0481 ± 0.0054 | 0.9333 ± 0.0151 | - | - | 0.8444 ± 0.0429 | 0.1740 ± 0.0015 | 0.2592 ± 0.0344 | 0.0038 ± 0.0007 | 0.1759 ± 0.1120 | 26.4497 |
| `CNN+ObservedStdNLL` | torchregress | 0.0546 ± 0.0010 | 0.0436 ± 0.0007 | 0.9453 ± 0.0020 | -1.4831 ± 0.0203 | 0.0308 ± 0.0006 | 0.8978 ± 0.0315 | 0.1823 ± 0.0188 | 0.2305 ± 0.0072 | 0.0041 ± 0.0006 | 0.1210 ± 0.0156 | 5.5664 |
| `CNN+PoissonGaussian` | torchregress | 0.2336 ± 0.0000 | 0.1920 ± 0.0000 | -0.0013 ± 0.0000 | - | - | - | - | - | - | - | 6.3032 |
| `CNN+SLS` | torchregress | 0.0565 ± 0.0052 † | 0.0454 ± 0.0043 | 0.9410 ± 0.0109 | - | - | 0.8711 ± 0.0267 | 0.1747 ± 0.0019 | 0.2318 ± 0.0180 | 0.0049 ± 0.0025 | 0.1944 ± 0.0675 | 12.0873 |
| `CNN+StudentT` | torchregress | 0.0541 ± 0.0007 | 0.0434 ± 0.0005 | 0.9463 ± 0.0014 | - | - | - | - | - | - | - | 5.2550 |
| `ResNet+Evidential` | torchregress | 0.0697 ± 0.0046 † | 0.0555 ± 0.0021 | 0.9105 ± 0.0119 | - | - | 0.1356 ± 0.0425 | 0.0476 ± 0.0205 | 0.8839 ± 0.0441 | 0.0256 ± 0.0194 | 0.8070 ± 0.0569 | 17.6215 |
| `ResNet+GaussianNLL` | torchregress | 0.0779 ± 0.0187 † | 0.0553 ± 0.0035 | 0.8844 ± 0.0570 | -1.1572 ± 0.0178 | 0.0395 ± 0.0020 | 0.7778 ± 0.0051 | 0.1778 ± 0.0220 | 0.3140 ± 0.0143 | 0.0047 ± 0.0014 | 0.1865 ± 0.0592 | 16.5235 |
| `ResNet+MDN_K3` | torchregress | 0.1548 ± 0.0618 † | 0.0638 ± 0.0074 | 0.5133 ± 0.3109 | - | - | 0.7856 ± 0.0762 | 0.3704 ± 0.1647 | 0.4937 ± 0.1673 | 0.0043 ± 0.0012 | 0.1357 ± 0.0474 | 16.9053 |
| `ResNet+MultiQuantile` | torchregress | 0.0714 ± 0.0101 † | 0.0554 ± 0.0054 | 0.9051 ± 0.0272 | - | - | 0.7511 ± 0.0255 | 0.1702 ± 0.0012 | 0.3172 ± 0.0176 | 0.0130 ± 0.0058 | 0.1638 ± 0.0409 | 15.9760 |
| `ResNet+NormalizingFlow` | torchregress | 0.0665 ± 0.0014 | 0.0534 ± 0.0011 | 0.9188 ± 0.0035 | - | - | 0.6678 ± 0.0460 | 0.1359 ± 0.0140 | 0.3644 ± 0.0257 | 0.0052 ± 0.0012 | 0.1757 ± 0.0171 | 32.8294 |
| `sklearn_GBR` | scikit-learn | 0.0569 ± 0.0002 | 0.0458 ± 0.0002 | 0.9406 ± 0.0004 | - | - | - | - | - | - | - | 15.3565 |
| `sklearn_RF` | scikit-learn | 0.0565 ± 0.0001 | 0.0453 ± 0.0001 | 0.9414 ± 0.0001 | - | - | - | - | - | - | - | 2.4455 |
| `timm_resnet18+GaussianNLL` | timm | 0.0679 ± 0.0027 | 0.0543 ± 0.0022 | 0.9154 ± 0.0068 | -1.0695 ± 0.1686 | 0.0392 ± 0.0023 | 0.7478 ± 0.0473 | 0.1659 ± 0.0074 | 0.3188 ± 0.0393 | 0.0048 ± 0.0016 | 0.1603 ± 0.0923 | 16.7903 |
| `torch_uncertainty` | torch-uncertainty | 0.2333 ± 0.0001 | 0.1918 ± 0.0000 | 8.399e-04 ± 6.505e-04 | -0.0365 ± 0.0004 | 0.1334 ± 0.0000 | 0.8878 ± 0.0019 | 0.7702 ± 0.0012 | 0.9043 ± 0.0005 | **0.0023 ± 0.0001** | **0.0434 ± 0.0004** | 6.6787 |

Failed / skipped:

- `CNN+RankNContrast` (failed, seed 0, 1, 2): failed: RuntimeError: Expected all tensors to be on the same device, but got b is on cuda:0, different from other tenso…

##### age_regression / udacity_steering

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CNN+Barron` | torchregress | 0.2333 ± 0.0210 † | 0.1590 ± 0.0149 | 0.5317 ± 0.0858 | - | - | - | - | - | - | - | 15.9631 |
| `CNN+BetaNLL` | torchregress | 0.2752 ± 0.0202 † | 0.1863 ± 0.0027 | 0.3493 ± 0.0969 | 0.1043 ± 0.1824 | 0.1370 ± 0.0046 | 0.7858 ± 0.0802 | 0.6589 ± 0.1082 | 1.1770 ± 0.1056 | 0.0053 ± 0.0023 | 0.1844 ± 0.0408 | 13.9432 |
| `CNN+CRPS` | torchregress | 0.2550 ± 0.0129 † | 0.1683 ± 0.0080 | 0.4426 ± 0.0563 | 0.1605 ± 0.1954 | 0.1244 ± 0.0065 | 0.8133 ± 0.0313 | 0.5862 ± 0.0770 | 1.1183 ± 0.0427 | **0.0053 ± 0.0010** | 0.1794 ± 0.0293 | 15.4858 |
| `CNN+Evidential` | torchregress | 0.2804 ± 0.0054 | 0.1820 ± 0.0023 | 0.3266 ± 0.0258 | - | - | 0.2692 ± 0.0808 | 0.1494 ± 0.0253 | 2.5377 ± 0.1742 | 0.0082 ± 0.0031 | 0.4179 ± 0.0565 | 18.0387 |
| `CNN+FaithfulGaussian` | torchregress | 0.3241 ± 0.0175 † | 0.2105 ± 0.0191 | 0.0991 ± 0.0981 | 0.0648 ± 0.1542 | 0.1591 ± 0.0136 | 0.8550 ± 0.0254 | 0.8702 ± 0.1180 | 1.3774 ± 0.1812 | 0.0083 ± 0.0012 | 0.2039 ± 0.0620 | 14.3345 |
| `CNN+GaussianNLL` | torchregress | 0.3268 ± 0.0086 | 0.1971 ± 0.0052 | 0.0854 ± 0.0483 | 0.0473 ± 0.1677 | 0.1533 ± 0.0053 | 0.8667 ± 0.0339 | 0.8771 ± 0.1566 | 1.3295 ± 0.0211 | 0.0083 ± 0.0013 | 0.1683 ± 0.0465 | 14.1645 |
| `CNN+Huber` | torchregress | **0.2313 ± 0.0159** † | 0.1561 ± 0.0114 | **0.5404 ± 0.0640** | - | - | - | - | - | - | - | 12.6265 |
| `CNN+MC_Dropout` | torchregress | 0.2598 ± 0.0147 † | 0.1702 ± 0.0149 | 0.4210 ± 0.0650 | 8.0719 ± 3.1654 | 0.1441 ± 0.0118 | 0.4508 ± 0.0767 | 0.1949 ± 0.0178 | 2.0468 ± 0.1596 | 0.0214 ± 0.0082 | 0.3487 ± 0.1153 | 12.7731 |
| `CNN+MDN_K2` | torchregress | 0.3346 ± 0.0389 † | 0.2012 ± 0.0239 | 0.0331 ± 0.2198 | - | - | 0.8658 ± 0.0635 | 0.9492 ± 0.1363 | 1.4491 ± 0.1822 | 0.0065 ± 0.0028 | 0.2727 ± 0.1004 | 15.7987 |
| `CNN+MDN_K3` | torchregress | 0.3422 ± 0.0024 | 0.2002 ± 0.0084 | -0.0026 ± 0.0142 | - | - | 0.8908 ± 0.0118 | 1.0575 ± 0.1730 | 1.6219 ± 0.2093 | 0.0102 ± 0.0019 | 0.2706 ± 0.0526 | 15.4051 |
| `CNN+MDN_K5` | torchregress | 0.3434 ± 0.0008 | 0.2003 ± 0.0101 | -0.0093 ± 0.0049 | - | - | 0.8933 ± 0.0128 | 1.1814 ± 0.1572 | 1.6937 ± 0.2105 | 0.0109 ± 0.0041 | 0.2717 ± 0.0456 | 15.0972 |
| `CNN+MultiQuantile` | torchregress | 0.2818 ± 0.0041 | 0.1691 ± 0.0035 | 0.3200 ± 0.0197 | - | - | 0.8808 ± 0.0029 | 0.7319 ± 0.0355 | **0.9984 ± 0.0210** | 0.0069 ± 0.0016 | **0.1552 ± 0.0681** | 14.1059 |
| `CNN+NF_Ensemble_K2` | torchregress | 0.3441 ± 0.0018 | 0.2009 ± 0.0034 | -0.0135 ± 0.0107 | - | - | 0.8792 ± 0.0210 | 1.1910 ± 0.0950 | 1.8652 ± 0.0466 | 0.0130 ± 0.0006 | 0.2773 ± 0.0298 | 122.4 |
| `CNN+NormalizingFlow` | torchregress | 0.3434 ± 0.0028 | 0.2089 ± 0.0140 | -0.0093 ± 0.0163 | - | - | 0.8958 ± 0.0181 | 1.3266 ± 0.0731 | 1.7380 ± 0.0330 | 0.0119 ± 0.0008 | 0.3595 ± 0.1256 | 61.2300 |
| `CNN+ObservedStdNLL` | torchregress | 0.3285 ± 0.0138 | 0.1958 ± 0.0040 | 0.0749 ± 0.0776 | **3.454e-04 ± 6.722e-02** | 0.1555 ± 0.0054 | 0.8567 ± 0.0123 | 0.8620 ± 0.0201 | 1.3648 ± 0.1250 | 0.0065 ± 0.0015 | 0.2284 ± 0.0459 | 14.9585 |
| `CNN+PoissonGaussian` | torchregress | 0.3437 ± 0.0000 | 0.1856 ± 0.0000 | -0.0112 ± 0.0000 | - | - | - | - | - | - | - | 15.2215 |
| `CNN+SLS` | torchregress | 0.3279 ± 0.0261 † | 0.1981 ± 0.0129 | 0.0757 ± 0.1431 | - | - | 0.5750 ± 0.2699 | 0.3503 ± 0.4335 | 2.6056 ± 1.2484 | 0.0311 ± 0.0283 | 0.3495 ± 0.1493 | 26.6094 |
| `CNN+StudentT` | torchregress | 0.2326 ± 0.0221 † | 0.1585 ± 0.0149 | 0.5339 ± 0.0883 | - | - | - | - | - | - | - | 12.9956 |
| `ResNet+Evidential` | torchregress | 0.2395 ± 0.0226 † | 0.1589 ± 0.0282 | 0.5061 ± 0.0950 | - | - | 0.1458 ± 0.1190 | 0.0318 ± 0.0011 | 2.9109 ± 0.5543 | 0.0207 ± 0.0126 | 0.5775 ± 0.1864 | 42.0476 |
| `ResNet+GaussianNLL` | torchregress | 0.2408 ± 0.0101 | 0.1621 ± 0.0141 | 0.5031 ± 0.0416 | 0.3541 ± 0.7815 | 0.1233 ± 0.0081 | 0.7333 ± 0.1767 | 0.4705 ± 0.1589 | 1.1766 ± 0.1699 | 0.0077 ± 0.0010 | 0.3426 ± 0.1400 | 39.1768 |
| `ResNet+MDN_K3` | torchregress | 0.2633 ± 0.0351 † | 0.1771 ± 0.0343 | 0.3997 ± 0.1521 | - | - | 0.6450 ± 0.2431 | 0.4496 ± 0.0483 | 1.3414 ± 0.2348 | 0.0079 ± 0.0025 | 0.3591 ± 0.1960 | 40.8047 |
| `ResNet+MultiQuantile` | torchregress | 0.2702 ± 0.0372 † | 0.1946 ± 0.0358 | 0.3669 ± 0.1653 | - | - | 0.7708 ± 0.1130 | 0.4374 ± 0.1958 | 1.3103 ± 0.3217 | 0.0168 ± 0.0049 | 0.2229 ± 0.0683 | 38.7718 |
| `ResNet+NormalizingFlow` | torchregress | 0.2427 ± 0.0233 † | 0.1514 ± 0.0178 | 0.4926 ± 0.0987 | - | - | 0.8325 ± 0.0391 | 0.4846 ± 0.1028 | 1.0308 ± 0.1721 | 0.0063 ± 0.0019 | 0.2555 ± 0.1208 | 76.7137 |
| `sklearn_GBR` | scikit-learn | 0.2534 ± 0.0013 | 0.1610 ± 0.0009 | 0.4502 ± 0.0057 | - | - | - | - | - | - | - | 369.2 |
| `sklearn_RF` | scikit-learn | 0.2505 ± 0.0021 | 0.1576 ± 0.0014 | 0.4630 ± 0.0091 | - | - | - | - | - | - | - | 69.2894 |
| `timm_resnet18+GaussianNLL` | timm | 0.2442 ± 0.0123 † | **0.1503 ± 0.0042** | 0.4888 ± 0.0519 | 2.1897 ± 1.8886 | **0.1192 ± 0.0044** | 0.7033 ± 0.0659 | 0.3082 ± 0.0567 | 1.4162 ± 0.1402 | 0.0071 ± 0.0013 | 0.2261 ± 0.0946 | 39.3160 |
| `torch_uncertainty` | torch-uncertainty | 0.3447 ± 0.0013 | 0.1925 ± 0.0041 | -0.0174 ± 0.0078 | 0.3505 ± 0.0064 | 0.1796 ± 0.0012 | 0.8817 ± 0.0038 | 1.1265 ± 0.0434 | 1.8198 ± 0.0115 | 0.0084 ± 0.0001 | 0.3315 ± 0.0123 | 37.8887 |

Failed / skipped:

- `CNN+RankNContrast` (failed, seed 0, 1, 2): failed: RuntimeError: Expected all tensors to be on the same device, but got b is on cuda:0, different from other tenso…

##### age_regression / udacity_steering_tiny

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CNN+Barron` | torchregress | 0.2861 ± 0.0034 | 0.1782 ± 0.0135 | 0.2854 ± 0.0169 | - | - | - | - | - | - | - | 1.7616 |
| `CNN+BetaNLL` | torchregress | 0.3340 ± 0.0111 | 0.2055 ± 0.0059 | 0.0258 ± 0.0654 | 0.4900 ± 0.1159 | 0.1686 ± 0.0059 | 0.9000 ± 0.0250 | 1.0599 ± 0.1132 | **1.5721 ± 0.1035** | 0.0285 ± 0.0040 | 0.1974 ± 0.0757 | 1.5487 |
| `CNN+CRPS` | torchregress | 0.3170 ± 0.0193 † | 0.2035 ± 0.0062 | 0.1208 ± 0.1069 | 0.4120 ± 0.0305 | 0.1641 ± 0.0141 | 0.8667 ± 0.0289 | 0.9296 ± 0.1176 | 1.6438 ± 0.1373 | 0.0258 ± 0.0028 | **0.1767 ± 0.0379** | 1.8620 |
| `CNN+Evidential` | torchregress | 0.3208 ± 0.0071 | 0.2098 ± 0.0110 | 0.1015 ± 0.0394 | - | - | 0.4167 ± 0.0144 | 0.4200 ± 0.1138 | 2.8581 ± 0.1216 | 0.0518 ± 0.0246 | 0.3370 ± 0.0723 | 2.0705 |
| `CNN+FaithfulGaussian` | torchregress | 0.3404 ± 0.0029 | 0.2052 ± 0.0097 | -0.0114 ± 0.0173 | 0.7587 ± 0.4860 | 0.1718 ± 0.0037 | 0.8833 ± 0.0382 | 1.0474 ± 0.1262 | 1.6120 ± 0.0503 | 0.0344 ± 0.0045 | 0.2017 ± 0.0380 | 1.6622 |
| `CNN+GaussianNLL` | torchregress | 0.3354 ± 0.0028 | 0.2109 ± 0.0096 | 0.0181 ± 0.0167 | 0.5483 ± 0.0628 | 0.1741 ± 0.0007 | 0.9000 ± 0.0250 | 1.1087 ± 0.0599 | 1.6089 ± 0.0370 | 0.0305 ± 0.0060 | 0.2260 ± 0.0582 | 2.1016 |
| `CNN+Huber` | torchregress | 0.2845 ± 0.0051 | 0.1807 ± 0.0093 | 0.2934 ± 0.0254 | - | - | - | - | - | - | - | 1.4207 |
| `CNN+MC_Dropout` | torchregress | 0.3011 ± 0.0149 | 0.1831 ± 0.0088 | 0.2074 ± 0.0784 | 7.6701 ± 1.4590 | **0.1540 ± 0.0080** | 0.5333 ± 0.0629 | 0.2392 ± 0.0167 | 2.1790 ± 0.1729 | 0.0514 ± 0.0233 | 0.2708 ± 0.0520 | 1.4557 |
| `CNN+MDN_K2` | torchregress | 0.3401 ± 0.0192 † | 0.2170 ± 0.0267 | -0.0117 ± 0.1149 | - | - | 0.9083 ± 0.0144 | 1.2283 ± 0.1668 | 1.6941 ± 0.1264 | 0.0235 ± 0.0021 | 0.2344 ± 0.0695 | 1.8268 |
| `CNN+MDN_K3` | torchregress | 0.3410 ± 0.0083 | 0.2078 ± 0.0085 | -0.0156 ± 0.0498 | - | - | 0.8417 ± 0.0382 | 1.0423 ± 0.1528 | 1.7982 ± 0.0458 | 0.0269 ± 0.0039 | 0.2375 ± 0.0583 | 1.7628 |
| `CNN+MDN_K5` | torchregress | 0.3379 ± 0.0025 | 0.2008 ± 0.0064 | 0.0037 ± 0.0149 | - | - | 0.9000 ± 0.0250 | 1.2582 ± 0.1153 | 1.7200 ± 0.1435 | 0.0285 ± 0.0021 | 0.2237 ± 0.0684 | 1.6677 |
| `CNN+MultiQuantile` | torchregress | 0.3411 ± 0.0009 | 0.1925 ± 0.0002 | -0.0157 ± 0.0051 | - | - | 0.9667 ± 0.0144 | 1.3200 ± 0.0497 | 1.6332 ± 0.0568 | 0.0229 ± 0.0008 | 0.2347 ± 0.0055 | 1.5231 |
| `CNN+NF_Ensemble_K2` | torchregress | 0.3445 ± 0.0049 | 0.2051 ± 0.0048 | -0.0360 ± 0.0292 | - | - | 0.9083 ± 0.0144 | 1.3857 ± 0.0858 | 1.7886 ± 0.1189 | 0.0296 ± 0.0057 | 0.2411 ± 0.0667 | 14.5901 |
| `CNN+NormalizingFlow` | torchregress | 0.3454 ± 0.0046 | 0.2063 ± 0.0044 | -0.0413 ± 0.0280 | - | - | 0.7250 ± 0.1090 | 0.6192 ± 0.3226 | 2.3654 ± 0.4150 | 0.0491 ± 0.0127 | 0.3112 ± 0.1487 | 7.3820 |
| `CNN+ObservedStdNLL` | torchregress | 0.3335 ± 0.0063 | 0.2093 ± 0.0063 | 0.0292 ± 0.0366 | **0.3281 ± 0.0710** | 0.1741 ± 0.0019 | 0.9417 ± 0.0144 | 1.2681 ± 0.0789 | 1.6030 ± 0.0314 | 0.0222 ± 0.0034 | 0.2292 ± 0.0194 | 1.5370 |
| `CNN+PoissonGaussian` | torchregress | 0.3433 ± 0.0000 | 0.1925 ± 0.0000 | -0.0289 ± 0.0000 | - | - | - | - | - | - | - | 1.8487 |
| `CNN+SLS` | torchregress | 0.3469 ± 0.0142 | 0.2120 ± 0.0162 | -0.0515 ± 0.0871 | - | - | 0.8917 ± 0.0144 | 1.2556 ± 0.0911 | 1.8222 ± 0.1052 | 0.0330 ± 0.0095 | 0.2783 ± 0.0487 | 3.5429 |
| `CNN+StudentT` | torchregress | 0.2855 ± 0.0065 | 0.1760 ± 0.0148 | 0.2884 ± 0.0322 | - | - | - | - | - | - | - | 1.4423 |
| `ResNet+Evidential` | torchregress | 0.3226 ± 0.0166 † | 0.2089 ± 0.0267 | 0.0901 ± 0.0932 | - | - | 0.2167 ± 0.0382 | 0.0997 ± 0.0198 | 3.4577 ± 0.5373 | 0.0360 ± 0.0192 | 0.4333 ± 0.1114 | 4.9187 |
| `ResNet+GaussianNLL` | torchregress | 0.3319 ± 0.0420 † | 0.2130 ± 0.0237 | 0.0281 ± 0.2500 | 1.6939 ± 0.6771 | 0.1675 ± 0.0173 | 0.6917 ± 0.1233 | 0.4230 ± 0.0779 | 1.9512 ± 0.2060 | 0.0355 ± 0.0185 | 0.1969 ± 0.1194 | 4.7017 |
| `ResNet+MDN_K3` | torchregress | 0.5421 ± 0.4425 † | 0.3071 ± 0.2318 | -2.7043 ± 5.1732 | - | - | 0.7333 ± 0.1588 | 1.0831 ± 0.9166 | 2.3091 ± 1.2914 | **0.0146 ± 0.0073** | 0.2961 ± 0.1822 | 4.7530 |
| `ResNet+MultiQuantile` | torchregress | 0.3273 ± 0.0235 † | 0.2051 ± 0.0124 | 0.0616 ± 0.1359 | - | - | 0.5917 ± 0.1283 | 0.3075 ± 0.0673 | 2.4298 ± 0.2316 | 0.0321 ± 0.0155 | 0.2372 ± 0.1000 | 4.5524 |
| `ResNet+NormalizingFlow` | torchregress | 0.3712 ± 0.0580 † | 0.2447 ± 0.0401 | -0.2225 ± 0.3651 | - | - | 0.6583 ± 0.1258 | 0.5887 ± 0.2591 | 2.0272 ± 0.4934 | 0.0247 ± 0.0073 | 0.2424 ± 0.0325 | 9.0324 |
| `sklearn_GBR` | scikit-learn | 0.2874 ± 0.0016 | 0.1845 ± 0.0014 | 0.2789 ± 0.0082 | - | - | - | - | - | - | - | 9.3040 |
| `sklearn_RF` | scikit-learn | **0.2801 ± 0.0058** | **0.1742 ± 0.0038** | **0.3148 ± 0.0286** | - | - | - | - | - | - | - | 1.6280 |
| `timm_resnet18+GaussianNLL` | timm | 0.3159 ± 0.0141 | 0.2190 ± 0.0222 | 0.1280 ± 0.0766 | 5.8408 ± 1.8613 | 0.1741 ± 0.0176 | 0.5833 ± 0.0144 | 0.4223 ± 0.1051 | 2.0699 ± 0.2680 | 0.0333 ± 0.0074 | 0.2626 ± 0.0381 | 4.6222 |
| `torch_uncertainty` | torch-uncertainty | 0.3397 ± 0.0009 | 0.2029 ± 0.0037 | -0.0072 ± 0.0055 | 0.3433 ± 0.0045 | 0.1808 ± 0.0005 | 0.9000 ± 0.0000 | 1.2259 ± 0.0203 | 1.7577 ± 0.0115 | 0.0210 ± 0.0009 | 0.2320 ± 0.0065 | 2.3557 |

Failed / skipped:

- `CNN+RankNContrast` (failed, seed 0, 1, 2): failed: RuntimeError: Expected all tensors to be on the same device, but got b is on cuda:0, different from other tenso…

#### cell_counting (image)

Group `image`; 2 dataset(s). Columns: RMSE ↓, MAE ↓, NLL ↓, Coverage ·, IntervalScore ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### cell_counting / bbbc_cellcount

| Method | Library | RMSE | MAE | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `CNN+EnhancedPG_fixed` | torchregress | 6.9960 ± 1.4502 † | 5.3155 ± 1.1152 | - | - | - | 23.4081 |
| `CNN+EnhancedPG_learned` | torchregress | 7.6539 ± 1.8333 † | 6.0355 ± 1.6740 | - | - | - | 24.0631 |
| `CNN+GaussianNLL` | torchregress | **4.8452 ± 1.2563** † | **3.6232 ± 1.0387** | 2.7459 ± 0.3951 | 0.8025 ± 0.1278 | 18.7652 ± 5.4002 | 19.6242 |
| `CNN+MSE` | torchregress | 5.0338 ± 0.8687 † | 3.7071 ± 0.7068 | - | - | - | 19.9361 |
| `CNN+NegBin` | torchregress | 9.8694 ± 3.0840 † | 7.1626 ± 1.9848 | - | - | - | 23.3312 |
| `CNN+Poisson` | torchregress | 6.7622 ± 0.7690 † | 5.3403 ± 0.7100 | - | - | - | 21.9290 |
| `CNN+PoissonGaussian` | torchregress | 7.3182 ± 0.0733 | 5.4819 ± 0.2317 | - | - | - | 22.8941 |
| `CNN+PoissonGaussianLR` | torchregress | 12.3355 ± 12.3920 † | 10.3285 ± 10.5700 | - | - | - | 22.8811 |
| `CNN+PoissonLR` | torchregress | 7.4861 ± 2.5226 † | 5.8233 ± 2.3750 | - | - | - | 21.9299 |
| `CNN+ZIP` | torchregress | 10.3701 ± 5.6292 † | 8.2535 ± 4.2724 | - | - | - | 22.9847 |
| `sklearn_RF` | scikit-learn | 12.1070 ± 0.0946 | 8.7232 ± 0.0874 | - | - | - | 55.1492 |

##### cell_counting / bbbc_cellcount_tiny

| Method | Library | RMSE | MAE | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `CNN+EnhancedPG_fixed` | torchregress | 13.5140 ± 0.6746 | 10.8644 ± 0.7284 | - | - | - | 1.9776 |
| `CNN+EnhancedPG_learned` | torchregress | 11.7537 ± 2.3404 † | 9.2381 ± 1.8477 | - | - | - | 2.0054 |
| `CNN+GaussianNLL` | torchregress | 32.2449 ± 4.7523 † | 27.5952 ± 4.9477 | 4.7306 ± 0.2639 | 0.9500 ± 0.0500 | 93.2160 ± 3.7110 | 1.6507 |
| `CNN+MSE` | torchregress | 14.2173 ± 0.7955 † | 10.9652 ± 0.7636 | - | - | - | 1.7645 |
| `CNN+NegBin` | torchregress | 19.0754 ± 9.2303 † | 14.4145 ± 6.7452 | - | - | - | 1.9009 |
| `CNN+Poisson` | torchregress | 14.8264 ± 4.4829 † | 12.0052 ± 3.6078 | - | - | - | 1.5651 |
| `CNN+PoissonGaussian` | torchregress | 26.6025 ± 5.5111 † | 21.7356 ± 5.9599 | - | - | - | 1.7817 |
| `CNN+PoissonGaussianLR` | torchregress | **10.9389 ± 3.5161** † | **8.5344 ± 2.9301** | - | - | - | 1.7894 |
| `CNN+PoissonLR` | torchregress | 14.9526 ± 5.6707 † | 12.1380 ± 4.8755 | - | - | - | 1.5599 |
| `CNN+ZIP` | torchregress | 13.8222 ± 3.5077 † | 11.1656 ± 3.3821 | - | - | - | 1.8138 |
| `sklearn_RF` | scikit-learn | 21.8014 ± 0.4625 | 16.5632 ± 0.2403 | - | - | - | 1.0422 |

#### censored_imaging (image)

Group `image`; 4 dataset(s). Columns: RMSE ↓, MAE ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### censored_imaging / bbbc_censored

| Method | Library | RMSE | MAE | train_s |
| --- | --- | ---: | ---: | ---: |
| `AFT` | torchregress | 47.0337 ± 0.0567 | 40.3302 ± 0.0567 | 10.1876 |
| `CensoredGaussianNLL` | torchregress | 218.1 ± 157.9 † | 195.8 ± 143.1 | 10.4235 |
| `CensoredQuantile` | torchregress | 5.8751 ± 0.1290 | 3.7628 ± 0.1017 | 9.0427 |
| `MSE` | torchregress | **4.4493 ± 0.4686** † | **3.4798 ± 0.3234** | 6.3202 |
| `lifelines_LogNormalAFT` | lifelines | 101.8 ± 0.9 | 58.8313 ± 0.4108 | 0.3393 |

##### censored_imaging / bbbc_censored_tiny

| Method | Library | RMSE | MAE | train_s |
| --- | --- | ---: | ---: | ---: |
| `AFT` | torchregress | 49.0376 ± 0.0387 | 44.2234 ± 0.0338 | 2.2728 |
| `CensoredGaussianNLL` | torchregress | 40.0353 ± 2.1367 † | 35.1287 ± 2.2914 | 2.4778 |
| `CensoredQuantile` | torchregress | 11.9636 ± 0.4822 | 9.8032 ± 0.2626 | 2.0071 |
| `MSE` | torchregress | **9.0006 ± 1.4171** † | **7.4631 ± 1.3484** | 1.8284 |
| `lifelines_LogNormalAFT` | lifelines | 155.5 ± 9.0 † | 84.8549 ± 3.5045 | 0.3525 |

##### censored_imaging / tcga_survival

| Method | Library | RMSE | MAE | train_s |
| --- | --- | ---: | ---: | ---: |
| `AFT` | torchregress | 796.8 ± 7.5 | **638.0 ± 11.5** | 3.2175 |
| `CensoredGaussianNLL` | torchregress | 1120.3 ± 90.4 † | 830.9 ± 91.7 | 3.4180 |
| `CensoredQuantile` | torchregress | 1017.0 ± 33.5 | 904.1 ± 29.2 | 2.8618 |
| `MSE` | torchregress | **792.2 ± 1.5** | 639.7 ± 2.8 | 2.5890 |
| `lifelines_LogNormalAFT` | lifelines | 4727.3 ± 57.6 | 3925.3 ± 27.5 | 0.5938 |

##### censored_imaging / tcga_survival_tiny

| Method | Library | RMSE | MAE | train_s |
| --- | --- | ---: | ---: | ---: |
| `AFT` | torchregress | 1217.9 ± 6.2 | 1078.6 ± 6.9 | 0.7793 |
| `CensoredGaussianNLL` | torchregress | 1209.3 ± 13.6 | 1069.1 ± 15.0 | 0.9469 |
| `CensoredQuantile` | torchregress | 977.9 ± 18.9 | 903.8 ± 16.3 | 0.6825 |
| `MSE` | torchregress | **522.3 ± 0.3** | **406.2 ± 0.7** | 1.0753 |
| `lifelines_LogNormalAFT` | lifelines | 2071.3 ± 4.8 | 1633.2 ± 2.5 | 0.3180 |

#### crowd_counting (image)

Group `image`; 3 dataset(s). Columns: CountMAE ↓, CountRMSE ↓, RMSE ↓, MAE ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: CountMAE.

##### crowd_counting / shanghai_crowd

| Method | Library | CountMAE | CountRMSE | RMSE | MAE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CSRNet` | csrnet-paper | **41.0445 ± 28.4621** † | **54.3139 ± 28.9063** | **0.0148 ± 0.0039** | **0.0053 ± 0.0024** | 57.1229 |
| `UNet+GaussianNLL` | torchregress | 74.4226 ± 36.5549 † | 92.1657 ± 37.2869 | 0.0222 ± 0.0014 | 0.0087 ± 0.0032 | 54.0430 |
| `UNet+Huber` | torchregress | 87.7098 ± 47.6151 † | 101.7 ± 50.0 | 0.0209 ± 0.0012 | 0.0097 ± 0.0016 | 54.2535 |
| `UNet+MSE` | torchregress | 49.0620 ± 12.4332 † | 63.5271 ± 13.8174 | 0.0166 ± 0.0030 | 0.0071 ± 0.0014 | 54.6813 |
| `sklearn_RF` | scikit-learn | 55.2608 ± 0.4214 | 72.1713 ± 1.1722 | 0.0212 ± 0.0000 | 0.0095 ± 0.0000 | 32.7418 |

##### crowd_counting / shanghai_crowd_tiny

| Method | Library | CountMAE | CountRMSE | RMSE | MAE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CSRNet` | csrnet-paper | 59.5493 ± 8.7311 † | 77.7117 ± 9.7069 | 0.0532 ± 0.0078 | 0.0230 ± 0.0024 | 4.8003 |
| `UNet+GaussianNLL` | torchregress | 65.1575 ± 36.3087 † | 78.6018 ± 36.0911 | 0.0500 ± 0.0185 | 0.0256 ± 0.0110 | 2.6413 |
| `UNet+Huber` | torchregress | 50.2544 ± 16.8793 † | 58.7951 ± 18.3358 | 0.0432 ± 0.0068 | 0.0216 ± 0.0041 | 2.6536 |
| `UNet+MSE` | torchregress | **39.1332 ± 9.0209** † | **45.5478 ± 9.8931** | **0.0372 ± 0.0034** | **0.0189 ± 0.0022** | 3.2036 |
| `sklearn_RF` | scikit-learn | 76.4739 ± 4.6283 † | 89.3634 ± 7.0217 | 0.0430 ± 0.0009 | 0.0326 ± 0.0015 | 0.7714 |

##### crowd_counting / ucf_qnrf

| Method | Library | CountMAE | CountRMSE | RMSE | MAE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CSRNet` | csrnet-paper | 452.0 ± 84.1 † | 732.5 ± 168.9 | 0.1022 ± 0.0110 | 0.0451 ± 0.0086 | 56.3757 |
| `UNet+GaussianNLL` | torchregress | 594.8 ± 211.4 † | 921.6 ± 244.0 | 0.1040 ± 0.0138 | 0.0463 ± 0.0141 | 53.8566 |
| `UNet+Huber` | torchregress | 416.6 ± 63.8 † | **624.9 ± 31.5** | **0.0867 ± 0.0046** | 0.0386 ± 0.0030 | 54.0775 |
| `UNet+MSE` | torchregress | **393.1 ± 46.2** † | 628.2 ± 33.6 | 0.0872 ± 0.0020 | **0.0370 ± 0.0025** | 54.4851 |
| `sklearn_RF` | scikit-learn | 898.5 ± 29.1 | 1254.6 ± 22.6 | 0.1281 ± 0.0008 | 0.0803 ± 0.0005 | 32.9347 |

#### depth_estimation (image)

Group `image`; 5 dataset(s). Columns: RMSE ↓, MAE ↓, NLL ↓, CRPS ↓, Coverage ·, ValidPixelFrac ↑ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### depth_estimation / kitti_depth

| Method | Library | RMSE | MAE | NLL | CRPS | Coverage | ValidPixelFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `UNet+BetaNLL` | torchregress | 9.9113 ± 1.8638 † | 5.3925 ± 1.6306 | 2.7650 ± 0.3839 | 3.6224 ± 1.0472 | 0.9378 ± 0.0748 | **0.1464 ± 0.0000** | 110.7 |
| `UNet+CRPS` | torchregress | 3.3828 ± 0.2806 † | 1.5435 ± 0.2401 | **1.7621 ± 0.2134** | 1.1349 ± 0.1716 | 0.9015 ± 0.0361 | **0.1464 ± 0.0000** | 110.9 |
| `UNet+DeepEnsemble_K3` | torchregress | **2.4155 ± 0.2431** † | **1.2729 ± 0.1630** | 62.1148 ± 18.5660 | **0.9970 ± 0.1361** | 0.6131 ± 0.0626 | **0.1464 ± 0.0000** | 336.2 |
| `UNet+Evidential` | torchregress | 4.6711 ± 0.7367 † | 1.8738 ± 0.3169 | - | - | 0.8416 ± 0.0640 | **0.1464 ± 0.0000** | 115.2 |
| `UNet+FaithfulGaussian` | torchregress | 2.9984 ± 0.9167 † | 1.7378 ± 0.6722 | 2.2059 ± 0.0953 | 3.3342 ± 1.1349 | 0.9762 ± 0.0119 | **0.1464 ± 0.0000** | 111.7 |
| `UNet+GaussianNLL` | torchregress | 15.4159 ± 0.4490 | 9.8296 ± 0.6981 | 3.2691 ± 0.2200 | 6.2620 ± 0.3913 | 0.9565 ± 0.0285 | **0.1464 ± 0.0000** | 110.8 |
| `UNet+Huber` | torchregress | 3.5094 ± 0.1926 † | 1.5230 ± 0.1836 | - | - | - | **0.1464 ± 0.0000** | 111.0 |
| `UNet+MDN_K3` | torchregress | 10.5790 ± 0.5788 † | 6.5665 ± 0.5660 | - | - | 0.8225 ± 0.0307 | **0.1464 ± 0.0000** | 113.3 |
| `UNet+MSE` | torchregress | 3.0239 ± 0.6180 † | 1.6824 ± 0.4209 | - | - | - | **0.1464 ± 0.0000** | 113.1 |
| `UNet+MultiQuantile` | torchregress | 3.4930 ± 0.2025 † | 1.5051 ± 0.1111 | - | - | 0.8800 ± 0.0160 | **0.1464 ± 0.0000** | 112.1 |
| `UNet+StudentT` | torchregress | 7.4241 ± 0.6263 † | 2.7984 ± 0.3243 | - | - | - | **0.1464 ± 0.0000** | 111.0 |
| `sklearn_RF` | scikit-learn | 16.1941 ± 0.0016 | 15.3605 ± 0.0015 | - | - | - | - | 18.5535 |

##### depth_estimation / kitti_depth_tiny

| Method | Library | RMSE | MAE | NLL | CRPS | Coverage | ValidPixelFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `UNet+BetaNLL` | torchregress | 19.3752 ± 0.3847 | 15.5856 ± 0.3346 | 25.2379 ± 8.0204 | 13.7538 ± 0.7701 | 0.0104 ± 0.0124 | **0.1290 ± 0.0000** | 2.5752 |
| `UNet+CRPS` | torchregress | 18.2784 ± 0.6805 | **14.1534 ± 0.8424** | 48.2073 ± 28.7451 | **11.5592 ± 1.8205** | 0.2678 ± 0.3352 | **0.1290 ± 0.0000** | 2.5867 |
| `UNet+DeepEnsemble_K3` | torchregress | 18.7274 ± 0.4668 | 15.2749 ± 0.3712 | 6222.8 ± 970.9 | 15.0148 ± 0.3942 | 4.206e-04 ± 3.643e-04 | **0.1290 ± 0.0000** | 8.0766 |
| `UNet+Evidential` | torchregress | 19.3612 ± 0.0488 | 14.4711 ± 0.2119 | - | - | 0.2906 ± 0.0963 | **0.1290 ± 0.0000** | 3.0023 |
| `UNet+FaithfulGaussian` | torchregress | 19.2595 ± 0.5107 | 15.7389 ± 0.3835 | 37.1176 ± 9.1616 | 14.5178 ± 0.4592 | 0.0016 ± 0.0018 | **0.1290 ± 0.0000** | 2.7148 |
| `UNet+GaussianNLL` | torchregress | 19.6100 ± 0.3599 | 15.6284 ± 0.3298 | **22.4356 ± 6.0906** | 14.0780 ± 0.3752 | 0.0029 ± 0.0016 | **0.1290 ± 0.0000** | 2.6894 |
| `UNet+Huber` | torchregress | 19.4157 ± 0.4229 | 15.1877 ± 0.4907 | - | - | - | **0.1290 ± 0.0000** | 2.5701 |
| `UNet+MDN_K3` | torchregress | 19.5386 ± 0.1563 | 15.6474 ± 0.1876 | - | - | 0.0024 ± 0.0016 | **0.1290 ± 0.0000** | 2.8300 |
| `UNet+MSE` | torchregress | 18.8905 ± 0.4431 | 15.4439 ± 0.3497 | - | - | - | **0.1290 ± 0.0000** | 3.6067 |
| `UNet+MultiQuantile` | torchregress | 19.5855 ± 0.1706 | 15.4477 ± 0.1909 | - | - | 0.0000 ± 0.0000 | **0.1290 ± 0.0000** | 2.7486 |
| `UNet+StudentT` | torchregress | 19.8564 ± 0.1424 | 14.1766 ± 0.4420 | - | - | - | **0.1290 ± 0.0000** | 2.5863 |
| `sklearn_RF` | scikit-learn | **17.6636 ± 0.0233** | 16.6455 ± 0.0216 | - | - | - | - | 0.5356 |

##### depth_estimation / nyu_depth_v2

| Method | Library | RMSE | MAE | NLL | CRPS | Coverage | ValidPixelFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `UNet+BetaNLL` | torchregress | 0.9495 ± 0.0390 | 0.6691 ± 0.0160 | 3.8935 ± 0.6771 | 0.5378 ± 0.0190 | 0.5061 ± 0.0205 | - | 196.6 |
| `UNet+CRPS` | torchregress | 0.9840 ± 0.0453 | 0.7038 ± 0.0424 | 3.5525 ± 0.2722 | 0.5632 ± 0.0366 | 0.5182 ± 0.0351 | - | 197.4 |
| `UNet+DeepEnsemble_K3` | torchregress | **0.8884 ± 0.0175** | **0.6246 ± 0.0174** | 40.3530 ± 4.1630 | **0.5092 ± 0.0149** | 0.4778 ± 0.0022 | - | 591.9 |
| `UNet+Evidential` | torchregress | 0.9957 ± 0.0490 | 0.6897 ± 0.0187 | - | - | 0.2710 ± 0.0427 | - | 199.1 |
| `UNet+FaithfulGaussian` | torchregress | 0.9491 ± 0.0245 | 0.6809 ± 0.0200 | **2.8704 ± 0.2931** | 0.5343 ± 0.0186 | 0.5687 ± 0.0250 | - | 197.8 |
| `UNet+GaussianNLL` | torchregress | 0.9713 ± 0.0168 | 0.6835 ± 0.0213 | 3.7943 ± 0.4422 | 0.5420 ± 0.0193 | 0.5163 ± 0.0366 | - | 197.1 |
| `UNet+Huber` | torchregress | 0.9706 ± 0.0218 | 0.6846 ± 0.0115 | - | - | - | - | 197.4 |
| `UNet+MDN_K3` | torchregress | 0.9817 ± 0.0528 † | 0.6958 ± 0.0343 | - | - | 0.5756 ± 0.0651 | - | 202.4 |
| `UNet+MSE` | torchregress | 0.9610 ± 0.0350 | 0.6817 ± 0.0321 | - | - | - | - | 198.3 |
| `UNet+MultiQuantile` | torchregress | 1.0203 ± 0.0489 | 0.7178 ± 0.0344 | - | - | 0.5944 ± 0.0457 | - | 197.8 |
| `UNet+StudentT` | torchregress | 0.9688 ± 0.0216 | 0.6854 ± 0.0119 | - | - | - | - | 197.4 |
| `sklearn_RF` | scikit-learn | 1.3714 ± 0.0000 | 1.0223 ± 0.0000 | - | - | - | - | 287.0 |

##### depth_estimation / nyu_depth_v2_tiny

| Method | Library | RMSE | MAE | NLL | CRPS | Coverage | ValidPixelFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `UNet+BetaNLL` | torchregress | 1.7746 ± 0.2482 † | 1.4697 ± 0.2470 | **2.1858 ± 0.2486** | 1.0304 ± 0.1683 | 0.7582 ± 0.0658 | - | 2.2149 |
| `UNet+CRPS` | torchregress | 1.8430 ± 0.1650 † | 1.5855 ± 0.1492 | 2.2257 ± 0.2464 | 1.1119 ± 0.1254 | 0.7904 ± 0.0415 | - | 2.2281 |
| `UNet+DeepEnsemble_K3` | torchregress | 1.9681 ± 0.2686 † | 1.7155 ± 0.2945 | 219.7 ± 239.2 | 1.5049 ± 0.2849 | 0.1848 ± 0.0811 | - | 7.0800 |
| `UNet+Evidential` | torchregress | 1.7144 ± 0.1709 † | 1.3743 ± 0.1510 | - | - | 0.8649 ± 0.0596 | - | 2.2533 |
| `UNet+FaithfulGaussian` | torchregress | 1.9357 ± 0.2810 † | 1.6619 ± 0.2980 | 2.3127 ± 0.1610 | 1.1769 ± 0.1920 | 0.7248 ± 0.0323 | - | 2.3470 |
| `UNet+GaussianNLL` | torchregress | 1.8667 ± 0.2663 † | 1.4755 ± 0.2169 | 2.4779 ± 0.4283 | **1.0288 ± 0.1120** | 0.7637 ± 0.0519 | - | 2.3065 |
| `UNet+Huber` | torchregress | 1.7318 ± 0.2917 † | 1.4731 ± 0.3467 | - | - | - | - | 2.2305 |
| `UNet+MDN_K3` | torchregress | 2.0681 ± 0.1629 † | 1.7761 ± 0.1895 | - | - | 0.6079 ± 0.0398 | - | 2.3194 |
| `UNet+MSE` | torchregress | 2.0818 ± 0.1067 † | 1.8158 ± 0.1208 | - | - | - | - | 2.8786 |
| `UNet+MultiQuantile` | torchregress | 2.1666 ± 0.1974 † | 1.8370 ± 0.2634 | - | - | 0.0767 ± 0.0634 | - | 2.2271 |
| `UNet+StudentT` | torchregress | 1.8403 ± 0.0893 | 1.4578 ± 0.1200 | - | - | - | - | 2.2274 |
| `sklearn_RF` | scikit-learn | **1.0913 ± 0.0010** | **0.8265 ± 0.0010** | - | - | - | - | 2.0429 |

##### depth_estimation / synthetic_image_depth

| Method | Library | RMSE | MAE | NLL | CRPS | Coverage | ValidPixelFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `UNet+BetaNLL` | torchregress | 0.0395 ± 0.0054 † | 0.0256 ± 0.0083 | **-1.9707 ± 0.1757** | 0.0187 ± 0.0042 | 0.9947 ± 0.0002 | - | 21.7619 |
| `UNet+CRPS` | torchregress | 0.0331 ± 0.0009 | 0.0136 ± 0.0020 | 5.6388 ± 12.6026 | 0.0194 ± 0.0016 | 0.9966 ± 0.0001 | - | 21.8137 |
| `UNet+DeepEnsemble_K3` | torchregress | **0.0303 ± 0.0001** | **0.0084 ± 0.0005** | 30.8368 ± 24.5825 | **0.0072 ± 0.0002** | 0.8611 ± 0.0507 | - | 65.3706 |
| `UNet+Evidential` | torchregress | 0.0334 ± 0.0015 | 0.0163 ± 0.0033 | - | - | 0.9785 ± 0.0015 | - | 21.9093 |
| `UNet+FaithfulGaussian` | torchregress | 0.3411 ± 0.3847 † | 0.2887 ± 0.3351 | 0.9129 ± 2.5579 | 0.2281 ± 0.2775 | 0.5715 ± 0.4033 | - | 22.3948 |
| `UNet+GaussianNLL` | torchregress | 0.0456 ± 0.0093 † | 0.0216 ± 0.0046 | -1.7989 ± 0.2558 | 0.0232 ± 0.0067 | 0.9987 ± 0.0004 | - | 21.8539 |
| `UNet+Huber` | torchregress | 0.0358 ± 0.0027 † | 0.0185 ± 0.0030 | - | - | - | - | 21.7821 |
| `UNet+MDN_K3` | torchregress | 0.0394 ± 0.0032 † | 0.0165 ± 0.0031 | - | - | 0.7993 ± 0.1012 | - | 22.2751 |
| `UNet+MSE` | torchregress | 0.0326 ± 0.0003 | 0.0134 ± 0.0003 | - | - | - | - | 22.4394 |
| `UNet+MultiQuantile` | torchregress | 0.0347 ± 0.0017 | 0.0134 ± 0.0033 | - | - | 0.9830 ± 0.0083 | - | 21.7380 |
| `UNet+StudentT` | torchregress | 0.0346 ± 0.0025 † | 0.0166 ± 0.0040 | - | - | - | - | 21.7966 |
| `sklearn_RF` | scikit-learn | 0.0883 ± 0.0000 | 0.0650 ± 0.0000 | - | - | - | - | 15.4123 |

#### image_calibration (image)

Group `image`; 1 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑, NLL ↓, CRPS ↓, Coverage ·, Width ·, IntervalScore ↓, ECE ↓, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### image_calibration / synthetic_image_regression

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `IsotonicMeanCalibrator` | torchregress | 0.0573 ± 0.0001 | 0.0456 ± 0.0006 | 0.9398 ± 0.0002 | -1.4197 ± 0.0264 | 0.0324 ± 0.0001 | 0.8656 ± 0.0375 | 0.1711 ± 0.0141 | 0.2425 ± 0.0083 | 0.0047 ± 0.0019 | **0.0677 ± 0.0228** | 4.4757 |
| `Uncalibrated` | torchregress | **0.0567 ± 0.0022** | **0.0451 ± 0.0017** | **0.9409 ± 0.0045** | **-1.4327 ± 0.0220** | **0.0320 ± 0.0012** | 0.8633 ± 0.0186 | 0.1711 ± 0.0141 | **0.2397 ± 0.0024** | **0.0046 ± 0.0016** | 0.1478 ± 0.0281 | 5.2424 |
| `VarianceTemperatureScaler` | torchregress | **0.0567 ± 0.0022** | **0.0451 ± 0.0017** | **0.9409 ± 0.0045** | **-1.4327 ± 0.0220** | **0.0320 ± 0.0012** | 0.8633 ± 0.0186 | 0.1711 ± 0.0141 | **0.2397 ± 0.0024** | **0.0046 ± 0.0016** | 0.1478 ± 0.0281 | 4.4221 |

#### image_conformal (image)

Group `image`; 1 dataset(s). Columns: Coverage ·, Width ·, IntervalScore ↓, InfiniteIntervalFrac · (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: IntervalScore.

##### image_conformal / synthetic_image_regression

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+CNN` | torchregress | 0.8867 ± 0.0410 | 0.1816 ± 0.0193 | 0.2385 ± 0.0036 | 0.0000 ± 0.0000 | 4.7271 |
| `MAPIE_Split+CNN` | MAPIE | 0.8889 ± 0.0135 | 0.1874 ± 0.0216 | 0.2483 ± 0.0142 † | 0.0000 ± 0.0000 | 3.8796 |
| `Split+CNN` | torchregress | 0.8889 ± 0.0135 | 0.1877 ± 0.0214 | 0.2483 ± 0.0142 † | 0.0000 ± 0.0000 | 4.2619 |
| `Zoo_Density+CNN` | torchregress | 0.8867 ± 0.0088 | 0.1918 ± 0.0198 | 0.2651 ± 0.0153 † | 0.0000 ± 0.0000 | 3.6957 |
| `Zoo_Distributional+CNN` | torchregress | 0.9133 ± 0.0240 | 0.1980 ± 0.0167 | 0.2388 ± 0.0071 | 0.0000 ± 0.0000 | 4.7536 |
| `Zoo_Prevalence+CNN` | torchregress | 0.9600 ± 0.0265 | 0.2601 ± 0.0330 | 0.2777 ± 0.0218 † | 0.0000 ± 0.0000 | 3.8276 |
| `Zoo_Split+CNN` | torchregress | 0.8889 ± 0.0135 | 0.1874 ± 0.0216 | 0.2483 ± 0.0142 † | 0.0000 ± 0.0000 | 3.6687 |
| `Zoo_UACQR+CNN` | torchregress | 0.8911 ± 0.0386 | 0.1833 ± 0.0202 | **0.2383 ± 0.0033** | 0.0000 ± 0.0000 | 4.5245 |
| `crepes_Split+CNN` | crepes | 0.8889 ± 0.0135 | 0.1874 ± 0.0216 | 0.2483 ± 0.0142 † | 0.0000 ± 0.0000 | 3.7979 |

Failed / skipped:

- `torchcp_Split+CNN` (skipped, seed 0, 1, 2): skipped: ModuleNotFoundError: No module named 'torchcp.regression.predictors'

#### image_density_parity (image)

Group `image`; 3 dataset(s). Columns: NLL ↓, CRPS ↓, ES ↓, Cov90_marg ·, Cov90_joint ·, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: NLL.

##### image_density_parity / ixi_brainage

| Method | Library | NLL | CRPS | ES | Cov90_marg | Cov90_joint | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sbi_MAF` | sbi | 1.5196 ± 0.0191 | 0.6471 ± 0.0081 | 0.6471 ± 0.0081 | 0.8750 ± 0.0331 | 0.8875 ± 0.0500 | 0.1500 ± 0.0238 | 2.3693 |
| `sbi_MDN` | sbi | 1.5082 ± 0.0013 | 0.6489 ± 0.0034 | 0.6489 ± 0.0034 | 0.9167 ± 0.0315 | 0.9250 ± 0.0125 | 0.1617 ± 0.0166 | 1.1794 |
| `sbi_NSF` | sbi | 1.3963 ± 0.0287 | 0.6303 ± 0.0079 | 0.6303 ± 0.0079 | 0.9042 ± 0.0072 | 0.9000 ± 0.0125 | 0.1067 ± 0.0289 | 4.7315 |
| `sbi_zuko_MAF` | sbi | 1.4069 ± 0.1835 † | 0.5787 ± 0.1151 | 0.5787 ± 0.1151 | 0.8750 ± 0.0331 | 0.9000 ± 0.0250 | 0.1242 ± 0.0600 | 2.0159 |
| `sbi_zuko_NSF` | sbi | **1.2435 ± 0.3060** † | **0.5480 ± 0.1430** | **0.5480 ± 0.1430** | 0.8750 ± 0.0451 | 0.8625 ± 0.0451 | 0.1158 ± 0.0326 | 8.6729 |
| `torchregress_MAF` | torchregress | 1.5087 ± 0.0010 | 0.6415 ± 0.0036 | 0.6415 ± 0.0036 | 0.9042 ± 0.0382 | 0.9292 ± 0.0072 | 0.1542 ± 0.0029 | 1.1000 |
| `torchregress_MDN` | torchregress | 1.3932 ± 0.0105 | 0.6355 ± 0.0096 | 0.6355 ± 0.0096 | 0.8292 ± 0.0382 | 0.8417 ± 0.0144 | 0.1108 ± 0.0243 | 2.2029 |
| `torchregress_NSF` | torchregress | 1.2656 ± 0.2684 † | 0.5642 ± 0.1149 | 0.5642 ± 0.1149 | 0.8792 ± 0.0260 | 0.8958 ± 0.0315 | **0.0875 ± 0.0565** | 8.2649 |

##### image_density_parity / synthetic_image_keypoint

| Method | Library | NLL | CRPS | ES | Cov90_marg | Cov90_joint | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sbi_MAF` | sbi | 11.3783 ± 0.0073 | 0.5892 ± 0.0009 | 1.9739 ± 0.0030 | 0.9237 ± 0.0050 | 0.9896 ± 0.0036 | 0.1014 ± 0.0012 | 3.8224 |
| `sbi_MDN` | sbi | 11.3806 ± 0.0049 | 0.5898 ± 0.0003 | 1.9765 ± 0.0012 | 0.9227 ± 0.0055 | 0.9813 ± 0.0108 | 0.1012 ± 0.0027 | 1.8593 |
| `sbi_NSF` | sbi | 10.5001 ± 0.0471 | 0.5817 ± 0.0013 | 1.9674 ± 0.0032 | 0.8914 ± 0.0016 | 0.9375 ± 0.0165 | 0.0741 ± 0.0002 | 12.2575 |
| `sbi_zuko_MAF` | sbi | 9.7852 ± 2.7926 † | **0.5546 ± 0.0613** | **1.8704 ± 0.1813** | 0.9094 ± 0.0068 | 0.9667 ± 0.0191 | 0.1012 ± 0.0090 | 6.9326 |
| `torchregress_MAF` | torchregress | 11.3877 ± 0.0048 | 0.5894 ± 0.0003 | 1.9738 ± 0.0006 | 0.9201 ± 0.0051 | 0.9792 ± 0.0072 | 0.1031 ± 0.0018 | 2.3852 |
| `torchregress_MDN` | torchregress | 11.4285 ± 0.0174 | 0.5892 ± 0.0008 | 1.9762 ± 0.0022 | 0.9297 ± 0.0089 | 0.9854 ± 0.0072 | 0.0960 ± 0.0030 | 2.2790 |
| `torchregress_NSF` | torchregress | **9.1681 ± 2.0979** † | 0.5553 ± 0.0482 | 1.8907 ± 0.1415 | 0.8826 ± 0.0063 | 0.8812 ± 0.0563 | **0.0682 ± 0.0060** | 20.2246 |

Failed / skipped:

- `sbi_zuko_NSF` (failed, seed 0, 1, 2): failed: RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at "/home/conda/feedstock_root/build_artifacts/libtorch_…

##### image_density_parity / udacity_steering

| Method | Library | NLL | CRPS | ES | Cov90_marg | Cov90_joint | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sbi_MAF` | sbi | 1.2557 ± 0.2189 † | 0.4899 ± 0.0113 | 0.4899 ± 0.0113 | 0.8733 ± 0.0072 | 0.8742 ± 0.0088 | 0.1917 ± 0.0161 | 18.2734 |
| `sbi_MDN` | sbi | 6.708e-04 ± 7.133e-02 † | 0.4772 ± 0.0108 | 0.4772 ± 0.0108 | 0.9033 ± 0.0038 | 0.9025 ± 0.0156 | 0.2075 ± 0.0869 | 28.5879 |
| `sbi_NSF` | sbi | **-0.5954 ± 0.8103** † | 0.5037 ± 0.0078 | 0.5037 ± 0.0078 | 0.7892 ± 0.0426 | 0.7375 ± 0.0548 | 0.2333 ± 0.0555 | 27.3174 |
| `sbi_zuko_MAF` | sbi | 104.0 ± 87.5 † | **0.4582 ± 0.0350** | **0.4582 ± 0.0350** | 0.8467 ± 0.0104 | 0.8483 ± 0.0115 | **0.1625 ± 0.0261** | 22.9243 |
| `sbi_zuko_NSF` | sbi | -0.4968 ± 0.1572 † | 0.4942 ± 0.0012 | 0.4942 ± 0.0012 | 0.8667 ± 0.0118 | 0.8692 ± 0.0255 | 0.3217 ± 0.0213 | 31.1521 |
| `torchregress_MAF` | torchregress | 9.4583 ± 13.8870 † | 0.4676 ± 0.0450 | 0.4676 ± 0.0450 | 0.8708 ± 0.0339 | 0.8717 ± 0.0332 | 0.1800 ± 0.0467 | 19.2669 |
| `torchregress_MDN` | torchregress | 0.1382 ± 0.3248 † | 0.4871 ± 0.0027 | 0.4871 ± 0.0027 | 0.8950 ± 0.0066 | 0.8925 ± 0.0090 | 0.2517 ± 0.0138 | 22.0923 |
| `torchregress_NSF` | torchregress | -0.5144 ± 0.2620 † | 0.4955 ± 0.0040 | 0.4955 ± 0.0040 | 0.8758 ± 0.0191 | 0.8533 ± 0.0419 | 0.2675 ± 0.1083 | 32.7184 |

#### image_ensembles_uq (image)

Group `image`; 1 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑, NLL ↓, CRPS ↓, Coverage ·, Width ·, IntervalScore ↓, ECE ↓, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### image_ensembles_uq / synthetic_image_regression

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `DeepEnsemble_K5` | torchregress | **0.0533 ± 0.0001** | **0.0426 ± 0.0002** | **0.9479 ± 0.0002** | 13.0248 ± 3.3541 | 0.0373 ± 0.0005 | 0.2500 ± 0.0273 | 0.0369 ± 0.0030 | 0.5724 ± 0.0187 | 0.0867 ± 0.0026 | 0.4372 ± 0.0102 | 23.2241 |
| `MC_Dropout` | torchregress | 0.0567 ± 0.0022 | 0.0446 ± 0.0021 | 0.9410 ± 0.0046 | -0.1096 ± 0.3600 | 0.0340 ± 0.0017 | 0.6156 ± 0.0336 | 0.1017 ± 0.0060 | 0.3500 ± 0.0282 | 0.0211 ± 0.0037 | 0.2250 ± 0.0328 | 4.9709 |
| `SWAG` | torchregress | 0.0545 ± 0.0007 | 0.0433 ± 0.0005 | 0.9455 ± 0.0013 | 5.9861 ± 5.3908 | 0.0368 ± 0.0012 | 0.2989 ± 0.0860 | 0.0458 ± 0.0122 | 0.5254 ± 0.0703 | 0.0344 ± 0.0058 | 0.3806 ± 0.0201 | 4.2535 |
| `luqbox_GaussianNLL` | lightning-uq-box | 0.0533 ± 0.0008 | 0.0426 ± 0.0009 | 0.9478 ± 0.0016 | **-1.5005 ± 0.0086** | **0.0301 ± 0.0005** | 0.8856 ± 0.0102 | 0.1747 ± 0.0054 | **0.2268 ± 0.0023** | **0.0043 ± 0.0005** | **0.0611 ± 0.0182** | 4.9448 |
| `luqbox_MCDropout` | lightning-uq-box | 0.0787 ± 0.0141 † | 0.0613 ± 0.0098 | 0.8839 ± 0.0402 | -0.8788 ± 0.0773 | 0.0494 ± 0.0058 | 0.9956 ± 0.0051 | 0.4744 ± 0.0228 | 0.4761 ± 0.0220 | 0.0080 ± 0.0015 | 0.1731 ± 0.0280 | 5.8806 |

#### image_robust (image)

Group `image`; 5 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### image_robust / ixi_brainage

| Method | Library | RMSE | MAE | R2 | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `BalancedMSE` | torchregress | 15.5848 ± 0.4140 | 13.7528 ± 0.3888 | 0.2525 ± 0.0395 | 2.4250 |
| `Barron` | torchregress | 16.1743 ± 0.3816 | 13.7731 ± 0.6362 | 0.1949 ± 0.0382 | 2.9959 |
| `Cauchy` | torchregress | 18.6545 ± 2.4705 † | 15.6027 ± 1.1409 | -0.0830 ± 0.2855 | 2.3733 |
| `DensityWeighted` | torchregress | 15.6322 ± 0.3566 | 13.8757 ± 0.5322 | 0.2480 ± 0.0341 | 2.5430 |
| `FDS` | torchregress | 17.1167 ± 0.2087 | 15.4720 ± 0.2150 | 0.0986 ± 0.0221 | 4.8387 |
| `FocalR` | torchregress | 15.6290 ± 0.4716 | 13.7941 ± 0.7305 | 0.2481 ± 0.0451 | 2.5902 |
| `Huber` | torchregress | 15.7968 ± 0.7835 | 13.6747 ± 0.8403 | 0.2311 ± 0.0754 | 2.4647 |
| `LDS` | torchregress | 15.6035 ± 0.6342 | 13.7271 ± 0.5062 | 0.2502 ± 0.0608 | 2.4169 |
| `LightGBM` | lightgbm | **11.1555 ± 0.0000** | **8.9083 ± 0.0000** | **0.6172 ± 0.0000** | 0.2147 |
| `LogCosh` | torchregress | 15.6005 ± 0.2953 | 13.6161 ± 0.5073 | 0.2511 ± 0.0283 | 2.4891 |
| `MSE` | torchregress | 16.2817 ± 0.4255 | 13.8836 ± 0.3825 | 0.1841 ± 0.0427 | 2.7345 |
| `XGBoost` | xgboost | 11.5686 ± 0.0000 | 9.3163 ± 0.0000 | 0.5883 ± 0.0000 | 0.7971 |

##### image_robust / ixi_brainage_tiny

| Method | Library | RMSE | MAE | R2 | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `BalancedMSE` | torchregress | 14.9151 ± 0.0600 | 13.1622 ± 0.0791 | 0.0564 ± 0.0076 | 0.4686 |
| `Barron` | torchregress | 14.9707 ± 0.2459 | 13.1097 ± 0.1100 | 0.0492 ± 0.0314 | 0.5627 |
| `Cauchy` | torchregress | 15.0353 ± 0.3037 | 13.2068 ± 0.1587 | 0.0409 ± 0.0390 | 0.4481 |
| `DensityWeighted` | torchregress | 14.9536 ± 0.1938 | 13.1146 ± 0.1091 | 0.0515 ± 0.0247 | 0.5260 |
| `FDS` | torchregress | 18.4012 ± 0.2721 | 15.3526 ± 0.1289 | -0.4364 ± 0.0426 | 0.9212 |
| `FocalR` | torchregress | 14.9148 ± 0.0675 | 13.1598 ± 0.0820 | 0.0565 ± 0.0086 | 0.4967 |
| `Huber` | torchregress | 14.9705 ± 0.2461 | 13.1094 ± 0.1101 | 0.0493 ± 0.0314 | 0.4891 |
| `LDS` | torchregress | 15.4140 ± 0.4239 | 13.5394 ± 0.3053 | -0.0082 ± 0.0552 | 0.4456 |
| `LightGBM` | lightgbm | 14.3749 ± 0.0000 | 12.0975 ± 0.0000 | 0.1236 ± 0.0000 | 0.0171 |
| `LogCosh` | torchregress | 14.9708 ± 0.2452 | 13.1098 ± 0.1091 | 0.0492 ± 0.0313 | 0.4621 |
| `MSE` | torchregress | 14.9135 ± 0.0631 | 13.1590 ± 0.0818 | 0.0567 ± 0.0080 | 0.8903 |
| `XGBoost` | xgboost | **12.7689 ± 0.0000** | **10.7442 ± 0.0000** | **0.3085 ± 0.0000** | 0.0559 |

##### image_robust / synthetic_image_regression

| Method | Library | RMSE | MAE | R2 | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `BalancedMSE` | torchregress | 0.0544 ± 0.0017 | 0.0434 ± 0.0014 | 0.9457 ± 0.0035 | 5.0602 |
| `Barron` | torchregress | **0.0535 ± 0.0004** | 0.0427 ± 0.0004 | **0.9475 ± 0.0008** | 6.6118 |
| `Cauchy` | torchregress | 0.0541 ± 0.0007 | 0.0434 ± 0.0005 | 0.9463 ± 0.0014 | 5.1893 |
| `DensityWeighted` | torchregress | 0.0539 ± 0.0014 | 0.0430 ± 0.0013 | 0.9466 ± 0.0028 | 4.2776 |
| `FDS` | torchregress | 0.0599 ± 0.0044 † | 0.0469 ± 0.0031 | 0.9340 ± 0.0097 | 8.9402 |
| `FocalR` | torchregress | 0.0541 ± 0.0010 | 0.0432 ± 0.0008 | 0.9462 ± 0.0020 | 5.4517 |
| `Huber` | torchregress | 0.0535 ± 0.0004 | 0.0428 ± 0.0005 | 0.9474 ± 0.0008 | 5.2209 |
| `LDS` | torchregress | 0.0623 ± 0.0124 † | 0.0495 ± 0.0103 | 0.9269 ± 0.0298 | 4.1768 |
| `LightGBM` | lightgbm | 0.0566 ± 0.0000 | 0.0461 ± 0.0000 | 0.9412 ± 0.0000 | 0.3635 |
| `LogCosh` | torchregress | 0.0535 ± 0.0005 | **0.0427 ± 0.0003** | 0.9474 ± 0.0010 | 5.1279 |
| `MSE` | torchregress | 0.0536 ± 0.0004 | 0.0429 ± 0.0005 | 0.9473 ± 0.0008 | 5.1964 |
| `XGBoost` | xgboost | 0.0606 ± 0.0000 | 0.0477 ± 0.0000 | 0.9326 ± 0.0000 | 0.6880 |

##### image_robust / udacity_steering

| Method | Library | RMSE | MAE | R2 | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `BalancedMSE` | torchregress | 0.2365 ± 0.0052 | 0.1731 ± 0.0009 | 0.5209 ± 0.0212 | 11.4875 |
| `Barron` | torchregress | 0.2333 ± 0.0210 † | 0.1590 ± 0.0149 | 0.5317 ± 0.0858 | 14.3868 |
| `Cauchy` | torchregress | 0.2326 ± 0.0221 † | 0.1585 ± 0.0149 | 0.5339 ± 0.0883 | 11.3596 |
| `DensityWeighted` | torchregress | 0.2427 ± 0.0080 | 0.1707 ± 0.0052 | 0.4952 ± 0.0333 | 11.4688 |
| `FDS` | torchregress | 0.3177 ± 0.0485 † | 0.2469 ± 0.0548 | 0.1226 ± 0.2728 | 14.9923 |
| `FocalR` | torchregress | 0.2268 ± 0.0063 | 0.1554 ± 0.0055 | 0.5595 ± 0.0245 | 11.9961 |
| `Huber` | torchregress | 0.2313 ± 0.0159 † | 0.1561 ± 0.0114 | 0.5404 ± 0.0640 | 11.7518 |
| `LDS` | torchregress | 0.2814 ± 0.0237 † | 0.2160 ± 0.0178 | 0.3190 ± 0.1154 | 11.3445 |
| `LightGBM` | lightgbm | 0.2443 ± 0.0000 | 0.1553 ± 0.0000 | 0.4891 ± 0.0000 | 7.4712 |
| `LogCosh` | torchregress | **0.2231 ± 0.0091** | **0.1526 ± 0.0074** | **0.5735 ± 0.0343** | 11.7167 |
| `MSE` | torchregress | 0.2273 ± 0.0111 | 0.1550 ± 0.0083 | 0.5570 ± 0.0426 | 10.8720 |
| `XGBoost` | xgboost | 0.2616 ± 0.0000 | 0.1676 ± 0.0000 | 0.4142 ± 0.0000 | 7.6302 |

##### image_robust / udacity_steering_tiny

| Method | Library | RMSE | MAE | R2 | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `BalancedMSE` | torchregress | 0.3146 ± 0.0102 | 0.2462 ± 0.0148 | 0.1357 ± 0.0553 | 1.4387 |
| `Barron` | torchregress | 0.2861 ± 0.0034 | 0.1782 ± 0.0135 | 0.2854 ± 0.0169 | 1.6929 |
| `Cauchy` | torchregress | 0.2855 ± 0.0065 | 0.1760 ± 0.0148 | 0.2884 ± 0.0322 | 1.3032 |
| `DensityWeighted` | torchregress | 0.3401 ± 0.0162 | 0.2303 ± 0.0220 | -0.0112 ± 0.0972 | 1.4651 |
| `FDS` | torchregress | 0.3052 ± 0.0015 | 0.1963 ± 0.0281 | 0.1870 ± 0.0078 | 1.6289 |
| `FocalR` | torchregress | 0.2894 ± 0.0287 † | 0.1904 ± 0.0080 | 0.2639 ± 0.1489 | 1.4958 |
| `Huber` | torchregress | 0.2845 ± 0.0051 | 0.1807 ± 0.0093 | 0.2934 ± 0.0254 | 1.4031 |
| `LDS` | torchregress | 0.3497 ± 0.0368 † | 0.2663 ± 0.0412 | -0.0751 ± 0.2189 | 1.3798 |
| `LightGBM` | lightgbm | 0.2879 ± 0.0000 | 0.1899 ± 0.0000 | 0.2765 ± 0.0000 | 0.1604 |
| `LogCosh` | torchregress | 0.2834 ± 0.0067 | 0.1783 ± 0.0099 | 0.2986 ± 0.0333 | 1.3807 |
| `MSE` | torchregress | **0.2782 ± 0.0046** | **0.1737 ± 0.0161** | **0.3243 ± 0.0222** | 1.7200 |
| `XGBoost` | xgboost | 0.2844 ± 0.0000 | 0.1765 ± 0.0000 | 0.2940 ± 0.0000 | 0.6863 |

#### keypoint_regression (image)

Group `image`; 1 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑, NLL ↓, CRPS ↓, Coverage ·, Width ·, IntervalScore ↓, ECE ↓, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### keypoint_regression / synthetic_image_keypoint

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CNN+LowRankGaussian` | torchregress | 0.1952 ± 0.0077 | 0.1700 ± 0.0066 | 0.0654 ± 0.0731 | -0.2118 ± 0.0411 | 0.1138 ± 0.0046 | 0.9451 ± 0.0131 | 0.6542 ± 0.0289 | 0.6841 ± 0.0228 | 0.0041 ± 0.0004 | 0.0846 ± 0.0056 | 9.7518 |
| `CNN+MSE` | torchregress | 0.1957 ± 0.0040 | 0.1711 ± 0.0035 | 0.0614 ± 0.0387 | - | - | - | - | - | - | - | 4.8643 |
| `CNN+MultivariateGaussian` | torchregress | 0.1920 ± 0.0023 | 0.1659 ± 0.0050 | 0.0970 ± 0.0214 | **-0.2354 ± 0.0159** | **0.1115 ± 0.0022** | 0.9451 ± 0.0114 | 0.6294 ± 0.0094 | **0.6698 ± 0.0313** | 0.0040 ± 0.0010 | 0.1006 ± 0.0111 | 14.1022 |
| `CNN+NF` | torchregress | 0.2041 ± 0.0028 | 0.1767 ± 0.0017 | -0.0203 ± 0.0275 | - | - | 0.8758 ± 0.0328 | 0.6087 ± 0.0235 | 0.6985 ± 0.0152 | 0.0035 ± 0.0007 | **0.0368 ± 0.0176** | 24.9762 |
| `ResNet+LowRankGaussian` | torchregress | 0.2050 ± 0.0088 | 0.1653 ± 0.0078 | -0.0301 ± 0.0895 | 0.5970 ± 0.2146 | 0.1436 ± 0.0171 | 0.8461 ± 0.0753 | 1.1607 ± 0.4028 | 1.5232 ± 0.2362 | 0.0054 ± 0.0020 | 0.1042 ± 0.0542 | 19.1008 |
| `ResNet+MultivariateGaussian` | torchregress | 0.2423 ± 0.0352 † | 0.1780 ± 0.0235 | -0.4585 ± 0.4082 | 0.0943 ± 0.0771 | 0.1341 ± 0.0173 | 0.8560 ± 0.1296 | 0.9157 ± 0.3424 | 1.1733 ± 0.1609 | **0.0028 ± 0.0002** | 0.1235 ± 0.0390 | 22.7140 |
| `sklearn_RF` | scikit-learn | **0.1894 ± 0.0004** | **0.1629 ± 0.0003** | **0.1211 ± 0.0036** | - | - | - | - | - | - | - | 5.3693 |

Failed / skipped:

- `CNN+GaussianNLL` (failed, seed 0, 1, 2): failed: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory…
- `ResNet+GaussianNLL` (failed, seed 0, 1, 2): failed: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory…

### Group `sbi`

#### sbi_parity (sbi)

Group `sbi`; 5 dataset(s). Columns: C2ST ↓, RefLogProb ↑, MeanRMSE ↓, SBC_Coverage95 ·, TARP_ATC |x|↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: C2ST.

##### sbi_parity / sbibm_bernoulli_glm

| Method | Library | C2ST | RefLogProb | MeanRMSE | SBC_Coverage95 | TARP_ATC | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | 0.8006 ± 0.0069 | - | 0.5768 ± 0.0566 | 0.9017 ± 0.0060 | -0.0281 ± 0.0159 | 211.8 |
| `sbi_FMPE` | sbi | **0.6552 ± 0.0260** | **-3.5104 ± 0.0765** | **0.3229 ± 0.0593** | 0.9447 ± 0.0025 | **-0.0035 ± 0.0179** | 32.9940 |
| `sbi_NPE` | sbi | 0.6750 ± 0.0113 | -3.9499 ± 0.0919 | 0.4756 ± 0.0689 | 0.9563 ± 0.0049 | -0.0193 ± 0.0122 | 86.5382 |
| `sbi_NPE_MAF` | sbi | 0.6641 ± 0.0214 | -3.5858 ± 0.1721 | 0.4392 ± 0.0605 | 0.9493 ± 0.0097 | -0.0138 ± 0.0140 | 32.8269 |
| `sbi_NPE_MDN` | sbi | 0.6992 ± 0.0265 | -3.8857 ± 0.1831 | 0.7588 ± 0.1002 | 0.9610 ± 0.0066 | -0.0162 ± 0.0240 | 42.1076 |
| `sbi_NPSE` | sbi | 0.6918 ± 0.0341 | -4.3144 ± 0.2207 | 0.6676 ± 0.1686 | 0.9627 ± 0.0068 | -0.0222 ± 0.0162 | 39.1034 |
| `torchregress_MAF` | torchregress | 0.7240 ± 0.0238 | -3.9640 ± 0.1409 | 0.6605 ± 0.1321 | 0.9353 ± 0.0112 | -0.0404 ± 0.0343 | 18.7368 |
| `torchregress_MDN` | torchregress | 0.7866 ± 0.0237 | -4.8653 ± 0.1587 | 1.1267 ± 0.0882 | 0.9467 ± 0.0085 | -0.0187 ± 0.0254 | 19.5083 |
| `torchregress_NSF` | torchregress | 0.7493 ± 0.0126 | -4.2213 ± 0.2009 | 0.7053 ± 0.0767 | 0.9350 ± 0.0050 | -0.0179 ± 0.0179 | 50.9661 |

##### sbi_parity / sbibm_gaussian_linear

| Method | Library | C2ST | RefLogProb | MeanRMSE | SBC_Coverage95 | TARP_ATC | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | 0.7928 ± 0.0119 | - | 0.2056 ± 0.0081 | 0.9017 ± 0.0025 | -0.0134 ± 0.0276 | 212.4 |
| `sbi_FMPE` | sbi | 0.5676 ± 0.0093 | 0.5328 ± 0.0058 | 0.1023 ± 0.0041 | 0.9327 ± 0.0051 | **-3.539e-04 ± 1.969e-02** | 27.1613 |
| `sbi_NPE` | sbi | 0.5335 ± 0.0079 | 0.4764 ± 0.0208 | **0.0691 ± 0.0105** | 0.9413 ± 0.0021 | 0.0024 ± 0.0127 | 21.4417 |
| `sbi_NPE_MAF` | sbi | **0.5272 ± 0.0139** | **0.6982 ± 0.0266** | 0.0851 ± 0.0099 | 0.9370 ± 0.0060 | -0.0074 ± 0.0229 | 12.1510 |
| `sbi_NPE_MDN` | sbi | 0.5568 ± 0.0056 | 0.5836 ± 0.0035 | 0.1140 ± 0.0103 | 0.9393 ± 0.0023 | 0.0063 ± 0.0088 | 6.6817 |
| `sbi_NPSE` | sbi | 0.5638 ± 0.0163 | 0.4080 ± 0.0739 | 0.1228 ± 0.0279 | 0.9350 ± 0.0040 | -0.0039 ± 0.0159 | 29.2974 |
| `torchregress_MAF` | torchregress | 0.5739 ± 0.0231 | 0.4349 ± 0.1111 | 0.1307 ± 0.0230 | 0.9307 ± 0.0080 | -0.0017 ± 0.0298 | 18.8532 |
| `torchregress_MDN` | torchregress | 0.5483 ± 0.0307 † | 0.6160 ± 0.0825 | 0.1005 ± 0.0214 | 0.9363 ± 0.0065 | -0.0050 ± 0.0054 | 17.2585 |
| `torchregress_NSF` | torchregress | 0.6285 ± 0.0207 | 0.1055 ± 0.1116 | 0.1991 ± 0.0155 | 0.9267 ± 0.0102 | -7.450e-04 ± 1.550e-02 | 55.8973 |

##### sbi_parity / sbibm_gaussian_mixture

| Method | Library | C2ST | RefLogProb | MeanRMSE | SBC_Coverage95 | TARP_ATC | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | 0.5546 ± 0.0221 | - | 0.1068 ± 0.0282 | 0.9283 ± 0.0076 | -0.0054 ± 0.0234 | 28.2447 |
| `sbi_FMPE` | sbi | 0.5504 ± 0.0124 | -1.1321 ± 0.0558 | 0.1017 ± 0.0103 | 0.9133 ± 0.0029 | 0.0172 ± 0.0170 | 50.3531 |
| `sbi_NPE` | sbi | **0.5130 ± 0.0184** | -1.0418 ± 0.0854 | **0.0909 ± 0.0186** | 0.9383 ± 0.0029 | **-0.0038 ± 0.0191** | 46.9315 |
| `sbi_NPE_MAF` | sbi | 0.6519 ± 0.0607 † | -1.7280 ± 0.2404 | 0.1861 ± 0.0395 | 0.9083 ± 0.0153 | 0.0049 ± 0.0017 | 30.6877 |
| `sbi_NPE_MDN` | sbi | 0.5405 ± 0.0200 | **-1.0111 ± 0.0539** | 0.1705 ± 0.0204 | 0.9233 ± 0.0275 | -0.0116 ± 0.0251 | 18.5238 |
| `sbi_NPSE` | sbi | 0.5642 ± 0.0252 | -1.4945 ± 0.0400 | 0.1943 ± 0.0319 | 0.9433 ± 0.0144 | -0.0199 ± 0.0057 | 38.8293 |
| `torchregress_MAF` | torchregress | 0.6927 ± 0.0083 | -1.9706 ± 0.0680 | 0.2389 ± 0.0592 | 0.9033 ± 0.0058 | 0.0262 ± 0.0178 | 18.2945 |
| `torchregress_MDN` | torchregress | 0.5505 ± 0.0262 | -1.0754 ± 0.0496 | 0.1231 ± 0.0155 | 0.9250 ± 0.0087 | -0.0080 ± 0.0120 | 10.0413 |
| `torchregress_NSF` | torchregress | 0.6236 ± 0.0354 † | -1.5095 ± 0.2671 | 0.1996 ± 0.0412 | 0.9133 ± 0.0076 | -0.0101 ± 0.0278 | 44.9824 |

##### sbi_parity / sbibm_slcp

| Method | Library | C2ST | RefLogProb | MeanRMSE | SBC_Coverage95 | TARP_ATC | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | 0.8591 ± 0.0074 | - | 1.0855 ± 0.1714 | 0.9273 ± 0.0023 | -0.0056 ± 0.0184 | 85.8979 |
| `sbi_FMPE` | sbi | 0.8388 ± 0.0190 | -5.1805 ± 0.2875 | 1.0991 ± 0.1830 | 0.9127 ± 0.0076 | -0.0274 ± 0.0075 | 44.7988 |
| `sbi_NPE` | sbi | 0.8514 ± 0.0182 | -5.4277 ± 0.2926 | 1.2247 ± 0.0817 | 0.9440 ± 0.0020 | **0.0013 ± 0.0113** | 89.4797 |
| `sbi_NPE_MAF` | sbi | 0.9292 ± 0.0344 | -7.0640 ± 0.6146 | 1.0748 ± 0.0449 | 0.9087 ± 0.0083 | -0.0629 ± 0.0323 | 38.8177 |
| `sbi_NPE_MDN` | sbi | 0.8716 ± 0.0411 | -6.0776 ± 0.7335 | 0.9831 ± 0.3514 | 0.9080 ± 0.0122 | -0.0163 ± 0.0098 | 49.4763 |
| `sbi_NPSE` | sbi | **0.7904 ± 0.0035** | -4.9718 ± 0.1679 | **0.8385 ± 0.1044** | 0.9213 ± 0.0081 | -0.0253 ± 0.0107 | 91.4137 |
| `torchregress_MAF` | torchregress | 0.9166 ± 0.0046 | -6.3620 ± 0.1696 | 1.0100 ± 0.2634 | 0.9533 ± 0.0153 | -0.0123 ± 0.0292 | 19.0401 |
| `torchregress_MDN` | torchregress | 0.8971 ± 0.0441 | -6.7439 ± 0.9806 | 1.2191 ± 0.3324 | 0.9473 ± 0.0058 | 0.0153 ± 0.0303 | 11.9859 |
| `torchregress_NSF` | torchregress | 0.8047 ± 0.0224 | **-4.8872 ± 0.6058** | 0.8948 ± 0.1996 | 0.9500 ± 0.0087 | 0.0061 ± 0.0026 | 51.1311 |

##### sbi_parity / sbibm_twomoons

| Method | Library | C2ST | RefLogProb | MeanRMSE | SBC_Coverage95 | TARP_ATC | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | 0.5167 ± 0.0170 | - | 0.0425 ± 0.0162 | 0.9500 ± 0.0132 | -0.0196 ± 0.0069 | 27.3200 |
| `sbi_FMPE` | sbi | 0.6273 ± 0.0447 † | 2.5716 ± 0.1662 | 0.0386 ± 0.0101 | 0.9600 ± 0.0350 | -0.0306 ± 0.0071 | 24.6127 |
| `sbi_NPE` | sbi | **0.4982 ± 0.0049** | **3.4799 ± 0.0204** | 0.0187 ± 0.0026 | 0.9383 ± 0.0029 | -0.0116 ± 0.0152 | 67.4755 |
| `sbi_NPE_MAF` | sbi | 0.6285 ± 0.0777 † | 2.8318 ± 0.4924 | 0.0241 ± 0.0114 | 0.9583 ± 0.0153 | -0.0292 ± 0.0192 | 28.4913 |
| `sbi_NPE_MDN` | sbi | 0.5338 ± 0.0462 † | 2.6585 ± 0.2098 | 0.0419 ± 0.0243 | 0.9600 ± 0.0132 | -0.0475 ± 0.0103 | 13.2864 |
| `sbi_NPSE` | sbi | 0.5305 ± 0.0247 | 3.1439 ± 0.1865 | 0.0441 ± 0.0501 | 0.9450 ± 0.0427 | -0.0241 ± 0.0118 | 47.2402 |
| `torchregress_MAF` | torchregress | 0.5889 ± 0.0704 † | 2.9173 ± 0.3063 | 0.0180 ± 0.0044 | 0.9700 ± 0.0132 | -0.0116 ± 0.0132 | 18.1221 |
| `torchregress_MDN` | torchregress | 0.6861 ± 0.2181 † | 1.7621 ± 1.3203 | 0.0269 ± 0.0180 | 1.0000 ± 0.0000 | -0.0220 ± 0.0327 | 8.0581 |
| `torchregress_NSF` | torchregress | 0.5885 ± 0.0594 † | 3.0157 ± 0.2460 | **0.0161 ± 0.0041** | 0.9533 ± 0.0225 | **-0.0027 ± 0.0280** | 47.2139 |

### Group `tabular`

#### bayesian_linear (tabular)

Group `tabular`; 3 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑, NLL ↓, CRPS ↓, ECE ↓, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### bayesian_linear / california_housing

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BLR+VTS` | torchregress | **0.8513 ± 0.0070** | 0.6657 ± 0.0092 | **0.4558 ± 0.0090** | 1.2825 ± 0.0380 | 0.4775 ± 0.0065 | 0.0046 ± 0.0006 | 0.1468 ± 0.0215 | 0.0015 |
| `BLR_Linear` | torchregress | 1.0393 ± 0.2335 † | 0.6352 ± 0.0551 | 0.1617 ± 0.3642 | 1.2377 ± 0.1443 | 0.4605 ± 0.0426 | 0.0041 ± 0.0012 | 0.0867 ± 0.0214 | 0.0286 |
| `BayesianLinearHead` | torchregress | 0.8750 ± 0.0385 | 0.6811 ± 0.0544 | 0.4244 ± 0.0499 | 5.4221 ± 0.4785 | 0.5666 ± 0.0467 | 0.0060 ± 0.0013 | 0.3453 ± 0.0673 | 0.0013 |
| `BoTorch_SingleTaskGP` | BoTorch | 1.0249 ± 0.1706 † | 0.7759 ± 0.1552 | 0.1967 ± 0.2713 | 2.0953 ± 0.3104 | 0.5802 ± 0.1144 | **0.0041 ± 0.0002** | 0.1657 ± 0.0899 | 0.3304 |
| `RecursiveBayesianHead` | torchregress | 0.8544 ± 0.0377 | 0.6632 ± 0.0613 | 0.4512 ± 0.0478 | 6.9626 ± 1.8248 | 0.5600 ± 0.0472 | 0.0068 ± 0.0019 | 0.3705 ± 0.0625 | 0.0011 |
| `sklearn_BayesianRidge` | sklearn | 1.0393 ± 0.2335 † | **0.6352 ± 0.0551** | 0.1617 ± 0.3642 | **1.2377 ± 0.1443** | **0.4605 ± 0.0426** | 0.0041 ± 0.0012 | **0.0867 ± 0.0214** | 0.0014 |

##### bayesian_linear / diabetes

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BLR+VTS` | torchregress | 57.8405 ± 0.4585 | 46.0656 ± 1.3111 | 0.3802 ± 0.0098 | 5.4903 ± 0.0124 | 32.9177 ± 0.1882 | 0.0059 ± 0.0025 | 0.0943 ± 0.0556 | 0.0018 |
| `BLR_Linear` | torchregress | 56.4870 ± 0.3699 | 45.1353 ± 1.2086 | 0.4089 ± 0.0077 | 5.4670 ± 0.0186 | 32.1853 ± 0.2836 | **0.0057 ± 0.0012** | 0.0888 ± 0.0178 | 0.0308 |
| `BayesianLinearHead` | torchregress | **56.0371 ± 0.9273** | **44.0254 ± 0.7436** | **0.4182 ± 0.0193** | 9.8561 ± 0.9778 | 37.1181 ± 0.5907 | 0.0065 ± 0.0024 | 0.2988 ± 0.0093 | 0.0016 |
| `BoTorch_SingleTaskGP` | BoTorch | 68.9601 ± 4.5601 † | 54.8722 ± 5.0682 | 0.1165 ± 0.1148 | 5.8545 ± 0.1266 | 39.6646 ± 2.6062 | 0.0068 ± 0.0004 | 0.1362 ± 0.0197 | 0.5198 |
| `RecursiveBayesianHead` | torchregress | 57.5036 ± 0.6477 | 44.9650 ± 1.5442 | 0.3874 ± 0.0138 | 9.9088 ± 0.7752 | 37.9145 ± 1.0073 | 0.0083 ± 0.0018 | 0.2946 ± 0.0066 | 0.0013 |
| `sklearn_BayesianRidge` | sklearn | 56.4869 ± 0.3699 | 45.1353 ± 1.2086 | 0.4089 ± 0.0077 | **5.4670 ± 0.0186** | **32.1853 ± 0.2836** | 0.0057 ± 0.0012 | **0.0888 ± 0.0178** | 0.0018 |

##### bayesian_linear / synthetic_sinusoidal

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BLR+VTS` | torchregress | 0.7824 ± 0.0301 | 0.6722 ± 0.0232 | -0.0286 ± 0.0790 | 1.1993 ± 0.0565 | 0.4547 ± 0.0166 | **0.0042 ± 0.0009** | **0.0775 ± 0.0169** | 0.0019 |
| `BLR_Linear` | torchregress | 0.7638 ± 0.0055 | 0.6609 ± 0.0038 | 0.0207 ± 0.0141 | **1.1522 ± 0.0042** | 0.4430 ± 0.0030 | 0.0045 ± 0.0014 | 0.0778 ± 0.0285 | 0.0312 |
| `BayesianLinearHead` | torchregress | 0.7710 ± 0.0161 | 0.6643 ± 0.0232 | 0.0018 ± 0.0413 | 9.0505 ± 0.5077 | 0.5762 ± 0.0205 | 0.0137 ± 0.0018 | 0.3822 ± 0.0300 | 0.0016 |
| `BoTorch_SingleTaskGP` | BoTorch | **0.6321 ± 0.2316** † | **0.5272 ± 0.2045** | **0.2694 ± 0.4563** | 1.1912 ± 0.1427 | **0.3673 ± 0.1282** | 0.0046 ± 0.0017 | 0.1156 ± 0.0638 | 0.1402 |
| `RecursiveBayesianHead` | torchregress | 0.7731 ± 0.0168 | 0.6652 ± 0.0230 | -0.0035 ± 0.0434 | 8.6921 ± 1.1406 | 0.5750 ± 0.0157 | 0.0128 ± 0.0019 | 0.3786 ± 0.0232 | 0.0013 |
| `sklearn_BayesianRidge` | sklearn | 0.7638 ± 0.0055 | 0.6609 ± 0.0038 | 0.0207 ± 0.0141 | 1.1522 ± 0.0042 | 0.4430 ± 0.0030 | 0.0045 ± 0.0014 | 0.0778 ± 0.0285 | 0.0017 |

#### calibration (tabular)

Group `tabular`; 4 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑, NLL ↓, CRPS ↓, ECE ↓, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### calibration / california_housing

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `IsotonicMeanCalibrator` | torchregress | **0.6001 ± 0.0032** | **0.4022 ± 0.0097** | **0.7296 ± 0.0029** | 41.7264 ± 66.7823 | **0.2930 ± 0.0055** | 0.0042 ± 0.0008 | 0.0671 ± 0.0156 | 0.0010 |
| `Uncalibrated` | torchregress | 0.6060 ± 0.0050 | 0.4073 ± 0.0031 | 0.7243 ± 0.0046 | 64.9294 ± 104.4956 | 0.2949 ± 0.0012 | **0.0037 ± 0.0004** | **0.0570 ± 0.0209** | 0.0021 |
| `VTS_Floor` | torchregress | 0.6060 ± 0.0050 | 0.4073 ± 0.0031 | 0.7243 ± 0.0046 | 0.9084 ± 0.1192 | 0.2972 ± 0.0009 | 0.0042 ± 0.0007 | 0.0816 ± 0.0156 | 0.0588 |
| `VTS_Floor_Clip` | torchregress | 0.6060 ± 0.0050 | 0.4073 ± 0.0031 | 0.7243 ± 0.0046 | **0.9041 ± 0.1130** | 0.2964 ± 0.0012 | 0.0039 ± 0.0005 | 0.0685 ± 0.0224 | 0.0472 |
| `VarianceTemperatureScaler` | torchregress | 0.6060 ± 0.0050 | 0.4073 ± 0.0031 | 0.7243 ± 0.0046 | 10.7293 ± 12.9015 | 0.3362 ± 0.0665 | 0.0092 ± 0.0077 | 0.1370 ± 0.1002 | 9.394e-04 |
| `sklearn_IsotonicRegression` | sklearn | 0.6001 ± 0.0032 | 0.4022 ± 0.0097 | 0.7296 ± 0.0029 | 41.7264 ± 66.7823 | 0.2930 ± 0.0055 | 0.0042 ± 0.0008 | 0.0671 ± 0.0156 | 0.0025 |

##### calibration / diabetes

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `IsotonicMeanCalibrator` | torchregress | 58.4691 ± 2.0929 | 46.0007 ± 1.6165 | 0.3662 ± 0.0451 | 5.6154 ± 0.0973 | 33.4787 ± 1.0207 | 0.0076 ± 0.0008 | 0.1666 ± 0.0458 | 4.064e-04 |
| `Uncalibrated` | torchregress | **54.2510 ± 1.5541** | **42.8794 ± 1.8981** | **0.4545 ± 0.0315** | 5.4779 ± 0.0937 | 30.6618 ± 1.2330 | 0.0065 ± 0.0014 | 0.0948 ± 0.0232 | 1.267e-04 |
| `VTS_Floor` | torchregress | **54.2510 ± 1.5541** | **42.8794 ± 1.8981** | **0.4545 ± 0.0315** | **5.3963 ± 0.0371** | 30.3972 ± 1.0579 | **0.0052 ± 0.0005** | **0.0608 ± 0.0042** | 0.0403 |
| `VTS_Floor_Clip` | torchregress | **54.2510 ± 1.5541** | **42.8794 ± 1.8981** | **0.4545 ± 0.0315** | **5.3963 ± 0.0371** | 30.3972 ± 1.0579 | **0.0052 ± 0.0005** | **0.0608 ± 0.0042** | 0.0405 |
| `VarianceTemperatureScaler` | torchregress | **54.2510 ± 1.5541** | **42.8794 ± 1.8981** | **0.4545 ± 0.0315** | 5.4015 ± 0.0408 | **30.3785 ± 1.0362** | 0.0063 ± 0.0012 | 0.0735 ± 0.0176 | 4.989e-04 |
| `sklearn_IsotonicRegression` | sklearn | 58.4691 ± 2.0929 | 46.0007 ± 1.6165 | 0.3662 ± 0.0451 | 5.6154 ± 0.0973 | 33.4787 ± 1.0207 | 0.0076 ± 0.0008 | 0.1666 ± 0.0458 | 9.707e-04 |

##### calibration / synthetic_hetero

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `IsotonicMeanCalibrator` | torchregress | 0.3972 ± 0.0115 | 0.3070 ± 0.0085 | 0.8147 ± 0.0107 | 0.5598 ± 0.0841 | 0.2208 ± 0.0074 | 0.0040 ± 0.0003 | 0.0622 ± 0.0105 | 6.859e-04 |
| `Uncalibrated` | torchregress | **0.3827 ± 0.0029** | **0.2925 ± 0.0018** | **0.8281 ± 0.0027** | 0.5015 ± 0.0583 | 0.2112 ± 0.0024 | 0.0039 ± 0.0005 | 0.0562 ± 0.0079 | 0.0017 |
| `VTS_Floor` | torchregress | **0.3827 ± 0.0029** | **0.2925 ± 0.0018** | **0.8281 ± 0.0027** | 0.4408 ± 0.0323 | 0.2107 ± 0.0024 | 0.0029 ± 0.0003 | 0.0589 ± 0.0149 | 0.0445 |
| `VTS_Floor_Clip` | torchregress | **0.3827 ± 0.0029** | **0.2925 ± 0.0018** | **0.8281 ± 0.0027** | **0.4392 ± 0.0338** | **0.2105 ± 0.0022** | 0.0030 ± 0.0003 | **0.0548 ± 0.0130** | 0.0437 |
| `VarianceTemperatureScaler` | torchregress | **0.3827 ± 0.0029** | **0.2925 ± 0.0018** | **0.8281 ± 0.0027** | 0.4507 ± 0.0353 | 0.2114 ± 0.0032 | **0.0025 ± 0.0006** | 0.0582 ± 0.0166 | 7.378e-04 |
| `sklearn_IsotonicRegression` | sklearn | 0.3972 ± 0.0115 | 0.3070 ± 0.0085 | 0.8147 ± 0.0107 | 0.5598 ± 0.0841 | 0.2208 ± 0.0074 | 0.0040 ± 0.0003 | 0.0622 ± 0.0105 | 0.0012 |

##### calibration / synthetic_outliers

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `IsotonicMeanCalibrator` | torchregress | 1.3095 ± 0.0187 | 0.7736 ± 0.0092 | 0.7813 ± 0.0062 | 2.1789 ± 0.3974 | 0.6345 ± 0.0081 | 0.0087 ± 0.0007 | 0.1230 ± 0.0339 | 5.456e-04 |
| `Uncalibrated` | torchregress | **1.2602 ± 0.0138** | **0.7073 ± 0.0212** | **0.7975 ± 0.0044** | 2.0914 ± 0.3930 | 0.5988 ± 0.0100 | 0.0094 ± 0.0008 | 0.1420 ± 0.0140 | 0.0014 |
| `VTS_Floor` | torchregress | **1.2602 ± 0.0138** | **0.7073 ± 0.0212** | **0.7975 ± 0.0044** | **1.6601 ± 0.0124** | 0.6134 ± 0.0135 | **0.0064 ± 0.0001** | 0.1898 ± 0.0236 | 0.0357 |
| `VTS_Floor_Clip` | torchregress | **1.2602 ± 0.0138** | **0.7073 ± 0.0212** | **0.7975 ± 0.0044** | 2.5301 ± 0.1677 | **0.5767 ± 0.0128** | 0.0066 ± 0.0007 | **0.0730 ± 0.0161** | 0.0357 |
| `VarianceTemperatureScaler` | torchregress | **1.2602 ± 0.0138** | **0.7073 ± 0.0212** | **0.7975 ± 0.0044** | 1.8069 ± 0.1817 | 0.6466 ± 0.0082 | 0.0075 ± 0.0015 | 0.2087 ± 0.0022 | 6.384e-04 |
| `sklearn_IsotonicRegression` | sklearn | 1.3095 ± 0.0187 | 0.7736 ± 0.0092 | 0.7813 ± 0.0062 | 2.1789 ± 0.3974 | 0.6345 ± 0.0081 | 0.0087 ± 0.0007 | 0.1230 ± 0.0339 | 9.304e-04 |

#### causal (tabular)

Group `tabular`; 2 dataset(s). Columns: ATEErr |x|↓, PEHE ↓, Covers ·, PolicyErr |x|↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: ATEErr.

##### causal / folktables_causal

| Method | Library | ATEErr | PEHE | Covers | PolicyErr | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `DR` | torchregress | 0.5473 ± 0.0539 † | 1.1839 ± 0.3695 | 0.0000 ± 0.0000 | - | 4.6853 |
| `EconML_DRLearner` | econml | 0.5349 ± 0.0137 | 1.1256 ± 0.2675 | 0.0000 ± 0.0000 | - | 2.0511 |
| `NaiveDiM` | sklearn | **0.4811 ± 0.0000** | **0.7202 ± 0.0000** | 0.0000 ± 0.0000 | - | 0.0020 |
| `Policy_AllTreat` | torchregress | - | - | - | 1.5932 ± 0.0134 | 5.4563 |
| `Policy_DR` | torchregress | - | - | - | **1.3793 ± 0.1509** | 10.1193 |
| `Policy_EconML` | torchregress | - | - | - | 1.4227 ± 0.0551 | 7.4066 |
| `TLearner` | sklearn | 0.5882 ± 0.0037 | 0.7293 ± 0.0044 | 0.0000 ± 0.0000 | - | 0.9710 |

##### causal / ihdp

| Method | Library | ATEErr | PEHE | Covers | PolicyErr | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `DR` | torchregress | 2.1759 ± 0.0751 | 2.3460 ± 0.0997 | 0.0000 ± 0.0000 | - | 3.1557 |
| `EconML_DRLearner` | econml | **2.0880 ± 0.0436** | **2.2465 ± 0.0322** | 0.0000 ± 0.0000 | - | 1.2059 |
| `NaiveDiM` | sklearn | 2.1964 ± 0.0000 | 2.3953 ± 0.0000 | 0.0000 ± 0.0000 | - | 0.0013 |
| `Policy_AllTreat` | torchregress | - | - | - | 4.9248 ± 0.0345 | 3.2375 |
| `Policy_DR` | torchregress | - | - | - | **4.8754 ± 0.0218** | 6.4231 |
| `Policy_EconML` | torchregress | - | - | - | 4.9258 ± 0.0360 | 4.4132 |
| `TLearner` | sklearn | 2.1917 ± 0.0023 | 2.3135 ± 0.0031 | 0.0000 ± 0.0000 | - | 0.7748 |

#### censored (tabular)

Group `tabular`; 2 dataset(s). Columns: RMSE ↓, MAE ↓, Concordance ↑, ObservedMAE ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### censored / survset_hdfail

| Method | Library | RMSE | MAE | Concordance | ObservedMAE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `AFT` | torchregress | 849.9 ± 0.5 | 669.1 ± 0.7 | 0.8913 ± 0.0015 | 711.7 ± 0.3 | 13.2462 |
| `CensoredGaussianNLL` | torchregress | 1108.8 ± 837.1 † | 1006.1 ± 839.9 | 0.7177 ± 0.0202 | 819.8 ± 712.1 | 15.2971 |
| `CensoredQuantile` | torchregress | 626.3 ± 6.1 | 465.2 ± 5.9 | 0.7922 ± 0.0035 | 362.3 ± 4.6 | 11.3040 |
| `MSE` | torchregress | **285.2 ± 0.2** | **201.4 ± 0.3** | 0.7320 ± 0.0022 | **230.7 ± 4.2** | 7.0742 |
| `lifelines_LogNormalAFT` | lifelines | 6353.3 ± 0.0 | 5523.9 ± 0.0 | **0.8943 ± 0.0000** | 1803.4 ± 0.0 | 0.6843 |

##### censored / synthetic_censored

| Method | Library | RMSE | MAE | Concordance | ObservedMAE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `AFT` | torchregress | 3.6900 ± 0.5031 † | 3.5052 ± 0.2778 | 0.8498 ± 0.0376 | 3.7662 ± 0.8035 | 1.4116 |
| `CensoredGaussianNLL` | torchregress | 0.5081 ± 0.0085 | 0.3277 ± 0.0053 | 0.9519 ± 0.0002 | 0.1733 ± 0.0023 | 1.4304 |
| `CensoredQuantile` | torchregress | 0.3206 ± 0.0106 | 0.2332 ± 0.0043 | 0.9509 ± 0.0002 | 0.1687 ± 0.0028 | 1.1709 |
| `MSE` | torchregress | **0.1811 ± 0.0028** | **0.1369 ± 0.0016** | 0.9506 ± 0.0010 | **0.1646 ± 0.0013** | 0.4416 |
| `lifelines_LogNormalAFT` | lifelines | 1.1939 ± 0.0000 | 0.6379 ± 0.0000 | **0.9567 ± 0.0000** | 0.1966 ± 0.0000 | 0.2839 |

#### classification_shift (tabular)

Group `tabular`; 1 dataset(s). Columns: Accuracy ↑, BalancedAccuracy ↑, LogLoss ↓, Brier ↓, TargetPriorL1 ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: Accuracy.

##### classification_shift / folktables_acs_income

| Method | Library | Accuracy | BalancedAccuracy | LogLoss | Brier | TargetPriorL1 | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBSE` | torchregress | **0.8239 ± 0.0053** | 0.7558 ± 0.0063 | 0.4025 ± 0.0122 | 0.1256 ± 0.0045 | 0.0357 ± 0.0255 | 0.0231 |
| `DomainClassifierRatio` | torchregress | 0.7632 ± 0.0018 | **0.7817 ± 0.0046** | 0.5021 ± 0.0100 | 0.1640 ± 0.0033 | 0.5150 ± 0.1265 | 0.0214 |
| `EM` | torchregress | 0.8216 ± 0.0069 | 0.7570 ± 0.0079 | 0.4018 ± 0.0119 | **0.1255 ± 0.0045** | 0.0041 ± 0.0047 | 0.0221 |
| `EstimatedWeights` | torchregress | 0.8216 ± 0.0071 | 0.7583 ± 0.0072 | 0.4019 ± 0.0118 | 0.1255 ± 0.0045 | 0.0078 ± 0.0041 | 0.0214 |
| `Oracle` | torchregress | 0.8217 ± 0.0069 | 0.7578 ± 0.0076 | **0.4018 ± 0.0118** | 0.1255 ± 0.0045 | **0.0000 ± 0.0000** | 0.0211 |
| `PosteriorAdapter` | torchregress | 0.8216 ± 0.0069 | 0.7570 ± 0.0079 | 0.4018 ± 0.0119 | **0.1255 ± 0.0045** | 0.0041 ± 0.0047 | 0.0217 |
| `Uncorrected` | torchregress | 0.7632 ± 0.0018 | **0.7817 ± 0.0046** | 0.5021 ± 0.0100 | 0.1640 ± 0.0033 | 0.5880 ± 0.0000 | 0.0255 |

#### conformal (tabular)

Group `tabular`; 10 dataset(s). Columns: Coverage ·, Width ·, IntervalScore ↓, InfiniteIntervalFrac · (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: IntervalScore.

##### conformal / california_housing

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9144 ± 0.0114 | 1.8035 ± 0.0810 | 2.4124 ± 0.0115 | 0.0000 ± 0.0000 | 1.3197 |
| `CQR+TabularMLP` | torchregress | 0.9040 ± 0.0174 | 1.7250 ± 0.0892 | **2.3785 ± 0.0480** | 0.0000 ± 0.0000 | 2.2595 |
| `MAPIE_CQR+GBR` | MAPIE | 0.9076 ± 0.0180 | 2.1353 ± 0.0796 | 2.6959 ± 0.0443 | 0.0000 ± 0.0000 | 1.5773 |
| `MAPIE_CrossCQR+GBR` | MAPIE | 0.9102 ± 0.0032 | 2.0115 ± 0.0070 | 2.6251 ± 0.0014 | 0.0000 ± 0.0000 | 5.7919 |
| `MAPIE_Split+Linear` | MAPIE | 0.9051 ± 0.0077 | 2.1499 ± 0.0387 | 3.5667 ± 0.0703 | 0.0000 ± 0.0000 | 0.0058 |
| `MAPIE_Split+MLP` | MAPIE | 0.9124 ± 0.0008 | 1.8759 ± 0.0713 | 2.8677 ± 0.0339 | 0.0000 ± 0.0000 | 0.4980 |
| `Split+Linear` | torchregress | 0.9018 ± 0.0030 | 2.1404 ± 0.0316 | 3.5556 ± 0.0172 | 0.0000 ± 0.0000 | 0.5259 |
| `Split+Linear(post-hoc)` | torchregress | 0.9051 ± 0.0077 | 2.1499 ± 0.0386 | 3.5668 ± 0.0702 | 0.0000 ± 0.0000 | 0.0093 |
| `Split+MLP` | torchregress | 0.9104 ± 0.0066 | 1.8583 ± 0.0344 | 2.8665 ± 0.0405 | 0.0000 ± 0.0000 | 0.9576 |
| `Zoo_CVPlus` | torchregress | 0.9151 ± 0.0014 | 1.8263 ± 0.0303 | 2.7461 ± 0.0171 | 0.0000 ± 0.0000 | 2.9361 |
| `Zoo_Density` | torchregress | 0.9989 ± 0.0008 | 11.4389 ± 0.7186 | 11.4912 ± 0.6754 † | 0.0000 ± 0.0000 | 0.8045 |
| `Zoo_Distributional` | torchregress | 0.9071 ± 0.0146 | 1.6763 ± 0.0535 | 2.4436 ± 0.0921 | 0.0000 ± 0.0000 | 2.1631 |
| `Zoo_EnbPI` | torchregress | 0.9042 ± 0.0063 | 1.7518 ± 0.0324 | 2.8581 ± 0.0689 | 0.0000 ± 0.0000 | 3.7173 |
| `Zoo_Local` | torchregress | 0.9993 ± 0.0000 | 8.3323 ± 0.0461 | 8.3352 ± 0.0447 | 0.0038 ± 0.0008 | 0.8136 |
| `Zoo_LocalMAD` | torchregress | 0.9998 ± 0.0004 | 16.2200 ± 1.3812 | 16.2241 ± 1.3809 † | 0.0038 ± 0.0008 | 1.1812 |
| `Zoo_NexCP` | torchregress | 0.9082 ± 0.0105 | 1.8307 ± 0.0860 | 2.8665 ± 0.0445 | 0.0000 ± 0.0000 | 0.8031 |
| `Zoo_Prevalence` | torchregress | 0.9993 ± 0.0007 | 8.2344 ± 0.0502 | 8.2412 ± 0.0594 | 0.0000 ± 0.0000 | 0.8039 |
| `Zoo_Split` | torchregress | 0.9989 ± 0.0004 | 8.3463 ± 0.0475 | 8.3604 ± 0.0568 | 0.0000 ± 0.0000 | 0.9626 |
| `Zoo_UACQR` | torchregress | 0.9178 ± 0.0048 | 1.8096 ± 0.0400 | 2.3842 ± 0.0132 | 0.0000 ± 0.0000 | 1.9250 |
| `crepes_CQR+GBR` | crepes | 0.9087 ± 0.0110 | 2.0506 ± 0.0809 | 2.6818 ± 0.0509 | 0.0000 ± 0.0000 | 1.4907 |
| `crepes_Split+Linear` | crepes | 0.9051 ± 0.0077 | 2.1499 ± 0.0387 | 3.5667 ± 0.0703 | 0.0000 ± 0.0000 | 0.0036 |
| `torchcp_Split+Linear` | torchcp | 0.9051 ± 0.0077 | 2.1499 ± 0.0387 | 3.5667 ± 0.0703 | 0.0000 ± 0.0000 | 0.0089 |

##### conformal / diabetes

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9298 ± 0.0355 | 202.6 ± 14.3 | 222.8 ± 7.6 | 0.0000 ± 0.0000 | 0.1301 |
| `CQR+TabularMLP` | torchregress | 0.9298 ± 0.0189 | 184.6 ± 8.3 | **201.1 ± 3.9** | 0.0000 ± 0.0000 | 0.2861 |
| `MAPIE_CQR+GBR` | MAPIE | 0.9624 ± 0.0199 | 227.7 ± 6.9 | 235.0 ± 7.1 | 0.0000 ± 0.0000 | 0.4487 |
| `MAPIE_CrossCQR+GBR` | MAPIE | 0.9424 ± 0.0157 | 200.2 ± 6.1 | 213.6 ± 4.8 | 0.0000 ± 0.0000 | 4.0166 |
| `MAPIE_Split+Linear` | MAPIE | 0.9424 ± 0.0189 | 198.2 ± 4.7 | 222.5 ± 0.5 | 0.0000 ± 0.0000 | 0.0048 |
| `MAPIE_Split+MLP` | MAPIE | 0.9549 ± 0.0075 | 218.1 ± 3.6 | 236.3 ± 11.7 | 0.0000 ± 0.0000 | 0.0657 |
| `Split+Linear` | torchregress | 0.9173 ± 0.0130 | 201.1 ± 17.6 | 233.0 ± 11.9 † | 0.0000 ± 0.0000 | 0.0527 |
| `Split+Linear(post-hoc)` | torchregress | 0.9424 ± 0.0189 | 198.2 ± 4.7 | 222.5 ± 0.5 | 0.0000 ± 0.0000 | 0.0079 |
| `Split+MLP` | torchregress | 0.9524 ± 0.0230 | 213.6 ± 10.0 | 234.8 ± 3.6 | 0.0000 ± 0.0000 | 0.0994 |
| `Zoo_CVPlus` | torchregress | 0.9373 ± 0.0264 | 193.1 ± 7.7 | 221.5 ± 4.0 | 0.0000 ± 0.0000 | 0.3549 |
| `Zoo_Density` | torchregress | 1.0000 ± 0.0000 | 653.2 ± 112.1 | 653.2 ± 112.1 † | 0.0000 ± 0.0000 | 0.0982 |
| `Zoo_Distributional` | torchregress | 0.9749 ± 0.0115 | 238.2 ± 16.8 | 246.0 ± 17.9 † | 0.0000 ± 0.0000 | 0.2319 |
| `Zoo_EnbPI` | torchregress | 0.9148 ± 0.0313 | 188.3 ± 7.2 | 227.1 ± 10.6 | 0.0000 ± 0.0000 | 0.4702 |
| `Zoo_Local` | torchregress | 1.0000 ± 0.0000 | 538.7 ± 10.6 | 538.7 ± 10.6 | 0.0000 ± 0.0000 | 0.0967 |
| `Zoo_LocalMAD` | torchregress | 1.0000 ± 0.0000 | 787.4 ± 89.9 | 787.4 ± 89.9 † | 0.0000 ± 0.0000 | 0.1373 |
| `Zoo_NexCP` | torchregress | 0.9449 ± 0.0339 | 207.9 ± 12.7 | 233.8 ± 2.5 | 0.0000 ± 0.0000 | 0.0952 |
| `Zoo_Prevalence` | torchregress | 1.0000 ± 0.0000 | non-finite (3) | non-finite (3) | 1.0000 ± 0.0000 | 0.0963 |
| `Zoo_Split` | torchregress | 1.0000 ± 0.0000 | 537.3 ± 11.7 | 537.3 ± 11.7 | 0.0000 ± 0.0000 | 0.0966 |
| `Zoo_UACQR` | torchregress | 0.9298 ± 0.0414 | 203.5 ± 19.0 | 220.4 ± 7.8 | 0.0000 ± 0.0000 | 0.2250 |
| `crepes_CQR+GBR` | crepes | 0.9549 ± 0.0199 | 217.2 ± 1.2 | 226.5 ± 6.1 | 0.0000 ± 0.0000 | 0.3788 |
| `crepes_Split+Linear` | crepes | 0.9424 ± 0.0189 | 198.2 ± 4.7 | 222.5 ± 0.5 | 0.0000 ± 0.0000 | 0.0057 |
| `torchcp_Split+Linear` | torchcp | 0.9424 ± 0.0189 | 198.2 ± 4.7 | 222.5 ± 0.5 | 0.0000 ± 0.0000 | 0.0044 |

##### conformal / house_sales

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9024 ± 0.0045 | 2.916e+05 ± 6.727e+01 | 4.261e+05 ± 1.333e+04 | 0.0000 ± 0.0000 | 6.3202 |
| `CQR+TabularMLP` | torchregress | 0.9015 ± 0.0015 | 2.809e+05 ± 1.480e+04 | **3.916e+05 ± 5.436e+03** | 0.0000 ± 0.0000 | 8.9481 |
| `MAPIE_CQR+GBR` | MAPIE | 0.8968 ± 0.0009 | 3.465e+05 ± 1.878e+03 | 4.925e+05 ± 2.809e+03 | 0.0000 ± 0.0000 | 5.2456 |
| `MAPIE_CrossCQR+GBR` | MAPIE | 0.9031 ± 0.0008 | 3.488e+05 ± 3.613e+02 | 4.794e+05 ± 1.830e+03 | 0.0000 ± 0.0000 | 13.0707 |
| `MAPIE_Split+Linear` | MAPIE | 0.8913 ± 0.0095 | 5.086e+05 ± 1.673e+04 | 9.322e+05 ± 3.333e+03 | 0.0000 ± 0.0000 | 0.0079 |
| `MAPIE_Split+MLP` | MAPIE | 0.8987 ± 0.0049 | 3.407e+05 ± 9.895e+03 | 6.410e+05 ± 1.962e+03 | 0.0000 ± 0.0000 | 3.1837 |
| `Split+Linear` | torchregress | 0.8916 ± 0.0095 | 5.025e+05 ± 1.515e+04 | 9.319e+05 ± 3.714e+03 | 0.0000 ± 0.0000 | 1.7582 |
| `Split+Linear(post-hoc)` | torchregress | 0.8913 ± 0.0095 | 5.086e+05 ± 1.671e+04 | 9.322e+05 ± 3.326e+03 | 0.0000 ± 0.0000 | 0.0075 |
| `Split+MLP` | torchregress | 0.8964 ± 0.0084 | 3.179e+05 ± 9.176e+03 | 6.137e+05 ± 1.182e+04 | 0.0000 ± 0.0000 | 4.9261 |
| `Zoo_CVPlus` | torchregress | 0.9360 ± 0.0025 | 3.627e+05 ± 7.609e+02 | 5.307e+05 ± 2.974e+03 | 0.0000 ± 0.0000 | 13.9789 |
| `Zoo_Density` | torchregress | 0.9959 ± 0.0028 | 1.794e+07 ± 5.459e+06 | 1.798e+07 ± 5.442e+06 † | 0.0000 ± 0.0000 | 4.4626 |
| `Zoo_Distributional` | torchregress | 0.8997 ± 0.0023 | 3.108e+05 ± 4.631e+03 | 4.268e+05 ± 1.266e+04 | 0.0000 ± 0.0000 | 9.4537 |
| `Zoo_EnbPI` | torchregress | 0.9024 ± 0.0008 | 3.103e+05 ± 4.042e+03 | 5.882e+05 ± 3.274e+03 | 0.0000 ± 0.0000 | 17.7886 |
| `Zoo_Local` | torchregress | 1.0000 ± 0.0000 | 1.161e+07 ± 3.186e+06 | 1.161e+07 ± 3.186e+06 † | 3.333e-04 ± 0.000e+00 | 4.9239 |
| `Zoo_LocalMAD` | torchregress | 1.0000 ± 0.0000 | 3.060e+07 ± 9.352e+06 | 3.060e+07 ± 9.352e+06 † | 3.333e-04 ± 0.000e+00 | 6.1763 |
| `Zoo_NexCP` | torchregress | 0.8952 ± 0.0106 | 3.160e+05 ± 1.166e+04 | 6.140e+05 ± 1.228e+04 | 0.0000 ± 0.0000 | 3.6467 |
| `Zoo_Prevalence` | torchregress | 1.0000 ± 0.0000 | 1.140e+07 ± 3.199e+06 | 1.140e+07 ± 3.199e+06 † | 0.0000 ± 0.0000 | 4.9810 |
| `Zoo_Split` | torchregress | 1.0000 ± 0.0000 | 1.161e+07 ± 3.187e+06 | 1.161e+07 ± 3.187e+06 † | 0.0000 ± 0.0000 | 3.7184 |
| `Zoo_UACQR` | torchregress | 0.9018 ± 0.0017 | 3.046e+05 ± 2.513e+03 | 4.221e+05 ± 1.413e+04 | 0.0000 ± 0.0000 | 7.4659 |
| `crepes_CQR+GBR` | crepes | 0.8976 ± 0.0005 | 3.468e+05 ± 5.149e+02 | 4.923e+05 ± 3.145e+03 | 0.0000 ± 0.0000 | 5.0648 |
| `crepes_Split+Linear` | crepes | 0.8913 ± 0.0095 | 5.086e+05 ± 1.673e+04 | 9.322e+05 ± 3.333e+03 | 0.0000 ± 0.0000 | 0.0059 |
| `torchcp_Split+Linear` | torchcp | 0.8913 ± 0.0095 | 5.086e+05 ± 1.673e+04 | 9.322e+05 ± 3.333e+03 | 0.0000 ± 0.0000 | 0.0205 |

##### conformal / insurance_claims

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9156 ± 0.0138 | 0.2678 ± 0.0482 | 1.0270 ± 0.0087 | 0.0000 ± 0.0000 | 7.6165 |
| `CQR+TabularMLP` | torchregress | 0.8944 ± 0.0100 | 0.3079 ± 0.1296 | **0.9989 ± 0.0097** | 0.0000 ± 0.0000 | 8.5983 |
| `MAPIE_CQR+GBR` | MAPIE | 0.9654 ± 0.0278 | 0.3659 ± 0.5880 | 1.0692 ± 0.0191 | 0.0000 ± 0.0000 | 1.5899 |
| `MAPIE_CrossCQR+GBR` | MAPIE | 0.9493 ± 0.0000 | 0.0000 ± 0.0000 | 1.0667 ± 0.0000 | 0.0000 ± 0.0000 | 4.3671 |
| `MAPIE_Split+Linear` | MAPIE | 0.8938 ± 0.0028 | 0.1820 ± 0.0075 | 1.1127 ± 0.0019 | 0.0000 ± 0.0000 | 0.0150 |
| `MAPIE_Split+MLP` | MAPIE | 0.8997 ± 0.0092 | 0.5432 ± 0.0516 | 1.3978 ± 0.0103 | 0.0000 ± 0.0000 | 4.4145 |
| `Split+Linear` | torchregress | 0.8963 ± 0.0067 | 0.2070 ± 0.0048 | 1.1307 ± 0.0009 | 0.0000 ± 0.0000 | 2.1907 |
| `Split+Linear(post-hoc)` | torchregress | 0.8937 ± 0.0026 | 0.1819 ± 0.0075 | 1.1127 ± 0.0019 | 0.0000 ± 0.0000 | 0.0090 |
| `Split+MLP` | torchregress | 0.8964 ± 0.0073 | 0.5489 ± 0.0239 | 1.4887 ± 0.0156 | 0.0000 ± 0.0000 | 5.6609 |
| `Zoo_CVPlus` | torchregress | 0.9446 ± 0.0017 | 0.6759 ± 0.0404 | 1.3429 ± 0.0183 | 0.0000 ± 0.0000 | 17.2090 |
| `Zoo_Density` | torchregress | 0.9885 ± 0.0012 | 4.2491 ± 0.0410 | 4.3398 ± 0.0271 | 0.0000 ± 0.0000 | 4.1855 |
| `Zoo_Distributional` | torchregress | 0.8980 ± 0.0114 | 0.5947 ± 0.5858 | 1.3839 ± 0.3875 † | 0.0000 ± 0.0000 | 9.6320 |
| `Zoo_EnbPI` | torchregress | 0.9101 ± 0.0041 | 0.5246 ± 0.0567 | 1.3850 ± 0.0069 | 0.0000 ± 0.0000 | 20.0071 |
| `Zoo_Local` | torchregress | 0.9994 ± 0.0003 | 4.1539 ± 0.0271 | 4.1572 ± 0.0268 | 0.0000 ± 0.0000 | 3.4561 |
| `Zoo_LocalMAD` | torchregress | 0.9989 ± 0.0002 | 59.6927 ± 30.8795 | 59.6993 ± 30.8811 † | 0.0000 ± 0.0000 | 5.8643 |
| `Zoo_NexCP` | torchregress | 0.8987 ± 0.0096 | 0.5654 ± 0.0452 | 1.4893 ± 0.0153 | 0.0000 ± 0.0000 | 4.4343 |
| `Zoo_Prevalence` | torchregress | 0.9994 ± 0.0003 | 4.1604 ± 0.0249 | 4.1637 ± 0.0247 | 0.0226 ± 0.0050 | 3.2939 |
| `Zoo_Split` | torchregress | 0.9994 ± 0.0003 | 4.1562 ± 0.0261 | 4.1595 ± 0.0259 | 0.0000 ± 0.0000 | 4.1292 |
| `Zoo_UACQR` | torchregress | 0.8961 ± 0.0022 | 0.2522 ± 0.0268 | 1.0246 ± 0.0089 | 0.0000 ± 0.0000 | 10.3048 |
| `crepes_CQR+GBR` | crepes | 0.9493 ± 0.0000 | 0.0326 ± 0.0170 | 1.0426 ± 0.0316 | 0.0000 ± 0.0000 | 1.4538 |
| `crepes_Split+Linear` | crepes | 0.8940 ± 0.0049 | 0.1821 ± 0.0071 | 1.1128 ± 0.0019 | 0.0000 ± 0.0000 | 0.0119 |
| `torchcp_Split+Linear` | torchcp | 0.8940 ± 0.0049 | 0.1821 ± 0.0071 | 1.1128 ± 0.0019 | 0.0000 ± 0.0000 | 0.0311 |

##### conformal / openml_year

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9066 ± 0.0066 | 0.5395 ± 0.0166 | 0.7079 ± 0.0176 | 0.0000 ± 0.0000 | 2.9976 |
| `CQR+TabularMLP` | torchregress | 0.9074 ± 0.0044 | 0.5114 ± 0.0128 | **0.6602 ± 0.0074** | 0.0000 ± 0.0000 | 4.8019 |
| `MAPIE_CQR+GBR` | MAPIE | 0.9134 ± 0.0019 | 0.6677 ± 0.0191 | 0.7964 ± 0.0213 | 0.0000 ± 0.0000 | 2.3021 |
| `MAPIE_CrossCQR+GBR` | MAPIE | 0.9134 ± 0.0004 | 0.6452 ± 0.0037 | 0.7712 ± 0.0059 | 0.0000 ± 0.0000 | 7.8401 |
| `MAPIE_Split+Linear` | MAPIE | 0.9046 ± 0.0016 | 0.6829 ± 0.0127 | 0.9256 ± 0.0084 | 0.0000 ± 0.0000 | 0.0055 |
| `MAPIE_Split+MLP` | MAPIE | 0.8974 ± 0.0076 | 0.5032 ± 0.0233 | 0.7552 ± 0.0019 | 0.0000 ± 0.0000 | 0.7756 |
| `Split+Linear` | torchregress | 0.9039 ± 0.0011 | 0.6850 ± 0.0141 | 0.9281 ± 0.0109 | 0.0000 ± 0.0000 | 1.0381 |
| `Split+Linear(post-hoc)` | torchregress | 0.9046 ± 0.0016 | 0.6830 ± 0.0127 | 0.9256 ± 0.0084 | 0.0000 ± 0.0000 | 0.0090 |
| `Split+MLP` | torchregress | 0.9089 ± 0.0071 | 0.5163 ± 0.0182 | 0.7409 ± 0.0144 | 0.0000 ± 0.0000 | 2.1821 |
| `Zoo_CVPlus` | torchregress | 0.9221 ± 0.0010 | 0.5445 ± 0.0006 | 0.7326 ± 0.0041 | 0.0000 ± 0.0000 | 5.0120 |
| `Zoo_Density` | torchregress | 0.9994 ± 0.0002 | 3.5755 ± 0.1459 | 3.5832 ± 0.1469 | 0.0000 ± 0.0000 | 1.5828 |
| `Zoo_Distributional` | torchregress | 0.9050 ± 0.0094 | 0.5202 ± 0.0337 | 0.7061 ± 0.0236 | 0.0000 ± 0.0000 | 3.8642 |
| `Zoo_EnbPI` | torchregress | 0.9101 ± 0.0043 | 0.5172 ± 0.0044 | 0.7376 ± 0.0011 | 0.0000 ± 0.0000 | 6.3106 |
| `Zoo_Local` | torchregress | 1.0000 ± 0.0000 | 2.6233 ± 0.1072 | 2.6233 ± 0.1072 | 0.0029 ± 0.0015 | 1.2803 |
| `Zoo_LocalMAD` | torchregress | 0.9998 ± 0.0002 | 3.4035 ± 0.3324 | 3.4042 ± 0.3328 † | 0.0029 ± 0.0015 | 1.8796 |
| `Zoo_NexCP` | torchregress | 0.9132 ± 0.0090 | 0.5262 ± 0.0207 | 0.7424 ± 0.0122 | 0.0000 ± 0.0000 | 1.2649 |
| `Zoo_Prevalence` | torchregress | 1.0000 ± 0.0000 | 2.5375 ± 0.0955 | 2.5375 ± 0.0955 | 0.0000 ± 0.0000 | 1.3522 |
| `Zoo_Split` | torchregress | 1.0000 ± 0.0000 | 2.6440 ± 0.1136 | 2.6440 ± 0.1136 | 0.0000 ± 0.0000 | 1.6638 |
| `Zoo_UACQR` | torchregress | 0.9062 ± 0.0044 | 0.5410 ± 0.0158 | 0.7204 ± 0.0104 | 0.0000 ± 0.0000 | 3.0294 |
| `crepes_CQR+GBR` | crepes | 0.9133 ± 0.0027 | 0.6684 ± 0.0220 | 0.7968 ± 0.0222 | 0.0000 ± 0.0000 | 2.6385 |
| `crepes_Split+Linear` | crepes | 0.9046 ± 0.0016 | 0.6829 ± 0.0127 | 0.9256 ± 0.0084 | 0.0000 ± 0.0000 | 0.0041 |
| `torchcp_Split+Linear` | torchcp | 0.9046 ± 0.0016 | 0.6829 ± 0.0127 | 0.9256 ± 0.0084 | 0.0000 ± 0.0000 | 0.0121 |

##### conformal / synthetic_hetero

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9278 ± 0.0067 | 1.4371 ± 0.0449 | 1.6541 ± 0.0498 | 0.0000 ± 0.0000 | 0.4504 |
| `CQR+TabularMLP` | torchregress | 0.9172 ± 0.0195 | 1.2634 ± 0.0062 | **1.5264 ± 0.0280** | 0.0000 ± 0.0000 | 0.8155 |
| `MAPIE_CQR+GBR` | MAPIE | 0.9172 ± 0.0107 | 1.5244 ± 0.0739 | 1.7842 ± 0.0373 | 0.0000 ± 0.0000 | 0.6783 |
| `MAPIE_CrossCQR+GBR` | MAPIE | 0.9172 ± 0.0048 | 1.4750 ± 0.0213 | 1.7127 ± 0.0050 | 0.0000 ± 0.0000 | 4.7047 |
| `MAPIE_Split+Linear` | MAPIE | 0.9156 ± 0.0082 | 1.3773 ± 0.0389 | 1.6882 ± 0.0079 | 0.0000 ± 0.0000 | 0.0050 |
| `MAPIE_Split+MLP` | MAPIE | 0.9156 ± 0.0178 | 1.4041 ± 0.0864 | 1.7648 ± 0.0455 | 0.0000 ± 0.0000 | 0.1668 |
| `Split+Linear` | torchregress | 0.9161 ± 0.0180 | 1.3892 ± 0.1124 | 1.7188 ± 0.0518 | 0.0000 ± 0.0000 | 0.1791 |
| `Split+Linear(post-hoc)` | torchregress | 0.9156 ± 0.0082 | 1.3772 ± 0.0389 | 1.6882 ± 0.0079 | 0.0000 ± 0.0000 | 0.0030 |
| `Split+MLP` | torchregress | 0.9211 ± 0.0082 | 1.3725 ± 0.0439 | 1.7036 ± 0.0350 | 0.0000 ± 0.0000 | 0.3273 |
| `Zoo_CVPlus` | torchregress | 0.9339 ± 0.0075 | 1.4282 ± 0.0254 | 1.6904 ± 0.0119 | 0.0000 ± 0.0000 | 1.0708 |
| `Zoo_Density` | torchregress | 1.0000 ± 0.0000 | 15.2321 ± 2.7828 | 15.2321 ± 2.7828 † | 0.0000 ± 0.0000 | 0.2637 |
| `Zoo_Distributional` | torchregress | 0.8983 ± 0.0044 | 1.3593 ± 0.0457 | 1.7156 ± 0.0291 | 0.0000 ± 0.0000 | 0.7700 |
| `Zoo_EnbPI` | torchregress | 0.9211 ± 0.0054 | 1.3556 ± 0.0293 | 1.6803 ± 0.0212 | 0.0000 ± 0.0000 | 1.2976 |
| `Zoo_Local` | torchregress | 1.0000 ± 0.0000 | 9.9518 ± 0.8211 | 9.9518 ± 0.8211 † | 0.0000 ± 0.0000 | 0.2667 |
| `Zoo_LocalMAD` | torchregress | 1.0000 ± 0.0000 | 12.8375 ± 1.0734 | 12.8375 ± 1.0734 † | 0.0000 ± 0.0000 | 0.3894 |
| `Zoo_NexCP` | torchregress | 0.9150 ± 0.0142 | 1.3551 ± 0.0994 | 1.7044 ± 0.0441 | 0.0000 ± 0.0000 | 0.2619 |
| `Zoo_Prevalence` | torchregress | 1.0000 ± 0.0000 | 9.7699 ± 0.6796 | 9.7699 ± 0.6796 † | 0.0000 ± 0.0000 | 0.2636 |
| `Zoo_Split` | torchregress | 1.0000 ± 0.0000 | 9.9552 ± 0.8209 | 9.9552 ± 0.8209 † | 0.0000 ± 0.0000 | 0.3269 |
| `Zoo_UACQR` | torchregress | 0.9222 ± 0.0177 | 1.4613 ± 0.0825 | 1.7024 ± 0.0398 | 0.0000 ± 0.0000 | 0.6329 |
| `crepes_CQR+GBR` | crepes | 0.9078 ± 0.0035 | 1.4978 ± 0.0370 | 1.7775 ± 0.0275 | 0.0000 ± 0.0000 | 0.7947 |
| `crepes_Split+Linear` | crepes | 0.9156 ± 0.0082 | 1.3773 ± 0.0389 | 1.6882 ± 0.0079 | 0.0000 ± 0.0000 | 0.0043 |
| `torchcp_Split+Linear` | torchcp | 0.9156 ± 0.0082 | 1.3773 ± 0.0389 | 1.6882 ± 0.0079 | 0.0000 ± 0.0000 | 0.0071 |

##### conformal / tabred_cooking

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.8953 ± 0.0112 | 1.6725 ± 0.0537 | 2.1535 ± 0.0152 | 0.0000 ± 0.0000 | 3.3693 |
| `CQR+TabularMLP` | torchregress | 0.8929 ± 0.0116 | 1.5671 ± 0.0507 | 2.0356 ± 0.0139 | 0.0000 ± 0.0000 | 4.3580 |
| `MAPIE_CQR+GBR` | MAPIE | 0.8927 ± 0.0121 | 1.5863 ± 0.0339 | 2.0190 ± 0.0131 | 0.0000 ± 0.0000 | 57.9729 |
| `MAPIE_CrossCQR+GBR` | MAPIE | 0.8936 ± 0.0008 | 1.5807 ± 0.0070 | **2.0046 ± 0.0012** | 0.0000 ± 0.0000 | 77.0150 |
| `MAPIE_Split+Linear` | MAPIE | 0.8979 ± 0.0048 | 1.5725 ± 0.0200 | 2.0637 ± 0.0048 | 0.0000 ± 0.0000 | 0.0360 |
| `MAPIE_Split+MLP` | MAPIE | 0.9017 ± 0.0078 | 2.0816 ± 0.0295 | 2.6848 ± 0.0388 | 0.0000 ± 0.0000 | 3.4088 |
| `Split+Linear` | torchregress | 0.8960 ± 0.0068 | 1.5629 ± 0.0218 | 2.0660 ± 0.0234 | 0.0000 ± 0.0000 | 1.0876 |
| `Split+Linear(post-hoc)` | torchregress | 0.8979 ± 0.0055 | 1.5745 ± 0.0236 | 2.0653 ± 0.0059 | 0.0000 ± 0.0000 | 0.2240 |
| `Split+MLP` | torchregress | 0.9023 ± 0.0052 | 2.0541 ± 0.0174 | 2.6804 ± 0.0072 | 0.0000 ± 0.0000 | 3.3663 |
| `Zoo_CVPlus` | torchregress | 0.9569 ± 0.0027 | 2.2528 ± 0.0135 | 2.4522 ± 0.0100 | 0.0000 ± 0.0000 | 10.7322 |
| `Zoo_Density` | torchregress | 0.9911 ± 0.0021 | 7.7993 ± 0.6007 | 7.9236 ± 0.5914 † | 0.0000 ± 0.0000 | 2.8561 |
| `Zoo_Distributional` | torchregress | 0.8993 ± 0.0153 | 2.3561 ± 0.1017 | 2.9366 ± 0.0467 | 0.0000 ± 0.0000 | 6.6247 |
| `Zoo_EnbPI` | torchregress | 0.9041 ± 0.0071 | 1.8014 ± 0.0145 | 2.2828 ± 0.0097 | 0.0000 ± 0.0000 | 13.6326 |
| `Zoo_Local` | torchregress | 0.9999 ± 0.0002 | 5.8002 ± 0.1926 | 5.8002 ± 0.1926 | 0.0000 ± 0.0000 | 2.8379 |
| `Zoo_LocalMAD` | torchregress | 0.9993 ± 0.0006 | 9.9329 ± 0.3885 | 9.9355 ± 0.3912 | 0.0000 ± 0.0000 | 3.7258 |
| `Zoo_NexCP` | torchregress | 0.8986 ± 0.0035 | 2.0339 ± 0.0365 | 2.6801 ± 0.0069 | 0.0000 ± 0.0000 | 2.8308 |
| `Zoo_Prevalence` | torchregress | 0.9999 ± 0.0002 | 6.1497 ± 0.1903 | 6.1499 ± 0.1899 | 0.0000 ± 0.0000 | 2.8247 |
| `Zoo_Split` | torchregress | 0.9999 ± 0.0002 | 5.8060 ± 0.1859 | 5.8060 ± 0.1859 | 0.0000 ± 0.0000 | 2.7293 |
| `Zoo_UACQR` | torchregress | 0.8947 ± 0.0045 | 1.7749 ± 0.0250 | 2.2978 ± 0.0515 | 0.0000 ± 0.0000 | 6.2246 |
| `crepes_CQR+GBR` | crepes | 0.8921 ± 0.0132 | 1.5837 ± 0.0413 | 2.0199 ± 0.0121 | 0.0000 ± 0.0000 | 43.4293 |
| `crepes_Split+Linear` | crepes | 0.8979 ± 0.0048 | 1.5725 ± 0.0200 | 2.0637 ± 0.0048 | 0.0000 ± 0.0000 | 0.0310 |
| `torchcp_Split+Linear` | torchcp | 0.8979 ± 0.0048 | 1.5725 ± 0.0200 | 2.0637 ± 0.0048 | 0.0000 ± 0.0000 | 0.0351 |

##### conformal / tabred_delivery

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.8983 ± 0.0084 | 1.9352 ± 0.0201 | 2.4996 ± 0.0029 | 0.0000 ± 0.0000 | 5.8472 |
| `CQR+TabularMLP` | torchregress | 0.8971 ± 0.0012 | 1.8345 ± 0.0258 | **2.2351 ± 0.0204** | 0.0000 ± 0.0000 | 5.8455 |
| `MAPIE_CQR+GBR` | MAPIE | 0.9063 ± 0.0108 | 1.9225 ± 0.0405 | 2.2826 ± 0.0103 | 0.0000 ± 0.0000 | 76.3522 |
| `MAPIE_CrossCQR+GBR` | MAPIE | 0.8980 ± 0.0007 | 1.8808 ± 0.0023 | 2.2614 ± 0.0007 | 0.0000 ± 0.0000 | 105.3 |
| `MAPIE_Split+Linear` | MAPIE | 0.8879 ± 0.0027 | 1.7811 ± 0.0163 | 2.3359 ± 0.0075 | 0.0000 ± 0.0000 | 0.0411 |
| `MAPIE_Split+MLP` | MAPIE | 0.8989 ± 0.0037 | 2.4362 ± 0.0088 | 3.1283 ± 0.0301 | 0.0000 ± 0.0000 | 3.3954 |
| `Split+Linear` | torchregress | 0.8924 ± 0.0058 | 1.8043 ± 0.0205 | 2.3379 ± 0.0060 | 0.0000 ± 0.0000 | 1.6101 |
| `Split+Linear(post-hoc)` | torchregress | 0.8910 ± 0.0019 | 1.7909 ± 0.0063 | 2.3412 ± 0.0097 | 0.0000 ± 0.0000 | 0.4138 |
| `Split+MLP` | torchregress | 0.9006 ± 0.0079 | 2.4113 ± 0.0288 | 3.1124 ± 0.0682 | 0.0000 ± 0.0000 | 4.4513 |
| `Zoo_CVPlus` | torchregress | 0.9703 ± 0.0035 | 2.6965 ± 0.0369 | 2.8380 ± 0.0211 | 0.0000 ± 0.0000 | 11.2717 |
| `Zoo_Density` | torchregress | 0.9873 ± 0.0047 | 9.3655 ± 0.4295 | 9.5635 ± 0.4043 | 0.0000 ± 0.0000 | 2.8299 |
| `Zoo_Distributional` | torchregress | 1.0000 ± 0.0000 | non-finite (3) | non-finite (3) | 1.0000 ± 0.0000 | 6.8784 |
| `Zoo_EnbPI` | torchregress | 0.9093 ± 0.0039 | 2.0631 ± 0.0320 | 2.5393 ± 0.0151 | 0.0000 ± 0.0000 | 14.2999 |
| `Zoo_Local` | torchregress | 0.9999 ± 0.0002 | 6.6650 ± 0.1378 | 6.6656 ± 0.1388 | 0.0000 ± 0.0000 | 2.8294 |
| `Zoo_LocalMAD` | torchregress | 0.9997 ± 0.0003 | 13.9895 ± 0.6677 | 13.9945 ± 0.6651 | 0.0000 ± 0.0000 | 3.8703 |
| `Zoo_NexCP` | torchregress | 0.9022 ± 0.0049 | 2.4249 ± 0.0574 | 3.1118 ± 0.0676 | 0.0000 ± 0.0000 | 2.8754 |
| `Zoo_Prevalence` | torchregress | 1.0000 ± 0.0000 | 6.9876 ± 0.0516 | 6.9876 ± 0.0516 | 0.0000 ± 0.0000 | 2.8164 |
| `Zoo_Split` | torchregress | 0.9999 ± 0.0002 | 6.6721 ± 0.1350 | 6.6727 ± 0.1360 | 0.0000 ± 0.0000 | 3.6193 |
| `Zoo_UACQR` | torchregress | 0.9083 ± 0.0132 | 2.2071 ± 0.0704 | 2.7841 ± 0.0503 | 0.0000 ± 0.0000 | 6.5201 |
| `crepes_CQR+GBR` | crepes | 0.9068 ± 0.0095 | 1.9190 ± 0.0403 | 2.2820 ± 0.0103 | 0.0000 ± 0.0000 | 60.3321 |
| `crepes_Split+Linear` | crepes | 0.8879 ± 0.0027 | 1.7811 ± 0.0163 | 2.3359 ± 0.0076 | 0.0000 ± 0.0000 | 0.0361 |
| `torchcp_Split+Linear` | torchcp | 0.8879 ± 0.0027 | 1.7811 ± 0.0163 | 2.3359 ± 0.0076 | 0.0000 ± 0.0000 | 0.0409 |

##### conformal / tabred_weather

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9137 ± 0.0078 | 6.1409 ± 0.1962 | 7.8753 ± 0.0759 | 0.0000 ± 0.0000 | 5.1480 |
| `CQR+TabularMLP` | torchregress | 0.8994 ± 0.0049 | 5.6475 ± 0.0906 | **7.4207 ± 0.0851** | 0.0000 ± 0.0000 | 5.3011 |
| `MAPIE_CQR+GBR` | MAPIE | 0.9064 ± 0.0066 | 6.0531 ± 0.0941 | 7.8249 ± 0.0582 | 0.0000 ± 0.0000 | 33.6847 |
| `MAPIE_CrossCQR+GBR` | MAPIE | 0.9116 ± 0.0018 | 6.0166 ± 0.0233 | 7.7157 ± 0.0281 | 0.0000 ± 0.0000 | 48.3465 |
| `MAPIE_Split+Linear` | MAPIE | 0.9047 ± 0.0089 | 5.6613 ± 0.2017 | 7.8437 ± 0.0094 | 0.0000 ± 0.0000 | 0.0191 |
| `MAPIE_Split+MLP` | MAPIE | 0.8959 ± 0.0017 | 6.9041 ± 0.1898 | 9.5814 ± 0.0767 | 0.0000 ± 0.0000 | 2.6412 |
| `Split+Linear` | torchregress | 0.9043 ± 0.0064 | 5.7343 ± 0.1667 | 7.9558 ± 0.0492 | 0.0000 ± 0.0000 | 1.3014 |
| `Split+Linear(post-hoc)` | torchregress | 0.9069 ± 0.0079 | 5.6862 ± 0.1953 | 7.8308 ± 0.0095 | 0.0000 ± 0.0000 | 0.1851 |
| `Split+MLP` | torchregress | 0.9050 ± 0.0017 | 6.6293 ± 0.1884 | 9.0419 ± 0.2243 | 0.0000 ± 0.0000 | 3.8647 |
| `Zoo_CVPlus` | torchregress | 0.9622 ± 0.0030 | 7.6558 ± 0.0666 | 8.6219 ± 0.1093 | 0.0000 ± 0.0000 | 9.9918 |
| `Zoo_Density` | torchregress | 0.9996 ± 0.0004 | 77.7904 ± 5.7326 | 77.8065 ± 5.7540 † | 0.0000 ± 0.0000 | 2.6875 |
| `Zoo_Distributional` | torchregress | 0.9077 ± 0.0138 | 7.6332 ± 0.3437 | 9.3692 ± 0.0584 | 0.0000 ± 0.0000 | 6.6486 |
| `Zoo_EnbPI` | torchregress | 0.9158 ± 0.0024 | 6.1523 ± 0.0475 | 8.1100 ± 0.1437 | 0.0000 ± 0.0000 | 12.5002 |
| `Zoo_Local` | torchregress | 1.0000 ± 0.0000 | 47.1480 ± 1.7098 | 47.1480 ± 1.7098 | 0.0000 ± 0.0000 | 3.1215 |
| `Zoo_LocalMAD` | torchregress | 1.0000 ± 0.0000 | 88.4257 ± 5.5575 | 88.4257 ± 5.5575 † | 0.0000 ± 0.0000 | 4.1362 |
| `Zoo_NexCP` | torchregress | 0.9029 ± 0.0028 | 6.5732 ± 0.1362 | 9.0401 ± 0.2218 | 0.0000 ± 0.0000 | 2.5247 |
| `Zoo_Prevalence` | torchregress | 1.0000 ± 0.0000 | 48.9529 ± 1.8667 | 48.9529 ± 1.8667 | 0.0000 ± 0.0000 | 3.2238 |
| `Zoo_Split` | torchregress | 1.0000 ± 0.0000 | 47.6093 ± 1.6136 | 47.6093 ± 1.6136 | 0.0000 ± 0.0000 | 2.9804 |
| `Zoo_UACQR` | torchregress | 0.9168 ± 0.0107 | 6.6373 ± 0.3550 | 8.1521 ± 0.1979 | 0.0000 ± 0.0000 | 5.8363 |
| `crepes_CQR+GBR` | crepes | 0.9074 ± 0.0072 | 6.0343 ± 0.0820 | 7.8050 ± 0.0606 | 0.0000 ± 0.0000 | 26.5357 |
| `crepes_Split+Linear` | crepes | 0.9047 ± 0.0089 | 5.6612 ± 0.2018 | 7.8437 ± 0.0094 | 0.0000 ± 0.0000 | 0.0174 |
| `torchcp_Split+Linear` | torchcp | 0.9047 ± 0.0089 | 5.6612 ± 0.2018 | 7.8437 ± 0.0094 | 0.0000 ± 0.0000 | 0.0239 |

##### conformal / zurich_delays

| Method | Library | Coverage | Width | IntervalScore | InfiniteIntervalFrac | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.8989 ± 0.0009 | 240.3 ± 3.1 | 376.4 ± 1.0 | 0.0000 ± 0.0000 | 24.2455 |
| `CQR+TabularMLP` | torchregress | 0.8998 ± 0.0016 | 238.5 ± 2.3 | **352.4 ± 1.0** | 0.0000 ± 0.0000 | 38.4368 |
| `MAPIE_CQR+GBR` | MAPIE | 0.9006 ± 0.0013 | 243.1 ± 3.2 | 386.0 ± 1.1 | 0.0000 ± 0.0000 | 28.0377 |
| `MAPIE_CrossCQR+GBR` | MAPIE | 0.9014 ± 0.0003 | 241.2 ± 0.2 | 382.6 ± 0.5 | 0.0000 ± 0.0000 | 42.4910 |
| `MAPIE_Split+Linear` | MAPIE | 0.9029 ± 0.0011 | 237.9 ± 1.4 | 476.4 ± 0.3 | 0.0000 ± 0.0000 | 0.1009 |
| `MAPIE_Split+MLP` | MAPIE | 0.9035 ± 0.0001 | 251.0 ± 3.0 | 480.5 ± 3.4 | 0.0000 ± 0.0000 | 18.9244 |
| `Split+Linear` | torchregress | 0.9030 ± 0.0016 | 243.2 ± 2.9 | 479.1 ± 1.5 | 0.0000 ± 0.0000 | 8.5197 |
| `Split+Linear(post-hoc)` | torchregress | 0.9026 ± 0.0009 | 237.7 ± 1.1 | 476.4 ± 0.3 | 0.0000 ± 0.0000 | 0.0262 |
| `Split+MLP` | torchregress | 0.9031 ± 0.0025 | 249.9 ± 0.6 | 485.3 ± 9.3 | 0.0000 ± 0.0000 | 20.0223 |
| `Zoo_CVPlus` | torchregress | 0.9321 ± 0.0012 | 274.9 ± 6.3 | 427.7 ± 2.0 | 0.0000 ± 0.0000 | 72.6348 |
| `Zoo_Density` | torchregress | 0.9972 ± 0.0022 | 30519.9 ± 34880.5 | 30548.9 ± 34861.8 † | 0.0000 ± 0.0000 | 18.1097 |
| `Zoo_Distributional` | torchregress | 0.8994 ± 0.0026 | 258.1 ± 4.8 | 409.6 ± 5.3 | 0.0000 ± 0.0000 | 42.4754 |
| `Zoo_EnbPI` | torchregress | 0.9052 ± 0.0009 | 247.9 ± 3.2 | 480.3 ± 17.6 | 0.0000 ± 0.0000 | 88.9138 |
| `Zoo_Local` | torchregress | 0.9999 ± 0.0001 | 23059.6 ± 26124.7 | 23061.5 ± 26123.0 † | 3.333e-05 ± 0.000e+00 | 18.1770 |
| `Zoo_LocalMAD` | torchregress | 1.0000 ± 0.0000 | 48460.6 ± 55258.6 | 48460.6 ± 55258.6 † | 3.333e-05 ± 0.000e+00 | 24.6952 |
| `Zoo_NexCP` | torchregress | 0.9037 ± 0.0019 | 250.8 ± 3.5 | 485.4 ± 9.4 | 0.0000 ± 0.0000 | 18.1209 |
| `Zoo_Prevalence` | torchregress | 0.9999 ± 0.0001 | 23110.8 ± 26142.3 | 23112.7 ± 26140.7 † | 0.0000 ± 0.0000 | 18.0359 |
| `Zoo_Split` | torchregress | 0.9999 ± 0.0001 | 23059.1 ± 26124.0 | 23061.0 ± 26122.3 † | 0.0000 ± 0.0000 | 20.1045 |
| `Zoo_UACQR` | torchregress | 0.8974 ± 0.0003 | 247.4 ± 0.5 | 374.9 ± 3.4 | 0.0000 ± 0.0000 | 41.2089 |
| `crepes_CQR+GBR` | crepes | 0.9005 ± 0.0017 | 241.1 ± 0.8 | 386.0 ± 1.1 | 0.0000 ± 0.0000 | 18.6145 |
| `crepes_Split+Linear` | crepes | 0.9029 ± 0.0010 | 237.8 ± 1.3 | 476.4 ± 0.3 | 0.0000 ± 0.0000 | 0.0476 |
| `torchcp_Split+Linear` | torchcp | 0.9029 ± 0.0010 | 237.8 ± 1.3 | 476.4 ± 0.3 | 0.0000 ± 0.0000 | 0.0836 |

#### constrained_outputs (tabular)

Group `tabular`; 5 dataset(s). Columns: RMSE ↓, ViolationRate ↓, Pinball ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### constrained_outputs / con_bounded

| Method | Library | RMSE | ViolationRate | Pinball | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `BoundedHead` | torchregress | **0.1289 ± 0.0022** | **0.0000 ± 0.0000** | - | 0.9347 |
| `MLP` | torchregress | 0.1321 ± 0.0014 | 0.0267 ± 0.0050 | - | 0.8913 |
| `MLP+clip` | torchregress | 0.1319 ± 0.0014 | **0.0000 ± 0.0000** | - | 0.8799 |
| `RandomForest` | scikit-learn | 0.1323 ± 0.0026 | **0.0000 ± 0.0000** | - | 0.3493 |

##### constrained_outputs / con_lipschitz

| Method | Library | RMSE | ViolationRate | Pinball | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `MLP` | torchregress | 0.2974 ± 0.0185 † | - | - | 0.7488 |
| `SpectralNormWrapper` | torchregress | **0.2916 ± 0.0123** | - | - | 1.4139 |

##### constrained_outputs / con_nonneg

| Method | Library | RMSE | ViolationRate | Pinball | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `HGB_Poisson` | scikit-learn | 0.9603 ± 0.0930 † | **0.0000 ± 0.0000** | - | 0.2497 |
| `MLP` | torchregress | 0.9285 ± 0.0531 † | 0.0043 ± 0.0049 | - | 0.8729 |
| `MLP+clip` | torchregress | **0.9285 ± 0.0531** † | **0.0000 ± 0.0000** | - | 0.8755 |
| `NonNegativeHead` | torchregress | 0.9286 ± 0.0514 † | **0.0000 ± 0.0000** | - | 0.9124 |

##### constrained_outputs / con_quantiles

| Method | Library | RMSE | ViolationRate | Pinball | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `LightGBM_quantile` | lightgbm | 1.4482 ± 0.0638 | 0.7100 ± 0.0171 | 0.3898 ± 0.0147 | 0.4447 |
| `MLP_MQ` | torchregress | 1.4069 ± 0.0764 † | 0.0110 ± 0.0085 | 0.3747 ± 0.0161 | 1.0406 |
| `MLP_MQ+sort` | torchregress | 1.4069 ± 0.0764 † | **0.0000 ± 0.0000** | 0.3747 ± 0.0161 | 1.0271 |
| `NonCrossingSort` | torchregress | **1.4035 ± 0.0774** † | **0.0000 ± 0.0000** | **0.3744 ± 0.0153** | 1.1231 |

##### constrained_outputs / con_simplex

| Method | Library | RMSE | ViolationRate | Pinball | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `MLP` | torchregress | 0.0741 ± 0.0007 | 0.9967 ± 0.0012 | - | 0.8883 |
| `MLP+renormalize` | torchregress | 0.0740 ± 0.0007 | **0.0000 ± 0.0000** | - | 0.8771 |
| `RandomForest` | scikit-learn | 0.0755 ± 0.0015 | **0.0000 ± 0.0000** | - | 0.4202 |
| `SimplexHead` | torchregress | **0.0730 ± 0.0013** | **0.0000 ± 0.0000** | - | 0.9394 |

#### density_parity (tabular)

Group `tabular`; 7 dataset(s). Columns: NLL ↓, CRPS ↓, ES ↓, Cov90_marg ·, Cov90_joint ·, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: NLL.

##### density_parity / california_housing

| Method | Library | NLL | CRPS | ES | Cov90_marg | Cov90_joint | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | 0.3155 ± 0.0615 † | **0.2349 ± 0.0094** | **0.2349 ± 0.0094** | 0.8607 ± 0.0367 | 0.8533 ± 0.0428 | 0.0430 ± 0.0185 | 16.0617 |
| `sbi_MAF` | sbi | 0.6135 ± 0.0685 † | 0.2513 ± 0.0077 | 0.2513 ± 0.0077 | 0.8933 ± 0.0096 | 0.8970 ± 0.0079 | 0.0600 ± 0.0312 | 7.1826 |
| `sbi_MDN` | sbi | 0.3103 ± 0.0273 † | 0.2356 ± 0.0059 | 0.2356 ± 0.0059 | 0.8837 ± 0.0057 | 0.8767 ± 0.0176 | 0.0457 ± 0.0119 | 5.6955 |
| `sbi_NPE_NSF` | sbi | 0.2516 ± 0.0242 † | 0.2414 ± 0.0098 | 0.2414 ± 0.0098 | 0.8947 ± 0.0172 | 0.8897 ± 0.0133 | **0.0333 ± 0.0090** | 23.8026 |
| `sbi_NSF` | sbi | **0.1581 ± 0.0495** † | 0.2358 ± 0.0127 | 0.2358 ± 0.0127 | 0.8800 ± 0.0050 | 0.8777 ± 0.0110 | 0.0863 ± 0.0335 | 16.4971 |
| `sbi_zuko_MAF` | sbi | 1.8541 ± 2.1737 † | 0.2459 ± 0.0032 | 0.2459 ± 0.0032 | 0.8750 ± 0.0262 | 0.8747 ± 0.0316 | 0.0557 ± 0.0159 | 5.9790 |
| `sbi_zuko_NSF` | sbi | 0.2313 ± 0.0421 † | 0.2400 ± 0.0093 | 0.2400 ± 0.0093 | 0.8803 ± 0.0102 | 0.8903 ± 0.0111 | 0.0557 ± 0.0256 | 8.2807 |
| `torchregress_MAF` | torchregress | 1.5171 ± 1.3816 † | 0.2489 ± 0.0094 | 0.2489 ± 0.0094 | 0.9037 ± 0.0200 | 0.9067 ± 0.0182 | 0.0703 ± 0.0352 | 5.2069 |
| `torchregress_MDN` | torchregress | 0.3228 ± 0.0148 | 0.2351 ± 0.0056 | 0.2351 ± 0.0056 | 0.8800 ± 0.0062 | 0.8830 ± 0.0053 | 0.0510 ± 0.0171 | 4.6000 |
| `torchregress_NSF` | torchregress | 0.2378 ± 0.0246 † | 0.2369 ± 0.0012 | 0.2369 ± 0.0012 | 0.8767 ± 0.0146 | 0.8657 ± 0.0031 | 0.0610 ± 0.0101 | 7.7952 |

##### density_parity / energy_efficiency

| Method | Library | NLL | CRPS | ES | Cov90_marg | Cov90_joint | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | **-2.1681 ± 0.2893** † | 0.1152 ± 0.0064 | 0.1753 ± 0.0099 | 0.8752 ± 0.0144 | 0.8485 ± 0.0354 | 0.0749 ± 0.0113 | 14.8143 |
| `sbi_MAF` | sbi | -1.3077 ± 0.1632 † | 0.1181 ± 0.0239 | 0.1805 ± 0.0352 | 0.9156 ± 0.0169 | 0.8788 ± 0.0173 | 0.1003 ± 0.0104 | 2.8793 |
| `sbi_MDN` | sbi | -1.3326 ± 0.0845 † | 0.1531 ± 0.0047 | 0.2308 ± 0.0067 | 0.9300 ± 0.0144 | 0.9408 ± 0.0175 | 0.0779 ± 0.0171 | 1.3865 |
| `sbi_NPE_NSF` | sbi | -1.5914 ± 0.0252 | 0.1455 ± 0.0017 | 0.2199 ± 0.0027 | 0.8997 ± 0.0175 | 0.9351 ± 0.0156 | **0.0737 ± 0.0060** | 6.9876 |
| `sbi_NSF` | sbi | -1.3605 ± 0.1678 † | 0.1528 ± 0.0106 | 0.2305 ± 0.0148 | 0.8838 ± 0.0238 | 0.9062 ± 0.0164 | 0.1016 ± 0.0280 | 3.9736 |
| `sbi_zuko_MAF` | sbi | -1.4867 ± 0.1028 † | **0.1048 ± 0.0105** | **0.1611 ± 0.0155** | 0.8636 ± 0.0270 | 0.8615 ± 0.0397 | 0.1047 ± 0.0343 | 3.1087 |
| `sbi_zuko_NSF` | sbi | -1.3614 ± 0.1244 † | 0.1452 ± 0.0097 | 0.2202 ± 0.0132 | 0.8911 ± 0.0152 | 0.9105 ± 0.0402 | 0.0840 ± 0.0214 | 3.5166 |
| `torchregress_MAF` | torchregress | -1.4311 ± 0.1361 † | 0.1141 ± 0.0118 | 0.1742 ± 0.0166 | 0.8478 ± 0.0274 | 0.8730 ± 0.0180 | 0.1202 ± 0.0282 | 2.1745 |
| `torchregress_MDN` | torchregress | -0.7622 ± 0.1956 † | 0.1543 ± 0.0190 | 0.2327 ± 0.0260 | 0.8997 ± 0.0214 | 0.9091 ± 0.0260 | 0.0990 ± 0.0222 | 0.7716 |
| `torchregress_NSF` | torchregress | -1.3509 ± 0.3836 † | 0.1397 ± 0.0124 | 0.2117 ± 0.0177 | 0.8853 ± 0.0226 | 0.8975 ± 0.0368 | 0.0803 ± 0.0158 | 1.8866 |

##### density_parity / mt_enb

| Method | Library | NLL | CRPS | ES | Cov90_marg | Cov90_joint | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | **-2.2723 ± 0.2263** † | 0.1101 ± 0.0061 | 0.1680 ± 0.0090 | 0.8680 ± 0.0068 | 0.8658 ± 0.0432 | 0.0886 ± 0.0129 | 15.9941 |
| `sbi_MAF` | sbi | -1.2897 ± 0.1185 † | 0.1242 ± 0.0108 | 0.1879 ± 0.0187 | 0.9026 ± 0.0344 | 0.8853 ± 0.0135 | 0.1193 ± 0.0079 | 2.7267 |
| `sbi_MDN` | sbi | -1.2402 ± 0.2400 † | 0.1538 ± 0.0107 | 0.2322 ± 0.0157 | 0.9048 ± 0.0050 | 0.9437 ± 0.0412 | 0.1088 ± 0.0184 | 1.7054 |
| `sbi_NPE_NSF` | sbi | -1.6486 ± 0.0745 | 0.1411 ± 0.0046 | 0.2141 ± 0.0066 | 0.9167 ± 0.0252 | 0.9004 ± 0.0135 | 0.0791 ± 0.0247 | 8.0994 |
| `sbi_NSF` | sbi | -1.5248 ± 0.2938 † | 0.1501 ± 0.0120 | 0.2259 ± 0.0170 | 0.9058 ± 0.0149 | 0.9091 ± 0.0130 | 0.1046 ± 0.0344 | 4.2492 |
| `sbi_zuko_MAF` | sbi | -1.7314 ± 0.1961 † | **0.0959 ± 0.0037** | **0.1491 ± 0.0051** | 0.8647 ± 0.0397 | 0.8788 ± 0.0228 | 0.1130 ± 0.0453 | 3.3315 |
| `sbi_zuko_NSF` | sbi | -1.4338 ± 0.2467 † | 0.1447 ± 0.0070 | 0.2191 ± 0.0099 | 0.8929 ± 0.0374 | 0.9156 ± 0.0337 | 0.0871 ± 0.0298 | 2.7273 |
| `torchregress_MAF` | torchregress | -1.7258 ± 0.1422 † | 0.1027 ± 0.0114 | 0.1584 ± 0.0160 | 0.8788 ± 0.0037 | 0.8961 ± 0.0260 | 0.0975 ± 0.0404 | 2.6137 |
| `torchregress_MDN` | torchregress | -0.8535 ± 0.5340 † | 0.1539 ± 0.0182 | 0.2322 ± 0.0261 | 0.9026 ± 0.0086 | 0.9091 ± 0.0065 | 0.1369 ± 0.0703 | 1.0036 |
| `torchregress_NSF` | torchregress | -1.7311 ± 0.1930 † | 0.1224 ± 0.0085 | 0.1877 ± 0.0114 | 0.9069 ± 0.0104 | 0.9329 ± 0.0320 | **0.0726 ± 0.0170** | 2.9274 |

##### density_parity / mt_jura

| Method | Library | NLL | CRPS | ES | Cov90_marg | Cov90_joint | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | 2.6439 ± 0.1138 | **0.2495 ± 0.0045** | **0.5223 ± 0.0072** | 0.7284 ± 0.0071 | 0.4630 ± 0.0401 | 0.1765 ± 0.0271 | 16.7661 |
| `sbi_MAF` | sbi | 2.0941 ± 0.2264 † | 0.3190 ± 0.0495 | 0.6880 ± 0.1197 | 0.8951 ± 0.0255 | 0.8380 ± 0.0765 | 0.1557 ± 0.0115 | 0.8434 |
| `sbi_MDN` | sbi | 1.9151 ± 0.1175 † | 0.2802 ± 0.0020 | 0.5919 ± 0.0091 | 0.8611 ± 0.0245 | 0.8333 ± 0.0417 | 0.1340 ± 0.0607 | 0.6091 |
| `sbi_NPE_NSF` | sbi | 1.8902 ± 0.0929 | 0.2759 ± 0.0145 | 0.5845 ± 0.0278 | 0.8858 ± 0.0315 | 0.8750 ± 0.0367 | **0.0827 ± 0.0175** | 4.2604 |
| `sbi_NSF` | sbi | 1.7478 ± 0.3066 † | 0.2802 ± 0.0295 | 0.5900 ± 0.0632 | 0.8580 ± 0.0193 | 0.8333 ± 0.0845 | 0.1096 ± 0.0098 | 1.4769 |
| `sbi_zuko_MAF` | sbi | 1.9892 ± 0.2222 † | 0.2874 ± 0.0209 | 0.6081 ± 0.0502 | 0.8873 ± 0.0149 | 0.8148 ± 0.0350 | 0.1376 ± 0.0328 | 0.7278 |
| `sbi_zuko_NSF` | sbi | 1.6576 ± 0.1630 † | 0.2701 ± 0.0108 | 0.5744 ± 0.0250 | 0.8565 ± 0.0122 | 0.8426 ± 0.0350 | 0.1286 ± 0.0381 | 1.1237 |
| `torchregress_MAF` | torchregress | 2.0979 ± 0.2458 † | 0.2860 ± 0.0117 | 0.6044 ± 0.0342 | 0.8750 ± 0.0282 | 0.8241 ± 0.0699 | 0.1567 ± 0.0080 | 0.6621 |
| `torchregress_MDN` | torchregress | 1.9480 ± 0.2986 † | 0.2917 ± 0.0222 | 0.6189 ± 0.0452 | 0.9059 ± 0.0096 | 0.8472 ± 0.0417 | 0.0969 ± 0.0213 | 0.5157 |
| `torchregress_NSF` | torchregress | **1.6085 ± 0.1478** † | 0.2670 ± 0.0122 | 0.5672 ± 0.0279 | 0.8519 ± 0.0046 | 0.8796 ± 0.0080 | 0.1385 ± 0.0161 | 0.8754 |

##### density_parity / mt_rf1

| Method | Library | NLL | CRPS | ES | Cov90_marg | Cov90_joint | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | -3.3006 ± 3.0687 † | non-finite (3) | non-finite (3) | 0.8961 ± 0.0125 | 0.8733 ± 0.0225 | 0.1246 ± 0.0418 | 30.1319 |
| `sbi_MAF` | sbi | -7.1889 ± 8.8559 † | 0.1292 ± 0.1542 | 0.4718 ± 0.5502 | 0.9080 ± 0.0069 | 0.8743 ± 0.0121 | 0.0875 ± 0.0233 | 21.2485 |
| `sbi_MDN` | sbi | 43.8466 ± 33.3220 † | 0.1494 ± 0.1425 | 0.5459 ± 0.5005 | 0.9189 ± 0.0086 | 0.9330 ± 0.0262 | 0.0909 ± 0.0245 | 18.7271 |
| `sbi_NPE_NSF` | sbi | -10.4751 ± 2.6543 † | 0.0873 ± 0.0272 | 0.3522 ± 0.1004 | 0.8978 ± 0.0109 | 0.9230 ± 0.0151 | **0.0758 ± 0.0046** | 63.9704 |
| `sbi_NSF` | sbi | -13.8070 ± 5.6022 † | 0.0522 ± 0.0458 | 0.2072 ± 0.1775 | 0.8859 ± 0.0085 | 0.8990 ± 0.0036 | 0.0847 ± 0.0094 | 98.3215 |
| `sbi_zuko_MAF` | sbi | 11116.2 ± 17025.5 † | 0.1146 ± 0.0697 | 0.4973 ± 0.3100 | 0.9083 ± 0.0045 | 0.9060 ± 0.0046 | 0.1100 ± 0.0373 | 15.1412 |
| `sbi_zuko_NSF` | sbi | **-15.1936 ± 1.4012** † | **0.0354 ± 0.0105** | **0.1453 ± 0.0450** | 0.9078 ± 0.0054 | 0.9280 ± 0.0131 | 0.1117 ± 0.0266 | 80.9173 |
| `torchregress_MAF` | torchregress | 35486.0 ± 61450.0 † | 0.1163 ± 0.0591 | 0.4854 ± 0.1667 | 0.8910 ± 0.0083 | 0.8753 ± 0.0245 | 0.0984 ± 0.0053 | 20.2205 |
| `torchregress_MDN` | torchregress | 33.6979 ± 49.9972 † | 0.1117 ± 0.0498 | 0.4099 ± 0.1727 | 0.9152 ± 0.0149 | 0.9137 ± 0.0025 | 0.0985 ± 0.0248 | 12.7672 |
| `torchregress_NSF` | torchregress | -15.0640 ± 1.4917 † | 0.0385 ± 0.0143 | 0.1574 ± 0.0609 | 0.9013 ± 0.0136 | 0.9287 ± 0.0121 | 0.0996 ± 0.0267 | 61.6301 |

##### density_parity / mt_scm20d

| Method | Library | NLL | CRPS | ES | Cov90_marg | Cov90_joint | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | -1.3180 ± 0.2181 † | **0.1570 ± 0.0040** | **0.8034 ± 0.0214** | 0.8141 ± 0.0091 | 0.3847 ± 0.0370 | 0.0754 ± 0.0033 | 259.3 |
| `sbi_MAF` | sbi | -1.5619 ± 0.1468 † | 0.1909 ± 0.0036 | 0.9425 ± 0.0184 | 0.8581 ± 0.0022 | 0.7260 ± 0.0233 | 0.0630 ± 0.0301 | 25.0304 |
| `sbi_MDN` | sbi | -0.0138 ± 0.3165 † | 0.2801 ± 0.0225 | 1.3447 ± 0.0956 | 0.8866 ± 0.0054 | 0.7640 ± 0.0241 | 0.0376 ± 0.0076 | 21.3380 |
| `sbi_NPE_NSF` | sbi | -0.3514 ± 0.1990 † | 0.2526 ± 0.0029 | 1.2388 ± 0.0161 | 0.8798 ± 0.0036 | 0.7660 ± 0.0130 | **0.0373 ± 0.0042** | 82.2383 |
| `sbi_NSF` | sbi | -2.2358 ± 0.1111 | 0.1797 ± 0.0054 | 0.8995 ± 0.0275 | 0.8459 ± 0.0143 | 0.6413 ± 0.0438 | 0.0724 ± 0.0139 | 83.3474 |
| `sbi_zuko_MAF` | sbi | -0.9904 ± 0.7228 † | 0.2018 ± 0.0189 | 0.9962 ± 0.0883 | 0.8645 ± 0.0104 | 0.7553 ± 0.0477 | 0.0550 ± 0.0087 | 27.2107 |
| `sbi_zuko_NSF` | sbi | -2.2784 ± 0.2413 † | 0.1877 ± 0.0072 | 0.9334 ± 0.0244 | 0.8590 ± 0.0057 | 0.7527 ± 0.0343 | 0.0742 ± 0.0085 | 73.5859 |
| `torchregress_MAF` | torchregress | -1.4803 ± 0.2740 † | 20043.6 ± 20823.6 | 2.996e+05 ± 3.298e+05 | 0.8690 ± 0.0074 | 0.7617 ± 0.0266 | 0.0598 ± 0.0107 | 28.6619 |
| `torchregress_MDN` | torchregress | 0.3321 ± 0.4257 † | 0.2885 ± 0.0049 | 1.3707 ± 0.0167 | 0.8921 ± 0.0089 | 0.7507 ± 0.0399 | 0.0375 ± 0.0028 | 20.4451 |
| `torchregress_NSF` | torchregress | **-2.5702 ± 0.1099** | 0.1925 ± 0.0038 | 0.9556 ± 0.0206 | 0.8669 ± 0.0063 | 0.7750 ± 0.0481 | 0.0615 ± 0.0054 | 53.7504 |

##### density_parity / pzflow_galaxies

| Method | Library | NLL | CRPS | ES | Cov90_marg | Cov90_joint | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | **-2.9965 ± 0.0299** | **0.0121 ± 0.0005** | **0.0121 ± 0.0005** | 0.9143 ± 0.0159 | 0.9287 ± 0.0051 | 0.0783 ± 0.0332 | 24.6259 |
| `sbi_MAF` | sbi | -2.0366 ± 0.2103 † | 0.0311 ± 0.0050 | 0.0311 ± 0.0050 | 0.9023 ± 0.0087 | 0.9047 ± 0.0057 | 0.0633 ± 0.0146 | 39.4199 |
| `sbi_MDN` | sbi | -2.1658 ± 0.2868 † | 0.0312 ± 0.0100 | 0.0312 ± 0.0100 | 0.9323 ± 0.0100 | 0.9303 ± 0.0154 | 0.0983 ± 0.0321 | 18.1718 |
| `sbi_NPE_NSF` | sbi | -2.8234 ± 0.0211 | 0.0282 ± 0.0036 | 0.0282 ± 0.0036 | 0.8890 ± 0.0148 | 0.8940 ± 0.0201 | 0.0833 ± 0.0199 | 97.2321 |
| `sbi_NSF` | sbi | -2.7175 ± 0.1893 † | 0.0235 ± 0.0038 | 0.0235 ± 0.0038 | 0.9097 ± 0.0136 | 0.9183 ± 0.0183 | **0.0410 ± 0.0204** | 67.3038 |
| `sbi_zuko_MAF` | sbi | -2.3600 ± 0.1260 † | 0.0211 ± 0.0045 | 0.0211 ± 0.0045 | 0.9020 ± 0.0261 | 0.9073 ± 0.0280 | 0.0867 ± 0.0172 | 26.7475 |
| `sbi_zuko_NSF` | sbi | -2.6498 ± 0.1503 † | 0.0190 ± 0.0053 | 0.0190 ± 0.0053 | 0.9233 ± 0.0110 | 0.9247 ± 0.0176 | 0.0573 ± 0.0257 | 50.1784 |
| `torchregress_MAF` | torchregress | -2.3702 ± 0.0380 | 0.0225 ± 0.0014 | 0.0225 ± 0.0014 | 0.9080 ± 0.0118 | 0.9153 ± 0.0091 | 0.1003 ± 0.0433 | 31.1596 |
| `torchregress_MDN` | torchregress | -2.4001 ± 0.1847 † | 0.0237 ± 0.0072 | 0.0237 ± 0.0072 | 0.9323 ± 0.0046 | 0.9300 ± 0.0053 | 0.1120 ± 0.0226 | 24.7820 |
| `torchregress_NSF` | torchregress | -2.6724 ± 0.0632 | 0.0167 ± 0.0006 | 0.0167 ± 0.0006 | 0.9090 ± 0.0082 | 0.9193 ± 0.0092 | 0.0883 ± 0.0369 | 55.5351 |

#### distributional_bins (tabular)

Group `tabular`; 3 dataset(s). Columns: CoverageDeviationAbsMean ↓, Coverage ·, IntervalScore ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: CoverageDeviationAbsMean.

##### distributional_bins / california_housing

| Method | Library | CoverageDeviationAbsMean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `AsymmetricLaplace` | torchregress | - | 0.9051 ± 0.0052 | 2.8744 ± 0.0599 | 1.4176 |
| `AsymmetricLaplace/dev_summary` | torchregress | 0.0155 ± 0.0054 † | - | - | - |
| `BetaRegression` | torchregress | - | 0.8989 ± 0.0021 | 17.8133 ± 0.3779 | 1.1053 |
| `BetaRegression/dev_summary` | torchregress | 0.0391 ± 0.0265 † | - | - | - |
| `CTI` | torchregress | - | 0.9198 ± 0.0053 | 1.9766 ± 0.0853 | 8.7589 |
| `CTI/dev_summary` | torchregress | 0.0198 ± 0.0053 † | - | - | - |
| `ContrastiveFlow` | torchregress | - | 0.8476 ± 0.0609 | 2.5941 ± 0.5280 | 15.8839 |
| `ContrastiveFlow/dev_summary` | torchregress | 0.0543 ± 0.0592 † | - | - | - |
| `Evidential` | torchregress | - | 0.9136 ± 0.0015 | 2.2323 ± 0.0617 | 1.5329 |
| `Evidential/dev_summary` | torchregress | 0.0136 ± 0.0015 † | - | - | - |
| `GEV` | torchregress | - | 0.9042 ± 0.0071 | 2.8038 ± 0.0287 | 1.4011 |
| `GEV/dev_summary` | torchregress | 0.0199 ± 0.0076 † | - | - | - |
| `GaussianNLL` | torchregress | - | 0.9096 ± 0.0048 | 2.1344 ± 0.0371 | 1.4266 |
| `GaussianNLL/dev_summary` | torchregress | 0.0100 ± 0.0041 † | - | - | - |
| `JohnsonSU` | torchregress | - | 0.8967 ± 0.0087 | 4.2260 ± 0.1879 | 1.1024 |
| `JohnsonSU/dev_summary` | torchregress | 0.0338 ± 0.0059 † | - | - | - |
| `MC_CP` | torchregress | - | 0.9118 ± 0.0119 | 2.0631 ± 0.1073 | 8.7908 |
| `MC_CP/dev_summary` | torchregress | 0.0148 ± 0.0099 † | - | - | - |
| `MDN_K3` | torchregress | - | 0.8751 ± 0.0077 | 2.0818 ± 0.0239 | 1.2663 |
| `MDN_K3/dev_summary` | torchregress | 0.0251 ± 0.0078 † | - | - | - |
| `NF_Ensemble_K10` | torchregress | - | 0.9107 ± 0.0053 | 1.8379 ± 0.0186 | 88.0623 |
| `NF_Ensemble_K10/dev_summary` | torchregress | 0.0107 ± 0.0053 † | - | - | - |
| `NF_Ensemble_K2` | torchregress | - | 0.8758 ± 0.0122 | 1.9022 ± 0.0584 | 17.6274 |
| `NF_Ensemble_K2/dev_summary` | torchregress | 0.0245 ± 0.0119 † | - | - | - |
| `NF_Ensemble_K20` | torchregress | - | 0.9147 ± 0.0050 | **1.8257 ± 0.0141** | 175.8 |
| `NF_Ensemble_K20/dev_summary` | torchregress | 0.0147 ± 0.0050 † | - | - | - |
| `NF_Ensemble_K5` | torchregress | - | 0.9011 ± 0.0060 | 1.8334 ± 0.0135 | 43.9546 |
| `NF_Ensemble_K5/dev_summary` | torchregress | **0.0056 ± 0.0032** † | - | - | - |
| `NormalizingFlow` | torchregress | - | 0.8520 ± 0.0175 | 1.9853 ± 0.0124 | 12.6503 |
| `NormalizingFlow/dev_summary` | torchregress | 0.0483 ± 0.0178 † | - | - | - |
| `NormalizingFlow_recipe` | torchregress | - | 0.8804 ± 0.0080 | 1.9943 ± 0.0461 | 7.7680 |
| `NormalizingFlow_recipe/dev_summary` | torchregress | 0.0203 ± 0.0074 † | - | - | - |
| `RandomPartition` | torchregress | - | 0.8916 ± 0.0004 | 2.3655 ± 0.0044 | 4.0986 |
| `RandomPartition/dev_summary` | torchregress | 0.0189 ± 0.0134 † | - | - | - |
| `SLS_CP` | torchregress | - | 0.9133 ± 0.0066 | 2.2618 ± 0.1847 | 3.3376 |
| `SLS_CP/dev_summary` | torchregress | 0.0133 ± 0.0066 † | - | - | - |
| `SQR` | torchregress | - | 0.8784 ± 0.0330 | 3.0394 ± 0.1248 | 1.1657 |
| `SQR/dev_summary` | torchregress | 0.0314 ± 0.0176 † | - | - | - |
| `SinhArcsinh` | torchregress | - | 0.9038 ± 0.0019 | 3.1375 ± 0.1245 | 1.1763 |
| `SinhArcsinh/dev_summary` | torchregress | 0.0182 ± 0.0087 † | - | - | - |
| `SkewNormal` | torchregress | - | 0.9073 ± 0.0013 | 2.7004 ± 0.0502 | 1.0079 |
| `SkewNormal/dev_summary` | torchregress | 0.0302 ± 0.0119 † | - | - | - |
| `SkewT` | torchregress | - | 0.9002 ± 0.0060 | 3.3630 ± 0.1799 | 8.9379 |
| `SkewT/dev_summary` | torchregress | 0.0182 ± 0.0060 † | - | - | - |
| `Split_CP` | torchregress | - | 0.9093 ± 0.0075 | 2.4642 ± 0.0338 | 0.9686 |
| `Split_CP/dev_summary` | torchregress | 0.0161 ± 0.0040 † | - | - | - |
| `crepes_CQR` | crepes | - | 0.9087 ± 0.0110 | 2.3068 ± 0.0438 | 1.0485 |
| `crepes_CQR/dev_summary` | crepes | 0.0114 ± 0.0089 † | - | - | - |

##### distributional_bins / diabetes

| Method | Library | CoverageDeviationAbsMean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `AsymmetricLaplace` | torchregress | - | 0.9273 ± 0.0174 | 3.0307 ± 0.1177 | 0.1155 |
| `AsymmetricLaplace/dev_summary` | torchregress | 0.0273 ± 0.0174 † | - | - | - |
| `BetaRegression` | torchregress | - | 0.8546 ± 0.0174 | 17.9157 ± 1.9579 | 0.1117 |
| `BetaRegression/dev_summary` | torchregress | 0.0454 ± 0.0174 † | - | - | - |
| `CTI` | torchregress | - | 0.9599 ± 0.0313 | 3.0762 ± 0.2378 | 0.8333 |
| `CTI/dev_summary` | torchregress | 0.0599 ± 0.0313 † | - | - | - |
| `ContrastiveFlow` | torchregress | - | 0.6416 ± 0.0998 | 4.4066 ± 1.4735 | 1.5111 |
| `ContrastiveFlow/dev_summary` | torchregress | 0.2584 ± 0.0998 † | - | - | - |
| `Evidential` | torchregress | - | 0.8070 ± 0.0544 | 3.0240 ± 0.2557 | 0.1532 |
| `Evidential/dev_summary` | torchregress | 0.0930 ± 0.0544 † | - | - | - |
| `GEV` | torchregress | - | 0.8822 ± 0.0157 | 3.0873 ± 0.0221 | 0.1299 |
| `GEV/dev_summary` | torchregress | 0.0178 ± 0.0157 † | - | - | - |
| `GaussianNLL` | torchregress | - | 0.7519 ± 0.0398 | 3.1465 ± 0.1655 | 0.1380 |
| `GaussianNLL/dev_summary` | torchregress | 0.1481 ± 0.0398 † | - | - | - |
| `JohnsonSU` | torchregress | - | 0.8822 ± 0.0157 | 3.1706 ± 0.3802 | 0.1106 |
| `JohnsonSU/dev_summary` | torchregress | 0.0178 ± 0.0157 † | - | - | - |
| `MC_CP` | torchregress | - | 0.9273 ± 0.0264 | 3.2645 ± 0.3257 | 0.8405 |
| `MC_CP/dev_summary` | torchregress | 0.0273 ± 0.0264 † | - | - | - |
| `MDN_K3` | torchregress | - | 0.7519 ± 0.0075 | 3.2080 ± 0.1919 | 0.1287 |
| `MDN_K3/dev_summary` | torchregress | 0.1481 ± 0.0075 † | - | - | - |
| `NF_Ensemble_K10` | torchregress | - | 0.7644 ± 0.0304 | 2.9461 ± 0.2359 | 9.9096 |
| `NF_Ensemble_K10/dev_summary` | torchregress | 0.1356 ± 0.0304 † | - | - | - |
| `NF_Ensemble_K2` | torchregress | - | 0.7519 ± 0.0651 | 3.1885 ± 0.5189 | 2.4187 |
| `NF_Ensemble_K2/dev_summary` | torchregress | 0.1481 ± 0.0651 † | - | - | - |
| `NF_Ensemble_K20` | torchregress | - | 0.7769 ± 0.0230 | 2.8663 ± 0.1741 | 16.7437 |
| `NF_Ensemble_K20/dev_summary` | torchregress | 0.1231 ± 0.0230 † | - | - | - |
| `NF_Ensemble_K5` | torchregress | - | 0.7544 ± 0.0264 | 3.0625 ± 0.1777 | 5.8687 |
| `NF_Ensemble_K5/dev_summary` | torchregress | 0.1456 ± 0.0264 † | - | - | - |
| `NormalizingFlow` | torchregress | - | 0.6867 ± 0.0674 | 3.6482 ± 0.7803 | 1.2206 |
| `NormalizingFlow/dev_summary` | torchregress | 0.2133 ± 0.0674 † | - | - | - |
| `NormalizingFlow_recipe` | torchregress | - | 0.9173 ± 0.0470 | 2.6113 ± 0.0981 | 0.7031 |
| `NormalizingFlow_recipe/dev_summary` | torchregress | 0.0308 ± 0.0350 † | - | - | - |
| `RandomPartition` | torchregress | - | 0.9424 ± 0.0087 | **2.5690 ± 0.0774** | 0.3481 |
| `RandomPartition/dev_summary` | torchregress | 0.0424 ± 0.0087 † | - | - | - |
| `SLS_CP` | torchregress | - | 0.9248 ± 0.0260 | 2.8804 ± 0.0523 | 0.3205 |
| `SLS_CP/dev_summary` | torchregress | 0.0283 ± 0.0200 † | - | - | - |
| `SQR` | torchregress | - | 0.7193 ± 0.0483 | 4.2636 ± 0.2567 | 0.0955 |
| `SQR/dev_summary` | torchregress | 0.1807 ± 0.0483 † | - | - | - |
| `SinhArcsinh` | torchregress | - | 0.9173 ± 0.0457 | 2.8605 ± 0.0299 | 0.1177 |
| `SinhArcsinh/dev_summary` | torchregress | 0.0409 ± 0.0061 † | - | - | - |
| `SkewNormal` | torchregress | - | 0.9048 ± 0.0242 | 2.8455 ± 0.0487 | 0.1021 |
| `SkewNormal/dev_summary` | torchregress | **0.0168 ± 0.0140** † | - | - | - |
| `SkewT` | torchregress | - | 0.9023 ± 0.0376 | 2.9430 ± 0.0749 | 0.5931 |
| `SkewT/dev_summary` | torchregress | 0.0258 ± 0.0205 † | - | - | - |
| `Split_CP` | torchregress | - | 0.9398 ± 0.0150 | 2.9394 ± 0.0876 | 0.0990 |
| `Split_CP/dev_summary` | torchregress | 0.0398 ± 0.0150 † | - | - | - |
| `crepes_CQR` | crepes | - | 0.9549 ± 0.0199 | 2.8886 ± 0.0777 | 0.1950 |
| `crepes_CQR/dev_summary` | crepes | 0.0549 ± 0.0199 † | - | - | - |

##### distributional_bins / synthetic_hetero

| Method | Library | CoverageDeviationAbsMean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `AsymmetricLaplace` | torchregress | - | 0.9139 ± 0.0086 | 1.8619 ± 0.0364 | 0.5869 |
| `AsymmetricLaplace/dev_summary` | torchregress | 0.0279 ± 0.0085 † | - | - | - |
| `BetaRegression` | torchregress | - | 0.9039 ± 0.0086 | 17.4976 ± 2.7263 | 0.5581 |
| `BetaRegression/dev_summary` | torchregress | 0.0873 ± 0.0132 † | - | - | - |
| `CTI` | torchregress | - | 0.8950 ± 0.0208 | 1.8930 ± 0.0987 | 3.5547 |
| `CTI/dev_summary` | torchregress | 0.0176 ± 0.0116 † | - | - | - |
| `ContrastiveFlow` | torchregress | - | 0.7828 ± 0.0529 | 2.3234 ± 0.1700 | 6.4406 |
| `ContrastiveFlow/dev_summary` | torchregress | 0.1173 ± 0.0528 † | - | - | - |
| `Evidential` | torchregress | - | 0.8617 ± 0.0044 | 1.8045 ± 0.0201 | 0.6414 |
| `Evidential/dev_summary` | torchregress | 0.0386 ± 0.0045 † | - | - | - |
| `GEV` | torchregress | - | 0.9044 ± 0.0098 | 1.9479 ± 0.2305 | 0.6655 |
| `GEV/dev_summary` | torchregress | 0.0371 ± 0.0135 † | - | - | - |
| `GaussianNLL` | torchregress | - | 0.8072 ± 0.0243 | 1.8027 ± 0.0792 | 0.4763 |
| `GaussianNLL/dev_summary` | torchregress | 0.0929 ± 0.0245 † | - | - | - |
| `JohnsonSU` | torchregress | - | 0.9167 ± 0.0159 | 1.9418 ± 0.0345 | 0.5630 |
| `JohnsonSU/dev_summary` | torchregress | 0.0314 ± 0.0141 † | - | - | - |
| `MC_CP` | torchregress | - | 0.8989 ± 0.0255 | 1.9605 ± 0.1166 | 3.6111 |
| `MC_CP/dev_summary` | torchregress | 0.0230 ± 0.0071 † | - | - | - |
| `MDN_K3` | torchregress | - | 0.8217 ± 0.0252 | 1.8447 ± 0.0845 | 0.5246 |
| `MDN_K3/dev_summary` | torchregress | 0.0810 ± 0.0245 † | - | - | - |
| `NF_Ensemble_K10` | torchregress | - | 0.8378 ± 0.0107 | 1.6939 ± 0.0134 | 36.3606 |
| `NF_Ensemble_K10/dev_summary` | torchregress | 0.0624 ± 0.0110 † | - | - | - |
| `NF_Ensemble_K2` | torchregress | - | 0.7606 ± 0.0100 | 1.9301 ± 0.0416 | 7.3124 |
| `NF_Ensemble_K2/dev_summary` | torchregress | 0.1396 ± 0.0099 † | - | - | - |
| `NF_Ensemble_K20` | torchregress | - | 0.8478 ± 0.0164 | **1.6519 ± 0.0379** | 76.0697 |
| `NF_Ensemble_K20/dev_summary` | torchregress | 0.0524 ± 0.0167 † | - | - | - |
| `NF_Ensemble_K5` | torchregress | - | 0.8194 ± 0.0063 | 1.7118 ± 0.0472 | 18.1331 |
| `NF_Ensemble_K5/dev_summary` | torchregress | 0.0816 ± 0.0075 † | - | - | - |
| `NormalizingFlow` | torchregress | - | 0.6822 ± 0.0377 | 2.3459 ± 0.2180 | 4.3738 |
| `NormalizingFlow/dev_summary` | torchregress | 0.2179 ± 0.0377 † | - | - | - |
| `NormalizingFlow_recipe` | torchregress | - | 0.8711 ± 0.0098 | 1.8404 ± 0.0373 | 1.8179 |
| `NormalizingFlow_recipe/dev_summary` | torchregress | 0.0326 ± 0.0131 † | - | - | - |
| `RandomPartition` | torchregress | - | 0.9406 ± 0.0207 | 2.4101 ± 0.1192 | 1.6703 |
| `RandomPartition/dev_summary` | torchregress | 0.0573 ± 0.0084 † | - | - | - |
| `SLS_CP` | torchregress | - | 0.9033 ± 0.0076 | 1.9491 ± 0.1213 | 1.3641 |
| `SLS_CP/dev_summary` | torchregress | **0.0112 ± 0.0047** † | - | - | - |
| `SQR` | torchregress | - | 0.8256 ± 0.0602 | 2.8705 ± 0.2542 | 0.4806 |
| `SQR/dev_summary` | torchregress | 0.0778 ± 0.0544 † | - | - | - |
| `SinhArcsinh` | torchregress | - | 0.9150 ± 0.0073 | 2.2301 ± 0.4093 | 0.6027 |
| `SinhArcsinh/dev_summary` | torchregress | 0.0400 ± 0.0256 † | - | - | - |
| `SkewNormal` | torchregress | - | 0.9156 ± 0.0241 | 1.8969 ± 0.0450 | 0.5514 |
| `SkewNormal/dev_summary` | torchregress | 0.0303 ± 0.0151 † | - | - | - |
| `SkewT` | torchregress | - | 0.9183 ± 0.0176 | 1.8310 ± 0.0329 | 3.0119 |
| `SkewT/dev_summary` | torchregress | 0.0346 ± 0.0094 † | - | - | - |
| `Split_CP` | torchregress | - | 0.9206 ± 0.0092 | 1.7771 ± 0.0348 | 0.3270 |
| `Split_CP/dev_summary` | torchregress | 0.0357 ± 0.0144 † | - | - | - |
| `crepes_CQR` | crepes | - | 0.9078 ± 0.0035 | 1.8571 ± 0.0288 | 0.5518 |
| `crepes_CQR/dev_summary` | crepes | 0.0206 ± 0.0131 † | - | - | - |

#### eiv (tabular)

Group `tabular`; 8 dataset(s). Columns: RMSE ↓, R2 ↑, Attenuation ·, RMSE_clean ↓, Coverage ·, IntervalScore ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### eiv / galaxiesml_eiv

| Method | Library | RMSE | R2 | Attenuation | RMSE_clean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EnsembleEIV` | torchregress | 0.3461 ± 0.0080 | 0.6487 ± 0.0161 | 0.6301 ± 0.0244 | - | - | - | 11.6907 |
| `FunctionalEIV` | torchregress | 0.3547 ± 0.0012 | 0.6312 ± 0.0024 | 0.6513 ± 0.0344 | - | - | - | 15.7499 |
| `FunctionalEIV_MC` | torchregress | 0.7900 ± 0.2535 † | -0.9549 ± 1.2518 | 0.8152 ± 0.3261 | - | - | - | 18.5490 |
| `GaussianNLL+InputNoiseMC` | torchregress | 0.3761 ± 0.0114 | 0.5852 ± 0.0252 | 0.5389 ± 0.0532 | - | - | - | 13.9538 |
| `InputNoiseBinnedPDF` | torchregress | 0.3835 ± 0.0035 | 0.5689 ± 0.0079 | 0.7449 ± 0.0071 | - | - | - | 10.2735 |
| `InputNoiseMC` | torchregress | 0.3454 ± 0.0077 | 0.6502 ± 0.0156 | 0.6288 ± 0.0234 | - | - | - | 12.4491 |
| `InputNoiseMDN` | torchregress | 0.3598 ± 0.0008 | 0.6205 ± 0.0017 | 0.5974 ± 0.0110 | - | - | - | 13.0861 |
| `LatentMarginalization` | torchregress | 0.3482 ± 0.0084 | 0.6444 ± 0.0171 | 0.6186 ± 0.0235 | - | - | - | 13.3824 |
| `LatentNN` | torchregress | 0.3661 ± 0.0054 | 0.6071 ± 0.0117 | 0.6644 ± 0.0233 | - | - | - | 5.6620 |
| `MSE` | torchregress | 0.3488 ± 0.0059 | 0.6432 ± 0.0120 | 0.6347 ± 0.0264 | - | - | - | 5.2308 |
| `MSE_NoisyPredict` | torchregress | 0.3488 ± 0.0060 | 0.6433 ± 0.0121 | 0.6344 ± 0.0260 | - | - | - | 3.4618 |
| `MSE_WarmupMC` | torchregress | 0.6357 ± 0.0920 † | -0.2012 ± 0.3527 | 0.6299 ± 0.1063 | - | - | - | 12.4192 |
| `NoiseAware` | torchregress | **0.3158 ± 0.0083** | **0.7075 ± 0.0155** | 0.6837 ± 0.0433 | - | - | - | 12.1520 |
| `ODR` | torchregress | 0.3764 ± 0.0318 † | 0.5826 ± 0.0718 | 0.7092 ± 0.0830 | - | - | - | 41.5864 |
| `RC` | torchregress | 0.3488 ± 0.0065 | 0.6433 ± 0.0133 | 0.6391 ± 0.0296 | - | - | - | 3.4372 |
| `RC+FunctionalEIV` | torchregress | 0.3496 ± 0.0035 | 0.6417 ± 0.0072 | 0.6452 ± 0.0492 | - | - | - | 13.1847 |
| `RC+GaussianNLL` | torchregress | 0.3754 ± 0.0114 | 0.5867 ± 0.0249 | 0.5346 ± 0.0672 | - | - | - | 4.8222 |
| `SIMEX` | torchregress | 0.3589 ± 0.0031 | 0.6223 ± 0.0066 | 0.6230 ± 0.0291 | - | - | - | 129.8 |
| `SciPyODR` | scipy | 0.4770 ± 0.0000 | 0.3330 ± 0.0000 | 0.3285 ± 0.0000 | - | - | - | 2.9433 |
| `SplitCP+FunctionalEIV` | torchregress+CP | 0.3643 ± 0.0068 | 0.6110 ± 0.0146 | 0.6065 ± 0.0353 | - | 0.8952 ± 0.0041 | 1.8931 ± 0.0456 | 9.8668 |
| `SplitCP+InputNoiseMC` | torchregress+CP | 0.3541 ± 0.0077 | 0.6323 ± 0.0161 | 0.6064 ± 0.0319 | - | 0.8939 ± 0.0038 | **1.8468 ± 0.0536** | 8.0914 |
| `SplitCP+MSE_WarmupMC` | torchregress+CP | 0.5845 ± 0.0561 † | -0.0075 ± 0.1952 | 0.6731 ± 0.0829 | - | 0.8979 ± 0.0046 | 2.9859 ± 0.2891 | 9.3308 |
| `StructuralEIV` | torchregress | 0.3547 ± 0.0012 | 0.6312 ± 0.0024 | 0.6513 ± 0.0344 | - | - | - | 15.8397 |
| `TLS` | numpy | 0.4770 ± 0.0000 | 0.3330 ± 0.0000 | 0.3285 ± 0.0000 | - | - | - | 0.0043 |
| `TLS_Init+FunctionalEIV` | torchregress | 0.4769 ± 0.0000 | 0.3334 ± 0.0000 | 0.3398 ± 0.0003 | - | - | - | 11.8559 |

##### eiv / photoz_eiv

| Method | Library | RMSE | R2 | Attenuation | RMSE_clean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EnsembleEIV` | torchregress | 0.1547 ± 0.0051 | 0.9270 ± 0.0048 | 1.0083 ± 0.0418 | - | - | - | 9.8091 |
| `FunctionalEIV` | torchregress | 0.1749 ± 0.0047 | 0.9067 ± 0.0051 | 0.8669 ± 0.0022 | - | - | - | 12.8803 |
| `FunctionalEIV_MC` | torchregress | 0.1738 ± 0.0077 | 0.9078 ± 0.0082 | 0.8887 ± 0.0190 | - | - | - | 15.2590 |
| `GaussianNLL+InputNoiseMC` | torchregress | 0.1887 ± 0.0048 | 0.8914 ± 0.0056 | 0.8226 ± 0.0193 | - | - | - | 13.5565 |
| `InputNoiseBinnedPDF` | torchregress | 0.1522 ± 0.0015 | 0.9293 ± 0.0014 | 0.9115 ± 0.0057 | - | - | - | 8.5683 |
| `InputNoiseMC` | torchregress | 0.1599 ± 0.0040 | 0.9221 ± 0.0039 | 0.8874 ± 0.0158 | - | - | - | 10.2474 |
| `InputNoiseMDN` | torchregress | 0.1583 ± 0.0063 | 0.9235 ± 0.0061 | 0.8830 ± 0.0124 | - | - | - | 11.3106 |
| `LatentMarginalization` | torchregress | 0.1524 ± 0.0016 | 0.9292 ± 0.0015 | 0.9090 ± 0.0134 | - | - | - | 12.9230 |
| `LatentNN` | torchregress | 0.1983 ± 0.0032 | 0.8801 ± 0.0038 | 0.8457 ± 0.0252 | - | - | - | 5.5980 |
| `MSE` | torchregress | 0.1237 ± 0.0106 † | 0.9531 ± 0.0082 | 0.9422 ± 0.0077 | - | - | - | 3.4462 |
| `MSE_NoisyPredict` | torchregress | 0.1627 ± 0.0050 | 0.9193 ± 0.0049 | 0.8493 ± 0.0115 | - | - | - | 3.4608 |
| `MSE_WarmupMC` | torchregress | 0.1688 ± 0.0042 | 0.9131 ± 0.0044 | 0.9083 ± 0.0119 | - | - | - | 12.1947 |
| `NoiseAware` | torchregress | **0.0776 ± 0.0171** † | **0.9810 ± 0.0086** | 0.9736 ± 0.0131 | - | - | - | 11.4108 |
| `ODR` | torchregress | 0.3942 ± 0.0303 † | 0.5245 ± 0.0724 | 0.5419 ± 0.0339 | - | - | - | 39.1102 |
| `RC` | torchregress | 0.1480 ± 0.0027 | 0.9332 ± 0.0024 | 0.9222 ± 0.0113 | - | - | - | 3.4541 |
| `RC+FunctionalEIV` | torchregress | 0.1575 ± 0.0049 | 0.9243 ± 0.0047 | 0.9206 ± 0.0074 | - | - | - | 13.0609 |
| `RC+GaussianNLL` | torchregress | 0.1613 ± 0.0064 | 0.9206 ± 0.0063 | 0.9276 ± 0.0139 | - | - | - | 5.1020 |
| `SIMEX` | torchregress | 0.2779 ± 0.0057 | 0.7645 ± 0.0096 | 1.1058 ± 0.0429 | - | - | - | 129.3 |
| `SciPyODR` | scipy | 3.9487 ± 0.0000 | -46.5317 ± 0.0000 | 0.8045 ± 0.0000 | - | - | - | 3.3686 |
| `SplitCP+FunctionalEIV` | torchregress+CP | 0.1827 ± 0.0037 | 0.8982 ± 0.0041 | 0.8476 ± 0.0089 | - | 0.9018 ± 0.0031 | 0.8886 ± 0.0149 | 9.8387 |
| `SplitCP+InputNoiseMC` | torchregress+CP | 0.1617 ± 0.0005 | 0.9203 ± 0.0005 | 0.8871 ± 0.0081 | - | 0.9067 ± 0.0021 | **0.7891 ± 0.0021** | 7.9046 |
| `SplitCP+MSE_WarmupMC` | torchregress+CP | 0.1720 ± 0.0008 | 0.9098 ± 0.0008 | 0.9027 ± 0.0157 | - | 0.9097 ± 0.0013 | 0.8362 ± 0.0075 | 9.0459 |
| `StructuralEIV` | torchregress | 0.1749 ± 0.0047 | 0.9067 ± 0.0051 | 0.8669 ± 0.0022 | - | - | - | 13.5952 |

Failed / skipped:

- `TLS` (skipped, seed 0, 1, 2): skipped: TLS ill-conditioned: \|V[-1]\|=0.0113, sigma_min([X\|y])/sigma_min(X)=0.9970 (collinear features / noise-domin…
- `TLS_Init+FunctionalEIV` (skipped, seed 0, 1, 2): skipped: TLS ill-conditioned: \|V[-1]\|=0.0113, sigma_min([X\|y])/sigma_min(X)=0.9970 (collinear features / noise-domin…

##### eiv / synthetic_eiv

| Method | Library | RMSE | R2 | Attenuation | RMSE_clean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EnsembleEIV` | torchregress | 0.2429 ± 0.0053 | 0.9480 ± 0.0023 | 0.9372 ± 0.0067 | 0.1355 ± 0.0064 | - | - | 1.0653 |
| `FunctionalEIV` | torchregress | 0.2439 ± 0.0037 | 0.9476 ± 0.0016 | 0.9357 ± 0.0023 | 0.1382 ± 0.0029 | - | - | 1.3462 |
| `FunctionalEIV_MC` | torchregress | 0.2439 ± 0.0058 | 0.9476 ± 0.0025 | 0.9364 ± 0.0021 | 0.1361 ± 0.0084 | - | - | 1.6677 |
| `GaussianNLL+InputNoiseMC` | torchregress | 0.2461 ± 0.0035 | 0.9466 ± 0.0015 | 0.9016 ± 0.0070 | - | - | - | 1.2884 |
| `InputNoiseBinnedPDF` | torchregress | 0.2731 ± 0.0034 | 0.9343 ± 0.0017 | 0.9634 ± 0.0087 | 0.1721 ± 0.0012 | - | - | 0.9183 |
| `InputNoiseMC` | torchregress | 0.2471 ± 0.0046 | 0.9462 ± 0.0020 | 0.9013 ± 0.0066 | 0.1525 ± 0.0071 | - | - | 1.1435 |
| `InputNoiseMDN` | torchregress | 0.2445 ± 0.0053 | 0.9473 ± 0.0023 | 0.9086 ± 0.0071 | 0.1467 ± 0.0060 | - | - | 1.1780 |
| `LatentMarginalization` | torchregress | 0.2430 ± 0.0058 | 0.9479 ± 0.0025 | 0.9390 ± 0.0103 | 0.1352 ± 0.0066 | - | - | 1.3935 |
| `LatentNN` | torchregress | 0.2501 ± 0.0059 | 0.9449 ± 0.0026 | 0.9769 ± 0.0052 | 0.1362 ± 0.0056 | - | - | 0.5905 |
| `MSE` | torchregress | 0.2439 ± 0.0039 | 0.9476 ± 0.0017 | 0.9385 ± 0.0113 | 0.1361 ± 0.0019 | - | - | 0.3578 |
| `MSE_NoisyPredict` | torchregress | 0.2423 ± 0.0037 | 0.9483 ± 0.0016 | 0.9383 ± 0.0114 | 0.1361 ± 0.0019 | - | - | 0.3600 |
| `MSE_WarmupMC` | torchregress | 0.2440 ± 0.0050 | 0.9475 ± 0.0022 | 0.9424 ± 0.0068 | 0.1351 ± 0.0068 | - | - | 1.2815 |
| `NoiseAware` | torchregress | 0.2515 ± 0.0044 | 0.9443 ± 0.0019 | 0.9196 ± 0.0046 | 0.1490 ± 0.0068 | - | - | 1.2305 |
| `ODR` | torchregress | 0.2476 ± 0.0070 | 0.9460 ± 0.0031 | 0.9570 ± 0.0046 | 0.1374 ± 0.0090 | - | - | 3.7388 |
| `RC` | torchregress | 0.2437 ± 0.0041 | 0.9477 ± 0.0018 | 0.9384 ± 0.0109 | 0.1277 ± 0.0057 | - | - | 0.3598 |
| `RC+FunctionalEIV` | torchregress | 0.2446 ± 0.0047 | 0.9473 ± 0.0020 | 0.9441 ± 0.0024 | 0.1274 ± 0.0080 | - | - | 1.3531 |
| `RC+GaussianNLL` | torchregress | 0.2415 ± 0.0038 | 0.9486 ± 0.0016 | 0.9470 ± 0.0109 | - | - | - | 0.5002 |
| `SIMEX` | torchregress | 0.3705 ± 0.0129 | 0.8790 ± 0.0084 | 0.9687 ± 0.0159 | 0.3102 ± 0.0071 | - | - | 14.9728 |
| `SciPyODR` | scipy | 0.2333 ± 0.0000 | 0.9521 ± 0.0000 | 0.9785 ± 0.0000 | - | - | - | 0.0134 |
| `SplitCP+FunctionalEIV` | torchregress+CP | 0.2484 ± 0.0043 | 0.9456 ± 0.0019 | 0.9380 ± 0.0028 | 0.1428 ± 0.0040 | 0.8861 ± 0.0051 | 1.0317 ± 0.0174 | 1.0397 |
| `SplitCP+InputNoiseMC` | torchregress+CP | 0.2497 ± 0.0046 | 0.9451 ± 0.0021 | 0.9012 ± 0.0046 | 0.1554 ± 0.0083 | 0.8933 ± 0.0192 | 1.0359 ± 0.0126 | 0.8629 |
| `SplitCP+MSE_WarmupMC` | torchregress+CP | 0.2463 ± 0.0029 | 0.9466 ± 0.0013 | 0.9372 ± 0.0050 | 0.1390 ± 0.0028 | 0.9006 ± 0.0102 | **1.0208 ± 0.0080** | 0.9864 |
| `StructuralEIV` | torchregress | 0.2439 ± 0.0037 | 0.9476 ± 0.0016 | 0.9357 ± 0.0023 | 0.1382 ± 0.0029 | - | - | 1.4383 |
| `TLS` | numpy | 0.2333 ± 0.0000 | 0.9521 ± 0.0000 | 0.9785 ± 0.0000 | - | - | - | 0.0013 |
| `TLS_Init+FunctionalEIV` | torchregress | **0.2322 ± 0.0000** | **0.9525 ± 0.0000** | 0.9408 ± 0.0001 | **0.1129 ± 0.0001** | - | - | 1.0488 |

##### eiv / synthetic_eiv_highnoise

| Method | Library | RMSE | R2 | Attenuation | RMSE_clean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EnsembleEIV` | torchregress | 0.7844 ± 0.0053 | 0.4580 ± 0.0073 | 0.4890 ± 0.0074 | 0.5254 ± 0.0176 | - | - | 1.0199 |
| `FunctionalEIV` | torchregress | 0.8177 ± 0.0270 | 0.4106 ± 0.0393 | 0.5905 ± 0.0155 | 0.5472 ± 0.0263 | - | - | 1.6380 |
| `FunctionalEIV_MC` | torchregress | 0.8039 ± 0.0119 | 0.4306 ± 0.0169 | 0.5896 ± 0.0068 | 0.4724 ± 0.0462 | - | - | 1.9080 |
| `GaussianNLL+InputNoiseMC` | torchregress | 0.8103 ± 0.0019 | 0.4217 ± 0.0027 | 0.3158 ± 0.0021 | - | - | - | 1.2635 |
| `InputNoiseBinnedPDF` | torchregress | 0.7957 ± 0.0010 | 0.4423 ± 0.0014 | 0.3596 ± 0.0035 | 0.6534 ± 0.0114 | - | - | 0.9205 |
| `InputNoiseMC` | torchregress | 0.8063 ± 0.0048 | 0.4273 ± 0.0068 | 0.3282 ± 0.0051 | 0.6982 ± 0.0075 | - | - | 1.0934 |
| `InputNoiseMDN` | torchregress | 0.8103 ± 0.0022 | 0.4217 ± 0.0032 | 0.3208 ± 0.0065 | 0.7091 ± 0.0121 | - | - | 1.1832 |
| `LatentMarginalization` | torchregress | 0.7941 ± 0.0061 | 0.4446 ± 0.0086 | 0.5632 ± 0.0056 | 0.4464 ± 0.0065 | - | - | 1.3425 |
| `LatentNN` | torchregress | 0.8011 ± 0.0054 | 0.4347 ± 0.0076 | 0.5318 ± 0.0113 | 0.5142 ± 0.0218 | - | - | 0.5796 |
| `MSE` | torchregress | 0.7853 ± 0.0101 | 0.4567 ± 0.0140 | 0.4883 ± 0.0074 | 0.5331 ± 0.0062 | - | - | 0.4489 |
| `MSE_NoisyPredict` | torchregress | **0.7749 ± 0.0030** | **0.4710 ± 0.0041** | 0.4736 ± 0.0054 | 0.5331 ± 0.0062 | - | - | 0.3574 |
| `MSE_WarmupMC` | torchregress | 0.8014 ± 0.0055 | 0.4343 ± 0.0077 | 0.5869 ± 0.0154 | 0.4681 ± 0.0269 | - | - | 1.2641 |
| `NoiseAware` | torchregress | 0.8595 ± 0.0098 | 0.3492 ± 0.0149 | 0.4711 ± 0.0163 | 0.6201 ± 0.0456 | - | - | 1.1897 |
| `ODR` | torchregress | 0.8027 ± 0.0106 | 0.4324 ± 0.0149 | 0.5321 ± 0.0073 | 0.5072 ± 0.0082 | - | - | 4.0388 |
| `RC` | torchregress | 0.7828 ± 0.0077 | 0.4603 ± 0.0107 | 0.4846 ± 0.0050 | 0.2871 ± 0.0092 | - | - | 0.3684 |
| `RC+FunctionalEIV` | torchregress | 0.9484 ± 0.0793 † | 0.2041 ± 0.1351 | 0.7334 ± 0.0233 | 1.2019 ± 0.2956 | - | - | 1.3751 |
| `RC+GaussianNLL` | torchregress | 0.7841 ± 0.0081 | 0.4584 ± 0.0112 | 0.4865 ± 0.0020 | - | - | - | 0.5070 |
| `SIMEX` | torchregress | 1.0268 ± 0.0212 | 0.0710 ± 0.0382 | 0.7210 ± 0.0424 | 0.6000 ± 0.0094 | - | - | 13.9085 |
| `SciPyODR` | scipy | 1.0436 ± 0.0000 | 0.0407 ± 0.0000 | 0.9488 ± 0.0000 | - | - | - | 0.2196 |
| `SplitCP+FunctionalEIV` | torchregress+CP | 0.8583 ± 0.0724 † | 0.3480 ± 0.1120 | 0.5862 ± 0.0352 | 0.5538 ± 0.0410 | 0.9150 ± 0.0246 | 3.5940 ± 0.2990 | 1.0567 |
| `SplitCP+InputNoiseMC` | torchregress+CP | 0.8092 ± 0.0081 | 0.4232 ± 0.0116 | 0.3252 ± 0.0157 | 0.7051 ± 0.0155 | 0.9000 ± 0.0217 | **3.3165 ± 0.0277** | 0.8498 |
| `SplitCP+MSE_WarmupMC` | torchregress+CP | 0.8092 ± 0.0099 | 0.4231 ± 0.0142 | 0.5810 ± 0.0097 | 0.4661 ± 0.0040 | 0.8883 ± 0.0277 | 3.3253 ± 0.1099 | 0.9762 |
| `StructuralEIV` | torchregress | 0.8177 ± 0.0270 | 0.4106 ± 0.0393 | 0.5905 ± 0.0155 | 0.5472 ± 0.0263 | - | - | 1.7235 |
| `TLS` | numpy | 1.0436 ± 0.0000 | 0.0407 ± 0.0000 | 0.9488 ± 0.0000 | - | - | - | 0.0012 |
| `TLS_Init+FunctionalEIV` | torchregress | 0.8663 ± 0.0001 | 0.3389 ± 0.0001 | 0.7509 ± 0.0001 | **0.2730 ± 0.0002** | - | - | 1.0536 |

##### eiv / synthetic_eiv_nonlinear

| Method | Library | RMSE | R2 | Attenuation | RMSE_clean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EnsembleEIV` | torchregress | 0.8440 ± 0.0066 | 0.0036 ± 0.0157 | 0.0597 ± 0.0037 | 0.8331 ± 0.0033 | - | - | 1.1856 |
| `FunctionalEIV` | torchregress | 1.1907 ± 0.1146 † | -0.9954 ± 0.3773 | 0.1753 ± 0.0655 | 1.0550 ± 0.0589 | - | - | 1.6181 |
| `FunctionalEIV_MC` | torchregress | 1.6194 ± 0.1187 † | -2.6813 ± 0.5336 | 0.2953 ± 0.1040 | 1.4154 ± 0.0886 | - | - | 1.9424 |
| `GaussianNLL+InputNoiseMC` | torchregress | 0.8292 ± 0.0004 | 0.0383 ± 0.0009 | 0.0497 ± 0.0013 | - | - | - | 1.3086 |
| `InputNoiseBinnedPDF` | torchregress | **0.8275 ± 0.0019** | **0.0422 ± 0.0044** | 0.0524 ± 0.0027 | **0.8258 ± 0.0023** | - | - | 1.0216 |
| `InputNoiseMC` | torchregress | 0.8328 ± 0.0027 | 0.0299 ± 0.0064 | 0.0503 ± 0.0013 | 0.8267 ± 0.0013 | - | - | 1.2562 |
| `InputNoiseMDN` | torchregress | 0.8290 ± 0.0038 | 0.0387 ± 0.0088 | 0.0511 ± 0.0036 | 0.8266 ± 0.0024 | - | - | 1.4018 |
| `LatentMarginalization` | torchregress | 0.8836 ± 0.0146 | -0.0922 ± 0.0362 | 0.0796 ± 0.0131 | 0.8619 ± 0.0081 | - | - | 1.3317 |
| `LatentNN` | torchregress | 1.3193 ± 0.0736 † | -1.4394 ± 0.2759 | 0.2190 ± 0.0345 | 1.2288 ± 0.0684 | - | - | 0.5842 |
| `MSE` | torchregress | 0.8481 ± 0.0096 | -0.0060 ± 0.0227 | 0.0575 ± 0.0093 | 0.8357 ± 0.0055 | - | - | 0.4446 |
| `MSE_NoisyPredict` | torchregress | 0.8413 ± 0.0074 | 0.0099 ± 0.0174 | 0.0587 ± 0.0075 | 0.8357 ± 0.0055 | - | - | 0.3648 |
| `MSE_WarmupMC` | torchregress | 1.5401 ± 0.0658 | -2.3218 ± 0.2810 | 0.1174 ± 0.0659 | 1.3980 ± 0.0350 | - | - | 1.2525 |
| `NoiseAware` | torchregress | 0.9081 ± 0.0101 | -0.1536 ± 0.0256 | 0.0593 ± 0.0212 | 0.8886 ± 0.0222 | - | - | 1.1858 |
| `ODR` | torchregress | 0.9062 ± 0.0038 | -0.1486 ± 0.0098 | 0.0633 ± 0.0080 | 0.8768 ± 0.0042 | - | - | 4.5290 |
| `RC` | torchregress | 0.8453 ± 0.0096 | 4.911e-04 ± 2.278e-02 | 0.0595 ± 0.0109 | 0.8465 ± 0.0073 | - | - | 0.3688 |
| `RC+FunctionalEIV` | torchregress | 0.9632 ± 0.0556 † | -0.3004 ± 0.1475 | 0.1119 ± 0.0242 | 1.0591 ± 0.0888 | - | - | 1.3871 |
| `RC+GaussianNLL` | torchregress | 0.8403 ± 0.0056 | 0.0123 ± 0.0132 | 0.0611 ± 0.0033 | - | - | - | 0.5065 |
| `SIMEX` | torchregress | 0.8968 ± 0.0040 | -0.1250 ± 0.0101 | 0.0789 ± 0.0096 | 0.8813 ± 0.0058 | - | - | 13.7227 |
| `SciPyODR` | scipy | 3.2487 ± 0.0000 | -13.7619 ± 0.0000 | 0.9548 ± 0.0000 | - | - | - | 0.2258 |
| `SplitCP+FunctionalEIV` | torchregress+CP | 1.2167 ± 0.1314 † | -1.0867 ± 0.4418 | 0.1730 ± 0.0734 | 1.0853 ± 0.0855 | 0.8911 ± 0.0126 | 5.3506 ± 0.8028 | 1.0509 |
| `SplitCP+InputNoiseMC` | torchregress+CP | 0.8350 ± 0.0024 | 0.0248 ± 0.0057 | 0.0489 ± 0.0049 | 0.8292 ± 0.0026 | 0.8544 ± 0.0212 | **3.0713 ± 0.0119** | 0.8332 |
| `SplitCP+MSE_WarmupMC` | torchregress+CP | 1.5383 ± 0.0398 | -2.3113 ± 0.1702 | 0.1464 ± 0.0727 | 1.3939 ± 0.0344 | 0.8922 ± 0.0255 | 6.7392 ± 0.1827 | 0.9924 |
| `StructuralEIV` | torchregress | 1.1907 ± 0.1146 † | -0.9954 ± 0.3773 | 0.1753 ± 0.0655 | 1.0550 ± 0.0589 | - | - | 1.7267 |
| `TLS` | numpy | 3.3461 ± 0.0000 | -14.6603 ± 0.0000 | 0.9726 ± 0.0000 | - | - | - | 0.0012 |
| `TLS_Init+FunctionalEIV` | torchregress | 3.2126 ± 0.0001 | -13.4354 ± 0.0012 | 0.9353 ± 0.0000 | 2.8858 ± 0.0001 | - | - | 1.0633 |

##### eiv / synthetic_eiv_nonlinear_highnoise

| Method | Library | RMSE | R2 | Attenuation | RMSE_clean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EnsembleEIV` | torchregress | 0.8446 ± 0.0028 | 0.0022 ± 0.0066 | 0.0421 ± 0.0009 | 0.8302 ± 0.0007 | - | - | 1.4403 |
| `FunctionalEIV` | torchregress | 1.0755 ± 0.0716 † | -0.6227 ± 0.2115 | 0.1305 ± 0.0447 | 0.8843 ± 0.0128 | - | - | 1.9499 |
| `FunctionalEIV_MC` | torchregress | 1.2020 ± 0.0532 | -1.0234 ± 0.1774 | 0.1603 ± 0.0596 | 0.9661 ± 0.0185 | - | - | 2.2757 |
| `GaussianNLL+InputNoiseMC` | torchregress | 0.8328 ± 0.0003 | 0.0299 ± 0.0008 | 0.0278 ± 0.0005 | - | - | - | 1.5984 |
| `InputNoiseBinnedPDF` | torchregress | 0.8330 ± 0.0016 | 0.0294 ± 0.0037 | 0.0279 ± 0.0016 | 0.8310 ± 0.0010 | - | - | 1.1378 |
| `InputNoiseMC` | torchregress | 0.8345 ± 0.0003 | 0.0260 ± 0.0008 | 0.0279 ± 0.0002 | **0.8301 ± 0.0007** | - | - | 1.5194 |
| `InputNoiseMDN` | torchregress | **0.8325 ± 0.0006** | **0.0307 ± 0.0015** | 0.0292 ± 0.0006 | 0.8310 ± 0.0009 | - | - | 1.6023 |
| `LatentMarginalization` | torchregress | 0.9777 ± 0.0034 | -0.3370 ± 0.0094 | 0.0605 ± 0.0081 | 0.9003 ± 0.0132 | - | - | 1.6715 |
| `LatentNN` | torchregress | 0.9831 ± 0.0285 | -0.3525 ± 0.0780 | 0.1031 ± 0.0211 | 0.8941 ± 0.0263 | - | - | 0.7303 |
| `MSE` | torchregress | 0.8518 ± 0.0054 | -0.0149 ± 0.0130 | 0.0446 ± 0.0035 | 0.8336 ± 0.0032 | - | - | 0.5336 |
| `MSE_NoisyPredict` | torchregress | 0.8396 ± 0.0034 | 0.0140 ± 0.0081 | 0.0452 ± 0.0032 | 0.8336 ± 0.0032 | - | - | 0.4525 |
| `MSE_WarmupMC` | torchregress | 1.2218 ± 0.0023 | -1.0880 ± 0.0078 | 0.0399 ± 0.0243 | 1.0354 ± 0.0141 | - | - | 1.5986 |
| `NoiseAware` | torchregress | 0.9161 ± 0.0209 | -0.1742 ± 0.0534 | 0.0484 ± 0.0241 | 0.8955 ± 0.0104 | - | - | 1.5082 |
| `ODR` | torchregress | 0.8974 ± 0.0121 | -0.1265 ± 0.0304 | 0.0522 ± 0.0141 | 0.8590 ± 0.0031 | - | - | 5.4863 |
| `RC` | torchregress | 0.8464 ± 0.0061 | -0.0021 ± 0.0145 | 0.0410 ± 0.0074 | 0.8521 ± 0.0093 | - | - | 0.3722 |
| `RC+FunctionalEIV` | torchregress | 1.1378 ± 0.0805 † | -0.8170 ± 0.2510 | 0.1247 ± 0.0202 | 1.9617 ± 0.2588 | - | - | 1.6877 |
| `RC+GaussianNLL` | torchregress | 0.8441 ± 0.0054 | 0.0033 ± 0.0127 | 0.0405 ± 0.0056 | - | - | - | 0.6221 |
| `SIMEX` | torchregress | 0.9322 ± 0.0123 | -0.2156 ± 0.0322 | 0.0779 ± 0.0127 | 0.8881 ± 0.0084 | - | - | 14.3489 |
| `SciPyODR` | scipy | 4.1027 ± 0.0000 | -22.5434 ± 0.0000 | 1.0619 ± 0.0000 | - | - | - | 0.3995 |
| `SplitCP+FunctionalEIV` | torchregress+CP | 1.1872 ± 0.1217 † | -0.9853 ± 0.4015 | 0.1412 ± 0.0547 | 0.9228 ± 0.0227 | 0.8994 ± 0.0274 | 5.1387 ± 0.6854 | 1.2872 |
| `SplitCP+InputNoiseMC` | torchregress+CP | 0.8362 ± 0.0004 | 0.0219 ± 0.0010 | 0.0276 ± 0.0043 | 0.8323 ± 0.0005 | 0.8756 ± 0.0222 | **3.0575 ± 0.0149** | 1.0674 |
| `SplitCP+MSE_WarmupMC` | torchregress+CP | 1.2213 ± 0.0133 | -1.0865 ± 0.0455 | 0.0642 ± 0.0340 | 1.0256 ± 0.0100 | 0.8961 ± 0.0158 | 5.0819 ± 0.1604 | 1.1990 |
| `StructuralEIV` | torchregress | 1.0755 ± 0.0716 † | -0.6227 ± 0.2115 | 0.1305 ± 0.0447 | 0.8843 ± 0.0128 | - | - | 2.0095 |

Failed / skipped:

- `TLS` (skipped, seed 0, 1, 2): skipped: TLS ill-conditioned: \|V[-1]\|=0.0219, sigma_min([X\|y])/sigma_min(X)=0.9921 (collinear features / noise-domin…
- `TLS_Init+FunctionalEIV` (skipped, seed 0, 1, 2): skipped: TLS ill-conditioned: \|V[-1]\|=0.0219, sigma_min([X\|y])/sigma_min(X)=0.9921 (collinear features / noise-domin…

##### eiv / synthetic_eiv_nonlinear_veryhighnoise

| Method | Library | RMSE | R2 | Attenuation | RMSE_clean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EnsembleEIV` | torchregress | 0.8459 ± 0.0020 | -9.176e-04 ± 4.697e-03 | 0.0248 ± 0.0020 | **0.8392 ± 0.0000** | - | - | 1.4219 |
| `FunctionalEIV` | torchregress | 0.9130 ± 0.0261 | -0.1666 ± 0.0671 | 0.0516 ± 0.0080 | 0.8439 ± 0.0019 | - | - | 2.1310 |
| `FunctionalEIV_MC` | torchregress | 1.1411 ± 0.0084 | -0.8215 ± 0.0270 | 0.0041 ± 0.0182 | 0.9122 ± 0.0038 | - | - | 2.5495 |
| `GaussianNLL+InputNoiseMC` | torchregress | 0.8388 ± 0.0010 | 0.0159 ± 0.0024 | 0.0131 ± 0.0010 | - | - | - | 1.7915 |
| `InputNoiseBinnedPDF` | torchregress | **0.8384 ± 0.0006** | **0.0168 ± 0.0015** | 0.0133 ± 0.0007 | 0.8409 ± 0.0002 | - | - | 1.4906 |
| `InputNoiseMC` | torchregress | 0.8393 ± 0.0008 | 0.0146 ± 0.0020 | 0.0131 ± 0.0009 | 0.8402 ± 0.0005 | - | - | 1.5712 |
| `InputNoiseMDN` | torchregress | 0.8389 ± 0.0012 | 0.0157 ± 0.0029 | 0.0134 ± 0.0014 | 0.8401 ± 0.0000 | - | - | 1.5184 |
| `LatentMarginalization` | torchregress | 1.0214 ± 0.0075 | -0.4593 ± 0.0213 | 0.0361 ± 0.0076 | 0.8772 ± 0.0122 | - | - | 1.8497 |
| `LatentNN` | torchregress | 0.9353 ± 0.0179 | -0.2238 ± 0.0469 | 0.0491 ± 0.0228 | 0.8484 ± 0.0113 | - | - | 0.9743 |
| `MSE` | torchregress | 0.8646 ± 0.0035 | -0.0457 ± 0.0085 | 0.0163 ± 0.0031 | 0.8435 ± 0.0020 | - | - | 0.5203 |
| `MSE_NoisyPredict` | torchregress | 0.8474 ± 0.0057 | -0.0043 ± 0.0135 | 0.0219 ± 0.0084 | 0.8435 ± 0.0020 | - | - | 0.5241 |
| `MSE_WarmupMC` | torchregress | 1.1140 ± 0.0067 | -0.7358 ± 0.0209 | 0.0067 ± 0.0139 | 0.9099 ± 0.0091 | - | - | 1.7831 |
| `NoiseAware` | torchregress | 0.9211 ± 0.0114 | -0.1867 ± 0.0293 | 0.0202 ± 0.0163 | 0.9651 ± 0.0064 | - | - | 1.8266 |
| `ODR` | torchregress | 0.9003 ± 0.0075 | -0.1338 ± 0.0188 | 0.0139 ± 0.0085 | 0.8567 ± 0.0056 | - | - | 5.4045 |
| `RC` | torchregress | 0.8481 ± 0.0008 | -0.0061 ± 0.0018 | 0.0209 ± 0.0027 | 0.9429 ± 0.0305 | - | - | 0.5272 |
| `RC+FunctionalEIV` | torchregress | 1.0161 ± 0.0452 | -0.4460 ± 0.1286 | 0.0469 ± 0.0564 | 2.3465 ± 0.2611 | - | - | 1.9246 |
| `RC+GaussianNLL` | torchregress | 0.8442 ± 0.0029 | 0.0032 ± 0.0069 | 0.0240 ± 0.0025 | - | - | - | 0.7512 |
| `SIMEX` | torchregress | 0.9577 ± 0.0030 | -0.2828 ± 0.0079 | 0.0548 ± 0.0049 | 0.9077 ± 0.0149 | - | - | 20.9286 |
| `SciPyODR` | scipy | 6.9028 ± 0.0000 | -65.6470 ± 0.0000 | 1.1812 ± 0.0000 | - | - | - | 0.4724 |
| `SplitCP+FunctionalEIV` | torchregress+CP | 0.9904 ± 0.0555 † | -0.3750 ± 0.1563 | 0.0672 ± 0.0279 | 0.8440 ± 0.0012 | 0.9006 ± 0.0206 | 3.9660 ± 0.2363 | 1.3338 |
| `SplitCP+InputNoiseMC` | torchregress+CP | 0.8409 ± 0.0015 | 0.0110 ± 0.0035 | 0.0134 ± 0.0042 | 0.8417 ± 0.0014 | 0.8789 ± 0.0142 | **3.0956 ± 0.0055** | 1.0263 |
| `SplitCP+MSE_WarmupMC` | torchregress+CP | 1.1240 ± 0.0194 | -0.7673 ± 0.0613 | -0.0087 ± 0.0146 | 0.9059 ± 0.0068 | 0.8756 ± 0.0184 | 4.5495 ± 0.1103 | 1.1842 |
| `StructuralEIV` | torchregress | 0.9130 ± 0.0261 | -0.1666 ± 0.0671 | 0.0516 ± 0.0080 | 0.8439 ± 0.0019 | - | - | 2.1543 |

Failed / skipped:

- `TLS` (skipped, seed 0, 1, 2): skipped: TLS ill-conditioned: \|V[-1]\|=0.00197, sigma_min([X\|y])/sigma_min(X)=0.9999 (collinear features / noise-domi…
- `TLS_Init+FunctionalEIV` (skipped, seed 0, 1, 2): skipped: TLS ill-conditioned: \|V[-1]\|=0.00197, sigma_min([X\|y])/sigma_min(X)=0.9999 (collinear features / noise-domi…

##### eiv / synthetic_eiv_veryhighnoise

| Method | Library | RMSE | R2 | Attenuation | RMSE_clean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EnsembleEIV` | torchregress | 0.9772 ± 0.0032 | 0.1589 ± 0.0055 | 0.1938 ± 0.0025 | 0.8470 ± 0.0156 | - | - | 1.0767 |
| `FunctionalEIV` | torchregress | 1.0443 ± 0.0066 | 0.0394 ± 0.0122 | 0.2611 ± 0.0505 | 0.9508 ± 0.0322 | - | - | 1.6346 |
| `FunctionalEIV_MC` | torchregress | 1.1340 ± 0.0114 | -0.1328 ± 0.0227 | 0.2893 ± 0.0142 | 0.8480 ± 0.0640 | - | - | 1.9580 |
| `GaussianNLL+InputNoiseMC` | torchregress | 0.9824 ± 0.0008 | 0.1499 ± 0.0014 | 0.1064 ± 0.0008 | - | - | - | 1.3338 |
| `InputNoiseBinnedPDF` | torchregress | 0.9766 ± 0.0003 | 0.1599 ± 0.0005 | 0.1192 ± 0.0017 | 0.9361 ± 0.0080 | - | - | 0.9165 |
| `InputNoiseMC` | torchregress | 0.9810 ± 0.0016 | 0.1522 ± 0.0028 | 0.1104 ± 0.0007 | 0.9435 ± 0.0060 | - | - | 1.1585 |
| `InputNoiseMDN` | torchregress | 0.9824 ± 0.0033 | 0.1499 ± 0.0058 | 0.1066 ± 0.0059 | 0.9482 ± 0.0111 | - | - | 1.1730 |
| `LatentMarginalization` | torchregress | 1.0397 ± 0.0015 | 0.0479 ± 0.0027 | 0.3184 ± 0.0021 | 0.6912 ± 0.0131 | - | - | 1.4113 |
| `LatentNN` | torchregress | 1.0421 ± 0.0096 | 0.0434 ± 0.0177 | 0.3048 ± 0.0137 | 0.7551 ± 0.0194 | - | - | 0.5864 |
| `MSE` | torchregress | 0.9866 ± 0.0074 | 0.1426 ± 0.0129 | 0.2013 ± 0.0050 | 0.8267 ± 0.0096 | - | - | 0.4442 |
| `MSE_NoisyPredict` | torchregress | **0.9723 ± 0.0033** | **0.1673 ± 0.0056** | 0.1916 ± 0.0020 | 0.8267 ± 0.0096 | - | - | 0.3617 |
| `MSE_WarmupMC` | torchregress | 1.1486 ± 0.0104 | -0.1621 ± 0.0210 | 0.2761 ± 0.0154 | 0.8445 ± 0.0098 | - | - | 1.3115 |
| `NoiseAware` | torchregress | 1.0360 ± 0.0173 | 0.0544 ± 0.0316 | 0.2224 ± 0.0046 | 0.8948 ± 0.0901 | - | - | 1.2402 |
| `ODR` | torchregress | 1.0073 ± 0.0141 | 0.1061 ± 0.0251 | 0.2196 ± 0.0041 | 0.8136 ± 0.0057 | - | - | 3.8233 |
| `RC` | torchregress | 0.9726 ± 0.0030 | 0.1668 ± 0.0051 | 0.1996 ± 0.0035 | 0.5115 ± 0.0493 | - | - | 0.3628 |
| `RC+FunctionalEIV` | torchregress | 1.1386 ± 0.1243 † | -0.1509 ± 0.2415 | 0.1661 ± 0.0887 | 2.0144 ± 0.6772 | - | - | 1.3668 |
| `RC+GaussianNLL` | torchregress | 0.9729 ± 0.0024 | 0.1662 ± 0.0041 | 0.1983 ± 0.0022 | - | - | - | 0.5071 |
| `SIMEX` | torchregress | 1.1833 ± 0.0142 | -0.2334 ± 0.0295 | 0.3510 ± 0.0179 | 0.7639 ± 0.0448 | - | - | 13.7155 |
| `SciPyODR` | scipy | 2.0700 ± 0.0000 | -2.7742 ± 0.0000 | 0.9286 ± 0.0000 | - | - | - | 0.2858 |
| `SplitCP+FunctionalEIV` | torchregress+CP | 1.0884 ± 0.0163 | -0.0436 ± 0.0314 | 0.2827 ± 0.0811 | 0.9273 ± 0.0202 | 0.9072 ± 0.0129 | 4.4916 ± 0.0526 | 1.0484 |
| `SplitCP+InputNoiseMC` | torchregress+CP | 0.9832 ± 0.0045 | 0.1484 ± 0.0078 | 0.1084 ± 0.0073 | 0.9471 ± 0.0029 | 0.9072 ± 0.0258 | **4.0141 ± 0.0348** | 0.8936 |
| `SplitCP+MSE_WarmupMC` | torchregress+CP | 1.1689 ± 0.0109 | -0.2036 ± 0.0225 | 0.2538 ± 0.0215 | 0.8515 ± 0.0278 | 0.8672 ± 0.0242 | 4.8269 ± 0.0313 | 1.0104 |
| `StructuralEIV` | torchregress | 1.0443 ± 0.0066 | 0.0394 ± 0.0122 | 0.2611 ± 0.0505 | 0.9508 ± 0.0322 | - | - | 1.7251 |
| `TLS` | numpy | 2.0700 ± 0.0000 | -2.7744 ± 0.0000 | 0.9287 ± 0.0000 | - | - | - | 0.0013 |
| `TLS_Init+FunctionalEIV` | torchregress | 1.6124 ± 0.0001 | -1.2902 ± 0.0004 | 0.7012 ± 0.0001 | **0.3589 ± 0.0000** | - | - | 1.0552 |

#### ensembles (tabular)

Group `tabular`; 10 dataset(s). Columns: RMSE ↓, NLL ↓, CRPS ↓, Coverage ·, IntervalScore ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### ensembles / california_housing

| Method | Library | RMSE | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBB` | torchregress | 0.6357 ± 0.0113 | 6.6884 ± 0.3003 | 0.3724 ± 0.0021 | 0.4031 ± 0.0097 | 5.1679 ± 0.0758 | 3.0246 |
| `BatchEnsemble` | torchregress | 0.5777 ± 0.0022 | 0.7064 ± 0.0336 | **0.2752 ± 0.0027** | 0.8773 ± 0.0041 | 2.3651 ± 0.0364 | 3.9312 |
| `BinnedPDFEnsemble` | torchregress | 0.6075 ± 0.0026 | 4.4922 ± 2.3718 | 0.4575 ± 0.0007 | 0.9716 ± 0.0014 | 5.3026 ± 0.0286 | 2.9707 |
| `DeepEnsemble_GaussianNLL` | torchregress | 0.5782 ± 0.0015 | **0.6741 ± 0.0075** | 0.2799 ± 0.0018 | - | - | 9.5939 |
| `DeepEnsemble_MSE` | torchregress | 0.5754 ± 0.0037 | - | - | - | - | 6.8860 |
| `DeepEnsemble_MultiQuantile` | torchregress | 0.5812 ± 0.0021 | - | - | 0.8178 ± 0.0073 | 2.4014 ± 0.0384 | 5.9702 |
| `HeteroBatchEnsemble` | torchregress | 0.6046 ± 0.0198 | 0.7352 ± 0.0540 | 0.4836 ± 0.3260 | 0.9044 ± 0.0119 | 5.1394 ± 4.5365 | 3.3278 |
| `HeteroscedasticBNN` | torchregress | 0.7177 ± 0.0934 † | 1.0741 ± 0.2166 | 0.6468 ± 0.3616 | 0.9504 ± 0.0114 | 6.8855 ± 5.3908 | 3.5094 |
| `HeteroscedasticEnsemble` | torchregress | 0.5856 ± 0.0076 | 0.6846 ± 0.0046 | 0.2860 ± 0.0040 | 0.9262 ± 0.0075 | 2.3924 ± 0.0520 | 5.5703 |
| `HeteroscedasticLaplace` | torchregress | 0.5875 ± 0.0039 | 0.7332 ± 0.0406 | 0.2880 ± 0.0018 | 0.9064 ± 0.0114 | 2.4044 ± 0.0956 | 2.1186 |
| `IVON` | torchregress | 0.6325 ± 0.0105 | 28.5029 ± 1.8494 | 0.4012 ± 0.0080 | 0.1938 ± 0.0180 | 6.6961 ± 0.1663 | 1.6511 |
| `LaplaceMAP` | torchregress | 0.5827 ± 0.0054 | 4.0590 ± 1.4534 | 8.8196 ± 6.8024 | 1.0000 ± 0.0000 | 124.0 ± 96.0 | 1.4769 |
| `MCDropoutWrapper` | torchregress | 0.6096 ± 0.0036 | 4.5794 ± 0.3698 | 0.3406 ± 0.0037 | 0.5056 ± 0.0116 | 4.2532 ± 0.0991 | 1.6518 |
| `MDNEnsemble` | torchregress | 0.5777 ± 0.0114 | - | - | - | - | 7.0349 |
| `MeanOnlyBatchEnsemble` | torchregress | 0.5899 ± 0.0122 | 1317.9 ± 170.2 | 0.3786 ± 0.0052 | 0.0558 ± 0.0132 | 7.2825 ± 0.0957 | 3.2295 |
| `MultiSWAG` | torchregress | 0.5817 ± 0.0056 | 0.6982 ± 0.0074 | 0.2832 ± 0.0040 | 0.9238 ± 0.0021 | **2.3633 ± 0.0208** | 6.0507 |
| `Snapshot` | torchregress | 0.5726 ± 0.0011 | 2.1540 ± 0.0502 | 0.2907 ± 0.0026 | 0.7104 ± 0.0180 | 3.2158 ± 0.0386 | 1.4175 |
| `Stacking` | torchregress | 0.5882 ± 0.0056 | - | - | - | - | 2.8394 |
| `VIDS` | torchregress | 0.7968 ± 0.0465 † | 1.3269 ± 0.0303 | 0.3901 ± 0.0063 | 0.8124 ± 0.0014 | 3.6490 ± 0.1111 | 3.1960 |
| `VariationalMLP` | torchregress | 0.7299 ± 0.1751 † | 0.8546 ± 0.0132 | 840.0 ± 1451.8 | 0.9584 ± 0.0038 | 11824.1 ± 20439.3 | 4.2980 |
| `sklearn_BaggingRegressor` | sklearn | 0.5834 ± 0.0017 | - | - | - | - | 1.0261 |
| `sklearn_StackingRegressor` | sklearn | **0.5695 ± 0.0021** | - | - | - | - | 2.2952 |
| `sklearn_VotingRegressor` | sklearn | 0.6056 ± 0.0012 | - | - | - | - | 0.7595 |

##### ensembles / diabetes

| Method | Library | RMSE | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBB` | torchregress | 52.9085 ± 0.4868 | 9.3717 ± 0.1869 | 34.4948 ± 0.4956 | 0.4637 ± 0.0157 | 447.7 ± 16.3 | 0.2868 |
| `BatchEnsemble` | torchregress | 53.9618 ± 2.6889 | 7.1862 ± 0.7874 | 31.8948 ± 1.3895 | 0.6441 ± 0.0217 | 319.9 ± 8.4 | 0.3688 |
| `BinnedPDFEnsemble` | torchregress | **51.4661 ± 0.5217** | 5.9930 ± 0.0115 | 42.4066 ± 0.3206 | 1.0000 ± 0.0000 | 505.1 ± 6.3 | 0.2946 |
| `DeepEnsemble_GaussianNLL` | torchregress | 51.9123 ± 0.3345 | **5.3313 ± 0.0027** | **28.9687 ± 0.1950** | - | - | 0.9233 |
| `DeepEnsemble_MSE` | torchregress | 51.7214 ± 0.3966 | - | - | - | - | 0.6843 |
| `DeepEnsemble_MultiQuantile` | torchregress | 52.4720 ± 0.3685 | - | - | 0.7895 ± 0.0075 | 231.9 ± 3.1 | 0.5621 |
| `HeteroBatchEnsemble` | torchregress | 52.5844 ± 1.7811 | 5.5899 ± 0.2012 | 29.8494 ± 1.1056 | 0.7719 ± 0.0414 | 243.4 ± 15.2 | 0.3056 |
| `HeteroscedasticBNN` | torchregress | 53.4424 ± 1.0225 | 5.3700 ± 0.0318 | 29.9883 ± 0.2959 | 0.9248 ± 0.0271 | 211.7 ± 14.4 | 0.3359 |
| `HeteroscedasticEnsemble` | torchregress | 51.9663 ± 0.3193 | 5.3800 ± 0.0218 | 29.2103 ± 0.1225 | 0.7995 ± 0.0157 | 218.8 ± 6.8 | 0.5357 |
| `HeteroscedasticLaplace` | torchregress | 55.0564 ± 4.0615 † | 5.6985 ± 0.2412 | 31.5440 ± 2.5848 | 0.7419 ± 0.0603 | 276.0 ± 47.0 | 0.2035 |
| `IVON` | torchregress | 53.4420 ± 1.1790 | 13.4976 ± 0.3995 | 36.4808 ± 0.4955 | 0.3383 ± 0.0075 | 544.3 ± 9.3 | 0.1617 |
| `LaplaceMAP` | torchregress | 53.0512 ± 1.1922 | 9.2363 ± 1.1138 | 1315.2 ± 913.6 | 1.0000 ± 0.0000 | 18508.8 ± 12868.4 | 0.1453 |
| `MCDropoutWrapper` | torchregress | 52.3736 ± 0.8983 | 14.3747 ± 0.3309 | 35.5365 ± 0.5967 | 0.3709 ± 0.0242 | 518.1 ± 10.0 | 0.1607 |
| `MDNEnsemble` | torchregress | 52.2292 ± 0.0679 | - | - | - | - | 0.6601 |
| `MeanOnlyBatchEnsemble` | torchregress | 54.6843 ± 3.9810 † | 266.0 ± 74.2 | 40.3597 ± 3.3605 | 0.1454 ± 0.0313 | 743.6 ± 66.3 | 0.3078 |
| `MultiSWAG` | torchregress | 51.9887 ± 0.3496 | 5.3349 ± 0.0200 | 28.9851 ± 0.3197 | 0.8095 ± 0.0157 | 210.4 ± 5.1 | 0.5810 |
| `Snapshot` | torchregress | 54.9964 ± 1.2200 | 6.5947 ± 0.3492 | 32.7487 ± 0.8857 | 0.6416 ± 0.0242 | 337.8 ± 16.4 | 0.1392 |
| `Stacking` | torchregress | 59.7686 ± 5.4024 † | - | - | - | - | 0.2677 |
| `VIDS` | torchregress | 53.1397 ± 0.8946 | 5.7458 ± 0.0457 | 31.4079 ± 0.5110 | 0.6742 ± 0.0087 | 287.7 ± 10.2 | 2.2457 |
| `VariationalMLP` | torchregress | 52.3448 ± 0.4503 | 5.3387 ± 0.0163 | 29.2426 ± 0.2831 | 0.9173 ± 0.0075 | **203.0 ± 3.3** | 0.4105 |
| `sklearn_BaggingRegressor` | sklearn | 53.8735 ± 0.1063 | - | - | - | - | 0.1279 |
| `sklearn_StackingRegressor` | sklearn | 51.6778 ± 0.1325 | - | - | - | - | 0.2046 |
| `sklearn_VotingRegressor` | sklearn | 51.5928 ± 0.1708 | - | - | - | - | 0.0595 |

##### ensembles / make_regression_synthetic

| Method | Library | RMSE | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBB` | torchregress | 15.8573 ± 0.2114 | 4.2455 ± 0.0115 | 9.1720 ± 0.1376 | 0.9304 ± 0.0119 | 71.9236 ± 0.7950 | 3.2144 |
| `BatchEnsemble` | torchregress | 17.3658 ± 0.3320 | 4.3484 ± 0.0511 | 9.9131 ± 0.2081 | 0.8189 ± 0.0233 | 76.4313 ± 4.1032 | 4.1555 |
| `BinnedPDFEnsemble` | torchregress | 45.8689 ± 1.2220 | 5607.6 ± 3840.4 | 28.5373 ± 0.6581 | 0.8233 ± 0.0037 | 399.6 ± 11.5 | 3.1358 |
| `DeepEnsemble_GaussianNLL` | torchregress | 15.5060 ± 0.0411 | **4.1697 ± 0.0036** | **8.7489 ± 0.0203** | - | - | 9.7600 |
| `DeepEnsemble_MSE` | torchregress | 15.4176 ± 0.0190 | - | - | - | - | 7.0476 |
| `DeepEnsemble_MultiQuantile` | torchregress | 15.4134 ± 0.0702 | - | - | 0.7802 ± 0.0047 | 70.5807 ± 0.8431 | 6.0918 |
| `HeteroBatchEnsemble` | torchregress | 16.5410 ± 0.3087 | 4.3084 ± 0.0264 | 9.4732 ± 0.1841 | 0.8309 ± 0.0062 | 72.4731 ± 1.4477 | 3.4093 |
| `HeteroscedasticBNN` | torchregress | 15.5914 ± 0.0803 | 4.1823 ± 0.0107 | 8.8389 ± 0.0543 | 0.9307 ± 0.0037 | 66.0921 ± 1.2265 | 3.6530 |
| `HeteroscedasticEnsemble` | torchregress | 15.7467 ± 0.0677 | 4.1969 ± 0.0008 | 8.9134 ± 0.0421 | 0.8880 ± 0.0064 | 66.3226 ± 0.2920 | 5.8079 |
| `HeteroscedasticLaplace` | torchregress | 16.6818 ± 0.2092 | 4.2974 ± 0.0303 | 9.5248 ± 0.1375 | 0.8336 ± 0.0132 | 72.5097 ± 2.1984 | 2.2472 |
| `IVON` | torchregress | 17.9025 ± 0.1430 | 5.0103 ± 0.2966 | 10.7386 ± 0.1204 | 0.6329 ± 0.0355 | 104.7 ± 7.3 | 1.7902 |
| `LaplaceMAP` | torchregress | 16.2936 ± 0.1380 | 10.1221 ± 0.6991 | 2705.7 ± 1569.9 | 1.0000 ± 0.0000 | 38090.9 ± 22101.9 | 1.5294 |
| `MCDropoutWrapper` | torchregress | 16.4613 ± 0.5523 | 4.5940 ± 0.0389 | 9.8941 ± 0.1733 | 0.8071 ± 0.0177 | 89.5106 ± 1.3527 | 1.7350 |
| `MDNEnsemble` | torchregress | 15.5741 ± 0.0444 | - | - | - | - | 7.1091 |
| `MeanOnlyBatchEnsemble` | torchregress | 17.8175 ± 0.1801 | 587.4 ± 68.7 | 13.7051 ± 0.2069 | 0.0676 ± 0.0056 | 257.7 ± 4.4 | 3.4592 |
| `MultiSWAG` | torchregress | 15.5913 ± 0.1656 | 4.1790 ± 0.0087 | 8.8129 ± 0.0956 | 0.8931 ± 0.0123 | **65.5587 ± 0.3157** | 6.2702 |
| `Snapshot` | torchregress | 15.7009 ± 0.0774 | 4.3524 ± 0.0122 | 9.1137 ± 0.0737 | 0.8778 ± 0.0084 | 73.9855 ± 0.7929 | 1.4687 |
| `Stacking` | torchregress | 15.7502 ± 0.1962 | - | - | - | - | 2.9316 |
| `VIDS` | torchregress | 15.3590 ± 0.0793 | 4.7794 ± 0.0074 | 12.5106 ± 0.0663 | 1.0000 ± 0.0000 | 147.3 ± 1.3 | 3.3776 |
| `VariationalMLP` | torchregress | 15.4075 ± 0.2299 | 4.2158 ± 0.0086 | 8.8985 ± 0.1117 | 0.9618 ± 0.0077 | 69.9648 ± 0.6203 | 4.5064 |
| `sklearn_BaggingRegressor` | sklearn | 24.7097 ± 0.0500 | - | - | - | - | 1.1613 |
| `sklearn_StackingRegressor` | sklearn | **14.9234 ± 0.0012** | - | - | - | - | 2.5231 |
| `sklearn_VotingRegressor` | sklearn | 23.5049 ± 0.0536 | - | - | - | - | 0.8541 |

##### ensembles / synthetic_hetero

| Method | Library | RMSE | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBB` | torchregress | 0.3930 ± 0.0006 | 1.4583 ± 0.0510 | 0.2312 ± 0.0004 | 0.6344 ± 0.0096 | 2.4147 ± 0.0101 | 1.2544 |
| `BatchEnsemble` | torchregress | 0.3967 ± 0.0053 | 0.7424 ± 0.0467 | 0.2224 ± 0.0041 | 0.7594 ± 0.0107 | 1.9140 ± 0.0673 | 1.6000 |
| `BinnedPDFEnsemble` | torchregress | 0.4871 ± 0.0065 | 1.2100 ± 0.0123 | 0.3609 ± 0.0021 | 0.9572 ± 0.0048 | 3.8939 ± 0.0136 | 1.2906 |
| `DeepEnsemble_GaussianNLL` | torchregress | **0.3667 ± 0.0018** | **0.3486 ± 0.0133** | **0.1984 ± 0.0009** | - | - | 4.7307 |
| `DeepEnsemble_MSE` | torchregress | 0.3688 ± 0.0005 | - | - | - | - | 3.4885 |
| `DeepEnsemble_MultiQuantile` | torchregress | 0.3683 ± 0.0029 | - | - | 0.7839 ± 0.0075 | 1.6452 ± 0.0261 | 2.4077 |
| `HeteroBatchEnsemble` | torchregress | 0.3797 ± 0.0051 | 0.5694 ± 0.0612 | 0.2107 ± 0.0032 | 0.8078 ± 0.0169 | 1.7409 ± 0.0512 | 1.3314 |
| `HeteroscedasticBNN` | torchregress | 0.3916 ± 0.0014 | 0.4895 ± 0.0146 | 0.2187 ± 0.0017 | 0.9411 ± 0.0079 | 1.7144 ± 0.0259 | 1.4448 |
| `HeteroscedasticEnsemble` | torchregress | 0.3690 ± 0.0039 | 0.3688 ± 0.0286 | 0.2002 ± 0.0025 | 0.8700 ± 0.0176 | **1.5306 ± 0.0406** | 2.3083 |
| `HeteroscedasticLaplace` | torchregress | 0.3882 ± 0.0046 | 0.5155 ± 0.0144 | 0.2145 ± 0.0024 | 0.8128 ± 0.0100 | 1.6960 ± 0.0279 | 0.8680 |
| `IVON` | torchregress | 0.3933 ± 0.0058 | 6.6297 ± 0.8012 | 0.2554 ± 0.0052 | 0.3567 ± 0.0296 | 3.6462 ± 0.1388 | 0.6901 |
| `LaplaceMAP` | torchregress | 0.3833 ± 0.0049 | 4.1185 ± 0.8373 | 7.4331 ± 6.6700 | 1.0000 ± 0.0000 | 104.6 ± 93.9 | 0.6178 |
| `MCDropoutWrapper` | torchregress | 0.3771 ± 0.0061 | 2.0809 ± 0.2342 | 0.2243 ± 0.0042 | 0.5828 ± 0.0228 | 2.5041 ± 0.0713 | 0.6943 |
| `MDNEnsemble` | torchregress | 0.3706 ± 0.0028 | - | - | - | - | 2.8494 |
| `MeanOnlyBatchEnsemble` | torchregress | 0.4133 ± 0.0117 | 463.6 ± 16.9 | 0.3089 ± 0.0064 | 0.0761 ± 0.0067 | 5.8136 ± 0.1452 | 1.3042 |
| `MultiSWAG` | torchregress | 0.3755 ± 0.0049 | 0.3888 ± 0.0190 | 0.2044 ± 0.0034 | 0.8789 ± 0.0082 | 1.5433 ± 0.0306 | 2.4787 |
| `Snapshot` | torchregress | 0.3905 ± 0.0081 | 1.6134 ± 0.0245 | 0.2278 ± 0.0039 | 0.6722 ± 0.0086 | 2.2535 ± 0.1128 | 0.5867 |
| `Stacking` | torchregress | 0.3814 ± 0.0107 | - | - | - | - | 1.1688 |
| `VIDS` | torchregress | 0.3876 ± 0.0001 | 0.4799 ± 0.0014 | 0.2149 ± 0.0001 | 0.8644 ± 0.0010 | 1.7066 ± 0.0069 | 2.7421 |
| `VariationalMLP` | torchregress | 0.3830 ± 0.0079 | 0.4839 ± 0.0246 | 0.2150 ± 0.0048 | 0.9678 ± 0.0025 | 1.7481 ± 0.0385 | 1.7510 |
| `sklearn_BaggingRegressor` | sklearn | 0.3886 ± 0.0017 | - | - | - | - | 0.3857 |
| `sklearn_StackingRegressor` | sklearn | 0.3736 ± 0.0004 | - | - | - | - | 0.7291 |
| `sklearn_VotingRegressor` | sklearn | 0.3834 ± 0.0003 | - | - | - | - | 0.2322 |

##### ensembles / synthetic_outliers

| Method | Library | RMSE | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBB` | torchregress | 1.2476 ± 0.0027 | 3.1599 ± 0.0438 | **0.5662 ± 0.0026** | 0.8611 ± 0.0035 | 7.3779 ± 0.0419 | 1.5433 |
| `BatchEnsemble` | torchregress | 1.3042 ± 0.0034 | 2.8561 ± 0.1681 | 0.6467 ± 0.0030 | 0.8272 ± 0.0042 | 7.7499 ± 0.1180 | 2.2542 |
| `BinnedPDFEnsemble` | torchregress | 1.3767 ± 0.0085 | 2.4770 ± 0.0695 | 0.9475 ± 0.0023 | 0.9728 ± 0.0010 | 10.3815 ± 0.0455 | 1.7983 |
| `DeepEnsemble_GaussianNLL` | torchregress | 1.2308 ± 0.0027 | 1.7507 ± 0.0181 | 0.5794 ± 0.0012 | - | - | 6.4623 |
| `DeepEnsemble_MSE` | torchregress | 1.2349 ± 0.0063 | - | - | - | - | 4.8199 |
| `DeepEnsemble_MultiQuantile` | torchregress | **1.2205 ± 0.0017** | - | - | 0.7772 ± 0.0059 | 7.6048 ± 0.0226 | 4.0097 |
| `HeteroBatchEnsemble` | torchregress | 1.2832 ± 0.0202 | 2.2940 ± 0.2175 | 0.6184 ± 0.0194 | 0.8594 ± 0.0212 | 7.4626 ± 0.1532 | 1.5761 |
| `HeteroscedasticBNN` | torchregress | 1.2429 ± 0.0071 | **1.6681 ± 0.0063** | 0.6099 ± 0.0027 | 0.9050 ± 0.0017 | 7.1649 ± 0.0084 | 1.5638 |
| `HeteroscedasticEnsemble` | torchregress | 1.2369 ± 0.0026 | 1.7963 ± 0.0520 | 0.5826 ± 0.0037 | 0.8967 ± 0.0044 | **7.1088 ± 0.0517** | 2.7733 |
| `HeteroscedasticLaplace` | torchregress | 1.2807 ± 0.0263 | 2.1391 ± 0.2366 | 0.6193 ± 0.0184 | 0.8744 ± 0.0048 | 7.3087 ± 0.0101 | 1.2356 |
| `IVON` | torchregress | 1.2676 ± 0.0009 | 9.6392 ± 1.3136 | 0.6097 ± 0.0026 | 0.6322 ± 0.0173 | 8.7039 ± 0.1492 | 0.9799 |
| `LaplaceMAP` | torchregress | 1.2690 ± 0.0084 | 4.7154 ± 1.2635 | 16.8006 ± 17.5220 | 1.0000 ± 0.0000 | 236.2 ± 247.0 | 0.9243 |
| `MCDropoutWrapper` | torchregress | 1.2565 ± 0.0023 | 5.7497 ± 0.2230 | 0.5784 ± 0.0050 | 0.7889 ± 0.0067 | 7.7342 ± 0.0813 | 0.8319 |
| `MDNEnsemble` | torchregress | 1.2230 ± 0.0063 | - | - | - | - | 4.5782 |
| `MeanOnlyBatchEnsemble` | torchregress | 1.3207 ± 0.0355 | 642.2 ± 67.7 | 0.7986 ± 0.0422 | 0.0894 ± 0.0129 | 14.9636 ± 0.8830 | 1.7742 |
| `MultiSWAG` | torchregress | 1.2374 ± 0.0025 | 1.7581 ± 0.0159 | 0.5819 ± 0.0025 | 0.8983 ± 0.0029 | 7.1476 ± 0.0057 | 3.1876 |
| `Snapshot` | torchregress | 1.2771 ± 0.0122 | 3.7390 ± 0.2655 | 0.5960 ± 0.0060 | 0.8350 ± 0.0083 | 7.5043 ± 0.0875 | 0.7124 |
| `Stacking` | torchregress | 1.2522 ± 0.0030 | - | - | - | - | 1.4411 |
| `VIDS` | torchregress | 1.3919 ± 0.0018 | 1.8640 ± 0.0058 | 0.6898 ± 0.0015 | 0.8567 ± 0.0017 | 7.6866 ± 0.0088 | 3.7284 |
| `VariationalMLP` | torchregress | 1.2553 ± 0.0029 | 1.6781 ± 0.0126 | 0.6395 ± 0.0024 | 0.9128 ± 0.0025 | 7.2914 ± 0.0660 | 2.1422 |
| `sklearn_BaggingRegressor` | sklearn | 1.3383 ± 0.0046 | - | - | - | - | 0.4075 |
| `sklearn_StackingRegressor` | sklearn | 1.2553 ± 0.0010 | - | - | - | - | 0.7380 |
| `sklearn_VotingRegressor` | sklearn | 1.3039 ± 0.0012 | - | - | - | - | 0.2379 |

##### ensembles / tabred_cooking

| Method | Library | RMSE | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBB` | torchregress | 0.5074 ± 0.0337 † | 7.6831 ± 0.8222 | 0.3384 ± 0.0210 | 0.3413 ± 0.0113 | 4.8138 ± 0.2142 | 12.2504 |
| `BatchEnsemble` | torchregress | 0.5523 ± 0.0109 | 3.4300 ± 0.5208 | 0.3362 ± 0.0077 | 0.5680 ± 0.0257 | 3.7157 ± 0.1826 | 14.1628 |
| `BinnedPDFEnsemble` | torchregress | 0.5280 ± 0.0065 | 1.1678 ± 0.0165 | 0.3588 ± 0.0011 | 0.9788 ± 0.0031 | 3.7576 ± 0.0331 | 12.5215 |
| `DeepEnsemble_GaussianNLL` | torchregress | 0.5216 ± 0.0032 | 0.9598 ± 0.0041 | 0.2951 ± 0.0008 | - | - | 38.0034 |
| `DeepEnsemble_MSE` | torchregress | 0.5346 ± 0.0038 | - | - | - | - | 37.2890 |
| `DeepEnsemble_MultiQuantile` | torchregress | 0.5204 ± 0.0015 | - | - | 0.5062 ± 0.0068 | 3.7655 ± 0.0971 | 24.5917 |
| `HeteroBatchEnsemble` | torchregress | 0.5530 ± 0.0054 | 4.0607 ± 0.2625 | 0.3354 ± 0.0029 | 0.5771 ± 0.0092 | 3.7095 ± 0.0183 | 11.9766 |
| `HeteroscedasticBNN` | torchregress | 0.4977 ± 0.0049 | 0.7361 ± 0.0467 | **0.2812 ± 0.0044** | 0.9366 ± 0.0344 | **2.1944 ± 0.1750** | 14.0010 |
| `HeteroscedasticEnsemble` | torchregress | 0.5129 ± 0.0032 | 1.1365 ± 0.0567 | 0.2929 ± 0.0022 | 0.7426 ± 0.0065 | 2.5892 ± 0.0452 | 22.0306 |
| `HeteroscedasticLaplace` | torchregress | 0.5485 ± 0.0104 | 3.0462 ± 0.4524 | 0.3301 ± 0.0113 | 0.5956 ± 0.0413 | 3.5895 ± 0.2818 | 8.5655 |
| `IVON` | torchregress | 0.4935 ± 0.0029 | 24.5944 ± 0.4119 | 0.3457 ± 0.0017 | 0.2224 ± 0.0020 | 5.7100 ± 0.0306 | 7.9333 |
| `LaplaceMAP` | torchregress | 0.6114 ± 0.0099 | 2.4074 ± 2.2543 | 4.9908 ± 8.0017 | 0.9919 ± 0.0070 | 69.0162 ± 113.6973 | 6.3860 |
| `MCDropoutWrapper` | torchregress | 0.5099 ± 0.0048 | 8.4658 ± 0.2084 | 0.3355 ± 0.0051 | 0.3553 ± 0.0214 | 4.7029 ± 0.1188 | 7.3226 |
| `MDNEnsemble` | torchregress | 0.5053 ± 0.0032 | - | - | - | - | 28.4382 |
| `MeanOnlyBatchEnsemble` | torchregress | 0.5992 ± 0.0111 | 6994.7 ± 688.7 | 0.4626 ± 0.0099 | 0.0173 ± 0.0009 | 9.1075 ± 0.1955 | 12.0584 |
| `MultiSWAG` | torchregress | 0.5309 ± 0.0033 | 1.1711 ± 0.0630 | 0.3042 ± 0.0025 | 0.7321 ± 0.0090 | 2.7404 ± 0.0585 | 24.0837 |
| `Snapshot` | torchregress | 0.5376 ± 0.0029 | 5.6517 ± 3.1050 | 0.3276 ± 0.0019 | 0.5293 ± 0.0104 | 3.7514 ± 0.0555 | 6.4360 |
| `Stacking` | torchregress | 0.5551 ± 0.0060 | - | - | - | - | 12.8664 |
| `VIDS` | torchregress | 0.4912 ± 0.0052 | 1.2077 ± 0.0364 | 0.2890 ± 0.0032 | 0.6763 ± 0.0085 | 2.7662 ± 0.0478 | 11.4605 |
| `VariationalMLP` | torchregress | 0.5048 ± 0.0184 | **0.7354 ± 0.0347** | 0.2993 ± 0.0221 | 0.9297 ± 0.0142 | 2.3641 ± 0.2573 | 18.7912 |
| `sklearn_BaggingRegressor` | sklearn | 0.4833 ± 0.0006 | - | - | - | - | 84.4665 |
| `sklearn_StackingRegressor` | sklearn | **0.4743 ± 0.0005** | - | - | - | - | 115.9 |
| `sklearn_VotingRegressor` | sklearn | 0.4814 ± 0.0004 | - | - | - | - | 39.1534 |

##### ensembles / tabred_delivery

| Method | Library | RMSE | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBB` | torchregress | 0.5583 ± 0.0014 | 7.5609 ± 1.0496 | 0.3745 ± 0.0039 | 0.3412 ± 0.0233 | 5.3753 ± 0.1818 | 14.3506 |
| `BatchEnsemble` | torchregress | 0.6503 ± 0.0104 | 7.0722 ± 1.3176 | 0.4120 ± 0.0055 | 0.4702 ± 0.0169 | 5.1245 ± 0.1226 | 15.5753 |
| `BinnedPDFEnsemble` | torchregress | 0.6337 ± 0.0068 | 1.3384 ± 0.0400 | 0.4195 ± 0.0033 | 0.9746 ± 0.0008 | 4.2728 ± 0.0279 | 13.1732 |
| `DeepEnsemble_GaussianNLL` | torchregress | 0.6103 ± 0.0031 | 1.2302 ± 0.0041 | 0.3522 ± 0.0019 | - | - | 38.5753 |
| `DeepEnsemble_MSE` | torchregress | 0.6184 ± 0.0001 | - | - | - | - | 32.7845 |
| `DeepEnsemble_MultiQuantile` | torchregress | 0.6137 ± 0.0021 | - | - | 0.3939 ± 0.0092 | 5.2884 ± 0.1351 | 25.7769 |
| `HeteroBatchEnsemble` | torchregress | 0.6311 ± 0.0140 | 5.3704 ± 0.7603 | 0.3904 ± 0.0067 | 0.5304 ± 0.0155 | 4.5477 ± 0.0954 | 13.0102 |
| `HeteroscedasticBNN` | torchregress | 0.5650 ± 0.0103 | 0.9124 ± 0.0912 | 0.3282 ± 0.0197 | 0.9668 ± 0.0161 | 2.6612 ± 0.3898 | 14.0944 |
| `HeteroscedasticEnsemble` | torchregress | 0.5901 ± 0.0020 | 1.3639 ± 0.0724 | 0.3414 ± 0.0013 | 0.7244 ± 0.0190 | 3.0508 ± 0.0825 | 24.9222 |
| `HeteroscedasticLaplace` | torchregress | 0.6281 ± 0.0085 | 4.6793 ± 0.3802 | 0.3856 ± 0.0076 | 0.5424 ± 0.0283 | 4.3772 ± 0.1910 | 9.1655 |
| `IVON` | torchregress | 0.5897 ± 0.0014 | 19.5234 ± 0.9595 | 0.4183 ± 0.0061 | 0.2183 ± 0.0143 | 6.7807 ± 0.1222 | 9.5587 |
| `LaplaceMAP` | torchregress | 0.7306 ± 0.0096 | 1.3001 ± 0.0319 | 0.4483 ± 0.0067 | 0.9924 ± 0.0040 | 4.1167 ± 0.2080 | 6.5017 |
| `MCDropoutWrapper` | torchregress | 0.5948 ± 0.0088 | 6.9546 ± 0.4347 | 0.3879 ± 0.0070 | 0.3794 ± 0.0085 | 5.2640 ± 0.1134 | 8.6660 |
| `MDNEnsemble` | torchregress | 0.5914 ± 0.0054 | - | - | - | - | 26.4397 |
| `MeanOnlyBatchEnsemble` | torchregress | 0.7450 ± 0.0074 | 12960.9 ± 4135.1 | 0.5808 ± 0.0076 | 0.0142 ± 0.0010 | 11.4682 ± 0.1541 | 13.2904 |
| `MultiSWAG` | torchregress | 0.6221 ± 0.0039 | 1.5480 ± 0.0602 | 0.3646 ± 0.0023 | 0.6812 ± 0.0076 | 3.4275 ± 0.0396 | 24.1756 |
| `Snapshot` | torchregress | 0.6442 ± 0.0048 | 3.3727 ± 0.2787 | 0.3935 ± 0.0032 | 0.5552 ± 0.0037 | 4.3430 ± 0.0489 | 6.5182 |
| `Stacking` | torchregress | 0.6517 ± 0.0048 | - | - | - | - | 12.5126 |
| `VIDS` | torchregress | 0.5739 ± 0.0032 | 1.3975 ± 0.0741 | 0.3412 ± 0.0046 | 0.6519 ± 0.0238 | 3.2718 ± 0.1106 | 14.3023 |
| `VariationalMLP` | torchregress | 0.5705 ± 0.0027 | **0.8740 ± 0.0126** | **0.3250 ± 0.0024** | 0.9226 ± 0.0138 | **2.4128 ± 0.0349** | 18.4912 |
| `sklearn_BaggingRegressor` | sklearn | 0.5644 ± 0.0011 | - | - | - | - | 75.7184 |
| `sklearn_StackingRegressor` | sklearn | **0.5545 ± 0.0006** | - | - | - | - | 137.1 |
| `sklearn_VotingRegressor` | sklearn | 0.5650 ± 0.0004 | - | - | - | - | 51.5331 |

##### ensembles / tabred_maps

| Method | Library | RMSE | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBB` | torchregress | 0.1757 ± 0.0069 | 1.1448 ± 0.5137 | 0.1063 ± 0.0077 | 0.5431 ± 0.0633 | 1.1869 ± 0.1427 | 37.8528 |
| `BatchEnsemble` | torchregress | 0.1960 ± 0.0009 | 5.8611 ± 0.6000 | 0.1236 ± 0.0016 | 0.4428 ± 0.0078 | 1.6052 ± 0.0376 | 41.9722 |
| `BinnedPDFEnsemble` | torchregress | 0.2567 ± 0.0052 | 97030.2 ± 80901.9 | 0.1619 ± 0.0028 | 0.8329 ± 0.0061 | 1.8372 ± 0.0456 | 42.4588 |
| `DeepEnsemble_GaussianNLL` | torchregress | 0.1741 ± 0.0008 | -0.0274 ± 0.0302 | **0.0975 ± 0.0005** | - | - | 121.1 |
| `DeepEnsemble_MSE` | torchregress | 0.1768 ± 0.0010 | - | - | - | - | 107.3 |
| `DeepEnsemble_MultiQuantile` | torchregress | 0.1754 ± 0.0009 | - | - | 0.3792 ± 0.0115 | 1.5629 ± 0.0329 | 70.8680 |
| `HeteroBatchEnsemble` | torchregress | 0.1890 ± 0.0011 | 4.5512 ± 0.0942 | 0.1162 ± 0.0011 | 0.5144 ± 0.0188 | 1.3990 ± 0.0330 | 36.3292 |
| `HeteroscedasticBNN` | torchregress | 0.1788 ± 0.0057 | **-0.2728 ± 0.0825** | 0.0997 ± 0.0063 | 0.9521 ± 0.0221 | **0.8497 ± 0.0949** | 41.4861 |
| `HeteroscedasticEnsemble` | torchregress | 0.1727 ± 0.0014 | 0.2029 ± 0.0431 | 0.0978 ± 0.0006 | 0.7309 ± 0.0050 | 0.9000 ± 0.0050 | 71.5235 |
| `HeteroscedasticLaplace` | torchregress | 0.1869 ± 0.0013 | 3.3781 ± 0.2956 | 0.1131 ± 0.0013 | 0.5492 ± 0.0179 | 1.3185 ± 0.0395 | 26.1897 |
| `IVON` | torchregress | 0.1730 ± 0.0004 | 7.7217 ± 0.4953 | 0.1139 ± 0.0008 | 0.3174 ± 0.0113 | 1.6658 ± 0.0258 | 32.2330 |
| `LaplaceMAP` | torchregress | 0.2100 ± 0.0014 | 4.6551 ± 1.5777 | 17.4206 ± 15.4416 | 1.0000 ± 0.0000 | 245.2 ± 217.4 | 21.4062 |
| `MCDropoutWrapper` | torchregress | 0.1782 ± 0.0007 | 6.3950 ± 0.9732 | 0.1109 ± 0.0004 | 0.4400 ± 0.0142 | 1.4381 ± 0.0340 | 22.5629 |
| `MDNEnsemble` | torchregress | 0.1721 ± 0.0018 | - | - | - | - | 76.8888 |
| `MeanOnlyBatchEnsemble` | torchregress | 0.2113 ± 0.0011 | 13884.2 ± 2232.4 | 0.1631 ± 0.0010 | 0.0130 ± 0.0007 | 3.2273 ± 0.0176 | 40.2591 |
| `MultiSWAG` | torchregress | 0.1793 ± 0.0010 | 0.3344 ± 0.0774 | 0.1028 ± 0.0003 | 0.7041 ± 0.0061 | 0.9826 ± 0.0140 | 73.8126 |
| `Snapshot` | torchregress | 0.1871 ± 0.0004 | 0.7361 ± 0.0603 | 0.1073 ± 0.0006 | 0.7124 ± 0.0139 | 1.0123 ± 0.0284 | 21.2707 |
| `Stacking` | torchregress | 0.1831 ± 0.0026 | - | - | - | - | 39.5933 |
| `VIDS` | torchregress | 0.3981 ± 0.0374 † | 0.5437 ± 0.1405 | 0.2213 ± 0.0220 | 0.8297 ± 0.0315 | 1.7633 ± 0.2207 | 91.2148 |
| `VariationalMLP` | torchregress | 0.3576 ± 0.1981 † | 0.0149 ± 0.2919 | 1.3444 ± 1.2335 | 0.9505 ± 0.0249 | 18.0423 ± 16.9633 | 48.7680 |
| `sklearn_BaggingRegressor` | sklearn | 0.1653 ± 0.0001 | - | - | - | - | 751.2 |
| `sklearn_StackingRegressor` | sklearn | **0.1611 ± 0.0002** | - | - | - | - | 2083.9 |
| `sklearn_VotingRegressor` | sklearn | 0.1649 ± 0.0001 | - | - | - | - | 859.6 |

##### ensembles / tabred_weather

| Method | Library | RMSE | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBB` | torchregress | 1.7718 ± 0.0198 | 5.6122 ± 0.6854 | 1.0859 ± 0.0288 | 0.4624 ± 0.0344 | 13.8950 ± 0.7592 | 10.8387 |
| `BatchEnsemble` | torchregress | 1.8944 ± 0.0240 | 3.1646 ± 0.1603 | 1.0829 ± 0.0165 | 0.6776 ± 0.0187 | 10.7934 ± 0.2474 | 13.4885 |
| `BinnedPDFEnsemble` | torchregress | 2.7516 ± 0.1101 | 6701.4 ± 3843.0 | 1.7948 ± 0.0771 | 0.8729 ± 0.0181 | 20.9136 ± 1.0146 | 11.2442 |
| `DeepEnsemble_GaussianNLL` | torchregress | 1.7324 ± 0.0056 | 1.9922 ± 0.0079 | **0.9343 ± 0.0041** | - | - | 34.1077 |
| `DeepEnsemble_MSE` | torchregress | 1.7727 ± 0.0126 | - | - | - | - | 30.8511 |
| `DeepEnsemble_MultiQuantile` | torchregress | 1.7279 ± 0.0077 | - | - | 0.6236 ± 0.0050 | 10.1990 ± 0.0824 | 22.0599 |
| `HeteroBatchEnsemble` | torchregress | 1.8744 ± 0.0121 | 2.9587 ± 0.0564 | 1.0574 ± 0.0065 | 0.6888 ± 0.0211 | 10.3146 ± 0.2280 | 11.2744 |
| `HeteroscedasticBNN` | torchregress | 1.7699 ± 0.0195 | **1.9718 ± 0.0274** | 0.9749 ± 0.0192 | 0.9459 ± 0.0163 | 7.9486 ± 0.6107 | 12.1796 |
| `HeteroscedasticEnsemble` | torchregress | 1.7279 ± 0.0192 | 2.0391 ± 0.0232 | 0.9346 ± 0.0129 | 0.8257 ± 0.0075 | **7.7051 ± 0.0555** | 20.1538 |
| `HeteroscedasticLaplace` | torchregress | 1.8495 ± 0.0105 | 3.0364 ± 0.1610 | 1.0474 ± 0.0059 | 0.6962 ± 0.0204 | 10.1130 ± 0.1281 | 7.8751 |
| `IVON` | torchregress | 1.7209 ± 0.0244 | 20.9581 ± 1.4649 | 1.1462 ± 0.0177 | 0.2427 ± 0.0091 | 18.2826 ± 0.3431 | 6.9623 |
| `LaplaceMAP` | torchregress | 2.0547 ± 0.0545 | 2.9917 ± 0.0407 | 2.0249 ± 0.0607 | 0.9997 ± 0.0000 | 25.5532 ± 1.1474 | 5.5884 |
| `MCDropoutWrapper` | torchregress | 1.7334 ± 0.0043 | 6.2404 ± 0.1821 | 1.0595 ± 0.0006 | 0.4529 ± 0.0040 | 13.4367 ± 0.0711 | 6.5273 |
| `MDNEnsemble` | torchregress | 1.7140 ± 0.0061 | - | - | - | - | 23.4510 |
| `MeanOnlyBatchEnsemble` | torchregress | 2.0274 ± 0.0186 | 5310.0 ± 741.9 | 1.5158 ± 0.0135 | 0.0194 ± 0.0046 | 29.7687 ± 0.2492 | 11.6729 |
| `MultiSWAG` | torchregress | 1.7560 ± 0.0084 | 2.0546 ± 0.0136 | 0.9522 ± 0.0093 | 0.8199 ± 0.0057 | 7.9287 ± 0.0929 | 21.4008 |
| `Snapshot` | torchregress | 1.8324 ± 0.0138 | 2.8801 ± 0.0532 | 1.0299 ± 0.0040 | 0.7209 ± 0.0110 | 9.7220 ± 0.1584 | 5.5780 |
| `Stacking` | torchregress | 1.8358 ± 0.0116 | - | - | - | - | 11.3574 |
| `VIDS` | torchregress | 1.8145 ± 0.0107 | 2.0410 ± 0.0081 | 0.9803 ± 0.0093 | 0.8698 ± 0.0073 | 8.0679 ± 0.0481 | 8.4396 |
| `VariationalMLP` | torchregress | 1.7735 ± 0.0123 | 1.9884 ± 0.0059 | 0.9913 ± 0.0165 | 0.9497 ± 0.0038 | 8.1687 ± 0.1920 | 15.4186 |
| `sklearn_BaggingRegressor` | sklearn | 1.7431 ± 0.0022 | - | - | - | - | 32.0068 |
| `sklearn_StackingRegressor` | sklearn | **1.6762 ± 0.0026** | - | - | - | - | 56.7440 |
| `sklearn_VotingRegressor` | sklearn | 1.7054 ± 0.0017 | - | - | - | - | 19.5939 |

##### ensembles / zurich_delays

| Method | Library | RMSE | NLL | CRPS | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBB` | torchregress | 113.9 ± 0.4 | 47.7607 ± 5.0594 | 55.4680 ± 0.2510 | 0.2935 ± 0.0138 | 899.4 ± 11.7 | 73.6715 |
| `BatchEnsemble` | torchregress | 112.2 ± 0.5 | 5.9628 ± 0.0931 | **43.4811 ± 0.1440** | 0.8920 ± 0.0118 | 401.8 ± 2.9 | 94.5998 |
| `BinnedPDFEnsemble` | torchregress | 1561.7 ± 71.0 | 817.3 ± 695.5 | 1019.2 ± 56.7 | 0.3660 ± 0.0190 | 14558.3 ± 949.4 | 79.6154 |
| `DeepEnsemble_GaussianNLL` | torchregress | 106.5 ± 0.3 | **5.8271 ± 0.0272** | 44.2826 ± 1.6598 | - | - | 237.1 |
| `DeepEnsemble_MSE` | torchregress | 108.6 ± 1.0 | - | - | - | - | 183.2 |
| `DeepEnsemble_MultiQuantile` | torchregress | **106.4 ± 0.2** | - | - | 0.7799 ± 0.0038 | **386.3 ± 1.8** | 161.8 |
| `HeteroBatchEnsemble` | torchregress | 108.8 ± 0.5 | 6.0464 ± 0.0535 | 46.0929 ± 4.2423 | 0.8982 ± 0.0055 | 436.2 ± 62.2 | 78.7267 |
| `HeteroscedasticBNN` | torchregress | 117.6 ± 1.4 | 5.9334 ± 0.0446 | 47.6201 ± 1.4000 | 0.9350 ± 0.0057 | 465.7 ± 15.1 | 85.0508 |
| `HeteroscedasticEnsemble` | torchregress | 109.0 ± 1.0 | 5.8823 ± 0.0194 | 1149.0 ± 1871.0 | 0.9117 ± 0.0007 | 15961.8 ± 26342.6 | 138.6 |
| `HeteroscedasticLaplace` | torchregress | 110.9 ± 0.2 (n=2) (1 non-finite) | 5.9744 ± 0.0158 (n=2) (1 non-finite) | 44.0068 ± 0.0560 (n=2) (1 non-finite) | 0.6048 ± 0.5238 | 404.9 ± 0.9 (n=2) (1 non-finite) | 54.8990 |
| `IVON` | torchregress | 141.8 ± 54.0 † | 300.5 ± 31.5 | 61.4666 ± 7.1827 | 0.0936 ± 0.0096 | 1144.3 ± 136.3 | 47.9917 |
| `LaplaceMAP` | torchregress | 112.3 ± 2.3 | 10.7010 ± 0.1223 | 4229.5 ± 526.4 | 1.0000 ± 0.0000 | 59539.5 ± 7411.4 | 40.0493 |
| `MCDropoutWrapper` | torchregress | 109.0 ± 0.4 | 4.916e+07 ± 3.785e+07 | 52.1034 ± 0.6218 | 0.2557 ± 0.0077 | 813.4 ± 11.0 | 45.5525 |
| `MDNEnsemble` | torchregress | 107.0 ± 0.3 | - | - | - | - | 172.2 |
| `MeanOnlyBatchEnsemble` | torchregress | 113.6 ± 4.9 | 23402.9 ± 7383.9 | 60.9101 ± 0.7767 | 0.0193 ± 0.0014 | 1199.3 ± 15.3 | 81.1313 |
| `MultiSWAG` | torchregress | 106.9 ± 0.6 | 5.8603 ± 0.0314 | 1583.3 ± 1333.8 | 0.9237 ± 0.0037 | 22068.6 ± 18776.5 | 154.3 |
| `Snapshot` | torchregress | 108.5 ± 0.2 | 16.8118 ± 8.7006 | 48.0127 ± 2.1540 | 0.5400 ± 0.1734 | 601.9 ± 101.1 | 40.4049 |
| `Stacking` | torchregress | 111.5 ± 2.5 | - | - | - | - | 79.0765 |
| `VIDS` | torchregress | 134.7 ± 24.4 † | 6.8937 ± 0.0061 | 48.6484 ± 0.5841 | 0.8594 ± 0.0047 | 491.5 ± 2.5 | 28.1535 |
| `VariationalMLP` | torchregress | 127.5 ± 8.5 † | 5.9946 ± 0.0264 | 8.775e+72 ± 1.520e+73 | 0.9534 ± 0.0112 | 1.235e+74 ± 2.140e+74 | 109.7 |
| `sklearn_BaggingRegressor` | sklearn | 134.9 ± 4.2 | - | - | - | - | 39.8655 |
| `sklearn_StackingRegressor` | sklearn | 111.1 ± 2.4 | - | - | - | - | 736.0 |
| `sklearn_VotingRegressor` | sklearn | 109.8 ± 0.6 | - | - | - | - | 259.1 |

#### label_shift (tabular)

Group `tabular`; 3 dataset(s). Columns: RMSE ↓, WeightRMSE ↓, TargetMAE ↓, ShiftCoverage ·, ShiftWidth ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### label_shift / california_housing

| Method | Library | RMSE | WeightRMSE | TargetMAE | ShiftCoverage | ShiftWidth | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBSE` | torchregress | **0.5843 ± 0.0075** | 0.1725 ± 0.0063 | 0.5493 ± 0.0121 | - | - | 1.3402 |
| `DomainClassifier` | sklearn | **0.5843 ± 0.0075** | 0.8611 ± 0.0722 | 0.5539 ± 0.0327 | - | - | 1.1389 |
| `GaussShiftCorrect` | torchregress | 0.9802 ± 0.0435 | - | 0.7972 ± 0.0201 | 0.4069 ± 0.0110 | - | 1.0565 |
| `LibDomainClassifier` | torchregress | **0.5843 ± 0.0075** | 0.4443 ± 0.0104 | 0.5006 ± 0.0093 | - | - | 1.1400 |
| `Oracle` | torchregress | **0.5843 ± 0.0075** | **0.0000 ± 0.0000** | 0.5788 ± 0.0121 | - | - | 1.1960 |
| `ShiftFactoredTransport` | torchregress | **0.5843 ± 0.0075** | - | **0.4940 ± 0.0103** | 0.8907 ± 0.0057 | 2.1165 ± 0.0415 | 0.8848 |
| `ShiftFactoredTransport+Conformal` | torchregress | **0.5843 ± 0.0075** | - | **0.4940 ± 0.0103** | 0.8907 ± 0.0057 | 2.1165 ± 0.0415 | 0.8904 |
| `SplitCP` | torchregress | **0.5843 ± 0.0075** | - | - | 0.8711 ± 0.0071 | 1.5535 ± 0.0032 | 0.8905 |
| `TT_BBSE` | torchregress | **0.5843 ± 0.0075** | 0.2062 ± 0.0649 | 0.5747 ± 0.0172 | - | - | 0.8917 |
| `TT_EM` | torchregress | **0.5843 ± 0.0075** | 0.2553 ± 0.0651 | 0.5679 ± 0.0106 | - | - | 0.8875 |
| `Unweighted` | torchregress | **0.5843 ± 0.0075** | 0.4202 ± 0.0060 | 0.4940 ± 0.0103 | - | - | 1.3566 |
| `WeightedCP` | torchregress | **0.5843 ± 0.0075** | - | - | 0.8704 ± 0.0071 | **1.5453 ± 0.0003** | 0.8891 |

##### label_shift / diabetes

| Method | Library | RMSE | WeightRMSE | TargetMAE | ShiftCoverage | ShiftWidth | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBSE` | torchregress | **53.1870 ± 1.2101** | 0.6788 ± 0.0410 | 40.8783 ± 3.7147 | - | - | 0.1304 |
| `DomainClassifier` | sklearn | **53.1870 ± 1.2101** | 0.5654 ± 0.0191 | 45.0889 ± 2.9452 | - | - | 0.1266 |
| `GaussShiftCorrect` | torchregress | 55.7949 ± 2.3020 | - | 47.5132 ± 2.3729 | 0.5815 ± 0.0753 | - | 0.1272 |
| `LibDomainClassifier` | torchregress | **53.1870 ± 1.2101** | 0.4737 ± 0.0285 | 45.8590 ± 3.1176 | - | - | 0.1280 |
| `Oracle` | torchregress | **53.1870 ± 1.2101** | **0.0000 ± 0.0000** | 48.7844 ± 2.0356 | - | - | 0.1267 |
| `ShiftFactoredTransport` | torchregress | **53.1870 ± 1.2101** | - | 45.5403 ± 2.9426 | 0.8847 ± 0.0313 | 183.4 ± 4.5 | 0.1288 |
| `ShiftFactoredTransport+Conformal` | torchregress | **53.1870 ± 1.2101** | - | 45.5403 ± 2.9426 | 0.9098 ± 0.0075 | 191.1 ± 3.2 | 0.1271 |
| `SplitCP` | torchregress | **53.1870 ± 1.2101** | - | - | 0.8070 ± 0.0657 | **151.3 ± 5.6** | 0.1303 |
| `TT_BBSE` | torchregress | **53.1870 ± 1.2101** | 1.3640 ± 0.2502 | 38.1207 ± 3.8550 | - | - | 0.1276 |
| `TT_EM` | torchregress | **53.1870 ± 1.2101** | 0.9990 ± 0.0074 | **36.9141 ± 2.9630** | - | - | 0.1273 |
| `Unweighted` | torchregress | **53.1870 ± 1.2101** | 0.3926 ± 0.0242 | 45.5496 ± 2.9498 | - | - | 0.1383 |
| `WeightedCP` | torchregress | **53.1870 ± 1.2101** | - | - | 0.8346 ± 0.0451 | 158.7 ± 4.1 | 0.1276 |

##### label_shift / folktables_income

| Method | Library | RMSE | WeightRMSE | TargetMAE | ShiftCoverage | ShiftWidth | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BBSE` | torchregress | **80208.4 ± 3081.8** | 0.2030 ± 0.0683 | 67863.1 ± 5688.9 | - | - | 4.4637 |
| `DomainClassifier` | sklearn | **80208.4 ± 3081.8** | 5.9586 ± 2.4917 | **52009.0 ± 526.2** | - | - | 4.7758 |
| `GaussShiftCorrect` | torchregress | 1.554e+05 ± 6.670e+03 | - | 1.097e+05 ± 5.411e+03 | 0.2146 ± 0.0219 | - | 4.7310 |
| `LibDomainClassifier` | torchregress | **80208.4 ± 3081.8** | 0.4135 ± 0.0012 | 55884.6 ± 1889.3 | - | - | 4.7579 |
| `Oracle` | torchregress | **80208.4 ± 3081.8** | **0.0000 ± 0.0000** | 76152.4 ± 1954.6 | - | - | 4.8377 |
| `ShiftFactoredTransport` | torchregress | **80208.4 ± 3081.8** | - | 55870.5 ± 1925.9 | 0.8931 ± 0.0054 | 2.590e+05 ± 1.928e+03 | 4.7550 |
| `ShiftFactoredTransport+Conformal` | torchregress | **80208.4 ± 3081.8** | - | 55870.5 ± 1925.9 | 0.8931 ± 0.0054 | 2.590e+05 ± 1.928e+03 | 4.6683 |
| `SplitCP` | torchregress | **80208.4 ± 3081.8** | - | - | 0.8276 ± 0.0248 | **1.200e+05 ± 3.604e+03** | 4.7731 |
| `TT_EM` | torchregress | **80208.4 ± 3081.8** | 0.6289 ± 0.0824 | 89945.6 ± 6809.1 | - | - | 4.6039 |
| `Unweighted` | torchregress | **80208.4 ± 3081.8** | 0.4135 ± 0.0012 | 55884.6 ± 1889.3 | - | - | 4.4012 |
| `WeightedCP` | torchregress | **80208.4 ± 3081.8** | - | - | 0.8280 ± 0.0246 | 1.203e+05 ± 3.748e+03 | 4.6822 |

Failed / skipped:

- `TT_BBSE` (failed, seed 1): failed: ValueError: confusion matrix ill-conditioned (cond=1.24e+17)
- `TT_BBSE` (failed, seed 2): failed: ValueError: confusion matrix ill-conditioned (cond=2.39e+16)
- `TT_BBSE` (failed, seed 0): failed: ValueError: confusion matrix ill-conditioned (cond=5.13e+16)

#### likelihoods (tabular)

Group `tabular`; 7 dataset(s). Columns: NLL ↓, NLLGapToTruth |x|↓, LocRMSE ↓, ScaleRMSE ↓, RMSE ↓, CRPS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: NLL.

##### likelihoods / insurance_claims

| Method | Library | NLL | NLLGapToTruth | LocRMSE | ScaleRMSE | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AsymmetricLaplace` | torchregress | -1.6358 ± 0.1419 † | - | - | - | 0.2386 ± 0.0003 | 0.0674 ± 0.0009 | 1.0550 |
| `CatBoost_Uncertainty` | catboost | 1.0145 ± 0.0356 | - | - | - | **0.2357 ± 0.0001** | 0.0847 ± 0.0003 | 0.7602 |
| `GaussianNLL` | torchregress | 362.0 ± 62.6 † | - | - | - | 0.2421 ± 0.0012 | 0.0869 ± 0.0032 | 1.1158 |
| `JohnsonSU` | torchregress | -3.3162 ± 0.4740 † | - | - | - | 0.2434 ± 0.0002 | 0.0543 ± 0.0006 | 1.0308 |
| `LightGBM_Point` | lightgbm | non-finite (3) | - | - | - | 0.2396 ± 0.0000 | - | 0.1899 |
| `MDN_K3` | torchregress | -1.5235 ± 2.6521 † | - | - | - | 0.2396 ± 0.0030 | 0.0843 ± 0.0279 | 1.1794 |
| `NSF` | torchregress | 9.0607 ± 1.7820 † | - | - | - | 0.3614 ± 0.0004 | 0.2723 ± 0.0008 | 6.0700 |
| `SinhArcsinh` | torchregress | -3.0847 ± 0.5208 † | - | - | - | 0.2430 ± 0.0007 | 0.0546 ± 0.0006 | 1.0557 |
| `SkewNormal` | torchregress | 1779.1 ± 1534.9 † | - | - | - | 0.2675 ± 0.0136 | 0.1017 ± 0.0047 | 1.0626 |
| `SkewT` | torchregress | **-4.3447 ± 0.5298** † | - | - | - | 0.2431 ± 0.0002 | **0.0540 ± 0.0004** | 5.1260 |

##### likelihoods / lik_beta

| Method | Library | NLL | NLLGapToTruth | LocRMSE | ScaleRMSE | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrueModel` | scipy | **-0.3367 ± 0.0302** † | **0.0000 ± 0.0000** | - | - | - | - | 0.0011 |
| `scipy_MLE_Linear` | scipy | -0.3354 ± 0.0299 † | 0.0013 ± 0.0014 | 0.0070 ± 0.0031 | 0.1291 ± 0.0756 | - | - | 0.0557 |
| `scipy_MLE_Unconditional` | scipy | -0.0536 ± 0.0107 † | 0.2831 ± 0.0201 | - | - | - | - | 6.198e-04 |
| `tr_GaussianNLL_MLP` | torchregress | -0.0296 ± 0.0209 † | 0.3072 ± 0.0475 | - | - | - | - | 0.8787 |
| `tr_Linear` | torchregress | -0.3354 ± 0.0299 † | 0.0013 ± 0.0014 | **0.0070 ± 0.0031** | **0.1290 ± 0.0754** | - | - | 0.0196 |
| `tr_MLP` | torchregress | -0.3272 ± 0.0282 † | 0.0095 ± 0.0032 | 0.0272 ± 0.0010 | 0.2757 ± 0.0565 | - | - | 1.0048 |

##### likelihoods / lik_gev

| Method | Library | NLL | NLLGapToTruth | LocRMSE | ScaleRMSE | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrueModel` | scipy | **1.0341 ± 0.0320** | **0.0000 ± 0.0000** | - | - | - | - | 0.0011 |
| `scipy_MLE_Linear` | scipy | 1.0364 ± 0.0327 | 0.0023 ± 0.0012 | **0.0264 ± 0.0073** | **0.0203 ± 0.0035** | - | - | 0.4339 |
| `scipy_MLE_Unconditional` | scipy | 1.7180 ± 0.0094 | 0.6839 ± 0.0306 | - | - | - | - | 0.0386 |
| `tr_GaussianNLL_MLP` | torchregress | 1.2393 ± 0.0619 | 0.2052 ± 0.0310 | - | - | - | - | 1.0759 |
| `tr_Linear` | torchregress | 1.0364 ± 0.0327 | 0.0023 ± 0.0012 | 0.0264 ± 0.0073 | 0.0203 ± 0.0035 | - | - | 0.0843 |
| `tr_MLP` | torchregress | 1.0502 ± 0.0294 | 0.0161 ± 0.0067 | 0.0819 ± 0.0100 | 0.0299 ± 0.0031 | - | - | 1.4633 |

##### likelihoods / lik_johnsonsu

| Method | Library | NLL | NLLGapToTruth | LocRMSE | ScaleRMSE | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrueModel` | scipy | **0.7591 ± 0.0294** | **0.0000 ± 0.0000** | - | - | - | - | 0.0012 |
| `scipy_MLE_Linear` | scipy | 0.7621 ± 0.0284 | 0.0031 ± 0.0016 | 0.0843 ± 0.0270 | **0.0620 ± 0.0268** | - | - | 0.3461 |
| `scipy_MLE_Unconditional` | scipy | 1.6239 ± 0.0073 | 0.8648 ± 0.0265 | - | - | - | - | 0.0844 |
| `tr_GaussianNLL_MLP` | torchregress | 0.9171 ± 0.0604 † | 0.1580 ± 0.0314 | - | - | - | - | 1.4664 |
| `tr_Linear` | torchregress | 0.7621 ± 0.0284 | 0.0031 ± 0.0016 | **0.0843 ± 0.0268** | 0.0620 ± 0.0267 | - | - | 0.0780 |
| `tr_MLP` | torchregress | 0.7792 ± 0.0252 | 0.0201 ± 0.0074 | 0.1578 ± 0.0873 | 0.0849 ± 0.0214 | - | - | 1.6801 |

##### likelihoods / lik_sinharcsinh

| Method | Library | NLL | NLLGapToTruth | LocRMSE | ScaleRMSE | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrueModel` | scipy | **0.5726 ± 0.0143** | **0.0000 ± 0.0000** | - | - | - | - | 9.600e-04 |
| `scipy_MLE_Linear` | scipy | 0.5759 ± 0.0128 | 0.0033 ± 0.0017 | **0.0437 ± 0.0232** | **0.0659 ± 0.0203** | - | - | 0.4350 |
| `scipy_MLE_Unconditional` | scipy | 1.5706 ± 0.0128 | 0.9980 ± 0.0192 | - | - | - | - | 0.0225 |
| `tr_GaussianNLL_MLP` | torchregress | 0.6744 ± 0.0113 | 0.1018 ± 0.0072 | - | - | - | - | 1.0300 |
| `tr_Linear` | torchregress | 0.5759 ± 0.0128 | 0.0032 ± 0.0017 | 0.0438 ± 0.0231 | 0.0661 ± 0.0201 | - | - | 0.0925 |
| `tr_MLP` | torchregress | 0.6306 ± 0.0475 † | 0.0580 ± 0.0342 | 0.1485 ± 0.0809 | 0.0999 ± 0.0200 | - | - | 1.2760 |

##### likelihoods / lik_skewnormal

| Method | Library | NLL | NLLGapToTruth | LocRMSE | ScaleRMSE | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrueModel` | scipy | **0.2908 ± 0.0167** † | **0.0000 ± 0.0000** | - | - | - | - | 9.230e-04 |
| `scipy_MLE_Linear` | scipy | 0.2947 ± 0.0189 † | 0.0040 ± 0.0025 | 0.0158 ± 0.0105 | **0.0193 ± 0.0038** | - | - | 0.2900 |
| `scipy_MLE_Unconditional` | scipy | 1.5254 ± 0.0116 | 1.2346 ± 0.0282 | - | - | - | - | 0.0128 |
| `tr_GaussianNLL_MLP` | torchregress | 0.3710 ± 0.0095 | 0.0803 ± 0.0106 | - | - | - | - | 0.8472 |
| `tr_Linear` | torchregress | 0.2947 ± 0.0188 † | 0.0039 ± 0.0025 | **0.0157 ± 0.0106** | 0.0193 ± 0.0038 | - | - | 0.0486 |
| `tr_MLP` | torchregress | 0.3260 ± 0.0250 † | 0.0352 ± 0.0087 | 0.0633 ± 0.0110 | 0.0478 ± 0.0130 | - | - | 0.9052 |

##### likelihoods / lik_skewt

| Method | Library | NLL | NLLGapToTruth | LocRMSE | ScaleRMSE | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrueModel` | scipy | **0.5601 ± 0.0083** | **0.0000 ± 0.0000** | - | - | - | - | 0.0032 |
| `scipy_MLE_Linear` | scipy | 0.5651 ± 0.0093 | 0.0050 ± 0.0021 | 0.0297 ± 0.0201 | 0.0414 ± 0.0166 | - | - | 3.5687 |
| `scipy_MLE_Unconditional` | scipy | 1.5775 ± 0.0119 | 1.0174 ± 0.0173 | - | - | - | - | 0.2283 |
| `tr_GaussianNLL_MLP` | torchregress | 0.7671 ± 0.0514 † | 0.2070 ± 0.0476 | - | - | - | - | 0.8290 |
| `tr_Linear` | torchregress | 0.5651 ± 0.0093 | 0.0050 ± 0.0021 | **0.0296 ± 0.0200** | **0.0414 ± 0.0167** | - | - | 0.8500 |
| `tr_MLP` | torchregress | 0.5824 ± 0.0069 | 0.0223 ± 0.0047 | 0.0776 ± 0.0246 | 0.0582 ± 0.0191 | - | - | 7.4143 |

#### missing_data (tabular)

Group `tabular`; 3 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### missing_data / california_housing

| Method | Library | RMSE | MAE | R2 | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `CompleteCase` | torchregress | **0.6558 ± 0.0658** † | **0.4139 ± 0.0135** | **0.6750 ± 0.0654** | 0.8967 |
| `OracleIPW` | torchregress | 0.7605 ± 0.2492 † | 0.4244 ± 0.0405 | 0.5346 ± 0.3108 | 1.4781 |
| `PropensityWeighted` | torchregress | 0.7527 ± 0.2200 † | 0.4254 ± 0.0387 | 0.5504 ± 0.2682 | 1.6134 |
| `sklearn_IPW` | sklearn | 0.8029 ± 0.0511 † | 0.5375 ± 0.0127 | 0.5147 ± 0.0622 | 0.0074 |

##### missing_data / diabetes

| Method | Library | RMSE | MAE | R2 | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `CompleteCase` | torchregress | 54.5509 ± 0.9795 | 43.5465 ± 1.5284 | 0.4486 ± 0.0197 | 0.0869 |
| `OracleIPW` | torchregress | 54.9032 ± 0.3109 | 43.8046 ± 1.3413 | 0.4416 ± 0.0063 | 0.1528 |
| `PropensityWeighted` | torchregress | 54.8692 ± 0.2637 | 43.8509 ± 1.2435 | 0.4423 ± 0.0054 | 0.1581 |
| `sklearn_IPW` | sklearn | **54.2739 ± 0.6038** | **43.2526 ± 0.8534** | **0.4543 ± 0.0122** | 0.0043 |

##### missing_data / folktables_income

| Method | Library | RMSE | MAE | R2 | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `CompleteCase` | torchregress | 79824.0 ± 858.2 | 40873.5 ± 697.4 | 0.1901 ± 0.0174 | 3.5536 |
| `OracleIPW` | torchregress | 83645.6 ± 1926.6 | 42932.9 ± 1159.7 | 0.1105 ± 0.0410 | 6.2166 |
| `PropensityWeighted` | torchregress | 82628.1 ± 2194.5 | 41923.4 ± 748.3 | 0.1319 ± 0.0458 | 6.5891 |
| `sklearn_IPW` | sklearn | **74295.5 ± 165.5** | **39587.5 ± 120.5** | **0.2985 ± 0.0031** | 0.0253 |

#### multivariate_intervals (tabular)

Group `tabular`; 5 dataset(s). Columns: JointCoverage ·, IntervalScore ↓, EnergyScore ↓, VariogramScore ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: IntervalScore.

##### multivariate_intervals / energy_efficiency

| Method | Library | JointCoverage | IntervalScore | EnergyScore | VariogramScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.9048 ± 0.0263 | 95.7645 ± 2.2602 | 18.8665 ± 0.6136 | 13.8792 ± 0.5104 | 0.3897 |
| `HeteroEnsembleCov` | torchregress | 0.9192 ± 0.0025 | 103.4 ± 2.6 | 19.4978 ± 0.2196 | 15.9648 ± 0.4340 | 0.9597 |
| `LowRankGaussian` | torchregress | 0.6508 ± 0.1825 | 74.0862 ± 17.3248 † | 14.0474 ± 1.9469 | 0.4098 ± 0.0112 | 1.2470 |
| `MDN` | torchregress | 0.9076 ± 0.0050 | 100.6 ± 11.1 † | 20.5051 ± 0.4526 | 0.4082 ± 0.0034 | 0.4400 |
| `MVScore` | torchregress | 0.9192 ± 0.0218 | 64.7671 ± 5.8503 † | - | - | 0.8172 |
| `MultiQuantile` | torchregress | 0.7013 ± 0.0087 | **17.6895 ± 0.5327** | - | - | 0.2149 |
| `MultiTargetCP` | torchregress | 0.8687 ± 0.0304 | 20.1413 ± 0.7496 | - | - | 0.2619 |
| `MultivariateGaussian` | torchregress | 0.4387 ± 0.1256 | 105.8 ± 31.1 † | 15.6205 ± 2.5247 | 0.5130 ± 0.0594 | 0.7800 |
| `MultivariateGaussian+Conformal` | torchregress | 0.9177 ± 0.0198 | 69.7466 ± 4.9744 † | - | - | 0.3491 |
| `MultivariateGaussian+LevelSetCP` | torchregress | 0.9177 ± 0.0198 | 69.2710 ± 4.7121 † | - | - | 0.3512 |
| `NF_Ensemble_K10` | torchregress | 0.0000 ± 0.0000 | 450.8 ± 0.4 | 33.2469 ± 0.0420 | 0.6548 ± 0.0051 | 33.3866 |
| `NF_Ensemble_K2` | torchregress | 0.0000 ± 0.0000 | 453.8 ± 2.5 | 33.3967 ± 0.1408 | 0.6582 ± 0.0598 | 8.6722 |
| `NF_Ensemble_K20` | torchregress | 0.0000 ± 0.0000 | 451.1 ± 0.2 | 33.2571 ± 0.0121 | 0.6608 ± 0.0021 | 58.4020 |
| `NF_Ensemble_K5` | torchregress | 0.0000 ± 0.0000 | 450.3 ± 1.3 | 33.2606 ± 0.0883 | 0.6423 ± 0.0060 | 14.2646 |
| `NormalizingFlow` | torchregress | 0.0000 ± 0.0000 | 454.4 ± 1.0 | 33.3748 ± 0.1150 | 0.6451 ± 0.0721 | 4.2511 |
| `SLS` | torchregress | 0.9942 ± 0.0100 | 51.9910 ± 3.4806 † | - | - | 2.7373 |
| `SLS+SLSConformal` | torchregress | 1.0000 ± 0.0000 | 62.3837 ± 11.3992 † | - | - | 2.1062 |
| `SLS_earlystop` | torchregress | 1.0000 ± 0.0000 | 82.4662 ± 5.8906 † | - | - | 0.9627 |
| `SLS_lowrank` | torchregress | 0.3896 ± 0.0327 | 84.1079 ± 8.4242 † | - | - | 2.8242 |
| `SLS_lowrank+SLSConformal` | torchregress | 0.9827 ± 0.0043 | 49.9644 ± 6.2486 † | - | - | 2.2222 |
| `TICovariance` | torchregress | 0.8167 ± 0.0090 | 50.8009 ± 4.9766 † | 9.6901 ± 0.4157 | **0.3464 ± 0.0065** | 2.5300 |
| `WassersteinBound` | torchregress | 0.7518 ± 0.0066 | 19.9299 ± 0.8288 | **3.2751 ± 0.0820** | 0.4564 ± 0.0052 | 0.6914 |

##### multivariate_intervals / synthetic_multivariate

| Method | Library | JointCoverage | IntervalScore | EnergyScore | VariogramScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.6506 ± 0.0100 | 0.9795 ± 0.0229 | 0.2594 ± 0.0049 | 0.0998 ± 0.0027 | 1.0340 |
| `HeteroEnsembleCov` | torchregress | 0.7311 ± 0.0025 | 0.9529 ± 0.0045 | 0.2557 ± 0.0021 | 0.0978 ± 0.0014 | 3.3619 |
| `LowRankGaussian` | torchregress | 0.6506 ± 0.0266 | 1.0181 ± 0.0286 | 0.2696 ± 0.0087 | 0.1061 ± 0.0046 | 2.5818 |
| `MDN` | torchregress | 0.6383 ± 0.0017 | 1.0354 ± 0.0434 | 0.2682 ± 0.0102 | 0.1037 ± 0.0066 | 1.0679 |
| `MVScore` | torchregress | 0.9261 ± 0.0155 | 1.1743 ± 0.0332 | - | - | 2.0953 |
| `MultiQuantile` | torchregress | 0.4494 ± 0.0125 | 1.1253 ± 0.0510 | - | - | 0.6091 |
| `MultiTargetCP` | torchregress | 0.6472 ± 0.0478 | 0.9759 ± 0.0082 | - | - | 0.6186 |
| `MultivariateGaussian` | torchregress | 0.6650 ± 0.0073 | 1.0316 ± 0.0270 | 0.2735 ± 0.0089 | 0.1067 ± 0.0039 | 2.3690 |
| `MultivariateGaussian+Conformal` | torchregress | 0.9722 ± 0.0127 | 1.4559 ± 0.1385 † | - | - | 0.9221 |
| `MultivariateGaussian+LevelSetCP` | torchregress | 0.9733 ± 0.0109 | 1.4196 ± 0.1273 † | - | - | 0.9294 |
| `NF_Ensemble_K10` | torchregress | 0.7017 ± 0.0060 | 0.9339 ± 0.0027 | 0.2531 ± 0.0003 | 0.0969 ± 0.0001 | 86.4344 |
| `NF_Ensemble_K2` | torchregress | 0.6167 ± 0.0076 | 0.9903 ± 0.0121 | 0.2620 ± 0.0006 | 0.1025 ± 0.0005 | 19.2351 |
| `NF_Ensemble_K20` | torchregress | 0.7156 ± 0.0019 | **0.9255 ± 0.0011** | **0.2512 ± 0.0006** | **0.0955 ± 0.0003** | 128.2 |
| `NF_Ensemble_K5` | torchregress | 0.6633 ± 0.0117 | 0.9525 ± 0.0067 | 0.2563 ± 0.0010 | 0.0987 ± 0.0004 | 44.8843 |
| `NormalizingFlow` | torchregress | 0.5444 ± 0.0095 | 1.0818 ± 0.0433 | 0.2741 ± 0.0018 | 0.1091 ± 0.0017 | 11.8277 |
| `SLS` | torchregress | 0.7522 ± 0.0461 | 1.1218 ± 0.0328 | - | - | 9.5740 |
| `SLS+SLSConformal` | torchregress | 0.9678 ± 0.0075 | 1.5483 ± 0.0613 | - | - | 6.1504 |
| `SLS_earlystop` | torchregress | 0.9678 ± 0.0010 | 1.8912 ± 0.1899 † | - | - | 1.7114 |
| `SLS_lowrank` | torchregress | 0.7206 ± 0.0407 | 1.1187 ± 0.0222 | - | - | 10.6459 |
| `SLS_lowrank+SLSConformal` | torchregress | 0.9650 ± 0.0017 | 1.5905 ± 0.0870 † | - | - | 6.1285 |
| `TICovariance` | torchregress | 0.9983 ± 0.0017 | 1.9677 ± 0.0089 | 0.3779 ± 0.0062 | 0.1896 ± 0.0049 | 9.3857 |
| `WassersteinBound` | torchregress | 0.9822 ± 0.0051 | 1.4271 ± 0.0027 | 0.2960 ± 0.0031 | 0.1329 ± 0.0024 | 1.6081 |

##### multivariate_intervals / synthetic_mv_bimodal

| Method | Library | JointCoverage | IntervalScore | EnergyScore | VariogramScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.7389 ± 0.0067 | 6.0679 ± 0.0112 | 1.9551 ± 0.0028 | 1.4797 ± 0.0090 | 0.9294 |
| `HeteroEnsembleCov` | torchregress | 0.7433 ± 0.0017 | 6.0157 ± 0.0061 | 1.9508 ± 0.0023 | 1.4773 ± 0.0025 | 3.1799 |
| `LowRankGaussian` | torchregress | 0.7450 ± 0.0058 | 6.0993 ± 0.0254 | 1.8628 ± 0.0036 | 1.3645 ± 0.0057 | 2.9540 |
| `MDN` | torchregress | 0.6594 ± 0.0144 | 5.9498 ± 0.0223 | **1.8296 ± 0.0016** | **1.3365 ± 0.0030** | 1.2733 |
| `MVScore` | torchregress | 0.9894 ± 0.0054 | 8.0721 ± 0.2290 | - | - | 2.5486 |
| `MultiQuantile` | torchregress | 0.4844 ± 0.0222 | 6.5117 ± 0.1014 | - | - | 0.6747 |
| `MultiTargetCP` | torchregress | 0.6861 ± 0.0208 | 6.0818 ± 0.0217 | - | - | 0.6750 |
| `MultivariateGaussian` | torchregress | 0.7522 ± 0.0125 | 6.0892 ± 0.0344 | 1.8612 ± 0.0061 | 1.3593 ± 0.0026 | 2.1419 |
| `MultivariateGaussian+Conformal` | torchregress | 0.9978 ± 0.0025 | 8.7501 ± 0.2533 | - | - | 1.1058 |
| `MultivariateGaussian+LevelSetCP` | torchregress | 0.9989 ± 0.0010 | 8.7239 ± 0.3242 | - | - | 1.1161 |
| `NF_Ensemble_K10` | torchregress | 0.6283 ± 0.0073 | 5.9802 ± 0.0039 | 1.8328 ± 0.0019 | 1.3395 ± 0.0013 | 61.2559 |
| `NF_Ensemble_K2` | torchregress | 0.6144 ± 0.0107 | 6.1040 ± 0.0204 | 1.8484 ± 0.0073 | 1.3506 ± 0.0077 | 21.7627 |
| `NF_Ensemble_K20` | torchregress | 0.6394 ± 0.0019 | **5.9479 ± 0.0045** | 1.8339 ± 0.0025 | 1.3447 ± 0.0064 | 143.3 |
| `NF_Ensemble_K5` | torchregress | 0.6217 ± 0.0100 | 6.0224 ± 0.0010 | 1.8405 ± 0.0042 | 1.3450 ± 0.0045 | 36.4746 |
| `NormalizingFlow` | torchregress | 0.5889 ± 0.0294 | 6.2316 ± 0.0354 | 1.8695 ± 0.0022 | 1.3666 ± 0.0017 | 10.4726 |
| `SLS` | torchregress | 0.8678 ± 0.0214 | 6.5810 ± 0.0422 | - | - | 11.6353 |
| `SLS+SLSConformal` | torchregress | 0.9944 ± 0.0010 | 9.8852 ± 0.2112 | - | - | 7.2480 |
| `SLS_earlystop` | torchregress | 0.9839 ± 0.0010 | 7.8748 ± 0.4071 † | - | - | 2.0429 |
| `SLS_lowrank` | torchregress | 0.8761 ± 0.0334 | 6.6953 ± 0.0915 | - | - | 9.6831 |
| `SLS_lowrank+SLSConformal` | torchregress | 0.9933 ± 0.0060 | 10.2270 ± 0.8461 † | - | - | 6.8331 |
| `TICovariance` | torchregress | 0.7283 ± 0.0132 | 6.9068 ± 0.0459 | 1.9686 ± 0.0072 | 1.4836 ± 0.0249 | 10.4912 |
| `WassersteinBound` | torchregress | 0.7089 ± 0.0095 | 6.0797 ± 0.0435 | 1.9629 ± 0.0074 | 1.4825 ± 0.0091 | 1.6823 |

##### multivariate_intervals / synthetic_mv_gaussian

| Method | Library | JointCoverage | IntervalScore | EnergyScore | VariogramScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.6639 ± 0.0069 | 4.4683 ± 0.0288 | 1.1873 ± 0.0055 | 0.5738 ± 0.0028 | 0.8834 |
| `HeteroEnsembleCov` | torchregress | 0.6667 ± 0.0044 | 4.4148 ± 0.0126 | 1.1774 ± 0.0016 | 0.5688 ± 0.0003 | 2.2873 |
| `LowRankGaussian` | torchregress | 0.6783 ± 0.0130 | 4.4141 ± 0.0643 | 1.1680 ± 0.0077 | 0.5500 ± 0.0044 | 2.5214 |
| `MDN` | torchregress | 0.6789 ± 0.0077 | 4.4660 ± 0.0550 | **1.1672 ± 0.0064** | **0.5467 ± 0.0046** | 1.0610 |
| `MVScore` | torchregress | 0.9394 ± 0.0111 | 4.9139 ± 0.0757 | - | - | 2.0966 |
| `MultiQuantile` | torchregress | 0.5072 ± 0.0075 | 4.9382 ± 0.0764 | - | - | 0.5257 |
| `MultiTargetCP` | torchregress | 0.6928 ± 0.0082 | 4.4031 ± 0.0309 | - | - | 0.6191 |
| `MultivariateGaussian` | torchregress | 0.6717 ± 0.0144 | **4.3770 ± 0.0091** | 1.1678 ± 0.0033 | 0.5482 ± 0.0023 | 2.0280 |
| `MultivariateGaussian+Conformal` | torchregress | 0.9711 ± 0.0111 | 5.5682 ± 0.1932 | - | - | 0.9234 |
| `MultivariateGaussian+LevelSetCP` | torchregress | 0.9722 ± 0.0102 | 5.5155 ± 0.1536 | - | - | 0.9300 |
| `NF_Ensemble_K10` | torchregress | 0.6678 ± 0.0010 | 4.4507 ± 0.0065 | 1.1762 ± 0.0029 | 0.5468 ± 0.0002 | 65.3850 |
| `NF_Ensemble_K2` | torchregress | 0.6411 ± 0.0095 | 4.5366 ± 0.0445 | 1.1858 ± 0.0080 | 0.5544 ± 0.0059 | 13.0981 |
| `NF_Ensemble_K20` | torchregress | 0.6694 ± 0.0019 | 4.4199 ± 0.0038 | 1.1738 ± 0.0019 | 0.5487 ± 0.0008 | 144.1 |
| `NF_Ensemble_K5` | torchregress | 0.6600 ± 0.0029 | 4.5050 ± 0.0191 | 1.1814 ± 0.0037 | 0.5502 ± 0.0013 | 30.3452 |
| `NormalizingFlow` | torchregress | 0.6117 ± 0.0067 | 4.6918 ± 0.0126 | 1.2003 ± 0.0038 | 0.5654 ± 0.0072 | 10.3336 |
| `SLS` | torchregress | 0.8011 ± 0.0069 | 4.7966 ± 0.0380 | - | - | 8.7433 |
| `SLS+SLSConformal` | torchregress | 0.9939 ± 0.0035 | 8.1389 ± 0.5914 † | - | - | 5.2375 |
| `SLS_earlystop` | torchregress | 0.9517 ± 0.0017 | 5.2089 ± 0.0962 | - | - | 2.0640 |
| `SLS_lowrank` | torchregress | 0.8028 ± 0.0077 | 4.7939 ± 0.1185 | - | - | 8.3024 |
| `SLS_lowrank+SLSConformal` | torchregress | 0.9883 ± 0.0050 | 7.7383 ± 0.0845 | - | - | 5.3133 |
| `TICovariance` | torchregress | 0.6633 ± 0.0044 | 4.6328 ± 0.0293 | 1.2134 ± 0.0017 | 0.5701 ± 0.0045 | 6.6297 |
| `WassersteinBound` | torchregress | 0.6844 ± 0.0079 | 4.4035 ± 0.0031 | 1.1865 ± 0.0019 | 0.5782 ± 0.0043 | 1.5696 |

##### multivariate_intervals / synthetic_mv_nongaussian

| Method | Library | JointCoverage | IntervalScore | EnergyScore | VariogramScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.7622 ± 0.0035 | 4.1209 ± 0.0144 | 1.1363 ± 0.0015 | 0.5715 ± 0.0014 | 0.8986 |
| `HeteroEnsembleCov` | torchregress | 0.7867 ± 0.0076 | 4.0989 ± 0.0168 | 1.1286 ± 0.0018 | 0.5693 ± 0.0016 | 2.8442 |
| `LowRankGaussian` | torchregress | 0.7389 ± 0.0079 | 4.1571 ± 0.0296 | 1.1378 ± 0.0025 | 0.5775 ± 0.0048 | 2.3970 |
| `MDN` | torchregress | 0.7072 ± 0.0410 | 3.9418 ± 0.0446 | 1.1195 ± 0.0039 | 0.5681 ± 0.0042 | 1.0802 |
| `MVScore` | torchregress | 0.9233 ± 0.0044 | 5.0468 ± 0.0966 | - | - | 1.9853 |
| `MultiQuantile` | torchregress | 0.4750 ± 0.0217 | 4.2003 ± 0.0177 | - | - | 0.5328 |
| `MultiTargetCP` | torchregress | 0.6911 ± 0.0051 | 4.0656 ± 0.0178 | - | - | 0.6117 |
| `MultivariateGaussian` | torchregress | 0.7711 ± 0.0134 | 4.1288 ± 0.0225 | 1.1339 ± 0.0062 | 0.5752 ± 0.0051 | 1.9580 |
| `MultivariateGaussian+Conformal` | torchregress | 0.9483 ± 0.0076 | 5.6914 ± 0.0712 | - | - | 0.9377 |
| `MultivariateGaussian+LevelSetCP` | torchregress | 0.9478 ± 0.0038 | 5.6404 ± 0.0972 | - | - | 0.9378 |
| `NF_Ensemble_K10` | torchregress | 0.6617 ± 0.0017 | **3.8620 ± 0.0049** | 1.1107 ± 0.0013 | 0.5601 ± 0.0014 | 63.8300 |
| `NF_Ensemble_K2` | torchregress | 0.6167 ± 0.0083 | 3.9315 ± 0.0265 | 1.1173 ± 0.0023 | 0.5642 ± 0.0033 | 12.3780 |
| `NF_Ensemble_K20` | torchregress | 0.6694 ± 0.0038 | 3.8625 ± 0.0038 | **1.1085 ± 0.0015** | **0.5597 ± 0.0009** | 143.0 |
| `NF_Ensemble_K5` | torchregress | 0.6461 ± 0.0038 | 3.8852 ± 0.0092 | 1.1087 ± 0.0014 | 0.5625 ± 0.0016 | 33.1073 |
| `NormalizingFlow` | torchregress | 0.5850 ± 0.0073 | 4.0018 ± 0.0301 | 1.1248 ± 0.0051 | 0.5660 ± 0.0015 | 9.9284 |
| `SLS` | torchregress | 0.8461 ± 0.0102 | 4.5425 ± 0.0512 | - | - | 11.1649 |
| `SLS+SLSConformal` | torchregress | 0.9822 ± 0.0054 | 8.2030 ± 0.7676 † | - | - | 7.2523 |
| `SLS_earlystop` | torchregress | 0.9317 ± 0.0104 | 5.5355 ± 0.3011 † | - | - | 2.0358 |
| `SLS_lowrank` | torchregress | 0.8211 ± 0.0243 | 4.5820 ± 0.0366 | - | - | 9.9936 |
| `SLS_lowrank+SLSConformal` | torchregress | 0.9756 ± 0.0111 | 8.1940 ± 0.5427 † | - | - | 5.9139 |
| `TICovariance` | torchregress | 0.7911 ± 0.0019 | 4.1168 ± 0.0267 | 1.1425 ± 0.0068 | 0.5727 ± 0.0038 | 7.9853 |
| `WassersteinBound` | torchregress | 0.8178 ± 0.0051 | 4.1053 ± 0.0075 | 1.1325 ± 0.0044 | 0.5668 ± 0.0023 | 1.4918 |

#### ordinal (tabular)

Group `tabular`; 2 dataset(s). Columns: MAE ↓, Accuracy ↑, QWK ↑, ClassMAE ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: MAE.

##### ordinal / folktables_decile

| Method | Library | MAE | Accuracy | QWK | ClassMAE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CORAL` | torchregress | 1.5226 ± 0.0063 | 0.2564 ± 0.0022 | 0.7143 ± 0.0034 | 1.5226 ± 0.0063 | 19.1584 |
| `CumulativeLink` | torchregress | 1.4714 ± 0.0046 | 0.2708 ± 0.0032 | 0.7227 ± 0.0006 | 1.4714 ± 0.0046 | 14.7369 |
| `CumulativeLinkEnsemble` | torchregress | **1.4358 ± 0.0045** | 0.2454 ± 0.0019 | 0.7254 ± 0.0020 | **1.4358 ± 0.0045** | 54.2916 |
| `DiscreteWasserstein1` | torchregress | 1.6376 ± 0.0788 | 0.1845 ± 0.0090 | 0.6414 ± 0.0408 | 1.6376 ± 0.0788 | 15.9725 |
| `MSE` | torchregress | 3.7202 ± 0.0022 | 0.1218 ± 0.0032 | 0.0181 ± 0.0022 | 3.7202 ± 0.0022 | 7.0604 |
| `OrdinalCrossEntropy` | torchregress | 1.5926 ± 0.0185 | 0.2773 ± 0.0043 | 0.6993 ± 0.0065 | 1.5926 ± 0.0185 | 13.6396 |
| `statsmodels_OrderedModel` | statsmodels | 1.5238 ± 0.0000 | **0.2778 ± 0.0000** | **0.7300 ± 0.0000** | 1.5238 ± 0.0000 | 25.8078 |

##### ordinal / synthetic_ordinal

| Method | Library | MAE | Accuracy | QWK | ClassMAE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CORAL` | torchregress | 0.2900 ± 0.0164 † | 0.7133 ± 0.0142 | 0.9255 ± 0.0048 | 0.2900 ± 0.0164 | 2.4940 |
| `CumulativeLink` | torchregress | 0.2642 ± 0.0095 | 0.7367 ± 0.0104 | 0.9322 ± 0.0024 | 0.2642 ± 0.0095 | 2.3021 |
| `CumulativeLinkEnsemble` | torchregress | 0.2550 ± 0.0109 | 0.7450 ± 0.0109 | 0.9349 ± 0.0028 | 0.2550 ± 0.0109 | 7.6384 |
| `DiscreteWasserstein1` | torchregress | 0.3283 ± 0.0794 † | 0.6742 ± 0.0794 | 0.9076 ± 0.0311 | 0.3283 ± 0.0794 | 2.6979 |
| `MSE` | torchregress | 1.2675 ± 0.0025 | 0.3250 ± 0.0025 | 0.2121 ± 0.0028 | 1.2675 ± 0.0025 | 0.5972 |
| `OrdinalCrossEntropy` | torchregress | 0.2850 ± 0.0025 | 0.7175 ± 0.0050 | 0.9255 ± 0.0011 | 0.2850 ± 0.0025 | 1.7072 |
| `statsmodels_OrderedModel` | statsmodels | **0.2500 ± 0.0000** | **0.7500 ± 0.0000** | **0.9357 ± 0.0000** | **0.2500 ± 0.0000** | 0.1277 |

#### orthogonal_inference (tabular)

Group `tabular`; 2 dataset(s). Columns: Bias |x|↓, RMSE ↓, Coverage95 ·, CIWidth ·, SEratio · (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: Bias.

##### orthogonal_inference / plr_ccddhnr

| Method | Library | Bias | RMSE | Coverage95 | CIWidth | SEratio | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DoubleML_PLR_Lasso` | doubleml | -8.721e-04 ± 2.777e-03 † | 0.0448 ± 0.0015 | 0.9400 ± 0.0250 | 0.1757 ± 0.0014 | 1.0005 ± 0.0253 | 0.3381 |
| `DoubleML_PLR_Poly3Ridge` | doubleml | -0.0036 ± 0.0057 † | 0.0538 ± 0.0012 | 0.9017 ± 0.0144 | 0.1792 ± 0.0012 | 0.8532 ± 0.0196 | 0.0221 |
| `DoubleML_PLR_RF` | doubleml | -0.0116 ± 0.0040 † | 0.0483 ± 0.0016 | 0.9250 ± 0.0265 | 0.1753 ± 0.0016 | 0.9537 ± 0.0201 | 1.6405 |
| `EconML_LinearDML_Lasso` | econml | -9.734e-04 ± 2.429e-03 † | **0.0444 ± 0.0015** | 0.9417 ± 0.0225 | 0.1758 ± 0.0014 | 1.0082 ± 0.0263 | 0.0730 |
| `tr_Naive` | torchregress | 0.1664 ± 0.0014 | 0.1695 ± 0.0013 | 0.0000 ± 0.0000 | 0.1283 ± 0.0002 | 1.0260 ± 0.0798 | 4.521e-04 |
| `tr_Orthogonal_NoCrossFit` | torchregress | -0.0025 ± 0.0029 † | 0.0471 ± 0.0018 | 0.9383 ± 0.0189 | 0.1763 ± 0.0012 | 0.9560 ± 0.0300 | 8.831e-04 |
| `tr_Orthogonal_Poly3` | torchregress | -0.0042 ± 0.0050 † | 0.0528 ± 0.0019 | 0.9183 ± 0.0153 | 0.1790 ± 0.0010 | 0.8694 ± 0.0329 | 0.0025 |
| `tr_Orthogonal_RFF` | torchregress | **2.612e-04 ± 4.854e-03** † | 0.0499 ± 0.0028 | 0.9200 ± 0.0229 | 0.1758 ± 0.0010 | 0.9010 ± 0.0444 | 0.1248 |
| `tr_Orthogonal_RFF_legacy` | torchregress | -6.356e-04 ± 2.640e-03 † | 0.0678 ± 0.0025 | 0.8083 ± 0.0076 | 0.1802 ± 0.0015 | 0.6773 ± 0.0201 | 0.0089 |

##### orthogonal_inference / plr_cubic

| Method | Library | Bias | RMSE | Coverage95 | CIWidth | SEratio | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DoubleML_PLR_Lasso` | doubleml | 0.4961 ± 0.0014 | 0.5092 ± 0.0020 | 0.0033 ± 0.0029 | 0.4008 ± 0.0046 | 0.8913 ± 0.0552 | 0.2712 |
| `DoubleML_PLR_Poly3Ridge` | doubleml | 0.0026 ± 0.0009 † | 0.0465 ± 0.0021 | 0.9367 ± 0.0153 | 0.1757 ± 0.0003 | 0.9637 ± 0.0445 | 0.0172 |
| `DoubleML_PLR_RF` | doubleml | 0.2198 ± 0.0066 | 0.2396 ± 0.0058 | 0.2150 ± 0.0218 | 0.3242 ± 0.0045 | 0.8679 ± 0.0467 | 0.7454 |
| `EconML_LinearDML_Lasso` | econml | 0.4965 ± 0.0005 | 0.5095 ± 0.0011 | 0.0067 ± 0.0029 | 0.4001 ± 0.0051 | 0.8918 ± 0.0606 | 0.0634 |
| `tr_Naive` | torchregress | 0.7720 ± 0.0077 | 0.7838 ± 0.0071 | 0.0000 ± 0.0000 | 0.2914 ± 0.0014 | 0.5486 ± 0.0351 | 2.045e-04 |
| `tr_Orthogonal_NoCrossFit` | torchregress | 0.0028 ± 0.0013 † | **0.0451 ± 0.0024** | 0.9400 ± 0.0180 | 0.1750 ± 0.0005 | 0.9916 ± 0.0553 | 4.634e-04 |
| `tr_Orthogonal_Poly3` | torchregress | 0.0027 ± 0.0019 † | 0.0461 ± 0.0038 | 0.9367 ± 0.0321 | 0.1753 ± 0.0006 | 0.9741 ± 0.0773 | 0.0012 |
| `tr_Orthogonal_RFF` | torchregress | **0.0020 ± 0.0019** † | 0.0463 ± 0.0038 | 0.9367 ± 0.0236 | 0.1756 ± 0.0006 | 0.9706 ± 0.0793 | 0.1094 |
| `tr_Orthogonal_RFF_legacy` | torchregress | 0.0683 ± 0.0080 † | 0.1276 ± 0.0048 | 0.7850 ± 0.0180 | 0.3072 ± 0.0092 | 0.7280 ± 0.0411 | 0.0107 |

#### ppi (tabular)

Group `tabular`; 3 dataset(s). Columns: Estimate ·, CIWidth ·, Covers ·, RMSE ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### ppi / california_housing

| Method | Library | Estimate | CIWidth | Covers | RMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Classical` | torchregress | 1.8143 ± 0.1049 | 4.2137 ± 0.2160 | 1.0000 ± 0.0000 | **0.5843 ± 0.0075** | 1.5402 |
| `PPI_CalibratedMean` | torchregress | 1.9846 ± 0.0364 | 0.2056 ± 0.0165 | 0.6667 ± 0.5774 | **0.5843 ± 0.0075** | 1.3305 |
| `PPI_Mean` | torchregress | 1.9805 ± 0.0374 | 0.2082 ± 0.0119 | 0.6667 ± 0.5774 | **0.5843 ± 0.0075** | 1.3542 |
| `PPI_OLS` | torchregress | 0.2135 ± 0.0530 | 0.6990 ± 0.1255 | 0.1852 ± 0.0642 | **0.5843 ± 0.0075** | 1.0548 |
| `PPI_PPMean` | torchregress | 1.9880 ± 0.0365 | 0.2096 ± 0.0064 | 0.6667 ± 0.5774 | **0.5843 ± 0.0075** | 1.1684 |
| `PPI_Quantile` | torchregress | 1.7457 ± 0.0578 | 0.4767 ± 0.2177 | 1.0000 ± 0.0000 | **0.5843 ± 0.0075** | 1.3244 |
| `PPI_py_Mean` | ppi_py | 1.9806 ± 0.0376 | 0.2102 ± 0.0076 | 0.6667 ± 0.5774 | **0.5843 ± 0.0075** | 0.9263 |
| `PPI_py_Quantile` | ppi_py | 1.7574 ± 0.0627 | 0.4751 ± 0.2171 | 1.0000 ± 0.0000 | **0.5843 ± 0.0075** | 0.9097 |

##### ppi / diabetes

| Method | Library | Estimate | CIWidth | Covers | RMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Classical` | torchregress | 139.5 ± 9.5 | 227.9 ± 1.4 | 1.0000 ± 0.0000 | **53.1870 ± 1.2101** | 0.1499 |
| `PPI_CalibratedMean` | torchregress | 150.0 ± 3.6 | 33.0862 ± 3.3412 | 1.0000 ± 0.0000 | **53.1870 ± 1.2101** | 0.1563 |
| `PPI_Mean` | torchregress | 150.6 ± 3.3 | 31.3126 ± 2.7100 | 1.0000 ± 0.0000 | **53.1870 ± 1.2101** | 0.1456 |
| `PPI_OLS` | torchregress | 17.2787 ± 7.2351 | 94.0100 ± 9.3289 | 0.6364 ± 0.0909 | **53.1870 ± 1.2101** | 0.1810 |
| `PPI_PPMean` | torchregress | 150.6 ± 3.7 | 27.4063 ± 1.4750 | 1.0000 ± 0.0000 | **53.1870 ± 1.2101** | 0.1387 |
| `PPI_Quantile` | torchregress | 129.2 ± 25.7 | 76.2911 ± 29.3582 | 1.0000 ± 0.0000 | **53.1870 ± 1.2101** | 0.1529 |
| `PPI_py_Mean` | ppi_py | 150.6 ± 3.7 | 27.1887 ± 1.4678 | 1.0000 ± 0.0000 | **53.1870 ± 1.2101** | 0.1452 |
| `PPI_py_Quantile` | ppi_py | 142.6 ± 12.2 | 76.7340 ± 29.2947 | 1.0000 ± 0.0000 | **53.1870 ± 1.2101** | 0.1664 |

##### ppi / folktables_income

| Method | Library | Estimate | CIWidth | Covers | RMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Classical` | torchregress | 36083.3 ± 9887.9 | 1.879e+05 ± 4.162e+04 | 1.0000 ± 0.0000 | **80208.4 ± 3081.8** | 7.1012 |
| `PPI_CalibratedMean` | torchregress | 68835.6 ± 7314.4 | 34146.6 ± 13003.6 | 1.0000 ± 0.0000 | **80208.4 ± 3081.8** | 5.7346 |
| `PPI_Mean` | torchregress | 68391.6 ± 5790.5 | 30454.0 ± 4033.3 | 1.0000 ± 0.0000 | **80208.4 ± 3081.8** | 5.8588 |
| `PPI_OLS` | torchregress | 78.0190 ± 1089.8872 | 61583.6 ± 37959.6 | 0.9061 ± 0.0709 | **80208.4 ± 3081.8** | 5.8410 |
| `PPI_PPMean` | torchregress | 68132.1 ± 6190.7 | 30848.8 ± 2378.3 | 1.0000 ± 0.0000 | **80208.4 ± 3081.8** | 5.7156 |
| `PPI_Quantile` | torchregress | 39529.2 ± 6527.9 | 24257.0 ± 16136.5 | 0.6667 ± 0.5774 | **80208.4 ± 3081.8** | 5.8964 |
| `PPI_py_Mean` | ppi_py | 67143.0 ± 5094.8 | 31298.5 ± 2511.4 | 1.0000 ± 0.0000 | **80208.4 ± 3081.8** | 5.7507 |
| `PPI_py_Quantile` | ppi_py | 39977.3 ± 5567.6 | 23974.6 ± 16076.5 | 0.6667 ± 0.5774 | **80208.4 ± 3081.8** | 5.4012 |

#### pzflow_parity (tabular)

Group `tabular`; 4 dataset(s). Columns: NLL ↓, Coverage ·, Width ·, IntervalScore ↓, EnergyScore ↓, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: NLL.

##### pzflow_parity / energy_efficiency

| Method | Library | NLL | Coverage | Width | IntervalScore | EnergyScore | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | **2.4945 ± 0.2502** † | 0.8939 ± 0.0078 | 27.2411 ± 0.4380 | 32.5117 ± 0.1437 | 7.9655 ± 0.0032 | **0.0401 ± 0.0085** | 11.6095 |
| `torchregress_NSF` | torchregress | 2.6066 ± 0.0336 | 0.9278 ± 0.0162 | 7.2840 ± 0.9635 | 7.9387 ± 0.7050 | 1.8459 ± 0.0673 | 0.1293 ± 0.0366 | 3.4200 |
| `torchregress_NSF_recipe` | torchregress | 2.6478 ± 0.2672 † | 0.9004 ± 0.0262 | 6.4799 ± 0.5826 | **7.4676 ± 1.1428** | **1.7048 ± 0.2679** | 0.0764 ± 0.0246 | 3.7297 |

Failed / skipped:

- `normflows_NSF` (failed, seed 0, 1, 2): known-bad: normflows_NSF pzflow_parity #broadcast error on 1-D tabular inputs

##### pzflow_parity / pzflow_galaxies

| Method | Library | NLL | Coverage | Width | IntervalScore | EnergyScore | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | 21.0829 ± 0.3060 | 0.8971 ± 0.0012 | 1.8383 ± 0.0031 | 2.0272 ± 0.0005 | 0.3274 ± 0.0000 | **0.0053 ± 0.0007** | 20.7900 |
| `torchregress_NSF` | torchregress | 11.3800 ± 0.5435 | 0.8887 ± 0.0282 | 0.0489 ± 0.0026 | **0.0818 ± 0.0042** | 0.0096 ± 0.0001 | 0.0973 ± 0.0589 | 53.9556 |
| `torchregress_NSF_recipe` | torchregress | **10.6687 ± 0.2013** | 0.9024 ± 0.0081 | 0.0520 ± 0.0017 | 0.0842 ± 0.0056 | **0.0095 ± 0.0006** | 0.0467 ± 0.0229 | 46.7871 |

Failed / skipped:

- `normflows_NSF` (failed, seed 0, 1, 2): known-bad: normflows_NSF pzflow_parity #broadcast error on 1-D tabular inputs

##### pzflow_parity / synthetic_cond_gaussian

| Method | Library | NLL | Coverage | Width | IntervalScore | EnergyScore | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | 7.1500 ± 0.4551 † | 0.9000 ± 0.0017 | 9.5015 ± 0.0759 | 11.6726 ± 0.0106 | 1.5948 ± 0.0003 | **0.0178 ± 0.0019** | 11.0114 |
| `torchregress_NSF` | torchregress | 5.9795 ± 0.2293 | 0.8306 ± 0.0204 | 0.9336 ± 0.0168 | 1.4719 ± 0.0614 | 0.1913 ± 0.0042 | 0.0822 ± 0.0327 | 5.6660 |
| `torchregress_NSF_recipe` | torchregress | **5.4035 ± 0.2656** | 0.8622 ± 0.0312 | 1.0101 ± 0.0518 | **1.4126 ± 0.0434** | **0.1885 ± 0.0030** | 0.0672 ± 0.0433 | 3.1409 |

Failed / skipped:

- `normflows_NSF` (failed, seed 0, 1, 2): known-bad: normflows_NSF pzflow_parity #broadcast error on 1-D tabular inputs

##### pzflow_parity / synthetic_cond_nongaussian

| Method | Library | NLL | Coverage | Width | IntervalScore | EnergyScore | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pzflow_NSF` | pzflow | 4.7856 ± 0.1471 | 0.8928 ± 0.0059 | 9.8278 ± 0.0340 | 12.5392 ± 0.0487 | 1.6889 ± 0.0010 | **0.0294 ± 0.0042** | 11.2366 |
| `torchregress_NSF` | torchregress | 4.0895 ± 0.1536 | 0.8150 ± 0.0548 | 2.7563 ± 0.1063 | 3.4259 ± 0.0903 | 0.5998 ± 0.0071 | 0.0656 ± 0.0135 | 5.0917 |
| `torchregress_NSF_recipe` | torchregress | **3.5683 ± 0.1168** | 0.8550 ± 0.0136 | 2.8767 ± 0.0686 | **3.3517 ± 0.1214** | **0.5992 ± 0.0095** | 0.0572 ± 0.0184 | 3.1893 |

Failed / skipped:

- `normflows_NSF` (failed, seed 0, 1, 2): known-bad: normflows_NSF pzflow_parity #broadcast error on 1-D tabular inputs

#### r2c_conformal (tabular)

Group `tabular`; 5 dataset(s). Columns: Coverage ·, Width ·, IntervalScore ↓, RMSE ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: IntervalScore.

##### r2c_conformal / california_housing

| Method | Library | Coverage | Width | IntervalScore | RMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9153 ± 0.0121 | 1.8068 ± 0.0819 | **2.4130 ± 0.0118** | 0.6651 ± 0.0210 | 0.8931 |
| `R2C+CORAL` | torchregress | 0.9307 ± 0.0105 | 1.7650 ± 0.0675 | 2.5588 ± 0.0771 | 0.6498 ± 0.0084 | 2.9665 |
| `R2C+CumulativeLink` | torchregress | 0.9664 ± 0.0068 | 2.2316 ± 0.1055 | 2.6552 ± 0.0966 | 0.7205 ± 0.0214 | 2.2719 |
| `R2C+Softmax` | torchregress | 0.9582 ± 0.0015 | 2.1722 ± 0.0206 | 2.6816 ± 0.0546 | 0.7984 ± 0.0496 | 0.7074 |
| `Split+MLP` | torchregress | 0.9107 ± 0.0064 | 1.8625 ± 0.0329 | 2.8669 ± 0.0407 | **0.6064 ± 0.0056** | 0.6428 |
| `torchcp_R2C` | torchcp | 0.9542 ± 0.0793 | 5.1647 ± 0.9334 | 5.6522 ± 0.0891 | 3.1232 ± 0.3879 | 0.9167 |

##### r2c_conformal / diabetes

| Method | Library | Coverage | Width | IntervalScore | RMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9424 ± 0.0339 | 207.2 ± 16.6 | **224.5 ± 8.2** | **54.4420 ± 1.4813** | 0.1530 |
| `R2C+CORAL` | torchregress | 0.9674 ± 0.0087 | 231.2 ± 9.6 | 242.3 ± 4.2 | 57.4898 ± 1.3579 | 0.4892 |
| `R2C+CumulativeLink` | torchregress | 0.9925 ± 0.0130 | 358.0 ± 15.4 | 359.6 ± 12.9 | 58.7623 ± 1.6251 | 0.3936 |
| `R2C+Softmax` | torchregress | 0.9950 ± 0.0043 | 288.8 ± 31.4 | 288.8 ± 31.3 † | 66.3408 ± 2.4764 | 0.1480 |
| `Split+MLP` | torchregress | 0.9674 ± 0.0174 | 223.8 ± 18.4 | 240.2 ± 10.1 | 54.8428 ± 2.0646 | 0.1114 |
| `torchcp_R2C` | torchcp | 1.0000 ± 0.0000 | 352.9 ± 7.0 | 352.9 ± 7.0 | 182.1 ± 19.2 | 0.1537 |

##### r2c_conformal / house_sales

| Method | Library | Coverage | Width | IntervalScore | RMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9027 ± 0.0042 | 2.917e+05 ± 1.727e+02 | **4.261e+05 ± 1.333e+04** | **1.285e+05 ± 4.992e+03** | 4.1364 |
| `R2C+CORAL` | torchregress | 0.9740 ± 0.0004 | 5.993e+05 ± 1.975e+03 | 6.844e+05 ± 6.627e+03 | 1.535e+05 ± 2.105e+03 | 13.6834 |
| `R2C+CumulativeLink` | torchregress | 0.9834 ± 0.0026 | 6.143e+05 ± 7.105e+03 | 6.980e+05 ± 6.785e+03 | 1.486e+05 ± 2.842e+03 | 10.8348 |
| `R2C+Softmax` | torchregress | 0.9917 ± 0.0008 | 6.499e+05 ± 9.704e+03 | 6.751e+05 ± 8.634e+03 | 1.640e+05 ± 2.023e+03 | 3.9636 |
| `Split+MLP` | torchregress | 0.8964 ± 0.0084 | 3.180e+05 ± 9.091e+03 | 6.137e+05 ± 1.181e+04 | 1.308e+05 ± 2.338e+03 | 3.0749 |
| `torchcp_R2C` | torchcp | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 4.697e+06 ± 0.000e+00 | 2.194e+06 ± 0.000e+00 | 4.2051 |

##### r2c_conformal / openml_year

| Method | Library | Coverage | Width | IntervalScore | RMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9076 ± 0.0062 | 0.5410 ± 0.0163 | **0.7080 ± 0.0176** | 0.1758 ± 0.0023 | 1.7892 |
| `R2C+CORAL` | torchregress | 0.9218 ± 0.0036 | 0.5408 ± 0.0122 | 0.7478 ± 0.0111 | 0.1726 ± 0.0034 | 6.1639 |
| `R2C+CumulativeLink` | torchregress | 0.9471 ± 0.0037 | 0.6026 ± 0.0167 | 0.7376 ± 0.0126 | 0.1870 ± 0.0049 | 5.5227 |
| `R2C+Softmax` | torchregress | 0.9471 ± 0.0032 | 0.6418 ± 0.0091 | 0.7881 ± 0.0006 | 0.2163 ± 0.0023 | 1.7480 |
| `Split+MLP` | torchregress | 0.9096 ± 0.0078 | 0.5173 ± 0.0184 | 0.7410 ± 0.0143 | **0.1625 ± 0.0034** | 1.2851 |
| `torchcp_R2C` | torchcp | 0.6020 ± 0.2879 | 0.9639 ± 0.3790 | 3.0270 ± 1.2323 † | 0.8216 ± 0.1237 | 1.8343 |

##### r2c_conformal / synthetic_hetero

| Method | Library | Coverage | Width | IntervalScore | RMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CQR+MLP` | torchregress | 0.9306 ± 0.0067 | 1.4484 ± 0.0485 | **1.6577 ± 0.0510** | 0.3978 ± 0.0086 | 0.3658 |
| `R2C+CORAL` | torchregress | 0.9278 ± 0.0170 | 1.5202 ± 0.0923 | 1.8120 ± 0.0257 | 0.4111 ± 0.0085 | 1.2416 |
| `R2C+CumulativeLink` | torchregress | 0.9611 ± 0.0010 | 1.9383 ± 0.0194 | 2.0832 ± 0.0080 | 0.4429 ± 0.0126 | 1.1290 |
| `R2C+Softmax` | torchregress | 0.9633 ± 0.0044 | 1.8779 ± 0.0500 | 2.0193 ± 0.0359 | 0.4606 ± 0.0060 | 0.3525 |
| `Split+MLP` | torchregress | 0.9256 ± 0.0111 | 1.3924 ± 0.0473 | 1.7088 ± 0.0363 | **0.3860 ± 0.0018** | 0.2636 |
| `torchcp_R2C` | torchcp | 0.9917 ± 0.0144 | 8.4732 ± 1.0810 | 8.6797 ± 0.7567 † | 3.5876 ± 1.0930 | 0.3753 |

#### robust_regression (tabular)

Group `tabular`; 3 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑, TailRMSE ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### robust_regression / california_housing

| Method | Library | RMSE | MAE | R2 | TailRMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `AdaptiveRobust` | torchregress | 0.5806 ± 0.0142 | 0.3763 ± 0.0033 | 0.7468 ± 0.0124 | 1.2331 ± 0.0749 | 3.7739 |
| `AdaptiveRobust_Linear` | torchregress | 0.8179 ± 0.0000 | 0.5165 ± 0.0000 | 0.4977 ± 0.0000 | 1.7407 ± 0.0001 | 0.0411 |
| `Barron` | torchregress | 0.5818 ± 0.0080 | 0.3794 ± 0.0065 | 0.7459 ± 0.0070 | 1.2366 ± 0.0525 | 2.4275 |
| `Barron_Linear` | torchregress | 0.8260 ± 0.0000 | 0.5173 ± 0.0000 | 0.4877 ± 0.0000 | 1.7162 ± 0.0000 | 0.0161 |
| `CVaR` | torchregress | 0.6400 ± 0.0081 | 0.4754 ± 0.0042 | 0.6925 ± 0.0078 | 1.2065 ± 0.0448 | 1.0252 |
| `CVaR_Linear` | torchregress | 0.8833 ± 0.0026 | 0.6983 ± 0.0017 | 0.4141 ± 0.0035 | 1.6230 ± 0.0074 | 0.0370 |
| `Cauchy` | torchregress | 0.5913 ± 0.0237 | 0.3757 ± 0.0046 | 0.7372 ± 0.0213 | 1.2916 ± 0.0886 | 1.6360 |
| `Cauchy_Linear` | torchregress | 0.8284 ± 0.0000 | 0.5067 ± 0.0000 | 0.4848 ± 0.0000 | 1.7038 ± 0.0000 | 0.0124 |
| `Charbonnier` | torchregress | 0.5956 ± 0.0207 | 0.3776 ± 0.0061 | 0.7335 ± 0.0187 | 1.3067 ± 0.0689 | 0.9306 |
| `Charbonnier_Linear` | torchregress | 0.8250 ± 0.0000 | 0.5084 ± 0.0000 | 0.4890 ± 0.0000 | 1.7372 ± 0.0001 | 0.0193 |
| `Expectile01` | torchregress | 0.7152 ± 0.0106 | 0.4728 ± 0.0150 | 0.6159 ± 0.0113 | 1.6717 ± 0.0073 | 0.9659 |
| `Expectile01_Linear` | torchregress | 0.9312 ± 0.0001 | 0.5954 ± 0.0000 | 0.3490 ± 0.0002 | 2.1719 ± 0.0004 | 0.0109 |
| `Expectile05` | torchregress | 0.5843 ± 0.0075 | 0.3900 ± 0.0102 | 0.7437 ± 0.0066 | 1.2094 ± 0.0635 | 0.9698 |
| `Expectile05_Linear` | torchregress | 0.7984 ± 0.0000 | 0.5357 ± 0.0000 | 0.5214 ± 0.0000 | 1.7575 ± 0.0000 | 0.0116 |
| `Expectile09` | torchregress | 0.7909 ± 0.0387 | 0.6119 ± 0.0528 | 0.5296 ± 0.0465 | **0.8820 ± 0.0676** | 0.9643 |
| `Expectile09_Linear` | torchregress | 1.0508 ± 0.0000 | 0.8860 ± 0.0000 | 0.1710 ± 0.0000 | 1.3880 ± 0.0000 | 0.0110 |
| `ExpectileCrossover` | torchregress | 0.6042 ± 0.0036 | 0.4091 ± 0.0041 | 0.7259 ± 0.0032 | 1.2761 ± 0.0298 | 2.1211 |
| `Huber` | torchregress | 0.5832 ± 0.0126 | 0.3826 ± 0.0075 | 0.7445 ± 0.0111 | 1.2351 ± 0.0689 | 1.5177 |
| `Huber_Linear` | torchregress | 0.8248 ± 0.0000 | 0.5176 ± 0.0000 | 0.4892 ± 0.0000 | 1.6912 ± 0.0001 | 0.0091 |
| `IRLS` | torchregress | 1.3286 ± 0.2246 † | 0.9870 ± 0.0910 | -0.3506 ± 0.4671 | 2.8444 ± 0.7003 | 0.0021 |
| `LightGBM` | LightGBM | 0.5761 ± 0.0069 | **0.3681 ± 0.0047** | 0.7508 ± 0.0060 | 1.3117 ± 0.0222 | 0.0845 |
| `LogCosh` | torchregress | 0.5810 ± 0.0083 | 0.3817 ± 0.0063 | 0.7465 ± 0.0072 | 1.2224 ± 0.0561 | 1.2949 |
| `LogCosh_Linear` | torchregress | 0.8274 ± 0.0000 | 0.5179 ± 0.0000 | 0.4859 ± 0.0000 | 1.7114 ± 0.0000 | 0.0127 |
| `MAE` | torchregress | 0.5945 ± 0.0190 | 0.3793 ± 0.0081 | 0.7345 ± 0.0171 | 1.2927 ± 0.0638 | 1.5460 |
| `MAE_Linear` | torchregress | 0.8248 ± 0.0005 | 0.5084 ± 0.0000 | 0.4893 ± 0.0006 | 1.7370 ± 0.0007 | 0.0284 |
| `MSE` | torchregress | 0.5843 ± 0.0075 | 0.3900 ± 0.0102 | 0.7437 ± 0.0066 | 1.2094 ± 0.0635 | 1.5206 |
| `MSE_Linear` | torchregress | 0.7984 ± 0.0000 | 0.5357 ± 0.0000 | 0.5214 ± 0.0000 | 1.7575 ± 0.0000 | 0.0182 |
| `MultiExpectile` | torchregress | 0.5980 ± 0.0053 | 0.3995 ± 0.0030 | 0.7315 ± 0.0048 | 1.2567 ± 0.0165 | 1.8163 |
| `PseudoHuber` | torchregress | 0.5817 ± 0.0080 | 0.3794 ± 0.0066 | 0.7459 ± 0.0070 | 1.2363 ± 0.0527 | 0.9982 |
| `PseudoHuber_Linear` | torchregress | 0.8260 ± 0.0000 | 0.5173 ± 0.0000 | 0.4877 ± 0.0000 | 1.7162 ± 0.0000 | 0.0121 |
| `Quantile05` | torchregress | 0.5961 ± 0.0187 | 0.3782 ± 0.0046 | 0.7330 ± 0.0169 | 1.3034 ± 0.0712 | 1.0104 |
| `Quantile05_Linear` | torchregress | 0.8249 ± 0.0007 | 0.5084 ± 0.0000 | 0.4891 ± 0.0009 | 1.7362 ± 0.0005 | 0.0349 |
| `QuantileCrossover` | torchregress | 0.6096 ± 0.0029 | 0.4030 ± 0.0021 | 0.7210 ± 0.0027 | 1.3291 ± 0.0201 | 1.8697 |
| `Quantile_Linear` | torchregress | 0.8249 ± 0.0007 | 0.5084 ± 0.0000 | 0.4891 ± 0.0009 | 1.7362 ± 0.0005 | 0.0245 |
| `Tukey` | torchregress | 0.5860 ± 0.0106 | 0.3852 ± 0.0068 | 0.7421 ± 0.0093 | 1.2324 ± 0.0772 | 1.1006 |
| `Tukey_Linear` | torchregress | 0.8800 ± 0.0000 | 0.5226 ± 0.0000 | 0.4186 ± 0.0000 | 1.6780 ± 0.0000 | 0.0133 |
| `XGBoost` | XGBoost | **0.5625 ± 0.0043** | 0.3710 ± 0.0045 | **0.7625 ± 0.0037** | 1.2080 ± 0.0206 | 0.0714 |
| `YeoJohnson` | torchregress | 0.5847 ± 0.0066 | 0.3900 ± 0.0102 | 0.7433 ± 0.0057 | 1.2118 ± 0.0598 | 1.6454 |
| `YeoJohnson_Linear` | torchregress | 0.7984 ± 0.0001 | 0.5357 ± 0.0000 | 0.5214 ± 0.0001 | 1.7572 ± 0.0004 | 0.0355 |
| `sklearn_QuantileRegressor` | sklearn | 0.8243 ± 0.0000 | 0.5084 ± 0.0000 | 0.4899 ± 0.0000 | 1.7354 ± 0.0000 | 0.8556 |
| `statsmodels_RLM` | statsmodels | 0.8260 ± 0.0000 | 0.5127 ± 0.0000 | 0.4878 ± 0.0000 | 1.7040 ± 0.0000 | 0.0334 |

##### robust_regression / synthetic_outliers

| Method | Library | RMSE | MAE | R2 | TailRMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `AdaptiveRobust` | torchregress | 1.2655 ± 0.0065 | 0.7154 ± 0.0037 | 0.7958 ± 0.0021 | 1.9467 ± 0.0388 | 1.4481 |
| `AdaptiveRobust_Linear` | torchregress | 32.6777 ± 43.5360 † | 27.1747 ± 36.4032 | -296.3 ± 502.1 | 38.4857 ± 51.9327 | 0.0434 |
| `Barron` | torchregress | 1.2571 ± 0.0092 | 0.7006 ± 0.0017 | 0.7985 ± 0.0029 | 1.9321 ± 0.0406 | 1.0173 |
| `Barron_Linear` | torchregress | 1.3857 ± 0.0000 | 0.8482 ± 0.0000 | 0.7552 ± 0.0000 | 2.2077 ± 0.0000 | 0.0065 |
| `CVaR` | torchregress | 1.3983 ± 0.0053 | 0.9369 ± 0.0064 | 0.7507 ± 0.0019 | 2.0676 ± 0.0439 | 0.5341 |
| `CVaR_Linear` | torchregress | 1.4455 ± 0.0007 | 0.9645 ± 0.0010 | 0.7336 ± 0.0002 | 2.0843 ± 0.0002 | 0.0234 |
| `Cauchy` | torchregress | 1.2425 ± 0.0068 | 0.6704 ± 0.0150 | 0.8032 ± 0.0021 | 1.9284 ± 0.0239 | 0.6849 |
| `Cauchy_Linear` | torchregress | 1.3879 ± 0.0000 | 0.8402 ± 0.0000 | 0.7544 ± 0.0000 | 2.2553 ± 0.0000 | 0.0061 |
| `Charbonnier` | torchregress | 1.2383 ± 0.0093 | 0.6666 ± 0.0211 | 0.8045 ± 0.0029 | 1.9138 ± 0.0358 | 0.5776 |
| `Charbonnier_Linear` | torchregress | 1.3974 ± 0.0000 | 0.8438 ± 0.0000 | 0.7510 ± 0.0000 | 2.3114 ± 0.0000 | 0.0096 |
| `Expectile01` | torchregress | 1.5729 ± 0.0416 | 1.1822 ± 0.0602 | 0.6844 ± 0.0166 | 2.5030 ± 0.0933 | 0.5106 |
| `Expectile01_Linear` | torchregress | 1.7184 ± 0.0000 | 1.2947 ± 0.0000 | 0.6235 ± 0.0000 | 2.9093 ± 0.0000 | 0.0053 |
| `Expectile05` | torchregress | 1.2655 ± 0.0092 | 0.7159 ± 0.0089 | 0.7958 ± 0.0030 | 1.9522 ± 0.0347 | 0.5108 |
| `Expectile05_Linear` | torchregress | 1.3885 ± 0.0000 | 0.8542 ± 0.0000 | 0.7542 ± 0.0000 | 2.1995 ± 0.0000 | 0.0042 |
| `Expectile09` | torchregress | 1.5637 ± 0.0275 | 1.1538 ± 0.0331 | 0.6882 ± 0.0110 | **1.6977 ± 0.0067** | 0.5059 |
| `Expectile09_Linear` | torchregress | 1.8157 ± 0.0000 | 1.4976 ± 0.0000 | 0.5797 ± 0.0000 | 1.7383 ± 0.0000 | 0.0057 |
| `ExpectileCrossover` | torchregress | 1.2494 ± 0.0061 | 0.6964 ± 0.0068 | 0.8010 ± 0.0020 | 1.9419 ± 0.0194 | 0.9023 |
| `Huber` | torchregress | 1.2592 ± 0.0098 | 0.7057 ± 0.0075 | 0.7978 ± 0.0032 | 1.9414 ± 0.0318 | 0.6381 |
| `Huber_Linear` | torchregress | 1.3855 ± 0.0000 | 0.8519 ± 0.0000 | 0.7553 ± 0.0000 | 2.1935 ± 0.0000 | 0.0038 |
| `IRLS` | torchregress | 2.8930 ± 0.4370 † | 2.3157 ± 0.3575 | -0.0833 ± 0.3320 | 4.5840 ± 0.8450 | 0.0019 |
| `LightGBM` | LightGBM | 1.2633 ± 0.0019 | 0.7188 ± 0.0008 | 0.7965 ± 0.0006 | 2.1124 ± 0.0443 | 0.0883 |
| `LogCosh` | torchregress | 1.2567 ± 0.0090 | 0.7004 ± 0.0056 | 0.7986 ± 0.0029 | 1.9342 ± 0.0454 | 0.6323 |
| `LogCosh_Linear` | torchregress | 1.3858 ± 0.0000 | 0.8492 ± 0.0000 | 0.7552 ± 0.0000 | 2.2047 ± 0.0000 | 0.0057 |
| `MAE` | torchregress | 1.2399 ± 0.0080 | 0.6671 ± 0.0201 | 0.8040 ± 0.0025 | 1.9194 ± 0.0301 | 0.6464 |
| `MAE_Linear` | torchregress | 1.3973 ± 0.0000 | 0.8438 ± 0.0000 | 0.7511 ± 0.0000 | 2.3114 ± 0.0001 | 0.0171 |
| `MSE` | torchregress | 1.2655 ± 0.0092 | 0.7159 ± 0.0089 | 0.7958 ± 0.0030 | 1.9522 ± 0.0347 | 0.6299 |
| `MSE_Linear` | torchregress | 1.3885 ± 0.0000 | 0.8542 ± 0.0000 | 0.7542 ± 0.0000 | 2.1995 ± 0.0000 | 0.0129 |
| `MultiExpectile` | torchregress | 1.2603 ± 0.0066 | 0.7142 ± 0.0041 | 0.7975 ± 0.0021 | 1.9378 ± 0.0407 | 0.7765 |
| `PseudoHuber` | torchregress | 1.2571 ± 0.0092 | 0.7006 ± 0.0017 | 0.7985 ± 0.0029 | 1.9321 ± 0.0406 | 0.5204 |
| `PseudoHuber_Linear` | torchregress | 1.3857 ± 0.0000 | 0.8482 ± 0.0000 | 0.7552 ± 0.0000 | 2.2077 ± 0.0000 | 0.0044 |
| `Quantile05` | torchregress | 1.2377 ± 0.0083 | **0.6658 ± 0.0215** | 0.8047 ± 0.0026 | 1.9215 ± 0.0245 | 0.5323 |
| `Quantile05_Linear` | torchregress | 1.3974 ± 0.0001 | 0.8439 ± 0.0001 | 0.7510 ± 0.0000 | 2.3114 ± 0.0001 | 0.0153 |
| `QuantileCrossover` | torchregress | **1.2363 ± 0.0079** | 0.6662 ± 0.0048 | **0.8051 ± 0.0025** | 1.9374 ± 0.0629 | 0.8065 |
| `Quantile_Linear` | torchregress | 1.3974 ± 0.0001 | 0.8439 ± 0.0001 | 0.7510 ± 0.0000 | 2.3114 ± 0.0001 | 0.0131 |
| `Tukey` | torchregress | 1.2615 ± 0.0085 | 0.7090 ± 0.0088 | 0.7971 ± 0.0028 | 1.9453 ± 0.0356 | 0.6202 |
| `Tukey_Linear` | torchregress | 1.3868 ± 0.0000 | 0.8519 ± 0.0000 | 0.7548 ± 0.0000 | 2.2002 ± 0.0000 | 0.0060 |
| `XGBoost` | XGBoost | 1.3163 ± 0.0072 | 0.8000 ± 0.0212 | 0.7791 ± 0.0024 | 2.1615 ± 0.0570 | 0.0796 |
| `YeoJohnson` | torchregress | 1.2659 ± 0.0088 | 0.7158 ± 0.0087 | 0.7957 ± 0.0029 | 1.9528 ± 0.0354 | 0.8641 |
| `YeoJohnson_Linear` | torchregress | 1.3885 ± 0.0000 | 0.8542 ± 0.0000 | 0.7542 ± 0.0000 | 2.1995 ± 0.0000 | 0.0071 |
| `sklearn_QuantileRegressor` | sklearn | 1.3975 ± 0.0000 | 0.8440 ± 0.0000 | 0.7510 ± 0.0000 | 2.3115 ± 0.0000 | 0.3130 |
| `statsmodels_RLM` | statsmodels | 1.3873 ± 0.0000 | 0.8415 ± 0.0000 | 0.7546 ± 0.0000 | 2.2441 ± 0.0000 | 0.0153 |

##### robust_regression / zurich_delays

| Method | Library | RMSE | MAE | R2 | TailRMSE | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `AdaptiveRobust` | torchregress | 249.2 ± 38.3 † | 190.7 ± 33.0 | -3.2614 ± 1.3445 | 407.9 ± 36.8 | 58.6582 |
| `AdaptiveRobust_Linear` | torchregress | 2529.9 ± 1715.8 † | 1190.1 ± 601.6 | -563.8 ± 569.6 | 6473.3 ± 5607.4 | 0.0839 |
| `Barron` | torchregress | 109.4 ± 0.2 | 58.6378 ± 0.0675 | 0.1917 ± 0.0025 | 257.4 ± 1.6 | 42.0566 |
| `Barron_Linear` | torchregress | 118.6 ± 0.0 | 60.2729 ± 0.0022 | 0.0503 ± 0.0001 | 311.4 ± 0.0 | 0.0729 |
| `CVaR` | torchregress | 121.4 ± 5.4 | 72.4389 ± 1.2536 | 0.0033 ± 0.0872 | 242.2 ± 3.4 | 29.8859 |
| `CVaR_Linear` | torchregress | 128.5 ± 0.1 | 87.1637 ± 0.1144 | -0.1148 ± 0.0010 | 262.2 ± 0.1 | 0.2323 |
| `Cauchy` | torchregress | 115.7 ± 0.4 | 58.4452 ± 0.0808 | 0.0966 ± 0.0064 | 298.3 ± 1.2 | 30.3825 |
| `Cauchy_Linear` | torchregress | 121.2 ± 0.0 | 59.0701 ± 0.0009 | 0.0083 ± 0.0000 | 325.9 ± 0.0 | 0.0910 |
| `Charbonnier` | torchregress | 110.6 ± 1.4 | 57.7380 ± 0.2543 | 0.1734 ± 0.0203 | 273.6 ± 6.9 | 27.5943 |
| `Charbonnier_Linear` | torchregress | 121.1 ± 0.0 | 59.0512 ± 0.0025 | 0.0098 ± 0.0002 | 325.5 ± 0.0 | 0.0847 |
| `Expectile01` | torchregress | 141.2 ± 21.5 † | 67.7638 ± 2.0294 | -0.3671 ± 0.4226 | 325.8 ± 9.9 | 28.5674 |
| `Expectile01_Linear` | torchregress | 131.3 ± 0.0 | 68.9218 ± 0.0036 | -0.1636 ± 0.0000 | 348.9 ± 0.0 | 0.0622 |
| `Expectile05` | torchregress | 114.0 ± 1.4 | 60.3195 ± 0.1307 | 0.1223 ± 0.0215 | 254.6 ± 3.0 | 27.4971 |
| `Expectile05_Linear` | torchregress | 117.6 ± 0.0 | 63.3154 ± 0.0010 | 0.0662 ± 0.0000 | 296.6 ± 0.0 | 0.0568 |
| `Expectile09` | torchregress | 147.5 ± 4.9 | 95.9170 ± 3.5754 | -0.4712 ± 0.0975 | **211.9 ± 7.5** | 30.4029 |
| `Expectile09_Linear` | torchregress | 161.3 ± 0.0 | 122.7 ± 0.0 | -0.7562 ± 0.0001 | 218.3 ± 0.0 | 0.0799 |
| `ExpectileCrossover` | torchregress | 113.8 ± 3.6 | 61.4892 ± 0.4523 | 0.1247 ± 0.0551 | 249.7 ± 2.8 | 45.1033 |
| `Huber` | torchregress | 109.6 ± 0.9 | 58.9955 ± 0.3191 | 0.1891 ± 0.0138 | 256.8 ± 3.5 | 27.0527 |
| `Huber_Linear` | torchregress | 118.5 ± 0.0 | 60.5592 ± 0.0022 | 0.0521 ± 0.0000 | 310.2 ± 0.0 | 0.0392 |
| `IRLS` | torchregress | 152.3 ± 5.1 | 98.8364 ± 5.3284 | -0.5671 ± 0.1054 | 333.4 ± 12.5 | 0.0135 |
| `LightGBM` | LightGBM | 115.5 ± 0.2 | **56.7611 ± 0.0482** | 0.0991 ± 0.0028 | 305.1 ± 0.8 | 0.4674 |
| `LogCosh` | torchregress | **109.2 ± 1.2** | 58.6916 ± 0.3303 | **0.1946 ± 0.0173** | 257.5 ± 3.9 | 29.7495 |
| `LogCosh_Linear` | torchregress | 118.5 ± 0.0 | 60.3809 ± 0.0021 | 0.0514 ± 0.0000 | 310.8 ± 0.0 | 0.0504 |
| `MAE` | torchregress | 110.7 ± 0.8 | 57.6267 ± 0.0678 | 0.1723 ± 0.0118 | 274.9 ± 5.5 | 27.3884 |
| `MAE_Linear` | torchregress | 121.1 ± 0.0 | 59.0513 ± 0.0027 | 0.0097 ± 0.0002 | 325.5 ± 0.0 | 0.0847 |
| `MSE` | torchregress | 114.0 ± 1.4 | 60.3195 ± 0.1307 | 0.1223 ± 0.0215 | 254.6 ± 3.0 | 29.9683 |
| `MSE_Linear` | torchregress | 117.6 ± 0.0 | 63.3154 ± 0.0010 | 0.0662 ± 0.0000 | 296.6 ± 0.0 | 0.0582 |
| `MultiExpectile` | torchregress | 113.6 ± 2.4 | 61.0001 ± 0.3102 | 0.1288 ± 0.0374 | 251.1 ± 4.0 | 35.5413 |
| `PseudoHuber` | torchregress | 109.8 ± 0.5 | 58.6305 ± 0.2019 | 0.1851 ± 0.0079 | 258.7 ± 2.1 | 28.9780 |
| `PseudoHuber_Linear` | torchregress | 118.6 ± 0.0 | 60.2731 ± 0.0020 | 0.0503 ± 0.0001 | 311.4 ± 0.0 | 0.0468 |
| `Quantile05` | torchregress | 110.7 ± 0.8 | 57.7519 ± 0.2189 | 0.1719 ± 0.0118 | 273.8 ± 6.8 | 29.4074 |
| `Quantile05_Linear` | torchregress | 121.1 ± 0.0 | 59.0513 ± 0.0027 | 0.0097 ± 0.0002 | 325.5 ± 0.0 | 0.1402 |
| `QuantileCrossover` | torchregress | 110.2 ± 0.8 | 57.6412 ± 0.0922 | 0.1801 ± 0.0122 | 265.8 ± 2.0 | 35.1872 |
| `Quantile_Linear` | torchregress | 121.1 ± 0.0 | 59.0513 ± 0.0027 | 0.0097 ± 0.0002 | 325.5 ± 0.0 | 0.0724 |
| `Tukey` | torchregress | 111.3 ± 0.1 | 59.0956 ± 0.0275 | 0.1629 ± 0.0021 | 269.4 ± 1.2 | 31.4040 |
| `Tukey_Linear` | torchregress | 119.2 ± 1.0 | 60.6649 ± 0.0582 | 0.0399 ± 0.0169 | 312.9 ± 4.0 | 0.0550 |
| `XGBoost` | XGBoost | 110.8 ± 0.1 | 57.5202 ± 0.0470 | 0.1706 ± 0.0017 | 282.0 ± 0.7 | 0.4579 |
| `YeoJohnson` | torchregress | 112.6 ± 4.5 | 60.4682 ± 0.7122 | 0.1421 ± 0.0693 | 254.7 ± 2.2 | 50.8941 |
| `YeoJohnson_Linear` | torchregress | 117.6 ± 0.0 | 63.3155 ± 0.0011 | 0.0662 ± 0.0000 | 296.6 ± 0.0 | 0.3370 |
| `sklearn_QuantileRegressor` | sklearn | 121.1 ± 0.0 | 59.0482 ± 0.0000 | 0.0099 ± 0.0000 | 325.4 ± 0.0 | 639.4 |
| `statsmodels_RLM` | statsmodels | 119.7 ± 0.0 | 59.4249 ± 0.0000 | 0.0329 ± 0.0000 | 318.6 ± 0.0 | 2.3954 |

#### semi_supervised (tabular)

Group `tabular`; 2 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### semi_supervised / diabetes

| Method | Library | RMSE | MAE | R2 | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `MLP_labeled` | scikit-learn | 70.3641 ± 3.7772 † | 56.8060 ± 2.1669 | 0.0811 ± 0.0999 | 0.0448 |
| `Oracle_FullyLabeled` | torchregress | 56.9987 ± 1.0273 | **43.5000 ± 0.9391** | 0.3980 ± 0.0217 | 0.4728 |
| `Ridge_labeled` | scikit-learn | **55.4803 ± 0.3236** | 44.4990 ± 0.6790 | **0.4298 ± 0.0067** | 0.0041 |
| `SAGERegLoss_SelfAgreement` | torchregress | 66.5572 ± 3.1797 | 53.2770 ± 2.5263 | 0.1782 ± 0.0773 | 4.7263 |
| `SelfTraining_RF` | scikit-learn | 59.8417 ± 3.7694 † | 47.2458 ± 3.6189 | 0.3349 ± 0.0828 | 0.3752 |
| `Supervised_GaussianNLL` | torchregress | 68.4292 ± 3.9567 † | 55.0502 ± 3.9648 | 0.1307 ± 0.0989 | 0.5884 |
| `TeacherStudent_BatchRelative` | torchregress | 67.2052 ± 4.0598 † | 53.8011 ± 3.0058 | 0.1613 ± 0.0999 | 3.1454 |
| `TeacherStudent_SAGEReg` | torchregress | 66.6575 ± 3.8605 † | 53.2445 ± 2.6537 | 0.1751 ± 0.0942 | 3.7087 |
| `TeacherStudent_UncertaintyWeight` | torchregress | 67.1834 ± 4.4469 † | 53.8374 ± 3.4373 | 0.1614 ± 0.1089 | 2.4285 |

##### semi_supervised / ssl_friedman

| Method | Library | RMSE | MAE | R2 | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `MLP_labeled` | scikit-learn | 0.5135 ± 0.0361 † | 0.3997 ± 0.0240 | 0.7410 ± 0.0363 | 0.0505 |
| `Oracle_FullyLabeled` | torchregress | **0.2546 ± 0.0097** | **0.2011 ± 0.0066** | **0.9364 ± 0.0065** | 0.4945 |
| `Ridge_labeled` | scikit-learn | 0.5461 ± 0.0091 | 0.4243 ± 0.0025 | 0.7078 ± 0.0183 | 0.0116 |
| `SAGERegLoss_SelfAgreement` | torchregress | 0.4792 ± 0.0475 † | 0.3709 ± 0.0338 | 0.7736 ± 0.0455 | 4.8345 |
| `SelfTraining_RF` | scikit-learn | 0.6221 ± 0.0193 | 0.5001 ± 0.0165 | 0.6214 ± 0.0131 | 1.0417 |
| `Supervised_GaussianNLL` | torchregress | 0.4698 ± 0.0461 † | 0.3622 ± 0.0283 | 0.7822 ± 0.0458 | 0.6949 |
| `TeacherStudent_BatchRelative` | torchregress | 0.4756 ± 0.0437 † | 0.3676 ± 0.0274 | 0.7770 ± 0.0444 | 3.1789 |
| `TeacherStudent_SAGEReg` | torchregress | 0.4766 ± 0.0433 † | 0.3681 ± 0.0287 | 0.7762 ± 0.0434 | 3.9560 |
| `TeacherStudent_UncertaintyWeight` | torchregress | 0.4762 ± 0.0439 † | 0.3679 ± 0.0271 | 0.7765 ± 0.0439 | 2.5116 |

#### sklearn_baselines (tabular)

Group `tabular`; 17 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑, NLL ↓, Coverage ·, IntervalScore ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### sklearn_baselines / california_housing

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.6010 ± 0.0059 | 0.4003 ± 0.0041 | 0.7288 ± 0.0054 | 1.5526 ± 1.1115 | - | - | 1.9533 |
| `GaussianNLL_Linear` | torchregress | 0.7984 ± 0.0000 | 0.5357 ± 0.0000 | 0.5214 ± 0.0000 | **1.2088 ± 0.0000** | - | - | 0.0149 |
| `MSE` | torchregress | 0.5843 ± 0.0075 | 0.3900 ± 0.0102 | 0.7437 ± 0.0066 | - | - | - | 1.2512 |
| `MultiQuantile` | torchregress | 0.5970 ± 0.0067 | 0.3872 ± 0.0028 | 0.7323 ± 0.0060 | - | 0.7749 ± 0.0337 | 2.5811 ± 0.1475 | 1.7477 |
| `OLS` | torchregress | 0.7984 ± 0.0000 | 0.5357 ± 0.0000 | 0.5214 ± 0.0000 | - | - | - | 0.0034 |
| `TabularMLP` | torchregress | 0.5797 ± 0.0137 | 0.3900 ± 0.0172 | 0.7476 ± 0.0118 | - | - | - | 2.3430 |
| `TabularMLP_Ens5` | torchregress | 0.5681 ± 0.0015 | 0.3779 ± 0.0012 | 0.7577 ± 0.0012 | - | - | - | 12.9285 |
| `sklearn_AdaBoost` | sklearn | 0.7897 ± 0.0061 | 0.6499 ± 0.0072 | 0.5317 ± 0.0073 | - | - | - | 0.2565 |
| `sklearn_BayesianRidge` | sklearn | 0.7980 ± 0.0000 | 0.5357 ± 0.0000 | 0.5219 ± 0.0000 | - | - | - | 0.0018 |
| `sklearn_DecisionTree` | sklearn | 0.7653 ± 0.0083 | 0.4988 ± 0.0037 | 0.5602 ± 0.0095 | - | - | - | 0.0150 |
| `sklearn_ElasticNet` | sklearn | 0.8201 ± 0.0000 | 0.6115 ± 0.0000 | 0.4950 ± 0.0000 | - | - | - | 0.0012 |
| `sklearn_ExtraTrees` | sklearn | 0.6318 ± 0.0051 | 0.4525 ± 0.0044 | 0.7002 ± 0.0049 | - | - | - | 0.2151 |
| `sklearn_GradientBoosting` | sklearn | 0.5557 ± 0.0002 | 0.3715 ± 0.0002 | 0.7682 ± 0.0002 | - | - | - | 0.9629 |
| `sklearn_HistGradientBoosting` | sklearn | **0.5453 ± 0.0000** | **0.3548 ± 0.0000** | **0.7767 ± 0.0000** | - | - | - | 0.2574 |
| `sklearn_HuberRegressor` | sklearn | 0.8295 ± 0.0000 | 0.5109 ± 0.0000 | 0.4834 ± 0.0000 | - | - | - | 0.0113 |
| `sklearn_KNeighbors` | sklearn | 0.6524 ± 0.0000 | 0.4394 ± 0.0000 | 0.6804 ± 0.0000 | - | - | - | 0.0031 |
| `sklearn_Lasso` | sklearn | 0.8461 ± 0.0000 | 0.6401 ± 0.0000 | 0.4624 ± 0.0000 | - | - | - | 0.0018 |
| `sklearn_LinearRegression` | sklearn | 0.7984 ± 0.0000 | 0.5357 ± 0.0000 | 0.5214 ± 0.0000 | - | - | - | 0.0041 |
| `sklearn_MLPRegressor` | sklearn | 0.5941 ± 0.0100 | 0.4032 ± 0.0101 | 0.7349 ± 0.0089 | - | - | - | 0.9325 |
| `sklearn_RandomForest` | sklearn | 0.5943 ± 0.0024 | 0.4048 ± 0.0020 | 0.7348 ± 0.0021 | - | - | - | 0.4506 |
| `sklearn_Ridge` | sklearn | 0.7982 ± 0.0000 | 0.5357 ± 0.0000 | 0.5217 ± 0.0000 | - | - | - | 0.0018 |
| `sklearn_SVR` | sklearn | 0.6030 ± 0.0000 | 0.4008 ± 0.0000 | 0.7270 ± 0.0000 | - | - | - | 0.3145 |

##### sklearn_baselines / diabetes

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 52.6474 ± 0.6842 | 41.3762 ± 0.3207 | 0.4865 ± 0.0133 | 5.3964 ± 0.0313 | - | - | 0.1865 |
| `GaussianNLL_Linear` | torchregress | 53.1200 ± 0.0004 | 41.9194 ± 0.0002 | 0.4773 ± 0.0000 | **5.3919 ± 0.0000** | - | - | 0.0122 |
| `MSE` | torchregress | 53.1870 ± 1.2100 | 41.8366 ± 0.9939 | 0.4758 ± 0.0239 | - | - | - | 0.1167 |
| `MultiQuantile` | torchregress | 53.4410 ± 0.7054 | 42.2517 ± 0.3136 | 0.4709 ± 0.0139 | - | 0.7744 ± 0.0130 | 232.8 ± 6.9 | 0.1629 |
| `OLS` | torchregress | 53.1202 ± 0.0000 | 41.9194 ± 0.0000 | 0.4773 ± 0.0000 | - | - | - | 0.0034 |
| `TabularMLP` | torchregress | 54.2005 ± 1.4958 | 42.1905 ± 1.6724 | 0.4555 ± 0.0298 | - | - | - | 0.3097 |
| `TabularMLP_Ens5` | torchregress | 52.7656 ± 0.3636 | 41.2077 ± 0.3689 | 0.4842 ± 0.0071 | - | - | - | 1.3734 |
| `sklearn_AdaBoost` | sklearn | 53.1622 ± 0.5816 | 42.7629 ± 0.3040 | 0.4764 ± 0.0115 | - | - | - | 0.0911 |
| `sklearn_BayesianRidge` | sklearn | 53.0561 ± 0.0000 | 41.8500 ± 0.0000 | 0.4785 ± 0.0000 | - | - | - | 0.0015 |
| `sklearn_DecisionTree` | sklearn | 74.0251 ± 0.5918 | 57.8520 ± 0.6152 | -0.0151 ± 0.0163 | - | - | - | 0.0024 |
| `sklearn_ElasticNet` | sklearn | 52.9443 ± 0.0000 | 42.2165 ± 0.0000 | 0.4807 ± 0.0000 | - | - | - | 0.0012 |
| `sklearn_ExtraTrees` | sklearn | **52.0420 ± 0.1683** | 42.1826 ± 0.5028 | **0.4983 ± 0.0032** | - | - | - | 0.2228 |
| `sklearn_GradientBoosting` | sklearn | 58.4395 ± 0.2014 | 45.6604 ± 0.1242 | 0.3674 ± 0.0044 | - | - | - | 0.1809 |
| `sklearn_HistGradientBoosting` | sklearn | 56.7700 ± 0.0000 | 44.2934 ± 0.0000 | 0.4030 ± 0.0000 | - | - | - | 0.1364 |
| `sklearn_HuberRegressor` | sklearn | 53.1864 ± 0.0000 | 41.9377 ± 0.0000 | 0.4760 ± 0.0000 | - | - | - | 0.0046 |
| `sklearn_KNeighbors` | sklearn | 56.9648 ± 0.0000 | 44.2760 ± 0.0000 | 0.3989 ± 0.0000 | - | - | - | 0.0013 |
| `sklearn_Lasso` | sklearn | 53.6755 ± 0.0000 | 43.2417 ± 0.0000 | 0.4663 ± 0.0000 | - | - | - | 0.0014 |
| `sklearn_LinearRegression` | sklearn | 53.1202 ± 0.0000 | 41.9194 ± 0.0000 | 0.4773 ± 0.0000 | - | - | - | 0.0021 |
| `sklearn_MLPRegressor` | sklearn | 53.8030 ± 1.3996 | 42.4968 ± 1.4706 | 0.4635 ± 0.0279 | - | - | - | 0.1613 |
| `sklearn_RandomForest` | sklearn | 53.5034 ± 0.1177 | 43.0442 ± 0.0921 | 0.4697 ± 0.0023 | - | - | - | 0.2964 |
| `sklearn_Ridge` | sklearn | 53.1035 ± 0.0000 | 41.8784 ± 0.0000 | 0.4776 ± 0.0000 | - | - | - | 0.0015 |
| `sklearn_SVR` | sklearn | 52.5452 ± 0.0000 | **40.5205 ± 0.0000** | 0.4885 ± 0.0000 | - | - | - | 0.0068 |

##### sklearn_baselines / house_sales

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 1.263e+05 ± 2.120e+03 | 70706.6 ± 916.2 | 0.8827 ± 0.0040 | **12.7320 ± 0.0521** | - | - | 7.0358 |
| `GaussianNLL_Linear` | torchregress | 2.075e+05 ± 1.654e-01 | 1.282e+05 ± 4.705e-01 | 0.6834 ± 0.0000 | 13.6630 ± 0.0000 | - | - | 0.0130 |
| `MSE` | torchregress | 1.302e+05 ± 2.658e+03 | 72099.2 ± 656.2 | 0.8753 ± 0.0051 | - | - | - | 4.1915 |
| `MultiQuantile` | torchregress | 1.235e+05 ± 2.413e+03 | 68157.9 ± 441.3 | 0.8879 ± 0.0044 | - | 0.7436 ± 0.0067 | 4.888e+05 ± 3.719e+03 | 6.7658 |
| `OLS` | torchregress | 2.075e+05 ± 4.485e-03 | 1.282e+05 ± 8.980e-03 | 0.6834 ± 0.0000 | - | - | - | 0.0049 |
| `TabularMLP` | torchregress | 1.195e+05 ± 1.355e+03 | 67735.0 ± 702.3 | 0.8950 ± 0.0024 | - | - | - | 9.4516 |
| `TabularMLP_Ens5` | torchregress | **1.133e+05 ± 9.053e+01** | **65277.4 ± 287.2** | **0.9056 ± 0.0002** | - | - | - | 43.1992 |
| `sklearn_AdaBoost` | sklearn | 2.540e+05 ± 4.178e+03 | 1.998e+05 ± 6.192e+03 | 0.5256 ± 0.0156 | - | - | - | 1.4587 |
| `sklearn_BayesianRidge` | sklearn | 2.075e+05 ± 0.000e+00 | 1.282e+05 ± 0.000e+00 | 0.6835 ± 0.0000 | - | - | - | 0.0033 |
| `sklearn_DecisionTree` | sklearn | 1.821e+05 ± 1.656e+03 | 94121.5 ± 138.6 | 0.7562 ± 0.0044 | - | - | - | 0.0638 |
| `sklearn_ElasticNet` | sklearn | 2.159e+05 ± 0.000e+00 | 1.273e+05 ± 0.000e+00 | 0.6573 ± 0.0000 | - | - | - | 0.0043 |
| `sklearn_ExtraTrees` | sklearn | 1.558e+05 ± 3.529e+02 | 83975.2 ± 225.6 | 0.8217 ± 0.0008 | - | - | - | 0.6665 |
| `sklearn_GradientBoosting` | sklearn | 1.311e+05 ± 3.640e+02 | 72536.5 ± 36.0 | 0.8737 ± 0.0007 | - | - | - | 4.1768 |
| `sklearn_HistGradientBoosting` | sklearn | 1.254e+05 ± 3.322e+03 | 69743.4 ± 2092.1 | 0.8844 ± 0.0061 | - | - | - | 0.2860 |
| `sklearn_HuberRegressor` | sklearn | 2.177e+05 ± 0.000e+00 | 1.185e+05 ± 0.000e+00 | 0.6516 ± 0.0000 | - | - | - | 0.0354 |
| `sklearn_KNeighbors` | sklearn | 1.855e+05 ± 0.000e+00 | 1.001e+05 ± 0.000e+00 | 0.7471 ± 0.0000 | - | - | - | 0.0014 |
| `sklearn_Lasso` | sklearn | 2.259e+05 ± 0.000e+00 | 1.335e+05 ± 0.000e+00 | 0.6249 ± 0.0000 | - | - | - | 0.0029 |
| `sklearn_LinearRegression` | sklearn | 2.075e+05 ± 0.000e+00 | 1.282e+05 ± 0.000e+00 | 0.6834 ± 0.0000 | - | - | - | 0.0036 |
| `sklearn_MLPRegressor` | sklearn | 1.353e+05 ± 8.351e+02 | 75752.8 ± 1429.9 | 0.8655 ± 0.0017 | - | - | - | 5.2480 |
| `sklearn_RandomForest` | sklearn | 1.406e+05 ± 6.423e+02 | 76168.0 ± 249.5 | 0.8547 ± 0.0013 | - | - | - | 1.1806 |
| `sklearn_Ridge` | sklearn | 2.075e+05 ± 0.000e+00 | 1.282e+05 ± 0.000e+00 | 0.6834 ± 0.0000 | - | - | - | 0.0018 |
| `sklearn_SVR` | sklearn | 1.848e+05 ± 0.000e+00 | 88621.5 ± 0.0 | 0.7490 ± 0.0000 | - | - | - | 4.0307 |

##### sklearn_baselines / insurance_claims

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.2408 ± 0.0004 | 0.0792 ± 0.0032 | -0.0328 ± 0.0033 | 165.0 ± 78.7 | - | - | 6.2944 |
| `GaussianNLL_Linear` | torchregress | 0.2362 ± 0.0000 | 0.0983 ± 0.0000 | 0.0063 ± 0.0000 | **-0.0235 ± 0.0000** | - | - | 0.0086 |
| `MSE` | torchregress | 0.2695 ± 0.0018 | 0.1306 ± 0.0022 | -0.2938 ± 0.0171 | - | - | - | 4.3058 |
| `MultiQuantile` | torchregress | 0.2431 ± 0.0007 | 0.0544 ± 0.0001 | -0.0525 ± 0.0058 | - | 0.9342 ± 0.0138 | 1.0542 ± 0.0035 | 5.9148 |
| `OLS` | torchregress | 0.2407 ± 0.0039 | 0.0714 ± 0.0233 | -0.0318 ± 0.0330 | - | - | - | 0.0067 |
| `TabularMLP` | torchregress | 0.2366 ± 0.0009 | 0.1013 ± 0.0007 | 0.0035 ± 0.0072 | - | - | - | 8.8391 |
| `TabularMLP_Ens5` | torchregress | 0.2358 ± 0.0001 | 0.0995 ± 0.0008 | 0.0101 ± 0.0007 | - | - | - | 40.1766 |
| `sklearn_AdaBoost` | sklearn | 0.3122 ± 0.0185 † | 0.2757 ± 0.0270 | -0.7394 ± 0.2025 | - | - | - | 0.4566 |
| `sklearn_BayesianRidge` | sklearn | 0.2363 ± 0.0000 | 0.0986 ± 0.0000 | 0.0058 ± 0.0000 | - | - | - | 0.0069 |
| `sklearn_DecisionTree` | sklearn | 0.2636 ± 0.0014 | 0.1005 ± 0.0006 | -0.2372 ± 0.0129 | - | - | - | 0.0333 |
| `sklearn_ElasticNet` | sklearn | 0.2370 ± 0.0000 | 0.0992 ± 0.0000 | -9.691e-05 ± 0.000e+00 | - | - | - | 0.0021 |
| `sklearn_ExtraTrees` | sklearn | 0.2382 ± 0.0003 | 0.0997 ± 0.0000 | -0.0106 ± 0.0025 | - | - | - | 0.9581 |
| `sklearn_GradientBoosting` | sklearn | 0.2371 ± 0.0001 | 0.0987 ± 0.0000 | -0.0013 ± 0.0007 | - | - | - | 2.2083 |
| `sklearn_HistGradientBoosting` | sklearn | **0.2355 ± 0.0002** | 0.0979 ± 0.0012 | **0.0121 ± 0.0013** | - | - | - | 0.1651 |
| `sklearn_HuberRegressor` | sklearn | 0.2429 ± 0.0000 | **0.0533 ± 0.0000** | -0.0507 ± 0.0000 | - | - | - | 0.2519 |
| `sklearn_KNeighbors` | sklearn | 0.2599 ± 0.0000 | 0.0953 ± 0.0000 | -0.2029 ± 0.0000 | - | - | - | 0.0012 |
| `sklearn_Lasso` | sklearn | 0.2370 ± 0.0000 | 0.0992 ± 0.0000 | -9.691e-05 ± 0.000e+00 | - | - | - | 0.0023 |
| `sklearn_LinearRegression` | sklearn | 0.2362 ± 0.0000 | 0.0983 ± 0.0000 | 0.0061 ± 0.0000 | - | - | - | 0.0056 |
| `sklearn_MLPRegressor` | sklearn | 0.2657 ± 0.0031 | 0.1365 ± 0.0048 | -0.2575 ± 0.0291 | - | - | - | 4.8846 |
| `sklearn_RandomForest` | sklearn | 0.2377 ± 0.0003 | 0.0989 ± 0.0002 | -0.0061 ± 0.0028 | - | - | - | 0.7707 |
| `sklearn_Ridge` | sklearn | 0.2362 ± 0.0000 | 0.0983 ± 0.0000 | 0.0063 ± 0.0000 | - | - | - | 0.0025 |
| `sklearn_SVR` | sklearn | 0.2391 ± 0.0000 | 0.0723 ± 0.0000 | -0.0179 ± 0.0000 | - | - | - | 1.7070 |

##### sklearn_baselines / make_regression_synthetic

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 16.3460 ± 0.2297 | 12.9789 ± 0.2033 | 0.9865 ± 0.0004 | 4.2957 ± 0.0374 | - | - | 1.7306 |
| `GaussianNLL_Linear` | torchregress | 14.9186 ± 0.0000 | 11.9373 ± 0.0000 | 0.9887 ± 0.0000 | **4.1218 ± 0.0000** | - | - | 0.0042 |
| `MSE` | torchregress | 16.6571 ± 0.1409 | 13.2900 ± 0.0830 | 0.9859 ± 0.0002 | - | - | - | 1.1064 |
| `MultiQuantile` | torchregress | 16.3224 ± 0.1867 | 13.0403 ± 0.2744 | 0.9865 ± 0.0003 | - | 0.7364 ± 0.0188 | 78.8020 ± 1.6107 | 1.5120 |
| `OLS` | torchregress | **14.9186 ± 0.0000** | **11.9373 ± 0.0000** | **0.9887 ± 0.0000** | - | - | - | 0.0019 |
| `TabularMLP` | torchregress | 15.4872 ± 0.1376 | 12.2905 ± 0.0985 | 0.9878 ± 0.0002 | - | - | - | 2.6324 |
| `TabularMLP_Ens5` | torchregress | 15.2924 ± 0.0115 | 12.1856 ± 0.0085 | 0.9882 ± 0.0000 | - | - | - | 12.4160 |
| `sklearn_AdaBoost` | sklearn | 50.4969 ± 0.8455 | 39.9134 ± 0.7366 | 0.8708 ± 0.0043 | - | - | - | 0.4947 |
| `sklearn_BayesianRidge` | sklearn | 14.9187 ± 0.0000 | 11.9374 ± 0.0000 | 0.9887 ± 0.0000 | - | - | - | 0.0016 |
| `sklearn_DecisionTree` | sklearn | 38.3701 ± 0.5571 | 30.0793 ± 0.4115 | 0.9254 ± 0.0022 | - | - | - | 0.0190 |
| `sklearn_ElasticNet` | sklearn | 25.0853 ± 0.0000 | 20.0817 ± 0.0000 | 0.9681 ± 0.0000 | - | - | - | 0.0011 |
| `sklearn_ExtraTrees` | sklearn | 26.0618 ± 0.0483 | 20.5348 ± 0.0730 | 0.9656 ± 0.0001 | - | - | - | 0.2139 |
| `sklearn_GradientBoosting` | sklearn | 19.5548 ± 0.0027 | 15.4610 ± 0.0038 | 0.9806 ± 0.0000 | - | - | - | 1.4397 |
| `sklearn_HistGradientBoosting` | sklearn | 19.9010 ± 0.0000 | 15.7002 ± 0.0000 | 0.9799 ± 0.0000 | - | - | - | 0.2530 |
| `sklearn_HuberRegressor` | sklearn | 14.9309 ± 0.0000 | 11.9504 ± 0.0000 | 0.9887 ± 0.0000 | - | - | - | 0.0066 |
| `sklearn_KNeighbors` | sklearn | 54.0228 ± 0.0000 | 41.9953 ± 0.0000 | 0.8522 ± 0.0000 | - | - | - | 0.0040 |
| `sklearn_Lasso` | sklearn | 31.5698 ± 0.0000 | 25.4066 ± 0.0000 | 0.9495 ± 0.0000 | - | - | - | 0.0014 |
| `sklearn_LinearRegression` | sklearn | 14.9186 ± 0.0000 | 11.9373 ± 0.0000 | 0.9887 ± 0.0000 | - | - | - | 0.0028 |
| `sklearn_MLPRegressor` | sklearn | 17.0253 ± 0.1045 | 13.4942 ± 0.0703 | 0.9853 ± 0.0002 | - | - | - | 1.1282 |
| `sklearn_RandomForest` | sklearn | 25.2256 ± 0.0329 | 19.3900 ± 0.0289 | 0.9678 ± 0.0001 | - | - | - | 0.4930 |
| `sklearn_Ridge` | sklearn | 14.9195 ± 0.0000 | 11.9380 ± 0.0000 | 0.9887 ± 0.0000 | - | - | - | 0.0016 |
| `sklearn_SVR` | sklearn | 23.5401 ± 0.0000 | 16.4617 ± 0.0000 | 0.9719 ± 0.0000 | - | - | - | 0.3436 |

##### sklearn_baselines / openml_year

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.1596 ± 0.0029 | 0.1154 ± 0.0027 | 0.7934 ± 0.0075 | **-0.3986 ± 0.0442** | - | - | 3.2891 |
| `GaussianNLL_Linear` | torchregress | 0.2156 ± 0.0000 | 0.1670 ± 0.0000 | 0.6229 ± 0.0000 | -0.1144 ± 0.0000 | - | - | 0.0093 |
| `MSE` | torchregress | 0.1591 ± 0.0019 | 0.1151 ± 0.0016 | 0.7946 ± 0.0048 | - | - | - | 2.1203 |
| `MultiQuantile` | torchregress | 0.1585 ± 0.0008 | 0.1134 ± 0.0019 | 0.7963 ± 0.0022 | - | 0.7884 ± 0.0185 | 0.7232 ± 0.0137 | 2.9275 |
| `OLS` | torchregress | 0.2156 ± 0.0000 | 0.1670 ± 0.0000 | 0.6229 ± 0.0000 | - | - | - | 0.0035 |
| `TabularMLP` | torchregress | 0.1549 ± 0.0009 | 0.1116 ± 0.0008 | 0.8053 ± 0.0022 | - | - | - | 5.3415 |
| `TabularMLP_Ens5` | torchregress | 0.1515 ± 0.0002 | 0.1089 ± 0.0001 | 0.8138 ± 0.0004 | - | - | - | 24.7094 |
| `sklearn_AdaBoost` | sklearn | 0.2215 ± 0.0029 | 0.1792 ± 0.0032 | 0.6020 ± 0.0105 | - | - | - | 0.6388 |
| `sklearn_BayesianRidge` | sklearn | 0.2156 ± 0.0000 | 0.1670 ± 0.0000 | 0.6229 ± 0.0000 | - | - | - | 0.0016 |
| `sklearn_DecisionTree` | sklearn | 0.2039 ± 0.0009 | 0.1429 ± 0.0002 | 0.6628 ± 0.0029 | - | - | - | 0.0243 |
| `sklearn_ElasticNet` | sklearn | 0.2441 ± 0.0000 | 0.1933 ± 0.0000 | 0.5165 ± 0.0000 | - | - | - | 0.0012 |
| `sklearn_ExtraTrees` | sklearn | 0.1787 ± 0.0016 | 0.1368 ± 0.0012 | 0.7408 ± 0.0045 | - | - | - | 0.3468 |
| `sklearn_GradientBoosting` | sklearn | 0.1477 ± 0.0001 | 0.1084 ± 0.0000 | 0.8230 ± 0.0003 | - | - | - | 2.0669 |
| `sklearn_HistGradientBoosting` | sklearn | **0.1407 ± 0.0000** | **0.1007 ± 0.0000** | **0.8395 ± 0.0000** | - | - | - | 0.2686 |
| `sklearn_HuberRegressor` | sklearn | 0.2111 ± 0.0000 | 0.1578 ± 0.0000 | 0.6386 ± 0.0000 | - | - | - | 0.0205 |
| `sklearn_KNeighbors` | sklearn | 0.1919 ± 0.0000 | 0.1411 ± 0.0000 | 0.7014 ± 0.0000 | - | - | - | 0.0071 |
| `sklearn_Lasso` | sklearn | 0.2568 ± 0.0000 | 0.2047 ± 0.0000 | 0.4651 ± 0.0000 | - | - | - | 0.0013 |
| `sklearn_LinearRegression` | sklearn | 0.2156 ± 0.0000 | 0.1670 ± 0.0000 | 0.6229 ± 0.0000 | - | - | - | 0.0023 |
| `sklearn_MLPRegressor` | sklearn | 0.1620 ± 0.0011 | 0.1161 ± 0.0014 | 0.7872 ± 0.0030 | - | - | - | 2.0299 |
| `sklearn_RandomForest` | sklearn | 0.1582 ± 0.0009 | 0.1147 ± 0.0009 | 0.7969 ± 0.0024 | - | - | - | 0.6300 |
| `sklearn_Ridge` | sklearn | 0.2156 ± 0.0000 | 0.1670 ± 0.0000 | 0.6229 ± 0.0000 | - | - | - | 0.0014 |
| `sklearn_SVR` | sklearn | 0.1706 ± 0.0000 | 0.1249 ± 0.0000 | 0.7639 ± 0.0000 | - | - | - | 1.6576 |

##### sklearn_baselines / synthetic_censored

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | **0.1765 ± 0.0021** | 0.1292 ± 0.0023 | **0.9682 ± 0.0008** | **-0.3552 ± 0.0354** | - | - | 0.6570 |
| `GaussianNLL_Linear` | torchregress | 0.3787 ± 0.0000 | 0.2998 ± 0.0000 | 0.8535 ± 0.0000 | 0.4486 ± 0.0000 | - | - | 0.0051 |
| `MSE` | torchregress | 0.1811 ± 0.0028 | 0.1369 ± 0.0016 | 0.9665 ± 0.0010 | - | - | - | 0.4170 |
| `MultiQuantile` | torchregress | 0.1794 ± 0.0003 | **0.1292 ± 0.0026** | 0.9671 ± 0.0001 | - | 0.7544 ± 0.0440 | 0.8052 ± 0.0275 | 0.5882 |
| `OLS` | torchregress | 0.3787 ± 0.0000 | 0.2998 ± 0.0000 | 0.8535 ± 0.0000 | - | - | - | 0.0029 |
| `TabularMLP` | torchregress | 0.1862 ± 0.0040 | 0.1439 ± 0.0030 | 0.9646 ± 0.0015 | - | - | - | 1.1312 |
| `TabularMLP_Ens5` | torchregress | 0.1839 ± 0.0017 | 0.1419 ± 0.0012 | 0.9655 ± 0.0006 | - | - | - | 6.0706 |
| `sklearn_AdaBoost` | sklearn | 0.3834 ± 0.0078 | 0.3240 ± 0.0043 | 0.8499 ± 0.0062 | - | - | - | 0.1775 |
| `sklearn_BayesianRidge` | sklearn | 0.3788 ± 0.0000 | 0.2999 ± 0.0000 | 0.8535 ± 0.0000 | - | - | - | 0.0015 |
| `sklearn_DecisionTree` | sklearn | 0.3355 ± 0.0090 | 0.2270 ± 0.0047 | 0.8850 ± 0.0061 | - | - | - | 0.0063 |
| `sklearn_ElasticNet` | sklearn | 0.3996 ± 0.0000 | 0.3288 ± 0.0000 | 0.8369 ± 0.0000 | - | - | - | 0.0011 |
| `sklearn_ExtraTrees` | sklearn | 0.2124 ± 0.0025 | 0.1594 ± 0.0017 | 0.9539 ± 0.0011 | - | - | - | 0.2216 |
| `sklearn_GradientBoosting` | sklearn | 0.2072 ± 0.0003 | 0.1568 ± 0.0002 | 0.9562 ± 0.0001 | - | - | - | 0.4606 |
| `sklearn_HistGradientBoosting` | sklearn | 0.2010 ± 0.0000 | 0.1484 ± 0.0000 | 0.9587 ± 0.0000 | - | - | - | 0.2020 |
| `sklearn_HuberRegressor` | sklearn | 0.3848 ± 0.0000 | 0.2836 ± 0.0000 | 0.8488 ± 0.0000 | - | - | - | 0.0041 |
| `sklearn_KNeighbors` | sklearn | 0.4104 ± 0.0000 | 0.3015 ± 0.0000 | 0.8280 ± 0.0000 | - | - | - | 0.0018 |
| `sklearn_Lasso` | sklearn | 0.4182 ± 0.0000 | 0.3455 ± 0.0000 | 0.8214 ± 0.0000 | - | - | - | 0.0013 |
| `sklearn_LinearRegression` | sklearn | 0.3787 ± 0.0000 | 0.2998 ± 0.0000 | 0.8535 ± 0.0000 | - | - | - | 0.0023 |
| `sklearn_MLPRegressor` | sklearn | 0.1868 ± 0.0029 | 0.1426 ± 0.0025 | 0.9644 ± 0.0011 | - | - | - | 0.4326 |
| `sklearn_RandomForest` | sklearn | 0.2266 ± 0.0024 | 0.1611 ± 0.0020 | 0.9476 ± 0.0011 | - | - | - | 0.3589 |
| `sklearn_Ridge` | sklearn | 0.3788 ± 0.0000 | 0.2999 ± 0.0000 | 0.8535 ± 0.0000 | - | - | - | 0.0014 |
| `sklearn_SVR` | sklearn | 0.2290 ± 0.0000 | 0.1758 ± 0.0000 | 0.9464 ± 0.0000 | - | - | - | 0.0693 |

##### sklearn_baselines / synthetic_eiv

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.2409 ± 0.0027 | 0.1906 ± 0.0033 | 0.9489 ± 0.0012 | 0.0551 ± 0.0308 | - | - | 0.8317 |
| `GaussianNLL_Linear` | torchregress | 0.2322 ± 0.0000 | 0.1850 ± 0.0000 | 0.9525 ± 0.0000 | **-0.0412 ± 0.0000** | - | - | 0.0077 |
| `MSE` | torchregress | 0.2439 ± 0.0039 | 0.1945 ± 0.0049 | 0.9476 ± 0.0017 | - | - | - | 0.4999 |
| `MultiQuantile` | torchregress | 0.2388 ± 0.0022 | 0.1904 ± 0.0021 | 0.9498 ± 0.0009 | - | 0.7811 ± 0.0158 | 1.1063 ± 0.0106 | 0.6978 |
| `OLS` | torchregress | 0.2322 ± 0.0000 | 0.1850 ± 0.0000 | 0.9525 ± 0.0000 | - | - | - | 0.0021 |
| `TabularMLP` | torchregress | 0.2381 ± 0.0012 | 0.1893 ± 0.0014 | 0.9501 ± 0.0005 | - | - | - | 0.8440 |
| `TabularMLP_Ens5` | torchregress | 0.2359 ± 0.0007 | 0.1874 ± 0.0005 | 0.9510 ± 0.0003 | - | - | - | 4.1842 |
| `sklearn_AdaBoost` | sklearn | 0.4608 ± 0.0049 | 0.3629 ± 0.0047 | 0.8129 ± 0.0040 | - | - | - | 0.1659 |
| `sklearn_BayesianRidge` | sklearn | 0.2322 ± 0.0000 | 0.1850 ± 0.0000 | 0.9525 ± 0.0000 | - | - | - | 0.0016 |
| `sklearn_DecisionTree` | sklearn | 0.4339 ± 0.0032 | 0.3300 ± 0.0031 | 0.8342 ± 0.0025 | - | - | - | 0.0060 |
| `sklearn_ElasticNet` | sklearn | 0.2742 ± 0.0000 | 0.2223 ± 0.0000 | 0.9338 ± 0.0000 | - | - | - | 0.0013 |
| `sklearn_ExtraTrees` | sklearn | 0.2904 ± 0.0016 | 0.2246 ± 0.0006 | 0.9257 ± 0.0008 | - | - | - | 0.2238 |
| `sklearn_GradientBoosting` | sklearn | 0.2892 ± 0.0002 | 0.2268 ± 0.0003 | 0.9263 ± 0.0001 | - | - | - | 0.4029 |
| `sklearn_HistGradientBoosting` | sklearn | 0.2948 ± 0.0000 | 0.2295 ± 0.0000 | 0.9235 ± 0.0000 | - | - | - | 0.1736 |
| `sklearn_HuberRegressor` | sklearn | **0.2320 ± 0.0000** | **0.1848 ± 0.0000** | **0.9526 ± 0.0000** | - | - | - | 0.0048 |
| `sklearn_KNeighbors` | sklearn | 0.4024 ± 0.0000 | 0.3164 ± 0.0000 | 0.8574 ± 0.0000 | - | - | - | 0.0018 |
| `sklearn_Lasso` | sklearn | 0.3037 ± 0.0000 | 0.2458 ± 0.0000 | 0.9188 ± 0.0000 | - | - | - | 0.0014 |
| `sklearn_LinearRegression` | sklearn | 0.2322 ± 0.0000 | 0.1850 ± 0.0000 | 0.9525 ± 0.0000 | - | - | - | 0.0022 |
| `sklearn_MLPRegressor` | sklearn | 0.2503 ± 0.0050 | 0.1994 ± 0.0039 | 0.9448 ± 0.0022 | - | - | - | 0.4453 |
| `sklearn_RandomForest` | sklearn | 0.3080 ± 0.0013 | 0.2387 ± 0.0011 | 0.9164 ± 0.0007 | - | - | - | 0.3525 |
| `sklearn_Ridge` | sklearn | 0.2322 ± 0.0000 | 0.1851 ± 0.0000 | 0.9525 ± 0.0000 | - | - | - | 0.0015 |
| `sklearn_SVR` | sklearn | 0.2912 ± 0.0000 | 0.2177 ± 0.0000 | 0.9253 ± 0.0000 | - | - | - | 0.0785 |

##### sklearn_baselines / synthetic_hetero

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.3747 ± 0.0048 | 0.2861 ± 0.0035 | 0.8351 ± 0.0042 | **0.4435 ± 0.0742** | - | - | 0.6760 |
| `GaussianNLL_Linear` | torchregress | 0.3850 ± 0.0000 | 0.2959 ± 0.0000 | 0.8260 ± 0.0000 | 0.4701 ± 0.0000 | - | - | 0.0043 |
| `MSE` | torchregress | 0.3802 ± 0.0035 | 0.2929 ± 0.0047 | 0.8303 ± 0.0031 | - | - | - | 0.4211 |
| `MultiQuantile` | torchregress | 0.3791 ± 0.0059 | 0.2885 ± 0.0048 | 0.8313 ± 0.0053 | - | 0.7456 ± 0.0162 | 1.7886 ± 0.0430 | 0.6040 |
| `OLS` | torchregress | 0.3850 ± 0.0000 | 0.2959 ± 0.0000 | 0.8260 ± 0.0000 | - | - | - | 0.0031 |
| `TabularMLP` | torchregress | 0.3655 ± 0.0044 | 0.2783 ± 0.0036 | 0.8432 ± 0.0038 | - | - | - | 0.8442 |
| `TabularMLP_Ens5` | torchregress | **0.3579 ± 0.0009** | **0.2708 ± 0.0007** | **0.8497 ± 0.0008** | - | - | - | 3.7163 |
| `sklearn_AdaBoost` | sklearn | 0.4646 ± 0.0052 | 0.3616 ± 0.0025 | 0.7466 ± 0.0057 | - | - | - | 0.1881 |
| `sklearn_BayesianRidge` | sklearn | 0.3850 ± 0.0000 | 0.2959 ± 0.0000 | 0.8260 ± 0.0000 | - | - | - | 0.0014 |
| `sklearn_DecisionTree` | sklearn | 0.5491 ± 0.0121 | 0.4088 ± 0.0042 | 0.6459 ± 0.0157 | - | - | - | 0.0062 |
| `sklearn_ElasticNet` | sklearn | 0.3997 ± 0.0000 | 0.3114 ± 0.0000 | 0.8124 ± 0.0000 | - | - | - | 0.0011 |
| `sklearn_ExtraTrees` | sklearn | 0.3816 ± 0.0019 | 0.2920 ± 0.0025 | 0.8291 ± 0.0017 | - | - | - | 0.2216 |
| `sklearn_GradientBoosting` | sklearn | 0.3883 ± 0.0024 | 0.2896 ± 0.0009 | 0.8230 ± 0.0022 | - | - | - | 0.4635 |
| `sklearn_HistGradientBoosting` | sklearn | 0.3834 ± 0.0000 | 0.2873 ± 0.0000 | 0.8275 ± 0.0000 | - | - | - | 0.1766 |
| `sklearn_HuberRegressor` | sklearn | 0.3850 ± 0.0000 | 0.2960 ± 0.0000 | 0.8260 ± 0.0000 | - | - | - | 0.0044 |
| `sklearn_KNeighbors` | sklearn | 0.4843 ± 0.0000 | 0.3719 ± 0.0000 | 0.7247 ± 0.0000 | - | - | - | 0.0020 |
| `sklearn_Lasso` | sklearn | 0.4170 ± 0.0000 | 0.3275 ± 0.0000 | 0.7959 ± 0.0000 | - | - | - | 0.0013 |
| `sklearn_LinearRegression` | sklearn | 0.3850 ± 0.0000 | 0.2959 ± 0.0000 | 0.8260 ± 0.0000 | - | - | - | 0.0021 |
| `sklearn_MLPRegressor` | sklearn | 0.3929 ± 0.0066 | 0.3012 ± 0.0032 | 0.8188 ± 0.0061 | - | - | - | 0.5628 |
| `sklearn_RandomForest` | sklearn | 0.3883 ± 0.0027 | 0.2943 ± 0.0018 | 0.8230 ± 0.0024 | - | - | - | 0.3249 |
| `sklearn_Ridge` | sklearn | 0.3850 ± 0.0000 | 0.2959 ± 0.0000 | 0.8260 ± 0.0000 | - | - | - | 0.0014 |
| `sklearn_SVR` | sklearn | 0.4148 ± 0.0000 | 0.3113 ± 0.0000 | 0.7980 ± 0.0000 | - | - | - | 0.0856 |

##### sklearn_baselines / synthetic_outliers

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 1.2598 ± 0.0159 | 0.7161 ± 0.0136 | 0.7976 ± 0.0051 | 1.9126 ± 0.2646 | - | - | 0.6676 |
| `GaussianNLL_Linear` | torchregress | 1.3885 ± 0.0000 | 0.8542 ± 0.0000 | 0.7542 ± 0.0000 | **1.7511 ± 0.0000** | - | - | 0.0052 |
| `MSE` | torchregress | 1.2655 ± 0.0092 | 0.7159 ± 0.0089 | 0.7958 ± 0.0030 | - | - | - | 0.4252 |
| `MultiQuantile` | torchregress | 1.2409 ± 0.0052 | 0.6669 ± 0.0085 | 0.8037 ± 0.0016 | - | 0.7278 ± 0.0129 | 7.9708 ± 0.0459 | 0.5922 |
| `OLS` | torchregress | 1.3885 ± 0.0000 | 0.8542 ± 0.0000 | 0.7542 ± 0.0000 | - | - | - | 0.0033 |
| `TabularMLP` | torchregress | 1.2373 ± 0.0071 | 0.6914 ± 0.0071 | 0.8048 ± 0.0022 | - | - | - | 0.8653 |
| `TabularMLP_Ens5` | torchregress | **1.2168 ± 0.0027** | **0.6474 ± 0.0037** | **0.8112 ± 0.0008** | - | - | - | 3.9621 |
| `sklearn_AdaBoost` | sklearn | 1.5590 ± 0.0135 | 1.0697 ± 0.0083 | 0.6901 ± 0.0054 | - | - | - | 0.1497 |
| `sklearn_BayesianRidge` | sklearn | 1.3886 ± 0.0000 | 0.8545 ± 0.0000 | 0.7542 ± 0.0000 | - | - | - | 0.0015 |
| `sklearn_DecisionTree` | sklearn | 1.6644 ± 0.0365 | 1.1089 ± 0.0222 | 0.6467 ± 0.0154 | - | - | - | 0.0063 |
| `sklearn_ElasticNet` | sklearn | 1.4219 ± 0.0000 | 0.9075 ± 0.0000 | 0.7422 ± 0.0000 | - | - | - | 0.0011 |
| `sklearn_ExtraTrees` | sklearn | 1.3304 ± 0.0039 | 0.8014 ± 0.0054 | 0.7743 ± 0.0013 | - | - | - | 0.2210 |
| `sklearn_GradientBoosting` | sklearn | 1.2943 ± 0.0023 | 0.7614 ± 0.0010 | 0.7864 ± 0.0008 | - | - | - | 0.4222 |
| `sklearn_HistGradientBoosting` | sklearn | 1.2909 ± 0.0000 | 0.7895 ± 0.0000 | 0.7875 ± 0.0000 | - | - | - | 0.1875 |
| `sklearn_HuberRegressor` | sklearn | 1.3916 ± 0.0000 | 0.8417 ± 0.0000 | 0.7531 ± 0.0000 | - | - | - | 0.0048 |
| `sklearn_KNeighbors` | sklearn | 1.5917 ± 0.0000 | 1.1377 ± 0.0000 | 0.6770 ± 0.0000 | - | - | - | 0.0016 |
| `sklearn_Lasso` | sklearn | 1.4434 ± 0.0000 | 0.9401 ± 0.0000 | 0.7344 ± 0.0000 | - | - | - | 0.0014 |
| `sklearn_LinearRegression` | sklearn | 1.3885 ± 0.0000 | 0.8542 ± 0.0000 | 0.7542 ± 0.0000 | - | - | - | 0.0026 |
| `sklearn_MLPRegressor` | sklearn | 1.2656 ± 0.0168 | 0.7383 ± 0.0162 | 0.7958 ± 0.0055 | - | - | - | 0.4010 |
| `sklearn_RandomForest` | sklearn | 1.3387 ± 0.0064 | 0.8211 ± 0.0034 | 0.7715 ± 0.0022 | - | - | - | 0.3666 |
| `sklearn_Ridge` | sklearn | 1.3885 ± 0.0000 | 0.8544 ± 0.0000 | 0.7542 ± 0.0000 | - | - | - | 0.0015 |
| `sklearn_SVR` | sklearn | 1.2638 ± 0.0000 | 0.7144 ± 0.0000 | 0.7963 ± 0.0000 | - | - | - | 0.0539 |

##### sklearn_baselines / synthetic_sinusoidal

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.7262 ± 0.0116 | 0.6189 ± 0.0138 | 0.1147 ± 0.0283 | **1.0827 ± 0.0258** | - | - | 0.6763 |
| `GaussianNLL_Linear` | torchregress | 0.7437 ± 0.0000 | 0.6393 ± 0.0000 | 0.0717 ± 0.0000 | 1.1228 ± 0.0000 | - | - | 0.0087 |
| `MSE` | torchregress | 0.7304 ± 0.0125 | 0.6244 ± 0.0132 | 0.1044 ± 0.0306 | - | - | - | 0.4507 |
| `MultiQuantile` | torchregress | 0.7550 ± 0.0023 | 0.6390 ± 0.0034 | 0.0433 ± 0.0059 | - | 0.7844 ± 0.0035 | 2.8687 ± 0.0175 | 0.5710 |
| `OLS` | torchregress | 0.7437 ± 0.0000 | 0.6393 ± 0.0000 | 0.0717 ± 0.0000 | - | - | - | 0.0021 |
| `TabularMLP` | torchregress | 0.2707 ± 0.0220 † | 0.2177 ± 0.0155 | 0.8765 ± 0.0200 | - | - | - | 1.5676 |
| `TabularMLP_Ens5` | torchregress | 0.2444 ± 0.0049 | 0.1982 ± 0.0039 | 0.8998 ± 0.0040 | - | - | - | 8.2344 |
| `sklearn_AdaBoost` | sklearn | 0.6298 ± 0.0135 | 0.5301 ± 0.0138 | 0.3340 ± 0.0288 | - | - | - | 0.1191 |
| `sklearn_BayesianRidge` | sklearn | 0.7436 ± 0.0000 | 0.6396 ± 0.0000 | 0.0720 ± 0.0000 | - | - | - | 0.0020 |
| `sklearn_DecisionTree` | sklearn | 0.3537 ± 0.0003 | 0.2891 ± 0.0010 | 0.7900 ± 0.0004 | - | - | - | 0.0058 |
| `sklearn_ElasticNet` | sklearn | 0.7447 ± 0.0000 | 0.6426 ± 0.0000 | 0.0690 ± 0.0000 | - | - | - | 0.0015 |
| `sklearn_ExtraTrees` | sklearn | 0.3382 ± 0.0043 | 0.2756 ± 0.0040 | 0.8080 ± 0.0049 | - | - | - | 0.1968 |
| `sklearn_GradientBoosting` | sklearn | 0.2261 ± 0.0002 | 0.1837 ± 0.0000 | 0.9142 ± 0.0001 | - | - | - | 0.3266 |
| `sklearn_HistGradientBoosting` | sklearn | **0.2081 ± 0.0000** | **0.1683 ± 0.0000** | **0.9273 ± 0.0000** | - | - | - | 0.1951 |
| `sklearn_HuberRegressor` | sklearn | 0.7444 ± 0.0000 | 0.6377 ± 0.0000 | 0.0698 ± 0.0000 | - | - | - | 0.0054 |
| `sklearn_KNeighbors` | sklearn | 0.7229 ± 0.0000 | 0.5870 ± 0.0000 | 0.1229 ± 0.0000 | - | - | - | 0.0021 |
| `sklearn_Lasso` | sklearn | 0.7482 ± 0.0000 | 0.6464 ± 0.0000 | 0.0605 ± 0.0000 | - | - | - | 0.0022 |
| `sklearn_LinearRegression` | sklearn | 0.7437 ± 0.0000 | 0.6393 ± 0.0000 | 0.0717 ± 0.0000 | - | - | - | 0.0045 |
| `sklearn_MLPRegressor` | sklearn | 0.7387 ± 0.0034 | 0.6312 ± 0.0048 | 0.0840 ± 0.0085 | - | - | - | 0.4894 |
| `sklearn_RandomForest` | sklearn | 0.2875 ± 0.0008 | 0.2361 ± 0.0007 | 0.8613 ± 0.0008 | - | - | - | 0.3135 |
| `sklearn_Ridge` | sklearn | 0.7437 ± 0.0000 | 0.6393 ± 0.0000 | 0.0717 ± 0.0000 | - | - | - | 0.0021 |
| `sklearn_SVR` | sklearn | 0.7393 ± 0.0000 | 0.6165 ± 0.0000 | 0.0827 ± 0.0000 | - | - | - | 0.0972 |

##### sklearn_baselines / synthetic_tweedie

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 1.1675 ± 0.0022 | 0.7496 ± 0.0046 | 0.4973 ± 0.0019 | **1.2561 ± 0.0186** | - | - | 0.6602 |
| `GaussianNLL_Linear` | torchregress | 1.2697 ± 0.0000 | 0.8529 ± 0.0000 | 0.4056 ± 0.0000 | 1.6619 ± 0.0000 | - | - | 0.0079 |
| `MSE` | torchregress | **1.1568 ± 0.0107** | 0.7497 ± 0.0063 | **0.5065 ± 0.0091** | - | - | - | 0.4246 |
| `MultiQuantile` | torchregress | 1.1937 ± 0.0064 | **0.7420 ± 0.0033** | 0.4746 ± 0.0056 | - | 0.7689 ± 0.0127 | 4.5871 ± 0.0035 | 0.5919 |
| `OLS` | torchregress | 1.2697 ± 0.0000 | 0.8529 ± 0.0000 | 0.4056 ± 0.0000 | - | - | - | 0.0026 |
| `TabularMLP` | torchregress | 1.1863 ± 0.0510 | 0.7641 ± 0.0321 | 0.4804 ± 0.0451 | - | - | - | 0.6247 |
| `TabularMLP_Ens5` | torchregress | 1.1655 ± 0.0094 | 0.7433 ± 0.0040 | 0.4990 ± 0.0080 | - | - | - | 3.1905 |
| `sklearn_AdaBoost` | sklearn | 1.2456 ± 0.0030 | 0.9263 ± 0.0097 | 0.4279 ± 0.0028 | - | - | - | 0.1363 |
| `sklearn_BayesianRidge` | sklearn | 1.2699 ± 0.0000 | 0.8517 ± 0.0000 | 0.4053 ± 0.0000 | - | - | - | 0.0016 |
| `sklearn_DecisionTree` | sklearn | 1.6077 ± 0.0254 | 0.9684 ± 0.0083 | 0.0467 ± 0.0301 | - | - | - | 0.0060 |
| `sklearn_ElasticNet` | sklearn | 1.2837 ± 0.0000 | 0.8287 ± 0.0000 | 0.3923 ± 0.0000 | - | - | - | 0.0013 |
| `sklearn_ExtraTrees` | sklearn | 1.1702 ± 0.0044 | 0.7519 ± 0.0041 | 0.4951 ± 0.0038 | - | - | - | 0.2238 |
| `sklearn_GradientBoosting` | sklearn | 1.1874 ± 0.0012 | 0.7715 ± 0.0011 | 0.4801 ± 0.0010 | - | - | - | 0.4016 |
| `sklearn_HistGradientBoosting` | sklearn | 1.2031 ± 0.0000 | 0.7871 ± 0.0000 | 0.4662 ± 0.0000 | - | - | - | 0.1498 |
| `sklearn_HuberRegressor` | sklearn | 1.3317 ± 0.0000 | 0.8166 ± 0.0000 | 0.3460 ± 0.0000 | - | - | - | 0.0042 |
| `sklearn_KNeighbors` | sklearn | 1.3228 ± 0.0000 | 0.8415 ± 0.0000 | 0.3548 ± 0.0000 | - | - | - | 0.0018 |
| `sklearn_Lasso` | sklearn | 1.2920 ± 0.0000 | 0.8273 ± 0.0000 | 0.3845 ± 0.0000 | - | - | - | 0.0015 |
| `sklearn_LinearRegression` | sklearn | 1.2697 ± 0.0000 | 0.8529 ± 0.0000 | 0.4056 ± 0.0000 | - | - | - | 0.0026 |
| `sklearn_MLPRegressor` | sklearn | 1.1659 ± 0.0076 | 0.7565 ± 0.0047 | 0.4987 ± 0.0066 | - | - | - | 0.4398 |
| `sklearn_RandomForest` | sklearn | 1.2044 ± 0.0106 | 0.7714 ± 0.0015 | 0.4651 ± 0.0094 | - | - | - | 0.3407 |
| `sklearn_Ridge` | sklearn | 1.2697 ± 0.0000 | 0.8527 ± 0.0000 | 0.4055 ± 0.0000 | - | - | - | 0.0016 |
| `sklearn_SVR` | sklearn | 1.2183 ± 0.0000 | 0.7610 ± 0.0000 | 0.4527 ± 0.0000 | - | - | - | 0.0762 |

##### sklearn_baselines / tabred_cooking

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.5879 ± 0.0069 | 0.4550 ± 0.0034 | 0.1788 ± 0.0194 | 4.2562 ± 0.9207 | - | - | 7.3808 |
| `GaussianNLL_Linear` | torchregress | 0.4766 ± 0.0000 | 0.3711 ± 0.0000 | 0.4602 ± 0.0000 | **0.6792 ± 0.0000** | - | - | 0.3914 |
| `MSE` | torchregress | 0.6216 ± 0.0033 | 0.4842 ± 0.0038 | 0.0819 ± 0.0098 | - | - | - | 3.8700 |
| `MultiQuantile` | torchregress | 0.5766 ± 0.0014 | 0.4486 ± 0.0023 | 0.2099 ± 0.0038 | - | 0.4589 ± 0.0079 | 4.4969 ± 0.0571 | 5.5395 |
| `OLS` | torchregress | 0.4824 ± 0.0050 | 0.3758 ± 0.0042 | 0.4471 ± 0.0115 | - | - | - | 0.0253 |
| `TabularMLP` | torchregress | 0.4835 ± 0.0027 | 0.3771 ± 0.0021 | 0.4446 ± 0.0061 | - | - | - | 6.0605 |
| `TabularMLP_Ens5` | torchregress | 0.4794 ± 0.0010 | 0.3744 ± 0.0010 | 0.4539 ± 0.0022 | - | - | - | 28.5966 |
| `sklearn_AdaBoost` | sklearn | 0.5071 ± 0.0003 | 0.4046 ± 0.0003 | 0.3889 ± 0.0007 | - | - | - | 14.7956 |
| `sklearn_BayesianRidge` | sklearn | **0.4749 ± 0.0000** | **0.3707 ± 0.0000** | **0.4642 ± 0.0000** | - | - | - | 0.0259 |
| `sklearn_DecisionTree` | sklearn | 0.5797 ± 0.0048 | 0.4463 ± 0.0029 | 0.2015 ± 0.0132 | - | - | - | 0.8416 |
| `sklearn_ElasticNet` | sklearn | 0.4851 ± 0.0000 | 0.3825 ± 0.0000 | 0.4409 ± 0.0000 | - | - | - | 0.0371 |
| `sklearn_ExtraTrees` | sklearn | 0.4807 ± 0.0005 | 0.3764 ± 0.0003 | 0.4511 ± 0.0011 | - | - | - | 2.6344 |
| `sklearn_GradientBoosting` | sklearn | 0.4799 ± 0.0001 | 0.3748 ± 0.0002 | 0.4528 ± 0.0003 | - | - | - | 38.4192 |
| `sklearn_HistGradientBoosting` | sklearn | 0.4791 ± 0.0000 | 0.3760 ± 0.0000 | 0.4546 ± 0.0000 | - | - | - | 0.9252 |
| `sklearn_HuberRegressor` | sklearn | 0.4772 ± 0.0000 | 0.3714 ± 0.0000 | 0.4590 ± 0.0000 | - | - | - | 1.3202 |
| `sklearn_KNeighbors` | sklearn | 0.5439 ± 0.0000 | 0.4303 ± 0.0000 | 0.2972 ± 0.0000 | - | - | - | 0.0027 |
| `sklearn_Lasso` | sklearn | 0.4911 ± 0.0000 | 0.3888 ± 0.0000 | 0.4269 ± 0.0000 | - | - | - | 0.0281 |
| `sklearn_LinearRegression` | sklearn | 0.4763 ± 0.0000 | 0.3709 ± 0.0000 | 0.4610 ± 0.0000 | - | - | - | 0.0196 |
| `sklearn_MLPRegressor` | sklearn | 0.6276 ± 0.0054 | 0.4914 ± 0.0053 | 0.0640 ± 0.0160 | - | - | - | 3.8691 |
| `sklearn_RandomForest` | sklearn | 0.4823 ± 0.0004 | 0.3785 ± 0.0004 | 0.4473 ± 0.0009 | - | - | - | 9.2476 |
| `sklearn_Ridge` | sklearn | 0.4764 ± 0.0000 | 0.3711 ± 0.0000 | 0.4607 ± 0.0000 | - | - | - | 0.0055 |
| `sklearn_SVR` | sklearn | 0.4929 ± 0.0000 | 0.3834 ± 0.0000 | 0.4227 ± 0.0000 | - | - | - | 4.4510 |

##### sklearn_baselines / tabred_delivery

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.7002 ± 0.0144 | 0.5531 ± 0.0106 | 0.0869 ± 0.0373 | 6.3972 ± 1.2102 | - | - | 7.0215 |
| `GaussianNLL_Linear` | torchregress | 0.5624 ± 0.0001 | 0.4494 ± 0.0000 | 0.4111 ± 0.0001 | **0.8463 ± 0.0001** | - | - | 0.7755 |
| `MSE` | torchregress | 0.7284 ± 0.0057 | 0.5747 ± 0.0035 | 0.0120 ± 0.0154 | - | - | - | 3.9983 |
| `MultiQuantile` | torchregress | 0.7038 ± 0.0129 | 0.5553 ± 0.0093 | 0.0774 ± 0.0339 | - | 0.3312 ± 0.0085 | 6.6825 ± 0.2012 | 5.6875 |
| `OLS` | torchregress | 0.6570 ± 0.0266 | 0.5188 ± 0.0309 | 0.1954 ± 0.0659 | - | - | - | 0.0243 |
| `TabularMLP` | torchregress | 0.5726 ± 0.0094 | 0.4615 ± 0.0147 | 0.3894 ± 0.0200 | - | - | - | 5.1525 |
| `TabularMLP_Ens5` | torchregress | **0.5569 ± 0.0016** | 0.4425 ± 0.0034 | **0.4225 ± 0.0032** | - | - | - | 29.5819 |
| `sklearn_AdaBoost` | sklearn | 0.6103 ± 0.0023 | 0.5114 ± 0.0025 | 0.3065 ± 0.0052 | - | - | - | 15.4969 |
| `sklearn_BayesianRidge` | sklearn | 0.5603 ± 0.0000 | 0.4459 ± 0.0000 | 0.4155 ± 0.0000 | - | - | - | 0.0260 |
| `sklearn_DecisionTree` | sklearn | 0.6754 ± 0.0047 | 0.5178 ± 0.0014 | 0.1507 ± 0.0119 | - | - | - | 0.8260 |
| `sklearn_ElasticNet` | sklearn | 0.5729 ± 0.0000 | 0.4506 ± 0.0000 | 0.3888 ± 0.0000 | - | - | - | 0.0264 |
| `sklearn_ExtraTrees` | sklearn | 0.5642 ± 0.0002 | 0.4492 ± 0.0004 | 0.4073 ± 0.0003 | - | - | - | 2.7701 |
| `sklearn_GradientBoosting` | sklearn | 0.5607 ± 0.0001 | 0.4478 ± 0.0001 | 0.4145 ± 0.0002 | - | - | - | 44.3563 |
| `sklearn_HistGradientBoosting` | sklearn | 0.5606 ± 0.0000 | 0.4456 ± 0.0000 | 0.4149 ± 0.0000 | - | - | - | 0.9906 |
| `sklearn_HuberRegressor` | sklearn | 0.5609 ± 0.0000 | **0.4365 ± 0.0000** | 0.4142 ± 0.0000 | - | - | - | 1.1719 |
| `sklearn_KNeighbors` | sklearn | 0.6990 ± 0.0000 | 0.5627 ± 0.0000 | 0.0903 ± 0.0000 | - | - | - | 0.0038 |
| `sklearn_Lasso` | sklearn | 0.5797 ± 0.0000 | 0.4532 ± 0.0000 | 0.3742 ± 0.0000 | - | - | - | 0.0429 |
| `sklearn_LinearRegression` | sklearn | 0.5626 ± 0.0000 | 0.4500 ± 0.0000 | 0.4105 ± 0.0000 | - | - | - | 0.0205 |
| `sklearn_MLPRegressor` | sklearn | 0.7489 ± 0.0079 | 0.5903 ± 0.0089 | -0.0445 ± 0.0222 | - | - | - | 4.3131 |
| `sklearn_RandomForest` | sklearn | 0.5626 ± 0.0009 | 0.4449 ± 0.0006 | 0.4106 ± 0.0018 | - | - | - | 9.2575 |
| `sklearn_Ridge` | sklearn | 0.5622 ± 0.0000 | 0.4485 ± 0.0000 | 0.4115 ± 0.0000 | - | - | - | 0.0063 |
| `sklearn_SVR` | sklearn | 0.5654 ± 0.0000 | 0.4384 ± 0.0000 | 0.4047 ± 0.0000 | - | - | - | 4.9998 |

##### sklearn_baselines / tabred_maps

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 0.2040 ± 0.0016 | 0.1568 ± 0.0012 | 0.6938 ± 0.0047 | 4.2684 ± 0.3946 | - | - | 16.9988 |
| `GaussianNLL_Linear` | torchregress | 0.1649 ± 0.0000 | 0.1247 ± 0.0000 | 0.8000 ± 0.0000 | **-0.3828 ± 0.0000** | - | - | 1.3498 |
| `MSE` | torchregress | 0.2085 ± 0.0010 | 0.1612 ± 0.0003 | 0.6799 ± 0.0030 | - | - | - | 12.1763 |
| `MultiQuantile` | torchregress | 0.2045 ± 0.0021 | 0.1576 ± 0.0013 | 0.6923 ± 0.0062 | - | 0.3381 ± 0.0124 | 1.9592 ± 0.0210 | 14.4093 |
| `OLS` | torchregress | 0.1830 ± 0.0005 | 0.1400 ± 0.0004 | 0.7536 ± 0.0013 | - | - | - | 0.2325 |
| `TabularMLP` | torchregress | 0.1693 ± 0.0019 | 0.1288 ± 0.0017 | 0.7890 ± 0.0048 | - | - | - | 35.4606 |
| `TabularMLP_Ens5` | torchregress | 0.1654 ± 0.0003 | 0.1254 ± 0.0005 | 0.7987 ± 0.0006 | - | - | - | 170.8 |
| `sklearn_AdaBoost` | sklearn | 0.1876 ± 0.0006 | 0.1478 ± 0.0003 | 0.7411 ± 0.0018 | - | - | - | 109.5 |
| `sklearn_BayesianRidge` | sklearn | 0.1638 ± 0.0000 | 0.1239 ± 0.0000 | 0.8026 ± 0.0000 | - | - | - | 0.4619 |
| `sklearn_DecisionTree` | sklearn | 0.1988 ± 0.0010 | 0.1452 ± 0.0004 | 0.7092 ± 0.0030 | - | - | - | 8.5714 |
| `sklearn_ElasticNet` | sklearn | 0.1709 ± 0.0000 | 0.1301 ± 0.0000 | 0.7852 ± 0.0000 | - | - | - | 1.0265 |
| `sklearn_ExtraTrees` | sklearn | 0.1656 ± 0.0001 | 0.1260 ± 0.0001 | 0.7981 ± 0.0002 | - | - | - | 23.3184 |
| `sklearn_GradientBoosting` | sklearn | **0.1632 ± 0.0000** | **0.1234 ± 0.0000** | **0.8041 ± 0.0000** | - | - | - | 400.2 |
| `sklearn_HistGradientBoosting` | sklearn | 0.1632 ± 0.0004 | 0.1237 ± 0.0005 | 0.8040 ± 0.0009 | - | - | - | 3.8786 |
| `sklearn_HuberRegressor` | sklearn | 0.1654 ± 0.0000 | 0.1252 ± 0.0000 | 0.7987 ± 0.0000 | - | - | - | 19.3795 |
| `sklearn_KNeighbors` | sklearn | 0.2022 ± 0.0000 | 0.1528 ± 0.0000 | 0.6991 ± 0.0000 | - | - | - | 0.0439 |
| `sklearn_Lasso` | sklearn | 0.1753 ± 0.0000 | 0.1340 ± 0.0000 | 0.7738 ± 0.0000 | - | - | - | 5.1039 |
| `sklearn_LinearRegression` | sklearn | 0.1652 ± 0.0000 | 0.1249 ± 0.0000 | 0.7992 ± 0.0000 | - | - | - | 0.2496 |
| `sklearn_MLPRegressor` | sklearn | 0.2079 ± 0.0020 | 0.1611 ± 0.0023 | 0.6819 ± 0.0062 | - | - | - | 20.7094 |
| `sklearn_RandomForest` | sklearn | 0.1660 ± 0.0001 | 0.1259 ± 0.0000 | 0.7972 ± 0.0002 | - | - | - | 70.9412 |
| `sklearn_Ridge` | sklearn | 0.1648 ± 0.0000 | 0.1247 ± 0.0000 | 0.8001 ± 0.0000 | - | - | - | 0.0678 |
| `sklearn_SVR` | sklearn | 0.1676 ± 0.0000 | 0.1269 ± 0.0000 | 0.7933 ± 0.0000 | - | - | - | 44.6404 |

##### sklearn_baselines / tabred_weather

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 1.9419 ± 0.0421 | 1.4689 ± 0.0270 | 0.7501 ± 0.0109 | 2.8376 ± 0.1820 | - | - | 7.4249 |
| `GaussianNLL_Linear` | torchregress | 1.7631 ± 0.0000 | 1.3120 ± 0.0000 | 0.7940 ± 0.0000 | **1.9863 ± 0.0000** | - | - | 0.2496 |
| `MSE` | torchregress | 2.0529 ± 0.0353 | 1.5564 ± 0.0330 | 0.7207 ± 0.0096 | - | - | - | 3.5141 |
| `MultiQuantile` | torchregress | 1.9206 ± 0.0109 | 1.4411 ± 0.0023 | 0.7556 ± 0.0028 | - | 0.5661 ± 0.0150 | 12.6577 ± 0.3256 | 5.8825 |
| `OLS` | torchregress | 2.0389 ± 0.1716 † | 1.5367 ± 0.1285 | 0.7233 ± 0.0472 | - | - | - | 0.0120 |
| `TabularMLP` | torchregress | 1.7325 ± 0.0105 | 1.2965 ± 0.0141 | 0.8011 ± 0.0024 | - | - | - | 5.2158 |
| `TabularMLP_Ens5` | torchregress | 1.6843 ± 0.0004 | 1.2567 ± 0.0027 | 0.8120 ± 0.0001 | - | - | - | 25.8918 |
| `sklearn_AdaBoost` | sklearn | 2.0578 ± 0.0147 | 1.5943 ± 0.0129 | 0.7194 ± 0.0040 | - | - | - | 6.4702 |
| `sklearn_BayesianRidge` | sklearn | 1.7702 ± 0.0000 | 1.3155 ± 0.0000 | 0.7924 ± 0.0000 | - | - | - | 0.0109 |
| `sklearn_DecisionTree` | sklearn | 2.2024 ± 0.0585 | 1.5950 ± 0.0212 | 0.6785 ± 0.0171 | - | - | - | 0.3932 |
| `sklearn_ElasticNet` | sklearn | 1.8557 ± 0.0000 | 1.3914 ± 0.0000 | 0.7718 ± 0.0000 | - | - | - | 0.0153 |
| `sklearn_ExtraTrees` | sklearn | 1.7731 ± 0.0028 | 1.3317 ± 0.0016 | 0.7917 ± 0.0006 | - | - | - | 1.5740 |
| `sklearn_GradientBoosting` | sklearn | 1.7001 ± 0.0021 | 1.2616 ± 0.0012 | 0.8085 ± 0.0005 | - | - | - | 22.8388 |
| `sklearn_HistGradientBoosting` | sklearn | **1.6742 ± 0.0000** | **1.2505 ± 0.0000** | **0.8143 ± 0.0000** | - | - | - | 0.5797 |
| `sklearn_HuberRegressor` | sklearn | 1.7631 ± 0.0000 | 1.3046 ± 0.0000 | 0.7940 ± 0.0000 | - | - | - | 0.6232 |
| `sklearn_KNeighbors` | sklearn | 2.0351 ± 0.0000 | 1.5725 ± 0.0000 | 0.7256 ± 0.0000 | - | - | - | 0.0019 |
| `sklearn_Lasso` | sklearn | 1.9459 ± 0.0000 | 1.4745 ± 0.0000 | 0.7491 ± 0.0000 | - | - | - | 0.0228 |
| `sklearn_LinearRegression` | sklearn | 1.7666 ± 0.0000 | 1.3157 ± 0.0000 | 0.7932 ± 0.0000 | - | - | - | 0.0096 |
| `sklearn_MLPRegressor` | sklearn | 2.1513 ± 0.0232 | 1.6367 ± 0.0180 | 0.6933 ± 0.0066 | - | - | - | 3.1286 |
| `sklearn_RandomForest` | sklearn | 1.7637 ± 0.0038 | 1.3222 ± 0.0038 | 0.7939 ± 0.0009 | - | - | - | 5.2693 |
| `sklearn_Ridge` | sklearn | 1.7636 ± 0.0000 | 1.3117 ± 0.0000 | 0.7939 ± 0.0000 | - | - | - | 0.0031 |
| `sklearn_SVR` | sklearn | 1.7501 ± 0.0000 | 1.2963 ± 0.0000 | 0.7971 ± 0.0000 | - | - | - | 2.9453 |

##### sklearn_baselines / zurich_delays

| Method | Library | RMSE | MAE | R2 | NLL | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GaussianNLL` | torchregress | 108.7 ± 0.6 | 59.5067 ± 0.6208 | 0.2021 ± 0.0085 | **5.9710 ± 0.0905** | - | - | 46.8181 |
| `GaussianNLL_Linear` | torchregress | 117.6 ± 0.0 | 63.3154 ± 0.0010 | 0.0662 ± 0.0000 | 6.2474 ± 0.0000 | - | - | 0.0474 |
| `MSE` | torchregress | 114.0 ± 1.4 | 60.3195 ± 0.1307 | 0.1223 ± 0.0215 | - | - | - | 26.1872 |
| `MultiQuantile` | torchregress | 109.2 ± 0.5 | 57.3302 ± 0.1551 | 0.1943 ± 0.0068 | - | 0.7693 ± 0.0131 | 402.9 ± 8.2 | 35.1271 |
| `OLS` | torchregress | 118.8 ± 2.1 | 64.2471 ± 1.6180 | 0.0467 ± 0.0338 | - | - | - | 0.0537 |
| `TabularMLP` | torchregress | 105.5 ± 0.6 | 58.1732 ± 0.6014 | 0.2477 ± 0.0087 | - | - | - | 50.7159 |
| `TabularMLP_Ens5` | torchregress | **103.1 ± 0.0** | **56.8826 ± 0.0535** | **0.2819 ± 0.0003** | - | - | - | 280.5 |
| `sklearn_AdaBoost` | sklearn | 143.7 ± 13.7 † | 82.9570 ± 7.7179 | -0.4034 ± 0.2699 | - | - | - | 8.8942 |
| `sklearn_BayesianRidge` | sklearn | 117.6 ± 0.0 | 63.1884 ± 0.0000 | 0.0659 ± 0.0000 | - | - | - | 0.0444 |
| `sklearn_DecisionTree` | sklearn | 161.4 ± 73.0 † | 61.5345 ± 1.1011 | -0.9997 ± 1.7984 | - | - | - | 0.2533 |
| `sklearn_ElasticNet` | sklearn | 119.9 ± 0.0 | 63.5844 ± 0.0000 | 0.0289 ± 0.0000 | - | - | - | 0.0105 |
| `sklearn_ExtraTrees` | sklearn | 121.2 ± 3.3 | 59.6710 ± 0.1049 | 0.0068 ± 0.0546 | - | - | - | 3.9383 |
| `sklearn_GradientBoosting` | sklearn | 163.6 ± 9.0 † | 60.7887 ± 0.0935 | -0.8117 ± 0.1961 | - | - | - | 16.9518 |
| `sklearn_HistGradientBoosting` | sklearn | 108.7 ± 0.2 | 60.2744 ± 0.0939 | 0.2019 ± 0.0032 | - | - | - | 0.7929 |
| `sklearn_HuberRegressor` | sklearn | 120.2 ± 0.0 | 59.1708 ± 0.0000 | 0.0236 ± 0.0000 | - | - | - | 0.7051 |
| `sklearn_KNeighbors` | sklearn | 123.4 ± 0.0 | 63.7756 ± 0.0000 | -0.0284 ± 0.0000 | - | - | - | 0.0145 |
| `sklearn_Lasso` | sklearn | 121.1 ± 0.0 | 64.4381 ± 0.0000 | 0.0097 ± 0.0000 | - | - | - | 0.0099 |
| `sklearn_LinearRegression` | sklearn | 117.6 ± 0.0 | 63.3105 ± 0.0000 | 0.0662 ± 0.0000 | - | - | - | 0.0334 |
| `sklearn_MLPRegressor` | sklearn | 113.5 ± 2.3 | 62.0244 ± 0.9577 | 0.1303 ± 0.0356 | - | - | - | 22.6045 |
| `sklearn_RandomForest` | sklearn | 128.1 ± 6.0 | 60.0812 ± 0.0763 | -0.1104 ± 0.1043 | - | - | - | 3.4574 |
| `sklearn_Ridge` | sklearn | 117.6 ± 0.0 | 63.3124 ± 0.0000 | 0.0662 ± 0.0000 | - | - | - | 0.0117 |
| `sklearn_SVR` | sklearn | 164.1 ± 0.0 | 114.6 ± 0.0 | -0.8178 ± 0.0000 | - | - | - | 25.3683 |

#### streaming_tta (tabular)

Group `tabular`; 3 dataset(s). Columns: RMSE ↓, Coverage ·, Width ·, IntervalScore ↓, AdaptGain ↑ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### streaming_tta / california_housing

| Method | Library | RMSE | Coverage | Width | IntervalScore | AdaptGain | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `COSAAdapted` | torchregress | 0.5014 ± 0.0063 | 0.9038 ± 0.0031 | 1.5159 ± 0.0202 | 2.4086 ± 0.0378 | 0.0011 ± 0.0022 | 1.1554 |
| `CausalHarnessCOSA` | torchregress | **0.5014 ± 0.0063** | 0.9038 ± 0.0031 | 1.5158 ± 0.0202 | 2.4086 ± 0.0378 | **0.0011 ± 0.0022** | 1.1158 |
| `CausalHarnessCOSA_Delay2` | torchregress | 0.5014 ± 0.0062 | 0.9036 ± 0.0034 | 1.5098 ± 0.0197 | 2.4014 ± 0.0347 | 0.0011 ± 0.0021 | 1.1087 |
| `CausalHarnessCOSA_Unlabeled` | torchregress | 0.5026 ± 0.0065 | 0.8987 ± 0.0018 | 1.4656 ± 0.0241 | 2.3931 ± 0.0469 | 9.934e-09 ± 1.721e-08 | 1.1032 |
| `Static` | torchregress | 0.5026 ± 0.0065 | 0.8987 ± 0.0018 | 1.4658 ± 0.0241 | **2.3931 ± 0.0469** | - | 1.3430 |

##### streaming_tta / diabetes

| Method | Library | RMSE | Coverage | Width | IntervalScore | AdaptGain | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `COSAAdapted` | torchregress | **0.6782 ± 0.0154** | 0.8747 ± 0.0157 | 2.1661 ± 0.0546 | 2.8891 ± 0.1181 | **7.669e-05 ± 5.243e-04** | 0.0950 |
| `CausalHarnessCOSA` | torchregress | **0.6782 ± 0.0154** | 0.8747 ± 0.0157 | 2.1660 ± 0.0545 | 2.8891 ± 0.1181 | **7.669e-05 ± 5.243e-04** | 0.0864 |
| `CausalHarnessCOSA_Delay2` | torchregress | 0.6803 ± 0.0149 | 0.8471 ± 0.0157 | 2.0555 ± 0.0462 | **2.8428 ± 0.1107** | -0.0020 ± 0.0005 | 0.0874 |
| `CausalHarnessCOSA_Unlabeled` | torchregress | 0.6783 ± 0.0154 | 0.8246 ± 0.0217 | 1.9226 ± 0.0450 | 2.9102 ± 0.0882 | 1.987e-08 ± 3.441e-08 | 0.0861 |
| `Static` | torchregress | 0.6783 ± 0.0154 | 0.8246 ± 0.0217 | 1.9228 ± 0.0450 | 2.9101 ± 0.0882 | - | 0.0912 |

##### streaming_tta / folktables_income

| Method | Library | RMSE | Coverage | Width | IntervalScore | AdaptGain | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `COSAAdapted` | torchregress | 0.9161 ± 0.0331 | 0.9439 ± 0.0014 | 3.0678 ± 0.1294 | 4.7461 ± 0.2056 | 5.777e-04 ± 2.127e-03 | 4.5741 |
| `CausalHarnessCOSA` | torchregress | 0.9161 ± 0.0331 | 0.9439 ± 0.0014 | 3.0675 ± 0.1294 | 4.7459 ± 0.2055 | 5.778e-04 ± 2.127e-03 | 4.4992 |
| `CausalHarnessCOSA_Delay2` | torchregress | **0.9160 ± 0.0333** | 0.9439 ± 0.0017 | 3.0635 ± 0.1285 | 4.7443 ± 0.2014 | **6.857e-04 ± 1.961e-03** | 5.1773 |
| `CausalHarnessCOSA_Unlabeled` | torchregress | 0.9167 ± 0.0352 | 0.8918 ± 0.0076 | 1.8999 ± 0.0162 | 4.4706 ± 0.1888 | -3.974e-08 ± 3.441e-08 | 4.9206 |
| `Static` | torchregress | 0.9167 ± 0.0352 | 0.8918 ± 0.0076 | 1.9001 ± 0.0162 | **4.4706 ± 0.1888** | - | 4.4369 |

#### stress_matrix (tabular)

Group `tabular`; 6 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑, NLL ↓, CRPS ↓, ECE ↓, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### stress_matrix / stress_imbalanced

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stress_imbalanced/BinReweighted_MLP` | torchregress | **0.4817 ± 0.0095** | **0.3535 ± 0.0134** | **0.7838 ± 0.0086** | - | - | - | - | 1.4840 |
| `stress_imbalanced/DensityWeighted_MLP` | torchregress | 0.5714 ± 0.0056 | 0.4343 ± 0.0037 | 0.6958 ± 0.0060 | - | - | - | - | 2.5220 |
| `stress_imbalanced/LDS_MLP` | torchregress | 0.5820 ± 0.0041 | 0.4441 ± 0.0041 | 0.6843 ± 0.0044 | - | - | - | - | 1.4023 |
| `stress_imbalanced/MSE_MLP` | torchregress | **0.4817 ± 0.0095** | **0.3535 ± 0.0134** | **0.7838 ± 0.0086** | - | - | - | - | 0.9903 |

##### stress_matrix / stress_noisy_inputs

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stress_noisy_inputs/InputNoise_MLP` | torchregress | 0.4863 ± 0.0080 | 0.3934 ± 0.0069 | 0.7848 ± 0.0071 | - | - | - | - | 1.0862 |
| `stress_noisy_inputs/MSE_MLP` | torchregress | **0.4135 ± 0.0046** | **0.3397 ± 0.0037** | **0.8444 ± 0.0034** | - | - | - | - | 1.0237 |

##### stress_matrix / stress_noisy_labels

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stress_noisy_labels/BetaNLL_MLP` | torchregress | 0.6169 ± 0.0075 | 0.4773 ± 0.0059 | 0.6508 ± 0.0085 | 1.0690 ± 0.0510 | 0.3467 ± 0.0059 | 0.0042 ± 0.0006 | 0.1056 ± 0.0293 | 1.6128 |
| `stress_noisy_labels/GaussianNLL_MLP` | torchregress | 0.6204 ± 0.0074 | 0.4816 ± 0.0023 | 0.6469 ± 0.0085 | 1.1584 ± 0.0804 | 0.3507 ± 0.0038 | 0.0038 ± 0.0006 | 0.1190 ± 0.0238 | 1.7320 |
| `stress_noisy_labels/NoisyTargetGaussianNLL_MLP` | torchregress | **0.6021 ± 0.0104** | **0.4618 ± 0.0078** | **0.6674 ± 0.0114** | **0.8767 ± 0.0141** | **0.3320 ± 0.0050** | **0.0030 ± 0.0008** | **0.0503 ± 0.0177** | 0.8936 |

##### stress_matrix / stress_ood

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stress_ood/DeepEnsemble_K10` | torchregress | 0.4616 ± 0.0104 | 0.3406 ± 0.0076 | 0.8000 ± 0.0090 | **0.6370 ± 0.0796** | 0.2466 ± 0.0071 | 0.0043 ± 0.0006 | **0.2839 ± 0.0157** | 11.2251 |
| `stress_ood/GaussianNLL_MLP` | torchregress | **0.4347 ± 0.0151** | **0.3123 ± 0.0099** | **0.8225 ± 0.0123** | 3.7152 ± 1.1924 | **0.2442 ± 0.0042** | 0.0034 ± 0.0000 | 0.3188 ± 0.0626 | 1.1036 |
| `stress_ood/MC_Dropout` | torchregress | 0.6794 ± 0.0394 † | 0.5396 ± 0.0339 | 0.5658 ± 0.0510 | 1.8226 ± 0.1619 | 0.3971 ± 0.0260 | **0.0026 ± 0.0007** | 0.3111 ± 0.0408 | 1.0853 |
| `stress_ood/luqbox_GaussianNLL` | lightning-uq-box | 0.4727 ± 0.0352 † | 0.3430 ± 0.0192 | 0.7895 ± 0.0316 | 7.5671 ± 2.8385 | 0.2769 ± 0.0099 | 0.0037 ± 0.0008 | 0.4098 ± 0.0533 | 4.2165 |
| `stress_ood/luqbox_MCDropout` | lightning-uq-box | 0.7663 ± 0.0911 † | 0.6161 ± 0.0772 | 0.4437 ± 0.1312 | 1.1330 ± 0.1146 | 0.4318 ± 0.0508 | 0.0061 ± 0.0004 | 0.3245 ± 0.0430 | 5.8023 |

##### stress_matrix / stress_semi_sup

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stress_semi_sup/SAGE_MLP` | torchregress | 0.5143 ± 0.0112 | 0.4056 ± 0.0067 | 0.7592 ± 0.0104 | **0.8492 ± 0.0315** | **0.2879 ± 0.0060** | **0.0034 ± 0.0012** | **0.0581 ± 0.0147** | 1.0660 |
| `stress_semi_sup/SupervisedOnly_MLP` | torchregress | **0.4489 ± 0.0045** | **0.3621 ± 0.0051** | **0.8166 ± 0.0037** | 5.4523 ± 2.7286 | 0.2902 ± 0.0059 | 0.0053 ± 0.0014 | 0.2639 ± 0.0134 | 0.8849 |

##### stress_matrix / stress_sparse_labels

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stress_sparse_labels/PseudoLabelNLL_MLP` | torchregress | 0.4157 ± 0.0331 † | 0.3374 ± 0.0324 | 0.8421 ± 0.0245 | **0.5889 ± 0.0600** | 0.2396 ± 0.0187 | 0.0080 ± 0.0009 | 0.2354 ± 0.0108 | 1.0860 |
| `stress_sparse_labels/SupervisedOnly_GaussianNLL` | torchregress | **0.3725 ± 0.0020** | **0.3049 ± 0.0045** | **0.8737 ± 0.0014** | 1.4049 ± 0.1538 | **0.2274 ± 0.0010** | **0.0036 ± 0.0009** | **0.1929 ± 0.0112** | 0.9171 |

#### test_time_adaptation (tabular)

Group `tabular`; 3 dataset(s). Columns: RMSE ↓, Coverage ·, Width ·, IntervalScore ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### test_time_adaptation / california_housing

| Method | Library | RMSE | Coverage | Width | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `FeatureNorm` | torchregress | 0.6579 ± 0.0075 | 0.8876 ± 0.0108 | 1.8388 ± 0.0361 | 2.7433 ± 0.1606 | 1.9357 |
| `JointTTA` | torchregress | 0.6235 ± 0.0232 | 0.9004 ± 0.0044 | 1.6245 ± 0.0518 | 2.5707 ± 0.1664 | 1.3434 |
| `JointTTA_LabelShiftEM` | torchregress | 0.6235 ± 0.0232 | 0.9002 ± 0.0030 | 1.6241 ± 0.0318 | 2.5696 ± 0.1661 | 1.2307 |
| `JointTTA_OTScore` | torchregress | 0.6235 ± 0.0232 | 0.9022 ± 0.0028 | 1.6304 ± 0.0360 | 2.5696 ± 0.1664 | 1.4786 |
| `SemiConformal` | torchregress | 0.6235 ± 0.0232 | 0.9011 ± 0.0037 | 1.6322 ± 0.0590 | 2.5442 ± 0.1796 | 1.2112 |
| `ShiftFactoredTransport` | torchregress | **0.6038 ± 0.0264** | 0.9513 ± 0.0077 | 2.0840 ± 0.1637 | 2.6465 ± 0.2157 | 1.5431 |
| `ShiftFactoredTransport+Conformal` | torchregress | 0.6236 ± 0.0232 | 0.9447 ± 0.0031 | 2.0540 ± 0.0636 | 2.6649 ± 0.1572 | 1.2418 |
| `ShiftInflator` | torchregress | 0.6038 ± 0.0264 | 0.9838 ± 0.0025 | 3.4942 ± 0.2695 | 3.6938 ± 0.3705 | 1.5729 |
| `Static` | torchregress | 0.6038 ± 0.0264 | 0.9104 ± 0.0062 | 1.6752 ± 0.1270 | **2.5117 ± 0.1837** | 1.9720 |
| `SubspaceAlign` | torchregress | 0.6211 ± 0.0216 | 0.8991 ± 0.0102 | 1.7381 ± 0.0380 | 2.6533 ± 0.2097 | 1.7272 |

##### test_time_adaptation / diabetes

| Method | Library | RMSE | Coverage | Width | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `FeatureNorm` | torchregress | 57.4755 ± 3.8166 † | 0.7794 ± 0.0528 | 136.4 ± 5.6 | 270.0 ± 32.5 | 0.1802 |
| `JointTTA` | torchregress | 54.3521 ± 5.7887 † | 0.8847 ± 0.0347 | 183.3 ± 22.0 | 228.5 ± 16.2 | 0.1466 |
| `JointTTA_LabelShiftEM` | torchregress | 54.3521 ± 5.7887 † | 0.9073 ± 0.0304 | 200.5 ± 6.3 | 228.2 ± 15.8 | 0.1454 |
| `JointTTA_OTScore` | torchregress | 54.3521 ± 5.7887 † | 0.9474 ± 0.0725 | 239.7 ± 39.8 | 259.1 ± 11.3 | 0.1454 |
| `SemiConformal` | torchregress | 54.3521 ± 5.7887 † | 0.9449 ± 0.0657 | 239.2 ± 37.3 | 257.9 ± 11.6 | 0.1403 |
| `ShiftFactoredTransport` | torchregress | 53.5804 ± 6.9521 † | 0.8446 ± 0.0441 | 162.3 ± 14.6 | **216.7 ± 23.6** | 0.1788 |
| `ShiftFactoredTransport+Conformal` | torchregress | 54.3495 ± 5.7917 † | 0.8446 ± 0.0157 | 158.0 ± 14.4 | 230.2 ± 16.1 | 0.1463 |
| `ShiftInflator` | torchregress | **53.5779 ± 6.9450** † | 0.9950 ± 0.0043 | 272.0 ± 25.0 | 272.5 ± 24.8 | 0.1788 |
| `Static` | torchregress | **53.5779 ± 6.9450** † | 0.7694 ± 0.0730 | 134.0 ± 12.1 | 239.3 ± 30.9 | 0.1906 |
| `SubspaceAlign` | torchregress | 57.6317 ± 2.6251 | 0.7845 ± 0.0489 | 137.2 ± 4.0 | 270.4 ± 23.7 | 0.1828 |

##### test_time_adaptation / folktables_income

| Method | Library | RMSE | Coverage | Width | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `FeatureNorm` | torchregress | **75944.6 ± 6260.5** † | 0.8337 ± 0.0554 | 3.066e+05 ± 2.971e+05 | 4.620e+05 ± 2.661e+05 | 6.3020 |
| `JointTTA` | torchregress | 76395.9 ± 6401.8 † | 0.8802 ± 0.0087 | 1.831e+05 ± 2.012e+04 | 3.101e+05 ± 3.445e+04 | 5.2076 |
| `JointTTA_LabelShiftEM` | torchregress | 76395.9 ± 6401.8 † | 0.8806 ± 0.0100 | 1.833e+05 ± 1.876e+04 | 3.101e+05 ± 3.399e+04 | 6.1575 |
| `JointTTA_OTScore` | torchregress | 76395.9 ± 6401.8 † | 0.8795 ± 0.0084 | 1.828e+05 ± 2.039e+04 | **3.100e+05 ± 3.453e+04** | 5.3815 |
| `SemiConformal` | torchregress | 76395.9 ± 6401.8 † | 0.8804 ± 0.0071 | 1.838e+05 ± 2.342e+04 | 3.102e+05 ± 3.556e+04 | 5.6632 |
| `ShiftFactoredTransport` | torchregress | 76491.9 ± 5470.4 † | 0.9247 ± 0.0306 | 2.393e+09 ± 4.145e+09 | 2.393e+09 ± 4.145e+09 | 6.2451 |
| `ShiftFactoredTransport+Conformal` | torchregress | 76418.3 ± 6437.3 † | 0.9087 ± 0.0220 | 1.945e+05 ± 1.347e+04 | 3.136e+05 ± 3.335e+04 | 5.1784 |
| `ShiftInflator` | torchregress | 76384.1 ± 5405.5 † | 0.9588 ± 0.0209 | 5.983e+09 ± 1.036e+10 | 5.983e+09 ± 1.036e+10 | 6.4244 |
| `Static` | torchregress | 76384.1 ± 5405.5 † | 0.8342 ± 0.0555 | 1.197e+09 ± 2.072e+09 | 1.197e+09 ± 2.072e+09 | 6.6654 |
| `SubspaceAlign` | torchregress | 76704.8 ± 5701.8 † | 0.8298 ± 0.0597 | 1.221e+09 ± 2.114e+09 | 1.221e+09 ± 2.114e+09 | 6.2419 |

#### tweedie (tabular)

Group `tabular`; 2 dataset(s). Columns: RMSE ↓, R2 ↑, TweedieDeviance ↓, ZeroFracPred · (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### tweedie / insurance_claims

| Method | Library | RMSE | R2 | TweedieDeviance | ZeroFracPred | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `BoxCox` | torchregress | 0.2429 ± 0.0000 | -0.0508 ± 0.0002 | non-finite (3) | 0.9973 ± 0.0003 | 6.2291 |
| `CompoundPoisson` | torchregress | 0.9905 ± 0.6059 † | -20.8309 ± 17.6735 | 187.2 ± 211.0 | 0.3498 ± 0.0189 | 9.8061 |
| `CompoundPoisson_Linear` | torchregress | **0.2361 ± 0.0000** | **0.0072 ± 0.0000** | **1.4131 ± 0.0000** | 0.0000 ± 0.0000 | 0.0236 |
| `LogT` | torchregress | 0.2429 ± 0.0000 | -0.0508 ± 0.0002 | non-finite (3) | 0.9973 ± 0.0003 | 6.2380 |
| `SqrtT` | torchregress | 0.2455 ± 0.0020 | -0.0737 ± 0.0172 | non-finite (3) | 0.3036 ± 0.0067 | 6.1235 |
| `Tweedie` | torchregress | 0.9905 ± 0.6059 † | -20.8309 ± 17.6735 | 187.2 ± 211.0 | 0.3498 ± 0.0189 | 10.2344 |
| `Tweedie_Linear` | torchregress | **0.2361 ± 0.0000** | **0.0072 ± 0.0000** | **1.4131 ± 0.0000** | 0.0000 ± 0.0000 | 0.0307 |
| `statsmodels_GLM` | statsmodels | 0.2361 ± 0.0000 | 0.0071 ± 0.0000 | 1.4133 ± 0.0000 | 0.0000 ± 0.0000 | 0.4928 |

Failed / skipped:

- `Gamma` (skipped, seed 0, 1, 2): skipped: Gamma requires strictly positive targets (support y > 0); y_train has 13323/14000 values <= 0. Not fitted: dro…
- `Gamma_Linear` (skipped, seed 0, 1, 2): skipped: Gamma_Linear requires strictly positive targets (support y > 0); y_train has 13323/14000 values <= 0. Not fitt…
- `InverseGaussian` (skipped, seed 0, 1, 2): skipped: InverseGaussian requires strictly positive targets (support y > 0); y_train has 13323/14000 values <= 0. Not f…

##### tweedie / synthetic_tweedie

| Method | Library | RMSE | R2 | TweedieDeviance | ZeroFracPred | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `BoxCox` | torchregress | 1.2690 ± 0.0224 | 0.4060 ± 0.0211 | 2.3987 ± 0.1515 | 0.0000 ± 0.0000 | 0.7694 |
| `CompoundPoisson` | torchregress | 1.1621 ± 0.0099 | 0.5020 ± 0.0085 | 0.7452 ± 0.0099 | 0.0000 ± 0.0000 | 0.8856 |
| `CompoundPoisson_Linear` | torchregress | 1.1060 ± 0.0000 | 0.5489 ± 0.0000 | **0.7174 ± 0.0000** | 0.0000 ± 0.0000 | 0.0064 |
| `LogT` | torchregress | 1.2690 ± 0.0224 | 0.4060 ± 0.0211 | 2.3987 ± 0.1515 | 0.0000 ± 0.0000 | 0.7626 |
| `SqrtT` | torchregress | 1.1654 ± 0.0083 | 0.4991 ± 0.0071 | 0.8198 ± 0.0186 | 0.0000 ± 0.0000 | 0.7713 |
| `Tweedie` | torchregress | 1.1621 ± 0.0099 | 0.5020 ± 0.0085 | 0.7452 ± 0.0099 | 0.0000 ± 0.0000 | 0.8999 |
| `Tweedie_Linear` | torchregress | 1.1060 ± 0.0000 | 0.5489 ± 0.0000 | **0.7174 ± 0.0000** | 0.0000 ± 0.0000 | 0.0146 |
| `statsmodels_GLM` | statsmodels | **1.1060 ± 0.0000** | **0.5489 ± 0.0000** | 0.7174 ± 0.0000 | 0.0000 ± 0.0000 | 0.0087 |

Failed / skipped:

- `Gamma` (skipped, seed 0, 1, 2): skipped: Gamma requires strictly positive targets (support y > 0); y_train has 79/1400 values <= 0. Not fitted: droppin…
- `Gamma_Linear` (skipped, seed 0, 1, 2): skipped: Gamma_Linear requires strictly positive targets (support y > 0); y_train has 79/1400 values <= 0. Not fitted: …
- `InverseGaussian` (skipped, seed 0, 1, 2): skipped: InverseGaussian requires strictly positive targets (support y > 0); y_train has 79/1400 values <= 0. Not fitte…

#### uq (tabular)

Group `tabular`; 5 dataset(s). Columns: RMSE ↓, MAE ↓, R2 ↑, NLL ↓, CRPS ↓, Coverage ·, Width ·, IntervalScore ↓, ECE ↓, PIT_KS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### uq / california_housing

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `DeepEnsemble_K10` | torchregress | 0.5815 ± 0.0018 | 0.3885 ± 0.0015 | 0.7461 ± 0.0016 | 0.6683 ± 0.0049 | 0.2807 ± 0.0004 | - | - | - | 0.0041 ± 0.0005 | 0.0798 ± 0.0097 | 14.1661 |
| `DeepEnsemble_K5` | torchregress | 0.5862 ± 0.0038 | 0.3952 ± 0.0037 | 0.7420 ± 0.0033 | 0.6828 ± 0.0191 | 0.2846 ± 0.0019 | - | - | - | 0.0037 ± 0.0004 | 0.0705 ± 0.0074 | 5.6749 |
| `DeepEnsemble_K5+CQR` | torchregress | 0.5862 ± 0.0038 | 0.3952 ± 0.0037 | 0.7420 ± 0.0033 | 0.6840 ± 0.0203 | 0.2840 ± 0.0021 | 0.9116 ± 0.0020 | 1.5830 ± 0.0252 | **2.3171 ± 0.0273** | 0.0036 ± 0.0003 | **0.0676 ± 0.0080** | 0.0014 |
| `DeepEnsemble_K5+VTS` | torchregress | 0.5862 ± 0.0038 | 0.3952 ± 0.0037 | 0.7420 ± 0.0033 | 0.6834 ± 0.0196 | 0.2849 ± 0.0022 | - | - | - | 0.0038 ± 0.0003 | 0.0724 ± 0.0108 | 0.0015 |
| `GaussianNLL` | torchregress | 0.6010 ± 0.0059 | 0.4003 ± 0.0041 | 0.7288 ± 0.0054 | 1.5526 ± 1.1115 | 0.2887 ± 0.0023 | - | - | - | 0.0036 ± 0.0002 | 0.0850 ± 0.0029 | 1.7566 |
| `MC_Dropout` | torchregress | 0.6451 ± 0.0030 | 0.4659 ± 0.0016 | 0.6876 ± 0.0029 | 2.7397 ± 0.1011 | 0.3602 ± 0.0014 | - | - | - | 0.0041 ± 0.0003 | 0.2381 ± 0.0031 | 1.4154 |
| `TabularMLP_Ens5_GaussianNLL` | torchregress | **0.5692 ± 0.0003** | **0.3749 ± 0.0014** | **0.7567 ± 0.0003** | **0.5955 ± 0.0048** | **0.2705 ± 0.0004** | - | - | - | 0.0044 ± 0.0000 | 0.0912 ± 0.0022 | 12.8187 |
| `TabularMLP_Ens5_GaussianNLL+VTS` | torchregress | 0.5793 ± 0.0042 | 0.3847 ± 0.0057 | 0.7480 ± 0.0037 | 0.6621 ± 0.0039 | 0.2784 ± 0.0026 | - | - | - | 0.0045 ± 0.0005 | 0.0831 ± 0.0172 | 0.0051 |
| `TabularMLP_Ens5_GaussianNLL_carve` | torchregress | 0.5793 ± 0.0042 | 0.3847 ± 0.0057 | 0.7480 ± 0.0037 | 0.6632 ± 0.0020 | 0.2783 ± 0.0020 | - | - | - | 0.0045 ± 0.0002 | 0.0831 ± 0.0154 | 9.1314 |
| `TabularMLP_GaussianNLL` | torchregress | 0.5740 ± 0.0056 | 0.3807 ± 0.0063 | 0.7526 ± 0.0048 | 0.6155 ± 0.0143 | 0.2743 ± 0.0033 | - | - | - | 0.0044 ± 0.0000 | 0.0882 ± 0.0276 | 2.5223 |
| `sklearn_GP` | sklearn | 0.5966 ± 0.0023 | 0.4040 ± 0.0039 | 0.7328 ± 0.0020 | 0.8520 ± 0.0043 | 0.3051 ± 0.0022 | - | - | - | 0.0039 ± 0.0004 | 0.0930 ± 0.0059 | 10.9060 |
| `sklearn_QuantileGBR` | sklearn | 0.5941 ± 0.0026 | 0.3819 ± 0.0012 | 0.7350 ± 0.0023 | - | - | 0.8533 ± 0.0018 | 1.8092 ± 0.0274 | 2.5517 ± 0.0296 | **0.0032 ± 0.0006** | 0.1574 ± 0.0085 | 2.6883 |

##### uq / diabetes

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `DeepEnsemble_K10` | torchregress | 51.9747 ± 0.0517 | 41.0397 ± 0.0741 | 0.4996 ± 0.0010 | **5.3291 ± 0.0073** | 29.0353 ± 0.0173 | - | - | - | 0.0055 ± 0.0007 | 0.0792 ± 0.0042 | 2.1006 |
| `DeepEnsemble_K5` | torchregress | 52.0375 ± 0.3654 | 41.2523 ± 0.3905 | 0.4984 ± 0.0070 | 5.3409 ± 0.0147 | 29.1059 ± 0.4224 | - | - | - | **0.0041 ± 0.0021** | 0.0920 ± 0.0287 | 0.8545 |
| `DeepEnsemble_K5+CQR` | torchregress | 52.0375 ± 0.3654 | 41.2523 ± 0.3905 | 0.4984 ± 0.0070 | 5.3449 ± 0.0161 | 29.0629 ± 0.3717 | 0.9348 ± 0.0674 | 182.6 ± 29.6 | **206.2 ± 3.3** | 0.0048 ± 0.0016 | 0.0711 ± 0.0083 | 0.0028 |
| `DeepEnsemble_K5+VTS` | torchregress | 52.0375 ± 0.3654 | 41.2523 ± 0.3905 | 0.4984 ± 0.0070 | 5.3292 ± 0.0099 | **28.9509 ± 0.3270** | - | - | - | 0.0046 ± 0.0016 | **0.0639 ± 0.0143** | 0.0016 |
| `GaussianNLL` | torchregress | 52.6474 ± 0.6842 | 41.3762 ± 0.3207 | 0.4865 ± 0.0133 | 5.3964 ± 0.0313 | 29.6126 ± 0.3868 | - | - | - | 0.0070 ± 0.0019 | 0.0859 ± 0.0113 | 0.2228 |
| `MC_Dropout` | torchregress | 53.4957 ± 0.4148 | 42.8157 ± 0.5435 | 0.4699 ± 0.0082 | 10.8605 ± 0.6345 | 35.3800 ± 0.2450 | - | - | - | 0.0090 ± 0.0040 | 0.2397 ± 0.0111 | 0.1850 |
| `TabularMLP_Ens5_GaussianNLL` | torchregress | 52.6170 ± 0.2189 | **40.7056 ± 0.3095** | 0.4871 ± 0.0043 | 5.3378 ± 0.0041 | 29.1368 ± 0.1204 | - | - | - | 0.0063 ± 0.0001 | 0.0850 ± 0.0067 | 1.7577 |
| `TabularMLP_Ens5_GaussianNLL+VTS` | torchregress | **51.5593 ± 0.3192** | 40.7962 ± 0.8676 | **0.5075 ± 0.0061** | 5.3419 ± 0.0112 | 28.9650 ± 0.3120 | - | - | - | 0.0079 ± 0.0005 | 0.0695 ± 0.0162 | 0.0040 |
| `TabularMLP_Ens5_GaussianNLL_carve` | torchregress | **51.5593 ± 0.3192** | 40.7962 ± 0.8676 | **0.5075 ± 0.0061** | 5.3473 ± 0.0112 | 28.9606 ± 0.1856 | - | - | - | 0.0082 ± 0.0005 | 0.0665 ± 0.0142 | 0.9491 |
| `TabularMLP_GaussianNLL` | torchregress | 54.6643 ± 1.7041 | 42.9190 ± 1.8295 | 0.4461 ± 0.0343 | 5.3956 ± 0.0460 | 30.5030 ± 1.1605 | - | - | - | 0.0046 ± 0.0009 | 0.0917 ± 0.0442 | 0.3284 |
| `sklearn_GP` | sklearn | 51.8194 ± 0.0000 | 40.9907 ± 0.0000 | 0.5026 ± 0.0000 | 5.3728 ± 0.0000 | 29.3299 ± 0.0000 | - | - | - | 0.0091 ± 0.0000 | 0.0652 ± 0.0000 | 0.2755 |
| `sklearn_QuantileGBR` | sklearn | 53.8687 ± 0.8520 | 43.4062 ± 0.7018 | 0.4624 ± 0.0170 | - | - | 0.8296 ± 0.0087 | 165.3 ± 7.6 | 220.8 ± 6.7 | 0.0076 ± 0.0012 | 0.1412 ± 0.0177 | 0.7287 |

##### uq / make_regression_synthetic

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `DeepEnsemble_K10` | torchregress | 15.3029 ± 0.0545 | 12.1728 ± 0.0374 | 0.9881 ± 0.0001 | 4.1535 ± 0.0047 | 8.6380 ± 0.0317 | - | - | - | 0.0017 ± 0.0002 | **0.0211 ± 0.0080** | 14.1048 |
| `DeepEnsemble_K5` | torchregress | 15.5605 ± 0.0890 | 12.3612 ± 0.1058 | 0.9877 ± 0.0001 | 4.1752 ± 0.0080 | 8.7911 ± 0.0488 | - | - | - | 0.0022 ± 0.0004 | 0.0408 ± 0.0249 | 5.3727 |
| `DeepEnsemble_K5+CQR` | torchregress | 15.5605 ± 0.0890 | 12.3612 ± 0.1058 | 0.9877 ± 0.0001 | 4.1745 ± 0.0071 | 8.7951 ± 0.0467 | 0.9140 ± 0.0085 | 53.7063 ± 0.8091 | **65.0150 ± 0.2746** | 0.0022 ± 0.0004 | 0.0463 ± 0.0236 | 0.0017 |
| `DeepEnsemble_K5+VTS` | torchregress | 15.5605 ± 0.0890 | 12.3612 ± 0.1058 | 0.9877 ± 0.0001 | 4.1759 ± 0.0071 | 8.7980 ± 0.0452 | - | - | - | 0.0022 ± 0.0004 | 0.0465 ± 0.0243 | 0.0017 |
| `GaussianNLL` | torchregress | 16.3460 ± 0.2297 | 12.9789 ± 0.2033 | 0.9865 ± 0.0004 | 4.2957 ± 0.0374 | 9.3204 ± 0.1477 | - | - | - | 0.0024 ± 0.0010 | 0.0554 ± 0.0169 | 1.6630 |
| `MC_Dropout` | torchregress | 21.8786 ± 2.3670 † | 16.7417 ± 1.7799 | 0.9756 ± 0.0054 | 4.5424 ± 0.0360 | 12.6775 ± 0.7824 | - | - | - | 0.0030 ± 0.0003 | 0.1072 ± 0.0037 | 1.4280 |
| `TabularMLP_Ens5_GaussianNLL` | torchregress | 15.3449 ± 0.0406 | 12.2275 ± 0.0324 | 0.9881 ± 0.0001 | 4.1649 ± 0.0028 | 8.6962 ± 0.0255 | - | - | - | **0.0017 ± 0.0003** | 0.0291 ± 0.0005 | 13.1441 |
| `TabularMLP_Ens5_GaussianNLL+VTS` | torchregress | 15.3963 ± 0.1035 | 12.2660 ± 0.1088 | 0.9880 ± 0.0002 | 4.1663 ± 0.0034 | 8.7200 ± 0.0503 | - | - | - | 0.0019 ± 0.0003 | 0.0283 ± 0.0050 | 0.0069 |
| `TabularMLP_Ens5_GaussianNLL_carve` | torchregress | 15.3963 ± 0.1035 | 12.2660 ± 0.1088 | 0.9880 ± 0.0002 | 4.1675 ± 0.0039 | 8.7242 ± 0.0557 | - | - | - | 0.0019 ± 0.0003 | 0.0289 ± 0.0009 | 10.4588 |
| `TabularMLP_GaussianNLL` | torchregress | 15.4913 ± 0.0507 | 12.3348 ± 0.0505 | 0.9878 ± 0.0001 | 4.1734 ± 0.0046 | 8.7736 ± 0.0310 | - | - | - | 0.0019 ± 0.0003 | 0.0471 ± 0.0156 | 2.5988 |
| `sklearn_GP` | sklearn | **14.9268 ± 0.0020** | **11.9317 ± 0.0076** | **0.9887 ± 0.0000** | **4.1230 ± 0.0005** | **8.4284 ± 0.0022** | - | - | - | 0.0018 ± 0.0004 | 0.0217 ± 0.0082 | 13.1509 |
| `sklearn_QuantileGBR` | sklearn | 22.6202 ± 0.4565 | 17.1931 ± 0.2518 | 0.9741 ± 0.0010 | - | - | 0.8304 ± 0.0015 | 103.3 ± 2.7 | 144.6 ± 3.2 | 0.0038 ± 0.0002 | 0.0979 ± 0.0013 | 2.7892 |

##### uq / synthetic_hetero

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `DeepEnsemble_K10` | torchregress | 0.3663 ± 0.0021 | 0.2783 ± 0.0018 | 0.8425 ± 0.0018 | 0.3501 ± 0.0125 | 0.1983 ± 0.0015 | - | - | - | 0.0030 ± 0.0008 | **0.0273 ± 0.0065** | 6.5948 |
| `DeepEnsemble_K5` | torchregress | 0.3732 ± 0.0086 | 0.2836 ± 0.0064 | 0.8365 ± 0.0075 | 0.3934 ± 0.0217 | 0.2027 ± 0.0047 | - | - | - | 0.0027 ± 0.0011 | 0.0296 ± 0.0054 | 2.7284 |
| `DeepEnsemble_K5+CQR` | torchregress | 0.3732 ± 0.0086 | 0.2836 ± 0.0064 | 0.8365 ± 0.0075 | 0.3955 ± 0.0059 | 0.2037 ± 0.0034 | 0.9244 ± 0.0171 | 1.3160 ± 0.0713 | **1.5837 ± 0.0013** | 0.0028 ± 0.0009 | 0.0536 ± 0.0205 | 0.0013 |
| `DeepEnsemble_K5+VTS` | torchregress | 0.3732 ± 0.0086 | 0.2836 ± 0.0064 | 0.8365 ± 0.0075 | 0.3946 ± 0.0140 | 0.2036 ± 0.0041 | - | - | - | 0.0026 ± 0.0011 | 0.0514 ± 0.0108 | 0.0014 |
| `GaussianNLL` | torchregress | 0.3747 ± 0.0048 | 0.2861 ± 0.0035 | 0.8351 ± 0.0042 | 0.4435 ± 0.0742 | 0.2052 ± 0.0042 | - | - | - | 0.0038 ± 0.0009 | 0.0547 ± 0.0188 | 0.6690 |
| `MC_Dropout` | torchregress | 0.4000 ± 0.0044 | 0.3088 ± 0.0051 | 0.8122 ± 0.0042 | 1.2769 ± 0.1424 | 0.2332 ± 0.0040 | - | - | - | 0.0034 ± 0.0011 | 0.1335 ± 0.0125 | 0.5618 |
| `TabularMLP_Ens5_GaussianNLL` | torchregress | **0.3577 ± 0.0002** | **0.2723 ± 0.0006** | **0.8498 ± 0.0002** | **0.2801 ± 0.0039** | **0.1919 ± 0.0003** | - | - | - | 0.0031 ± 0.0001 | 0.0363 ± 0.0032 | 4.2001 |
| `TabularMLP_Ens5_GaussianNLL+VTS` | torchregress | 0.3615 ± 0.0005 | 0.2750 ± 0.0007 | 0.8466 ± 0.0004 | 0.2965 ± 0.0089 | 0.1940 ± 0.0006 | - | - | - | 0.0027 ± 0.0006 | 0.0386 ± 0.0172 | 0.0047 |
| `TabularMLP_Ens5_GaussianNLL_carve` | torchregress | 0.3615 ± 0.0005 | 0.2750 ± 0.0007 | 0.8466 ± 0.0004 | 0.2967 ± 0.0074 | 0.1940 ± 0.0004 | - | - | - | **0.0025 ± 0.0006** | 0.0390 ± 0.0177 | 3.5704 |
| `TabularMLP_GaussianNLL` | torchregress | 0.3648 ± 0.0039 | 0.2783 ± 0.0031 | 0.8438 ± 0.0033 | 0.3237 ± 0.0215 | 0.1970 ± 0.0024 | - | - | - | 0.0027 ± 0.0012 | 0.0595 ± 0.0227 | 0.9288 |
| `sklearn_GP` | sklearn | 0.3808 ± 0.0000 | 0.2918 ± 0.0000 | 0.8297 ± 0.0000 | 0.4468 ± 0.0000 | 0.2108 ± 0.0000 | - | - | - | 0.0027 ± 0.0000 | 0.0422 ± 0.0000 | 4.5582 |
| `sklearn_QuantileGBR` | sklearn | 0.3744 ± 0.0027 | 0.2821 ± 0.0033 | 0.8354 ± 0.0024 | - | - | 0.8383 ± 0.0017 | 1.2022 ± 0.0130 | 1.7322 ± 0.0122 | 0.0038 ± 0.0009 | 0.0834 ± 0.0029 | 1.3563 |

##### uq / synthetic_sinusoidal

| Method | Library | RMSE | MAE | R2 | NLL | CRPS | Coverage | Width | IntervalScore | ECE | PIT_KS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `DeepEnsemble_K10` | torchregress | 0.7303 ± 0.0018 | 0.6245 ± 0.0022 | 0.1048 ± 0.0044 | 1.0946 ± 0.0026 | 0.4213 ± 0.0012 | - | - | - | 0.0039 ± 0.0003 | 0.0642 ± 0.0023 | 5.1151 |
| `DeepEnsemble_K5` | torchregress | 0.7353 ± 0.0026 | 0.6288 ± 0.0025 | 0.0924 ± 0.0064 | 1.1054 ± 0.0033 | 0.4250 ± 0.0015 | - | - | - | 0.0037 ± 0.0005 | 0.0642 ± 0.0043 | 2.1040 |
| `DeepEnsemble_K5+CQR` | torchregress | 0.7353 ± 0.0026 | 0.6288 ± 0.0025 | 0.0924 ± 0.0064 | 1.1064 ± 0.0043 | 0.4254 ± 0.0018 | 0.9061 ± 0.0082 | 2.3178 ± 0.0187 | 2.6595 ± 0.0148 | 0.0036 ± 0.0005 | 0.0672 ± 0.0066 | 0.0020 |
| `DeepEnsemble_K5+VTS` | torchregress | 0.7353 ± 0.0026 | 0.6288 ± 0.0025 | 0.0924 ± 0.0064 | 1.1058 ± 0.0024 | 0.4240 ± 0.0017 | - | - | - | 0.0040 ± 0.0005 | 0.0543 ± 0.0061 | 0.0010 |
| `GaussianNLL` | torchregress | 0.7262 ± 0.0116 | 0.6189 ± 0.0138 | 0.1147 ± 0.0283 | 1.0827 ± 0.0258 | 0.4182 ± 0.0085 | - | - | - | **0.0029 ± 0.0005** | 0.0652 ± 0.0032 | 0.5180 |
| `MC_Dropout` | torchregress | 0.7443 ± 0.0009 | 0.6425 ± 0.0006 | 0.0702 ± 0.0022 | 43.1071 ± 10.8871 | 0.5887 ± 0.0035 | - | - | - | 0.0220 ± 0.0018 | 0.4092 ± 0.0057 | 0.3973 |
| `TabularMLP_Ens5_GaussianNLL` | torchregress | **0.2489 ± 0.0047** | 0.2032 ± 0.0042 | **0.8960 ± 0.0039** | **0.0647 ± 0.0182** | **0.1430 ± 0.0027** | - | - | - | 0.0054 ± 0.0002 | 0.0614 ± 0.0011 | 5.4510 |
| `TabularMLP_Ens5_GaussianNLL+VTS` | torchregress | 0.2689 ± 0.0033 | 0.2176 ± 0.0020 | 0.8786 ± 0.0030 | 0.1163 ± 0.0236 | 0.1535 ± 0.0023 | - | - | - | 0.0033 ± 0.0010 | **0.0352 ± 0.0075** | 0.0040 |
| `TabularMLP_Ens5_GaussianNLL_carve` | torchregress | 0.2689 ± 0.0033 | 0.2176 ± 0.0020 | 0.8786 ± 0.0030 | 0.1531 ± 0.0322 | 0.1554 ± 0.0033 | - | - | - | 0.0048 ± 0.0004 | 0.0622 ± 0.0177 | 4.4239 |
| `TabularMLP_GaussianNLL` | torchregress | 0.2741 ± 0.0067 | 0.2218 ± 0.0039 | 0.8738 ± 0.0061 | 0.1147 ± 0.0187 | 0.1551 ± 0.0031 | - | - | - | 0.0038 ± 0.0004 | 0.0501 ± 0.0078 | 1.0279 |
| `sklearn_GP` | sklearn | 0.7205 ± 0.0399 † | 0.6103 ± 0.0510 | 0.1269 ± 0.0949 | 1.0808 ± 0.0726 | 0.4146 ± 0.0283 | - | - | - | 0.0035 ± 0.0000 | 0.0553 ± 0.0099 | 4.7264 |
| `sklearn_QuantileGBR` | sklearn | 0.2499 ± 0.0068 | **0.2026 ± 0.0062** | 0.8952 ± 0.0057 | - | - | 0.8628 ± 0.0100 | 1.5256 ± 0.0139 | **1.8946 ± 0.0326** | 0.0039 ± 0.0003 | 0.0832 ± 0.0062 | 0.7341 |

#### uq_layer (tabular)

Group `tabular`; 10 dataset(s). Columns: Coverage ·, Width ·, IntervalScore ↓, RMSE ↓, CRPS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: IntervalScore.

##### uq_layer / california_housing

| Method | Library | Coverage | Width | IntervalScore | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | 0.9124 ± 0.0098 | 1.6535 ± 0.0910 | 2.4059 ± 0.0299 | 0.5920 ± 0.0038 | - | 6.2399 |
| `MAPIE_CQR+LightGBM` | MAPIE | 0.9049 ± 0.0083 | 1.7488 ± 0.0443 | 2.4918 ± 0.0310 | 0.5414 ± 0.0029 | - | 0.5587 |
| `MAPIE_CQR+RF` | MAPIE | 0.9098 ± 0.0170 | 1.6373 ± 0.0921 | 2.3999 ± 0.0353 | 0.5986 ± 0.0028 | - | 1.4003 |
| `MAPIE_CQR+TabularMLP` | MAPIE | 0.9098 ± 0.0183 | 1.5819 ± 0.0575 | 2.3323 ± 0.0264 | 0.5800 ± 0.0035 | - | 9.2319 |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | 0.9136 ± 0.0078 | 1.7246 ± 0.0278 | 2.3767 ± 0.0325 | 0.5920 ± 0.0038 | - | 6.2399 |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | 0.9078 ± 0.0114 | 1.8458 ± 0.0485 | 2.4873 ± 0.0321 | 0.5414 ± 0.0029 | - | 0.5587 |
| `MAPIE_CQR_asym+RF` | MAPIE | 0.9078 ± 0.0113 | 1.6740 ± 0.0568 | 2.3988 ± 0.0277 | 0.5986 ± 0.0028 | - | 1.4004 |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | 0.9073 ± 0.0116 | 1.6894 ± 0.0652 | **2.3100 ± 0.0378** | 0.5800 ± 0.0035 | - | 9.2320 |
| `MAPIE_CVPlus+LightGBM` | MAPIE | 0.9111 ± 0.0023 | 1.5881 ± 0.0117 | 2.4956 ± 0.0119 | **0.5295 ± 0.0007** | - | 1.5737 |
| `MAPIE_CVPlus+RF` | MAPIE | 0.9104 ± 0.0008 | 1.8197 ± 0.0075 | 2.7675 ± 0.0131 | 0.5869 ± 0.0005 | - | 10.9390 |
| `MAPIE_Split+DeepEnsK5` | MAPIE | 0.9138 ± 0.0060 | 1.8909 ± 0.0441 | 2.8208 ± 0.0165 | 0.5920 ± 0.0038 | - | 6.2398 |
| `MAPIE_Split+LightGBM` | MAPIE | 0.9036 ± 0.0113 | 1.6116 ± 0.0585 | 2.5854 ± 0.0235 | 0.5414 ± 0.0029 | - | 0.5596 |
| `MAPIE_Split+RF` | MAPIE | 0.9093 ± 0.0068 | 1.8372 ± 0.0855 | 2.8344 ± 0.0014 | 0.5986 ± 0.0028 | - | 1.4001 |
| `MAPIE_Split+TabularMLP` | MAPIE | 0.9029 ± 0.0091 | 1.7687 ± 0.0748 | 2.8026 ± 0.0193 | 0.5800 ± 0.0035 | - | 9.2319 |
| `Raw+DeepEnsK5` | torchregress | 0.9207 ± 0.0013 | 1.7162 ± 0.0556 | 2.4149 ± 0.0230 | 0.5920 ± 0.0038 | 0.2899 ± 0.0027 | 6.2386 |
| `Raw+LightGBM` | lightgbm | 0.8016 ± 0.0056 | 1.5281 ± 0.0147 | 2.5642 ± 0.0417 | 0.5414 ± 0.0029 | **0.2611 ± 0.0002** | 0.5574 |
| `Raw+RF` | sklearn | 0.8309 ± 0.0034 | 1.3873 ± 0.0204 | 2.4463 ± 0.0290 | 0.5986 ± 0.0028 | 0.2908 ± 0.0020 | 1.3991 |
| `Raw+TabularMLP` | torchregress | 0.9242 ± 0.0014 | 1.6484 ± 0.0240 | 2.3390 ± 0.0196 | 0.5800 ± 0.0035 | 0.2785 ± 0.0029 | 9.2307 |
| `crepes_Normalized+DeepEnsK5` | crepes | 0.9122 ± 0.0080 | 1.6432 ± 0.0903 | 2.4076 ± 0.0257 | 0.5920 ± 0.0038 | - | 6.2387 |
| `crepes_Normalized+LightGBM` | crepes | 0.9102 ± 0.0117 | 1.5377 ± 0.0696 | 2.3359 ± 0.0543 | 0.5414 ± 0.0029 | - | 0.5575 |
| `crepes_Normalized+RF` | crepes | 0.9053 ± 0.0181 | 1.5950 ± 0.1264 | 2.4500 ± 0.0395 | 0.5986 ± 0.0028 | - | 1.3991 |
| `crepes_Normalized+TabularMLP` | crepes | 0.9122 ± 0.0174 | 1.5789 ± 0.0684 | 2.3355 ± 0.0241 | 0.5800 ± 0.0035 | - | 9.2307 |
| `crepes_Split+DeepEnsK5` | crepes | 0.9138 ± 0.0060 | 1.8909 ± 0.0441 | 2.8208 ± 0.0165 | 0.5920 ± 0.0038 | - | 6.2387 |
| `crepes_Split+LightGBM` | crepes | 0.9036 ± 0.0113 | 1.6116 ± 0.0585 | 2.5854 ± 0.0235 | 0.5414 ± 0.0029 | - | 0.5575 |
| `crepes_Split+RF` | crepes | 0.9093 ± 0.0068 | 1.8372 ± 0.0855 | 2.8344 ± 0.0014 | 0.5986 ± 0.0028 | - | 1.3992 |
| `crepes_Split+TabularMLP` | crepes | 0.9029 ± 0.0091 | 1.7687 ± 0.0748 | 2.8026 ± 0.0193 | 0.5800 ± 0.0035 | - | 9.2308 |
| `tr_CQR+DeepEnsK5` | torchregress | 0.9124 ± 0.0098 | 1.6535 ± 0.0910 | 2.4059 ± 0.0299 | 0.5920 ± 0.0038 | - | 6.2389 |
| `tr_CQR+LightGBM` | torchregress | 0.9049 ± 0.0083 | 1.7488 ± 0.0443 | 2.4918 ± 0.0310 | 0.5414 ± 0.0029 | - | 0.5589 |
| `tr_CQR+RF` | torchregress | 0.9098 ± 0.0170 | 1.6373 ± 0.0921 | 2.3999 ± 0.0353 | 0.5986 ± 0.0028 | - | 1.3997 |
| `tr_CQR+TabularMLP` | torchregress | 0.9098 ± 0.0183 | 1.5819 ± 0.0575 | 2.3323 ± 0.0264 | 0.5800 ± 0.0035 | - | 9.2309 |
| `tr_CVPlus+LightGBM` | torchregress | 0.9111 ± 0.0023 | 1.5881 ± 0.0117 | 2.4956 ± 0.0119 | **0.5295 ± 0.0007** | - | 1.4287 |
| `tr_CVPlus+RF` | torchregress | 0.9104 ± 0.0008 | 1.8197 ± 0.0075 | 2.7675 ± 0.0131 | 0.5869 ± 0.0005 | - | 9.1519 |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | 0.9347 ± 0.0013 | 1.8438 ± 0.0492 | 2.4897 ± 0.0312 | 0.5965 ± 0.0049 | 0.2928 ± 0.0036 | 6.2396 |
| `tr_IsoMean+VTS+LightGBM` | torchregress | 0.9324 ± 0.0126 | 1.8055 ± 0.1742 | 2.4581 ± 0.1158 | 0.5517 ± 0.0014 | 0.2716 ± 0.0037 | 0.5587 |
| `tr_IsoMean+VTS+RF` | torchregress | 0.9300 ± 0.0217 | 1.9331 ± 0.2779 | 2.6086 ± 0.1609 † | 0.6091 ± 0.0058 | 0.2990 ± 0.0088 | 1.4003 |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | 0.9373 ± 0.0024 | 1.8393 ± 0.0612 | 2.4244 ± 0.0281 | 0.5892 ± 0.0062 | 0.2859 ± 0.0055 | 9.2316 |
| `tr_Normalized+DeepEnsK5` | torchregress | 0.9122 ± 0.0080 | 1.6432 ± 0.0903 | 2.4076 ± 0.0257 | 0.5920 ± 0.0038 | - | 6.2388 |
| `tr_Normalized+LightGBM` | torchregress | 0.9102 ± 0.0117 | 1.5377 ± 0.0696 | 2.3359 ± 0.0543 | 0.5414 ± 0.0029 | - | 0.5576 |
| `tr_Normalized+RF` | torchregress | 0.9053 ± 0.0181 | 1.5950 ± 0.1264 | 2.4500 ± 0.0395 | 0.5986 ± 0.0028 | - | 1.3993 |
| `tr_Normalized+TabularMLP` | torchregress | 0.9122 ± 0.0174 | 1.5789 ± 0.0684 | 2.3355 ± 0.0241 | 0.5800 ± 0.0035 | - | 9.2309 |
| `tr_Split+DeepEnsK5` | torchregress | 0.9138 ± 0.0060 | 1.8909 ± 0.0441 | 2.8208 ± 0.0165 | 0.5920 ± 0.0038 | - | 6.2388 |
| `tr_Split+LightGBM` | torchregress | 0.9036 ± 0.0113 | 1.6116 ± 0.0585 | 2.5854 ± 0.0235 | 0.5414 ± 0.0029 | - | 0.5577 |
| `tr_Split+RF` | torchregress | 0.9093 ± 0.0068 | 1.8372 ± 0.0855 | 2.8344 ± 0.0014 | 0.5986 ± 0.0028 | - | 1.3993 |
| `tr_Split+TabularMLP` | torchregress | 0.9029 ± 0.0091 | 1.7687 ± 0.0748 | 2.8026 ± 0.0193 | 0.5800 ± 0.0035 | - | 9.2309 |
| `tr_VTS+DeepEnsK5` | torchregress | 0.9364 ± 0.0064 | 1.8569 ± 0.0568 | 2.4515 ± 0.0326 | 0.5920 ± 0.0038 | 0.2913 ± 0.0025 | 6.2390 |
| `tr_VTS+LightGBM` | torchregress | 0.9391 ± 0.0100 | 1.8184 ± 0.1750 | 2.4382 ± 0.1095 | 0.5414 ± 0.0029 | 0.2661 ± 0.0043 | 0.5582 |
| `tr_VTS+RF` | torchregress | 0.9347 ± 0.0151 | 1.8683 ± 0.2180 | 2.5244 ± 0.1276 † | 0.5986 ± 0.0028 | 0.2930 ± 0.0050 | 1.4000 |
| `tr_VTS+TabularMLP` | torchregress | 0.9404 ± 0.0081 | 1.7912 ± 0.0805 | 2.3813 ± 0.0520 | 0.5800 ± 0.0035 | 0.2801 ± 0.0032 | 9.2311 |

Failed / skipped:

- `MAPIE_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=3500 models (> 500)
- `tr_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=3500 models (> 500)

##### uq_layer / diabetes

| Method | Library | Coverage | Width | IntervalScore | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | 0.9524 ± 0.0230 | 200.3 ± 14.0 | 211.1 ± 12.9 † | 53.3180 ± 1.7121 | - | 0.6240 |
| `MAPIE_CQR+LightGBM` | MAPIE | 0.9424 ± 0.0189 | 200.8 ± 9.6 | 220.4 ± 3.4 | 57.6029 ± 2.0502 | - | 0.1437 |
| `MAPIE_CQR+RF` | MAPIE | 0.9524 ± 0.0264 | 198.9 ± 18.7 | 215.9 ± 11.7 † | 53.1818 ± 1.0729 | - | 0.1751 |
| `MAPIE_CQR+TabularMLP` | MAPIE | 0.9449 ± 0.0313 | 192.6 ± 18.6 | 215.6 ± 3.3 | 52.3900 ± 0.0510 | - | 0.7404 |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | 0.9398 ± 0.0526 | 203.6 ± 24.2 | 219.4 ± 14.0 † | 53.3180 ± 1.7121 | - | 0.6241 |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | 0.9474 ± 0.0457 | 202.8 ± 20.7 | 224.9 ± 6.3 | 57.6029 ± 2.0502 | - | 0.1437 |
| `MAPIE_CQR_asym+RF` | MAPIE | 0.9574 ± 0.0230 | 207.4 ± 23.1 | 223.7 ± 15.9 † | 53.1818 ± 1.0729 | - | 0.1752 |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | 0.9148 ± 0.0891 | 190.9 ± 20.3 | 231.4 ± 23.2 † | 52.3900 ± 0.0510 | - | 0.7405 |
| `MAPIE_CVPlus+LightGBM` | MAPIE | 0.9499 ± 0.0043 | 222.3 ± 3.2 | 250.3 ± 2.1 | 54.6632 ± 0.5673 | - | 0.3925 |
| `MAPIE_CVPlus+RF` | MAPIE | 0.9524 ± 0.0043 | 195.1 ± 3.9 | 219.6 ± 2.6 | 51.8597 ± 0.2115 | - | 1.3616 |
| `MAPIE_JackknifePlus+LightGBM` | MAPIE | 0.9398 ± 0.0000 | 216.1 ± 0.0 | 251.9 ± 0.0 | 58.0981 ± 0.0000 | - | 10.6596 |
| `MAPIE_Split+DeepEnsK5` | MAPIE | 0.9549 ± 0.0199 | 208.4 ± 16.4 | 227.4 ± 12.1 † | 53.3180 ± 1.7121 | - | 0.6239 |
| `MAPIE_Split+LightGBM` | MAPIE | 0.9549 ± 0.0075 | 234.4 ± 9.4 | 259.5 ± 11.1 | 57.6029 ± 2.0502 | - | 0.1442 |
| `MAPIE_Split+RF` | MAPIE | 0.9524 ± 0.0115 | 213.0 ± 15.7 | 233.2 ± 11.5 | 53.1818 ± 1.0729 | - | 0.1751 |
| `MAPIE_Split+TabularMLP` | MAPIE | 0.9624 ± 0.0199 | 210.3 ± 8.2 | 229.6 ± 4.7 | 52.3900 ± 0.0510 | - | 0.7403 |
| `Raw+DeepEnsK5` | torchregress | 0.7995 ± 0.0217 | 140.3 ± 4.7 | 220.3 ± 13.7 † | 53.3180 ± 1.7121 | 30.0680 ± 1.1248 | 0.6234 |
| `Raw+LightGBM` | lightgbm | 0.6667 ± 0.0157 | 111.8 ± 7.7 | 265.4 ± 15.2 † | 57.6029 ± 2.0502 | 33.5153 ± 1.7656 | 0.1430 |
| `Raw+RF` | sklearn | 0.7419 ± 0.0304 | 123.5 ± 4.1 | 228.1 ± 10.1 | 53.1818 ± 1.0729 | 30.1938 ± 0.9116 | 0.1745 |
| `Raw+TabularMLP` | torchregress | 0.9449 ± 0.0217 | 190.4 ± 14.6 | 213.0 ± 6.0 | 52.3900 ± 0.0510 | 29.3609 ± 0.2004 | 0.7398 |
| `crepes_Normalized+DeepEnsK5` | crepes | 0.9474 ± 0.0226 | 198.8 ± 14.7 | 210.1 ± 14.3 † | 53.3180 ± 1.7121 | - | 0.6234 |
| `crepes_Normalized+LightGBM` | crepes | 0.9749 ± 0.0189 | 259.4 ± 24.2 | 270.1 ± 16.8 † | 57.6029 ± 2.0502 | - | 0.1430 |
| `crepes_Normalized+RF` | crepes | 0.9323 ± 0.0271 | 192.7 ± 26.1 | 220.8 ± 17.9 † | 53.1818 ± 1.0729 | - | 0.1746 |
| `crepes_Normalized+TabularMLP` | crepes | 0.9449 ± 0.0313 | 191.8 ± 18.1 | 214.9 ± 3.6 | 52.3900 ± 0.0510 | - | 0.7398 |
| `crepes_Split+DeepEnsK5` | crepes | 0.9549 ± 0.0199 | 208.4 ± 16.4 | 227.4 ± 12.1 † | 53.3180 ± 1.7121 | - | 0.6234 |
| `crepes_Split+LightGBM` | crepes | 0.9549 ± 0.0075 | 234.4 ± 9.4 | 259.5 ± 11.1 | 57.6029 ± 2.0502 | - | 0.1430 |
| `crepes_Split+RF` | crepes | 0.9524 ± 0.0115 | 213.0 ± 15.7 | 233.2 ± 11.5 | 53.1818 ± 1.0729 | - | 0.1746 |
| `crepes_Split+TabularMLP` | crepes | 0.9624 ± 0.0199 | 210.3 ± 8.2 | 229.6 ± 4.7 | 52.3900 ± 0.0510 | - | 0.7398 |
| `tr_CQR+DeepEnsK5` | torchregress | 0.9524 ± 0.0230 | 200.3 ± 14.0 | 211.1 ± 12.9 † | 53.3180 ± 1.7121 | - | 0.6239 |
| `tr_CQR+LightGBM` | torchregress | 0.9424 ± 0.0189 | 200.8 ± 9.6 | 220.4 ± 3.4 | 57.6029 ± 2.0502 | - | 0.1443 |
| `tr_CQR+RF` | torchregress | 0.9524 ± 0.0264 | 198.9 ± 18.7 | 215.9 ± 11.7 † | 53.1818 ± 1.0729 | - | 0.1751 |
| `tr_CQR+TabularMLP` | torchregress | 0.9449 ± 0.0313 | 192.6 ± 18.6 | 215.6 ± 3.3 | 52.3900 ± 0.0510 | - | 0.7400 |
| `tr_CVPlus+LightGBM` | torchregress | 0.9499 ± 0.0043 | 222.3 ± 3.2 | 250.3 ± 2.1 | 54.6628 ± 0.5650 | - | 0.3486 |
| `tr_CVPlus+RF` | torchregress | 0.9524 ± 0.0043 | 195.1 ± 3.9 | 219.6 ± 2.6 | **51.8576 ± 0.2132** | - | 1.1751 |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | 0.9148 ± 0.0855 | 197.1 ± 27.2 | 246.0 ± 39.5 † | 60.5743 ± 3.0118 | 34.0084 ± 1.3641 | 0.6240 |
| `tr_IsoMean+VTS+LightGBM` | torchregress | 0.9148 ± 0.0603 | 237.1 ± 28.0 | 292.7 ± 20.0 † | 65.1909 ± 2.2873 | 37.0106 ± 2.4608 | 0.1439 |
| `tr_IsoMean+VTS+RF` | torchregress | 0.9173 ± 0.0398 | 202.3 ± 23.5 | 237.7 ± 20.3 † | 58.8638 ± 1.2345 | 32.9995 ± 0.8576 | 0.1753 |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | 0.8647 ± 0.0813 | 180.3 ± 19.8 | 247.5 ± 42.7 † | 59.6462 ± 4.8178 | 33.9377 ± 3.4971 | 0.7404 |
| `tr_JackknifePlus+LightGBM` | torchregress | 0.9398 ± 0.0000 | 216.1 ± 0.0 | 251.9 ± 0.0 | 58.0981 ± 0.0000 | - | 12.6751 |
| `tr_Normalized+DeepEnsK5` | torchregress | 0.9474 ± 0.0226 | 198.8 ± 14.7 | 210.1 ± 14.3 † | 53.3180 ± 1.7121 | - | 0.6235 |
| `tr_Normalized+LightGBM` | torchregress | 0.9749 ± 0.0189 | 259.4 ± 24.2 | 270.1 ± 16.8 † | 57.6029 ± 2.0502 | - | 0.1432 |
| `tr_Normalized+RF` | torchregress | 0.9323 ± 0.0271 | 192.7 ± 26.1 | 220.8 ± 17.9 † | 53.1818 ± 1.0729 | - | 0.1747 |
| `tr_Normalized+TabularMLP` | torchregress | 0.9449 ± 0.0313 | 191.8 ± 18.1 | 214.9 ± 3.6 | 52.3900 ± 0.0510 | - | 0.7399 |
| `tr_Split+DeepEnsK5` | torchregress | 0.9549 ± 0.0199 | 208.4 ± 16.4 | 227.4 ± 12.1 † | 53.3180 ± 1.7121 | - | 0.6235 |
| `tr_Split+LightGBM` | torchregress | 0.9549 ± 0.0075 | 234.4 ± 9.4 | 259.5 ± 11.1 | 57.6029 ± 2.0502 | - | 0.1432 |
| `tr_Split+RF` | torchregress | 0.9524 ± 0.0115 | 213.0 ± 15.7 | 233.2 ± 11.5 | 53.1818 ± 1.0729 | - | 0.1747 |
| `tr_Split+TabularMLP` | torchregress | 0.9624 ± 0.0199 | 210.3 ± 8.2 | 229.6 ± 4.7 | 52.3900 ± 0.0510 | - | 0.7399 |
| `tr_VTS+DeepEnsK5` | torchregress | 0.9398 ± 0.0260 | 196.8 ± 15.7 | **209.3 ± 14.6** † | 53.3180 ± 1.7121 | 29.7899 ± 1.0363 | 0.6236 |
| `tr_VTS+LightGBM` | torchregress | 0.9499 ± 0.0304 | 229.4 ± 21.0 | 250.8 ± 8.9 | 57.6029 ± 2.0502 | 32.3401 ± 1.1876 | 0.1440 |
| `tr_VTS+RF` | torchregress | 0.9398 ± 0.0226 | 197.1 ± 23.8 | 221.8 ± 18.0 † | 53.1818 ± 1.0729 | 29.6864 ± 0.8986 | 0.1749 |
| `tr_VTS+TabularMLP` | torchregress | 0.9398 ± 0.0271 | 186.7 ± 16.3 | 212.6 ± 3.1 | 52.3900 ± 0.0510 | **29.3441 ± 0.1183** | 0.7401 |

##### uq_layer / house_sales

| Method | Library | Coverage | Width | IntervalScore | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | 0.9024 ± 0.0033 | 2.679e+05 ± 1.055e+03 | 3.945e+05 ± 4.836e+03 | 1.191e+05 ± 1.229e+03 | - | 23.2356 |
| `MAPIE_CQR+LightGBM` | MAPIE | 0.8984 ± 0.0019 | 3.022e+05 ± 2.211e+03 | 4.291e+05 ± 2.397e+03 | 1.203e+05 ± 2.578e+03 | - | 1.9444 |
| `MAPIE_CQR+RF` | MAPIE | 0.8963 ± 0.0015 | 3.066e+05 ± 1.017e+03 | 4.306e+05 ± 4.588e+03 | 1.386e+05 ± 2.529e+03 | - | 9.7247 |
| `MAPIE_CQR+TabularMLP` | MAPIE | 0.8975 ± 0.0046 | 2.648e+05 ± 6.932e+03 | 3.691e+05 ± 2.535e+03 | **1.127e+05 ± 1.361e+02** | - | 36.1806 |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | 0.9012 ± 0.0028 | 2.682e+05 ± 7.575e+02 | 3.934e+05 ± 5.566e+03 | 1.191e+05 ± 1.229e+03 | - | 23.2355 |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | 0.8989 ± 0.0019 | 3.041e+05 ± 1.941e+03 | 4.286e+05 ± 2.305e+03 | 1.203e+05 ± 2.578e+03 | - | 1.9445 |
| `MAPIE_CQR_asym+RF` | MAPIE | 0.8969 ± 0.0030 | 3.054e+05 ± 1.369e+03 | 4.300e+05 ± 4.691e+03 | 1.386e+05 ± 2.529e+03 | - | 9.7246 |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | 0.8984 ± 0.0043 | 2.656e+05 ± 5.844e+03 | **3.687e+05 ± 2.122e+03** | **1.127e+05 ± 1.361e+02** | - | 36.1806 |
| `MAPIE_CVPlus+LightGBM` | MAPIE | 0.9108 ± 0.0021 | 3.072e+05 ± 7.705e+02 | 5.235e+05 ± 2.895e+03 | 1.166e+05 ± 4.336e+02 | - | 8.2209 |
| `MAPIE_CVPlus+RF` | MAPIE | 0.9129 ± 0.0018 | 3.293e+05 ± 2.806e+03 | 5.968e+05 ± 9.138e+02 | 1.326e+05 ± 2.754e+02 | - | 63.6579 |
| `MAPIE_Split+DeepEnsK5` | MAPIE | 0.8982 ± 0.0008 | 2.923e+05 ± 4.901e+03 | 5.624e+05 ± 5.144e+03 | 1.191e+05 ± 1.229e+03 | - | 23.2330 |
| `MAPIE_Split+LightGBM` | MAPIE | 0.9003 ± 0.0057 | 2.983e+05 ± 1.259e+04 | 5.693e+05 ± 9.263e+03 | 1.203e+05 ± 2.578e+03 | - | 1.9413 |
| `MAPIE_Split+RF` | MAPIE | 0.9005 ± 0.0045 | 3.139e+05 ± 6.107e+03 | 6.415e+05 ± 3.029e+03 | 1.386e+05 ± 2.529e+03 | - | 9.7215 |
| `MAPIE_Split+TabularMLP` | MAPIE | 0.8979 ± 0.0008 | 2.777e+05 ± 4.488e+03 | 5.375e+05 ± 1.900e+03 | **1.127e+05 ± 1.361e+02** | - | 36.1777 |
| `Raw+DeepEnsK5` | torchregress | 0.9036 ± 0.0039 | 2.692e+05 ± 2.715e+03 | 3.945e+05 ± 4.901e+03 | 1.191e+05 ± 1.229e+03 | 48401.1 ± 382.0 | 23.2300 |
| `Raw+LightGBM` | lightgbm | 0.8505 ± 0.0032 | 2.717e+05 ± 4.845e+02 | 4.364e+05 ± 3.295e+03 | 1.203e+05 ± 2.578e+03 | 49360.4 ± 310.7 | 1.9364 |
| `Raw+RF` | sklearn | 0.8552 ± 0.0017 | 2.807e+05 ± 6.100e+02 | 4.363e+05 ± 4.656e+03 | 1.386e+05 ± 2.529e+03 | 52640.0 ± 108.2 | 9.7179 |
| `Raw+TabularMLP` | torchregress | 0.9125 ± 0.0021 | 2.760e+05 ± 4.633e+03 | 3.696e+05 ± 2.737e+03 | **1.127e+05 ± 1.361e+02** | 46004.6 ± 351.9 | 36.1751 |
| `crepes_Normalized+DeepEnsK5` | crepes | 0.9025 ± 0.0034 | 2.677e+05 ± 1.340e+03 | 3.947e+05 ± 4.993e+03 | 1.191e+05 ± 1.229e+03 | - | 23.2300 |
| `crepes_Normalized+LightGBM` | crepes | 0.9041 ± 0.0066 | 2.985e+05 ± 7.054e+03 | 3.957e+05 ± 2.453e+03 | 1.203e+05 ± 2.578e+03 | - | 1.9365 |
| `crepes_Normalized+RF` | crepes | 0.8973 ± 0.0032 | 3.158e+05 ± 1.960e+03 | 4.323e+05 ± 3.324e+03 | 1.386e+05 ± 2.529e+03 | - | 9.7179 |
| `crepes_Normalized+TabularMLP` | crepes | 0.8976 ± 0.0037 | 2.610e+05 ± 6.536e+03 | 3.692e+05 ± 2.058e+03 | **1.127e+05 ± 1.361e+02** | - | 36.1751 |
| `crepes_Split+DeepEnsK5` | crepes | 0.8982 ± 0.0008 | 2.923e+05 ± 4.901e+03 | 5.624e+05 ± 5.144e+03 | 1.191e+05 ± 1.229e+03 | - | 23.2300 |
| `crepes_Split+LightGBM` | crepes | 0.9003 ± 0.0057 | 2.983e+05 ± 1.259e+04 | 5.693e+05 ± 9.263e+03 | 1.203e+05 ± 2.578e+03 | - | 1.9365 |
| `crepes_Split+RF` | crepes | 0.9005 ± 0.0045 | 3.139e+05 ± 6.107e+03 | 6.415e+05 ± 3.029e+03 | 1.386e+05 ± 2.529e+03 | - | 9.7180 |
| `crepes_Split+TabularMLP` | crepes | 0.8979 ± 0.0008 | 2.777e+05 ± 4.488e+03 | 5.375e+05 ± 1.900e+03 | **1.127e+05 ± 1.361e+02** | - | 36.1751 |
| `tr_CQR+DeepEnsK5` | torchregress | 0.9024 ± 0.0033 | 2.679e+05 ± 1.055e+03 | 3.945e+05 ± 4.836e+03 | 1.191e+05 ± 1.229e+03 | - | 23.2303 |
| `tr_CQR+LightGBM` | torchregress | 0.8984 ± 0.0019 | 3.022e+05 ± 2.211e+03 | 4.291e+05 ± 2.397e+03 | 1.203e+05 ± 2.578e+03 | - | 1.9371 |
| `tr_CQR+RF` | torchregress | 0.8963 ± 0.0015 | 3.066e+05 ± 1.017e+03 | 4.306e+05 ± 4.588e+03 | 1.386e+05 ± 2.529e+03 | - | 9.7184 |
| `tr_CQR+TabularMLP` | torchregress | 0.8975 ± 0.0046 | 2.648e+05 ± 6.932e+03 | 3.691e+05 ± 2.535e+03 | **1.127e+05 ± 1.361e+02** | - | 36.1754 |
| `tr_CVPlus+LightGBM` | torchregress | 0.9108 ± 0.0021 | 3.072e+05 ± 7.705e+02 | 5.235e+05 ± 2.895e+03 | 1.166e+05 ± 4.336e+02 | - | 6.6492 |
| `tr_CVPlus+RF` | torchregress | 0.9129 ± 0.0018 | 3.293e+05 ± 2.806e+03 | 5.968e+05 ± 9.138e+02 | 1.326e+05 ± 2.754e+02 | - | 54.0644 |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | 0.9141 ± 0.0021 | 2.850e+05 ± 3.100e+03 | 4.081e+05 ± 1.137e+04 | 1.273e+05 ± 4.782e+03 | 50008.6 ± 564.1 | 23.2314 |
| `tr_IsoMean+VTS+LightGBM` | torchregress | 0.9187 ± 0.0040 | 3.231e+05 ± 5.256e+03 | 4.157e+05 ± 1.064e+04 | 1.422e+05 ± 2.028e+04 | 52301.5 ± 1155.9 | 1.9392 |
| `tr_IsoMean+VTS+RF` | torchregress | 0.9102 ± 0.0007 | 3.362e+05 ± 2.098e+03 | 4.379e+05 ± 6.368e+03 | 1.448e+05 ± 1.008e+04 | 54116.9 ± 751.9 | 9.7198 |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | 0.9077 ± 0.0056 | 2.782e+05 ± 8.337e+03 | 3.840e+05 ± 1.138e+04 | 1.252e+05 ± 1.149e+04 | 47996.2 ± 1101.8 | 36.1765 |
| `tr_Normalized+DeepEnsK5` | torchregress | 0.9025 ± 0.0034 | 2.677e+05 ± 1.340e+03 | 3.947e+05 ± 4.993e+03 | 1.191e+05 ± 1.229e+03 | - | 23.2302 |
| `tr_Normalized+LightGBM` | torchregress | 0.9041 ± 0.0066 | 2.985e+05 ± 7.054e+03 | 3.957e+05 ± 2.453e+03 | 1.203e+05 ± 2.578e+03 | - | 1.9369 |
| `tr_Normalized+RF` | torchregress | 0.8973 ± 0.0032 | 3.158e+05 ± 1.960e+03 | 4.323e+05 ± 3.324e+03 | 1.386e+05 ± 2.529e+03 | - | 9.7182 |
| `tr_Normalized+TabularMLP` | torchregress | 0.8976 ± 0.0037 | 2.610e+05 ± 6.536e+03 | 3.692e+05 ± 2.058e+03 | **1.127e+05 ± 1.361e+02** | - | 36.1754 |
| `tr_Split+DeepEnsK5` | torchregress | 0.8982 ± 0.0008 | 2.923e+05 ± 4.901e+03 | 5.624e+05 ± 5.144e+03 | 1.191e+05 ± 1.229e+03 | - | 23.2303 |
| `tr_Split+LightGBM` | torchregress | 0.9003 ± 0.0057 | 2.983e+05 ± 1.259e+04 | 5.693e+05 ± 9.263e+03 | 1.203e+05 ± 2.578e+03 | - | 1.9370 |
| `tr_Split+RF` | torchregress | 0.9005 ± 0.0045 | 3.139e+05 ± 6.107e+03 | 6.415e+05 ± 3.029e+03 | 1.386e+05 ± 2.529e+03 | - | 9.7183 |
| `tr_Split+TabularMLP` | torchregress | 0.8979 ± 0.0008 | 2.777e+05 ± 4.488e+03 | 5.375e+05 ± 1.900e+03 | **1.127e+05 ± 1.361e+02** | - | 36.1753 |
| `tr_VTS+DeepEnsK5` | torchregress | 0.9126 ± 0.0038 | 2.785e+05 ± 3.571e+03 | 3.940e+05 ± 4.945e+03 | 1.191e+05 ± 1.229e+03 | 48424.7 ± 368.3 | 23.2304 |
| `tr_VTS+LightGBM` | torchregress | 0.9184 ± 0.0037 | 3.143e+05 ± 4.593e+03 | 3.985e+05 ± 3.298e+03 | 1.203e+05 ± 2.578e+03 | 49583.4 ± 324.6 | 1.9381 |
| `tr_VTS+RF` | torchregress | 0.9104 ± 0.0052 | 3.327e+05 ± 4.682e+03 | 4.350e+05 ± 4.220e+03 | 1.386e+05 ± 2.529e+03 | 52890.8 ± 133.2 | 9.7194 |
| `tr_VTS+TabularMLP` | torchregress | 0.9099 ± 0.0014 | 2.731e+05 ± 2.947e+03 | 3.693e+05 ± 2.420e+03 | **1.127e+05 ± 1.361e+02** | **45987.1 ± 334.1** | 36.1755 |

Failed / skipped:

- `MAPIE_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=14000 models (> 500)
- `tr_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=14000 models (> 500)

##### uq_layer / insurance_claims

| Method | Library | Coverage | Width | IntervalScore | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | 0.8995 ± 0.0075 | 0.4412 ± 0.0317 | 1.2177 ± 0.0124 | 0.2405 ± 0.0009 | - | 23.3752 |
| `MAPIE_CQR+LightGBM` | MAPIE | 0.9494 ± 0.0002 | 0.2410 ± 0.1731 | **0.9492 ± 0.0216** | 0.2382 ± 0.0009 | - | 0.9678 |
| `MAPIE_CQR+RF` | MAPIE | 0.9498 ± 0.0004 | 0.3096 ± 0.0082 | 0.9917 ± 0.0078 | 0.2382 ± 0.0004 | - | 5.9348 |
| `MAPIE_CQR+TabularMLP` | MAPIE | 0.8986 ± 0.0073 | 0.1799 ± 0.0146 | 1.0941 ± 0.0115 | **0.2363 ± 0.0002** | - | 18.9075 |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | 0.9006 ± 0.0008 | 0.5630 ± 0.1684 | 1.2170 ± 0.0095 | 0.2405 ± 0.0009 | - | 23.3742 |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | 0.9521 ± 0.0048 | 0.2410 ± 0.1731 | 0.9492 ± 0.0216 | 0.2382 ± 0.0009 | - | 0.9678 |
| `MAPIE_CQR_asym+RF` | MAPIE | 0.9528 ± 0.0048 | 0.4548 ± 0.2434 | 0.9972 ± 0.0127 | 0.2382 ± 0.0004 | - | 5.9348 |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | 0.9018 ± 0.0047 | 0.4089 ± 0.4982 | 1.0940 ± 0.0097 | **0.2363 ± 0.0002** | - | 18.9074 |
| `MAPIE_CVPlus+LightGBM` | MAPIE | 0.9302 ± 0.0007 | 0.2803 ± 0.0049 | 1.1480 ± 0.0022 | 0.2364 ± 0.0001 | - | 6.1422 |
| `MAPIE_CVPlus+RF` | MAPIE | 0.9219 ± 0.0024 | 0.2992 ± 0.0043 | 1.1561 ± 0.0014 | 0.2368 ± 0.0000 | - | 41.2130 |
| `MAPIE_Split+DeepEnsK5` | MAPIE | 0.8985 ± 0.0090 | 0.1855 ± 0.0291 | 1.1902 ± 0.0051 | 0.2405 ± 0.0009 | - | 23.3681 |
| `MAPIE_Split+LightGBM` | MAPIE | 0.8977 ± 0.0070 | 0.2646 ± 0.0044 | 1.2004 ± 0.0135 | 0.2382 ± 0.0009 | - | 0.9607 |
| `MAPIE_Split+RF` | MAPIE | 0.8932 ± 0.0057 | 0.2828 ± 0.0078 | 1.1936 ± 0.0092 | 0.2382 ± 0.0004 | - | 5.9256 |
| `MAPIE_Split+TabularMLP` | MAPIE | 0.8963 ± 0.0065 | 0.1370 ± 0.0025 | 1.0870 ± 0.0065 | **0.2363 ± 0.0002** | - | 18.9001 |
| `Raw+DeepEnsK5` | torchregress | 0.9511 ± 0.0017 | 0.4853 ± 0.0417 | 1.2330 ± 0.0138 | 0.2405 ± 0.0009 | **0.0819 ± 0.0024** | 23.3619 |
| `Raw+LightGBM` | lightgbm | 0.9494 ± 0.0002 | 0.2410 ± 0.1731 | **0.9492 ± 0.0216** | 0.2382 ± 0.0009 | 0.0962 ± 0.0040 | 0.9498 |
| `Raw+RF` | sklearn | 0.9498 ± 0.0004 | 0.3096 ± 0.0082 | 0.9917 ± 0.0078 | 0.2382 ± 0.0004 | 0.0828 ± 0.0011 | 5.9185 |
| `Raw+TabularMLP` | torchregress | 0.9493 ± 0.0000 | 0.7568 ± 0.0345 | 1.3704 ± 0.0158 | **0.2363 ± 0.0002** | 0.0990 ± 0.0023 | 18.8951 |
| `crepes_Normalized+DeepEnsK5` | crepes | 0.9002 ± 0.0078 | 0.1428 ± 0.0109 | 1.1005 ± 0.0071 | 0.2405 ± 0.0009 | - | 23.3619 |
| `crepes_Normalized+LightGBM` | crepes | 0.9033 ± 0.0083 | 35976.2 ± 19213.2 | 35976.8 ± 19213.0 † | 0.2382 ± 0.0009 | - | 0.9499 |
| `crepes_Normalized+RF` | crepes | 0.8927 ± 0.0052 | 0.1619 ± 0.0056 | 1.0930 ± 0.0047 | 0.2382 ± 0.0004 | - | 5.9185 |
| `crepes_Normalized+TabularMLP` | crepes | 0.8958 ± 0.0060 | 0.1180 ± 0.0031 | 1.0718 ± 0.0049 | **0.2363 ± 0.0002** | - | 18.8951 |
| `crepes_Split+DeepEnsK5` | crepes | 0.8985 ± 0.0090 | 0.1855 ± 0.0291 | 1.1902 ± 0.0051 | 0.2405 ± 0.0009 | - | 23.3619 |
| `crepes_Split+LightGBM` | crepes | 0.8977 ± 0.0070 | 0.2646 ± 0.0044 | 1.2004 ± 0.0135 | 0.2382 ± 0.0009 | - | 0.9499 |
| `crepes_Split+RF` | crepes | 0.8932 ± 0.0057 | 0.2828 ± 0.0078 | 1.1936 ± 0.0092 | 0.2382 ± 0.0004 | - | 5.9185 |
| `crepes_Split+TabularMLP` | crepes | 0.8963 ± 0.0065 | 0.1370 ± 0.0025 | 1.0870 ± 0.0065 | **0.2363 ± 0.0002** | - | 18.8951 |
| `tr_CQR+DeepEnsK5` | torchregress | 0.8995 ± 0.0075 | 0.4412 ± 0.0317 | 1.2177 ± 0.0124 | 0.2405 ± 0.0009 | - | 23.3622 |
| `tr_CQR+LightGBM` | torchregress | 0.9494 ± 0.0002 | 0.2410 ± 0.1731 | **0.9492 ± 0.0216** | 0.2382 ± 0.0009 | - | 0.9503 |
| `tr_CQR+RF` | torchregress | 0.9498 ± 0.0004 | 0.3096 ± 0.0082 | 0.9917 ± 0.0078 | 0.2382 ± 0.0004 | - | 5.9188 |
| `tr_CQR+TabularMLP` | torchregress | 0.8986 ± 0.0073 | 0.1799 ± 0.0146 | 1.0941 ± 0.0115 | **0.2363 ± 0.0002** | - | 18.8954 |
| `tr_CVPlus+LightGBM` | torchregress | 0.9302 ± 0.0007 | 0.2803 ± 0.0049 | 1.1480 ± 0.0022 | 0.2364 ± 0.0001 | - | 4.8989 |
| `tr_CVPlus+RF` | torchregress | 0.9219 ± 0.0024 | 0.2992 ± 0.0043 | 1.1561 ± 0.0014 | 0.2368 ± 0.0000 | - | 35.1884 |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | 0.9768 ± 0.0031 | 2.9409 ± 0.5697 | 3.1952 ± 0.5455 † | 0.2371 ± 0.0004 | 0.2372 ± 0.0388 | 23.3633 |
| `tr_IsoMean+VTS+LightGBM` | torchregress | 0.9672 ± 0.0073 | 1.358e+05 ± 6.763e+04 | 1.358e+05 ± 6.763e+04 † | 0.2368 ± 0.0004 | 9646.2 ± 4804.6 | 0.9524 |
| `tr_IsoMean+VTS+RF` | torchregress | 0.9579 ± 0.0011 | 1.0935 ± 0.0417 | 1.4912 ± 0.0352 | 0.2370 ± 0.0004 | 0.1164 ± 0.0025 | 5.9202 |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | 0.9493 ± 0.0001 | 0.7382 ± 0.0645 | 1.3548 ± 0.0234 | 0.2365 ± 0.0008 | 0.0986 ± 0.0042 | 18.8965 |
| `tr_Normalized+DeepEnsK5` | torchregress | 0.9002 ± 0.0078 | 0.1428 ± 0.0109 | 1.1005 ± 0.0071 | 0.2405 ± 0.0009 | - | 23.3622 |
| `tr_Normalized+LightGBM` | torchregress | 0.9033 ± 0.0083 | 35976.2 ± 19213.2 | 35976.8 ± 19213.0 † | 0.2382 ± 0.0009 | - | 0.9503 |
| `tr_Normalized+RF` | torchregress | 0.8927 ± 0.0052 | 0.1619 ± 0.0056 | 1.0930 ± 0.0047 | 0.2382 ± 0.0004 | - | 5.9188 |
| `tr_Normalized+TabularMLP` | torchregress | 0.8958 ± 0.0060 | 0.1180 ± 0.0031 | 1.0718 ± 0.0049 | **0.2363 ± 0.0002** | - | 18.8954 |
| `tr_Split+DeepEnsK5` | torchregress | 0.8985 ± 0.0090 | 0.1855 ± 0.0291 | 1.1902 ± 0.0051 | 0.2405 ± 0.0009 | - | 23.3622 |
| `tr_Split+LightGBM` | torchregress | 0.8977 ± 0.0070 | 0.2646 ± 0.0044 | 1.2004 ± 0.0135 | 0.2382 ± 0.0009 | - | 0.9504 |
| `tr_Split+RF` | torchregress | 0.8932 ± 0.0057 | 0.2828 ± 0.0078 | 1.1936 ± 0.0092 | 0.2382 ± 0.0004 | - | 5.9189 |
| `tr_Split+TabularMLP` | torchregress | 0.8963 ± 0.0065 | 0.1370 ± 0.0025 | 1.0870 ± 0.0065 | **0.2363 ± 0.0002** | - | 18.8954 |
| `tr_VTS+DeepEnsK5` | torchregress | 0.9751 ± 0.0033 | 2.7013 ± 0.5316 | 2.9929 ± 0.5073 † | 0.2405 ± 0.0009 | 0.2205 ± 0.0359 | 23.3623 |
| `tr_VTS+LightGBM` | torchregress | 0.9671 ± 0.0075 | 1.380e+05 ± 7.109e+04 | 1.380e+05 ± 7.109e+04 † | 0.2382 ± 0.0009 | 9806.4 ± 5049.9 | 0.9512 |
| `tr_VTS+RF` | torchregress | 0.9632 ± 0.0063 | 1.2796 ± 0.3121 | 1.6217 ± 0.2275 † | 0.2382 ± 0.0004 | 0.1273 ± 0.0186 | 5.9194 |
| `tr_VTS+TabularMLP` | torchregress | 0.9493 ± 0.0000 | 0.7558 ± 0.0606 | 1.3698 ± 0.0301 | **0.2363 ± 0.0002** | 0.0990 ± 0.0029 | 18.8955 |

Failed / skipped:

- `MAPIE_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=14000 models (> 500)
- `tr_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=14000 models (> 500)

##### uq_layer / openml_year

| Method | Library | Coverage | Width | IntervalScore | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | 0.9074 ± 0.0145 | 0.4963 ± 0.0269 | 0.6754 ± 0.0122 | 0.1610 ± 0.0024 | - | 12.3964 |
| `MAPIE_CQR+LightGBM` | MAPIE | 0.9124 ± 0.0062 | 0.5563 ± 0.0172 | 0.6902 ± 0.0098 | 0.1446 ± 0.0021 | - | 0.9652 |
| `MAPIE_CQR+RF` | MAPIE | 0.9134 ± 0.0074 | 0.5000 ± 0.0192 | 0.6651 ± 0.0064 | 0.1575 ± 0.0002 | - | 3.3652 |
| `MAPIE_CQR+TabularMLP` | MAPIE | 0.9036 ± 0.0052 | 0.4742 ± 0.0166 | 0.6451 ± 0.0042 | 0.1549 ± 0.0006 | - | 19.2533 |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | 0.9058 ± 0.0120 | 0.4989 ± 0.0260 | 0.6754 ± 0.0099 | 0.1610 ± 0.0024 | - | 12.3964 |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | 0.9133 ± 0.0059 | 0.5566 ± 0.0160 | 0.6899 ± 0.0093 | 0.1446 ± 0.0021 | - | 0.9652 |
| `MAPIE_CQR_asym+RF` | MAPIE | 0.9158 ± 0.0088 | 0.5063 ± 0.0207 | 0.6666 ± 0.0072 | 0.1575 ± 0.0002 | - | 3.3652 |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | 0.9083 ± 0.0078 | 0.4883 ± 0.0161 | 0.6443 ± 0.0055 | 0.1549 ± 0.0006 | - | 19.2535 |
| `MAPIE_CVPlus+LightGBM` | MAPIE | 0.9133 ± 0.0003 | 0.4608 ± 0.0012 | 0.6467 ± 0.0012 | **0.1408 ± 0.0005** | - | 3.2986 |
| `MAPIE_CVPlus+RF` | MAPIE | 0.9158 ± 0.0010 | 0.5139 ± 0.0010 | 0.7159 ± 0.0002 | 0.1541 ± 0.0000 | - | 21.4773 |
| `MAPIE_Split+DeepEnsK5` | MAPIE | 0.9064 ± 0.0108 | 0.5143 ± 0.0263 | 0.7320 ± 0.0102 | 0.1610 ± 0.0024 | - | 12.3962 |
| `MAPIE_Split+LightGBM` | MAPIE | 0.9044 ± 0.0132 | 0.4565 ± 0.0269 | 0.6693 ± 0.0137 | 0.1446 ± 0.0021 | - | 0.9665 |
| `MAPIE_Split+RF` | MAPIE | 0.9132 ± 0.0089 | 0.5213 ± 0.0237 | 0.7343 ± 0.0022 | 0.1575 ± 0.0002 | - | 3.3646 |
| `MAPIE_Split+TabularMLP` | MAPIE | 0.9088 ± 0.0077 | 0.5042 ± 0.0170 | 0.7202 ± 0.0038 | 0.1549 ± 0.0006 | - | 19.2528 |
| `Raw+DeepEnsK5` | torchregress | 0.9234 ± 0.0062 | 0.5270 ± 0.0116 | 0.6797 ± 0.0113 | 0.1610 ± 0.0024 | 0.0846 ± 0.0015 | 12.3944 |
| `Raw+LightGBM` | lightgbm | 0.8543 ± 0.0064 | 0.5020 ± 0.0177 | 0.6960 ± 0.0082 | 0.1446 ± 0.0021 | 0.0765 ± 0.0002 | 0.9629 |
| `Raw+RF` | sklearn | 0.8618 ± 0.0063 | 0.4343 ± 0.0076 | 0.6703 ± 0.0072 | 0.1575 ± 0.0002 | 0.0813 ± 0.0005 | 3.3633 |
| `Raw+TabularMLP` | torchregress | 0.9139 ± 0.0090 | 0.4926 ± 0.0107 | 0.6472 ± 0.0027 | 0.1549 ± 0.0006 | 0.0799 ± 0.0003 | 19.2515 |
| `crepes_Normalized+DeepEnsK5` | crepes | 0.9082 ± 0.0136 | 0.4936 ± 0.0289 | 0.6750 ± 0.0124 | 0.1610 ± 0.0024 | - | 12.3945 |
| `crepes_Normalized+LightGBM` | crepes | 0.9020 ± 0.0130 | 0.4612 ± 0.0224 | **0.6425 ± 0.0044** | 0.1446 ± 0.0021 | - | 0.9629 |
| `crepes_Normalized+RF` | crepes | 0.9120 ± 0.0061 | 0.4810 ± 0.0165 | 0.6641 ± 0.0048 | 0.1575 ± 0.0002 | - | 3.3633 |
| `crepes_Normalized+TabularMLP` | crepes | 0.9028 ± 0.0066 | 0.4718 ± 0.0201 | 0.6447 ± 0.0043 | 0.1549 ± 0.0006 | - | 19.2515 |
| `crepes_Split+DeepEnsK5` | crepes | 0.9064 ± 0.0108 | 0.5143 ± 0.0263 | 0.7320 ± 0.0102 | 0.1610 ± 0.0024 | - | 12.3945 |
| `crepes_Split+LightGBM` | crepes | 0.9044 ± 0.0132 | 0.4565 ± 0.0269 | 0.6693 ± 0.0137 | 0.1446 ± 0.0021 | - | 0.9630 |
| `crepes_Split+RF` | crepes | 0.9132 ± 0.0089 | 0.5213 ± 0.0237 | 0.7343 ± 0.0022 | 0.1575 ± 0.0002 | - | 3.3633 |
| `crepes_Split+TabularMLP` | crepes | 0.9088 ± 0.0077 | 0.5042 ± 0.0170 | 0.7202 ± 0.0038 | 0.1549 ± 0.0006 | - | 19.2516 |
| `tr_CQR+DeepEnsK5` | torchregress | 0.9074 ± 0.0145 | 0.4963 ± 0.0269 | 0.6754 ± 0.0122 | 0.1610 ± 0.0024 | - | 12.3947 |
| `tr_CQR+LightGBM` | torchregress | 0.9124 ± 0.0062 | 0.5563 ± 0.0172 | 0.6902 ± 0.0098 | 0.1446 ± 0.0021 | - | 0.9634 |
| `tr_CQR+RF` | torchregress | 0.9134 ± 0.0074 | 0.5000 ± 0.0192 | 0.6651 ± 0.0064 | 0.1575 ± 0.0002 | - | 3.3636 |
| `tr_CQR+TabularMLP` | torchregress | 0.9036 ± 0.0052 | 0.4742 ± 0.0166 | 0.6451 ± 0.0042 | 0.1549 ± 0.0006 | - | 19.2518 |
| `tr_CVPlus+LightGBM` | torchregress | 0.9133 ± 0.0003 | 0.4608 ± 0.0012 | 0.6467 ± 0.0012 | **0.1408 ± 0.0005** | - | 2.7254 |
| `tr_CVPlus+RF` | torchregress | 0.9158 ± 0.0010 | 0.5139 ± 0.0010 | 0.7159 ± 0.0002 | 0.1541 ± 0.0000 | - | 19.8455 |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | 0.9073 ± 0.0075 | 0.5074 ± 0.0296 | 0.6867 ± 0.0175 | 0.1618 ± 0.0032 | 0.0849 ± 0.0021 | 12.3958 |
| `tr_IsoMean+VTS+LightGBM` | torchregress | 0.9122 ± 0.0067 | 0.4920 ± 0.0179 | 0.6559 ± 0.0044 | 0.1464 ± 0.0025 | 0.0772 ± 0.0006 | 0.9647 |
| `tr_IsoMean+VTS+RF` | torchregress | 0.9253 ± 0.0117 | 0.5259 ± 0.0258 | 0.6879 ± 0.0049 | 0.1591 ± 0.0003 | 0.0825 ± 0.0004 | 3.3652 |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | 0.9140 ± 0.0120 | 0.5027 ± 0.0256 | 0.6611 ± 0.0092 | 0.1572 ± 0.0010 | 0.0815 ± 0.0006 | 19.2526 |
| `tr_Normalized+DeepEnsK5` | torchregress | 0.9082 ± 0.0136 | 0.4936 ± 0.0289 | 0.6750 ± 0.0124 | 0.1610 ± 0.0024 | - | 12.3947 |
| `tr_Normalized+LightGBM` | torchregress | 0.9020 ± 0.0130 | 0.4612 ± 0.0224 | **0.6425 ± 0.0044** | 0.1446 ± 0.0021 | - | 0.9632 |
| `tr_Normalized+RF` | torchregress | 0.9120 ± 0.0061 | 0.4810 ± 0.0165 | 0.6641 ± 0.0048 | 0.1575 ± 0.0002 | - | 3.3635 |
| `tr_Normalized+TabularMLP` | torchregress | 0.9028 ± 0.0066 | 0.4718 ± 0.0201 | 0.6447 ± 0.0043 | 0.1549 ± 0.0006 | - | 19.2517 |
| `tr_Split+DeepEnsK5` | torchregress | 0.9064 ± 0.0108 | 0.5143 ± 0.0263 | 0.7320 ± 0.0102 | 0.1610 ± 0.0024 | - | 12.3947 |
| `tr_Split+LightGBM` | torchregress | 0.9044 ± 0.0132 | 0.4565 ± 0.0269 | 0.6693 ± 0.0137 | 0.1446 ± 0.0021 | - | 0.9633 |
| `tr_Split+RF` | torchregress | 0.9132 ± 0.0089 | 0.5213 ± 0.0237 | 0.7343 ± 0.0022 | 0.1575 ± 0.0002 | - | 3.3636 |
| `tr_Split+TabularMLP` | torchregress | 0.9088 ± 0.0077 | 0.5042 ± 0.0170 | 0.7202 ± 0.0038 | 0.1549 ± 0.0006 | - | 19.2517 |
| `tr_VTS+DeepEnsK5` | torchregress | 0.9211 ± 0.0032 | 0.5221 ± 0.0106 | 0.6786 ± 0.0113 | 0.1610 ± 0.0024 | 0.0845 ± 0.0015 | 12.3949 |
| `tr_VTS+LightGBM` | torchregress | 0.9197 ± 0.0035 | 0.5014 ± 0.0118 | 0.6491 ± 0.0053 | 0.1446 ± 0.0021 | **0.0764 ± 0.0005** | 0.9642 |
| `tr_VTS+RF` | torchregress | 0.9306 ± 0.0044 | 0.5228 ± 0.0114 | 0.6737 ± 0.0098 | 0.1575 ± 0.0002 | 0.0817 ± 0.0003 | 3.3644 |
| `tr_VTS+TabularMLP` | torchregress | 0.9251 ± 0.0014 | 0.5125 ± 0.0068 | 0.6515 ± 0.0043 | 0.1549 ± 0.0006 | 0.0801 ± 0.0004 | 19.2519 |

Failed / skipped:

- `MAPIE_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=7000 models (> 500)
- `tr_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=7000 models (> 500)

##### uq_layer / synthetic_hetero

| Method | Library | Coverage | Width | IntervalScore | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | 0.9222 ± 0.0108 | 1.2960 ± 0.0448 | 1.5803 ± 0.0418 | 0.3756 ± 0.0061 | - | 2.0473 |
| `MAPIE_CQR+LightGBM` | MAPIE | 0.9067 ± 0.0088 | 1.4217 ± 0.0332 | 1.7357 ± 0.0138 | 0.3986 ± 0.0050 | - | 0.5284 |
| `MAPIE_CQR+RF` | MAPIE | 0.9167 ± 0.0176 | 1.4047 ± 0.0656 | 1.7054 ± 0.0302 | 0.3984 ± 0.0060 | - | 0.6241 |
| `MAPIE_CQR+TabularMLP` | MAPIE | 0.9267 ± 0.0050 | 1.2139 ± 0.0125 | 1.4205 ± 0.0216 | **0.3629 ± 0.0021** | - | 3.1352 |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | 0.9233 ± 0.0100 | 1.3169 ± 0.0409 | 1.5846 ± 0.0438 | 0.3756 ± 0.0061 | - | 2.0473 |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | 0.9106 ± 0.0079 | 1.4550 ± 0.0361 | 1.7460 ± 0.0198 | 0.3986 ± 0.0050 | - | 0.5284 |
| `MAPIE_CQR_asym+RF` | MAPIE | 0.9167 ± 0.0186 | 1.4368 ± 0.0732 | 1.7259 ± 0.0305 | 0.3984 ± 0.0060 | - | 0.6242 |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | 0.9272 ± 0.0069 | 1.2413 ± 0.0239 | 1.4272 ± 0.0251 | **0.3629 ± 0.0021** | - | 3.1352 |
| `MAPIE_CVPlus+LightGBM` | MAPIE | 0.9256 ± 0.0054 | 1.3964 ± 0.0124 | 1.6963 ± 0.0073 | 0.3799 ± 0.0023 | - | 1.3255 |
| `MAPIE_CVPlus+RF` | MAPIE | 0.9294 ± 0.0025 | 1.4189 ± 0.0097 | 1.7566 ± 0.0064 | 0.3892 ± 0.0005 | - | 4.1474 |
| `MAPIE_Split+DeepEnsK5` | MAPIE | 0.9206 ± 0.0086 | 1.3324 ± 0.0616 | 1.6622 ± 0.0313 | 0.3756 ± 0.0061 | - | 2.0473 |
| `MAPIE_Split+LightGBM` | MAPIE | 0.9039 ± 0.0155 | 1.3281 ± 0.0432 | 1.7697 ± 0.0485 | 0.3986 ± 0.0050 | - | 0.5290 |
| `MAPIE_Split+RF` | MAPIE | 0.9233 ± 0.0087 | 1.4446 ± 0.0497 | 1.8105 ± 0.0244 | 0.3984 ± 0.0060 | - | 0.6239 |
| `MAPIE_Split+TabularMLP` | MAPIE | 0.9283 ± 0.0101 | 1.3021 ± 0.0411 | 1.6243 ± 0.0254 | **0.3629 ± 0.0021** | - | 3.1351 |
| `Raw+DeepEnsK5` | torchregress | 0.8883 ± 0.0088 | 1.1746 ± 0.0198 | 1.5719 ± 0.0387 | 0.3756 ± 0.0061 | 0.2044 ± 0.0043 | 2.0466 |
| `Raw+LightGBM` | lightgbm | 0.7600 ± 0.0100 | 1.0351 ± 0.0061 | 1.9463 ± 0.0241 | 0.3986 ± 0.0050 | 0.2187 ± 0.0008 | 0.5273 |
| `Raw+RF` | sklearn | 0.8322 ± 0.0113 | 1.0456 ± 0.0082 | 1.7684 ± 0.0318 | 0.3984 ± 0.0060 | 0.2169 ± 0.0032 | 0.6231 |
| `Raw+TabularMLP` | torchregress | 0.9028 ± 0.0170 | 1.1412 ± 0.0305 | **1.4116 ± 0.0264** | **0.3629 ± 0.0021** | **0.1951 ± 0.0016** | 3.1345 |
| `crepes_Normalized+DeepEnsK5` | crepes | 0.9189 ± 0.0067 | 1.2932 ± 0.0191 | 1.5817 ± 0.0388 | 0.3756 ± 0.0061 | - | 2.0466 |
| `crepes_Normalized+LightGBM` | crepes | 0.8933 ± 0.0093 | 1.3770 ± 0.0423 | 1.7236 ± 0.0276 | 0.3986 ± 0.0050 | - | 0.5274 |
| `crepes_Normalized+RF` | crepes | 0.9183 ± 0.0117 | 1.4095 ± 0.0164 | 1.7063 ± 0.0207 | 0.3984 ± 0.0060 | - | 0.6232 |
| `crepes_Normalized+TabularMLP` | crepes | 0.9244 ± 0.0025 | 1.2217 ± 0.0189 | 1.4250 ± 0.0263 | **0.3629 ± 0.0021** | - | 3.1345 |
| `crepes_Split+DeepEnsK5` | crepes | 0.9206 ± 0.0086 | 1.3324 ± 0.0616 | 1.6622 ± 0.0313 | 0.3756 ± 0.0061 | - | 2.0467 |
| `crepes_Split+LightGBM` | crepes | 0.9039 ± 0.0155 | 1.3281 ± 0.0432 | 1.7697 ± 0.0485 | 0.3986 ± 0.0050 | - | 0.5274 |
| `crepes_Split+RF` | crepes | 0.9233 ± 0.0087 | 1.4446 ± 0.0497 | 1.8105 ± 0.0244 | 0.3984 ± 0.0060 | - | 0.6232 |
| `crepes_Split+TabularMLP` | crepes | 0.9283 ± 0.0101 | 1.3021 ± 0.0411 | 1.6243 ± 0.0254 | **0.3629 ± 0.0021** | - | 3.1345 |
| `tr_CQR+DeepEnsK5` | torchregress | 0.9222 ± 0.0108 | 1.2960 ± 0.0448 | 1.5803 ± 0.0418 | 0.3756 ± 0.0061 | - | 2.0468 |
| `tr_CQR+LightGBM` | torchregress | 0.9067 ± 0.0088 | 1.4217 ± 0.0332 | 1.7357 ± 0.0138 | 0.3986 ± 0.0050 | - | 0.5289 |
| `tr_CQR+RF` | torchregress | 0.9167 ± 0.0176 | 1.4047 ± 0.0656 | 1.7054 ± 0.0302 | 0.3984 ± 0.0060 | - | 0.6237 |
| `tr_CQR+TabularMLP` | torchregress | 0.9267 ± 0.0050 | 1.2139 ± 0.0125 | 1.4205 ± 0.0216 | **0.3629 ± 0.0021** | - | 3.1346 |
| `tr_CVPlus+LightGBM` | torchregress | 0.9256 ± 0.0054 | 1.3964 ± 0.0124 | 1.6963 ± 0.0073 | 0.3799 ± 0.0023 | - | 1.1977 |
| `tr_CVPlus+RF` | torchregress | 0.9294 ± 0.0025 | 1.4189 ± 0.0097 | 1.7566 ± 0.0064 | 0.3892 ± 0.0005 | - | 3.8871 |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | 0.9161 ± 0.0242 | 1.3590 ± 0.1397 | 1.6514 ± 0.0635 | 0.3972 ± 0.0180 | 0.2172 ± 0.0103 | 2.0473 |
| `tr_IsoMean+VTS+LightGBM` | torchregress | 0.9061 ± 0.0051 | 1.4644 ± 0.0546 | 1.7814 ± 0.0512 | 0.4288 ± 0.0242 | 0.2319 ± 0.0067 | 0.5285 |
| `tr_IsoMean+VTS+RF` | torchregress | 0.9122 ± 0.0142 | 1.4664 ± 0.0651 | 1.8251 ± 0.0289 | 0.4300 ± 0.0156 | 0.2338 ± 0.0076 | 0.6239 |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | 0.9006 ± 0.0180 | 1.2317 ± 0.0362 | 1.5177 ± 0.0616 | 0.3877 ± 0.0069 | 0.2093 ± 0.0050 | 3.1351 |
| `tr_Normalized+DeepEnsK5` | torchregress | 0.9189 ± 0.0067 | 1.2932 ± 0.0191 | 1.5817 ± 0.0388 | 0.3756 ± 0.0061 | - | 2.0467 |
| `tr_Normalized+LightGBM` | torchregress | 0.8933 ± 0.0093 | 1.3770 ± 0.0423 | 1.7236 ± 0.0276 | 0.3986 ± 0.0050 | - | 0.5276 |
| `tr_Normalized+RF` | torchregress | 0.9183 ± 0.0117 | 1.4095 ± 0.0164 | 1.7063 ± 0.0207 | 0.3984 ± 0.0060 | - | 0.6233 |
| `tr_Normalized+TabularMLP` | torchregress | 0.9244 ± 0.0025 | 1.2217 ± 0.0189 | 1.4250 ± 0.0263 | **0.3629 ± 0.0021** | - | 3.1346 |
| `tr_Split+DeepEnsK5` | torchregress | 0.9206 ± 0.0086 | 1.3324 ± 0.0616 | 1.6622 ± 0.0313 | 0.3756 ± 0.0061 | - | 2.0467 |
| `tr_Split+LightGBM` | torchregress | 0.9039 ± 0.0155 | 1.3281 ± 0.0432 | 1.7697 ± 0.0485 | 0.3986 ± 0.0050 | - | 0.5276 |
| `tr_Split+RF` | torchregress | 0.9233 ± 0.0087 | 1.4446 ± 0.0497 | 1.8105 ± 0.0244 | 0.3984 ± 0.0060 | - | 0.6233 |
| `tr_Split+TabularMLP` | torchregress | 0.9283 ± 0.0101 | 1.3021 ± 0.0411 | 1.6243 ± 0.0254 | **0.3629 ± 0.0021** | - | 3.1346 |
| `tr_VTS+DeepEnsK5` | torchregress | 0.9156 ± 0.0206 | 1.2715 ± 0.0835 | 1.5822 ± 0.0456 | 0.3756 ± 0.0061 | 0.2049 ± 0.0046 | 2.0469 |
| `tr_VTS+LightGBM` | torchregress | 0.8867 ± 0.0044 | 1.3522 ± 0.0490 | 1.7188 ± 0.0264 | 0.3986 ± 0.0050 | 0.2181 ± 0.0010 | 0.5280 |
| `tr_VTS+RF` | torchregress | 0.9128 ± 0.0082 | 1.3784 ± 0.0152 | 1.6990 ± 0.0130 | 0.3984 ± 0.0060 | 0.2168 ± 0.0026 | 0.6236 |
| `tr_VTS+TabularMLP` | torchregress | 0.9161 ± 0.0108 | 1.1930 ± 0.0329 | 1.4183 ± 0.0239 | **0.3629 ± 0.0021** | 0.1952 ± 0.0015 | 3.1347 |

Failed / skipped:

- `MAPIE_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=1400 models (> 500)
- `tr_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=1400 models (> 500)

##### uq_layer / tabred_cooking

| Method | Library | Coverage | Width | IntervalScore | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | 0.8961 ± 0.0025 | 1.7427 ± 0.0253 | 2.2589 ± 0.0143 | 0.5275 ± 0.0020 | - | 26.4593 |
| `MAPIE_CQR+LightGBM` | MAPIE | 0.8947 ± 0.0102 | 1.5566 ± 0.0422 | 1.9922 ± 0.0034 | 0.4809 ± 0.0003 | - | 15.1378 |
| `MAPIE_CQR+RF` | MAPIE | 0.9004 ± 0.0080 | 1.5481 ± 0.0276 | 1.9659 ± 0.0023 | 0.4832 ± 0.0006 | - | 107.4 |
| `MAPIE_CQR+TabularMLP` | MAPIE | 0.8922 ± 0.0075 | 1.5189 ± 0.0304 | 1.9871 ± 0.0125 | 0.4841 ± 0.0016 | - | 25.1999 |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | 0.8971 ± 0.0031 | 1.7470 ± 0.0283 | 2.2574 ± 0.0158 | 0.5275 ± 0.0020 | - | 26.4614 |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | 0.8972 ± 0.0072 | 1.5638 ± 0.0366 | 1.9853 ± 0.0047 | 0.4809 ± 0.0003 | - | 15.1390 |
| `MAPIE_CQR_asym+RF` | MAPIE | 0.9023 ± 0.0073 | 1.5534 ± 0.0276 | **1.9636 ± 0.0011** | 0.4832 ± 0.0006 | - | 107.4 |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | 0.8937 ± 0.0077 | 1.5231 ± 0.0288 | 1.9826 ± 0.0135 | 0.4841 ± 0.0016 | - | 25.2004 |
| `MAPIE_CVPlus+LightGBM` | MAPIE | 0.8979 ± 0.0013 | 1.5519 ± 0.0052 | 2.0325 ± 0.0016 | **0.4758 ± 0.0002** | - | 34.0991 |
| `MAPIE_CVPlus+RF` | MAPIE | 0.8994 ± 0.0008 | 1.5541 ± 0.0005 | 2.0462 ± 0.0012 | 0.4804 ± 0.0001 | - | 737.1 |
| `MAPIE_Split+DeepEnsK5` | MAPIE | 0.8939 ± 0.0017 | 1.7177 ± 0.0131 | 2.2758 ± 0.0103 | 0.5275 ± 0.0020 | - | 26.4392 |
| `MAPIE_Split+LightGBM` | MAPIE | 0.8950 ± 0.0060 | 1.5623 ± 0.0252 | 2.0585 ± 0.0089 | 0.4809 ± 0.0003 | - | 15.1106 |
| `MAPIE_Split+RF` | MAPIE | 0.9034 ± 0.0044 | 1.5862 ± 0.0283 | 2.0642 ± 0.0021 | 0.4832 ± 0.0006 | - | 107.4 |
| `MAPIE_Split+TabularMLP` | MAPIE | 0.8963 ± 0.0041 | 1.5750 ± 0.0379 | 2.0662 ± 0.0106 | 0.4841 ± 0.0016 | - | 25.1742 |
| `Raw+DeepEnsK5` | torchregress | 0.7313 ± 0.0031 | 1.1563 ± 0.0175 | 2.7033 ± 0.0492 | 0.5275 ± 0.0020 | 0.3022 ± 0.0013 | 26.4213 |
| `Raw+LightGBM` | lightgbm | 0.8299 ± 0.0034 | 1.3464 ± 0.0186 | 2.0653 ± 0.0148 | 0.4809 ± 0.0003 | 0.2676 ± 0.0007 | 15.0810 |
| `Raw+RF` | sklearn | 0.8351 ± 0.0022 | 1.3273 ± 0.0082 | 2.0253 ± 0.0017 | 0.4832 ± 0.0006 | 0.2690 ± 0.0003 | 107.4 |
| `Raw+TabularMLP` | torchregress | 0.9019 ± 0.0035 | 1.5672 ± 0.0301 | 1.9856 ± 0.0111 | 0.4841 ± 0.0016 | 0.2686 ± 0.0008 | 25.1590 |
| `crepes_Normalized+DeepEnsK5` | crepes | 0.8969 ± 0.0062 | 1.8128 ± 0.0378 | 2.3223 ± 0.0208 | 0.5275 ± 0.0020 | - | 26.4213 |
| `crepes_Normalized+LightGBM` | crepes | 0.8922 ± 0.0055 | 1.5334 ± 0.0214 | 1.9821 ± 0.0011 | 0.4809 ± 0.0003 | - | 15.0811 |
| `crepes_Normalized+RF` | crepes | 0.8969 ± 0.0054 | 1.5304 ± 0.0183 | 1.9732 ± 0.0031 | 0.4832 ± 0.0006 | - | 107.4 |
| `crepes_Normalized+TabularMLP` | crepes | 0.8932 ± 0.0065 | 1.5200 ± 0.0298 | 1.9874 ± 0.0128 | 0.4841 ± 0.0016 | - | 25.1590 |
| `crepes_Split+DeepEnsK5` | crepes | 0.8939 ± 0.0017 | 1.7177 ± 0.0131 | 2.2758 ± 0.0103 | 0.5275 ± 0.0020 | - | 26.4213 |
| `crepes_Split+LightGBM` | crepes | 0.8950 ± 0.0060 | 1.5623 ± 0.0252 | 2.0585 ± 0.0089 | 0.4809 ± 0.0003 | - | 15.0811 |
| `crepes_Split+RF` | crepes | 0.9034 ± 0.0044 | 1.5862 ± 0.0283 | 2.0642 ± 0.0021 | 0.4832 ± 0.0006 | - | 107.4 |
| `crepes_Split+TabularMLP` | crepes | 0.8963 ± 0.0041 | 1.5750 ± 0.0379 | 2.0662 ± 0.0106 | 0.4841 ± 0.0016 | - | 25.1590 |
| `tr_CQR+DeepEnsK5` | torchregress | 0.8961 ± 0.0025 | 1.7427 ± 0.0253 | 2.2589 ± 0.0143 | 0.5275 ± 0.0020 | - | 26.4216 |
| `tr_CQR+LightGBM` | torchregress | 0.8947 ± 0.0102 | 1.5566 ± 0.0422 | 1.9922 ± 0.0034 | 0.4809 ± 0.0003 | - | 15.0857 |
| `tr_CQR+RF` | torchregress | 0.9004 ± 0.0080 | 1.5481 ± 0.0276 | 1.9659 ± 0.0023 | 0.4832 ± 0.0006 | - | 107.4 |
| `tr_CQR+TabularMLP` | torchregress | 0.8922 ± 0.0075 | 1.5189 ± 0.0304 | 1.9871 ± 0.0125 | 0.4841 ± 0.0016 | - | 25.1593 |
| `tr_CVPlus+LightGBM` | torchregress | 0.8979 ± 0.0013 | 1.5519 ± 0.0052 | 2.0325 ± 0.0016 | **0.4758 ± 0.0002** | - | 29.5317 |
| `tr_CVPlus+RF` | torchregress | 0.8994 ± 0.0008 | 1.5541 ± 0.0005 | 2.0462 ± 0.0012 | 0.4804 ± 0.0001 | - | 629.5 |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | 0.9034 ± 0.0079 | 1.8000 ± 0.0534 | 2.2402 ± 0.0159 | 0.5141 ± 0.0018 | 0.2901 ± 0.0018 | 26.4228 |
| `tr_IsoMean+VTS+LightGBM` | torchregress | 0.9029 ± 0.0028 | 1.5859 ± 0.0138 | 1.9904 ± 0.0072 | 0.4835 ± 0.0011 | 0.2681 ± 0.0007 | 15.0830 |
| `tr_IsoMean+VTS+RF` | torchregress | 0.8999 ± 0.0045 | 1.5571 ± 0.0163 | 1.9842 ± 0.0035 | 0.4862 ± 0.0013 | 0.2695 ± 0.0010 | 107.4 |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | 0.8958 ± 0.0109 | 1.5546 ± 0.0452 | 1.9975 ± 0.0073 | 0.4862 ± 0.0016 | 0.2697 ± 0.0013 | 25.1605 |
| `tr_Normalized+DeepEnsK5` | torchregress | 0.8969 ± 0.0062 | 1.8128 ± 0.0378 | 2.3223 ± 0.0208 | 0.5275 ± 0.0020 | - | 26.4216 |
| `tr_Normalized+LightGBM` | torchregress | 0.8922 ± 0.0055 | 1.5334 ± 0.0214 | 1.9821 ± 0.0011 | 0.4809 ± 0.0003 | - | 15.0815 |
| `tr_Normalized+RF` | torchregress | 0.8969 ± 0.0054 | 1.5304 ± 0.0183 | 1.9732 ± 0.0031 | 0.4832 ± 0.0006 | - | 107.4 |
| `tr_Normalized+TabularMLP` | torchregress | 0.8932 ± 0.0065 | 1.5200 ± 0.0298 | 1.9874 ± 0.0128 | 0.4841 ± 0.0016 | - | 25.1593 |
| `tr_Split+DeepEnsK5` | torchregress | 0.8939 ± 0.0017 | 1.7177 ± 0.0131 | 2.2758 ± 0.0103 | 0.5275 ± 0.0020 | - | 26.4216 |
| `tr_Split+LightGBM` | torchregress | 0.8950 ± 0.0060 | 1.5623 ± 0.0252 | 2.0585 ± 0.0089 | 0.4809 ± 0.0003 | - | 15.0817 |
| `tr_Split+RF` | torchregress | 0.9034 ± 0.0044 | 1.5862 ± 0.0283 | 2.0642 ± 0.0021 | 0.4832 ± 0.0006 | - | 107.4 |
| `tr_Split+TabularMLP` | torchregress | 0.8963 ± 0.0041 | 1.5750 ± 0.0379 | 2.0662 ± 0.0106 | 0.4841 ± 0.0016 | - | 25.1592 |
| `tr_VTS+DeepEnsK5` | torchregress | 0.9018 ± 0.0060 | 1.8420 ± 0.0375 | 2.3258 ± 0.0205 | 0.5275 ± 0.0020 | 0.2958 ± 0.0013 | 26.4218 |
| `tr_VTS+LightGBM` | torchregress | 0.8979 ± 0.0038 | 1.5571 ± 0.0111 | 1.9815 ± 0.0019 | 0.4809 ± 0.0003 | **0.2665 ± 0.0005** | 15.0823 |
| `tr_VTS+RF` | torchregress | 0.8996 ± 0.0043 | 1.5447 ± 0.0170 | 1.9728 ± 0.0024 | 0.4832 ± 0.0006 | 0.2677 ± 0.0004 | 107.4 |
| `tr_VTS+TabularMLP` | torchregress | 0.8966 ± 0.0041 | 1.5354 ± 0.0203 | 1.9860 ± 0.0115 | 0.4841 ± 0.0016 | 0.2686 ± 0.0008 | 25.1596 |

Failed / skipped:

- `MAPIE_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=10000 models (> 500)
- `tr_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=10000 models (> 500)

##### uq_layer / tabred_delivery

| Method | Library | Coverage | Width | IntervalScore | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | 0.8897 ± 0.0064 | 2.0021 ± 0.0125 | 2.6244 ± 0.0115 | 0.6163 ± 0.0054 | - | 23.8328 |
| `MAPIE_CQR+LightGBM` | MAPIE | 0.9079 ± 0.0079 | 1.9063 ± 0.0350 | 2.2659 ± 0.0039 | 0.5643 ± 0.0025 | - | 13.0743 |
| `MAPIE_CQR+RF` | MAPIE | 0.8881 ± 0.0072 | 1.7619 ± 0.0274 | 2.2637 ± 0.0122 | 0.5630 ± 0.0023 | - | 101.7 |
| `MAPIE_CQR+TabularMLP` | MAPIE | 0.8951 ± 0.0065 | 1.7612 ± 0.0260 | 2.2563 ± 0.0120 | 0.5596 ± 0.0014 | - | 23.9471 |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | 0.8912 ± 0.0079 | 2.0346 ± 0.0185 | 2.6036 ± 0.0205 | 0.6163 ± 0.0054 | - | 23.8304 |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | 0.9070 ± 0.0101 | 1.9144 ± 0.0368 | 2.2663 ± 0.0041 | 0.5643 ± 0.0025 | - | 13.0727 |
| `MAPIE_CQR_asym+RF` | MAPIE | 0.8884 ± 0.0093 | 1.7918 ± 0.0274 | 2.2629 ± 0.0105 | 0.5630 ± 0.0023 | - | 101.7 |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | 0.8938 ± 0.0008 | 1.8024 ± 0.0111 | **2.2513 ± 0.0069** | 0.5596 ± 0.0014 | - | 23.9526 |
| `MAPIE_CVPlus+LightGBM` | MAPIE | 0.8974 ± 0.0044 | 1.7879 ± 0.0094 | 2.2827 ± 0.0036 | **0.5564 ± 0.0006** | - | 33.7578 |
| `MAPIE_CVPlus+RF` | MAPIE | 0.8946 ± 0.0021 | 1.7820 ± 0.0031 | 2.3205 ± 0.0030 | 0.5596 ± 0.0001 | - | 712.0 |
| `MAPIE_Split+DeepEnsK5` | MAPIE | 0.8961 ± 0.0077 | 2.0031 ± 0.0366 | 2.5846 ± 0.0211 | 0.6163 ± 0.0054 | - | 23.8050 |
| `MAPIE_Split+LightGBM` | MAPIE | 0.8882 ± 0.0062 | 1.7753 ± 0.0108 | 2.3189 ± 0.0185 | 0.5643 ± 0.0025 | - | 13.0480 |
| `MAPIE_Split+RF` | MAPIE | 0.8907 ± 0.0031 | 1.7715 ± 0.0048 | 2.3388 ± 0.0154 | 0.5630 ± 0.0023 | - | 101.7 |
| `MAPIE_Split+TabularMLP` | MAPIE | 0.8891 ± 0.0055 | 1.7686 ± 0.0221 | 2.2917 ± 0.0086 | 0.5596 ± 0.0014 | - | 23.9261 |
| `Raw+DeepEnsK5` | torchregress | 0.6973 ± 0.0038 | 1.2870 ± 0.0144 | 3.2880 ± 0.0172 | 0.6163 ± 0.0054 | 0.3599 ± 0.0031 | 23.7860 |
| `Raw+LightGBM` | lightgbm | 0.8426 ± 0.0042 | 1.6680 ± 0.0100 | 2.3163 ± 0.0080 | 0.5643 ± 0.0025 | 0.3174 ± 0.0012 | 13.0257 |
| `Raw+RF` | sklearn | 0.8439 ± 0.0011 | 1.5938 ± 0.0142 | 2.3189 ± 0.0173 | 0.5630 ± 0.0023 | 0.3156 ± 0.0014 | 101.7 |
| `Raw+TabularMLP` | torchregress | 0.9136 ± 0.0194 | 1.8545 ± 0.0966 | 2.2669 ± 0.0072 | 0.5596 ± 0.0014 | 0.3134 ± 0.0007 | 23.9107 |
| `crepes_Normalized+DeepEnsK5` | crepes | 0.8899 ± 0.0042 | 2.1034 ± 0.0111 | 2.7527 ± 0.0076 | 0.6163 ± 0.0054 | - | 23.7861 |
| `crepes_Normalized+LightGBM` | crepes | 0.9009 ± 0.0054 | 1.8198 ± 0.0297 | 2.2773 ± 0.0146 | 0.5643 ± 0.0025 | - | 13.0257 |
| `crepes_Normalized+RF` | crepes | 0.8937 ± 0.0084 | 1.7632 ± 0.0222 | 2.3120 ± 0.0199 | 0.5630 ± 0.0023 | - | 101.7 |
| `crepes_Normalized+TabularMLP` | crepes | 0.8946 ± 0.0057 | 1.7581 ± 0.0250 | 2.2544 ± 0.0135 | 0.5596 ± 0.0014 | - | 23.9107 |
| `crepes_Split+DeepEnsK5` | crepes | 0.8961 ± 0.0077 | 2.0031 ± 0.0366 | 2.5846 ± 0.0211 | 0.6163 ± 0.0054 | - | 23.7861 |
| `crepes_Split+LightGBM` | crepes | 0.8882 ± 0.0062 | 1.7753 ± 0.0108 | 2.3189 ± 0.0185 | 0.5643 ± 0.0025 | - | 13.0258 |
| `crepes_Split+RF` | crepes | 0.8907 ± 0.0031 | 1.7715 ± 0.0048 | 2.3388 ± 0.0154 | 0.5630 ± 0.0023 | - | 101.7 |
| `crepes_Split+TabularMLP` | crepes | 0.8891 ± 0.0055 | 1.7686 ± 0.0221 | 2.2917 ± 0.0086 | 0.5596 ± 0.0014 | - | 23.9107 |
| `tr_CQR+DeepEnsK5` | torchregress | 0.8897 ± 0.0064 | 2.0021 ± 0.0125 | 2.6244 ± 0.0115 | 0.6163 ± 0.0054 | - | 23.7864 |
| `tr_CQR+LightGBM` | torchregress | 0.9079 ± 0.0079 | 1.9063 ± 0.0350 | 2.2659 ± 0.0039 | 0.5643 ± 0.0025 | - | 13.0263 |
| `tr_CQR+RF` | torchregress | 0.8881 ± 0.0072 | 1.7619 ± 0.0274 | 2.2637 ± 0.0122 | 0.5630 ± 0.0023 | - | 101.7 |
| `tr_CQR+TabularMLP` | torchregress | 0.8951 ± 0.0065 | 1.7612 ± 0.0260 | 2.2563 ± 0.0120 | 0.5596 ± 0.0014 | - | 23.9110 |
| `tr_CVPlus+LightGBM` | torchregress | 0.8974 ± 0.0044 | 1.7879 ± 0.0094 | 2.2827 ± 0.0036 | **0.5564 ± 0.0006** | - | 28.4102 |
| `tr_CVPlus+RF` | torchregress | 0.8946 ± 0.0021 | 1.7820 ± 0.0031 | 2.3205 ± 0.0030 | 0.5596 ± 0.0001 | - | 580.7 |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | 0.8956 ± 0.0008 | 2.1009 ± 0.0378 | 2.6827 ± 0.0442 | 0.5979 ± 0.0043 | 0.3410 ± 0.0025 | 23.7875 |
| `tr_IsoMean+VTS+LightGBM` | torchregress | 0.9012 ± 0.0012 | 1.8320 ± 0.0312 | 2.2949 ± 0.0228 | 0.5680 ± 0.0022 | 0.3185 ± 0.0016 | 13.0275 |
| `tr_IsoMean+VTS+RF` | torchregress | 0.9019 ± 0.0079 | 1.8183 ± 0.0396 | 2.3395 ± 0.0272 | 0.5652 ± 0.0040 | 0.3154 ± 0.0027 | 101.7 |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | 0.9019 ± 0.0031 | 1.8049 ± 0.0230 | 2.2722 ± 0.0162 | 0.5642 ± 0.0017 | 0.3162 ± 0.0013 | 23.9120 |
| `tr_Normalized+DeepEnsK5` | torchregress | 0.8899 ± 0.0042 | 2.1034 ± 0.0111 | 2.7527 ± 0.0076 | 0.6163 ± 0.0054 | - | 23.7863 |
| `tr_Normalized+LightGBM` | torchregress | 0.9009 ± 0.0054 | 1.8198 ± 0.0297 | 2.2773 ± 0.0146 | 0.5643 ± 0.0025 | - | 13.0261 |
| `tr_Normalized+RF` | torchregress | 0.8937 ± 0.0084 | 1.7632 ± 0.0222 | 2.3120 ± 0.0199 | 0.5630 ± 0.0023 | - | 101.7 |
| `tr_Normalized+TabularMLP` | torchregress | 0.8946 ± 0.0057 | 1.7581 ± 0.0250 | 2.2544 ± 0.0135 | 0.5596 ± 0.0014 | - | 23.9109 |
| `tr_Split+DeepEnsK5` | torchregress | 0.8961 ± 0.0077 | 2.0031 ± 0.0366 | 2.5846 ± 0.0211 | 0.6163 ± 0.0054 | - | 23.7863 |
| `tr_Split+LightGBM` | torchregress | 0.8882 ± 0.0062 | 1.7753 ± 0.0108 | 2.3189 ± 0.0185 | 0.5643 ± 0.0025 | - | 13.0261 |
| `tr_Split+RF` | torchregress | 0.8907 ± 0.0031 | 1.7715 ± 0.0048 | 2.3388 ± 0.0154 | 0.5630 ± 0.0023 | - | 101.7 |
| `tr_Split+TabularMLP` | torchregress | 0.8891 ± 0.0055 | 1.7686 ± 0.0221 | 2.2917 ± 0.0086 | 0.5596 ± 0.0014 | - | 23.9109 |
| `tr_VTS+DeepEnsK5` | torchregress | 0.9006 ± 0.0029 | 2.1793 ± 0.0187 | 2.7641 ± 0.0058 | 0.6163 ± 0.0054 | 0.3496 ± 0.0028 | 23.7867 |
| `tr_VTS+LightGBM` | torchregress | 0.9044 ± 0.0034 | 1.8393 ± 0.0202 | 2.2779 ± 0.0154 | 0.5643 ± 0.0025 | 0.3165 ± 0.0011 | 13.0269 |
| `tr_VTS+RF` | torchregress | 0.9046 ± 0.0020 | 1.8152 ± 0.0096 | 2.3129 ± 0.0205 | 0.5630 ± 0.0023 | 0.3145 ± 0.0013 | 101.7 |
| `tr_VTS+TabularMLP` | torchregress | 0.9067 ± 0.0038 | 1.8146 ± 0.0217 | 2.2553 ± 0.0150 | 0.5596 ± 0.0014 | **0.3133 ± 0.0007** | 23.9113 |

Failed / skipped:

- `MAPIE_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=10000 models (> 500)
- `tr_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=10000 models (> 500)

##### uq_layer / tabred_weather

| Method | Library | Coverage | Width | IntervalScore | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | 0.9089 ± 0.0034 | 5.6204 ± 0.1141 | 7.5074 ± 0.0658 | 1.7551 ± 0.0239 | - | 20.0693 |
| `MAPIE_CQR+LightGBM` | MAPIE | 0.9092 ± 0.0040 | 5.8276 ± 0.0704 | 7.5487 ± 0.0281 | 1.6916 ± 0.0140 | - | 5.8211 |
| `MAPIE_CQR+RF` | MAPIE | 0.9028 ± 0.0059 | 5.5364 ± 0.0729 | 7.3896 ± 0.0493 | 1.7536 ± 0.0032 | - | 40.9951 |
| `MAPIE_CQR+TabularMLP` | MAPIE | 0.9082 ± 0.0048 | 5.4131 ± 0.0158 | 7.1083 ± 0.0371 | 1.7049 ± 0.0190 | - | 19.4194 |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | 0.9063 ± 0.0072 | 5.6307 ± 0.1470 | 7.4861 ± 0.0766 | 1.7551 ± 0.0239 | - | 20.0691 |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | 0.9097 ± 0.0062 | 5.8619 ± 0.1067 | 7.5684 ± 0.0420 | 1.6916 ± 0.0140 | - | 5.8211 |
| `MAPIE_CQR_asym+RF` | MAPIE | 0.9049 ± 0.0047 | 5.5948 ± 0.1219 | 7.4055 ± 0.0369 | 1.7536 ± 0.0032 | - | 40.9951 |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | 0.9071 ± 0.0032 | 5.4475 ± 0.0482 | 7.1200 ± 0.0650 | 1.7049 ± 0.0190 | - | 19.4193 |
| `MAPIE_CVPlus+LightGBM` | MAPIE | 0.9127 ± 0.0006 | 5.4030 ± 0.0013 | 7.3757 ± 0.0102 | **1.6617 ± 0.0013** | - | 15.8709 |
| `MAPIE_CVPlus+RF` | MAPIE | 0.9042 ± 0.0007 | 5.5701 ± 0.0243 | 7.6916 ± 0.0025 | 1.7368 ± 0.0005 | - | 307.8 |
| `MAPIE_Split+DeepEnsK5` | MAPIE | 0.9078 ± 0.0095 | 5.6137 ± 0.1680 | 7.7778 ± 0.1079 | 1.7551 ± 0.0239 | - | 20.0604 |
| `MAPIE_Split+LightGBM` | MAPIE | 0.9072 ± 0.0104 | 5.3710 ± 0.1704 | 7.5120 ± 0.0557 | 1.6916 ± 0.0140 | - | 5.8119 |
| `MAPIE_Split+RF` | MAPIE | 0.9040 ± 0.0100 | 5.5929 ± 0.1903 | 7.7544 ± 0.0146 | 1.7536 ± 0.0032 | - | 40.9858 |
| `MAPIE_Split+TabularMLP` | MAPIE | 0.9071 ± 0.0082 | 5.4590 ± 0.1268 | 7.5293 ± 0.0209 | 1.7049 ± 0.0190 | - | 19.4100 |
| `Raw+DeepEnsK5` | torchregress | 0.8223 ± 0.0074 | 4.4146 ± 0.0265 | 7.8722 ± 0.1160 | 1.7551 ± 0.0239 | 0.9492 ± 0.0136 | 20.0530 |
| `Raw+LightGBM` | lightgbm | 0.8394 ± 0.0010 | 4.9016 ± 0.0654 | 7.7560 ± 0.0459 | 1.6916 ± 0.0140 | **0.9046 ± 0.0085** | 5.8013 |
| `Raw+RF` | sklearn | 0.8797 ± 0.0030 | 5.1305 ± 0.0649 | 7.4211 ± 0.0502 | 1.7536 ± 0.0032 | 0.9402 ± 0.0020 | 40.9789 |
| `Raw+TabularMLP` | torchregress | 0.9246 ± 0.0012 | 5.7008 ± 0.0828 | 7.1565 ± 0.0350 | 1.7049 ± 0.0190 | 0.9157 ± 0.0145 | 19.4032 |
| `crepes_Normalized+DeepEnsK5` | crepes | 0.9059 ± 0.0033 | 5.6958 ± 0.0796 | 7.5652 ± 0.0601 | 1.7551 ± 0.0239 | - | 20.0530 |
| `crepes_Normalized+LightGBM` | crepes | 0.9099 ± 0.0088 | 5.4530 ± 0.0986 | 7.2129 ± 0.0740 | 1.6916 ± 0.0140 | - | 5.8013 |
| `crepes_Normalized+RF` | crepes | 0.9070 ± 0.0092 | 5.5231 ± 0.1211 | 7.3362 ± 0.0403 | 1.7536 ± 0.0032 | - | 40.9789 |
| `crepes_Normalized+TabularMLP` | crepes | 0.9091 ± 0.0052 | 5.3846 ± 0.0432 | **7.1016 ± 0.0381** | 1.7049 ± 0.0190 | - | 19.4032 |
| `crepes_Split+DeepEnsK5` | crepes | 0.9078 ± 0.0095 | 5.6137 ± 0.1680 | 7.7778 ± 0.1079 | 1.7551 ± 0.0239 | - | 20.0531 |
| `crepes_Split+LightGBM` | crepes | 0.9072 ± 0.0104 | 5.3710 ± 0.1704 | 7.5120 ± 0.0557 | 1.6916 ± 0.0140 | - | 5.8013 |
| `crepes_Split+RF` | crepes | 0.9040 ± 0.0100 | 5.5929 ± 0.1903 | 7.7544 ± 0.0146 | 1.7536 ± 0.0032 | - | 40.9790 |
| `crepes_Split+TabularMLP` | crepes | 0.9071 ± 0.0082 | 5.4590 ± 0.1268 | 7.5293 ± 0.0209 | 1.7049 ± 0.0190 | - | 19.4033 |
| `tr_CQR+DeepEnsK5` | torchregress | 0.9089 ± 0.0034 | 5.6204 ± 0.1141 | 7.5074 ± 0.0658 | 1.7551 ± 0.0239 | - | 20.0533 |
| `tr_CQR+LightGBM` | torchregress | 0.9092 ± 0.0040 | 5.8276 ± 0.0704 | 7.5487 ± 0.0281 | 1.6916 ± 0.0140 | - | 5.8018 |
| `tr_CQR+RF` | torchregress | 0.9028 ± 0.0059 | 5.5364 ± 0.0729 | 7.3896 ± 0.0493 | 1.7536 ± 0.0032 | - | 40.9792 |
| `tr_CQR+TabularMLP` | torchregress | 0.9082 ± 0.0048 | 5.4131 ± 0.0158 | 7.1083 ± 0.0371 | 1.7049 ± 0.0190 | - | 19.4035 |
| `tr_CVPlus+LightGBM` | torchregress | 0.9127 ± 0.0006 | 5.4030 ± 0.0013 | 7.3757 ± 0.0102 | **1.6617 ± 0.0013** | - | 12.9444 |
| `tr_CVPlus+RF` | torchregress | 0.9042 ± 0.0007 | 5.5701 ± 0.0243 | 7.6916 ± 0.0025 | 1.7368 ± 0.0005 | - | 252.4 |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | 0.9240 ± 0.0087 | 6.2178 ± 0.1634 | 7.7547 ± 0.0942 | 1.7884 ± 0.0517 | 0.9680 ± 0.0206 | 20.0542 |
| `tr_IsoMean+VTS+LightGBM` | torchregress | 0.9258 ± 0.0147 | 5.8862 ± 0.2903 | 7.3968 ± 0.0317 | 1.7212 ± 0.0115 | 0.9252 ± 0.0042 | 5.8031 |
| `tr_IsoMean+VTS+RF` | torchregress | 0.9257 ± 0.0123 | 6.0017 ± 0.2201 | 7.5145 ± 0.0713 | 1.7713 ± 0.0018 | 0.9520 ± 0.0028 | 40.9804 |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | 0.9288 ± 0.0143 | 5.9169 ± 0.3044 | 7.3128 ± 0.0755 | 1.7205 ± 0.0057 | 0.9242 ± 0.0017 | 19.4044 |
| `tr_Normalized+DeepEnsK5` | torchregress | 0.9059 ± 0.0033 | 5.6958 ± 0.0796 | 7.5652 ± 0.0601 | 1.7551 ± 0.0239 | - | 20.0532 |
| `tr_Normalized+LightGBM` | torchregress | 0.9099 ± 0.0088 | 5.4530 ± 0.0986 | 7.2129 ± 0.0740 | 1.6916 ± 0.0140 | - | 5.8016 |
| `tr_Normalized+RF` | torchregress | 0.9070 ± 0.0092 | 5.5231 ± 0.1211 | 7.3362 ± 0.0403 | 1.7536 ± 0.0032 | - | 40.9791 |
| `tr_Normalized+TabularMLP` | torchregress | 0.9091 ± 0.0052 | 5.3846 ± 0.0432 | **7.1016 ± 0.0381** | 1.7049 ± 0.0190 | - | 19.4034 |
| `tr_Split+DeepEnsK5` | torchregress | 0.9078 ± 0.0095 | 5.6137 ± 0.1680 | 7.7778 ± 0.1079 | 1.7551 ± 0.0239 | - | 20.0532 |
| `tr_Split+LightGBM` | torchregress | 0.9072 ± 0.0104 | 5.3710 ± 0.1704 | 7.5120 ± 0.0557 | 1.6916 ± 0.0140 | - | 5.8017 |
| `tr_Split+RF` | torchregress | 0.9040 ± 0.0100 | 5.5929 ± 0.1903 | 7.7544 ± 0.0146 | 1.7536 ± 0.0032 | - | 40.9792 |
| `tr_Split+TabularMLP` | torchregress | 0.9071 ± 0.0082 | 5.4590 ± 0.1268 | 7.5293 ± 0.0209 | 1.7049 ± 0.0190 | - | 19.4034 |
| `tr_VTS+DeepEnsK5` | torchregress | 0.9257 ± 0.0047 | 6.1420 ± 0.0334 | 7.6614 ± 0.0565 | 1.7551 ± 0.0239 | 0.9490 ± 0.0114 | 20.0534 |
| `tr_VTS+LightGBM` | torchregress | 0.9212 ± 0.0054 | 5.7342 ± 0.1195 | 7.2621 ± 0.0674 | 1.6916 ± 0.0140 | 0.9080 ± 0.0081 | 5.8026 |
| `tr_VTS+RF` | torchregress | 0.9238 ± 0.0059 | 5.8584 ± 0.0546 | 7.3800 ± 0.0297 | 1.7536 ± 0.0032 | 0.9425 ± 0.0017 | 40.9798 |
| `tr_VTS+TabularMLP` | torchregress | 0.9281 ± 0.0053 | 5.7819 ± 0.1172 | 7.1803 ± 0.0243 | 1.7049 ± 0.0190 | 0.9165 ± 0.0138 | 19.4036 |

Failed / skipped:

- `MAPIE_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=10000 models (> 500)
- `tr_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=10000 models (> 500)

##### uq_layer / zurich_delays

| Method | Library | Coverage | Width | IntervalScore | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | 0.8992 ± 0.0012 | 265.0 ± 34.1 | 412.6 ± 38.8 † | 107.9 ± 0.1 | - | 162.6 |
| `MAPIE_CQR+LightGBM` | MAPIE | 0.9008 ± 0.0007 | 231.7 ± 0.5 | 365.3 ± 2.4 | 108.6 ± 1.8 | - | 7.9667 |
| `MAPIE_CQR+RF` | MAPIE | 0.9008 ± 0.0020 | 226.4 ± 1.6 | 359.0 ± 3.0 | 107.8 ± 3.0 | - | 50.4358 |
| `MAPIE_CQR+TabularMLP` | MAPIE | 0.8983 ± 0.0013 | 258.0 ± 15.3 | 387.6 ± 11.3 | 106.8 ± 0.9 | - | 268.6 |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | 0.8990 ± 0.0002 | 277.6 ± 35.2 | 402.2 ± 39.1 † | 107.9 ± 0.1 | - | 162.6 |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | 0.9004 ± 0.0009 | 235.7 ± 1.5 | 365.3 ± 2.3 | 108.6 ± 1.8 | - | 7.9661 |
| `MAPIE_CQR_asym+RF` | MAPIE | 0.9021 ± 0.0020 | 238.2 ± 3.6 | **357.0 ± 3.1** | 107.8 ± 3.0 | - | 50.4343 |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | 0.9007 ± 0.0011 | 270.6 ± 16.2 | 380.5 ± 11.8 | 106.8 ± 0.9 | - | 268.6 |
| `MAPIE_CVPlus+LightGBM` | MAPIE | 0.9080 ± 0.0002 | 237.6 ± 0.2 | 436.7 ± 0.9 | 107.3 ± 0.0 | - | 88.1011 |
| `MAPIE_CVPlus+RF` | MAPIE | 0.9100 ± 0.0005 | 240.8 ± 0.9 | 427.0 ± 0.5 | **106.3 ± 0.1** | - | 472.6 |
| `MAPIE_Split+DeepEnsK5` | MAPIE | 0.9014 ± 0.0007 | 233.1 ± 2.6 | 456.2 ± 2.4 | 107.9 ± 0.1 | - | 162.5 |
| `MAPIE_Split+LightGBM` | MAPIE | 0.9008 ± 0.0021 | 232.9 ± 3.6 | 457.3 ± 8.3 | 108.6 ± 1.8 | - | 7.8523 |
| `MAPIE_Split+RF` | MAPIE | 0.9018 ± 0.0003 | 237.4 ± 1.1 | 450.5 ± 5.4 | 107.8 ± 3.0 | - | 50.3594 |
| `MAPIE_Split+TabularMLP` | MAPIE | 0.9028 ± 0.0010 | 232.1 ± 1.3 | 443.7 ± 1.3 | 106.8 ± 0.9 | - | 268.5 |
| `Raw+DeepEnsK5` | torchregress | 0.9250 ± 0.0052 | 289.4 ± 38.7 | 415.9 ± 39.9 † | 107.9 ± 0.1 | 45.1019 ± 2.6040 | 162.5 |
| `Raw+LightGBM` | lightgbm | 0.8893 ± 0.0006 | 225.6 ± 0.6 | 365.6 ± 2.4 | 108.6 ± 1.8 | 43.5771 ± 0.5398 | 7.7515 |
| `Raw+RF` | sklearn | 0.7492 ± 0.0014 | 167.8 ± 1.1 | 396.0 ± 3.1 | 107.8 ± 3.0 | **43.0358 ± 0.2247** | 50.3076 |
| `Raw+TabularMLP` | torchregress | 0.9335 ± 0.0036 | 288.0 ± 11.9 | 393.0 ± 10.0 | 106.8 ± 0.9 | 43.5997 ± 0.7163 | 268.4 |
| `crepes_Normalized+DeepEnsK5` | crepes | 0.8996 ± 0.0010 | 253.4 ± 28.0 | 407.0 ± 33.8 † | 107.9 ± 0.1 | - | 162.5 |
| `crepes_Normalized+LightGBM` | crepes | 0.9005 ± 0.0011 | 224.3 ± 3.0 | 394.1 ± 8.8 | 108.6 ± 1.8 | - | 7.7517 |
| `crepes_Normalized+RF` | crepes | 0.9003 ± 0.0006 | 241.2 ± 3.0 | 376.5 ± 4.4 | 107.8 ± 3.0 | - | 50.3076 |
| `crepes_Normalized+TabularMLP` | crepes | 0.8996 ± 0.0026 | 245.3 ± 15.2 | 382.2 ± 8.6 | 106.8 ± 0.9 | - | 268.4 |
| `crepes_Split+DeepEnsK5` | crepes | 0.9014 ± 0.0007 | 233.1 ± 2.6 | 456.2 ± 2.4 | 107.9 ± 0.1 | - | 162.5 |
| `crepes_Split+LightGBM` | crepes | 0.9008 ± 0.0021 | 232.9 ± 3.6 | 457.3 ± 8.3 | 108.6 ± 1.8 | - | 7.7516 |
| `crepes_Split+RF` | crepes | 0.9018 ± 0.0003 | 237.4 ± 1.1 | 450.5 ± 5.4 | 107.8 ± 3.0 | - | 50.3075 |
| `crepes_Split+TabularMLP` | crepes | 0.9028 ± 0.0010 | 232.1 ± 1.3 | 443.7 ± 1.3 | 106.8 ± 0.9 | - | 268.4 |
| `tr_CQR+DeepEnsK5` | torchregress | 0.8992 ± 0.0012 | 265.0 ± 34.1 | 412.6 ± 38.8 † | 107.9 ± 0.1 | - | 162.5 |
| `tr_CQR+LightGBM` | torchregress | 0.9008 ± 0.0007 | 231.7 ± 0.5 | 365.3 ± 2.4 | 108.6 ± 1.8 | - | 7.7549 |
| `tr_CQR+RF` | torchregress | 0.9008 ± 0.0020 | 226.4 ± 1.6 | 359.0 ± 3.0 | 107.8 ± 3.0 | - | 50.3087 |
| `tr_CQR+TabularMLP` | torchregress | 0.8983 ± 0.0013 | 258.0 ± 15.3 | 387.6 ± 11.3 | 106.8 ± 0.9 | - | 268.4 |
| `tr_CVPlus+LightGBM` | torchregress | 0.9080 ± 0.0002 | 237.6 ± 0.2 | 436.7 ± 0.9 | 107.3 ± 0.0 | - | 42.5965 |
| `tr_CVPlus+RF` | torchregress | 0.9100 ± 0.0005 | 240.8 ± 0.9 | 427.0 ± 0.5 | **106.3 ± 0.1** | - | 343.5 |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | 0.9428 ± 0.0095 | 330.5 ± 17.8 | 433.7 ± 27.7 † | 108.5 ± 0.6 | 46.3191 ± 2.0252 | 162.5 |
| `tr_IsoMean+VTS+LightGBM` | torchregress | 0.9494 ± 0.0228 | 320.6 ± 83.6 | 442.4 ± 79.7 † | 118.8 ± 17.8 | 46.1495 ± 3.9925 | 7.7616 |
| `tr_IsoMean+VTS+RF` | torchregress | 0.9396 ± 0.0086 | 298.9 ± 17.1 | 409.0 ± 26.6 † | 122.4 ± 25.9 | 44.4316 ± 1.4846 | 50.3137 |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | 0.9569 ± 0.0332 | 447.7 ± 245.9 | 538.3 ± 234.2 † | 122.0 ± 26.5 | 51.7606 ± 12.7237 | 268.4 |
| `tr_Normalized+DeepEnsK5` | torchregress | 0.8996 ± 0.0010 | 253.4 ± 28.0 | 407.0 ± 33.8 † | 107.9 ± 0.1 | - | 162.5 |
| `tr_Normalized+LightGBM` | torchregress | 0.9005 ± 0.0011 | 224.3 ± 3.0 | 394.1 ± 8.8 | 108.6 ± 1.8 | - | 7.7534 |
| `tr_Normalized+RF` | torchregress | 0.9003 ± 0.0006 | 241.2 ± 3.0 | 376.5 ± 4.4 | 107.8 ± 3.0 | - | 50.3086 |
| `tr_Normalized+TabularMLP` | torchregress | 0.8996 ± 0.0026 | 245.3 ± 15.2 | 382.2 ± 8.6 | 106.8 ± 0.9 | - | 268.4 |
| `tr_Split+DeepEnsK5` | torchregress | 0.9014 ± 0.0007 | 233.1 ± 2.6 | 456.2 ± 2.4 | 107.9 ± 0.1 | - | 162.5 |
| `tr_Split+LightGBM` | torchregress | 0.9008 ± 0.0021 | 232.9 ± 3.6 | 457.3 ± 8.3 | 108.6 ± 1.8 | - | 7.7534 |
| `tr_Split+RF` | torchregress | 0.9018 ± 0.0003 | 237.4 ± 1.1 | 450.5 ± 5.4 | 107.8 ± 3.0 | - | 50.3086 |
| `tr_Split+TabularMLP` | torchregress | 0.9028 ± 0.0010 | 232.1 ± 1.3 | 443.7 ± 1.3 | 106.8 ± 0.9 | - | 268.4 |
| `tr_VTS+DeepEnsK5` | torchregress | 0.9597 ± 0.0345 | 673.0 ± 594.6 | 749.6 ± 535.5 † | 107.9 ± 0.1 | 65.4726 ± 32.8778 | 162.5 |
| `tr_VTS+LightGBM` | torchregress | 0.9583 ± 0.0303 | 403.2 ± 206.9 | 495.8 ± 149.0 † | 108.6 ± 1.8 | 49.1082 ± 8.1086 | 7.7539 |
| `tr_VTS+RF` | torchregress | 0.9424 ± 0.0157 | 306.8 ± 35.4 | 399.0 ± 14.9 | 107.8 ± 3.0 | 43.7757 ± 0.6337 | 50.3095 |
| `tr_VTS+TabularMLP` | torchregress | 0.9559 ± 0.0326 | 410.4 ± 185.5 | 483.1 ± 139.2 † | 106.8 ± 0.9 | 48.4560 ± 7.3949 | 268.4 |

Failed / skipped:

- `MAPIE_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=70000 models (> 500)
- `tr_JackknifePlus+LightGBM` (skipped, seed 0, 1, 2): skipped: Jackknife+ refits n=70000 models (> 500)

### Group `tabular_big`

#### eiv (tabular_big)

Group `tabular_big`; 1 dataset(s). Columns: RMSE ↓, R2 ↑, Attenuation ·, RMSE_clean ↓, Coverage ·, IntervalScore ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: RMSE.

##### eiv / folktables_eiv

| Method | Library | RMSE | R2 | Attenuation | RMSE_clean | Coverage | IntervalScore | train_s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EnsembleEIV` | torchregress | 58681.2 ± 42.9 | 0.3505 ± 0.0010 | 0.3742 ± 0.0152 | 58530.9 ± 54.2 | - | - | 53.9879 |
| `FunctionalEIV` | torchregress | 59437.5 ± 167.4 | 0.3336 ± 0.0038 | 0.3840 ± 0.0160 | 59477.3 ± 290.2 | - | - | 136.1 |
| `FunctionalEIV_MC` | torchregress | 61388.8 ± 59.0 | 0.2891 ± 0.0014 | 0.3342 ± 0.0209 | 59612.5 ± 465.9 | - | - | 123.8 |
| `GaussianNLL+InputNoiseMC` | torchregress | 58773.4 ± 93.5 | 0.3484 ± 0.0021 | 0.3331 ± 0.0079 | - | - | - | 68.6805 |
| `InputNoiseBinnedPDF` | torchregress | 1.134e+05 ± 1.210e+03 | -1.4263 ± 0.0518 | 0.9873 ± 0.0073 | 1.132e+05 ± 1.220e+03 | - | - | 55.5264 |
| `InputNoiseMC` | torchregress | 58502.8 ± 49.7 | 0.3544 ± 0.0011 | 0.3687 ± 0.0148 | 58395.5 ± 44.1 | - | - | 50.2925 |
| `InputNoiseMDN` | torchregress | **58192.2 ± 156.9** | **0.3612 ± 0.0034** | 0.3715 ± 0.0085 | **58082.9 ± 159.6** | - | - | 83.7023 |
| `LatentMarginalization` | torchregress | 90834.7 ± 5824.9 † | -0.5606 ± 0.1988 | 0.5226 ± 0.0393 | 91593.4 ± 6373.8 | - | - | 62.3922 |
| `LatentNN` | torchregress | 65672.5 ± 2717.6 | 0.1855 ± 0.0681 | 0.4277 ± 0.0387 | 63862.5 ± 2592.7 | - | - | 68.2521 |
| `MSE` | torchregress | 58697.3 ± 309.0 | 0.3501 ± 0.0069 | 0.3861 ± 0.0270 | 58571.6 ± 266.9 | - | - | 37.1245 |
| `MSE_NoisyPredict` | torchregress | 58624.2 ± 298.8 | 0.3517 ± 0.0066 | 0.3835 ± 0.0266 | 58571.6 ± 266.9 | - | - | 42.7420 |
| `MSE_WarmupMC` | torchregress | 60826.9 ± 512.5 | 0.3020 ± 0.0117 | 0.3558 ± 0.0285 | 59418.8 ± 422.6 | - | - | 93.8261 |
| `NoiseAware` | torchregress | 58718.1 ± 151.6 | 0.3496 ± 0.0034 | 0.3865 ± 0.0111 | 58598.3 ± 165.9 | - | - | 85.8252 |
| `ODR` | torchregress | 1.148e+05 ± 1.545e+04 † | -1.5168 ± 0.6807 | 0.5006 ± 0.1423 | 91545.8 ± 14425.9 | - | - | 477.7 |
| `RC` | torchregress | 58767.6 ± 369.9 | 0.3485 ± 0.0082 | 0.3893 ± 0.0277 | 58601.5 ± 360.8 | - | - | 43.5110 |
| `RC+FunctionalEIV` | torchregress | 58960.2 ± 299.7 | 0.3443 ± 0.0067 | 0.3877 ± 0.0263 | 58848.8 ± 277.1 | - | - | 129.9 |
| `RC+GaussianNLL` | torchregress | 58720.2 ± 83.0 | 0.3496 ± 0.0018 | 0.3373 ± 0.0051 | - | - | - | 56.2310 |
| `SIMEX` | torchregress | 60550.6 ± 372.1 | 0.3084 ± 0.0085 | 0.4027 ± 0.0196 | 60296.0 ± 354.6 | - | - | 1081.8 |
| `SciPyODR` | scipy | 62134.0 ± 0.0 | 0.2718 ± 0.0000 | 0.2943 ± 0.0000 | - | - | - | 9187.1 |
| `SplitCP+FunctionalEIV` | torchregress+CP | 60016.5 ± 481.3 | 0.3205 ± 0.0109 | 0.3752 ± 0.0101 | 59743.3 ± 332.8 | 0.8997 ± 0.0019 | 2.744e+05 ± 1.582e+03 | 99.4871 |
| `SplitCP+InputNoiseMC` | torchregress+CP | 58788.8 ± 306.9 | 0.3481 ± 0.0068 | 0.3735 ± 0.0165 | 58701.8 ± 308.5 | 0.8992 ± 0.0020 | **2.706e+05 ± 2.846e+03** | 39.2224 |
| `SplitCP+MSE_WarmupMC` | torchregress+CP | 61017.5 ± 564.9 | 0.2977 ± 0.0130 | 0.3488 ± 0.0277 | 59884.2 ± 499.2 | 0.9005 ± 0.0022 | 2.834e+05 ± 3.413e+03 | 71.5374 |
| `StructuralEIV` | torchregress | 59437.5 ± 167.4 | 0.3336 ± 0.0038 | 0.3840 ± 0.0160 | 59477.3 ± 290.2 | - | - | 147.5 |

Failed / skipped:

- `TLS` (skipped, seed 0, 1, 2): skipped: TLS ill-conditioned: \|V[-1]\|=0.378, sigma_min([X\|y])/sigma_min(X)=0.9992 (collinear features / noise-domina…
- `TLS_Init+FunctionalEIV` (skipped, seed 0, 1, 2): skipped: TLS ill-conditioned: \|V[-1]\|=0.378, sigma_min([X\|y])/sigma_min(X)=0.9992 (collinear features / noise-domina…

#### likelihoods (tabular_big)

Group `tabular_big`; 2 dataset(s). Columns: NLL ↓, RMSE ↓, CRPS ↓ (↓ lower is better, ↑ higher, |x|↓ closer to 0, · not ranked). Seed-spread metric: NLL.

##### likelihoods / folktables_all

| Method | Library | NLL | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `AsymmetricLaplace` | torchregress | 0.2045 ± 0.0011 | 0.8633 ± 0.0911 | 0.2649 ± 0.0019 | 442.8 |
| `CatBoost_Uncertainty` | catboost | 0.7021 ± 0.0017 | 0.7921 ± 0.0002 | 0.2921 ± 0.0002 | 21.0745 |
| `GaussianNLL` | torchregress | 0.7344 ± 0.0115 | 0.7937 ± 0.0022 | 0.2899 ± 0.0003 | 352.7 |
| `JohnsonSU` | torchregress | 0.1653 ± 0.0013 | 0.8082 ± 0.0081 | 0.2620 ± 0.0005 | 425.0 |
| `LightGBM_Point` | lightgbm | non-finite (3) | **0.7593 ± 0.0002** | - | 4.2929 |
| `MDN_K3` | torchregress | 0.1740 ± 0.0016 | 0.8125 ± 0.0062 | **0.2616 ± 0.0003** | 426.6 |
| `NSF` ‡ | torchregress | 0.3336 ± 0.0015 | 1.0319 ± 0.0010 | 0.3966 ± 0.0001 | 634.2 |
| `SinhArcsinh` | torchregress | 0.2034 ± 0.0015 | 0.8156 ± 0.0058 | 0.2635 ± 0.0005 | 468.2 |
| `SkewNormal` | torchregress | 0.4054 ± 0.0027 | 0.7949 ± 0.0013 | 0.2814 ± 0.0007 | 389.1 |
| `SkewT` | torchregress | **0.1638 ± 0.0010** | 0.8074 ± 0.0045 | 0.2619 ± 0.0002 | 3968.6 |

‡ ran on CPU after a CUDA failure (cpu retry).

##### likelihoods / tlc_fare

| Method | Library | NLL | RMSE | CRPS | train_s |
| --- | --- | ---: | ---: | ---: | ---: |
| `AsymmetricLaplace` | torchregress | 1.0241 ± 0.0041 | 5.1663 ± 1.6527 | 0.7205 ± 0.0386 | 788.5 |
| `CatBoost_Uncertainty` | catboost | 1.8425 ± 0.0060 | 3.6982 ± 0.0092 | 0.7985 ± 0.0017 | 25.3895 |
| `GaussianNLL` | torchregress | 2.0135 ± 0.0567 | 52.1273 ± 45.7261 | 1.5046 ± 0.4347 | 623.6 |
| `JohnsonSU` | torchregress | 0.8758 ± 0.0091 | 4.1761 ± 0.1764 | 0.6830 ± 0.0109 | 757.7 |
| `LightGBM_Point` | lightgbm | non-finite (3) | **3.4747 ± 0.0089** | - | 2.7576 |
| `MDN_K3` | torchregress | **0.8619 ± 0.0307** | 4.2390 ± 0.9595 | **0.6624 ± 0.0501** | 738.0 |
| `NSF` ‡ | torchregress | 14.2597 ± 0.5399 | 25.3644 ± 0.0048 | 17.9479 ± 0.0201 | 985.1 |
| `SinhArcsinh` | torchregress | 0.9801 ± 0.0380 | 1119.9 ± 1435.4 | 25498.0 ± 43233.8 | 821.6 |
| `SkewNormal` | torchregress | 1.6772 ± 0.0167 | 4.1573 ± 0.1353 | 0.8987 ± 0.0174 | 683.6 |
| `SkewT` | torchregress | 0.8870 ± 0.0380 | 4.1499 ± 0.2067 | 0.6961 ± 0.0260 | 6501.1 |

‡ ran on CPU after a CUDA failure (cpu retry).

## Seed spread

Cells marked † have a relative standard deviation (std / |mean|) above 5% on the suite's seed-spread metric: rank them with error bars only. 1847 cells flagged.

| Suite | Flagged cells |
| --- | ---: |
| `age_regression` | 45 |
| `bayesian_linear` | 5 |
| `causal` | 1 |
| `cell_counting` | 18 |
| `censored` | 2 |
| `censored_imaging` | 6 |
| `conformal` | 27 |
| `constrained_outputs` | 8 |
| `crowd_counting` | 13 |
| `density_parity` | 59 |
| `depth_estimation` | 27 |
| `distributional_bins` | 72 |
| `eiv` | 23 |
| `ensembles` | 12 |
| `fm_multitarget` | 26 |
| `fm_uq_scenarios` | 1298 |
| `image_conformal` | 6 |
| `image_density_parity` | 13 |
| `image_ensembles_uq` | 1 |
| `image_robust` | 10 |
| `keypoint_regression` | 1 |
| `likelihoods` | 21 |
| `missing_data` | 4 |
| `multivariate_intervals` | 22 |
| `ordinal` | 2 |
| `orthogonal_inference` | 13 |
| `pzflow_parity` | 3 |
| `r2c_conformal` | 3 |
| `robust_regression` | 6 |
| `sbi_parity` | 9 |
| `semi_supervised` | 12 |
| `sklearn_baselines` | 6 |
| `stress_matrix` | 4 |
| `test_time_adaptation` | 19 |
| `tweedie` | 2 |
| `uq` | 2 |
| `uq_layer` | 46 |

Largest 25:

| Suite | Dataset | Method | Metric | Rel. std |
| --- | --- | --- | --- | ---: |
| `image_density_parity` | udacity_steering | `sbi_MDN` | NLL | 10634.05% |
| `density_parity` | mt_scm20d | `sbi_MDN` | NLL | 2287.55% |
| `orthogonal_inference` | plr_ccddhnr | `tr_Orthogonal_RFF` | Bias | 1858.37% |
| `orthogonal_inference` | plr_ccddhnr | `tr_Orthogonal_RFF_legacy` | Bias | 415.34% |
| `orthogonal_inference` | plr_ccddhnr | `DoubleML_PLR_Lasso` | Bias | 318.40% |
| `orthogonal_inference` | plr_ccddhnr | `EconML_LinearDML_Lasso` | Bias | 249.57% |
| `image_density_parity` | udacity_steering | `torchregress_MDN` | NLL | 235.00% |
| `likelihoods` | insurance_claims | `MDN_K3` | NLL | 174.08% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_ResidualDist+LightGBM [heavy_tail]` | IntervalScore | 173.21% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_Mondrian+LightGBM [heavy_tail]` | IntervalScore | 173.21% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_Split+LightGBM [heavy_tail]` | IntervalScore | 173.21% |
| `density_parity` | mt_rf1 | `torchregress_MAF` | NLL | 173.17% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_ResidualDist+KumoTabular [heavy_tail]` | IntervalScore | 172.91% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_PITRecal+KumoTabular [heavy_tail]` | IntervalScore | 172.90% |
| `fm_uq_scenarios` | house_sales@heavy_tail | `tr_Mondrian+LightGBM [heavy_tail]` | IntervalScore | 172.83% |
| `fm_uq_scenarios` | house_sales@heavy_tail | `tr_ResidualDist+LightGBM [heavy_tail]` | IntervalScore | 172.83% |
| `fm_uq_scenarios` | house_sales@heavy_tail | `tr_Split+LightGBM [heavy_tail]` | IntervalScore | 172.83% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_Mondrian+TabICL [heavy_tail]` | IntervalScore | 172.83% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_Mondrian+KumoTabular [heavy_tail]` | IntervalScore | 172.83% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_PITRecal+TabICL [heavy_tail]` | IntervalScore | 172.77% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_ResidualDist+TabICL [heavy_tail]` | IntervalScore | 172.76% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_Normalized+KumoTabular [heavy_tail]` | IntervalScore | 172.69% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_Split+KumoTabular [heavy_tail]` | IntervalScore | 172.68% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_DCP+TabICL [heavy_tail]` | IntervalScore | 172.66% |
| `fm_uq_scenarios` | zurich_delays@heavy_tail | `tr_CQR+TabICL [heavy_tail]` | IntervalScore | 172.65% |

## Appendix A: failure and skip ledger

39 failed and 99 skipped rows (dataset (seeds)).

| Status | Suite | Method | Dataset (seeds) | Rows | Reason |
| --- | --- | --- | --- | ---: | --- |
| failed | `age_regression` | `CNN+RankNContrast` | ixi_brainage (0,1,2); ixi_brainage_tiny (0,1,2); synthetic_image_regression (0,1,2); udacity_steering (0,1,2); udacity_steering_tiny (0,1,2) | 15 | failed: RuntimeError: Expected all tensors to be on the same device, but got b is on cuda:0, different from other tenso… |
| failed | `image_density_parity` | `sbi_zuko_NSF` | synthetic_image_keypoint (0,1,2) | 3 | failed: RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at "/home/conda/feedstock_root/build_artifacts/libtorch_… |
| failed | `keypoint_regression` | `CNN+GaussianNLL` | synthetic_image_keypoint (0,1,2) | 3 | failed: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory… |
| failed | `keypoint_regression` | `ResNet+GaussianNLL` | synthetic_image_keypoint (0,1,2) | 3 | failed: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory… |
| failed | `label_shift` | `TT_BBSE` | folktables_income (1) | 1 | failed: ValueError: confusion matrix ill-conditioned (cond=1.24e+17) |
| failed | `label_shift` | `TT_BBSE` | folktables_income (2) | 1 | failed: ValueError: confusion matrix ill-conditioned (cond=2.39e+16) |
| failed | `label_shift` | `TT_BBSE` | folktables_income (0) | 1 | failed: ValueError: confusion matrix ill-conditioned (cond=5.13e+16) |
| failed | `pzflow_parity` | `normflows_NSF` | energy_efficiency (0,1,2); pzflow_galaxies (0,1,2); synthetic_cond_gaussian (0,1,2); synthetic_cond_nongaussian (0,1,2) | 12 | known-bad: normflows_NSF pzflow_parity #broadcast error on 1-D tabular inputs |
| skipped | `eiv` | `TLS` | synthetic_eiv_nonlinear_veryhighnoise (0,1,2) | 3 | skipped: TLS ill-conditioned: \|V[-1]\|=0.00197, sigma_min([X\|y])/sigma_min(X)=0.9999 (collinear features / noise-domi… |
| skipped | `eiv` | `TLS` | photoz_eiv (0,1,2) | 3 | skipped: TLS ill-conditioned: \|V[-1]\|=0.0113, sigma_min([X\|y])/sigma_min(X)=0.9970 (collinear features / noise-domin… |
| skipped | `eiv` | `TLS` | synthetic_eiv_nonlinear_highnoise (0,1,2) | 3 | skipped: TLS ill-conditioned: \|V[-1]\|=0.0219, sigma_min([X\|y])/sigma_min(X)=0.9921 (collinear features / noise-domin… |
| skipped | `eiv` | `TLS` | folktables_eiv (0,1,2) | 3 | skipped: TLS ill-conditioned: \|V[-1]\|=0.378, sigma_min([X\|y])/sigma_min(X)=0.9992 (collinear features / noise-domina… |
| skipped | `eiv` | `TLS_Init+FunctionalEIV` | synthetic_eiv_nonlinear_veryhighnoise (0,1,2) | 3 | skipped: TLS ill-conditioned: \|V[-1]\|=0.00197, sigma_min([X\|y])/sigma_min(X)=0.9999 (collinear features / noise-domi… |
| skipped | `eiv` | `TLS_Init+FunctionalEIV` | photoz_eiv (0,1,2) | 3 | skipped: TLS ill-conditioned: \|V[-1]\|=0.0113, sigma_min([X\|y])/sigma_min(X)=0.9970 (collinear features / noise-domin… |
| skipped | `eiv` | `TLS_Init+FunctionalEIV` | synthetic_eiv_nonlinear_highnoise (0,1,2) | 3 | skipped: TLS ill-conditioned: \|V[-1]\|=0.0219, sigma_min([X\|y])/sigma_min(X)=0.9921 (collinear features / noise-domin… |
| skipped | `eiv` | `TLS_Init+FunctionalEIV` | folktables_eiv (0,1,2) | 3 | skipped: TLS ill-conditioned: \|V[-1]\|=0.378, sigma_min([X\|y])/sigma_min(X)=0.9992 (collinear features / noise-domina… |
| skipped | `image_conformal` | `torchcp_Split+CNN` | synthetic_image_regression (0,1,2) | 3 | skipped: ModuleNotFoundError: No module named 'torchcp.regression.predictors' |
| skipped | `tweedie` | `Gamma` | insurance_claims (0,1,2) | 3 | skipped: Gamma requires strictly positive targets (support y > 0); y_train has 13323/14000 values <= 0. Not fitted: dro… |
| skipped | `tweedie` | `Gamma` | synthetic_tweedie (0,1,2) | 3 | skipped: Gamma requires strictly positive targets (support y > 0); y_train has 79/1400 values <= 0. Not fitted: droppin… |
| skipped | `tweedie` | `Gamma_Linear` | insurance_claims (0,1,2) | 3 | skipped: Gamma_Linear requires strictly positive targets (support y > 0); y_train has 13323/14000 values <= 0. Not fitt… |
| skipped | `tweedie` | `Gamma_Linear` | synthetic_tweedie (0,1,2) | 3 | skipped: Gamma_Linear requires strictly positive targets (support y > 0); y_train has 79/1400 values <= 0. Not fitted: … |
| skipped | `tweedie` | `InverseGaussian` | insurance_claims (0,1,2) | 3 | skipped: InverseGaussian requires strictly positive targets (support y > 0); y_train has 13323/14000 values <= 0. Not f… |
| skipped | `tweedie` | `InverseGaussian` | synthetic_tweedie (0,1,2) | 3 | skipped: InverseGaussian requires strictly positive targets (support y > 0); y_train has 79/1400 values <= 0. Not fitte… |
| skipped | `uq_layer` | `MAPIE_JackknifePlus+LightGBM` | tabred_cooking (0,1,2); tabred_delivery (0,1,2); tabred_weather (0,1,2) | 9 | skipped: Jackknife+ refits n=10000 models (> 500) |
| skipped | `uq_layer` | `MAPIE_JackknifePlus+LightGBM` | synthetic_hetero (0,1,2) | 3 | skipped: Jackknife+ refits n=1400 models (> 500) |
| skipped | `uq_layer` | `MAPIE_JackknifePlus+LightGBM` | house_sales (0,1,2); insurance_claims (0,1,2) | 6 | skipped: Jackknife+ refits n=14000 models (> 500) |
| skipped | `uq_layer` | `MAPIE_JackknifePlus+LightGBM` | california_housing (0,1,2) | 3 | skipped: Jackknife+ refits n=3500 models (> 500) |
| skipped | `uq_layer` | `MAPIE_JackknifePlus+LightGBM` | openml_year (0,1,2) | 3 | skipped: Jackknife+ refits n=7000 models (> 500) |
| skipped | `uq_layer` | `MAPIE_JackknifePlus+LightGBM` | zurich_delays (0,1,2) | 3 | skipped: Jackknife+ refits n=70000 models (> 500) |
| skipped | `uq_layer` | `tr_JackknifePlus+LightGBM` | tabred_cooking (0,1,2); tabred_delivery (0,1,2); tabred_weather (0,1,2) | 9 | skipped: Jackknife+ refits n=10000 models (> 500) |
| skipped | `uq_layer` | `tr_JackknifePlus+LightGBM` | synthetic_hetero (0,1,2) | 3 | skipped: Jackknife+ refits n=1400 models (> 500) |
| skipped | `uq_layer` | `tr_JackknifePlus+LightGBM` | house_sales (0,1,2); insurance_claims (0,1,2) | 6 | skipped: Jackknife+ refits n=14000 models (> 500) |
| skipped | `uq_layer` | `tr_JackknifePlus+LightGBM` | california_housing (0,1,2) | 3 | skipped: Jackknife+ refits n=3500 models (> 500) |
| skipped | `uq_layer` | `tr_JackknifePlus+LightGBM` | openml_year (0,1,2) | 3 | skipped: Jackknife+ refits n=7000 models (> 500) |
| skipped | `uq_layer` | `tr_JackknifePlus+LightGBM` | zurich_delays (0,1,2) | 3 | skipped: Jackknife+ refits n=70000 models (> 500) |

## Appendix B: method descriptions

First non-empty note per (suite, method) among ok rows, truncated.

### age_regression (methods)

| Method | Library | Description |
| --- | --- | --- |
| `CNN+Barron` | torchregress | 4-layer CNN + Barron adaptive robust loss (point prediction) |
| `CNN+BetaNLL` | torchregress | 4-layer CNN + BetaNLL(beta=0.5) anti-collapse variant (same 2-head) |
| `CNN+CRPS` | torchregress | 4-layer CNN + Gaussian CRPS (proper score, tail-robust NLL alternative) |
| `CNN+Evidential` | torchregress | 4-layer CNN backbone + NIG evidential regression |
| `CNN+FaithfulGaussian` | torchregress | 4-layer CNN + Faithful MSE/NLL-decoupled Gaussian (same 2-head) |
| `CNN+GaussianNLL` | torchregress | 4-layer CNN backbone + diagonal Gaussian NLL loss |
| `CNN+Huber` | torchregress | 4-layer CNN + PseudoHuber robust loss (point prediction) |
| `CNN+MC_Dropout` | torchregress | dropout CNN + MSE, 20 MC passes (train mode at predict) |
| `CNN+MDN_K2` | torchregress | 4-layer CNN + 2-component MDN (K-sweep minimum) |
| `CNN+MDN_K3` | torchregress | 4-layer CNN backbone + 3-component mixture density network |
| `CNN+MDN_K5` | torchregress | 4-layer CNN + 5-component MDN (K-sweep mid) |
| `CNN+MultiQuantile` | torchregress | 4-layer CNN + 3-quantile head (0.05/0.5/0.95) → median & intervals |
| `CNN+NF_Ensemble_K2` | torchregress | 2 independent NSF flows, pooled samples (K-sweep minimum) |
| `CNN+NormalizingFlow` | torchregress | 4-layer CNN context + zuko NSF normalizing flow |
| `CNN+ObservedStdNLL` | torchregress | 4-layer CNN + NoisyTarget NLL on observed rater std (APPA-REAL aleatoric gold) |
| `CNN+PoissonGaussian` | torchregress | 4-layer CNN + Poisson-Gaussian imaging-noise NLL (log-lambda head, point) |
| `CNN+SLS` | torchregress | 4-layer CNN context + SLS min-volume level-set regions (tau=0.9) |
| `CNN+StudentT` | torchregress | 4-layer CNN + Student-t NLL (heavy-tailed) |
| `ResNet+Evidential` | torchregress | ResNet18 backbone + NIG evidential regression |
| `ResNet+GaussianNLL` | torchregress | ResNet18 backbone + diagonal Gaussian NLL loss |
| `ResNet+MDN_K3` | torchregress | ResNet18 backbone + 3-component MDN |
| `ResNet+MultiQuantile` | torchregress | ResNet18 + 3-quantile head (0.05/0.5/0.95) |
| `ResNet+NormalizingFlow` | torchregress | ResNet18 context + zuko NSF normalizing flow |
| `sklearn_GBR` | scikit-learn | GradientBoostingRegressor on flattened pixels — point prediction only |
| `sklearn_RF` | scikit-learn | RandomForestRegressor on flattened pixels (100 trees) — point prediction only |
| `timm_resnet18+GaussianNLL` | timm | timm resnet18 backbone + Gaussian NLL (unpretrained) |
| `torch_uncertainty` | torch-uncertainty | torch-uncertainty RegressionRoutine (>=0.13) with distributional output — git pin in EXTERNAL_PINS.md |

### bayesian_linear (methods)

| Method | Library | Description |
| --- | --- | --- |
| `BLR+VTS` | torchregress | RBF BLR fitted on 35 of the 50 labels + VarianceTemperatureScaler on the other 15; ECE from the same Gaussian PIT as Bo… |
| `BLR_Linear` | torchregress | BLR (no RBF), hyperparams matched to sklearn BayesianRidge |
| `BayesianLinearHead` | torchregress | BLR with RBF(20) kernel, γ auto-tuned, auto_noise |
| `BoTorch_SingleTaskGP` | BoTorch | GP hyperparameter marginal-likelihood fit; capacity not matched |
| `RecursiveBayesianHead` | torchregress | RBF BLR via fit(half) + partial_fit(half) streaming |
| `sklearn_BayesianRidge` | sklearn | conjugate Bayesian linear regression (ARD priors) |

### calibration (methods)

| Method | Library | Description |
| --- | --- | --- |
| `IsotonicMeanCalibrator` | torchregress | PAVA isotonic regression on predicted means |
| `Uncalibrated` | torchregress | raw Gaussian NLL model; no calibration |
| `VTS_Floor` | torchregress | VarianceTemperatureScaler(fit_floor=True): T * var + f |
| `VTS_Floor_Clip` | torchregress | VarianceTemperatureScaler(fit_floor=True, clip=5): robust to outliers |
| `VarianceTemperatureScaler` | torchregress | scalar temperature on predictive variance (NLL optim) |
| `sklearn_IsotonicRegression` | sklearn | sklearn IsotonicRegression on predicted means |

### causal (methods)

| Method | Library | Description |
| --- | --- | --- |
| `DR` | torchregress | torchregress dr_ate/dr_cate (cross-fitted DR, RF nuisances) |
| `EconML_DRLearner` | econml | econml DRLearner (external SOTA) |
| `NaiveDiM` | sklearn | difference-in-means (confounded baseline) |
| `Policy_AllTreat` | torchregress | AIPW value of treat-everyone vs truth (baseline) |
| `Policy_DR` | torchregress | AIPW value of treat-if-CATE>0 (DR CATE) vs truth |
| `Policy_EconML` | torchregress | AIPW value of treat-if-CATE>0 (econml CATE) vs truth |
| `TLearner` | sklearn | sklearn RF T-learner (no debiasing) |

### cell_counting (methods)

| Method | Library | Description |
| --- | --- | --- |
| `CNN+EnhancedPG_fixed` | torchregress | CNN + calibrated PG (gain=1, read=2.0, shot=1.0 fixed) |
| `CNN+EnhancedPG_learned` | torchregress | CNN + PG with learned gain/shot noise |
| `CNN+GaussianNLL` | torchregress | CNN + diagonal Gaussian NLL (2-head comparison row) |
| `CNN+MSE` | torchregress | CNN + MSE on raw counts (point baseline, misspecified variance) |
| `CNN+NegBin` | torchregress | CNN log-link + NB NLL with learned dispersion (overdispersed counts) |
| `CNN+Poisson` | torchregress | CNN log-rate + Poisson deviance (var=mean) |
| `CNN+PoissonGaussian` | torchregress | CNN log-lambda + shot+readout imaging-noise NLL |
| `CNN+PoissonGaussianLR` | torchregress | CNN log-lambda + PG likelihood-ratio (diagnostic form) |
| `CNN+PoissonLR` | torchregress | CNN log-rate + Poisson likelihood-ratio (goodness-of-fit form) |
| `CNN+ZIP` | torchregress | CNN log-rate + zero-inflation logit (excess-zero counts) |
| `sklearn_RF` | scikit-learn | RandomForestRegressor on flattened pixels — point prediction only |

### censored (methods)

| Method | Library | Description |
| --- | --- | --- |
| `AFT` | torchregress | MLP + AFTLoss — log-normal accelerated failure time model |
| `CensoredGaussianNLL` | torchregress | MLP + CensoredGaussianNLLLoss — censored Gaussian NLL with right-censoring support |
| `CensoredQuantile` | torchregress | MLP + CensoredQuantileLoss (τ=0.5) — censored median regression |
| `MSE` | torchregress | MLP + WeightedMSELoss — plain baseline (ignores censoring) |
| `lifelines_LogNormalAFT` | lifelines | lifelines.LogNormalAFTFitter — log-normal AFT (parametric, external) |

### censored_imaging (methods)

| Method | Library | Description |
| --- | --- | --- |
| `AFT` | torchregress | CNN + log-normal AFT on counts |
| `CensoredGaussianNLL` | torchregress | CNN + censored Gaussian NLL (detection-limit aware) |
| `CensoredQuantile` | torchregress | CNN + censored median regression (τ=0.5) |
| `MSE` | torchregress | CNN + WeightedMSELoss — ignores censoring (baseline) |
| `lifelines_LogNormalAFT` | lifelines | lifelines LogNormalAFT on 20 PCA components of the pixels (external) |

### classification_shift (methods)

| Method | Library | Description |
| --- | --- | --- |
| `BBSE` | torchregress | estimate_target_prior_bbse plus posterior correction |
| `DomainClassifierRatio` | torchregress | DomainClassifierRatioEstimator feature density ratio |
| `EM` | torchregress | estimate_target_prior_em plus apply_label_shift_correction |
| `EstimatedWeights` | torchregress | estimate_label_shift_weights and its EM estimate |
| `Oracle` | torchregress | known target prior correction; not an unlabeled deployment method |
| `PosteriorAdapter` | torchregress | PosteriorLabelShiftAdapter estimate and transform |
| `Uncorrected` | torchregress | source posterior without label-shift correction |

### conformal (methods)

| Method | Library | Description |
| --- | --- | --- |
| `CQR+MLP` | torchregress | quantile MLP + MultiQuantileLoss + post-hoc CQR calibration (ConformalLoss(cqr) not used — it would be 2× loss scaling,… |
| `CQR+TabularMLP` | torchregress | torchregress.models TabularMLP (library defaults, MultiQuantileLoss alpha/2, 1-alpha/2) + fit_tabular + post-hoc CQR ca… |
| `MAPIE_CQR+GBR` | MAPIE | real mapie.regression.ConformalizedQuantileRegressor on sklearn GradientBoostingRegressor(loss='quantile') — MAPIE cano… |
| `MAPIE_CrossCQR+GBR` | MAPIE | real mapie.regression.CrossConformalizedQuantileRegressor (K-fold cross-conformal CQR, Romano et al. 2019) on sklearn G… |
| `MAPIE_Split+Linear` | MAPIE | real mapie.regression.SplitConformalRegressor (absolute-residual score) on sklearn LinearRegression — MAPIE canonical |
| `MAPIE_Split+MLP` | MAPIE | real mapie.regression.SplitConformalRegressor (absolute-residual score) on sklearn MLPRegressor (matched 3-layer ReLU) |
| `Split+Linear` | torchregress | single linear layer + ConformalLoss(split); matched-capacity baseline |
| `Split+Linear(post-hoc)` | torchregress | single linear layer + MSE → post-hoc residual calibration; parity with MAPIE/crepes/torchcp |
| `Split+MLP` | torchregress | point MLP + MSE → post-hoc residual calibration (ConformalLoss(split) ≡ MSE by design) |
| `Zoo_CVPlus` | torchregress | torchregress CVPlus (K=5 refits, 1-2α guarantee) — JackknifePlus is its K=n alias |
| `Zoo_Density` | torchregress | torchregress DensityConformal (density-adaptive) on MLP base |
| `Zoo_Distributional` | torchregress | torchregress DistributionalConformal (PIT-band) on MLP base |
| `Zoo_EnbPI` | torchregress | torchregress EnsembleBatchCP on bootstrap-OOB predictions |
| `Zoo_Local` | torchregress | torchregress LocalConformal (kernel-weighted, x-passed) on MLP base |
| `Zoo_LocalMAD` | torchregress | torchregress LocalConformalMAD (MAD-scaled) on Gaussian base |
| `Zoo_NexCP` | torchregress | torchregress NonExchangeableCP (recency-weighted threshold) on MLP base |
| `Zoo_Prevalence` | torchregress | torchregress PrevalenceAdjustedCP (group-adjusted) on MLP base |
| `Zoo_Split` | torchregress | torchregress SplitConformal class on MLP point base |
| `Zoo_UACQR` | torchregress | torchregress UACQR (band-normalized) on MLP base |
| `crepes_CQR+GBR` | crepes | sklearn GradientBoostingRegressor + crepes ConformalRegressor |
| `crepes_Split+Linear` | crepes | sklearn LinearRegression + crepes.ConformalRegressor |
| `torchcp_Split+Linear` | torchcp | sklearn LinearRegression + torchcp.regression.predictor.SplitPredictor (wrapped in torch nn.Module adapter for 1.x API) |

### constrained_outputs (methods)

| Method | Library | Description |
| --- | --- | --- |
| `BoundedHead` | torchregress | BoundedHead (scaled sigmoid) inside the model |
| `HGB_Poisson` | scikit-learn | HistGradientBoostingRegressor(loss='poisson'), positive by construction |
| `LightGBM_quantile` | lightgbm | one LightGBM model per quantile level (can cross) |
| `MLP` | torchregress | unconstrained MLP (MSE) |
| `MLP+clip` | torchregress | unconstrained MLP, predictions clipped to the feasible set after training |
| `MLP+renormalize` | torchregress | unconstrained MLP, negatives clipped then rows renormalised |
| `MLP_MQ` | torchregress | MultiQuantileLoss head, no ordering constraint |
| `MLP_MQ+sort` | torchregress | MultiQuantileLoss head, quantiles sorted after training |
| `NonCrossingSort` | torchregress | NonCrossingSort inside the model (sorted during training) |
| `NonNegativeHead` | torchregress | NonNegativeHead (softplus) inside the model |
| `RandomForest` | scikit-learn | RandomForestRegressor: averages of feasible targets stay feasible |
| `SimplexHead` | torchregress | SimplexHead (softmax) inside the model |
| `SpectralNormWrapper` | torchregress | SpectralNormWrapper on every linear layer |

### crowd_counting (methods)

| Method | Library | Description |
| --- | --- | --- |
| `CSRNet` | csrnet-paper | native CSRNet reimplementation (VGG-16 frontend + dilated backend, unpretrained) at 1/8 resolution; counts via map sums |
| `UNet+GaussianNLL` | torchregress | UNet 2ch density mean+logvar per pixel |
| `UNet+Huber` | torchregress | UNet density map + PseudoHuber (robust to congested outliers) |
| `UNet+MSE` | torchregress | UNet density map + MSE (point baseline) |
| `sklearn_RF` | scikit-learn | RF count regressor → uniform maps (count metrics honest; no spatial model) |

### density_parity (methods)

| Method | Library | Description |
| --- | --- | --- |
| `pzflow_NSF` | pzflow | pzflow Flow (default bijector), conditional columns = x; pzflow's own trainer |
| `sbi_MAF` | sbi | sbi posterior_nn('maf') = nflows MAF (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer |
| `sbi_MDN` | sbi | sbi posterior_nn('mdn') K=10 full covariance, shared encoder as embedding_net; shared trainer |
| `sbi_NPE_NSF` | sbi | reference: sbi NPE trainer and defaults (5 transforms, 50 hidden, own early stopping) |
| `sbi_NSF` | sbi | sbi posterior_nn('nsf') = nflows NSF (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer |
| `sbi_zuko_MAF` | sbi | sbi posterior_nn('zuko_maf') (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer |
| `sbi_zuko_NSF` | sbi | sbi posterior_nn('zuko_nsf') (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer |
| `torchregress_MAF` | torchregress | zuko MAF (3 transforms, 64 hidden) on shared encoder; shared trainer |
| `torchregress_MDN` | torchregress | MixtureDensityLoss K=10, full covariance; shared encoder + 2x64 hidden head (as sbi MDN); shared trainer |
| `torchregress_NSF` | torchregress | zuko NSF (3 transforms, 64 hidden) on shared encoder; shared trainer |

### depth_estimation (methods)

| Method | Library | Description |
| --- | --- | --- |
| `UNet+BetaNLL` | torchregress | UNet + per-pixel BetaNLL(beta=0.5) anti-collapse variant |
| `UNet+CRPS` | torchregress | UNet + per-pixel Gaussian CRPS (tail-robust proper score) |
| `UNet+DeepEnsemble_K3` | torchregress | 3 independent UNets + MSE; mean/var across members |
| `UNet+Evidential` | torchregress | UNet backbone + per-pixel NIG evidential regression |
| `UNet+FaithfulGaussian` | torchregress | UNet + per-pixel Faithful decoupled Gaussian NLL |
| `UNet+GaussianNLL` | torchregress | UNet backbone + per-pixel Gaussian NLL (mean + log-var per pixel) |
| `UNet+Huber` | torchregress | UNet + PseudoHuber robust loss per pixel (point prediction) |
| `UNet+MDN_K3` | torchregress | UNet + 3-component per-pixel mixture density network |
| `UNet+MSE` | torchregress | Lightweight UNet backbone + MSE loss (point prediction, no uncertainty) |
| `UNet+MultiQuantile` | torchregress | UNet + 3-quantile head per pixel (0.05/0.5/0.95) |
| `UNet+StudentT` | torchregress | UNet + Student-t NLL per pixel (heavy-tailed) |
| `sklearn_RF` | scikit-learn | RandomForestRegressor on flattened pixels (100 trees, 10 depth) — point prediction only |

### distributional_bins (methods)

| Method | Library | Description |
| --- | --- | --- |
| `AsymmetricLaplace` | torchregress | asymmetric-Laplace (quantile-like) MLE → location ± conformal intervals |
| `AsymmetricLaplace/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `BetaRegression` | torchregress | beta MLE on min-max-normalized targets → location ± conformal intervals |
| `BetaRegression/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `CTI` | torchregress | NSF flow density + Conformal Thresholded Intervals (level sets) |
| `CTI/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `ContrastiveFlow` | torchregress | NSF + in-batch negative contexts (contrastive shaping) |
| `ContrastiveFlow/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `Evidential` | torchregress | NIG evidential regression → Student-t predictive intervals |
| `Evidential/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `GEV` | torchregress | [μ,σ,ξ] extreme-value MLE → location ± conformal intervals |
| `GEV/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `GaussianNLL` | torchregress | diagonal Gaussian NLL → mean ± z·σ intervals |
| `GaussianNLL/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `JohnsonSU` | torchregress | [ξ,λ,γ,δ] Johnson-SU MLE → location ± conformal intervals |
| `JohnsonSU/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `MC_CP` | torchregress | NSF flow samples + MonteCarloConformal (mean center) |
| `MC_CP/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `MDN_K3` | torchregress | 3-component mixture density network → sample-based quantile intervals |
| `MDN_K3/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `NF_Ensemble_K10` | torchregress | 10 independent NormalizingFlows (seed offsets) with pooled samples (10×n_samples) for quantile intervals — code-level p… |
| `NF_Ensemble_K10/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `NF_Ensemble_K2` | torchregress | 2 independent NormalizingFlows (seed offsets) with pooled samples (2×n_samples) for quantile intervals — K-sweep minimu… |
| `NF_Ensemble_K2/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `NF_Ensemble_K20` | torchregress | 20 independent NormalizingFlows (seed offsets) with pooled samples (20×n_samples) for quantile intervals — K-sweep satu… |
| `NF_Ensemble_K20/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `NF_Ensemble_K5` | torchregress | 5 independent NormalizingFlows (seed offsets) with pooled samples (5×n_samples) for quantile intervals — K-sweep §13 ba… |
| `NF_Ensemble_K5/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `NormalizingFlow` | torchregress | zuko NSF normalizing flow → sample-based quantile intervals |
| `NormalizingFlow/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `NormalizingFlow_recipe` | torchregress | NormalizingFlow trained with the docs/losses/nflows.md recipe: 15% validation split, early stopping (patience 10, best … |
| `NormalizingFlow_recipe/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `RandomPartition` | torchregress | 3× binned heads on jittered partitions → CDF-average mean/var |
| `RandomPartition/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `SLS_CP` | torchregress | SLS frontier (d=1) + SLSConformal grid predictor |
| `SLS_CP/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `SQR` | torchregress | 32-level sorted quantiles → own 5/95% intervals (Duan et al. 2020) |
| `SQR/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `SinhArcsinh` | torchregress | sinh-arcsinh MLE → location ± conformal intervals |
| `SinhArcsinh/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `SkewNormal` | torchregress | [ξ,ω,α] skew-normal MLE → location ± conformal intervals |
| `SkewNormal/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `SkewT` | torchregress | [ξ,ω,α,ν] skew-t MLE → location ± conformal intervals |
| `SkewT/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `Split_CP` | torchregress | post-hoc split conformal — baseline, also defines bins |
| `Split_CP/dev_summary` | torchregress | Aggregate deviation across 3 bins (size-weighted) |
| `crepes_CQR` | crepes | sklearn quantile GBRs + crepes score calibration (external) |
| `crepes_CQR/dev_summary` | crepes | Aggregate deviation across 3 bins (size-weighted) |

### eiv (methods)

| Method | Library | Description |
| --- | --- | --- |
| `EnsembleEIV` | torchregress | MLP + EnsembleEIVLoss — Monte-Carlo input perturbation averaging [per-obs σ on heteroscedastic sets, per-band fallback] |
| `FunctionalEIV` | torchregress | MLP + FunctionalEIVLoss — analytical gradient-based variance propagation [per-obs σ on heteroscedastic sets, per-band f… |
| `FunctionalEIV_MC` | torchregress | MLP + FunctionalEIVLoss(monte_carlo=True) — MC sample covariance, no Jacobian [per-obs σ on heteroscedastic sets, per-b… |
| `GaussianNLL+InputNoiseMC` | torchregress | MLP(mean+logvar) + InputNoiseMC→GaussianNLL — MC perturbation of probabilistic model; test-time moment-matching [per-ob… |
| `InputNoiseBinnedPDF` | torchregress | MLP + InputNoiseBinnedPDFLoss — nonparametric binned-PDF marginalization [per-obs σ on heteroscedastic sets, per-band f… |
| `InputNoiseMC` | torchregress | MLP + InputNoiseMarginalizationLoss(MSE) — proper MC marginalisation over input noise [per-obs σ on heteroscedastic set… |
| `InputNoiseMDN` | torchregress | MDN head + InputNoiseMDNLoss — multimodal EIV marginalization [per-obs σ on heteroscedastic sets, per-band fallback] |
| `LatentMarginalization` | torchregress | MLP + LatentMarginalizationLoss — analytic Gaussian marginalization |
| `LatentNN` | torchregress | LatentNN — jointly optimises model + per-sample latent clean inputs (Gaussian penalty); test-time MC marginalization |
| `MSE` | torchregress | MLP + WeightedMSELoss — plain baseline (no EIV correction) |
| `MSE_NoisyPredict` | torchregress | MSE-trained MLP + NoisyInputPredictor — MC averaging at inference time only [per-obs σ on heteroscedastic sets, per-ban… |
| `MSE_WarmupMC` | torchregress | MSE pre-training (config.warmup_epochs, default 20) → FunctionalEIV_MC — warmup avoids early-phase MC gradient noise; d… |
| `NoiseAware` | torchregress | NoiseAwareRegressor — error-aware feature encoder (SNR, quality gates) + MLP backbone |
| `ODR` | torchregress | MLP + OrthogonalDistanceRegressionLoss — iterative latent-x optimisation [per-obs σ on heteroscedastic sets, per-band f… |
| `RC` | torchregress | RegressionCalibration — pre-processing input calibration via reliability matrix |
| `RC+FunctionalEIV` | torchregress | RC pre-processing + FunctionalEIVLoss on calibrated inputs — algorithmic chaining: does EIV loss help after RC? |
| `RC+GaussianNLL` | torchregress | RC pre-processing + GaussianNLL on calibrated inputs — pipeline approach to probabilistic EIV |
| `SIMEX` | torchregress | SIMEX — simulation-extrapolation with matched-noise prediction fix (perturbs test inputs per λ level) |
| `SciPyODR` | scipy | scipy.odr.ODR — classical orthogonal distance regression (external, linear model) |
| `SplitCP+FunctionalEIV` | torchregress+CP | FunctionalEIV + split-conformal intervals (90% q_quantile on 25% held-out calibration residuals) |
| `SplitCP+InputNoiseMC` | torchregress+CP | InputNoiseMC + split-conformal intervals (90% q_quantile on 25% held-out calibration residuals) |
| `SplitCP+MSE_WarmupMC` | torchregress+CP | MSE_WarmupMC + split-conformal intervals (90% q_quantile on 25% held-out calibration residuals) — tests whether warmup … |
| `StructuralEIV` | torchregress | MLP + StructuralEIVLoss — full cross-covariance structural model [per-obs σ on heteroscedastic sets, per-band fallback] |
| `TLS` | numpy | Noise-weighted Total Least Squares via SVD (columns scaled by sigma_x / sigma_y; Deming for one feature) — self-contain… |
| `TLS_Init+FunctionalEIV` | torchregress | Linear model initialised to TLS solution → FunctionalEIV fine-tune — ⚠️ INVESTIGATED: TLS β is NOT preserved; β drifts … |

### ensembles (methods)

| Method | Library | Description |
| --- | --- | --- |
| `BBB` | torchregress | Bayes-by-Backprop VI (ELBO) → posterior MC mean/std [contract: epistemic_only] |
| `BatchEnsemble` | torchregress | BatchEnsembleRegressor over BatchEnsembleMLPBackbone (4 members, heteroscedastic) → epi+alea intervals |
| `BinnedPDFEnsemble` | torchregress | 2× binned-PDF heads → ensemble mean/var over 10 bins |
| `DeepEnsemble_GaussianNLL` | torchregress | 5× MLP + BetaNLL → mixture mean & std [contract: epistemic_plus_aleatoric] |
| `DeepEnsemble_MSE` | torchregress | 5× MLP + MSE → mean of point predictions |
| `DeepEnsemble_MultiQuantile` | torchregress | 5× MLP + MultiQuantile → mean of quantiles |
| `HeteroBatchEnsemble` | torchregress | HeteroscedasticBatchEnsembleModel on a BatchEnsembleLinear MLP backbone (4 members) → epi+alea intervals |
| `HeteroscedasticBNN` | torchregress | BBB + variance head → epi/alea decomposition |
| `HeteroscedasticEnsemble` | torchregress | native hetero ensemble (3× Gaussian heads) → epi+alea split [contract: epistemic_plus_aleatoric] |
| `HeteroscedasticLaplace` | torchregress | last-layer Laplace on frozen MSE features (tuple head) |
| `IVON` | torchregress | natural-gradient VI optimizer + posterior MC sampling |
| `LaplaceMAP` | torchregress | MAP MLP + FullNetworkLaplace post-hoc (mean, var, intervals) |
| `MCDropoutWrapper` | torchregress | MC-Dropout wrapper over dropout MLP (20 draws) → mean/std [contract: epistemic_only] |
| `MDNEnsemble` | torchregress | 3× MDN heads (K=2) → mixture-mean point prediction |
| `MeanOnlyBatchEnsemble` | torchregress | BatchEnsembleRegressor mean-only (4 members, epistemic variance only — expect undercoverage: no aleatoric term) |
| `MultiSWAG` | torchregress | 3× heteroscedastic SWAGs → mean/epistemic/aleatoric split |
| `Snapshot` | torchregress | cyclical snapshots (3) → mean/var predictive intervals |
| `Stacking` | torchregress | torchregress StackingEnsemble: 3 MSE MLPs → Linear meta on held-out slice (trails proper-CV sklearn stacking) |
| `VIDS` | torchregress | adaptive-prior VI (bootstrap environments) → mean/std intervals |
| `VariationalMLP` | torchregress | MLP of VariationalLinear layers + Gaussian head, NLL + KL/n (Bayes-by-Backprop), 30 posterior draws → mixture mean/std |
| `sklearn_BaggingRegressor` | sklearn | Bagging(DecisionTreeRegressor, n=100) |
| `sklearn_StackingRegressor` | sklearn | Stacking(LinearRegression, RandomForest, SVR, KNeighbors) → RidgeCV |
| `sklearn_VotingRegressor` | sklearn | Voting(LinearRegression, RandomForest, SVR, KNeighbors) |

### fm_multitarget (methods)

| Method | Library | Description |
| --- | --- | --- |
| `GaussCopula+KumoTabular` | sdm | GaussCopula: 2 per-target KumoTabular fits |
| `GaussCopula+LightGBM` | lightgbm | GaussCopula: 2 per-target LightGBM fits |
| `GaussCopula+TabFM` | sdm | GaussCopula: 2 per-target TabFM fits (point-only: conformal residual marginals) |
| `GaussCopula+TabICL` | tabicl | GaussCopula: 2 per-target TabICL fits |
| `GaussCopula+TabPFN` | tabpfn | GaussCopula: 8 per-target TabPFN fits |
| `GaussCopula+TabularMLP` | torchregress | GaussCopula: 2 per-target TabularMLP fits |
| `Indep+KumoTabular` | sdm | Indep: 2 per-target KumoTabular fits |
| `Indep+LightGBM` | lightgbm | Indep: 2 per-target LightGBM fits |
| `Indep+TabFM` | sdm | Indep: 2 per-target TabFM fits (point-only: conformal residual marginals) |
| `Indep+TabICL` | tabicl | Indep: 2 per-target TabICL fits |
| `Indep+TabPFN` | tabpfn | Indep: 8 per-target TabPFN fits |
| `Indep+TabularMLP` | torchregress | Indep: 2 per-target TabularMLP fits |
| `torchregress_MDN` | torchregress | joint density (density_parity trainer) |
| `torchregress_NSF` | torchregress | joint density (density_parity trainer) |
| `tr_PITRecal+GaussCopula+KumoTabular` | torchregress | tr_PITRecal+GaussCopula: 2 per-target KumoTabular fits |
| `tr_PITRecal+GaussCopula+LightGBM` | torchregress | tr_PITRecal+GaussCopula: 2 per-target LightGBM fits |
| `tr_PITRecal+GaussCopula+TabICL` | torchregress | tr_PITRecal+GaussCopula: 2 per-target TabICL fits |
| `tr_PITRecal+GaussCopula+TabPFN` | torchregress | tr_PITRecal+GaussCopula: 8 per-target TabPFN fits |
| `tr_PITRecal+GaussCopula+TabularMLP` | torchregress | tr_PITRecal+GaussCopula: 2 per-target TabularMLP fits |

### fm_uq_scenarios (methods)

| Method | Library | Description |
| --- | --- | --- |
| `Raw+KumoTabular` | sdm | Raw on KumoTabular (clean) |
| `Raw+LightGBM` | lightgbm | Raw on LightGBM (clean) |
| `Raw+TabICL` | tabicl | Raw on TabICL (clean) |
| `Raw+TabularMLP` | torchregress | Raw on TabularMLP (clean) |
| `tr_CQR+KumoTabular` | torchregress | tr_CQR on KumoTabular (clean) |
| `tr_CQR+LightGBM` | torchregress | tr_CQR on LightGBM (clean) |
| `tr_CQR+TabICL` | torchregress | tr_CQR on TabICL (clean) |
| `tr_CQR+TabularMLP` | torchregress | tr_CQR on TabularMLP (clean) |
| `tr_DCP+KumoTabular` | torchregress | tr_DCP on KumoTabular (clean) |
| `tr_DCP+LightGBM` | torchregress | tr_DCP on LightGBM (clean) |
| `tr_DCP+TabICL` | torchregress | tr_DCP on TabICL (clean) |
| `tr_DCP+TabularMLP` | torchregress | tr_DCP on TabularMLP (clean) |
| `tr_Mondrian+KumoTabular` | torchregress | tr_Mondrian on KumoTabular (clean) |
| `tr_Mondrian+LightGBM` | torchregress | tr_Mondrian on LightGBM (clean) |
| `tr_Mondrian+TabFM` | torchregress | tr_Mondrian on TabFM (clean) |
| `tr_Mondrian+TabICL` | torchregress | tr_Mondrian on TabICL (clean) |
| `tr_Mondrian+TabularMLP` | torchregress | tr_Mondrian on TabularMLP (clean) |
| `tr_Normalized+KumoTabular` | torchregress | tr_Normalized on KumoTabular (clean) |
| `tr_Normalized+LightGBM` | torchregress | tr_Normalized on LightGBM (clean) |
| `tr_Normalized+TabICL` | torchregress | tr_Normalized on TabICL (clean) |
| `tr_Normalized+TabularMLP` | torchregress | tr_Normalized on TabularMLP (clean) |
| `tr_PITRecal+KumoTabular` | torchregress | tr_PITRecal on KumoTabular (clean) |
| `tr_PITRecal+LightGBM` | torchregress | tr_PITRecal on LightGBM (clean) |
| `tr_PITRecal+TabICL` | torchregress | tr_PITRecal on TabICL (clean) |
| `tr_PITRecal+TabularMLP` | torchregress | tr_PITRecal on TabularMLP (clean) |
| `tr_ResidualDist+KumoTabular` | torchregress | tr_ResidualDist on KumoTabular (clean) |
| `tr_ResidualDist+LightGBM` | torchregress | tr_ResidualDist on LightGBM (clean) |
| `tr_ResidualDist+TabFM` | torchregress | tr_ResidualDist on TabFM (clean) |
| `tr_ResidualDist+TabICL` | torchregress | tr_ResidualDist on TabICL (clean) |
| `tr_ResidualDist+TabularMLP` | torchregress | tr_ResidualDist on TabularMLP (clean) |
| `tr_Split+KumoTabular` | torchregress | tr_Split on KumoTabular (clean) |
| `tr_Split+LightGBM` | torchregress | tr_Split on LightGBM (clean) |
| `tr_Split+TabFM` | torchregress | tr_Split on TabFM (clean) |
| `tr_Split+TabICL` | torchregress | tr_Split on TabICL (clean) |
| `tr_Split+TabularMLP` | torchregress | tr_Split on TabularMLP (clean) |
| `tr_WeightedCQR+KumoTabular` | torchregress | tr_WeightedCQR on KumoTabular (shift) |
| `tr_WeightedCQR+LightGBM` | torchregress | tr_WeightedCQR on LightGBM (shift) |
| `tr_WeightedCQR+TabICL` | torchregress | tr_WeightedCQR on TabICL (shift) |
| `tr_WeightedCQR+TabularMLP` | torchregress | tr_WeightedCQR on TabularMLP (shift) |
| `tr_WeightedSplit+KumoTabular` | torchregress | tr_WeightedSplit on KumoTabular (shift) |
| `tr_WeightedSplit+LightGBM` | torchregress | tr_WeightedSplit on LightGBM (shift) |
| `tr_WeightedSplit+TabFM` | torchregress | tr_WeightedSplit on TabFM (shift) |
| `tr_WeightedSplit+TabICL` | torchregress | tr_WeightedSplit on TabICL (shift) |
| `tr_WeightedSplit+TabularMLP` | torchregress | tr_WeightedSplit on TabularMLP (shift) |

### foundation_model (methods)

| Method | Library | Description |
| --- | --- | --- |
| `GaussianNLL` | torchregress | 2-head MLP + BetaNLLLoss(beta=0.5) → mean & std |
| `KumoTabular` | structured-data-models | sdm KumoTabular (large, NVIDIA 2026), 999 predictive quantiles |
| `MDN` | torchregress | 3-component MixtureDensityLoss → mean & std via predict_mean_std() |
| `TabFM` | structured-data-models | sdm TabFM (Google 2026), point prediction only (single regression output); weights are non-commercial (accept_license=T… |
| `TabICL` | tabicl | TabICLRegressor (v2 checkpoint, tabicl>=2.2), 99 predictive quantiles |
| `TabPFN` | tabpfn | TabPFNRegressor, 99 predictive quantiles. TabPFN 9 weights are gated: needs HF_TOKEN with the licence accepted |

### image_calibration (methods)

| Method | Library | Description |
| --- | --- | --- |
| `IsotonicMeanCalibrator` | torchregress | Isotonic regression on predicted means (cal split) |
| `Uncalibrated` | torchregress | Frozen CNN GaussianNLL, no post-hoc calibration |
| `VarianceTemperatureScaler` | torchregress | Single temperature on predictive variance (cal split) |

### image_conformal (methods)

| Method | Library | Description |
| --- | --- | --- |
| `CQR+CNN` | torchregress | CNN 3-quantile head + CQR calibration on cal split |
| `MAPIE_Split+CNN` | MAPIE | MAPIE SplitConformalRegressor post-hoc on CNN residuals |
| `Split+CNN` | torchregress | CNN point predictor + split-conformal residual correction (α=0.1) |
| `Zoo_Density+CNN` | torchregress | torchregress DensityConformal on CNN point base |
| `Zoo_Distributional+CNN` | torchregress | torchregress DistributionalConformal (PIT-band) on CNN Gaussian base |
| `Zoo_Prevalence+CNN` | torchregress | torchregress PrevalenceAdjustedCP on CNN point base |
| `Zoo_Split+CNN` | torchregress | torchregress SplitConformal class on CNN point base |
| `Zoo_UACQR+CNN` | torchregress | torchregress UACQR on CNN quantile base |
| `crepes_Split+CNN` | crepes | crepes conformal regressor post-hoc on CNN residuals |

### image_density_parity (methods)

| Method | Library | Description |
| --- | --- | --- |
| `sbi_MAF` | sbi | sbi posterior_nn('maf') = nflows MAF (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer |
| `sbi_MDN` | sbi | sbi posterior_nn('mdn') K=10 full covariance, shared encoder as embedding_net; shared trainer |
| `sbi_NSF` | sbi | sbi posterior_nn('nsf') = nflows NSF (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer |
| `sbi_zuko_MAF` | sbi | sbi posterior_nn('zuko_maf') (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer |
| `sbi_zuko_NSF` | sbi | sbi posterior_nn('zuko_nsf') (3 transforms, 64 hidden), shared encoder as embedding_net; shared trainer |
| `torchregress_MAF` | torchregress | zuko MAF (3 transforms, 64 hidden) on shared encoder; shared trainer |
| `torchregress_MDN` | torchregress | MixtureDensityLoss K=10, full covariance; shared encoder + 2x64 hidden head (as sbi MDN); shared trainer |
| `torchregress_NSF` | torchregress | zuko NSF (3 transforms, 64 hidden) on shared encoder; shared trainer |

### image_ensembles_uq (methods)

| Method | Library | Description |
| --- | --- | --- |
| `DeepEnsemble_K5` | torchregress | 5× CNN + MSE; mean/var across members |
| `MC_Dropout` | torchregress | dropout CNN (p=0.2), 20 MC passes at predict |
| `SWAG` | torchregress | SWA snapshots (last-30%) ensemble mean/var for intervals |
| `luqbox_GaussianNLL` | lightning-uq-box | LUQ-Box DeterministicRegression+NLL on CNN features (64-d) |
| `luqbox_MCDropout` | lightning-uq-box | LUQ-Box MCDropoutRegression on CNN features (64-d, dropout_p=0.5) |

### image_robust (methods)

| Method | Library | Description |
| --- | --- | --- |
| `BalancedMSE` | torchregress | CNN + binned inverse-frequency MSE (fit on train targets) |
| `Barron` | torchregress | CNN + Barron adaptive robust loss |
| `Cauchy` | torchregress | CNN + Cauchy robust loss |
| `DensityWeighted` | torchregress | CNN + MSE with density-based reweighting |
| `FDS` | torchregress | CNN features → FDS smoothing → Linear head (train-time calibration) |
| `FocalR` | torchregress | CNN + focal-R down-weighted easy samples (mse base) |
| `Huber` | torchregress | CNN + PseudoHuber (delta=1.0) |
| `LDS` | torchregress | CNN + label-distribution-smoothing reweighting |
| `LightGBM` | lightgbm | LightGBM on flattened pixels — model-family gap vs CNN |
| `LogCosh` | torchregress | CNN + LogCosh robust loss |
| `MSE` | torchregress | CNN + MSE (non-robust baseline) |
| `XGBoost` | xgboost | XGBoost on flattened pixels — model-family gap vs CNN |

### keypoint_regression (methods)

| Method | Library | Description |
| --- | --- | --- |
| `CNN+LowRankGaussian` | torchregress | CNN + low-rank (r=1) + diag Gaussian (3D head, scalable MV) |
| `CNN+MSE` | torchregress | CNN + MSE on (N,D) targets (point baseline) |
| `CNN+MultivariateGaussian` | torchregress | CNN + full-covariance Gaussian via Cholesky params |
| `CNN+NF` | torchregress | CNN context + zuko NSF flow on D-dim targets |
| `ResNet+LowRankGaussian` | torchregress | ResNet18 + low-rank (r=1) + diag Gaussian |
| `ResNet+MultivariateGaussian` | torchregress | ResNet18 + full-covariance Gaussian |
| `sklearn_RF` | scikit-learn | RandomForest on flattened pixels, multi-output (point only) |

### label_shift (methods)

| Method | Library | Description |
| --- | --- | --- |
| `BBSE` | torchregress | BinnedLabelShiftEstimator (EM) weights from source + target preds |
| `DomainClassifier` | sklearn | sklearn domain classifier density-ratio weights (external) |
| `GaussShiftCorrect` | torchregress | Gaussian predictions corrected for label shift (EM priors) |
| `LibDomainClassifier` | torchregress | torchregress MLP discriminator density-ratio weights |
| `Oracle` | torchregress | TRUE importance weights (upper bound) |
| `ShiftFactoredTransport` | torchregress | ShiftFactoredPredictiveTransport of the Gaussian predictive |
| `ShiftFactoredTransport+Conformal` | torchregress | transport + conformal on the last 20% of train rows |
| `SplitCP` | torchregress | split conformal on source cal — undercovers shifted target |
| `TT_BBSE` | torchregress | test_time estimate_target_prior_bbse on 5 target bins (prior-ratio weights) |
| `TT_EM` | torchregress | test_time estimate_target_prior_em on 5 target bins (prior-ratio weights) |
| `Unweighted` | torchregress | no correction — biased target-risk baseline |
| `WeightedCP` | torchregress | WeightedConformalRegressionAdapter (density-ratio CP) |

### likelihoods (methods)

| Method | Library | Description |
| --- | --- | --- |
| `AsymmetricLaplace` | torchregress | ALD NLL (3-head: loc, scale, kappa) |
| `CatBoost_Uncertainty` | catboost | CatBoost RMSEWithUncertainty (mean+var), scored Gaussian |
| `GaussianNLL` | torchregress | diagonal Gaussian NLL baseline (2-head MLP) |
| `JohnsonSU` | torchregress | Johnson-SU NLL (4-head: loc, scale, gamma, delta) |
| `LightGBM_Point` | lightgbm | LightGBM L2 point baseline (no distribution) |
| `MDN_K3` | torchregress | 3-component diagonal mixture density network |
| `NSF` | torchregress | zuko NSF flow; internal z-score (spline bound) with native-unit NLL correction |
| `SinhArcsinh` | torchregress | Jones-Pewsey sinh-arcsinh NLL (4-head) |
| `SkewNormal` | torchregress | Azzalini skew-normal NLL (3-head: loc, scale, alpha) |
| `SkewT` | torchregress | Azzalini skew-t NLL (4-head: + dof) |
| `TrueModel` | scipy | scipy.stats NLL at the true parameters; TruthNLLParity = \|library - scipy\| |
| `scipy_MLE_Linear` | scipy | same linear model, scipy.stats log-density + scipy.optimize L-BFGS-B |
| `scipy_MLE_Unconditional` | scipy | scipy.stats distribution fitted by MLE to y_train (ignores x) |
| `tr_GaussianNLL_MLP` | torchregress | GaussianNLLLoss on the same MLP (misspecified baseline) |
| `tr_Linear` | torchregress | torchregress loss on a linear head, float64 L-BFGS (conditional MLE) |
| `tr_MLP` | torchregress | torchregress loss on the harness MLP (Adam) |

### missing_data (methods)

| Method | Library | Description |
| --- | --- | --- |
| `CompleteCase` | torchregress | MLP + MSE on observed targets only (selection-bias baseline) |
| `OracleIPW` | torchregress | MLP + PropensityWeightedLoss with TRUE propensities (upper bound) |
| `PropensityWeighted` | torchregress | MLP + PropensityWeightedLoss with estimated propensities |
| `sklearn_IPW` | sklearn | sklearn LogisticRegression propensity + Ridge IPW (external) |

### multivariate_intervals (methods)

| Method | Library | Description |
| --- | --- | --- |
| `GaussianNLL` | torchregress | diagonal Gaussian NLL → marginal μ ± z·σ intervals per output dim |
| `HeteroEnsembleCov` | torchregress | heteroscedastic ensemble → full epi+alea covariance → marginals + samples |
| `LowRankGaussian` | torchregress | low-rank + diag Gaussian NLL (factor samples) → scalable MV |
| `MDN` | torchregress | multivariate mixture (3 full-cov Gaussians) → sampled intervals + full scoring |
| `MVScore` | torchregress | Mahalanobis-score conformal regions (bounding-box intervals) |
| `MultiQuantile` | torchregress | marginal 10/50/90% quantiles per dim → own intervals (no joint samples) |
| `MultiTargetCP` | torchregress | per-dimension conformal thresholds on a point MLP |
| `MultivariateGaussian` | torchregress | full-covariance Gaussian NLL via Cholesky factor → marginal std from cov diagonal |
| `MultivariateGaussian+Conformal` | torchregress | MultivariateScoreConformal (Mahalanobis) ellipsoid box; model fitted on the training data minus a 25% calibration carve… |
| `MultivariateGaussian+LevelSetCP` | torchregress | LevelSetConformalPredictor on the Gaussian negative log-density (HPD ellipsoid, volume-adaptive radius), grid-searched … |
| `NF_Ensemble_K10` | torchregress | 10 independent NSF NormalizingFlows (seed offsets, [64, 64] hidden, 3 transforms) with pooled samples (10×512 samples/t… |
| `NF_Ensemble_K2` | torchregress | 2 independent NSF NormalizingFlows (seed offsets, [64, 64] hidden, 3 transforms) with pooled samples (2×512 samples/tes… |
| `NF_Ensemble_K20` | torchregress | 20 independent NSF NormalizingFlows (seed offsets, [64, 64] hidden, 3 transforms) with pooled samples (20×512 samples/t… |
| `NF_Ensemble_K5` | torchregress | 5 independent NSF NormalizingFlows (seed offsets, [64, 64] hidden, 3 transforms) with pooled samples (5×512 samples/tes… |
| `NormalizingFlow` | torchregress | zuko NSF flow → sample-based quantile intervals per output dim |
| `SLS` | torchregress | Super-Level-Set regression → boundary sampling + inverse flow → marginal intervals per output dim |
| `SLS+SLSConformal` | torchregress | SLSConformal on G/q_tau scores, SLS region box; model fitted on the training data minus a 25% calibration carve (_carve… |
| `SLS_earlystop` | torchregress | SLS (same settings as the SLS row) trained on 85% of the training rows with early stopping on a 15% validation carve (h… |
| `SLS_lowrank` | torchregress | SLS with low-rank frontier (dense-Cholesky boundary sampling) → scalable SLS |
| `SLS_lowrank+SLSConformal` | torchregress | SLSConformal on G/q_tau scores, low-rank SLS region box; model fitted on the training data minus a 25% calibration carv… |
| `TICovariance` | torchregress | Taylor-induced covariance head (ICML24) → Jacobian/Hessian-tied full cov |
| `WassersteinBound` | torchregress | supervised mean+var (pseudo-labeled cov) via Wasserstein surrogate |

### ordinal (methods)

| Method | Library | Description |
| --- | --- | --- |
| `CORAL` | torchregress | MLP + CORALHead + CORALLoss — shared-weight monotonic-bias head (b₁≥…≥b_{K-1}) per Cao et al. 2020; rank-consistent pre… |
| `CumulativeLink` | torchregress | MLP + CumulativeLinkLoss — K-1 binary classifiers on cumulative logits |
| `CumulativeLinkEnsemble` | torchregress | 3× cumulative-link MLPs — PMF-space averaging + support-moment classes |
| `DiscreteWasserstein1` | torchregress | MLP + DiscreteWasserstein1Loss — K logits, W1 to the one-hot class on the class grid (= expected class distance); PMF-m… |
| `MSE` | torchregress | MLP + WeightedMSELoss — treats ordinal classes as continuous (baseline) |
| `OrdinalCrossEntropy` | torchregress | MLP + OrdinalCrossEntropyLoss — K-class softmax cross-entropy |
| `statsmodels_OrderedModel` | statsmodels | statsmodels OrderedModel(distr='logit') — cumulative-link ordinal logistic |

### orthogonal_inference (methods)

| Method | Library | Description |
| --- | --- | --- |
| `DoubleML_PLR_Lasso` | doubleml | DoubleMLPLR, LassoCV nuisances on raw covariates |
| `DoubleML_PLR_Poly3Ridge` | doubleml | DoubleMLPLR, same cubic ridge basis; pair: tr_Orthogonal_Poly3 |
| `DoubleML_PLR_RF` | doubleml | DoubleMLPLR, random-forest nuisances (100 trees, depth 5) |
| `EconML_LinearDML_Lasso` | econml | econml LinearDML, LassoCV nuisances, constant effect |
| `tr_Naive` | torchregress | naive_linear_estimate: OLS of y on d, ignores x (biased baseline) |
| `tr_Orthogonal_NoCrossFit` | torchregress | folds=1 (no cross-fitting): diagnostic only |
| `tr_Orthogonal_Poly3` | torchregress | orthogonal_partially_linear, cubic per-covariate ridge, 5 folds |
| `tr_Orthogonal_RFF` | torchregress | orthogonal_partially_linear with inference.random_fourier_features (256 features, median bandwidth, polynomial_degree=3… |
| `tr_Orthogonal_RFF_legacy` | torchregress | pre-d67289a3 hand-rolled recipe for comparison: 256 random Fourier features, W ~ N(0, 1/K) (unit bandwidth, unstandardi… |

### ppi (methods)

| Method | Library | Description |
| --- | --- | --- |
| `Classical` | torchregress | labeled-only mean/median CIs (no model, no PPI) |
| `PPI_CalibratedMean` | torchregress | ppi_calibrated_mean_ci: affine-calibrated rectifier |
| `PPI_Mean` | torchregress | ppi_mean_ci: rectified mean + bootstrap CI |
| `PPI_OLS` | torchregress | ppi_ols_ci: prediction-powered OLS coefficients |
| `PPI_PPMean` | torchregress | ppi_pp_mean_ci: PPI++ power-tuned mean CI |
| `PPI_Quantile` | torchregress | ppi_quantile_ci (q=0.5): rectified median + CI |
| `PPI_py_Mean` | ppi_py | OFFICIAL ppi_py ppi_mean_ci (PPI++ power tuning, external) |
| `PPI_py_Quantile` | ppi_py | OFFICIAL ppi_py ppi_quantile_ci (external) |

### pzflow_parity (methods)

| Method | Library | Description |
| --- | --- | --- |
| `pzflow_NSF` | pzflow | pzflow Flow with default bijector, conditional_columns, Normal(1) latent (JAX). Architecture: default (opaque, not manu… |
| `torchregress_NSF` | torchregress | zuko Neural Spline Flow (3 transforms, 64-dim hidden) via NormalizingFlowLoss (PyTorch) |
| `torchregress_NSF_recipe` | torchregress | torchregress_NSF trained with the docs/losses/nflows.md recipe: 15% validation split, early stopping (patience 10, best… |

### r2c_conformal (methods)

| Method | Library | Description |
| --- | --- | --- |
| `CQR+MLP` | torchregress | MLP quantile head + MultiQuantileLoss → post-hoc CQR baseline. Conformalized quantile regression for comparison. |
| `R2C+CORAL` | torchregress | MLP backbone + CORALHead(shared-weight, monotonic biases) + CORALLoss → R2CConformal. Rank-consistent ordinal architect… |
| `R2C+CumulativeLink` | torchregress | MLP(49 cumulative logits) + CumulativeLinkLoss → R2CConformal. Ordinal regression backbone with sigmoid-based bin proba… |
| `R2C+Softmax` | torchregress | MLP(50-class softmax) + CE → R2CConformal. Discretises target into uniform bins, trains classifier, builds APS-style co… |
| `Split+MLP` | torchregress | MLP + MSE → post-hoc split conformal baseline. Standard approach — no classification discretisation. |
| `torchcp_R2C` | torchcp | torchcp R2CCP score + R2ccpLoss + SplitPredictor. Uses torchcp's build_regression_model and ordinal-aware binned loss p… |

### robust_regression (methods)

| Method | Library | Description |
| --- | --- | --- |
| `AdaptiveRobust` | torchregress | learns α dynamically |
| `AdaptiveRobust_Linear` | torchregress | linear model + AdaptiveRobustLoss via LBFGS — parity reference |
| `Barron` | torchregress | general Barron family α=1 |
| `Barron_Linear` | torchregress | linear model + BarronLoss via LBFGS — parity reference |
| `CVaR` | torchregress | mean over worst-10% sample losses (tail-risk) |
| `CVaR_Linear` | torchregress | linear model + CVaRLoss via LBFGS — parity reference |
| `Cauchy` | torchregress | very robust, non-convex |
| `Cauchy_Linear` | torchregress | linear model + CauchyLoss via LBFGS — parity reference |
| `Charbonnier` | torchregress | smooth L1 alternative |
| `Charbonnier_Linear` | torchregress | linear model + CharbonnierLoss via LBFGS — parity reference |
| `Expectile01` | torchregress | asymmetric L2, τ=0.1 (ALS family) |
| `Expectile01_Linear` | torchregress | linear model + Expectile01Loss via LBFGS — parity reference |
| `Expectile05` | torchregress | τ=0.5 ≡ MSE/ALS — expectile-MSE parity anchor |
| `Expectile05_Linear` | torchregress | linear model + Expectile05Loss via LBFGS — parity reference |
| `Expectile09` | torchregress | asymmetric L2, τ=0.9 (ALS family) |
| `Expectile09_Linear` | torchregress | linear model + Expectile09Loss via LBFGS — parity reference |
| `ExpectileCrossover` | torchregress | MLP 3-expectile head + crossing penalty — ordered expectiles |
| `Huber` | torchregress | balanced robustness default |
| `Huber_Linear` | torchregress | linear model + HuberLoss via LBFGS — parity reference |
| `IRLS` | torchregress | iteratively_reweighted_least_squares on _Linear (Huber weights) |
| `LightGBM` | LightGBM | gradient boosting (objective='regression_l1' for Huber-like) |
| `LogCosh` | torchregress | smooth log(cosh) approximation |
| `LogCosh_Linear` | torchregress | linear model + LogCoshLoss via LBFGS — parity reference |
| `MAE` | torchregress | L1 absolute error baseline |
| `MAE_Linear` | torchregress | linear model + MAELoss via LBFGS — parity reference |
| `MSE` | torchregress | outlier-sensitive baseline |
| `MSE_Linear` | torchregress | linear model + MSELoss via LBFGS — parity reference |
| `MultiExpectile` | torchregress | MLP 3-expectile head [0.1,0.5,0.9] — median point + outer interval |
| `PseudoHuber` | torchregress | smooth Huber approximation |
| `PseudoHuber_Linear` | torchregress | linear model + PseudoHuberLoss via LBFGS — parity reference |
| `Quantile05` | torchregress | single-quantile pinball τ=0.5 (median) |
| `Quantile05_Linear` | torchregress | linear model + Quantile05Loss via LBFGS — parity reference |
| `QuantileCrossover` | torchregress | MLP 3-quantile head + crossing penalty — ordered intervals |
| `Quantile_Linear` | torchregress | linear model + MultiQuantileLoss(q=0.5) via LBFGS — parity with sklearn QuantileRegressor |
| `Tukey` | torchregress | redescending, ignores far outliers |
| `Tukey_Linear` | torchregress | linear model + TukeyLoss via LBFGS — parity reference |
| `XGBoost` | XGBoost | gradient boosting (objective='reg:pseudohubererror') |
| `YeoJohnson` | torchregress | signed Yeo-Johnson warp + MSE (only signed transform) |
| `YeoJohnson_Linear` | torchregress | linear model + YeoJohnsonLoss via LBFGS — parity reference |
| `sklearn_QuantileRegressor` | sklearn | linear median regression via linear programming (solver=highs) |
| `statsmodels_RLM` | statsmodels | IRLS + HuberT(delta=1.345) — gold-standard ground truth |

### sbi_parity (methods)

| Method | Library | Description |
| --- | --- | --- |
| `pzflow_NSF` | pzflow | pzflow Flow, default bijector (JAX) |
| `sbi_FMPE` | sbi | sbi flow-matching posterior estimation (reference, no pair) |
| `sbi_NPE` | sbi | sbi NPE, NSF density estimator (matched capacity) |
| `sbi_NPE_MAF` | sbi | sbi NPE, MAF density estimator (matched capacity) |
| `sbi_NPE_MDN` | sbi | sbi NPE, MDN with 10 components |
| `sbi_NPSE` | sbi | sbi score-based posterior estimation, VE SDE (reference, no pair) |
| `torchregress_MAF` | torchregress | zuko MAF conditional flow; pair: sbi_NPE_MAF |
| `torchregress_MDN` | torchregress | MDNLoss, 10 diagonal components; pair: sbi_NPE_MDN |
| `torchregress_NSF` | torchregress | zuko NSF conditional flow (PyTorch), spline tail bound from recommended_tail_bound(theta_train); pair: sbi_NPE |

### semi_supervised (methods)

| Method | Library | Description |
| --- | --- | --- |
| `MLP_labeled` | scikit-learn | MLPRegressor(64, 64) on labeled rows only |
| `Oracle_FullyLabeled` | torchregress | Gaussian MLP with every training target (upper bound) |
| `Ridge_labeled` | scikit-learn | RidgeCV on labeled rows only |
| `SAGERegLoss_SelfAgreement` | torchregress | SAGERegLoss on 4 noise views of the student (no teacher) |
| `SelfTraining_RF` | scikit-learn | RandomForest self-training, low-spread pseudo-labels, 2 rounds |
| `Supervised_GaussianNLL` | torchregress | Gaussian MLP on labeled rows only (same step budget) |
| `TeacherStudent_BatchRelative` | torchregress | TeacherStudentTrainer, disagreement_to_weight(batch_relative_mode) |
| `TeacherStudent_SAGEReg` | torchregress | TeacherStudentTrainer, EMA teacher, 4 noise views, default weights |
| `TeacherStudent_UncertaintyWeight` | torchregress | TeacherStudentTrainer, uncertainty_to_weight(consensus std) |

### sklearn_baselines (methods)

| Method | Library | Description |
| --- | --- | --- |
| `GaussianNLL` | torchregress | MLP + BetaNLLLoss(beta=0.5) → mean & std |
| `GaussianNLL_Linear` | torchregress | _Linear + BetaNLLLoss(beta=0.0) via two-stage: MSE→OLS (LBFGS) + analytic residual variance; exact mean parity with skl… |
| `MSE` | torchregress | 3-layer MLP + Adam + MSE — Kaiming init (sklearn uses Glorot); converges to same optimum at 200+ epochs |
| `MultiQuantile` | torchregress | MLP + MultiQuantileLoss (q=0.1,0.5,0.9) → median & quantile intervals |
| `OLS` | torchregress | analytic linear regression (torch.linalg.lstsq) → point prediction |
| `TabularMLP` | torchregress | torchregress.models TabularMLP + fit_tabular with library defaults (robust scaling, periodic embeddings, AdamW one-cycl… |
| `TabularMLP_Ens5` | torchregress | 5 TabularMLP members (fit_tabular_ensemble, seeds seed..seed+4), MSE head, member predictions averaged; library default… |
| `sklearn_AdaBoost` | sklearn | AdaBoost on shallow decision stubs |
| `sklearn_BayesianRidge` | sklearn | Bayesian linear regression with automatic relevance determination |
| `sklearn_DecisionTree` | sklearn | single decision tree |
| `sklearn_ElasticNet` | sklearn | L1+L2-regularized linear regression |
| `sklearn_ExtraTrees` | sklearn | extremely randomized trees (100 estimators) |
| `sklearn_GradientBoosting` | sklearn | classic gradient boosting (100 estimators) |
| `sklearn_HistGradientBoosting` | sklearn | histogram-based gradient boosting (100 iterations) |
| `sklearn_HuberRegressor` | sklearn | robust linear regression with Huber loss |
| `sklearn_KNeighbors` | sklearn | k-nearest neighbors (k=5, distance-weighted) |
| `sklearn_Lasso` | sklearn | L1-regularized linear regression |
| `sklearn_LinearRegression` | sklearn | OLS linear regression |
| `sklearn_MLPRegressor` | sklearn | 3-layer MLP (ReLU) + Adam + MSE — Glorot init (torchregress uses Kaiming); converges to same optimum at 200+ epochs |
| `sklearn_RandomForest` | sklearn | bagged trees (100 estimators) |
| `sklearn_Ridge` | sklearn | L2-regularized linear regression |
| `sklearn_SVR` | sklearn | support vector regression (RBF kernel) |

### streaming_tta (methods)

| Method | Library | Description |
| --- | --- | --- |
| `COSAAdapted` | torchregress | DelayedLabelResidualAdapter, 1-chunk label delay |
| `CausalHarnessCOSA` | torchregress | CausalTTAHarness + DelayedLabelResidualAdapter, 1-chunk delay |
| `CausalHarnessCOSA_Delay2` | torchregress | CausalTTAHarness + adapter, 2-chunk label delay |
| `CausalHarnessCOSA_Unlabeled` | torchregress | CausalTTAHarness + adapter, no labels ever (delay=-1) |
| `Static` | torchregress | MLP point + train-residual std, no adaptation (ablation) |

### stress_matrix (methods)

| Method | Library | Description |
| --- | --- | --- |
| `stress_imbalanced/BinReweighted_MLP` | torchregress | long-tailed target density |
| `stress_imbalanced/DensityWeighted_MLP` | torchregress | long-tailed target density |
| `stress_imbalanced/LDS_MLP` | torchregress | long-tailed target density |
| `stress_imbalanced/MSE_MLP` | torchregress | long-tailed target density |
| `stress_noisy_inputs/InputNoise_MLP` | torchregress | EIV: noisy train features, clean test |
| `stress_noisy_inputs/MSE_MLP` | torchregress | EIV: noisy train features, clean test |
| `stress_noisy_labels/BetaNLL_MLP` | torchregress | heteroscedastic known label-noise variance (eval adds known noise to predictive variance) |
| `stress_noisy_labels/GaussianNLL_MLP` | torchregress | heteroscedastic known label-noise variance (eval adds known noise to predictive variance) |
| `stress_noisy_labels/NoisyTargetGaussianNLL_MLP` | torchregress | heteroscedastic known label-noise variance (eval adds known noise to predictive variance) |
| `stress_ood/DeepEnsemble_K10` | torchregress | covariate shift along x0 |
| `stress_ood/GaussianNLL_MLP` | torchregress | covariate shift along x0 |
| `stress_ood/MC_Dropout` | torchregress | covariate shift along x0 |
| `stress_ood/luqbox_GaussianNLL` | lightning-uq-box | covariate shift along x0 |
| `stress_ood/luqbox_MCDropout` | lightning-uq-box | covariate shift along x0 |
| `stress_semi_sup/SAGE_MLP` | torchregress | 200 labeled + 3000 unlabeled (SAGE) |
| `stress_semi_sup/SupervisedOnly_MLP` | torchregress | 200 labeled + 3000 unlabeled (SAGE) |
| `stress_sparse_labels/PseudoLabelNLL_MLP` | torchregress | 20% labeled; variance-weighted pseudo-labels |
| `stress_sparse_labels/SupervisedOnly_GaussianNLL` | torchregress | 20% labeled; variance-weighted pseudo-labels |

### test_time_adaptation (methods)

| Method | Library | Description |
| --- | --- | --- |
| `FeatureNorm` | torchregress | FeatureStatNormalizer source-to-shifted-target moment alignment |
| `JointTTA` | torchregress | weighted joint conformal TTA (domain-clf weights); pseudo-label rounds disabled |
| `JointTTA_LabelShiftEM` | torchregress | weighted joint conformal TTA (label-shift EM weights); pseudo-label rounds disabled |
| `JointTTA_OTScore` | torchregress | weighted joint conformal TTA (OT score weights); pseudo-label rounds disabled |
| `SemiConformal` | torchregress | SemiConformalCalibrator: weighted split conformal on normalized residuals \|y - mu\| / sigma of the static model, w(x) … |
| `ShiftFactoredTransport` | torchregress | unlabeled predictive prior transport + shift inflation |
| `ShiftFactoredTransport+Conformal` | torchregress | shift-factored transport + source-calibrated conformal |
| `ShiftInflator` | torchregress | RepresentationShiftInflator scales predictive std by target novelty |
| `Static` | torchregress | Gaussian MLP without test-time adaptation (shifted-target ablation) |
| `SubspaceAlign` | torchregress | WeightedSubspaceMomentAligner using source target relevance |

### tweedie (methods)

| Method | Library | Description |
| --- | --- | --- |
| `BoxCox` | torchregress | MLP + BoxCoxTransformLoss (λ=0 log-warp + MSE) |
| `CompoundPoisson` | torchregress | MLP + torchregress.CompoundPoissonLoss on log-mean |
| `CompoundPoisson_Linear` | torchregress | linear model + CompoundPoissonLoss via LBFGS — parity with statsmodels GLM |
| `LogT` | torchregress | MLP + LogTransformLoss (log warp + MSE, inverted at predict) |
| `SqrtT` | torchregress | MLP + SqrtTransformLoss (variance-stabilizing warp + MSE) |
| `Tweedie` | torchregress | MLP + torchregress.TweedieLoss on log-mean |
| `Tweedie_Linear` | torchregress | linear model + TweedieLoss via LBFGS — parity with statsmodels GLM |
| `statsmodels_GLM` | statsmodels | GLM(family=Tweedie, link=log) via IRLS — gold-standard ground truth |

### uq (methods)

| Method | Library | Description |
| --- | --- | --- |
| `DeepEnsemble_K10` | torchregress | 10× MLP + GaussianNLL → mixture mean & total variance |
| `DeepEnsemble_K5` | torchregress | 5× MLP + BetaNLL on the training data minus a 20% validation split, uncalibrated (control for the +VTS/+CQR rows) |
| `DeepEnsemble_K5+CQR` | torchregress | DeepEnsemble_K5 + torchregress CQR on the Gaussian 5%/95% quantiles, calibrated on the validation split; distribution =… |
| `DeepEnsemble_K5+VTS` | torchregress | DeepEnsemble_K5 + torchregress VarianceTemperatureScaler fitted on the validation split (std × sqrt(T)); ECE from the s… |
| `GaussianNLL` | torchregress | single MLP with BetaNLLLoss(beta=0.5) → aleatoric uncertainty |
| `MC_Dropout` | torchregress | MLP with 50% dropout, 20 stochastic passes → empirical mean & std |
| `TabularMLP_Ens5_GaussianNLL` | torchregress | 5 TabularMLP Gaussian members (fit_tabular_ensemble, library defaults) combined as a Gaussian mixture (mixture mean and… |
| `TabularMLP_Ens5_GaussianNLL+VTS` | torchregress | TabularMLP_Ens5_GaussianNLL_carve + torchregress VarianceTemperatureScaler fitted on the carve (std x sqrt(T)); train_s… |
| `TabularMLP_Ens5_GaussianNLL_carve` | torchregress | TabularMLP_Ens5_GaussianNLL fitted on the training data minus a 20% carve, uncalibrated (control for the +VTS row) |
| `TabularMLP_GaussianNLL` | torchregress | torchregress.models TabularMLP (Gaussian head) + fit_tabular, library defaults, epochs = min(100, config.epochs); 15% i… |
| `sklearn_GP` | sklearn | GaussianProcessRegressor(RBF+WhiteKernel) → posterior mean & std |
| `sklearn_QuantileGBR` | sklearn | GradientBoostingRegressor(q=0.05,0.5,0.95) → median & 90% intervals |

### uq_layer (methods)

| Method | Library | Description |
| --- | --- | --- |
| `MAPIE_CQR+DeepEnsK5` | MAPIE | mapie ConformalizedQuantileRegressor(prefit), symmetric_correction=True \| base: 5x Gaussian MLP (BetaNLL 0.5) mixture |
| `MAPIE_CQR+LightGBM` | MAPIE | mapie ConformalizedQuantileRegressor(prefit), symmetric_correction=True \| base: LGBMRegressor L2 point + quantile(alph… |
| `MAPIE_CQR+RF` | MAPIE | mapie ConformalizedQuantileRegressor(prefit), symmetric_correction=True \| base: RandomForest mean; quantiles/std over … |
| `MAPIE_CQR+TabularMLP` | MAPIE | mapie ConformalizedQuantileRegressor(prefit), symmetric_correction=True \| base: 5x torchregress.models TabularMLP Gaus… |
| `MAPIE_CQR_asym+DeepEnsK5` | MAPIE | mapie ConformalizedQuantileRegressor(prefit), default asymmetric correction \| base: 5x Gaussian MLP (BetaNLL 0.5) mixt… |
| `MAPIE_CQR_asym+LightGBM` | MAPIE | mapie ConformalizedQuantileRegressor(prefit), default asymmetric correction \| base: LGBMRegressor L2 point + quantile(… |
| `MAPIE_CQR_asym+RF` | MAPIE | mapie ConformalizedQuantileRegressor(prefit), default asymmetric correction \| base: RandomForest mean; quantiles/std o… |
| `MAPIE_CQR_asym+TabularMLP` | MAPIE | mapie ConformalizedQuantileRegressor(prefit), default asymmetric correction \| base: 5x torchregress.models TabularMLP … |
| `MAPIE_CVPlus+LightGBM` | MAPIE | mapie CrossConformalRegressor(method='plus', same KFold) on train+cal \| base: LGBMRegressor L2 point + quantile(alpha/… |
| `MAPIE_CVPlus+RF` | MAPIE | mapie CrossConformalRegressor(method='plus', same KFold) on train+cal \| base: RandomForest mean; quantiles/std over pe… |
| `MAPIE_JackknifePlus+LightGBM` | MAPIE | mapie CrossConformalRegressor(plus, LeaveOneOut) on train+cal \| base: LGBMRegressor L2 point + quantile(alpha/2, 0.5, … |
| `MAPIE_Split+DeepEnsK5` | MAPIE | mapie SplitConformalRegressor(prefit, absolute) on the cached base \| base: 5x Gaussian MLP (BetaNLL 0.5) mixture |
| `MAPIE_Split+LightGBM` | MAPIE | mapie SplitConformalRegressor(prefit, absolute) on the cached base \| base: LGBMRegressor L2 point + quantile(alpha/2, … |
| `MAPIE_Split+RF` | MAPIE | mapie SplitConformalRegressor(prefit, absolute) on the cached base \| base: RandomForest mean; quantiles/std over per-t… |
| `MAPIE_Split+TabularMLP` | MAPIE | mapie SplitConformalRegressor(prefit, absolute) on the cached base \| base: 5x torchregress.models TabularMLP Gaussian … |
| `Raw+DeepEnsK5` | torchregress | uncalibrated base interval; NLL/CRPS/ECE from the base's Gaussian view \| base: 5x Gaussian MLP (BetaNLL 0.5) mixture |
| `Raw+LightGBM` | lightgbm | uncalibrated base interval; NLL/CRPS/ECE from the base's Gaussian view \| base: LGBMRegressor L2 point + quantile(alpha… |
| `Raw+RF` | sklearn | uncalibrated base interval; NLL/CRPS/ECE from the base's Gaussian view \| base: RandomForest mean; quantiles/std over p… |
| `Raw+TabularMLP` | torchregress | uncalibrated base interval; NLL/CRPS/ECE from the base's Gaussian view \| base: 5x torchregress.models TabularMLP Gauss… |
| `crepes_Normalized+DeepEnsK5` | crepes | crepes ConformalRegressor(sigmas=spread) \| base: 5x Gaussian MLP (BetaNLL 0.5) mixture |
| `crepes_Normalized+LightGBM` | crepes | crepes ConformalRegressor(sigmas=spread) \| base: LGBMRegressor L2 point + quantile(alpha/2, 0.5, 1-alpha/2) |
| `crepes_Normalized+RF` | crepes | crepes ConformalRegressor(sigmas=spread) \| base: RandomForest mean; quantiles/std over per-tree predictions |
| `crepes_Normalized+TabularMLP` | crepes | crepes ConformalRegressor(sigmas=spread) \| base: 5x torchregress.models TabularMLP Gaussian head (library defaults) mi… |
| `crepes_Split+DeepEnsK5` | crepes | crepes ConformalRegressor on the cached base residuals \| base: 5x Gaussian MLP (BetaNLL 0.5) mixture |
| `crepes_Split+LightGBM` | crepes | crepes ConformalRegressor on the cached base residuals \| base: LGBMRegressor L2 point + quantile(alpha/2, 0.5, 1-alpha… |
| `crepes_Split+RF` | crepes | crepes ConformalRegressor on the cached base residuals \| base: RandomForest mean; quantiles/std over per-tree predicti… |
| `crepes_Split+TabularMLP` | crepes | crepes ConformalRegressor on the cached base residuals \| base: 5x torchregress.models TabularMLP Gaussian head (librar… |
| `tr_CQR+DeepEnsK5` | torchregress | torchregress CQR on the base quantiles \| base: 5x Gaussian MLP (BetaNLL 0.5) mixture |
| `tr_CQR+LightGBM` | torchregress | torchregress CQR on the base quantiles \| base: LGBMRegressor L2 point + quantile(alpha/2, 0.5, 1-alpha/2) |
| `tr_CQR+RF` | torchregress | torchregress CQR on the base quantiles \| base: RandomForest mean; quantiles/std over per-tree predictions |
| `tr_CQR+TabularMLP` | torchregress | torchregress CQR on the base quantiles \| base: 5x torchregress.models TabularMLP Gaussian head (library defaults) mixt… |
| `tr_CVPlus+LightGBM` | torchregress | torchregress CVPlus, K=5 refits on train+cal (same KFold as MAPIE) \| base: LGBMRegressor L2 point + quantile(alpha/2, … |
| `tr_CVPlus+RF` | torchregress | torchregress CVPlus, K=5 refits on train+cal (same KFold as MAPIE) \| base: RandomForest mean; quantiles/std over per-t… |
| `tr_IsoMean+VTS+DeepEnsK5` | torchregress | IsotonicMeanCalibrator on cal half A, VarianceTemperatureScaler on half B \| base: 5x Gaussian MLP (BetaNLL 0.5) mixture |
| `tr_IsoMean+VTS+LightGBM` | torchregress | IsotonicMeanCalibrator on cal half A, VarianceTemperatureScaler on half B \| base: LGBMRegressor L2 point + quantile(al… |
| `tr_IsoMean+VTS+RF` | torchregress | IsotonicMeanCalibrator on cal half A, VarianceTemperatureScaler on half B \| base: RandomForest mean; quantiles/std ove… |
| `tr_IsoMean+VTS+TabularMLP` | torchregress | IsotonicMeanCalibrator on cal half A, VarianceTemperatureScaler on half B \| base: 5x torchregress.models TabularMLP Ga… |
| `tr_JackknifePlus+LightGBM` | torchregress | torchregress JackknifePlus, LOO refits on train+cal \| base: LGBMRegressor L2 point + quantile(alpha/2, 0.5, 1-alpha/2) |
| `tr_Normalized+DeepEnsK5` | torchregress | torchregress SplitConformal(normalize_fn=spread) (residual/spread) \| base: 5x Gaussian MLP (BetaNLL 0.5) mixture |
| `tr_Normalized+LightGBM` | torchregress | torchregress SplitConformal(normalize_fn=spread) (residual/spread) \| base: LGBMRegressor L2 point + quantile(alpha/2, … |
| `tr_Normalized+RF` | torchregress | torchregress SplitConformal(normalize_fn=spread) (residual/spread) \| base: RandomForest mean; quantiles/std over per-t… |
| `tr_Normalized+TabularMLP` | torchregress | torchregress SplitConformal(normalize_fn=spread) (residual/spread) \| base: 5x torchregress.models TabularMLP Gaussian … |
| `tr_Split+DeepEnsK5` | torchregress | torchregress SplitConformal (absolute residual) \| base: 5x Gaussian MLP (BetaNLL 0.5) mixture |
| `tr_Split+LightGBM` | torchregress | torchregress SplitConformal (absolute residual) \| base: LGBMRegressor L2 point + quantile(alpha/2, 0.5, 1-alpha/2) |
| `tr_Split+RF` | torchregress | torchregress SplitConformal (absolute residual) \| base: RandomForest mean; quantiles/std over per-tree predictions |
| `tr_Split+TabularMLP` | torchregress | torchregress SplitConformal (absolute residual) \| base: 5x torchregress.models TabularMLP Gaussian head (library defau… |
| `tr_VTS+DeepEnsK5` | torchregress | torchregress VarianceTemperatureScaler fitted on the calibration split \| base: 5x Gaussian MLP (BetaNLL 0.5) mixture |
| `tr_VTS+LightGBM` | torchregress | torchregress VarianceTemperatureScaler fitted on the calibration split \| base: LGBMRegressor L2 point + quantile(alpha… |
| `tr_VTS+RF` | torchregress | torchregress VarianceTemperatureScaler fitted on the calibration split \| base: RandomForest mean; quantiles/std over p… |
| `tr_VTS+TabularMLP` | torchregress | torchregress VarianceTemperatureScaler fitted on the calibration split \| base: 5x torchregress.models TabularMLP Gauss… |

## Appendix C: methodology

- **Rows.** One row per (suite, dataset, method, seed) from the job JSONs. A row is *failed* when its `Notes` start with `failed:`, `known-bad:` or `error`, *skipped* when they start with `skipped:`, otherwise *ok*. Per-bin `distributional_bins` rows (`.../bin_k`) are counted but not tabulated. For scenario suites the scenario is appended to the method name (`Method [scenario]`).
- **Means and spread.** Cells are the mean ± sample standard deviation (n−1) over the ok seeds with a finite value; `(n=k)` marks fewer than 3 seeds and `(k non-finite)` counts excluded inf/nan values. `train_s` is the mean training time in seconds.
- **Best value.** Bold marks the best mean per column among ≥2 methods: lowest for error-type metrics, highest for Accuracy, AdaptGain, BalancedAccuracy, Concordance, Corr, EvalRowsPerSec, LibAccuracy, QWK, R2, R2_clean, RefLogProb, SBI_SBC_KSpMin, TARP_KSp, ValidPixelFrac, WeightESS, smallest |x| for ATEErr, Bias, CoverageDeviation, CoverageDeviationMean, Diag_Shift, NLLGapToTruth, PIT_Mean, PolicyErr, TARP_ATC. Coverage, widths and other target-valued columns are not ranked (see the column legend in each suite).
- **Seed spread.** † when std / |mean| > 5% on the suite's first ranked column.
- **Parity verdicts.** For each pair and key metric the relative difference |a−b| / max(|a|, |b|) of the seed means a, b (seeds where both rows are ok) is computed; for log-scale metrics (EnsembleNLL, LogLoss, NLL, RefLogProb, TrueNLL) it is |a−b| / max(|b|, 1) (nats). The verdict uses the largest value: exact ≤ 1e-06, close ≤ 2%, differs above. z = |a−b| / sqrt(s_a²/n_a + s_b²/n_b); *within noise* when z ≤ 2 for every key metric.

### Pairing rules

Method names are matched exactly (full-regex on the name without the suite prefix); comparator names are templates filled from the regex groups. No fuzzy matching.

| Suites | torchregress (regex) | Comparators | Key metrics | Kind | Rationale |
| --- | --- | --- | --- | --- | --- |
| `conformal` | `Split\+Linear\(post-hoc\)` | `MAPIE_Split+Linear`, `crepes_Split+Linear`, `torchcp_Split+Linear` | IntervalScore, Coverage, Width | deterministic | same LinearRegression base, absolute-residual split conformal |
| `conformal` | `Split\+MLP` | `MAPIE_Split+MLP` | IntervalScore, Coverage, Width | matched | split conformal on the harness MLP |
| `image_conformal` | `Split\+CNN` | `MAPIE_Split+CNN`, `crepes_Split+CNN`, `torchcp_Split+CNN` | IntervalScore, Coverage, Width | deterministic | split conformal on the same CNN predictions |
| `uq_layer` | `tr_(Split\|Normalized\|CQR\|CVPlus\|JackknifePlus)\+(.+)` | `MAPIE_{1}+{2}`, `crepes_{1}+{2}` | IntervalScore, Coverage, Width | deterministic | same conformal method on the same fitted base model (`tr_X+Base` vs `MAPIE_X+Base` / `crepes_X+Base`) |
| `r2c_conformal` | `R2C\+Softmax` | `torchcp_R2C` | IntervalScore, Coverage, Width | method | R2C conformal; torchcp trains with its own R2ccp loss |
| `sbi_parity` | `torchregress_NSF` | `sbi_NPE` | C2ST, MeanRMSE | matched | NSF posterior, matched capacity (row note `pair: sbi_NPE`) |
| `sbi_parity` | `torchregress_(MAF\|MDN)` | `sbi_NPE_{1}` | C2ST, MeanRMSE | matched | matched-capacity posterior estimator (row note `pair: sbi_NPE_X`) |
| `sbi_parity` | `torchregress_NSF` | `pzflow_NSF` | C2ST, MeanRMSE | method | NSF posterior; pzflow uses its default bijector and trainer |
| `density_parity`, `image_density_parity` | `torchregress_(NSF\|MAF)` | `sbi_{1}`, `sbi_zuko_{1}` | NLL, CRPS, Cov90_marg | matched | same flow size, shared encoder and shared trainer |
| `density_parity`, `image_density_parity` | `torchregress_MDN` | `sbi_MDN` | NLL, CRPS, Cov90_marg | matched | mixture density network, shared encoder and trainer |
| `density_parity` | `torchregress_NSF` | `pzflow_NSF`, `sbi_NPE_NSF` | NLL, CRPS, Cov90_marg | method | NSF with the comparator's own defaults and trainer |
| `pzflow_parity` | `torchregress_NSF` | `pzflow_NSF`, `normflows_NSF` | NLL, IntervalScore, Coverage | method | conditional NSF; comparator uses its default architecture and trainer |
| `orthogonal_inference` | `tr_Orthogonal_Poly3` | `DoubleML_PLR_Poly3Ridge` | RMSE, Coverage95, CIWidth | matched | partially linear DML, same cubic ridge basis (row note `pair: tr_Orthogonal_Poly3`) |
| `tweedie` | `(Tweedie\|CompoundPoisson)_Linear` | `statsmodels_GLM` | TweedieDeviance, RMSE, R2 | deterministic | linear log-link Tweedie GLM: L-BFGS vs IRLS |
| `robust_regression` | `Huber_Linear` | `statsmodels_RLM` | RMSE, MAE | deterministic | linear Huber regression vs statsmodels RLM(HuberT) |
| `robust_regression` | `IRLS` | `statsmodels_RLM` | RMSE, MAE | method | library IRLS (Huber weights, max_iter=10, float32); tuning not matched to RLM |
| `robust_regression` | `Quantile_Linear` | `sklearn_QuantileRegressor` | RMSE, MAE | deterministic | linear median regression (L-BFGS vs LP) |
| `bayesian_linear` | `BLR_Linear` | `sklearn_BayesianRidge` | RMSE, NLL, CRPS | deterministic | Bayesian linear regression, hyper-parameters matched |
| `likelihoods` | `tr_Linear` | `scipy_MLE_Linear` | NLL, LocRMSE, ScaleRMSE | deterministic | same linear conditional likelihood: torchregress loss vs scipy.stats log-density |
| `calibration` | `IsotonicMeanCalibrator` | `sklearn_IsotonicRegression` | RMSE, MAE | deterministic | PAVA isotonic regression on the same predicted means |
| `sklearn_baselines` | `OLS\|GaussianNLL_Linear` | `sklearn_LinearRegression` | RMSE, MAE | deterministic | ordinary least squares mean |
| `causal` | `DR` | `EconML_DRLearner` | ATEErr, PEHE | method | cross-fitted doubly robust learner; nuisances not identical |
| `ppi` | `PPI_(Mean\|Quantile)` | `PPI_py_{1}` | Estimate, CIWidth | method | prediction-powered inference; ppi_py mean uses PPI++ power tuning |
| `ordinal` | `CumulativeLink` | `statsmodels_OrderedModel` | MAE, QWK | method | cumulative-logit ordinal model; MLP vs linear |
| `censored`, `censored_imaging` | `AFT` | `lifelines_LogNormalAFT` | RMSE, Concordance | method | log-normal AFT; MLP vs linear |
| `label_shift` | `LibDomainClassifier` | `DomainClassifier` | WeightRMSE, RMSE | method | domain-classifier density ratio; MLP vs sklearn classifier |
| `missing_data` | `PropensityWeighted` | `sklearn_IPW` | RMSE | method | inverse propensity weighting; MLP vs Ridge |
| `image_ensembles_uq` | `MC_Dropout` | `luqbox_MCDropout` | RMSE, NLL | method | MC dropout; lightning-uq-box head on CNN features |
| `stress_matrix` | `stress_ood/MC_Dropout` | `stress_ood/luqbox_MCDropout` | RMSE, NLL | method | MC dropout under covariate shift |
| `ensembles` | `Stacking` | `sklearn_StackingRegressor` | RMSE | method | stacking; different base learners |
| `age_regression` | `ResNet\+GaussianNLL` | `timm_resnet18+GaussianNLL` | RMSE, NLL | method | ResNet + Gaussian NLL; timm backbone |
| `age_regression` | `CNN\+GaussianNLL` | `torch_uncertainty` | RMSE, NLL | method | distributional CNN; torch-uncertainty RegressionRoutine |
