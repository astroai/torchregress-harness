# 0.3.0 release run — harness status

Tracks plan Phase 1 (harness) for the torchregress 0.3.0 PyPI release.
Last updated: 2026-10-07. Algorithm development in torchregress is frozen for 0.3.0.

## Done (CPU, no network data)

| Plan step | What changed | Commit |
|---|---|---|
| 1.1 | `TORCHREGRESS_HARNESS_STRICT_DATA=1` makes `load_dataset` raise instead of returning a synthetic proxy for a real dataset | `a429faf` |
| 1.1 | `tools/check_provenance.py` gate (in `pixi run ci`) and `tools/seed_spread.py` flaky-row tool | `a429faf` |
| 1.1 | Suites turn any per-method exception into a failed row (a refused TabICL download used to drop the whole dataset) | `22fe584` |
| 1.2 | Re-pinned MAPIE `3b84b8212d`, torch-uncertainty `3f82fe5d15`, LUQ-Box v0.4.0, TabPFN 9.1.0, TabICL 2.2.0, scikit-learn 1.9.1, sbi 0.27.0; new extras `sbi`, `causal`, `distributional`, `scoring` | `22fe584` |
| 1.3 | `sbi_parity`: 9 methods (torchregress NSF/MAF/MDN paired with sbi NPE NSF/MAF/MDN, pzflow, sbi FMPE/NPSE), sbi SBC/TARP cross-check, 100 SBC simulations, frozen sbibm tasks | `f22eecb` |
| 1.4 | New `orthogonal_inference` suite (vs DoubleML, EconML) | `1c93095` |
| 1.4 | New `constrained_outputs` suite (all 5 `constraints` exports) | `6c944bc` |
| 1.4 | `calibration` suite benchmarks the VarianceTemperatureScaler floor and clip options | `e3f7110` |
| 1.5 | `--seed` for `leaderboard_regression.py`; `BENCH_PROFILE=release` in `scripts/skaha_paper_benchmark.sh` | `10cb870` |
| 1.2 | sbi from git `main` (`17ef00b8dd`); per-model estimator options for the new sbi API | this commit |
| 1.4 | KumoTabular, TabICLv2, TabFM (NVIDIA structured-data-models) in `foundation_model`; all foundation models scored on predictive quantiles | this commit |
| 1.5 | `scripts/canfar_launch_release.sh` + `skaha_release.sh` + `skaha_release_report.sh`; `tools/release_jobs.py` | `f73aa3c` and earlier |
| 1.4 | `run_all.py` and `leaderboard_regression.py` skip suite x dataset pairs a suite does not declare (`tools/_suite_registry.split_supported`); skipped pairs listed in the artifact notes / `metadata.skipped_unsupported`, no failed rows (fixes `multivariate_intervals` on univariate data) | this change |
| 1.4 | Audit follow-up: interval metrics survive the post-audit conformal `±inf` bounds. `Coverage` counts infinite intervals as covering; `Width` and `IntervalScore` average the finite intervals; new `InfiniteIntervalFrac` column (+ row note, report and leaderboard footnotes); PIT proxy and Gaussian NLL/CRPS skip non-finite rows (`NonFiniteStdFrac`); leaderboard means use finite cells only and rank methods with infinite intervals after all-finite ones | this change |
| 1.4 | New `semi_supervised` suite: `TeacherStudentTrainer` (default, batch-relative `disagreement_to_weight`, `uncertainty_to_weight`), `SAGERegLoss` self-agreement vs labeled-only Gaussian MLP, RidgeCV, MLPRegressor, random-forest self-training and a fully-labeled oracle; `ssl_friedman` (100 labels) and `diabetes` (50 labels, strict-data aware) | this change |
| 1.4 | `likelihoods`: six synthetic true-model designs (`lik_skewnormal`, `lik_skewt`, `lik_johnsonsu`, `lik_sinharcsinh`, `lik_gev`, `lik_beta`): library NLL vs `scipy.stats` at the truth, linear-head conditional MLE (torchregress L-BFGS vs `scipy.optimize`), MLP, Gaussian baseline, unconditional `scipy.stats` MLE; parameter recovery | this change |
| 1.4 | `label_shift`: `TT_BBSE` / `TT_EM` (`test_time.estimate_target_prior_bbse` / `_em` on binned Gaussian predictions, `PriorTV` vs the oracle prior) and `ShiftFactoredTransport` (+`Conformal`) on the label-shifted target | this change |
| 1.4 | `ensembles`: `HeteroBatchEnsemble` (`HeteroscedasticBatchEnsembleModel` on a `BatchEnsembleLinear` stack) and `VariationalMLP` (`VariationalLinear` layers, NLL + KL/n) | this change |
| 1.4 | `age_regression`: `CNN+RankNContrast` (`RankNContrastLoss` encoder + least-squares linear probe) | this change |
| 2b | Wired the torchregress `d67289a3` features into the suites: `TabularMLP` rows (`sklearn_baselines`, `uq`, `uq_layer` base, `conformal`), library `random_fourier_features` + `ridge="gcv"` in `orthogonal_inference` (old recipe kept as `tr_Orthogonal_RFF_legacy`), the documented NSF training recipe (`pzflow_parity`, `distributional_bins`); `models` added to the export-coverage modules | this change |
| 1.4 | `tools/export_coverage.py` → `EXPORT_COVERAGE.md` / `export_coverage.json`: 360 public exports of 10 modules, 213 covered by a suite, 147 allowlisted with a reason (`tools/export_coverage_allowlist.py`); `tests/test_export_coverage.py` fails on any export neither covered nor allowlisted, and on stale allowlist entries | this change |

## Findings so far

1. **Library bug, fixed in torchregress `63ecc3f`.** `orthogonal_partially_linear`
   used different cross-fitting splits for its two nuisances, biasing theta by
   −0.026 (≈7.7 SE) on the CCDDHNR-2018 design; DoubleML on the same basis
   −0.007. One shared split: −0.003. **Re-run on post-audit torchregress
   (`efbaf2b`, 200 replications, CPU, run 2026-10-07):**

   | Design | Method | Bias | Coverage95 | SE ratio |
   |---|---|---:|---:|---:|
   | plr_ccddhnr | tr_Orthogonal_Poly3 | +0.006 | 0.875 | 0.80 |
   | plr_ccddhnr | DoubleML_PLR_Poly3Ridge (same basis) | +0.005 | 0.900 | 0.82 |
   | plr_ccddhnr | DoubleML_PLR_Lasso | +0.007 | 0.940 | 0.97 |
   | plr_cubic | tr_Orthogonal_Poly3 | +0.002 | 0.930 | 0.96 |
   | plr_cubic | DoubleML_PLR_Poly3Ridge | +0.003 | 0.930 | 0.96 |
   | plr_cubic | DoubleML_PLR_Lasso / EconML Lasso | +0.48 | 0.02 | 0.88 |

   torchregress now matches DoubleML on the same nuisance basis on both designs;
   the under-coverage on CCDDHNR is shared by both (cubic ridge basis, SE ratio
   0.8), and on the confounded cubic design the Lasso-on-raw-covariates
   comparators fail while the cubic basis does not.
2. **Silent synthetic rows in committed artifacts.** `results/latest/harness_conformal.json`
   and `_cuda_smoke.json` hold 86 rows where TabReD (cooking, delivery, maps,
   sberbank, weather) and `insurance_claims` ran on their synthetic fallbacks.
   Neither file feeds `PARITY_REPORT.md`; the published `leaderboard_results.json`
   is clean. The release run uses strict data mode, so this cannot recur.
3. **TabPFN 9 needs a Hugging Face token** (gated `Prior-Labs/tabpfn_3_5`);
   TabICL downloads its checkpoint from the Hub too.
4. **Per-level LightGBM quantile models cross** on 73% of test rows
   (`con_quantiles`); torchregress `NonCrossingSort` has none at equal pinball loss.
5. **`VarianceTemperatureScaler(clip=5)`** gives the best CRPS on
   `synthetic_outliers` but raises NLL (1.64 → 2.71): test outliers meet a
   variance fitted without them. Document the trade-off (Phase 2.5).
6. **Resolved by torchregress `d67289a3`.** `tr_Orthogonal_RFF` (hand-rolled random
   Fourier nuisances, unit bandwidth, ridge 1e-2) covered only 59% at the time. The library now ships
   `inference.random_fourier_features` (median bandwidth, linear columns) and
   `ridge="gcv"`; the suite's `tr_Orthogonal_RFF` uses both, the old recipe stays as
   `tr_Orthogonal_RFF_legacy`. 200 replications, seed 42, CPU (2026-10-07):

   | Design | Method | Bias | RMSE | Coverage95 | SE ratio |
   |---|---|---:|---:|---:|---:|
   | plr_ccddhnr | tr_Orthogonal_RFF (new) | +0.005 | 0.048 | 0.925 | 0.93 |
   | plr_ccddhnr | tr_Orthogonal_RFF_legacy | +0.011 | 0.075 | 0.780 | 0.62 |
   | plr_ccddhnr | tr_Orthogonal_Poly3 | +0.006 | 0.057 | 0.875 | 0.80 |
   | plr_ccddhnr | DoubleML_PLR_Lasso / RF | +0.007 / -0.005 | 0.047 / 0.048 | 0.940 / 0.935 | 0.97 / 0.93 |
   | plr_cubic | tr_Orthogonal_RFF (new) | +0.015 | 0.074 | 0.895 | 0.84 |
   | plr_cubic | tr_Orthogonal_RFF_legacy | +0.065 | 0.119 | 0.810 | 0.78 |
   | plr_cubic | tr_Orthogonal_Poly3 | +0.002 | 0.046 | 0.930 | 0.96 |
   | plr_cubic | DoubleML_PLR_Poly3Ridge | +0.003 | 0.046 | 0.930 | 0.96 |

   The legacy recipe now reads 0.78 / 0.81 rather than 0.59 because the shared cross-fitting split
   (`63ecc3f`) already removed part of its bias. The new RFF is within MC error of DoubleML's Lasso on
   CCDDHNR (0.925 vs 0.940) and, unlike Lasso on raw covariates, does not fail the confounded cubic design
   (0.895 vs 0.02), but it is less efficient there than the exactly specified cubic basis (RMSE 0.074 vs 0.046).
   Cost: 0.23-0.25 s per replication (gcv SVD per fold) vs 0.002-0.004 s for Poly3.
7. **Library bug (not fixed, torchregress 065d48d).** `SkewTLoss` / `skew_t_nll`
   has an exactly zero gradient in the skewness `alpha` at `alpha = 0`
   (the Student-t CDF term clamps its incomplete-beta argument at
   `1 - 1e-15` when `w = 0`); finite differences give -0.29 on a 3-row
   example. A zero-initialised skewness head never learns skew. The
   `likelihoods` designs start skew-t fits at `alpha = 0.1`;
   `tests/test_likelihood_families.py` holds a strict xfail that flips when
   the library is fixed.
8. **Resolved by torchregress `d67289a3`** (transport ranks confident rows on the in-grid peak
   mass; the EM prior no longer collapses onto the margin bins). Originally
   `ShiftFactoredPredictiveTransport` was inert under label shift: on the `label_shift` target (and a
   Ridge predictor on diabetes) the EM prior collapsed onto the two margin support bins, which hold no
   source mass, giving `PriorTV = 1.0` and `TransportApplied = 0` for every `n_support` in 16-256.
   Re-run of `label_shift` on diabetes (full, seed 42, `/tmp/claude-0/harness_smoke/b_label_shift.json`):
   `ShiftFactoredTransport` `PriorTV = 0.438`, `TransportApplied = 1` (TT_BBSE 0.436, TT_EM 0.467).
   Other datasets and seeds are not re-run.
9. Test-time EM vs BBSE on binned Gaussian predictions: EM treats the
   Gaussian-bin probabilities as calibrated posteriors and stops at a biased
   fixed point (prior TV 0.26 vs 0.32 uncorrected on a synthetic 0.8 SD
   shift), BBSE needs only the confusion matrix (TV 0.06). BBSE raises on an
   ill-conditioned confusion, which the 2-epoch smoke model produces on
   diabetes (failed `TT_BBSE` row in smoke only; fine at 60 epochs).
10. On the synthetic family designs the linear-head torchregress fit matches
    the `scipy.optimize` MLE of the same model to < 0.01 nats for every
    family, and the library NLL equals the `scipy.stats` density at the true
    parameters to 1e-6 (smoke, seed 0). GEV needed a finite barrier for
    infeasible L-BFGS steps (a target outside the support gives an infinite
    NLL and stalls torch's strong-Wolfe line search).
11. Skipping unsupported pairs changes the default leaderboard: with the
    default `diabetes,california_housing,synthetic_hetero,house_sales`, a
    suite now runs only on the datasets it declares (e.g. `uq` no longer runs
    on `house_sales`, `multivariate_intervals` on none of them). Cell counts
    in a regenerated `leaderboard_results.json` drop accordingly;
    `metadata.skipped_unsupported` lists the pairs.

## Phase 2b: torchregress as the UQ layer (A2, A3)

New suite `uq_layer` (`suites/tabular/uq_layer.py`, 9 tabular datasets, 133 tabular release jobs).
Each base model is fitted once; torchregress, MAPIE and crepes are applied to the same cached base
predictions on the same train / calibration / test split (alpha = 0.1, 25% calibration).
Bases: LightGBM (point + quantile 0.05/0.5/0.95), RandomForest (per-tree spread), TabICL (quantiles;
failed rows when the checkpoint cannot be downloaded, as here), 5-member Gaussian MLP ensemble, and a 5-member `TabularMLP` Gaussian ensemble (library defaults, see the follow-up below).
Methods: split conformal (tr / MAPIE / crepes), normalized (tr / crepes), CQR (tr / MAPIE symmetric and
asymmetric), CV+ (tr / MAPIE, LightGBM and RF), Jackknife+ (tr / MAPIE, LightGBM, pooled n <= 500),
`VarianceTemperatureScaler`, `IsotonicMeanCalibrator`+VTS. No crepes CQR row: crepes'
`ConformalRegressor.fit` takes `np.abs(residuals)`, so signed CQR scores come out wider (see below).
Recipe rows: `uq` gains `DeepEnsemble_K5`, `+VTS`, `+CQR` (VTS / CQR fitted on a 20% validation carve of
the training data); `bayesian_linear` gains `BLR+VTS`. ECE is the harness Gaussian PIT, as for GP / BoTorch.
Tests: `tests/test_uq_layer.py`. Full run (60 epochs, seed 42, diabetes + synthetic_hetero):

| Dataset | Base | Method | Coverage | Width | IntervalScore |
|---|---|---|---:|---:|---:|
| diabetes | LightGBM | Raw | 0.647 | 107.0 | 279.9 |
| diabetes | LightGBM | tr_Split = MAPIE_Split = crepes_Split | 0.917 | 202.0 | 257.7 |
| diabetes | LightGBM | tr_CQR = MAPIE_CQR | 0.887 | 161.9 | 213.2 |
| diabetes | LightGBM | tr_CVPlus = MAPIE_CVPlus | 0.962 | 230.1 | 254.8 |
| diabetes | LightGBM | tr_JackknifePlus = MAPIE_JackknifePlus | 0.940 | 216.1 | 251.9 |
| diabetes | RF | tr_Split = MAPIE = crepes | 0.925 | 169.4 | 209.9 |
| diabetes | RF | tr_Normalized = crepes_Normalized | 0.865 | 150.5 | 202.5 |
| diabetes | RF | tr_CQR = MAPIE_CQR | 0.895 | 153.5 | 193.6 |
| diabetes | DeepEnsK5 | tr_Split = MAPIE = crepes | 0.925 | 184.8 | 225.2 |
| diabetes | DeepEnsK5 | tr_VTS | 0.902 | 175.1 | 203.3 |
| synthetic_hetero | LightGBM | tr_Split = MAPIE = crepes | 0.875 | 1.239 | 1.779 |
| synthetic_hetero | LightGBM | tr_CQR = MAPIE_CQR | 0.922 | 1.479 | 1.728 |
| synthetic_hetero | RF | tr_CQR = MAPIE_CQR | 0.913 | 1.350 | 1.659 |
| synthetic_hetero | DeepEnsK5 | tr_Split = MAPIE = crepes | 0.908 | 1.251 | 1.636 |
| synthetic_hetero | DeepEnsK5 | tr_VTS | 0.913 | 1.254 | 1.550 |

On identical base predictions torchregress matches MAPIE and crepes to 1e-6 on every shared method
(split, normalized, CQR, CV+, Jackknife+); the interval quality is set by the base model and the score,
not by the library. Jackknife+ is skipped (row note) on synthetic_hetero (n = 1400 > 500). TabICL rows fail
with `ProxyError: 403` here (no Hugging Face access).

Recipe-row ECE (full run, diabetes; lower is better):

| Row | RMSE | NLL | CRPS | ECE |
|---|---:|---:|---:|---:|
| DeepEnsemble_K5 | 53.21 | 5.3735 | 29.77 | 0.0063 |
| DeepEnsemble_K5+VTS | 53.21 | 5.3429 | 29.51 | 0.0080 |
| DeepEnsemble_K5+CQR (Gaussian view) | 53.21 | 5.3397 | 29.52 | 0.0073 (coverage 0.917) |
| sklearn_GP | 51.82 | 5.3728 | 29.33 | 0.0091 |
| BLR+VTS | 55.29 | 5.4633 | 31.74 | 0.0065 |
| BayesianLinearHead (no VTS) | 56.79 | 9.8232 | 39.29 | 0.0139 |
| BoTorch_SingleTaskGP | 69.37 | 5.6553 | 40.07 | 0.0074 |

VTS improves NLL / CRPS everywhere but ECE is already ~0.006 on the K5 ensemble, so it does not
improve (0.0063 to 0.0080 on diabetes; synthetic_hetero unchanged at 0.0021). BLR+VTS cuts NLL from 9.82 to 5.46.

Finding 12 (harness, pre-existing rows): `conformal.py` (`crepes_CQR+GBR`) and `distributional_bins.py`
(`crepes_CQR`) feed signed CQR scores to crepes, which takes absolute values, so those rows
over-widen (not changed here).

### Phase 2b follow-up: strong-default TabularMLP, library RFF, NSF recipe

Torchregress `d67289a3` ships `torchregress.models` (`TabularMLP`, `TabularPreprocessor`, `fit_tabular`,
`fit_tabular_ensemble`), `inference.random_fourier_features` and the NSF training recipe. The harness
uses the **library defaults** (hidden 256x3, periodic embeddings, AdamW one-cycle, 15% validation split,
patience 20) seeded with the harness seed (ensemble members `seed .. seed+4`); only the loss head changes,
and `epochs = min(100, config.epochs)` (60 in the release run). The harness's `lr` / `batch_size` are not
forwarded. Shared helper: `suites/_tabular_mlp.py`.

New rows: `sklearn_baselines` `TabularMLP`, `TabularMLP_Ens5`; `uq` `TabularMLP_GaussianNLL`,
`TabularMLP_Ens5_GaussianNLL`, `TabularMLP_Ens5_GaussianNLL_carve` (control: ensemble on the training data
minus a 20% carve) and `TabularMLP_Ens5_GaussianNLL+VTS` (VTS fitted on the carve); `uq_layer` base
`TabularMLP` (5-member Gaussian ensemble, mixture; 11 UQ methods, no refit methods); `conformal`
`CQR+TabularMLP` (MultiQuantile head at alpha/2, 1-alpha/2, post-hoc CQR); `pzflow_parity`
`torchregress_NSF_recipe` and `distributional_bins` `NormalizingFlow_recipe` (`suites/_flow_recipe.py`:
15% validation, patience 10, restore best, `bound=recommended_tail_bound(y_std)`, `bins=16`, up to 200
epochs); `orthogonal_inference` `tr_Orthogonal_RFF` now library RFF + `ridge="gcv"`, old recipe
`tr_Orthogonal_RFF_legacy`.

Full run (60-epoch cap, seed 42, one split, so single-seed numbers; `/tmp/claude-0/harness_full/b_*.json`).
Point RMSE, `sklearn_baselines` (LightGBM is not a row of this suite: 300 trees, lr 0.05, 15 leaves from
`uq_layer`, fitted on the same split in a scratch script):

| Model | diabetes RMSE | synthetic_hetero RMSE |
|---|---:|---:|
| TabularMLP_Ens5 | 51.97 | 0.3594 |
| TabularMLP | 52.65 | 0.3649 |
| MSE (harness MLP, Kaiming) | 54.02 | 0.3793 |
| sklearn_MLPRegressor | 53.01 | 0.3949 |
| sklearn_ExtraTrees | 51.46 | 0.3797 |
| sklearn_RandomForest | 53.58 | 0.3893 |
| sklearn_HistGradientBoosting | 56.77 | 0.3834 |
| sklearn_GradientBoosting | 58.53 | 0.3865 |
| LightGBM (scratch) | 57.83 | 0.3927 |
| sklearn_Ridge | 53.10 | 0.3850 |

TabularMLP beats the old harness MLP, sklearn MLP, HistGBR and LightGBM on both datasets; ExtraTrees
is ahead of the single TabularMLP on diabetes (n_test = 89, within noise) and the ensemble is first on
synthetic_hetero. Training cost on synthetic_hetero: TabularMLP 1.6 s, Ens5 9.3 s, MSE MLP 1.3 s.

`uq` (distribution metrics, lower is better; ECE is the harness Gaussian PIT):

| Dataset | Row | RMSE | NLL | CRPS | ECE |
|---|---|---:|---:|---:|---:|
| diabetes | TabularMLP_GaussianNLL | 51.88 | 5.3370 | 28.94 | 0.0047 |
| diabetes | TabularMLP_Ens5_GaussianNLL | 51.46 | 5.3203 | 28.60 | 0.0073 |
| diabetes | TabularMLP_Ens5_GaussianNLL_carve | 51.41 | 5.3330 | 28.70 | 0.0073 |
| diabetes | TabularMLP_Ens5_GaussianNLL+VTS | 51.41 | 5.3270 | 28.66 | 0.0071 |
| diabetes | sklearn_GP | 51.82 | 5.3728 | 29.33 | 0.0091 |
| diabetes | DeepEnsemble_K5 / +VTS | 53.21 | 5.3735 / 5.3429 | 29.77 / 29.51 | 0.0063 / 0.0080 |
| diabetes | DeepEnsemble_K10 | 51.49 | 5.3087 | 28.69 | 0.0084 |
| synthetic_hetero | TabularMLP_GaussianNLL | 0.3673 | 0.3168 | 0.1972 | 0.0024 |
| synthetic_hetero | TabularMLP_Ens5_GaussianNLL | 0.3595 | 0.2828 | 0.1926 | 0.0047 |
| synthetic_hetero | TabularMLP_Ens5_GaussianNLL_carve | 0.3676 | 0.3006 | 0.1953 | 0.0024 |
| synthetic_hetero | TabularMLP_Ens5_GaussianNLL+VTS | 0.3676 | 0.3003 | 0.1953 | 0.0024 |
| synthetic_hetero | sklearn_GP | 0.3808 | 0.4468 | 0.2108 | 0.0027 |
| synthetic_hetero | DeepEnsemble_K5 / +VTS | 0.3846 | 0.4173 / 0.4157 | 0.2091 / 0.2092 | 0.0021 / 0.0021 |
| synthetic_hetero | DeepEnsemble_K10 | 0.3682 | 0.3521 | 0.1992 | 0.0033 |

Every TabularMLP row beats `sklearn_GP` on NLL and CRPS on both datasets; ECE is within 0.003 of the GP
everywhere (diabetes better, synthetic_hetero slightly worse for the ensemble). VTS again changes little
(diabetes NLL -0.006, synthetic_hetero -0.0003): the ensembles are already calibrated. Note the 20%
carve costs the ensemble 0.012 NLL (diabetes) and 0.018 (synthetic_hetero) against using all the data.

`uq_layer` interval score (alpha = 0.1, 25% calibration; diabetes / synthetic_hetero; TabularMLP rows
trained on the same 75% as the other bases):

| Method | LightGBM | RF | DeepEnsK5 | TabularMLP |
|---|---:|---:|---:|---:|
| Raw | 279.9 / 1.924 | 207.3 / 1.706 | 216.6 / 1.543 | 214.3 / 1.394 |
| tr_Split | 257.7 / 1.779 | 209.9 / 1.754 | 225.2 / 1.636 | 217.5 / 1.605 |
| tr_CQR | 213.2 / 1.728 | 193.6 / 1.659 | 204.0 / 1.549 | 207.4 / 1.399 |
| tr_VTS | 254.6 / 1.662 | 202.2 / 1.636 | 203.3 / 1.550 | 209.1 / 1.398 |
| base RMSE | 59.48 / 0.3935 | 52.67 / 0.3934 | 54.35 / 0.3710 | 52.59 / 0.3606 |

On synthetic_hetero TabularMLP is the best base for every method (tr_CQR 1.399 vs 1.549 DeepEnsK5, 1.659
RF, 1.728 LightGBM) and its raw 90% interval already covers 0.898. On diabetes (n_cal = 77) it is
second to RF on CQR (207.4 vs 193.6) and its split / CQR coverage is 0.865 / 0.857 against the nominal
0.90 (one calibration split; torchregress, MAPIE and crepes agree on identical base predictions, see
`tests/test_uq_layer.py`). Conformal suite, `CQR+TabularMLP` vs `CQR+MLP` (full, seed 42, coverage / width /
interval score): diabetes 0.895 / 187.6 / 217.9 vs 0.895 / 179.7 / 224.2; synthetic_hetero 0.902 / 1.209 /
1.461 vs 0.915 / 1.366 / 1.644.

NSF recipe (`pzflow_parity`, full, seed 42, NLL / 90% coverage / interval score):
synthetic_cond_nongaussian: fixed-budget `torchregress_NSF` 3.996 / 0.877 / 3.343, `_recipe` 3.697 / 0.837 /
3.327, pzflow 4.930 / 0.892 / 12.60; energy_efficiency: 0.476 / 0.811 / 1.243, `_recipe` 0.437 / 0.885 /
1.302, pzflow 2.838 / 0.883 / 3.188. The recipe lowers NLL on both (-0.30, -0.04) and trains faster
(9.2 s vs 13.1 s; 4.1 s vs 7.4 s); coverage moves toward nominal on energy_efficiency and away from it on
the non-Gaussian design.

Finding 13 (library, torchregress `d67289a3`): `fit_tabular` with `scheduler="onecycle"` raises
`ZeroDivisionError` when `epochs * ceil(n_train / batch_size)` is exactly 10 (torch `OneCycleLR` with
`pct_start=0.1`: the first phase ends at step 0). The harness bumps the epoch count by one in that case
(`suites/_tabular_mlp.py::_onecycle_safe_epochs`); the library should do the same or clamp `total_steps`.

SLS interval rows (`multivariate_intervals`, 90% nominal, 60 epochs, seed 42, one split; box of the region,
joint coverage / marginal coverage / mean width; `/tmp/claude-0/harness_full/sls.json`). New rows
`SLS_earlystop` (15% validation carve of the training rows, stop on held-out region coverage vs tau,
patience 10, best epoch restored), `SLS+SLSConformal` and `SLS_lowrank+SLSConformal` (25% calibration
carve, `SLSConformal`, box of the calibrated region) plus the matched `MultivariateGaussian+Conformal`
(`MultivariateScoreConformal` ellipsoid box, same carve); tests `tests/test_mv_conformal_rows.py`. The raw
SLS rows are unchanged, so the raw threshold still under-covers held-out regions (`docs/losses/sls.md`,
"Intervals from a level set"); `NF_Ensemble_K{5,10,20}` were skipped in this run (their pooled sampling
exceeds the 8 GB sandbox; the suite itself is unchanged).

| Design | GaussianNLL | MultivariateGaussian | SLS raw | SLS_earlystop | SLS+SLSConformal |
|---|---:|---:|---:|---:|---:|
| synthetic_mv_gaussian | 0.657 / 0.852 / 3.11 | 0.685 / 0.863 / 3.15 | 0.748 / 0.898 / 3.87 | 0.952 / 0.983 / 5.27 | 0.987 / 0.996 / 8.19 |
| synthetic_mv_nongaussian | 0.783 / 0.922 / 3.03 | 0.763 / 0.916 / 3.02 | 0.847 / 0.943 / 3.99 | 0.938 / 0.979 / 5.37 | 0.990 / 0.997 / 7.72 |
| synthetic_mv_bimodal | 0.740 / 0.902 / 5.23 | 0.767 / 0.912 / 5.44 | 0.880 / 0.956 / 6.22 | 0.993 / 0.998 / 8.43 | 0.997 / 0.999 / 10.20 |

Only the conformal rows carry a guarantee, and it is on the region, not the box: the box of a 90%-region
ellipsoid has joint coverage 0.99, so these intervals are conservative (the matched
`MultivariateGaussian+Conformal` box is 5.51 / 4.99 / 8.53 wide at joint 0.975 / 0.938 / 0.988, i.e. 0.67x
/ 0.65x / 0.84x of `SLS+SLSConformal`). `SLS_earlystop` stops at the first epochs where held-out region
coverage is closest to 0.9, which gives a wide, under-trained box (joint 0.94-0.99) without a guarantee.
`PARITY_REPORT.md` SLS wording ("guarantees coverage", "1.4-1.8x GaussianNLL") is stale pending
regeneration: raw SLS is 1.2-1.3x GaussianNLL width at joint coverage 0.75-0.88, not a guarantee.

## Still to do

- [ ] **Re-run the CANFAR release run on the post-audit library.** Run
      `release_20261005_1650` used torchregress `5a334d3` (pre-audit: conformal
      bounds, ensemble eval mode, VarianceTemperatureScaler clamp, IRLS weights,
      SIMEX predict and per-output R² all changed in the 0.3.0 audit, 99
      findings). Its results must be re-run on torchregress `065d48d` or later
      before `PARITY_REPORT.md` is regenerated; do not mix the two.
- [x] Re-run `orthogonal_inference` after the torchregress fix (table under finding 1).
- [ ] Remaining coverage gaps from plan 1.4: (level-set conformal in
      `multivariate_intervals` is done: `SLS+SLSConformal` rows), metrics parity
      tests against `scoringrules` (the allowlisted metric exports in
      `tools/export_coverage_allowlist.py` point here),
      `SemiConformalCalibrator`, `DiscreteWasserstein1Loss`.
- [ ] Library follow-up for finding 7 (skew-t alpha gradient at 0). Finding 8 is resolved
      by `d67289a3`; finding 13 (OneCycleLR) is new.
- [ ] Full release run on CANFAR (below), then regenerate `PARITY_REPORT.md`
      and `EXPORT_COVERAGE.md` (`python tools/export_coverage.py`).

## Launching the release run on CANFAR

From your Mac (CANFAR is not reachable from the agent sandboxes), with both
forks pushed, `canfar auth login` done, `gh` logged in and `HF_TOKEN` exported:

```bash
cd ~/src/astroai/torchregress-harness
DRY_RUN=1 scripts/canfar_launch_release.sh launch   # check the four commands
scripts/canfar_launch_release.sh launch             # submit
scripts/canfar_launch_release.sh status             # watch
scripts/canfar_launch_release.sh report             # after all groups finish
```

`launch` submits four headless groups on `astroai/base:latest`, each running
`scripts/skaha_release.sh` on the pushed SHAs (checked against `origin/main`):

| Group | Jobs (suite x dataset) | Default replicas | Resources |
|---|---:|---:|---|
| tabular | 150 | 12 | 8 cores, 32 GB |
| sbi | 5 | 5 | 8 cores, 32 GB |
| fm (foundation models) | 4 | 2 | 8 cores, 64 GB, 1 GPU |
| image | 31 | 6 | 8 cores, 64 GB, 1 GPU |

Every job runs seeds 0, 1, 2 at 60 epochs in strict data mode. Results land in
`/arc/projects/ots/torchregress-harness/runs/<RUN_TAG>/<group>/`, one JSON per
suite, dataset and seed, plus per-replica logs and installed versions. A
replica skips jobs whose JSON already exists, so relaunching a group resumes
it. Skaha sometimes refuses a few replicas of a batch; `launch` resubmits those
shards once as single sessions, and `fill` (or `fill tabular 3,6`) retries any
that are still missing. `report` runs the provenance gate, per-group seed spread and a job ledger
into `<RUN_TAG>/report/`.

The sessions fetch the private harness fork with your `gh` token, passed as a
session environment variable (visible in your own session listing).
