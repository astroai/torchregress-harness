# 0.3.0 release run — harness status

Tracks plan Phase 1 (harness) for the torchregress 0.3.0 PyPI release.
Last updated: 2026-10-06. Algorithm development in torchregress is frozen for 0.3.0.

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
6. `tr_Orthogonal_RFF` (random Fourier nuisances, default bandwidth) covers
   only 59%: the feature map needs a bandwidth choice before it is shown as
   a recommended nuisance.
7. **Library bug (not fixed, torchregress 065d48d).** `SkewTLoss` / `skew_t_nll`
   has an exactly zero gradient in the skewness `alpha` at `alpha = 0`
   (the Student-t CDF term clamps its incomplete-beta argument at
   `1 - 1e-15` when `w = 0`); finite differences give -0.29 on a 3-row
   example. A zero-initialised skewness head never learns skew. The
   `likelihoods` designs start skew-t fits at `alpha = 0.1`;
   `tests/test_likelihood_families.py` holds a strict xfail that flips when
   the library is fixed.
8. **`ShiftFactoredPredictiveTransport` is inert under label shift.** On the
   `label_shift` target (and a Ridge predictor on diabetes) the EM prior
   collapses onto the two margin support bins, which hold no source mass
   (`source_prior` there is `eps`, so the prior ratio explodes):
   `PriorTV = 1.0` and `TransportApplied = 0` for every `n_support` in
   16-256. The rows report the predictive unchanged; worth a library look
   before transport is advertised for label shift.
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
failed rows when the checkpoint cannot be downloaded, as here), 5-member Gaussian MLP ensemble.
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

## Still to do

- [ ] **Re-run the CANFAR release run on the post-audit library.** Run
      `release_20261005_1650` used torchregress `5a334d3` (pre-audit: conformal
      bounds, ensemble eval mode, VarianceTemperatureScaler clamp, IRLS weights,
      SIMEX predict and per-output R² all changed in the 0.3.0 audit, 99
      findings). Its results must be re-run on torchregress `065d48d` or later
      before `PARITY_REPORT.md` is regenerated; do not mix the two.
- [x] Re-run `orthogonal_inference` after the torchregress fix (table under finding 1).
- [ ] Remaining coverage gaps from plan 1.4: level-set conformal
      (`LevelSetConformalPredictor`) in `multivariate_intervals`, metrics parity
      tests against `scoringrules` (the allowlisted metric exports in
      `tools/export_coverage_allowlist.py` point here),
      `SemiConformalCalibrator`, `DiscreteWasserstein1Loss`.
- [ ] Library follow-ups for findings 7 and 8 (skew-t alpha gradient at 0;
      transport EM on empty margin bins).
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
| tabular | 133 | 12 | 8 cores, 32 GB |
| sbi | 5 | 5 | 8 cores, 32 GB |
| fm (foundation models) | 4 | 2 | 8 cores, 64 GB, 1 GPU |
| image | 49 | 6 | 8 cores, 64 GB, 1 GPU |

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
