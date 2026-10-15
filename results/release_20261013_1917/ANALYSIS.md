# Release run #6 analysis (`release_20261013_1917`): tabular groups

Harness `550acb3` for the main run, fills at `5b4a0f2` (zurich MAPIE blocked
predict, tabred MultiQuantile), `1e2a0d5` (tweedie, other MultiQuantile
datasets) and `fdcb483` (multivariate_intervals). torchregress `051ac39`.
Artifacts: `/arc/projects/ots/torchregress-harness/runs/release_20261013_1917/{tabular,tabular_big}/`.
Scope: the `tabular` and `tabular_big` groups only (sbi, fm and image groups are
covered separately, except for the reruns in section 5). Status at 2026-10-14
15:10 UTC; section 5 and the corrections pointing to it updated 2026-10-15.

## 0. Completion

| Group | Expected (`.sessions/run6_jobs2.txt`) | Landed | Missing |
|---|---:|---:|---|
| tabular | 420 | 420 | - |
| tabular_big | 9 | 9 | - (`eiv / folktables_eiv` landed after the redesign, section 5) |

142 suite x dataset jobs, 2 935 method cells, 8 805 rows. Against run #5
(`release_20261008_1756`) all 122 earlier jobs are present; 20 are new
(TabReD cooking / delivery / maps / weather, `zurich_delays`, which replaces
TabReD Sberbank, and `r2c_conformal / california_housing`).

Failed rows after the fills (all others ok or deliberately skipped):

| Suite / method | Dataset | Seeds | Cause |
|---|---|---:|---|
| likelihoods / NSF | tlc_fare, folktables_all | 3 + 3 | Was the `NVML_SUCCESS == r INTERNAL ASSERT` / CUDA OOM on the GPU slice. **Resolved 2026-10-15**: CPU retry, all 6 rows ok (section 5). |
| label_shift / TT_BBSE | folktables_income | 3 | BBSE confusion matrix ill-conditioned (cond 5e16): some target bins never predicted. By design BBSE raises; EM rows are fine. |

Fixed during this check-in (all reruns have landed, section 5):

- **`sklearn_baselines / MultiQuantile`** crossed quantile heads failed the
  interval metric on 14 seed rows (6 tabred, 8 on five other datasets).
  Monotone rearrangement (`e33c385`); all 14 now produce intervals.
- **`tweedie / LogT, BoxCox, SqrtT`** failed on every dataset and seed: the
  0.3.0 library transform losses warp both arguments and reject a negative
  prediction, while the harness MLP emitted the warped value directly. The MLP
  output now goes through `loss_fn.inverse` (`1e2a0d5`).
- **`multivariate_intervals / MDN`** sampled from the precision factor as if it
  were the covariance Cholesky factor (library default changed to the
  precision parameterisation in 0.3.0): coverage 0.00 on `energy_efficiency`,
  0.98 with 2.3x the width on the synthetic designs. Fixed in `fdcb483` and
  rerun on all 5 datasets x 3 seeds: synthetic designs back to run #5
  (0.851-0.891, e.g. `synthetic_mv_gaussian` 0.861 in both runs),
  `energy_efficiency` 0.925 (run #5, one seed: 0.753).
- **MAPIE cross-conformal OOM on zurich_delays**: interval predictions in
  test-row blocks (`7b2cd8d`, `5b4a0f2`); conformal, robust_regression and
  uq_layer on `zurich_delays` all landed. `MAPIE_CVPlus+LightGBM` covers 0.908,
  `MAPIE_CrossCQR+GBR` 0.90.

Report gates (`runs/release_20261013_1917/report/`): provenance OK (15 494
rows, no synthetic fallback on a real dataset). Seed spread at 5% relative
tolerance flags 573 of 2 471 tabular rows (bayesian_linear, calibration NLL on
california and the heavy-tailed likelihood rows lead); rank those with error
bars only.

## 1. New real-data benchmarks (TabReD time split, Zurich delays)

Mean of 3 seeds; interval rows at 90% nominal. Best interval score among
methods with coverage in [0.88, 0.93]:

| Dataset | conformal | uq_layer | Best RMSE |
|---|---|---|---|
| tabred_cooking | MAPIE_CrossCQR+GBR 0.894 / IS 2.005 | MAPIE_CQR_asym+RF 0.902 / 1.964 | StackingRegressor 0.474 |
| tabred_delivery | CQR+TabularMLP 0.897 / 2.235 | MAPIE_CQR_asym+TabularMLP 0.894 / 2.251 | StackingRegressor 0.555 |
| tabred_weather | CQR+TabularMLP 0.899 / 7.421 | tr_Normalized+TabularMLP 0.909 / 7.102 | tr/MAPIE_CVPlus+LightGBM 1.662 |
| zurich_delays | CQR+TabularMLP 0.900 / 352.4 | MAPIE_CQR_asym+RF 0.902 / 357.0 | TabularMLP_Ens5 103.1 |

(`tabred_maps` runs only in sklearn_baselines and ensembles: best RMSE
StackingRegressor 0.161.)

Findings:

- **Conformal holds under the TabReD time split.** Every conformal method
  covers 0.89-1.00 on all four new datasets; the uncalibrated base models in
  `uq_layer` under-cover (Raw+DeepEnsK5 0.70-0.82 on TabReD, Raw+RF 0.75 on
  Zurich, Raw+LightGBM 0.83-0.89). `TabularMLP`'s own intervals are the only
  raw ones at or above nominal (0.90-0.93).
- **Exact parity with MAPIE and crepes.** All 18 torchregress / MAPIE / crepes
  pairs on the same base model in `uq_layer` agree to 0.00% in interval score
  on every new dataset (Split, Normalized, CQR, CV+).
- **Unconformalised quantile MLPs fail under shift.** `sklearn_baselines /
  MultiQuantile` (q 0.1-0.9, 80% nominal) covers 0.33-0.57 on TabReD and 0.77
  on Zurich, against 0.74-0.93 on the i.i.d. datasets. Use CQR on top.
- `ensembles` coverage spans 0.01-1.00 on the new datasets: several Bayesian /
  ensemble heads are badly mis-scaled at this size (VariationalMLP is the only
  one near nominal on TabReD; `HeteroscedasticEnsemble` on Zurich has IS 16 000
  vs 400 for BatchEnsemble). Not a ranking; worth a look per method.

## 2. Against run #5 (same jobs, 2 364 common cells)

144 cells move by more than 5% (RMSE / interval score / CRPS), 0.02 in
coverage or 0.1 nats. 117 of them had one seed in run #5 and three now
(`distributional_bins` per-bin coverage on california, `uq_layer`,
`ensembles`): seed noise, not a change. The 27 cells with three seeds in both
runs (MDN in `multivariate_intervals` is the bug fixed above):

- **MAPIE_CrossCQR+GBR** moved to nominal: coverage 0.97-0.99 → 0.91-0.94 and
  intervals ~30% narrower on california, diabetes, openml_year,
  synthetic_hetero. The harness code for this row is unchanged; the comparator
  moved: the run #6 prep (`2583e83`) bumped the MAPIE pin `3b84b82` → `929740a`
  and scikit-learn went 1.9.0 → 1.9.1 (`versions_replica_*.txt`). Both runs
  are pinned; the change is upstream MAPIE between those commits.
- **torchregress MDN** (precision parameterisation now default) in
  `density_parity`: NLL improves on `mt_scm20d` (3.11 → 0.33) and `mt_jura`
  (2.53 → 1.95) and gets worse on `energy_efficiency` (-1.32 → -0.76) and
  `mt_enb` (-1.18 → -0.85). sbi's MDN still leads on all four (-0.01, 1.92,
  -1.33, -1.24). Library action item stays open.
- `eiv / LatentNN` RMSE improves on photoz_eiv (0.48 → 0.20) and synthetic_eiv
  (0.34 → 0.25), worse on synthetic_eiv_nonlinear (1.02 → 1.32).
- `likelihoods / tlc_fare`: SinhArcsinh and GaussianNLL means are dominated by
  single exploding seeds in both runs (SinhArcsinh RMSE 5 / 615 / 2 740 by
  seed, GaussianNLL 209 424 on one run #5 seed). Heavy-tail instability;
  report medians or drop these two from the table.
- Small moves: house_sales OLS RMSE -6%, synthetic_censored MultiQuantile
  coverage 0.78 → 0.75 (rearrangement), tlc_fare SkewT / AsymmetricLaplace
  within ±12%.

Status changes: 11 rows failed in run #5 and pass now (conformal Zoo_Local /
LocalMAD / Prevalence, robust_regression XGBoost, distributional_bins
RandomPartition, r2c_conformal R2C+CORAL). One regressed: likelihoods NSF on
tlc_fare (GPU assert above;
ok again after the CPU retry, section 5).

## 3. Open harness / library items from this run

1. **Tweedie-family MLP rows do not train.** *(2026-10-15, section 5: they
   now train on synthetic_tweedie but not on insurance_claims; Gamma /
   InverseGaussian are skipped on zero-valued targets.)* `tweedie / Tweedie, Gamma,
   CompoundPoisson, InverseGaussian` (MLP + Adam, exp output) give R² -0.05 to
   -0.69 and deviances 46-14 000, in run #5 too, while the linear L-BFGS
   versions match statsmodels' GLM exactly (R² 0.549 / deviance 0.717 on
   synthetic_tweedie). The new warped rows (SqrtT R² 0.50, LogT 0.41) train
   fine on the same backbone, so the problem is the exp-output loss path.
2. ~~GPU NVML assert in NSF on the 1.5M-row likelihood datasets.~~ Handled by
   the CPU retry (section 5); the assert remains on one image suite.
3. MDN precision parameterisation: mixed in `density_parity` (above).

## 4. eiv / folktables_eiv did not fit the 6 h job budget (resolved, section 5)

*Superseded*: the redesigned job (noise on AGEP / WKHP only, 250k rows) fits
with ~3.4 h of training per seed; see section 5. Original diagnosis below.

Seed 0 on harness `550acb3` hit the 6 h per-job timeout with no output; the
session was stopped at 14:00 UTC. No single method hangs. Profile of all 25
methods on a folktables_eiv-shaped synthetic set at 240k training rows (above
the 100k big-data threshold, so batch 1024 as in the real run), 2 CPU cores:
~30 min with 10 features. The real design has ~71 features (2 numeric, 2
integer-coded high-cardinality categoricals, ~67 one-hot columns), and the MC
input-noise methods scale with the feature count (InputNoiseMC 100 s → 350 s
at 71 features). Scaled to 1.2M rows that is several hours per seed, at or
over the budget. The like-for-like `likelihoods / folktables_all` job (same
rows, 10 methods) takes ~2.5 h per seed, so data loading is not the problem.

The design itself is questionable: Gaussian measurement noise is injected
into every standardised column, including the one-hot indicators and the
integer category codes for occupation and place of birth, which have no
meaningful measurement error. Options: inject noise only into the continuous
columns (AGEP, WKHP), drop or embed the categoricals, and subsample the
training rows for the EIV suite.

## 5. Reruns after the follow-up fixes (2026-10-15)

Final run #6 artifacts, all groups (`tabular` 420, `tabular_big` 9, `sbi` 15,
`fm` 42, `image` 93 job JSONs; 15 616 rows). Means over 3 seeds.

**eiv / folktables_eiv (redesigned: noise on AGEP / WKHP only, 250k rows).**
All 3 seeds landed, 25 methods x 3 = 75 rows. Training time (sum of `train_s`)
3.42 / 3.41 / 3.42 h for seeds 0 / 1 / 2, inside the 6 h budget. TLS and
TLS_Init+FunctionalEIV are skipped on every seed (TLS ill-conditioned,
sigma ratio 0.9992). RMSE (USD):

| Method | RMSE | R² |
|---|---:|---:|
| InputNoiseMDN | 58 192 | 0.361 |
| InputNoiseMC | 58 503 | 0.354 |
| MSE_NoisyPredict | 58 624 | 0.352 |
| EnsembleEIV | 58 681 | 0.350 |
| MSE (baseline) | 58 697 | 0.350 |

The best EIV method gains 0.9% over MSE: with noise only on two of ~71
columns the correction has little to do. LatentMarginalization (90 835),
InputNoiseBinnedPDF (113 409) and ODR (114 820) are far worse than MSE.

**sbi_parity / torchregress_NSF on gaussian_mixture.** C2ST 0.662 / 0.593 /
0.616 (mean 0.624; was 0.997), against sbi_NPE 0.502 / 0.534 / 0.503 (mean
0.513). Fixed, but still behind sbi's NPE. No earlier per-task sbi numbers are
in this file to compare the other four tasks against; current torchregress_NSF
means: bernoulli_glm 0.749, gaussian_linear 0.628, slcp 0.805, twomoons 0.588
(sbi_NPE 0.675, 0.533, 0.851, 0.498).

**tweedie.** On `synthetic_tweedie` the MLP rows now train: Tweedie and
CompoundPoisson (identical rows) R² 0.502, deviance 0.745, against
statsmodels GLM R² 0.549 / 0.717 (Tweedie_Linear matches the GLM exactly). On
`insurance_claims` they still fail: R² -20.8 (seeds -29.3 / -32.7 / -0.52),
deviance 187 against GLM 0.007 / 1.413. Item 1 in section 3 stays open for
the zero-inflated real data. Gamma, Gamma_Linear and InverseGaussian are now
skipped on both datasets (targets <= 0: 13 323 / 14 000 on insurance_claims,
79 / 1 400 on synthetic_tweedie) instead of being fitted.

**test_time_adaptation (corrected; earlier run #6 numbers were scored against
the wrong targets).** Coverage at 90% nominal / RMSE:

| Method | california_housing | diabetes | folktables_income |
|---|---|---|---|
| Static | 0.910 / 0.604 | 0.769 / 53.6 | 0.834 / 76 384 |
| FeatureNorm | 0.888 / 0.658 | 0.779 / 57.5 | 0.834 / 75 945 |
| SubspaceAlign | 0.899 / 0.621 | 0.784 / 57.6 | 0.830 / 76 705 |
| ShiftInflator | 0.984 / 0.604 | 0.995 / 53.6 | 0.959 / 76 384 |
| JointTTA | 0.900 / 0.624 | 0.885 / 54.4 | 0.880 / 76 396 |
| JointTTA_LabelShiftEM | 0.900 / 0.624 | 0.907 / 54.4 | 0.881 / 76 396 |
| JointTTA_OTScore | 0.902 / 0.624 | 0.947 / 54.4 | 0.880 / 76 396 |
| SemiConformal (new) | 0.901 / 0.624 | 0.945 / 54.4 | 0.880 / 76 396 |
| ShiftFactoredTransport | 0.951 / 0.604 | 0.845 / 53.6 | 0.925 / 76 492 |
| ShiftFactoredTransport+Conformal | 0.945 / 0.624 | 0.845 / 54.3 | 0.909 / 76 418 |

The JointTTA family and SemiConformal are the rows near nominal on all three
datasets. On folktables_income Static, SubspaceAlign, ShiftInflator and
ShiftFactoredTransport have mean widths of 1e9-6e9 (a few unbounded seeds);
use the conformal variants there.

**label_shift / ShiftFactoredTransport (new oracle columns).** Prior TV against
the true target prior (Source = no adaptation, Raw = unconstrained estimate,
Adapted = what the method uses) and shift coverage vs the base model:

| Dataset | Source / Raw / Adapted TV | BaseShiftCoverage | ShiftCoverage (+Conformal) |
|---|---|---:|---|
| california_housing | 0.206 / 0.053 / 0.204 | 0.842 | 0.891 (0.891) |
| diabetes | 0.216 / 0.359 / 0.214 | 0.769 | 0.885 (0.910) |
| folktables_income | 0.157 / 0.185 / 0.154 | 0.831 | 0.893 (0.893) |

Transport lifts coverage by 0.05-0.12 on all three. The adapted prior barely
moves from the source prior (TV within 0.003); on california the raw estimate
is much closer to the truth (0.053) and is not used.

**New rows.**

- `multivariate_intervals / MultivariateGaussian+LevelSetCP`: matches the
  existing +Conformal row in joint coverage on all 5 datasets (0.918-0.999)
  with intervals 0.3-2.5% tighter (energy_efficiency IS 69.27 vs 69.75).
  The raw MultivariateGaussian covers 0.44-0.77 jointly.
- `ordinal / DiscreteWasserstein1`: last of the ordinal-aware methods on both
  datasets: MAE 0.328 / QWK 0.908 on synthetic_ordinal (statsmodels
  OrderedModel 0.250 / 0.936), MAE 1.638 / QWK 0.641 on folktables_decile
  (CumulativeLinkEnsemble 1.436 / 0.725). Ahead of plain MSE only.
- `fm_uq_scenarios / tr_Mondrian+*`: all 720 rows finite
  (`InfiniteIntervalFrac` <= 0.013). Median 90% coverage 0.89-0.95 in the
  clean / outliers / shift / noise_all / heavy_tail scenarios; `noise_test`
  breaks it (median 0.565, 0.00 on insurance_claims and zurich_delays with
  the FM backbones), as expected for test-only noise.

**CUDA OOM handled by CPU retry.** 30 rows carry "this row ran on cpu":

| Suite / method | Rows |
|---|---:|
| likelihoods / NSF (tlc_fare, folktables_all) | 6 |
| fm_multitarget / GaussCopula+TabFM | 6 |
| fm_multitarget / Indep+TabFM | 6 |
| fm_multitarget / torchregress_NSF | 3 |
| foundation_model / TabFM | 6 |
| foundation_model / KumoTabular | 3 |

**Remaining failed rows: 27.** Not only TT_BBSE:

| Suite / method | Dataset(s) | Rows | Cause |
|---|---|---:|---|
| label_shift / TT_BBSE | folktables_income | 3 | confusion matrix ill-conditioned (cond 2e16-1e17), by design |
| age_regression / CNN+RankNContrast | 5 image datasets | 15 | device mismatch in `linalg_lstsq` (cuda vs cpu tensor) |
| keypoint_regression / CNN+GaussianNLL, ResNet+GaussianNLL | synthetic_image_keypoint | 6 | `.numpy()` on a cuda tensor |
| image_density_parity / sbi_zuko_NSF | synthetic_image_keypoint | 3 | NVML assert in the CUDA caching allocator (not retried) |

The 24 image rows are harness device-handling bugs on GPU, outside this
file's tabular scope but open before release.
