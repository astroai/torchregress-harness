# Release run #6 analysis (`release_20261013_1917`): tabular groups

Harness `550acb3` for the main run, fills at `5b4a0f2` (zurich MAPIE blocked
predict, tabred MultiQuantile), `1e2a0d5` (tweedie, other MultiQuantile
datasets) and `fdcb483` (multivariate_intervals). torchregress `051ac39`.
Artifacts: `/arc/projects/ots/torchregress-harness/runs/release_20261013_1917/{tabular,tabular_big}/`.
Scope: the `tabular` and `tabular_big` groups only (sbi, fm and image groups are
covered separately). Status at 2026-10-14 15:10 UTC.

## 0. Completion

| Group | Expected (`.sessions/run6_jobs2.txt`) | Landed | Missing |
|---|---:|---:|---|
| tabular | 420 | 420 | - |
| tabular_big | 9 | 6 | `eiv / folktables_eiv` seeds 0-2 (section 5) |

142 suite x dataset jobs, 2 935 method cells, 8 805 rows. Against run #5
(`release_20261008_1756`) all 122 earlier jobs are present; 20 are new
(TabReD cooking / delivery / maps / weather, `zurich_delays`, which replaces
TabReD Sberbank, and `r2c_conformal / california_housing`).

Failed rows after the fills (all others ok or deliberately skipped):

| Suite / method | Dataset | Seeds | Cause |
|---|---|---:|---|
| likelihoods / NSF | tlc_fare, folktables_all | 3 + 3 | `NVML_SUCCESS == r INTERNAL ASSERT` in the CUDA caching allocator (GPU node). Ran in run #5 on tlc_fare; the same assert hit one image row in run #5. |
| label_shift / TT_BBSE | folktables_income | 3 | BBSE confusion matrix ill-conditioned (cond 5e16): some target bins never predicted. By design BBSE raises; EM rows are fine. |

Fixed during this check-in (reruns landed or running):

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
  moved: MAPIE git `3b84b82` → `929740a` (the sessions install MAPIE `main`)
  and scikit-learn 1.9.0 → 1.9.1 (`versions_replica_*.txt`). Pin MAPIE to
  one commit for the published table.
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
tlc_fare (GPU assert above).

## 3. Open harness / library items from this run

1. **Tweedie-family MLP rows do not train.** `tweedie / Tweedie, Gamma,
   CompoundPoisson, InverseGaussian` (MLP + Adam, exp output) give R² -0.05 to
   -0.69 and deviances 46-14 000, in run #5 too, while the linear L-BFGS
   versions match statsmodels' GLM exactly (R² 0.549 / deviance 0.717 on
   synthetic_tweedie). The new warped rows (SqrtT R² 0.50, LogT 0.41) train
   fine on the same backbone, so the problem is the exp-output loss path.
2. GPU NVML assert in NSF on the 1.5M-row likelihood datasets (rerun on a
   fresh node, or `PYTORCH_CUDA_ALLOC_CONF` / smaller batch).
3. MDN precision parameterisation: mixed in `density_parity` (above).

## 4. eiv / folktables_eiv does not fit the 6 h job budget

Seed 0 on harness `550acb3` (device-side batches) hit the 6 h per-job timeout
with no output; seeds 1-2 would have done the same, so the session was stopped
at 14:00 UTC. It is not one method hanging: a profile of all 25 eiv methods on
a folktables_eiv-shaped synthetic set (20k training rows, 10 features, 60
epochs, 2 CPU cores) takes 19 min in total, roughly linear in n; the slowest
are SIMEX (330 s), ODR (130 s), FunctionalEIV_MC (75 s). Scaled to the 1.2M
training rows that is ~60x, i.e. many hours per seed even on 8 cores.
Options: subsample folktables_eiv (e.g. 200k training rows), or split the job
per method, or a longer per-job budget for this one dataset.
