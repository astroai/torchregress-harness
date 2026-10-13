# Release run #5 analysis (`release_20261008_1756`)

Harness `4971b99` → `fb4bdb2` (refills), torchregress `5d66913`, sbi git `29af137`.
Artifacts: `/arc/projects/ots/torchregress-harness/runs/release_20261008_1756/`.
Status at 2026-10-13 00:40 UTC: every shard DONE with 0 failed jobs except
tabular shard 3: `eiv / folktables_eiv` (1.5M rows) hung for 2.5 days with no output (one EIV method does not scale; per-job `JOB_TIMEOUT` added in `c8d4f43`). The fm group rerun (TabFM fix, TabPFN) is pending a CANFAR auth outage (sessions fail at start since 2026-10-13 00:36 UTC).
Silent session deaths on 5 shards (CANFAR outage aftermath) were refilled once;
all but the 1.5M-row shard completed on refill.

## 1. Density estimators on real data (`density_parity`, `image_density_parity`)

Shared encoder and training loop; scores in standardised target units; mean of 3 seeds.

Head-to-head, torchregress wins / datasets compared:

| Pair | NLL | Energy score | CRPS | Variogram | PIT KS | CorrErr |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| torchregress NSF vs sbi NSF (nflows) | 6/10 | 8/10 | 8/10 | 3/6 | 6/10 | 4/6 |
| torchregress NSF vs sbi zuko NSF | 5/9 | 5/9 | 5/9 | 3/5 | 6/9 | 3/5 |
| torchregress MAF vs sbi MAF (nflows) | 4/10 | 8/10 | 8/10 | 3/6 | 3/10 | 3/6 |
| torchregress MAF vs sbi zuko MAF | 4/10 | 2/10 | 1/10 | 2/6 | 2/10 | 1/6 |
| torchregress MDN vs sbi MDN | 3/10 | 2/10 | 2/10 | 3/6 | 4/10 | 2/6 |

Findings:

- **NSF is torchregress's strongest estimator**: better than sbi's default (nflows)
  NSF on most datasets and level with sbi's zuko NSF (same backend).
- **MDN (full covariance) lags sbi's MDN** at equal parameter count; the gap is
  systematic on `mt_scm20d` (16 targets): NLL 2.8-3.6 vs -0.4-0.2 on every seed.
  Candidate causes: Cholesky diagonal parameterisation (softplus + min_std) or
  initialisation. Action item for the library.
- **zuko MAF is unstable under this trainer** (both torchregress and sbi's zuko
  wrapper): single seeds diverge (udacity NLL 25 / 156; california 3.1 / 4.4),
  while nflows MAF stays stable. Action item: gradient clipping is already on;
  check MAF permutation/initialisation defaults in `create_flow_model`.
- `mt_rf1` breaks most estimators on some seed (test targets far outside the
  training range); only sbi's nflows MAF stays finite on all seeds. Report it as
  a tail-robustness stress case, not a ranking.
- pzflow often has the best energy score but poor joint HPD coverage
  (`mt_jura` 0.46, `mt_scm20d` 0.38 vs ~0.8-0.9 for the others).
- One failed row: `synthetic_image_keypoint / sbi_zuko_NSF` (NVML internal
  assert on the GPU node, infrastructure).

Corner plots (test row 0, seed 0; torchregress blue, sbi orange, star = truth):
`mt_jura__seed0__row0.png`, `mt_enb__seed0__row0.png`. All 210 sample dumps:
`runs/release_20261008_1756/tabular/density_samples/`.

## 2. sbibm posterior estimation (`sbi_parity`, balanced C2ST, 3 seeds)

C2ST (0.5 = indistinguishable from the reference posterior):

| Task | tr NSF | sbi NPE | tr MAF | sbi MAF | tr MDN | sbi MDN | pzflow | FMPE | NPSE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bernoulli_glm | 0.744 | 0.675 | 0.724 | 0.664 | 0.896 | 0.699 | 0.801 | 0.655 | 0.692 |
| gaussian_linear | 0.628 | 0.533 | 0.574 | 0.527 | 0.572 | 0.557 | 0.793 | 0.568 | 0.564 |
| gaussian_mixture | 0.997 | 0.513 | 0.693 | 0.652 | 0.559 | 0.540 | 0.555 | 0.550 | 0.564 |
| slcp | 0.805 | 0.851 | 0.917 | 0.929 | 0.958 | 0.872 | 0.859 | 0.839 | 0.790 |
| twomoons | 0.588 | 0.498 | 0.589 | 0.629 | 0.523 | 0.534 | 0.517 | 0.627 | 0.531 |

sbi's NPE leads on most tasks here. Unlike `density_parity`, this suite trains
torchregress with the generic harness loop and sbi with its own tuned trainer,
so it mixes estimator and training-recipe differences. `torchregress_NSF` on
`gaussian_mixture` (0.997) is a training failure to investigate (likely the NSF
tail bound vs the wide mixture component).

## 3. Foundation models (`foundation_model`, 3 seeds)

| Dataset | GaussianNLL | MDN | TabICL | TabICLv2 | KumoTabular | TabFM |
|:--|--:|--:|--:|--:|--:|--:|
| california_housing R² / CRPS | 0.732 / 0.289 | 0.742 / 0.283 | 0.831 / 0.204 | 0.832 / 0.203 | **0.860 / 0.183** | 0.854 / 0.26 |
| diabetes | 0.484 / 29.7 | 0.458 / 30.3 | 0.511 / 28.4 | **0.515 / 28.3** | 0.512 / 28.4 | 0.514 / 40.8 |
| openml_year | 0.783 / 0.086 | 0.793 / 0.082 | 0.879 / 0.058 | 0.881 / 0.058 | **0.899 / 0.052** | - |

Foundation models lead clearly on mid-sized tables. The torchregress rows here
are the plain 2-head MLP; adding `TabularMLP` / `fit_tabular` (0.3.0) is the
fair torchregress entry for this suite.

## 4. Against the pre-release leaderboard

Only 92 (suite, dataset, method) cells overlap with `leaderboard_results.json`
(1,748 cells are new). Of those: 67 within 2%, 9 better, 16 worse. The changes
are small and partly confounded by data changes (deterministic OLS moved 6.7% on
`house_sales`). Worth a look: `calibration/VarianceTemperatureScaler` CRPS +15%
on `california_housing` (changed for 0.3.0).

## 5. Bugs found and fixed during runs #3-#5

Harness: comparators never installed (no pip in the conda env); PyPI torch
replacing the conda build; torchsort build; predictive samples not
inverse-transformed (TabICL / flows / SLS rows on the wrong scale); sdm
generator device; imbalanced C2ST (floor 0.833); diagonal-vs-full MDN pairing;
TCGA re-download per seed; lifelines AFT on 49k raw pixels (silent OOM);
folktables re-download; credentials/launcher for agent use.
Library: full-covariance `MixtureDensityLoss.sample`; PRD-001 tail mass;
DML `polynomial_degree` default 3.
## 6. Foundation models + torchregress UQ layer (`fm_uq_scenarios`, `fm_multitarget`)

Harness `6c52221`, sdm git `6bc86cf`, tabicl git `c91f00d`; 5 datasets x 3 seeds (single target), 4 MULAN sets x 3 seeds (multi-target). Standardised target units; means over datasets and seeds.

**TabFM excluded below.** In sdm its regression head is a single point output (`out_channels=1`), so its "quantile grid" was a point mass (coverage 0). Fixed in `c8d4f43` (point-only bases get split-conformal residual distributions, `tr_ResidualDist`); the TabFM and TabPFN rows will come from the fm rerun.

### 6.1 90% coverage / coverage above the training p90 / interval score

| Scenario | Base | Raw (own) | tr_CQR | tr_DCP | tr_Split | tr_Mondrian |
|:--|:--|:--|:--|:--|:--|:--|
| clean | TabICL | 0.90 / 0.84 / 2.18 | 0.90 / 0.85 / 2.19 | 0.90 / 0.85 / 2.24 | 0.90 / 0.75 / 2.62 | 0.90 / 0.82 / 2.53 |
| clean | KumoTabular | 0.91 / 0.87 / 2.07 | 0.90 / 0.86 / 2.07 | 0.90 / 0.87 / 2.15 | 0.90 / 0.74 / 2.53 | 0.90 / 0.82 / 2.45 |
| clean | LightGBM | 0.69 / 0.49 / 2.83 | 0.91 / 0.76 / 2.55 | 0.80 / 0.63 / 3.80 | 0.90 / 0.65 / 2.88 | 0.90 / 0.75 / 2.74 |
| clean | TabularMLP | 0.91 / 0.81 / 2.83 | 0.89 / 0.79 / 2.56 | 0.89 / 0.79 / 2.54 | 0.90 / 0.69 / 2.77 | 0.89 / 0.78 / 2.64 |
| outliers | TabICL | 0.96 / 0.92 / 3.05 | 0.94 / 0.90 / 3.00 | 0.94 / 0.89 / 2.75 | 0.95 / 0.82 / 3.00 | 0.95 / 0.91 / 3.29 |
| outliers | KumoTabular | 0.97 / 0.96 / 2.80 | 0.95 / 0.87 / 2.74 | 0.95 / 0.93 / 2.39 | 0.95 / 0.83 / 2.87 | 0.95 / 0.92 / 3.34 |
| outliers | LightGBM | 0.87 / 0.74 / 3.51 | 0.95 / 0.84 / 3.54 | 0.95 / 0.83 / 3.86 | 0.94 / 0.77 / 3.21 | 0.95 / 0.86 / 3.47 |
| outliers | TabularMLP | 0.99 / 0.98 / 6.32 | 0.95 / 0.79 / 3.55 | 0.95 / 0.77 / 3.38 | 0.95 / 0.75 / 3.39 | 0.95 / 0.84 / 3.83 |
| shift | TabICL | 0.90 / 0.86 / 2.22 | 0.90 / 0.84 / 2.23 | 0.90 / 0.85 / 2.24 | 0.90 / 0.74 / 2.62 | 0.90 / 0.85 / 2.58 |
| shift | KumoTabular | 0.91 / 0.89 / 2.02 | 0.90 / 0.87 / 2.04 | 0.90 / 0.88 / 2.12 | 0.91 / 0.73 / 2.52 | 0.89 / 0.84 / 2.49 |
| shift | LightGBM | 0.69 / 0.54 / 2.90 | 0.91 / 0.78 / 2.72 | 0.79 / 0.63 / 4.90 | 0.91 / 0.68 / 2.81 | 0.91 / 0.80 / 2.70 |
| shift | TabularMLP | 0.92 / 0.86 / 2.75 | 0.91 / 0.84 / 2.52 | 0.91 / 0.83 / 2.49 | 0.91 / 0.76 / 2.74 | 0.90 / 0.84 / 2.64 |
| extrapolate | TabICL | 0.93 / 0.86 / 2.51 | 0.93 / 0.85 / 2.52 | 0.93 / 0.85 / 2.54 | 0.85 / 0.70 / 2.95 | 0.85 / 0.77 / 2.88 |
| extrapolate | KumoTabular | 0.94 / 0.87 / 2.27 | 0.92 / 0.85 / 2.28 | 0.93 / 0.86 / 2.31 | 0.84 / 0.68 / 2.78 | 0.85 / 0.77 / 2.73 |
| extrapolate | LightGBM | 0.68 / 0.47 / 3.00 | 0.89 / 0.74 / 2.95 | 0.76 / 0.54 / 5.04 | 0.88 / 0.68 / 3.22 | 0.87 / 0.77 / 3.17 |
| extrapolate | TabularMLP | 0.90 / 0.80 / 3.07 | 0.88 / 0.76 / 2.81 | 0.89 / 0.78 / 2.77 | 0.87 / 0.71 / 3.17 | 0.85 / 0.78 / 3.08 |
| noise_test | TabICL | 0.94 / 0.91 / 5.92 | 0.95 / 0.91 / 5.94 | 0.90 / 0.86 / 5.47 | 0.47 / 0.31 / 9.41 | 0.45 / 0.35 / 8.86 |
| noise_test | KumoTabular | 0.95 / 0.91 / 9.09 | 0.95 / 0.90 / 9.10 | 0.92 / 0.89 / 6.59 | 0.49 / 0.31 / 6.15 | 0.47 / 0.37 / 6.14 |
| noise_test | LightGBM | 0.64 / 0.46 / 3.61 | 0.86 / 0.72 / 3.32 | 0.76 / 0.61 / 4.26 | 0.72 / 0.55 / 4.93 | 0.73 / 0.65 / 4.52 |
| noise_test | TabularMLP | 0.77 / 0.64 / 4.88 | 0.75 / 0.65 / 4.70 | 0.75 / 0.65 / 4.68 | 0.72 / 0.53 / 5.12 | 0.71 / 0.62 / 5.10 |
| noise_all | TabICL | 0.90 / 0.78 / 2.94 | 0.90 / 0.78 / 2.96 | 0.90 / 0.79 / 2.97 | 0.91 / 0.68 / 3.38 | 0.91 / 0.74 / 3.30 |
| noise_all | KumoTabular | 0.90 / 0.80 / 2.95 | 0.90 / 0.79 / 2.95 | 0.90 / 0.80 / 2.96 | 0.91 / 0.67 / 3.37 | 0.91 / 0.72 / 3.29 |
| noise_all | LightGBM | 0.69 / 0.40 / 3.44 | 0.92 / 0.71 / 3.15 | 0.80 / 0.59 / 4.24 | 0.91 / 0.64 / 3.50 | 0.91 / 0.70 / 3.35 |
| noise_all | TabularMLP | 0.91 / 0.74 / 3.43 | 0.90 / 0.73 / 3.18 | 0.90 / 0.73 / 3.16 | 0.90 / 0.64 / 3.31 | 0.90 / 0.72 / 3.20 |
| imbalance | TabICL | 0.86 / 0.72 / 2.56 | 0.85 / 0.71 / 2.52 | 0.85 / 0.72 / 2.52 | 0.82 / 0.58 / 3.08 | 0.83 / 0.63 / 2.99 |
| imbalance | KumoTabular | 0.87 / 0.73 / 2.45 | 0.85 / 0.72 / 2.42 | 0.86 / 0.74 / 2.40 | 0.81 / 0.57 / 2.94 | 0.83 / 0.62 / 2.86 |
| imbalance | LightGBM | 0.52 / 0.23 / 4.01 | 0.83 / 0.56 / 3.35 | 0.70 / 0.41 / 4.65 | 0.82 / 0.52 / 3.63 | 0.83 / 0.58 / 3.49 |
| imbalance | TabularMLP | 0.86 / 0.68 / 3.13 | 0.84 / 0.66 / 3.00 | 0.83 / 0.65 / 2.99 | 0.81 / 0.55 / 3.30 | 0.82 / 0.61 / 3.18 |
| heavy_tail | TabICL | 0.93 / 0.86 / 1.35 | 0.90 / 0.83 / 1.35 | 0.91 / 0.83 / 1.36 | 0.90 / 0.62 / 2.02 | 0.91 / 0.77 / 1.74 |
| heavy_tail | KumoTabular | 0.93 / 0.89 / 1.25 | 0.90 / 0.83 / 1.27 | 0.90 / 0.84 / 1.27 | 0.91 / 0.65 / 1.91 | 0.91 / 0.81 / 1.67 |
| heavy_tail | LightGBM | 0.65 / 0.49 / 2.00 | 0.91 / 0.75 / 1.79 | 0.79 / 0.62 / 6.31 | 0.91 / 0.47 / 2.24 | 0.91 / 0.70 / 1.86 |
| heavy_tail | TabularMLP | 0.96 / 0.84 / 279.44 | 0.91 / 0.79 / 279.16 | 0.91 / 0.77 / 9.17 | 0.91 / 0.63 / 2.22 | 0.90 / 0.74 / 1.81 |

### 6.2 Full predictive distribution: CRPS / PIT KS, own vs torchregress `PITCalibrator`

| Scenario | TabICL Raw | TabICL tr_PITRecal | Kumo Raw | Kumo tr_PITRecal | LightGBM Raw | LightGBM tr_PITRecal |
|:--|:--|:--|:--|:--|:--|:--|
| clean | 0.222 / 0.070 | 0.221 / 0.054 | 0.212 / 0.109 | 0.212 / 0.054 | 0.260 / 0.259 | 0.266 / 0.218 |
| outliers | 0.237 / 0.111 | 0.238 / 0.062 | 0.225 / 0.126 | 0.222 / 0.060 | 0.276 / 0.223 | 0.303 / 0.129 |
| shift | 0.213 / 0.074 | 0.215 / 0.061 | 0.202 / 0.115 | 0.205 / 0.069 | 0.250 / 0.275 | 0.254 / 0.238 |
| extrapolate | 0.241 / 0.137 | 0.239 / 0.108 | 0.221 / 0.177 | 0.220 / 0.120 | 0.272 / 0.304 | 0.283 / 0.294 |
| noise_test | 0.607 / 0.289 | 0.571 / 0.310 | 0.614 / 0.206 | 0.474 / 0.250 | 0.400 / 0.329 | 0.409 / 0.284 |
| noise_all | 0.328 / 0.072 | 0.327 / 0.048 | 0.325 / 0.113 | 0.325 / 0.051 | 0.349 / 0.265 | 0.349 / 0.220 |
| imbalance | 0.253 / 0.154 | 0.253 / 0.115 | 0.242 / 0.209 | 0.242 / 0.109 | 0.314 / 0.374 | 0.315 / 0.282 |
| heavy_tail | 0.148 / 0.147 | 0.149 / 0.054 | 0.141 / 0.208 | 0.141 / 0.052 | 0.176 / 0.317 | 0.181 / 0.191 |

### 6.3 Multi-target: energy score / CorrErr / 90% ellipsoid coverage

| Dataset (targets) | torchregress_NSF | torchregress_MDN | Indep+KumoTabular | GaussCopula+KumoTabular | Indep+TabICL | GaussCopula+TabICL | Indep+LightGBM | GaussCopula+LightGBM |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| mt_enb (2) | 0.203 / 0.04 / 0.87 | 0.236 / 0.07 / 0.90 | 0.036 / 0.04 / 0.88 | 0.036 / 0.06 / 0.89 | 0.079 / 0.57 / 0.89 | 0.077 / 0.39 / 0.92 | 0.078 / 0.03 / 0.95 | 0.077 / 0.02 / 0.96 |
| mt_jura (3) | 0.558 / 0.13 / 0.90 | 0.619 / 0.14 / 0.94 | 0.472 / 0.15 / 0.89 | 0.473 / 0.20 / 0.88 | 0.477 / 0.16 / 0.88 | 0.477 / 0.18 / 0.90 | 0.514 / 0.15 / 0.75 | 0.516 / 0.12 / 0.74 |
| mt_scm20d (16) | failed (GPU NVML assert) | 2.118 / 0.05 / 0.90 | 0.610 / 0.10 / 0.81 | 0.609 / 0.06 / 0.80 | 0.651 / 0.12 / 0.80 | 0.649 / 0.06 / 0.79 | 1.335 / 0.43 / 0.91 | 1.270 / 0.09 / 0.94 |
| mt_rf1 (8) | 0.187 / 0.09 / 0.91 | 0.632 / 0.16 / 0.96 | 0.031 / 0.02 / 0.89 | 0.031 / 0.03 / 0.89 | 0.033 / 0.03 / 0.87 | 0.033 / 0.03 / 0.87 | 0.237 / 0.27 / 1.00 | 0.231 / 0.20 / 1.00 |

### 6.4 Findings

- **TabICL v2 and KumoTabular are already well calibrated at 90% on clean, shifted and noisy-everywhere data** (0.90-0.91). The torchregress layer does not improve their 90% intervals; `tr_PITRecal` does fix the shape of the distribution (PIT KS 0.07-0.11 -> ~0.05) at unchanged CRPS. KumoTabular has the lowest interval score and CRPS in every scenario except `noise_test`, where its own intervals get very wide (interval score 9.1 vs TabICL 5.9).
- **Feature noise at test time only is the case where the models' own uncertainty matters most.** TabICL/Kumo widen their own intervals for corrupted inputs and stay at 0.94-0.95 coverage; constant-width conformal (`tr_Split`, `tr_Mondrian`) calibrated on clean features collapses to ~0.46. `tr_CQR` inherits the adaptivity (0.95). Any conformal layer on FMs should be adaptive (CQR / DCP), not split.
- **Outliers in train + calibration:** the models' own intervals over-cover (0.96-0.97) and are wide; `tr_DCP` / `tr_PITRecal` tighten them (Kumo interval score 2.80 -> 2.39) while keeping 0.95. TabularMLP's Gaussian ensemble is the most damaged by outliers (IS 6.3 -> 3.4 after conformal).
- **Extrapolation (test outside the PC1 support):** own intervals hold (0.93) and CQR/DCP preserve them; split and Mondrian under-cover (0.84-0.85): exchangeability is broken and only the model's own uncertainty grows out of support.
- **Imbalance (rare high targets thinned in train and calibration):** every method under-covers (0.81-0.87 overall, 0.56-0.75 on rare targets). This is label shift between calibration and test, which plain conformal cannot repair. Candidate for the library's label-shift tools (`BinnedLabelShiftEstimator`, `PrevalenceAdjustedCP`) in this suite.
- **Conditional coverage is the weak point everywhere:** worst target-decile coverage is 0.62-0.65 for the best methods on clean data (0.54-0.55 for split conformal); Mondrian bins on the predicted value do not fix it.
- **LightGBM quantile regression is badly miscalibrated on its own** (0.52-0.69); CQR brings it to ~0.91. `tr_DCP` on LightGBM's coarse 15-level grid is poor (interval score up to 6.3); use CQR there.
- **TabularMLP diverged on one heavy-tail dataset** (interval score 279, CRPS 19.9 averaged in); a robustness issue of the Gaussian ensemble on sinh-transformed targets, not of the UQ layer.
- **Multi-target:** fitting the FMs once per target beats torchregress's joint NSF / MDN on energy score on every dataset: 5-6x on `mt_enb` (0.036 vs 0.20) and `mt_rf1` (0.031 vs 0.19), 3.5x on `mt_scm20d` (0.61 vs MDN 2.1), but only 1.2x on `mt_jura` (0.47 vs 0.56). The Gaussian copula fixes the dependence where independence is wrong (`mt_scm20d`: CorrErr 0.10 -> 0.06 for Kumo, 0.43 -> 0.09 for LightGBM) but changes ES little: on these sets the marginals dominate the score. torchregress MDN's NLL explodes on `mt_rf1` (as in `density_parity`). torchregress_NSF failed on `mt_scm20d` (GPU NVML assert).
- **Failures:** TabFM (see above); 21 TabFM / NSF rows hit the PyTorch `NVML_SUCCESS` CUDA allocator assert on the GPU nodes (infrastructure); 1 Mondrian empty-bin error (fixed in `c8d4f43`).
