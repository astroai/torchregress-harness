# Release run #5 analysis (`release_20261008_1756`)

Harness `4971b99` → `fb4bdb2` (refills), torchregress `5d66913`, sbi git `29af137`.
Artifacts: `/arc/projects/ots/torchregress-harness/runs/release_20261008_1756/`.
Status at 2026-10-10 12:30 UTC: every shard DONE with 0 failed jobs except
tabular shard 3 (`eiv / folktables_eiv`, 1.5M rows, GPU, still running).
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
