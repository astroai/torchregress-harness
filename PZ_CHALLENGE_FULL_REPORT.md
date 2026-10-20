# LSST DESC PZ Data Challenge — methods, tasks, n(z), and flow comparison

## Scope

The challenge (April–September 2026) has four p(z) task sets (tasksets 1–4),
each with Cardinal/Flagship simulations and 1yr/10yr scenarios, plus an n(z) /
tomography part.  The public p(z) test files have no labels; the repository's
accepted submissions are estimates, and the reserved truth is not in the public
repository.  We therefore benchmark on held-out official training data and use
the post-challenge `reserved` n(z) WFD truth (object-id matched) for the n(z)
diagnostics.  This is an internal benchmark, not a challenge submission.

## Accepted submissions found on GitHub

`LSSTDESC/pz_data_challenge` contains 37 accepted submission modules
(latest collection PR #92, `collect/1016`).  There is no committed numeric
leaderboard: `docs_results/high_level_summary.rst` points at a generated
`results/scores_summary.csv` that is absent from the repository and is generated
only during the private evaluation build.  No challenge results paper was found
on arXiv as of 2026-10-19 (queries for "PZ Data Challenge" / LSST photo-z data
challenge).  The accepted methods are:

- Template/SED fitting: BPZ (two template sets), LePhare, MaxOptPZ, TPZ-colors
  (curvature and no-curvature), RAIL-BPZ, FZBoost (base/downsampled/KNN/faint).
- Machine learning: AION, Ascention, Cin_zs, Conclave, COSM, DNF (LSST/Roman
  fallbacks and task-3/4 clean variants), EasyForest, Expiation, GraySmoke,
  KernelPool, LSST-v3, MLP-pretrained, MLPVAE (three variants), NN augmentation,
  OwenQueen, PZ-ResNet-flow, RAIL-KNN, WhiteSmoke, ZNN, PATCH.
- Augmentation: NN augmentation.

Their presence means a full official ranking cannot be reconstructed from the
public repo.  References obtained from DS: `docs/pz_challenge.pdf`,
`github.com/LSSTDESC/pz_data_challenge`, `docs_results/high_level_summary.rst`.

## Task definitions (from the challenge document)

- Task 1: representative training; uniform i<23 selection; test same distribution.
- Task 2: non-representative training (spectroscopic selection emulation);
  test all objects to i<25.4.
- Task 3: as task 2 plus narrowband/COSMOS2020 reference redshifts
  (`redshift_manyband`, survey flags); test labels include neither truth nor
  manyband redshifts.
- Task 4: as task 3 plus unrecognized blends (friends-of-friends, summed flux,
  brightest-object redshift); task-4 training is 1-year only in the official
  description.

Metrics: point bias, outlier rate, sigma_MAD; PDF CDE loss, PIT KS/RMSE/KL;
training/estimation time and output size.  n(z) adds ensemble n(z), tomographic
bin assignment, and inference.

## Normalizing-flow comparison

Reference: Ren et al. 2026, ApJ 997, 65, DOI 10.3847/1538-4357/ae201d
("Photo-z Estimation with Normalizing Flow", nflow-z).  Their cINN uses 25
affine-coupling layers; cNSF uses 3 spline layers.  On the CSST mock they report
cINN σ_NMAD=0.018±0.0004 and η=1.60%±0.16%; cNSF σ_NMAD=0.017±0.002 and
η=0.83%±0.07%.  cNSF is faster (~6 min vs ~25 min training+test) and beats RF,
MLP and CNN on the CSST mock; cINN has slightly lower bias and cleaner PIT;
cNSF's PIT has mild edge peaks.  They also report ~30% improvement over official
faint DESCaLS results.

Our harness implements a direct proxy comparison:
`NormalizingFlow` = zuko cNSF (3 NSF transforms, bound 8) and
`NormalizingFlow_cINN` = conditional RealNVP coupling (4 transforms), on the
same PZ features and training budgets.  This is not the paper's exact FrEIA
architecture but is an apples-to-apples flow-family comparison.

## PZRecipe run (2026-10-20)

PZRecipe was added to `PZChallengeSuite` and run on **all 16 p(z) challenge
datasets** (task sets 1–4, Cardinal/Flagship, 1yr/10yr), **3 seeds**, **60
epochs**, on a CANFAR GPU run sharded eight ways (`release_20261020_1127`).
Two variants were run:

- `PZRecipe`: CE+CDF+RnC-N FT-Transformer stage-1 p(z), 100 quantile bins.
- `PZRecipe_TabM`: same stage 1, plus the paper-style TabM stage-2 classifier
  on frozen features with 200 bins.

Mean over 16 datasets × 3 seeds:

| method | mean RMSE | median NLL | mean CRPS | coverage90 | σ_NMAD | train_s |
|---|--:|--:|--:|--:|--:|--:|
| PZRecipe | 0.0581 | −2.002 | 0.0196 | 0.8762 | 0.0131 | 693 |
| PZRecipe_TabM | 0.0520 | −2.114 | 0.0174 | 0.8702 | 0.0119 | 733 |

For comparison, existing `NormalizingFlow` mean is RMSE 0.0480 / CRPS 0.0174
/ σ_NMAD 0.0128 at 2129 s, and `MDN_K3` is RMSE 0.0594 / CRPS 0.0199 at 22 s.
TabM stage 2 improves RMSE, CRPS and σ_NMAD over stage 1 in **48/48**
dataset×seed pairs, but PZRecipe remains a cheap reference rather than a
state-of-the-art result in this unsearched configuration.  Coverage is ~0.87,
slightly under nominal 0.90.  Binned NLL is not a continuous-density NLL; one
task3 flagship seed has a positive-NLL edge-bin outlier (1.00 and 5.01) and is
retained in the table.

Full table and caveats: [`reports/pz_recipe_summary.md`](reports/pz_recipe_summary.md);
merged artifact: [`reports/pz_recipe_all_results_long.json`](reports/pz_recipe_all_results_long.json).

## n(z), inference and PPI

The n(z) archives contain 10 DDF fields with reference redshifts and a 1M-object
WFD catalog without public labels, for tasksets 1 and 2, Cardinal/Flagship,
1yr/4yr.  The reserved archive supplies WFD truth for post-challenge evaluation.
The harness trains per-object p(z) on all DDF fields, predicts WFD p(z), and
aggregates to n(z) and four tomographic bins.  PPI (`torchregress.inference`:
`ppi_mean_ci`) rectifies a small labeled WFD slice plus model predictions on the
rest of WFD for mean-z and bin fractions, with 90% CI coverage against truth.
The existing generic `ppi` suite remains the reference implementation for
prediction-powered mean/median/OLS.  The n(z) suite additionally reports
prediction-powered median-z CIs (`ppi_quantile_ci`) next to mean-z and
bin-fraction CIs; classical (labeled-only) estimates are the baseline.

### Implementation notes / findings

- `DensityWeightedLoss.fit_density` used a full `query @ train.T` matrix.  At
  the n(z) scale (and even at 100k PZ rows) this allocates 40+ GB and the process
  is SIGKILLed.  Fixed in torchregress by chunking the query axis; this is now
  covered by `tests/losses/test_imbalanced.py::test_kde_score_samples_chunked_large`.
- Independent quantile heads crossed on PZ task sets 3/4; the PZ suite sorts
  quantiles before constructing CQR/split intervals.
- `InputNoiseAugmentationLoss` must be given a mask-aware base loss
  (`WeightedMSELoss`, not raw `nn.MSELoss`) when the input-noise path forwards
  masks.  This is now the PZ EIV implementation.
- One-million-row WFD inference requires batched prediction; the PZ suite now
  chunks test predictions at 50k.
- The OOD benchmark initially compared scaled predictions to unscaled shifted
  targets, producing RMSE≈1; it now applies the task-1 target scaler to every
  shifted view.

### Current benchmark scope

`pz_core` / `pz_cp` / `pz_extended` cover all 16 p(z) datasets (4 task sets ×
2 simulations × 2 scenarios); `pz_flow` covers task 1 and task 4 (8 datasets);
`pz_ood` covers task1→task2/3/4; `pz_removal` covers task 1 and task 4; the n(z)
suite covers tasksets 1/2 × Cardinal/Flagship × 1yr/4yr.

## Reproduction

Data: `public_nz_t1.tgz`, `public_nz_t2.tgz`, `nz_reserved.tgz`, and the p(z)
`public.tgz` from `https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/`.

Suites: `pz_core` (8 point/distributional/TabM heads), `pz_cp` (split, CQR,
Mondrian, CV+, local, normalized CP), `pz_extended` (evidential, beta-NLL,
input-noise EIV, density-weighted, BMC), `pz_flow` (cNSF vs cINN), `pz_removal`
and `pz_ood` (paper-style robustness), `nz_challenge` (n(z) + PPI).

All suites are registered in `tools/_suite_registry.py`; jobs are generated by
`tools/release_jobs.py`.

## Results produced by the harness (2026-10-19)

- `reports/pz_core_summary.md` — 8 point/distributional/TabM heads on all 16 p(z)
  datasets.  Mean RMSE: NSF 0.048, TabM 0.050, MSE 0.054, MDN 0.059, Gaussian
  0.066; NSF median NLL −2.22 and TabM Gaussian NLL mean coverage 0.921.
- `reports/pz_cp_extended_summary.md` — six CP variants on all 16 datasets and
  evidential/beta/EIV/imbalanced heads on all 16.  CV+ is conservative
  (0.925 mean 90% coverage), CQR has the lowest CP RMSE (0.055), normalized CP
  the narrowest mean width; evidential NIG over-covers (0.997) and density-weighted
  MSE under-performs (0.091), while BMC recovers to 0.058.
- `reports/pz_flow_summary.md` — cNSF vs cINN-style RealNVP on 8 datasets.
  cINN is ~2.3× faster (896 s vs 2,115 s mean training) and usually has better
  NLL/coverage, but has one catastrophic-RMSE tail on task4 flagship 10yr;
  cNSF has one NLL underflow outlier on task4 flagship 1yr.
- `reports/pz_ood_summary.md` — task1-trained model on tasksets 2/3/4: RMSE
  0.026→0.064/0.113/0.129 (MSE), TabM 0.024→0.054/0.092/0.106; SplitCP coverage
  collapses from 0.899 to 0.775/0.770/0.737, while Gaussian coverage is more
  stable (0.900→0.862/0.891/0.872).
- `reports/pz_removal_summary.md` — paired removal test.  Removing the faint
  third increases task-4 RMSE from 0.18 to 0.62–0.80 and drops 90% coverage to
  0.70–0.83; the colour and field omissions are milder.  This reproduces the
  paper's qualitative result that calibration and scatter must be assessed
  separately.
- `reports/pz_nz_summary.md` — n(z) / tomographic-bin / PPI diagnostics.  On
  task1 cardinal 1yr, n(z) L1 is 0.042–0.097, n(z) KS 0.006–0.025 and bin
  accuracy 0.966–0.975; PPI mean-z and bin-fraction CIs cover the truth
  (1.0 in 15/16 cases; one 0.5) but are conservative at this labeled-subset
  size.  Taskset 2 n(z) (5.2M DDF rows) is the slowest benchmark; its
  Cardinal/Flagship 1yr/4yr jobs are running on CANFAR in separate shards.
  Two taskset-2 shards (cardinal 1yr, flagship 4yr) completed at 10 epochs with
  RMSE ~0.16–0.19, much worse than taskset 1; the remaining two shards were
  blocked by CANFAR admission and remain queued.  Artifact pull for the two
  completed shards also failed on admission, though their per-job metrics are
  in the session logs.

## Bugs / missing functionality surfaced

1. `DensityWeightedLoss.fit_density` allocated a dense N×N distance matrix; at
   100k rows this is 40 GB and the process is SIGKILLed.  Fixed with chunked
   query blocks in torchregress; new regression test added.
2. Independent quantile heads crossed on task sets 3/4, producing inverted
   CQR intervals; the PZ suite now sorts quantiles before all CP construction.
3. `InputNoiseAugmentationLoss` requires a mask-aware base loss; raw
   `nn.MSELoss` fails with an unexpected `mask` keyword.  Use
   `WeightedMSELoss` (documented in the suite).
4. Million-object WFD prediction OOMs in a single forward; PZ predictions are
   now chunked at 50k.
5. OOD benchmark initially mixed standardized predictions with raw shifted
   targets; fixed by applying the task-1 target scaler to all views.
6. Bounded NSF can produce extreme NLL underflow on task4 flagship 1yr; cINN
   RealNVP can produce catastrophic point outliers on task4 flagship 10yr.
   Both need robust/median reporting and tail diagnostics.

## Public leaderboard / results note

No numeric public leaderboard was found.  The challenge repository has 37
accepted submission modules (latest collection PR #92, `collect/1016`) but the
summary CSV and individual score files are generated by a private evaluation
build and are not committed.  `PZDC1paper` is the older DC1 paper, not the 2026
challenge results.  The current challenge technical note was not on arXiv as of
2026-10-19.  Our report therefore compares against the challenge task/method
definitions and the published nflow-z paper, not against unpublished scores.
