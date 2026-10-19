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
