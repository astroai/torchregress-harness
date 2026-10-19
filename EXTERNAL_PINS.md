# External-library pins (MAPIE / torch-uncertainty / Lightning-UQ-Box)

The harness compares torchregress against **the latest git versions** of MAPIE,
torch-uncertainty, and Lightning-UQ-Box, not stale PyPI releases.  This file
records the exact commits a paper-grade run must be benchmarked against, and
how to install / re-pin them.

> **Why git, not PyPI?** At the time of writing, PyPI lags upstream:
>
> - **mapie** — PyPI (1.5.0, 2026-08-05) lags `master`
>   (Cross-CQR, Automatically Adaptive Conformal Risk Control, LLM-as-Judge
>   risk control, docs migrated to Read the Docs).  The harness calls the real
>   `mapie.regression` v1 API (`SplitConformalRegressor` /
>   `ConformalizedQuantileRegressor`, `fit → conformalize → predict_interval`).
> - **torch-uncertainty** — v0.13.0 (2026-07-22) reworked the regression
>   routine's loss/metric contract (breaking change): the routine now calls
>   `loss(distribution, targets)` and probabilistic models must return a dict
>   of distribution parameters.  The harness's image suite requires this API;
>   v0.12.x is unsupported.

## Pinned commits (current)

| Library | Source | Commit | Date | Reason for the pin |
|---------|--------|--------|------|--------------------|
| **MAPIE** | `https://github.com/scikit-learn-contrib/MAPIE` (branch `master`) | `929740af07` | 2026-10-13 | Newest `master` (`1.5.1.dev13`), ahead of PyPI 1.5.0 and tag v1.5.0rc1; smoke-tested with `conformal` and `uq_layer`. Previous: `3b84b8212d` |
| **torch-uncertainty** | `https://github.com/torch-uncertainty/torch-uncertainty` (branch `main`) | `3f82fe5d15` | 2026-08-07 (still `main` HEAD on 2026-10-13) | `main` HEAD after tag v0.13.0 (`94e511a737`, the previous pin); same v0.13 regression-routine API |
| **Lightning-UQ-Box** | `https://github.com/lightning-uq-box/lightning-uq-box` (branch `main`) | `d1b7f6a499` | 2026-10-13 | `main` (`0.5.0.dev0`), ahead of tag v0.4.0 (previous pin); `stress_matrix` smoke-tested |
| **TabPFN** | git `PriorLabs/TabPFN` `main` (fallback: newest PyPI) | latest at launch | 2026-10-10 | Foundation-model rows when a Prior Labs API key is present (`TABPFN_TOKEN`; the launcher reads `~/.canfar/tabpfn_token`). The account must have accepted the TabPFN-3.5 licence (non-commercial, non-production) at https://ux.priorlabs.ai; the Hub repo `Prior-Labs/tabpfn_3_5` is not gated, so no `HF_TOKEN`. Sessions set `TABPFN_NO_BROWSER=1` and cache weights on /arc (`TABPFN_MODEL_CACHE_DIR`). Last PyPI checked: 9.1.0 |
| **TabICL** | git `soda-inria/tabicl` `main` (fallback: newest PyPI) | latest at launch | 2026-10-10 | `TabICL` rows (v2 regressor checkpoint `tabicl-regressor-v2-*`, from the Hub `jingang/TabICL`). Last PyPI checked: 2.2.0 |
| **structured-data-models (sdm)** | `https://github.com/NVIDIA/structured-data-models` (branch `main`, Apache-2.0) | latest `main` at launch (was `98f61289c7`) | 2026-10-10 | One interface to the 2026 tabular foundation models: NVIDIA **KumoTabular** (large; weights OpenMDW-1.1) and Google **TabFM** v1.0.0 (weights `tabfm-non-commercial-v1.0`: benchmarking only, the suite passes `accept_license=True`). KumoTabular returns 999 regression quantiles; **TabFM's regression head is a single point output** (`out_channels=1`), so it is scored as a point model and the FM UQ suites give it split-conformal residual distributions. sdm's TabICLv2 is not benchmarked: `tabicl>=2.2` loads the same v2 checkpoint (row `TabICL`). TabFM does not fit in 7 GB RAM: GPU sessions only |
| **scikit-learn** | PyPI | `1.9.1` | 2026-09-10 | Baselines and MAPIE/crepes backends. Previous: 1.9.0 |
| **timm** | `https://github.com/huggingface/pytorch-image-models` (PyPI `timm`) | PyPI latest (no git pin; `timm_resnet18` row is unpretrained) | — | timm backbone for `timm_resnet18+GaussianNLL` (image suite); version self-reported via `LibraryVersion` |
| **lifelines** | PyPI `lifelines` | PyPI latest | — | `lifelines_LogNormalAFT` comparator (tabular censored suite) |
| **engression** | PyPI `engression` | `1.1.0` | — | Official Engression distributional-regression package (Shen & Meinshausen, JRSS-B 2024); `engression_parity` rows, torch-based and GPU-capable |
| **TabM** | PyPI `tabm` | `0.0.3` | — | Official TabM (ICLR 2025) point/NLL baseline for `pz_challenge`; needs `rtdl_num_embeddings` |
| **crepes** | PyPI `crepes` | PyPI latest | — | `crepes_CQR` comparator (distributional_bins) + conformal suite |
| **ppi-python** | PyPI `ppi-python` (`ppi_py`, Angelopoulos et al.) | `0.2.3` | — | Official PPI package: `PPI_py_Mean/Quantile` comparator rows (ppi suite) |
| **survset** | PyPI `survset` (Drysdale 2022) | PyPI latest | — | SurvSet time-to-event repository loader (`survset_hdfail` real survival data) |
| **folktables** | PyPI `folktables` (Ding et al., NeurIPS 2021) | `0.0.12` | — | ACS PUMS loader (`folktables_income`, `folktables_decile` modern benchmarks) |
| **openpyxl** | PyPI `openpyxl` | PyPI latest | — | xlsx reader for UCI Energy Efficiency (`energy_efficiency`) |
| **h5py** | PyPI `h5py` | PyPI latest | — | HDF5 reader for GalaxiesML photometry (`galaxiesml_eiv`) |
| **pzflow** | `https://github.com/jfcrenshaw/pzflow` (branch `main`) | `bf9c70d6d1` | 2026-10-13 | JAX conditional flows + bundled photo-z galaxies (`pzflow_NSF`, `pzflow_galaxies`, `density_parity`); 4.0.0, smoke-tested |
| **sbibm** | PyPI `sbibm` (Lueckmann et al., NeurIPS 2021, `--no-deps` + pyro-ppl/future) | `1.1.0` | — | SBI tasks, reference posteriors/C2ST, and held-out simulator draws for SBC (`two_moons`, `slcp`, Gaussian Linear/Mixture, Bernoulli GLM). Draws are frozen to `$TORCHREGRESS_HARNESS_CACHE/sbibm/` on first use (`tools/freeze_sbibm.py`); later runs never import sbibm |
| **sbi** | `https://github.com/sbi-dev/sbi` (branch `main`) | `29af137501` | 2026-10-06 (still `main` HEAD on 2026-10-13) | Latest git, ahead of PyPI 0.27.0: amortized NPE (NSF/MAF/MDN), FMPE, NPSE comparators and SBC/TARP diagnostics (`sbi_parity`). `main` rejects estimator options a model does not use, so the suite passes per-model options |
| **econml** | PyPI `econml` (Microsoft) | PyPI latest | — | DRLearner causal comparator (`EconML_DRLearner`) |
| **Kaggle datasets** | Kaggle API (user token, `~/.kaggle/kaggle.json`) | — | — | IHDP (`konradb/ihdp-data`), UCF-QNRF mirrors; cached under the harness cache, never home |
| **CSRNet** | native reimplementation from Li, Zhang, Chen (CVPR 2016), no external package | paper | — | VGG-16 frontend (conv1–conv3) + 6 dilated backend convs, 1/8-resolution density maps, MSE; unpretrained; `crowd_counting/CSRNet` with `LibraryVersion: Li-Zhang-Chen-2016-native-unpretrained` |
| **CatBoost** | PyPI `catboost` | 1.2.10 | — | SOTA gradient-boosting probabilistic comparator: `RMSEWithUncertainty` (mean+var) + point baseline; native categorical support; CPU/GPU (`task_type`) |
| **NYC TLC** | TLC public S3 (`d37ci6vzurychx.cloudfront.net/trip-data`) | `yellow_tripdata_2024-01.parquet` (2.96M rows) | 2026-09-11 | Right-skewed `fare_amount`, raw targets; staged under `staged/tlc` on scratch + ARC; pinned month + schema validation |
| **Udacity steering (staged)** | Kaggle `andy8744/udacity-self-driving-car-behavioural-cloning` (CC0-1.0) | 2022-04-12 snapshot | 2026-09-09 | headerless `driving_log.csv` + `IMG/` frames (7,334 pairs across jungle/make); staged under `staged/udacity` on scratch + ARC allocation |
| **Preprocessed IXI (staged)** | Kaggle `kingpowa/preprocessed-ixi-dataset-with-fs8` (Apache-2.0) | 2025-09-10 snapshot | 2026-09-09 | MNI-registered T1 volumes + `subjects.csv` ages (560 usable after -1.0 sentinel filter); staged under `staged/ixi`; requires `nibabel` |
| **KITTI Eigen split (staged)** | Kaggle `aalniak/kitti-eigen-split-for-monocular-depth-estimation` | 2025-03-03 snapshot | 2026-09-09 | `gt_locations.txt` (train, single-col depth refs) + `test_locations.txt` (rgb/depth pairs); 652+652 resolved pairs; staged under `staged/kitti` |
| **UCF-QNRF (staged)** | Kaggle `faihajalamtopu/ucf-qnrf` (4.5 GB) | 2024-11-14 snapshot | 2026-09-10 | `UCF-QNRF_ECCV18/{Train,Test}` jpg + `_ann.mat` pairs (3,072 files); staged under `staged/qnrf` on scratch + ARC allocation; loader prefers staged mirrors, Kaggle fetch is last resort |

## Install commands

```bash
# MAPIE from git (records the commit for provenance via direct_url.json)
pip install "git+https://github.com/scikit-learn-contrib/MAPIE.git@929740af07f53a9b1856c8d5b37356f75b5096db"

# torch-uncertainty from git (main HEAD after v0.13.0)
pip install "git+https://github.com/torch-uncertainty/torch-uncertainty.git@3f82fe5d15"
pip install lightning   # required by torch-uncertainty routines

# Lightning-UQ-Box (stress-matrix comparator)
pip install "git+https://github.com/lightning-uq-box/lightning-uq-box.git@d1b7f6a49995c70eeef9fc878f2967f83273c49d"

# Engression (PyPI)
pip install "engression>=1.1.0"

# sbi + sbibm (sbibm 1.1.0 pins sbi<0.22, so install it without its deps)
pip install "git+https://github.com/sbi-dev/sbi.git@29af13750100b61ad528c3f8a49daa0ef8c96881" pyro-ppl future
pip install --no-deps "sbibm==1.1.0"

# NVIDIA structured-data-models: KumoTabular, TabFM (latest main)
pip install "git+https://github.com/NVIDIA/structured-data-models.git@main"
```

In the pixi env (linux CI box):

```bash
pixi run pip install "git+https://github.com/scikit-learn-contrib/MAPIE.git@929740af07f53a9b1856c8d5b37356f75b5096db"
pixi run pip install "git+https://github.com/torch-uncertainty/torch-uncertainty.git@3f82fe5d15" lightning
```

> **⚠️ Git pins are enforced by these install commands, not by the lockfile.**
> A plain `pip install -e ".[conformal]"` / `.[image]` resolves the PyPI floors
> (`mapie 1.5.0`, `torch-uncertainty 0.13.0`, `lightning-uq-box 0.4.0`) which **work** with the harness but
> are NOT the latest-git builds the manifest pins.  Only the git installs above
> produce commit-annotated `LibraryVersion` rows; run paper benchmarks with the
> git pins installed.

## Reproducibility & self-reporting

Result rows for external libraries carry a **`LibraryVersion`** field (e.g.
`1.4.1 (git 6478f16c9a)`) so artifacts are self-describing without needing
this file.  Rows are produced by:

- `suites/tabular/conformal.py` → `MAPIE_*` methods (real `mapie.regression`
  API; `mapie.metrics` used for MAPIE's own coverage/width scores).
- `suites/image/age_regression.py` → `torch_uncertainty` method
  (`RegressionRoutine` with `DistributionNLLLoss`, `dist_family="normal"`).
- `suites/tabular/stress_matrix.py` → `luqbox_*` methods
  (`MCDropoutRegression` / `DeterministicRegression` wrapped around `LUQMLP`;
  predict mirrors the library's `predict_step`, see `_LUQBOX_PIN` in-file).
- `suites/image/image_ensembles_uq.py` → `luqbox_*` methods (same LUQ-Box
  classes on flattened pixels; documented model-family gap vs CNN rows).
- `suites/image/age_regression.py` → `timm_resnet18+GaussianNLL`
  (timm backbone, unpretrained; version self-reported).

## How to bump to a newer commit

1. Update the commit SHA(s) above and in the install commands.
2. Re-run the affected suite smoke tests locally and the full suite on the
   linux box:
   ```bash
   pixi run python tools/run_canonical.py --scenario mapie_reference --smoke
   pixi run python tools/run_all.py --suites conformal --datasets synthetic_hetero
   pixi run python tools/run_all.py --suites age_regression --datasets synthetic_image_regression
   ```
3. If upstream changed an API, update `suites/tabular/conformal.py` /
   `suites/image/age_regression.py` accordingly and record the new commit here.

## Offline (Skaha) install for Lightning-UQ-Box

Skaha compute nodes cannot reach pypi, so `scripts/skaha_stress_matrix.sh`
installs the pin from a git clone on `/arc`: package contents are copied into
site-packages with a hand-written dist-info, plus offline stubs for its
optional deps (timm, omegaconf, ema_pytorch, …). The copy is idempotent on
Cavern-backed mounts (`7a46993`).

## Status of older artifacts

> **2026-10-05 re-pin.** The pins above moved for the 0.3.0 release run (MAPIE
> `3b84b8212d`, torch-uncertainty `3f82fe5d15`, LUQ-Box v0.4.0, TabPFN 9.1.0,
> TabICL 2.2.0, scikit-learn 1.9.1, sbi 0.27.0). Every artifact below predates
> it and must be regenerated before the release (plan step 1.5). Later on
> 2026-10-05: sbi moved to git `main` `17ef00b8dd` (2026-10-08: `main` head `29af137501`, verified for NSF/MAF/MDN and zuko NSF/MAF estimators); KumoTabular, TabICLv2 and
> TabFM added through structured-data-models `98f61289c7`. Tabular smoke
> runs against the new pins pass for MAPIE, crepes, torchcp, BoTorch and
> LUQ-Box; TabPFN/TabICL need their Hugging Face weights staged first.

`PARITY_REPORT.md` and `results/` were regenerated on **2026-08-26**
against the previous pins (`6478f16c9a` / `94e511a737`) on the
CUDA phase and were **fresh** for external-comparison claims until the 2026-10-05 re-pin. Older
artifacts (pre-2026-08-04) generated against MAPIE 1.4.1 PyPI are stale.

Stress-matrix artifacts (`results/latest/stress_matrix_luqboxfix.json` and
the `/arc` runs under `runs/stress_*`) were fresh as of **2026-09-03** against
the LUQ-box pin `df6366e` (harness `539e3a2`+).
