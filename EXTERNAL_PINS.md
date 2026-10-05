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
| **MAPIE** | `https://github.com/scikit-learn-contrib/MAPIE` (branch `master`) | `3b84b8212d` | 2026-09-30 | Newest `master` (`1.5.1.dev6`), ahead of PyPI 1.5.0 (2026-08-05); includes Cross-CQR + Adaptive Conformal Risk Control. Previous pin `6478f16c9a` (2026-08-04) |
| **torch-uncertainty** | `https://github.com/torch-uncertainty/torch-uncertainty` (branch `main`) | `3f82fe5d15` | 2026-08-07 | `main` HEAD after tag v0.13.0 (`94e511a737`, the previous pin); same v0.13 regression-routine API |
| **Lightning-UQ-Box** | `https://github.com/lightning-uq-box/lightning-uq-box` (tag `v0.4.0`) | `2032b2e87a` | 2026-09-28 | Stress-matrix + image comparator (`luqbox_*` rows). v0.4.0 verified with the existing adapters on 2026-10-05 (MLP signature unchanged). Previous pin v0.3.0 `df6366e` |
| **TabPFN** | PyPI `tabpfn` | `9.1.0` | 2026-10-02 | Foundation-model rows. **9.x downloads gated weights (`Prior-Labs/tabpfn_3_5`) and needs a Hugging Face token (`HF_TOKEN`) that has accepted the model terms.** Previous: 8.4.0 |
| **TabICL** | PyPI `tabicl` | `2.2.0` | 2026-09-02 | `Split+TabICL` rows; checkpoint `tabicl-regressor-v2-20260212.ckpt` comes from the Hugging Face Hub (`jingang/TabICL`), so stage it for offline nodes. Previous: 2.1.1 |
| **scikit-learn** | PyPI | `1.9.1` | 2026-09-10 | Baselines and MAPIE/crepes backends. Previous: 1.9.0 |
| **timm** | `https://github.com/huggingface/pytorch-image-models` (PyPI `timm`) | PyPI latest (no git pin; `timm_resnet18` row is unpretrained) | — | timm backbone for `timm_resnet18+GaussianNLL` (image suite); version self-reported via `LibraryVersion` |
| **lifelines** | PyPI `lifelines` | PyPI latest | — | `lifelines_LogNormalAFT` comparator (tabular censored suite) |
| **crepes** | PyPI `crepes` | PyPI latest | — | `crepes_CQR` comparator (distributional_bins) + conformal suite |
| **ppi-python** | PyPI `ppi-python` (`ppi_py`, Angelopoulos et al.) | `0.2.3` | — | Official PPI package: `PPI_py_Mean/Quantile` comparator rows (ppi suite) |
| **survset** | PyPI `survset` (Drysdale 2022) | PyPI latest | — | SurvSet time-to-event repository loader (`survset_hdfail` real survival data) |
| **folktables** | PyPI `folktables` (Ding et al., NeurIPS 2021) | `0.0.12` | — | ACS PUMS loader (`folktables_income`, `folktables_decile` modern benchmarks) |
| **openpyxl** | PyPI `openpyxl` | PyPI latest | — | xlsx reader for UCI Energy Efficiency (`energy_efficiency`) |
| **h5py** | PyPI `h5py` | PyPI latest | — | HDF5 reader for GalaxiesML photometry (`galaxiesml_eiv`) |
| **pzflow** | PyPI `pzflow` (Crenshaw et al., LSST-DESC) | PyPI latest | — | JAX conditional flows + bundled photo-z galaxies (`pzflow_NSF`, `pzflow_galaxies`) |
| **sbibm** | PyPI `sbibm` (Lueckmann et al., NeurIPS 2021, `--no-deps` + pyro-ppl/future) | `1.1.0` | — | SBI tasks, reference posteriors/C2ST, and held-out simulator draws for SBC (`two_moons`, `slcp`, Gaussian Linear/Mixture, Bernoulli GLM) |
| **sbi** | PyPI `sbi` (mackelab) | `0.27.0` | 2026-08-04 | Amortized posterior comparators (`sbi_parity`) and SBC/TARP diagnostics. Previously unpinned |
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
pip install "git+https://github.com/scikit-learn-contrib/MAPIE.git@3b84b8212d"

# torch-uncertainty from git (main HEAD after v0.13.0)
pip install "git+https://github.com/torch-uncertainty/torch-uncertainty.git@3f82fe5d15"
pip install lightning   # required by torch-uncertainty routines

# Lightning-UQ-Box (stress-matrix comparator)
pip install "git+https://github.com/lightning-uq-box/lightning-uq-box.git@v0.4.0"

# sbi + sbibm (sbibm 1.1.0 pins sbi<0.22, so install it without its deps)
pip install "sbi==0.27.0" pyro-ppl future
pip install --no-deps "sbibm==1.1.0"
```

In the pixi env (linux CI box):

```bash
pixi run pip install "git+https://github.com/scikit-learn-contrib/MAPIE.git@3b84b8212d"
pixi run pip install "git+https://github.com/torch-uncertainty/torch-uncertainty.git@3f82fe5d15" lightning
```

> **⚠️ Git pins are enforced by these install commands, not by the lockfile.**
> A plain `pip install -e ".[conformal]"` / `.[image]` resolves the PyPI floors
> (`mapie 1.4.1`, `torch-uncertainty 0.13.0`) which **work** with the harness but
> are NOT the latest-git builds the manifest pins.  Only the git installs below
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
> it and must be regenerated before the release (plan step 1.5). Tabular smoke
> runs against the new pins pass for MAPIE, crepes, torchcp, BoTorch and
> LUQ-Box; TabPFN/TabICL need their Hugging Face weights staged first.

`PARITY_REPORT.md` and `results/` were regenerated on **2026-08-26**
against the previous pins (`6478f16c9a` / `94e511a737`) on the
CUDA phase and were **fresh** for external-comparison claims until the 2026-10-05 re-pin. Older
artifacts (pre-2026-08-04) generated against MAPIE 1.4.1 PyPI are stale.

Stress-matrix artifacts (`results/latest/stress_matrix_luqboxfix.json` and
the `/arc` runs under `runs/stress_*`) were fresh as of **2026-09-03** against
the LUQ-box pin `df6366e` (harness `539e3a2`+).
