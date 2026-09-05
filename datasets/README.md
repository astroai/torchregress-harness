# Datasets directory

Datasets are downloaded or generated at runtime and cached here.
This directory is gitignored except for this README.

Generated synthetic datasets are small and fast.
Real tabular datasets (California Housing, Diabetes) are fetched via sklearn.
OpenML Year is fetched from OpenML and may take a few seconds on first run.

## Image datasets

Synthetic (`synthetic_image_regression`, `synthetic_image_depth`,
`synthetic_image_keypoint`, `biwi_head_pose` proxy) generate in-memory —
no downloads, CI-safe.

Real image sets cache under `$TORCHREGRESS_HARNESS_CACHE`
(default `~/.cache/torchregress-harness`):

* `nyu_depth_v2[_fullres]` — `nyu_depth_v2_labeled.mat` (~2.7 GB, mirrors
  tried in order; place manually if all mirrors fail).
* `imdb_wiki[_fullres]` — `imdb_crop.tar` (6.6 GB) + `wiki_crop.tar`
  (774 MB) + metadata tars. The ETH mirror 404s `wiki_meta.tar`
  (2026-08); the loader falls back to the HuggingFace mirror, and a
  missing `wiki_meta.tar` only drops the wiki subset (imdb still runs).
* `biwi_head_pose[_tiny]` — real data via `BIWI_ROOT` env (dir of `*.jpg` +
  `*.txt` poses); synthetic proxy otherwise (flagged `SyntheticFallback`).
  Tiny = 200 samples at 32×32 for CPU smoke.
* `utkface_age[_tiny]` — UTKFace age (0–100y) via `UTKFACE_ROOT` env
  (`<age>_<gender>_<race>_*.jpg`); synthetic face-blob proxy otherwise.
  Small-real P0 default (avoids 8 GB IMDB download for debug).
* `boneage` — RSNA bone-age maturity (0–228 months) via `BONEAGE_ROOT`
  env (csv + X-rays); synthetic gradient+blob proxy otherwise.
  Imbalanced/high-stakes P0 axis.
* `nyu_depth_v2[_tiny|_fullres]` — tiny = 50 samples at 64×64 for CPU smoke.
* `bbbc_cellcount[_tiny]` — BBBC005 microscopy counts (1–100) via auto-download
  (~1.8 GB Broad zip) or `BBBC_ROOT` (zip or dir); synthetic Poisson-blob
  proxy otherwise. Counts never standardized. Tiny = 200 at 64×64.
* `appa_age[_tiny]` — APPA-REAL apparent age (mean + observed rater std) via
  auto-download (~845 MB ChaLearn zip, no login) or `APPA_ROOT`;
  synthetic heteroscedastic proxy otherwise. Tiny = 200 at 32×32.
