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
* `biwi_head_pose` — real data via `BIWI_ROOT` env (dir of `*.jpg` +
  `*.txt` poses); synthetic proxy otherwise (flagged `SyntheticFallback`).
