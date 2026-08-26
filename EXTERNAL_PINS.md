# External-library pins (MAPIE / torch-uncertainty)

The harness compares torchregress against **the latest git versions** of MAPIE
and torch-uncertainty, not stale PyPI releases.  This file records the exact
commits a paper-grade run must be benchmarked against, and how to install /
re-pin them.

> **Why git, not PyPI?** At the time of writing, PyPI lags upstream:
>
> - **mapie** — PyPI is stuck at `1.4.1`, while `master` is far ahead
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
| **MAPIE** | `https://github.com/scikit-learn-contrib/MAPIE` (branch `master`) | `6478f16c9a` | 2026-08-04 | Newest `master`; includes Cross-CQR + Adaptive Conformal Risk Control |
| **torch-uncertainty** | `https://github.com/torch-uncertainty/torch-uncertainty` (tag `v0.13.0`, main HEAD) | `94e511a737` | 2026-07-22 | v0.13.0 regression-routine rework the image suite is written against |

## Install commands

```bash
# MAPIE from git (records the commit for provenance via direct_url.json)
pip install "git+https://github.com/scikit-learn-contrib/MAPIE.git@6478f16c9a"

# torch-uncertainty from git (v0.13.0 tag = main HEAD)
pip install "git+https://github.com/torch-uncertainty/torch-uncertainty.git@94e511a737"
pip install lightning   # required by torch-uncertainty routines
```

In the pixi env (linux CI box):

```bash
pixi run pip install "git+https://github.com/scikit-learn-contrib/MAPIE.git@6478f16c9a"
pixi run pip install "git+https://github.com/torch-uncertainty/torch-uncertainty.git@94e511a737" lightning
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

## Status of older artifacts

`PARITY_REPORT.md` and `results/` were regenerated on **2026-08-26**
against the pinned commits above (`6478f16c9a` / `94e511a737`) on the
CUDA phase and are **fresh** for external-comparison claims. Older
artifacts (pre-2026-08-04) generated against MAPIE 1.4.1 PyPI are stale.
