# Paper Benchmark Run — Status & Plan

Live status of the full-paper benchmark against the pinned external libraries
(MAPIE / torch-uncertainty), run on CANFAR Skaha headless sessions, and of the
PARITY_REPORT.md regeneration.

**Last updated:** 2026-08-04 (during the run).

---

## 1. Goal

Regenerate every number that feeds [`PARITY_REPORT.md`](PARITY_REPORT.md) against
the **latest git builds** of the external comparison libraries, and record the
exact pinned commits in every artifact:

| Library | Pin (commit) | Why |
|---|---|---|
| MAPIE | `6478f16c9a` (master) | Real `mapie.regression` v1 API + new `CrossConformalizedQuantileRegressor`; PyPI lags |
| torch-uncertainty | `94e511a737` (v0.13.0) | v0.13 regression-routine rework the image suite is written against |

See [`EXTERNAL_PINS.md`](EXTERNAL_PINS.md) for rationale and install commands.

## 2. Where the work lives

* This repo (`torchregress-harness`): benchmark code + results + this status doc.
* `scripts/skaha_paper_benchmark.sh` — the Skaha headless job (clones both
  private repos via `GH_TOKEN`, builds the pixi env, installs the pins, runs the
  full benchmark, writes JSON/MD artifacts).
* `scripts/skaha_publish_results.sh` — emits run artifacts (gzip+base64) through
  the Skaha logs endpoint for retrieval.
* CANFAR Skaha production (`ws-uv.canfar.net`, image `astroai/openworker:latest`)
  — compute.  Persistent data lives under `/arc/projects/ots/torchregress-harness/`
  (the container's `$HOME` is `/arc/home/sfabbro`).

## 3. Current state (2026-08-04)

### Done

1. **Code: MAPIE comparison updated to git main** (commits `ec1e043` → `d9f0ae6`,
   all pushed to `origin/main`):
   * `MAPIE_*` rows call the **real** `mapie.regression` API
     (`SplitConformalRegressor` / `ConformalizedQuantileRegressor`,
     `fit → conformalize → predict_interval`).
   * **`MAPIE_CrossCQR+GBR` added** (`CrossConformalizedQuantileRegressor`,
     `fit_conformalize` over K-fold CV out-of-fold scores on the full train+cal
     data — no calibration split; gated by `_MAPIE_CROSS_CQR_AVAILABLE`).
   * **Fixed MAPIE canonical metrics**: git main moved them to
     `mapie.metrics.regression` with interval-array signatures; the old
     top-level import was silently swallowed, so `MAPIE_Coverage`/`MAPIE_Width`
     were missing from every MAPIE row. `_mapie_coverage_and_width` dispatches
     across both APIs (verified populating).
2. **Code: torch-uncertainty updated to v0.13.0** (`DistributionNLLLoss`,
   `{"loc","scale"}` predict parsing, bare-tensor loader, `>=0.13` guard) —
   part of the earlier `ec1e043` commit.
3. **Local verification (macOS `.venv`, mapie git pin installed)**: ruff clean;
   conformal/MAPIE tests pass (incl. the new Cross-CQR helper + metrics guard).
4. **Skaha plumbing validated**: production server registered in the canfar
   client, `astroai/openworker:latest` image probed (has python 3.13, git, curl,
   pixi 0.76), `/arc` mounted + writable, logs endpoint round-trips, `GH_TOKEN`
   env propagation works, pixi/uv caches redirected to `/arc/home/sfabbro`.
5. **Full benchmark launched** — session **`nur3clql`** (`bench-paper`, 16 cores
   / 32 GB), run dir
   `arc://projects/ots/torchregress-harness/runs/paperbench20260804`.

### In flight

* **Full benchmark session `nur3clql` is Running.**  Step progress so far:
  1. `run_canonical --scenario mapie_reference` ✅ (artifact written)
  2. `run_canonical --scenario crepes_reference` ✅
  3. `leaderboard_conformal.py --epochs 60` ✅ (MAPIE_CQR+GBR ranked #2, IS 124.5k)
  4. `leaderboard_regression.py` ⏳ (was in the `ordinal` suite at last check)
  5. `benchmark_pit_ensemble.py --epochs 60` — pending
  6. `run_all --suites conformal` — pending
  7. `run_all --suites age_regression` — pending

  **Note:** this session launched *before* the Cross-CQR row existed, so its
  results cover the 3 original MAPIE rows (not `MAPIE_CrossCQR+GBR`).

### Decided (user)

Finish the current run (all non-conformal numbers stay valid), then run a
**conformal-only pass** on the updated repo so the conformal/MAPIE tables
include `MAPIE_CrossCQR+GBR`. The PARITY_REPORT.md conformal/MAPIE sections will
come from that pass; everything else from the current run.

## 4. Next steps

1. Let `nur3clql` finish (monitor `canfar ps` / session logs).
2. Launch the conformal-only rerun on `/arc` (already at `d9f0ae6`):
   `bash scripts/skaha_paper_benchmark.sh conformal20260804` after temporarily
   limiting it to the conformal steps (canonical mapie + leaderboard_conformal),
   or a small dedicated script.
3. Fetch artifacts via `scripts/skaha_publish_results.sh` (gzip+base64 through
   `canfar logs`) and decode into `results/latest/` + a versioned run dir.
4. Verify `LibraryVersion` on the MAPIE rows shows `… (git 6478f16c9a)`.
5. Regenerate [`PARITY_REPORT.md`](PARITY_REPORT.md) from the new JSONs
   (all sections; header date/config/pins updated).
6. Commit + push results and the updated report.

## 5. Useful commands

```bash
# Monitor the run
canfar ps --all
canfar logs nur3clql            # or: curl https://ws-uv.canfar.net/skaha/v1/session/nur3clql?view=logs

# Relaunch the full benchmark (resumable; env cached on /arc)
bash /arc/projects/ots/torchregress-harness/src/torchregress-harness/scripts/skaha_paper_benchmark.sh <RUN_ID>

# Publish a run's artifacts through the logs endpoint
bash scripts/skaha_publish_results.sh <RUN_ID>
```

## 6. Session log (scratch)

| Session | Purpose | Status |
|---|---|---|
| `nur3clql` | full paper benchmark (16c/32G) | Running |
| `t3i5osqn` / `frljd6xl` | `/arc` repo pull to Cross-CQR commits | Completed |
| earlier probes | image/API validation | Deleted |
