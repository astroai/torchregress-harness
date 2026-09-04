# Paper Benchmark Run — Status & Plan

Live status of the full-paper benchmark against the pinned external libraries
(MAPIE / torch-uncertainty), run on CANFAR Skaha headless sessions, and of the
PARITY_REPORT.md regeneration.

**Last updated:** 2026-09-03. **Both tracks are closed**: the paper-benchmark
track (PARITY_REPORT regeneration) finished 2026-08-26, and the stress-matrix
diversity track finished 2026-09-03 with a fully verified LUQ-box comparator
re-run. All verification sessions complete — nothing in flight.

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
* CANFAR Skaha production (`ws-uv.canfar.net`, image `astroai/notebook:latest`,
  headless kind) — compute.  Persistent data lives under
  `/arc/projects/ots/torchregress-harness/` (the container's `$HOME` is
  `/arc/home/sfabbro`).
  **⚠️ `astroai/openworker:latest` — used by every run before 2026-09-03 — was
  removed from the registry that day** (new sessions die in ImagePullBackOff).
  `astroai/notebook:latest` is the validated drop-in (runs
  `stress_luqboxfix20260903` + seed checks used it).

## 3. Track status (2026-09-03)

### Track 1 — Paper benchmark + PARITY_REPORT: CLOSED (2026-08-26)

`nur3clql` finished all steps; the follow-up conformal-only pass added the
`MAPIE_CrossCQR+GBR` rows. [`PARITY_REPORT.md`](PARITY_REPORT.md) was fully
generated on **2026-08-26** against the pinned externals (MAPIE `6478f16c9a`,
torch-uncertainty `94e511a737`): 943-row / 208-method leaderboard
(`leaderboard_results.json`) + canonical scenarios on the CUDA phase.
Exact-match parity with MAPIE and four-way split-conformal agreement
(torchregress = MAPIE = crepes = torchcp) — see the report header.

### Track 2 — Stress-matrix diversity suite: CLOSED (2026-09-03)

Built and hardened across `4bce8ff` → `d6fd67a` (all pushed):

1. `StressMatrixSuite` (6 axes × method families × metrics) + offline
   Lightning-UQ-Box install (`4bce8ff`, `cdff7b8`); first full headless run
   `stress_20260903_011458` exposed six failure modes, all fixed (`07ad839`).
2. NoisyTargetNLL eval-protocol fix + SAGE tuning (`3b4b156`); 3-seed
   stability 48/48 rows clean.
3. LUQ-box comparator rows landed in `stress_20260903_174930` but the
   MC-dropout row was degenerate (`LUQMLP` default `dropout_p=0.0` → no
   dropout layers → MC std collapsed to the clip floor, NLL 1.8e15, point
   metrics bit-identical to the deterministic sibling). Fixed in `539e3a2`
   (dropout_p=0.5, fit/predict split with honest timing, library-faithful
   `predict_step` semantics).
4. **Final full run `stress_luqboxfix20260903`** (session `aljbqdg9`,
   harness `539e3a2`): **18/18 rows OK, 0 errors, 0 skips**; LUQ-box rows
   now show non-degenerate uncertainty (MC-dropout NLL 1.07 vs 7.49 for its
   deterministic sibling on the covariate-shift axis). Artifact:
   `results/latest/stress_matrix_luqboxfix.json`; narrative in
   [`EXHAUSTIVE_REPORT.md`](EXHAUSTIVE_REPORT.md) §4.
5. Regression guards: `tests/test_stress_matrix.py` — 14 tests including
   `test_luqbox_mcdropout_has_real_dropout` and the 3-seed stability check
   `test_luqbox_rows_stable_across_seeds` (`d6fd67a`).

## 4. Next steps

None outstanding. The LUQ-box 3-seed check (session `bx25d3p3`, harness
`d6fd67a`) **passed**: `test_luqbox_rows_stable_across_seeds`,
`test_luqbox_mcdropout_has_real_dropout` and
`test_luqbox_methods_import_gated` all green with pinned LUQ-box v0.3.0 —
per-seed finite bounded NLL, non-collapsed MC std, MC-dropout means distinct
from the deterministic sibling on every seed; across seeds no RMSE blow-up
(max < 3× min) and CV < 0.5. Optional follow-ups (not scheduled):

* Full-scale (60-epoch) 3-seed sweep of the LUQ-box rows if the paper needs
  seed-spread error bars for the comparator.
* Repair the `/arc` clones' git remotes: the embedded `x-access-token` URL on
  `origin` (upstream astroai) prompts for credentials; headless refresh falls
  back to the operator-synced checkout (warn-only). Fetch from the local
  `/scratch/src` clones instead, or reset `origin` to the fork URL.

## 5. Useful commands

```bash
# Monitor sessions
canfar ps --all
canfar logs <session-id>       # or: curl https://ws-uv.canfar.net/skaha/v1/session/<id>?view=logs
canfar events <session-id>     # ImagePullBackOff & friends

# Relaunch the stress matrix (env cached on /arc; image must exist in the registry)
CANFAR_TIMEOUT=300 canfar create headless astroai/notebook:latest -n stress-<id> \
  -c 8 -m 16 -- bash /arc/projects/ots/torchregress-harness/src/torchregress-harness/scripts/skaha_stress_matrix.sh <RUN_ID>

# Sync the /arc harness checkout from the local clone (the /arc origin remote
# carries an expired embedded token; fetch over the filesystem instead)
git -C /arc/projects/ots/torchregress-harness/src/torchregress-harness fetch /scratch/src/torchregress-harness main
git -C /arc/projects/ots/torchregress-harness/src/torchregress-harness reset --hard FETCH_HEAD

# Publish a run's artifacts through the logs endpoint
bash scripts/skaha_publish_results.sh <RUN_ID>
```

## 6. Session log (scratch)

| Session | Purpose | Status |
|---|---|---|
| `nur3clql` | full paper benchmark (16c/32G) | Completed 2026-08; report regenerated 08-26 |
| `w03k5rek` | first full stress matrix | Completed; run `stress_20260903_011458` |
| `aljbqdg9` | LUQ-box fix re-run (`stress_luqboxfix20260903`) | Completed; 18/18 OK |
| `uq83j0wr` / `jyiijcnd` | first fix-run attempt | Deleted — image `openworker:latest` gone from registry |
| `htze4vvm` | seed check, stale checkout | Deleted — test not in fetched tree (exit 4) |
| `bx25d3p3` | LUQ-box 3-seed check (`d6fd67a`) | Completed — 3 passed (56 min) |
| earlier probes | image/API validation | Deleted |
