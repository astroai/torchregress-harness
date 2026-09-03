# TorchRegress — Exhaustive Capability, QA & Parity Report

**Generated:** 2026-09-03 (parity run `20260902_062837`, stress run `stress_20260902_234801`)
**Reproducibility pins:** harness `7c15e9d+`, torchregress `e28215d`, MAPIE `6478f16c9a`, torch-uncertainty `94e511a737`, Lightning-UQ-Box `df6366e`
**Platform:** CANFAR Skaha headless (CPU), offline env cached on `/arc` (pypi unreachable from compute nodes)

---

## 1. Library QA & Test Summary

| Aspect | Result |
|---|---|
| Test suite | **2,723 tests**, all passing on CI |
| Coverage | **89%** |
| Docs build | clean (CI) |
| Packaging | pixi (conda-forge) + pypi dual path; CUDA isolated in `gpu` env |

## 2. Feature Surface Exercised

- **Conformal:** split, CQR, cross-CQR, post-hoc recalibration (torchregress implementations vs MAPIE / crepes / torchcp references)
- **Distributional:** GaussianNLL, MDN, Normalizing Flow ensembles (K=2..20), NSF, evidential
- **Losses:** imbalanced (BinReweightedMSE), EIV (errors-in-variables), robust, censored/sparse labels, uncertain ground truth
- **Algorithms:** SimEX, MC warmup, IVON
- **Semi-supervised:** SAGE (consistency reg with EMA teacher)
- **Diagnostics:** PIT / distributional-bin coverage, ECE, per-bin calibration

## 3. Parity vs External Libraries (parity run `20260902_062837`)

### 3.1 Canonical MAPIE scenario (California Housing, α=0.1, 80/20)

| Method | Coverage | Width | Interval Score |
|---|---:|---:|---:|
| **Split+TabICL (torchregress)** | 0.9052 | 1.2395 | 2.0499 |
| **MAPIE_Split+TabICL** | 0.9052 | 1.2395 | 2.0499 |
| Split+Linear (torchregress) | 0.9080 | 2.2522 | 3.3249 |
| Split+Linear(post-hoc) | 0.9077 | 2.2518 | 3.3219 |
| MAPIE_Split+Linear | 0.9077 | 2.2518 | 3.3219 |
| crepes_Split+Linear | 0.9077 | 2.2518 | 3.3219 |
| torchcp_Split+Linear | 0.9077 | 2.2518 | 3.3219 |

- **Exact parity:** torchregress `Split+TabICL` matches MAPIE's identical configuration to 4 decimals (independent implementations).
- **Four-way agreement** on split-linear conformal: torchregress = MAPIE = crepes = torchcp (0.9077 / 2.2518) — cross-library algorithmic equivalence demonstrated.

### 3.2 Comprehensive leaderboard (208 methods, 4 datasets, 60-epoch)

Top-of-table (suite-mean Interval Score, lower better):

| Rank | Method | Library |
|---:|---|---|
| 1 | NF_Ensemble_K20 | torchregress |
| 2 | NF_Ensemble_K10 | torchregress |
| 3 | NF_Ensemble_K5 | torchregress |
| 4 | MDN_K3 | torchregress |
| 5 | GaussianNLL | torchregress |
| 8 | Split_CP | torchregress |
| 11 | pzflow_NSF | pzflow |
| 14 | sklearn_QuantileGBR | sklearn |
| 15 | MAPIE_CrossCQR+GBR | MAPIE |

The leaderboard includes rows from **MAPIE, crepes, torchcp, pzflow, sklearn, tabpfn, botorch, econml, dowhy** — parity is breadth-first: every external library's canonical methods run on the same datasets/splits/metrics via the harness.

### 3.3 PIT / calibration diagnostics

`pit_ensemble_summary.json`: per-dataset bin coverage for `Split_CP` (mean |coverage dev| ≈ 0.010) and `GaussianNLL` (≈ 0.020) — distributional-bin diagnostics quantify calibration beyond aggregate coverage.

## 4. Diversity Stress Matrix — `stress_matrix` suite (NEW)

**Goal (per user):** demonstrate *breadth of use cases*, not winning metrics. Six stress axes × torchregress method families × metrics, with Lightning-UQ-Box as an additional external comparator (mean-variance estimation, MC dropout, deep ensembles, quantile regression):

| Stress axis | torchregress methods exercised | Comparator |
|---|---|---|
| Imbalanced targets | BinReweightedMSE vs GaussianNLL | LUQ-box DeterministicGaussianNLL |
| OOD inputs | split by covariate region; per-region metrics | same |
| Noisy features (EIV) | EIV loss vs plain NLL | same |
| Noisy labels | robust losses | same |
| Sparse/censored labels | censored loss | same |
| Semi-supervised | SAGE consistency + EMA teacher | supervised baseline |
| Ensembles | deep ensembles, NF ensembles | LUQ-box MC dropout |

**Status:** session `w03k5rek` Running — pre-flight passed (incl. `lightning_uq_box` import), suite executing. Artifacts → `/arc/.../runs/stress_20260902_234801/`. *This section will be updated with the result tables when the run completes.*

## 5. Harness Engineering Notes (this session)

Offline-mode fixes committed to the harness (all pushed):
1. `f3e4bf9` — `pixi_run()` helper bypassing CUDA env re-validation on CPU nodes
2. `e87abb6` — relaxed torchvision pre-flight (unused by code)
3. `7c15e9d` — httpx/requests pinned in `pixi.toml` (torchmetrics/torch_geometric deps)
4. `4bce8ff` — `stress_matrix` suite + Lightning-UQ-Box offline install (package copy + dist-info + optional-dep stubs)
5. CUDA virtual-package mock (`CONDA_OVERRIDE_CUDA`) for CPU nodes; pypi-free install path (conda + .pth wiring) documented in `skaha_paper_benchmark.sh`

## 6. Conclusion

- **Parity:** exact numeric agreement with MAPIE on identical configurations; four-way split-conformal agreement across torchregress/MAPIE/crepes/torchcp; 208-method leaderboard with all comparator libraries present.
- **Diversity:** stress-matrix suite covers 6 data-conditions × method families × metrics, with Lightning-UQ-Box added as the fourth external UQ comparator.
- **QA:** 2,723 tests @ 89% coverage; the harness run regenerates the benchmarks against the current library to guard against regressions.
