# 0.3.0 release run — harness status

Tracks plan Phase 1 (harness) for the torchregress 0.3.0 PyPI release.
Last updated: 2026-10-05.

## Done (CPU, no network data)

| Plan step | What changed | Commit |
|---|---|---|
| 1.1 | `TORCHREGRESS_HARNESS_STRICT_DATA=1` makes `load_dataset` raise instead of returning a synthetic proxy for a real dataset | `a429faf` |
| 1.1 | `tools/check_provenance.py` gate (in `pixi run ci`) and `tools/seed_spread.py` flaky-row tool | `a429faf` |
| 1.1 | Suites turn any per-method exception into a failed row (a refused TabICL download used to drop the whole dataset) | `22fe584` |
| 1.2 | Re-pinned MAPIE `3b84b8212d`, torch-uncertainty `3f82fe5d15`, LUQ-Box v0.4.0, TabPFN 9.1.0, TabICL 2.2.0, scikit-learn 1.9.1, sbi 0.27.0; new extras `sbi`, `causal`, `distributional`, `scoring` | `22fe584` |
| 1.3 | `sbi_parity`: 9 methods (torchregress NSF/MAF/MDN paired with sbi NPE NSF/MAF/MDN, pzflow, sbi FMPE/NPSE), sbi SBC/TARP cross-check, 100 SBC simulations, frozen sbibm tasks | `f22eecb` |
| 1.4 | New `orthogonal_inference` suite (vs DoubleML, EconML) | `1c93095` |
| 1.4 | New `constrained_outputs` suite (all 5 `constraints` exports) | `6c944bc` |
| 1.4 | `calibration` suite benchmarks the VarianceTemperatureScaler floor and clip options | `e3f7110` |
| 1.5 | `--seed` for `leaderboard_regression.py`; `BENCH_PROFILE=release` in `scripts/skaha_paper_benchmark.sh` | this commit |

## Findings so far

1. **Library bug, fixed in torchregress `da4b3d0`.** `orthogonal_partially_linear`
   used different cross-fitting splits for its two nuisances, biasing theta by
   −0.026 (≈7.7 SE) on the CCDDHNR-2018 design; DoubleML on the same basis
   −0.007. One shared split: −0.003.
2. **Silent synthetic rows in committed artifacts.** `results/latest/harness_conformal.json`
   and `_cuda_smoke.json` hold 86 rows where TabReD (cooking, delivery, maps,
   sberbank, weather) and `insurance_claims` ran on their synthetic fallbacks.
   Neither file feeds `PARITY_REPORT.md`; the published `leaderboard_results.json`
   is clean. The release run uses strict data mode, so this cannot recur.
3. **TabPFN 9 needs a Hugging Face token** (gated `Prior-Labs/tabpfn_3_5`);
   TabICL downloads its checkpoint from the Hub too.
4. **Per-level LightGBM quantile models cross** on 73% of test rows
   (`con_quantiles`); torchregress `NonCrossingSort` has none at equal pinball loss.
5. **`VarianceTemperatureScaler(clip=5)`** gives the best CRPS on
   `synthetic_outliers` but raises NLL (1.64 → 2.71): test outliers meet a
   variance fitted without them. Document the trade-off (Phase 2.5).
6. `tr_Orthogonal_RFF` (random Fourier nuisances, default bandwidth) covers
   only 59%: the feature map needs a bandwidth choice before it is shown as
   a recommended nuisance.

## Still to do

- [ ] Re-run `orthogonal_inference` after the torchregress fix (bias table above is pre-fix).
- [ ] Remaining coverage gaps from plan 1.4: `semi_supervised`, likelihood families
      (Johnson SU, sinh-arcsinh, skew-normal, GEV, beta), `RankNContrastLoss` in
      `age_regression`, level-set conformal in `multivariate_intervals`,
      `test_time` label-shift and transport, BatchEnsemble/`VariationalLinear`,
      metrics parity tests against `scoringrules`.
- [ ] `multivariate_intervals` fails noisily when the leaderboard hands it a
      univariate dataset; skip unsupported datasets instead.
- [ ] Full release run on CANFAR (below), then regenerate `PARITY_REPORT.md`.

## Launching the release run on CANFAR

Headless job on an image with git + curl (e.g. `astroai/notebook:latest`):

```bash
export GH_TOKEN=...          # read access to the private repos
export HF_TOKEN=...          # accepted the TabPFN 3.5 model terms
export BENCH_PROFILE=release SEEDS="0 1 2"
bash scripts/skaha_paper_benchmark.sh release_$(date +%Y%m%d)
# GPU node, image suites too:
IMAGE_SUITES=1 bash scripts/skaha_paper_benchmark.sh release_img_$(date +%Y%m%d)
```

Outputs land in `$ARC_ROOT/runs/<RUN_ID>/`: per-seed leaderboards and suite
JSONs, `seed_spread_*.md`, and the provenance gate's verdict in the log.
The script refreshes both repos from `sfabbro/*` `main`, so push the fork
first.
