# torchregress-harness

[![CI](https://github.com/astroai/torchregress-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/torchregress-harness/actions/workflows/ci.yml)

**Benchmarking harness for [torchregress](https://github.com/astroai/torchregress).**

Compares every torchregress regression method against external software on known tabular and image datasets. Produces machine-readable JSON artifacts and human-readable Markdown reports.

## Quick Start

Requires a sibling checkout of [`torchregress`](https://github.com/astroai/torchregress) at `../torchregress` (installed editable by pixi).

```bash
git clone git@github.com:astroai/torchregress.git
git clone git@github.com:astroai/torchregress-harness.git
cd torchregress-harness
```

### With Pixi (recommended)

The pixi workspace targets `linux-64` only (CI and CANFAR). Environments: `default` (CPU) and `gpu` (CUDA 12).

```bash
pixi install
pixi run test
pixi run lint
pixi run typecheck
pixi run ci          # lint + typecheck + test + provenance check
```

### Run harness

```bash
pixi run python tools/run_all.py --smoke --output results/latest/harness_smoke.json
pixi run python tools/render_report.py results/latest/harness_smoke.json
pixi run python tools/run_all.py --suites conformal --datasets diabetes --output results/latest/conformal_diabetes.json
pixi run python tools/run_canonical.py --scenario mapie_reference --smoke
```

Config-driven sweeps: `pixi run experiments-smoke` (also `experiments-tabular`, `experiments-image`, `experiments-ablations`, `experiments-all`; configs in `experiments/configs/`).

Likelihood-ratio/SBI and attenuation-free EIV coverage (self-contained synthetic tasks,
no optional `sbi`/`sbibm` dependency):

```bash
# NRE vs contrastive flow vs NPE vs sklearn NRE on exact-reference SBI tasks
pixi run python tools/run_all.py --suites likelihood_ratio \
  --datasets synthetic_sbi_gaussian_linear,synthetic_sbi_bimodal --epochs 40 \
  --output results/latest/likelihood_ratio.json
# Forward inversion, split-measurement and corrected-score EIV vs MSE/sklearn OLS
pixi run python tools/run_all.py --suites eiv_estimators \
  --datasets synthetic_eiv,synthetic_eiv_highnoise --epochs 30 \
  --output results/latest/eiv_estimators.json
# PRIME / ConR / continuous group DRO on a long-tailed target
pixi run python tools/run_all.py --suites imbalanced_representation \
  --datasets stress_imbalanced,california_housing --epochs 60 \
  --output results/latest/imbalanced_representation.json
# Conditional invertible network (cINN) parity via FrEIA on real tabular data
pixi run python tools/run_all.py --suites freia_parity \
  --datasets diabetes,california_housing --epochs 60 \
  --output results/latest/freia_parity.json
# Contrastive normalizing flow (CNF) reproduction (FAIR-Universe toy + sample)
pixi run python tools/run_all.py --suites cnf_parity \
  --datasets cnf_toy_gaussian,fair_universe_sample --epochs 60 \
  --output results/latest/cnf_parity.json
# Engression vs torchregress regression/UQ baselines
pixi run python tools/run_all.py --suites engression_parity \
  --datasets synthetic_hetero,stress_noisy_inputs --epochs 60 \
  --output results/latest/engression_parity.json
# Regression UQ under covariate/mean/noise shifts
pixi run python tools/run_all.py --suites uq_shift \
  --datasets stress_ood,synthetic_hetero --epochs 60 \
  --output results/latest/uq_shift.json
# LSST DESC PZ Data Challenge (task set 1, held-out official training split)
pixi run python tools/run_all.py --suites pz_challenge \
  --datasets pz_challenge_cardinal_10yr --epochs 60 \
  --output results/latest/pz_challenge.json
# Astronomical flow empirical-Bayes EIV benchmark (suite-generated design)
pixi run python tools/run_all.py --suites flow_eiv \
  --datasets synthetic_flow_eiv --epochs 60 \
  --output results/latest/flow_eiv.json
```

### Optional benchmark extras (pip)

```bash
pip install -e ".[all]"
# or a subset:
pip install -e ".[conformal,bayesian,foundation]"
```

See [OPTIONAL_DEPS.md](OPTIONAL_DEPS.md) for the coverage map and [EXTERNAL_PINS.md](EXTERNAL_PINS.md) for the git pins required for paper-grade runs.

## Documents

| File | Contents |
|------|----------|
| [OPTIONAL_DEPS.md](OPTIONAL_DEPS.md) | Optional extras and the suites they gate |
| [EXTERNAL_PINS.md](EXTERNAL_PINS.md) | External-library commits, install commands, provenance |
| [PARITY_REPORT.md](PARITY_REPORT.md) | Generated parity report vs external libraries |
| [leaderboard_results.md](leaderboard_results.md) | Generated leaderboard |
| [EXPORT_COVERAGE.md](EXPORT_COVERAGE.md) | Generated torchregress export coverage (`tools/export_coverage.py`) |
| [BENCHMARK_RESEARCH.md](BENCHMARK_RESEARCH.md) | Original-paper datasets and comparator-package research for the benchmark campaign |
| [BENCHMARK_GPU_RESULTS.md](BENCHMARK_GPU_RESULTS.md) | Measured CANFAR GPU results (sbi/pzflow parity, FrEIA cINN, real image/tabular data) |
| [RELEASE_0.3_STATUS.md](RELEASE_0.3_STATUS.md) | 0.3.0 release run status and CANFAR launch |
| [PAPER_RUN_STATUS.md](PAPER_RUN_STATUS.md) | Paper benchmark run log |
| [EXHAUSTIVE_REPORT.md](EXHAUSTIVE_REPORT.md) | Capability, QA and parity summary |
| [PROPOSALS.md](PROPOSALS.md) | Improvement proposals for torchregress |
| [datasets/README.md](datasets/README.md) | Dataset sources and caching |
