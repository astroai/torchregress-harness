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
| [RELEASE_0.3_STATUS.md](RELEASE_0.3_STATUS.md) | 0.3.0 release run status and CANFAR launch |
| [PAPER_RUN_STATUS.md](PAPER_RUN_STATUS.md) | Paper benchmark run log |
| [EXHAUSTIVE_REPORT.md](EXHAUSTIVE_REPORT.md) | Capability, QA and parity summary |
| [PROPOSALS.md](PROPOSALS.md) | Improvement proposals for torchregress |
| [datasets/README.md](datasets/README.md) | Dataset sources and caching |
