# torchregress-harness

[![CI](https://github.com/sfabbro/torchregress-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/sfabbro/torchregress-harness/actions/workflows/ci.yml)

**Benchmarking harness for [torchregress](https://github.com/sfabbro/torchregress).**

Compares every torchregress regression method against external software on known tabular and image datasets. Produces machine-readable JSON artifacts and human-readable Markdown reports.

**Related:** [torchregress](https://github.com/sfabbro/torchregress) (library)

## Quick Start

Requires sibling checkout of [`torchregress`](https://github.com/sfabbro/torchregress) at `../torchregress`.

### With Pixi (recommended)

```bash
pixi install
pixi run test
pixi run lint
pixi run typecheck
pixi run ci
```

### Run harness

```bash
pixi run python tools/run_all.py --smoke --output results/latest/harness_smoke.json
pixi run python tools/render_report.py results/latest/harness_smoke.json
pixi run python tools/run_all.py --suites conformal --datasets diabetes --output results/latest/conformal_diabetes.json
```

### Optional benchmark extras (pip)

```bash
pip install -e ".[all]"
# or a subset:
pip install -e ".[conformal,bayesian,foundation]"
```

See [OPTIONAL_DEPS.md](OPTIONAL_DEPS.md) for the coverage map.
