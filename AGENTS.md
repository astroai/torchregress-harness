# AGENTS.md

This file provides guidance to coding agents (Claude, Gemini, Codex) when working in this repository.

## Project Overview

**torchregress-harness** is a standalone benchmarking harness for [torchregress](https://github.com/sfabbro/torchregress). It compares torchregress regression methods against external software on known tabular and image datasets, producing structured JSON artifacts and Markdown reports.

## Sibling repositories

| Repo | Scope |
|------|-------|
| **torchregress** | Library (losses, conformal, ensembles, metrics) |
| **torchregress-harness** (this repo) | torchregress vs MAPIE, LightGBM, TabPFN, BoTorch, … |

Local/CI expect `../torchregress` as an editable, unpinned path dependency (via pixi).

## Development Commands

This project uses [pixi](https://pixi.sh). `pyproject.toml` is package metadata for pip consumers.

### Setup
```bash
pixi install
```

### Testing / quality
```bash
pixi run test
pixi run lint
pixi run typecheck
pixi run ci
```

### Run Harness
```bash
pixi run python tools/run_all.py --smoke --output results/latest/harness_smoke.json
pixi run python tools/run_all.py --suites conformal --datasets diabetes
pixi run python tools/render_report.py results/latest/harness_smoke.json
```

### Optional extras
Use `pip install -e ".[conformal]"` (etc.) for benchmark libraries not in the default pixi env. See `OPTIONAL_DEPS.md`.
