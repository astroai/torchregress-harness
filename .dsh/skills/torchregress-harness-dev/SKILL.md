---
name: torchregress-harness-dev
description: Develop torchregress-harness — parity benchmarking of torchregress regression methods against external software (MAPIE, torch-uncertainty, LightGBM, TabPFN, BoTorch), JSON/Markdown report generation, parity gates
metadata:
  project: torchregress-harness
  stack: python, pytorch, pixi
---

# torchregress-harness development

Standalone benchmarking harness comparing every torchregress regression
method against external software on known tabular and image datasets,
producing machine-readable JSON artifacts and human-readable Markdown
reports. See `AGENTS.md`, `PAPER_RUN_STATUS.md`, `PARITY_REPORT.md`.

## Contract

- Sibling checkout `../torchregress` (editable, unpinned pixi path dep).
- Optional benchmark libraries are pinned by commit: `EXTERNAL_PINS.md`
  (MAPIE `6478f16c9a`, torch-uncertainty `94e511a737`), install surface in
  `OPTIONAL_DEPS.md`.
- Results land under `results/latest/`; `PARITY_REPORT.md` is the
  human-readable parity gate, `leaderboard_results.json/md` the conformal
  leaderboard.

## Run

- Smoke (smallest verifier): `pixi run python tools/run_all.py --smoke --output results/latest/harness_smoke.json`
- Suite: `pixi run python tools/run_all.py --suites conformal --datasets diabetes --output results/latest/out.json`
- Report: `pixi run python tools/render_report.py results/latest/out.json`
- Gates: `pixi run test`, `pixi run lint`, `pixi run typecheck`, `pixi run ci`

## Parity discipline (statistics-principled)

- Report widths AND coverage together (coverage alone is meaningless — a
  trivially wide interval hits any coverage); lock in fixes with
  cross-suite checks (cross-CQR uses full train+cal data).
- Pin external-library commits in every artifact; regenerate every number
  that feeds `PARITY_REPORT.md` against the latest git builds.
- Honest comparisons: fixed seeds, per-dataset/suite config, no
  cherry-picked subsets; record the pinned commits in the report.
- Full paper/parity runs are `verify_full` — never auto-run in an agent
  turn without the explicit task; `--smoke` is the fast tier.

## Benchmarks

- `leaderboard_conformal.py` — conformal leaderboard across datasets.
- Image suite against torch-uncertainty v0.13 regression routines.
- Never regress committed baseline numbers without a stated statistical
  justification and a re-run of the affected suites.