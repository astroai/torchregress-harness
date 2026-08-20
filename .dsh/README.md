# torchregress-harness × DeepSeek Harness (dsh)

Repo-local [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
configuration. dsh auto-discovers skills from `.dsh/skills/` at the project
root (top project priority).

> dsh is a **developer preview** — expect breaking changes. Requires a DeepSeek
> API key, or any OpenAI-compatible endpoint via `DEEPSEEK_BASE_URL`.

## Install

```bash
npm install -g @deepseek-ai/dsh
```

> Do not use `npx -y @deepseek-ai/dsh ...`: on npm ≥ 10/11, npm's arg parser
> swallows launcher flags (`--profile`, `--patch`, `--dump-config` — only
> `--help`/`--version` pass through). A global install avoids the shim
> entirely. `--patch` paths are resolved relative to the CLI package dir, so
> always pass an **absolute** path (use `$PWD/.dsh/cordis.patch.yml`).

## Run

Web UI (needs localhost):

```bash
dsh web --patch "$PWD/.dsh/cordis.patch.yml"
```

Headless one-shot:

```bash
export DEEPSEEK_API_KEY=sk-...
dsh --profile headless --patch "$PWD/.dsh/cordis.patch.yml" \
  "Run the conformal+diabetes suite, render the report, and flag parity regressions vs PARITY_REPORT.md"
```

Sanity check before booting the app (prints the composed plugin tree):

```bash
dsh web --patch "$PWD/.dsh/cordis.patch.yml" --dump-config
```

Environment knobs:

- `DEEPSEEK_API_KEY` — required.
- `DEEPSEEK_BASE_URL` — OpenAI-compatible proxy (e.g. local vLLM server).
- `DSH_MODEL` — model override (default `deepseek-v4-flash`).
- `DSH_PERMISSION_MODE=danger-full-access` — for one-shot headless runs on a
  disposable CANFAR VM where `ask` approvals would stall (default is
  `workspace-write` + ask). Never use on a shared machine.
- `DSH_TOOLS_MODE=code` — Code Mode (tools exposed as a typed TypeScript SDK).

## Python SDK (scripted pipelines)

```python
from deepseek_harness import DeepSeekHarness

with DeepSeekHarness(
    model="deepseek-v4-flash",
    cwd="/Users/fabbros/src/torchregress-harness",
    session_root="/Users/fabbros/src/.dsh-sessions",
    env={"DEEPSEEK_API_KEY": "..."},
) as h:
    r = h.run(
        "Run tools/run_all.py --suites conformal --datasets diabetes --smoke, "
        "render the report, and summarize the parity table.",
        session_id="parity-2026-08-19",
    )
    print(r.final_response)
```

## Skills

- `science-core` — shared correctness / math / statistics / data-driven / science-impact discipline
- `stats-uq` — statistics-principled UQ: conformal coverage, proper scoring, calibration
- `torchregress-harness-dev` — harness domain: suites/datasets, run_all, reports, parity gates

## Notes

- Requires sibling checkout `../torchregress` (editable pixi path dep) — run
  dsh from a workspace that contains both.
- Optional benchmark extras (MAPIE, torch-uncertainty, …) are pinned in
  `EXTERNAL_PINS.md` / `OPTIONAL_DEPS.md`; install before parity runs.
- Never auto-run the full paper/parity suites from an agent turn without the
  explicit task; `--smoke` is the smallest verifier.
- Sessions are append-only logs (provenance); reuse a session id to continue a
  durable conversation with persistent bash state.
- `cordis.patch.yml`: 10-minute bash timeout + durable full-text session search.