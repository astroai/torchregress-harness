#!/usr/bin/env bash
# Gates and summaries for a finished release run (scripts/skaha_release.sh).
#
# Runs on one small headless session after every group has finished:
#   - tools/check_provenance.py over every job artifact (strict-data gate)
#   - tools/seed_spread.py per group (flaky rows across seeds)
#   - a job ledger: which (suite, dataset, seed) produced rows and which failed
# Env: RUN_TAG, HARNESS_REF, LIB_REF, GH_TOKEN (as for skaha_release.sh).
set -uo pipefail
: "${RUN_TAG:?}" "${HARNESS_REF:?}" "${LIB_REF:?}"
ARC_PROJ="${ARC_PROJ:-/arc/projects/ots/torchregress-harness}"
RUN_DIR="${OUT_ROOT:-$ARC_PROJ/runs}/$RUN_TAG"
mkdir -p "$RUN_DIR/report"
exec > >(tee "$RUN_DIR/report/report.log") 2>&1
export NO_COLOR=1 TERM=dumb PIXI_NO_PROGRESS=true PYTHONNOUSERSITE=1
export PIXI_CACHE_DIR="$ARC_PROJ/cache/pixi" UV_CACHE_DIR="$ARC_PROJ/cache/uv"
export PIXI_UV_CACHE_DIR="$UV_CACHE_DIR"

WORK="/scratch/release-report-$RUN_TAG"
mkdir -p "$WORK" && cd "$WORK" || exit 2
for spec in "torchregress $LIB_REF" "torchregress-harness $HARNESS_REF"; do
  set -- $spec
  rm -rf "$1" && git init -q "$1"
  git -C "$1" -c http.extraheader="Authorization: Bearer ${GH_TOKEN:-}" fetch -q --depth 1 \
    "https://github.com/sfabbro/$1.git" "$2" && git -C "$1" checkout -q FETCH_HEAD || exit 2
done
cd torchregress-harness || exit 2
command -v pixi >/dev/null 2>&1 || { curl -fsSL https://pixi.sh/install.sh | bash; export PATH="$HOME/.pixi/bin:$PATH"; }
pixi install -e default >/dev/null 2>&1 || exit 3

pixi run python - "$RUN_DIR" <<'PY' | tee "$RUN_DIR/report/ledger.md"
import json, sys, collections
from pathlib import Path
run = Path(sys.argv[1])
print(f"# Release run {run.name}: job ledger\n")
print("| Group | Jobs | With rows | Failed rows | Missing |")
print("|---|---:|---:|---:|---:|")
for group in sorted(p for p in run.iterdir() if p.is_dir() and p.name != "report"):
    files = sorted(group.glob("*__*__seed*.json"))
    with_rows = failed = 0
    for f in files:
        try:
            rows = json.loads(f.read_text()).get("rows", [])
        except Exception:
            rows = []
        ok = [r for r in rows if not str(r.get("Notes", "")).startswith(("failed", "skipped"))]
        with_rows += bool(ok)
        failed += len(rows) - len(ok)
    print(f"| {group.name} | {len(files)} | {with_rows} | {failed} | see logs/ |")
PY

echo "== provenance gate =="
pixi run python tools/check_provenance.py "$RUN_DIR"/*/*.json | tee "$RUN_DIR/report/provenance.txt"
for group in "$RUN_DIR"/*/; do
  g=$(basename "$group"); [ "$g" = report ] && continue
  pixi run python - "$group" "$RUN_DIR/report" "$g" <<'PY'
import subprocess, sys, collections
from pathlib import Path
group, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
by_job = collections.defaultdict(list)
for f in sorted(group.glob("*__*__seed*.json")):
    by_job[f.name.rsplit("__seed", 1)[0]].append(str(f))
runs = [fs for fs in by_job.values() if len(fs) >= 2]
seeds = collections.defaultdict(list)
for fs in runs:
    for f in fs:
        seeds[f.rsplit("__seed", 1)[1]].append(f)
# Merge per seed so seed_spread compares the same rows across seeds.
import json
merged = []
for seed, fs in sorted(seeds.items()):
    rows = []
    for f in fs:
        rows += json.loads(Path(f).read_text()).get("rows", [])
    p = out / f"{name}_seed{seed.removesuffix('.json')}.json"
    p.write_text(json.dumps({"rows": rows}))
    merged.append(str(p))
if len(merged) >= 2:
    subprocess.run([sys.executable, "tools/seed_spread.py", *merged,
                    "--json-out", str(out / f"seed_spread_{name}.json"),
                    "--markdown-out", str(out / f"seed_spread_{name}.md")], check=False)
PY
done
echo "report written to $RUN_DIR/report"
