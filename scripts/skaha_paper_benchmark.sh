#!/usr/bin/env bash
# Full paper benchmark for torchregress-harness, run inside a CANFAR Skaha
# headless session.
#
# Clones torchregress + torchregress-harness as siblings, builds the pixi env,
# installs the EXTERNAL_PINS.md git pins (MAPIE + torch-uncertainty), then
# re-runs every benchmark that feeds PARITY_REPORT.md:
#
#   1. run_canonical.py  --scenario mapie_reference      (external conformal)
#   2. run_canonical.py  --scenario crepes_reference      (external conformal)
#   3. leaderboard_conformal.py  (4 datasets, 60 epochs)  (conformal matrix)
#   4. leaderboard_regression.py (all suites x datasets)  (Tier-1 ranking)
#   5. benchmark_pit_ensemble.py (3 datasets, 60 epochs)  (distributional bins)
#   6. run_all.py  conformal + age_regression             (external rows)
#
# Artifacts land in $OUT_ROOT/$RUN_ID (JSON + Markdown + versions.txt +
# run_meta.txt).  Suites whose optional deps fail to install are skipped by
# the tools themselves (WARNING: unknown suite / skipped) -- the run still
# completes and reports what ran.
#
# Usage (from the container, after cloning the harness repo):
#   bash scripts/skaha_paper_benchmark.sh [RUN_ID]
#
# Optional env:
#   RUN_ID       unique run suffix (default: YYYYMMDD_HHMMSS)
#   OUT_ROOT     results root (default: $HOME/parity_runs)
#   SKIP_PIXI_INSTALL=1  reuse a prebuilt pixi env (container rebuilds only)
#
# Pins (keep in sync with EXTERNAL_PINS.md):
#   MAPIE          6478f16c9a   (branch master)
#   torch-uncertainty 94e511a737 (tag v0.13.0 / main HEAD)
set -euo pipefail

RUN_ID="${1:-${RUN_ID:-$(date +%Y%m%d_%H%M%S)}}"
OUT_ROOT="${OUT_ROOT:-$HOME/parity_runs}"
OUT_DIR="$OUT_ROOT/$RUN_ID"
MAPIE_PIN="${MAPIE_PIN:-6478f16c9a}"
TU_PIN="${TU_PIN:-94e511a737}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "$OUT_DIR"
log "RUN_ID=$RUN_ID  OUT_DIR=$OUT_DIR"
log "pins: mapie@$MAPIE_PIN torch-uncertainty@$TU_PIN"

# --- 0. Baseline tools -------------------------------------------------------
for tool in git curl bash; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    log "FATAL: missing '$tool' -- cannot bootstrap"
    exit 1
  fi
done

# --- 1. Clone repos as siblings (pixi path dep expects ../torchregress) ------
SRC="$HOME/src"
mkdir -p "$SRC"
if [ ! -d "$SRC/torchregress/.git" ]; then
  log "cloning astroai/torchregress"
  git clone --depth 1 https://github.com/astroai/torchregress.git "$SRC/torchregress"
fi
if [ ! -d "$SRC/torchregress-harness/.git" ]; then
  log "cloning astroai/torchregress-harness"
  git clone --depth 1 https://github.com/astroai/torchregress-harness.git "$SRC/torchregress-harness"
fi
cd "$SRC/torchregress-harness"
HARNESS_SHA=$(git rev-parse HEAD)
TORCHREGRESS_SHA=$(git -C "$SRC/torchregress" rev-parse HEAD)
log "harness @ $HARNESS_SHA | torchregress @ $TORCHREGRESS_SHA"
cat > "$OUT_DIR/run_meta.txt" <<EOF
run_id: $RUN_ID
harness_sha: $HARNESS_SHA
torchregress_sha: $TORCHREGRESS_SHA
mapie_pin: $MAPIE_PIN
torch_uncertainty_pin: $TU_PIN
date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

# --- 2. Pixi env --------------------------------------------------------------
export PATH="$HOME/.pixi/bin:$PATH"
if ! command -v pixi >/dev/null 2>&1; then
  log "installing pixi"
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi
if [ "${SKIP_PIXI_INSTALL:-}" != "1" ]; then
  log "pixi install (locked env; may take several minutes)"
  pixi install
fi

# --- 3. Pinned external libraries (EXTERNAL_PINS.md) --------------------------
log "installing MAPIE git pin"
pixi run pip install --quiet "git+https://github.com/scikit-learn-contrib/MAPIE.git@$MAPIE_PIN"
log "installing torch-uncertainty git pin + lightning"
pixi run pip install --quiet "git+https://github.com/torch-uncertainty/torch-uncertainty.git@$TU_PIN" lightning

# --- 4. Optional extras (best-effort; skips degrade gracefully) ---------------
log "installing optional extras [all] (best-effort)"
pixi run pip install --quiet -e ".[all]" || log "WARN: [all] extras install failed"
log "installing pzflow (best-effort)"
pixi run pip install --quiet pzflow || log "WARN: pzflow install failed"

log "recording installed versions"
pixi run python - <<'PY' > "$OUT_DIR/versions.txt" 2>&1 || true
import importlib.metadata as md
for pkg in ("mapie", "torch-uncertainty", "torch", "lightning", "torchregress",
            "torchregress-harness", "pzflow", "jax", "crepes", "torchcp",
            "lightgbm", "xgboost", "botorch", "gpytorch", "scikit-lego",
            "tabpfn", "tabicl", "econml", "dowhy", "pytorch-lightning"):
    try:
        d = md.distribution(pkg)
        url = d.read_text("direct_url.json") or ""
        print(f"{pkg}\t{d.version}\t{url}")
    except Exception:
        print(f"{pkg}\tUNAVAILABLE")
PY
cat "$OUT_DIR/versions.txt"

# --- 5. Benchmarks ------------------------------------------------------------
run_bench() {
  log "===== $* ====="
  "$@" || log "WARN: step failed (continuing): $*"
}

run_bench pixi run python tools/run_canonical.py --scenario mapie_reference \
  --output "$OUT_DIR/canonical_mapie.json" --report "$OUT_DIR/canonical_mapie.md"
run_bench pixi run python tools/run_canonical.py --scenario crepes_reference \
  --output "$OUT_DIR/canonical_crepes.json" --report "$OUT_DIR/canonical_crepes.md"
run_bench pixi run python leaderboard_conformal.py \
  --output "$OUT_DIR/leaderboard_conformal.json" --epochs 60
run_bench pixi run python tools/leaderboard_regression.py \
  --output "$OUT_DIR/leaderboard_results.json" --markdown "$OUT_DIR/leaderboard_results.md"
run_bench pixi run python tools/benchmark_pit_ensemble.py \
  --epochs 60 --output "$OUT_DIR/pit_ensemble.json"
run_bench pixi run python tools/run_all.py --suites conformal,age_regression \
  --datasets california_housing,synthetic_image_regression \
  --output "$OUT_DIR/run_all_external.json"

# --- 6. Summary ---------------------------------------------------------------
log "===== ARTIFACTS ====="
ls -la "$OUT_DIR" || true

log "===== EXTERNAL LIBRARY ROWS (LibraryVersion) ====="
pixi run python - "$OUT_DIR" <<'PY' || true
import json, sys, glob, os
root = sys.argv[1]
def rows_of(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("rows") or data.get("results") or []
    return data if isinstance(data, list) else []
seen = set()
for path in sorted(glob.glob(os.path.join(root, "*.json"))):
    for r in rows_of(path):
        m = r.get("Method", "")
        lib = r.get("Library", "")
        if not (("MAPIE" in m) or ("torch_uncertainty" in m) or "mapie" in lib.lower()):
            continue
        key = (path, m)
        if key in seen:
            continue
        seen.add(key)
        print(f"{os.path.basename(path)} | {m} | lib={lib} | "
              f"libver={r.get('LibraryVersion')} | cov={r.get('Coverage')} | "
              f"w={r.get('Width')} | notes={(r.get('Notes') or '')[:60]}")
PY

if command -v vcp >/dev/null 2>&1; then
  VOS="vos:sfabbro/torchregress-harness/runs/$RUN_ID"
  log "pushing results to VOS: $VOS"
  vcp --target "$VOS" "$OUT_DIR" && log "PUSHED to $VOS" || log "WARN: vcp push failed"
else
  log "vcp not present in image; artifacts remain in $OUT_DIR"
fi

log "===== DONE $RUN_ID ====="
