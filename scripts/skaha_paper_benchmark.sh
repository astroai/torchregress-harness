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
# The script is RESUMMABLE: existing clones / pixi env / artifacts are reused,
# so a session that dies can be relaunched and it continues where it stopped.
#
# Usage (from the container):
#   bash scripts/skaha_paper_benchmark.sh [RUN_ID]
#
# Optional env:
#   RUN_ID       unique run suffix (default: YYYYMMDD_HHMMSS)
#   OUT_ROOT     results root (default: persistent /arc when writable)
#   SKIP_PIXI_INSTALL=1  reuse a prebuilt pixi env
#
# Pins (keep in sync with EXTERNAL_PINS.md):
#   MAPIE           6478f16c9a    (branch master)
#   torch-uncertainty 94e511a737  (tag v0.13.0 / main HEAD)
set -euo pipefail

# Skaha containers run as a non-root user without write access to the
# system pixi/uv caches -- redirect every cache to $HOME (or /arc).
# NOTE: some images pre-set PIXI_HOME to a non-writable path (e.g.
# /usr/local/share/pixi); force a writable location instead of honoring it.
ensure_writable() { [ -d "$1" ] && [ -w "$1" ] || mkdir -p "$1" 2>/dev/null && [ -w "$1" ]; }
export PIXI_HOME="$HOME/.pixi"
ensure_writable "$PIXI_HOME" || PIXI_HOME="/arc/home/sfabbro/.pixi"
export PIXI_HOME
export PIXI_CACHE_DIR="$HOME/.cache/pixi"
export UV_CACHE_DIR="$HOME/.cache/uv"
export XDG_CACHE_HOME="$HOME/.cache"
ensure_writable "$UV_CACHE_DIR" || { export UV_CACHE_DIR="/arc/home/sfabbro/.cache/uv"; ensure_writable "$UV_CACHE_DIR"; }
export PIXI_UV_CACHE_DIR="$UV_CACHE_DIR"
echo "[caches] PIXI_HOME=$PIXI_HOME UV_CACHE_DIR=$UV_CACHE_DIR"

RUN_ID="${1:-${RUN_ID:-$(date +%Y%m%d_%H%M%S)}}"
MAPIE_PIN="${MAPIE_PIN:-6478f16c9a}"
TU_PIN="${TU_PIN:-94e511a737}"

# --- persistent root ---------------------------------------------------------
pick_root() {
  for cand in /arc/projects/ots/torchregress-harness /arc/home/sfabbro/torchregress-harness; do
    if mkdir -p "$cand/.wtest" 2>/dev/null; then
      rmdir "$cand/.wtest" 2>/dev/null || true
      echo "$cand"
      return 0
    fi
  done
  echo "$HOME"
}
ARC_ROOT="$(pick_root)"
SRC="$ARC_ROOT/src"
OUT_ROOT="${OUT_ROOT:-$ARC_ROOT/runs}"
OUT_DIR="$OUT_ROOT/$RUN_ID"

log() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "$OUT_DIR" "$SRC"
log "RUN_ID=$RUN_ID"
log "ARC_ROOT=$ARC_ROOT"
log "OUT_DIR=$OUT_DIR"
log "pins: mapie@$MAPIE_PIN torch-uncertainty@$TU_PIN"

# --- 0. Baseline tools -------------------------------------------------------
for tool in git curl bash; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    log "FATAL: missing '$tool' -- cannot bootstrap"
    exit 1
  fi
done

# --- 1. Clone repos as siblings (pixi path dep expects ../torchregress) ------
# Both astroai repos are PRIVATE: use GH_TOKEN when provided (e.g. from the
# Skaha session env).  Origins are scrubbed to the clean https URL afterwards
# so the token never persists on disk.
GH_BASE="https://github.com"
if [ -n "${GH_TOKEN:-}" ]; then
  GH_BASE="https://x-access-token:${GH_TOKEN}@github.com"
fi
clone_repo() { # <repo> <dest>
  git clone --depth 1 "$GH_BASE/$1" "$2"
  git -C "$2" remote set-url origin "https://github.com/$1"
}
refresh_repo() { # <dest> <repo>
  local url="https://github.com/$2"
  if [ -n "${GH_TOKEN:-}" ]; then
    git -C "$1" -c http.extraheader="Authorization: Bearer $GH_TOKEN" fetch --depth 1 "$url" main
  else
    git -C "$1" fetch --depth 1 "$url" main
  fi
  git -C "$1" reset --hard FETCH_HEAD
}

if [ ! -d "$SRC/torchregress/.git" ]; then
  log "cloning astroai/torchregress"
  clone_repo astroai/torchregress "$SRC/torchregress"
else
  log "torchregress clone exists; refreshing"
  refresh_repo "$SRC/torchregress" sfabbro/torchregress 2>/dev/null || true
fi
if [ ! -d "$SRC/torchregress-harness/.git" ]; then
  log "cloning astroai/torchregress-harness"
  clone_repo astroai/torchregress-harness "$SRC/torchregress-harness"
else
  log "harness clone exists; refreshing"
  refresh_repo "$SRC/torchregress-harness" sfabbro/torchregress-harness 2>/dev/null || true
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
# Mock the CUDA virtual package for every pixi invocation (install AND run):
# the committed lock carries a CUDA pseudo-platform solve, which CPU-only
# headless nodes reject with "unsupported-platform ... __cuda". Offsetting the
# mock with CONDA_OVERRIDE_CUDA lets CPU nodes install and run the CPU parts.
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.0}"
if ! command -v pixi >/dev/null 2>&1; then
  log "installing pixi"
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi
if [ "${SKIP_PIXI_INSTALL:-}" != "1" ]; then
  log "pixi install (locked env; may take several minutes)"
  # -e default: plain `pixi install` installs EVERY env incl. the CUDA `gpu`
  # env, which fails on CPU-only headless nodes. Only default is needed here.
  # CONDA_OVERRIDE_CUDA: the lock may carry a CUDA pseudo-platform solve from
  # GPU machines; mock the virtual package so CPU nodes can install.
  CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.0}" pixi install -e default
fi
# OFFLINE_MODE=1: Skaha headless nodes cannot reach pypi.org (egress blocked);
# reuse the cached pixi env on /arc, take torchvision from conda-forge, and
# wire torchregress/harness into the env with a .pth instead of pip -e.
OFFLINE_MODE="${OFFLINE_MODE:-0}"
# pixi_run: drop-in for `pixi run` that skips pixi's environment re-validation
# (`pixi run` re-solves the lock and fails on CPU-only nodes when the lock
# carries a CUDA pseudo-platform). In OFFLINE_MODE, invoke the cached env's
# binaries directly instead.
pixi_run() {
  if [ "$OFFLINE_MODE" = "1" ]; then
    local envbin="$SRC/torchregress-harness/.pixi/envs/default/bin"
    case "$1" in
      python) "$envbin/python" "${@:2}" ;;
      pip)    "$envbin/python" -m pip "${@:2}" ;;
      *)      "$envbin/$1" "${@:2}" ;;
    esac
  else
    pixi run "$@"
  fi
}
# The openworker image ships a read-only system venv on PATH; make sure the
# pixi env owns pip/setuptools so `pixi_run pip` installs into the writable
# pixi env, not /opt/openworker/venv.
# NOTE: `pixi add` re-solves every environment and fails on CPU-only nodes
# when the lock carries a CUDA pseudo-platform; the cached offline env already
# contains pip/setuptools/torchvision, so skip adds there.
if [ "$OFFLINE_MODE" != "1" ]; then
  pixi add pip setuptools wheel >/dev/null 2>&1 || log "WARN: pixi add pip/setuptools/wheel failed"
  # torch must pair with torchvision from the SAME source.  The conda-forge
  # torch in the pixi env is ABI-incompatible with a PyPI torchvision wheel
  # (torchvision::nms does not exist).  Install torchvision from conda-forge and
  # drop any pip-installed one so pip does not reinstall its own wheel during
  # the extras step.
  pixi add torchvision >/dev/null 2>&1 || log "WARN: pixi add torchvision failed"
fi
if [ "$OFFLINE_MODE" = "1" ]; then
  log "OFFLINE_MODE: wiring local packages via .pth (no pypi)"
  # Direct interpreter call: `pixi run` re-validates every environment and
  # fails on CPU nodes when the lock carries a CUDA pseudo-platform.
  PYBIN="$PWD/.pixi/envs/default/bin/python"
  [ -x "$PYBIN" ] || { log "FATAL: cached pixi env python missing at $PYBIN"; exit 1; }
  "$PYBIN" - <<'PY'
import pathlib, sysconfig
sp = pathlib.Path(sysconfig.get_paths()["purelib"])
root = pathlib.Path.cwd().parent  # $SRC (torchregress-harness's parent)
lines = [str(root / "torchregress" / "src"), str(root / "torchregress-harness" / "src")]
(sp / "harness_local.pth").write_text("\n".join(lines) + "\n")
print("wrote", sp / "harness_local.pth")
PY
fi

# --- 3. Pinned external libraries (EXTERNAL_PINS.md) --------------------------
if [ "$OFFLINE_MODE" = "1" ]; then
  log "OFFLINE_MODE: skipping pypi/git pip installs (already in cached env)"
else
  log "installing MAPIE git pin"
  pixi_run pip install --quiet "git+https://github.com/scikit-learn-contrib/MAPIE.git@$MAPIE_PIN"
  log "installing torch-uncertainty git pin + lightning"
  pixi_run pip install --quiet "git+https://github.com/torch-uncertainty/torch-uncertainty.git@$TU_PIN" lightning
fi

# --- 4. Optional extras (best-effort; skips degrade gracefully) ---------------
if [ "$OFFLINE_MODE" = "1" ]; then
  log "OFFLINE_MODE: skipping [all] extras and pzflow pip installs"
else
  log "installing optional extras [all] (best-effort)"
  pixi_run pip install --quiet -e ".[all]" || log "WARN: [all] extras install failed"
  log "installing pzflow (best-effort)"
  pixi_run pip install --quiet pzflow || log "WARN: pzflow install failed"
fi

log "recording installed versions"
pixi_run python - <<'PY' > "$OUT_DIR/versions.txt" 2>&1 || true
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

# --- 4b. Pre-flight import check (fail fast instead of 7 failing steps) ------
if ! pixi_run python -c "import torch, torchvision, torchregress, harness, mapie"; then
  log "FATAL: pre-flight import check failed (torch/torchvision ABI mismatch?) -- aborting"
  exit 1
fi
log "pre-flight imports OK"

# --- 5. Benchmarks ------------------------------------------------------------
run_bench() {
  log "===== $* ====="
  "$@" || log "WARN: step failed (continuing): $*"
}

run_bench pixi_run python tools/run_canonical.py --scenario mapie_reference \
  --output "$OUT_DIR/canonical_mapie.json" --report "$OUT_DIR/canonical_mapie.md"
run_bench pixi_run python tools/run_canonical.py --scenario crepes_reference \
  --output "$OUT_DIR/canonical_crepes.json" --report "$OUT_DIR/canonical_crepes.md"
run_bench pixi_run python leaderboard_conformal.py \
  --output "$OUT_DIR/leaderboard_conformal.json" --epochs 60
run_bench pixi_run python tools/leaderboard_regression.py \
  --output "$OUT_DIR/leaderboard_results.json" --markdown "$OUT_DIR/leaderboard_results.md"
run_bench pixi_run python tools/benchmark_pit_ensemble.py \
  --epochs 60 --output "$OUT_DIR/pit_ensemble.json"
run_bench pixi_run python tools/run_all.py --suites conformal \
  --datasets california_housing --output "$OUT_DIR/run_all_conformal.json"
run_bench pixi_run python tools/run_all.py --suites age_regression \
  --datasets synthetic_image_regression --output "$OUT_DIR/run_all_image.json"

# --- 6. Summary ---------------------------------------------------------------
log "===== ARTIFACTS ====="
ls -la "$OUT_DIR" || true

log "===== EXTERNAL LIBRARY ROWS (LibraryVersion) ====="
pixi_run python - "$OUT_DIR" <<'PY' || true
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

log "===== DONE $RUN_ID ====="
