#!/usr/bin/env bash
# Stress-matrix diversity benchmark (torchregress-harness) on CANFAR Skaha
# headless, OFFLINE (pypi unreachable from Skaha nodes).
#
# Reuses the persistent /arc clones + pixi env prepared by
# skaha_paper_benchmark.sh, additionally installs Lightning-UQ-Box v0.3.0
# from a git clone (package copy + dist-info + optional-dep stubs), then runs
# tools/run_all.py --suites stress_matrix over all six stress axes.
#
# Artifacts: $OUT_DIR/run_all_stress_matrix.json (+ .md)
#
# Usage (from the container):
#   bash scripts/skaha_stress_matrix.sh [RUN_ID]
set -euo pipefail

RUN_ID="${1:-${RUN_ID:-$(date +%Y%m%d_%H%M%S)}}"
LUQBOX_REF="${LUQBOX_REF:-v0.3.0}"

ensure_writable() { [ -d "$1" ] && [ -w "$1" ] || mkdir -p "$1" 2>/dev/null && [ -w "$1" ]; }
export PIXI_HOME="$HOME/.pixi"
ensure_writable "$PIXI_HOME" || PIXI_HOME="/arc/home/sfabbro/.pixi"
export PIXI_HOME
export PIXI_CACHE_DIR="$HOME/.cache/pixi"
export UV_CACHE_DIR="$HOME/.cache/uv"
export XDG_CACHE_HOME="$HOME/.cache"
export PIXI_UV_CACHE_DIR="$UV_CACHE_DIR"
# CPU-only nodes + a lock carrying a CUDA pseudo-platform solve:
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.0}"
# torch on MIG-partitioned nodes:
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

ARC_ROOT="/arc/projects/ots/torchregress-harness"
SRC="$ARC_ROOT/src"
HARNESS_DIR="$SRC/torchregress-harness"
TR_DIR="$SRC/torchregress"
OUT_DIR="$ARC_ROOT/runs/stress_$RUN_ID"
mkdir -p "$OUT_DIR"

log "RUN_ID=$RUN_ID"
cd "$HARNESS_DIR"
log "harness @ $(git rev-parse HEAD) | torchregress @ $(git -C "$TR_DIR" rev-parse HEAD)"
cat > "$OUT_DIR/run_meta.txt" <<EOF
run_id: $RUN_ID
harness_sha: $(git rev-parse HEAD)
torchregress_sha: $(git -C "$TR_DIR" rev-parse HEAD)
luqbox_ref: $LUQBOX_REF
date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

export PATH="$HOME/.pixi/bin:$PATH"
PYBIN="$HARNESS_DIR/.pixi/envs/default/bin/python"
SITE="$HARNESS_DIR/.pixi/envs/default/lib/python3.13/site-packages"
[ -x "$PYBIN" ] || { log "FATAL: pixi env python missing at $PYBIN"; exit 1; }

# --- 0. Refresh clones to latest main (keeps /arc in sync with the repo) -----
# Fetch from the sfabbro fork explicitly: the clone's origin may point at the
# upstream astroai remote with a gh credential helper that does not exist in
# the Skaha image (expired x-access-token URLs abort the fetch). A refresh
# failure must not kill the session — the /arc clone is kept current by the
# operator, so fall back to whatever HEAD is checked out.
refresh() { # <dest> <repo>
  local dest="$1" repo="$2"
  rm -f "$dest/.git/shallow.lock" "$dest/.git/index.lock" 2>/dev/null || true
  if git -C "$dest" fetch --depth 1 "https://github.com/$repo" main \
    && git -C "$dest" reset --hard FETCH_HEAD; then
    log "refreshed $repo -> $(git -C "$dest" rev-parse --short HEAD)"
  else
    log "WARN: could not refresh $repo (offline?); using $dest at $(git -C "$dest" rev-parse --short HEAD)"
  fi
}
refresh "$HARNESS_DIR" sfabbro/torchregress-harness
refresh "$TR_DIR" sfabbro/torchregress

# --- 1. Wire local packages via .pth (idempotent) ----------------------------
"$PYBIN" - <<'PY'
import pathlib, sysconfig
sp = pathlib.Path(sysconfig.get_paths()["purelib"])
root = pathlib.Path("/arc/projects/ots/torchregress-harness/src")
lines = [str(root / "torchregress" / "src"), str(root / "torchregress-harness" / "src")]
(sp / "harness_local.pth").write_text("\n".join(lines) + "\n")
print("pth ok", flush=True)
PY

# --- 2. Install Lightning-UQ-Box offline (clone + copy + stubs) --------------
LUQ_DIR="$SRC/lightning-uq-box"
if [ ! -d "$LUQ_DIR" ]; then
  git clone --depth 1 --branch "$LUQBOX_REF" https://github.com/lightning-uq-box/lightning-uq-box.git "$LUQ_DIR"
fi
LUQ_SHA=$(git -C "$LUQ_DIR" rev-parse HEAD)
log "luqbox @ $LUQ_SHA"
rm -rf "$SITE/lightning_uq_box" "$SITE/lightning_uq_box-0.3.0.dist-info"
cp -r "$LUQ_DIR/lightning_uq_box" "$SITE/"
mkdir -p "$SITE/lightning_uq_box-0.3.0.dist-info"
printf 'Metadata-Version: 2.1\nName: lightning-uq-box\nVersion: 0.3.0\n' > "$SITE/lightning_uq_box-0.3.0.dist-info/METADATA"
touch "$SITE/lightning_uq_box-0.3.0.dist-info/RECORD"

stub_pkg() { # <import-name> <heredoc-file>
  mkdir -p "$SITE/$1"
  cp "$2" "$SITE/$1/__init__.py"
}
STUBS="$OUT_DIR/stubs"
mkdir -p "$STUBS"
cat > "$STUBS/automock.py" <<'PY'
"""Offline auto-mock stub: import-time symbols resolve; real use raises."""
from unittest.mock import MagicMock


def __getattr__(name):
    return MagicMock()
PY
for m in timm h5py kornia vbll laplace_torch curvlinops curvlinops_for_pytorch \
         vector_quantize_pytorch uncertainty_toolbox jsonargparse; do
  stub_pkg "$m" "$STUBS/automock.py"
done
# smp needs submodules that vae.py imports directly
mkdir -p "$SITE/segmentation_models_pytorch/base/modules"
for sub in segmentation_models_pytorch segmentation_models_pytorch/base segmentation_models_pytorch/base/modules; do
  cp "$STUBS/automock.py" "$SITE/$sub/__init__.py"
done
cat > "$SITE/omegaconf/__init__.py" <<'PY'
"""Offline stub: LUQ-box imports OmegaConf at import time; the harness never
exercises lightning-CLI config parsing."""


class OmegaConf:  # noqa: N801 - mirror upstream class name
    def __init__(self, *args, **kwargs):
        raise RuntimeError("omegaconf stub: real OmegaConf unavailable offline")

    @staticmethod
    def create(*args, **kwargs):
        raise RuntimeError("omegaconf stub: real OmegaConf unavailable offline")

    @staticmethod
    def load(*args, **kwargs):
        raise RuntimeError("omegaconf stub: real OmegaConf unavailable offline")


__all__ = ["OmegaConf"]
PY
cat > "$SITE/ema_pytorch/__init__.py" <<'PY'
"""Offline stub: only cards.py (not used by the harness) imports EMA."""


class EMA:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("ema_pytorch stub: real EMA unavailable offline")


__all__ = ["EMA"]
PY
cat > "$SITE/laplace/__init__.py" <<'PY'
"""Offline stub: laplace_model.py (not used by the harness) imports Laplace."""


class Laplace:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("laplace stub: real Laplace unavailable offline")


__all__ = ["Laplace"]
PY
mkdir -p "$SITE/torchvision/transforms" "$SITE/torchvision/datasets" "$SITE/torchvision/models"
cp "$STUBS/automock.py" "$SITE/torchvision/__init__.py"
for sub in transforms datasets models; do
  cp "$STUBS/automock.py" "$SITE/torchvision/$sub/__init__.py"
done
log "offline stubs installed"

# --- 3. Import pre-flight -----------------------------------------------------
if ! "$PYBIN" -c "import torch, torchregress, harness, mapie, lightning_uq_box"; then
  log "FATAL: pre-flight import check failed"
  exit 1
fi
log "pre-flight imports OK (incl. lightning_uq_box)"

# --- 4. Stress matrix (all six axes) ------------------------------------------
run_bench() {
  log "===== $* ====="
  "$@" || log "WARN: step failed (continuing): $*"
}

run_bench "$PYBIN" tools/run_all.py --suites stress_matrix \
  --output "$OUT_DIR/run_all_stress_matrix.json"

log "===== ARTIFACTS ====="
ls -la "$OUT_DIR" || true
log "===== DONE stress_$RUN_ID ====="
