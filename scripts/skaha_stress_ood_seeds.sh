#!/usr/bin/env bash
# 3-seed x 60-epoch sweep of the stress_ood axis for paper error bars.
#
# Runs the FULL stress_ood method set (torchregress + LUQ-box comparators)
# at production scale (no --smoke, 60 epochs) for seeds {42, 43, 44}, one
# run_all invocation each, then merges the three runs into per-method
# mean/std summaries (paper error bars).
#
# Artifacts: $OUT_DIR/seed{42,43,44}.json + seed_errorbars.json + .md
#
# Usage (from the container):
#   bash scripts/skaha_stress_ood_seeds.sh [RUN_ID]
set -euo pipefail

RUN_ID="${1:-${RUN_ID:-$(date +%Y%m%d_%H%M%S)}}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-60}"

ensure_writable() { [ -d "$1" ] && [ -w "$1" ] || mkdir -p "$1" 2>/dev/null && [ -w "$1" ]; }
export PIXI_HOME="$HOME/.pixi"
ensure_writable "$PIXI_HOME" || PIXI_HOME="/arc/home/sfabbro/.pixi"
export PIXI_HOME
export PIXI_CACHE_DIR="$HOME/.cache/pixi"
export UV_CACHE_DIR="$HOME/.cache/uv"
export XDG_CACHE_HOME="$HOME/.cache"
export PIXI_UV_CACHE_DIR="$UV_CACHE_DIR"
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.0}"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

ARC_ROOT="/arc/projects/ots/torchregress-harness"
SRC="$ARC_ROOT/src"
HARNESS_DIR="$SRC/torchregress-harness"
TR_DIR="$SRC/torchregress"
OUT_DIR="$ARC_ROOT/runs/stressseeds_$RUN_ID"
mkdir -p "$OUT_DIR"

log "RUN_ID=$RUN_ID seeds=[$SEEDS] epochs=$EPOCHS"
cd "$HARNESS_DIR"
log "harness @ $(git rev-parse HEAD) | torchregress @ $(git -C "$TR_DIR" rev-parse HEAD)"
cat > "$OUT_DIR/run_meta.txt" <<EOF
run_id: $RUN_ID
harness_sha: $(git rev-parse HEAD)
torchregress_sha: $(git -C "$TR_DIR" rev-parse HEAD)
luqbox_ref: v0.3.0
seeds: $SEEDS
epochs: $EPOCHS
date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

export PATH="$HOME/.pixi/bin:$PATH"
PYBIN="$HARNESS_DIR/.pixi/envs/default/bin/python"
SITE="$HARNESS_DIR/.pixi/envs/default/lib/python3.13/site-packages"
[ -x "$PYBIN" ] || { log "FATAL: pixi env python missing at $PYBIN"; exit 1; }

# --- 0. Offline Lightning-UQ-Box install (same as skaha_stress_matrix.sh) ----
# The pixi env carries no lightning/luqbox; install package copy + stubs.
LUQ_DIR="$SRC/lightning-uq-box"
if [ ! -d "$LUQ_DIR" ]; then
  git clone --depth 1 --branch v0.3.0 \
    https://github.com/lightning-uq-box/lightning-uq-box.git "$LUQ_DIR" \
    || log "FATAL: cannot clone lightning-uq-box"
fi
LUQ_SHA=$(git -C "$LUQ_DIR" rev-parse HEAD)
log "luqbox @ $LUQ_SHA"
rm -rf "$SITE/lightning_uq_box" "$SITE/lightning_uq_box-0.3.0.dist-info" \
  || log "WARN: rm of old lightning_uq_box incomplete (Cavern lag?) — will copy over it"
mkdir -p "$SITE/lightning_uq_box"
cp -a "$LUQ_DIR/lightning_uq_box/." "$SITE/lightning_uq_box/" \
  || log "WARN: luqbox package copy reported errors (existing copy retained)"
mkdir -p "$SITE/lightning_uq_box-0.3.0.dist-info"
printf 'Metadata-Version: 2.1\nName: lightning-uq-box\nVersion: 0.3.0\n' \
  > "$SITE/lightning_uq_box-0.3.0.dist-info/METADATA" \
  || log "WARN: could not write luqbox dist-info METADATA"
touch "$SITE/lightning_uq_box-0.3.0.dist-info/RECORD" 2>/dev/null || true

stub_pkg() { mkdir -p "$SITE/$1"; cp "$2" "$SITE/$1/__init__.py"; }
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

# --- 1. Wire local packages via .pth (idempotent) ----------------------------
"$PYBIN" - <<'PY'
import pathlib, sysconfig
sp = pathlib.Path(sysconfig.get_paths()["purelib"])
root = pathlib.Path("/arc/projects/ots/torchregress-harness/src")
lines = [str(root / "torchregress" / "src"), str(root / "torchregress-harness" / "src")]
(sp / "harness_local.pth").write_text("\n".join(lines) + "\n")
print("pth ok", flush=True)
PY

# --- 2. Import pre-flight ------------------------------------------------------
if ! "$PYBIN" -c "import torch, torchregress, harness, lightning_uq_box, lightning"; then
  log "FATAL: pre-flight import check failed"
  exit 1
fi
log "pre-flight imports OK (incl. lightning_uq_box)"

# --- 3. One full stress_ood run per seed --------------------------------------
run_bench() {
  log "===== $* ====="
  "$@" || log "WARN: step failed (continuing): $*"
}
for seed in $SEEDS; do
  run_bench "$PYBIN" tools/run_all.py --suites stress_matrix \
    --datasets stress_ood --seed "$seed" --epochs "$EPOCHS" \
    --output "$OUT_DIR/seed${seed}.json"
done

# --- 4. Merge seeds into per-method mean/std error bars -----------------------
"$PYBIN" - "$OUT_DIR" $SEEDS <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

out_dir = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]]

per_seed: dict[int, dict[str, dict]] = {}
for seed in seeds:
    d = json.loads((out_dir / f"seed{seed}.json").read_text())
    rows = {r["Method"]: r for r in d["rows"]}
    bad = [m for m, r in rows.items() if r.get("RMSE") is None]
    if bad:
        print(f"WARN seed {seed}: rows without metrics: {bad}")
    per_seed[seed] = rows

methods = sorted({m for rows in per_seed.values() for m in rows})
summary = []
for m in methods:
    entry: dict = {"Method": m}
    for key in ("RMSE", "MAE", "R2", "NLL", "CRPS", "PIT_KS", "ECE"):
        vals = np.array(
            [per_seed[s][m][key] for s in seeds if m in per_seed[s] and per_seed[s][m].get(key) is not None],
            dtype=float,
        )
        if vals.size == 0:
            continue
        entry[f"{key}_mean"] = float(vals.mean())
        entry[f"{key}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        entry[f"{key}_n"] = int(vals.size)
    summary.append(entry)

agg = {
    "artifact": "seed_errorbars.json",
    "run_ids": [f"stressseeds (seed {s})" for s in seeds],
    "seeds": seeds,
    "epochs": 60,
    "dataset": "stress_ood",
    "n_methods": len(summary),
    "summary": summary,
}
(out_dir / "seed_errorbars.json").write_text(json.dumps(agg, indent=2) + "\n")

# Markdown table: mean +/- std for RMSE and NLL
lines = [
    "# stress_ood — 3-seed x 60-epoch error bars",
    "",
    f"Seeds: {seeds} · epochs: 60 · dataset: stress_ood (covariate shift along x0)",
    "",
    "| Method | RMSE | NLL | PIT_KS | ECE |",
    "|---|---:|---:|---:|---:|",
]
for e in summary:
    def cell(key: str) -> str:
        if f"{key}_mean" not in e:
            return "—"
        return f"{e[f'{key}_mean']:.4f} ± {e[f'{key}_std']:.4f}"

    lines.append(
        f"| {e['Method']} | {cell('RMSE')} | {cell('NLL')} | {cell('PIT_KS')} | {cell('ECE')} |"
    )
(out_dir / "seed_errorbars.md").write_text("\n".join(lines) + "\n")
print(f"wrote seed_errorbars.json/md for {len(summary)} methods")
PY

log "===== ARTIFACTS ====="
ls -la "$OUT_DIR" || true
log "===== DONE stressseeds_$RUN_ID ====="
