#!/usr/bin/env bash
# Image-parity benchmark for torchregress-harness on CANFAR Skaha headless.
#
# Image: astroai/base:latest (per operator instruction — use verbatim even
# though `canfar image ls` does not list it).
# Code delivery: /arc mirror preferred (no secrets in the job spec;
# the operator syncs it from the pushed SHA before launching).
#
# LAUNCH NOTE (2026-09-04): `canfar create headless` word-splits CMD args at
# runtime, so `bash -c "script with spaces"` arrives broken (bash executes
# only the first word). Also sfabbro/* repos are private → unauthenticated
# `git clone https://github.com/...` fails in the container. Working launch:
# a space-free `python3 -c exec(chr(..)+..)` bootstrap that clones from the
# /arc mirror path and execs this script, e.g. (generated, see bootstrap):
#   canfar create headless astroai/base:latest --cpu 4 --memory 16 \
#     --replicas 3 --name torchregress-img-smoke \
#     --env RUN_ID=img-tiny10-03 --env REPO_REF=main --env SMOKE=1 \
#     -- python3 -c "$ARG"
# where $ARG is the exec(chr()+...) payload for:
#   import subprocess
#   subprocess.run(["git","clone","-q",
#     "/arc/projects/ots/torchregress-harness/src/torchregress-harness",
#     "/scratch/bootstrap-harness"],check=True)
#   subprocess.run(["bash",
#     "/scratch/bootstrap-harness/scripts/skaha_image_parity.sh"],check=True)
# Sync the mirror first:
#   git -C /arc/projects/ots/torchregress-harness/src/torchregress-harness \
#     fetch https://github.com/sfabbro/torchregress-harness.git main --depth 20 \
#     && git -C ... reset --hard FETCH_HEAD   (same for ../torchregress)
#
# Usage (from the container):
#   REPO_REF=<branch-or-sha> bash scripts/skaha_image_parity.sh [RUN_ID] [--smoke]
#   bash scripts/skaha_image_parity.sh img-full-001 main
#
# Env:
#   REPO_REF   git ref to benchmark (required — the pushed PR head)
#   RUN_ID     unique run suffix (default: YYYYMMDD_HHMMSS)
#   OUT_ROOT   results root (default: /arc/projects/ots/torchregress-harness/runs)
#   GH_TOKEN   GitHub token for private clones
#   REPLICA_ID / REPLICA_COUNT  set by headless replicas — shard suite/dataset
#   MAPIE_PIN / TU_PIN / LUQBOX_REF  pins (defaults match EXTERNAL_PINS.md)
set -euo pipefail

RUN_ID="${1:-${RUN_ID:-$(date +%Y%m%d_%H%M%S)}}"
REPO_REF="${2:-${REPO_REF:-main}}"
SMOKE="${SMOKE:-0}"
[[ "${3:-}" == "--smoke" ]] && SMOKE=1
[[ "${1:-}" == "--smoke" ]] && SMOKE=1

MAPIE_PIN="${MAPIE_PIN:-6478f16c9a}"
TU_PIN="${TU_PIN:-94e511a737}"
LUQBOX_REF="${LUQBOX_REF:-v0.3.0}"

export PIXI_HOME="$HOME/.pixi"
export PIXI_CACHE_DIR="$HOME/.cache/pixi"
export UV_CACHE_DIR="$HOME/.cache/uv"
export XDG_CACHE_HOME="$HOME/.cache"
export PIXI_UV_CACHE_DIR="$UV_CACHE_DIR"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Plain output: the `canfar logs` renderer crashes on ANSI/box-drawing.
export NO_COLOR=1
export TERM=dumb
export PIXI_NO_PROGRESS=true

log() { echo "[$(date +%H:%M:%S)] $*"; }

OUT_ROOT="${OUT_ROOT:-/arc/projects/ots/torchregress-harness/runs}"
OUT_DIR="$OUT_ROOT/img_$RUN_ID"
mkdir -p "$OUT_DIR"
# Persist everything to /arc (survives session cleanup + broken renderers).
exec > >(tee "$OUT_DIR/run.log") 2>&1

WORK=/scratch/image-parity
mkdir -p "$WORK"
cd "$WORK"

ARC_SRC="/arc/projects/ots/torchregress-harness/src"

for repo in torchregress torchregress-harness; do
  if [ -d "$ARC_SRC/$repo/.git" ]; then
    # Preferred: operator-synced /arc mirror (no secrets in the job spec;
    # the operator syncs it from the pushed SHA before launching).
    log "cloning $repo from /arc mirror"
    rm -rf "$repo"
    git clone -q "$ARC_SRC/$repo" "$repo"
    git -C "$repo" rev-parse HEAD | tee "$OUT_DIR/${repo}_sha.txt"
    continue
  fi
  if [ -n "${GH_TOKEN:-}" ]; then
    GH_BASE="https://x-access-token:${GH_TOKEN}@github.com"
  else
    GH_BASE="https://github.com"
  fi
  if [ ! -d "$repo/.git" ]; then
    log "cloning $repo @ $REPO_REF"
    git clone --depth 50 "$GH_BASE/sfabbro/$repo.git" "$repo"
  fi
  log "fetching $repo ref $REPO_REF"
  git -C "$repo" fetch origin "$REPO_REF" --depth 20 || git -C "$repo" fetch origin main --depth 20
  git -C "$repo" checkout FETCH_HEAD
  git -C "$repo" rev-parse HEAD | tee "$OUT_DIR/${repo}_sha.txt"
done

cd "$WORK/torchregress-harness"
log "pixi install (5 attempts; transient mirror timeouts are common)"
for attempt in 1 2 3 4 5; do
  if pixi install 2>&1 | tee "$OUT_DIR/pixi_install_attempt${attempt}.log" | tail -n 3; then
    break
  fi
  log "pixi install attempt $attempt failed; retrying in 60s"
  sleep 60
  if [[ "$attempt" == 5 ]]; then
    log "FATAL: pixi install failed 5x — aborting (infra, not code)"
    exit 3
  fi
done

log "installing git pins (MAPIE/TU/LUQ-Box + image stack)"
pixi run pip install -q "git+https://github.com/scikit-learn-contrib/MAPIE.git@$MAPIE_PIN" || log "MAPIE pin failed (offline?)"
pixi run pip install -q "git+https://github.com/torch-uncertainty/torch-uncertainty.git@$TU_PIN" lightning || log "TU pin failed"
pixi run pip install -q "git+https://github.com/lightning-uq-box/lightning-uq-box.git@$LUQBOX_REF" timm torchvision h5py Pillow requests nibabel || log "LUQ/timm stack failed"
log "installing comparator/data deps (lifelines/crepes/survset/folktables/openpyxl)"
pixi run pip install -q lifelines crepes survset folktables openpyxl || log "comparator deps failed (offline?)"
log "installing likelihood deps (catboost/lightgbm/pyarrow for TLC parquet)"
pixi run pip install -q catboost lightgbm pyarrow || log "likelihood deps failed (offline?)"
log "installing h5py (GalaxiesML HDF5)"
pixi run pip install -q h5py || log "h5py failed (offline?)"
log "installing kaggle CLI (Kaggle dataset fetch, needs KAGGLE_USERNAME/KEY)"
pixi run pip install -q kaggle || log "kaggle failed (offline?)"
log "installing econml (causal comparator)"
pixi run pip install -q econml || log "econml failed (offline?)"
# ppi-python pulls numba (~slow); only sessions that run ppi jobs need it.
if [[ -z "${JOB_FILTER:-}" || "${JOB_FILTER}" == *ppi* ]]; then
  log "installing ppi-python (official PPI comparator)"
  pixi run pip install -q ppi-python || log "ppi-python failed (offline?)"
fi
# JAX + pzflow (photo-z flows); only flow-parity sessions need them.
if [[ -z "${JOB_FILTER:-}" || "${JOB_FILTER}" == *pzflow* || "${JOB_FILTER}" == *sbi* ]]; then
  log "installing JAX + pzflow (flow parity)"
  pixi run pip install -q "jax[cpu]" pzflow || log "jax/pzflow failed (offline?)"
fi
# SBI stack (sbibm tasks + sbi NPE + pyro); only sbi-parity sessions.
if [[ -z "${JOB_FILTER:-}" || "${JOB_FILTER}" == *sbi* ]]; then
  log "installing SBI stack (sbibm/sbi/pyro)"
  pixi run pip install -q --no-deps sbibm pyro-ppl future || log "sbibm failed (offline?)"
  pixi run pip install -q sbi || log "sbi failed (offline?)"
fi

# Replica sharding: split the suite×dataset matrix across headless replicas.
# Tiny variants (*_tiny) are CPU-smoke friendly; full-real sets are nightly.
ALL_JOBS=(
  "age_regression synthetic_image_regression"
  "age_regression utkface_age_tiny"
  "age_regression utkface_age"
  "age_regression boneage"
  "age_regression imdb_wiki"
  "depth_estimation synthetic_image_depth"
  "depth_estimation nyu_depth_v2_tiny"
  "depth_estimation nyu_depth_v2"
  "keypoint_regression synthetic_image_keypoint"
  "keypoint_regression biwi_head_pose_tiny"
  "keypoint_regression biwi_head_pose"
  "cell_counting bbbc_cellcount_tiny"
  "cell_counting bbbc_cellcount"
  "crowd_counting shanghai_crowd_tiny"
  "crowd_counting shanghai_crowd"
  "age_regression appa_age_tiny"
  "age_regression appa_age"
  "age_regression ixi_brainage_tiny"
  "age_regression ixi_brainage"
  "image_robust udacity_steering_tiny"
  "image_robust udacity_steering"
  "depth_estimation kitti_depth_tiny"
  "depth_estimation kitti_depth"
  "censored_imaging bbbc_censored_tiny"
  "censored_imaging bbbc_censored"
  "censored_imaging tcga_survival_tiny"
  "censored_imaging tcga_survival"
  "image_conformal utkface_age_tiny"
  "conformal diabetes"
  "conformal california_housing"
  "bayesian_linear diabetes"
  "bayesian_linear california_housing"
  "image_ensembles_uq utkface_age_tiny"
  "image_robust utkface_age_tiny"
  "image_calibration utkface_age_tiny"
  "robust_regression synthetic_outliers"
  "robust_regression california_housing"
  "tweedie synthetic_tweedie"
  "tweedie insurance_claims"
  "distributional_bins diabetes"
  "distributional_bins california_housing"
  "ensembles diabetes"
  "ensembles california_housing"
  "multivariate_intervals synthetic_multivariate"
  "multivariate_intervals energy_efficiency"
  "eiv synthetic_eiv"
  "censored synthetic_censored"
  "censored survset_hdfail"
  "ordinal synthetic_ordinal"
  "ordinal folktables_decile"
  "ppi diabetes"
  "ppi california_housing"
  "ppi folktables_income"
  "label_shift diabetes"
  "label_shift california_housing"
  "label_shift folktables_income"
  "classification_shift folktables_acs_income"
  "missing_data diabetes"
  "missing_data california_housing"
  "missing_data folktables_income"
  "pzflow_parity synthetic_cond_gaussian"
  "pzflow_parity synthetic_cond_nongaussian"
  "pzflow_parity pzflow_galaxies"
  "pzflow_parity energy_efficiency"
  "sbi_parity sbibm_twomoons"
  "sbi_parity sbibm_slcp"
  "sbi_parity sbibm_gaussian_linear"
  "sbi_parity sbibm_gaussian_mixture"
  "sbi_parity sbibm_bernoulli_glm"
  "photoz_imaging galaxiesml_cutouts"
  "image_robust galaxiesml_cutouts"
  "crowd_counting ucf_qnrf"
  "causal ihdp"
  "causal folktables_causal"
  "eiv photoz_eiv"
  "eiv galaxiesml_eiv"
  "eiv folktables_eiv"
  "likelihoods folktables_all"
  "likelihoods tlc_fare"
  "likelihoods insurance_claims"
  "streaming_tta diabetes"
  "streaming_tta california_housing"
  "streaming_tta folktables_income"
  "test_time_adaptation diabetes"
  "test_time_adaptation california_housing"
  "test_time_adaptation folktables_income"
)
RID_RAW="${REPLICA_ID:-1}"
RCNT="${REPLICA_COUNT:-1}"
# Platform replica IDs are 1-indexed (single sessions report 1/1);
# normalize to 0-indexed shards (a lone session takes shard 0 = all jobs).
RID=$(( (RID_RAW - 1) % RCNT ))
if [[ "$RID" -lt 0 ]]; then RID=0; fi
# JOB_FILTER (optional substring, e.g. "bbbc", "appa", "shanghai"): restrict
# to matching "suite dataset" pairs — one session per dataset policy.
JOB_FILTER="${JOB_FILTER:-}"
log "replica $RID/$RCNT, n_jobs=${#ALL_JOBS[@]}, filter='${JOB_FILTER}'"
JOBS=()
for i in "${!ALL_JOBS[@]}"; do
  if (( i % RCNT == RID )); then
    if [[ -z "$JOB_FILTER" || "${ALL_JOBS[$i]}" == *"$JOB_FILTER"* ]]; then
      JOBS+=("${ALL_JOBS[$i]}")
    fi
  fi
done
if [[ "${#JOBS[@]}" == 0 ]]; then
  log "FATAL: JOB_FILTER='${JOB_FILTER}' matched no jobs — aborting"
  exit 4
fi

EPOCHS=10
EXTRA_ARGS="--epochs $EPOCHS"
if [[ "$SMOKE" != 1 ]]; then EPOCHS=30; EXTRA_ARGS="--epochs $EPOCHS"; fi
# EPOCHS_OVERRIDE (optional): exact epoch budget, e.g. 60 for R2C-scale runs.
if [[ -n "${EPOCHS_OVERRIDE:-}" ]]; then
  EPOCHS="$EPOCHS_OVERRIDE"; EXTRA_ARGS="--epochs $EPOCHS"
fi

# shellcheck disable=SC2086
for job in "${JOBS[@]}"; do
  # Defensive: re-anchor the project dir per job (a prior job must never be
  # able to leave us in a cwd where pixi cannot find pixi.toml).
  cd "$WORK/torchregress-harness"
  # shellcheck disable=SC2086
  suite="${job%% *}"; dataset="${job##* }"
  log "=== $suite / $dataset ==="
  pixi run python tools/run_all.py --suites "$suite" --datasets "$dataset" \
    $EXTRA_ARGS --output "$OUT_DIR/${suite}_${dataset}.json" 2>&1 | tail -n 5 || log "job failed: $job"
done

log "run complete → $OUT_DIR"
ls -lh "$OUT_DIR"
