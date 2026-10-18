#!/usr/bin/env bash
# torchregress 0.3.0 release benchmark: one headless replica of one job group.
#
# Launched by scripts/canfar_launch_release.sh (on the operator's machine) on
# CANFAR Skaha headless sessions, image astroai/base:latest.  Each replica
# clones the pushed fork SHAs, builds the pixi env (package caches on /scratch, HF weights on /arc), installs
# the pinned comparators, then runs its shard of the group's (suite, dataset)
# jobs for every seed.  Real datasets run in strict mode: a missing mirror
# fails that job instead of silently benchmarking a synthetic stand-in.
#
# Env (set by the launcher):
#   RUN_TAG        shared run name, e.g. release_20261005_1600 (required)
#   GROUP          tabular | sbi | fm | image (required)
#   HARNESS_REF    torchregress-harness commit to run (required)
#   LIB_REF        torchregress commit to run (required)
#   GH_TOKEN       read token for the private sfabbro/torchregress-harness fork
#   HF_TOKEN       optional Hugging Face token (not needed for TabICL / sdm / TabPFN weights)
#   TABPFN_TOKEN   optional Prior Labs API key (account that accepted the TabPFN licence);
#                  when set, TabPFN is installed and benchmarked
#   SEEDS          comma-separated, default "0,1,2"
#   EPOCHS         default 60
#   REPLICA_ID / REPLICA_COUNT   set by Skaha for --replicas sessions
#   SHARD_ID / SHARD_COUNT       override them (one session redoing one shard)
#   RERUN_SUITES   comma-separated suites whose existing artifacts are overwritten
#   JOB_TIMEOUT    wall-clock limit per (suite, dataset, seed) job, default 6h
#   JOB_FILTER     extended regex on "suite dataset": run only matching jobs
#                  (JOB_FILTER_HEX: the same, hex-encoded, as the launcher sends it)
#   OUT_ROOT       default /arc/projects/ots/torchregress-harness/runs
set -uo pipefail

: "${RUN_TAG:?RUN_TAG is required}" "${GROUP:?GROUP is required}"
: "${HARNESS_REF:?HARNESS_REF is required}" "${LIB_REF:?LIB_REF is required}"
SEEDS="${SEEDS:-0,1,2}"
EPOCHS="${EPOCHS:-60}"
ARC_PROJ="${ARC_PROJ:-/arc/projects/ots/torchregress-harness}"
OUT_ROOT="${OUT_ROOT:-$ARC_PROJ/runs}"
OUT_DIR="$OUT_ROOT/$RUN_TAG/$GROUP"

# Pins (keep in sync with EXTERNAL_PINS.md).
MAPIE_PIN="${MAPIE_PIN:-929740af07f53a9b1856c8d5b37356f75b5096db}"
TU_PIN="${TU_PIN:-3f82fe5d15a7bf877a821731baadef1e4731c31d}"
LUQBOX_REF="${LUQBOX_REF:-d1b7f6a49995c70eeef9fc878f2967f83273c49d}"  # main (was v0.4.0)
SBI_PIN="${SBI_PIN:-29af13750100b61ad528c3f8a49daa0ef8c96881}"
# Tabular foundation models track their latest code: git default branch
# (override with a tag / SHA), falling back to the newest PyPI release when the
# branch does not build.  The installed commit lands in versions_replica_*.txt.
PZFLOW_PIN="${PZFLOW_PIN:-bf9c70d6d1c2ea5d9678868d3d77f1bd203d33df}"  # git main
SDM_REF="${SDM_REF:-main}"
TABICL_REF="${TABICL_REF:-main}"
TABPFN_REF="${TABPFN_REF:-main}"

# SHARD_ID/SHARD_COUNT (set by `canfar_launch_release.sh fill`) override Skaha's
# REPLICA_ID/REPLICA_COUNT so a single session can redo one shard of a group.
RID_RAW="${SHARD_ID:-${REPLICA_ID:-1}}"
RCNT="${SHARD_COUNT:-${REPLICA_COUNT:-1}}"
RID=$(( (RID_RAW - 1) % RCNT ))
(( RID < 0 )) && RID=0

mkdir -p "$OUT_DIR/logs"
exec > >(tee "$OUT_DIR/logs/replica_${RID}_of_${RCNT}.log") 2>&1

export NO_COLOR=1 TERM=dumb PIXI_NO_PROGRESS=true
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIXI_HOME="$HOME/.pixi"
# Always override: astroai/base presets PIXI_CACHE_DIR to a read-only
# /usr/local/share/pixi/cache, so `${VAR:-default}` kept the image value and
# every pixi install failed with "Permission denied" creating uv-cache.
# Package caches live on per-session local /scratch (fast, same filesystem as
# the env, no 25-writer contention on CephFS); only HF weights are shared on /arc.
CACHE_ROOT="/scratch/cache-$RUN_TAG-$GROUP-$RID_RAW"
export PIXI_CACHE_DIR="$CACHE_ROOT/pixi"
export RATTLER_CACHE_DIR="$PIXI_CACHE_DIR"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export PIXI_UV_CACHE_DIR="$UV_CACHE_DIR"
if ! mkdir -p "$PIXI_CACHE_DIR" "$UV_CACHE_DIR" 2>/dev/null || [ ! -w "$PIXI_CACHE_DIR" ]; then
  export PIXI_CACHE_DIR="$HOME/.cache/pixi" RATTLER_CACHE_DIR="$HOME/.cache/pixi"
  export UV_CACHE_DIR="$HOME/.cache/uv" PIXI_UV_CACHE_DIR="$HOME/.cache/uv"
fi
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME="$ARC_PROJ/cache/huggingface"  # override image presets too
export TORCHREGRESS_HARNESS_STRICT_DATA=1
export DENSITY_PARITY_DUMP_DIR="$OUT_DIR/density_samples"
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.0}"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$PIXI_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME"

log() { echo "[$(date +%H:%M:%S)] [$GROUP $RID/$RCNT] $*"; }
log "run=$RUN_TAG harness=$HARNESS_REF lib=$LIB_REF seeds=[$SEEDS] epochs=$EPOCHS"

# --- 1. code at the pushed SHAs -------------------------------------------------
WORK="/scratch/release-$RUN_TAG-$GROUP-$RID"
mkdir -p "$WORK" && cd "$WORK" || exit 2
fetch_repo() { # <repo> <ref>
  local url="https://github.com/sfabbro/$1.git"
  rm -rf "$1" && git init -q "$1"
  if [ -n "${GH_TOKEN:-}" ]; then
    git -C "$1" -c http.extraheader="Authorization: Basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 | tr -d '\n')" fetch -q --depth 1 "$url" "$2"
  else
    git -C "$1" fetch -q --depth 1 "$url" "$2"
  fi && git -C "$1" checkout -q FETCH_HEAD
}
for attempt in 1 2 3; do
  fetch_repo torchregress "$LIB_REF" && fetch_repo torchregress-harness "$HARNESS_REF" && break
  log "git fetch attempt $attempt failed; retrying in 30s"; sleep 30
  [ "$attempt" = 3 ] && { log "FATAL: cannot fetch code"; exit 2; }
done
cd "$WORK/torchregress-harness" || exit 2

# --- 2. environment -------------------------------------------------------------
# GPU groups use the pixi `gpu` env (CUDA pytorch); the default env is CPU-only.
case "$GROUP" in fm|image|tabular_big) PENV=gpu ;; *) PENV=default ;; esac
[ "${USE_GPU:-0}" = 1 ] && PENV=gpu  # ALL_GPU=1 launches: CUDA env for every group
if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi
for attempt in 1 2 3 4 5; do
  pixi install -e "$PENV" 2>&1 | tail -n 3 && break
  log "pixi install attempt $attempt failed; retrying in 60s"; sleep 60
  [ "$attempt" = 5 ] && { log "FATAL: pixi install failed 5x (infra)"; exit 3; }
done
prun() { pixi run -e "$PENV" "$@"; }
# The conda-forge env ships without pip; bootstrap it or every comparator
# install silently fails (run #3 benchmarked torchregress alone).
if ! prun python -m pip --version >/dev/null 2>&1; then
  prun python -m ensurepip --upgrade >/dev/null 2>&1 \
    || { curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$WORK/get-pip.py" && prun python "$WORK/get-pip.py" -q; }
  prun python -m pip --version || { log "FATAL: cannot bootstrap pip in env $PENV"; exit 3; }
fi
pip_install() { prun python -m pip install -q "$@" || log "WARN: pip install failed: $*"; }
# Freeze the conda-forge torch stack so no comparator's pip deps replace it with
# a PyPI wheel built for another CUDA.
prun python -m pip list --format=freeze 2>/dev/null \
  | grep -iE '^(torch|torchvision|numpy|scipy)==' > "$WORK/constraints.txt"
export PIP_CONSTRAINT="$WORK/constraints.txt"
# torchcp needs torchsort, a CUDA extension: build it against the env's torch
# (build isolation pulls the newest PyPI torch, CUDA 13 vs the node's 12.9).
pip_install setuptools wheel ninja
# The gpu env carries nvcc + cudart/cccl headers (pixi.toml) so torchsort's
# CUDA kernel builds against the env's own torch and CUDA 12.9.
if [ "$PENV" = gpu ]; then
  ENVP="$WORK/torchregress-harness/.pixi/envs/gpu"
  CUDA_HOME="$ENVP" CPATH="$ENVP/targets/x86_64-linux/include:${CPATH:-}" pip_install --no-build-isolation torchsort
else
  pip_install --no-build-isolation torchsort
fi

log "installing pinned comparators"
pip_install "git+https://github.com/scikit-learn-contrib/MAPIE.git@$MAPIE_PIN"
pip_install "git+https://github.com/torch-uncertainty/torch-uncertainty.git@$TU_PIN" lightning
pip_install "git+https://github.com/lightning-uq-box/lightning-uq-box.git@$LUQBOX_REF"
# One extra per call: a single unbuildable package must not drop every extra.
for extra in conformal bayesian tweedie boosting causal distributional scoring; do
  pip_install -e ".[$extra]"
done
pip_install lifelines survset folktables openpyxl h5py pyarrow kaggle ppi-python catboost
case "$GROUP" in
  sbi)
    pip_install "git+https://github.com/sbi-dev/sbi.git@$SBI_PIN" pyro-ppl future
    pip_install --no-deps "sbibm==1.1.0"
    pip_install "jax[cpu]" "git+https://github.com/jfcrenshaw/pzflow.git@$PZFLOW_PIN"
    ;;
  fm)
    latest_install() { # <pypi name> <git url> <ref>
      prun python -m pip install -q --upgrade "git+$2@$3" \
        || { log "WARN: $1 @ $3 failed to build; using the newest PyPI release"; pip_install --upgrade "$1"; }
    }
    latest_install tabicl https://github.com/soda-inria/tabicl.git "$TABICL_REF"
    latest_install structured-data-models https://github.com/NVIDIA/structured-data-models.git "$SDM_REF"
    if [ -n "${TABPFN_TOKEN:-}" ]; then
      export TORCHREGRESS_HARNESS_WITH_TABPFN=1
    fi
    if [ "${TORCHREGRESS_HARNESS_WITH_TABPFN:-0}" = 1 ]; then
      # Licence check runs against the Prior Labs API with TABPFN_TOKEN; never a
      # browser in a headless session.  Weights are cached on /arc once.
      export TABPFN_NO_BROWSER=1 TABPFN_MODEL_CACHE_DIR="$ARC_PROJ/models/tabpfn"
      mkdir -p "$TABPFN_MODEL_CACHE_DIR"
      latest_install tabpfn https://github.com/PriorLabs/TabPFN.git "$TABPFN_REF"
    fi
    ;;
  image)
    pip_install timm torchvision Pillow requests nibabel
    pip_install "git+https://github.com/sbi-dev/sbi.git@$SBI_PIN"  # image_density_parity
    ;;
  tabular|tabular_big)
    pip_install "jax[cpu]" "git+https://github.com/jfcrenshaw/pzflow.git@$PZFLOW_PIN"
    pip_install "git+https://github.com/sbi-dev/sbi.git@$SBI_PIN"  # density_parity
    pip_install "FrEIA>=0.2"  # freia_parity (cINN)
    ;;
esac

case "$GROUP" in
  sbi) REQ="mapie sbi sbibm" ;; fm) REQ="mapie tabicl" ;;
  image) REQ="mapie timm nibabel crepes torchcp lightgbm xgboost sbi" ;; *) REQ="mapie torch_uncertainty lifelines crepes torchcp sbi pzflow" ;;
esac
for mod in $REQ; do
  err=$(prun python -c "import $mod" 2>&1) || { echo "$err" | tail -n 5; log "FATAL: required comparator '$mod' not importable"; exit 4; }
done
log "comparators OK: $REQ"

prun python - > "$OUT_DIR/versions_replica_${RID}.txt" 2>&1 <<'PY' || true
import importlib.metadata as md, platform, torch
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
for pkg in ("torchregress", "torchregress-harness", "mapie", "torch-uncertainty", "lightning-uq-box",
            "sbi", "sbibm", "structured-data-models", "tabpfn", "tabicl", "doubleml", "econml",
            "ngboost", "laplace-torch", "scoringrules", "scikit-learn", "lightgbm", "xgboost",
            "catboost", "crepes", "torchcp", "botorch", "pzflow", "numpy", "scipy"):
    try:
        d = md.distribution(pkg)
        print(f"{pkg}\t{d.version}\t{d.read_text('direct_url.json') or ''}")
    except md.PackageNotFoundError:
        print(f"{pkg}\tNOT INSTALLED")
PY

if [ "$GROUP" = "fm" ]; then
  log "prefetching foundation-model weights into $HF_HOME"
  prun python - <<'PY' || log "WARN: weight prefetch failed (rows will fail individually)"
import os
import numpy as np
X, y = np.random.default_rng(0).normal(size=(64, 4)), np.arange(64.0)
from tabicl import TabICLRegressor
TabICLRegressor().fit(X, y).predict(X[:2])
if os.environ.get("TORCHREGRESS_HARNESS_WITH_TABPFN") == "1":
    from tabpfn import TabPFNRegressor
    TabPFNRegressor().fit(X, y).predict(X[:2])
print("weights cached")
PY
fi
if [ "$GROUP" = "sbi" ]; then
  log "freezing sbibm tasks"
  prun python tools/freeze_sbibm.py || log "WARN: sbibm freeze failed"
fi

# --- 3. this replica's shard ----------------------------------------------------
mapfile -t ALL_JOBS < <(prun python tools/release_jobs.py "$GROUP")
# JOB_FILTER: extended regex on "suite dataset"; only matching jobs are sharded
# (targeted reruns that must not overlap shards already running).
# The launcher sends it hex-encoded (JOB_FILTER_HEX) so any regex survives Skaha.
if [ -n "${JOB_FILTER_HEX:-}" ]; then
  JOB_FILTER=$(printf '%s' "$JOB_FILTER_HEX" | python3 -c 'import sys; print(bytes.fromhex(sys.stdin.read().strip()).decode(), end="")')
fi
if [ -n "${JOB_FILTER:-}" ]; then
  mapfile -t ALL_JOBS < <(printf '%s\n' "${ALL_JOBS[@]}" | grep -E -- "$JOB_FILTER")
  log "JOB_FILTER '$JOB_FILTER': ${#ALL_JOBS[@]} jobs"
fi
JOBS=()
for i in "${!ALL_JOBS[@]}"; do
  (( i % RCNT == RID )) && JOBS+=("${ALL_JOBS[$i]}")
done
log "${#JOBS[@]} of ${#ALL_JOBS[@]} jobs on this replica"

n_ok=0; n_fail=0
for seed in ${SEEDS//,/ }; do
  for job in "${JOBS[@]}"; do
    suite="${job%% *}"; dataset="${job##* }"
    out="$OUT_DIR/${suite}__${dataset}__seed${seed}.json"
    if [ -s "$out" ]; then
      case ",${RERUN_SUITES:-}," in
        *",$suite,"*) log "rerun (RERUN_SUITES): $out" ;;
        *) log "skip (exists): $out"; continue ;;
      esac
    fi
    log "=== $suite / $dataset / seed $seed ==="
    # JOB_TIMEOUT bounds one (suite, dataset, seed) job: run #5's eiv/folktables_eiv
    # (1.5M rows) hung for 2.5 days without output and blocked its whole shard.
    if timeout --kill-after=5m "${JOB_TIMEOUT:-6h}" pixi run -e "$PENV" python tools/run_all.py --suites "$suite" --datasets "$dataset" \
        --seed "$seed" --epochs "$EPOCHS" --output "$out" 2>&1 | tail -n 4; then
      n_ok=$((n_ok + 1))
      # GPU slices (~11 GiB, NVML denied) report CUDA OOM as an NVML assert:
      # rerun the job on cpu and take those rows from the cpu run (tools/cpu_retry.py).
      if prun python tools/cpu_retry.py needs "$out"; then
        log "CUDA OOM rows in $out: rerunning on cpu"
        if timeout --kill-after=5m "${JOB_TIMEOUT:-6h}" pixi run -e "$PENV" python tools/run_all.py --suites "$suite" --datasets "$dataset" \
            --seed "$seed" --epochs "$EPOCHS" --device cpu --output "$out.cpu.json" 2>&1 | tail -n 2; then
          prun python tools/cpu_retry.py merge "$out" "$out.cpu.json"
        else
          log "cpu retry failed: $job seed $seed"
        fi
        rm -f "$out.cpu.json"
      fi
    else
      n_fail=$((n_fail + 1)); log "job failed: $job seed $seed"
    fi
  done
done
log "DONE: $n_ok ok, $n_fail failed -> $OUT_DIR"
