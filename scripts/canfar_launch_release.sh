#!/usr/bin/env bash
# Launch the torchregress 0.3.0 release benchmark on CANFAR (run on your machine).
#
#   scripts/canfar_launch_release.sh launch   # submit all job groups
#   scripts/canfar_launch_release.sh status   # canfar ps for this run
#   scripts/canfar_launch_release.sh report   # gates + summaries once groups finish
#   DRY_RUN=1 scripts/canfar_launch_release.sh launch   # parse only, submit nothing
#
# Needs: `canfar` CLI already logged in (`canfar auth login`), GH_TOKEN exported or
# `gh` logged in (the token lets the sessions fetch the private harness fork), both forks
# pushed (the script checks).  HF_TOKEN is optional (public TabICL/sdm checkpoints
# download without it; TabPFN is paused for this run, set WITH_TABPFN=1 and a token
# that accepted the TabPFN 3.5 terms to bring it back).
#
# Sizing (override with env): replicas x cores/RAM/GPU per group.
#   tabular  TAB_REPLICAS=12  8 cores 32 GB          (116 suite x dataset jobs x seeds)
#   sbi      SBI_REPLICAS=5   8 cores 32 GB          (5 sbibm tasks x seeds)
#   fm       FM_REPLICAS=2    8 cores 64 GB, 1 GPU   (foundation models)
#   image    IMG_REPLICAS=6   8 cores 64 GB, 1 GPU   (image suites)
# Results: /arc/projects/ots/torchregress-harness/runs/$RUN_TAG/<group>/
set -euo pipefail
# macOS ships bash 3.2: empty arrays are expanded as ${a[@]+"${a[@]}"} so set -u
# does not abort on them.

IMAGE="${IMAGE:-astroai/base:latest}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LIB_DIR="${LIB_DIR:-$HERE/../torchregress}"
STATE="$HERE/.release_run"
SEEDS="${SEEDS:-0,1,2}"  # comma-separated: env values must not contain spaces
EPOCHS="${EPOCHS:-60}"

die() { echo "ERROR: $*" >&2; exit 1; }
command -v canfar >/dev/null || die "canfar CLI not found (pip install canfar; canfar auth login)"
gh_token() { # GH_TOKEN from the environment, else from a logged-in gh CLI
  if [ -n "${GH_TOKEN:-}" ]; then echo "$GH_TOKEN"; return; fi
  command -v gh >/dev/null || die "set GH_TOKEN (read access to sfabbro/torchregress-harness) or log in with gh"
  gh auth token
}

pushed_sha() { # <repo dir> -> HEAD sha, after checking it is on origin/main
  local dir="$1" head remote
  head=$(git -C "$dir" rev-parse HEAD)
  remote=$(git -C "$dir" ls-remote origin refs/heads/main | cut -f1)
  [ "$head" = "$remote" ] || die "$dir: HEAD $head is not origin/main ($remote); push first"
  echo "$head"
}

# python3 -c payload without spaces: `canfar create headless` word-splits the
# command line, so the bootstrap travels hex-encoded and exec'd in the session.
bootstrap() { # <script path inside the harness repo>
  # Single-quoted literal, not a heredoc inside $(...): bash 3.2 misparses those.
  local code='import os, subprocess
w = "/scratch/bootstrap-" + os.environ["RUN_TAG"] + "-" + os.environ.get("GROUP", "report") + "-" + os.environ.get("REPLICA_ID", "1")
subprocess.run(["rm", "-rf", w], check=True)
subprocess.run(["git", "init", "-q", w], check=True)
subprocess.run(["git", "-C", w, "-c", "http.extraheader=Authorization: Bearer " + os.environ.get("GH_TOKEN", ""), "fetch", "-q", "--depth", "1", "https://github.com/sfabbro/torchregress-harness.git", os.environ["HARNESS_REF"]], check=True)
subprocess.run(["git", "-C", w, "checkout", "-q", "FETCH_HEAD"], check=True)
os.execvp("bash", ["bash", w + "/SCRIPT"])'
  code="${code/SCRIPT/$1}"
  # No spaces (canfar word-splits the command) and no double quotes (Skaha
  # embeds the args in a double-quoted YAML list).
  printf "exec(bytes.fromhex('%s').decode())" "$(printf '%s' "$code" | od -An -tx1 | tr -d ' \n')"
}

# Skaha session names allow only letters, digits and '-'.
name_tag() { local t="${RUN_TAG#release_}"; echo "${t//_/}"; }

ENV_ARGS=()
set_env_args() {
  ENV_ARGS=(-e "RUN_TAG=$RUN_TAG" -e "HARNESS_REF=$HARNESS_REF" -e "LIB_REF=$LIB_REF"
    -e "GH_TOKEN=$GH_TOKEN" -e "SEEDS=$SEEDS" -e "EPOCHS=$EPOCHS")
  [ -n "${HF_TOKEN:-}" ] && ENV_ARGS+=(-e "HF_TOKEN=$HF_TOKEN")
  [ "${WITH_TABPFN:-0}" = 1 ] && ENV_ARGS+=(-e "TORCHREGRESS_HARNESS_WITH_TABPFN=1")
  return 0
}
DRY=()
[ "${DRY_RUN:-0}" = 1 ] && DRY=(--dry-run)

launch_group() { # <group> <replicas> <cpu> <mem GB> <gpu>
  local group="$1" reps="$2" cpu="$3" mem="$4" gpu="$5" gpu_args=()
  [ "$gpu" -gt 0 ] && gpu_args=(--gpu "$gpu")
  echo ">> $group: $reps replicas, ${cpu}c/${mem}G, gpu=$gpu"
  canfar create headless "$IMAGE" ${DRY[@]+"${DRY[@]}"} --name "trr-$group-$(name_tag)" \
    --cpu "$cpu" --memory "$mem" ${gpu_args[@]+"${gpu_args[@]}"} --replicas "$reps" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -e "GROUP=$group" \
    -- python3 -c "$(bootstrap scripts/skaha_release.sh)"
}

case "${1:-}" in
  launch)
    HARNESS_REF=$(pushed_sha "$HERE")
    LIB_REF=$(pushed_sha "$LIB_DIR")
    GH_TOKEN=$(gh_token)
    RUN_TAG="release_$(date +%Y%m%d_%H%M)"
    printf 'RUN_TAG=%s\nHARNESS_REF=%s\nLIB_REF=%s\n' "$RUN_TAG" "$HARNESS_REF" "$LIB_REF" > "$STATE"
    set_env_args
    echo "run $RUN_TAG  harness $HARNESS_REF  torchregress $LIB_REF  image $IMAGE"
    launch_group tabular "${TAB_REPLICAS:-12}" 8 32 0
    launch_group sbi "${SBI_REPLICAS:-5}" 8 32 0
    launch_group fm "${FM_REPLICAS:-2}" 8 64 1
    launch_group image "${IMG_REPLICAS:-6}" 8 64 1
    echo "state saved to $STATE; check progress with: $0 status"
    ;;
  status)
    [ -f "$STATE" ] || die "no run launched from here ($STATE missing)"
    # shellcheck disable=SC1090
    . "$STATE"
    canfar ps --all | grep -E "NAME|trr-.*-$(name_tag)" || true
    echo "logs: /arc/projects/ots/torchregress-harness/runs/$RUN_TAG/<group>/logs/"
    ;;
  report)
    [ -f "$STATE" ] || die "no run launched from here ($STATE missing)"
    # shellcheck disable=SC1090
    . "$STATE"
    GH_TOKEN=$(gh_token)
    set_env_args
    canfar create headless "$IMAGE" ${DRY[@]+"${DRY[@]}"} --name "trr-report-$(name_tag)" \
      --cpu 4 --memory 16 ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -- python3 -c "$(bootstrap scripts/skaha_release_report.sh)"
    echo "report -> /arc/projects/ots/torchregress-harness/runs/$RUN_TAG/report/"
    ;;
  *)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
