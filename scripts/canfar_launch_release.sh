#!/usr/bin/env bash
# Launch the torchregress 0.3.0 release benchmark on CANFAR (run on your machine).
#
#   scripts/canfar_launch_release.sh launch   # submit all job groups
#   scripts/canfar_launch_release.sh status   # canfar ps for this run
#   scripts/canfar_launch_release.sh report   # gates + summaries once groups finish
#   scripts/canfar_launch_release.sh logs     # events + log tail of each group -> release_logs.txt
#   scripts/canfar_launch_release.sh fill [group [ids]]  # (re)submit missing replicas as single sessions
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
  local code='import base64, os, subprocess
w = "/scratch/bootstrap-" + os.environ["RUN_TAG"] + "-" + os.environ.get("GROUP", "report") + "-" + os.environ.get("REPLICA_ID", "1")
subprocess.run(["rm", "-rf", w], check=True)
subprocess.run(["git", "init", "-q", w], check=True)
auth = base64.b64encode(("x-access-token:" + os.environ.get("GH_TOKEN", "")).encode()).decode()
subprocess.run(["git", "-C", w, "-c", "http.extraheader=Authorization: Basic " + auth, "fetch", "-q", "--depth", "1", "https://github.com/sfabbro/torchregress-harness.git", os.environ["HARNESS_REF"]], check=True)
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

group_spec() { # <group> -> "replicas cpu memGB gpu"
  case "$1" in
    tabular) echo "${TAB_REPLICAS:-12} 8 32 0" ;;
    sbi) echo "${SBI_REPLICAS:-5} 8 32 0" ;;
    fm) echo "${FM_REPLICAS:-2} 8 64 1" ;;
    image) echo "${IMG_REPLICAS:-6} 8 64 1" ;;
    *) die "unknown group: $1" ;;
  esac
}

record_sessions() { # <group> <canfar output>: remember session ids for `logs`
  echo "$2" | sed -n 's/^ *- \([a-z0-9]\{6,\}\)$/\1/p' | sed "s/^/SESSION_$1=/" >> "$STATE"
}

failed_replicas() { # <group> <canfar output> -> replica numbers Skaha refused to create
  echo "$2" | sed -n "s/.*Failed to create session.*'trr-$1-[0-9]*-\([0-9][0-9]*\)'.*/\1/p" | sort -n | uniq
}

create() { # <name> <cpu> <mem> <gpu> <replicas> [-e K=V ...]: one canfar create, output on stdout
  local name="$1" cpu="$2" mem="$3" gpu="$4" reps="$5" gpu_args=()
  shift 5
  [ "$gpu" -gt 0 ] && gpu_args=(--gpu "$gpu")
  canfar create headless "$IMAGE" ${DRY[@]+"${DRY[@]}"} --name "$name" \
    --cpu "$cpu" --memory "$mem" ${gpu_args[@]+"${gpu_args[@]}"} --replicas "$reps" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} "$@" \
    -- python3 -c "$(bootstrap scripts/skaha_release.sh)" 2>&1
}

fill_group() { # <group> <replica ids...>: one single-replica session per missing shard
  local group="$1" reps cpu mem gpu id out still=""
  shift
  read -r reps cpu mem gpu <<< "$(group_spec "$group")"
  for id in "$@"; do
    echo ">> $group: shard $id of $reps (single session)"
    out=$(create "trr-$group-$(name_tag)-s$id" "$cpu" "$mem" "$gpu" 1 \
      -e "GROUP=$group" -e "SHARD_ID=$id" -e "SHARD_COUNT=$reps") || true
    echo "$out" | redact
    if echo "$out" | grep -q "Successfully created"; then
      record_sessions "$group" "$out"
    else
      still="${still:+$still,}$id"
    fi
  done
  # Forget the old failure list for this group, keep what is still missing.
  sed -i.bak "/^FAILED_$group=/d" "$STATE" && rm -f "$STATE.bak"
  if [ -n "$still" ]; then
    echo "FAILED_$group=$still" >> "$STATE"
    echo "!! $group shards still not created: $still (retry: $0 fill $group)"
  fi
}

launch_group() { # <group>
  local group="$1" reps cpu mem gpu out failed
  read -r reps cpu mem gpu <<< "$(group_spec "$group")"
  echo ">> $group: $reps replicas, ${cpu}c/${mem}G, gpu=$gpu"
  out=$(create "trr-$group-$(name_tag)" "$cpu" "$mem" "$gpu" "$reps" -e "GROUP=$group") \
    || { echo "$out" | redact; die "$group: canfar create failed"; }
  echo "$out" | redact
  record_sessions "$group" "$out"
  # Skaha sometimes refuses a few replicas of a batch: resubmit those shards once.
  failed=$(failed_replicas "$group" "$out")
  if [ -n "$failed" ]; then
    echo "!! $group: replicas not created: $(echo $failed); resubmitting them in 30 s"
    sleep "${FILL_DELAY:-30}"
    # shellcheck disable=SC2086
    fill_group "$group" $failed
  fi
}

load_state() {
  [ -f "$STATE" ] || die "no run launched from here ($STATE missing)"
  # shellcheck disable=SC1090
  . "$STATE"
}

# canfar prints full request URLs (with the GH_TOKEN env) on errors: never echo the token.
redact() { sed -E 's/(gh[opsu]_)[A-Za-z0-9]{20,}/\1REDACTED/g; s/(hf_)[A-Za-z0-9]{20,}/\1REDACTED/g'; }

case "${1:-}" in
  launch)
    HARNESS_REF=$(pushed_sha "$HERE")
    LIB_REF=$(pushed_sha "$LIB_DIR")
    GH_TOKEN=$(gh_token)
    RUN_TAG="release_$(date +%Y%m%d_%H%M)"
    printf 'RUN_TAG=%s\nHARNESS_REF=%s\nLIB_REF=%s\n' "$RUN_TAG" "$HARNESS_REF" "$LIB_REF" > "$STATE"
    set_env_args
    echo "run $RUN_TAG  harness $HARNESS_REF  torchregress $LIB_REF  image $IMAGE"
    for group in tabular sbi fm image; do launch_group "$group"; done
    echo "state saved to $STATE; check progress with: $0 status"
    ;;
  status)
    load_state
    canfar ps --all | grep -E "NAME|trr-.*-$(name_tag)" || true
    grep '^FAILED_' "$STATE" | sed 's/^FAILED_/never created: /' || true
    echo "logs: /arc/projects/ots/torchregress-harness/runs/$RUN_TAG/<group>/logs/"
    ;;
  logs)
    # Logs + events of the first session of each group (or of every session with ALL=1)
    # into release_logs.txt, tokens redacted: paste or share that file.
    load_state
    out="$HERE/release_logs.txt"; : > "$out"
    for group in tabular sbi fm image report; do
      ids=$(sed -n "s/^SESSION_$group=//p" "$STATE")
      [ -n "$ids" ] || continue
      [ "${ALL:-0}" = 1 ] || ids=$(echo "$ids" | head -n 1)
      for id in $ids; do
        { echo "===== $group $id ====="; canfar events "$id" 2>&1; canfar logs "$id" 2>&1 | tail -n "${TAIL:-80}"; } | redact >> "$out"
      done
    done
    echo "wrote $out"
    ;;
  fill)
    # fill                 every group's FAILED_<group> shards from the state file
    # fill tabular 3,6     those replica numbers (1-based, as in the session names)
    load_state
    # Fill sessions run the harness HEAD (it must know SHARD_ID); the benchmark
    # code itself is unchanged when only scripts/ moved since the launch.
    HARNESS_REF=$(pushed_sha "$HERE")
    echo "FILL_HARNESS_REF=$HARNESS_REF" >> "$STATE"
    GH_TOKEN=$(gh_token)
    set_env_args
    echo "run $RUN_TAG  fill harness $HARNESS_REF  torchregress $LIB_REF"
    if [ -n "${2:-}" ]; then
      ids="${3:-$(sed -n "s/^FAILED_$2=//p" "$STATE")}"
      [ -n "$ids" ] || die "no missing shards recorded for $2; pass ids, e.g. $0 fill $2 3,6"
      # shellcheck disable=SC2086
      fill_group "$2" ${ids//,/ }
    else
      found=0
      for group in tabular sbi fm image; do
        ids=$(sed -n "s/^FAILED_$group=//p" "$STATE")
        [ -n "$ids" ] || continue
        found=1
        # shellcheck disable=SC2086
        fill_group "$group" ${ids//,/ }
      done
      [ "$found" = 1 ] || echo "nothing to fill (no FAILED_<group> lines in $STATE)"
    fi
    ;;
  report)
    load_state
    GH_TOKEN=$(gh_token)
    set_env_args
    out=$(canfar create headless "$IMAGE" ${DRY[@]+"${DRY[@]}"} --name "trr-report-$(name_tag)" \
      --cpu 4 --memory 16 ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -- python3 -c "$(bootstrap scripts/skaha_release_report.sh)" 2>&1) \
      || { echo "$out" | redact; die "report: canfar create failed"; }
    echo "$out" | redact
    record_sessions report "$out"
    echo "report -> /arc/projects/ots/torchregress-harness/runs/$RUN_TAG/report/"
    ;;
  *)
    sed -n '2,21p' "$0"; exit 1 ;;
esac
