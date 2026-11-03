#!/usr/bin/env bash
# Run a test/CI command for torchregress, torchregress-harness or torchregress-research
# on a CANFAR headless session, against commits already pushed to the sfabbro forks.
#
#   canfar_ci.sh launch <name> <repo> <ref> [lib_ref] -- <command...>
#   canfar_ci.sh status            # canfar ps for ci sessions
#   canfar_ci.sh logs <session-id> # session log (also kept on /arc)
#
# <repo>    torchregress | torchregress-harness | torchregress-research
# <ref>     commit SHA or branch of that repo on github.com/sfabbro/<repo>
# [lib_ref] torchregress SHA/branch checked out beside it as ../torchregress (the harness
#           and research pixi envs install it from there). Defaults to <ref> for torchregress itself.
# command   run from the repo root, e.g.  pixi run test   or  ./scripts/ci_local.sh
#
# Env: CPU (8), MEM (32), GPU (0), IMAGE (astroai/base:latest), DRY_RUN=1.
# Logs: /arc/projects/ots/torchregress-ci/<name>/{run.log,exit_code}
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
IMAGE="${IMAGE:-astroai/base:latest}"
die() { echo "ERROR: $*" >&2; exit 1; }
hex() { printf '%s' "$1" | od -An -tx1 | tr -d ' \n'; }

if [ -z "${CANFAR_CERTIFICATE:-}" ] && [ -s "$HOME/.canfar/cadcproxy.pem" ]; then
  export CANFAR_CERTIFICATE="$HOME/.canfar/cadcproxy.pem"
fi

# The in-session runner (bash), shipped hex-encoded: canfar word-splits the
# command line and Skaha embeds args in a double-quoted YAML list.
read -r -d '' RUNNER <<'EOS' || true
set -uo pipefail
OUT="/arc/projects/ots/torchregress-ci/$CI_NAME"
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1
echo "== $CI_NAME: $CI_REPO@$CI_REF lib@$CI_LIB_REF on $(hostname) $(date -u +%FT%TZ)"
W="/scratch/ci-$CI_NAME"; rm -rf "$W"; mkdir -p "$W"
export HOME="${HOME:-/scratch}" PIXI_HOME="$W/.pixi" PIXI_CACHE_DIR="/scratch/pixi-cache" RATTLER_CACHE_DIR="/scratch/pixi-cache"
AUTH=()
[ -n "${GH_TOKEN:-}" ] && AUTH=(-c "http.extraheader=Authorization: Basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 | tr -d '\n')")
fetch() { # <repo> <ref> <dir>; public forks need no token
  git init -q "$3" && git -C "$3" "${AUTH[@]}" fetch -q --depth 1 "https://github.com/sfabbro/$1.git" "$2" \
    && git -C "$3" checkout -q FETCH_HEAD && echo "   $1 -> $(git -C "$3" rev-parse HEAD)"
}
case "$CI_REPO" in
  torchregress) fetch torchregress "$CI_REF" "$W/torchregress" || exit 2; cd "$W/torchregress" ;;
  torchregress-harness) fetch torchregress "$CI_LIB_REF" "$W/torchregress" || exit 2  # pixi path ../torchregress
    fetch torchregress-harness "$CI_REF" "$W/torchregress-harness" || exit 2; cd "$W/torchregress-harness" ;;
  torchregress-research) fetch torchregress "$CI_LIB_REF" "$W/torchregress" || exit 2
    fetch torchregress-research "$CI_REF" "$W/torchregress-research" || exit 2; cd "$W/torchregress-research" ;;
  *) echo "unknown repo $CI_REPO"; exit 2 ;;
esac
command -v pixi >/dev/null || { curl -fsSL https://pixi.sh/install.sh | bash >/dev/null; export PATH="$PIXI_HOME/bin:$PATH"; }
for a in 1 2 3; do pixi install 2>&1 | tail -n 3 && break; sleep 60; done
CMD="$(printf '%s' "$CI_CMD_HEX" | xxd -r -p 2>/dev/null || python3 -c "import sys;print(bytes.fromhex(sys.argv[1]).decode())" "$CI_CMD_HEX")"
echo "== running: $CMD"
export PYTHONNOUSERSITE=1; unset PYTHONPATH
bash -c "$CMD"; rc=$?
echo "$rc" > "$OUT/exit_code"
echo "== exit $rc $(date -u +%FT%TZ)"
exit $rc
EOS

case "${1:-}" in
  launch)
    shift; [ $# -ge 5 ] || die "usage: launch <name> <repo> <ref> [lib_ref] -- <command...>"
    name="$1" repo="$2" ref="$3"; shift 3
    lib_ref="$ref"; if [ "$1" != "--" ]; then lib_ref="$1"; shift; fi
    [ "$1" = "--" ] || die "missing -- before command"; shift
    case "$name" in *[!a-z0-9-]*) die "name: lowercase letters, digits and - only" ;; esac
    token="${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}"
    # A stale token makes GitHub refuse even public fetches: send only a valid one.
    if [ -n "$token" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $token" https://api.github.com/user)" != 200 ]; then
      echo "warning: GitHub token is invalid; launching without it" >&2; token=""
    fi
    [ -n "$token" ] || [ "$repo" = torchregress ] || die "no GitHub token (gh auth login); $repo is private"
    boot="exec(bytes.fromhex('$(hex "import os;open('/tmp/ci_runner.sh','w').write(bytes.fromhex(os.environ['CI_RUNNER_HEX']).decode());os.execvp('bash',['bash','/tmp/ci_runner.sh'])")').decode())"
    dry=(); [ "${DRY_RUN:-0}" = 1 ] && dry=(--dry-run)
    gpu=(); [ "${GPU:-0}" -gt 0 ] && gpu=(--gpu "$GPU")
    canfar create headless "$IMAGE" ${dry[@]+"${dry[@]}"} --name "ci-$name" --cpu "${CPU:-8}" --memory "${MEM:-32}" \
      ${gpu[@]+"${gpu[@]}"} -e "CI_NAME=$name" -e "CI_REPO=$repo" -e "CI_REF=$ref" -e "CI_LIB_REF=$lib_ref" \
      -e "CI_CMD_HEX=$(hex "$*")" -e "CI_RUNNER_HEX=$(hex "$RUNNER")" -e "GH_TOKEN=$token" \
      -- python3 -c "$boot" 2>&1 | sed -E "s/(GH_TOKEN=)[^ ]+/\1***/g; s/('GH_TOKEN': ')[^']+/\1***/g; s/(CI_RUNNER_HEX': ')[0-9a-f]+/\1.../g"
    ;;
  status) canfar ps --all 2>&1 | grep -E "NAME|ci-" || true ;;
  logs) [ -n "${2:-}" ] || die "usage: logs <session-id>"; canfar logs "$2" ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
