#!/usr/bin/env bash
# Run a test/CI command for torchregress, torchregress-harness or torchregress-research
# on a CANFAR headless session. The commits are shipped as `git archive` tarballs to
# /arc/projects/ots/torchregress-ci/src (scripts/_ship_src.sh), so sessions need no
# GitHub access and no GitHub credentials are ever sent to CANFAR.
#
#   canfar_ci.sh launch <name> <repo> <ref> [lib_ref] -- <command...>
#   canfar_ci.sh status            # canfar ps for ci sessions
#   canfar_ci.sh logs <session-id> # session log (also kept on /arc)
#
# <repo>    torchregress | torchregress-harness | torchregress-research
# <ref>     commit SHA or branch, resolved in the local clone (../<repo> next to the harness,
#           or the harness itself); it needs no push
# [lib_ref] torchregress commit unpacked beside it as ../torchregress (the harness and
#           research pixi envs install it from there). Defaults to <ref> for torchregress itself.
# command   run from the repo root, e.g.  pixi run test   or  ./scripts/ci_local.sh
#
# Env: CPU (8), MEM (32), GPU (0), IMAGE (astroai/base:latest), DRY_RUN=1.
# Logs: /arc/projects/ots/torchregress-ci/<name>/{run.log,exit_code}
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/_ship_src.sh
. "$HERE/scripts/_ship_src.sh"
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
unpack() { # <repo> <sha>: committed tree shipped by the launcher to /arc
  tar -xzf "$CI_SRC_DIR/$1-$2.tar.gz" -C "$W" && echo "   $1 -> $2"
}
case "$CI_REPO" in
  torchregress) unpack torchregress "$CI_REF" || exit 2; cd "$W/torchregress" ;;
  torchregress-harness | torchregress-research)
    unpack torchregress "$CI_LIB_REF" || exit 2  # pixi path ../torchregress
    unpack "$CI_REPO" "$CI_REF" || exit 2; cd "$W/$CI_REPO" ;;
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
    case "$repo" in
      torchregress) repo_dir="$HERE/../torchregress" ;;
      torchregress-harness) repo_dir="$HERE" ;;
      torchregress-research) repo_dir="$HERE/../torchregress-research" ;;
      *) die "unknown repo $repo" ;;
    esac
    sha=$(ship_src "$repo_dir" "$repo" "$ref") || die "cannot ship $repo@$ref"
    lib_sha=$sha
    if [ "$repo" != torchregress ]; then
      lib_sha=$(ship_src "$HERE/../torchregress" torchregress "$lib_ref") || die "cannot ship torchregress@$lib_ref"
    fi
    boot="exec(bytes.fromhex('$(hex "import os;open('/tmp/ci_runner.sh','w').write(bytes.fromhex(os.environ['CI_RUNNER_HEX']).decode());os.execvp('bash',['bash','/tmp/ci_runner.sh'])")').decode())"
    dry=(); [ "${DRY_RUN:-0}" = 1 ] && dry=(--dry-run)
    gpu=(); [ "${GPU:-0}" -gt 0 ] && gpu=(--gpu "$GPU")
    canfar create headless "$IMAGE" ${dry[@]+"${dry[@]}"} --name "ci-$name" --cpu "${CPU:-8}" --memory "${MEM:-32}" \
      ${gpu[@]+"${gpu[@]}"} -e "CI_NAME=$name" -e "CI_REPO=$repo" -e "CI_REF=$sha" -e "CI_LIB_REF=$lib_sha" -e "CI_SRC_DIR=$SRC_ARC_DIR" \
      -e "CI_CMD_HEX=$(hex "$*")" -e "CI_RUNNER_HEX=$(hex "$RUNNER")" \
      -- python3 -c "$boot" 2>&1 | sed -E "s/(CI_RUNNER_HEX': ')[0-9a-f]+/\1.../g"
    ;;
  status) canfar ps --all 2>&1 | grep -E "NAME|ci-" || true ;;
  logs) [ -n "${2:-}" ] || die "usage: logs <session-id>"; canfar logs "$2" ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac
