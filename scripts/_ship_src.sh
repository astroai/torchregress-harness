# Sourced by canfar_ci.sh and canfar_launch_release.sh (bash).
#
# Ship a commit to CANFAR storage instead of letting sessions fetch it from GitHub:
# sessions then need no GitHub credentials at all (the CANFAR proxy certificate does
# the upload), and nothing secret is passed in session environments.
#
#   sha=$(ship_src <local repo dir> <repo name> <ref>)
#
# Uploads `git archive --prefix=<repo>/ <sha>` to
#   arc:/projects/ots/torchregress-ci/src/<repo>-<sha>.tar.gz
# unless it is already there, and prints the full SHA. Sessions unpack it with
#   tar -xzf /arc/projects/ots/torchregress-ci/src/<repo>-<sha>.tar.gz -C <dir>
# (see src_tarball). The archive holds exactly the committed tree of <ref>.

SRC_ARC_DIR="${SRC_ARC_DIR:-/arc/projects/ots/torchregress-ci/src}"

src_tarball() { # <repo name> <sha> -> path of the tarball as seen inside a session
  echo "$SRC_ARC_DIR/$1-$2.tar.gz"
}

ship_src() { # <local repo dir> <repo name> <ref> -> full SHA on stdout
  local dir="$1" repo="$2" ref="$3" sha tmp dest
  sha=$(git -C "$dir" rev-parse --verify "$ref^{commit}") || {
    echo "ERROR: $dir: cannot resolve $ref" >&2
    return 1
  }
  # canfar data names /arc/<path> as arc:/<path>.
  dest="arc:$(src_tarball "$repo" "$sha" | sed 's#^/arc##')"
  if ! canfar data ls "$dest" >/dev/null 2>&1; then
    canfar data mkdir "arc:${SRC_ARC_DIR#/arc}" >/dev/null 2>&1 || true
    tmp=$(mktemp -t "$repo-src.XXXXXX")
    git -C "$dir" archive --format=tar.gz --prefix="$repo/" "$sha" > "$tmp" || {
      rm -f "$tmp"
      return 1
    }
    canfar data cp "local:$tmp" "$dest" >/dev/null || {
      rm -f "$tmp"
      echo "ERROR: upload of $repo@$sha to $dest failed" >&2
      return 1
    }
    rm -f "$tmp"
  fi
  echo "$sha"
}
