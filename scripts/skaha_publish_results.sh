#!/usr/bin/env bash
# Print the artifacts of a Skaha benchmark run to stdout, so they can be
# recovered through the session logs endpoint (`canfar logs <session>`).
#
# Usage (from the container):
#   bash scripts/skaha_publish_results.sh RUN_ID [ARC_ROOT]
#
# Each artifact is emitted as:
#   @@ <filename> <size_bytes> <base64-of-gzip> @@
# `.md` reports and `versions.txt` are emitted raw (small, human-readable);
# `.json` files are gzipped then base64-encoded.  Files larger than
# MAX_RAW (default 24 MB) are skipped with a `!! SKIPPED` line.
set -euo pipefail

RUN_ID="${1:?usage: skaha_publish_results.sh RUN_ID [ARC_ROOT]}"
ARC_ROOT="${2:-}"
if [ -z "$ARC_ROOT" ]; then
  for cand in /arc/projects/ots/torchregress-harness /arc/home/sfabbro/torchregress-harness; do
    if [ -d "$cand" ]; then ARC_ROOT="$cand"; break; fi
  done
  ARC_ROOT="${ARC_ROOT:-$HOME}"
fi
OUT_DIR="$ARC_ROOT/runs/$RUN_ID"
MAX_RAW="${MAX_RAW:-25165824}"   # 24 MB

if [ ! -d "$OUT_DIR" ]; then
  echo "!! NO SUCH RUN: $OUT_DIR"
  exit 1
fi

echo "## publish run=$RUN_ID dir=$OUT_DIR"
ls -la "$OUT_DIR"

for f in "$OUT_DIR"/*; do
  [ -f "$f" ] || continue
  name="$(basename "$f")"
  size="$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")"
  case "$name" in
    *.md|*.txt|run_meta.txt)
      echo "@@ $name $size raw"
      cat "$f"
      echo "@@ end-$name"
      ;;
    *.json)
      if [ "$size" -gt "$MAX_RAW" ]; then
        echo "!! SKIPPED $name ($size bytes > ${MAX_RAW})"
        continue
      fi
      echo "@@ $name $size gz64"
      gzip -c "$f" | base64 -w 0
      echo
      echo "@@ end-$name"
      ;;
    *)
      echo "!! SKIPPED (unknown type) $name $size"
      ;;
  esac
done
echo "## publish done"
