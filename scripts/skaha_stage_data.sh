#!/usr/bin/env bash
# Stage slow-to-download real datasets on /arc once, for every release session.
#
# Runs on one headless CANFAR session (scripts/canfar_launch_release.sh stage).
# - Folktables ACS 2018 1-Year person files for all 50 states + PR (several GB
#   from census.gov; folktables_eiv and the all-states likelihood datasets read
#   them via harness.datasets._folktables_root).
# - TabReD regression sets (sberbank-housing, cooking-time, delivery-eta,
#   maps-routing, weather) with the official `tabred download` (Kaggle API;
#   uses the session user's ~/.kaggle/kaggle.json on /arc/home, never copied).
# STAGE_WHAT=folktables,tabred (default both). Idempotent: present data is skipped.
set -uo pipefail
ARC_PROJ="${ARC_PROJ:-/arc/projects/ots/torchregress-harness}"
STAGED="$ARC_PROJ/datasets/staged"
DEST="$STAGED/folktables"
mkdir -p "$DEST" "$STAGED/tabred"
exec > >(tee -a "$STAGED/stage.log") 2>&1
WHAT=",${STAGE_WHAT:-folktables,tabred},"
log() { echo "[$(date +%H:%M:%S)] [stage] $*"; }
log "staging Folktables ACS 2018 1-Year person files into $DEST"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
VENV=/scratch/stage-venv
python3 -m venv "$VENV" || { log "FATAL: python3 -m venv failed"; exit 2; }
"$VENV/bin/python" -m pip install -q --upgrade pip folktables pandas || { log "FATAL: pip install failed"; exit 3; }

rc=0
if [[ "$WHAT" == *",folktables,"* ]]; then
"$VENV/bin/python" - "$DEST" <<'PY'
import sys, time
from folktables import ACSDataSource
from folktables.load_acs import state_list
root = sys.argv[1]
ds = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=root)
failed = []
for st in state_list:
    t = time.time()
    for attempt in range(3):
        try:
            ds.get_data(states=[st], download=True)
            print(f"{st} ok ({time.time() - t:.0f}s)", flush=True)
            break
        except Exception as exc:  # noqa: BLE001 — retry flaky census.gov
            print(f"{st} attempt {attempt + 1} failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(30)
    else:
        failed.append(st)
print("FAILED:", failed if failed else "none", flush=True)
sys.exit(1 if failed else 0)
PY
rc=$?
log "folktables rc=$rc"
fi

if [[ "$WHAT" == *",tabred,"* ]]; then
  if [ ! -s "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_API_TOKEN:-}${KAGGLE_KEY:-}" ]; then
    log "WARN: no Kaggle credentials in $HOME/.kaggle; skipping TabReD"
  else
    "$VENV/bin/python" -m pip install -q uv || log "WARN: pip install uv failed"
    for ds in sberbank-housing cooking-time delivery-eta maps-routing weather; do
      if [ -d "$STAGED/tabred/$ds" ]; then log "tabred $ds present"; continue; fi
      log "tabred download $ds"
      # astroai/base presets uv dirs to read-only /usr/local/share/uv: override all.
      UV_CACHE_DIR=/scratch/uv/cache UV_TOOL_DIR=/scratch/uv/tools UV_TOOL_BIN_DIR=/scratch/uv/bin \
        UV_PYTHON_INSTALL_DIR=/scratch/uv/python \
        "$VENV/bin/uv" tool run --python 3.13 tabred download "$ds" \
        --output-path "$STAGED/tabred" || { log "tabred $ds FAILED (competition rules not accepted?)"; rc=1; }
    done
    ls -la "$STAGED/tabred"
  fi
fi
log "DONE rc=$rc"
exit $rc
