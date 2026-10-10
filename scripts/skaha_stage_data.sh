#!/usr/bin/env bash
# Stage slow-to-download real datasets on /arc once, for every release session.
#
# Runs on one headless CANFAR session (scripts/canfar_launch_release.sh stage).
# Currently: Folktables ACS 2018 1-Year person files for all 50 states + PR
# (several GB from census.gov; folktables_eiv and the all-states likelihood
# datasets read them via harness.datasets._folktables_root).
# Idempotent: files already present are skipped by folktables itself.
set -uo pipefail
ARC_PROJ="${ARC_PROJ:-/arc/projects/ots/torchregress-harness}"
STAGED="$ARC_PROJ/datasets/staged"
DEST="$STAGED/folktables"
mkdir -p "$DEST"
exec > >(tee -a "$DEST/stage.log") 2>&1
log() { echo "[$(date +%H:%M:%S)] [stage] $*"; }
log "staging Folktables ACS 2018 1-Year person files into $DEST"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
VENV=/scratch/stage-venv
python3 -m venv "$VENV" || { log "FATAL: python3 -m venv failed"; exit 2; }
"$VENV/bin/python" -m pip install -q --upgrade pip folktables pandas || { log "FATAL: pip install failed"; exit 3; }

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
log "DONE rc=$rc"
exit $rc
