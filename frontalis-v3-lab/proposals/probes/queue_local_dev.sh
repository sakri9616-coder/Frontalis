#!/usr/bin/env bash
# Local dev-seed queue during the v5 evaluation (dev seeds 77380-77389 only). Resumable: a finished step (result file present)
# is skipped; relgate resumes from its own checkpoint.
cd "$(dirname "$0")"
PY=/c/Users/sakur/AppData/Local/ASIST-She-necheia/Frontalis/frontalis-phase3d-lab/.venv/Scripts/python.exe
SCR="C:/Users/sakur/AppData/Local/Temp/claude/C--Users-sakur-AppData-Local-ASIST-She-necheia/512780de-422b-4ef6-a34c-e3a52504368d/scratchpad/slp_32000.pt"
step() { out=$1; shift; [ -f "$out" ] && { echo "SKIP $out" >> queue_local_dev.log; return; }; echo "START $(date '+%F %T') $*" >> queue_local_dev.log; $PY -B "$@" >> queue_local_dev.log 2>&1; echo "END $(date '+%F %T') rc=$? $*" >> queue_local_dev.log; }
step p34_j3v5base.json p34_local_dev.py j3v5base "$SCR"
step p34_relgate.json p34_local_dev.py relgate
echo "QUEUE DONE $(date '+%F %T')" >> queue_local_dev.log
