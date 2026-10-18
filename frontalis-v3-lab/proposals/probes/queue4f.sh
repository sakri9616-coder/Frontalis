#!/usr/bin/env bash
# Sequential job queue for the draft-4 dev runs (one GPU job at a time: the GPU is shared with another agent).
cd "$(dirname "$0")"
PY=/c/Users/sakur/AppData/Local/ASIST-She-necheia/Frontalis/frontalis-phase3d-lab/.venv/Scripts/python.exe
while read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  echo "$(date '+%F %T') START $line" >> queue4f.log
  PYTHONUTF8=1 $PY -B $line >> queue4f_jobs.out 2>> queue4f_jobs.err
  echo "$(date '+%F %T') END rc=$? $line" >> queue4f.log
done < "${1:-queue4f.txt}"
