#!/usr/bin/env bash
# Sequential job queue for a cloud GPU (proposals/saturn_offload.md section 3). Same form as queue4.sh, plus:
#  - the interpreter comes from $PY (default: python on PATH);
#  - lines that already finished with rc=0 (listed in queue_cloud.done) are skipped, so the queue can be restarted after
#    the machine stops; an interrupted cell resumes from its last checkpoint (cells.run_cell) when the line is rerun;
#  - checkpoints go to $TMPDIR (python's tempfile.gettempdir()); set TMPDIR to persistent storage before starting.
# Usage: PY=/path/to/python TMPDIR=/persistent/tmp nohup bash queue_cloud.sh queue_cloud.txt > /dev/null 2>&1 &
cd "$(dirname "$0")"
PY=${PY:-python}
Q=${1:-queue_cloud.txt}
touch queue_cloud.done
while read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  if grep -qxF "$line" queue_cloud.done; then echo "$(date '+%F %T') SKIP(done) $line" >> queue_cloud.log; continue; fi
  echo "$(date '+%F %T') START $line" >> queue_cloud.log
  PYTHONUTF8=1 $PY -B $line >> queue_cloud_jobs.out 2>> queue_cloud_jobs.err
  rc=$?
  echo "$(date '+%F %T') END rc=$rc $line" >> queue_cloud.log
  [ $rc -eq 0 ] && echo "$line" >> queue_cloud.done
done < "$Q"
