#!/usr/bin/env bash
# Queue F replaces the rest of queue B (stopped 2026-10-17 ~20:00 so that the P23 speed measurement runs alone;
# queue B's running job 'p23 equiv base' is left to finish first).
cd "$(dirname "$0")"
until [ -f p23_equiv_base.json ]; do sleep 30; done
echo "$(date '+%F %T') queue B's last job (equiv base) finished; starting queue F" >> queue4b.log
bash queue4f.sh
