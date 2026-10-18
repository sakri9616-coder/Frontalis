#!/usr/bin/env bash
# Cause investigation queue (dev seeds 77380-77389 only), sequential to avoid concurrent writes to d3_results.json.
cd "$(dirname "$0")"
PY=/c/Users/sakur/AppData/Local/ASIST-She-necheia/Frontalis/frontalis-phase3d-lab/.venv/Scripts/python.exe
run() { echo "START $(date +%T) $*" >> d3.log; $PY -B d3_exp.py "$@" >> d3.log 2>&1; echo "END $(date +%T) rc=$? $1" >> d3.log; }
run S0_base set mid 6 '{}'
run S1_wgate set mid 6 '{"wgate":1}'
run E1_krr_flat krr prod A '{"krr_flat_b":1}' C-SLp
run E2_set_belr set prod A '{"bel_r":1}' C-SLp
run S2_wgate_belr set mid 6 '{"wgate":1,"bel_r":1}'
echo "QUEUE DONE $(date +%T)" >> d3.log
