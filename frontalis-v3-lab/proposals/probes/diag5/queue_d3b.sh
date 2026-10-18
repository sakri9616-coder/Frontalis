#!/usr/bin/env bash
# Second queue: E1 rerun with the production partner scale of KRR (0.0, P16); E1_repro checks that sim5 with the switch off
# reproduces the P15 KRR C-SLp cell. Starts after queue_d3.sh is done.
cd "$(dirname "$0")"
PY=/c/Users/sakur/AppData/Local/ASIST-She-necheia/Frontalis/frontalis-phase3d-lab/.venv/Scripts/python.exe
until grep -q "QUEUE DONE" d3.log; do sleep 20; done
run() { echo "START $(date +%T) $*" >> d3.log; $PY -B d3_exp.py "$@" >> d3.log 2>&1; echo "END $(date +%T) rc=$? $1" >> d3.log; }
run E1_repro_krr krr prod A '{}' C-SLp
run E1b_krr_flat krr prod A '{"krr_flat_b":1}' C-SLp
echo "QUEUE2 DONE $(date +%T)" >> d3.log
