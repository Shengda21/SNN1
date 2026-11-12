#!/bin/bash
# Stage 2: waits for the moment dumps, then E23 dumps (GPU), E21 on paired noise, E22, E23, E24 rerun.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python3}
until grep -q ALLDONE ../logs/e2x_dumps.log; do sleep 30; done
until grep -q ALLDONE ../logs/e2x_cpu1.log; do sleep 30; done
set -x
OMP_NUM_THREADS=4 $PY e23_dump.py --platform sim --data_dir ../data_sim --runs ../runs25/sim --out ../results/sim
OMP_NUM_THREADS=4 $PY e23_dump.py --platform sarcos --data_dir .. --runs ../runs25/sarcos --out ../results/sarcos
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY e22_path.py --platform sim --res ../results/sim --workers 4
$PY e22_path.py --platform sarcos --res ../results/sarcos --workers 4
$PY e21_ladder.py --platform sim --res ../results/sim --source paired --workers 4
$PY e21_ladder.py --platform sarcos --res ../results/sarcos --source paired --workers 4
$PY e2x_verify.py --platform sim --res ../results/sim --orig ../results/sim > ../logs/verify_sim.log 2>&1
$PY e2x_verify.py --platform sarcos --res ../results/sarcos --orig ../results/sarcos > ../logs/verify_sarcos.log 2>&1
$PY e23_labelfree.py --platform sim --res ../results/sim
$PY e23_labelfree.py --platform sarcos --res ../results/sarcos
$PY e24_certificate.py --platform sim --res ../results/sim --workers 4
$PY e24_certificate.py --platform sarcos --res ../results/sarcos --workers 4
echo ALLDONE
