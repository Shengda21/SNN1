#!/bin/bash
cd "$(dirname "$0")" || exit 1
PY=${PY:-python3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set -x
$PY e21_ladder.py --platform sarcos --res ../results/sarcos --workers 3
$PY e2x_verify.py --platform sarcos --res ../results/sarcos --orig ../results/sarcos > ../logs/verify_sarcos.log 2>&1
$PY e24_certificate.py --platform sim --res ../results/sim --workers 3
$PY e24_certificate.py --platform sarcos --res ../results/sarcos --workers 3
$PY e25_budget.py --platform sim --res ../results/sim --workers 3
$PY e25_budget.py --platform sarcos --res ../results/sarcos --workers 3
$PY e26_select.py --platform sim --res ../results/sim --workers 3
$PY e26_select.py --platform sarcos --res ../results/sarcos --workers 3
echo ALLDONE
