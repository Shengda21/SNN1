#!/bin/bash
# Moment dumps for E21-E26 (orig conditions, estimator variants, paired rho grid).
cd "$(dirname "$0")" || exit 1
PY=${PY:-python3}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
set -x
$PY e2x_dump_paired.py --platform sim --data_dir ../data_sim --runs ../runs25/sim --out ../results/sim --mode orig \
   --conds N,O1:0.35,O1:0.7,O1:1.0,O2B:0.7 --variants raw --members full --check
$PY e2x_dump_paired.py --platform sim --data_dir ../data_sim --runs ../runs25/sim --out ../results/sim --mode orig \
   --conds N,O1:0.35,O1:0.7,O1:1.0,O2B:0.7 --variants wfe,wfo --members core --xblocks "" --check
$PY e2x_dump_paired.py --platform sarcos --data_dir .. --runs ../runs25/sarcos --out ../results/sarcos --mode orig \
   --conds N,O1:0.25,O1:0.5,O1:0.7,O1qqd:0.5 --variants raw --members full --check
$PY e2x_dump_paired.py --platform sarcos --data_dir .. --runs ../runs25/sarcos --out ../results/sarcos --mode orig \
   --conds N,O1:0.25,O1:0.5,O1:0.7,O1qqd:0.5 --variants wfe,wfo --members core --xblocks "" --check
$PY e2x_dump_paired.py --platform sarcos --data_dir .. --runs ../runs25/sarcos --out ../results/sarcos --mode grid \
   --members full --xblocks "" --check
$PY e2x_dump_paired.py --platform sim --data_dir ../data_sim --runs ../runs25/sim --out ../results/sim --mode grid \
   --members full --xblocks "" --check
echo ALLDONE
