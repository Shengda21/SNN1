#!/bin/bash
# Moment dumps for the E16-E18 members: tag w16 (w16_* incl. ultra-wide, with D-M and D-S of the same seed),
# tag w16d (detached-reset w16d_*), tag e18 (w18_*, w17_*, w16c_* as single models; original conditions only).
# Pass --overwrite to replace existing files.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python3}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
OW="$1"
set -x
for TAG in w16 w16d e18; do
  case $TAG in
    w16) M='glob:w16_*,block:dm,block:ds';;
    w16d) M='glob:w16d_*';;
    e18) M='glob:w18_*,glob:w17_*,glob:w16c_*';;
  esac
  $PY e2x_dump_paired.py --platform sim --data_dir ../data_sim --runs ../runs/sim,../runs25/sim --out ../results/sim --mode orig \
     --conds N,O1:0.35,O1:0.7,O1:1.0,O2B:0.7 --variants raw --members "$M" --xblocks "" --seg 1 --tag $TAG --check $OW
  $PY e2x_dump_paired.py --platform sarcos --data_dir .. --runs ../runs/sarcos,../runs25/sarcos --out ../results/sarcos --mode orig \
     --conds N,O1:0.25,O1:0.5,O1:0.7,O1qqd:0.5 --variants raw --members "$M" --xblocks "" --tag $TAG --check $OW
  if [ $TAG != e18 ]; then
    $PY e2x_dump_paired.py --platform sarcos --data_dir .. --runs ../runs/sarcos,../runs25/sarcos --out ../results/sarcos --mode grid \
       --members "$M" --xblocks "" --tag $TAG --check $OW
    $PY e2x_dump_paired.py --platform sim --data_dir ../data_sim --runs ../runs/sim,../runs25/sim --out ../results/sim --mode grid \
       --members "$M" --xblocks "" --tag $TAG --check $OW
  fi
done
echo ALLDONE
