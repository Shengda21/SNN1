#!/bin/bash
# E15 driver: position, flip and stationarity for one platform.
# usage: bash e15_run.sh <platform> [runs] [members-glob] [suffix]
set -u
PLAT=$1
RUNS=${2:-../runs25/$PLAT}
MEM=${3:-*}
SUF=${4:-}
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
OUT=../results/$PLAT
mkdir -p $OUT ../logs
EXTRA=""
if [ -n "$SUF" ]; then EXTRA="--no_traj_check"; fi
$PY e15_position.py --platform $PLAT --runs $RUNS --members "$MEM" --out $OUT --suffix "$SUF" $EXTRA
$PY e15_flip.py --platform $PLAT --runs $RUNS --members "$MEM" --out $OUT --suffix "$SUF"
$PY e15_stationarity.py --platform $PLAT --runs $RUNS --members "$MEM" --out $OUT --suffix "$SUF"
echo "E15_RUN_DONE $PLAT $SUF"
