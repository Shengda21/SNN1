#!/bin/bash
# Freeze the development decisions, then train the formal blocks (D-M, D-S, D-T, B-L, B-A, B-N)
# of all five seeds on one platform.
# usage: bash launch_formal.sh <platform> <data_dir>
P=$1; DD=$2; PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
$PY dev_decisions.py --platform $P --data_dir $DD --runs ../runs/$P --out ../runs/$P/frozen.json || exit 1
CONDS=$($PY -c "import json;print(','.join(json.load(open('../runs/$P/frozen.json'))['conds']))")
echo "conds=$CONDS"
for s in 0 1 2 3 4; do
  nohup bash -c "for b in dm ds dt bl ba bn; do $PY run_train.py --platform $P --data_dir $DD --block \$b --seed $s --out ../runs/$P --widths ../runs/$P/frozen.json --conds $CONDS --max_epochs 300 --patience 30; done; echo FORMAL_DONE_s$s" > ../logs/${P}_formal_s$s.log 2>&1 &
done
echo launched
