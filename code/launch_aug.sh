#!/bin/bash
# E09 training: augmented counterparts of D-M, D-S and the MLP.
# usage: bash launch_aug.sh <platform> <data_dir> <family: all|qqd>
P=$1; DD=$2; FAM=$3; PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
CONDS=$($PY -c "import json;print(','.join(json.load(open('../runs/$P/frozen.json'))['conds']))")
echo "conds=$CONDS family=$FAM"
for s in 0 1 2 3 4; do
  nohup bash -c "for b in dma dsa bna; do $PY run_train.py --platform $P --data_dir $DD --block \$b --seed $s --out ../runs/$P --widths ../runs/$P/frozen.json --conds $CONDS --max_epochs 300 --patience 30 --aug_family $FAM; done; echo AUG_DONE_s$s" > ../logs/${P}_aug_${FAM}_s$s.log 2>&1 &
done
echo launched
