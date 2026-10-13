#!/bin/bash
# E09, second block: the augmented strong-surrogate network and the augmented MLP with model
# selection on the augmented development set (B-A+, B-NA+).
# usage: bash launch_aug2.sh <platform> <data_dir> <family: all|qqd>
P=$1; DD=$2; FAM=$3; PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
CONDS=$($PY -c "import json;print(','.join(json.load(open('../runs/$P/frozen.json'))['conds']))")
echo "conds=$CONDS family=$FAM"
for s in 0 1 2 3 4; do
  nohup bash -c "$PY run_train.py --platform $P --data_dir $DD --block dma --only strong --dev_sel aug --seed $s --out ../runs/$P --widths ../runs/$P/frozen.json --conds $CONDS --max_epochs 300 --patience 30 --aug_family $FAM; $PY run_train.py --platform $P --data_dir $DD --block bna --dev_sel aug --seed $s --out ../runs/$P --widths ../runs/$P/frozen.json --conds $CONDS --max_epochs 300 --patience 30 --aug_family $FAM; echo AUG2_DONE_s$s" > ../logs/${P}_aug2_${FAM}_s$s.log 2>&1 &
done
echo launched
