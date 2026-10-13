#!/bin/bash
# E14: replay the first epochs of every D-M member (same seed, initialization and batch order),
# keep every epoch state, and save the early checkpoints that bracket the straight-through
# member's development error. Then run the analysis. Use throttle.sh to limit concurrency.
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
for P in sarcos sim; do
  if [ "$P" = sarcos ]; then DD=..; else DD=../data_sim; fi
  CONDS=$($PY -c "import json;print(','.join(json.load(open('../runs/$P/frozen.json'))['conds']))")
  A="--platform $P --data_dir $DD --block traj --out ../runs/$P --widths ../runs/$P/frozen.json --conds $CONDS"
  for s in 0 1 2 3 4; do
    if [ "$P" = sarcos ]; then
      # the straight-through member needs about a hundred epochs on SARCOS to reach its best state
      nohup bash -c "$PY run_train.py $A --seed $s --only ste --traj_epochs 125; $PY run_train.py $A --seed $s --traj_epochs 25; echo E14_TRAIN_DONE_s$s" > ../logs/${P}_e14_s$s.log 2>&1 &
    else
      nohup bash -c "$PY run_train.py $A --seed $s --traj_epochs 25; echo E14_TRAIN_DONE_s$s" > ../logs/${P}_e14_s$s.log 2>&1 &
    fi
  done
done
echo launched
