#!/bin/bash
# Complete pass of the E09-E12 analysis after every augmented block (including the networks
# selected on the augmented development set) and post_aug.sh have finished.
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
while true; do
  a=$(grep -l AUG_DONE ../logs/*_aug_all_s*.log ../logs/*_aug_qqd_s*.log 2>/dev/null | wc -l)
  b=$(grep -l AUG2_DONE ../logs/*_aug2_*_s*.log 2>/dev/null | wc -l)
  [ "$a" -ge 15 ] && [ "$b" -ge 15 ] && [ -f ../logs/aug_pipeline_done.txt ] && break
  sleep 30
done
echo "all augmented training done $(date)"
for P in sarcos sim; do
  if [ "$P" = sarcos ]; then DD=..; else DD=../data_sim; fi
  $PY frontend_preds.py --platform $P --data_dir $DD --runs ../runs/$P --frozen ../runs/$P/frozen.json --out ../results/$P/e11_frontend_geometry.json > ../logs/rerun_aug_frontend_$P.log 2>&1
  $PY analysis_aug.py --platform $P --data_dir $DD --runs ../runs/$P --frozen ../runs/$P/frozen.json --out ../results/$P > ../logs/rerun_aug_analysis_$P.log 2>&1
  $PY forward_response.py --platform $P --data_dir $DD --runs ../runs/$P --frozen ../runs/$P/frozen.json --out ../results/$P/forward_response_aug.json > ../logs/rerun_aug_fr_$P.log 2>&1
  echo "$P final analysis done $(date)"
done
echo "RERUN_AUG_DONE $(date)" > ../logs/rerun_aug_done.txt
