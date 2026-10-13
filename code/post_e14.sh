#!/bin/bash
# Wait for the E14 trajectory replays, then run the underfitting analysis on both platforms.
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
while true; do
  c=$(grep -l E14_TRAIN_DONE ../logs/sarcos_e14_s*.log ../logs/sim_e14_s*.log 2>/dev/null | wc -l)
  [ "$c" -ge 10 ] && break
  sleep 30
done
echo "E14 replays done $(date)"
$PY analysis_underfit.py --platform sarcos --data_dir .. --runs ../runs/sarcos --frozen ../runs/sarcos/frozen.json --out ../results/sarcos > ../logs/e14_analysis_sarcos.log 2>&1
$PY analysis_underfit.py --platform sim --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim > ../logs/e14_analysis_sim.log 2>&1
echo "E14_PIPELINE_DONE $(date)" > ../logs/e14_done.txt
