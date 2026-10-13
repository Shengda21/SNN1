#!/bin/bash
# E09-E12 post-processing: waits for the augmented training blocks, then dumps the
# front-end predictions, runs the analysis and the forward-response measurement.
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
wait_done() { while true; do c=$(grep -l AUG_DONE ../logs/$1_aug_$2_s*.log 2>/dev/null | wc -l); [ "$c" -ge 5 ] && break; sleep 60; done; }
wait_done sarcos all
echo "sarcos aug training done $(date)"
$PY frontend_preds.py --platform sarcos --data_dir .. --runs ../runs/sarcos --frozen ../runs/sarcos/frozen.json --out ../results/sarcos/e11_frontend_geometry.json > ../logs/frontend_sarcos.log 2>&1
$PY analysis_aug.py --platform sarcos --data_dir .. --runs ../runs/sarcos --frozen ../runs/sarcos/frozen.json --out ../results/sarcos > ../logs/analysis_aug_sarcos.log 2>&1
$PY forward_response.py --platform sarcos --data_dir .. --runs ../runs/sarcos --frozen ../runs/sarcos/frozen.json --out ../results/sarcos/forward_response_aug.json > ../logs/fr_aug_sarcos.log 2>&1
echo "sarcos aug analysis done $(date)"
wait_done sim all
wait_done sim qqd
echo "sim aug training done $(date)"
$PY frontend_preds.py --platform sim --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim/e11_frontend_geometry.json > ../logs/frontend_sim.log 2>&1
$PY analysis_aug.py --platform sim --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim > ../logs/analysis_aug_sim.log 2>&1
$PY forward_response.py --platform sim --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim/forward_response_aug.json > ../logs/fr_aug_sim.log 2>&1
echo "sim aug analysis done $(date)"
echo "AUG_PIPELINE_DONE $(date)" > ../logs/aug_pipeline_done.txt
