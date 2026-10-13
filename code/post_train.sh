#!/bin/bash
# Post-processing of the formal blocks on both platforms: frozen predictions, open-loop analysis,
# and forward-response measurement.
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
wait_done() { P=$1; while true; do c=$(grep -l FORMAL_DONE ../logs/${P}_formal_s*.log 2>/dev/null | wc -l); [ "$c" -ge 5 ] && break; sleep 120; done; }
conds() { $PY -c "import json;print(','.join(json.load(open('../runs/$1/frozen.json'))['conds']))"; }
wait_done sarcos
echo "sarcos training done $(date)"
$PY dump_preds.py --platform sarcos --data_dir .. --runs ../runs/sarcos --conds $(conds sarcos) > ../logs/dump_sarcos.log 2>&1
$PY analysis_openloop.py --platform sarcos --data_dir .. --runs ../runs/sarcos --frozen ../runs/sarcos/frozen.json --out ../results/sarcos > ../logs/analysis_sarcos.log 2>&1
$PY forward_response.py --platform sarcos --data_dir .. --runs ../runs/sarcos --frozen ../runs/sarcos/frozen.json --out ../results/sarcos/forward_response.json > ../logs/fr_sarcos.log 2>&1
echo "sarcos analysis done $(date)"
wait_done sim
echo "sim training done $(date)"
$PY dump_preds.py --platform sim --data_dir ../data_sim --runs ../runs/sim --conds $(conds sim) > ../logs/dump_sim.log 2>&1
$PY analysis_openloop.py --platform sim --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim > ../logs/analysis_sim.log 2>&1
$PY forward_response.py --platform sim --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim/forward_response.json > ../logs/fr_sim.log 2>&1
echo "sim analysis done $(date)"
$PY run_closedloop.py --data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim > ../logs/closedloop.log 2>&1
echo "closed loop done $(date)"
mkdir -p ../results/runtime
$PY profile_runtime.py --out ../results/runtime/runtime.json > ../logs/runtime.log 2>&1
echo "PIPELINE_DONE $(date)" > ../logs/pipeline_done.txt
