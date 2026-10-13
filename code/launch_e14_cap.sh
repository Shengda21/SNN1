#!/bin/bash
# E14, second control: reduced-capacity members (hidden 4, 8, 16) of the development-selected
# surrogate, trained under the unchanged protocol; then the underfitting analysis is rerun.
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
for P in sarcos sim; do
  if [ "$P" = sarcos ]; then DD=..; else DD=../data_sim; fi
  CONDS=$($PY -c "import json;print(','.join(json.load(open('../runs/$P/frozen.json'))['conds']))")
  for s in 0 1 2 3 4; do
    nohup bash -c "$PY run_train.py --platform $P --data_dir $DD --block cap --hidden 4,8,16 --seed $s --out ../runs/$P --widths ../runs/$P/frozen.json --conds $CONDS --max_epochs 300 --patience 30; echo E14_CAP_DONE_s$s" > ../logs/${P}_e14cap_s$s.log 2>&1 &
  done
done
(nohup bash -c "while true; do c=\$(grep -l E14_CAP_DONE ../logs/sarcos_e14cap_s*.log ../logs/sim_e14cap_s*.log 2>/dev/null | wc -l); [ \"\$c\" -ge 10 ] && break; sleep 30; done; $PY analysis_underfit.py --platform sarcos --data_dir .. --runs ../runs/sarcos --frozen ../runs/sarcos/frozen.json --out ../results/sarcos > ../logs/e14_analysis_sarcos.log 2>&1; $PY analysis_underfit.py --platform sim --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim > ../logs/e14_analysis_sim.log 2>&1; echo E14_CAP_PIPELINE_DONE > ../logs/e14_cap_done.txt" > ../logs/post_e14_cap.log 2>&1 &)
echo launched
