#!/bin/bash
# E28 follow-up runs (after the main runs): admittance with B_d = 50 x torque std (shared feedback, N/O1-M/P1),
# and the true-state-feedback diagnostic at B_d = 20 x torque std (O1-M), one process per seed.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python3}
COMMON="--data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda --runs ../runs25/sim --frozen ../runs25/sim/frozen.json --cfg e27_candidates.json"
for s in 0 1 2 3 4; do
  if [ $s = 0 ]; then O=""; else O="--no_oracle"; fi
  ( OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY e28_admittance.py $COMMON --out ../results/sim/e28_parts/e28_episodes_tfb_s$s.csv \
      --seeds $s --conds O1M --fb_true --no_extras $O \
      --names "D-S:W-Q5,D-S+null:W-Q5,D-S|est:W-Q5,D-M:W-Q5,B-N,B-N|est,B-A,D-S:e21-W3,D-M:e21-W3"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY e28_admittance.py $COMMON --out ../results/sim/e28_parts/e28_episodes_c50_s$s.csv \
      --seeds $s --c_bd 50 --no_extras $O --names "D-S:W-Q5,D-S+null:W-Q5,D-S|est:W-Q5,D-M:W-Q5,B-N,B-A"
  ) > ../logs/e28_followup_s$s.log 2>&1 < /dev/null &
done
wait
echo FOLLOWUP_DONE
