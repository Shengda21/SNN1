#!/bin/bash
# E28 follow-up: admittance with B_d = 50 x torque std (M_d = 0.1 B_d), human model identical to the
# main runs (K_h = 100 x torque std, B_h = 20/3 x torque std): kh_ratio = 2, bh_ratio = 2/15.
# (A first attempt that scaled K_h = 5 B_d with B_d was unstable in every intent episode and was discarded.)
cd "$(dirname "$0")" || exit 1
PY=${PY:-python3}
COMMON="--data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda --runs ../runs25/sim --frozen ../runs25/sim/frozen.json --cfg e27_candidates.json"
for s in 0 1 2 3 4; do
  if [ $s = 0 ]; then O=""; else O="--no_oracle"; fi
  ( OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY e28_admittance.py $COMMON --out ../results/sim/e28_parts/e28_episodes_c50_s$s.csv \
      --seeds $s --c_bd 50 --kh_ratio 2.0 --bh_ratio 0.13333333 --no_extras $O --names "D-S:W-Q5,D-S+null:W-Q5,D-S|est:W-Q5,D-M:W-Q5,B-N,B-A"
  ) > ../logs/e28_c50_s$s.log 2>&1 < /dev/null &
done
wait
echo C50_DONE
