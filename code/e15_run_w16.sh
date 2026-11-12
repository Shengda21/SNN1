#!/bin/bash
# E15 on the new E16 members (w16_*): position + flip for both platforms, then stationarity.
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
for P in sim sarcos; do
  $PY e15_position.py --platform $P --runs ../runs/$P --members 'w16_*' --out ../results/$P --suffix _w16 --no_traj_check
  $PY e15_flip.py --platform $P --runs ../runs/$P --members 'w16_*' --out ../results/$P --suffix _w16
  $PY e15_summary.py --platform $P --suffix _w16
done
echo E15_W16_POSFLIP_DONE
for P in sim sarcos; do
  $PY e15_stationarity.py --platform $P --runs ../runs/$P --members 'w16_*' --out ../results/$P --suffix _w16
  $PY e15_summary.py --platform $P --suffix _w16
done
echo E15_W16_ALL_DONE
