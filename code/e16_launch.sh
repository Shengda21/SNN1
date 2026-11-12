#!/bin/bash
# Start E16-E18 training workers that share one priority-ordered task queue.
# usage: bash e16_launch.sh [n_workers=5] [tasks_file=../logs/e16_tasks.jsonl]
# Only processes matching "python e16_train.py" (started by this script) are counted;
# no other process on the machine is ever signalled. Claims of dead workers are removed
# (liveness checked with kill -0, which sends no signal) so their tasks are re-queued.
N=${1:-5}; TF=${2:-../logs/e16_tasks.jsonl}
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
PAT='python[0-9.]* e16_train.py'
mkdir -p ../logs/e16_claims
for f in ../logs/e16_claims/*.claim; do
  [ -e "$f" ] || continue
  pid=$(cut -d' ' -f1 "$f")
  if ! kill -0 "$pid" 2>/dev/null; then echo "stale claim removed: $f"; rm -f "$f"; fi
done
running=$(pgrep -u "$USER" -f "$PAT" | wc -l)
echo "running workers: $running, target: $N"
for i in $(seq 0 $((N - 1))); do
  [ "$running" -ge "$N" ] && break
  if pgrep -u "$USER" -f "$PAT .*--worker $i\$" > /dev/null; then continue; fi
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup $PY e16_train.py --tasks "$TF" --worker $i >> ../logs/e16_worker$i.log 2>&1 &
  echo "started worker $i pid $!"
  running=$((running + 1))
  sleep 2
done
