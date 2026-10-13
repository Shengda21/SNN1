#!/bin/bash
# Keep at most N training processes running; the others are paused with SIGSTOP and
# resumed in PID order as slots free up. N is re-read from ../logs/throttle_n on every
# pass, so the load can be changed without restarting anything. Exits when no training
# process is left (and resumes everything on exit).
F=${THROTTLE_FILE:-$(dirname "$0")/../logs/throttle_n}
PAT='python[0-9.]* run_train.py'
resume_all() { for p in $(pgrep -u "$USER" -f "$PAT"); do kill -CONT "$p" 2>/dev/null; done; }
trap 'resume_all; exit 0' INT TERM
empty=0
while true; do
  N=$(cat "$F" 2>/dev/null || echo 4)
  mapfile -t P < <(pgrep -u "$USER" -f "$PAT" | sort -n)
  if [ ${#P[@]} -eq 0 ]; then
    empty=$((empty + 1)); [ "$empty" -ge 4 ] && break
    sleep 15; continue
  fi
  empty=0; i=0
  for p in "${P[@]}"; do
    if [ "$i" -lt "$N" ]; then kill -CONT "$p" 2>/dev/null; else kill -STOP "$p" 2>/dev/null; fi
    i=$((i + 1))
  done
  sleep 10
done
echo "throttle finished $(date)"
