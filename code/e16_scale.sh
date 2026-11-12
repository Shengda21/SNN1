#!/bin/bash
# Wait until the E28 admittance simulation (CPU processes of another agent) has finished,
# then raise the number of E16-E18 training workers to N (default 7) via e16_launch.sh.
# Only observes other processes (pgrep); never signals them.
N=${1:-7}
cd "$(dirname "$0")" || exit 1
while pgrep -u "$USER" -f 'python[0-9.]* e28_admittance' > /dev/null; do sleep 60; done
echo "$(date) e28 finished, scaling to $N workers"
bash e16_launch.sh "$N"
