#!/bin/bash
# Run all jobs of a work directory prepared by prepare_inputs.py.
# Usage: run_all.sh WORK_DIR [N_PARALLEL]   (default 10 parallel single-thread runs)
# Start it detached on a remote host, e.g.
#   nohup setsid analysis/dem/run_all.sh WORK_DIR 10 > WORK_DIR/run_all.log 2>&1 < /dev/null &
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(cd "$1" && pwd) || exit 1
NP=${2:-10}
: > "$WORK/done.log"
nice -n 10 xargs -P "$NP" -I{} "$HERE/run_one.sh" "$WORK" {} < "$WORK/jobs.txt"
echo "finished: $(grep -c '^0 ' "$WORK/done.log") of $(wc -l < "$WORK/jobs.txt") runs exited with 0"
