#!/bin/bash
# Run one soot-dem restructuring simulation on a single thread.
# Usage: run_one.sh WORK_DIR RUN_DIR   (RUN_DIR relative to WORK_DIR, one line of jobs.txt)
# Appends "<exit code> <RUN_DIR>" to WORK_DIR/done.log.
WORK=$(cd "$1" && pwd) || exit 1
cd "$WORK/$2" || exit 1
rm -rf run
OMP_NUM_THREADS=1 "$WORK/bin/restructuring" restructuring_thin_coating_input.xml > stdout.txt 2>&1
echo "$? $2" >> "$WORK/done.log"
