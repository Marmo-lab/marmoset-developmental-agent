#!/bin/bash
# Print the status and (when ready) the results of the level-audit array job.
# Safe to run repeatedly; does nothing destructive.
#
#   cd ~/marmoset-developmental-agent && bash tools/cluster/show_levels.sh [JOBID]
#
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source tools/cluster/env.sh

JOB="${1:-}"
N_DONE=$(ls "$WORK_DIR"/analysis/level_*.json 2>/dev/null | wc -l)
N_WANT=$(wc -l < "$WORK_DIR/level_files.txt" 2>/dev/null || echo "?")

echo "=== level audit: $N_DONE / $N_WANT files analysed ==="
if [ -n "$JOB" ]; then
    RUNNING=$(squeue -h -u "$USER" -r -j "$JOB" -o "%T" 2>/dev/null | sort | uniq -c)
    if [ -n "$RUNNING" ]; then
        echo "still queued/running:"; echo "$RUNNING"
    else
        echo "job $JOB finished. Final states:"
        sacct -j "$JOB" --format=State -n 2>/dev/null | grep -v '^ *$' | sort | uniq -c
    fi
fi

if [ "$N_DONE" -gt 0 ]; then
    echo
    python tools/analysis/aggregate_levels.py "$WORK_DIR/analysis"
else
    echo "No results yet -- nothing to aggregate."
fi
