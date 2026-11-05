#!/bin/bash
# Short high call candidates (tools/analysis/find_short_high.py). Reads raw recordings, the corpus run
# and a synthetic set; writes only $WORK_DIR/short_high/.
#   SET=v1 sbatch --array=0 -o ~/marmoset_work/logs/%x-selftest-%j.out tools/cluster/short_high.sbatch selftest
#   J=$(RUN=v4_calls sbatch --parsable -o ~/marmoset_work/logs/%x-%A_%a.out tools/cluster/short_high.sbatch scan)
#   RUN=v4_calls sbatch --dependency=afterok:$J --array=0 -o ~/marmoset_work/logs/%x-select-%j.out tools/cluster/short_high.sbatch select
#SBATCH --job-name=short-high
#SBATCH --partition=elscn.q
#SBATCH --array=0-19
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --requeue
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
STEP="${1:?selftest, scan or select}"; shift
OUT="$WORK_DIR/short_high"
cd "$PROJECT_DIR"
if [ "$STEP" = selftest ]; then
    SRC="$WORK_DIR/synth/sets/${SET:?set SET=<name>}"
else
    SRC="$WORK_DIR/corpus/${RUN:?set RUN=<name>, e.g. RUN=v4_calls}"
fi
PYTHONPATH="$PROJECT_DIR" python tools/analysis/find_short_high.py "$STEP" "$SRC" "$OUT" \
    --task-index "${SLURM_ARRAY_TASK_ID:-0}" --n-tasks 20 "$@"
