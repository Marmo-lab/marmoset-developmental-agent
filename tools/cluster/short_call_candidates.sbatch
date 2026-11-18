#!/bin/bash
# Short-call candidates for hand labelling (tools/labels/short_call_candidates.py). Reads a corpus run and raw recordings;
# writes only $WORK_DIR/short_call_candidates/.
#   S=$(RUN=v5_calls sbatch --parsable -o ~/marmoset_work/logs/%x-%A_%a.out tools/cluster/short_call_candidates.sbatch scan)
#   RUN=v5_calls sbatch --dependency=afterok:$S --array=0 -o ~/marmoset_work/logs/%x-select-%j.out tools/cluster/short_call_candidates.sbatch select
#SBATCH --job-name=short-cands
#SBATCH --partition=elscn.q
#SBATCH --array=0-19
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
STEP="${1:?scan or select}"
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" python tools/labels/short_call_candidates.py "$STEP" "$WORK_DIR/corpus/${RUN:?}" \
    "$WORK_DIR/short_call_candidates" --work "$WORK_DIR" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --n-tasks 20
