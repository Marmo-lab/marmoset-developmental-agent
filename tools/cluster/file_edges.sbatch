#!/bin/bash
# Detections at recording-file edges (tools/analysis/file_edges.py). Reads a corpus run and raw recordings;
# writes only $WORK_DIR/file_edges/.
#   S=$(RUN=v5_calls sbatch --parsable --array=0 -o ~/marmoset_work/logs/%x-stats-%j.out tools/cluster/file_edges.sbatch stats)
#   C=$(RUN=v5_calls sbatch --parsable --dependency=afterok:$S -o ~/marmoset_work/logs/%x-%A_%a.out tools/cluster/file_edges.sbatch context)
#   sbatch --dependency=afterok:$C --array=0 -o ~/marmoset_work/logs/%x-summary-%j.out tools/cluster/file_edges.sbatch summary
#SBATCH --job-name=file-edges
#SBATCH --partition=elscn.q
#SBATCH --array=0-19
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
STEP="${1:?stats, context or summary}"
OUT="$WORK_DIR/file_edges"
cd "$PROJECT_DIR"
case "$STEP" in
    stats) PYTHONPATH="$PROJECT_DIR" python tools/analysis/file_edges.py stats "$WORK_DIR/corpus/${RUN:?}" "$OUT" ;;
    context) PYTHONPATH="$PROJECT_DIR" python tools/analysis/file_edges.py context "$WORK_DIR/corpus/${RUN:?}" "$OUT" \
                 --task "${SLURM_ARRAY_TASK_ID:-0}" --n-tasks 20 ;;
    summary) PYTHONPATH="$PROJECT_DIR" python tools/analysis/file_edges.py summary "$OUT" ;;
esac
