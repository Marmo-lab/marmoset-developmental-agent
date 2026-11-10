#!/bin/bash
# Quality check of one raw recording folder (tools/analysis/recording_qc.py). Reads the folder read-only;
# writes only $WORK_DIR/recording_qc/<FOLDER>/.
#   J=$(FOLDER=Rec_2026_11_8-10 sbatch --parsable -o ~/marmoset_work/logs/%x-%A_%a.out tools/cluster/recording_qc.sbatch scan)
#   FOLDER=Rec_2026_11_8-10 sbatch --dependency=afterok:$J --array=0 -o ~/marmoset_work/logs/%x-summary-%j.out tools/cluster/recording_qc.sbatch summary
#SBATCH --job-name=recording-qc
#SBATCH --partition=elscn.q
#SBATCH --array=0-29
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --requeue
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
STEP="${1:?scan or summary}"
OUT="$WORK_DIR/recording_qc/${FOLDER:?set FOLDER=<Rec_... folder>}"
cd "$PROJECT_DIR"
if [ "$STEP" = scan ]; then
    PYTHONPATH="$PROJECT_DIR" python tools/analysis/recording_qc.py scan "$RAW_RECORDINGS/$FOLDER" "$OUT" \
        --task "${SLURM_ARRAY_TASK_ID:-0}" --n-tasks 30
else
    PYTHONPATH="$PROJECT_DIR" python tools/analysis/recording_qc.py summary "$OUT"
fi
