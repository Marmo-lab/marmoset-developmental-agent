#!/bin/bash
# Score the detector on a synthetic set (tools/synth/score.py): detections as an array, then the score.
# Reads the set; writes only SET/detections/ and SET/scores.*.
#   J=$(SET=v1 sbatch --parsable -o ~/marmoset_work/logs/%x-%A_%a.out tools/cluster/synth_score.sbatch detect)
#   SET=v1 sbatch --dependency=afterok:$J --array=0 -o ~/marmoset_work/logs/%x-score-%j.out tools/cluster/synth_score.sbatch score
#   (another checkpoint: append --model PATH --name NAME to both)
#SBATCH --job-name=synth-score
#SBATCH --partition=elscn.q
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --requeue
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
STEP="${1:?detect or score}"; shift
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" python tools/synth/score.py "$STEP" "$WORK_DIR/synth/sets/${SET:?set SET=<name>}" \
    --task-index "${SLURM_ARRAY_TASK_ID:-0}" --n-tasks 20 "$@"
