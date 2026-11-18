#!/bin/bash
# Detectors on the held-out short-call labels (tools/analysis/score_short_labels.py). Reads the labels and raw recordings;
# writes only $WORK_DIR/detector/short_labels/.
#   sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/score_short_labels.sbatch --model v2a=models/call_detector_v2a.pt:0 \
#       --model v3=$WORK_DIR/detector/v3/call_detector_v3.pt:162 [...]
#SBATCH --job-name=score-short
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" python tools/analysis/score_short_labels.py score "${LABELS:-$WORK_DIR/short_call_candidates}" \
    "$WORK_DIR/detector/short_labels" "$@"
