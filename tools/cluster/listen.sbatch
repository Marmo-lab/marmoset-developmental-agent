#!/bin/bash
# Stitch detected calls from fresh daytime audio into one WAV for listening.
#
#   sbatch tools/cluster/listen.sbatch
#   rsync -a elsc-login-direct:marmoset_work/listen/ labeling/listen/
#
#SBATCH --job-name=listen
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"

python "$PROJECT_DIR/tools/analysis/listen_detections.py" "$WORK_DIR/manifests/recordings.csv" \
    --exclude "$WORK_DIR/labeling/clips.csv" --out "$WORK_DIR/listen" "$@"
