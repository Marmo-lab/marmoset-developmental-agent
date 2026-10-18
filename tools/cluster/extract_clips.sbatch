#!/bin/bash
# Cut the raw 4-capsule audio under each labelled clip (plus context) for detector
# scoring. Reads the raw recordings, writes only under $WORK_DIR.
#
#   sbatch tools/cluster/extract_clips.sbatch            # clips.csv from make_labeling
#   rsync -a elsc-login-direct:marmoset_work/clip_audio/ labeling/clip_audio/
#
#SBATCH --job-name=extract-clips
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"

CLIPS="${1:-$WORK_DIR/labeling/clips.csv}"
python "$PROJECT_DIR/tools/analysis/score_detectors.py" extract "$CLIPS" --out "$WORK_DIR/clip_audio"
