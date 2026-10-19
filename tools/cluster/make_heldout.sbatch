#!/bin/bash
# Build the held-out labelling set: 15 day clips from recordings not used for calibration.
#   sbatch tools/cluster/make_heldout.sbatch
#SBATCH --job-name=make-heldout
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
OUT="$WORK_DIR/labeling_heldout"; mkdir -p "$OUT/labels"
python "$PROJECT_DIR/tools/labels/make_clips.py" "$WORK_DIR/manifests/recordings.csv" -o "$OUT" \
    --day 15 --night 0 --seed 20261020 --exclude "$WORK_DIR/labeling/clips.csv" --prefix ho
cp "$PROJECT_DIR/LABELING.md" "$WORK_DIR/labeling/guide.png" "$OUT/"
du -sh "$OUT"
