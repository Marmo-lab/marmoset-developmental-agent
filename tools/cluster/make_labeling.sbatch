#!/bin/bash
# Build the hand-labelling package: stratified clips + visual guide.
#   sbatch tools/cluster/make_labeling.sbatch
#SBATCH --job-name=make-labeling
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
OUT="$WORK_DIR/labeling"; mkdir -p "$OUT/labels"
python "$PROJECT_DIR/tools/labels/make_clips.py" "$WORK_DIR/manifests/recordings.csv" -o "$OUT" --day 24 --night 6
python "$PROJECT_DIR/tools/labels/make_guide.py" "$RAW_RECORDINGS/Rec_2021_05_03/210503_001.WAV" -o "$OUT/guide.png"
cp "$PROJECT_DIR/LABELING.md" "$OUT/"
du -sh "$OUT"
