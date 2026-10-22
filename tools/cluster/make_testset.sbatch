#!/bin/bash
# Build the blind test set (2026-10-22): 15 day clips from recordings and, where possible,
# dates not in the calibration or held-out sets, plus the raw 4-capsule audio under each
# clip for scoring. Reads the raw recordings, writes only under $WORK_DIR.
#   sbatch tools/cluster/make_testset.sbatch
#   rsync -a elsc-login-direct:marmoset_work/labeling_test/ labeling/test/
#SBATCH --job-name=make-testset
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
OUT="$WORK_DIR/labeling_test"; mkdir -p "$OUT/labels"
python "$PROJECT_DIR/tools/labels/make_clips.py" "$WORK_DIR/manifests/recordings.csv" -o "$OUT" \
    --day 15 --night 0 --seed 20261022 --prefix ts \
    --exclude "$WORK_DIR/labeling/clips.csv" "$WORK_DIR/labeling_heldout/clips.csv"
cp "$PROJECT_DIR/docs/LABELING.md" "$WORK_DIR/labeling/guide.png" "$OUT/"
python "$PROJECT_DIR/tools/analysis/score_detectors.py" extract "$OUT/clips.csv" --out "$OUT/audio48"
du -sh "$OUT"
