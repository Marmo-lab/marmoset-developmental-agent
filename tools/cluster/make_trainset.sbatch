#!/bin/bash
# A second labelling set for training (2026-10-29): 40 day clips, each placed around a call the
# learned detector found in the v3_calls corpus, from recordings not in the calibration, held-out
# or test sets. Reads the raw recordings, writes only under $WORK_DIR.
#   sbatch tools/cluster/make_trainset.sbatch
#   rsync -a elsc-login-indirect:marmoset_work/labeling_train2/ labeling/train2/
#SBATCH --job-name=make-trainset
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
OUT="$WORK_DIR/labeling_train2"; mkdir -p "$OUT/labels"
python "$PROJECT_DIR/tools/labels/make_clips.py" "$WORK_DIR/manifests/recordings.csv" -o "$OUT" \
    --day 40 --night 0 --seed 20261029 --prefix tr --around-calls "$WORK_DIR/corpus/v3_calls" \
    --exclude "$WORK_DIR/labeling/clips.csv" "$WORK_DIR/labeling_heldout/clips.csv" "$WORK_DIR/labeling_test/clips.csv"
du -sh "$OUT"
