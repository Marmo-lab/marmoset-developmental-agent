#!/bin/bash
# Array job: run the |STFT| level audit over a list of raw recordings.
#
#   bash tools/cluster/sample_files.sh 6 > ~/marmoset_work/level_files.txt
#   sbatch --array=1-$(wc -l < ~/marmoset_work/level_files.txt) \
#          tools/cluster/level_audit.sbatch ~/marmoset_work/level_files.txt
#
#SBATCH --job-name=level-audit
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail

# sbatch copies this script to a spool dir, so $(dirname "$0") is NOT the repo.
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"

LIST="${1:?usage: sbatch level_audit.sbatch <file-list>}"
WAV="$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$LIST")"
[ -n "$WAV" ] || { echo "no entry $SLURM_ARRAY_TASK_ID in $LIST" >&2; exit 1; }

OUT="$WORK_DIR/analysis"
mkdir -p "$OUT"
STEM="$(basename "$(dirname "$WAV")")_$(basename "$WAV" .WAV)"

echo "task $SLURM_ARRAY_TASK_ID -> $WAV"
python "$PROJECT_DIR/tools/analysis/level_audit.py" "$WAV" \
    --windows 40 --window-s 15 --json-out "$OUT/level_${STEM}.json"
