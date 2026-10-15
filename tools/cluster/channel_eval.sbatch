#!/bin/bash
# Array job: five-strategy capsule evaluation over a list of raw recordings.
#
#   sbatch --array=1-$(wc -l < ~/marmoset_work/level_files.txt)%20 \
#          tools/cluster/channel_eval.sbatch ~/marmoset_work/level_files.txt
#   python tools/analysis/channel_eval.py aggregate ~/marmoset_work/channel_eval
#
#SBATCH --job-name=channel-eval
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"

LIST="${1:?usage: sbatch channel_eval.sbatch <file-list>}"
WAV="$(sed -n "${SLURM_ARRAY_TASK_ID:-1}p" "$LIST")"
OUT="$WORK_DIR/channel_eval"; mkdir -p "$OUT"
STEM="$(basename "$(dirname "$WAV")")_$(basename "$WAV" .WAV)"
python "$PROJECT_DIR/tools/analysis/channel_eval.py" run "$WAV" --json-out "$OUT/chan_${STEM}.json"
