#!/bin/bash
# Noise-event candidates (tools/analysis/find_noise_events.py). Reads raw recordings and a corpus run;
# writes only $WORK_DIR/noise_events/.
#   J=$(RUN=v5_calls sbatch --parsable -o ~/marmoset_work/logs/%x-%A_%a.out tools/cluster/noise_events.sbatch scan)
#   RUN=v5_calls sbatch --dependency=afterok:$J --array=0 -o ~/marmoset_work/logs/%x-select-%j.out tools/cluster/noise_events.sbatch select
#SBATCH --job-name=noise-events
#SBATCH --partition=elscn.q
#SBATCH --array=0-19
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --requeue
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
STEP="${1:?scan or select}"; shift
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" python tools/analysis/find_noise_events.py "$STEP" "$WORK_DIR/corpus/${RUN:?set RUN=<name>}" \
    "$WORK_DIR/noise_events" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --n-tasks 20 "$@"
