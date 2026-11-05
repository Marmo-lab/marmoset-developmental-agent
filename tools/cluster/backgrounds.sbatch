#!/bin/bash
# Background candidates (tools/synth/backgrounds.py): scan call-free night recordings (and a day
# reference) as an array, then select. Reads raw recordings; writes only $WORK_DIR/synth/backgrounds/.
#   J=$(RUN=v4_calls sbatch --parsable -o ~/marmoset_work/logs/%x-%A_%a.out tools/cluster/backgrounds.sbatch scan)
#   RUN=v4_calls sbatch --dependency=afterok:$J --array=0 -o ~/marmoset_work/logs/%x-select-%j.out tools/cluster/backgrounds.sbatch select
#SBATCH --job-name=backgrounds
#SBATCH --partition=elscn.q
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --requeue
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
RUN="${RUN:?set RUN=<name>, e.g. RUN=v4_calls}"
STEP="${1:?scan or select}"; shift
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" python tools/synth/backgrounds.py "$STEP" "$WORK_DIR/corpus/$RUN" "$WORK_DIR/synth/backgrounds" \
    --task-index "${SLURM_ARRAY_TASK_ID:-0}" --n-tasks 20 "$@"
