#!/bin/bash
# Call map (tools/analysis/call_map.py): sample calls, embed them in 2-D, write RUN/call_map/map.json,
# which the call library serves at /map. Reads the run's call audio; writes only RUN/call_map/.
#   RUN=v4_calls sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/call_map.sbatch [--n 30000] [--version clean]
#SBATCH --job-name=call-map
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
RUN="${RUN:?set RUN=<name>, e.g. RUN=v4_calls}"
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" python tools/analysis/call_map.py "$WORK_DIR/corpus/$RUN" "$@"
