#!/bin/bash
# Repertoire-figure candidates (tools/analysis/figure_candidates.py). Reads a corpus run and its raw recordings;
# writes only $WORK_DIR/figure_candidates/.
#   RUN=v5_calls sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/figure_candidates.sbatch
#SBATCH --job-name=figure-candidates
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" python tools/analysis/figure_candidates.py "$WORK_DIR/corpus/${RUN:?set RUN=<name>}" "$WORK_DIR/figure_candidates"
