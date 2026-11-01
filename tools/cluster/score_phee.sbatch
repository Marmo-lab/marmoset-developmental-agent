#!/bin/bash
# Phee-database score of the production detector (tools/analysis/score_phee.py, detect step).
# Reads the lab's phee database and experiment recordings, writes only under $WORK_DIR/phee_score.
# Re-submitting resumes: experiments with exps/<name>.json are skipped.
#   sbatch -o ~/marmoset_work/logs/%x-%A_%a.out --array=0-5 tools/cluster/score_phee.sbatch --n-tasks 6
#   python tools/analysis/score_phee.py report ~/marmoset_work/phee_score     # after all tasks
#SBATCH --job-name=score-phee
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
DB="${DB:-/ems/elsc-labs/omer-d/Omer-Lab-Shared/data/analysis/phee_database/pheeDB_15122024_164547.csv}"
python tools/analysis/score_phee.py detect "$DB" "$WORK_DIR/phee_score" --task-index "${SLURM_ARRAY_TASK_ID:-0}" "$@"
