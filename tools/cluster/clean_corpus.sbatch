#!/bin/bash
# Spectral cleaning of a corpus run's calls (tools/corpus/clean_corpus.py). Reads the raw
# recordings, writes only call_audio_clean/ and clean/ under $WORK_DIR/corpus/$RUN.
# Re-submitting resumes: recordings with clean/<id>.json are skipped.
#   RUN=v3_calls sbatch -o ~/marmoset_work/logs/%x-%A_%a.out --array=0-15 tools/cluster/clean_corpus.sbatch \
#       --n-tasks 16 --one-per-date
#SBATCH --job-name=clean-corpus
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=08:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN="${RUN:?set RUN=<name>, e.g. RUN=v3_calls}"
python "$PROJECT_DIR/tools/corpus/clean_corpus.py" "$WORK_DIR/corpus/$RUN" --task-index "${SLURM_ARRAY_TASK_ID:-0}" "$@"
