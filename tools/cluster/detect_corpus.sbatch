#!/bin/bash
# Production front-end over the raw corpus -> call catalogue (tools/corpus/detect_corpus.py).
# Reads the raw recordings, writes only under $WORK_DIR/corpus/$RUN. Re-submitting the same
# command resumes: recordings with files/<id>.json are skipped.
#
#   # pilot: the recordings behind the labelled clips (paths listed one per line)
#   RUN=pilot sbatch -o ~/marmoset_work/logs/%x-%A_%a.out --array=0-3 tools/cluster/detect_corpus.sbatch \
#       --n-tasks 4 --only ~/marmoset_work/manifests/pilot_paths.txt
#   # production (v3_calls): full corpus, day only, learned detector, each recording's calls as one
#   # 48 kHz FLAC. 136 tasks of ~20 recordings, at most 16 at once (shared file system)
#   RUN=v3_calls sbatch -o ~/marmoset_work/logs/%x-%A_%a.out --array=0-135%16 tools/cluster/detect_corpus.sbatch \
#       --n-tasks 136 --config config/corpus_day_learned.yaml --export-calls
#   # then, on a compute node: merge into calls.tsv, then clean (clean_corpus.sbatch)
#   python tools/corpus/merge_calls.py $WORK_DIR/corpus/$RUN
#   # GmSLM comparison arm (v2_day): --config config/corpus_day.yaml --export-audio, then merge_corpus.py
#
#SBATCH --job-name=detect-corpus
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=08:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # one core per task

RUN="${RUN:?set RUN=<name>, e.g. RUN=pilot}"
python "$PROJECT_DIR/tools/corpus/detect_corpus.py" "$WORK_DIR/manifests/recordings.csv" \
    "$WORK_DIR/corpus/$RUN" --task-index "${SLURM_ARRAY_TASK_ID:-0}" "$@"
