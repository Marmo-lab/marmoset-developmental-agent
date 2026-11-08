#!/bin/bash
# Learned call detector training (tools/analysis/train_call_detector.py) on the cluster: the hand
# clips from $WORK_DIR (--work), extra sources as given. Reads only; writes under $WORK_DIR/detector/.
#   NAME=v2 sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/train_detector.sbatch fit [--synth DIR --real DIR]
#   J=$(NAME=v2 sbatch --parsable --array=0-8 -o ~/marmoset_work/logs/%x-%A_%a.out tools/cluster/train_detector.sbatch cv [extra args])
#   NAME=v2 sbatch --dependency=afterok:$J -o ~/marmoset_work/logs/%x-%j.out tools/cluster/train_detector.sbatch merge
#SBATCH --job-name=train-detector
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
STEP="${1:?fit, cv or merge}"; shift
OUT="$WORK_DIR/detector/${NAME:?set NAME=<model name>}"
mkdir -p "$OUT"
cd "$PROJECT_DIR"
T="python tools/analysis/train_call_detector.py"
case "$STEP" in
    fit)   PYTHONPATH="$PROJECT_DIR" $T fit --heads 1 --work "$WORK_DIR" --out "$OUT/call_detector_$NAME.pt" "$@" ;;
    cv)    PYTHONPATH="$PROJECT_DIR" $T cv --heads 1 --work "$WORK_DIR" --out "$OUT/cv" --fold "${SLURM_ARRAY_TASK_ID:?array task}" "$@" ;;
    merge) PYTHONPATH="$PROJECT_DIR" $T cv --heads 1 --work "$WORK_DIR" --out "$OUT/cv" --merge ;;
esac
