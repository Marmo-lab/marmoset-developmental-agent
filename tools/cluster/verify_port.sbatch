#!/bin/bash
# Verification gate: our "talia" port vs her call_indicies.pkl on 210503_001.
#SBATCH --job-name=verify-port
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"

WAV="${1:-$RAW_RECORDINGS/Rec_2021_05_03/210503_001.WAV}"
PKL="${2:-$TALIA_FIRSTTRIAL/Rec_2021_05_03/210503_001/call_indicies.pkl}"
python "$PROJECT_DIR/tools/analysis/verify_talia_port.py" "$WAV" "$PKL" --strategy "${3:-mean}" ${VERIFY_ARGS:-}
