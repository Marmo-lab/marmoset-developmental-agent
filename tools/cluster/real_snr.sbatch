#!/bin/bash
# SNR of the hand-labelled calls, synthetic definition (tools/analysis/real_snr.py). Reads raw recordings;
# writes only $WORK_DIR/real_snr/.
#   sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/real_snr.sbatch
#SBATCH --job-name=real-snr
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
OUT="$WORK_DIR/real_snr"
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" python tools/analysis/real_snr.py measure "$OUT"
PYTHONPATH="$PROJECT_DIR" python tools/analysis/real_snr.py summary "$OUT"
