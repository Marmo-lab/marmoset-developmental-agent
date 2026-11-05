#!/bin/bash
# Synthetic recordings (tools/synth/generate.py). Reads the donors and the approved backgrounds;
# writes only under $WORK_DIR/synth/.
#   sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/synth.sbatch extract        # approved backgrounds -> mono FLAC
#   SET=v1 sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/synth.sbatch make [--n 300] [--snr 1.5]
#SBATCH --job-name=synth
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
S="$WORK_DIR/synth"
STEP="${1:?extract or make}"; shift
cd "$PROJECT_DIR"
if [ "$STEP" = extract ]; then
    PYTHONPATH="$PROJECT_DIR" python tools/synth/generate.py extract "$S/backgrounds"
else
    PYTHONPATH="$PROJECT_DIR" python tools/synth/generate.py make "$S/donors/turesson" "$S/backgrounds" "$S/sets/${SET:?set SET=<name>}" "$@"
fi
