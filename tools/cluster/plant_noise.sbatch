#!/bin/bash
# Planted noise (tools/synth/plant_noise.py). Reads the noise review, the raw recordings and a synthetic set;
# writes only under $WORK_DIR/synth/.
#   RUN=v5_calls sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/plant_noise.sbatch extract
#   SET=v2test OUT=v2test_noise sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/plant_noise.sbatch make [--snr 1.5]
#SBATCH --job-name=plant-noise
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
S="$WORK_DIR/synth"
STEP="${1:?extract or make}"; shift
cd "$PROJECT_DIR"
if [ "$STEP" = extract ]; then
    PYTHONPATH="$PROJECT_DIR" python tools/synth/plant_noise.py extract "$WORK_DIR/noise_events" \
        "$WORK_DIR/corpus/${RUN:?set RUN=<name>}" "$S/noises"
else
    PYTHONPATH="$PROJECT_DIR" python tools/synth/plant_noise.py make "$S/noises" "$S/sets/${SET:?set SET=<name>}" \
        "$S/sets/${OUT:?set OUT=<name>}" "$@"
fi
