#!/bin/bash
# Probe every raw recording's header and write the corpus manifest.
#
#   sbatch tools/cluster/build_manifest.sbatch
#
# Header-only: a few KB per file, not the 2 GB of audio. Single task, because the cost
# is network-filesystem metadata latency rather than CPU.
#SBATCH --job-name=build-manifest
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail

export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
mkdir -p "$WORK_DIR/manifests"

python "$PROJECT_DIR/tools/build_manifest.py" "$RAW_RECORDINGS" \
    -o "$WORK_DIR/manifests/recordings.csv" \
    -r "$WORK_DIR/manifests/recordings_rejected.csv"
