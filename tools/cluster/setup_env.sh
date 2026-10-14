#!/bin/bash
# One-time: build the project venv on a compute node. Idempotent -- safe to re-run.
#
#   sbatch tools/cluster/setup_env.sh        # or: bash tools/cluster/setup_env.sh
#
# Run this on a COMPUTE node, not the login node: the login node has 2 cores and
# ~11 GB free, and a torch install there is both slow and inconsiderate to others.
#SBATCH --job-name=marmoset-setup-env
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Locate the project. Under `sbatch`, SLURM copies this script into a spool directory,
# so `$(dirname "$0")` points at /var/spool/slurm/... and NOT at the repo -- use
# SLURM_SUBMIT_DIR instead, falling back to the script path for a plain `bash` run.
# Every batch script in this project needs this preamble.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    export PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    export PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
fi
source "$PROJECT_DIR/tools/cluster/env.sh"

echo "host=$(hostname)  python=$(python3 -V)  venv=$VENV_DIR"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv "$VENV_DIR"
    echo "created venv"
fi
source "$VENV_DIR/bin/activate"

python -m pip install -q --upgrade pip
# Stage 1 (preprocessing, call detection) is pure CPU work, so the CPU-only torch
# wheel is enough and ~2 GB smaller. Switch to a CUDA build before GPU training --
# see tools/cluster/README.md for the driver-matched index URL.
python -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -q -r "$PROJECT_DIR/requirements.txt"

echo "--- versions ---"
python - <<'PY'
import torch, numpy, scipy, sklearn, soundfile, yaml
print("torch     ", torch.__version__, "cuda:", torch.cuda.is_available())
print("numpy     ", numpy.__version__)
print("scipy     ", scipy.__version__)
print("sklearn   ", sklearn.__version__)
print("soundfile ", soundfile.__version__)
PY
echo "setup_env OK"
