#!/bin/bash
# Listening page for the phee offsets (tools/analysis/listen_phee_edges.py). Reads the experiment
# recordings and $WORK_DIR/phee_score, writes only $WORK_DIR/listen_phee_edges.
#   sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/listen_phee_edges.sbatch
#   rsync -a elsc-login-direct:marmoset_work/listen_phee_edges/ labeling/listen_phee_edges/
#SBATCH --job-name=listen-phee-edges
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
python "$PROJECT_DIR/tools/analysis/listen_phee_edges.py" "$WORK_DIR/phee_score" "$WORK_DIR/listen_phee_edges" "$@"
