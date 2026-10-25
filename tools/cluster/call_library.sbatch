#!/bin/bash
# Call library server (tools/corpus/call_library.py) on a compute node, for the user's browser.
# It only reads the run (audio + catalogue) and appends verdicts to RUN/library_feedback.jsonl.
#   RUN=v3_calls sbatch -o ~/marmoset_work/logs/%x-%j.out tools/cluster/call_library.sbatch [--all]
#   grep -E "^(node|open):" ~/marmoset_work/logs/call-library-<job>.out      # node, port and token
#   ssh -N -L 8765:<node>:<port> elsc-login-indirect                         # on the Mac, then open the URL
#SBATCH --job-name=call-library
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=12:00:00
set -euo pipefail
export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/marmoset-developmental-agent}"
source "$PROJECT_DIR/tools/cluster/env.sh"
RUN="${RUN:?set RUN=<name>, e.g. RUN=v3_calls}"
PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))
echo "node: $(hostname) port: $PORT"
python "$PROJECT_DIR/tools/corpus/call_library.py" "$WORK_DIR/corpus/$RUN" --host 0.0.0.0 --port "$PORT" "$@"
