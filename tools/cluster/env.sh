#!/bin/bash
# Shared environment for every ELSC cluster job in this project. Source, don't run.
#
#   source tools/cluster/env.sh
#
# Why not `module load python/3.11.13`: the `module` function is defined in the
# interactive bashrc and is unavailable in non-interactive shells (SLURM batch
# scripts, `ssh host bash -s`). The module only prepends PATH and LD_LIBRARY_PATH,
# so we set those directly -- equivalent and robust. Note that the interpreter will
# NOT start without the LD_LIBRARY_PATH entry (libpython3.11.so.1.0 lives there).

PY_ROOT=/ems/elsc-labs/apps/python/3.11.13
export PATH="$PY_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$PY_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export PROJECT_DIR="${PROJECT_DIR:-$HOME/marmoset-developmental-agent}"
export VENV_DIR="${VENV_DIR:-$HOME/venvs/marmoset311}"

# Make the package importable. Running `python $PROJECT_DIR/tools/foo.py` puts
# `tools/` on sys.path, not the repo root, so `import marmoset_developmental_agent`
# fails inside batch jobs even though it works from an interactive shell sitting in
# the repo. Setting PYTHONPATH makes both cases behave the same.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Data roots (gmslm_audit/GMSLM_CODE_AUDIT.md S4). Read-only; never write here.
export RAW_RECORDINGS=/ems/elsc-labs/omer-d/Omer-Lab-Shared/data/experiments/AnimalRoomRecordings
export TALIA_FIRSTTRIAL=/ems/elsc-labs/omer-d/talia.sternberg/MonkeyPhones/segmentations_results/firstTrial
export PHEEDB_ROOT=/ems/elsc-labs/omer-d/talia.sternberg/phee_exp

# Our own outputs.
export WORK_DIR="${WORK_DIR:-$HOME/marmoset_work}"

# Activate the venv if it exists. Written as an `if` rather than `[ ... ] && source`
# because the exit status of the last command in a sourced file becomes the status of
# `source` itself -- a failing test would then abort any caller running `set -e`
# (which is how setup_env.sh died the first time). The trailing `:` guarantees 0.
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi
:
