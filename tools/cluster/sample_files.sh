#!/bin/bash
# Pick one raw WAV per Nth recording folder, spread across the corpus date range.
# Writes absolute paths, one per line, for consumption by level_audit.sbatch.
#
#   bash tools/cluster/sample_files.sh [stride] > files.txt
#
# Excludes Rec_2021_06_28_ultrasonic_recordings_colony: a different instrument
# (250 kHz mono, no bext chunk) that must never go through a 48 kHz-assuming path.
set -euo pipefail
STRIDE="${1:-6}"
RAW="${RAW_RECORDINGS:-/ems/elsc-labs/omer-d/Omer-Lab-Shared/data/experiments/AnimalRoomRecordings}"
for d in $(ls -d "$RAW"/Rec_* | grep -v ultrasonic | sort | awk -v s="$STRIDE" 'NR%s==1'); do
    # middle file of the folder, to avoid recorder start-up transients
    mapfile -t f < <(ls "$d"/*.WAV 2>/dev/null | sort)
    [ "${#f[@]}" -eq 0 ] && continue
    echo "${f[$(( ${#f[@]} / 2 ))]}"
done
