# Session state

Rolling handoff note. Updated whenever work pauses, the SSH link drops, or a milestone
lands. Read this first on reconnect: it should be enough to resume without re-deriving
anything. Long-lived decisions live in `OPEN_DECISIONS.md`; this file is "where are we
right now".

**Last updated:** 2026-10-14

---

## Where we are

Stage 1 (audio preprocessing and call detection), rebuilding GmSLM's front-end after
auditing Talia Sternberg's legacy tree.

### Done

- **Audit reconciled into code.** Every Stage-1 constant in `PreprocessingConfig` is
  tagged `[paper]` / `[talia]` / `[ours]` under a four-tier policy (see `CLAUDE.md`).
  Durations are in seconds, not STFT frames; a test pins each back to her frame counts.
- **Detector is switchable** via `frame_features`: `"talia"` (faithful port of
  `segment_processor.py`) vs `"spectral"` (ours, level-relative). 27 tests pass;
  `run_pipeline.py --mock` runs in ~42 s.
- **Cluster access working.** `ssh elsc-login-direct` (lab network only). venv at
  `~/venvs/marmoset311`, repo clone at `~/marmoset-developmental-agent`,
  outputs in `~/marmoset_work`. See `tools/cluster/env.sh`.
- **Git live:** `github.com/Zvimarmor/marmoset-developmental-agent`, branch `main`.
- **Level audit complete** over 54 files (one per recording folder), job 61620268.

### Key empirical findings

1. **Her `AMP_THRESHOLD = 0.02` is an absolute |STFT| magnitude** (librosa scaling, which
   is 1024x scipy's -- do not confuse them). **23/54 files never cross it at all.**
2. That is **driven by time of day, not calendar date.** Day files (06:00-19:00) median
   max |STFT| 0.2523 with 7/40 silent windows; night files 0.0055 with 40/40 silent.
   Correlation of level with day/night = **+0.859**, with days-since-start = +0.238.
   An earlier claim of a "25x gain drift across the campaign" was WRONG -- an artefact
   of a 9-file sample that caught quiet May nights and loud September days. The noise
   floor is stable (-180 to -198 dB, no trend), so **no recorder gain change occurred.**
3. **~46% of the corpus is night audio contributing almost no calls.** This matters for
   the exposure schedule: "hours of audio per simulated day" is not "hours of vocal
   input", and the day/night mix varies per folder.
4. **Best single capsule beats the 4-capsule mean in 32/54 files, never worse.**
   Median bin-survival gain 3.05x (min 1.29x). The `mean` default looks wrong.
5. Verification target decoded: `210503_001` -> 23 calls, all within the first 358 s of
   a 3,727 s file, all durations inside our adjudicated [0.2322, 4.0] s bounds.

### In progress

Streaming 4-channel reader + `bext` manifest builder (`marmoset_developmental_agent/io/`).

### Next

1. Run the manifest builder over all 2,750 raw files -> `path,start_time,...` CSV.
2. Channel-strategy ablation against hand labels.
3. `call_indicies.pkl` verification gate: run the `"talia"` port end-to-end on
   `210503_001.WAV` and compare detections to hers.

### Blocked

- **Hand-labelled hold-out set does not exist** (audit S3.2) and nothing can be called
  an improvement without it. `OPEN_DECISIONS.md` C9-C13 are unanswered.
- `OPEN_DECISIONS.md` sections B, C, D, E are unanswered (A1-A5 are answered).

---

## How to resume

```bash
ssh elsc-login-direct                      # lab network only
cd ~/marmoset-developmental-agent && git pull
source tools/cluster/env.sh                # sets PATH, venv, data roots
bash tools/cluster/show_levels.sh 61620268 # reprint the level-audit table (instant)
```

Locally: `pytest tests/ -q` and `python run_pipeline.py --mock`.

## Cluster facts worth not rediscovering

- `gpu.q`: 6 d walltime, **one** a100 node (4 cards) plus 15 `rtxpro6000` (96 GB,
  Blackwell, less contended -- benchmark before assuming A100). `elscn.q` = CPU.
- **`PreemptMode = REQUEUE`** -> long jobs need checkpoint/resume.
- Login node is 2 cores / 11 GB: never compute or pip-install there.
- `module` is a shell function, absent in non-interactive shells -> set `PATH` and
  `LD_LIBRARY_PATH` directly (python 3.11 will not start without the latter).
- **`sbatch` copies the script to a spool dir**, so `$(dirname "$0")` is not the repo --
  use `$SLURM_SUBMIT_DIR`.
- `[ -f x ] && source x` as the last line of a sourced file returns 1 and kills any
  `set -e` caller.
- A stale pre-audit snapshot sits at `~/Omer_Lab_Project` (Sept 27, not a git repo).
  Untouched; delete when convenient.
