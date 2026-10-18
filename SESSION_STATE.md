# Session state

Rolling handoff note. Updated whenever work pauses, the SSH link drops, or a milestone
lands. Read this first on reconnect: it should be enough to resume without re-deriving
anything. Long-lived decisions live in `OPEN_DECISIONS.md`; this file is "where are we
right now".

**Last updated:** 2026-10-18

---

## Where we are

Stage 1 (audio preprocessing and call detection), rebuilding GmSLM's front-end after
auditing Talia Sternberg's legacy tree.

### Done

- **Audit reconciled into code.** Every Stage-1 constant in `PreprocessingConfig` is
  tagged `[paper]` / `[talia]` / `[ours]` under a four-tier policy (see `docs/CONVENTIONS.md`).
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

6. **Corpus manifest built** (job 61620693, 3 min 51 s):
   `$WORK_DIR/manifests/recordings.csv` -- **2,719 files, 2,778 h**, 2021-05-03T18:58:29
   to 2021-10-12T15:00:18, chronologically sorted by bext start time.
   Rejects in `recordings_rejected.csv`: 31 = **25 zero-frame files** (recorder start/stop
   artefacts, mostly `_001` or the last file of a session) + **6 ultrasonic** files.
   Recording is effectively 24/7 and uniform over the clock (~115-125 h per hour-of-day
   bucket), so **day 06-19 = 1,509 h (54.3%), night = 1,269 h (45.7%)**. 137 recording
   dates over a 163-day span; 26 dates have no audio; median 23.8 h per recording date.
7. **Bug found and fixed in our own port (commit 751050c):** `filter_5khz._stft` used
   scipy's normalised STFT, which is 1024x smaller than librosa's, so our `"talia"`
   detector was ~1024x stricter than hers. Now in librosa units, pinned by a test.

8. **VERIFICATION GATE PASSED (job 61629425):** after re-porting from her source, the
   `"talia"` detector reproduces her `call_indicies.pkl` for `210503_001` exactly --
   23/23 calls, IoU 1.00 each, no extras. (The first port got 13/23: it was written from
   the audit's prose and had seven divergences, chiefly the "abrupt up" rule.)
9. **Channel evaluation (job 61629442, 54 files x 10 windows, gain-invariant):** daytime
   call-band dynamic range vs `mean`: ambisonic_beam +2.45 dB (wins 260/286 windows),
   per_channel +2.19, max_energy +2.14, best_snr +1.67. Night: ~0. Spectral-detector
   yield: per_channel +7% calls / +15% call-s by day, others ~flat. **Correction:** the
   earlier "3x bin survival for the best capsule" was mostly the 6 dB on-axis *gain*
   difference under an absolute threshold; the gain-invariant gain is ~2 dB.
   Provisional default for the training stream: `ambisonic_beam`.
10. **Unit sweep implemented** (D15): `tools/sweep_units.py`, k in {50,100,200}; mock runs.

11. **Labelling package delivered** to the laptop at `labeling/` (gitignored, ~170 MB):
    30 one-minute clips (24 day / 6 night, May 9 - Oct 12, one per date where possible),
    `clips.csv`, `guide.png`, `LABELING.md`. Zvi is labelling in Audacity; exports go to
    `labeling/labels/clip_XX.txt`. Import with `tools/labels/import_audacity.py`.
12. **Two findings from inspecting her detections visually** (while building the guide):
    (a) her "abrupt up" click rule rejected a clear rising call at 210503_001 t=41.0 s;
    (b) her call boundaries cover only *fragments* of longer, overlapping calling (e.g.
    t=61-62 s, 42-43 s): the first minute of that file is near-continuous calling. Both
    suggest her real recall is lower than detection counts imply -- the labels will tell.
    The guide is therefore hand-picked, with no detector boundaries.

13. **HAND LABELS COMPLETE + FIRST REAL DETECTOR SCORES (2026-10-18).** 30/30 clips,
    319 labels; 13 clips empty (all 6 night + 7 day). Full numbers and method:
    `docs/DETECTOR_EVAL.md`. Headlines (24 day clips, event-overlap matching):
    - Talia's detector as she ran it: **P 0.97, R 0.40** (paper: 0.975 / 0.78). Only 69%
      of labelled calls ever land in a training chunk. Recall on 0.3-0.5 s calls is 0.12.
      Root cause, by ablation: mainly the 0.2322 s floor applied to the *detected core*
      (shorter than the call), then the absolute threshold, then the abrupt-up rule.
    - Spectral detector @15 dB: P 0.76-0.78, R 0.82-0.85; ~95% of calls reach a chunk.
      Its false alarms are mostly vertical cage-noise transients in busy periods.
    - Detection at 48 kHz with a 16 kHz band ceiling (new `spectral_max_freq_hz`):
      consistent but small gain (+3 F1 on both clip halves). Calls at 10-12 kHz exist.
    - Channel strategy is second-order: gain-invariantly beam ~ best_snr ~ max_energy ~
      mean; per_channel adds 5-10 pts recall at 4x cost. The +2.45 dB of finding 9 does
      not translate into detection gains. (The training-audio channel is a separate question.)
    - Night: 0 calls in 6/6 night clips, 0 false alarms at sane settings -> pre-registered
      rule branch 3 (exclusion optional, compute only).
    - Top configs differ by < 5 F1 and swap rank between clip halves: need a held-out set
      to choose finely.

### In progress

Nothing running. Labels done (see finding 13). Cluster clone reset to the rewritten history
(at 3127efb or later); local backup branch `backup/pre-trailer-removal-2026-10-15`.
**Never add Claude co-author trailers to commits.**

Already landed and verified on real cluster data:
- `marmoset_developmental_agent/io/raw_reader.py` -- streaming blocks (bounded memory),
  `probe()` format validation, `bext` timestamps. Verified: `210503_001.WAV` reads as
  48 kHz / 4 ch / PCM_24 / 3727 s, start 2021-05-03T18:58:29; 3 x 60 s blocks in 3.1 s.
- `marmoset_developmental_agent/io/channels.py` -- `mean` / `max_energy` / `best_snr` /
  `per_channel`. The ambisonic-W candidate was dropped: for A-format capsules
  W = FLU+FRD+BLD+BRU, i.e. `mean` up to scale, so it would duplicate the baseline.
- `tools/build_manifest.py`, 43 tests passing.

**Audit correction:** the ultrasonic file *does* carry a bext chunk (GMSLM_CODE_AUDIT
S4.1 says it does not). It is still correctly rejected, on rate (250 kHz) and channel
count (1), not on the missing timestamp.

### Next (as of 2026-10-18; supersedes the list below where they conflict)

A. Spectral detector: add a broadband-transient rejection, then calibrate threshold /
   floor / rate on the current labels. B. Label a fresh held-out set (~15 day clips) to
   confirm. C. Wire `io/raw_reader` + channel strategy into `dataset_builder`. D. Full-corpus
   detection array job -> chunks/day, night chunks, duration distribution vs labels.

### Next (older)

1. Run the gate on a few more files with `firstTrial` output (daytime, busy ones) to
   confirm the exact match generalises beyond one quiet file.
2. Wire `io/raw_reader` into `dataset_builder` (replace the eager `audio.mean(axis=1)`
   loader) and add `channel_strategy` to `PreprocessingConfig` (default ambisonic_beam).
3. Labels (C9) -> real precision/recall for each channel strategy and detector.
4. Frozen-HuBERT path (D14): HF checkpoint needed -- stock facebook/hubert-base-ls960 for
   a first timing run, then convert Talia's marmoset checkpoint (layer question open).

### Blocked

- **Hand-labelled hold-out set does not exist** (audit S3.2) and nothing can be called
  an improvement without it. `OPEN_DECISIONS.md` C9-C13 are unanswered.
- OPEN_DECISIONS.md restructured: 9 open questions at the top (C10-C13, D18, D19, E20-E23), decided log below.

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
