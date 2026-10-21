# Session state

Rolling handoff note. Updated whenever work pauses, the SSH link drops, or a milestone
lands. Read this first on reconnect: it should be enough to resume without re-deriving
anything. Long-lived decisions live in `OPEN_DECISIONS.md`; this file is "where are we
right now".

**Last updated:** 2026-10-21 (night)

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

### Latest (2026-10-21, night; supersedes everything below where they conflict)

**User verdict on the harmonic rule detector:** better, but still not accurate and not matching the labels. Agreed plan:
a detector learned from the labels.

**Done (commit after 062efdd):** the learned detector, `PreprocessingConfig.learned()`, `config/learned.yaml`, and
`config/corpus_day_learned.yaml`. Code in `preprocessing/learned_detector.py`; training and evaluation in
`tools/analysis/train_call_detector.py` (`cv` / `fit` / `evaluate`). The checkpoint is `models/call_detector_v1.pt`
(fit on all 45 clips).
- 9-fold clip-grouped cross-validation, `c` labels: P 0.93, R 0.93, IoU≥.5 P/R 0.83 / 0.84, time P 0.88, chunk P 1.00.
  The harmonic rule scores 0.78 / 0.97, 0.53 / 0.75, 0.70 and 0.86.
- Splits fall from 12% to 4% of labels. Median onset error is 20 ms (was 37 ms). Both edges are within 50 ms for 47%
  of labels (was 23%). 0 night detections. Details in DETECTOR_EVAL finding 12.
- Visual comparison: `labeling/compare_learned/index.html`.
- **Not yet the default.** Waiting for the user's look at the comparison page. Switching is one line
  (`frame_features`) plus the preset's fields. The talia path and the port gate are untouched.
- Streaming works: about 1.5 s per minute of audio on CPU.

**Pending:**
1. User verdict on `labeling/compare_learned`, then make `learned()` the production default.
2. Training rate, 48 kHz or 32 kHz (see below). Still open.
3. `v3_day` corpus run with `config/corpus_day_learned.yaml` plus the chosen rate.
4. More labels (10-20 clips): an untouched test set, and more training data.

### Earlier on 2026-10-21 (evening)

**User verdicts on `labeling/listen_v2`:**
- **The 10,875 Hz "device tone" is calls**, not a device. Do not filter it.
- The detections are not accurate: many calls are missed, and some non-calls are marked.
- **No reason to stay faithful to GmSLM**: its preprocessing is not good. Deviate wherever the data says so.

**Done 2026-10-21 (commit 0d44c45, pushed):**
- Inventory of all 527 hand labels on the raw 48 kHz audio:
  - The calls are harmonic. The lowest band has median 7.0 kHz, with the second band at 1.9-2.0x.
  - In 29% of calls the lowest band is above 8 kHz, and 87% have energy above 8 kHz.
  - The detector's 50 misses: 31 short faint notes, removed by the 0.2322 s floor, and 15 loud pulsed stacks, removed
    by the 0.35 density cap.
- **New production detector:**
  - Harmonic-comb frames, scored on the second-best partial over 5-24 kHz, skip the density cap and count as tonal.
  - Call limits are now 0.15-8 s.
  - Held-out: R 0.79 -> 0.93, call-time R 0.65 -> 0.85, P 0.80 -> 0.78, chunk P 0.94, chunk R 0.92 -> 0.99, no night
    false alarms. Details in DETECTOR_EVAL finding 11.
  - The held-out set has now been scored several times and is no longer untouched; a fresh test set needs new labels.
  - The Talia preset is identical field by field (checked locally). pytest 68 pass, ruff clean, mock 69 s.

**Pending, in order:**
1. DONE: **port gate passed** after this change (job 63944349): 23/23 matched, recall and precision vs hers 1.000.
2. **Decision: training sample rate.** 16 kHz training audio removes the fundamental of 29% of calls and the
   harmonics of nearly all. The user has said GmSLM is not binding. Options:
   - 48 kHz, which keeps everything.
   - 32 kHz, which keeps the fundamentals (up to ~11.5 kHz) and part of the harmonics.

   Either way, `target_sample_rate` already flows through `read_chunks`, `chunk_recording` and `merge_corpus`. The
   encoders hard-code 16 kHz (`BaseAudioEncoder.sample_rate`, conv stride 320 = 20 ms). At 48 kHz a 20 ms frame needs
   total stride 960, for example a first conv of (30, 15). A pretrained 16 kHz HuBERT then cannot be used, so the
   encoder is trained from scratch on our corpus, and the vocoder must change rate too. Not changed yet: this changes
   the model, not only preprocessing.
3. **Re-run the corpus as `RUN=v3_day`** with the new detector, at the rate chosen in step 2. Run it once, not twice.
   `v2_day` stays as is: do not delete it without approval.
4. Precision is unchanged: broadband cage noise is still the main false alarm. Calls under 0.15 s are still dropped.

### Next (as of 2026-10-20)

Done 2026-10-20: held-out set labelled (15 clips, 208 labels, `data/labels/heldout_labels.csv`) and scored once.
The pre-registered choice is **spectral @12 dB, 48 kHz with a 16 kHz ceiling, floor 0.23, tonal >= 0.5**. Held-out:
P 0.80 / R 0.79 (c only R 0.87), chunk P 0.94 / chunk R 0.92; Talia as run R 0.30 (DETECTOR_EVAL, held-out). Port
gate 23/23 after the tonality change. Repo root tidied (docs in docs/, local material in references/ and labeling/).
The 1kD extraction pipeline is summarised in docs/UPSTREAM_1KD.md: the agent trains on TTS re-synthesis, not on
recordings.

A. DONE (0b0d794): `PreprocessingConfig()` is the chosen detector; `PreprocessingConfig.talia()` /
   `config/gmslm_arm.yaml` = Talia's front-end (port gate and analysis tools use it). Held-out re-score identical.
B. DONE (0b0d794): `preprocessing/streaming.py`: 60 s blocks + 5 s context, `CallRecord` catalogue (wall-clock time,
   SNR, tonal fraction, dominant frequency + IQR), chunk audio read per window with `training_channel_strategy`
   (ambisonic_beam; detection on mean). `dataset_builder` streams `RecordingFile`s.
C. DONE: `tools/corpus/detect_corpus.py` + `tools/cluster/detect_corpus.sbatch` (resumable array job),
   `summarize_corpus.py`, `merge_corpus.py` (a523397). Pilot = the 45 recordings behind the labelled clips (`RUN=pilot`).
   2026-10-20 user decisions: night is not processed, chunk audio is stored under `~/marmoset_work`; `RUN=v1` (catalogue
   with night) was cancelled at 59/2719 and replaced by **`RUN=v2_day`** (job 63105484: `--config config/corpus_day.yaml
   --export-audio`, 136 tasks, 16 at once; finished overnight, **2719/2719, 0 errors**, 1114 all-night files skipped
   unread). Outputs in `~/marmoset_work/corpus/v2_day/`: `calls/`, `files/`, `chunks/<recording_id>/*.wav + chunks.tsv`,
   `summary.json`, `per_date.csv`, `fixed_tone_examples.csv`, `fixed_tone_10875.png`, and the **training manifest**
   `manifest.tsv` + `day_sizes.json` (job 63901147; loads with `dataset_builder.load_manifest`).
   Results (after dropping duplicates):
   - **344,900 chunks = 958 h** of 10 s training audio from 2,602 recordings, 1.81 M calls (361 h of calls), 137 dates
     2021-05-03 .. 2021-10-12; chunk-hours per date p5 1.0 / median 7.9 / p95 9.8. Default F(d) cuts it into 265
     simulated days (720 chunks/day at 2 h). Disk: 254 GB allocated for ~115 GB of WAVs: the share allocates
     736 KB per 320 KB chunk file (2.3x). Packing chunks per recording (one file + offsets) would halve it; the 117
     duplicate recordings' chunks (~11 GB) can be deleted, both pending the user's OK.
   - **The production path reproduces the evaluation** on the labelled minutes (`?` counted as calls): held-out P 0.81 /
     R 0.77 / chunk P 0.96 / chunk R 0.93; calibration P 0.87 / R 0.88 / chunk R 0.97; 0 detections in night clips.
   - Calls per audio-hour by clock hour: 06h 458, 07h 1665, 10-12h ~1000, 15-17h ~1600-1700, 18h 800; night 0 by design.
   - Durations: detected p5/p50/p95 0.30/0.55/1.82 s vs labels 0.25/0.52/2.04 (short-call gap, D26). SNR median 24 dB
     (p5 17); tonal fraction median 0.88.
   - **Duplicate copies on the shared drive:** `Rec_2021_07_14-17` holds byte-identical copies of all 116 files of
     `Rec_2021_07_08-13` (identical catalogues), so the real July 14-17 recordings are probably missing from the share
     (tell the lab). `210627_075` is in `Rec_2021_06_24-27` and in `Rec_2021_06_29`, where it is zeros after ~10 min
     (a truncated copy). `merge_corpus.deduplicate` keeps one copy (most calls, then earliest folder); `summary.json`
     and `per_date.csv` still count both copies (Jul 8-13 show ~47 h audio/date).
   - **Fixed tone found: 10,875 Hz**, a flat pulsed line with no harmonics (`fixed_tone_10875.png`): an electronic
     device, not a call. ~770 detections, mostly 2021-09-10..12 (434/229/104), a few elsewhere; 0.04% of calls. Not
     filtered yet: the catalogue has `peak_freq_hz`/`peak_freq_iqr_hz`, so a filter is a decision, not a re-run. The
     other top flat bins (7.7-9.2 kHz, spread over neighbouring bins) are phee-like calls, not device tones.
D. Optional: D26 (floor 0.1) needs ~10 fresh clips.
E. wandb: build `LoggingConfig` + logger (E21) before the first GPU training run.
F. Frozen-HuBERT units + D24 metrics, then the agent.

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
