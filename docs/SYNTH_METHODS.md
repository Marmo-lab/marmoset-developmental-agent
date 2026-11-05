# Synthetic evaluation recordings: methods

Paper-ready description of how the synthetic evaluation set is built and scored. The plan
and its checkpoints are in `SYNTH_EVAL.md`; this file says exactly what is done, with the
numbers. Every item is marked **[done]** (implemented, the numbers are final),
**[decided]** (agreed with the user, not yet implemented), or **[open]** (not yet decided).
Sources follow the CLAUDE.md convention: `[paper]`, `[ours]`.

## 1. Purpose
Hand labels (527 calls in 45 one-minute clips, `DETECTOR_EVAL.md`) give an absolute but
small test of the call detector. The synthetic set complements them. Known calls are planted
into call-free colony background at controlled signal-to-noise ratios (SNRs), with and
without interfering noise events. The detector is then scored against an exact answer on
thousands of calls, per call type, SNR, background kind and noise type. A planted call is
not a call recorded in our cages, so the synthetic scores serve relative comparisons:
SNR curves, robustness to noise, and detector versions. The hand labels remain the
reference for absolute performance.

## 2. The signal **[done]**
The colony recordings are 4-capsule ambisonic, 48 kHz, in files of about one hour. The
production detector hears the mean of the four capsules (`detection_channel_strategy =
"mean"`). The synthetic recordings are built from that same mean signal: mono, 48 kHz. The
detector then receives a synthetic file exactly as it receives the mean of a real one.

## 3. Donor calls **[done]**
- **Source** [paper]: Turesson, H. K., Ribeiro, S., Pereira, D. R., Papa, J. P., & de
  Albuquerque, V. H. C. (2016). Machine learning algorithms for automatic classification of
  marmoset vocalizations. *PLoS ONE* 11(9): e0163041. The set holds 321 single-call clips
  in 11 types, 44.1 kHz mono:

  | type | clips | median call duration (s) |
  |---|---|---|
  | alarm | 30 | 0.12 |
  | chirp | 30 | 1.06 |
  | loud shrill | 27 | 2.08 |
  | phee (2 notes) | 30 | 3.65 |
  | phee (3 notes) | 30 | 4.78 |
  | phee (4 notes) | 30 | 3.94 |
  | seep | 30 | 0.10 |
  | trill | 27 | 0.22 |
  | tsik | 27 | 0.14 |
  | tsik-ek | 30 | 0.33 |
  | twitter | 30 | 1.19 |

- **Rate** [ours]: polyphase resampling from 44.1 to 48 kHz by 160/147
  (`scipy.signal.resample_poly`, default Kaiser window).
- **Call bounds** [ours]. The clips are padded with silence, so each call's onset and
  offset within its clip are found by energy:
  - STFT: 1024-point Hann window, 480-sample (10 ms) hop. This is the detector's frame.
  - Band power: summed over 2–20 kHz, in dB, smoothed by a 3-frame moving average.
  - Floor: the clip's 10th percentile. Rise: the clip's maximum minus the floor.
  - Active frames: those exceeding the floor by rise/3, held to 3–12 dB.
  - Runs of fewer than 2 active frames are discarded.
  - The call spans from the first to the last remaining active frame, so the silent gaps
    between the syllables of a multi-syllable call stay inside it. Times are frame centres.
- **Verification**. On 2026-11-05 the user looked over all 321 clips on a listening page.
  Each showed a spectrogram with the bounds, playback of the call alone and of the whole
  clip. The user accepted every clip as a single call and every bound as correct.
- Code: `tools/synth/donor_calls.py`.

## 4. Background
### 4a. Night background **[done: method; numbers pending]**
- **Pool.** Recordings of the v4 corpus run in which the production detector (threshold
  0.5) found no call. Of 2,719 recordings, 832 have no detected call, 826 of them starting
  19:00–05:59. For each recording date, one recording is drawn at random (seed 0). It must
  start at 21:00–03:59 and be at least 10 minutes long.
- **Scan** of each pooled recording:
  1. The production detector is rerun with its frame threshold lowered from 0.5 to 0.1.
     Any faintly call-like event is listed.
  2. The background level is measured every second: power summed over 5–20 kHz, in dB
     relative to full scale. The STFT uses a 1024-point window and a 20 ms hop.
  3. A long-term spectrum (250 Hz bands × 1 s) is kept for display.
- **Day reference level.** 30 daytime recordings (start 09:00–16:59, at least one
  detected call) are scanned the same way. Each one's 10th-percentile per-second level is
  the background between calls; the reference is the median over the 30.
- **Selection.** Night recordings with no detection at threshold 0.1 are ranked by how far
  their median level lies from the day reference, with at most one per date. The top 10 are
  proposed. The user listens to each and approves or rejects it. **[pending: numbers, approvals]**
- Code: `tools/synth/backgrounds.py`.

### 4b. Day background **[decided]**
Stretches of daytime recordings with no detected call, at least about 10 s long, with a
margin of several seconds on each side. They are checked the same way: no detection at
threshold 0.1, then the user listens to a sample. Day background is less certain to be
call-free. Any faint call left in it shows up as false alarms on day background only, so
the two kinds are reported side by side. At least half of all synthetic recordings use
night background.

## 5. Synthesis **[done: code (`tools/synth/generate.py`, tested); pending: approved backgrounds, the run]**
Parameters fixed with the user on 2026-11-05.
- **Format**: 60 s per recording; mono, 48 kHz, written as 24-bit FLAC.
- **Background excerpt**: approved backgrounds are first extracted once as mono files (the
  4-capsule mean). Each recording takes one background at random and a 60 s excerpt at a
  uniform random offset. A background is described by its usable spans: a night recording is
  one span, the day background (4b) many short ones. When no span holds a full minute, random
  spans are joined with 50 ms raised-cosine crossfades.
- **Number of calls**: Poisson with mean 21.2 per minute. That is the daytime rate of the
  v4 corpus: the mean over the hours 07:00–17:59 of the per-hour means over 138 dates,
  1,271 detected calls per recorded hour (`figures/v4_calls_per_hour.png`). It counts every
  detection, quality flags included.
- **Call types**: each call draws its type uniformly over the 11 types, then a donor uniformly
  within the type. Every type is therefore planted about equally often, whatever its number
  of donors.
- **Timing**: no two planted calls overlap. Consecutive calls are separated by at least
  0.5 s: closer calls are a segmentation question, measured on the hand labels. Each call is
  at least 0.5 s from either end of the recording.
  - Given the calls' durations, the free time is split at sorted uniform random points, so
    every non-overlapping arrangement is equally likely.
  - A draw whose calls cannot fit is redrawn. Under these settings that happens when the
    Poisson count and long phees coincide. It slightly favours short calls in very dense
    draws, and the realised count is in the manifest.
- **Donors are planted raw** (decided by ear, 2026-11-05): the donor clip between its bounds,
  with its own recording noise. Spectral cleaning of the donors (the corpus's `clean_calls.py`,
  noise profile from the clip's padding) was tried and rejected: it made the calls sound
  unnatural, and natural-sounding calls with a little noise are preferred. Known cost: at
  high SNR the donor's noise is scaled with the call, so quiet-in-band donors (alarm,
  twitter) at +20 dB carry a faint broadband rectangle of their own noise. Cleaning remains
  available (`--clean-donors`) for comparison.
- **Planted call**: the donor between its bounds (section 3), with 5 ms raised-cosine ramps
  at both ends, scaled by a gain *g* and added: *y = b + g·s*.
- **SNR**: one level per recording. Let *bp* be a 4th-order Butterworth band-pass, 5–20 kHz,
  applied forward and backward (zero phase). For each call, *g* is set so that

  RMS(*bp*(*g·s*)) / RMS(*bp*(*b*)) = level,

  with *b* the background under the call, *s* the faded donor, and RMS taken over the call's
  samples. The level is an amplitude ratio. Levels, as amplitude ratio / dB:

  | ratio | 10 | 3.2 | 2 | 1.5 | 1 | 0.7 |
  |---|---|---|---|---|---|---|
  | dB | +20 | +10 | +6 | +3.5 | 0 | −3 |

  The donor's own residual noise (its clip floor, at least about 18 dB below the call for most
  types) is part of *s*, so the call proper sits marginally below the nominal level.
- **Overall gain**: if a recording's peak exceeds 0.99 full scale, the whole recording is
  scaled down. Background and calls are scaled together, so every SNR is unchanged; the
  factor is recorded.
- **Seeds**: recording *k* at level index *l* uses `numpy.random.default_rng([seed, l, k])`,
  so any single recording can be regenerated alone.
- **Scale**: 300 recordings per level (5 h) by default. Ten examples are approved by ear
  before mass generation.
- **Answer files**:
  - `recordings.tsv`, per recording: SNR, background id, kind and offset, number of calls,
    scale, seed.
  - `events.tsv`, per planted call: onset and offset (s, including the 5 ms ramps), donor,
    type, SNR, gain.
- **Background kinds**: night only, until the day background is built (4b). At least half of
  all recordings stay on night background.
- **Noise events** (step 3) **[decided: kinds and two placements; open: rates and levels]**.
  Door, speech, beep and the like are cut from our recordings and approved by ear. They are
  planted both apart from calls (false-alarm test) and overlapping calls (missed-call test).

## 6. Scoring **[decided]**
- **Detector**: production settings (learned detector, frame threshold 0.5, `merge_calls`
  quality flags), run on each synthetic file as a mono recording. Scored twice: the
  detector's raw detections, and the full front-end output.
- **Matching**: planted calls against detections. One-to-one greedy matching, best pairs
  first, at IoU ≥ 0.5, the same rule as `DETECTOR_EVAL.md` (`score_detectors.py`). Also
  reported: any-overlap matching.
- **Main view** **[decided, 2026-11-05]**: IoU ≥ 0.5, except for phees (2, 3 and 4 notes),
  which count as found when any detection overlaps them. Every detection overlapping a phee
  counts as correct. The detector splits multi-note phees at the note gaps; the notes still
  reach the corpus, which is what the developmental agent needs.
- **Short high calls** (tsik, tsik-ek, seep; peaks about 13–14 kHz) stay in the evaluation
  although the examples suggest the detector misses them; their miss rate is reported as a
  limitation.
- **Metrics**:
  - precision, recall, F1;
  - false alarms per hour, separately for noise-event overlaps and plain background;
  - onset and offset error (median, ms) on matched calls.
- **Breakdowns**: by call type, SNR level, background kind (night or day) and noise type.
- **Uncertainty** **[open]**: bootstrap over recordings, 95% intervals.

## 7. Later
The same procedure on other labs' marmoset recordings in `shared/data/DataFromOtherLabs`
(Landman et al. 2019; Feng 2019; the "Dialects" set). These are labelled long recordings
from other setups, a test of generalisation.
