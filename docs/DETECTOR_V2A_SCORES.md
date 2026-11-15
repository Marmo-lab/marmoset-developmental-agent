# Detector v2a: scores

The production call detector (`models/call_detector_v2a.pt`, sha256 `bbf29f92945877d3…`) scored two ways:
1. **Hand labels.** One-minute colony clips labelled by the user.
2. **Synthetic recordings.** The latest version: unseen donor calls planted into unseen call-free night background,
   plus real colony noises.

Written 2026-11-15. Nothing here was re-run or tuned to produce it: every number comes from runs already recorded in
`DETECTOR_EVAL.md`, `SYNTH_EVAL.md`, `TEST_PROTOCOL.md` and `analysis/`.

## How to read the scores

| term | meaning |
|---|---|
| **Precision (P)** | of the detections, the share that are calls |
| **Recall (R)** | of the calls, the share the detector found |
| **F1** | the harmonic mean of P and R |
| **event, overlap** | a call counts as found if any detection overlaps it |
| **event, IoU ≥ 0.5** | a call and a detection must overlap by at least half of their union. This is the strict view: it also tests the boundaries |
| **event, main view** (synthetic) | IoU ≥ 0.5 one-to-one, except phee, twitter and trill, which count as found on any overlap (a decision with the user, 2026-11-05) |
| **time P / R** | the same questions measured in seconds of audio rather than in calls |
| **presence, 1 s / 10 s** | each minute cut into 1 s or 10 s segments. Each segment is call / no call, for the reference and for the detector |
| **Cohen's kappa** | agreement on presence beyond what chance would give (0 = chance, 1 = perfect) |
| **onset / offset error** | the median absolute difference between detected and true call edges |
| **front-end view** | detections flagged `short_low_conf` (under 0.18 s with confidence under 0.6) dropped, as the training corpus does |

Intervals are 95% bootstrap intervals over minutes (hand labels) or recordings (synthetic).

## 1. Against hand labels

### 1a. Blind test set: the headline numbers
- 15 one-minute day clips that no decision ever touched.
- Labelled to the visible call edges.
- Scored once, with the frozen model (`TEST_PROTOCOL.md`).
- 200 labelled calls, 265 detections. The 34 calls labelled `?` are ignored.

| metric | v2a | 95% interval |
|---|---|---|
| Precision (overlap) | 0.91 | 0.82 – 0.98 |
| Recall (overlap) | 0.95 | 0.90 – 0.99 |
| F1 (overlap) | 0.92 | 0.88 – 0.96 |
| Precision, IoU ≥ 0.5 | 0.62 | 0.51 – 0.74 |
| Recall, IoU ≥ 0.5 | 0.83 | 0.74 – 0.90 |
| Time precision | 0.93 | 0.89 – 0.97 |
| Time recall | 0.87 | 0.81 – 0.92 |

How labelled calls were matched:

| outcome | share of labelled calls |
|---|---|
| one detection, one call | 0.78 |
| split over several detections | 0.155 |
| merged with a neighbour | 0.01 |
| missed | 0.055 |

| boundaries | value |
|---|---|
| median onset error | 17 ms |
| median offset error | 48 ms |
| both edges within 30 ms | 24% |
| both edges within 50 ms | 47% |

| recall by call length | n | recall |
|---|---|---|
| 0.18 – 0.5 s | 65 | 0.89 |
| ≥ 0.5 s | 135 | 0.97 |
| < 0.18 s | 0 | (none labelled) |

### 1b. Cross-validated hand labels
- 45 minutes: 30 calibration clips and 15 held-out clips.
- Each clip is scored by a model trained without it. These clips shaped earlier decisions, so the blind set above is
  the unbiased number.
- Calls labelled `c` and `?` both count.

| P | R | F1 | IoU P / R | split | merged | missed | onset / offset error |
|---|---|---|---|---|---|---|---|
| 0.91 | 0.97 | 0.93 | 0.84 / 0.90 | 0.03 | 0.04 | 0.03 | 28 / 42 ms |

No detection in any of the 6 night minutes.

### 1c. Presence per segment (the Hasson-style table)

| set | segment | prevalence | P | R | F1 | kappa |
|---|---|---|---|---|---|---|
| blind test (15 min) | 1 min | 0.73 | 1.00 | 1.00 | 1.00 | 1.00 |
| blind test | 10 s | 0.59 | 0.96 (0.88–1.00) | 0.98 (0.94–1.00) | 0.97 | 0.93 (0.80–1.00) |
| blind test | 1 s | 0.39 | 0.95 (0.90–0.99) | 0.96 (0.93–0.98) | 0.95 | 0.92 (0.87–0.97) |
| cross-validated (45 min) | 10 s | 0.45 | 0.99 (0.97–1.00) | 0.97 (0.93–1.00) | 0.98 | 0.97 (0.92–1.00) |
| cross-validated | 1 s | 0.25 | 0.94 (0.90–0.96) | 0.98 (0.96–0.99) | 0.96 | 0.94 (0.92–0.96) |

Human-human kappa is not measured yet: it needs a second rater.

## 2. Against the synthetic recordings (latest version: with planted noise)

**The set.** The held-out half of the synthetic set, at SNR 1.5 (+3.5 dB):
- 300 one-minute recordings.
- 6,004 planted calls of 11 types (Turesson 2016 donors), raw, at about 21 calls a minute.
- Background: call-free colony night recordings.
- Neither the donors nor the backgrounds were seen in training.

**The noise.** On top, 2,106 real colony noises (`tools/synth/plant_noise.py`, about 7 a minute):
- 1,921 cage, 126 human, 59 door.
- Planted at their recorded level, anywhere in the minute.
- 1,441 of them land on a call.

The same 300 minutes without noise are shown alongside, for context.

### 2a. Events (front-end view)

| metric | with noise | without noise |
|---|---|---|
| Recall, overlap | 0.87 | 0.92 |
| Precision, overlap | 1.00 | 1.00 |
| Recall, main view | 0.72 | 0.74 |
| Precision, main view | 0.82 | 0.82 |
| Recall, IoU ≥ 0.5 (all types) | 0.63 | 0.65 |
| Precision, IoU ≥ 0.5 | 0.68 | 0.74 |
| False alarms per hour (detections touching no call) | 4.2 | 4.8 |
| Median onset / offset error | 28 / 58 ms | 29 / 58 ms |

Detector view, before the `short_low_conf` filter:

| metric | with noise | without noise |
|---|---|---|
| Recall, overlap | 0.89 | 0.94 |
| Main recall | 0.74 | 0.75 |
| False alarms per hour | 12.4 | 15.2 |

### 2b. Presence per segment

| set | segment | prevalence | P | R | F1 | kappa |
|---|---|---|---|---|---|---|
| synthetic + noise | 1 s | 0.80 | 0.99 (0.99–0.99) | 0.92 (0.91–0.92) | 0.95 | 0.79 (0.77–0.80) |
| synthetic, no noise | 1 s | 0.80 | 0.98 (0.98–0.99) | 0.96 (0.95–0.96) | 0.97 | 0.86 (0.85–0.87) |

The 10 s segments are uninformative here, since every 10 s segment holds a planted call.

### 2c. Recall by call type (front-end view, with noise; without noise in brackets)

| type | n | main recall | overlap recall |
|---|---|---|---|
| phee 2 / 3 / 4 | 502 / 500 / 529 | 1.00 (1.00) | 1.00 (1.00) |
| twitter | 567 | 0.98 (1.00) | 0.98 (1.00) |
| loud shrill | 533 | 0.79 (0.73) | 0.99 (1.00) |
| trill | 609 | 0.73 (0.78) | 0.73 (0.78) |
| chirp | 554 | 0.65 (0.69) | 0.98 (1.00) |
| tsik | 540 | 0.56 (0.61) | 0.63 (0.72) |
| seep | 585 | 0.55 (0.61) | 0.74 (0.87) |
| alarm | 530 | 0.38 (0.39) | 0.77 (0.85) |
| tsik-ek | 555 | 0.32 (0.39) | 0.79 (0.91) |

Of the 665 noises clear of every call, a detection fired on 7% of the cage noises, and on none of the people or doors.

## 3. Reading

1. **Real colony audio.** v2a finds about 95% of the calls the user hears, and about 9 in 10 of its detections are
   calls. Both hold on the blind set.
2. **Boundaries are the weak point.** Only half of the calls get both edges within 50 ms, and 15% of the labelled
   calls are split into pieces.
3. **Tonal calls are near perfect.** Phee, twitter and loud shrill are found almost always.
4. **Short and broadband calls are the hard ones.** Alarm, tsik-ek, tsik and seep are usually *detected*, but with
   boundaries that miss the strict match. Alarm's main recall of 0.38 against its overlap recall of 0.77 shows this.
5. **Colony noise costs recall, not precision.** It hides calls (1 s recall 0.96 to 0.92), but rarely raises a
   false call.
6. **Limits of these tests:**
   - The blind set has no calls under 0.18 s and no night minutes.
   - The synthetic donors come from another lab, and the background is night only.
   - Both sets are small for interval width: 15 minutes for the blind set, a single SNR level for the noise set.

## Sources
| item | file |
|---|---|
| blind test | `analysis/detector_v2/testset/test_scores.json` |
| cross-validation | `analysis/detector_v2/cv_v2a/` and `SYNTH_EVAL.md` |
| presence | `analysis/presence_table/presence_table.json` |
| synthetic | `analysis/synth_eval/v2test/scores_v2a.json` and `analysis/synth_eval/v2test_noise/scores_v2a.json` |
| figures | `figures/presence_table_v2a.png` |

## 4. Detections at recording-file edges (checked 2026-11-15)
**What was seen.** Picking calls for a figure, many of the strongest detections ended exactly at a recording file's end.
`tools/analysis/file_edges.py` ran on v5, results in `analysis/file_edges/`.

**How many.**

| measure | value |
|---|---|
| detections ending within 50 ms of their file's end | 377 (in 14.5% of files) |
| detections starting within 50 ms of their file's start | 324 |
| rate in the last 0.1 s of a file, against the rate 1–30 s from the end | 8× |
| end-touching detections whose file continues seamlessly into the next one | 355 |
| of those, with a start-touching detection in the next file | 228 |

**What they are: real calls cut by the recorder's file split, not an artifact.** The test ran the production detector
on the last 70 s of a file twice: alone, and with the next file's first 10 s appended (150 file ends).

| case | edge detection when cut | vanishes with the continuation (artifact) | continues across the junction (cut call) |
|---|---|---|---|
| file ends | 0.99 | 0.02 | 0.78 |
| control: random interior cut points | 0.35 | 0.00 | 0.85 |

**Effect.**
- A call that spans a file split becomes two partial detections, one at the end of each file.
- They are about 700 of 1.85 million calls (0.04%) and 0.18 of 405 call-hours. That is negligible for the corpus and
  for training.
- None of the detector's evaluations is affected: hand and synthetic clips always carry context around their calls.
- Their median SNR (23.8 dB, against 19.5 overall) puts them high on any SNR-ranked list, so tools that rank calls
  should skip file edges, as `figure_candidates.py` now does.
- A cure, if wanted later (Track 1): join an end-touching call with a start-touching call in the seamless next file in
  `merge_calls`.
