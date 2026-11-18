# Blind test protocol for `labeling/test` (written 2026-10-25, before any test label exists)

**Purpose.** All 45 earlier clips have shaped the detector. The 30 calibration clips set the rule thresholds; the 15
held-out clips were scored several times; and all 45 trained the learned model. Its 0.93 / 0.93 is a
cross-validated estimate on those same clips. The 15 test clips (`ts_01`–`ts_15`) are the only audio no decision has
touched. This file fixes, in advance, what is scored and how, so the score cannot be tuned to them.

## What is frozen

| item | value |
|---|---|
| model | `models/call_detector_v1.pt`, sha256 prefix `da94fdadead78ed3` (trained on the 45 clips only) |
| configuration | `PreprocessingConfig.learned()` at commit `8d815bb`: threshold 0.5, 50 ms gap bridging, calls of 0.05–8 s, 1 kHz high-pass, mean of the 4 capsules |
| reference | the harmonic rule detector, `PreprocessingConfig()` at the same commit |
| code | `tools/analysis/train_call_detector.py evaluate` plus the boundary metrics below |

## Rules

1. **Nobody looks at detector output on the test clips before the labels are complete.** The user labels without it.
   Claude does not run any detector on `labeling/test` until the import step below.
2. **The test clips are never in training.** `train_call_detector.py` reads only the calibration and held-out sets
   (`SETS`), and the test labels go to their own file, `data/labels/test_labels.csv`.
3. **Score once,** with the frozen model and configuration, on every metric below. No threshold, setting or model
   choice may change in response to these scores, for this report.
4. **Report everything,** including whatever looks bad.

## Metrics

All are computed against the user's `c` labels, with `?` ignored:

- P, R, F1 (overlap); P and R at IoU ≥ 0.5; time P and time R.
- Per label: matched one-to-one, split, merged with a neighbour, or missed.
- Boundary error of one-to-one matches: median |onset| and |offset|; share with both edges within 30 ms and within
  50 ms. These test labels were drawn to the visible edges, so these are the first trustworthy boundary numbers.
- Recall by SNR band, and by shape cluster (same clustering as the 45-clip analysis).

## After scoring

- The test labels may then join the training data (a v2 model), but **v2 cannot be evaluated on them.** Any later
  claim about v2 needs new, untouched clips.
- The scores and this protocol go into `DETECTOR_EVAL.md` as the test-set result.

## Addendum, 2026-10-25 (before any test scoring)

The user's listening check of the pilot corpus led to a quality flag, `merge_calls.quality`: `short_low_conf` marks
a call shorter than 0.18 s with detector confidence below 0.6. It came from pilot audio, not from the test clips. At
this point the test labels are partly made (ts_01-ts_07), and no detector has run on the test clips.

The test is scored twice, and both results are reported:
1. the frozen detector as above (primary);
2. the same detections without those flagged `short_low_conf`.

Clip ts_07 contains a dense series that the user labelled as fewer, longer calls. Its scores are also reported on
their own.

## Addendum, 2026-11-08: scoring v2a once (written before scoring)

**Model.** `models/call_detector_v2a.pt`, sha256 prefix `bbf29f92945877d3`, frozen today as the detector candidate
(docs/SYNTH_EVAL.md, v2a). It was trained on the 45 calibration and held-out clips, a synthetic train half and 145
real day minutes; never on test labels. Checked before scoring: none of the 15 test clips comes from a recording
whose minutes v2a trained on, and the synthetic backgrounds are night recordings. No v2a decision used the test set.

**What runs.** The same scorer and metrics as the first scoring (`score_testset.py`), from the current commit, with
v2a and, for a same-code comparison, v1 again. `PreprocessingConfig.learned()` as of today. Both label modes, both
views (all detections; without `short_low_conf`), the three subsets, bootstrap intervals.

**Rules.** Score once. Report everything. No model, threshold or setting changes in response; the result decides
only whether v2a is confirmed or the decision returns to the user. After this the test set is spent for model choice
between v1 and v2a; any later model needs new untouched clips.

**Result (scored once, 2026-11-08; `analysis/detector_v2/testset/`).** `c` labels, all 15 clips, ± half the 95%
bootstrap interval over clips:

| | P | R | F1 | R@IoU.5 | time R | one-to-one | edges ≤ 50 ms | onset / offset |
|---|---|---|---|---|---|---|---|---|
| v1 | 0.94±0.04 | 0.93±0.05 | 0.93±0.02 | 0.77±0.10 | 0.80±0.08 | 0.76 | 0.43 | 14 / 53 ms |
| v2a | 0.91±0.08 | 0.94±0.04 | 0.92±0.04 | 0.82±0.08 | 0.87±0.05 | 0.78 | 0.47 | 17 / 48 ms |

- v1 reproduces its first scoring exactly, so the code path is unchanged.
- **v2a is confirmed**: no score is worse beyond the intervals; recall at IoU 0.5 (+5 points) and time recall (+7)
  are better, precision 3 points lower (295 detections against 279).
- The test set holds **no label shorter than 0.18 s**, so it cannot speak to the short high calls v2a was retrained
  for. Some of v2a's 16 extra detections may be such calls, unlabelled; that was not checked (no listening, by the
  rules above).
- Divergence: under the current default configuration the `rule (reference)` row runs the learned v1 (the default
  detector changed after the frozen commit), so it duplicates v1 and is not a rule-detector result.

## Addendum, 2026-11-17: scoring v3 once (written before scoring)

**Why the set is used again.** The addendum above spent the test set on the v1 / v2a choice and asked for new clips
for any later model. The user decided (2026-11-17) to use it once more, as the last of six criteria fixed before
training v3, not to choose between candidates. Its limits for this question are stated in advance: it holds no label
shorter than 0.18 s, so it cannot show v3's purpose (short calls); it checks only that nothing else broke. v2a's
result on it is already known, so this is a second look, and is reported as such.

**Model.** `call_detector_v3.pt` (`$WORK_DIR/detector/v3/`), sha256 prefix `17573e57657bb9ba`: v2a's recipe plus the
user's short-call labels from the train half of `short_call_candidates` (`train_call_detector.py fit --short`, mix
0.20 / 0.15 / 0.15, seed 0), named the candidate before any result. Checked before scoring: none of its 74 labelled
source files is one of the 11 the test clips come from (`short_call_candidates.excluded_paths` leaves them out); three
recording dates are shared (210807, 210909, 211005), in other files. Criteria 1-5 were read before this scoring
(`analysis/detector_v3/`): 1, 3, 4 hold; 2 (3 of 57 rejected candidates fired on, against 1) and 5 (synthetic
accurate-bounds recall 1-2 points lower without noise) are within or near their intervals.

**What runs.** `score_testset.py` from the current tree, v2a and v3, both label modes, both views, the same metrics
and bootstrap. **Rule.** v3 is confirmed if no score is worse than v2a's beyond the intervals; otherwise the decision
returns to the user. Score once, report everything, change nothing in response.

**Result (scored once, 2026-11-17; `analysis/detector_v3/testset/`).** `c` labels, all 15 clips, ± half the 95%
bootstrap interval over clips:

| | P | R | F1 | R@IoU.5 | time R | one-to-one | edges ≤ 50 ms | onset / offset |
|---|---|---|---|---|---|---|---|---|
| v2a | 0.91±0.08 | 0.94±0.04 | 0.92±0.04 | 0.82±0.08 | 0.87±0.05 | 0.78 | 0.47 | 17 / 48 ms |
| v3 | 0.90±0.09 | 0.95±0.03 | 0.93±0.04 | 0.84±0.06 | 0.88±0.04 | 0.79 | 0.48 | 17 / 47 ms |

- v2a reproduces its 2026-11-08 scoring exactly, so the code path is unchanged.
- **v3 is confirmed** by the rule above: no score is worse beyond the intervals (293 detections against 295). The
  largest change is the dense series ts_07: recall 0.83 → 0.91, one-to-one 0.52 → 0.65. With `c+?` labels the two
  are equal within 0.02 everywhere.
- As stated in advance, this set has no label under 0.18 s; v3's gain on short calls rests on criterion 1.
