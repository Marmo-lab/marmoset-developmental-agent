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
