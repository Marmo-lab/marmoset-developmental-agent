# Detector v3: short calls from our own colony

The production call detector since 2026-11-17 (`models/call_detector_v3.pt`, sha256 `17573e57657bb9ba…`). It replaced
v2a (`DETECTOR_V2A_SCORES.md`), which stays in `models/` for comparison. Terms (precision, recall, IoU, main view,
front-end view) are defined in `DETECTOR_V2A_SCORES.md`.

## Why

v2a missed many short high calls (tsik, seep, alarm: 0.1-0.3 s): 20-37% of them on the synthetic set, and in our own
recordings about 39% of the short high candidates a plain energy rule finds (`SYNTH_EVAL.md`). It had met such calls
only as other labs' donors and as unchecked rule candidates. The fix: real short calls from our colony, checked by ear,
with their edges set by hand.

## The labels

- **Candidates** (`tools/labels/short_call_candidates.py`): the energy rule on 80 day recordings no earlier label set
  or training scan used (seed 2026_11_16). 43,863 candidates; 4,597 overlap no v5 detection.
- **Items:** 400 two-second clips, 320 of candidates v2a missed and 80 it caught (controls), spread over level
  terciles, at most 6 per recording. Recordings split into halves with `generate.half`; the test half never trains.
- **Labelling** (user, 2026-11-16/17, `review.html` with edge clicks): call (with its start and end), not a call, or
  unsure. Only the call under the candidate is marked; a call's smeared echo tail is left out. Frozen at 346 answers
  (`$WORK_DIR/short_labels_2026-11-17`).

| half | calls | not calls | unsure |
|---|---|---|---|
| train | 96 | 66 | 22 |
| test | 88 | 57 | 17 |

Of the 320 candidates v2a missed, 114 were calls, 121 not calls, 37 unsure: about 42% of the rule's "misses" are real
calls, so the real-world gain is about 24 missed calls per day-hour (≈ +2% corpus calls at best).

## Training

v2a's recipe (hand-label clips, synthetic train half `train50`, 145 real short-high minutes with gaps taught as
background; 1 head, 1000 steps, seed 0) plus the train-half labels (`train_call_detector.py fit --short`): a marked
call is a positive between its edges, a rejected candidate is background over its span, everything else is ignored.
Batch shares: synthetic 0.20, real minutes 0.15, short labels 0.15; the hand clips keep 0.50.

## Six criteria, fixed before training

| # | criterion | v2a | v3 | held |
|---|---|---|---|---|
| 1 | test-half calls caught (front end) | 0.42 [0.27-0.54] | 0.68 [0.54-0.79] | yes: paired +0.26 [+0.16, +0.38]; 23 gained, 0 lost |
| 1 | of those v2a missed in the corpus run | 0.14 | 0.53 | paired +0.40 [+0.24, +0.54] |
| 2 | test-half "not a call" fired on | 1 / 57 | 3 / 57 | within the interval: paired +0.035 [-0.03, +0.11] |
| 3 | hand-label CV, 45 min: P / R / F1 | 0.91 / 0.97 / 0.93 | 0.91 / 0.97 / 0.94 | yes; split 0.03 → 0.04, merged 0.04 → 0.05, onset 28 → 23 ms |
| 4 | night detections (10.1 h) | 0 / h | 0 / h | yes; day: rule candidates caught 0.897 → 0.916, detections/h +0.4% |
| 5 | synthetic ×1.0-×1.5 with noise, found | 0.805-0.870 | 0.838-0.893 | yes; false alarms/h lower; main view equal with noise, 1-2 points lower without |
| 6 | blind test set, once | F1 0.92, R@IoU.5 0.82 | F1 0.93, R@IoU.5 0.84 | yes (`TEST_PROTOCOL.md`, addendum 2026-11-17) |

Intervals: 95% bootstrap over recordings; "paired" resamples recordings and differences the two models on the same
items. Short types on the synthetic set (noise, ×1.5, found): tsik 0.58 → 0.64, seep 0.75 → 0.82, tsik-ek 0.80 →
0.87. Synthetic onset / offset errors are 6 / 11 ms larger; on the hand labels they are equal or smaller.

**Check of the method.** v2a itself catches 0.14 of the test calls the v5 corpus run had missed. The score runs the
production stream on a 10 s window around each item, not the whole recording, and a hand-marked call can extend past
the rule's candidate. Both models are scored the same way, so the comparison holds; the absolute levels are not the
corpus run's.

## Learning curve and seed

`figures/learning_curve_v3.png`: the same recipe with 25% (40 items), 50% (81) and all 162 train items, and all 162
with seed 1.

| model | train items | calls caught | v2a's misses caught | "not" fired on |
|---|---|---|---|---|
| v2a | 0 | 0.42 | 0.14 | 0.02 |
| v3_f25 | 40 | 0.64 | 0.47 | 0.05 |
| v3_f50 | 81 | 0.64 | 0.48 | 0.11 |
| v3 | 162 | 0.68 | 0.53 | 0.05 |
| v3_s1 (seed 1) | 162 | 0.58 | 0.38 | 0.04 |

- Most of the gain comes with the first 40 items. More labels of the same kind would add little.
- The seed spread at 162 items (0.58-0.68) is as large as the gain from 40 to 162. Every run beats v2a, and none loses
  a call v2a caught. v3 (seed 0) was named the candidate before any result.

## Files

| what | where |
|---|---|
| short-label scores, per-item hits | `analysis/detector_v3/short_labels/` |
| hand-label CV detections | `analysis/detector_v3/cv_v3/` |
| real-recording evaluation | `analysis/detector_v3/real/realscore_v3.json` |
| synthetic sweep | `analysis/synth_eval/v2sweep*/scores_v3.json` |
| blind test | `analysis/detector_v3/testset/` |
| code | `tools/labels/short_call_candidates.py`, `tools/analysis/train_call_detector.py --short`, `tools/analysis/score_short_labels.py` |
