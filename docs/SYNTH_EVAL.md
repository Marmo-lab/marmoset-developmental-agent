# Synthetic evaluation: planted calls in call-free background

Goal: score the front-end (the learned detector alone, then detector + merge_calls + quality
flags) against an exact answer, on many recordings, per call type, SNR and noise type. It
complements the hand labels (`DETECTOR_EVAL.md`), it does not replace them: a planted call
is not a call recorded in our cages, so the synthetic numbers are for relative comparisons
(SNR curves, robustness to noise events, detector versions), the hand labels stay the gold
standard. Agreed with the user 2026-11-05; each step ends with a listening checkpoint.

## Steps
1. **Donor calls.** Turesson et al. 2016 (PLoS ONE 11(9): e0163041), 11 types x ~30 clips,
   44.1 kHz mono, in `shared/data/DataFromOtherLabs/Phee_calls/Turesson 2016/` (read only).
   The clips are padded; `tools/synth/donor_calls.py` resamples to 48 kHz and proposes call
   bounds; the user checks them on the library's `/donors` page.
   **Done 2026-11-05:** the user looked over all 321 and accepted every donor as a call and
   every proposed bound as correct; no verdicts needed (`donor_feedback.jsonl` stays empty).
2. **Background.** Two kinds, compared against each other: (a) 1-5 night recordings with no
   calls at all (detector at a very low threshold finds nothing; the user listens), looped
   from a random start with a short crossfade; (b) day gaps between calls, several seconds
   clear on each side, checked the same way. At least half the synthetic recordings use (a).
3. **Noise events.** Candidates found automatically (loud in the call band, scored as non-call
   by the detector); the user listens and labels them (door, speech, beep, discard). Planted
   both apart from calls (false alarms) and on top of calls (missed calls).
4. **Generation.** One-minute recordings, call rate matched to v4, no overlapping calls at
   first. SNR = call RMS over background RMS, both in the call band (5-20 kHz) and over the
   call's duration. Levels (amplitude ratio / dB): 10/+20, 3.2/+10, 2/+6, 1.5/+3.5, 1/0,
   0.7/-3; about 300 recordings per level. Ten examples are listened to before mass
   generation. Written to `$WORK_DIR/synth/` with a manifest of every planted event and a fixed seed.
5. **Scoring.** Event-level precision, recall, F1 (matching by overlap, as `DETECTOR_EVAL.md`),
   false alarms per hour, onset and offset error in ms; split by call type, SNR, background
   kind and noise type.

## Later
The same with other labs' marmoset data (Landman 2017, Feng 2019, the "Dialects" set in the
same folder): long labelled recordings from other setups, a test of generalization.

## Results

### v1, night background (2026-11-05)
1,800 one-minute recordings (300 per SNR level), 36,701 planted calls, raw donors. Results in
`analysis/synth_eval/v1/` (figures: `figures/synth_v1_*.png`; 95% bootstrap intervals over
recordings are within ±0.012 of every recall and precision below). Front-end view, main matching
(IoU >= 0.5; phee, twitter and trill by overlap):

| SNR (dB) | +20 | +10 | +6 | +3.5 | 0 | -3 |
|---|---|---|---|---|---|---|
| recall | 0.54 | 0.47 | 0.43 | 0.39 | 0.32 | 0.23 |
| precision | 0.91 | 0.94 | 0.96 | 0.96 | 0.90 | 0.86 |
| false alarms / h | 0.2 | 0 | 0 | 0 | 0 | 0 |

Overlap precision is 1.00 at every level; main precision is lower because pieces of split
non-phee calls (chirp, alarm) fail IoU. By type at +20 dB: phees and loud shrill 1.00, twitter
0.87, trill 0.53, alarm 0.45, seep 0.13, tsik 0.09, tsik-ek 0.05, chirp 0.03 (overlap 0.48:
chirps are found but cut short or split).

### Why the short high calls are missed (probe, 2026-11-05)
`tools/analysis/synth_shift_probe.py` (`figures/synth_v1_frequency_shift.png`): every donor planted alone in 20 s of night background at +20 and +6 dB; the
detector's maximum call probability inside the call.
- **Hand labels.** Of 761 labels (calibration, held-out, test), none is both shorter than
  0.2 s and peaked above 12 kHz (5-20 kHz band). 7 are shorter than 0.2 s, 6 peak above
  12 kHz. Label duration p5 is 0.25 s. The detector never saw a tsik-like call as a call.
- **Frequency is part of it, not all.** Shifting other calls up 5 kHz (single-sideband, to
  12.6-13.5 kHz) cuts detection at +20 dB: twitter 0.90 to 0.33, chirp 0.63 to 0.17, loud
  shrill 1.00 to 0.41; phees hold (0.60-0.93). Shifting tsik, tsik-ek and seep down 5 kHz (to
  8-9 kHz, the usual call band) does not rescue them (0.07-0.30). Their short, sweeping shape
  is itself outside what the detector learned.
- **The threshold is not the cause.** At +20 dB, found at threshold 0.5 / 0.3 / 0.2 / 0.1:
  tsik 0.11 / 0.15 / 0.19 / 0.30, tsik-ek 0.13 / 0.13 / 0.20 / 0.20, seep 0.23 / 0.27 / 0.30 / 0.37.
- **Open:** whether our colony produces such calls (and the labeller did not hear or mark
  them) or rarely does. Not separable from the labels alone.

### Search for short high calls in our recordings (2026-11-05)
`tools/analysis/find_short_high.py`: a plain energy rule, independent of the learned detector
(11.5-20 kHz at least 15 dB above its median and 8 dB above the call band, 0.03-0.5 s).
Selftest on v1 (+20 / +6 dB): it finds tsik 0.68 / 0.65, tsik-ek 0.93 / 0.80, seep 0.81 /
0.74, twitter 0.46 / 0.21, every other type at most 0.11, and no false alarm on the night
background. Scan: 60 day recordings (07:00-17:59) of v4; the strongest candidates are cut
to 2 s clips for the library's `/short_high` page, where the user listens.

Result: 60 day recordings (60.1 h), 30,158 candidates (502/h), 7,671 outside every v4 call
(128/h, against v4's 1,271 detected calls per day hour). Listening (user, 2026-11-08, a random
look through the 150 strongest missed candidates): **all are calls.** So the colony does make
short high calls, the v4 corpus misses them, and the hand labels hold none. By eye most are
broadband 10-20 kHz bursts with sharp edges, some tonal sweeps at 15-20 kHz: not the same
shape as the Turesson tsiks (13-14 kHz sweeps).

Since the colony does make them, the plan (user, 2026-11-05): fine-tune the detector on a small
share of synthetic recordings so it meets these calls, then rerun the evaluation on the
rest. The donors must be split between the two shares (a donor heard in training would make
the test easy), and the test recordings must not share a background minute with training.

### Retraining with short high calls: v2 (2026-11-08)
Decided with the user: train on the 45 hand clips plus a small synthetic share (50 recordings per
level from the **train half** of donors and backgrounds, `generate.py --half train`, 6,000 calls)
plus real minutes (145 minutes from the train half of the 60 scanned recordings, rule candidates
and v4 calls as positives, the rest ignored). Each batch: 50% hand, 25% synthetic, 25% real. One
head, 1,000 steps, as production. Tested on what training never saw. Results in
`analysis/detector_v2/` and `analysis/synth_eval/v2test/`; `figures/synth_v2test_models.png`.

| test | production (v1) | v2 |
|---|---|---|
| synthetic held-out, recall +20 / +6 / -3 dB | 0.56 / 0.44 / 0.21 | 0.83 / 0.73 / 0.63 |
| synthetic held-out, precision +20 / +6 / -3 dB | 0.90 / 0.97 / 0.85 | 0.82 / 0.79 / 0.81 |
| synthetic, false alarms per hour | 0-0.2 | 2.8-8.4 (short pieces, median 0.35 s from a call) |
| tsik / tsik-ek / seep recall at +6 dB | 0.02 / 0.01 / 0.05 | 0.56 / 0.36 / 0.48 |
| real test half: rule candidates caught (all / missed by v4) | 0.74 / 0.00 | 0.95 / 0.81 |
| real test half: detections per day hour; call seconds per hour | 1,354; 1,038 | 1,363; 1,293 |
| call-free nights (10 h): detections per hour | 0 | 0.1 |
| hand labels, cross-validated: P / R / F1 | 0.94 / 0.90 / 0.92 | 0.91 / 0.93 / 0.92 |
| hand labels: IoU recall; merged; split; offset error | 0.81; 0.05; 0.05; 44 ms | 0.77; 0.13; 0.02; 54 ms |

Reading: v2 finds most of the short high calls, synthetic and real, at no cost in F1 on the hand
labels and without night false alarms. Its cost is merging: on the real test half the number of
detections barely moves while call time grows 25%, because many of the newly caught calls sit
next to another call and extend its detection; on the hand labels merges rise from 5% to 13%.
Open: how many of those merges join two labelled calls through a short high call between them
that the hand labels never marked (then v2 is right to see sound there, wrong only to join), and
whether the real minutes' ignored gaps teach the model to bridge. v2 is not production;
`learned_model_path` still points at v1.

### Fixing the merges: v2a, v2b (2026-11-08)
v2 merged neighbouring calls. The user: merging close calls is acceptable, not critical. Two
variants were trained anyway, identical to v2 except: **v2a**, gaps of 0.05-1 s between real
positives taught as background (`--real-gaps`); **v2b**, the same plus onset / offset heads
(`HEADS=3`). Hand labels, day clips, `c`+`?`, cross-validated:

| model | P | R | F1 | IoU P / R | split | merged | missed | onset / offset error |
|---|---|---|---|---|---|---|---|---|
| v1 (production) | 0.94 | 0.90 | 0.92 | 0.81 / 0.81 | 0.05 | 0.05 | 0.10 | 18 / 44 ms |
| v2 | 0.91 | 0.93 | 0.92 | 0.81 / 0.77 | 0.02 | 0.13 | 0.07 | 22 / 54 ms |
| v2a | 0.91 | 0.97 | 0.93 | 0.84 / 0.90 | 0.03 | 0.04 | 0.03 | 28 / 42 ms |
| v2b | 0.91 | 0.96 | 0.93 | 0.81 / 0.89 | 0.05 | 0.04 | 0.04 | 25 / 44 ms |

Held-out synthetic (front end, main matching), +20 / +6 / -3 dB:

| model | recall | precision | false alarms / h |
|---|---|---|---|
| v1 | 0.56 / 0.44 / 0.21 | 0.90 / 0.97 / 0.85 | 0.2 / 0 / 0 |
| v2 | 0.83 / 0.73 / 0.63 | 0.82 / 0.79 / 0.81 | 2.8 / 5.6 / 7.2 |
| v2a | 0.86 / 0.77 / 0.65 | 0.88 / 0.81 / 0.83 | 1.0 / 4.6 / 4.4 |
| v2b | 0.88 / 0.79 / 0.67 | 0.92 / 0.83 / 0.82 | 0.8 / 5.6 / 7.0 |

Recall at +6 dB, v1 / v2 / v2a / v2b: tsik 0.02 / 0.56 / 0.65 / 0.67; tsik-ek 0.01 / 0.36 /
0.43 / 0.31; seep 0.05 / 0.48 / 0.64 / 0.72; alarm 0.02 / 0.47 / 0.44 / 0.58; chirp 0.00 / 0.68 /
0.71 / 0.74; loud shrill 0.90 / 0.71 / 0.79 / 0.85; phees 1.00 throughout.

Real test half (30 day hours, 10 call-free night hours):

| model | candidates caught (all / missed by v4) | detections per day hour | per night hour |
|---|---|---|---|
| v1 | 0.74 / 0.00 | 1,354 | 0 |
| v2 | 0.95 / 0.81 | 1,363 | 0.1 |
| v2a | 0.90 / 0.61 | 1,480 | 0 |
| v2b | 0.91 / 0.66 | 1,511 | 0.3 |

Reading: the gap fix removes v2's merges (13% to 4%) and lifts every hand-label score above
production (IoU recall 0.81 to 0.90). v2's higher share of caught real candidates is partly its
merging: a long bridged detection covers candidates between calls. v2a and v2b now report the
newly caught calls as separate detections (+9-12% detections per hour).

### Decision: v2a frozen (user, 2026-11-08)
v2a (`models/call_detector_v2a.pt`, sha256 `bbf29f92945877d3...`) is the detector candidate. No further variants
(architecture, data, steps) until the developmental agent shows a problem that traces to the front end: each
further variant chosen on the same 45 clips and synthetic set makes those tests less honest. Known limitation, on
record: v2a still misses about 39% of the short high calls v4 missed, so the corpus under-represents tsik and seep;
on the strict IoU test its short-call boundaries are weak (alarm 0.44, tsik-ek 0.43 at +6 dB) though it touches
91-95% of them. Confirmed once on the blind test set (`docs/TEST_PROTOCOL.md`, addendum 2026-11-08): against v1,
P 0.91 vs 0.94, R 0.94 vs 0.93, F1 0.92 vs 0.93, R@IoU.5 0.82 vs 0.77, time recall 0.87 vs 0.80; nothing worse
beyond the bootstrap intervals. The test set has no label under 0.18 s. Production (`learned_model_path`) still points at v1 until the user switches it.
