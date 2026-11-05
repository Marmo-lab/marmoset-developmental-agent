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
1,800 one-minute recordings (300 per SNR level), 36,701 planted calls, raw donors.
`$WORK_DIR/synth/sets/v1/scores.{json,png}`. Front-end view, main matching
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
Scratch probe: every donor planted alone in 20 s of night background at +20 and +6 dB; the
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
