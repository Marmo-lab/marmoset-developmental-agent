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
