# Detector evaluation against hand labels (2026-10-18)

**Ground truth:** 30 one-minute clips (24 day, 6 night; 2021-05-09 to 2021-10-12), labelled by ear in Audacity under `LABELING.md`. That gives 319 labels (298 `c`, 21 `?`) after import normalisation (merge gaps under 0.1 s, drop labels under 30 ms). 13 clips have no calls: all 6 night clips and 7 day clips.

**How to reproduce:**
```bash
sbatch tools/cluster/extract_clips.sbatch                       # cluster: raw 4-ch audio under each clip, 48 kHz
rsync -a elsc-login-direct:marmoset_work/clip_audio48/ labeling/clip_audio48/
python tools/analysis/score_detectors.py score labeling/clips.csv data/labels/handlabels.csv \
    --audio labeling/clip_audio48 --json-out outputs/detector_scores.json            # threshold x strategy
python tools/analysis/score_detectors.py score ... --sweep rate --strategies mean per_channel
python tools/labels/review_labels.py labeling clip_16 --scores outputs/detector_scores.json \
    --show talia@0.02/mean --show spectral@15/mean                                   # visual check
```

## Metrics
All numbers are on day clips, with `?` counted as calls unless noted.
- **Event (P/R):** a label is found if any detection overlaps it; a detection is correct if it overlaps any label.
- **IoU:** one-to-one matching at IoU ≥ 0.5.
- **chunk P / chunk R:** run the production `chunk_recording` over the detections. chunk P is the share of 10 s chunks that contain a labelled call. chunk R is the share of labelled calls whose midpoint lands in some chunk. **This is what decides the training data.**

## Results (mean of the 4 capsules unless noted)

| configuration | P | R | F1 | R_iou | chunk P | chunk R |
|---|---|---|---|---|---|---|
| Talia, as run (22 kHz, amp 0.02) | 0.97 | **0.40** | 0.57 | 0.28 | 1.00 | **0.69** |
| Talia, per_channel | 0.96 | 0.59 | 0.73 | 0.49 | 1.00 | 0.84 |
| Talia relaxed (no abrupt-up, floor 0.1 s), per_channel | 0.77 | 0.87 | 0.81 | 0.67 | 0.93 | 0.97 |
| spectral @15 dB, 22 kHz | 0.76 | 0.82 | 0.79 | 0.60 | 0.85 | 0.96 |
| spectral @15 dB, 48 kHz, 16 kHz ceiling | 0.78 | 0.85 | 0.81 | 0.64 | 0.86 | 0.95 |
| spectral @18 dB, 48 kHz, 16 kHz ceiling | 0.84 | 0.75 | 0.79 | 0.61 | 0.94 | 0.92 |
| spectral @18 dB, 22 kHz, floor 0.1 s | 0.69 | 0.95 | 0.80 | 0.71 | 0.82 | 0.98 |

## Findings
1. **Talia's precision reproduces the paper, her recall does not.** We measure 0.97 / 0.40 against the paper's 0.975 / 0.78. Recall depends heavily on call duration:

   | duration | recall |
   |---|---|
   | < 0.3 s | 0.04 |
   | 0.3–0.5 s (a third of all labels) | 0.12 |
   | 0.5–1 s | 0.49 |
   | 1–2 s | 0.78 |
   | > 2 s | 0.95 |

   So GmSLM's training data under-represents short calls.
2. **Root cause, from ablation on mean @0.02:**

   | change | effect |
   |---|---|
   | Disable the abrupt-up rule | R 0.40 → 0.46 |
   | Widen the narrow-high noise filter | R 0.40 → 0.42 |
   | Lower the 0.2322 s floor to 0.1 s (with both changes above) | R → 0.67, with P falling to 0.89 |

   The floor is tested on the span of the *detected* frames, which is shorter than the call whenever only its loudest part crosses the absolute threshold.
3. **Low-SNR calls:** Talia's recall is 0.2 below 6 dB in-band SNR; the spectral detector's is 0.65–0.78.
4. **The spectral detector's false alarms** (about 1 in 4 at 15 dB) are mostly isolated, vertical, broadband cage-noise transients in busy periods. They are almost absent in quiet clips (3 in 7 empty day clips) and absent at night. It has no shape rejection; that is the next fix.
5. **Detection rate:** 48 kHz with a 16 kHz band ceiling gives +2–5 points of recall and +4–8 points of time-recall. The F1 gain (+3) holds on both clip halves. Some calls sit at 10–12 kHz, at or above the 11 kHz Nyquist of 22050 Hz (e.g. clip_23).
6. **Channel strategy is second-order for detection.** Matched for gain (mean @0.01 vs beam @0.02), mean ≈ best_snr ≈ max_energy ≈ ambisonic_beam. per_channel (union over capsules) adds 5–10 points of recall at 4× the cost, for both detectors.
7. **Night:** no calls in any of the 6 night clips, and 0 false alarms per minute for every configuration above. Under the pre-registered rule this is branch 3: exclusion is optional and saves compute only.
8. **Uncertainty:** the top configurations differ by less than 5 F1 and change rank between the odd and even clip halves. Talia-as-run is worst on both halves (R 0.26 / 0.48). The fine choice needs a held-out set.
9. **Tonality separates calls from false alarms; frequency does not** (2026-10-19, preliminary scratch analysis on the
   rate-sweep detections, day clips). Per detection, on the 4-capsule mean at 48 kHz, 4–16 kHz, 10 ms frames: a frame
   is *tonal* if its peak bin is > 10 dB above that bin's median and the band within 10 dB of the peak is < 2 kHz wide.

   | feature (true vs false medians) | AUC |
   |---|---|
   | tonal-frame fraction (0.84 vs 0.21) | 0.96 |
   | peak-to-band contrast (23.6 vs 13.7 dB) | 0.95 |
   | -10 dB bandwidth (0.9 vs 3.2 kHz) | 0.94 (inverted) |
   | peak-frequency continuity | 0.91 |
   | dominant frequency (8.1 vs 9.1 kHz) | 0.38, i.e. useless |

   A rule "tonal fraction ≥ threshold", with the threshold fit leave-one-clip-out to keep 97% of the true detections:

   | configuration | before: P / R / chunk P / chunk R | after |
   |---|---|---|
   | spectral @15, 48k-16k, floor 0.23 | 0.77 / 0.87 / 0.86 / 0.95 | 0.92 / 0.85 / 0.96 / 0.95 |
   | spectral @12, 48k-16k, floor 0.23 | 0.62 / 0.92 / 0.72 / 0.96 | 0.89 / 0.90 / 0.89 / 0.94 |
   | spectral @12, 48k-16k, floor 0.10 | 0.46 / 0.99 / 0.63 / 1.00 | 0.75 / 0.97 / 0.79 / 0.99 |

   F1 is stable across the clip halves. Variants (tonal-frame count, longest tonal run) are equivalent within noise. A
   flat-line (machine-tone) rule removed only one detection, a true call, so it was dropped. Visual check
   (`labeling/review/tonality_filter_check.png`):
   - The real calls the filter drops are tonal notes inside a long detection that is mostly cage noise.
   - Many unlabelled detections that survive look call-shaped in the spectrogram. Checked by ear, only 2 of the 16 most
     tonal ones are calls (`labeling/review/unlabelled_tonal_detections.wav`, items 3 and 5). So the measured precision
     stands and is not a lower bound.

   **Implemented** as `min_tonal_fraction` (spectral path; `filter_5khz.py` computes the per-frame tonal flag,
   `segmentation.py` rejects). Reproduce with `score_detectors.py score ... --sweep tonal --strategies mean`. Production
   numbers (2048 / 512 STFT, 5–16 kHz band, `?` counted as calls):

   | configuration (48k-16k) | P | R | F1 | chunk P | chunk R | F1 even / odd halves |
   |---|---|---|---|---|---|---|
   | @12, floor 0.23, no rejection | 0.62 | 0.92 | 0.74 | 0.72 | 0.96 | 0.74 / 0.75 |
   | @12, floor 0.23, tonal ≥ 0.5 | 0.87 | 0.91 | 0.89 | 0.86 | 0.97 | 0.90 / 0.86 |
   | @12, floor 0.23, tonal ≥ 0.6 | 0.91 | 0.85 | 0.88 | 0.93 | 0.96 | 0.88 / 0.84 |
   | @15, floor 0.23, tonal ≥ 0.5 | 0.89 | 0.86 | 0.87 | 0.92 | 0.95 | 0.88 / 0.85 |
   | @15, floor 0.23, tonal ≥ 0.6 | 0.92 | 0.84 | 0.88 | 0.97 | 0.94 | 0.88 / 0.85 |

   Recall-first under the C13 constraint (chunk P ≥ 0.85) points to @12 / tonal 0.5, but its chunk P is 0.84 on the odd
   half. @15 / tonal 0.5 leaves a margin. Default set provisionally to 0.5; choose between the two on the held-out set.
   Night false alarms are 0 for every rejection ≥ 0.3.
10. **The entrance-door beep passes the tonality filter.** It is a stationary harmonic stack (e.g. 2.8 / 5.6 / 8.4 kHz in
    clip_24 @ 1.9 s; a line near 8.6 kHz in clip_18) and is heard among the surviving false alarms. Two generic rules
    fail on the labelled detections (scratch analysis, 2026-10-19):
    - "Fixed peak frequency in ≥ 40% of frames" flags 17 survivors, and 12 of them are true calls: phees are near-flat too.
    - "A 2.8 kHz line in ≥ 5 frames" flags 21, and 18 of them are true calls.

    A beep is one fixed device sound, so an exact-frequency template is legitimate here, unlike call templates. It is
    worth building only if the full-corpus run shows beeps are common. In the labelled set they are a handful of 520
    detections.

## Held-out test (pre-registered 2026-10-20, written before scoring)
**Set:** 15 day clips (`ho_01`–`ho_15`), from recordings not used for calibration and mostly from new dates. The user
labelled them by ear: 208 labels (163 `c`, 45 `?`); 5 clips have no calls. Imported to `data/labels/heldout_labels.csv`.

**Rule:** score once, with no tuning on this set.
- Choose between the two calibrated candidates (48k-16k, floor 0.23, tonal ≥ 0.5): **@12** if its chunk P ≥ 0.85 with `?`
  counted as calls (C13: recall first under that constraint), otherwise **@15**.
- Everything else (Talia as run, the no-rejection configurations) is reported for context only.

**Result (scored once, 2026-10-20).** `?` is counted as a call unless noted.

| configuration (48k-16k unless noted) | P | R | F1 | chunk P | chunk R |
|---|---|---|---|---|---|
| Talia as run (22 kHz, amp 0.02) | 0.91 | 0.30 | 0.45 | 0.93 | 0.61 |
| @12, floor 0.23, no rejection | 0.59 | 0.83 | 0.69 | 0.82 | 0.92 |
| **@12, floor 0.23, tonal ≥ 0.5 (chosen)** | 0.80 | 0.79 | 0.80 | **0.94** | 0.92 |
| @15, floor 0.23, tonal ≥ 0.5 | 0.78 | 0.75 | 0.76 | 0.92 | 0.92 |
| same chosen config, `c` only | 0.78 | 0.87 | 0.82 | 0.94 | 0.94 |

- **Decision, by the pre-registered rule:** @12, chunk P 0.94 ≥ 0.85. It also beats @15 on every metric. The tonality
  rejection holds out of sample: F1 +11, chunk P +12. Talia's recall reproduces at 0.30.
- **Recall is lower than on the calibration set (0.79 vs 0.91)** because of the `?` labels, which are short and faint
  (21 of the 45 are in ho_15). Counting `c` only, recall is 0.87.
- **The remaining weakness is short calls.** Recall by duration: < 0.3 s 16/39, 0.3–0.5 s 60/72, 0.5–1 s 60/66, > 1 s
  28/31.
- **Floor 0.1 (not a candidate; reported, not chosen).** At floor 0.1 / tonal 0.6, held-out chunk P is 0.94 and chunk R
  1.00. Pooled over all 39 day clips, P 0.80 / R 0.88 / chunk P 0.88 / chunk R 0.99, against P 0.85 / R 0.85 / chunk
  P 0.89 / chunk R 0.94 for the chosen config. It is the next candidate, but no untouched test set is left to confirm
  it, so it needs fresh labels (OPEN_DECISIONS D26).

## Finding 11: calls are harmonic, not narrow (2026-10-21)
**What the labels look like.** All 527 labels (461 `c`, 66 `?`, calibration + held-out) were measured on the raw
4-capsule mean at 48 kHz, 0–24 kHz, against the 2026-10-20 production detector:
- The lowest energy band has median 7.0 kHz (p5–p95 4.6–9.7). In 29% of calls it is above 8 kHz.
- The second band sits at 1.9–2.0× the first: the calls are harmonic. 87% of calls have energy above 8 kHz, which
  16 kHz training audio removes.
- Call shapes vary widely (flat phees, short up-sweeps, arches, loud pulsed stacks), but all are a fundamental with
  integer partials and a smooth contour. Cage noise is a vertical smear with no comb.

**What the detector missed** (event recall 0.89 on `c`; 50 misses):
- 31 are short, faint tonal notes of 0.2–0.35 s. The 0.2322 s floor, tested on the detected span, removes them: recall
  0.44 at 0.2–0.3 s.
- 15 are the loudest calls: pulsed harmonic stacks with 0.5–0.65 of the 5–16 kHz bins active. The 0.35 density cap,
  meant for cage noise, discards them. That is why recall at > 40 dB peak SNR (0.86) is below recall at 30–40 dB (0.95).

**Fix** (`harmonic_min_contrast_db`, `filter_5khz._harmonic_frames`):
- A per-frame harmonic-comb test over 5–24 kHz, f0 from 4 to 12 kHz. Each partial is scored against the louder of its
  two neighbouring troughs, so a spectrum that drops off at a band edge does not pass. The frame's score is the
  *second-best* partial, so a single band of noise or a lone line does not pass either.
- A comb frame skips the density cap and counts as tonal.
- With the comb, stacks join into runs longer than 4 s, which the old ceiling dropped whole. The ceiling is now 8 s and
  the floor 0.15 s. The talia preset keeps 0.2322 / 4 s and no comb.

Two earlier versions were superseded after unit tests caught them:
- Mean over partials, trough above each partial only: band-limited 5–16 kHz noise passed.
- Mean over partials, trough on both sides: a single 6–10 kHz noise band passed, as cage noise in the call band would.

Their sweep numbers are not reported.

**Calibration sweep** (`--sweep harmonic --strategies mean`, `?` counted as calls):

| configuration | P | R | F1 | call-time R | chunk P | chunk R | night FA/min |
|---|---|---|---|---|---|---|---|
| no comb, floor 0.23, max 4 (2026-10-20 production) | 0.87 | 0.88 | 0.88 | 0.76 | 0.86 | 0.96 | 0 |
| no comb, floor 0.15 | 0.82 | 0.94 | 0.88 | 0.79 | 0.82 | 0.97 | 0 |
| comb 8 dB, floor 0.15, max 8 | 0.73 | 0.96 | 0.83 | 0.92 | 0.79 | 0.98 | 0 |
| **comb 10 dB, floor 0.15, max 8 (chosen)** | 0.80 | 0.95 | 0.87 | 0.90 | 0.82 | 0.99 | 0 |
| comb 12 dB, floor 0.15, max 8 | 0.81 | 0.95 | 0.88 | 0.88 | 0.81 | 0.97 | 0 |
| comb 6 dB, floor 0.15, max 8 | 0.54 | 0.98 | 0.69 | 0.93 | 0.68 | 1.00 | 0.83 |

10 dB was chosen over 12 dB for call-time recall and chunk recall at the same precision. Against the 2026-10-20
detector, event precision is lower. It is accepted because it keeps the calls the user flagged as missing: call-time
recall is up 14 points, and the loud stacks are kept.

**Held-out (no longer an untouched set).** The set was first used on 2026-10-20. Here it was scored once for each of
the two superseded versions and once for the final one. Final scores:

| configuration | P | R | F1 | call-time R | chunk P | chunk R | night FA/min |
|---|---|---|---|---|---|---|---|
| 2026-10-20 production | 0.80 | 0.79 | 0.80 | 0.65 | 0.94 | 0.92 | 0 |
| no comb, floor 0.15, max 8 | 0.79 | 0.91 | 0.84 | 0.71 | 0.94 | 0.99 | 0 |
| **comb 10 dB, floor 0.15, max 8 (new production)** | 0.78 | 0.93 | 0.85 | 0.85 | 0.94 | 0.99 | 0 |
| same, `c` only | 0.75 | 0.96 | 0.84 | 0.87 | 0.94 | 0.99 | 0 |

Reproduce with `score_detectors.py score <clips> <labels> --audio <dir> --sweep harmonic --strategies mean`.

**Still open:**
- Precision has not improved. The false alarms are still broadband cage noise.
- Short calls under 0.15 s are still dropped.
- Training audio at 16 kHz removes most of the harmonic structure measured here.
