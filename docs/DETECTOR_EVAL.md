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
