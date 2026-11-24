# Parameter checks: detector and cleaning (2026-11-24)

Why (user, 2026-11-24): before the developmental agent, check that the detector's and the cleaning's main settings
are the right ones, and which can be left alone. Results and code below; the data is in `analysis/param_checks/`.

## Detector (v3)

| check | what | result |
|---|---|---|
| D1 threshold | Frame threshold 0.2-0.8 on the out-of-fold probabilities of the v3 cross-validation (day clips, `c` labels; the blind test set untouched) | F1 0.93-0.94 throughout; 0 night detections at every threshold. 0.5 is best for IoU≥.5 recall (0.89) and edges within 50 ms (0.49). 0.3: merges 4.8% → 8.7%; 0.7-0.8: recall 0.97 → 0.94-0.95, more splits. **Kept at 0.5.** |
| D3 frames at 0.3-0.5 | Where they lie; how close the calls missed at 0.5 come | 59% background, 34% inside `c` labels, 7% inside `?`. Of 15 missed calls, 1 reaches 0.3-0.5, 7 never exceed 0.1 (4 pass 0.5 too briefly). Lowering the threshold recovers almost nothing. |
| D4 250 Hz bands | Contrast of each labelled call in 47 Hz bins vs the detector's 250 Hz bands, missed against caught | Pooling costs the same for missed and caught calls: hand labels 4.3 vs 4.5 dB (missed are fainter, 13 vs 20 dB, and shorter); 88 held-out short calls 4.4 vs 5.0 dB (missed are not fainter). **Band width kept; retraining at 125 Hz (D5) not needed.** |
| D2 confidence | Confidence over v6 day calls; blind listening, 10 calls per bin | Confidence follows duration: 67% of calls under 0.18 s have confidence < 0.6, almost none over 0.5 s. A cut at 0.6 removes 6.2% of calls, 1.2% of call time. By ear: ≥ 0.7, 29 of 30 calls, 0 not; < 0.7, 63% calls, 10% not. **No call-level confidence filter.** |
| D6 `short_low_conf` | Blind listening: 25 flagged calls (+5 flagged among D2's) and 15 short calls just above the cut (0.6-0.7) | Flagged: 20 of 30 calls, 8 unsure, 2 not. Just above the cut: 10 of 17 calls, 4 not. The flag removed mostly real calls (≈4.9% of v6 day calls). **The flag is a marker only; `calls_train.tsv` keeps every call** (`merge_calls.TRAIN_EXCLUDES`, user 2026-11-24). |

Code: `tools/explorations/detector_checks.py` (D1, D3, D4), `tools/explorations/confidence_review.py` (D2, D6; verdicts
in `analysis/param_checks/confidence_review_verdicts.tsv`).

## Cleaning

| check | what | result |
|---|---|---|
| C1 noise, synthetic vs real | Cleaning front end on 600 random v6 day calls and 300 planted calls of `v2test` (4-capsule mean, ±3 s) | The synthetic (night) background is quieter (2-20 kHz: -82.1 vs -75.0 dB; 4-8 kHz: -93.2 vs -84.7), steadier (frame IQR 1.5 vs 2.0 dB) and click-free (0 vs 1.9% of frames). Real call SNR (loudest cell over N(f)): median 23.9 dB, p10-p90 16.0-38.5. **`v2test` is not used to tune the cleaning.** |
| C2 frequency speed | Dominant ridge of 422 real calls (SNR ≥ 20 dB) and 311 donors: step between 5 ms frames | Phee, loud shrill, alarm: 0-2% of steps > 500 Hz; trill 2.5%; twitter, tsik, chirp, tsik-ek, seep 11-21%; real calls 11%. The ridge mask still keeps ~100% of the dominant ridge at every link from 250 Hz (broken pieces stay ≥ 20 ms). **`LINK_HZ` is insensitive; kept at 500.** |
| C3 ridge leakage | Share of call-free cells the ridge mask keeps, real day background around 150 calls, peak × link × minimum length | 6 dB / 20 ms: 8.8% of cells; 9 dB: 4.0%; 3 dB: 31%. Link 500 → 1000: 8.8% → 9.9%. Upper bound (the "background" may hold faint calls and echo). **Peak threshold and minimum length are the parameters that matter.** |
| C4 examples page | C2's measures on the 44 calls of the examples page | Long calls lose 2-5% of their peak energy to tracks under 20 ms, short calls 20-48% (6 dB). |
| C8 listening | The 44 calls in five versions (`ridge_listen.py compare`) | User: ridges at **9 dB / 20 ms** are the best ridge version. |
| C5 + C7 shadow filtering | 550 mixtures (11 types × 10 donors × 5 SNRs at C1's real quantiles) of a donor and real day background on a random beam; the cleaning's per-cell gain applied to call and noise separately (`tools/analysis/clean_eval.py`) | Below. |

C7 medians (dB; call SDR higher = less damage; noise left lower = cleaner):

| setting | call SDR | call kept | noise left |
|---|---|---|---|
| spectral subtraction (α = 3) | **14.5** | -0.49 | **-20.7** |
| ridges 6 dB / 20 ms | 11.4 | -0.34 | -11.0 |
| ridges 9 dB / 20 ms | 10.5 | -0.41 | -17.6 |
| ridges 12 dB / 20 ms | 10.0 | -0.48 | -20.8 |
| ridges 9 dB / 10 ms | 11.2 | -0.35 | -15.2 |
| ridges 9 dB / 40 ms | 8.0 | -0.76 | -20.4 |

- Among ridge settings, 9 dB / 20 ms is not beaten on both measures by a neighbour.
- Spectral subtraction beats it on both, per mixture: less call damage in 77% (median +1.45 dB), less noise left in
  94% (median -2.2 dB), at every SNR; the gap is largest for fast-FM types (twitter, tsik, seep, chirp). Ridges keep
  long tonal calls (phee 2, loud shrill) slightly more intact, with more noise left.
- Caveats: the donor's own recording noise counts as call, a small bias against ridges; the mixtures' background is
  ~2 dB quieter than around real calls (-77.2 vs -75.0 dB).
- **The corpus keeps spectral subtraction** (`clean_calls.py`, unchanged). Figure: `figures/clean_eval_2026-11-24.png`.

Code: `tools/explorations/cleaning_checks.py` (C1-C4), `tools/explorations/ridge_listen.py` (pages; ridge default now
9 dB), `tools/analysis/clean_eval.py` + `tests/test_clean_eval.py` (the per-cell gain reproduces `clean_call` and
`ridge_clean` exactly, and splits the output exactly into call and noise).
