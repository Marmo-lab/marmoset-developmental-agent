# Figures

Figures worth keeping (for the paper, talks, or the record), each with the data it was drawn
from in `data/` and the command that redraws it. Large or regenerable run outputs stay in
`outputs/` (git-ignored); a figure moves here once it is worth citing.

| figure | what it shows | data | redraw |
|---|---|---|---|
| `v4_calls_per_hour.png` | Detected calls per recorded hour by hour of day: one line per recording date (138), and the mean. Day 07:00-17:59 averages 1,271 calls/h, night 20:00-04:59 under 5/h. Basis of `CALLS_PER_MIN` in `tools/synth/generate.py`. | `data/v4_calls_per_hour.tsv` (date, hour, n_calls, recorded_s; from `calls_per_hour.py aggregate` on the v4 run, 2026-11-05) | `python tools/analysis/calls_per_hour.py plot figures/data/v4_calls_per_hour.tsv figures/v4_calls_per_hour.png` |
| `synth_v1_snr_curve.png` | Learned detector on the synthetic set v1 (1,800 one-minute recordings, 36,701 planted calls, night background): front-end recall and precision against SNR, main matching (IoU ≥ 0.5; phee, twitter, trill by overlap), 95% bootstrap bands over recordings; false alarms per hour beneath. | `analysis/synth_eval/v1/` | `python -m tools.analysis.synth_figures analysis/synth_eval/v1 figures` |
| `synth_v1_recall_by_type.png` | Same set: recall per call type and SNR, heat map. The short high calls (seep, tsik, tsik-ek) are missed at every level; chirp is found by overlap (0.48 at +20 dB) but fails IoU. | `analysis/synth_eval/v1/scores.json` | as above |
| `synth_v1_frequency_shift.png` | Each donor planted alone at +20 dB, as recorded and shifted 5 kHz: usual calls lose detection when moved up to ~13 kHz, but tsik, tsik-ek and seep stay missed when moved down to 8-9 kHz. Frequency is part of the blind spot, not all of it. | `analysis/synth_eval/v1/shift_probe.tsv` | as above |
| `synth_v2test_models.png` | Production (v1) against the detector retrained with synthetic (train half) and real short high calls (v2), on the held-out synthetic test set (1,800 recordings): recall and precision against SNR with bootstrap bands; recall per call type at +6 dB. | `analysis/synth_eval/v2test/` | `python -m tools.analysis.synth_figures analysis/synth_eval/v2test figures --compare v2` |

- `presence_table_v2a.png`: that table drawn in the paper layout. Redraw: `python tools/analysis/presence_table_figure.py`
