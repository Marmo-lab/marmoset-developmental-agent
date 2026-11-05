# Figures

Figures worth keeping (for the paper, talks, or the record), each with the data it was drawn
from in `data/` and the command that redraws it. Large or regenerable run outputs stay in
`outputs/` (git-ignored); a figure moves here once it is worth citing.

| figure | what it shows | data | redraw |
|---|---|---|---|
| `v4_calls_per_hour.png` | Detected calls per recorded hour by hour of day: one line per recording date (138), and the mean. Day 07:00-17:59 averages 1,271 calls/h, night 20:00-04:59 under 5/h. Basis of `CALLS_PER_MIN` in `tools/synth/generate.py`. | `data/v4_calls_per_hour.tsv` (date, hour, n_calls, recorded_s; from `calls_per_hour.py aggregate` on the v4 run, 2026-11-05) | `python tools/analysis/calls_per_hour.py plot figures/data/v4_calls_per_hour.tsv figures/v4_calls_per_hour.png` |
