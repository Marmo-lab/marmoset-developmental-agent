# Analysis results

Results worth keeping, small enough for git, each folder with what produced it. Figures drawn
from them live in `figures/`; large or regenerable run outputs stay in `outputs/` (git-ignored)
or on the cluster under `$WORK_DIR`.

| folder | what | produced by |
|---|---|---|
| `synth_eval/v2test/` | Held-out synthetic test set (test half of donors and backgrounds), scored with v1 and the retrained v2, v2a, v2b (2026-11-08). In git: the manifest, the answer, the v2a detections and every model's scores; the other detections stay on the cluster. | `generate.py make --half test --seed 2 --n 300`; `score.py` with and without `--model ... --name v2*` |
| `synth_eval/v2test_noise/` | The same 300 minutes at SNR 1.5 with real colony noises planted (2026-11-10), scored with v2a. | `plant_noise.py make`; `score.py --name v2a` |
| `detector_v2/` | Retraining with synthetic and real short high calls: hand-label cross-validation with (`cv_v2`) and without (`cv_base`) the extra sources, identical settings; the real test half and call-free nights (`real/`). Checkpoint kept in git: `models/call_detector_v2a.pt` (production); v2 and v2b stay on the cluster. `testset/`: v1 and v2a scored once on the blind test set (TEST_PROTOCOL addendum 2026-11-08). | `train_detector.sbatch` (fit, cv x 9 folds, merge); `find_short_high.py realeval/realscore` |
| `synth_eval/v2sweep/`, `synth_eval/v2sweep_noise/` | The low-SNR sweep on the test half: 1,800 one-minute recordings, 300 at each SNR ×1.0-×1.5, 36,341 planted calls; `_noise` adds real colony noises (7 per minute: cage, door, human; `plant.json`). Scored with v2a and v3 (criterion 5 in `docs/DETECTOR_V3.md`). `bootstrap_sweep.json` (v2a) and `bootstrap_sweep_v3.json` (v3): per SNR, found / main / precision as [estimate, 95% low, 95% high], resampling recordings, for both sets. | `generate.py make --half test`; `plant_noise.py make`; `score.py --name v2a/v3` |
| `synth_eval/v1/` | Synthetic evaluation of the learned detector, night background, raw donors (2026-11-05): 1,800 one-minute recordings (300 per SNR level), 36,701 planted calls. | see below |

## `synth_eval/v1/`
- `recordings.tsv`: the set's manifest (`tools/synth/generate.py make`, seed 0). The answer (`events.tsv`), the
  detections and the audio stay on the cluster, `$WORK_DIR/synth/sets/v1/`.
- `scores.json`: `tools/synth/score.py score analysis/synth_eval/v1` (views `detector` and
  `front_end`; by SNR and by call type x SNR; main, overlap and IoU matching).
- `bootstrap.json`: front-end main recall and precision per SNR with 95% intervals over
  recordings (`tools/analysis/synth_figures.py`).
- `shift_probe.tsv`: every donor planted alone, as recorded and shifted 5 kHz, with the
  detector's frame probabilities inside the call (`tools/explorations/synth_shift_probe.py`).

Interpretation: `docs/SYNTH_EVAL.md`, results. Method: `docs/SYNTH_METHODS.md`.

- `presence_table/presence_table.json`: segment-level call presence (1 min / 10 s / 1 s) of v2a vs. hand labels, P/R/F1/lift/kappa, in the format of Hasson et al. Table 1. `python tools/analysis/presence_table.py`
- `detector_v3/`: the six criteria that made v3 production (`docs/DETECTOR_V3.md`): `short_labels/` (held-out short-call labels, `tools/analysis/score_short_labels.py`), `cv_v3/` (hand-label cross-validation), `real/` (rule candidates and night false alarms), `testset/` (blind test, scored once).
- `v6_check/`: calls per date and hour of the v6 run (detector v3, 2021 and the November 2026 recordings) and of v5, from `tools/analysis/calls_per_hour.py aggregate`; the 2026 rows are the low-gain recordings kept out of v6 (`docs/DETECTOR_V3.md`, corpus v6).
