# Analysis results

Results worth keeping, small enough for git, each folder with what produced it. Figures drawn
from them live in `figures/`; large or regenerable run outputs stay in `outputs/` (git-ignored)
or on the cluster under `$WORK_DIR`.

| folder | what | produced by |
|---|---|---|
| `synth_eval/v2test/` | Held-out synthetic test set (test half of donors and backgrounds), scored with production (v1) and the retrained v2 detector (2026-11-08). | `generate.py make --half test --seed 2 --n 300`; `score.py` with and without `--model ... --name v2` |
| `detector_v2/` | Retraining with synthetic and real short high calls: hand-label cross-validation with (`cv_v2`) and without (`cv_base`) the extra sources, identical settings; the real test half and call-free nights (`real/`). Checkpoints `models/call_detector_v2*.pt`. `testset/`: v1 and v2a scored once on the blind test set (TEST_PROTOCOL addendum 2026-11-08). | `train_detector.sbatch` (fit, cv x 9 folds, merge); `find_short_high.py realeval/realscore` |
| `synth_eval/v1/` | Synthetic evaluation of the learned detector, night background, raw donors (2026-11-05): 1,800 one-minute recordings (300 per SNR level), 36,701 planted calls. | see below |

## `synth_eval/v1/`
- `recordings.tsv`, `events.tsv`: the set's manifest and answer (`tools/synth/generate.py make`,
  seed 0, cluster `$WORK_DIR/synth/sets/v1/`; the audio stays there).
- `detections/all.tsv`: the production detector on every recording (`tools/synth/score.py detect`,
  20 array tasks merged).
- `scores.json`: `tools/synth/score.py score analysis/synth_eval/v1` (views `detector` and
  `front_end`; by SNR and by call type x SNR; main, overlap and IoU matching).
- `bootstrap.json`: front-end main recall and precision per SNR with 95% intervals over
  recordings (`tools/analysis/synth_figures.py`).
- `shift_probe.tsv`: every donor planted alone, as recorded and shifted 5 kHz, with the
  detector's frame probabilities inside the call (`tools/analysis/synth_shift_probe.py`).

Interpretation: `docs/SYNTH_EVAL.md`, results. Method: `docs/SYNTH_METHODS.md`.
