# Marmoset Developmental Agent

An unsupervised developmental model of marmoset vocal learning (Omer lab, ELSC × Hasson lab, Princeton). A learned
call detector finds the calls in ~2,700 hours of raw 4-channel colony recordings. An agent with an audio encoder and a
unit language model then learns from them day by day, following the Hasson "First 1000 Days" framework.

```bash
pip install -r requirements.txt
pytest tests/                      # must pass
python -m ruff check .             # must be clean
python run_pipeline.py --mock      # end-to-end on synthetic data, CPU, ~1.5 min
```

## Layout

| Path | What |
|---|---|
| `run_pipeline.py`, `config/` | entry point; production config (`default.yaml`) and the GmSLM comparison arm |
| `marmoset_developmental_agent/` | the package: io, preprocessing (incl. the learned call detector), encoders, decoders, agent, evaluation |
| `tools/corpus/` | the production corpus path on the cluster: detect → merge → clean; the call library browser |
| `tools/synth/` | the synthetic evaluation: donor calls planted into colony background and noise, with an exact answer |
| `tools/analysis/` | detector training and scoring, and the analyses behind the results in `docs/` |
| `tools/labels/`, `apps/labeler/` | the hand-labelling workflow (Audacity import, Android app) |
| `tools/cluster/` | ELSC SLURM jobs for the above |
| `tools/explorations/` | one-off investigations, kept because the docs cite their results (see its README) |
| `tests/` | pytest suite |
| `data/labels/` | hand labels: calibration, held-out and blind test sets |
| `models/` | detector checkpoints: `call_detector_v2a.pt` (production) and `v1` (the blind-test baseline) |
| `analysis/`, `figures/` | small results and figures worth keeping, each with what produced it |
| `docs/` | method and results; see below |

Git-ignored and local only: `references/` (read-only upstream code), `labeling/` (labelling clips), `outputs/` (run
outputs). Raw audio never enters git.

## Where to read

- `docs/ROADMAP.md`: the science plan. `docs/OPEN_DECISIONS.md`: what is still open.
- `docs/DETECTOR_V2A_SCORES.md`: the production detector's scores, against hand labels and synthetic recordings.
- `docs/DETECTOR_EVAL.md`, `docs/TEST_PROTOCOL.md`: how the detector was chosen and tested.
- `docs/SYNTH_METHODS.md`, `docs/SYNTH_EVAL.md`: the synthetic evaluation.
- `docs/LABELING.md`: the hand-labelling guide.
- `docs/CONVENTIONS.md`: guardrails and gotchas; read it before changing code.
