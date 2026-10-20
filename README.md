# Marmoset Developmental Agent

An unsupervised developmental model of marmoset vocal learning (Omer lab, ELSC × Hasson lab, Princeton). Raw colony
recordings are preprocessed into call-dense 10 s chunks (GmSLM front-end). An agent with an encoder and a unit
language model then learns from them day by day, following the Hasson "First 1000 Days" framework.

```bash
pip install -r requirements.txt
pytest tests/                      # must pass
python -m ruff check .             # must be clean
python run_pipeline.py --mock      # end-to-end on synthetic data, CPU, ~1.5 min
```

| Path | What |
|---|---|
| `run_pipeline.py`, `config/default.yaml` | entry point and paper-scale config |
| `marmoset_developmental_agent/` | the package: io, preprocessing, encoders, decoders, agent, evaluation |
| `tools/` | analysis scripts, the hand-label workflow, ELSC cluster jobs |
| `tests/` | pytest suite |
| `data/labels/` | hand labels (calibration and held-out sets) |
| `docs/` | conventions, roadmap, open decisions, session state, detector evaluation, labelling guide, code audit |
| `CLAUDE.md`, `REVIEW.md` | working rules and review standard |

These are git-ignored and local only: `references/` (read-only upstream code and audit material), `labeling/` (clips
and audio for labelling) and `outputs/` (run outputs).

Start with `docs/SESSION_STATE.md` for where things stand, and `docs/CONVENTIONS.md` before changing code.
