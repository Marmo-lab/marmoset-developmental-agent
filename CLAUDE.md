# CLAUDE.md: Marmoset Developmental Agent

Unsupervised developmental model of marmoset vocal learning (Omer lab, ELSC × Hasson lab, Princeton). Front-end and units follow **GmSLM** (Sternberg et al., EMNLP Findings 2025). Day-by-day training follows the **Hasson "First 1000 Days"** framework.

Details in `docs/`: `CONVENTIONS.md` (gotchas, layout, cluster), `ROADMAP.md` (science), `OPEN_DECISIONS.md`, `SESSION_STATE.md` (resume point), `DETECTOR_EVAL.md`, `LABELING.md`. Review standard: `REVIEW.md`.

## Stack
Python 3.11 · PyTorch · transformers (HuBERT) · scikit-learn (k-means) · scipy / librosa / soundfile · pytest · ruff. Real runs happen on the ELSC SLURM cluster; everything also runs on CPU in mock mode. No fairseq, no torchaudio.

## Goals
1. **Now:** a correct, verified preprocessing front-end on the raw colony recordings (4-ch ambisonic, 48 kHz, ~2,800 h), scored against hand labels.
2. **Next:** the developmental agent: chronological training with exposure F(d), replay, and a weekly trajectory.
3. GmSLM reproduction serves only as a comparison arm.

## Non-goals
- Call-type classification from our labels (we label onset and offset only).
- Collapsing repeated units, or a separate duration stream (see guardrail 1).
- Importing from `references/` (read-only upstream code).
- Any write to shared cluster data.

## Commands
```bash
pip install -r requirements.txt
pytest tests/                                   # ~55 tests, ~30 s; must pass
python -m ruff check .                          # lint (bug rules only); must be clean
python run_pipeline.py --mock                   # end-to-end on synthetic data, must finish < ~2 min
python run_pipeline.py --mock --encoder hubert  # frozen-HuBERT path
sbatch tools/cluster/verify_port.sbatch         # cluster only: Talia port gate, must stay 23/23 IoU 1.00
```
There is no build step. **Verify before handing over:** run pytest and ruff on every change, add the mock run for anything on the pipeline path, and add the port gate for anything touching `filter_5khz.py` or `segmentation.py`.

## Guardrails (violations are review failures)
1. **No deduplication.** Repeated units are never collapsed on any path to the uLM or the evaluations. Encoders return exactly `num_frames(L)` units, and the trainer calls `check_no_dedup`.
2. **Strict typing and config.** Every public function and class gets full type hints. Config lives only in the `agent_config.py` dataclasses: no raw dicts, and unknown YAML keys raise.
3. **Mock compatibility.** Every feature has a mock path, is exercised by `--mock`, and has a test.
4. **Decoupled stages.** Stages share only `Chunk`, `CallSegment`, `BaseAudioEncoder`, `UnitLanguageModel` and the configs.
5. **Chronology.** Never shuffle across days; only the sleep-replay portion of a day is shuffled.
6. **Provenance.** Every constant or formula cites its source (`[paper]` / `[talia]` / `[ours]`, a paper section or a `references/` file), and divergences are stated.

## Workflow: Two-Track Gatekeeper (user rule, 2026-11-10)
Classify every change before touching a file. If it is ambiguous, it is Track 1.
1. **Track 1: core research.** This covers preprocessing, detector and models, pipelines, cleaning and resynthesis: `marmoset_developmental_agent/`, the processing stages in `tools/corpus/`, and the detector training.
   - Do not change files first. Outline the logic, the math and the biological reasoning in chat (STFT sizes, thresholds, tensor operations), and wait for the user's approval of the concept.
   - Then stage the code, uncommitted, so the user can read it line by line in the app's diff viewer. Commit only after they accept it.
   - In core code, inline comments over-explain the *why* of every step. This overrides "minimal inline comments" below.
2. **Track 2: analysis, plots, glue.** This covers reports, figures, viewers, sbatch wrappers and one-off tools. Write, run and commit quickly; what matters is correct output.

## Conventions
- Paper vs. code: a small gap means the code wins, a large gap means the paper wins, and a paper typo means the code wins. Details in `docs/CONVENTIONS.md`.
- Durations are stored in seconds and converted with `to_frames()`.
- Match the surrounding code: dense docstrings explaining *why* and citing sources. Inline comments are minimal in Track 2 code and explanatory in Track 1 code (see Workflow).
- Git: work on `main` and push to `origin`. **Commit messages carry no Claude co-author or session trailers.**
