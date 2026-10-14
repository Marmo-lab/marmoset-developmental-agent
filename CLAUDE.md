# CLAUDE.md: Marmoset Developmental Agent

Unsupervised developmental model of marmoset (*Callithrix jacchus*) vocal learning. A joint project of the Omer lab (ELSC, HUJI) and the Hasson lab (Princeton). It combines:
- **GmSLM** (Sternberg, London, Omer, Adi, Findings of EMNLP 2025): acoustic front-end, marmoset HuBERT units (layer 1, 50 clusters, **no dedup**), transformer uLM, unit HiFi-GAN, zero-shot benchmarks.
- **Hasson "First 1000 Days" framework** (`hassonlab/1kd_learning_model`): chronological day-by-day training with exposure schedule F(d), DINO-SR encoder (512 units), sleep replay, sigmoid-gated decoder, weekly trajectory and lexicon analysis.

Science and phase plan: `ROADMAP.md`. Baseline code and papers: `references/README.md`.
File-by-file code maps (shapes, formulas, gotchas, cross-project Rosetta Stone): `CODEMAP_DEVELOPMENTAL_AGENT.md`, `references/hasson_dev/CODEMAP.md`, `references/gmslm/CODEMAP.md`.

## Setup
```bash
python -m venv venv && source venv/bin/activate      # or: conda create -n marmoset python=3.11
pip install -r requirements.txt                     # torch, transformers, scipy, scikit-learn, librosa, soundfile, pyyaml, pytest
```
No fairseq or torchaudio needed. Everything runs on CPU in mock mode.

## Commands
```bash
python run_pipeline.py --mock                         # full dry-run on synthetic data (~80 s CPU); writes outputs/mock/
python run_pipeline.py --mock --encoder hubert        # same with frozen HuBERT units (~20 s)
python run_pipeline.py --mock --days 2 --skip-extended
python run_pipeline.py --config config/default.yaml   # real run (set data.* paths in the YAML first)
pytest tests/                                         # 21 tests, ~25 s
python -m marmoset_developmental_agent.preprocessing.filter_5khz --help        # in.wav out.wav [--gate]
python -m marmoset_developmental_agent.evaluation.zero_shot_benchmarks --help  # --checkpoint outputs/mock/checkpoints/day_0003.pt --mock
```
Outputs: `outputs/<run>/results.json`, `config_resolved.yaml`, `checkpoints/day_XXXX.pt` (weekly, plus the last day), `generated_continuation.wav`.

## Layout
```
run_pipeline.py                 CLI orchestrating stages 1-7
config/default.yaml             paper-scale overrides (provenance-tagged [GmSLM] / [1kd])
marmoset_developmental_agent/
  config/agent_config.py        all dataclasses; AgentConfig.mock() = tiny CPU config; load_config/save_config (unknown YAML keys raise)
  preprocessing/                filter_5khz (HPF + gate) -> segmentation (calls, 10 s Chunk) -> dataset_builder (ChronologicalDataset, F(d))
  encoders/                     BaseAudioEncoder ABC; MarmosetHuBERTEncoder (frozen, k-means); DinoSREncoder (trainable, EMA teacher, online codebooks)
  decoders/transformer_ulm.py   UnitLanguageModel (PAD=0, BOS=1, EOS=2, unit u -> id u+3)
  agent/                        lr_scheduler (sigmoid gate), replay_buffer (wake + sleep), trainer (day loop), factory (build/save/load)
  vocoder/hifigan_u2v.py        unit HiFi-GAN generator; mock "additive" tone renderer when no checkpoint
  evaluation/                   scoring (shared), zero_shot_benchmarks, developmental_eval (tracker + lexicon), downstream_probes, new_metrics
  utils/                        synthetic_data (mock colony + PHEEDB), mock_pipeline, log
references/                     read-only upstream code (never import from it)
```

## Guardrails (do not violate)
1. **No deduplication.** Consecutive identical units are never collapsed anywhere that feeds the uLM or the evaluations. Call duration is biologically meaningful (GmSLM Table 4). Encoders return exactly `num_frames(L)` units (499 for 10 s at 16 kHz). The trainer calls `check_no_dedup` on every batch. `run_lengths()` exists for analysis only. Note that Hasson's `multimodel.py` *does* dedup (`unique_consecutive` + duration stream); that part is deliberately not ported.
2. **Strict typing.** Full type hints on every public function or class. Config goes through dataclasses in `agent_config.py`: add fields there, never read raw dicts.
3. **Mock compatibility.** Every module must run on synthetic data on CPU. New features need a mock path, must be exercised by `run_pipeline.py --mock`, and need a test. Keep `--mock` under about 2 min.
4. **Decoupled stages.** Preprocessing, encoders, decoder, agent, vocoder and evaluation don't import each other's internals, except for the shared types (`Chunk`, `CallSegment`, `BaseAudioEncoder`, `UnitLanguageModel`, configs). New encoders subclass `BaseAudioEncoder` (`encode`, `extract_features`, `get_vocab_size`, `frame_rate`; trainable ones also `training_step` / `after_optimizer_step`).
5. **Chronology.** Never shuffle across days. Only the sleep-replay portion of a day is shuffled.
6. **Provenance.** Docstrings cite the paper section or the `references/` file a formula or hyper-parameter comes from. Divergences from official code are stated explicitly.

## Conventions and gotchas
- **Layer indexing differs.** HuBERT `layer` is 1-based (output of layer i; `hidden_states[i]`). DINO-SR `layer_for_tokens` is **0-based**, matching the official code: 9 means the 10th of 12 layers, and it must lie in the top-K codebook layers.
- **Replay.** `passes_per_day` X counts *all* passes: a day = n_D wake + (X-1)·n_D sleep samples, drawn from days <= D including today. `kernel: exponential` is the project spec. `inverse_recency` with `size_weighted: true` reproduces `past_sampling_dataset.py` exactly (there is a test for this).
- **Sigmoid.** Official code gates the uLM *loss weight* (`sigmoid-X0-K-Y`, per update, with a shared Adam). We gate the decoder *LR* with a separate optimizer. Same curve and parameters; see the `lr_scheduler.py` docstring for why.
- The GmSLM code is **not public**, but Talia's working tree was audited: `gmslm_audit/GMSLM_CODE_AUDIT.md`. Stage-1 constants are now reconciled against it and tagged `[paper]` / `[talia]` / `[ours]` in `PreprocessingConfig`. Resolution policy: where the two differ only by rounding, **the code wins** (her 10-frame call floor = 0.2322 s, which the paper rounds to 0.25 s); where they differ wildly, the paper wins; where the paper's number is physically implausible and the code's is field-standard, the code wins as a **paper typo** (mask_prob 0.80 not 8%, LR 3.5e-4 not 3.5e-3 — the paper also prints an impossible "5e-45"); where neither source made a decision at all, we decide (see the ROADMAP ablations).
- **Durations live in seconds, never in STFT frames.** Her thresholds were frame counts that only mean anything at hop 512 / 22050 Hz. `PreprocessingConfig.to_frames()` converts, and a test pins every constant to the exact frame count she used.
- `frame_features: "talia" | "spectral"` selects the detector. `"talia"` is a faithful port of `segment_processor.py` (freq_range / bandwidth / 3-class) whose purpose is to reproduce her `call_indicies.pkl` as a verification gate; `"spectral"` is ours, to be calibrated against hand labels. Mock runs use `"spectral"`, because her `amp_threshold` is an **absolute** |STFT| threshold tied to the colony's recording gain and so under-detects on synthetic audio.
- Detection runs at `detection_sample_rate` (22050 Hz — librosa's default resample in her code, *not* the native 48 kHz), and the 5 kHz high-pass applies to **detection only**: the audio handed downstream is unfiltered unless `highpass_training_audio` is set. Both reproduce her pipeline; both are flagged for ablation.
- **No labelled call-detection hold-out set exists** anywhere in her tree — the paper's P=0.975 / R=0.78 cannot currently be reproduced or checked. Building one is the blocking prerequisite for any Stage-1 optimisation.
- HuBERT without `pretrained_path` is random-init, so its units are meaningless (the pipeline logs a warning). A real marmoset HuBERT must be converted from fairseq to HF format first.
- Mock results near 50% on the zero-shot tasks are expected: the models are tiny and train for about 40 steps. The mock checks plumbing, not science.
- `references/hasson_dev` and `references/dinosr` are nested git clones. If this project becomes a git repo, add them as submodules or ignore them.
