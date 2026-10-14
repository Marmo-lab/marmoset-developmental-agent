# CODEMAP: Marmoset Developmental Agent (`marmoset_developmental_agent/`)

Our integrated codebase: GmSLM's marmoset front-end, units and evaluations, trained under the Hasson lab's developmental regime (chronological days, sleep replay, gated decoder, weekly trajectories).
Companion maps: [`references/hasson_dev/CODEMAP.md`](references/hasson_dev/CODEMAP.md) · [`references/gmslm/CODEMAP.md`](references/gmslm/CODEMAP.md) · science plan in [`ROADMAP.md`](ROADMAP.md) · commands in [`CLAUDE.md`](CLAUDE.md).

State documented: 2026-09-29. 30 Python files, about 3,700 lines. `python run_pipeline.py --mock` passes (about 70 s CPU) and `pytest tests/` passes (21 tests).

---

## 1. Directory overview

```text
Omer_Lab_Project/
├── run_pipeline.py                 CLI: runs stages 1–7 end to end (--mock or --config)
├── config/default.yaml             paper-scale overrides of the dataclass defaults
├── tests/test_pipeline.py          21 pytest checks
├── pytest.ini, requirements.txt, .gitignore, CLAUDE.md, ROADMAP.md
├── references/                     read-only upstream code (never imported)
└── marmoset_developmental_agent/
    ├── config/agent_config.py      ALL hyperparameters (dataclasses) + YAML load/save
    ├── preprocessing/              PHASE 1   filter_5khz → segmentation → dataset_builder
    ├── encoders/                   PHASE 2   base_encoder, marmoset_hubert, dino_sr
    ├── decoders/                   (uLM)     transformer_ulm
    ├── agent/                      PHASE 3   lr_scheduler, replay_buffer, trainer, factory
    ├── vocoder/                    PHASE 4   hifigan_u2v
    ├── evaluation/                 PHASE 5   scoring, zero_shot_benchmarks, developmental_eval,
    │                                         downstream_probes, new_metrics
    └── utils/                      synthetic_data, mock_pipeline, log
```

**Dependency direction** (arrows mean "imports"). Stages never import each other's internals; they share only the types in the middle row.

```text
 run_pipeline.py ─► agent/factory ─► encoders/*, decoders/*
        │                 ▲
        ├─► agent/trainer ─► agent/{lr_scheduler, replay_buffer}
        ├─► evaluation/* ─► evaluation/scoring
        └─► utils/mock_pipeline ─► utils/synthetic_data, preprocessing/dataset_builder
 shared types:  config.agent_config.*Config · segmentation.{Chunk, CallSegment} · BaseAudioEncoder · UnitLanguageModel
```

---

## 2. End-to-end data flow

```text
 WAV recording(s), any rate (mock: 48 kHz synthetic colony, file order shuffled on purpose)
   │  Recording(audio (L_native,), sample_rate, start_time, recording_id, annotations?)
   ▼ dataset_builder.build_chronological_dataset
   │   sort by start_time  ─────────────────────────────────────────── chronology is never broken after this
   │   filter_5khz.preprocess_waveform:  Butterworth HPF 5 kHz (order 8, zero-phase) at native rate
   │        compute_frame_activity: STFT 2048/512 → per-bin noise floor → density & variance → active[t]
   │   segmentation.activity_to_calls:  merge gaps < 0.12 s, keep 0.25–4 s  → [CallSegment]
   │   filter_5khz.resample_audio → 16 kHz
   │   segmentation.chunk_recording:  10 s windows packed with calls → Chunk(waveform (160000,), calls, start_time)
   │   DailyExposureSchedule F(d): n(d) = ⌊hours(d)·3600 / 10⌋ chunks  → ChronologicalDataset(chunks, day_sizes)
   ▼
 ENCODERS  wav (B, 160000) float32 @16 kHz
   │  conv stack ×7, stride 320 → (B, 499, D) frames @ 50 Hz
   │  HuBERT: hidden_states[1] (B, 499, 768) → nearest of 50 k-means centroids ─┐
   │  DINO-SR: EMA-teacher layer `layer_for_tokens` → instance-norm → nearest of 512 codewords ─┤→ units (B, 499) int64
   │  check_no_dedup: units.shape[-1] == 499  (repeats kept)                                    ┘
   ▼
 TRAINER (per simulated day D)
   │  SleepReplayBuffer.build_day_schedule: wake = today's n_D chunks (in order)
   │                                        sleep = (X−1)·n_D chunks, day d ~ P(d|D) ∝ exp(−λ(D−d))  [or n_d/(D−d+1)]
   │  per batch of B chunk indices:
   │     (DINO-SR only) L_enc = CE(student heads @ masked frames, teacher codes) → AdamW → EMA teacher update
   │     units = encoder.encode(wav)             (B, 499)
   │     lr_dec = lr_max·(σ(k(t − T_p)) + floor) → UnitLanguageModel.loss(units)  = mean NLL over 500 targets (499 units + EOS)
   │  DevelopmentalTracker (every eval_every_days): surprisal / PPL on days D+1..D+7, OOD PPL, LearnedLexicon
   │  checkpoint every checkpoint_every_days → outputs/<run>/checkpoints/day_XXXX.pt
   ▼
 VOCODER  ulm.generate(prompt 100 units, +100) → HiFiGANU2V.synthesize → wav (1, 200·320) → generated_continuation.wav
   ▼
 EVALUATION on held-out days (test chunks)
      zero-shot: Shuffle / Reversal / Concat / CallerChange / ReceiverChange → accuracy of log p(real) > log p(fake)
      probes: per-call mean⊕var of uLM states (2·E) → ProbeMLP → macro-F1 (call type, caller)
      new metrics: ICI ×{0.25, 0.5, 2, 4}, pitch ±{1, 2} semitones → accuracy + Δ surprisal (bits/unit)
   ▼
 outputs/<run>/results.json · config_resolved.yaml · checkpoints/ · generated_continuation.wav
```

---

## 3. Key hyperparameters and tensor contracts

"Default" means paper scale (dataclass defaults / `config/default.yaml`). "Mock" means `AgentConfig.mock()`.

| Quantity | Default | Mock | Defined in |
|---|---|---|---|
| Model sample rate | 16,000 Hz | same | `PreprocessingConfig.target_sample_rate`, `BaseAudioEncoder.sample_rate` |
| High-pass | 5 kHz, Butterworth order 8, `sosfiltfilt` | same | `PreprocessingConfig` |
| Activity STFT | n_fft 2048, hop 512 (native rate) | same | `PreprocessingConfig` |
| Call duration | 0.25–4.0 s, merge gap 0.12 s | same | `PreprocessingConfig` |
| Chunk | 10 s = **160,000 samples** | same | `chunk_duration_s` |
| Conv stack | kernels (10,3,3,3,3,2,2), strides (5,2,2,2,2,2,2) → stride 320 | same | `base_encoder.CONV_LAYERS` |
| Frames per chunk | **499** = `conv_output_length(160000)`; frame rate **50 Hz (20 ms)** | same | `BaseAudioEncoder.num_frames / frame_rate` |
| HuBERT vocab | **K = 50**, layer **1** (1-based) | 50, 2-layer 64-d random model | `HuBERTEncoderConfig` |
| DINO-SR vocab | **K = 512**, 12 layers, 768-d, top-8 codebooks, `layer_for_tokens` 9 (**0-based**) | 512, 3 layers, 64-d, top-2, layer 2 | `DinoSRConfig` |
| DINO-SR masking | prob 0.8, span 10 | same | `DinoSRConfig` |
| EMA τ | 0.999 → 0.9999 over 40k steps | over 100 steps | `DinoSRConfig` |
| uLM | 12 layers × 1024, 16 heads, FFN 4096, max 3072 positions | 2 × 64, 4 heads, 1024 positions | `DecoderConfig` |
| uLM vocab layout | PAD 0, BOS 1, EOS 2, unit *u* → *u + 3* | same | `UnitLanguageModel` |
| Replay X | 25 passes/day; kernel exponential, λ = 0.1 | 25 | `ReplayConfig` |
| Decoder gate | lr_max 5e-4, T_p 279,000 steps, k 3e-4, floor 1e-6 | 1e-3, T_p 20, k 0.3, floor 1e-3 | `SigmoidScheduleConfig` |
| F(d) | hours/day by month [2,2,2,3,4,4,5,5,6,6,7,7,8], 30 days/period | [0.0084, 0.0112, 0.014] h, 1 day/period → 3, 4, 5 chunks | `ExposureScheduleConfig` |
| Batch | 22 | 8 | `TrainingConfig.batch_size` |
| Vocoder hop | upsample (5,4,4,2,2) → 320 samples/unit | channels 32 | `VocoderConfig` |
| Evaluation period | 7 days | 1 day | `EvaluationConfig.eval_every_days` |

---

## 4. Gotchas, traps and non-obvious assumptions

1. **No deduplication, ever.** Encoders return one unit per 20 ms frame. `DevelopmentalTrainer._tokens` calls `check_no_dedup` on every batch (it raises if the length ≠ 499 for a 10 s chunk). `run_lengths()` exists **only** for statistics. Hasson's `multimodel.py` *does* dedup; we deliberately did not port that part.
2. **Two layer-index conventions.** HuBERT `layer=1` → `hidden_states[1]` (output of the first transformer layer; `[0]` is the pre-transformer input). DINO-SR `layer_for_tokens=9` is **0-based** like the official code (the 10th layer). It must satisfy `num_layers − K ≤ layer ≤ num_layers − 1`, otherwise `DinoSREncoder.__init__` raises.
3. **Replay maths.** `P(d | D) = w(D−d) · n_d^s / Σ …`, where `w` is `exp(−λΔ)` (spec, default) or `1/(Δ+1)` (official), and `s = 1` iff `size_weighted`. Today is included (`include_current_day`). X counts **all** passes: sleep = (X−1)·n_D samples, drawn with replacement. Replicate Hasson exactly with `kernel: inverse_recency, size_weighted: true`; `test_replay_inverse_recency_matches_official` pins this.
4. **The sigmoid is on the decoder LR, not on its loss.** It deviates from Hasson on purpose (see the `agent/lr_scheduler.py` docstring). `t` is the global update count by default; `unit: "day"` uses fractional days.
5. **Known divergences from the official DINO-SR (not yet fixed):**
   - **Loss scale.** `DinoSREncoder.training_step` *sums* CE over the K codebooks, then divides by #masked frames. The official code averages over codebooks (`losses / n_codebooks`), so our loss is K× larger. With Adam this mostly affects logged values and gradient clipping.
   - **Target tensor.** We use each layer's full output (after the residual and LayerNorm). The official code uses the FFN output before the residual (`layer_result`, index 2).
   - **EMA scope.** We average every teacher parameter; the official code copies `pos_conv` instead (`ema_layers_only`).
   - **Masking.** One number of spans for the whole batch, overlaps allowed. The official code uses fairseq's `compute_mask_indices` with `require_same_masks`.
6. **Frame ↔ time mapping** (lexicon, probes). Call times map to frames as `int(start_s·50)` and `ceil(end_s·50)`. We ignore the 25 ms receptive field of the first frame, which Hasson corrects with a 5 ms shift. The error is at most one frame.
7. **HuBERT is random in mock mode.** Without `pretrained_path` the HuBERT weights are random-init, so its units are meaningless (the pipeline logs a warning). k-means is fitted in `run_pipeline.py` on the training chunks. **Centroids live in a buffer** (`centroids`) so they are saved in checkpoints; `load_state_dict` resizes that buffer before loading.
8. **Frozen encoders are cached.** When the encoder is not trained (HuBERT, or DINO-SR with `train_encoder: false`), the trainer caches units per chunk index. A trained DINO-SR re-encodes every batch, because its codebook and teacher keep changing.
9. **Chunks keep the gaps.** The noise gate is used for *detection* only; training chunks are not silenced between calls (GmSLM: 60% gaps). `noise_gate()` is only for the standalone CLI (`--gate`).
10. **The last day may be partial.** `DailyExposureSchedule.day_sizes` cuts the chronological stream exactly; the final day takes whatever is left. `run_pipeline.py` treats days after `num_days` (or the last `eval_holdout_days`) as the held-out test set.
11. **Scoring reduction.** Zero-shot comparisons default to the **mean** log-prob per unit (`score_reduction: mean`). Shuffle, Reversal and Concat pairs have equal length, so sum and mean agree there; Phee responses differ in length, so mean avoids a length bias.
12. **Segmentation thresholds are guesses.** `filter_5khz` and `segmentation` re-implement the GmSLM paper's prose; the thresholds are uncalibrated. On synthetic data P = R = 1 by construction, which says nothing about real colony audio.
13. **Nyquist.** After 16 kHz resampling only 5–8 kHz remains. The synthetic generator keeps call energy below 8 kHz, and the vocoder's additive mock maps units to 5–8 kHz tones.

---

## 5. Rosetta Stone: GmSLM ↔ Hasson ↔ our codebase

| Functionality | Talia's GmSLM (paper §; code not public) | Uri Hasson's framework (`references/hasson_dev/…`) | Our integrated codebase |
| :--- | :--- | :--- | :--- |
| High-pass and noise gating | §3.1, A.1: 5 kHz HPF, STFT 2048/512 variance + density noise filter | N/A (clean child-centred speech; no filtering) | `preprocessing/filter_5khz.py` |
| Call extraction / chunking | A.1–A.2: calls 0.25–4 s; 10 s segments @ 16 kHz | N/A (utterance files from transcripts; 2–20 s clips, random crop) | `preprocessing/segmentation.py` |
| Chronological dataset | N/A (80/10/10 random split) | `data_preparation/scripts/generate_fairseq_manifest.py` (order = CSV order) | `preprocessing/dataset_builder.py` (`build_chronological_dataset`) |
| Daily exposure F(d) | N/A | `data_preparation/scripts/generate_cycle_sizes.py` + `daily_hours_per_month.json` (words/h → files) | `dataset_builder.DailyExposureSchedule` (hours → 10 s chunks) |
| Encoder interface | textlesslib / fairseq HuBERT | `models/multimodel.py` → `self.speech2units` | `encoders/base_encoder.py` (`BaseAudioEncoder`) |
| Discretisation (HuBERT) | §3.2, A.3: HuBERT L1 + k-means 50 | N/A | `encoders/marmoset_hubert.py` |
| Discretisation (DINO-SR) | N/A | `training/speech_learning/models/dinosr.py` | `encoders/dino_sr.py` |
| Deduplication | **none** (Table 4) | `multimodel.units_to_units_durations` (**dedup + duration stream**) | **none**: `base_encoder.check_no_dedup` |
| Causal uLM | fairseq `transformer_lm_big` | fairseq `TransformerUnitLanguageModel` (MS-TLM), built in `multimodel.py` | `decoders/transformer_ulm.py` |
| Joint loss / LM gating | N/A (uLM trained after V2U) | patched `criterions/model_criterion.py` (`sigmoid-X0-K-Y` loss weight) | `agent/lr_scheduler.py` (LR gate) |
| Sleep replay | N/A | `dataset/past_sampling_dataset.py` (+ `base_cycle_dataset.py`) | `agent/replay_buffer.py` |
| Day-by-day loop | N/A | fairseq trainer + `tasks/multimodel_task.py` (per-cycle strategy, weekly saves) | `agent/trainer.py` |
| Checkpoint / reload | fairseq | fairseq `checkpoint_<epoch>_<update>.pt` | `agent/factory.py` (`save_checkpoint`, `load_checkpoint`) |
| Vocoder | §3.2, A.3: unit HiFi-GAN, no spk/F0 | N/A | `vocoder/hifigan_u2v.py` |
| Zero-shot eval | §3.3: Shuffle, Concat, Reversal, Phee eval | N/A | `evaluation/zero_shot_benchmarks.py` |
| Next-week surprisal / PPL | PPL only (static) | `multimodel_extraction/*` (`--extract-next-cycle-only`) | `evaluation/developmental_eval.py` (`DevelopmentalTracker`) |
| Lexicon discovery | N/A | `analysis/audio_postprocessing/2_smooth_probs.py`, `3_add_words_learned.py`, `4_generate_dictionary.py` | `developmental_eval.py` (`smooth_probs`, `call_is_learned`, `LearnedLexicon`) |
| Behavioural comparison | N/A | `analysis/behavioral_data/words/generate_model_eval_statistics.py` (CDI, sigmoid fit) | not yet (no marmoset equivalent of CDI) |
| Unit purity | §5.4 (low purity) | `analysis/behavioral_data/phonemes/generate_model_phoneme_eval_statistics.py` | not yet |
| Probes | §5.2, A.5: mean ⊕ var → 3 FC + LN + ReLU | N/A (UMAP of word embeddings instead) | `evaluation/downstream_probes.py` |
| ICI / pitch sensitivity | N/A | N/A | `evaluation/new_metrics.py` (new) |
| Synthetic data | N/A | N/A | `utils/synthetic_data.py`, `utils/mock_pipeline.py` |

---

## 6. File-by-file reference

### 6.1 Entry points and configuration

#### `run_pipeline.py`
- **Primary Purpose**: One command that runs the whole agent (data → preprocessing → both encoders → developmental training → vocoder → evaluation → summary) on synthetic or real data.
- **Architectural Role**: Orchestration / CLI.
- **Upstream & Downstream**: Imports almost every module. Writes `outputs/<run>/{results.json, config_resolved.yaml, checkpoints/, generated_continuation.wav}`. `tests/test_pipeline.py::test_pipeline_smoke` calls `main()`.
- **Core Functions**:
  - `main(argv) -> int`
    - **Inputs & Outputs**: flags `--mock`, `--config PATH`, `--encoder {dinosr,hubert}`, `--days N`, `--output-dir`, `--seed`, `--skip-extended`. Returns 0 on success.
    - **Key Logic**: Loads the config (mock defaults if `--mock`) and applies CLI overrides. Asserts chunk chronology. Splits days into train (first `num_days`, or all minus `eval_holdout_days`) and test (the rest). Builds **both** encoders, fits HuBERT k-means on the train chunks, and logs token stats for both. Trains with the `--encoder` one through `DevelopmentalTrainer` + `DevelopmentalTracker`. Generates 100 units from a 100-unit prompt and vocodes them. Runs zero-shot, probes and new metrics on the test days. Prints a summary.
  - `load_data(cfg, mock) -> (ChronologicalDataset, info, phee_pairs)`: mock uses `mock_dataset` + `generate_mock_phee_dialogues`. Real uses `data.chunk_manifest` (lazy) or `data.recordings_manifest` (preprocess, and export if `data.export_dir` is set); the PHEEDB loader is not implemented.
  - `token_stats(name, encoder, wav (B, L)) -> dict`: vocab, frame rate, token shape, units used, mean run length (repeats kept). Calls `check_no_dedup`.
  - `stage(i, name)`: log banner.

#### `config/default.yaml`
- **Primary Purpose**: Paper-scale settings with provenance tags `[GmSLM]` / `[1kd]` and TODO paths (`data.*`, `encoder.hubert.pretrained_path`, `vocoder.checkpoint_path`).
- **Architectural Role**: Configuration. Loaded by `load_config(path)`; unknown keys raise `KeyError`.

#### `marmoset_developmental_agent/config/agent_config.py`
- **Primary Purpose**: The single source of truth for every hyperparameter, as typed dataclasses, plus YAML (de)serialisation.
- **Architectural Role**: Configuration for all stages.
- **Upstream & Downstream**: `yaml`. Imported by every module.
- **Core Classes & Functions**:
  - `PreprocessingConfig`: sample rate, HPF cutoff/order, STFT, activity thresholds (`bin_energy_threshold_db 12`, `min/max_active_bin_fraction 0.005/0.35`, `min_spectral_variance_db2 20`), call duration limits, `merge_gap_s`, chunk length / pre-pad / min calls.
  - `ExposureScheduleConfig`: `hours_per_day_by_period`, `days_per_period`, `day_assignment ∈ {exposure, calendar}`.
  - `HuBERTEncoderConfig`: `pretrained_path`, `layer`, `n_clusters`, k-means settings, `normalize_waveform=False`, architecture fields for the random-init model.
  - `DinoSRConfig`: conv/embed dims, layers, heads, FFN, pos-conv, dropout, `codebook_size`, `average_top_k_layers`, `layer_for_tokens` (0-based), `codebook_decay`, masking, EMA schedule, `normalize_waveform=True`, `lr`, `weight_decay`.
  - `EncoderConfig`: `active ∈ {dinosr, hubert}` + both sub-configs.
  - `DecoderConfig`: uLM size.
  - `ReplayConfig`: `passes_per_day`, `kernel`, `decay_lambda`, `size_weighted`, `include_current_day`.
  - `SigmoidScheduleConfig`: `lr_max, t_p, k, floor, unit`.
  - `TrainingConfig`: `num_days, batch_size, drop_last, grad_clip, decoder_weight_decay, replay, decoder_lr, train_encoder, checkpoint_every_days, log_every_steps`.
  - `VocoderConfig`: `checkpoint_path`, HiFi-GAN shapes, `mock_mode ∈ {additive, neural}`.
  - `EvaluationConfig`: `score_reduction`, evaluation period, lexicon thresholds (Hasson defaults), probe settings (GmSLM), ICI factors, pitch shifts.
  - `DataConfig`: `recordings_manifest`, `chunk_manifest`, `export_dir`, `eval_holdout_days`.
  - `AgentConfig`: top level (+ `seed`, `device`, `output_dir`). `AgentConfig.mock()` shrinks everything for CPU. `to_dict()`.
  - `_from_dict(cls, data, base)`: recursive overlay onto dataclasses; unwraps `Optional`; **raises `KeyError` on unknown keys**.
  - `load_config(path=None, mock=False) -> AgentConfig`, `save_config(cfg, path)`.

---

### 6.2 `preprocessing/`: Phase 1

#### `preprocessing/__init__.py`
- Docstring only. It deliberately does not import submodules, so `python -m …filter_5khz` runs without a `RuntimeWarning`.

#### `preprocessing/filter_5khz.py`
- **Primary Purpose**: GmSLM's acoustic front-end: 5 kHz high-pass, spectrogram-based vocal-activity detection, optional noise gate, resampling. Also a standalone CLI.
- **Architectural Role**: Raw audio preprocessing.
- **Upstream & Downstream**: `scipy.signal`, `PreprocessingConfig`, `soundfile` (CLI). Used by `dataset_builder.preprocess_recording`; `segmentation.activity_to_calls` consumes `FrameActivity`.
- **Core Classes & Functions**:
  - `FrameActivity` (dataclass): `frame_times_s (T,)`, `active (T,) bool`, `density (T,)`, `variance_db2 (T,)`, `hop_s`.
  - `highpass_filter(audio (L,), sr, cutoff_hz=5000, order=8) -> (L,) float32`
    - **Key Logic**: `scipy.signal.butter(order, fc/nyq, 'highpass', output='sos')` + `sosfiltfilt` (zero phase, effective order 16). Raises if the cutoff is not below Nyquist.
  - `resample_audio(audio, orig_sr, target_sr) -> float32`: `resample_poly(up = target/g, down = orig/g)` with g = gcd.
  - `compute_frame_activity(audio, sr, cfg) -> FrameActivity`
    - **Key Logic**: STFT (n_fft 2048, hop 512, no boundary padding); keep bins ≥ cutoff; `S_db = 10 log10|X|²`; `floor_f = median_t S_db`; `rel = S_db − floor`; `density_t = mean_f[rel > θ]`; `variance_t = var_f(rel)`; `active = ρ_min ≤ density ≤ ρ_max ∧ variance ≥ v_min`. Tonal calls are narrow and peaky; clicks are broadband; hiss is flat.
  - `noise_gate(audio, sr, cfg, attenuation_db=40, activity=None) -> (gated (L,), FrameActivity)`: frame gains (1 or 10^(−40/20)) linearly interpolated to samples.
  - `preprocess_waveform(audio, sr, cfg) -> (filtered_native (L,), FrameActivity)`: HPF, then activity.
  - `main(argv)` / `_read_audio`: CLI `input output [--cutoff --order --target-sr --gate --attenuation-db]`; mixes to mono; prints the fraction of active frames.

#### `preprocessing/segmentation.py`
- **Primary Purpose**: Turns frame activity into call events and packs calls into fixed 10 s, 16 kHz training chunks.
- **Architectural Role**: Raw audio preprocessing (call extraction, chunking).
- **Upstream & Downstream**: `FrameActivity`, `PreprocessingConfig`. `Chunk` and `CallSegment` are the **shared currency** of the trainer, evaluation and dataset builder.
- **Core Classes & Functions**:
  - `CallSegment(start_s, end_s, call_type=None, caller_id=None, receiver_id=None)`, with `duration_s`.
  - `Chunk(waveform (160000,) | None, sample_rate, recording_id, start_time, offset_s, calls, path=None, recording_day=None)`: `audio()` returns the in-memory array or lazily reads `path`.
  - `activity_to_calls(activity, cfg) -> List[CallSegment]`
    - **Key Logic**: indices of active frames → split where the gap between consecutive active frames exceeds `merge_gap_s + hop` → onset/offset = first/last frame centre ∓ hop/2 → keep durations in [0.25, 4] s.
  - `chunk_recording(audio_16k, calls, cfg, recording_id, start_time, recording_day) -> List[Chunk]`
    - **Key Logic**: Greedy loop over sorted calls. A window starts at `call.start − pre_pad` (never overlapping the previous chunk) and spans 10 s. Every call ending inside it is annotated with times relative to the chunk. The next window starts at the first call that did not fit; a call that overruns a window is skipped. The tail is zero-padded. Chunks with fewer than `min_calls_per_chunk` calls are dropped.
  - `label_calls_from_annotations(detected, annotations, min_overlap=0.5)`: copies ground-truth labels by maximum overlap (mock data / labelled sets).
  - `detection_scores(detected, annotations, min_overlap=0.5) -> {precision, recall, n_detected, n_annotated}`.

#### `preprocessing/dataset_builder.py`
- **Primary Purpose**: Sorts recordings in time, runs the front-end, and cuts the chunk stream into simulated days with F(d). It also handles disk manifests for real-scale data.
- **Architectural Role**: Dataset construction (the developmental timeline).
- **Upstream & Downstream**: `filter_5khz`, `segmentation`, configs, `soundfile`. Produces `ChronologicalDataset` for `trainer`, `DevelopmentalTracker` and `run_pipeline`.
- **Core Classes & Functions**:
  - `Recording(audio (L,), sample_rate, start_time, recording_id, day=None, annotations=[])`.
  - `DailyExposureSchedule(cfg, seconds_per_sample)`
    - `hours(d)`: `list[min(d // days_per_period, last)]`.
    - `samples_for_day(d) = ⌊hours(d)·3600 / seconds_per_sample⌋` (raises if 0).
    - `day_sizes(total)`: consumes the stream day by day; the last day may be partial.
  - `ChronologicalDataset(chunks, day_sizes)`: `num_days`, `day_indices(d) -> np.ndarray` (global indices), `day(d)`, `days(start, stop)`, `day_of(index)`, `waveforms(indices) -> (B, 160000)`, `summary()`.
  - `preprocess_recording(rec, cfg) -> (chunks, detection_stats)`: HPF + activity at the native rate → calls (+ labels if annotated) → resample → chunk.
  - `build_chronological_dataset(recordings, pre_cfg, exp_cfg) -> (dataset, info)`: sorts by `start_time`, preprocesses, re-sorts chunks by time, and assigns days (`exposure` via `DailyExposureSchedule`, or `calendar` via runs of `recording_day` / date). `info` carries the pooled detection precision and recall when annotations exist.
  - `load_recordings_from_manifest(csv)`: CSV `path,start_time[,recording_id]`, eager load, mono.
  - `export_manifest(dataset, out_dir) -> Path`: `chunks/%08d.wav` + `manifest.tsv` (`path, n_samples, day, recording_id, start_time, calls "a-b;c-d"`) + `day_sizes.json`.
  - `load_manifest(tsv, sr)`: lazy `Chunk(waveform=None, path=…)` dataset.

---

### 6.3 `encoders/`: Phase 2 (vocalization → units)

#### `encoders/__init__.py`
- Re-exports `BaseAudioEncoder, DinoSREncoder, MarmosetHuBERTEncoder, check_no_dedup, run_lengths`.

#### `encoders/base_encoder.py`
- **Primary Purpose**: The abstract contract every discretiser must satisfy, plus the no-dedup guard.
- **Architectural Role**: Feature extraction / tokenisation interface.
- **Upstream & Downstream**: Subclassed by both encoders. `check_no_dedup` is called by the trainer and `run_pipeline`.
- **Core Classes & Functions**:
  - `CONV_LAYERS = ((10,5),(3,2),(3,2),(3,2),(3,2),(2,2),(2,2))`.
  - `conv_output_length(num_samples, layers) -> int`: `L ← ⌊(L − k)/s⌋ + 1` per layer (160000 → 499).
  - `BaseAudioEncoder(nn.Module, ABC)`
    - Class attributes: `sample_rate = 16000`, `is_trainable = False`.
    - `frame_rate` = 16000 / Π strides = 50.0; `num_frames(L)`.
    - Abstract: `get_vocab_size() -> int`, `extract_features(wav (B, L)) -> (B, T, D)`, `encode(wav (B, L)) -> LongTensor (B, T)`.
    - Hooks: `fit(wavs)`, `training_step(wav) -> loss | None`, `after_optimizer_step()`, `trainable_parameters()`.
    - Helpers: `_as_batch` (1-D → (1, L); checks the dimensions), `_normalize` (per-utterance `layer_norm`, same as fairseq `normalize: true`).
  - `check_no_dedup(tokens, num_samples, encoder)`: raises `AssertionError` unless `tokens.shape[-1] == num_frames(num_samples)`.
  - `run_lengths(tokens (T,)) -> (values, counts)`: `unique_consecutive`, **analysis only**.

#### `encoders/marmoset_hubert.py`
- **Primary Purpose**: GmSLM's V2U final step: HuBERT layer-1 features → k-means (50) → units. The model is frozen.
- **Architectural Role**: Tokenisation (the "innate or pre-learned" perception option).
- **Upstream & Downstream**: `transformers.HubertModel/HubertConfig`, `sklearn.cluster.MiniBatchKMeans`. Built by `factory.build_encoder("hubert")`.
- **Core Classes & Functions**:
  - `MarmosetHuBERTEncoder(cfg, seed=0)`
    - `__init__`: `HubertModel.from_pretrained(pretrained_path)`, or a random model built from the config fields (conv stack = `CONV_LAYERS`, `apply_spec_augment=False`). Freezes all weights and registers an empty `centroids` buffer (0, D).
    - `extract_features(wav (B, L)) -> (B, T, D)`: optional normalisation → `hidden_states[cfg.layer]`.
    - `fit(wavs: Iterable[(1|B, L)])`: extracts features, keeps a random `kmeans_fit_fraction` of frames, runs `MiniBatchKMeans(n_clusters, batch_size, n_init, k-means++)` and stores `cluster_centers_` in `centroids` (K, D).
    - `encode(wav) -> (B, T)`: `torch.cdist(features, centroids).argmin(-1)`. **One unit per frame.** Raises if not fitted.
    - `is_fitted`, `load_kmeans(centroids (K, D))` (import external k-means), `train(mode)` (keeps HuBERT in eval), `load_state_dict` (resizes `centroids` first).
    - `is_trainable = False`.

#### `encoders/dino_sr.py`
- **Primary Purpose**: Hasson's developmental encoder. DINO-SR self-distillation with online codebooks; it learns *during* the developmental run.
- **Architectural Role**: Tokenisation + self-supervised encoder training.
- **Upstream & Downstream**: torch only. Called by the trainer (`training_step`, `after_optimizer_step`, `encode`), evaluation (`encode`, `extract_features`) and the factory.
- **Core Classes & Functions**:
  - `ConvFeatureExtractor(dim)`: 7 × (Conv1d without bias → LayerNorm over channels → GELU). `(B, L) → (B, T, dim)`.
  - `_TransformerStack(cfg)`: grouped conv positional embedding (odd kernel, GELU, residual) → LayerNorm → dropout → N post-LN `nn.TransformerEncoderLayer` (GELU, batch_first). `forward(x (B, T, D)) -> (final, [each layer's output (B, T, D)])`.
  - `compute_span_mask(B, T, p, span, generator) -> (B, T) bool`: `n_spans = max(1, ⌊p·T/span + U(0,1)⌋)` random starts per row, overlaps allowed.
  - `DinoSREncoder(cfg, seed)`
    - Modules: `feature_extractor`, `feat_norm`, `post_extract_proj` (conv_dim → D), `mask_emb` (D), `student`, `teacher` (deep copy, no grads), K `heads` (D → V).
    - Buffers: `codebook_sums (K, V, D)` initialised to randn/√V; `codebook_counts (K, V)` ones; `num_updates`.
    - `token_codebook = layer_for_tokens − num_layers + K` (validated).
    - `_pre_encoder(wav) -> (B, T, D)`: normalise → conv → LN → projection.
    - `_teacher_targets(feats) -> K × (B, T, D)`: teacher in eval, no grad; top-K layer outputs, each instance-normalised over time.
    - `_nearest(x (N, D), codewords (V, D)) -> (N,)`: argmin of `‖x‖² + ‖c‖² − 2x·c`.
    - `training_step(wav) -> scalar`
      - **Key Logic**: feats → span mask (p 0.8, span 10) → masked positions replaced by `mask_emb` → student → for each top-K layer i: `z = nearest(y_i[mask], c_i)`; `loss += CE(heads[i](student_out[mask]), z, reduction='sum')`; EMA codebook update; returns `loss / #masked`. **This sums over K; the official code averages over K. See Gotcha 5.**
    - `_update_codebook(i, y (M, D), ids (M,))`: `α = decay` where count > 0, else 1; `counts ← α·counts + (1−α)·count`; `sums ← α·sums + (1−α)·Σy`.
    - `after_optimizer_step()`: `θ_T ← τθ_T + (1−τ)θ_S` with τ linearly annealed from `ema_decay` to `ema_end_decay` over `ema_anneal_end_step`; `num_updates += 1`.
    - `extract_features(wav, layer=None) -> (B, T, D)`: teacher output of the 0-based `layer` (default `layer_for_tokens`), instance-normed.
    - `encode(wav) -> (B, T)`: nearest codeword in `token_codebook`. **No dedup.**
    - `trainable_parameters()`: everything except `teacher.*`. `is_trainable = True`.

---

### 6.4 `decoders/`

#### `decoders/__init__.py`
- Re-exports `UnitLanguageModel`.

#### `decoders/transformer_ulm.py`
- **Primary Purpose**: Causal Transformer unit language model (GmSLM's uLM, frame-level units).
- **Architectural Role**: Causal sequence modelling. Provides the training loss and every likelihood used in evaluation.
- **Upstream & Downstream**: `DecoderConfig`. Trained by `trainer`; scored by `evaluation/scoring`; hidden states feed `downstream_probes`; `generate` feeds the vocoder.
- **Core Classes & Functions**:
  - `UnitLanguageModel(n_units, cfg)`: `vocab = n_units + 3` (PAD 0, BOS 1, EOS 2); token embedding + learned positions (`max_positions + 1`); pre-LN `nn.TransformerEncoder` with a causal mask; final LayerNorm; LM head **tied** to the embedding.
  - `_prepare(units (B, T), lengths (B,)?) -> (inp, tgt, pad)`: `inp = [BOS, u+3 …]`, `tgt = [u+3 …, EOS]`; positions after `lengths` are PAD. Sequences are truncated to `max_positions`.
  - `forward(ids (B, T'), pad_mask (B, T')?, return_hidden=False) -> logits (B, T', V) | hidden (B, T', E)`.
  - `loss(units, lengths=None) -> scalar`: mean next-token CE (nats), EOS included, PAD ignored.
  - `token_log_probs(units, lengths) -> (logp (B, T), valid (B, T))`: `log p(u_t | u_<t)` for each unit (EOS excluded).
  - `sequence_log_prob(units, lengths, reduction='sum'|'mean') -> (B,)`.
  - `hidden_states(units, lengths) -> (B, T, E)`: final-layer state *after reading* each unit.
  - `generate(prompt (T0,), max_new_tokens, temperature=1.0, top_k=None, generator=None) -> (T0 + n,)`: ancestral sampling (specials excluded from sampling). GmSLM used beam search instead.

---

### 6.5 `agent/`: Phase 3 (developmental training)

#### `agent/__init__.py`
- Re-exports `DayLog, DaySchedule, DevelopmentalTrainer, SigmoidLRScheduler, SleepReplayBuffer, sigmoid_gate`.

#### `agent/lr_scheduler.py`
- **Primary Purpose**: Keeps the decoder nearly frozen early on, then opens it smoothly around T_p.
- **Architectural Role**: Training schedule (staged plasticity).
- **Core Classes & Functions**:
  - `sigmoid_gate(t, t_p, k, floor=0) -> float`: `1/(1+e^{−k(t−t_p)}) + floor`, computed in an overflow-safe way for both signs of z.
  - `SigmoidLRScheduler(optimizer, cfg)`
    - `lr_at(t) = lr_max·sigmoid_gate(t)`. `step(t) -> lr` writes that LR into every param group. `get_last_lr()`, `state_dict()`, `load_state_dict()`.
    - **Key Logic**: Same (x0, k, y_shift) triple as Hasson's `sigmoid-X0-K-Y`, applied to the LR instead of the loss weight.

#### `agent/replay_buffer.py`
- **Primary Purpose**: Sleep consolidation. Decides which past chunks are replayed each simulated day.
- **Architectural Role**: Training data ordering / memory replay.
- **Core Classes & Functions**:
  - `DaySchedule(day, wake (n_D,), sleep ((X−1)·n_D,))`, with `order = concat(wake, sleep)`.
  - `SleepReplayBuffer(cfg, seed)`
    - `add_day(day, indices)`, `days`.
    - `day_probabilities(D) -> (days, p)`
      - **Key Logic**: `Δ = D − d`; `w = e^{−λΔ}` or `1/(Δ+1)`; optionally `× n_d`; normalised. Uses days ≤ D (or < D).
    - `sample_replay(D, n) -> (n,)`: day ids ~ `p`, then chunk indices uniform with replacement inside each day, shuffled.
    - `build_day_schedule(D, today_indices) -> DaySchedule`: registers today and draws `(X−1)·n_D` replay samples.

#### `agent/trainer.py`
- **Primary Purpose**: The chronological day loop: wake + sleep batches, encoder self-supervision, decoder next-unit prediction, evaluation hooks and checkpoints.
- **Architectural Role**: Training loop.
- **Upstream & Downstream**: `SleepReplayBuffer`, `SigmoidLRScheduler`, `BaseAudioEncoder`, `UnitLanguageModel`, `ChronologicalDataset`. Called by `run_pipeline`. `on_day_end` is typically a `DevelopmentalTracker`.
- **Core Classes & Functions**:
  - `DayLog`: `day, n_wake, n_sleep, steps, global_step, encoder_loss, ulm_loss_wake, ulm_loss_sleep, decoder_lr_start/end, seconds`.
  - `DevelopmentalTrainer(encoder, ulm, dataset, cfg, encoder_lr, encoder_weight_decay, device, seed, checkpoint_dir, checkpoint_fn)`
    - Checks that the encoder vocab equals the uLM `n_units`. Uses separate **AdamW** optimisers (β = (0.9, 0.98), eps 1e-6); the encoder optimiser exists only if the encoder is trainable and `train_encoder` is set.
    - `_schedule_t(day, frac)`: global step, or `day + frac`.
    - `_tokens(idx, wav)`: encodes, or reads the per-chunk cache for frozen encoders, then calls `check_no_dedup`.
    - `train_step(idx, day, frac) -> {encoder_loss?, decoder_lr, ulm_loss}`
      - **Key Logic**: `wav = dataset.waveforms(idx)` (B, 160000). If the encoder trains: `training_step` → backward → clip → step → `after_optimizer_step` (EMA). Then `units = encode(wav)` (B, 499) → set the decoder LR → `ulm.loss(units)` → backward → clip → step. Increments `global_step`.
    - `_batches(order)`: fixed-size slices (remainder kept unless `drop_last`).
    - `train_day(D) -> DayLog`: builds the schedule, iterates the batches, and averages losses separately for wake and sleep.
    - `run(num_days=None, on_day_end=None) -> List[DayLog]`: days 0..n−1. After each day it calls `on_day_end(day, log)`, and checkpoints every `checkpoint_every_days` and on the last day.
    - `history_dicts()`.

#### `agent/factory.py`
- **Primary Purpose**: Builds encoders and uLMs from a config, and saves/loads complete agent checkpoints.
- **Architectural Role**: Construction and persistence.
- **Core Functions**:
  - `resolve_device(name)`: `auto` → CUDA if available, else CPU.
  - `build_encoder(cfg, name=None) -> BaseAudioEncoder`; `build_ulm(cfg, n_units) -> UnitLanguageModel` (seeded).
  - `save_checkpoint(path, cfg, encoder_name, encoder, ulm, extra)`: `{config (dict), encoder_name, encoder (state), ulm (state), n_units, extra}`.
  - `load_checkpoint(path, device) -> (cfg, encoder, ulm)`: rebuilds from the stored config, then loads the state dicts.

---

### 6.6 `vocoder/`: Phase 4

#### `vocoder/__init__.py`
- Re-exports `HiFiGANU2V, UnitHiFiGANGenerator`.

#### `vocoder/hifigan_u2v.py`
- **Primary Purpose**: Unit → waveform synthesis (a Polyak-style unit HiFi-GAN generator), with a mock renderer when there is no trained checkpoint.
- **Architectural Role**: Audio reconstruction.
- **Core Classes & Functions**:
  - `ResBlock(ch, kernel, dilations)`: HiFi-GAN v1 dilated conv pairs with residual connections, LeakyReLU 0.1.
  - `UnitHiFiGANGenerator(n_units, cfg)`: `Embedding(V, 128)` → `Conv1d(128→C, 7)` → for each (rate, kernel): LeakyReLU → `ConvTranspose1d(ch → ch/2, k, stride r, padding (k−r)/2)` → multi-receptive-field fusion (mean of ResBlocks with kernels 3/7/11) → LeakyReLU → `Conv1d(→1, 7)` → tanh. `forward(units (B, T)) -> (B, T·320)`.
  - `HiFiGANU2V(n_units, cfg, sample_rate=16000)`: loads `checkpoint_path` (key `generator` or a raw state dict); `is_mock` if none; `hop_length` = 320.
    - `synthesize(units (T,) | (B, T)) -> (B, T·320)`: generator, or `_additive` when mocking.
    - `_additive(units)`: phase-continuous sine with `f(u) = 5 kHz + 3 kHz·u/(V−1)`, amplitude 0.3.
  - Training (discriminators, losses) is **not** implemented.

---

### 6.7 `evaluation/`: Phase 5

#### `evaluation/__init__.py`
- Docstring only.

#### `evaluation/scoring.py`
- **Primary Purpose**: Shared "waveform → units → log-likelihood" utilities, so every benchmark scores the same way.
- **Core Classes & Functions**:
  - `encode_waveforms(encoder, wavs: List[(L_i,)], batch_size=16) -> List[(T_i,)]`: batches waveforms of equal length together.
  - `token_log_probs(ulm, units: List[(T_i,)], batch_size) -> List[(T_i,) np]`: pads, uses `lengths`, and returns only the valid positions.
  - `score_waveforms(encoder, ulm, wavs, reduction='mean'|'sum') -> (N,)`.
  - `SurprisalReport(n_sequences, n_tokens, mean_nll_nats, mean_surprisal_bits, perplexity)`; `surprisal_report(log_probs)`: token-weighted. `S = NLL / ln 2`, `PPL = e^{NLL}`.
  - `pairwise_accuracy(real, fake) -> float`: `mean[(r > f) + ½(r == f)]`.
  - `LN2` constant.

#### `evaluation/zero_shot_benchmarks.py`
- **Primary Purpose**: GmSLM's zero-shot tests: does the uLM prefer real marmoset audio over controlled distractors? Also a CLI.
- **Core Classes & Functions**:
  - `_spans(calls, sr, n)`: sorted, non-overlapping sample spans.
  - `shuffle_calls(wav, calls, sr, rng) -> wav | None`: permutes the call snippets into the existing call slots; gaps stay in place; guaranteed to differ from the identity; same length; None if fewer than 2 calls.
  - `reverse_audio(wav)`, `concat_halves(a, b)` (first half of A + second half of B at the **time midpoint**, same length).
  - `_dialogue(call, gap_s, response, sr)`: call + silence + response.
  - `build_phee_pairs(pairs, sr, rng) -> {"CallerChange": (real, fake), "ReceiverChange": (real, fake)}`: the pool is every vocalisation labelled (emitter, addressee). CallerChange uses a response from an individual that is neither the caller nor the true responder. ReceiverChange uses a response from the same responder that was addressed to someone other than the caller.
  - `ZeroShotBenchmarks(encoder, ulm, cfg, seed)`: `_acc(real, fake) -> {accuracy, n_pairs, mean_delta_logp}`. `run(chunks, phee_pairs) -> {task: …}` (Concat partners drawn from other eligible chunks; at most `max_eval_chunks`).
  - `format_results(results) -> str` (table); `main(argv)`: CLI `--checkpoint`, `--manifest`, `--mock`, `--encoder`, `--output`.

#### `evaluation/developmental_eval.py`
- **Primary Purpose**: Weekly developmental trajectory: next-period surprisal and PPL, out-of-domain PPL, and the learned call lexicon (a marmoset adaptation of Hasson's word-learning rule).
- **Core Classes & Functions**:
  - `smooth_probs(p (T,), factor=0.5, weights=None) -> (T,)`: Hasson's `mean_weighted_average`. With no dedup, weights are 1 per frame; Hasson uses encoder confidence × run length.
  - `call_is_learned(m, confidence, patience, stability, min_coverage) -> bool`: break where `m < conf` or (`m < patience` and `Δm < stability`). The call is learned if its longest unbroken, confident run covers at least `min_coverage` of its frames.
  - `LexiconEntry(call_type, first_learned_day, history)`.
  - `LearnedLexicon(cfg, frame_rate)`: `update(day, chunks, log_probs) -> {call_type: learned_fraction}`. A type enters the lexicon the first time its learned fraction ≥ `lexicon_min_learned_fraction`. Also `size`, `as_dict()`.
  - `DevelopmentalTracker(encoder, ulm, dataset, cfg, ood_waveforms)`: a callable `on_day_end`. Every `eval_every_days` it scores the **unseen** days D+1..D+`eval_every_days` (at most `max_eval_chunks`), updates the lexicon and computes OOD PPL (the mock uses concatenated phee pairs). `history` holds a list of dicts.

#### `evaluation/downstream_probes.py`
- **Primary Purpose**: GmSLM-style classifiers on pooled representations, for call type and caller identity.
- **Core Classes & Functions**:
  - `pool_mean_var(x (T, D)) -> (2D,)`.
  - `ProbeMLP(in_dim, hidden=[256,128,64], n_classes)`: 3 × (Linear → LayerNorm → ReLU) → Linear.
  - `ProbeResult`: `task, accuracy, precision, recall, f1 (macro), n_train, n_val, n_classes, chance (majority rate)`.
  - `extract_call_representations(encoder, ulm, chunks, source='ulm'|'encoder') -> (X (N, 2D), call_types, caller_ids)`: per labelled call, frames `[⌊s·50⌋, ⌈e·50⌉)` of `ulm.hidden_states` or `encoder.extract_features`.
  - `train_probe(X, labels, cfg, task, seed) -> ProbeResult`: stratified 90/10 split when possible, z-scored on the train split, Adam + `PolynomialLR` (power 1), minibatch 64, `probe_epochs` epochs.
  - `run_probes(...)`: both tasks, skipping any with fewer than 10 samples or fewer than 2 classes.

#### `evaluation/new_metrics.py`
- **Primary Purpose**: New sensitivity tests for inter-call timing and pitch.
- **Core Functions**:
  - `_fit(x, n)`: tile or truncate to length n.
  - `rescale_intercall_intervals(wav, calls, factor, sr) -> wav | None`: scales each gap between calls by `factor` (gap audio tiled, not zeroed); calls unchanged; trimmed or padded (with tiled post-call background) back to the original length.
  - `pitch_shift_calls(wav, calls, semitones, sr)`: `librosa.effects.pitch_shift` on call regions only (skips snippets shorter than 512 samples).
  - `_condition(encoder, ulm, orig, pert) -> {accuracy, delta_surprisal_bits, n}`: mean-reduced scores; Δ = (log p_orig − log p_pert) / ln 2.
  - `ici_distortion_sensitivity(...)` → keys `x0.25`, `x0.5`, `x2`, `x4`; `pitch_shift_sensitivity(...)` → keys `-2st` … `+2st`.

---

### 6.8 `utils/`

#### `utils/__init__.py`
- Empty.

#### `utils/synthetic_data.py`
- **Primary Purpose**: Synthetic colony audio with known ground truth, so every code path runs without data.
- **Architectural Role**: Mock data generation.
- **Core Classes & Functions**:
  - `CALL_TYPES = (phee, trill, twitter, tsik)`. `CALL_TRANSITIONS` (4 × 4 Markov matrix) gives the call order structure that Shuffle and Concat should detect.
  - `CallAnnotation`, `SyntheticRecording(audio, sample_rate, day, start_time, recording_id, calls)`, `PheeExchange(call_audio, response_audio, gap_s, caller_id, responder_id, response_receiver_id)`.
  - `_tone(freq (L,), sr)`: phase-integrated sine. `_envelope(n, attack)`.
  - `synth_call(type, sr, rng, caller_id, receiver_id) -> (L,)`: phee 0.8–2 s around 6.5 kHz with an FM rate set by the **receiver** (toy vocal labelling); trill 28 Hz FM at 7 kHz; twitter 5–9 upward sweeps; tsik a 0.25–0.35 s down-sweep. The caller shifts f0 by 150 Hz · (id mod 6).
  - `_background(n, sr, rng, human_speech)`: hiss + 50 Hz hum + harmonic "voice" below 3 kHz + 5 ms clicks roughly every 4 s (targets for the HPF and the duration gate).
  - `generate_recording(duration_s, sr, rng, day, start_time, n_individuals, human_speech)`: Markov-ordered calls, 80% short ICIs (0.3–1.2 s) and 20% long pauses (2–5 s).
  - `generate_mock_colony(n_days=3, seconds_per_day=60, recordings_per_day=2, native_sample_rate=48000, …, shuffle_order=True)`: recordings returned in **shuffled** order on purpose.
  - `generate_mock_phee_dialogues(n_individuals=4, n_pairs=24, sr=16000, seed)`: real pairs (the response is addressed to the caller).
  - `mock_daily_metadata(recordings)`: manifest-like rows.

#### `utils/mock_pipeline.py`
- `to_recording(SyntheticRecording) -> Recording` (annotations → `CallSegment`); `mock_recordings(cfg, n_calendar_days=4, seconds_per_day=70)`; `mock_dataset(cfg, …) -> (ChronologicalDataset, info)`. Shared by `run_pipeline`, the zero-shot CLI and the tests.

#### `utils/log.py`
- `setup_logging(level)`: `HH:MM:SS | LEVEL | logger | msg` to stdout; quiets numba, matplotlib, urllib3 and transformers.

---

### 6.9 `tests/test_pipeline.py`
- **Primary Purpose**: 21 fast sanity checks (about 22 s). Fixtures: `cfg` (mock), `dataset` (2 calendar days × 40 s), `hubert` (k-means fitted on 4 chunks).

| Test | Guards |
|---|---|
| `test_conv_frame_count` | 160,000 samples → 499 frames |
| `test_hubert_shapes_and_repeats_preserved` | (2, 499) long, vocab 50, runs > 10 frames survive, 50 Hz |
| `test_check_no_dedup_rejects_collapsed` | a deduplicated sequence raises |
| `test_dinosr_training_step_and_encode` | finite loss, gradients reach the heads but never the teacher, EMA moves the teacher, tokens (2, 499) < 512 |
| `test_dinosr_rejects_layer_without_codebook` | an invalid `layer_for_tokens` raises |
| `test_ulm_loss_shapes_and_causality` | changing token 30 leaves log p of tokens < 30 unchanged |
| `test_ulm_respects_lengths` | padded scoring equals unpadded scoring |
| `test_replay_exponential_probabilities` | p ∝ e^{−λΔ}, increasing with recency |
| `test_replay_inverse_recency_matches_official` | equals the `past_sampling_dataset.py` formula |
| `test_replay_day_schedule_counts_and_sources` | wake = today, \|sleep\| = 24·n_D, only days ≤ D, recency bias |
| `test_replay_exclude_current_day` | `include_current_day=False` |
| `test_sigmoid_lr_schedule` | lr(T_p) = lr_max·(0.5 + floor), monotone, overflow-safe |
| `test_exposure_schedule_day_sizes` | F(d) arithmetic, last value repeats, partial last day |
| `test_highpass_5khz` | 1 kHz attenuated by > 60 dB, 7 kHz within 0.5 dB |
| `test_dataset_chronological_and_chunk_shapes` | sorted times, 160,000 samples, call limits |
| `test_zero_shot_perturbations` | shuffle preserves the audio material, reverse∘reverse = id, concat halves |
| `test_ici_rescale` | factor 1 = identity; factor 2 moves call 2 correctly |
| `test_lexicon_learning_rule` | flat high probability → learned; a dip → not learned; smoothing |
| `test_vocoder_output_length` | T·320 samples in both mock modes |
| `test_config_yaml_roundtrip` | save → load is identical; unknown key → `KeyError` |
| `test_pipeline_smoke` | `run_pipeline.main --mock --encoder hubert --days 2` → rc 0, all 5 zero-shot tasks, checkpoint written |

### 6.10 Package `__init__.py` files
- `marmoset_developmental_agent/__init__.py`: docstring + `__version__ = "0.1.0"`.
- `config/__init__.py`: re-exports `AgentConfig, load_config, save_config`.
- The others are listed in their sections above.
