# Conventions and gotchas

Detail moved out of `CLAUDE.md` to keep that file short. Read the section that matches the code you're touching.

## Layout
```
run_pipeline.py                 CLI orchestrating stages 1-7
config/default.yaml             paper-scale overrides (provenance-tagged)
marmoset_developmental_agent/
  config/agent_config.py        all dataclasses; AgentConfig.mock() = tiny CPU config; load_config/save_config (unknown YAML keys raise)
  io/                           raw_reader (bext timestamps, probe, streaming 4-ch blocks), channels (5 channel strategies)
  preprocessing/                filter_5khz (HPF + frame activity) -> segmentation (calls, night window, 10 s Chunk) -> dataset_builder (ChronologicalDataset, F(d))
  encoders/                     BaseAudioEncoder ABC; MarmosetHuBERTEncoder (frozen, k-means); DinoSREncoder (trainable, EMA teacher, online codebooks)
  decoders/transformer_ulm.py   UnitLanguageModel (PAD=0, BOS=1, EOS=2, unit u -> id u+3)
  agent/                        lr_scheduler (sigmoid gate), replay_buffer (wake + sleep), trainer (day loop), factory (build/save/load)
  vocoder/hifigan_u2v.py        unit HiFi-GAN generator; mock "additive" tone renderer when no checkpoint
  evaluation/                   scoring (shared), zero_shot_benchmarks, developmental_eval (tracker + lexicon), downstream_probes, new_metrics
  utils/                        synthetic_data (mock colony + PHEEDB), mock_pipeline, log
tools/
  build_manifest.py, sweep_units.py
  analysis/                     level_audit, aggregate_levels, verify_talia_port (the port gate), channel_eval
  labels/                       make_clips, make_guide, import_audacity, review_labels (hand-label workflow, see LABELING.md)
  cluster/                      env.sh, setup_env.sh, *.sbatch (ELSC SLURM jobs)
references/                     read-only upstream code, git-ignored (never import from it)
```
Other outputs: `outputs/<run>/results.json`, `config_resolved.yaml`, `checkpoints/day_XXXX.pt` (weekly, plus the last day), `generated_continuation.wav`.

## Paper vs. code: who wins
The GmSLM code is not public, but Talia's working tree was audited (`gmslm_audit/GMSLM_CODE_AUDIT.md`). Constants are tagged `[paper]` / `[talia]` / `[ours]` in `PreprocessingConfig`.
- **Rounding-level gap: the code wins.** Example: her 10-frame call floor is 0.2322 s, which the paper rounds to 0.25 s.
- **Large gap: the paper wins.**
- **Paper physically implausible, code field-standard: the code wins (paper typo).** Examples: mask_prob 0.80, not 8%; LR 3.5e-4, not 3.5e-3. The paper also prints an impossible "5e-45".
- **Neither source decided** (library defaults): we decide. See the ROADMAP ablations and `OPEN_DECISIONS.md`.

## Preprocessing
- **Durations live in seconds, never in STFT frames.** Her thresholds were frame counts that only mean anything at hop 512 / 22050 Hz. `PreprocessingConfig.to_frames()` converts, and a test pins every constant to the exact frame count she used.
- **`frame_features: "talia" | "spectral"`** selects the detector.
  - `"talia"` is a faithful port of `segment_processor.py`. It reproduces her `call_indicies.pkl` exactly (23/23 at IoU 1.00 on 210503_001), and that match is the verification gate.
  - `"spectral"` is ours and is calibrated against hand labels. Mock runs use it, because her `amp_threshold` is an **absolute** |STFT| threshold (in librosa scaling) tied to the colony's recording gain.
- **Detection rate and filtering.** Detection runs at `detection_sample_rate` (22050 Hz, librosa's default resample, not the native 48 kHz). The 5 kHz high-pass applies to **detection only**: downstream audio is unfiltered unless `highpass_training_audio` is set.
- **STFT scaling.** librosa's |STFT| is 1024× scipy's normalised STFT. `_stft` multiplies by the window sum, and a test pins this. Any new level or threshold code must use the same scaling.
- **Recording level tracks day vs. night** (r = 0.86), not date. About 46% of the corpus is night and near-silent. `exclude_night` is opt-in (default off) until the hand labels decide it.
- **Channel strategies** (`io/channels.py`): `ambisonic_beam` has about 2× the on-axis gain of `mean`, so with absolute thresholds always compare strategies gain-invariantly.
- **Hand labels:** `LABELING.md`; imported to `data/labels/handlabels.csv`. They are the only ground truth for detector precision and recall. Her tree has no hold-out set.

## Encoders and decoder
- **Layer indexing differs.** HuBERT `layer` is 1-based (output of layer i; `hidden_states[i]`). DINO-SR `layer_for_tokens` is **0-based**, matching the official code: 9 means the 10th of 12 layers, and it must lie in the top-K codebook layers.
- HuBERT without `pretrained_path` is random-init, so its units are meaningless (the pipeline logs a warning). A real marmoset HuBERT must first be converted from fairseq to HF format.

## Agent
- **Replay.** `passes_per_day` X counts *all* passes: a day = n_D wake + (X-1)·n_D sleep samples, drawn from days ≤ D including today. `kernel: exponential` is the project spec. `inverse_recency` with `size_weighted: true` reproduces `past_sampling_dataset.py` exactly (a test checks this).
- **Sigmoid.** Official code gates the uLM *loss weight* (`sigmoid-X0-K-Y`, per update, with a shared Adam). We gate the decoder *LR* with a separate optimizer. The curve and parameters are the same; see the `lr_scheduler.py` docstring for why.
- Mock results near 50% on the zero-shot tasks are expected: the models are tiny and train for about 40 steps. The mock checks plumbing, not science.

## Cluster (ELSC)
- Login: `ssh elsc-login-direct` (lab network only). Source `tools/cluster/env.sh` (there is no `module` in batch shells).
- Write only inside `$HOME` / `WORK_DIR`. Raw recordings and other labs' trees are read-only: never move, delete or rewrite them.
- Heavy work goes through `sbatch`, not the login node. Use `$SLURM_SUBMIT_DIR`, not `$(dirname $0)`, inside jobs.
- `SESSION_STATE.md` is the resume point after a dropped connection; update it before stopping.
