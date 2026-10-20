# GmSLM Code & Data Audit — Talia Sternberg's working folder

Audited folder (TALIA_PATH): `/ems/elsc-labs/omer-d/talia.sternberg` (~8 TB, almost all data)
Code mirror used for reading: `/ems/elsc-labs/omer-d/zvi.marmor/gmslm_audit/code_mirror`
All paths below are ORIGINAL paths under TALIA_PATH unless marked otherwise.

---

## 1. Summary

Talia's folder contains a real, mostly-working implementation of the GmSLM pipeline, but it is a snapshot of many parallel experiments rather than one clean pipeline, and several pieces needed to reproduce the paper's reported numbers are either missing, contradictory, or only reconstructable by inference. The preprocessing/call-detection stage is fully recovered as working frame-based code (Butterworth 5th-order high-pass at 5000 Hz, run on librosa's default 22050 Hz mono mix-down of the raw audio — notably collapsing the raw recorder's 4 ambisonic channels into one signal with no documented rationale), and the **final dataset is confirmed**: `firstTrial`'s 10-second segments give exactly 216,660 segments in an 80/10/10 split, matching the paper's COLONYDB figure exactly. Three things we did not expect: (1) the **labelled hold-out set behind the paper's reported 0.975 precision / 0.78 recall for call detection does not exist anywhere in this folder** — it was evaluated and never saved, or lives elsewhere entirely; (2) what the paper calls HuBERT "fine-tuning" is actually **HuBERT pretraining from scratch** via fairseq's iterative relabeling recipe, and none of the paper's quoted fine-tuning hyperparameters (mask span 10, p=8%, LR 3.5e-3, 150k steps) match any config found; (3) **no trace of the unit language model (transformer_lm_big) training exists anywhere** — no script, log, or checkpoint — even though the vocoder and k-means stages around it are well documented. The raw colony recordings themselves live entirely **outside** TALIA_PATH, on shared lab storage (`/ems/elsc-labs/omer-d/Omer-Lab-Shared/...`, roughly 5–6 TB, readable by our account with no extra permission needed), which is not counted in the "8 TB" figure for Talia's folder. A separate, differently-instrumented (250 kHz ultrasonic) recording folder sits inside the same directory glob the detection pipeline iterates over with a hardcoded 48 kHz assumption — a real risk that some outputs from that folder are nonsense. Overall: usable for a from-scratch reimplementation of preprocessing, k-means units, and the vocoder; the uLM and the detection-evaluation hold-out set must be rebuilt or recovered from Talia directly.

---

## 2. Inventory

From `index/by_folder.txt` and `index/by_extension.txt` (describe the ~8 TB TALIA_PATH tree without scanning it).

**Biggest folders (3 levels deep):**

| Folder | Size | Files | What it is |
|---|---|---|---|
| `MonkeyPhones/code/fairseq` | 9.12 TB | 21,151 | fairseq repo clone — hosts checkpoints/training artifacts dumped inside the code dir, not actual code size. (Flag: this single folder exceeds the "8 TB total" figure given for the whole tree — likely stale size estimate or counting quirk; unresolved.) |
| `MonkeyPhones/segmentations_results/second_trial` | 835.8 GB | 1,349,990 | second-pass detection output (30s + 10s segments, pre- and post-downsample) |
| `MonkeyPhones/segmentations_results/firstTrial` | 420.9 GB | 223,547 | **final** first-pass detection output (10s segments) — produced the paper's 216,660-segment dataset |
| `MonkeyPhones/code/speech-resynthesis` | 246.1 GB | 361,773 | unit HiFi-GAN vocoder repo (facebookresearch), with training artifacts dumped inside |
| `MonkeyPhones/code/dataset_classifier` | 63.1 GB | 6,098 | classification probes |
| `dataset/InfantMarmosetsVox/data` | 36.8 GB | 352 | third-party public dataset (not colony data) |
| `dataset` (top level) | 16.4 GB | 8 | mostly InfantMarmosetsVox subfolders |
| `dataset/InfantMarmosetsVox/partitioned_data` | 18.7 GB | 21,325 | same, partitioned |
| `phee_exp/banchmark/caller_full_manipulation` | 10.4 GB | 3,166 | PHEE CallerChange distractor set |
| `phee_exp/banchmark/call_to_full_manipulation` | 10.2 GB | 3,098 | PHEE ReceiverChange distractor set |
| `phee_exp/2022_08_17`, `2022_12_14`, `2023_01_22` | 2.5 / 1.9 / 1.85 GB | 6 / 111 / 60 | **PHEEDB raw sessions** (one `ExpAudio.wav` + labels each) |
| `checkpoints` | 1.14 GB | 1 | `hubert_base_ls960.pt` — stock HuBERT-base checkpoint |
| `MonkeyPhones/code/general_code` | 6.95 GB | 660 | main pipeline glue scripts, manifests, SLURM scripts, logs |
| `MonkeyPhones/code/didisDATA` | 2.21 GB | 35,775 | detection pipeline + benchmark-creation code |
| `MonkeyPhones/code/textlesslib` | 1.62 GB | 1,518 | textless-NLP unit extraction / k-means tooling |

**By extension (top):** `.npy` 9.1 TB/70,167 files (feature caches, mostly under fairseq/textlesslib), `.wav` 1.45 TB/1.86M files (raw + processed audio), `.pth` 64 GB/2,747 (checkpoints), `.pt` 41.4 GB/38 (fairseq checkpoints — all confirmed HuBERT, none transformer_lm), `.km` 11.3 GB/2,533 (k-means models), `.units` 9.5 GB/338 (unit sequence files), `.pkl` 0.14 GB/2,928 (detection indices, k-means models, misc). Code extensions (`.py` 1.83 GB/128,228 files, `.ipynb` 0.12 GB/351, `.sh` 0.01 GB/1,018) are the small slice mirrored into `code_mirror`.

**What's data vs. code vs. scratch:** `fairseq`, `speech-resynthesis`, `textlesslib` are vendored library clones (code, but with huge data/checkpoints dumped inside); `segmentations_results/*`, `dataset/*`, `phee_exp/banchmark/*`, `phee_exp/2022_*`/`2023_*` are pure data; `general_code`, `didisDATA`, `dataset_classifier`, `fromPklToHubert` are Talia's own pipeline code; `.cache`, `envs`, `.venv*`, `cache_backup`, `wandb` (run artifacts), `.pycharm_helpers`, `tmp` are environment/IDE scratch, not pipeline-relevant. `MonkeyPhones/MickeysDataForDebug` and `segmentations_results/mickey*` are confirmed **unrelated debug data** (different experiment, different dates/names) — exclude from any data inventory.

---

## 3. Findings per stage

### 3.1 Preprocessing and call detection (most important stage)

Entry point: `MonkeyPhones/code/didisDATA/refactored_code/main.py`, launched by `bash_scripts/submit_monkey.sh:16` with `--type_of_animal Monkey`. CONFIRMED live (real SLURM launch script, input = raw `AnimalRoomRecordings`, output = `segmentations_results/firstTrial/<folder>`). Calls `wav_processor.py` → `segment_processor.py`.

**Full algorithm, with every constant (CONFIRMED unless noted):**

1. Load audio: `librosa.load(file_path)` with no `sr=` argument → **resamples to librosa's default 22050 Hz and mixes down to mono** (`wav_processor.py:96,118,142`). The raw files are actually 48 kHz, 4-channel ambisonic (Zoom H3-VR); librosa's default mono mixdown silently averages all 4 capsules — no explicit channel-handling code exists anywhere in the pipeline. **Not something the paper or any script addresses.**
2. Split into chunks of `length_of_sub_audios = 1,000,000` samples (`main.py:196`, `wav_processor.py:162-172`) — memory chunking only, not a detection parameter.
3. Per chunk, `SegmentProcessor` constants for `type_of_animal == 'Monkey'` (`segment_processor.py:44-65`):
   a. High-pass Butterworth filter, **order 5, cutoff 5000 Hz**, applied zero-phase via `scipy.signal.butter` + `filtfilt` at the 22050 Hz rate (`segment_processor.py:56,104-106,280-291`). Matches paper's 5 kHz high-pass exactly.
   b. Spectrogram: `librosa.stft(filtered_audio)` with **no explicit `n_fft`/`hop_length`** (`segment_processor.py:109`) → uses librosa defaults (n_fft=2048, hop=512), which happen to match the paper, but this was never set explicitly — INFERRED, not a deliberate choice recorded in code.
   c. Per-frame feature extraction (`find_max_and_min_freqs_by_amp`, `segment_processor.py:207-236`): keep FFT bins with amplitude > `AMP_THRESHOLD = 0.02`; `max_max_freq`/`min_max_freq` = highest/lowest surviving bin frequency per frame.
   d. `freq_range[frame] = max_max_freq − min_max_freq` (0 if no surviving bin) (`segment_processor.py:253-254`).
   e. `bandwidth[frame]` = count of occupied 1000 Hz-wide bins between `MIN_BANDWIDTH=3000` Hz and `MAX_BANDWIDTH=11000` Hz (`segment_processor.py:239-255`).
   f. Candidate-call frame classification (`cluster_frame_features`, `segment_processor.py:127-156`), a frame qualifies if `bandwidth>0` (or a neighbor does), then splits into three classes:
      - **bandwidth_smaller_than_range**: `1000*len(bandwidth[i]) + 3000 < freq_range[i]` (broadband signal).
      - **bandwidth_equals_range_low_freq**: `0 < freq_range[i] <= 3000` (narrow low-freq content).
      - **bandwidth_equals_range_high_freq**: `freq_range[i] > 3000` and `len(bandwidth[i]) < 6` (narrowband high-freq tone), grouped into runs with gaps ≤20 frames (`DISTANCE_BETWEEN_NEI_FOR_NOISE_SUSPECTED`), keeping only runs with **20 < length < 80 frames** (`MIN/MAX_CALL_DURATION_FOR_NOISE_CANCELANTION`) — at 22050 Hz / 512-sample hop this is **≈0.46–1.86 s**, the code's equivalent of the paper's "<0.5s or >2s = noise" rule. **Discrepancy**: 0.46/1.86 s, not exactly 0.5/2.0 s.
   g. Union + sort the three classes, group into runs allowing gaps ≤4 frames (`MAX_DISTANCE_BETWEEN_NEI_IN_CALLS`, merging nearby detections), drop runs of length ≤2 (`identify_suspected_frame_calls`, `segment_processor.py:159-167,257-278`). **Call boundary = first/last frame of each surviving run.**
   h. `filter_calls` (`segment_processor.py:169-185`) rejects a run if:
      - duration ≤ 10 frames (**≈0.23 s**, closest match to paper's 0.25 s floor — small discrepancy) OR duration ≥ 4 s (exact match to paper's 4 s ceiling); OR
      - bandwidth "crossing count" ≥ 3 (threshold line = 1000 Hz) — rejects flickery/noisy bandwidth; OR
      - an abrupt jump in `freq_range` > 2000 Hz within 20 frames — rejects clicks/transients.
4. `wav_processor.save_calls` (`wav_processor.py:176-186`): convert surviving frame runs to sample indices at 22050 Hz, pad **±500 samples**, offset by chunk index, **rescale to the original 48,000 Hz rate** for storage in `call_indicies.pkl`.
5. No handling anywhere for human speech, overlapping calls, clipping, or day/night differences — confirmed absent by exhaustive grep (`human|speech|overlap|clip|day.*night`) across `didisDATA`/`general_code`/`fromPklToHubert`.
6. 10-second segments and resampling are built in a **separate script**, `from_pkl_to_wavs_for_hubert.py` (also mirrored in `fromPklToHubert/code/fromPklToaudio.py`): greedily packs consecutive calls into a segment while `(call_end − segment_start) ≤ max_size_of_segment_seconds × sr` (`from_pkl_to_wavs_for_hubert.py:21-56`), writes `{start}_{end}.wav` named by 48 kHz sample indices (`from_pkl_to_wavs_for_hubert.py:136-141`). Segments are written **at 48 kHz** — the 48→16 kHz downsample happens later, at HuBERT feature extraction (`fromPklToHubert/code/AudioToUnits.py:27,41,134`, `ORIGINAL_SAMPLE_RATE=48000 NEW_SAMPLE_RATE=16000` set in `fromPklToHubert/submit.sh:18-19`).
7. With `--with_silence False` (the value actually launched, `submit_segments_builder.sh:16`), `split_wav_with_no_silence` just slices `waveform[first_call_start:last_call_end]` — the segment includes raw in-between audio, not silence-padded concatenated calls (an unused `with_silence=True` path would concatenate calls with synthetic zero-silence gaps; never invoked by any launch script found).

**Which version is final:** Two segment-builder configurations exist — `submit_segments_builder.sh:16` (`--max_size_of_segment 30`, writes to `second_trial`) vs. a commented-out 10 s example at the bottom of `from_pkl_to_wavs_for_hubert.py:230-237` (matches the function's own 10 s default, writes to `firstTrial`). The on-disk manifests (`segmentations_results/firstTrial/manifests_and_logs/{train,validation,test}_files.txt`) list `10_seconds_segments/<start>_<end>.wav` paths and total exactly **173,328 / 21,666 / 21,666 = 216,660** — an exact match to the paper's COLONYDB figure and an exact 80/10/10 split. **`firstTrial` + 10 s segments is CONFIRMED final**; `second_trial` + 30 s segments is a parallel/later trial not used for the reported numbers (segment length doesn't match the paper, no manifest references it). Caveat: a log (`pitch_filter.log`, 2024-07-29) shows paths under `second_trial/.../10_seconds_segments/...` too — folder names alone are not reliable; the manifest file counts are the authoritative signal.

### 3.2 Detection evaluation (labelled hold-out set behind precision 0.975 / recall 0.78)

**Not found anywhere in the folder.** Exhaustively grepped for `precision|recall|0.975|0.78|97.5|78%|holdout|ground_truth|annotat|false positive/negative|TP/FP/FN|test_set|gold|manual_check` across every `.py`/`.ipynb`, plus run logs under `segmentations_results/debugging` and `mickey_first_trial`, and job logs under `didisDATA/refactored_code/logs`.

Ruled out candidates, with evidence:
- `didisDATA/refactored_code/banchmark_creation/{BanchMarks.py,createBanchMark.py}` and `stats_banchmarks.ipynb` — despite the name, this is the Phee caller/receiver **distractor** benchmark (Section 3.7), not a detection-accuracy hold-out set.
- `dataset_classifier/play_with_embedding.ipynb:10,38-39` computes `precision_score`/`recall_score` from sklearn, but on a caller/call-type **classifier**, unrelated to call detection.
- `MickeysDataForDebug/{BehavioralExp,behav2,behav3}` (referenced by `didisDATA/refactored_code/bash_scripts/submit.sh:15`, `main.py:92`) are debug input audio with no label file — not ground truth.
- `segmentations_results/{debugging,mickey_first_trial}/*/log.txt` are run logs of the detector itself (call counts, timing) with no comparison to any label set.
- `phee_exp/create_call_df.py`, `call_df.py` belong to the separate PHEE experiment, unrelated to the COLONYDB hold-out.

**Conclusion:** the labelled hold-out set and the script that computed the paper's 0.975/0.78 numbers live outside this folder entirely, were computed interactively and never saved, or were deleted. This is a hard open question for Talia (see Section 8).

### 3.3 Dataset building (80/10/10 split, naming, manifests)

The script that actually produced `firstTrial/manifests_and_logs/{train,validation,test}_files.txt` was **not found** — two candidate split utilities exist but neither matches the on-disk format:
- `didisDATA/split_data_to_train_val_test.py` — standalone `split_files(files, 0.8, 0.1, 0.1)` with `np.random.shuffle` (no fixed seed by default); no importers found anywhere in code_mirror — likely dead code.
- `fromPklToHubert/code/fromPklToaudio.py:152-170` (`save_partitioned_data`) splits already-HuBERT-encoded segments via `sklearn.train_test_split` (80/10/10, no fixed seed), writing `train_encoded.txt`/`val_encoded.txt`/`test_encoded.txt` as `"{file_path}: {unit_string}"` — but this script uses `hubert-base-ls960`, `vocab_size=100`, `deduplicate=True`, none of which match the final pipeline (layer 1, 50 clusters, dedup off) — an early experimental script, not the final one.

The manifest's counts match the paper exactly (CONFIRMED artifact), but the generating script's provenance is unresolved (open question).

**File naming convention** (from the manifests): `segmentations_results/firstTrial/Rec_YYYY_MM_DD[-DD]/<YYMMDD>_<NNN>/10_seconds_segments/<start_sample>_<end_sample>.wav`, where `Rec_YYYY_MM_DD[-DD]` is the recording-date folder, `<YYMMDD>_<NNN>` is the raw wav's basename (date + file index), and `<start>_<end>` are 48 kHz sample offsets into that raw wav — so a segment's original timestamp is recoverable as `start_sample/48000` seconds into the named raw file.

### 3.4 Vocalization-to-unit: HuBERT + k-means

**HuBERT is pretrained from scratch (iterative relabeling), not fine-tuned from a checkpoint — this contradicts how the paper's "fine-tuning" wording reads.**

- Active launch: `general_code/old_bash/fine_tune_hubertsynt.sh:44-46` runs `fairseq_cli/hydra_train.py --config-dir .../examples/hubert/config/pretrain --config-name hubert_base_librispeech_layer_6`. The genuine CTC fine-tune config (`config/finetune/base_10h.yaml`, which would initialize from the `checkpoints/hubert_base_ls960.pt` 1.14 GB file) exists only **commented out** (lines 32-41 of the same script).
- `fairseq/examples/hubert/config/pretrain/hubert_base_librispeech_layer_6.yaml`: `task.data` = marmoset manifests, `mask_prob: 0.80` (standard HuBERT pretrain default, **not** the paper's "mask span 10, p=8%"), `lr: [0.00035]` = 3.5e-4 (**not** paper's 3.5e-3), `max_update: 1,600,000` (**not** 150k), `warmup_updates: 40,000`. Sibling configs exist for layers 3, 9, 12 — standard HuBERT iterative-relabeling generations, not independent fine-tuning variants.
- The genuine finetune config (`finetune/base_10h.yaml`) is unmodified fairseq ASR-CTC defaults: `mask_length: 10`, `mask_prob: 0.75`, `lr: [2e-5]`, `max_update: 25,000` — still does not match the paper's quoted numbers. **None of the three configs found match the paper's fine-tuning hyperparameters** — likely paper garbling (same pattern as its known "5e-45" LR typo for the uLM). Open question for Talia.
- Which HuBERT layer/iteration is "final" (layer 6, the one actually active in the launch script, vs. layer 12, which other scripts target repeatedly) could not be confirmed from code alone — open question.

**K-means (final pipeline, CONFIRMED):**
- `n_cluster=50` confirmed in two launch scripts (`general_code/old_bash/done/submit_learn_kmeans_synt.sh:17`, feature dir `layer_12`; `.../done/submit_kmeans_and_units_of_data_synth.sh:13`, MFCC first-pass) — matches the paper.
- `fairseq/examples/hubert/simple_kmeans/learn_kmeans.py` is a **modified** copy of fairseq's stock script: `load_feature()` hardcodes loading shards 0-4 of 10 from train+val only (lines 73-80) — exactly reproduces "50% of train+valid," but via a hardcoded loop, not the `--percent` CLI flag (which was disabled). `init='k-means++'`, `n_init=20` are CLI defaults, not overridden — matches paper. **`batch_size` default is 5000, not overridden — paper says 10,000, a real discrepancy.**
- "First k-means" = MFCC features (`submit_kmeans_and_units_of_data_synth.sh`, standard HuBERT iter-0 bootstrap). Later iterations use the HuBERT model's own layer features (layer_3/6/9/12 variants, ~35 near-identical launch scripts under `general_code/old_bash/done/submit_features_for_k_means_synt_layer_*`) — not individually diffed, treated as hyperparameter/layer sweeps of the same mechanism.

**Deduplication — OFF in the final unit files, via an unusual two-stage mechanism:**
- `textlesslib`'s `SpeechEncoder` deduplicates by default everywhere (`deduplicate=True` throughout example scripts, `distributed_transcribe/local.sh:19`, `single.sh:34`), including the production `transcribe.py`.
- `textlesslib/tools/distributed_transcribe/from_deduplicate_to_duplicate.py` **re-expands** each deduplicated `.units` file back to full frame-rate using its paired `.durations` file (repeats each unit `duration` times), writing to a sibling `duplicate/` subfolder. Hardcoded on `full_data/k50`. File sizes confirm a complete, successful run: `train.units` 78.85 MB (deduplicated) → `duplicate/train.units` 181.34 MB (~2.3× expansion); same pattern for val/test, all three splits complete.
- A later run (`single.sh`, mtime 2025-02-05) wrote to `full_data/k50/final_manifests_bigger_dataset/` but only has `_partial_0` shards for train/val and no test split — an incomplete/abandoned attempt.
- **Conclusion:** `hubert_output/16khz_sr/full_data/k50/duplicate/{train,val,test}.units` is the complete, non-deduplicated unit set matching the paper's claim — CONFIRMED by mechanism + file-size arithmetic. The deduplicated parent `k50/` set and the incomplete `final_manifests_bigger_dataset/` are dead/intermediate copies.
- Other dead ends: `fromPklToHubert/code/AudioToUnits.py` uses generic `hubert-base-ls960` + `vocab_size=100` + `deduplicate=True`, and its core writing loop is commented out with a hardcoded `self.files[:102]` slice — non-functional debug script.

### 3.5 Unit language model (fairseq transformer_lm)

**Essentially unrecoverable — the single biggest gap in this audit.**

- `MonkeyPhones/code/fairseq` (9.1 TB listed) and `MonkeyPhones/code/speech-resynthesis` (246 GB listed) are present in the index but **do not exist on disk in code_mirror** (verified via `stat`/`find` — directories legitimately absent, not a permissions issue). This blocks reading their configs/training scripts directly from code_mirror; the original paths were read directly instead where possible.
- Searched `text_files.tsv`-listed logs for any run invoking `language_modeling` task or `transformer_lm*` architecture: **zero hits**. The one hydra log found, `fairseq/None/hydra_train.log` (9,864 lines, Oct 26–Nov 25 2024), is entirely `HubertPretrainingTask`/`HubertCriterion` (useful cross-reference for Section 3.4: shows `label_dir: .../layer_9_50k/merged_shards`, lr=5e-4 peak, warmup=32,000, polynomial_decay, max_update=400,000 — another HuBERT run, not the uLM).
- All 38 `.pt` checkpoints in the index are HuBERT pretraining checkpoints or stock pretrained models (`hubert_base_ls960*`, FAD models) — **no transformer_lm/uLM checkpoint exists anywhere** in the indexed files.
- No launch script anywhere references `fairseq-train`/`fairseq-generate` with a language-modeling task.
- **Conclusion:** no code, log, or checkpoint for the uLM survives in this folder. Either it was trained outside Talia's account (a collaborator's, or a different project space) or its artifacts were deleted. High-priority open question for Talia.
- Generation settings are only weakly inferable: vocoder test scripts (`test_synt_3_sec_temp_1.sh:33`) reference filenames like `real_3_sec_temp_1.5_9_1.txt`, implying **temperature = 1.5** (matches paper) — INFERRED from filenames only, not a logged argument. **Beam size: no evidence found anywhere.**

### 3.6 Vocoder (unit HiFi-GAN)

Well documented, CONFIRMED by multiple independent sources:
- Repo: `speech-resynthesis` (facebookresearch), adapted LJSpeech HuBERT-LUT recipe. Launch: `general_code/old_bash/done/train_synt.sh:28-31` (`torch.distributed.run`, 3 GPUs) → `speech-resynthesis/train.py --config speech-resynthesis/configs/LJSpeech/hubert100_lut.json --checkpoint_path .../checkpoints/data_pad_no_spkr/hub_3_emb_1`.
- 5 wandb run configs (`general_code/code/wandb/run-20241201_*/files/config.yaml`, `general_code/logs/wandb/run-20241125_120324-l1so54aj`, `run-20250115_133325-torxru6n`, `run-20250115_140712-gb1syoh5`) all agree: `training_steps: 400000` (exact match to paper), `checkpoint_interval: 10000`, `training_epochs: 2000`.
- Checkpoint path evolved over time: generic `checkpoints/` (Nov 25) → `cp_hifigan` (Dec 1) → `checkpoints/data_pad_no_spkr/hub_9_emb_1` (Jan 15 2025, newest). **Final = `hub_9_emb_1`**, directly referenced by the real inference scripts `test_synt.sh:29-33` and `test_synt_3_sec_temp_1.sh:30-34`, which load checkpoint `g_00400000` (full 400k steps, matches paper).
- Directory name `data_pad_no_spkr` = no speaker embedding, matching the paper's "no speaker or F0 input" claim — INFERRED from naming; the JSON config content itself could not be read (excluded extension, and the repo directory is missing from the mirror).
- **Flag:** `hub_9_emb_1` implies HuBERT layer 9 features feed the vocoder, which conflicts with the paper's final HuBERT layer=1 choice (Section 3.4) and with sibling dir `hub_3_emb_1` — i.e. the vocoder may have been trained on units from a different HuBERT layer/k-means run than the one deemed "final" for the uLM. Needs reconciliation with Talia.

**Checkpoint inventory (this stage):**

| Path (original) | Size | What | Status |
|---|---|---|---|
| `speech-resynthesis/checkpoints/data_pad_no_spkr/hub_9_emb_1/g_00400000` | — (not indexed; dir missing from mirror, confirmed to exist via launch script reference) | Vocoder, full 400k steps | **Final** |
| `speech-resynthesis/checkpoints/data_pad_no_spkr/hub_9_emb_1/g_00220000` | — | Vocoder, intermediate checkpoint | superseded |
| `cp_hifigan/` (16 files, ~0 GB) | tiny | Early/intermediate vocoder run artifacts | dead/superseded |

### 3.7 Evaluation: Shuffle/Concat/Reversal/Phee distractors, scoring, PHEEDB

**COLONYDB distractors** — `didisDATA/refactored_code/banchmark_creation/BanchMarks.py`, launched by `createBanchMark.py` via `submit_banchmarks.sh` (CONFIRMED real SLURM job):
- **Shuffle** (`shuffle_sample`, L98-134): loads a 10s sample at 48 kHz, extracts call boundaries from `call_indicies.pkl`, randomly permutes call order, re-inserts silences drawn from the shuffled original inter-call-gap distribution, writes to `shuffle_samples_v1`. A near-duplicate function `shuffle_samples_full_data` is defined **twice** in the same file (L141 and L196, same name — the second silently shadows the first in Python); writes to `full_shuffle_samples`/`shuffle_samples_full_data`. Which definition ran for a given job depends on which was live at submit time — **unresolved, flag as open question.**
- **Reversal** (`reverse_sample`, L176-192): full-waveform time reversal, saved to `reverse`.
- **Concat** (`cut_and_concat_samples`, L249-280): pairs of 10s samples with exactly 6 calls each, split at the midpoint call boundary, second halves swapped, saved to `cut_concat_samples/6-6`. 10,000 random pairs by default (`createBanchMark.py:55-61`).
- **Scoring method (summed vs. mean log-probability): NOT FOUND.** No custom scoring/eval script surfaced beyond vanilla fairseq's `SequenceScorer`/`fairseq-eval-lm` — no wrapper pins down the reduction used. Open question.

**PHEE distractors (CallerChange/ReceiverChange)** — `phee_exp/banchmark/*.py` (CONFIRMED, outputs match by_folder.txt sizes):
- 3 raw sessions, each one `ExpAudio.wav`, **96 kHz, 24-bit stereo** (not the paper's stated 16 kHz), durations 4324s/3176s/3036s (~72/53/51 min, not exactly "3×1h" but close). 4 marmosets confirmed: Adonis, Dionysus, Bhumi, Dia. Channels merged as `left_norm + right_norm/2` — an **asymmetric** mono mixdown (right channel halved, left not), possibly intentional, possibly a bug — flag for Talia.
- Labels: `phee_exp/df_merged.csv`, `all_calls_splited.csv` — columns `phee_calls_start_times, phee_calls_end_times, phee_calls_channels, start_ind, end_ind, speaker, recording_path, call_to`.
- **CallerChange** = `full_caller.py:42-61` → `caller_full_manipulation` (10.37 GB/3,166 files): keeps the real response, prepends a call from a different caller identity (any `speaker != caller`).
- **ReceiverChange** = `full_call_to.py:41-57` → `call_to_full_manipulation` (10.20 GB/3,098 files): keeps the real response, prepends a call whose original addressee differs from the true addressee.
- Four "half manipulation" variants add extra matching constraints beyond the paper's binary description — likely exploratory, not the ones reported (INFERRED: the "full_*" versions match the paper's binary description with no extra constraint).
- All distractor/true audio saved at native 96 kHz, not resampled to 16 kHz in these scripts.

**Probes** — `dataset_classifier/embedding_classifier.py` (CONFIRMED via `submit_classifier.sh`): `AudioLinearSmartClassifier` (the one actually used, `--is_smart` defaults True) does mean+variance pooling concatenated, then 3× `Linear → LayerNorm → ReLU` (hidden dims default `[128,64,32]`), final linear layer — matches the paper's architecture exactly. **Discrepancy: `submit_classifier.sh:28` sets `NUM_EPOCHS=30`, not the paper's 20.** Run per HuBERT layer `1,3,6,9,12`; `INPUT_DIM=768`, LR `0.001`, batch size `200`.

**FAD** — no driver script found anywhere in code_mirror; only two installed pip environments exist (`fad_env`, `fad_env_love`). Most likely run interactively and never saved. Open question: what embeddings/reference set FAD was computed against.

---

## 4. Data locations

### 4.1 Raw colony recordings — `/ems/elsc-labs/omer-d/Omer-Lab-Shared/data/experiments/AnimalRoomRecordings/` (outside TALIA_PATH)

Found at `didisDATA/refactored_code/bash_scripts/submit_segments_builder.sh:15` (`FOLDERS=(.../AnimalRoomRecordings/*)`, `--array=0-55`) and `BanchMarks.py:11` (`BASE_DATA_DIR`).

- **Role:** raw colony recordings, input to the detection pipeline.
- **Layout:** 55 day-range folders named `Rec_YYYY_MM_DD[-DD]/`, e.g. `Rec_2021_07_26-27/210726_001.WAV … 210727_025.WAV`. One oddly-named folder, `Rec_2021_06_28_ultrasonic_recordings_colony/`, uses different naming/recorder: `M12_F14_YYMMDDHHMMSS_NNN.WAV` + `FILES.LOG`.
- **Audio format** (read from WAV `fmt `/`bext` chunks, 3 files sampled): normal files are **48 kHz, 4-channel, 24-bit PCM**, ~62 min each (3,727 s), from a **Zoom H3-VR** ambisonic recorder (one device with 4 capsules in one room, confirmed via `bext`/`iXML` tags `zTRK1-4 = FLU/FRD/BLD/BRU`) — not 4 separate mics/rooms. `bext` "OriginationDate/Time" gives each file's exact wall-clock start (e.g. `210726_001.WAV` → 2021-07-26 16:56:37), so per-sample absolute timestamps are recoverable. The `ultrasonic_recordings_colony` folder is a different instrument: **250 kHz, mono, 16-bit**, no `bext` chunk, tracking two named individuals (M12/F14).
- **Risk:** this ultrasonic folder sits inside the same `AnimalRoomRecordings/*` glob (array 0-55) fed to the detection pipeline, which hardcodes `--sr 48000`. If it was actually processed, every sample index/duration computed on it is wrong by ~5.2×.
- **Count/size:** 2,750 WAV files across 55 day-folders; sampled folder sizes range 5.8–126 GB; full-tree size not computed (avoided unbounded `du`), extrapolated to roughly **5–6 TB total, ~2,600–2,700 raw channel-hours**. Continuous within a recorder-rotation chunk (~62 min each, final chunk of a run shorter), not pre-cut.
- **Date range:** 2021-05-03 → 2021-10-12 (~5.3 months). Visible gaps: 2021-05-24→05-29, 2021-06-07→06-14 (~7 days), 2021-06-18→06-21, 2021-07-27→07-30, 2021-08-03→08-05. Recording ran in multi-day bursts, not continuously across the full span.
- **Permissions:** owner `omerlab.storadmin`, group `omer-lab`, mode 0750 — readable by our account (`zvi.marmor`, group `omer-lab`) with no extra request needed.
- **Mapping to processed segments:** explicit and recoverable — see 4.2.

### 4.2 Processed segments — `TALIA_PATH/MonkeyPhones/segmentations_results/{firstTrial,second_trial}/`

- **Role:** detected-call / segmented outputs of the preprocessing pipeline.
- **Layout:** `firstTrial/<Rec_folder>/<raw_stem>/{call_indicies.pkl, log.txt, final_segments/, 10_seconds_segments/, online_plots/}`; `second_trial/<Rec_folder>/<raw_stem>/{10_seconds_segments/, 30_seconds_segments/}` (48 kHz/16-bit mono, pre-resample); `second_trial/downsample/<Rec_folder>/<raw_stem>/10_seconds_segments/*.wav` (**16 kHz, 32-bit float, mono** — confirmed by hand-parsing the `fmt ` chunk since stdlib `wave` can't read float WAVs). This `downsample` tree is what the fairseq-style `train.tsv`/`valid.tsv` manifests point at.
- **Count:** 197,141 segment files counted in `second_trial/downsample` — **short of the paper's 216,660** (which is matched instead by `firstTrial`'s manifests, Section 3.1/3.3). Worth flagging as a possible incomplete run of the downsample/second-pass path, or evidence that `firstTrial` (not `second_trial/downsample`) is truly what fed the final HuBERT/unit pipeline — unresolved, see open questions.
- **Segment→raw mapping:** explicit — the folder name under `firstTrial`/`second_trial` IS the raw file's stem, and the segment filename `<start>_<end>.wav` encodes the 48 kHz sample offset into that raw file (verified: `(end-start)/48000` matches segment duration at 16 kHz to within rounding). Implemented in code at `BanchMarks.py:20-32` (`from_sample_name_to_pkl`, `from_sample_name_to_original_wav`). Full timestamp recovery = raw file start time (from `bext`) + `start_sample/48000`.
- **Permissions:** inside TALIA_PATH, readable by our account.

### 4.3 Manifests — `TALIA_PATH/MonkeyPhones/code/general_code/{train,valid}.tsv`

Format: a `root_dir` header line followed by `relpath<TAB>num_frames` rows, for the unit/HuBERT pipeline. Readable.

### 4.4 Detection labels (hold-out set)

**UNKNOWN — not found anywhere under TALIA_PATH.** See Sections 3.2 and 8.

### 4.5 PHEEDB — `TALIA_PATH/phee_exp/{2022_08_17,2022_12_14,2023_01_22}/`

- **Role:** PHEEDB raw sessions.
- **Layout:** each folder = one `ExpAudio.wav` + `config.mat` + `calls.csv`/`calls.pkl` (per-call labels).
- **Audio format:** 96 kHz, 24-bit, stereo (not the paper's stated 16 kHz) — see Section 3.7.
- **Size:** 2.49 / 1.94 / 1.85 GB (really 1.83/1.75 GB after discounting ancillary files), each session ≈ 1 hour, roughly matching "3×1h."
- **Owner:** `david.omer` (PI), group `omer-lab`, rwx — accessible, no extra permission needed.

### 4.6 InfantMarmosetsVox — `TALIA_PATH/dataset/InfantMarmosetsVox/`

Third-party public dataset (Sarkar et al., Idiap), used for probe/classifier work, not colony data. `infantmarmosetsvox.py:79` reads `labels.csv`. Readable.

### 4.7 Other

- `/ems/elsc-labs/london-m/Lab-Shared/Talia/*` — referenced at `bash_scripts/submit_mice.sh:15` ("mice" data, apparently a different lab/project). **Permission denied to our account** — open question whether it's relevant to GmSLM and who can grant access.
- `/cs/labs/adiyoss/talia_123/...` — stale comment reference (`general_code/remove_names_of_files.py:1-2`) to Talia's old HUJI CS-lab account. Does not exist from this cluster.
- `TALIA_PATH/MonkeyPhones/MickeysDataForDebug/` — unrelated debug data (different experiment/dates/names), referenced only as a commented-out debug path in `main.py:92-93`. Exclude from any real data inventory.

### 4.8 YAML paths block

```yaml
raw_recordings: /ems/elsc-labs/omer-d/Omer-Lab-Shared/data/experiments/AnimalRoomRecordings
processed_segments:
  first_pass: /ems/elsc-labs/omer-d/talia.sternberg/MonkeyPhones/segmentations_results/firstTrial
  second_pass: /ems/elsc-labs/omer-d/talia.sternberg/MonkeyPhones/segmentations_results/second_trial
  final_16khz: /ems/elsc-labs/omer-d/talia.sternberg/MonkeyPhones/segmentations_results/second_trial/downsample
manifests: /ems/elsc-labs/omer-d/talia.sternberg/MonkeyPhones/code/general_code/{train,valid}.tsv
detection_labels: UNKNOWN - not found under TALIA_PATH, see Section 8
pheedb: /ems/elsc-labs/omer-d/talia.sternberg/phee_exp/{2022_08_17,2022_12_14,2023_01_22}
infant_marmosets_vox: /ems/elsc-labs/omer-d/talia.sternberg/dataset/InfantMarmosetsVox
other:
  ultrasonic_special_recording: /ems/elsc-labs/omer-d/Omer-Lab-Shared/data/experiments/AnimalRoomRecordings/Rec_2021_06_28_ultrasonic_recordings_colony
  unrelated_debug_data: /ems/elsc-labs/omer-d/talia.sternberg/MonkeyPhones/MickeysDataForDebug
  inaccessible_external: /ems/elsc-labs/london-m/Lab-Shared/Talia
```

---

## 5. Sample plan

**Note on budget:** the task spec's `<SAMPLE_BUDGET_GB>` placeholder was never filled in for this session. The plan below defaults to **~9 GB**, with an explicit trim-to-~5 GB fallback noted; confirm the real budget before copying.

No files are listed under Priority 1 (labelled detection hold-out set) — it does not exist in this folder (Sections 3.2, 4.4). Flagged as the top open question rather than silently skipped.

| # | Path | Size | Date/time | Why chosen |
|---|---|---|---|---|
| P2.1 | `AnimalRoomRecordings/Rec_2021_05_03/210503_001.WAV` | 2.05 GB | 2021-05-03 18:58 (day, earliest date) | temporal spread; has full matching processed output below |
| P2.2 | `AnimalRoomRecordings/Rec_2021_07_26-27/210727_009.WAV` | 2.05 GB | 2021-07-27 01:13 (night, mid-range date) | day/night contrast |
| P2.3 | `AnimalRoomRecordings/Rec_2021_10_10-12/211012_047.WAV` | 2.05 GB | 2021-10-12 11:53 (day, latest date) | temporal spread, dataset end |
| P2.4 | `AnimalRoomRecordings/Rec_2021_06_28_ultrasonic_recordings_colony/M12_F14_210628103427_001.WAV` + `FILES.LOG` | 1.05 GB | 2021-06-28 10:34 | different equipment/sample-rate — stress-tests resampling assumptions, surfaces the 48 kHz-hardcode risk |
| P3 | `segmentations_results/firstTrial/Rec_2021_05_03/210503_001/**` (pkl, log, final_segments/, 10_seconds_segments/, online_plots/) | 74 MB | — | Talia's first-pass detections on raw file P2.1, for direct comparison |
| P3 | `segmentations_results/second_trial/Rec_2021_05_03/210503_001/{30,10}_seconds_segments/**` | 13 MB | — | second-pass pre-resample output, same raw file |
| P3 | `segmentations_results/second_trial/downsample/Rec_2021_05_03/210503_001/10_seconds_segments/**` | 1.05 MB | — | final 16 kHz segments as actually fed to HuBERT, same raw file |
| P4 | `phee_exp/2023_01_22/{ExpAudio.wav,config.mat,calls.pkl,final_calls_0.05.csv,merged_df_calls_0.05.csv}` | 1.75 GB | 2023-01-22 | smallest of the 3 PHEEDB sessions, full audio+labels |

**Total ≈ 9.0 GB.** If the real budget is tighter (~5 GB), drop P2.2 and P2.3 first — keep P2.1 (full raw↔processed comparison), P2.4 (equipment anomaly), and P4 (PHEEDB).

**rsync template (not run):**
```bash
rsync -avP --relative --files-from=<(grep -v '^#' data_sample_list.txt) / <user>@<cluster-login-node>:/ ./gmslm_sample/
```
(Paths in the list are absolute; `--relative` with source `/` preserves each file's full original path under `./gmslm_sample/` on the laptop. The `grep -v '^#'` strips the comment/priority header lines.)

---

## 6. Reusable artifacts

| Artifact | Path (original) | Size | Status |
|---|---|---|---|
| Stock HuBERT-base checkpoint | `checkpoints/hubert_base_ls960.pt` | 1.14 GB | base model for pretrain init |
| HuBERT pretrain checkpoints (layers 3/6/9/9_adj_lr/12) | `fairseq/None/best_checkpoints/pre_train_hubert/layer_{3,6,9,9_adj_lr,12}/...` | ~1.1 GB each | multiple iterative-relabeling generations; layer 6 confirmed "active" per launch script, layer 12 targeted by more downstream scripts — **which is "final" unresolved** |
| k-means models (first-pass MFCC, later HuBERT-layer passes) | `.km` files under `general_code/.../k_means_models/` | 11.3 GB/2,533 files total across tree | 50 clusters confirmed; final layer unresolved (see above) |
| Non-deduplicated unit files (final) | `textlesslib/.../hubert_output/16khz_sr/full_data/k50/duplicate/{train,val,test}.units` | train 181.34 MB, val/test ~22.8/22.6 MB | **final**, confirmed complete 3-split set |
| Deduplicated unit files (intermediate) | `.../full_data/k50/{train,val,test}.units` | train 78.85 MB, val/test ~9.9/9.8 MB | intermediate input to the re-expansion script above |
| Incomplete "bigger dataset" units | `.../full_data/k50/final_manifests_bigger_dataset/` | partial | abandoned — only `_partial_0` shards, no test split |
| Vocoder checkpoint, final | `speech-resynthesis/checkpoints/data_pad_no_spkr/hub_9_emb_1/g_00400000` | — (dir missing from code_mirror; path confirmed via launch script) | **final**, 400k steps |
| Vocoder checkpoint, intermediate | `.../hub_9_emb_1/g_00220000` | — | superseded |
| Unit language model checkpoint | — | — | **does not exist anywhere in the folder** |
| Train/valid manifests | `general_code/{train,valid}.tsv` | small | current pipeline input |
| firstTrial detection manifests | `segmentations_results/firstTrial/manifests_and_logs/{train,validation,test}_files.txt` | small | **final** 216,660-segment 80/10/10 split |
| Probe checkpoints | `dataset_classifier/checkpoints/layer_{1,3,6,9,12}/big_net/...epoch_{N}.pth` | part of 63 GB `dataset_classifier` total | one classifier per HuBERT layer swept |

---

## 7. Differences from the paper

| Area | Paper | Found in code | Verdict |
|---|---|---|---|
| High-pass filter | 5 kHz | Butterworth order 5, cutoff 5000 Hz, zero-phase (`filtfilt`) | Match |
| Spectrogram window/hop | 2048 / 512 | librosa STFT defaults (n_fft=2048, hop=512), **never set explicitly** | Match, but accidental not deliberate |
| Noise rule | <0.5 s or >2 s | code thresholds are frame-count based, giving ≈0.46–1.86 s | Close but not exact |
| Call rule | 0.25–4 s | code floor ≈0.23 s, ceiling exactly 4 s | Floor close but not exact; ceiling exact |
| Downsample | 16 kHz | raw mixdown resampled to 22050 Hz for detection; segments stored at 48 kHz; downsampled to 16 kHz only at HuBERT feature-extraction time | Different pipeline mechanics, same final rate |
| Segment length | 10 s | confirmed 10 s (`firstTrial`), with a 30 s variant (`second_trial`) also present | Match (final), extra untested variant exists |
| Split | 80/10/10 | confirmed exactly 173,328/21,666/21,666 = 216,660 | Exact match |
| COLONYDB size | 216,660 segments, ~365h (155h calls) | 216,660 confirmed via manifests; hours not independently verified; `second_trial/downsample` only has 197,141 files (short) | Segment count matches; hours unverified; a parallel tree is short |
| PHEEDB | 3×1h at 16 kHz, 4 marmosets, 56 call-response pairs | 3 sessions ~51–72 min each at **96 kHz, 24-bit stereo**; 4 marmosets confirmed (Adonis, Dionysus, Bhumi, Dia); 56-pairs count not verified | Sample rate/bit depth/channels differ from paper |
| HuBERT base | hubert-base | `hubert_base_ls960.pt` present but referenced only in a commented-out finetune path; active path is pretrain-from-scratch | Mismatch in training regime |
| HuBERT fine-tune: mask span/p/LR/steps | 10 / 8% / 3.5e-3 / 150k | pretrain config: mask_prob 0.80, LR 3.5e-4, 1.6M steps; finetune config (unused): mask_length 10, mask_prob 0.75, LR 2e-5, 25k steps | No config matches the paper's numbers |
| Units: layer / clusters | 1 / 50 | 50 clusters confirmed; which layer is "final" unresolved (layer 6 active in launch script, layer 12 and others also present); vocoder checkpoint name implies layer 9 | Clusters match; layer unresolved/possibly inconsistent across stages |
| k-means | MiniBatchKMeans, 50% train+valid, batch 10k, k-means++, 20 inits | 50% via hardcoded 5/10-shard loop (not `--percent`), k-means++ and n_init=20 match, **batch_size=5000 not 10,000** | One real discrepancy (batch size) |
| Deduplication | none | OFF in final `duplicate/` unit set via re-expansion script; ON by default everywhere else in the tooling | Match in final artifact, fragile mechanism |
| uLM architecture/training | transformer_lm_big, 50k steps, inverse_sqrt 4k warmup | **no script, log, or checkpoint found anywhere** | Cannot verify — total gap |
| uLM peak LR | printed as "5e-45" (typo) | real value not found | Cannot verify |
| uLM generation | beam 5, temperature 1.5 | temperature 1.5 weakly inferred from filenames; beam size never found | Partially unverifiable |
| Vocoder | 400k steps, no speaker/F0 | 400k steps confirmed (5 wandb configs + checkpoint name `g_00400000`); no-speaker/F0 inferred from `data_pad_no_spkr` directory naming only | Steps match exactly; speaker/F0 claim plausible but not read from an actual config file |
| Probes | mean+variance pooling, 3 FC + LayerNorm + ReLU, 20 epochs | architecture matches exactly; **epochs = 30 in the launch script, not 20** | One real discrepancy (epoch count) |
| Shuffle/Concat/Reversal/Phee distractors | described qualitatively | all four implemented and located; Shuffle has an ambiguous duplicate-function-name issue; PHEE "half_*" variants beyond the paper's binary description exist alongside the "full_*" ones that match it | Mostly consistent, with code hygiene caveats |
| Scoring (summed vs mean log-prob) | not specified beyond implication | no scoring script found | Cannot verify |
| FAD | used for evaluation | no driver script survives; only pip envs | Cannot verify |

---

## 8. Open questions only Talia (or a data/storage owner) can answer

1. **Where is the manually-labelled call-detection hold-out set** behind the paper's reported 0.975 precision / 0.78 recall? Not found anywhere under TALIA_PATH; what format is it in, how many hours, who labelled it, and how was a detection matched to a label?
2. **Where was the unit language model (transformer_lm_big) actually trained?** No script, log, or checkpoint for it exists in this folder — was it trained under a different account/project space, and can those artifacts be recovered? What is the real peak learning rate (the paper prints the obviously-wrong "5e-45")? What beam size was used at generation?
3. **Which HuBERT layer/iteration is the "final" one** feeding the paper's reported k-means/units — the launch script's active layer 6, or layer 12 (targeted by more downstream scripts), or layer 9 (implied by the vocoder checkpoint name `hub_9_emb_1`)? These three signals disagree.
4. **Was `Rec_2021_06_28_ultrasonic_recordings_colony/`** (250 kHz, different recorder) actually run through the 48 kHz-hardcoded detection pipeline, given it sits inside the same directory glob as the other 55 folders? If so, its outputs are likely wrong.
5. **How should the raw Zoom H3-VR's 4 ambisonic channels be combined** before detection? The current code silently lets `librosa.load`'s default mono-mixdown average all 4 — was this ever evaluated against a deliberate channel choice?
6. **Why does `second_trial/downsample` have 197,141 segments, short of the paper's 216,660** (which is matched by `firstTrial`'s manifests instead)? Is `second_trial/downsample` an incomplete/abandoned run, or is `firstTrial` (at 48 kHz, pre-downsample) actually what fed the final HuBERT pipeline, with downsampling happening at a different stage than assumed?
7. **Which `shuffle_samples_full_data` function definition** (two identical-named definitions exist in `BanchMarks.py`, lines 141 and 196) actually ran for the distractor sets used in the paper?
8. **Are the "full_*" PHEE manipulation scripts (CallerChange/ReceiverChange) the ones reported in the paper**, with the "half_*" variants being unused exploratory conditions? And is the asymmetric `left_norm + right_norm/2` channel mixdown for PHEEDB audio intentional?
9. **What tool/config was used for FAD**, and does any log of that run survive anywhere (including outside this cluster, e.g. a personal laptop or a collaborator's machine)?
10. **What is `/ems/elsc-labs/london-m/Lab-Shared/Talia/*`** (referenced in `submit_mice.sh`, permission-denied to our account) — is it relevant to GmSLM, and who can grant access if so?
11. **Why does `index/by_folder.txt` show `MonkeyPhones/code/fairseq` alone at 9.1 TB**, exceeding the stated ~8 TB total for the whole folder? Possible double-counting (hardlinks, nested checkpoint copies) — worth a sanity check before relying on any size estimate in this report.
12. Can Talia point to **specific days/times known to be especially busy, quiet, or noisy** (cleaning, feeding) to improve the sample selection in Section 5 beyond the arbitrary day/night spread chosen here?
13. The task's sample budget (`<SAMPLE_BUDGET_GB>`) was never specified for this session — please confirm the real laptop-copy budget so Section 5 can be resized accurately.
