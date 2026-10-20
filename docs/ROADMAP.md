# Marmoset Developmental Agent: Roadmap

*Omer lab (ELSC, Hebrew University) and Hasson lab (Princeton). Living document, last updated 2026-09-27.*

## 1. Scientific motivation

Marmoset vocal behaviour is not purely innate. Infant vocalisations mature faster with contingent parental feedback (Takahashi et al., 2015). Adults take turns in phee exchanges (Takahashi, Narayanan & Ghazanfar, 2013). Adults also use distinct phee calls to address specific individuals (Oren et al., 2024). Marmosets are therefore a tractable non-human primate model for asking how vocal-communication structure is *learned from experience*.

Standard spoken-language-model (SLM) pretraining, GmSLM included, treats the corpus as one stationary distribution: segments are shuffled i.i.d., and every sample is visited under the same learning rules. That removes exactly the variables a developmental account cares about:

- **Order.** An infant hears day 1 before day 300. What can be learned on day *d* depends on representations built from days 1..*d−1*.
- **Growing exposure.** Daily input rises through infancy. The Hasson framework models this with a schedule of 2 → 8 h/day, F(d).
- **Consolidation.** Experience is replayed offline, and older memories fade.
- **Staged plasticity.** Low-level perceptual units have to stabilise before sequence-level (lexical, grammatical) learning can work.

Hasson's "First 1000 Days" framework builds these constraints into a self-supervised speech learner and yields learning *trajectories* (surprisal, lexicon growth) that can be compared with behaviour. GmSLM shows that a textless SLM pipeline adapted to marmosets captures meaningful structure in colony audio, and it provides marmoset-specific front-end, tokenisation and evaluation tools. **The Marmoset Developmental Agent combines the two.** It is a textless marmoset vocal learner trained chronologically under developmental constraints and evaluated with marmoset-specific zero-shot probes over simulated development.

### Core questions
1. Does chronological training with sleep replay reach GmSLM-level structure (Shuffle, Concat, Reversal, Phee) with the same data budget, and along which trajectory?
2. In what order do call types become predictable ("enter the lexicon"), and does that order match known marmoset vocal development (for example, the transition from infant cries and immature phees to adult-like phees)?
3. When during development does the model become sensitive to social structure (CallerChange, ReceiverChange), to inter-call timing, and to caller-specific pitch?
4. Which constraints matter? Compare chronology vs. shuffling, replay kernel and X, the timing of decoder plasticity (T_p), and online (DINO-SR) vs. frozen (HuBERT) units.

## 2. Architectural lineage

| Component | Origin | Module | Notes |
|---|---|---|---|
| 5 kHz high-pass, spectrogram noise gate (2048/512) | GmSLM §3.1, A.1 | `preprocessing/filter_5khz.py` | Re-implemented from the paper's prose (code not public) |
| 0.25–4 s call extraction, 10 s chunks at 16 kHz | GmSLM A.1–A.2 | `preprocessing/segmentation.py` | Greedy packing: "as many calls as possible" |
| Chronological day-by-day dataset, F(d) exposure | Hasson `generate_cycle_sizes.py` | `preprocessing/dataset_builder.py` | Hours/day schedule converted to 10 s chunks/day |
| Marmoset HuBERT, layer 1, k = 50, no dedup | GmSLM §3.2, A.3, §5.3 | `encoders/marmoset_hubert.py` | Loads an HF-converted checkpoint; k-means via sklearn |
| DINO-SR, 512-entry online codebooks, EMA teacher | Liu et al. 2023; Hasson `models/dinosr.py` | `encoders/dino_sr.py` | Compact re-implementation, 0-based `layer_for_tokens` |
| Causal Transformer uLM | GmSLM A.3 (`transformer_lm_big`) | `decoders/transformer_ulm.py` | Frame-level units (no MS-TLM duration stream) |
| Sleep replay, recency-decaying day sampling, X passes | Hasson `past_sampling_dataset.py` | `agent/replay_buffer.py` | Exponential (spec) and inverse-recency (official) kernels |
| Sigmoid decoder gate (T_p) | Hasson `fairseq_changes.patch` (`sigmoid-X0-K-Y`) | `agent/lr_scheduler.py` | Applied as a decoder LR gate, not a loss weight |
| Day loop with wake/sleep phases | Hasson `multimodel_task.py` + cycled datasets | `agent/trainer.py` | Separate encoder/decoder optimisers |
| Unit HiFi-GAN (no speaker, no F0) | GmSLM §3.2; Polyak et al. 2021 | `vocoder/hifigan_u2v.py` | Inference only; training TBD |
| Shuffle / Reversal / Concat / Phee eval | GmSLM §3.3 | `evaluation/zero_shot_benchmarks.py` | |
| Weekly next-period surprisal/PPL, learned lexicon | Hasson `multimodel_extraction/`, `analysis/audio_postprocessing/` | `evaluation/developmental_eval.py` | Words become calls; MFA alignment replaced by the segmenter |
| Call-type / caller probes | GmSLM §5.2, A.5 | `evaluation/downstream_probes.py` | Mean+var pooling, 3 FC + LN + ReLU |
| Inter-call interval and pitch-shift sensitivity | **New** (this project) | `evaluation/new_metrics.py` | |

### Deliberate divergences from the official code

1. **No deduplication.** Hasson's `MultiModel.units_to_units_durations` collapses repeats into (unit, duration) pairs, and its published config gives the duration loss weight 0. GmSLM shows that for marmosets, keeping repeats improves every zero-shot task (Table 4). We keep frame-level units.
2. **Sigmoid gate on the decoder LR instead of on the uLM loss.** Hasson's code has a single Adam over encoder + uLM. Adam normalises gradient scale, so a loss weight w(t) only suppresses uLM learning where w·|g| ≲ ε (1e-6). An explicit LR gate implements the intended "decoder waits for the encoder" behaviour independent of gradient scale, with the same (T_p, k, floor) triple.
3. **Replay kernel.** The spec asks for P(d | D) ∝ exp(−λ(D−d)). The official code uses P(d | D) ∝ n_d/(D−d+1). Both are implemented (`replay.kernel`, `replay.size_weighted`), and a unit test pins the official formula.

## 3. Phases

Status legend: ✅ implemented and tested on synthetic data · 🟡 scaffolded, needs real data or calibration · ⬜ not started.

### Phase 1: Preprocessing and chronological dataset

| Step | Spec | Status |
|---|---|---|
| High-pass | Butterworth, order 8, zero-phase, $f_c = 5$ kHz at the native rate | ✅ |
| Noise gating | STFT 2048/512. Per-bin floor = median over time; active bins exceed the floor by θ dB; frame is vocal if bin density ∈ [ρ_min, ρ_max] and spectral variance ≥ v_min | 🟡 thresholds uncalibrated |
| Call extraction | Merge active runs across gaps < 0.12 s; keep durations ∈ [0.25, 4] s | 🟡 validate against GmSLM P = 0.975 / R = 0.78 on the labelled set |
| Chunking | Resample to 16 kHz; 10 s windows greedily packed with calls; inter-call gaps retained | ✅ |
| Chronology | Sort by wall-clock time; never shuffle across days | ✅ |
| Day split | $n(d) = \lfloor F(d)\cdot 3600 / 10 \rfloor$ chunks, with $F(d)$ = `hours_per_day_by_period[d // days_per_period]` | ✅ (time-scale TBD, see §5) |
| Scale-out | `export_manifest` writes WAV chunks, `manifest.tsv` and `day_sizes.json`; `load_manifest` loads lazily | ✅ |

**Deliverable:** a chronological COLONYDB manifest with per-day sizes, plus a detection-quality report.

### Phase 2: Dual encoders

**Option A, marmoset HuBERT (GmSLM).** hubert-base → k-means on COLONYDB features → HuBERT fine-tuned with those targets (mask span 10, p = 8%, 150k steps) → units from **layer 1**, **k = 50**, 50 Hz. Frozen during developmental training, so it serves as an "innate" or pre-learned perceptual front-end.

**Option B, DINO-SR (Hasson).** Conv extractor (stride 320, 20 ms) → 12-layer Transformer student. The EMA teacher, with $\tau$ annealed 0.999 → 0.9999, provides targets from its top-8 layers after instance norm. These are quantised against online codebooks (V = 512) updated by EMA with decay 0.9. The student predicts teacher codes at masked frames (p = 0.8, span 10):

$$\mathcal{L}_{\text{DINO-SR}} = \frac{1}{|M|}\sum_{l\in\text{top-}K}\sum_{t\in M} -\log p_{\theta}\big(z^{(l)}_t \mid \tilde x\big),\qquad z^{(l)}_t=\arg\min_v \lVert y^{(l)}_t-c^{(l)}_v\rVert^2 .$$

Units for the uLM come from the codebook of layer `layer_for_tokens` (0-based 9, i.e. the 10th layer). DINO-SR **learns online** during development, so the unit inventory itself develops.

Both encoders implement `BaseAudioEncoder` (`encode`, `extract_features`, `get_vocab_size`, `frame_rate`) and must return one unit per frame (the no-dedup check). Status: ✅ architecture, 🟡 needs the converted marmoset HuBERT checkpoint and a full-scale DINO-SR run.

### Phase 3: Developmental training loop

For each simulated day $D$:

1. **Wake.** Train once on the day's $n_D$ chunks in chronological order.
2. **Sleep.** Train on $(X-1)\,n_D$ replayed chunks, with $X \in [25, 30]$ (official 25). A day $d \le D$ is drawn with
   $$P(d\mid D)=\frac{w(D-d)\;n_d^{\,s}}{\sum_{d'\le D} w(D-d')\;n_{d'}^{\,s}},\qquad w(\Delta)=e^{-\lambda\Delta}\ \ \text{or}\ \ \frac{1}{\Delta+1},$$
   then a chunk is drawn uniformly within that day. The sleep phase is shuffled.
3. **Per batch.** A DINO-SR step (if the encoder is trainable) and an EMA teacher update; tokenise with the teacher; then a uLM next-unit NLL step with
   $$\eta_{\text{dec}}(t)=\eta_{\max}\Big(\sigma\big(k\,(t-T_p)\big)+\epsilon\Big),$$
   using the official values $T_p = 279{,}000$ updates, $k = 3\times10^{-4}$, $\epsilon = 10^{-6}$.
4. **End of day.** Developmental evaluation hooks run weekly, and a weekly checkpoint is saved.

Status: ✅ on synthetic data (3 days, X = 25). 🟡 compute budget: at X = 25 and 8 h/day, one simulated day is about 72k chunks of 10 s, so throughput planning is needed (§6).

### Phase 4: Audio reconstruction (U2V)

Unit HiFi-GAN (Polyak et al., 2021): Embedding(V, 128) → Conv → 5 transposed-conv upsamplers (5·4·4·2·2 = 320 samples per unit) with MRF residual blocks → tanh. There is no speaker or F0 conditioning, since COLONYDB is unlabelled. Uses: resynthesis, uLM continuations (prompted by the first 3 s), FAD against real audio (VGGish), and listening tests of what the agent "would say" at each developmental stage.

Status: ✅ generator + mock renderer. ⬜ training (MPD/MSD discriminators, mel + feature-matching losses), or import of GmSLM's trained vocoder. A separate vocoder is needed for each unit inventory (HuBERT-50 and DINO-SR-512).

### Phase 5: Evaluation and trajectory tracking

| Family | Metric | Timing | Status |
|---|---|---|---|
| Zero-shot structure (GmSLM) | Shuffle, Reversal, Concat: accuracy of $\log p(\text{real}) > \log p(\text{distractor})$ | weekly + final | ✅ |
| Social structure (GmSLM) | Phee CallerChange / ReceiverChange on PHEEDB | weekly + final | ✅ mock; ⬜ PHEEDB loader |
| Predictive trajectory (Hasson) | Next-week surprisal $S=-\tfrac1N\sum\log_2 p(u_t\mid u_{<t})$ and PPL on unseen data; out-of-domain PPL | weekly | ✅ |
| Lexicon discovery (Hasson) | Per-call smoothed-probability rule (smooth 0.5, confidence 0.05, patience 1.0, stability −0.25, coverage 0.9); a call type enters the lexicon when ≥ 50% of its instances are learned | weekly | ✅ mock; 🟡 needs call-type labels or unsupervised call clustering on real data |
| Representation probes (GmSLM) | Call type and caller identity from mean+var pooled uLM states; 3 FC + LN + ReLU; macro-F1 | per checkpoint | ✅ mock; ⬜ InfantMarmosetsVox loader |
| **New:** ICI distortion | Scale inter-call gaps by $s \in \{0.25, 0.5, 2, 4\}$; accuracy and Δ surprisal (bits/unit) | per checkpoint | ✅ |
| **New:** pitch-shift sensitivity | Shift calls ±1–2 semitones; accuracy and Δ surprisal | per checkpoint | ✅ |
| Generation quality | FAD (VGGish) of resynthesised and generated audio | final | ⬜ |

## 4. Experimental design

**Main comparison:** a *developmental* agent (chronological, F(d), sleep replay, sigmoid gate, DINO-SR) versus a *GmSLM-style* model (the same data shuffled, a static uLM, frozen HuBERT), matched for number of updates and total audio.

**Ablations** (each changes one factor):

| Factor | Levels |
|---|---|
| Data order | chronological · shuffled across days · reversed chronology |
| Replay | none (X = 1) · X ∈ {10, 25, 30} · kernel exponential (λ ∈ {0.05, 0.1, 0.3}) vs. inverse recency |
| Decoder gate | none (constant LR) · T_p ∈ {early, official, late} |
| Units | HuBERT-50 (frozen) · DINO-SR-512 (online) · DINO-SR frozen after day N |
| Dedup | none (default) · dedup (replicates GmSLM Table 4 inside the developmental setting) |
| Exposure | F(d) ramp · flat 8 h/day |

**Predictions.** Chronology plus replay yields a staged trajectory, with Reversal and Shuffle learned before Concat and Phee. Removing replay causes forgetting on early-day probes. An early T_p (the decoder learning before units stabilise) hurts final Concat and Phee accuracy.

## 5. Open questions and risks

1. **GmSLM code.** It is not public. We need Talia's segmentation, HuBERT fine-tuning, k-means and vocoder checkpoints to calibrate Phase 1 and to use the real Option A encoder.
2. **Developmental time-scale.** Hasson's schedule is per month of human infancy. Marmoset vocal development is compressed into roughly the first 2–3 months, and `days_per_period` has to be set from the marmoset literature.
3. **Whose experience?** COLONYDB is a colony-level recording, not an infant's-eye (or infant-ear) stream. Longitudinal recordings centred on specific infants (and their family cage) would make "Day 1..N" literal and would allow comparison with individual infants' vocal development.
4. **Lexicon units.** Hasson aligns words with MFA. Marmoset "words" are calls, which need call-type labels or an unsupervised call-type clustering step.
5. **Compute.** Paper scale (96M-parameter DINO-SR + 150M uLM) × X = 25 × hundreds of hours needs a GPU-cluster budget and possibly mixed precision or distributed training (not yet implemented).

## 6. Milestones

| ID | Milestone | Exit criterion |
|---|---|---|
| M0 | Scaffold + mock pipeline | `python run_pipeline.py --mock` exits 0; `pytest` green ✅ |
| M1 | Real preprocessing | COLONYDB manifest built; detection P/R within 5 pts of GmSLM on the labelled set |
| M2 | GmSLM reproduction (static) | Frozen HuBERT-50 + shuffled uLM reproduces Table 1 within ±3 pts |
| M3 | Developmental run v1 | Full chronological DINO-SR run with weekly checkpoints and trajectory plots |
| M4 | Ablation grid | §4 factors; figures for trajectory × factor |
| M5 | Vocoder + generation | Unit HiFi-GAN per unit inventory; FAD per developmental stage |
| M6 | Paper draft | Joint Omer–Hasson manuscript |

## 7. References

- Sternberg, T., London, M., Omer, D., & Adi, Y. (2025). GmSLM: Generative Marmoset Spoken Language Modeling. *Findings of EMNLP 2025*. arXiv:2509.09198.
- Hasson lab. First 1,000 Days learning-modeling framework. Code: github.com/hassonlab/1kd_learning_model (2026).
- Liu, A. H., Chang, H.-J., Auli, M., Hsu, W.-N., & Glass, J. (2023). DinoSR: Self-distillation and online clustering for self-supervised speech representation learning. *NeurIPS 36*.
- Hsu, W.-N., et al. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. *IEEE/ACM TASLP*.
- Lakhotia, K., et al. (2021). On generative spoken language modeling from raw audio. *TACL*.
- Kharitonov, E., et al. (2022). Text-free prosody-aware generative spoken language modeling. *ACL*.
- Polyak, A., et al. (2021). Speech resynthesis from discrete disentangled self-supervised representations. *Interspeech*.
- Kong, J., Kim, J., & Bae, J. (2020). HiFi-GAN. *NeurIPS*.
- Takahashi, D. Y., Narayanan, D. Z., & Ghazanfar, A. A. (2013). Coupled oscillator dynamics of vocal turn-taking in monkeys. *Current Biology*.
- Takahashi, D. Y., et al. (2015). The developmental dynamics of marmoset monkey vocal production. *Science*.
- Oren, G., et al. (2024). Vocal labeling of others by nonhuman primates. *Science*.
- Sarkar, E., & Magimai-Doss, M. (2023). InfantMarmosetsVox dataset.
