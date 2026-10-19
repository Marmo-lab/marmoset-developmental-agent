"""Typed configuration for every stage of the Marmoset Developmental Agent.

Each dataclass groups the hyper-parameters of one decoupled stage. Defaults are
the *paper-scale* values (GmSLM, EMNLP 2025 Findings; Hasson lab 1kd_learning_model).
:meth:`AgentConfig.mock` returns a tiny CPU-friendly variant used for smoke tests.

Sources for defaults
--------------------
* GmSLM (Sternberg et al., 2025), Sec. 3 & Appendix A.1-A.5: 5 kHz high-pass,
  STFT n_fft=2048 / hop=512 noise filtering, calls in [0.25, 4] s, 10 s chunks at
  16 kHz, HuBERT-base layer 1, 50 k-means units, no deduplication, transformer_lm_big
  uLM, probes on mean+variance pooled features.
* Hasson lab ``references/hasson_dev`` (training/speech_learning/model_training_config/
  default.yaml, train.sh, data_preparation/data_examples/daily_hours_per_month.json):
  DINO-SR codebook 512, mask_prob 0.8 / mask_length 10, 12-layer encoder, top-8
  codebooks, layer_for_tokens 9, dataset_cycle_repeats 25, sigmoid schedule
  ``sigmoid-279_000-0.0003-0.000001``, weekly (7-cycle) checkpoints.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Phase 1: preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessingConfig:
    """Acoustic front-end (GmSLM Appendix A.1-A.2).

    Provenance of every constant is tagged in its comment:

    ``[paper]``   stated in GmSLM (Sternberg et al., 2025).
    ``[talia]``   read off her working code (``gmslm_audit/GMSLM_CODE_AUDIT.md`` S3.1);
                  preferred over the paper wherever the two differ only by rounding.
    ``[ours]``    our decision, because neither source makes one (audit tier D).

    Durations are stored in **seconds**, never in STFT frames. Talia's code hard-coded
    frame counts that are only meaningful at hop 512 / 22050 Hz (so e.g. her 10-frame
    call floor is 0.2322 s, which the paper rounds to 0.25 s). Keeping seconds canonical
    means :attr:`detection_sample_rate` can change without silently moving every
    duration threshold. Use :meth:`hop_s` / :meth:`to_frames` to convert.
    """

    target_sample_rate: int = 16_000  # [paper] rate everything downstream trains at
    # [talia] Detection ran on librosa's default resample, NOT the native 48 kHz. This
    # caps Nyquist at 11025 Hz -- flush against max_bandwidth_hz below, which is very
    # likely an accident rather than a design. Kept for fidelity; see ROADMAP ablation.
    detection_sample_rate: int = 22_050
    highpass_cutoff_hz: float = 5_000.0  # [paper] + [talia], exact agreement
    highpass_order: int = 5  # [talia] order 5, zero-phase -> effective order 10
    n_fft: int = 2048  # [paper]; [talia] librosa default, never set explicitly
    hop_length: int = 512  # [paper]; [talia] librosa default

    # [talia] Her pipeline high-pass filtered for *detection only*: the segments written
    # for HuBERT are slices of the unfiltered waveform. False reproduces that.
    highpass_training_audio: bool = False

    # Which per-frame feature set drives detection.
    #   "talia"    -- faithful port of segment_processor.py (freq_range / bandwidth).
    #                 Use this to validate against her call_indicies.pkl.
    #   "spectral" -- our noise-floor + density + variance detector, to be calibrated
    #                 against hand labels once those exist.
    frame_features: Literal["talia", "spectral"] = "talia"

    # --- "talia" feature path (segment_processor.py, read from source) ------------ #
    amp_threshold: float = 0.02  # [talia] AMP_THRESHOLD on librosa |STFT|
    # [talia] MIN_BANDWIDTH / MAX_BANDWIDTH. These are loop bounds, not band edges: she
    # iterates j over range(3000, 11000, 1000) and tests the open band (j-1000, j), so
    # the bands actually counted span 2-10 kHz. See filter_5khz._talia_frame_activity.
    min_bandwidth_hz: float = 3_000.0
    max_bandwidth_hz: float = 11_000.0
    bandwidth_bin_width_hz: float = 1_000.0  # [talia] delta_freq_for_bandwidth_calc
    min_freq_range_hz: float = 3_000.0  # [talia] MIN_FREQ_RANGE, splits narrow-low / -high
    narrowband_max_bins: int = 6  # [talia] MINIMAL_BANDWIDTH_FOR_EQUAL_RANGE (< 6 bands)
    # [talia] Noise-cancellation on the narrow-high class: group with gaps <= 20 frames,
    # keep groups whose span (last - first) is 20 < span < 80 frames. The paper rounds
    # these to 0.5 s / 2 s.
    noise_merge_gap_s: float = 0.4644
    noise_min_duration_s: float = 0.4644
    noise_max_duration_s: float = 1.8576
    # [talia] filter_calls rejections.
    bandwidth_crossing_hz: float = 1_000.0  # BANDWIDTH_CROSSING_LINE (= 1 band)
    max_bandwidth_crossings: int = 3  # LOW_THRESHOLD_FOR_BANDWIDTH_CROSSING; reject at >=
    # "Abrupt up": reject a *short* candidate (span < 20 frames) whose freq_range ever
    # exceeds 2 kHz -- i.e. a brief broadband click. Not a sliding-window jump test,
    # which an earlier port of ours wrongly implemented from the audit's paraphrase and
    # which rejected long calls instead.
    abrupt_up_freq_range_hz: float = 2_000.0  # UPPER_THRESHOLD_FOR_ABBRUPT_UP
    abrupt_up_max_span_s: float = 0.4644  # DURATION_FOR_ABBRUPT_UP_CALC = 20 frames
    boundary_pad_s: float = 0.02268  # [talia] WRAPPING_SAMPLES, +-500 samples at 22050 Hz

    # --- "spectral" feature path (ours) ---------------------------------------- #
    # A time-frequency bin is "active" if it exceeds the per-bin noise floor
    # (median over time) by this many dB.
    bin_energy_threshold_db: float = 12.0  # [ours]
    # Tonal marmoset calls occupy a narrow band; broadband transients fill many bins.
    min_active_bin_fraction: float = 0.005  # [ours]
    max_active_bin_fraction: float = 0.35  # [ours]
    # Frames whose (log) spectral variance is below this are treated as noise.
    min_spectral_variance_db2: float = 20.0  # [ours]
    # Upper edge of the band the spectral detector looks at; None = Nyquist of the
    # detection rate. Lets detection run at a higher rate (marmoset calls reach 12-16 kHz,
    # above the 11 kHz Nyquist of 22050 Hz) without the empty band above the calls
    # diluting the active-bin fractions.
    spectral_max_freq_hz: Optional[float] = None  # [ours]
    # Tonality rejection (docs/DETECTOR_EVAL.md findings 9-10). A frame is *tonal* if its
    # loudest in-band bin is > tonal_peak_db above that bin's floor and the bins within
    # tonal_peak_db of that peak span < tonal_max_bandwidth_hz: a narrow line, not a
    # broadband smear. A call is rejected if fewer than min_tonal_fraction of the frames
    # across its unpadded span are tonal. Cage-noise transients sit in the same band as
    # calls (dominant-frequency AUC 0.38), but are broad: tonal fraction separates them
    # with AUC 0.96. None disables the rejection. 0.5 is provisional, from the 30 labelled
    # clips (12 dB, 48 kHz / 16 kHz: F1 0.74 -> 0.89, chunk P 0.72 -> 0.86); confirm on the
    # held-out set.
    tonal_peak_db: float = 10.0  # [ours]
    tonal_max_bandwidth_hz: float = 2_000.0  # [ours]
    min_tonal_fraction: Optional[float] = 0.5  # [ours]

    # --- shared run grouping and duration limits -------------------------------- #
    # [talia] Gaps <= 4 frames between active frames are bridged into one call
    # (find_consecutive_series splits where diff > 4).
    merge_gap_s: float = 0.0929
    # [talia] find_consecutive_series drops groups with <= 2 *members*. A count of
    # active frames, not a duration, so it is not stored in seconds.
    min_run_members: int = 3
    # [talia] Duration is tested on the unpadded frame span (last - first), strictly:
    # span > 10 frames and span < 4 s. The paper rounds the floor to 0.25 s. Padding
    # (boundary_pad_s) is applied only after these tests, as in her save_calls.
    min_call_duration_s: float = 0.2322
    max_call_duration_s: float = 4.0  # [paper] + [talia]

    # --- output chunking (GmSLM A.2) -------------------------------------------- #
    chunk_duration_s: float = 10.0  # [paper] + [talia]
    chunk_pre_pad_s: float = 0.1  # [ours]
    min_calls_per_chunk: int = 1  # [ours]

    # --- night exclusion [ours] ------------------------------------------------- #
    # Recording level tracks time of day (r = +0.859 over 54 files) and ~46% of the
    # corpus is night audio with almost no calls. When enabled, calls whose wall-clock
    # time falls in [night_start_hour, night_end_hour) -- a window that wraps midnight --
    # are dropped, and recordings lying entirely inside it are skipped without running
    # detection. Raw files are never touched. Off by default: whether to enable it is
    # decided from the night clips in the hand-labelled set (are there real calls at
    # night, and does the detector raise false alarms there?).
    exclude_night: bool = False
    night_start_hour: float = 19.0
    night_end_hour: float = 6.0

    def hop_s(self) -> float:
        """Seconds per STFT frame at the detection rate."""
        return self.hop_length / self.detection_sample_rate

    def to_frames(self, seconds: float) -> int:
        """Convert a duration in seconds to a frame count at the detection rate."""
        return int(round(seconds / self.hop_s()))


@dataclass
class ExposureScheduleConfig:
    """Daily exposure schedule F(d) (Hasson lab, generate_cycle_sizes.py).

    ``hours_per_day_by_period[p]`` gives the audio hours consumed per simulated day
    during period ``p`` (a period lasts ``days_per_period`` days). After the list is
    exhausted the final value repeats. Hasson's infant schedule rises 2 h -> 8 h/day
    over 13 months; marmoset development is faster, so ``days_per_period`` is a
    free parameter to be calibrated.
    """

    hours_per_day_by_period: List[float] = field(
        default_factory=lambda: [2, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]
    )
    days_per_period: int = 30
    # "exposure": re-cut the chronological stream by F(d) (official behaviour).
    # "calendar": one simulated day per recording day.
    day_assignment: Literal["exposure", "calendar"] = "exposure"


# --------------------------------------------------------------------------- #
# Phase 2: encoders
# --------------------------------------------------------------------------- #
@dataclass
class HuBERTEncoderConfig:
    """Marmoset-adapted HuBERT (GmSLM Sec. 3.2, Appendix A.3)."""

    # HF model id / local dir of a *converted* marmoset HuBERT. None -> random init
    # from the architecture fields below (mock mode).
    pretrained_path: Optional[str] = None
    # [paper] layer 1. Her code gives three disagreeing signals (launch script layer 6,
    # downstream scripts layer 12, vocoder checkpoint name implies layer 9), so there is
    # no coherent code value to defer to -- audit S3.4, open question 3.
    layer: int = 1
    n_clusters: int = 50  # [paper] + [talia]
    # Unit-vocabulary sweep run by tools/sweep_units.py (OPEN_DECISIONS D15): 50 is
    # GmSLM's value, 200 the upper end GSLM found useful for English HuBERT units.
    n_clusters_sweep: List[int] = field(default_factory=lambda: [50, 100, 200])
    kmeans_batch_size: int = 5_000  # [talia] MiniBatchKMeans default; paper says 10,000
    kmeans_n_init: int = 20  # [paper] + [talia]
    kmeans_fit_fraction: float = 0.5  # [paper]; [talia] hard-coded 5-of-10 shards
    normalize_waveform: bool = False
    # Marmoset adaptation. The paper calls this "fine-tuning", but the only fine-tune
    # config in her tree is fairseq's ASR-CTC recipe, which needs transcripts that do
    # not exist for marmosets -- so the real regime is HuBERT *pretraining from scratch*
    # via iterative relabeling (audit S3.4). Two of the paper's quoted numbers are
    # decimal-point typos (it also prints an impossible "5e-45" uLM LR), so we take the
    # code's field-standard values instead:
    mask_prob: float = 0.80  # [talia]; paper says 8% -- 0.08 would not train
    mask_length: int = 10  # [paper] + [talia] (fairseq default)
    pretrain_lr: float = 3.5e-4  # [talia]; paper says 3.5e-3 -- would diverge
    pretrain_max_update: int = 150_000  # [paper]; her configs show 1.6M / 400k
    pretrain_warmup_updates: int = 40_000  # [talia]
    # Architecture used when pretrained_path is None (hubert-base shape).
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    conv_dim: int = 512
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16


@dataclass
class DinoSRConfig:
    """DINO-SR encoder (Liu et al., NeurIPS 2023; Hasson default.yaml)."""

    conv_dim: int = 512
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    conv_pos_kernel: int = 95
    conv_pos_groups: int = 16
    dropout: float = 0.1
    codebook_size: int = 512
    average_top_k_layers: int = 8
    layer_for_tokens: int = 9  # 0-based transformer layer (official convention) -> 10th layer
    codebook_decay: float = 0.9
    mask_prob: float = 0.8
    mask_length: int = 10
    ema_decay: float = 0.999
    ema_end_decay: float = 0.9999
    ema_anneal_end_step: int = 40_000
    normalize_waveform: bool = True
    lr: float = 1e-4
    weight_decay: float = 0.01


@dataclass
class EncoderConfig:
    """Which discretizer drives training, plus both encoders' settings."""

    active: Literal["dinosr", "hubert"] = "dinosr"
    hubert: HuBERTEncoderConfig = field(default_factory=HuBERTEncoderConfig)
    dinosr: DinoSRConfig = field(default_factory=DinoSRConfig)


# --------------------------------------------------------------------------- #
# Decoder (uLM)
# --------------------------------------------------------------------------- #
@dataclass
class DecoderConfig:
    """Causal unit language model (GmSLM A.3: fairseq transformer_lm_big).

    No uLM script, log or checkpoint survives in Talia's tree (audit S3.5), so these
    are the paper's values with nothing to cross-check them against. Note that
    ``transformer_lm_big`` is a GPT-2-Large-shaped model designed for a ~50k-token
    vocabulary, whereas ours is 53 (50 units + PAD/BOS/EOS) -- the embedding and output
    stacks are wildly overparameterised at that scale. Unit-vocabulary size and the
    matching model shape are an open research question for this project, not settled
    here; see ROADMAP.
    """

    embed_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    ffn_dim: int = 4096
    dropout: float = 0.1
    max_positions: int = 3072


# --------------------------------------------------------------------------- #
# Phase 3: developmental training
# --------------------------------------------------------------------------- #
@dataclass
class ReplayConfig:
    """Sleep-consolidation replay.

    ``passes_per_day`` (X) mirrors Hasson's ``dataset_cycle_repeats``: a simulated
    day contains the day's own samples once ("wake") plus ``(X - 1) * n_D`` samples
    replayed from days 1..D ("sleep").
    """

    passes_per_day: int = 25
    # "exponential": w(d) = exp(-lambda (D - d)); "inverse_recency": w(d) = 1 / (D - d + 1)
    # (the latter is the official 1kd_learning_model kernel).
    kernel: Literal["exponential", "inverse_recency"] = "exponential"
    decay_lambda: float = 0.1
    # Multiply kernel by the number of samples in day d (official behaviour).
    size_weighted: bool = False
    include_current_day: bool = True


@dataclass
class SigmoidScheduleConfig:
    """Decoder LR gate: lr(t) = lr_max * (sigmoid(k (t - T_p)) + floor).

    Hasson's example ``sigmoid-279_000-0.0003-0.000001`` -> T_p=279000, k=3e-4,
    floor=1e-6, with t counted in optimizer updates.
    """

    lr_max: float = 5e-4
    t_p: float = 279_000.0
    k: float = 3e-4
    floor: float = 1e-6
    unit: Literal["step", "day"] = "step"


@dataclass
class TrainingConfig:
    num_days: Optional[int] = None  # None -> all available days
    batch_size: int = 22
    drop_last: bool = False
    grad_clip: float = 1.0
    decoder_weight_decay: float = 0.01
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    decoder_lr: SigmoidScheduleConfig = field(default_factory=SigmoidScheduleConfig)
    train_encoder: bool = True  # only meaningful for trainable encoders (DINO-SR)
    checkpoint_every_days: int = 7
    log_every_steps: int = 50


# --------------------------------------------------------------------------- #
# Phase 4: vocoder
# --------------------------------------------------------------------------- #
@dataclass
class VocoderConfig:
    """Unit HiFi-GAN (Polyak et al., 2021) without speaker / F0 conditioning."""

    checkpoint_path: Optional[str] = None
    unit_embed_dim: int = 128
    upsample_initial_channel: int = 512
    upsample_rates: List[int] = field(default_factory=lambda: [5, 4, 4, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [11, 8, 8, 4, 4])
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilations: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    # With no checkpoint: "additive" renders each unit as a 5-8 kHz tone (interpretable
    # mock); "neural" runs the randomly initialised generator (shape check only).
    mock_mode: Literal["additive", "neural"] = "additive"


# --------------------------------------------------------------------------- #
# Phase 5: evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationConfig:
    score_reduction: Literal["sum", "mean"] = "mean"
    eval_every_days: int = 7  # "weekly"
    max_eval_chunks: int = 256
    concat_min_calls: int = 1  # GmSLM used segments with six calls
    # Lexicon discovery (Hasson extract_word_dictionary.sh defaults).
    lexicon_smooth_factor: float = 0.5
    lexicon_confidence: float = 0.05
    lexicon_patience: float = 1.0
    lexicon_stability: float = -0.25
    lexicon_min_coverage: float = 0.9
    lexicon_min_learned_fraction: float = 0.5
    # Probes (GmSLM A.5).
    probe_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    probe_epochs: int = 30  # [talia] submit_classifier.sh; paper says 20
    probe_lr: float = 1e-3
    probe_val_fraction: float = 0.1
    # New metrics.
    ici_scale_factors: List[float] = field(default_factory=lambda: [0.25, 0.5, 2.0, 4.0])
    pitch_shift_semitones: List[float] = field(default_factory=lambda: [-2.0, -1.0, 1.0, 2.0])


@dataclass
class DataConfig:
    """Real-data inputs (ignored with --mock)."""

    # CSV with columns path,start_time[,recording_id] -> raw recordings to preprocess.
    recordings_manifest: Optional[str] = None
    # manifest.tsv from dataset_builder.export_manifest -> skip preprocessing.
    chunk_manifest: Optional[str] = None
    # Where to export preprocessed chunks when building from raw recordings.
    export_dir: Optional[str] = None
    # Days held out after training for final zero-shot evaluation.
    eval_holdout_days: int = 7


@dataclass
class AgentConfig:
    seed: int = 0
    device: str = "auto"
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    exposure: ExposureScheduleConfig = field(default_factory=ExposureScheduleConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def mock(cls) -> "AgentConfig":
        """Tiny configuration that runs the full pipeline on CPU in about a minute."""
        cfg = cls()
        cfg.output_dir = "outputs/mock"
        cfg.device = "cpu"
        cfg.training.num_days = 3
        # The "talia" detector keys off an absolute |STFT| threshold (amp_threshold),
        # implicitly calibrated to the colony recordings' gain, so it under-detects on
        # synthetic audio (recall ~0.4 vs ~0.98 on the mock colony). Mock runs use our
        # level-relative detector; both paths are covered by tests/test_pipeline.py.
        cfg.preprocessing.frame_features = "spectral"
        # ~3-5 ten-second chunks per simulated day; one day per "period".
        cfg.exposure.hours_per_day_by_period = [0.0084, 0.0112, 0.014]
        cfg.exposure.days_per_period = 1

        h = cfg.encoder.hubert
        h.n_clusters, h.kmeans_n_init, h.kmeans_batch_size = 50, 3, 2048
        h.hidden_size, h.num_hidden_layers, h.num_attention_heads = 64, 2, 4
        h.intermediate_size, h.conv_dim = 128, 32
        h.num_conv_pos_embeddings, h.num_conv_pos_embedding_groups = 16, 4

        d = cfg.encoder.dinosr
        d.conv_dim, d.embed_dim, d.num_layers, d.num_heads, d.ffn_dim = 32, 64, 3, 4, 128
        d.conv_pos_kernel, d.conv_pos_groups = 15, 4
        d.average_top_k_layers, d.layer_for_tokens = 2, 2
        d.ema_anneal_end_step = 100
        d.lr = 5e-4

        cfg.decoder = DecoderConfig(embed_dim=64, num_layers=2, num_heads=4, ffn_dim=128, max_positions=1024)

        cfg.training.batch_size = 8
        cfg.training.log_every_steps = 5
        cfg.training.checkpoint_every_days = 1
        cfg.training.replay.passes_per_day = 25
        # Gate opens during day 2 of the mock run (t counted in updates).
        cfg.training.decoder_lr = SigmoidScheduleConfig(lr_max=1e-3, t_p=20, k=0.3, floor=1e-3)

        cfg.vocoder.upsample_initial_channel = 32
        cfg.evaluation.eval_every_days = 1
        cfg.evaluation.max_eval_chunks = 32
        cfg.evaluation.probe_hidden_sizes = [32, 16, 8]
        cfg.evaluation.probe_epochs = 30
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# YAML (de)serialisation
# --------------------------------------------------------------------------- #
def _from_dict(cls: Type[T], data: Dict[str, Any], base: Optional[T] = None) -> T:
    """Recursively overlay ``data`` onto dataclass ``cls`` (or onto ``base``)."""
    obj = base if base is not None else cls()
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for key, value in data.items():
        ftype = hints[key]
        if get_origin(ftype) is Union:  # Optional[...]
            ftype = next(a for a in get_args(ftype) if a is not type(None))
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            value = _from_dict(ftype, value, getattr(obj, key))
        setattr(obj, key, value)
    return obj


def load_config(path: Union[str, Path, None] = None, mock: bool = False) -> AgentConfig:
    """Load an :class:`AgentConfig`, overlaying YAML values on paper or mock defaults."""
    base = AgentConfig.mock() if mock else AgentConfig()
    if path is None:
        return base
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _from_dict(AgentConfig, data, base)


def save_config(cfg: AgentConfig, path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
