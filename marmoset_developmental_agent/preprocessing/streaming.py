"""The production front-end over a whole recording: blocked detection, the call catalogue,
and chunk audio read back from the raw file.

Why blocks. A raw colony file is 48 kHz, 4 capsules, ~62 min: 2.9 GB as float32. The
detector therefore runs on blocks of ``stream_block_s`` (60 s), each read with
``stream_context_s`` (5 s) of extra audio on both sides, and a call is kept by the block
whose *core* holds its onset. The context is longer than the longest call (4 s), so a call
is never cut at a block edge, and it absorbs the high-pass filter's edge transient. Calls
that two blocks both report (an onset that moves across a core edge between them) are
merged. The block length also fixes the spectral detector's noise-floor window (a per-bin
median over ~70 s), which is close to the 64 s clips it was evaluated on
(docs/DETECTOR_EVAL.md), so the evaluated numbers apply to this path.

Why a catalogue. Every detected call becomes a :class:`CallRecord`: its position in the
file, its wall-clock time (from the file's ``bext`` start time), and quality features
measured on the detector's own spectrogram (SNR, tonal fraction, dominant frequency and its
spread). The catalogue is what the corpus run writes. Chunk counts per day, the time-of-day
profile, duration distributions and corpus-wide quality filters all come from it without
touching audio again; audio is read only for the chunks that are finally kept
(:func:`read_chunks`). This mirrors the 1kD pipeline, which carries per-minute metadata
(time, room, who is present) alongside each transcript (docs/UPSTREAM_1KD.md).

Two signals come from the four capsules (``io/channels.py``): the one detection runs on
(``detection_channel_strategy``) and the one training chunks are cut from
(``training_channel_strategy``). In-memory mono audio (synthetic data, tests) skips the
combination and goes through the same code via :class:`ArraySource`.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.io.channels import ChannelStrategy, combine, pick_best_channel
from marmoset_developmental_agent.preprocessing.filter_5khz import (
    FrameActivity,
    compute_frame_activity,
    highpass_filter,
    resample_audio,
)
from marmoset_developmental_agent.preprocessing.segmentation import (
    CallSegment,
    Chunk,
    ChunkPlan,
    activity_to_calls,
    in_night_window,
    recording_entirely_at_night,
)

# Extra native-rate audio read around each chunk so the resampler's edge effects fall
# outside the kept 10 s.
_CHUNK_MARGIN_S = 0.05


class AudioSource(Protocol):
    """Random access to a recording's samples, as ``(n, channels)`` float32."""

    sample_rate: int
    frames: int
    channels: int

    def read(self, start: int, n: int) -> np.ndarray: ...


class ArraySource:
    """An in-memory recording: ``(n,)`` mono or ``(n, channels)``."""

    def __init__(self, audio: np.ndarray, sample_rate: int) -> None:
        a = np.asarray(audio, dtype=np.float32)
        self._audio = a[:, None] if a.ndim == 1 else a
        self.sample_rate = sample_rate
        self.frames, self.channels = self._audio.shape

    def read(self, start: int, n: int) -> np.ndarray:
        return self._audio[max(0, start) : start + n]


class FileSource:
    """A WAV on disk, read by seeking; memory is bounded by the largest read."""

    def __init__(self, path: str) -> None:
        import soundfile as sf

        info = sf.info(path)
        self.path = path
        self.sample_rate, self.frames, self.channels = info.samplerate, info.frames, info.channels

    def read(self, start: int, n: int) -> np.ndarray:
        import soundfile as sf

        with sf.SoundFile(self.path) as f:
            f.seek(max(0, start))
            return f.read(n, dtype="float32", always_2d=True)


@dataclass
class CallRecord:
    """One detected call with its provenance and quality features (a catalogue row).

    ``start_s`` / ``end_s`` are padded boundaries within the recording, as in
    :class:`CallSegment`. The features are measured over the call's unpadded frames on the
    spectral detector's spectrogram, and are NaN on the ``"talia"`` path, which does not
    compute them:

    * ``snr_db``: median over frames of the loudest in-band bin above that bin's noise floor.
    * ``tonal_fraction``: fraction of tonal (narrow-line) frames, the quantity the tonality
      rejection thresholds.
    * ``peak_freq_hz`` / ``peak_freq_iqr_hz``: median and interquartile range of the
      dominant frequency over the tonal frames. A fixed device tone (the door beep,
      DETECTOR_EVAL finding 10) has an IQR of about one STFT bin and sits at the same
      frequency every time, so it shows up as a spike in the corpus-wide histogram.
    * ``confidence``: the learned detector's mean call probability over the call (NaN for
      the rule detectors).
    * ``notes``: the notes inside the call, ``"start-end;..."`` in seconds from
      ``start_s`` (``note_min_db`` / ``note_gap_s``); empty on the ``"talia"`` path.
    """

    start_s: float
    end_s: float
    start_time: dt.datetime
    snr_db: float
    tonal_fraction: float
    peak_freq_hz: float
    peak_freq_iqr_hz: float
    confidence: float = math.nan
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def segment(self) -> CallSegment:
        return CallSegment(self.start_s, self.end_s)


def _signals(block: np.ndarray, strategy: ChannelStrategy, sample_rate: int, channel: Optional[int]) -> np.ndarray:
    """``(n_out, n)`` signals for a strategy; mono input is passed through unchanged."""
    if block.shape[1] == 1:
        return block.T
    return combine(block, strategy, sample_rate, channel=channel)


def best_channel(src: AudioSource, strategy: ChannelStrategy, n: int = 8, seconds: float = 5.0) -> Optional[int]:
    """The fixed ``best_snr`` capsule, from ``n`` excerpts spread across the recording.

    Spread, as in ``io.raw_reader._sample_excerpts``, so that one silent night stretch
    cannot decide it. None for every other strategy and for mono sources.
    """
    if strategy != "best_snr" or src.channels == 1:
        return None
    span = int(round(seconds * src.sample_rate))
    starts = np.linspace(0, max(0, src.frames - span), n).astype(np.int64)
    excerpts = np.concatenate([src.read(int(s), span) for s in starts], axis=0)
    return pick_best_channel(excerpts, src.sample_rate)


def detection_activity(audio: np.ndarray, sample_rate: int, cfg: PreprocessingConfig) -> FrameActivity:
    """Resample to the detection rate, high-pass, and classify frames (as ``preprocess_waveform``)."""
    x = resample_audio(audio, sample_rate, cfg.detection_sample_rate)
    x = highpass_filter(x, cfg.detection_sample_rate, cfg.highpass_cutoff_hz, cfg.highpass_order)
    return compute_frame_activity(x, cfg.detection_sample_rate, cfg)


def _notes(activity: FrameActivity, call: CallSegment, cfg: PreprocessingConfig) -> str:
    """Notes inside a call as ``"start-end;..."``, seconds from the call's (padded) start."""
    if activity.peak_db is None:
        return ""
    t = activity.frame_times_s
    sel = np.flatnonzero((t >= call.start_s) & (t <= call.end_s) & (activity.peak_db > cfg.note_min_db))
    if not sel.size:
        return ""
    runs = np.split(sel, np.flatnonzero(np.diff(sel) * activity.hop_s > cfg.note_gap_s) + 1)
    runs = [r for r in runs if len(r) * activity.hop_s >= cfg.note_gap_s]  # a lone frame is not a note
    half = activity.hop_s / 2
    return ";".join(f"{max(0.0, t[r[0]] - half - call.start_s):.3f}-{t[r[-1]] + half - call.start_s:.3f}"
                    for r in runs)


def _features(activity: FrameActivity, call: CallSegment, cfg: PreprocessingConfig) -> List[object]:
    """``[snr_db, tonal_fraction, peak_freq_hz, peak_freq_iqr_hz, confidence, notes]`` over
    the call's unpadded frames (notes over its padded span)."""
    t, tol = activity.frame_times_s, activity.hop_s / 2
    conf = math.nan
    if activity.call_prob is not None:
        inside = (t >= call.start_s - tol) & (t <= call.end_s + tol)
        conf = float(activity.call_prob[inside].mean()) if inside.any() else math.nan
    notes = _notes(activity, call, cfg)
    if activity.peak_db is None or activity.tonal is None or activity.peak_freq_hz is None:
        return [math.nan] * 4 + [conf, notes]
    t, tol = activity.frame_times_s, activity.hop_s / 2
    lo = call.start_s + cfg.boundary_pad_s if call.start_s > 0 else 0.0
    sel = (t >= lo - tol) & (t <= call.end_s - cfg.boundary_pad_s + tol)
    if not sel.any():
        return [math.nan] * 4 + [conf, notes]
    tonal = activity.tonal[sel]
    freqs = activity.peak_freq_hz[sel][tonal] if tonal.any() else activity.peak_freq_hz[sel]
    q1, med, q3 = np.percentile(freqs, [25, 50, 75])
    return [float(np.median(activity.peak_db[sel])), float(tonal.mean()), float(med), float(q3 - q1), conf, notes]


def _merge_overlapping(found: List[CallRecord]) -> List[CallRecord]:
    """Union overlapping calls, keeping the features of the one with the higher SNR.

    Overlaps arise only from a call reported by two neighbouring blocks, or by two capsules
    under ``per_channel``.
    """
    out: List[CallRecord] = []
    for r in sorted(found, key=lambda r: r.start_s):
        if out and r.start_s < out[-1].end_s:  # sorted, so the earlier start is prev's
            prev = out[-1]
            keep = r if r.snr_db > prev.snr_db else prev
            out[-1] = CallRecord(prev.start_s, max(prev.end_s, r.end_s), prev.start_time, keep.snr_db,
                                 keep.tonal_fraction, keep.peak_freq_hz, keep.peak_freq_iqr_hz,
                                 keep.confidence, keep.notes)
        else:
            out.append(r)
    return out


def detect_recording(src: AudioSource, start_time: dt.datetime, cfg: PreprocessingConfig) -> List[CallRecord]:
    """Detect every call in a recording, block by block; the result is chronological.

    With ``cfg.exclude_night``, a recording lying entirely in the night window is skipped
    before any audio is read, and calls with a night onset are dropped from the rest.
    """
    sr = src.sample_rate
    if cfg.exclude_night and recording_entirely_at_night(start_time, src.frames / sr, cfg):
        return []
    core = int(round(cfg.stream_block_s * sr))
    ctx = int(round(cfg.stream_context_s * sr))
    channel = best_channel(src, cfg.detection_channel_strategy)
    found: List[CallRecord] = []
    for s0 in range(0, src.frames, core):
        a, b = max(0, s0 - ctx), min(src.frames, s0 + core + ctx)
        block = src.read(a, b - a)
        core_lo, core_hi = s0 / sr, min(s0 + core, src.frames) / sr
        for x in _signals(block, cfg.detection_channel_strategy, sr, channel):
            activity = detection_activity(x, sr, cfg)
            for c in activity_to_calls(activity, cfg):
                onset = a / sr + c.start_s
                if core_lo <= onset < core_hi:
                    found.append(CallRecord(onset, a / sr + c.end_s, start_time + dt.timedelta(seconds=onset),
                                            *_features(activity, c, cfg)))
    calls = _merge_overlapping(found)
    if cfg.exclude_night:
        calls = [r for r in calls if not in_night_window(r.start_time, cfg)]
    return calls


def read_call_audio(src: AudioSource, records: Sequence[CallRecord], cfg: PreprocessingConfig) -> List[np.ndarray]:
    """Each call's audio: the training channel at ``call_audio_sample_rate``, from
    ``start_s - call_audio_margin_s`` to ``end_s + call_audio_margin_s`` (clipped to the file).

    Unlike :func:`read_chunks` nothing is padded or gated: a call's audio starts
    ``min(margin, start_s)`` before its catalogued start.
    """
    if cfg.training_channel_strategy == "per_channel":
        raise ValueError("training_channel_strategy 'per_channel' does not give one training signal")
    sr = src.sample_rate
    channel = best_channel(src, cfg.training_channel_strategy)
    out: List[np.ndarray] = []
    for r in records:
        a = max(0, int(round((r.start_s - cfg.call_audio_margin_s) * sr)))
        b = min(src.frames, int(round((r.end_s + cfg.call_audio_margin_s) * sr)))
        x = _signals(src.read(a, b - a), cfg.training_channel_strategy, sr, channel)[0]
        out.append(resample_audio(x, sr, cfg.call_audio_sample_rate))
    return out


def read_chunks(
    src: AudioSource,
    plans: Sequence[ChunkPlan],
    cfg: PreprocessingConfig,
    recording_id: str,
    start_time: dt.datetime,
    recording_day: Optional[int] = None,
) -> List[Chunk]:
    """Cut each planned chunk from the source at ``target_sample_rate``.

    Each window is read at the native rate with a small margin, reduced to one signal with
    ``training_channel_strategy`` (a per-chunk choice for ``ambisonic_beam`` and
    ``max_energy``, so the 10 s are phase-continuous), optionally high-passed (at the native
    rate; her code filtered at the detection rate, a negligible difference above 5 kHz), and
    resampled.
    """
    if cfg.training_channel_strategy == "per_channel":
        raise ValueError("training_channel_strategy 'per_channel' does not give one training signal")
    sr, tsr = src.sample_rate, cfg.target_sample_rate
    n_chunk = int(round(cfg.chunk_duration_s * tsr))
    n_native = int(round(cfg.chunk_duration_s * sr))
    margin = int(round(_CHUNK_MARGIN_S * sr))
    channel = best_channel(src, cfg.training_channel_strategy)
    chunks: List[Chunk] = []
    for plan in plans:
        # Start on the target-rate sample grid, as slicing whole-recording 16 kHz audio would.
        s0 = int(round(round(plan.offset_s * tsr) * sr / tsr))
        a = max(0, s0 - margin)
        x = _signals(src.read(a, s0 - a + n_native + margin), cfg.training_channel_strategy, sr, channel)[0]
        if cfg.highpass_training_audio:
            x = highpass_filter(x, sr, cfg.highpass_cutoff_hz, cfg.highpass_order)
        lead = int(round((s0 - a) * tsr / sr))
        wav = resample_audio(x, sr, tsr)[lead : lead + n_chunk]
        if len(wav) < n_chunk:
            wav = np.pad(wav, (0, n_chunk - len(wav)))
        chunks.append(
            Chunk(
                waveform=wav.astype(np.float32),
                sample_rate=tsr,
                recording_id=recording_id,
                start_time=start_time + dt.timedelta(seconds=plan.offset_s),
                offset_s=plan.offset_s,
                calls=plan.calls,
                recording_day=recording_day,
            )
        )
    return chunks
