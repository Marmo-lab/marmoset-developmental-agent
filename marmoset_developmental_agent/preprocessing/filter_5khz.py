"""5 kHz high-pass filtering and per-frame vocal-activity detection (GmSLM A.1).

The front-end, as the paper describes it:

1. High-pass at 5 kHz: marmoset calls sit mostly above 5 kHz (energy peaks in
   5-8 kHz), while lower frequencies carry room noise and human speech.
2. Spectrogram with window 2048 / hop 512 for frame-wise filtering.
3. Time-frequency bins below an energy threshold are discarded; frames are flagged
   as noise from the remaining bins.
4. Surviving frames are grouped over time and kept only if their duration is
   plausible for a call (see ``segmentation.py``).

Two interchangeable implementations of step 3 live here, selected by
``PreprocessingConfig.frame_features``:

``"talia"`` (default)
    Faithful port of ``segment_processor.py`` from Talia Sternberg's working tree, as
    recovered in ``gmslm_audit/GMSLM_CODE_AUDIT.md`` S3.1. Per frame it keeps STFT bins
    with magnitude above ``amp_threshold``, then derives two features:

        freq_range[t] = f_max[t] - f_min[t]    over surviving bins (0 if none)
        bandwidth[t]  = number of occupied ``bandwidth_bin_width_hz``-wide bands
                        between min_bandwidth_hz and max_bandwidth_hz

    A frame must have ``bandwidth > 0`` (or an immediate neighbour that does), and then
    falls into one of three candidate classes, whose union is the active set:

        broadband        bandwidth_bin_width * bandwidth + min_bandwidth < freq_range
        narrow, low f    0 < freq_range <= min_bandwidth
        narrow, high f   freq_range > min_bandwidth and bandwidth < narrowband_max_bins
                         -- plus a noise-cancellation pass: group these with gaps up to
                         noise_merge_gap_s and keep only runs lasting between
                         noise_min_duration_s and noise_max_duration_s

    This path exists to reproduce her ``call_indicies.pkl`` bit-for-bit, which is the
    only available check that we understand what the published pipeline actually did.
    It inherits her uncalibrated ``amp_threshold`` and heuristics.

``"spectral"``
    Our own detector, to be calibrated against hand labels:

        S_db[f, t]  = 10 log10 |STFT|^2, restricted to f >= cutoff
        floor[f]    = median_t S_db[f, t]                   (stationary noise floor)
        A[f, t]     = S_db[f, t] > floor[f] + bin_energy_threshold_db
        density[t]  = mean_f A[f, t]                        (fraction of active bins)
        variance[t] = var_f (S_db[f, t] - floor[f])         (peakiness)
        active[t]   = min_frac <= density[t] <= max_frac and variance[t] >= min_var

    Broadband transients (clicks, cage noise) have high density; silence and hiss have
    low variance; tonal or FM marmoset calls are narrow-band and peaky. Per frame it also
    records whether the frame is tonal,

        peak[t]   = max_f (S_db[f, t] - floor[f])
        width[t]  = bandwidth of the bins within tonal_peak_db of peak[t]
        tonal[t]  = peak[t] > tonal_peak_db and width[t] < tonal_max_bandwidth_hz

    which ``segmentation.py`` uses to reject calls that are mostly not tonal.

Note that detection runs at ``detection_sample_rate`` (22050 Hz, reproducing librosa's
default resample in her code), not at the recordings' native 48 kHz, and that the
high-pass filter applies to detection only -- the audio handed downstream is unfiltered
unless ``highpass_training_audio`` is set. Both choices follow her code; both are
flagged for ablation in ROADMAP.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import gcd
from typing import Optional, Tuple

import numpy as np
from scipy import signal

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig


@dataclass
class FrameActivity:
    """Per-STFT-frame vocal-activity decision and the features behind it.

    Which feature fields are populated depends on the detector that produced it:
    ``freq_range_hz`` / ``bandwidth_bins`` for the ``"talia"`` path (and they are
    required by :func:`~...segmentation.filter_calls`), ``density`` / ``variance_db2``
    for the ``"spectral"`` path.
    """

    frame_times_s: np.ndarray  # (T,) centre time of each frame
    active: np.ndarray  # (T,) bool
    hop_s: float
    freq_range_hz: Optional[np.ndarray] = None  # (T,) f_max - f_min over kept bins
    bandwidth_bins: Optional[np.ndarray] = None  # (T,) int, count of occupied bands
    density: Optional[np.ndarray] = None  # (T,) fraction of above-floor bins
    variance_db2: Optional[np.ndarray] = None  # (T,) spectral variance in dB^2
    tonal: Optional[np.ndarray] = None  # (T,) bool, narrow-line frame


def highpass_filter(
    audio: np.ndarray, sample_rate: int, cutoff_hz: float = 5_000.0, order: int = 5
) -> np.ndarray:
    """Zero-phase Butterworth high-pass (``sosfiltfilt``, so effective order is 2x).

    Order 5 + zero-phase matches Talia's ``butter(5, ...)`` + ``filtfilt``.
    """
    nyq = sample_rate / 2
    if not 0 < cutoff_hz < nyq:
        raise ValueError(f"cutoff {cutoff_hz} Hz must lie in (0, {nyq}) for sr={sample_rate}")
    sos = signal.butter(order, cutoff_hz / nyq, btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, audio.astype(np.float64)).astype(np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling (anti-aliased). GmSLM trains everything at 16 kHz."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    g = gcd(orig_sr, target_sr)
    return signal.resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def _stft(audio: np.ndarray, sample_rate: int, cfg: PreprocessingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude STFT with the paper's window/hop, in **librosa's scaling**.

    Returns ``(freqs, times, |STFT|)``.

    The scaling is not a detail. ``scipy.signal.stft`` divides by the window sum (1024
    for a 2048-point periodic Hann) while ``librosa.stft`` does not, so the two differ
    by exactly 1024x. Talia's ``AMP_THRESHOLD = 0.02`` is an absolute magnitude taken
    against *librosa's* output, so leaving scipy's normalisation in place would make
    our port ~1024x stricter than hers and detect essentially nothing. We undo the
    normalisation here so the threshold means what she meant by it.

    The ``"spectral"`` path is unaffected either way: it compares each bin to a
    per-bin median in dB, which is scale-invariant. Only the ``"talia"`` path, with its
    absolute threshold, depends on this -- which is precisely why the absolute
    threshold is the thing we intend to replace.
    """
    n_fft, hop = cfg.n_fft, cfg.hop_length
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    freqs, times, spec = signal.stft(
        audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False
    )
    window_sum = signal.get_window("hann", n_fft, fftbins=True).sum()
    return freqs, times, np.abs(spec) * window_sum


def _consecutive_series(idx: np.ndarray, max_gap: int) -> list:
    """Port of her ``find_consecutive_series``: split sorted indices where the gap
    exceeds ``max_gap`` and drop groups with <= 2 members. Groups keep only their
    member indices -- gaps are *not* filled in."""
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > max_gap) + 1)
    return [g for g in groups if len(g) > 2]


def _talia_frame_activity(
    audio: np.ndarray, sample_rate: int, cfg: PreprocessingConfig
) -> FrameActivity:
    """Line-by-line port of ``segment_processor.py`` (calculate_spectrogram,
    extract_features, cluster_frame_features), read from source in
    ``gmslm_audit/code_mirror/MonkeyPhones/code/didisDATA/refactored_code``.

    It reproduces her behaviour including its quirks, because the point of this path
    is to be checked against her ``call_indicies.pkl``:

    * the STFT is ``librosa.stft`` with ``center=True``, so frame ``t`` is centred on
      sample ``t * hop`` -- which is how ``save_calls`` maps frames back to samples;
    * bandwidth bands are the *open* intervals ``(j - 1000, j)`` for
      ``j in range(MIN_BANDWIDTH, MAX_BANDWIDTH, 1000)``, i.e. 2-10 kHz, not 3-11;
    * a frame's sub-threshold bins are filled with the mean surviving frequency before
      the band test (``find_max_and_min_freqs_by_amp``), so that mean can occupy one
      extra band;
    * a frame "has enough bandwidth" only if it *and* a neighbour are non-empty, and
      the first and last frames of a chunk never qualify;
    * the narrow-high class excludes frames already classed broadband, and its noise
      filter keeps groups by span (``last - first``), retaining member frames only.
    """
    import librosa

    mag = np.abs(librosa.stft(audio.astype(np.float32), n_fft=cfg.n_fft, hop_length=cfg.hop_length))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=cfg.n_fft)
    n_frames = mag.shape[1]
    times = np.arange(n_frames) * cfg.hop_length / sample_rate
    kept = mag > cfg.amp_threshold  # (F, T)

    # freq_range[t] = highest minus lowest surviving bin frequency, 0 if none survive.
    any_kept = kept.any(axis=0)
    f_hi = np.where(kept, freqs[:, None], -np.inf).max(axis=0)
    f_lo = np.where(kept, freqs[:, None], np.inf).min(axis=0)
    freq_range = np.where(any_kept, f_hi - f_lo, 0.0)

    # Mean surviving (non-zero) frequency per frame; stands in for sub-threshold bins.
    kept_nz = kept & (freqs[:, None] > 0)
    n_nz = kept_nz.sum(axis=0)
    mean_f = np.where(n_nz > 0, (kept_nz * freqs[:, None]).sum(axis=0) / np.maximum(n_nz, 1), 0.0)
    has_fill = (n_nz > 0) & (~kept).any(axis=0)

    width = cfg.bandwidth_bin_width_hz
    bandwidth = np.zeros(n_frames, dtype=np.int64)
    for j in np.arange(cfg.min_bandwidth_hz, cfg.max_bandwidth_hz, width):
        in_band = (freqs > j - width) & (freqs < j)
        occupied = kept[in_band].any(axis=0) | (has_fill & (mean_f > j - width) & (mean_f < j))
        bandwidth += occupied

    nonempty = bandwidth > 0
    enough = np.zeros(n_frames, dtype=bool)
    enough[1:-1] = nonempty[1:-1] & (nonempty[:-2] | nonempty[2:])

    broadband = enough & (width * bandwidth + cfg.min_bandwidth_hz < freq_range)
    narrow_low = enough & ~broadband & (freq_range <= cfg.min_freq_range_hz) & (freq_range != 0)
    narrow_high_all = (enough & ~broadband & (freq_range > cfg.min_freq_range_hz)
                       & (bandwidth < cfg.narrowband_max_bins))

    lo, hi = cfg.to_frames(cfg.noise_min_duration_s), cfg.to_frames(cfg.noise_max_duration_s)
    narrow_high = np.zeros(n_frames, dtype=bool)
    for g in _consecutive_series(np.flatnonzero(narrow_high_all), cfg.to_frames(cfg.noise_merge_gap_s)):
        if lo < g[-1] - g[0] < hi:
            narrow_high[g] = True

    return FrameActivity(
        frame_times_s=times,
        active=broadband | narrow_low | narrow_high,
        hop_s=cfg.hop_length / sample_rate,
        freq_range_hz=freq_range,
        bandwidth_bins=bandwidth,
    )


def _spectral_frame_activity(
    audio: np.ndarray, sample_rate: int, cfg: PreprocessingConfig
) -> FrameActivity:
    """Our noise-floor / density / variance frame classifier."""
    freqs, times, mag = _stft(audio, sample_rate, cfg)
    band = freqs >= cfg.highpass_cutoff_hz
    if cfg.spectral_max_freq_hz is not None:
        band &= freqs <= cfg.spectral_max_freq_hz
    s_db = 10 * np.log10(mag[band] ** 2 + 1e-12)  # (F_band, T)
    rel = s_db - np.median(s_db, axis=1, keepdims=True)
    density = (rel > cfg.bin_energy_threshold_db).mean(axis=0)
    variance = rel.var(axis=0)
    peak = rel.max(axis=0)
    width_hz = (rel > peak - cfg.tonal_peak_db).sum(axis=0) * (freqs[1] - freqs[0])
    tonal = (peak > cfg.tonal_peak_db) & (width_hz < cfg.tonal_max_bandwidth_hz)
    active = (
        (density >= cfg.min_active_bin_fraction)
        & (density <= cfg.max_active_bin_fraction)
        & (variance >= cfg.min_spectral_variance_db2)
    )
    return FrameActivity(
        frame_times_s=times,
        active=active,
        hop_s=cfg.hop_length / sample_rate,
        density=density,
        variance_db2=variance,
        tonal=tonal,
    )


def compute_frame_activity(
    audio: np.ndarray, sample_rate: int, cfg: PreprocessingConfig
) -> FrameActivity:
    """Classify each STFT frame as vocal or noise, via ``cfg.frame_features``."""
    if cfg.frame_features == "talia":
        return _talia_frame_activity(audio, sample_rate, cfg)
    return _spectral_frame_activity(audio, sample_rate, cfg)


def noise_gate(
    audio: np.ndarray,
    sample_rate: int,
    cfg: PreprocessingConfig,
    attenuation_db: float = 40.0,
    activity: Optional[FrameActivity] = None,
) -> Tuple[np.ndarray, FrameActivity]:
    """Attenuate non-vocal frames by ``attenuation_db``.

    The main pipeline does *not* gate the chunks it trains on: GmSLM keeps inter-call
    gaps (about 60% of the audio). The gate is here for standalone use and for
    visualisation.
    """
    act = activity if activity is not None else compute_frame_activity(audio, sample_rate, cfg)
    gain_frames = np.where(act.active, 1.0, 10 ** (-attenuation_db / 20))
    sample_t = np.arange(len(audio)) / sample_rate
    gain = np.interp(sample_t, act.frame_times_s, gain_frames, left=gain_frames[0], right=gain_frames[-1])
    return (audio * gain).astype(np.float32), act


def preprocess_waveform(
    audio: np.ndarray, sample_rate: int, cfg: PreprocessingConfig
) -> Tuple[np.ndarray, FrameActivity]:
    """Run the front-end on one recording.

    Returns ``(training_audio, activity)`` where ``training_audio`` is at
    ``cfg.target_sample_rate`` and ``activity`` is indexed in detection-rate frames.
    Detection happens at ``cfg.detection_sample_rate`` on the high-passed signal; the
    training audio is high-passed only if ``cfg.highpass_training_audio`` is set (her
    pipeline filtered for detection and wrote unfiltered slices -- audit S3.1 step 6).
    """
    detect_audio = resample_audio(audio, sample_rate, cfg.detection_sample_rate)
    filtered = highpass_filter(
        detect_audio, cfg.detection_sample_rate, cfg.highpass_cutoff_hz, cfg.highpass_order
    )
    activity = compute_frame_activity(filtered, cfg.detection_sample_rate, cfg)
    if cfg.highpass_training_audio:
        training = resample_audio(filtered, cfg.detection_sample_rate, cfg.target_sample_rate)
    else:
        training = resample_audio(audio, sample_rate, cfg.target_sample_rate)
    return training, activity


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _read_audio(path: str) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Apply the GmSLM 5 kHz high-pass + noise gate to a WAV file and resample to 16 kHz."
    )
    parser.add_argument("input", help="Input audio file (any rate/channels; mixed to mono).")
    parser.add_argument("output", help="Output WAV path.")
    parser.add_argument("--cutoff", type=float, default=5_000.0, help="High-pass cutoff in Hz (default 5000).")
    parser.add_argument("--order", type=int, default=5, help="Butterworth order (default 5).")
    parser.add_argument("--target-sr", type=int, default=16_000, help="Output sample rate (default 16000).")
    parser.add_argument("--gate", action="store_true", help="Also attenuate frames classified as noise.")
    parser.add_argument("--attenuation-db", type=float, default=40.0)
    args = parser.parse_args(argv)

    import soundfile as sf

    cfg = PreprocessingConfig(
        highpass_cutoff_hz=args.cutoff, highpass_order=args.order, target_sample_rate=args.target_sr
    )
    audio, sr = _read_audio(args.input)
    out, act = preprocess_waveform(audio, sr, cfg)
    if args.gate:
        out, _ = noise_gate(out, args.target_sr, cfg, args.attenuation_db, activity=act)
    sf.write(args.output, out, args.target_sr)
    print(
        f"{args.input}: {len(audio) / sr:.1f}s @ {sr} Hz -> {args.output} @ {args.target_sr} Hz; "
        f"vocal-active frames: {act.active.mean():.1%}"
    )


if __name__ == "__main__":
    main()
