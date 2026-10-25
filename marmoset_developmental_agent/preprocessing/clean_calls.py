"""Spectral cleaning of detected calls: room noise and cage clicks removed, silence outside the call.

The user's goal (2026-10-22) is clean calls with exact times, the colony analogue of the
1kD pipeline's clean TTS speech (docs/UPSTREAM_1KD.md). Harmonic re-synthesis was tried
and rejected by ear (it cannot render infant cries or chatter, and loses pulsing); this
module is the chosen spectral route. All parameters live in ``CleaningConfig``.

For one call, given the audio around it and where every detected call lies in that audio:

1. High-pass at ``highpass_hz`` and take an STFT (``n_fft`` / ``hop_length``).
2. **Noise profile:** per bin, ``median_to_mean`` x the median power over the frames that
   lie outside every detected call (by ``guard_s``). The median ignores the occasional
   loud frame, so neither calls nor clicks inflate it.
3. **Clicks:** short frames in which much of ``click_band_hz`` jumps above the noise at
   once are capped, bin by bin, at the level of their non-click neighbours. A call line
   running through a click keeps its level; the vertical spike goes.
4. **Gain:** each cell is scaled by ``max(1 - over_subtraction * N / P, gain_floor)``,
   smoothed, and the whole signal fades to zero outside the call.
5. Inverse STFT with the original phase.

Known limits (listening demo, 2026-10-25): clicks inside calls are reduced, not always
removed; nothing here separates two animals calling at once.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

from marmoset_developmental_agent.config.agent_config import CleaningConfig

Interval = Tuple[float, float]


def _stft(x: np.ndarray, sr: int, cfg: CleaningConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return signal.stft(x, sr, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length)


def noise_profile(power: np.ndarray, t: np.ndarray, calls: Sequence[Interval], cfg: CleaningConfig) -> Tuple[np.ndarray, float]:
    """Per-bin noise power from the frames outside every call; also the seconds of noise used."""
    outside = np.ones(len(t), bool)
    for a, b in calls:
        outside &= ~((t >= a - cfg.guard_s) & (t <= b + cfg.guard_s))
    if not outside.any():  # wall-to-wall calling: fall back to the quietest fifth of frames
        level = power.sum(axis=0)
        outside = level <= np.percentile(level, 20)
    hop = float(t[1] - t[0]) if len(t) > 1 else 0.0
    return cfg.median_to_mean * np.median(power[:, outside], axis=1), float(outside.sum() * hop)


def declick(X: np.ndarray, f: np.ndarray, noise: np.ndarray, cfg: CleaningConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Cap click frames at their non-click neighbours' level; returns ``(X, click mask)``."""
    mag = np.abs(X)
    rel = 10 * np.log10(mag ** 2 / (noise[:, None] + 1e-20) + 1e-12)
    band = (f >= cfg.click_band_hz[0]) & (f <= cfg.click_band_hz[1])
    frac = (rel[band] > cfg.click_bin_db).mean(axis=0)
    base = median_filter(frac, size=cfg.click_context_frames)
    click = (frac > cfg.click_min_fraction) & (frac - base > cfg.click_min_jump)
    click = np.convolve(click.astype(float), [1.0, 1.0, 1.0], "same") > 0
    if not click.any():
        return X, click
    cap = mag.copy()
    n, w = mag.shape[1], cfg.click_fill_frames
    for j in np.flatnonzero(click):
        nb = np.arange(max(0, j - w), min(n, j + w + 1))
        nb = nb[~click[nb]]
        if nb.size:
            cap[:, j] = np.minimum(mag[:, j], np.median(mag[:, nb], axis=1))
    return X * (cap / (mag + 1e-20)), click


def clean_call(audio: np.ndarray, sr: int, call: Interval, calls: Sequence[Interval], cfg: CleaningConfig) -> Tuple[np.ndarray, dict]:
    """Clean one call. ``audio`` is mono; ``call`` and ``calls`` (every detected call in the
    audio, ``call`` included) are in seconds from its start. Returns the cleaned audio, the
    same length, silent outside ``call`` beyond ``fade_s``, and a few diagnostics."""
    x = signal.sosfiltfilt(signal.butter(5, cfg.highpass_hz, "highpass", fs=sr, output="sos"), audio.astype(np.float64))
    f, t, X = _stft(x, sr, cfg)
    noise, noise_s = noise_profile(np.abs(X) ** 2, t, calls, cfg)
    X, click = declick(X, f, noise, cfg)
    P = np.abs(X) ** 2
    G = np.clip(1 - cfg.over_subtraction * noise[:, None] / (P + 1e-20), cfg.gain_floor, 1.0)
    G = uniform_filter1d(median_filter(G, size=tuple(cfg.gain_median)), cfg.gain_smooth_frames, axis=1)
    a, b = call
    fade = np.clip(np.minimum(t - a, b - t) / cfg.fade_s + 1, 0, 1)
    y = signal.istft(X * G * fade[None, :], sr, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length)[1]
    y = np.pad(y, (0, max(0, len(audio) - len(y))))[: len(audio)]
    inside = (t >= a) & (t <= b)
    info = {"noise_s": noise_s, "click_frames_in_call": int((click & inside).sum())}
    return y.astype(np.float32), info


def calls_in_window(calls: Sequence[Interval], lo: float, hi: float) -> List[Interval]:
    """The calls overlapping ``[lo, hi]``, shifted to start at ``lo``."""
    return [(a - lo, b - lo) for a, b in calls if b > lo and a < hi]
