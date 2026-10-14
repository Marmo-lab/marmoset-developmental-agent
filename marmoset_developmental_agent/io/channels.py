"""Combining the Zoom H3-VR's four ambisonic capsules into the signal we detect on.

The raw colony recordings are 48 kHz, 4-channel A-format from a single Zoom H3-VR
(capsules tagged FLU / FRD / BLD / BRU in the file's iXML). Talia's pipeline never made
a choice here: ``librosa.load`` defaults to a mono mixdown, so all four capsules were
averaged. Our own ``dataset_builder`` reproduced that by accident. Averaging capsules
that see a directional source at different delays comb-filters the signal while the
uncorrelated noise between them simply adds, so the mixdown can cost real SNR.

Measured on 54 recordings (one per folder, `tools/analysis/level_audit.py`): the best
single capsule yields more surviving spectral bins than the 4-capsule mean in 32/54
files and fewer in none, median gain 3.05x. So ``mean`` is a baseline to beat, not a
sensible default -- hence this module, and the ablation it enables.

A note on what is deliberately *absent*. The obvious fifth candidate, "reconstruct the
ambisonic W (omnidirectional) component", is not here because for A-format capsules
W = FLU + FRD + BLD + BRU, which is exactly ``mean`` up to a constant factor. It would
have been a duplicate column in the ablation table. The directional B-format components
(X/Y/Z) could in principle steer a beam toward a calling animal, but that needs to know
where the animal is, which we do not; ``max_energy`` is the cheap approximation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ChannelStrategy = Literal["mean", "max_energy", "best_snr", "per_channel"]

STRATEGIES: tuple[str, ...] = ("mean", "max_energy", "best_snr", "per_channel")


def combine(
    block: np.ndarray,
    strategy: ChannelStrategy = "mean",
    sample_rate: int = 48_000,
    band_hz: tuple[float, float] = (5_000.0, 14_000.0),
    channel: int | None = None,
) -> np.ndarray:
    """Reduce a multi-channel block to the signal(s) the detector should run on.

    Args:
        block: ``(n_samples, n_channels)`` float array, or ``(n_samples,)`` for mono.
        strategy:
            ``mean``        -- average all capsules. Talia's accidental behaviour, and
                               equal to the ambisonic W component up to scale. Baseline.
            ``max_energy``  -- the single capsule with the most energy in ``band_hz``
                               *within this block*. Tracks a moving caller, but the
                               chosen capsule can change between blocks, so boundaries
                               are not phase-continuous; keep blocks long (>= 10 s).
            ``best_snr``    -- a fixed capsule chosen once for the whole file and passed
                               in as ``channel``. Phase-continuous. Use
                               :func:`pick_best_channel` on a sample of the file first.
            ``per_channel`` -- no reduction: returns every capsule so the detector can
                               run on each and the detections be unioned. Highest
                               recall, highest cost, and needs overlap-merging downstream.
        sample_rate: of ``block``, for the band-energy measurement.
        band_hz: the frequency band marmoset calls occupy, used by ``max_energy``.
        channel: required by ``best_snr``; ignored otherwise.

    Returns:
        ``(n_out, n_samples)`` float32, where ``n_out`` is 1 for every strategy except
        ``per_channel``, which returns ``n_channels`` rows. Always 2-D so that callers
        have one shape to handle.
    """
    x = np.asarray(block, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected (n_samples, n_channels), got shape {x.shape}")
    x = x.T  # -> (n_channels, n_samples)
    n_ch = x.shape[0]

    if strategy == "mean":
        return x.mean(axis=0, keepdims=True)
    if strategy == "per_channel":
        return x
    if strategy == "best_snr":
        if channel is None:
            raise ValueError("strategy 'best_snr' requires `channel` (see pick_best_channel)")
        if not 0 <= channel < n_ch:
            raise ValueError(f"channel {channel} out of range for {n_ch} channels")
        return x[channel : channel + 1]
    if strategy == "max_energy":
        return x[int(np.argmax(band_energy(x, sample_rate, band_hz)))][None, :]
    raise ValueError(f"unknown channel strategy {strategy!r}; expected one of {STRATEGIES}")


def band_energy(
    channels: np.ndarray, sample_rate: int, band_hz: tuple[float, float] = (5_000.0, 14_000.0)
) -> np.ndarray:
    """Energy of each channel inside ``band_hz``, via a real FFT. Shape ``(n_channels,)``.

    Measured in the call band rather than broadband so that low-frequency room rumble,
    which is common to all capsules anyway, does not decide which capsule looks best.
    """
    x = np.atleast_2d(np.asarray(channels, dtype=np.float64))
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / sample_rate)
    sel = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not sel.any():  # band lies entirely above Nyquist
        return spec.sum(axis=-1)
    return spec[:, sel].sum(axis=-1)


def pick_best_channel(
    samples: np.ndarray, sample_rate: int, band_hz: tuple[float, float] = (5_000.0, 14_000.0)
) -> int:
    """Index of the capsule with the best in-band peak-to-median ratio.

    Used once per file to fix the ``best_snr`` channel. Peak-to-median rather than raw
    energy: a capsule sitting next to the ventilation has the most energy but the worst
    SNR, and what we want is the one where transient calls stand out furthest above the
    stationary background.

    Args:
        samples: ``(n_samples, n_channels)`` -- a representative excerpt, not the whole
            file (the caller should concatenate a few windows spread across it).
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        return 0
    env = np.abs(x.T)  # (n_channels, n_samples)
    peak = np.percentile(env, 99.9, axis=1)
    median = np.median(env, axis=1)
    return int(np.argmax(peak / (median + 1e-12)))
