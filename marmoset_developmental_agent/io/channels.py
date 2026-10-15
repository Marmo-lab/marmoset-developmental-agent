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

Why there is no "ambisonic W" strategy: for A-format capsules W = FLU + FRD + BLD + BRU,
which is exactly ``mean`` up to a constant, so it would duplicate the baseline. The fifth
strategy, ``ambisonic_beam``, uses the *directional* B-format components instead.

**Gain is a confound when comparing strategies.** For a plane wave arriving on a
capsule's axis, that cardioid capsule has gain 1, but the mean of the four has gain 1/2
(only their shared omnidirectional half adds coherently) -- 6 dB lower before any comb
filtering. A detector with an *absolute* threshold therefore favours single-capsule
strategies partly through gain alone. Compare strategies with gain-invariant measures
(peak-to-floor dynamic range, a level-relative detector); see
``tools/analysis/channel_eval.py``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

ChannelStrategy = Literal["mean", "max_energy", "best_snr", "per_channel", "ambisonic_beam"]

STRATEGIES: tuple[str, ...] = ("mean", "max_energy", "best_snr", "per_channel", "ambisonic_beam")

# Horizontal look directions for ambisonic_beam (8 x 45 degrees). The colony room is a
# floor-level space, so steering in azimuth only is the useful freedom.
BEAM_AZIMUTHS_DEG: tuple[float, ...] = tuple(range(0, 360, 45))


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
            ``ambisonic_beam`` -- convert to B-format and pick, per block, the loudest
                               of 8 virtual horizontal cardioids (:func:`ambisonic_beams`).
                               Unlike ``max_energy`` it can look *between* capsules and
                               is not tilted +-35 degrees in elevation.
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
    if strategy == "ambisonic_beam":
        if n_ch != 4:
            raise ValueError(f"ambisonic_beam needs 4 A-format capsules, got {n_ch}")
        beams = ambisonic_beams(x)
        return beams[int(np.argmax(band_energy(beams, sample_rate, band_hz)))][None, :]
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


def ambisonic_beams(capsules: np.ndarray, azimuths_deg: tuple[float, ...] = BEAM_AZIMUTHS_DEG) -> np.ndarray:
    """Virtual horizontal cardioids from the Zoom H3-VR's A-format capsules.

    Assumes capsule order FLU, FRD, BLD, BRU (channels 0-3), as tagged ``zTRK1-4`` in the
    raw files' iXML chunk (audit S4.1). Treating each capsule as an ideal cardioid
    0.5 (p + v . d_i) pointing along a tetrahedral direction d_i = (+-1, +-1, +-1)/sqrt(3),
    the sums

        W = FLU+FRD+BLD+BRU = 2p
        X = FLU+FRD-BLD-BRU = (2/sqrt3) v_x       (front-back)
        Y = FLU-FRD+BLD-BRU = (2/sqrt3) v_y       (left-right)

    recover pressure p and the horizontal velocity components, and a cardioid steered to
    azimuth theta is 0.5 (p + v_x cos theta + v_y sin theta), unit gain on axis -- the same
    on-axis gain as a single capsule, so beam and capsule strategies compare fairly.

    Args:
        capsules: ``(4, n_samples)``.
    Returns:
        ``(len(azimuths_deg), n_samples)`` float32.
    """
    flu, frd, bld, bru = np.asarray(capsules, dtype=np.float64)
    p = 0.5 * (flu + frd + bld + bru)
    k = math.sqrt(3.0) / 2.0
    vx, vy = k * (flu + frd - bld - bru), k * (flu - frd + bld - bru)
    th = np.deg2rad(np.asarray(azimuths_deg, dtype=np.float64))[:, None]
    return (0.5 * (p[None, :] + vx[None, :] * np.cos(th) + vy[None, :] * np.sin(th))).astype(np.float32)
