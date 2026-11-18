"""Streaming 4-channel reader, bext parsing, and capsule-combination strategies.

Everything here runs on synthetic WAVs written to a tmp dir, so the suite stays
CPU-only and cluster-independent (CONVENTIONS.md, guardrail 3). The real-data counterpart
is tools/build_manifest.py run against AnimalRoomRecordings on the cluster.
"""

from __future__ import annotations

import datetime as dt
import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from marmoset_developmental_agent.io.channels import (
    STRATEGIES, band_energy, combine, pick_best_channel)
from marmoset_developmental_agent.io.raw_reader import (
    EXPECTED_CHANNELS, EXPECTED_SAMPLE_RATE, iter_blocks, probe, read_bext_datetime)

SR = EXPECTED_SAMPLE_RATE


def _bext_chunk(when: dt.datetime) -> bytes:
    """A minimal 602-byte bext chunk carrying OriginationDate/Time."""
    body = bytearray(602)
    body[320:330] = when.strftime("%Y-%m-%d").encode()
    body[330:338] = when.strftime("%H:%M:%S").encode()
    return b"bext" + struct.pack("<I", len(body)) + bytes(body)


def _write_wav(path: Path, data: np.ndarray, sr: int = SR, when: dt.datetime | None = None) -> Path:
    """Write a WAV, optionally splicing a bext chunk in after the header."""
    sf.write(path, data, sr, subtype="PCM_24")
    if when is None:
        return path
    raw = path.read_bytes()
    out = raw[:12] + _bext_chunk(when) + raw[12:]
    # RIFF size field must count everything after it
    out = out[:4] + struct.pack("<I", len(out) - 8) + out[8:]
    path.write_bytes(out)
    return path


def _four_channel(n: int, loud_channel: int = 2) -> np.ndarray:
    """4 capsules of noise; one also carries a 7 kHz tone burst (a "call").

    The burst occupies the middle third of the file, so the helper works for any ``n``
    rather than assuming at least two seconds.
    """
    rng = np.random.default_rng(0)
    x = 0.001 * rng.standard_normal((n, EXPECTED_CHANNELS)).astype(np.float32)
    s0, s1 = n // 3, 2 * n // 3
    t = np.arange(s1 - s0) / SR
    x[s0:s1, loud_channel] += 0.2 * np.sin(2 * np.pi * 7000 * t).astype(np.float32)
    return x


@pytest.fixture(scope="module")
def good_wav(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("raw")
    return _write_wav(d / "210503_001.WAV", _four_channel(5 * SR),
                      when=dt.datetime(2021, 5, 3, 16, 56, 37))


# --------------------------------------------------------------------------- #
# bext timestamps
# --------------------------------------------------------------------------- #
def test_bext_roundtrip(good_wav):
    assert read_bext_datetime(good_wav) == dt.datetime(2021, 5, 3, 16, 56, 37)


def test_bext_absent_returns_none(tmp_path):
    p = _write_wav(tmp_path / "nobext.WAV", _four_channel(SR))
    assert read_bext_datetime(p) is None


def test_bext_on_non_wav_returns_none(tmp_path):
    p = tmp_path / "notaudio.WAV"
    p.write_bytes(b"this is not a RIFF file at all")
    assert read_bext_datetime(p) is None


# --------------------------------------------------------------------------- #
# probe / format validation
# --------------------------------------------------------------------------- #
def test_probe_accepts_colony_format(good_wav):
    i = probe(good_wav)
    assert i.ok and i.reason == ""
    assert (i.sample_rate, i.channels) == (EXPECTED_SAMPLE_RATE, EXPECTED_CHANNELS)
    assert i.duration_s == pytest.approx(5.0, abs=0.01)
    assert i.recording_id.endswith("_210503_001")


def test_probe_rejects_the_ultrasonic_format(tmp_path):
    """250 kHz mono, no bext -- the Rec_2021_06_28_ultrasonic folder (audit Q4).

    It must be rejected, not silently processed with a 48 kHz assumption, which would
    make every derived sample index wrong by ~5.2x.
    """
    p = _write_wav(tmp_path / "M12_F14_210628103427_001.WAV",
                   np.zeros((250_000, 1), np.float32), sr=250_000)
    i = probe(p)
    assert not i.ok
    assert "sample rate" in i.reason and "channels" in i.reason and "bext" in i.reason


def test_probe_accepts_the_96khz_colony_format(tmp_path):
    """Rec_2026_11_8-10: the same recorder at 96 kHz. Accepted, and its duration comes from its own rate."""
    p = _write_wav(tmp_path / "261108_001.WAV", np.zeros((2 * 96_000, EXPECTED_CHANNELS), np.float32), sr=96_000,
                   when=dt.datetime(2026, 11, 8, 15, 30, 0))
    i = probe(p)
    assert i.ok and i.sample_rate == 96_000
    assert i.duration_s == pytest.approx(2.0, abs=0.01)


def test_96khz_and_48khz_recordings_detect_alike():
    """The production detector on one recording at 96 kHz and on its 1:2 decimation gives the same calls.

    Three synthetic frequency sweeps (0.15-1.2 s, 6.5-12 kHz) in 4-capsule noise; the default (learned) detector must
    find them, so the comparison is not between two empty lists."""
    from scipy.signal import resample_poly

    from marmoset_developmental_agent.config.agent_config import load_config
    from marmoset_developmental_agent.preprocessing.streaming import ArraySource, detect_recording

    sr = 96_000
    t = np.arange(20 * sr) / sr
    x = 0.003 * np.random.default_rng(0).standard_normal((len(t), EXPECTED_CHANNELS))
    for t0, f0, f1, d in ((4.0, 7000, 9000, 0.6), (9.0, 8000, 12000, 0.15), (14.0, 6500, 7500, 1.2)):
        m = (t >= t0) & (t < t0 + d)
        u = t[m] - t0
        x[m] += (0.1 * np.sin(np.pi * u / d) ** 0.5 * np.sin(2 * np.pi * (f0 * u + (f1 - f0) * u ** 2 / (2 * d))))[:, None]
    x96 = x.astype(np.float32)
    x48 = resample_poly(x96, 1, 2, axis=0).astype(np.float32)
    cfg = load_config().preprocessing
    when = dt.datetime(2026, 11, 9, 12, 0, 0)
    a = detect_recording(ArraySource(x96, 96_000), when, cfg)
    b = detect_recording(ArraySource(x48, 48_000), when, cfg)
    assert len(a) == len(b) == 3
    for c96, c48 in zip(a, b):
        assert abs(c96.start_s - c48.start_s) <= 0.01 and abs(c96.end_s - c48.end_s) <= 0.01
        assert abs(c96.confidence - c48.confidence) <= 0.01


def test_probe_never_raises_on_garbage(tmp_path):
    p = tmp_path / "broken.WAV"
    p.write_bytes(b"\x00" * 64)
    i = probe(p)
    assert not i.ok and "unreadable" in i.reason


# --------------------------------------------------------------------------- #
# channel strategies
# --------------------------------------------------------------------------- #
def test_combine_shapes_and_strategies():
    x = _four_channel(2 * SR, loud_channel=2)
    for s in STRATEGIES:
        kw = {"channel": 2} if s == "best_snr" else {}
        out = combine(x, s, SR, **kw)
        assert out.ndim == 2 and out.dtype == np.float32
        assert out.shape[1] == x.shape[0]
        assert out.shape[0] == (EXPECTED_CHANNELS if s == "per_channel" else 1)


def test_max_energy_picks_the_capsule_carrying_the_call():
    x = _four_channel(2 * SR, loud_channel=3)
    picked = combine(x, "max_energy", SR)[0]
    assert np.allclose(picked, x[:, 3])


def test_best_snr_and_pick_best_channel_find_the_loud_capsule():
    x = _four_channel(2 * SR, loud_channel=1)
    assert pick_best_channel(x, SR) == 1
    assert np.allclose(combine(x, "best_snr", SR, channel=1)[0], x[:, 1])


def test_single_capsule_beats_the_four_capsule_mean_in_band_energy():
    """The empirical finding this module exists for: averaging dilutes a call that is
    present on one capsule, because the other three contribute only noise."""
    x = _four_channel(2 * SR, loud_channel=0)
    e_mean = band_energy(combine(x, "mean", SR), SR)[0]
    e_best = band_energy(combine(x, "max_energy", SR), SR)[0]
    assert e_best > 4 * e_mean


def test_combine_rejects_unknown_strategy_and_missing_channel():
    x = _four_channel(SR)
    with pytest.raises(ValueError, match="unknown channel strategy"):
        combine(x, "telepathy", SR)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires `channel`"):
        combine(x, "best_snr", SR)


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #
def test_iter_blocks_covers_the_file_with_the_requested_overlap(good_wav):
    blocks = list(iter_blocks(good_wav, block_s=2.0, overlap_s=0.5, strategy="mean"))
    assert len(blocks) >= 2
    starts = [s for _, s in blocks]
    assert starts[0] == 0
    assert starts[1] - starts[0] == int(round(1.5 * SR))      # block - overlap
    assert starts[-1] + blocks[-1][0].shape[1] >= probe(good_wav).frames
    for sig, _ in blocks:
        assert sig.ndim == 2 and sig.shape[0] == 1


def test_iter_blocks_per_channel_yields_every_capsule(good_wav):
    sig, start = next(iter(iter_blocks(good_wav, block_s=2.0, strategy="per_channel")))
    assert sig.shape[0] == EXPECTED_CHANNELS and start == 0


def test_iter_blocks_bounded_memory(good_wav):
    """Each block must be the requested size, never the whole file."""
    for sig, _ in iter_blocks(good_wav, block_s=1.0, overlap_s=0.25):
        assert sig.shape[1] <= int(round(1.0 * SR))


def test_iter_blocks_rejects_overlap_ge_block(good_wav):
    with pytest.raises(ValueError, match="must exceed"):
        next(iter(iter_blocks(good_wav, block_s=1.0, overlap_s=1.0)))


def test_iter_blocks_best_snr_autoselects_when_channel_not_given(good_wav):
    sig, _ = next(iter(iter_blocks(good_wav, block_s=2.0, strategy="best_snr")))
    assert sig.shape[0] == 1


# --------------------------------------------------------------------------- #
# STFT scaling (regression: our port must speak librosa's units)
# --------------------------------------------------------------------------- #
def test_stft_scaling_matches_librosa_not_scipy():
    """Talia's AMP_THRESHOLD = 0.02 is an absolute magnitude against librosa.stft.

    scipy.signal.stft divides by the window sum (1024 for a 2048-point Hann) and
    librosa does not, so a port that leaves scipy's normalisation in place is ~1024x
    stricter than hers and detects almost nothing. This pins the convention.
    """
    import librosa

    from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
    from marmoset_developmental_agent.preprocessing.filter_5khz import _stft

    cfg = PreprocessingConfig.talia()
    sr = cfg.detection_sample_rate
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 7_000 * t)).astype(np.float32)

    _, _, ours = _stft(x, sr, cfg)
    theirs = np.abs(librosa.stft(x, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
    assert ours.max() == pytest.approx(theirs.max(), rel=0.02)
    assert ours.max() > 100 * cfg.amp_threshold  # a loud tone must clear it comfortably


def test_ambisonic_beam_has_unit_on_axis_gain_and_beats_capsules_and_mean():
    """A horizontal plane wave seen by four ideal tetrahedral cardioids (FLU/FRD/BLD/BRU):
    the steered beam recovers unit gain; the best capsule is off-axis by ~35 deg of
    elevation (gain ~0.79); the 4-capsule mean keeps only the omni half (0.5)."""
    from marmoset_developmental_agent.io.channels import ambisonic_beams

    d = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    for az in (0, 45, 90, 180, 270):
        u = np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), 0.0])
        caps = np.array([0.5 * (1 + u @ di) for di in d])[:, None] * np.ones((1, 8))
        best_beam = ambisonic_beams(caps)[:, 0].max()
        assert best_beam == pytest.approx(1.0, abs=1e-6)
        assert best_beam > caps[:, 0].max() > caps[:, 0].mean() == pytest.approx(0.5)


def test_ambisonic_beam_requires_four_capsules():
    with pytest.raises(ValueError, match="4 A-format"):
        combine(np.zeros((100, 2), np.float32), "ambisonic_beam", SR)
