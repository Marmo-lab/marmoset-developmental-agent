"""tools/analysis/recording_qc.py: raw 24-bit reading (including an unfinalised file) and window stats."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import soundfile as sf

import tools.analysis.recording_qc as qc

SR = 96_000


def _signal(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 8_000 * t)
    return np.stack([tone, 0.5 * tone, 0.1 * tone, tone], axis=1).astype(np.float32)


def test_reads_what_soundfile_wrote(tmp_path: Path) -> None:
    x = _signal(1.0)
    sf.write(tmp_path / "a.WAV", x, SR, subtype="PCM_24", format="WAV")
    raw, sr, ch = qc.pcm_view(tmp_path / "a.WAV")
    assert (sr, ch, len(raw)) == (SR, 4, len(x))
    np.testing.assert_allclose(qc.decode(np.asarray(raw), ch), x, atol=1e-6)


def test_unfinalised_file_is_read_from_its_bytes(tmp_path: Path) -> None:
    x = _signal(0.5)
    p = tmp_path / "b.WAV"
    sf.write(p, x, SR, subtype="PCM_24", format="WAV")
    b = bytearray(p.read_bytes())
    k = b.find(b"data")
    b[4:8] = struct.pack("<I", 0)                       # the recorder never wrote the sizes
    b[k + 4:k + 8] = struct.pack("<I", 0)
    p.write_bytes(bytes(b))
    raw, _, ch = qc.pcm_view(p)
    assert len(raw) == len(x)
    np.testing.assert_allclose(qc.decode(np.asarray(raw), ch), x, atol=1e-6)


def test_window_stats_put_a_tone_in_its_band() -> None:
    s = qc.window_stats(_signal(1.0), SR)
    assert s["band2"] > s["band0"] + 40 and s["band2"] > s["band4"] + 40      # 8 kHz is in 5-12 kHz
    assert abs(s["rms0"] - s["rms1"] - 20 * np.log10(2)) < 0.1
    assert s["clip_frac"] == 0.0
