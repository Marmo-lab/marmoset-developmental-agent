"""tools/synth/donor_calls.py: call bounds in padded clips, and the library's /donors data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from tools.synth.donor_calls import SR, call_bounds, prepare, to_48k


def _clip(parts: list[tuple[float, float, float]], total: float, seed: int = 0) -> np.ndarray:
    """``total`` s of faint noise with tones (start_s, dur_s, freq_hz) at 48 kHz."""
    rng = np.random.default_rng(seed)
    x = 0.001 * rng.standard_normal(int(total * SR))
    for start, dur, freq in parts:
        t = np.arange(int(dur * SR)) / SR
        i = int(start * SR)
        x[i:i + len(t)] += 0.3 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t)) ** 0.2
    return x.astype(np.float32)


def test_bounds_of_a_padded_call() -> None:
    onset, offset, rise, peak = call_bounds(_clip([(0.6, 0.15, 7_000)], 1.2))
    assert abs(onset - 0.6) < 0.03 and abs(offset - 0.75) < 0.03
    assert rise > 30 and abs(peak - 7_000) < 100


def test_bounds_keep_the_gaps_between_syllables() -> None:
    onset, offset, _, _ = call_bounds(_clip([(0.5, 0.06, 9_000), (0.65, 0.06, 9_500), (0.8, 0.06, 10_000)], 1.5))
    assert abs(onset - 0.5) < 0.03 and abs(offset - 0.86) < 0.03


def test_resample_44k1_keeps_length_and_tone() -> None:
    t = np.arange(44_100) / 44_100
    y = to_48k(np.sin(2 * np.pi * 7_000 * t).astype(np.float32), 44_100)
    assert len(y) == SR
    assert abs(np.fft.rfftfreq(len(y), 1 / SR)[np.abs(np.fft.rfft(y)).argmax()] - 7_000) < 2


def test_prepare_and_the_library_serves_it(tmp_path: Path) -> None:
    from tools.corpus.call_library import Donors
    for k, typ in enumerate(["phee", "tsik"]):
        (tmp_path / "src" / typ).mkdir(parents=True)
        for j in range(2):
            x = _clip([(0.4, 1.0 if typ == "phee" else 0.1, 7_000 + 2_000 * k)], 2.0, seed=j)
            sf.write(tmp_path / "src" / typ / f"c{j}.wav", x, SR)
    donors = prepare(tmp_path / "src", tmp_path / "out")
    assert [d.donor_id for d in donors] == ["phee_000", "phee_001", "tsik_000", "tsik_001"]
    lib = Donors(tmp_path / "out")
    call, sr = lib.audio("tsik_000", "call")
    assert sr == SR and abs(len(call) / SR - lib.by_id["tsik_000"]["duration_s"]) < 0.01
    lib.add_feedback({"donor_id": "tsik_000", "verdict": "good", "note": ""})
    lib.add_feedback({"donor_id": "phee_001", "verdict": "edges", "note": "too long"})
    lib.add_feedback({"donor_id": "tsik_000", "verdict": None, "note": ""})
    assert set(lib.feedback()) == {"phee_001"}
