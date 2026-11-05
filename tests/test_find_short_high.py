"""tools/analysis/find_short_high.py: the energy rule finds short high sweeps and leaves normal calls and clicks."""

from __future__ import annotations

import numpy as np

import tools.analysis.find_short_high as fsh

SR = fsh.SR


def _bed(rng: np.random.Generator, seconds: float = 20.0) -> np.ndarray:
    return 0.01 * rng.standard_normal(int(seconds * SR))


def _found(y: np.ndarray, a: float, b: float) -> bool:
    return any(fsh._ov(a, b, c["start_s"], c["start_s"] + c["duration_s"]) for c in fsh.find(lambda s, n: y[s:s + n], len(y)))


def test_a_short_high_sweep_is_found() -> None:
    rng = np.random.default_rng(0)
    y = _bed(rng)
    t = np.arange(int(0.12 * SR)) / SR
    y[5 * SR:5 * SR + len(t)] += 0.1 * np.sin(2 * np.pi * (16_000 * t - 15_000 * t ** 2))   # 16 -> 12.4 kHz
    c = [c for c in fsh.find(lambda s, n: y[s:s + n], len(y))]
    assert len(c) == 1 and abs(c[0]["start_s"] - 5.0) < 0.03 and 12_000 < c[0]["peak_hz"] < 16_500


def test_a_harmonic_call_and_a_click_are_not() -> None:
    rng = np.random.default_rng(1)
    y = _bed(rng)
    t = np.arange(int(0.8 * SR)) / SR
    y[3 * SR:3 * SR + len(t)] += sum(0.1 / k * np.sin(2 * np.pi * 7_000 * k * t) for k in (1, 2, 3))
    y[10 * SR:10 * SR + 200] += rng.standard_normal(200)                                   # broadband click
    assert not _found(y, 3.0, 3.8) and not _found(y, 10.0, 10.01)


def test_a_call_across_a_block_edge_is_kept_once() -> None:
    rng = np.random.default_rng(2)
    y = _bed(rng, 2 * fsh.BLOCK_S + 5)
    t = np.arange(int(0.1 * SR)) / SR
    i = fsh.BLOCK_S * SR - 1000
    y[i:i + len(t)] += 0.1 * np.sin(2 * np.pi * 14_000 * t)
    c = fsh.find(lambda s, n: y[s:s + n], len(y))
    assert len(c) == 1


def test_library_serves_the_candidates(tmp_path) -> None:
    import csv

    import pytest
    import soundfile as sf

    import tools.corpus.call_library as cl
    (tmp_path / "clips").mkdir()
    sf.write(tmp_path / "clips" / "r_0000.flac", np.zeros(2 * SR, np.float32), SR)
    with open(tmp_path / "candidates.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fsh.FIELDS + ["clip_offset_s"], delimiter="\t")
        w.writeheader()
        w.writerow({"cand_id": "r_0000", "recording_id": "r", "start_s": 10, "duration_s": 0.1, "peak_hz": 13000,
                    "high_db": 20, "lead_db": 10, "in_run_call": 0, "clip_offset_s": 0.95})
    sh = cl.ShortHigh(tmp_path)
    cl.SHORT_HIGH[:] = [sh]
    assert sh.audio("r_0000")[0].shape == (2 * SR,) and cl._short_high_png("r_0000")[:4] == b"\x89PNG"
    with pytest.raises(KeyError):
        sh.audio("../candidates")
    sh.add_feedback({"cand_id": "r_0000", "verdict": "noise"})
    assert sh.feedback()["r_0000"]["verdict"] == "noise"
