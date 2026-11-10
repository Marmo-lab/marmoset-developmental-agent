"""tools/analysis/find_noise_events.py: loud events away from run calls, suspect detections, a varied selection."""

from __future__ import annotations

import numpy as np

import tools.analysis.find_noise_events as fne

SR = fne.SR


def _recording() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = 0.001 * rng.standard_normal(150 * SR)
    t = np.arange(int(0.5 * SR)) / SR
    x[20 * SR:20 * SR + len(t)] += 0.05 * np.sin(2 * np.pi * 7_000 * t)          # a "call" the run found
    x[40 * SR:40 * SR + int(0.02 * SR)] += 0.2 * rng.standard_normal(int(0.02 * SR))  # a click: wide
    x[61 * SR:61 * SR + int(0.3 * SR)] += 0.05 * np.sin(2 * np.pi * 3_000 * np.arange(int(0.3 * SR)) / SR)  # a beep
    return x


def test_finds_noise_and_skips_run_calls() -> None:
    x = _recording()
    loud = fne.find(lambda s, n: x[s:s + n], len(x))
    rec = {"recording_id": "r", "half": "test", "start_time": "2021-05-03T10:00:00"}
    calls = [(20.0, 20.5, 0.9), (100.0, 110.0, 0.9), (130.0, 130.4, 0.3)]
    rows = fne.candidates(rec, loud, calls)
    lo = [r for r in rows if r["pool"] == "loud"]
    starts = sorted(round(r["start_s"]) for r in lo)
    assert 40 in starts and 61 in starts and 20 not in starts
    click = next(r for r in lo if round(r["start_s"]) == 40)
    beep = next(r for r in lo if round(r["start_s"]) == 61)
    assert click["width_khz"] > 5 * beep["width_khz"] and abs(beep["peak_hz"] - 3_000) < 500
    sus = sorted((r["start_s"], r["confidence"]) for r in rows if r["pool"] == "suspect")
    assert sus == [(100.0, 0.9), (130.0, 0.3)]                      # the 10 s detection and the unsure one
    assert all(r["hour"] == 10 and r["half"] == "test" for r in rows)


def test_choice_spreads_over_cells_and_recordings(monkeypatch) -> None:
    monkeypatch.setattr(fne, "N_SUSPECT", 2)
    rows = [{"pool": "loud", "recording_id": f"r{k % 3}", "peak_hz": str(2_000 if k < 20 else 15_000),
             "duration_s": "0.1", "level_db": str(30 - k % 20)} for k in range(40)]
    rows += [{"pool": "suspect", "recording_id": "r9", "duration_s": str(d), "confidence": str(c)}
             for d, c in [(12, 0.9), (0.2, 0.1), (0.3, 0.5)]]
    got = fne.choose(rows, 8)
    loud = [r for r in got if r["pool"] == "loud"]
    assert len(loud) == 6 and {r["peak_hz"] for r in loud} == {"2000", "15000"}
    assert max(sum(r["recording_id"] == f"r{k}" for r in loud) for k in range(3)) <= fne.PER_RECORDING
    assert [r["duration_s"] for r in got if r["pool"] == "suspect"] == ["12", "0.2"]
