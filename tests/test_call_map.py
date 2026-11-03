"""tools/analysis/call_map.py: call images and the map on three synthetic call types."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.analysis.call_map import N_BANDS, N_STEPS, build, call_image

SR = 48_000


def _tone(f0: float, f1: float, dur: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * dur))) * np.hanning(len(t))
    return (x * rng.uniform(0.05, 0.5) + 0.002 * rng.standard_normal(len(t))).astype(np.float32)


def test_call_image_is_fixed_size_and_independent_of_gain() -> None:
    rng = np.random.default_rng(0)
    x = _tone(7_000, 7_000, 0.3, rng)
    a, b = call_image(x, SR), call_image(10 * x, SR)
    assert a.shape == (N_BANDS, N_STEPS) == call_image(_tone(7_000, 9_000, 2.5, rng), SR).shape
    assert a.min() >= 0 and a.max() == 1
    assert np.allclose(a, b, atol=1e-4)
    assert call_image(np.zeros(100, np.float32), SR).shape == (N_BANDS, N_STEPS)


def test_three_call_types_form_three_pure_clusters(tmp_path: Path) -> None:
    """Short flat 6 kHz, long flat 9 kHz, and rising 5 -> 12 kHz calls, 60 of each."""
    rng = np.random.default_rng(1)
    kinds = {"flat6": (6_000, 6_000, 0.2), "flat9": (9_000, 9_000, 1.5), "rise": (5_000, 12_000, 0.6)}
    audio, rows, truth = [], [], {}
    pos = 0.0
    for k in range(60):
        for name, (f0, f1, dur) in kinds.items():
            d = dur * rng.uniform(0.8, 1.2)
            x = _tone(f0, f1, d, rng)
            gap = np.zeros(int(0.1 * SR), np.float32)
            cid = f"r_{len(rows):05d}"
            rows.append({"call_id": cid, "start_time": f"2021-05-{1 + k % 3:02d}T10:00:00", "duration_s": f"{len(x) / SR}",
                         "audio_path": "call_audio/r.flac", "audio_offset_s": f"{pos}", "audio_lead_s": "0.1",
                         "peak_freq_hz": f"{f0}", "snr_db": "30"})
            truth[cid] = name
            audio += [gap, x]
            pos += (len(gap) + len(x)) / SR
    (tmp_path / "call_audio").mkdir()
    sf.write(tmp_path / "call_audio" / "r.flac", np.concatenate(audio + [np.zeros(SR, np.float32)]), SR)
    with open(tmp_path / "calls_train.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    out = build(tmp_path, 1000, 0, "orig", perplexity=15, min_cluster=20)
    assert json.loads((tmp_path / "call_map" / "map.json").read_text())["meta"]["n"] == len(rows) == out["meta"]["n"]
    by_cluster = {}
    for p in out["points"]:
        by_cluster.setdefault(p[3], Counter())[truth[p[0]]] += 1
    clustered = {c: n for c, n in by_cluster.items() if c >= 0}
    assert len(clustered) == 3
    assert all(n.most_common(1)[0][1] / sum(n.values()) > 0.95 for n in clustered.values())
    assert {n.most_common(1)[0][0] for n in clustered.values()} == set(kinds)


def test_compare_redraws_the_maps_sample(tmp_path: Path) -> None:
    test_three_call_types_form_three_pure_clusters(tmp_path)
    from tools.analysis.call_map import compare
    assert compare(tmp_path, isomap_n=100).stat().st_size > 10_000
    assert (tmp_path / "call_map" / "images_orig.npy").exists()
