"""tools/synth/generate.py: placement, SNR, excerpts, and a small set written end to end."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import tools.synth.generate as g

SR = g.SR


def _donors(rng: np.random.Generator) -> tuple[dict, dict]:
    by_type, audio = {}, {}
    for t, (f, d) in {"phee": (7_000, 1.5), "tsik": (12_000, 0.1), "trill": (9_000, 0.3)}.items():
        for k in range(3):
            did = f"{t}_{k:03d}"
            tt = np.arange(int(d * SR)) / SR
            audio[did] = (0.2 * np.sin(2 * np.pi * f * tt) + 0.001 * rng.standard_normal(len(tt))).astype(np.float32)
            by_type.setdefault(t, []).append({"donor_id": did, "call_type": t})
    return by_type, audio


def test_place_keeps_gaps_and_edges() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        durs = list(rng.uniform(0.1, 4, rng.integers(1, 15)))
        on = g.place(durs, rng)
        if on is None:
            assert g.LENGTH_S - 2 * g.EDGE_S - sum(durs) - g.GAP_S * (len(durs) - 1) < 0
            continue
        assert on[0] >= g.EDGE_S - 1e-9 and on[-1] + durs[-1] <= g.LENGTH_S - g.EDGE_S + 1e-9
        assert all(on[k + 1] - (on[k] + durs[k]) >= g.GAP_S - 1e-9 for k in range(len(durs) - 1))
    assert g.place([30.0, 30.0], rng) is None


def test_every_call_meets_its_snr_and_types_are_uniform() -> None:
    rng = np.random.default_rng(1)
    donors, audio = _donors(rng)
    bg = g.ArrayBackground(0.01 * rng.standard_normal(90 * SR))
    types = Counter()
    for k in range(8):
        y, events, start = g.make_recording(f"r{k}", 1.5, donors, audio, bg, [(0, bg.frames)], rng)
        b = bg.read(start, len(y))
        assert len(y) == g.LENGTH_S * SR and 0 <= start <= bg.frames - len(y)
        b_bp, call_bp = g.bandpass(b), g.bandpass(y - b)
        for e in events:
            i, j = int(round(e.onset_s * SR)), int(round(e.offset_s * SR))
            assert abs(g.rms(call_bp[i:j]) / g.rms(b_bp[i:j]) - 1.5) < 0.02
            types[e.call_type] += 1
        assert all(b2.onset_s - a.offset_s >= g.GAP_S - 1e-3 for a, b2 in zip(events, events[1:]))
    n = sum(types.values())
    assert 8 * 21 * 0.7 < n < 8 * 21 * 1.3
    assert all(abs(c / n - 1 / 3) < 0.1 for c in types.values())


def test_short_spans_are_joined_to_a_minute() -> None:
    rng = np.random.default_rng(2)
    bg = g.ArrayBackground(np.ones(30 * SR))
    x, start = g.excerpt(bg, [(0, 12 * SR), (15 * SR, 27 * SR)], rng)
    assert len(x) == g.LENGTH_S * SR and start in (0, 15 * SR)
    assert np.allclose(x, 1.0)                    # crossfades of a constant leave it constant


def test_make_writes_audio_and_the_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(3)
    donors, audio = _donors(rng)
    (tmp_path / "donors" / "audio").mkdir(parents=True)
    rows = []
    for t, ds in donors.items():
        for d in ds:
            clip = np.concatenate([np.zeros(SR // 10, np.float32), audio[d["donor_id"]], np.zeros(SR // 10, np.float32)])
            sf.write(tmp_path / "donors" / "audio" / f"{d['donor_id']}.flac", clip, SR)
            rows.append({"donor_id": d["donor_id"], "call_type": t, "onset_s": 0.1, "offset_s": 0.1 + len(audio[d["donor_id"]]) / SR})
    with open(tmp_path / "donors" / "donors.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    bgdir = tmp_path / "bg"
    (bgdir / "audio").mkdir(parents=True)
    sf.write(bgdir / "audio" / "n1.flac", (0.01 * rng.standard_normal(70 * SR)).astype(np.float32), SR, subtype="PCM_24")
    g.make(tmp_path / "donors", bgdir, tmp_path / "out", n=2, levels=[1.5, 1.0], seed=0)
    recs = list(csv.DictReader(open(tmp_path / "out" / "recordings.tsv"), delimiter="\t"))
    events = list(csv.DictReader(open(tmp_path / "out" / "events.tsv"), delimiter="\t"))
    assert [r["rec_id"] for r in recs] == ["snr1.5_0000", "snr1.5_0001", "snr1_0000", "snr1_0001"]
    assert sum(int(r["n_calls"]) for r in recs) == len(events)
    assert sf.info(tmp_path / "out" / "audio" / "snr1_0001.flac").duration == g.LENGTH_S


def test_approved_reads_the_last_verdict(tmp_path: Path) -> None:
    (tmp_path / "candidates.json").write_text(json.dumps({"candidates": [{"recording_id": "a"}, {"recording_id": "b"}]}))
    (tmp_path / "bg_feedback.jsonl").write_text("\n".join(json.dumps(d) for d in (
        {"recording_id": "a", "verdict": "approve"}, {"recording_id": "b", "verdict": "approve"},
        {"recording_id": "b", "verdict": "reject"})))
    assert [c["recording_id"] for c in g.approved(tmp_path)] == ["a"]
