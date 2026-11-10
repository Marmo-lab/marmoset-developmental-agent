"""tools/synth/plant_noise.py: noises added at their natural level, placed apart from each other, in the heard shares."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import tools.synth.plant_noise as pn

SR = pn.SR


def _tsv(path: Path, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def _fixture(tmp: Path) -> tuple:
    rng = np.random.default_rng(0)
    noises, src = tmp / "noises", tmp / "src"
    for d in (noises / "audio", src / "audio"):
        d.mkdir(parents=True)
    rows = []
    for k, (t, dur) in enumerate([("cage", 0.4), ("cage", 1.5), ("human", 0.8)]):
        sf.write(noises / "audio" / f"n{k}.flac", 0.05 * rng.standard_normal(int(dur * SR)), SR, subtype="PCM_24")
        rows.append({"noise_id": f"n{k}", "type": t, "duration_s": dur, "recording_id": "r", "start_s": 0, "share": 0})
    _tsv(noises / "noises.tsv", rows)
    (noises / "shares.json").write_text(json.dumps({"all_halves": {"cage": 0.9, "human": 0.05, "beep": 0.05}}))
    recs, events = [], []
    for k in range(4):
        y = 0.001 * rng.standard_normal(60 * SR)
        for on in (5.0, 20.0, 40.0):
            events.append({"rec_id": f"x{k}", "kind": "call", "donor_id": "d", "call_type": "phee_3", "onset_s": on,
                           "offset_s": on + 1, "snr": "1.5", "gain": 0.5})
        sf.write(src / "audio" / f"x{k}.flac", 0.5 * y, SR, subtype="PCM_24")
        recs.append({"rec_id": f"x{k}", "snr": "1.5", "snr_db": 3.52, "background": "B", "background_kind": "night",
                     "half": "test", "background_start_s": 0, "n_calls": 3, "scale": 0.5, "seed": "0"})
    recs.append({**recs[0], "rec_id": "y0", "snr": "10.0"})                  # another level: left out
    _tsv(src / "recordings.tsv", recs)
    _tsv(src / "events.tsv", events)
    return noises, src


def test_make_adds_noises_and_keeps_the_answer(tmp_path: Path) -> None:
    noises, src = _fixture(tmp_path)
    out = tmp_path / "out"
    pn.make(noises, src, out, levels=[1.5], seed=0)
    recs = list(csv.DictReader(open(out / "recordings.tsv"), delimiter="\t"))
    assert [r["rec_id"] for r in recs] == ["x0", "x1", "x2", "x3"]
    src_ev = list(csv.DictReader(open(src / "events.tsv"), delimiter="\t"))
    new_ev = list(csv.DictReader(open(out / "events.tsv"), delimiter="\t"))
    assert [(e["rec_id"], e["onset_s"]) for e in new_ev] == [(e["rec_id"], e["onset_s"]) for e in src_ev]
    nz = list(csv.DictReader(open(out / "noises.tsv"), delimiter="\t"))
    assert len(nz) == sum(int(r["n_noises"]) for r in recs) > 0
    by_rec: dict = {}
    for z in nz:
        by_rec.setdefault(z["rec_id"], []).append((float(z["onset_s"]), float(z["offset_s"])))
    for spans in by_rec.values():                       # never two noises fused, never at an edge
        spans.sort()
        assert spans[0][0] >= 0.5 - 1e-3 and spans[-1][1] <= 59.5 + 1e-3
        assert all(b[0] - a[1] >= 0.5 - 1e-3 for a, b in zip(spans, spans[1:]))
    # natural level: what was added is the clip times the source's scale (0.5). Onsets are stored to
    # 0.1 ms (5 samples), so the exact sample is found within that tolerance.
    src_y, _ = sf.read(src / "audio" / "x0.flac")
    new_y, _ = sf.read(out / "audio" / "x0.flac")
    for z in (z for z in nz if z["rec_id"] == "x0"):
        clip, _ = sf.read(noises / "audio" / f"{z['noise_id']}.flac")
        ref = 0.5 * pn.faded(clip)[len(clip) // 4:len(clip) // 2]
        on = float(z["onset_s"])
        best = []
        for i in range(int(on * SR) - 5, int(on * SR) + 6):
            d = (new_y - src_y)[i + len(clip) // 4:i + len(clip) // 2]
            best.append((np.corrcoef(d, ref)[0, 1], np.std(d) / np.std(ref)))
        r, ratio = max(best)
        assert r > 0.99 and abs(ratio - 1) < 0.05


def test_draw_follows_the_heard_shares_and_drops_missing_types(tmp_path: Path) -> None:
    noises, _ = _fixture(tmp_path)
    by_type, shares = pn.load_noises(noises)
    assert set(shares) == {"cage", "human"} and abs(shares["cage"] - 0.9 / 0.95) < 1e-9     # no beep clip: dropped
    picks = pn.draw(20_000, by_type, shares, np.random.default_rng(1))
    assert abs(np.mean([t == "cage" for t, *_ in picks]) - shares["cage"]) < 0.01


def test_clipping_guard_scales_the_answer_with_the_audio(tmp_path: Path) -> None:
    noises, src = _fixture(tmp_path)
    sf.write(src / "audio" / "x1.flac", 0.98 * np.ones(60 * SR), SR, subtype="PCM_24")   # any noise clips it
    out = tmp_path / "out"
    pn.make(noises, src, out, levels=[1.5], seed=0)
    r = next(r for r in csv.DictReader(open(out / "recordings.tsv"), delimiter="\t") if r["rec_id"] == "x1")
    y, _ = sf.read(out / "audio" / "x1.flac")
    s = float(r["scale"]) / 0.5
    assert int(r["n_noises"]) > 0 and s < 1 and np.abs(y).max() <= 0.99 + 1e-4
    gains = [float(e["gain"]) for e in csv.DictReader(open(out / "events.tsv"), delimiter="\t") if e["rec_id"] == "x1"]
    assert all(abs(g - 0.5 * s) < 1e-5 for g in gains)
