"""tools/analysis/find_short_high.py: the energy rule finds short high sweeps and leaves normal calls and clicks."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

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


def _tiny_run(tmp_path: Path) -> Tuple[Path, Path]:
    """Four 3-minute day recordings (4 capsules), each with one run call and one missed candidate."""
    import csv
    import json

    import soundfile as sf
    run, out = tmp_path / "run", tmp_path / "out"
    for d in ("files", "calls", "audio"):
        (run / d).mkdir(parents=True)
    (out / "scan").mkdir(parents=True)
    rng = np.random.default_rng(3)
    for k in range(4):
        rid = f"rec{k}"
        sf.write(run / "audio" / f"{rid}.wav", 0.01 * rng.standard_normal((3 * 60 * SR, 4)).astype(np.float32), SR)
        (run / "files" / f"{rid}.json").write_text(json.dumps({"recording_id": rid, "path": str(run / "audio" / f"{rid}.wav"),
                                                               "start_time": "2021-05-10T10:00:00", "duration_s": 3600}))
        (run / "calls" / f"{rid}.csv").write_text("start_s,end_s\n70.0,71.0\n")
        with open(out / "scan" / f"{rid}.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fsh.FIELDS, delimiter="\t")
            w.writeheader()
            w.writerow({"cand_id": f"{rid}_0000", "recording_id": rid, "start_s": 75.0, "duration_s": 0.1, "peak_hz": 15000,
                        "high_db": 20, "lead_db": 10, "in_run_call": 0})
    return run, out


def test_minutes_cut_the_train_half_with_positives(tmp_path: Path) -> None:
    import csv

    import soundfile as sf
    run, out = _tiny_run(tmp_path)
    train = [r["recording_id"] for r in fsh.recording_half(run, "train")]
    test = [r["recording_id"] for r in fsh.recording_half(run, "test")]
    assert len(train) == 2 and not set(train) & set(test)
    assert fsh.minutes(run, out) == 2
    pos = list(csv.DictReader(open(out / "minutes" / "positives.tsv"), delimiter="\t"))
    assert {p["minute_id"] for p in pos} == {f"{r}_m01" for r in train}
    assert sorted((p["source"], p["start_s"], p["end_s"]) for p in pos if p["minute_id"] == f"{train[0]}_m01") == [
        ("candidate", "15.0", "15.1"), ("run_call", "10.0", "11.0")]
    assert sf.info(out / "minutes" / f"{train[0]}_m01.flac").duration == 60


def test_realeval_runs_a_model_on_the_test_half_and_nights(tmp_path: Path) -> None:
    import soundfile as sf
    run, out = _tiny_run(tmp_path)
    nights = tmp_path / "nights"
    nights.mkdir()
    sf.write(nights / "n0.flac", np.zeros(60 * SR, np.float32), SR)
    fsh.realeval(run, out, "models/call_detector_v1.pt", "v1", nights, 0, 1)
    files = sorted(p.stem for p in (out / "realeval_v1").glob("*.tsv"))
    assert files == sorted([r["recording_id"] for r in fsh.recording_half(run, "test")] + ["n0"])
    r = fsh.realscore(run, out, "v1")
    assert r["candidates"] == 2 and r["missed_by_v4"] == 2 and r["candidates_caught"] == 0.0
    assert abs(r["day_hours"] - 0.1) < 0.01 and r["detections_per_night_hour"] == 0.0


def test_library_serves_a_generic_review(tmp_path: Path) -> None:
    import json

    import pytest
    import soundfile as sf

    import tools.corpus.call_library as cl
    (tmp_path / "clips").mkdir()
    sf.write(tmp_path / "clips" / "i0.flac", np.zeros(SR, np.float32), SR)
    (tmp_path / "review.json").write_text(json.dumps({"title": "t", "intro": "i", "options": [["a", "A"]]}))
    (tmp_path / "items.tsv").write_text('item_id\tmarks\ni0\t[[0.1, 0.5, "label", "x"], [0.2, 0.9, "b", "v2"]]\n')
    rv = cl.Review(tmp_path)
    cl.REVIEW[:] = [rv]
    assert rv.audio("i0")[0].shape == (SR,) and cl._review_png("i0")[:4] == b"\x89PNG"
    with pytest.raises(KeyError):
        rv.audio("../items")
    rv.add_feedback({"item_id": "i0", "verdict": "a"})
    assert rv.feedback()["i0"]["verdict"] == "a"


def test_merge_cases_are_new_joins_on_day_clips() -> None:
    from tools.analysis.merge_review import cases
    labels = {"c": [(1.0, 1.5), (1.8, 2.2), (5.0, 5.5), (5.7, 6.0)], "n": [(1.0, 1.5), (1.8, 2.2)]}
    a = {"c": [(0.9, 1.6), (1.7, 2.3), (4.9, 6.1)]}                 # A already joins the second pair
    b = {"c": [(0.9, 2.3), (4.9, 6.1)], "n": [(0.9, 2.3)]}
    found = cases(a, b, labels, {"c": 1, "n": 0})
    assert [(f["clip_id"], f["det"]) for f in found] == [("c", (0.9, 2.3))]
    assert found[0]["gaps"] == [(1.5, 1.8)]
