"""tools/synth/backgrounds.py: the pool, the scan, and the selection of call-free nights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import tools.synth.backgrounds as bg
from marmoset_developmental_agent.config.agent_config import load_config

SR = 48_000


def _recording(path: Path, seconds: float, noise: float, calls: list[float], seed: int) -> None:
    """Four capsules of noise, with 0.5 s phee-like tones (7 kHz) at ``calls`` seconds."""
    rng = np.random.default_rng(seed)
    x = noise * rng.standard_normal((int(seconds * SR), 4)).astype(np.float32)
    t = np.arange(SR // 2) / SR
    for c in calls:
        i = int(c * SR)
        x[i:i + len(t)] += (0.3 * np.sin(2 * np.pi * 7_000 * t) * np.hanning(len(t)))[:, None].astype(np.float32)
    sf.write(path, x, SR, subtype="PCM_16")


@pytest.fixture()
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(bg, "MIN_DURATION_S", 5)
    monkeypatch.setattr(bg, "BLOCK_S", 4)
    (tmp_path / "run" / "files").mkdir(parents=True)
    specs = [("n1", "2021-05-01T22:00:00", 0, 0.01, []), ("n2", "2021-05-02T23:00:00", 0, 0.002, []),
             ("n3", "2021-05-03T01:00:00", 0, 0.01, [3.0]), ("d1", "2021-05-01T10:00:00", 2, 0.01, [2.0, 6.0]),
             ("e1", "2021-05-01T19:30:00", 0, 0.01, [])]       # 19:30: outside the night hours
    for k, (rid, start, n_calls, noise, calls) in enumerate(specs):
        _recording(tmp_path / f"{rid}.wav", 10, noise, calls, k)
        (tmp_path / "run" / "files" / f"{rid}.json").write_text(json.dumps(
            {"recording_id": rid, "path": str(tmp_path / f"{rid}.wav"), "start_time": start, "duration_s": 10.0,
             "n_calls": n_calls}))
    return tmp_path


def test_pool_takes_one_night_per_date_and_the_day_reference(run: Path) -> None:
    p = bg.pool(run / "run")
    assert sorted((r["recording_id"], r["kind"]) for r in p) == [("d1", "day"), ("n1", "night"), ("n2", "night"),
                                                                  ("n3", "night")]


def test_scan_and_select(run: Path) -> None:
    cfg = load_config().preprocessing
    out = run / "out"
    (out / "scan").mkdir(parents=True)
    for rec in bg.pool(run / "run"):
        s = bg.scan(rec, cfg)
        assert len(s["level_db"]) == 9 and s["ltsa"].shape == (96, 9)    # every second but the last
        np.save(out / "scan" / f"{rec['recording_id']}_ltsa.npy", s.pop("ltsa").astype(np.float16))
        (out / "scan" / f"{rec['recording_id']}.json").write_text(json.dumps(s))
    n3 = json.loads((out / "scan" / "n3.json").read_text())
    assert any(abs(d["start_s"] - 3.0) < 0.3 for d in n3["detections"])
    r = bg.select(out, n=1)
    picked = [c["recording_id"] for c in r["candidates"] if c["selected"]]
    assert picked == ["n1"]                      # call-free, and its level matches the day background (n2 is quieter)
    assert (out / "png" / "n1.png").exists()


def test_library_serves_the_candidates(run: Path) -> None:
    from tools.corpus.call_library import Backgrounds
    test_scan_and_select(run)
    lib = Backgrounds(run / "out")
    assert [r["recording_id"] for r in lib.rows] == ["n1"] and "level_db" not in lib.rows[0]
    x, sr = lib.audio("n1", 2.0, 3.0)
    assert sr == SR and x.shape == (3 * SR,)
    lib.add_feedback({"recording_id": "n1", "verdict": "approve", "note": ""})
    from tools.synth.generate import approved
    assert [c["recording_id"] for c in approved(run / "out")] == ["n1"]
