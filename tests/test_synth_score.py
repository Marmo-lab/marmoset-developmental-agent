"""tools/synth/score.py: matching, and detect + score end to end on a small synthetic set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.synth.score import detect, match_iou, quality, score, score_recording


def test_matching_and_counts() -> None:
    calls = [(1.0, 2.0), (5.0, 5.2), (9.0, 10.0)]
    dets = [(1.1, 2.1), (4.0, 4.1), (9.0, 9.3), (9.4, 10.0)]   # a hit, a false alarm, a call split in two
    assert match_iou(dets, calls) == [(0, 0), (3, 2)]
    s = score_recording(dets, calls)
    assert s["call_hit"] == [True, False, True] and s["call_iou"] == [True, False, True]
    assert s["det_hit"] == [True, False, True, True] and s["n_iou"] == 2
    assert [round(x, 3) for x in s["onset_err"]] == [0.1, 0.4]
    assert quality(0.1, 0.5) == "short_low_conf" and quality(0.1, 0.7) == "ok" and quality(9, 0.1) == "long"


def test_detect_and_score_a_small_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_generate import test_make_writes_audio_and_the_answer
    test_make_writes_audio_and_the_answer(tmp_path, monkeypatch)
    folder = tmp_path / "out"
    detect(folder, 0, 2)
    detect(folder, 1, 2)
    s = score(folder)
    assert set(s["views"]) == {"detector", "front_end"}
    assert set(s["views"]["front_end"]["by_snr"]) == {"1.5", "1.0"}
    assert (folder / "scores.png").exists() and json.loads((folder / "scores.json").read_text())["n_recordings"] == 4
