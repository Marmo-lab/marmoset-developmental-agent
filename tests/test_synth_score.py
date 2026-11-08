"""tools/synth/score.py: matching, and detect + score end to end on a small synthetic set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.synth.score import detect, match_iou, quality, score, score_recording, split_ok


def test_matching_and_counts() -> None:
    calls = [(1.0, 2.0), (5.0, 5.2), (9.0, 10.0)]
    dets = [(1.1, 2.1), (4.0, 4.1), (9.0, 9.3), (9.4, 10.0)]   # a hit, a false alarm, a call split in two
    assert match_iou(dets, calls) == [(0, 0), (3, 2)]
    s = score_recording(dets, calls)
    assert s["call_hit"] == [True, False, True] and s["call_iou"] == [True, False, True]
    assert s["det_hit"] == [True, False, True, True] and s["n_iou"] == 2
    assert [round(x, 3) for x in s["onset_err"]] == [0.1, 0.4]
    s = score_recording(dets, calls, [False, False, False])
    assert s["call_main"] == s["call_iou"] and s["det_main"] == [True, False, False, True]
    s = score_recording(dets, calls, [False, False, True])        # the split call is a phee: both pieces right
    assert s["call_main"] == [True, False, True] and s["det_main"] == [True, False, True, True]
    s = score_recording([(9.0, 9.3), (9.4, 9.6)], calls, [False, False, True])   # no piece reaches IoU 0.5
    assert s["call_iou"] == [False, False, False] and s["call_main"] == [False, False, True]
    assert split_ok("phee_3") and not split_ok("tsik")
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


def test_figures_of_a_small_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.analysis import synth_figures as sf_
    test_detect_and_score_a_small_set(tmp_path, monkeypatch)
    folder = tmp_path / "out"
    rows = sf_.per_recording(folder)
    assert len(rows) == 4 and sum(r["n_calls"] for r in rows) == sum(1 for _ in open(folder / "events.tsv")) - 1
    b = sf_.bootstrap(rows)
    assert set(b) == {1.5, 1.0} and all(lo <= est <= hi for v in b.values() for est, lo, hi in v.values())
    sf_.snr_curve(folder, tmp_path / "curve.png")
    sf_.recall_by_type(folder, tmp_path / "types.png")
    assert (tmp_path / "curve.png").exists() and (tmp_path / "types.png").exists()
    assert sf_._db(20 * np.log10(1.5)) == "+3.5" and sf_._db(20 * np.log10(0.7)) == "-3"


def test_another_checkpoint_writes_its_own_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_generate import test_make_writes_audio_and_the_answer
    test_make_writes_audio_and_the_answer(tmp_path, monkeypatch)
    folder = tmp_path / "out"
    detect(folder, 0, 1, model="models/call_detector_v1.pt", name="copy")
    s = score(folder, "copy")
    assert (folder / "detections_copy" / "000.tsv").exists() and not (folder / "detections").exists()
    assert s["model"] == "copy" and (folder / "scores_copy.json").exists() and (folder / "scores_copy.png").exists()


def test_model_comparison_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.analysis import synth_figures as sf_
    test_detect_and_score_a_small_set(tmp_path, monkeypatch)
    folder = tmp_path / "out"
    detect(folder, 0, 1, model="models/call_detector_v1.pt", name="copy")
    score(folder, "copy")
    sf_.compare_models(folder, "copy", tmp_path / "models.png", snr="1.5")
    assert (tmp_path / "models.png").exists()
