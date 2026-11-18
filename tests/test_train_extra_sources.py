"""tools/analysis/train_call_detector.py: synthetic and real extra sources, their targets, and mixed batches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import tools.analysis.train_call_detector as tcd
from marmoset_developmental_agent.config.agent_config import PreprocessingConfig

SR = 48_000


@pytest.fixture()
def cfg() -> PreprocessingConfig:
    return PreprocessingConfig.learned()


def _minute(rng: np.random.Generator) -> np.ndarray:
    return (0.01 * rng.standard_normal(60 * SR)).astype(np.float32)


def test_synthetic_targets_are_calls_and_background(tmp_path: Path, cfg: PreprocessingConfig) -> None:
    rng = np.random.default_rng(0)
    (tmp_path / "audio").mkdir()
    sf.write(tmp_path / "audio" / "r0.flac", _minute(rng), SR)
    (tmp_path / "recordings.tsv").write_text("rec_id\tsnr\nr0\t2.0\n")
    (tmp_path / "events.tsv").write_text("rec_id\tonset_s\toffset_s\nr0\t10.0\t11.0\n")
    c = tcd.load_synth(cfg, tmp_path)["synth:r0"]
    y, on, t = c["y"][0], c["y"][1], c["t"]
    assert (y[(t > 10.05) & (t < 10.95)] == 1).all() and (y[(t < 9.9) | (t > 11.1)] == 0).all()
    assert abs(t[np.argmax(on)] - 10.0) < 0.011


def test_real_minutes_label_positives_and_ignore_the_rest(tmp_path: Path, cfg: PreprocessingConfig) -> None:
    rng = np.random.default_rng(1)
    sf.write(tmp_path / "m0.flac", _minute(rng), SR)
    sf.write(tmp_path / "m1.flac", _minute(rng), SR)
    (tmp_path / "positives.tsv").write_text("minute_id\tstart_s\tend_s\tsource\nm0\t20.0\t20.2\tcandidate\n")
    clips = tcd.load_real(cfg, tmp_path)
    assert list(clips) == ["real:m0"]                       # a minute without positives teaches nothing
    c = clips["real:m0"]
    assert set(np.unique(c["y"][0])) == {-1.0, 1.0} and (c["y"][1:] == -1).all()
    starts = [tcd._crop_start(c, np.random.default_rng(k)) for k in range(50)]
    assert all(s <= c["pos"][0] < s + tcd.CROP or s <= c["pos"][-1] < s + tcd.CROP for s in starts)


def test_training_mixes_the_sources(tmp_path: Path, cfg: PreprocessingConfig) -> None:
    rng = np.random.default_rng(2)
    t = np.arange(600) * 0.01
    hand = {"h": {"x": rng.standard_normal((92, 600)).astype(np.float32),
                  "y": np.zeros((3, 600), np.float32), "t": t, "day": np.int64(0), "lead": np.float64(0)}}
    real = {"real:m": {"x": rng.standard_normal((92, 600)).astype(np.float32),
                       "y": np.stack([np.where((t > 2) & (t < 2.2), 1, -1)] + [np.full(600, -1)] * 2).astype(np.float32),
                       "t": t, "day": np.int64(-1), "lead": np.float64(0), "pos": np.flatnonzero((t > 2) & (t < 2.2))}}
    net = tcd.train({**hand, **real}, ["h"], steps=3, seed=0, n_out=1, extra=(([], 0.25), (["real:m"], 0.25)))
    assert np.isfinite(tcd.predict(net, hand["h"])).all()


def test_hand_sets_map_to_the_cluster_layout(tmp_path: Path) -> None:
    assert tcd.hand_sets() == tcd.SETS
    sets = tcd.hand_sets(tmp_path)
    assert sets[0] == (str(tmp_path / "labeling" / "clips.csv"), tcd.SETS[0][1], str(tmp_path / "clip_audio48"))
    assert sets[1][2] == str(tmp_path / "heldout_audio48")


def test_one_fold_then_merge_equals_the_files(tmp_path: Path, cfg: PreprocessingConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(4)
    t = np.arange(600) * 0.01
    fake = {f"c{k}": {"x": rng.standard_normal((92, 600)).astype(np.float32), "y": np.zeros((3, 600), np.float32),
                      "t": t, "day": np.int64(k), "lead": np.float64(0)} for k in range(4)}
    monkeypatch.setattr(tcd, "load_clips", lambda cfg, sets=tcd.SETS: {k: dict(v) for k, v in fake.items()})
    for k in range(2):
        tcd.cv(cfg, tmp_path, folds=2, steps=2, n_out=1, only=k)
    assert sorted(p.name for p in tmp_path.glob("fold_*.npz")) == ["fold_0.npz", "fold_1.npz"]
    tcd.cv(cfg, tmp_path, folds=2, steps=2, n_out=1, merge=True)
    assert sorted(np.load(tmp_path / "oof_probs.npz").files) == ["c0", "c1", "c2", "c3"]
    assert (tmp_path / "learned_detections.json").exists()


def _short_folder(tmp_path: Path, rng: np.random.Generator) -> Path:
    """A review folder of four items in one raw 4-capsule file: a train call, a train "not a call", a train
    "unsure" and a test-half call."""
    raw = tmp_path / "rec.wav"
    sf.write(raw, (0.01 * rng.standard_normal((30 * SR, 4))).astype(np.float32), SR)
    (tmp_path / "meta.tsv").write_text("item_id\tfile_path\tclip_start_s\thalf\n" + "".join(
        f"{k}\t{raw}\t{start}\t{half}\n" for k, start, half in
        (("sc_000", 5.0, "train"), ("sc_001", 10.0, "train"), ("sc_002", 15.0, "train"), ("sc_003", 20.0, "test"))))
    marks = '[[0.95, 1.05, "a", "candidate"]]'
    (tmp_path / "items.tsv").write_text("item_id\tclip_s\tmarks\n" + "".join(f"sc_00{k}\t2.0\t{marks}\n" for k in range(4)))
    lines = [{"item_id": "sc_000", "verdict": "call", "note": "", "bounds": [0.9, 1.2]},
             {"item_id": "sc_001", "verdict": "call", "note": "", "bounds": [0.5, 0.6]},
             {"item_id": "sc_001", "verdict": "not", "note": ""},          # changed answer: the last one counts
             {"item_id": "sc_002", "verdict": "unsure", "note": ""},
             {"item_id": "sc_003", "verdict": "call", "note": "", "bounds": [0.9, 1.1]}]
    (tmp_path / "review_feedback.jsonl").write_text("".join(json.dumps(d) + "\n" for d in lines))
    return tmp_path


def test_short_labels_teach_the_call_and_the_rejected_candidate_only(tmp_path: Path, cfg: PreprocessingConfig) -> None:
    clips = tcd.load_short(cfg, _short_folder(tmp_path, np.random.default_rng(6)))
    assert sorted(clips) == ["short:sc_000", "short:sc_001"]           # unsure and the test half stay out
    call, rejected = clips["short:sc_000"], clips["short:sc_001"]
    t, y = call["t"], call["y"][0]
    assert (y[(t > 0.92) & (t < 1.18)] == 1).all() and (y[(t < 0.85) | (t > 1.25)] == -1).all()
    assert set(np.unique(rejected["y"][0])) == {-1.0, 0.0}             # its old "call" answer is gone
    assert (rejected["y"][0][(t > 0.97) & (t < 1.03)] == 0).all() and (rejected["y"][1:] == -1).all()
    assert len(t) >= tcd.CROP
    starts = [tcd._crop_start(call, np.random.default_rng(k)) for k in range(30)]
    assert all(s <= call["pos"][0] < s + tcd.CROP for s in starts)
    assert sorted(tcd.load_short(cfg, tmp_path, half="test")) == ["short:sc_003"]
    assert len(tcd.load_short(cfg, tmp_path, frac=0.5)) == 1


def test_mix_of_three_sources_pads_missing_shares(tmp_path: Path, cfg: PreprocessingConfig) -> None:
    folder = _short_folder(tmp_path, np.random.default_rng(7))
    clips, groups = tcd.load_extra(cfg, [], [], (0.2, 0.15, 0.15), short=[folder])
    assert [round(w, 2) for _, w in groups] == [0.2, 0.15, 0.15] and groups[2][0] == ["short:sc_000", "short:sc_001"]
    _, groups = tcd.load_extra(cfg, [], [], (0.25, 0.25))
    assert len(groups) == 3 and groups[2] == ([], 0.0)


def test_real_gaps_between_positives_are_taught_as_background(tmp_path: Path, cfg: PreprocessingConfig) -> None:
    rng = np.random.default_rng(5)
    sf.write(tmp_path / "m0.flac", _minute(rng), SR)
    (tmp_path / "positives.tsv").write_text("minute_id\tstart_s\tend_s\tsource\n"
                                           "m0\t10.0\t10.5\trun_call\nm0\t10.8\t10.9\tcandidate\n"
                                           "m0\t30.0\t30.2\tcandidate\nm0\t30.1\t30.6\trun_call\n")
    plain = tcd.load_real(cfg, tmp_path)["real:m0"]
    gaps = tcd.load_real(cfg, tmp_path, gaps=True)["real:m0"]
    t = gaps["t"]
    mid = (t > 10.55) & (t < 10.75)
    assert (plain["y"][0][mid] == -1).all() and (gaps["y"][0][mid] == 0).all()
    assert (gaps["y"][0][(t > 30.05) & (t < 30.55)] == 1).all()          # overlapping spans are one call, no gap
    assert (gaps["y"][0][(t > 12) & (t < 29)] == -1).all()               # a wide gap stays ignored
