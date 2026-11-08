"""tools/analysis/train_call_detector.py: synthetic and real extra sources, their targets, and mixed batches."""

from __future__ import annotations

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
