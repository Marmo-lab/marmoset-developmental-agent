"""Blocked detection over a recording, the call catalogue, and chunk audio read back
from the source (preprocessing/streaming.py). Synthetic WAVs in a tmp dir; CPU only."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig, load_config
from marmoset_developmental_agent.preprocessing.dataset_builder import (
    Recording,
    RecordingFile,
    preprocess_recording,
)
from marmoset_developmental_agent.preprocessing.filter_5khz import resample_audio
from marmoset_developmental_agent.preprocessing.segmentation import chunk_recording, plan_chunks
from marmoset_developmental_agent.preprocessing.streaming import (
    ArraySource,
    FileSource,
    detect_recording,
    read_chunks,
)

SR = 48_000
START = dt.datetime(2021, 6, 1, 10, 0, 0)
# (onset s, duration s): one call straddles the 4 s block edge used below, one the 8 s edge.
CALLS = [(1.0, 0.6), (3.8, 0.5), (7.7, 0.8), (12.0, 0.4), (16.5, 1.2)]


def _colony(seconds: float = 20.0) -> np.ndarray:
    """``(n, 4)``: FM sweeps (9 -> 7 kHz) at ``CALLS`` over independent capsule noise.

    Capsule gains differ, as for a source off to one side, so channel strategies differ.
    """
    rng = np.random.default_rng(0)
    n = int(seconds * SR)
    x = 1e-3 * rng.standard_normal((n, 4))
    for onset, dur in CALLS:
        m = int(dur * SR)
        sweep = 0.05 * signal.chirp(np.arange(m) / SR, 9_000, dur, 7_000)
        s = int(onset * SR)
        x[s : s + m] += sweep[:, None] * np.array([1.0, 0.7, 0.5, 0.3])
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def wav(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("stream") / "210601_001.WAV"
    sf.write(p, _colony(), SR, subtype="PCM_24")
    return p


def _onsets(records) -> list:
    return [round(r.start_s, 1) for r in records]


def test_file_and_array_sources_give_identical_calls(wav):
    cfg = PreprocessingConfig()
    from_file = detect_recording(FileSource(str(wav)), START, cfg)
    audio, _ = sf.read(wav, dtype="float32", always_2d=True)
    from_array = detect_recording(ArraySource(audio, SR), START, cfg)
    assert from_file == from_array
    assert len(from_file) == len(CALLS)


def test_blocks_find_the_same_calls_as_one_block_and_never_split_or_duplicate(wav):
    """Calls straddling a block edge are kept once, whole, with the same boundaries."""
    src = FileSource(str(wav))
    whole = detect_recording(src, START, dataclasses.replace(PreprocessingConfig(), stream_block_s=1_000.0))
    blocked = detect_recording(src, START, dataclasses.replace(PreprocessingConfig(), stream_block_s=4.0))
    assert len(blocked) == len(whole) == len(CALLS)
    for a, b in zip(whole, blocked):
        assert a.start_s == pytest.approx(b.start_s, abs=0.03)
        assert a.end_s == pytest.approx(b.end_s, abs=0.03)
    for (onset, dur), r in zip(CALLS, blocked):
        assert r.start_s == pytest.approx(onset, abs=0.08)
        assert r.duration_s == pytest.approx(dur, abs=0.12)


def test_catalogue_carries_wall_clock_time_and_quality_features(wav):
    for r in detect_recording(FileSource(str(wav)), START, PreprocessingConfig()):
        assert r.start_time == START + dt.timedelta(seconds=r.start_s)
        assert 7_000 - 100 < r.peak_freq_hz < 9_000 + 100  # the sweep's range
        assert r.tonal_fraction > 0.8
        assert r.snr_db > 20
        assert r.peak_freq_iqr_hz > 2 * SR / 2048  # a sweep, not a fixed tone


def test_talia_path_catalogue_has_no_features(wav):
    records = detect_recording(FileSource(str(wav)), START, PreprocessingConfig.talia())
    assert all(np.isnan(r.snr_db) and np.isnan(r.peak_freq_hz) for r in records)


def test_read_chunks_matches_slicing_in_memory_audio():
    """On a mono source, chunks read window by window equal slices of the whole resampled
    recording, away from the resampler's edges."""
    mono = _colony()[:, 0]
    cfg = PreprocessingConfig()
    calls = [r.segment() for r in detect_recording(ArraySource(mono, SR), START, cfg)]
    old = chunk_recording(resample_audio(mono, SR, 16_000), calls, cfg, "r", START)
    new = read_chunks(ArraySource(mono, SR), plan_chunks(calls, cfg, len(mono) / SR), cfg, "r", START)
    assert [c.offset_s for c in new] == [c.offset_s for c in old]
    for a, b in zip(old, new):
        assert a.start_time == b.start_time and a.calls == b.calls
        assert len(b.waveform) == 160_000
        np.testing.assert_allclose(a.waveform[800:-800], b.waveform[800:-800], atol=1e-4)


def test_training_strategies_shape_and_per_channel_refused(wav):
    src = FileSource(str(wav))
    cfg = PreprocessingConfig()
    plans = plan_chunks([r.segment() for r in detect_recording(src, START, cfg)], cfg, 20.0)
    for strategy in ("ambisonic_beam", "mean", "best_snr", "max_energy"):
        c = dataclasses.replace(cfg, training_channel_strategy=strategy)
        chunks = read_chunks(src, plans, c, "r", START)
        assert all(ch.waveform.shape == (160_000,) for ch in chunks)
    with pytest.raises(ValueError, match="per_channel"):
        read_chunks(src, plans, dataclasses.replace(cfg, training_channel_strategy="per_channel"), "r", START)


def test_file_and_memory_recordings_preprocess_identically(wav):
    audio, _ = sf.read(wav, dtype="float32", always_2d=True)
    cfg = PreprocessingConfig()
    a, _ = preprocess_recording(RecordingFile(str(wav), START, "r"), cfg)
    b, _ = preprocess_recording(Recording(audio, SR, START, "r"), cfg)
    assert len(a) == len(b) > 0
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.waveform, y.waveform)


def test_gmslm_arm_yaml_is_the_talia_preset():
    root = Path(__file__).resolve().parents[1]
    assert load_config(root / "config" / "gmslm_arm.yaml").preprocessing == PreprocessingConfig.talia()
    assert load_config(root / "config" / "default.yaml").preprocessing == PreprocessingConfig()


def test_corpus_run_writes_the_catalogue_and_scores_labelled_minutes(wav, tmp_path):
    """tools/corpus: per-recording catalogue + summary, and the label check reads them back."""
    from tools.corpus.detect_corpus import process
    from tools.corpus.summarize_corpus import label_check

    rec = RecordingFile(str(wav), START, "R_210601_001")
    s = process(rec, tmp_path, PreprocessingConfig(), export_audio=True)
    assert s["n_calls"] == len(CALLS) and s["n_chunks"] >= 1
    rows = (tmp_path / "calls" / "R_210601_001.csv").read_text().splitlines()
    assert len(rows) == 1 + len(CALLS) and rows[1].split(",")[3].startswith("2021-06-01T10:00:00.99")
    assert len(list((tmp_path / "chunks" / "R_210601_001").glob("*.wav"))) == s["n_chunks"]

    clips = tmp_path / "clips.csv"
    clips.write_text(f"clip_id,source_path,offset_s,day\nc1,{wav},0.0,1\n")
    labels = tmp_path / "labels.csv"
    labels.write_text("clip_id,label,clip_start_s,clip_end_s\n" +
                      "".join(f"c1,c,{o},{o + d}\n" for o, d in CALLS))
    files = {"R_210601_001": {"path": str(wav)}}
    res = label_check(files, tmp_path, str(clips), str(labels))
    assert res["day_c+?"]["P"] == res["day_c+?"]["R"] == 1.0


def test_merge_builds_a_chronological_manifest_and_drops_duplicate_copies(wav, tmp_path):
    """tools/corpus/merge_corpus: per-recording chunk lists -> one manifest the agent loads.

    The same file under a second folder, with a later recording in between alphabetically,
    must appear once, and rows must follow wall-clock time rather than recording ids.
    """
    import shutil

    from marmoset_developmental_agent.preprocessing.dataset_builder import load_manifest
    from tools.corpus.detect_corpus import process
    from tools.corpus.merge_corpus import merge

    copy = tmp_path / "Rec_B" / wav.name
    copy.parent.mkdir()
    shutil.copy(wav, copy)
    out = tmp_path / "run"
    for path, start, rid in [(wav, START + dt.timedelta(hours=1), "Rec_A_001"),
                             (copy, START + dt.timedelta(hours=1), "Rec_B_001"),
                             (wav, START, "Rec_C_001")]:
        process(RecordingFile(str(path), start, rid), out, PreprocessingConfig(), export_audio=True)
    s = merge(out)
    assert s["duplicates_dropped"] == ["Rec_B_001"] and s["n_recordings"] == 2
    ds = load_manifest(out / "manifest.tsv")
    assert len(ds) == s["n_chunks"] == sum(ds.day_sizes)
    assert [c.start_time for c in ds.chunks] == sorted(c.start_time for c in ds.chunks)
    assert ds.chunks[0].recording_id == "Rec_C_001"
    assert ds.chunks[0].audio().shape == (160_000,) and ds.chunks[0].calls
