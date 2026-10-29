"""Phone labelling server: labels from the app land in the Audacity format, and nothing is overwritten unseen."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "labels"))
from import_audacity import read_audacity  # noqa: E402
from phone_server import manifest, parse_labels, write_labels  # noqa: E402


def _set(tmp_path: Path) -> Path:
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "ts_01.wav").write_bytes(b"RIFF....")
    (tmp_path / "clips.csv").write_text("clip_id,file,source_path,offset_s,duration_s,wall_time,day,beam_index,gain\n"
                                        "ts_01,ts_01.wav,x,0,60.0,2021-05-06T15:12:51,1,4,1\n"
                                        "ts_02,ts_02.wav,x,0,60.0,2021-05-06T16:00:00,1,4,1\n")
    return tmp_path


def test_phone_labels_read_back_through_import_audacity(tmp_path):
    folder = _set(tmp_path)
    text = parse_labels("1.5\t2.25\tc\n3\t3.4\t?\n")
    write_labels(folder, "ts_01", text)
    assert read_audacity(folder / "labels" / "ts_01.txt") == [(1.5, 2.25, "c"), (3.0, 3.4, "?")]


def test_none_is_accepted_and_empty_refused(tmp_path):
    assert parse_labels("0\t0\tnone\n") == "0.000000\t0.000000\tnone\n"
    with pytest.raises(ValueError):
        parse_labels("")  # would read as "checked, no calls": the app must say `none`
    with pytest.raises(ValueError):
        parse_labels("1\t2\tphee\n")
    with pytest.raises(ValueError):
        parse_labels("2\t1\tc\n")


def test_a_different_existing_file_is_backed_up(tmp_path):
    folder = _set(tmp_path)
    (folder / "labels").mkdir()
    (folder / "labels" / "ts_01.txt").write_text("5.0\t6.0\tc\n")
    assert write_labels(folder, "ts_01", parse_labels("5.0\t6.0\tc\n")) is not None  # normalised text differs
    backups = list((folder / "labels" / ".bak").glob("ts_01.*.txt"))
    assert [b.read_text() for b in backups] == ["5.0\t6.0\tc\n"]
    assert write_labels(folder, "ts_01", parse_labels("5.0\t6.0\tc\n")) is None  # identical: no new backup


def test_manifest_lists_only_clips_with_audio(tmp_path):
    folder = _set(tmp_path)
    m = manifest({"test": folder})
    assert [c["id"] for c in m["sets"][0]["clips"]] == ["ts_01"]
    assert m["sets"][0]["clips"][0]["labels"] is None
