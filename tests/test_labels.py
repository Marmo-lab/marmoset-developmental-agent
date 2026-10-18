"""Audacity label import: conventions from LABELING.md."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "labels"))
from import_audacity import read_audacity  # noqa: E402


def test_reads_labels_skips_spectral_lines_and_defaults_empty_text_to_call(tmp_path):
    f = tmp_path / "clip_01.txt"
    f.write_text("1.25\t1.90\tc\n\\\t5000.0\t9000.0\n10.5\t11.0\t?\n20.0\t20.3\t\n")
    assert read_audacity(f) == [(1.25, 1.90, "c"), (10.5, 11.0, "?"), (20.0, 20.3, "c")]


def test_none_marks_a_checked_clip_with_no_calls(tmp_path):
    """Audacity refuses to export an empty label track, so `none` stands for 'no calls'."""
    f = tmp_path / "clip_05.txt"
    f.write_text("12.0\t12.0\tnone\n")
    assert read_audacity(f) == []


def test_empty_file_is_also_no_calls(tmp_path):
    f = tmp_path / "clip_01.txt"
    f.write_text("")
    assert read_audacity(f) == []


def test_rejects_unknown_label_text(tmp_path):
    f = tmp_path / "clip_02.txt"
    f.write_text("1.0\t2.0\tphee\n")
    with pytest.raises(ValueError, match="not one of"):
        read_audacity(f)
