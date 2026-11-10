"""tools/corpus/call_library.py --raw: files listed with their wall start, windows read (also unfinalised), spectrogram drawn."""

from __future__ import annotations

import datetime as dt
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus"))
import call_library as cl  # noqa: E402
from tests.test_raw_io import _write_wav  # noqa: E402

SR = 96_000
T0 = dt.datetime(2026, 11, 8, 15, 35, 54)


def _folder(tmp: Path) -> Path:
    rng = np.random.default_rng(0)
    for k, sec in enumerate([3.0, 2.0, 0.2]):              # the last is a false start: left out
        x = 0.01 * rng.standard_normal((int(sec * SR), 4)).astype(np.float32)
        x[:, 2] *= 4
        _write_wav(tmp / f"26110{k}_00{k}.WAV", x, SR, T0 + dt.timedelta(seconds=3 * k))
    p = tmp / "261101_001.WAV"                           # unfinalised: sizes left at 0
    b = bytearray(p.read_bytes())
    b[4:8] = struct.pack("<I", 0)
    k = b.find(b"data")
    b[k + 4:k + 8] = struct.pack("<I", 0)
    p.write_bytes(bytes(b))
    return tmp


def test_raw_folder_lists_reads_and_draws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rw = cl.RawFolder(_folder(tmp_path))
    assert [f["file"] for f in rw.files] == ["261100_000", "261101_001"]
    assert rw.files[1]["start"] == (T0 + dt.timedelta(seconds=3)).isoformat() and rw.files[1]["duration_s"] == 2.0
    x, sr = rw.audio("261101_001", 0.5, 1.0, "mean")
    assert sr == SR and len(x) == SR
    loud, _ = rw.audio("261100_000", 0.0, 1.0, "3")
    quiet, _ = rw.audio("261100_000", 0.0, 1.0, "1")
    assert loud.std() > 3 * quiet.std()
    with pytest.raises(KeyError):
        rw.audio("261100_000", 10.0, 1.0, "mean")
    monkeypatch.setattr(cl, "RAW", [rw])
    assert cl._raw_png("261101_001", 0.0, 2.0, "mean", 48)[:4] == b"\x89PNG"
