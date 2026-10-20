"""Streaming reader for the raw colony recordings, with format validation.

Why streaming. One raw file is 48 kHz, 4-channel, 24-bit PCM, ~62 minutes -- 2.0 GB on
disk and **2.9 GB as float32 in RAM**, and there are ~2,750 of them, so nothing reads a
whole file. The production front-end streams through ``preprocessing/streaming.py``;
:func:`iter_blocks` here is a simpler block reader for ad-hoc use, not on the pipeline path.

Why validation. ``AnimalRoomRecordings/`` contains one folder,
``Rec_2021_06_28_ultrasonic_recordings_colony``, recorded on a different instrument:
250 kHz, mono, 16-bit, no ``bext`` chunk, named ``M12_F14_*``. It sits inside the same
glob Talia's array job iterated over with a hardcoded 48 kHz assumption, so any sample
index computed from it is wrong by ~5.2x (audit open question 4). :func:`probe` rejects
anything that does not match the expected format rather than silently mis-processing it.

Why ``bext``. The broadcast-extension chunk carries each file's wall-clock start time,
so a sample offset maps to an absolute timestamp. We need that for chronological
ordering, and -- given that recording level correlates with time of day at r = +0.859
while barely correlating with calendar date -- for stratifying and interpreting
detection yield.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from marmoset_developmental_agent.io.channels import ChannelStrategy, combine, pick_best_channel

# The format every genuine colony recording has (Zoom H3-VR A-format ambisonic).
EXPECTED_SAMPLE_RATE = 48_000
EXPECTED_CHANNELS = 4

# bext layout, EBU Tech 3285: Description 256B, Originator 32B, OriginatorReference 32B,
# then OriginationDate (10B, "yyyy-mm-dd") and OriginationTime (8B, "hh:mm:ss").
_BEXT_DATE_OFFSET = 320
_BEXT_TIME_OFFSET = 330


@dataclass(frozen=True)
class RawFileInfo:
    """Format and provenance of one raw recording, as read from its header."""

    path: str
    sample_rate: int
    channels: int
    subtype: str
    frames: int
    start_time: Optional[dt.datetime]  # None when the file carries no bext chunk
    ok: bool                           # matches the expected colony format
    reason: str                        # why not, when ``ok`` is False

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def recording_id(self) -> str:
        """``<folder>_<stem>`` -- unique across the corpus, unlike the stem alone."""
        p = Path(self.path)
        return f"{p.parent.name}_{p.stem}"


def read_bext_datetime(path: Union[str, Path]) -> Optional[dt.datetime]:
    """Wall-clock start time from the WAV's ``bext`` chunk, or None if absent/unparsable.

    Walks the RIFF chunk list rather than assuming a fixed layout, because the H3-VR
    writes ``iXML`` too and chunk order is not guaranteed.
    """
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF":
            return None
        f.seek(8)
        if f.read(4) != b"WAVE":
            return None
        while True:
            header = f.read(8)
            if len(header) < 8:
                return None
            chunk_id, size = header[:4], int.from_bytes(header[4:8], "little")
            if chunk_id == b"bext":
                body = f.read(size)
                date = body[_BEXT_DATE_OFFSET:_BEXT_DATE_OFFSET + 10].decode("ascii", "ignore")
                time = body[_BEXT_TIME_OFFSET:_BEXT_TIME_OFFSET + 8].decode("ascii", "ignore")
                try:
                    return dt.datetime.strptime(
                        f"{date.strip(chr(0) + ' ')} {time.strip(chr(0) + ' ')}",
                        "%Y-%m-%d %H:%M:%S",
                    )
                except ValueError:
                    return None
            f.seek(size + (size & 1), 1)  # RIFF chunks are word-aligned


def probe(path: Union[str, Path]) -> RawFileInfo:
    """Read one file's header and decide whether it is a usable colony recording.

    Never raises on a malformed or foreign file -- returns ``ok=False`` with a reason,
    so a manifest build over 2,750 files reports problems instead of dying on one.
    """
    path = str(path)
    try:
        info = sf.info(path)
    except Exception as exc:  # unreadable, truncated, or not audio at all
        return RawFileInfo(path, 0, 0, "", 0, None, False, f"unreadable: {exc}")

    reasons = []
    if info.samplerate != EXPECTED_SAMPLE_RATE:
        reasons.append(f"sample rate {info.samplerate} != {EXPECTED_SAMPLE_RATE}")
    if info.channels != EXPECTED_CHANNELS:
        reasons.append(f"{info.channels} channels != {EXPECTED_CHANNELS}")
    if info.frames <= 0:
        reasons.append("empty")

    start = read_bext_datetime(path)
    if start is None:
        reasons.append("no bext timestamp")

    return RawFileInfo(
        path=path,
        sample_rate=info.samplerate,
        channels=info.channels,
        subtype=info.subtype or "",
        frames=info.frames,
        start_time=start,
        ok=not reasons,
        reason="; ".join(reasons),
    )


def iter_blocks(
    path: Union[str, Path],
    block_s: float = 60.0,
    overlap_s: float = 1.0,
    strategy: ChannelStrategy = "mean",
    best_channel: Optional[int] = None,
    band_hz: Tuple[float, float] = (5_000.0, 14_000.0),
) -> Iterator[Tuple[np.ndarray, int]]:
    """Stream a recording as ``(signals, start_sample)`` pairs.

    ``signals`` is ``(n_out, n_samples)`` float32 as returned by
    :func:`~...io.channels.combine` -- one row for every strategy except
    ``per_channel``. ``start_sample`` is the offset of column 0 within the file, at the
    file's native rate, so detections can be mapped back to absolute sample indices
    (and, with :attr:`RawFileInfo.start_time`, to wall-clock time).

    Consecutive blocks overlap by ``overlap_s`` so that a call straddling a boundary is
    wholly inside at least one block. The caller is responsible for de-duplicating
    detections in the overlap region -- this function does not merge anything.

    Memory is bounded by ``block_s``: at 60 s, 48 kHz, 4 channels that is ~46 MB.
    """
    info = sf.info(str(path))
    block = int(round(block_s * info.samplerate))
    overlap = int(round(overlap_s * info.samplerate))
    if block <= overlap:
        raise ValueError(f"block_s ({block_s}) must exceed overlap_s ({overlap_s})")

    if strategy == "best_snr" and best_channel is None:
        best_channel = pick_best_channel(
            _sample_excerpts(str(path), info), info.samplerate, band_hz
        )

    step = block - overlap
    with sf.SoundFile(str(path)) as f:
        start = 0
        while start < info.frames:
            f.seek(start)
            data = f.read(block, dtype="float32", always_2d=True)
            if data.shape[0] == 0:
                break
            yield combine(data, strategy, info.samplerate, band_hz, best_channel), start
            if start + block >= info.frames:
                break
            start += step


def _sample_excerpts(
    path: str, info: "sf._SoundFileInfo", n: int = 8, seconds: float = 5.0
) -> np.ndarray:
    """Concatenate ``n`` short windows spread across the file, for channel selection.

    Spread rather than contiguous because a single excerpt could land in a silent night
    stretch (nearly half the corpus is night audio with almost no vocal activity), which
    would pick a capsule on noise alone.
    """
    span = int(round(seconds * info.samplerate))
    starts = np.linspace(0, max(0, info.frames - span), n).astype(np.int64)
    out = []
    with sf.SoundFile(path) as f:
        for s in starts:
            f.seek(int(s))
            out.append(f.read(span, dtype="float32", always_2d=True))
    return np.concatenate(out, axis=0) if out else np.zeros((0, info.channels), np.float32)
