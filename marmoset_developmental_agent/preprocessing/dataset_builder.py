"""Chronological, day-by-day dataset construction (Hasson developmental framework).

Recordings are sorted by wall-clock start time. They are never shuffled, since the
agent must experience its "life" in order. Each recording goes through the GmSLM
front-end (``filter_5khz`` + ``segmentation``) and becomes a list of 10 s chunks.
Chunks are then assigned to simulated days in one of two ways:

* ``exposure`` (default; mirrors ``data_preparation/scripts/generate_cycle_sizes.py``):
  the chronological chunk stream is cut so that day ``d`` holds
  ``n(d) = floor(F(d) * 3600 / chunk_duration_s)`` chunks, where F(d) is the
  daily-hours schedule. This models exposure growing through infancy.
* ``calendar``: one simulated day per recording day.

At real scale the chunks do not fit in memory, so ``export_manifest`` writes them as
WAV files plus a TSV and a ``day_sizes.json`` (the analogue of Hasson's
``cycle_sizes/train.json``).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from marmoset_developmental_agent.config.agent_config import ExposureScheduleConfig, PreprocessingConfig
from marmoset_developmental_agent.preprocessing.filter_5khz import preprocess_waveform
from marmoset_developmental_agent.preprocessing.segmentation import (
    CallSegment,
    Chunk,
    activity_to_calls,
    chunk_recording,
    detection_scores,
    label_calls_from_annotations,
)

logger = logging.getLogger(__name__)


@dataclass
class Recording:
    """A continuous recording plus optional ground-truth call annotations."""

    audio: np.ndarray
    sample_rate: int
    start_time: dt.datetime
    recording_id: str
    day: Optional[int] = None
    annotations: List[CallSegment] = field(default_factory=list)


class DailyExposureSchedule:
    """F(d): chunks consumed on simulated day ``d`` (0-indexed)."""

    def __init__(self, cfg: ExposureScheduleConfig, seconds_per_sample: float) -> None:
        if not cfg.hours_per_day_by_period or any(h < 0 for h in cfg.hours_per_day_by_period):
            raise ValueError("hours_per_day_by_period must be a non-empty list of non-negative values")
        self.cfg = cfg
        self.seconds_per_sample = seconds_per_sample

    def hours(self, day: int) -> float:
        period = min(day // self.cfg.days_per_period, len(self.cfg.hours_per_day_by_period) - 1)
        return float(self.cfg.hours_per_day_by_period[period])

    def samples_for_day(self, day: int) -> int:
        n = int(self.hours(day) * 3600 / self.seconds_per_sample)
        if n <= 0:
            raise ValueError(f"F(d={day}) = {self.hours(day)} h gives 0 samples of {self.seconds_per_sample}s")
        return n

    def day_sizes(self, total_samples: int) -> List[int]:
        """Cut ``total_samples`` chronologically; the last day may be partial."""
        sizes: List[int] = []
        assigned = 0
        while assigned < total_samples:
            n = min(self.samples_for_day(len(sizes)), total_samples - assigned)
            sizes.append(n)
            assigned += n
        return sizes


class ChronologicalDataset:
    """Chunks in chronological order with contiguous per-day index ranges."""

    def __init__(self, chunks: List[Chunk], day_sizes: Sequence[int]) -> None:
        if sum(day_sizes) != len(chunks):
            raise ValueError(f"day sizes sum to {sum(day_sizes)} but there are {len(chunks)} chunks")
        self.chunks = chunks
        self.day_sizes = list(day_sizes)
        ends = np.cumsum(self.day_sizes)
        self._bounds: List[Tuple[int, int]] = list(zip([0, *ends[:-1].tolist()], ends.tolist()))

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def num_days(self) -> int:
        return len(self.day_sizes)

    def day_indices(self, day: int) -> np.ndarray:
        s, e = self._bounds[day]
        return np.arange(s, e)

    def day(self, day: int) -> List[Chunk]:
        s, e = self._bounds[day]
        return self.chunks[s:e]

    def days(self, start: int, stop: int) -> List[Chunk]:
        stop = min(stop, self.num_days)
        return [c for d in range(start, stop) for c in self.day(d)]

    def day_of(self, index: int) -> int:
        return int(np.searchsorted(np.cumsum(self.day_sizes), index, side="right"))

    def waveforms(self, indices: Iterable[int]) -> np.ndarray:
        return np.stack([self.chunks[i].audio() for i in indices])

    def summary(self) -> Dict[str, object]:
        n_calls = sum(len(c.calls) for c in self.chunks)
        return {
            "n_chunks": len(self.chunks),
            "n_days": self.num_days,
            "day_sizes": self.day_sizes,
            "n_calls": n_calls,
            "calls_per_chunk": n_calls / max(1, len(self.chunks)),
        }


def preprocess_recording(
    rec: Recording, cfg: PreprocessingConfig
) -> Tuple[List[Chunk], Dict[str, float]]:
    """GmSLM front-end for one recording: detect calls -> 16 kHz audio -> 10 s chunks.

    ``preprocess_waveform`` owns both rate conversions (detection rate for the
    detector, target rate for the audio), so there is no resampling here.
    """
    audio_16k, activity = preprocess_waveform(rec.audio, rec.sample_rate, cfg)
    calls = activity_to_calls(activity, cfg)
    stats: Dict[str, float] = {}
    if rec.annotations:
        stats = detection_scores(calls, rec.annotations)
        calls = label_calls_from_annotations(calls, rec.annotations)
    chunks = chunk_recording(audio_16k, calls, cfg, rec.recording_id, rec.start_time, rec.day)
    return chunks, stats


def build_chronological_dataset(
    recordings: Iterable[Recording],
    pre_cfg: PreprocessingConfig,
    exp_cfg: ExposureScheduleConfig,
) -> Tuple[ChronologicalDataset, Dict[str, object]]:
    """Sort recordings by time, preprocess, and assign chunks to simulated days."""
    recs = sorted(recordings, key=lambda r: r.start_time)
    all_chunks: List[Chunk] = []
    det: List[Dict[str, float]] = []
    for rec in recs:
        chunks, stats = preprocess_recording(rec, pre_cfg)
        all_chunks.extend(chunks)
        if stats:
            det.append(stats)
        logger.debug("%s: %d chunks", rec.recording_id, len(chunks))
    if not all_chunks:
        raise RuntimeError("No chunks produced: check preprocessing thresholds / input audio")
    # Stable chronological order even if chunk_recording is later parallelised.
    all_chunks.sort(key=lambda c: c.start_time)

    if exp_cfg.day_assignment == "exposure":
        sizes = DailyExposureSchedule(exp_cfg, pre_cfg.chunk_duration_s).day_sizes(len(all_chunks))
    elif exp_cfg.day_assignment == "calendar":
        keys = [c.recording_day if c.recording_day is not None else c.start_time.date() for c in all_chunks]
        sizes = [sum(1 for _ in grp) for grp in _runs(keys)]
    else:
        raise ValueError(exp_cfg.day_assignment)

    info: Dict[str, object] = {"n_recordings": len(recs)}
    if det:
        n_ann = sum(d["n_annotated"] for d in det)
        n_det = sum(d["n_detected"] for d in det)
        info["detection_precision"] = float(sum(d["precision"] * d["n_detected"] for d in det) / max(1, n_det))
        info["detection_recall"] = float(sum(d["recall"] * d["n_annotated"] for d in det) / max(1, n_ann))
    return ChronologicalDataset(all_chunks, sizes), info


def _runs(keys: Sequence[object]) -> Iterable[List[object]]:
    run: List[object] = []
    for k in keys:
        if run and k != run[-1]:
            yield run
            run = []
        run.append(k)
    if run:
        yield run


# --------------------------------------------------------------------------- #
# Real-data I/O
# --------------------------------------------------------------------------- #
def load_recordings_from_manifest(manifest_csv: Union[str, Path]) -> List[Recording]:
    """Load recordings listed in a CSV with columns ``path,start_time[,recording_id]``.

    ``start_time`` is ISO-8601. Relative paths resolve against the CSV's folder.
    Everything is loaded eagerly; for 24/7 colony data, run this per day and
    ``export_manifest`` incrementally.
    """
    import soundfile as sf

    root = Path(manifest_csv).parent
    recs: List[Recording] = []
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            path = Path(row["path"])
            path = path if path.is_absolute() else root / path
            audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
            recs.append(
                Recording(
                    audio=audio.mean(axis=1),
                    sample_rate=sr,
                    start_time=dt.datetime.fromisoformat(row["start_time"]),
                    recording_id=row.get("recording_id") or path.stem,
                )
            )
    return recs


def export_manifest(dataset: ChronologicalDataset, out_dir: Union[str, Path]) -> Path:
    """Write chunks as WAVs, a chronological ``manifest.tsv`` and ``day_sizes.json``."""
    import soundfile as sf

    out = Path(out_dir)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.tsv", "w") as f:
        f.write("path\tn_samples\tday\trecording_id\tstart_time\tcalls\n")
        for i, c in enumerate(dataset.chunks):
            p = out / "chunks" / f"{i:08d}.wav"
            wav = c.audio()
            sf.write(str(p), wav, c.sample_rate)
            calls = ";".join(f"{x.start_s:.3f}-{x.end_s:.3f}" for x in c.calls)
            f.write(f"{p.relative_to(out)}\t{len(wav)}\t{dataset.day_of(i)}\t{c.recording_id}\t{c.start_time.isoformat()}\t{calls}\n")
    (out / "day_sizes.json").write_text(json.dumps(dataset.day_sizes))
    return out / "manifest.tsv"


def load_manifest(manifest_tsv: Union[str, Path], sample_rate: int = 16_000) -> ChronologicalDataset:
    """Lazy counterpart of :func:`export_manifest` (waveforms read on access)."""
    root = Path(manifest_tsv).parent
    chunks: List[Chunk] = []
    with open(manifest_tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            calls = [
                CallSegment(float(a), float(b))
                for a, b in (x.split("-") for x in row["calls"].split(";") if x)
            ]
            chunks.append(
                Chunk(
                    waveform=None,
                    sample_rate=sample_rate,
                    recording_id=row["recording_id"],
                    start_time=dt.datetime.fromisoformat(row["start_time"]),
                    offset_s=0.0,
                    calls=calls,
                    path=str(root / row["path"]),
                )
            )
    sizes = json.loads((root / "day_sizes.json").read_text())
    return ChronologicalDataset(chunks, sizes)
