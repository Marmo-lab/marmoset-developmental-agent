"""Call extraction and 10 s chunking at 16 kHz (GmSLM Appendix A.1-A.2).

Call detection groups consecutive vocal-active STFT frames (see ``filter_5khz.py``).
Gaps up to ``merge_gap_s`` are bridged, so twitter syllables form one call; groups with
fewer than ``min_run_members`` frames are dropped; a group survives only if its unpadded
frame span lies strictly inside (``min_call_duration_s``, ``max_call_duration_s``) and it
passes the two shape rejections of Talia's ``filter_calls`` (or, on the spectral path, our
tonality rejection); survivors are then padded by
``boundary_pad_s``. All of this is ported from her source, not the audit's summary. The paper's "calls are 0.25-4 s" is the rounded
form of the code's 0.2322-4 s; we follow the code (audit S7, tier A).

Chunking follows GmSLM: each chunk is exactly ``chunk_duration_s`` (10 s) at 16 kHz
and packs "as many marmoset calls as possible". A chunk starts just before an
uncovered call and spans 10 s. Every call that ends inside the window is annotated
in it, and the next chunk starts at the first call that did not fit. Inter-call
gaps are kept, not silenced.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.preprocessing.filter_5khz import FrameActivity


@dataclass
class CallSegment:
    """A detected (or annotated) call, in seconds. Labels are optional (mock / PHEEDB only)."""

    start_s: float
    end_s: float
    call_type: Optional[str] = None
    caller_id: Optional[int] = None
    receiver_id: Optional[int] = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Chunk:
    """A fixed-length 16 kHz training segment with call boundaries relative to its start."""

    waveform: Optional[np.ndarray]  # (chunk_samples,) float32; None if stored on disk
    sample_rate: int
    recording_id: str
    start_time: dt.datetime  # absolute wall-clock time of sample 0
    offset_s: float  # offset of sample 0 within its recording
    calls: List[CallSegment] = field(default_factory=list)
    path: Optional[str] = None
    recording_day: Optional[int] = None

    def audio(self) -> np.ndarray:
        if self.waveform is not None:
            return self.waveform
        if self.path is None:
            raise ValueError("Chunk has neither an in-memory waveform nor a path")
        import soundfile as sf

        wav, _ = sf.read(self.path, dtype="float32")
        return wav

    @property
    def num_samples(self) -> int:
        return len(self.audio())


def _frames(seconds: float, activity: FrameActivity) -> int:
    """Seconds -> frame count at the hop the activity was actually computed with."""
    return int(round(seconds / activity.hop_s))


def _active_runs(activity: FrameActivity, cfg: PreprocessingConfig) -> List[np.ndarray]:
    """Candidate calls as arrays of their *member* frame indices.

    Port of her ``identify_suspected_frame_calls``: active frames are split wherever the
    gap between consecutive members exceeds ``merge_gap_s`` (4 frames), and groups with
    fewer than ``min_run_members`` members are dropped. Members are returned rather
    than (start, end) because her shape rejections are computed over the member frames
    only, not over the gaps between them.
    """
    idx = np.flatnonzero(activity.active)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > _frames(cfg.merge_gap_s, activity)) + 1)
    return [g for g in groups if len(g) >= cfg.min_run_members]


def _split_and_refine(groups: List[np.ndarray], activity: FrameActivity, cfg: PreprocessingConfig) -> List[np.ndarray]:
    """Use a learned model's onset / offset heads on each candidate (``learned_*`` fields).

    * **Split** [ours]: where an offset peak and, within ``learned_split_max_gap_s`` after
      it, an onset peak both reach ``learned_split_prob``, the candidate is two calls that
      the call head ran together.
    * **Refine** [ours]: each piece's first / last frame moves to the strongest onset /
      offset frame within ``learned_edge_search_s``, if that frame reaches
      ``learned_edge_min_prob``.

    Pieces are returned as contiguous frame ranges: after refinement the edge frames need
    not be members of the original run.
    """
    on, off = activity.onset_prob, activity.offset_prob
    assert on is not None and off is not None
    n = len(on)
    w = _frames(cfg.learned_edge_search_s, activity)
    gap = _frames(cfg.learned_split_max_gap_s, activity)
    out: List[np.ndarray] = []
    for g in groups:
        pieces, start = [], int(g[0])
        if cfg.learned_split_prob is not None:
            i = start
            while i < g[-1]:
                if off[i] >= cfg.learned_split_prob and off[i] == off[max(start, i - w):i + w + 1].max():
                    j_hi = min(int(g[-1]), i + gap)
                    later = np.arange(i + 1, j_hi + 1)
                    hits = later[on[later] >= cfg.learned_split_prob] if later.size else later
                    if hits.size:
                        j = int(hits[np.argmax(on[hits])])
                        pieces.append((start, i))
                        start, i = j, j
                        continue
                i += 1
        pieces.append((start, int(g[-1])))
        for a, b in pieces:
            lo, hi = max(0, a - w), min(n - 1, a + w)
            if on[lo:hi + 1].max() >= cfg.learned_edge_min_prob:
                a = lo + int(np.argmax(on[lo:hi + 1]))
            lo, hi = max(0, b - w), min(n - 1, b + w)
            if off[lo:hi + 1].max() >= cfg.learned_edge_min_prob:
                b = lo + int(np.argmax(off[lo:hi + 1]))
            if b > a:
                out.append(np.arange(a, b + 1))
    return out


def _rejected_by_shape(members: np.ndarray, activity: FrameActivity, cfg: PreprocessingConfig) -> bool:
    """Her two ``filter_calls`` shape rejections, ported from source.

    * **Bandwidth crossings.** Count how often the number of occupied bands crosses
      ``bandwidth_crossing_hz / bandwidth_bin_width_hz`` (= 1 band) across the members;
      reject at ``max_bandwidth_crossings`` (3) or more. Faithful to her use of
      ``np.roll``, which also compares the last member with the first.
    * **Abrupt up.** Reject a *short* candidate -- span under ``abrupt_up_max_span_s``
      (20 frames) -- whose ``freq_range`` exceeds ``abrupt_up_freq_range_hz`` (2 kHz)
      anywhere: a brief broadband click.

    Only the ``"talia"`` feature path populates the fields these need. The ``"spectral"``
    path has its own rejection instead:

    * **Not tonal** [ours]. With ``min_tonal_fraction`` set, reject a candidate if fewer
      than that fraction of the frames from its first to its last member (gaps included)
      are tonal (see ``filter_5khz.py``). Calls are narrow lines; the false alarms this
      removes are broadband cage noise in the same band (docs/DETECTOR_EVAL.md finding 9).

    The ``"learned"`` path has no shape rejection: the classifier was trained to reject.
    """
    if cfg.frame_features == "learned":
        return False  # the classifier already decided; its tonal flag is catalogue-only
    if activity.tonal is not None and cfg.min_tonal_fraction is not None:
        span = activity.tonal[members[0]:members[-1] + 1]
        if span.mean() < cfg.min_tonal_fraction:
            return True
    if activity.bandwidth_bins is None or activity.freq_range_hz is None:
        return False
    bands = activity.bandwidth_bins[members]
    thr = cfg.bandwidth_crossing_hz / cfg.bandwidth_bin_width_hz
    prev = np.roll(bands, 1)
    crossings = int(np.sum(((prev < thr) & (bands >= thr)) | ((prev >= thr) & (bands < thr))))
    if crossings >= cfg.max_bandwidth_crossings:
        return True
    span = int(members[-1] - members[0])
    return bool(span < _frames(cfg.abrupt_up_max_span_s, activity)
                and activity.freq_range_hz[members].max() > cfg.abrupt_up_freq_range_hz)


def activity_to_calls(activity: FrameActivity, cfg: PreprocessingConfig) -> List[CallSegment]:
    """Group active frames into calls, apply duration and shape rejections, then pad.

    As in her ``filter_calls`` / ``save_calls``: duration is tested on the unpadded frame
    span (``last - first``), strictly, against ``min_call_duration_s`` (10 frames) and
    ``max_call_duration_s`` (4 s); only survivors are padded by ``boundary_pad_s``. The
    call boundary is the first/last member's frame time. A learned model with edge heads
    first splits and refines the runs (:func:`_split_and_refine`).
    """
    calls: List[CallSegment] = []
    floor = _frames(cfg.min_call_duration_s, activity)
    runs = _active_runs(activity, cfg)
    if activity.onset_prob is not None and activity.offset_prob is not None:
        runs = _split_and_refine(runs, activity, cfg)
    for members in runs:
        span = int(members[-1] - members[0])
        if not (span > floor and span * activity.hop_s < cfg.max_call_duration_s):
            continue
        if _rejected_by_shape(members, activity, cfg):
            continue
        t0 = max(0.0, activity.frame_times_s[members[0]] - cfg.boundary_pad_s)
        t1 = activity.frame_times_s[members[-1]] + cfg.boundary_pad_s
        calls.append(CallSegment(float(t0), float(t1)))
    return calls


def in_night_window(t: dt.datetime, cfg: PreprocessingConfig) -> bool:
    """True if wall-clock time ``t`` lies in [night_start_hour, night_end_hour).

    The window may wrap midnight (19 -> 6 is the default); a non-wrapping window such
    as 0 -> 5 also works.
    """
    h = t.hour + t.minute / 60 + t.second / 3600
    lo, hi = cfg.night_start_hour, cfg.night_end_hour
    return (lo <= h or h < hi) if lo > hi else (lo <= h < hi)


def recording_entirely_at_night(start: dt.datetime, duration_s: float, cfg: PreprocessingConfig) -> bool:
    """True if every minute of a recording falls inside the night window.

    Checked on a one-minute grid (plus the last instant), which is exact for any
    window whose edges sit on whole minutes. Used to skip detection outright.
    """
    n = int(duration_s // 60)
    stamps = [start + dt.timedelta(minutes=m) for m in range(n + 1)]
    stamps.append(start + dt.timedelta(seconds=duration_s))
    return all(in_night_window(t, cfg) for t in stamps)


def drop_night_calls(
    calls: Sequence[CallSegment], start: dt.datetime, cfg: PreprocessingConfig
) -> List[CallSegment]:
    """Keep only calls whose onset (recording start + ``start_s``) is outside the night."""
    return [c for c in calls if not in_night_window(start + dt.timedelta(seconds=c.start_s), cfg)]


@dataclass
class ChunkPlan:
    """Where a chunk sits in its recording and which calls it holds, before any audio is read."""

    offset_s: float  # chunk start within the recording
    calls: List[CallSegment]  # relative to the chunk start


def plan_chunks(calls: Sequence[CallSegment], cfg: PreprocessingConfig, rec_len_s: float) -> List[ChunkPlan]:
    """Greedily pack calls into non-overlapping ``chunk_duration_s`` windows.

    Needs only the call times, so the corpus run can count chunks from the call catalogue
    and read audio for the chunks alone (``streaming.read_chunks``).
    """
    calls = sorted(calls, key=lambda c: c.start_s)
    plans: List[ChunkPlan] = []
    i = 0
    while i < len(calls):
        c0 = max(0.0, calls[i].start_s - cfg.chunk_pre_pad_s)
        if plans:  # never overlap the previous chunk
            c0 = max(c0, plans[-1].offset_s + cfg.chunk_duration_s)
        c1 = c0 + cfg.chunk_duration_s
        inside: List[CallSegment] = []
        j = i
        while j < len(calls) and calls[j].end_s <= c1:
            if calls[j].start_s >= c0:
                c = calls[j]
                inside.append(CallSegment(c.start_s - c0, c.end_s - c0, c.call_type, c.caller_id, c.receiver_id))
            j += 1
        if j == i:  # call starts in window but overruns it -> skip it
            j = i + 1
        i = j
        if len(inside) < cfg.min_calls_per_chunk or c0 >= rec_len_s:
            continue
        plans.append(ChunkPlan(c0, inside))
    return plans


def chunk_recording(
    audio_16k: np.ndarray,
    calls: Sequence[CallSegment],
    cfg: PreprocessingConfig,
    recording_id: str,
    start_time: dt.datetime,
    recording_day: Optional[int] = None,
) -> List[Chunk]:
    """:func:`plan_chunks`, then slice each window out of in-memory 16 kHz audio."""
    sr = cfg.target_sample_rate
    n_chunk = int(round(cfg.chunk_duration_s * sr))
    chunks: List[Chunk] = []
    for plan in plan_chunks(calls, cfg, len(audio_16k) / sr):
        s0 = int(round(plan.offset_s * sr))
        wav = audio_16k[s0 : s0 + n_chunk]
        if len(wav) < n_chunk:
            wav = np.pad(wav, (0, n_chunk - len(wav)))
        chunks.append(
            Chunk(
                waveform=wav.astype(np.float32),
                sample_rate=sr,
                recording_id=recording_id,
                start_time=start_time + dt.timedelta(seconds=plan.offset_s),
                offset_s=plan.offset_s,
                calls=plan.calls,
                recording_day=recording_day,
            )
        )
    return chunks


def label_calls_from_annotations(
    detected: Sequence[CallSegment], annotations: Sequence[CallSegment], min_overlap: float = 0.5
) -> List[CallSegment]:
    """Copy labels from ground-truth annotations onto detected calls (max-overlap match).

    Used only when labels exist (synthetic data, PHEEDB). A detected call gets a label
    when its overlap with the best annotation covers at least ``min_overlap`` of the
    shorter of the two.
    """
    out: List[CallSegment] = []
    for d in detected:
        best, best_ov = None, 0.0
        for a in annotations:
            ov = min(d.end_s, a.end_s) - max(d.start_s, a.start_s)
            if ov > best_ov:
                best, best_ov = a, ov
        if best is not None and best_ov >= min_overlap * min(d.duration_s, best.duration_s):
            out.append(CallSegment(d.start_s, d.end_s, best.call_type, best.caller_id, best.receiver_id))
        else:
            out.append(CallSegment(d.start_s, d.end_s))
    return out


def detection_scores(
    detected: Sequence[CallSegment], annotations: Sequence[CallSegment], min_overlap: float = 0.5
) -> dict:
    """Precision / recall of call detection (GmSLM reports P=0.975, R=0.78)."""

    def matched(x: CallSegment, pool: Sequence[CallSegment]) -> bool:
        return any(
            (min(x.end_s, p.end_s) - max(x.start_s, p.start_s)) >= min_overlap * min(x.duration_s, p.duration_s)
            for p in pool
        )

    tp_d = sum(matched(d, annotations) for d in detected)
    tp_a = sum(matched(a, detected) for a in annotations)
    return {
        "precision": tp_d / max(1, len(detected)),
        "recall": tp_a / max(1, len(annotations)),
        "n_detected": len(detected),
        "n_annotated": len(annotations),
    }
