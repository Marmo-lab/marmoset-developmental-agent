"""Short high calls (tsik, tsik-ek, seep-like) in our recordings, found without the learned detector.

Why. On the synthetic set the learned detector misses tsik, tsik-ek and seep (Turesson 2016
donors: about 0.1-0.3 s, peak 13-14 kHz) even at +20 dB, and none of the 761 hand labels is
both shorter than 0.2 s and peaked above 12 kHz (docs/SYNTH_EVAL.md, results). Either the
colony rarely makes such calls, or it does and nobody labelled them. A detector trained on
those labels cannot tell; a plain energy rule can propose candidates for the user to hear.

Rule [ours], on the 4-capsule mean (what the learned detector hears), per frame of a 1024-point
STFT with a 10 ms hop: power in 250 Hz bands (as the learned detector), each band in dB above
its median over the ``BLOCK_S`` block. ``high`` is the loudest 3-band (750 Hz) average inside
``HIGH_HZ``, ``low`` the loudest inside ``LOW_HZ``. A frame is active when ``high`` is at
least ``MIN_HIGH_DB`` and exceeds ``low`` by ``MIN_LEAD_DB``: energy concentrated above the
usual call band. A normal call (fundamental 4.6-9.7 kHz, harmonics above) fails the lead,
and so does a broadband click. Runs bridged over ``MAX_GAP_S``, kept between
``MIN_DUR_S`` and ``MAX_DUR_S``.

Steps:
* ``selftest SET``: the rule on a synthetic set (``tools/synth/generate.py``) at +20 and +6 dB;
  per call type, the share of planted calls it finds (any overlap), and its false alarms per
  hour on night background. It must find the short high types and leave the rest alone.
* ``scan RUN OUT`` (array): ``N_RECORDINGS`` day recordings (07:00-17:59, seeded) of a corpus
  run; every candidate, flagged when it overlaps a call of the run.
* ``select RUN OUT``: counts, and the strongest ``--n`` candidates (at most ``PER_RECORDING``
  per recording, missed by the run first) cut to 2 s clips for the library's ``/short_high``
  page, where the user listens.

* ``minutes RUN OUT``: training material for the learned detector (user, 2026-11-08). The 60
  recordings are halved (``generate.half``: the train half gives minutes, the test half is
  kept for evaluation). From each train recording, the ``MINUTES_PER_RECORDING`` minutes with
  the most candidates missed by the run, as mono FLACs (the 4-capsule mean), with
  ``positives.tsv``: every candidate and every run call in the minute, all calls (the user
  heard the candidates as calls). The rest of the minute is not labelled: it may hold calls
  nobody found, so training ignores it rather than calling it background.

Writes under OUT: ``selftest.json``, ``scan/<recording_id>.tsv``, ``candidates.tsv``,
``summary.json``, ``clips/<cand_id>.flac``, ``minutes/<minute_id>.flac`` and ``minutes/positives.tsv``.

Usage (on the cluster, via tools/cluster/short_high.sbatch):
    python tools/analysis/find_short_high.py selftest SET OUT
    python tools/analysis/find_short_high.py scan RUN OUT [--task-index I --n-tasks N]
    python tools/analysis/find_short_high.py select RUN OUT [--n 150]
    python tools/analysis/find_short_high.py minutes RUN OUT
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

SR = 48_000
HOP = 480
N_FFT = 1024
BAND_HZ = 250
HIGH_HZ = (11_500, 20_000)
LOW_HZ = (3_000, 10_500)
MIN_HIGH_DB = 15.0
MIN_LEAD_DB = 8.0
MAX_GAP_S = 0.03
MIN_DUR_S, MAX_DUR_S = 0.03, 0.5
BLOCK_S = 60
MARGIN_S = 1.0                                   # > MAX_DUR_S: a call at a block edge is whole in one block
DAY_HOURS = range(7, 18)
N_RECORDINGS = 60
PER_RECORDING = 5
MINUTES_PER_RECORDING = 5
CLIP_S = 2.0
FIELDS = ["cand_id", "recording_id", "start_s", "duration_s", "peak_hz", "high_db", "lead_db", "in_run_call"]


def frame_scores(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(times, high_db, low_db, peak_hz)`` per frame of one mono 48 kHz block."""
    f, t, z = signal.stft(x, SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
    p = np.abs(z) ** 2
    edges = np.arange(0, SR / 2 + 1, BAND_HZ)
    bands = np.stack([p[(f >= lo) & (f < hi)].mean(0) for lo, hi in zip(edges[:-1], edges[1:])])
    db = 10 * np.log10(bands + 1e-20)
    db -= np.median(db, axis=1, keepdims=True)
    sm = (db[:-2] + db[1:-1] + db[2:]) / 3                    # 750 Hz: a band and its two neighbours
    centre = (edges[1:-2] + edges[2:-1]) / 2

    def loudest(lo_hi: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        sel = (centre >= lo_hi[0]) & (centre < lo_hi[1])
        return sm[sel].max(0), centre[sel][sm[sel].argmax(0)]

    high, peak = loudest(HIGH_HZ)
    low, _ = loudest(LOW_HZ)
    return t, high, low, peak


def runs(t: np.ndarray, high: np.ndarray, low: np.ndarray, peak: np.ndarray) -> List[Dict]:
    """Active runs as candidates (start, duration, median peak, max high, median lead)."""
    act = np.flatnonzero((high >= MIN_HIGH_DB) & (high - low >= MIN_LEAD_DB))
    out = []
    if not act.size:
        return out
    gap = int(round(MAX_GAP_S * SR / HOP)) + 1
    for r in np.split(act, np.flatnonzero(np.diff(act) > gap) + 1):
        a, b = t[r[0]] - HOP / SR / 2, t[r[-1]] + HOP / SR / 2
        if MIN_DUR_S <= b - a <= MAX_DUR_S:
            out.append({"start_s": round(float(a), 3), "duration_s": round(float(b - a), 3),
                        "peak_hz": int(np.median(peak[r])), "high_db": round(float(high[r].max()), 1),
                        "lead_db": round(float(np.median(high[r] - low[r])), 1)})
    return out


def find(read: Callable[[int, int], np.ndarray], frames: int) -> List[Dict]:
    """Candidates over a whole recording; ``read(start, n)`` returns mono 48 kHz samples.

    Each block is read with ``MARGIN_S`` on both sides and keeps the runs that start inside
    it, so a call across a block edge is seen whole, once."""
    out: List[Dict] = []
    step, margin = BLOCK_S * SR, int(MARGIN_S * SR)
    for s0 in range(0, frames, step):
        a = max(0, s0 - margin)
        x = read(a, min(s0 + step + margin, frames) - a)
        if len(x) < SR:
            break
        for c in runs(*frame_scores(x)):
            c["start_s"] = round(c["start_s"] + a / SR, 3)
            if s0 / SR <= c["start_s"] < (s0 + step) / SR:
                out.append(c)
    return out


def _ov(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) > max(a0, b0)


def selftest(folder: Path, out: Path, levels: Sequence[str] = ("10.0", "2.0")) -> Dict:
    recs = [r for r in csv.DictReader(open(folder / "recordings.tsv"), delimiter="\t") if r["snr"] in levels]
    events: Dict[str, List[Dict]] = defaultdict(list)
    for e in csv.DictReader(open(folder / "events.tsv"), delimiter="\t"):
        events[e["rec_id"]].append(e)
    hits: Dict[Tuple[str, str], List[bool]] = defaultdict(list)
    fa: Dict[str, int] = defaultdict(int)
    for r in recs:
        x, sr = sf.read(folder / "audio" / f"{r['rec_id']}.flac", dtype="float32")
        cands = find(lambda s, n, x=x: x[s:s + n], len(x))
        ev = events[r["rec_id"]]
        for e in ev:
            a, b = float(e["onset_s"]), float(e["offset_s"])
            hits[(e["call_type"], r["snr"])].append(any(_ov(a, b, c["start_s"], c["start_s"] + c["duration_s"]) for c in cands))
        fa[r["snr"]] += sum(not any(_ov(float(e["onset_s"]), float(e["offset_s"]), c["start_s"], c["start_s"] + c["duration_s"])
                                    for e in ev) for c in cands)
    res = {"levels": list(levels), "n_recordings": len(recs),
           "found": {f"{t}@{snr}": round(float(np.mean(v)), 3) for (t, snr), v in sorted(hits.items())},
           "false_alarms_per_h": {snr: round(fa[snr] * 3600 / (60 * sum(r["snr"] == snr for r in recs)), 1) for snr in levels}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "selftest.json").write_text(json.dumps(res, indent=1))
    return res


def pool(run: Path, seed: int = 0) -> List[Dict]:
    files = sorted((json.loads(p.read_text()) for p in (run / "files").glob("*.json")), key=lambda f: f["recording_id"])
    days = [f for f in files if dt.datetime.fromisoformat(f["start_time"]).hour in DAY_HOURS and f["duration_s"] >= 600]
    return sorted(random.Random(seed).sample(days, min(N_RECORDINGS, len(days))), key=lambda f: f["recording_id"])


def run_calls(run: Path, rid: str) -> List[Tuple[float, float]]:
    p = run / "calls" / f"{rid}.csv"
    return [(float(c["start_s"]), float(c["end_s"])) for c in csv.DictReader(open(p))] if p.exists() else []


def scan(run: Path, out: Path, task: int, n_tasks: int) -> None:
    (out / "scan").mkdir(parents=True, exist_ok=True)
    for rec in pool(run)[task::n_tasks]:
        with sf.SoundFile(rec["path"]) as f:
            assert f.samplerate == SR, (rec["path"], f.samplerate)

            def read(s: int, n: int) -> np.ndarray:
                f.seek(s)
                return f.read(n, dtype="float32", always_2d=True).mean(axis=1)

            cands = find(read, f.frames)
        calls = run_calls(run, rec["recording_id"])
        rows = [{"cand_id": f"{rec['recording_id']}_{k:04d}", "recording_id": rec["recording_id"], **c,
                 "in_run_call": int(any(_ov(c["start_s"], c["start_s"] + c["duration_s"], a, b) for a, b in calls))}
                for k, c in enumerate(cands)]
        _write(out / "scan" / f"{rec['recording_id']}.tsv", rows)
        print(f"{rec['recording_id']}: {len(rows)} candidates, {sum(r['in_run_call'] for r in rows)} inside run calls", flush=True)


def _write(path: Path, rows: List[Dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def select(run: Path, out: Path, n: int) -> Dict:
    paths = {f["recording_id"]: f["path"] for f in pool(run)}
    rows = [r for p in sorted((out / "scan").glob("*.tsv")) for r in csv.DictReader(open(p), delimiter="\t")]
    hours = sum(sf.info(paths[rid]).duration for rid in {r["recording_id"] for r in rows} | set(
        p.stem for p in (out / "scan").glob("*.tsv"))) / 3600
    missed = [r for r in rows if r["in_run_call"] == "0"]
    per: Dict[str, int] = defaultdict(int)
    chosen = []
    for r in sorted(rows, key=lambda r: (r["in_run_call"], -float(r["high_db"]))):
        if per[r["recording_id"]] < PER_RECORDING and len(chosen) < n:
            per[r["recording_id"]] += 1
            chosen.append(r)
    (out / "clips").mkdir(exist_ok=True)
    for r in chosen:
        mid = float(r["start_s"]) + float(r["duration_s"]) / 2
        with sf.SoundFile(paths[r["recording_id"]]) as f:
            s = max(0, int((mid - CLIP_S / 2) * SR))
            f.seek(s)
            x = f.read(int(CLIP_S * SR), dtype="float32", always_2d=True).mean(axis=1)
        r["clip_offset_s"] = round(float(r["start_s"]) - s / SR, 3)
        sf.write(out / "clips" / f"{r['cand_id']}.flac", x, SR, subtype="PCM_24")
    with open(out / "candidates.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["clip_offset_s"], delimiter="\t")
        w.writeheader()
        w.writerows(chosen)
    summary = {"n_recordings": len(list((out / "scan").glob("*.tsv"))), "hours": round(hours, 1),
               "n_candidates": len(rows), "n_missed_by_run": len(missed),
               "per_hour": round(len(rows) / hours, 1) if hours else None,
               "missed_per_hour": round(len(missed) / hours, 1) if hours else None, "n_selected": len(chosen),
               "rule": {"high_hz": HIGH_HZ, "low_hz": LOW_HZ, "min_high_db": MIN_HIGH_DB, "min_lead_db": MIN_LEAD_DB,
                        "duration_s": (MIN_DUR_S, MAX_DUR_S)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def recording_half(run: Path, which: str) -> List[Dict]:
    """The scanned recordings of one half (``generate.half`` over their ids)."""
    from tools.synth.generate import half
    recs = pool(run)
    keep = set(half([r["recording_id"] for r in recs], which))
    return [r for r in recs if r["recording_id"] in keep]


def minutes(run: Path, out: Path) -> int:
    (out / "minutes").mkdir(parents=True, exist_ok=True)
    pos = []
    for rec in recording_half(run, "train"):
        p = out / "scan" / f"{rec['recording_id']}.tsv"
        cands = list(csv.DictReader(open(p), delimiter="\t")) if p.exists() else []
        per_min: Dict[int, int] = defaultdict(int)
        for c in cands:
            if c["in_run_call"] == "0":
                per_min[int(float(c["start_s"]) // 60)] += 1
        top = sorted(per_min, key=lambda m: -per_min[m])[:MINUTES_PER_RECORDING]
        calls = run_calls(run, rec["recording_id"])
        with sf.SoundFile(rec["path"]) as f:
            for m in sorted(top):
                f.seek(m * 60 * SR)
                x = f.read(60 * SR, dtype="float32", always_2d=True).mean(axis=1)
                if len(x) < 60 * SR:
                    continue
                mid = f"{rec['recording_id']}_m{m:02d}"
                sf.write(out / "minutes" / f"{mid}.flac", x, SR, subtype="PCM_24")
                spans = [(float(c["start_s"]), float(c["start_s"]) + float(c["duration_s"]), "candidate") for c in cands]
                spans += [(a, b, "run_call") for a, b in calls]
                for a, b, src in spans:
                    if _ov(a, b, m * 60, (m + 1) * 60):
                        pos.append({"minute_id": mid, "start_s": round(max(0.0, a - m * 60), 3),
                                    "end_s": round(min(60.0, b - m * 60), 3), "source": src})
    with open(out / "minutes" / "positives.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["minute_id", "start_s", "end_s", "source"], delimiter="\t")
        w.writeheader()
        w.writerows(pos)
    n = len(list((out / "minutes").glob("*.flac")))
    print(f"{n} minutes, {len(pos)} positive spans", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["selftest", "scan", "select", "minutes"])
    ap.add_argument("src", type=Path, help="a synthetic set (selftest) or a corpus run (scan, select)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    if a.step == "selftest":
        print(json.dumps(selftest(a.src, a.out), indent=1))
    elif a.step == "scan":
        scan(a.src, a.out, a.task_index, a.n_tasks)
    elif a.step == "select":
        print(json.dumps(select(a.src, a.out, a.n), indent=1))
    else:
        minutes(a.src, a.out)


if __name__ == "__main__":
    main()
