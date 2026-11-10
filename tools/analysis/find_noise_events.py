"""Noise-event candidates in the colony recordings, for the user to hear and name (SYNTH_EVAL step 3).

Why. The synthetic set plants calls into call-free night background only. Real recordings also
hold loud non-call sounds (doors, people, cage rattles, beeps). To test the detector against
them, they must be found, heard and named first, then planted (``tools/synth/generate.py``,
later). Two pools [ours]:

* ``loud``: loud events the production run did *not* detect. Rule on the 4-capsule mean (what
  the detector hears), per frame of a 1024-point STFT with a 10 ms hop, power in 250 Hz bands,
  each band in dB above its median over the ``BLOCK_S`` block, smoothed over 3 bands (750 Hz),
  as ``find_short_high.frame_scores``. A frame is active when its loudest smoothed band inside
  ``BAND_HZ`` (1-20 kHz: the detector high-passes at 1 kHz; calls reach 20 kHz) is at least
  ``MIN_DB`` above median. Runs bridged over ``MAX_GAP_S``, kept between ``MIN_DUR_S`` and
  ``MAX_DUR_S``, dropped when within ``CALL_PAD_S`` of a run call. Per candidate: peak
  frequency and level (at its loudest frame) and width (the median over its frames of how many
  kHz are ``WIDE_DB`` above median: a door is wide, a beep narrow; the median, because an abrupt
  onset splatters even a pure tone across the spectrum for a frame or two). Calls the run missed will be among them, so "a call" is one
  of the answers.
* ``suspect``: run detections that may be noise the detector fires on: the longest
  (>= ``LONG_S``, the library's "long" flag) and the least confident (< ``LOW_CONF``).

Recordings: ``N_RECORDINGS`` drawn from the whole run (day and night, seeded), each tagged
with its ``generate.half`` half, so planted noises can later come from the held-out half only.

Steps (on the cluster, tools/cluster/noise_events.sbatch):
* ``scan RUN OUT`` (array): ``scan/<recording_id>.tsv``, every candidate of both pools.
* ``select RUN OUT [--n 300]``: a varied listening set: ``loud`` candidates stratified by peak
  frequency x duration (strongest first in each cell, at most ``PER_RECORDING`` per recording),
  plus ``N_SUSPECT`` suspects; cut to clips for the library's ``/review`` page
  (``review.json``, ``items.tsv``, ``clips/``) and ``summary.json`` (counts per hour).

Usage:
    python tools/analysis/find_noise_events.py scan RUN OUT [--task-index I --n-tasks N]
    python tools/analysis/find_noise_events.py select RUN OUT [--n 300]
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
from typing import Callable, Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

SR = 48_000
HOP = 480
N_FFT = 1024
BAND_W = 250
BAND_HZ = (1_000, 20_000)
MIN_DB = 15.0
WIDE_DB = 10.0
MAX_GAP_S = 0.05
MIN_DUR_S = 0.03
MAX_DUR_S = 5.0
CALL_PAD_S = 0.2
BLOCK_S = 60
MARGIN_S = 6.0                                   # > MAX_DUR_S: an event at a block edge is whole in one block
N_RECORDINGS = 80
PER_RECORDING = 4
LONG_S, LOW_CONF = 8.0, 0.6                      # merge_calls.quality / call_library
N_SUSPECT = 60
PAD_S, MAX_CLIP_S = 1.0, 8.0
FREQ_BINS = (1_000, 3_000, 6_000, 12_000, 20_001)
DUR_BINS = (0.0, 0.2, 1.0, MAX_DUR_S + 1)
FIELDS = ["cand_id", "pool", "recording_id", "half", "hour", "start_s", "duration_s", "peak_hz", "width_khz",
          "level_db", "confidence"]
QUESTION = {
    "title": "רעשים במושבה: מה זה?",
    "intro": ("כל כרטיס: אירוע חזק שהגלאי לא סימן כקריאה (פס כחול), או זיהוי חשוד של הגלאי: ארוך מאוד או בביטחון נמוך "
              "(גם הוא כחול). אפור = קריאות שהגלאי כן זיהה בסביבה. מה שומעים בפס הכחול?"),
    "options": [["door", "דלת / חבטה"], ["human", "דיבור / אדם"], ["beep", "צפצוף / מכשיר"],
                ["cage", "כלוב / שקשוק / גירוד"], ["other_noise", "רעש אחר"], ["call", "זו קריאה"],
                ["discard", "לא ברור / לזרוק"]],
}


def frame_scores(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(times, loudest_db, peak_hz, width_khz)`` per frame of one mono 48 kHz block."""
    f, t, z = signal.stft(x, SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
    p = np.abs(z) ** 2
    edges = np.arange(0, SR / 2 + 1, BAND_W)
    bands = np.stack([p[(f >= lo) & (f < hi)].mean(0) for lo, hi in zip(edges[:-1], edges[1:])])
    db = 10 * np.log10(bands + 1e-20)
    db -= np.median(db, axis=1, keepdims=True)
    sm = (db[:-2] + db[1:-1] + db[2:]) / 3
    centre = (edges[1:-2] + edges[2:-1]) / 2
    sel = (centre >= BAND_HZ[0]) & (centre < BAND_HZ[1])
    s = sm[sel]
    return t, s.max(0), centre[sel][s.argmax(0)], (s >= WIDE_DB).sum(0) * BAND_W / 1000


def runs(t: np.ndarray, loud: np.ndarray, peak: np.ndarray, width: np.ndarray) -> List[Dict]:
    act = np.flatnonzero(loud >= MIN_DB)
    out: List[Dict] = []
    if not act.size:
        return out
    gap = int(round(MAX_GAP_S * SR / HOP)) + 1
    for r in np.split(act, np.flatnonzero(np.diff(act) > gap) + 1):
        a, b = t[r[0]] - HOP / SR / 2, t[r[-1]] + HOP / SR / 2
        if MIN_DUR_S <= b - a <= MAX_DUR_S:
            k = r[np.argmax(loud[r])]
            out.append({"start_s": round(float(a), 3), "duration_s": round(float(b - a), 3), "peak_hz": int(peak[k]),
                        "width_khz": round(float(np.median(width[r])), 2), "level_db": round(float(loud[k]), 1)})
    return out


def find(read: Callable[[int, int], np.ndarray], frames: int) -> List[Dict]:
    """Candidates over a whole recording; each block read with ``MARGIN_S`` on both sides, runs kept once."""
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


def near(a: float, b: float, calls: List[Tuple[float, float, float]], pad: float) -> bool:
    return any(min(b, e + pad) > max(a, s - pad) for s, e, _ in calls)


def run_calls(run: Path, rid: str) -> List[Tuple[float, float, float]]:
    p = run / "calls" / f"{rid}.csv"
    return [(float(c["start_s"]), float(c["end_s"]), float(c["confidence"] or 1)) for c in csv.DictReader(open(p))] \
        if p.exists() else []


def pool(run: Path, seed: int = 3) -> List[Dict]:
    from tools.synth.generate import half
    files = sorted((json.loads(p.read_text()) for p in (run / "files").glob("*.json")), key=lambda f: f["recording_id"])
    files = [f for f in files if f["duration_s"] >= 600]
    recs = sorted(random.Random(seed).sample(files, min(N_RECORDINGS, len(files))), key=lambda f: f["recording_id"])
    train = set(half([r["recording_id"] for r in recs], "train"))
    for r in recs:
        r["half"] = "train" if r["recording_id"] in train else "test"
    return recs


def candidates(rec: Dict, loud: List[Dict], calls: List[Tuple[float, float, float]]) -> List[Dict]:
    """Both pools of one recording: ``loud`` runs away from run calls, and ``suspect`` run calls."""
    hour = dt.datetime.fromisoformat(rec["start_time"]).hour
    base = {"recording_id": rec["recording_id"], "half": rec["half"], "hour": hour}
    rows = [{**base, "pool": "loud", **c, "confidence": ""} for c in loud
            if not near(c["start_s"], c["start_s"] + c["duration_s"], calls, CALL_PAD_S)]
    rows += [{**base, "pool": "suspect", "start_s": round(s, 3), "duration_s": round(e - s, 3), "peak_hz": "",
              "width_khz": "", "level_db": "", "confidence": conf}
             for s, e, conf in calls if e - s >= LONG_S or conf < LOW_CONF]
    for k, r in enumerate(rows):
        r["cand_id"] = f"{rec['recording_id']}_{r['pool'][0]}{k:04d}"
    return rows


def scan(run: Path, out: Path, task: int, n_tasks: int) -> None:
    (out / "scan").mkdir(parents=True, exist_ok=True)
    for rec in pool(run)[task::n_tasks]:
        with sf.SoundFile(rec["path"]) as f:
            assert f.samplerate == SR, (rec["path"], f.samplerate)

            def read(s: int, n: int) -> np.ndarray:
                f.seek(s)
                return f.read(n, dtype="float32", always_2d=True).mean(axis=1)

            loud = find(read, f.frames)
        rows = candidates(rec, loud, run_calls(run, rec["recording_id"]))
        _write(out / "scan" / f"{rec['recording_id']}.tsv", rows, FIELDS)
        print(f"{rec['recording_id']}: {sum(r['pool'] == 'loud' for r in rows)} loud, "
              f"{sum(r['pool'] == 'suspect' for r in rows)} suspect", flush=True)


def _write(path: Path, rows: List[Dict], fields: List[str]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _cell(r: Dict) -> Tuple[int, int]:
    return (int(np.digitize(float(r["peak_hz"]), FREQ_BINS)), int(np.digitize(float(r["duration_s"]), DUR_BINS)))


def choose(rows: List[Dict], n: int, seed: int = 0) -> List[Dict]:
    """``loud``: round-robin over (frequency x duration) cells, strongest first in each, ``PER_RECORDING`` cap;
    ``suspect``: half the longest, half the least confident."""
    loud = [r for r in rows if r["pool"] == "loud"]
    cells: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for r in sorted(loud, key=lambda r: -float(r["level_db"])):
        cells[_cell(r)].append(r)
    per: Dict[str, int] = defaultdict(int)
    chosen: List[Dict] = []
    n_loud = n - N_SUSPECT
    while len(chosen) < n_loud and any(cells.values()):
        for key in sorted(cells):
            while cells[key]:
                r = cells[key].pop(0)
                if per[r["recording_id"]] < PER_RECORDING:
                    per[r["recording_id"]] += 1
                    chosen.append(r)
                    break
            if len(chosen) >= n_loud:
                break
    sus = [r for r in rows if r["pool"] == "suspect"]
    longest = sorted(sus, key=lambda r: -float(r["duration_s"]))[:N_SUSPECT // 2]
    rest = [r for r in sus if r not in longest]
    chosen += longest + sorted(rest, key=lambda r: float(r["confidence"]))[:N_SUSPECT - len(longest)]
    return chosen


def select(run: Path, out: Path, n: int) -> Dict:
    recs = {r["recording_id"]: r for r in pool(run)}
    rows = [r for p in sorted((out / "scan").glob("*.tsv")) for r in csv.DictReader(open(p), delimiter="\t")]
    scanned = {p.stem for p in (out / "scan").glob("*.tsv")}
    hours = sum(recs[rid]["duration_s"] for rid in scanned) / 3600
    chosen = choose(rows, n)
    (out / "clips").mkdir(exist_ok=True)
    items = []
    for r in chosen:
        a, d = float(r["start_s"]), float(r["duration_s"])
        t0 = max(0.0, a - PAD_S)
        t1 = min(t0 + MAX_CLIP_S, a + d + PAD_S)
        with sf.SoundFile(recs[r["recording_id"]]["path"]) as f:
            f.seek(int(t0 * SR))
            x = f.read(int((t1 - t0) * SR), dtype="float32", always_2d=True).mean(axis=1)
        sf.write(out / "clips" / f"{r['cand_id']}.flac", x, SR, subtype="PCM_24")
        calls = [c for c in run_calls(run, r["recording_id"]) if min(c[1], t1) > max(c[0], t0)
                 and not (r["pool"] == "suspect" and abs(c[0] - a) < 1e-3)]
        marks = [[round(a - t0, 3), round(min(t1, a + d) - t0, 3), "b", r["pool"]]]
        marks += [[round(max(0.0, s - t0), 3), round(min(t1, e) - t0, 3), "a", "run call"] for s, e, _ in calls]
        items.append({"item_id": r["cand_id"], "pool": r["pool"], "hour": r["hour"], "half": r["half"],
                      "dur_s": r["duration_s"], "peak_khz": round(float(r["peak_hz"]) / 1000, 1) if r["peak_hz"] else "",
                      "width_khz": r["width_khz"], "conf": r["confidence"], "marks": json.dumps(marks, ensure_ascii=False),
                      "recording_id": r["recording_id"], "start_s": r["start_s"]})
    fields = ["item_id", "pool", "hour", "half", "dur_s", "peak_khz", "width_khz", "conf", "marks", "recording_id", "start_s"]
    _write(out / "items.tsv", items, fields)
    (out / "review.json").write_text(json.dumps(QUESTION | {"models": ["run call", "candidate"]}, ensure_ascii=False, indent=1))
    loud = [r for r in rows if r["pool"] == "loud"]
    summary = {"n_recordings": len(scanned), "hours": round(hours, 1),
               "loud": len(loud), "loud_per_hour": round(len(loud) / hours, 1) if hours else None,
               "suspect": len(rows) - len(loud), "n_selected": len(items),
               "loud_by_cell": {f"f{k[0]}_d{k[1]}": v for k, v in sorted(_count(loud).items())},
               "rule": {"band_hz": BAND_HZ, "min_db": MIN_DB, "duration_s": (MIN_DUR_S, MAX_DUR_S), "call_pad_s": CALL_PAD_S}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _count(rows: List[Dict]) -> Dict[Tuple[int, int], int]:
    c: Dict[Tuple[int, int], int] = defaultdict(int)
    for r in rows:
        c[_cell(r)] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["scan", "select"])
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    if a.step == "scan":
        scan(a.run, a.out, a.task_index, a.n_tasks)
    else:
        print(json.dumps(select(a.run, a.out, a.n), indent=1))


if __name__ == "__main__":
    main()
