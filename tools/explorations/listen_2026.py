"""Listening check of the detector on the November 2026 recordings (96 kHz), before they join the v6 corpus.

Why (user, 2026-11-17). The 2026 folders are the colony recorder set to 96 kHz; the reader now accepts that rate and the
stream resamples to 48 kHz (``io/raw_reader.py``). A test shows a recording and its 1:2 decimation detect alike, but
not that the detector behaves on this room, these weeks and these animals. So: ``N_ITEMS`` detections of the v6 run
from the 2026 recordings, drawn with ``SEED``, spread over call-duration terciles and over day and night, each a clip
around the detection, for the review page. The user answers: a call with the right edges, a call with wrong edges, or
not a call.

Usage (cluster): python tools/explorations/listen_2026.py RUN OUT
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from marmoset_developmental_agent.preprocessing.filter_5khz import resample_audio

SEED = 2026_11_17
N_ITEMS = 30
SR_OUT = 48_000
PAD_S = 0.6
DAY_HOURS = range(7, 18)
QUESTION = {
    "title": "הקלטות 2026: בדיקת הגלאי",
    "intro": ("כל פריט: זיהוי אחד של הגלאי מההקלטות החדשות (96 קילו-הרץ), עם חצי שנייה לפניו ואחריו. "
              "הפס הכחול = הזיהוי. האם זו קריאה, והאם הגבולות נכונים?"),
    "options": [["ok", "קריאה, גבולות נכונים"], ["edges", "קריאה, גבולות שגויים"], ["not", "לא קריאה"],
                ["unsure", "לא בטוח"]],
}


def detections(run: Path) -> List[Dict]:
    out = []
    for p in sorted((run / "files").glob("Rec_2026_*.json")):
        f = json.loads(p.read_text())
        for r in csv.DictReader(open(run / "calls" / f"{f['recording_id']}.csv")):
            out.append({**r, "path": f["path"], "recording_id": f["recording_id"]})
    return out


def select(rows: List[Dict], rng: random.Random) -> List[Dict]:
    """Half day, half night where night has enough; within each, even over duration terciles."""
    dur = np.array([float(r["duration_s"]) for r in rows])
    cuts = np.percentile(dur, [100 / 3, 200 / 3])
    cells: Dict[tuple, List[Dict]] = {}
    for r, d in zip(rows, dur):
        day = dt.datetime.fromisoformat(r["start_time"]).hour in DAY_HOURS
        cells.setdefault((day, int(np.searchsorted(cuts, d))), []).append(r)
    for c in cells.values():
        rng.shuffle(c)
    chosen: List[Dict] = []
    while len(chosen) < N_ITEMS and any(cells.values()):
        for k in sorted(cells):
            if cells[k] and len(chosen) < N_ITEMS:
                chosen.append(cells[k].pop())
    return chosen


def build(run: Path, out: Path) -> None:
    rows = detections(run)
    chosen = select(rows, random.Random(SEED))
    (out / "clips").mkdir(parents=True, exist_ok=True)
    items, meta = [], []
    for n, r in enumerate(chosen):
        item_id = f"n26_{n:03d}"
        s, e = float(r["start_s"]), float(r["end_s"])
        with sf.SoundFile(r["path"]) as f:
            t0 = max(0.0, s - PAD_S)
            f.seek(int(t0 * f.samplerate))
            x = f.read(int((e - t0 + PAD_S) * f.samplerate), dtype="float32", always_2d=True).mean(axis=1)
            x = resample_audio(x, f.samplerate, SR_OUT)
        sf.write(out / "clips" / f"{item_id}.flac", x, SR_OUT, subtype="PCM_24")
        clip_s = len(x) / SR_OUT
        items.append({"item_id": item_id, "clip_s": round(clip_s, 3),
                      "marks": json.dumps([[round(s - t0, 3), round(min(clip_s, e - t0), 3), "b", "detector"]])})
        meta.append({"item_id": item_id, "recording_id": r["recording_id"], "start_time": r["start_time"],
                     "start_s": s, "duration_s": r["duration_s"], "confidence": r.get("confidence", ""),
                     "quality": r.get("quality", "")})
    for name, data in (("items.tsv", items), ("meta.tsv", meta)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), delimiter="\t")
            w.writeheader()
            w.writerows(data)
    (out / "review.json").write_text(json.dumps(QUESTION | {"models": ["detector"]}, ensure_ascii=False, indent=1))
    print(json.dumps({"n_detections": len(rows), "n_items": len(items),
                      "n_day": sum(dt.datetime.fromisoformat(m["start_time"]).hour in DAY_HOURS for m in meta)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    build(a.run, a.out)


if __name__ == "__main__":
    main()
