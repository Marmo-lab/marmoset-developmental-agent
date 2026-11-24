"""D2 and D6: does the detector's call confidence say which v6 calls are real? (a listening review, 2026-11-24)

D2. Every call carries ``confidence``, the detector's mean frame probability over the call. If low-confidence calls
are mostly not calls, a call-level filter (no re-run of the detector) would clean the corpus; if they are calls, the
frame threshold is already right. Printed first: the distribution of confidence over the v6 day calls, by duration,
and what share of calls and call time each cut would remove. Then ``N_BIN`` random day calls per confidence bin.
D6. ``short_low_conf`` (``merge_calls.py``: shorter than 0.18 s and confidence under 0.6) drops calls from the training
corpus; it was set from 17 verdicts. ``N_FLAG`` random flagged calls, and ``N_NEAR`` short calls just above the
confidence cut (0.6-0.7), to see whether the cut sits where calls begin.

Items are shuffled together and shown blind (no confidence, no group): the key is in ``key.tsv``, which the page does
not serve. Each clip is the loudest beam (as the training audio) ``PAD_S`` around the call, the call marked.
Writes a review folder for ``call_library.py --review``.

Usage (cluster): python tools/explorations/confidence_review.py RUN OUT
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from tools.explorations.ridge_listen import _beam

BINS = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
N_BIN, N_FLAG, N_NEAR = 10, 25, 15
PAD_S = 0.6
SEED = 2026_11_24
QUESTION = {
    "title": "האם זו קריאה?",
    "intro": ("כל כרטיס: קריאה אחת שהדטקטור מצא בקורפוס v6 (הפס הכחול), עם 0.6 שניות סביבה. "
              "הכרטיסים מעורבבים ואין רמז לביטחון של הדטקטור, כדי שהשמיעה תהיה עיוורת. "
              "השאלה: האם מה שבתוך הפס הוא קריאה של מרמוסט?"),
    "options": [["call", "✓ קריאה"], ["not", "✗ לא קריאה (רעש / כלוב / אחר)"], ["unsure", "? לא בטוח"]],
}


def summary(rows: List[Dict]) -> None:
    c = np.array([float(r["confidence"]) for r in rows])
    d = np.array([float(r["duration_s"]) for r in rows])
    print(f"v6 day calls: {len(c)}, call hours {d.sum() / 3600:.1f}")
    print("confidence percentiles 1/5/10/25/50:", np.round(np.percentile(c, [1, 5, 10, 25, 50]), 3))
    for lo, hi in BINS:
        s = (c >= lo) & (c < hi)
        print(f"  [{lo:.1f}, {hi:.1f}): {s.mean():6.1%} of calls, {d[s].sum() / d.sum():6.1%} of call time, "
              f"median duration {np.median(d[s]) if s.any() else float('nan'):.2f} s")
    for cut in (0.55, 0.6, 0.65, 0.7):
        s = c < cut
        print(f"  cut confidence < {cut}: removes {s.mean():.1%} of calls, {d[s].sum() / d.sum():.1%} of call time")
    for lo, hi in ((0, 0.18), (0.18, 0.5), (0.5, 2), (2, 1e9)):
        s = (d >= lo) & (d < hi)
        print(f"  duration [{lo}, {hi}) s: {s.mean():6.1%} of calls, confidence median {np.median(c[s]):.3f}, "
              f"< 0.6: {(c[s] < 0.6).mean():.1%}")


def main(run: Path, out: Path) -> None:
    rows = [r for r in csv.DictReader(open(run / "calls.tsv"), delimiter="\t")
            if 7 <= dt.datetime.fromisoformat(r["start_time"]).hour < 18 and r["confidence"] not in ("", "nan")]
    summary(rows)
    rng = random.Random(SEED)
    picks = []
    for lo, hi in BINS:
        pool = [r for r in rows if lo <= float(r["confidence"]) < hi]
        picks += [("D2", f"{lo:.1f}-{hi:.1f}", r) for r in rng.sample(pool, N_BIN)]
    flagged = [r for r in rows if r["quality"] == "short_low_conf"]
    near = [r for r in rows if float(r["duration_s"]) < 0.18 and 0.6 <= float(r["confidence"]) < 0.7]
    picks += [("D6", "flagged", r) for r in rng.sample(flagged, N_FLAG)]
    picks += [("D6", "near 0.6-0.7", r) for r in rng.sample(near, N_NEAR)]
    rng.shuffle(picks)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    items, key = [], []
    for n, (check, group, r) in enumerate(picks):
        info = json.loads((run / "files" / f"{r['recording_id']}.json").read_text())
        s, e = float(r["start_s"]), float(r["end_s"])
        with sf.SoundFile(info["path"]) as fh:
            sr = fh.samplerate
            lo = max(0.0, s - PAD_S)
            fh.seek(int(lo * sr))
            caps = fh.read(int((e + PAD_S - lo) * sr), dtype="float32", always_2d=True).T
        x = _beam(caps, sr, s - lo, e - lo)
        item = f"c{n:03d}"
        sf.write(out / "clips" / f"{item}.flac", x / max(float(np.abs(x).max()), 1e-6) * 0.5, sr, subtype="PCM_24")
        items.append({"item_id": item, "marks": json.dumps([[round(s - lo, 3), round(e - lo, 3), "b", "זיהוי"]],
                                                           ensure_ascii=False)})
        key.append({"item_id": item, "check": check, "group": group, "call_id": r["call_id"],
                    "confidence": r["confidence"], "duration_s": r["duration_s"], "snr_db": r["snr_db"],
                    "quality": r["quality"]})
    for name, data in (("items.tsv", items), ("key.tsv", key)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), delimiter="\t")
            w.writeheader()
            w.writerows(data)
    (out / "review.json").write_text(json.dumps(QUESTION, ensure_ascii=False, indent=1))
    print(f"{len(items)} items -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
