"""Candidate calls for a repertoire figure: strong, clean detections of varied kinds, cut for listening.

Why. A presentation figure shows a handful of the colony's call kinds as detected by the production
run. We do not classify calls; the kinds below are rough acoustic bins [ours] that only make the
candidate list varied. The user hears the candidates on the library's ``/review`` page, picks one per
kind and names it by ear.

Bins (duration s, peak frequency kHz, number of sub-segments in the run's ``notes``):
``CATEGORIES``. Within a bin: day hours (07-18), quality ``ok``, confidence >= ``MIN_CONF``, ranked by
SNR, at most one call per recording (variety), ``PER_CAT`` kept. Calls within ``PAD_S`` + ``EDGE_S`` of either
end of their recording file are left out: the clip would lack its margin, and a first try showed 17 of 72
top candidates ending exactly at a file's end (likely cut calls, a detector edge effect worth a later look). Each is cut from the raw recording,
the 4-capsule mean (what the detector hears), ``PAD_S`` on each side.

Writes OUT/review.json, OUT/items.tsv, OUT/clips/<call_id>.flac.

Usage: python tools/analysis/figure_candidates.py RUN OUT
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf

SR = 48_000
PAD_S = 0.4
PER_CAT = 12
MIN_CONF = 0.85
EDGE_S = 0.2
# name: (min dur, max dur, min peak kHz, max peak kHz, min segments, max segments)
CATEGORIES: Dict[str, Tuple[float, float, float, float, int, int]] = {
    "phee-like (long tonal)": (0.8, 3.0, 5.5, 10.0, 1, 2),
    "twitter-like (note series)": (0.4, 2.0, 6.0, 14.0, 4, 99),
    "trill-like (short tonal)": (0.25, 0.8, 5.0, 9.0, 1, 1),
    "short high (tsik-like)": (0.05, 0.3, 11.0, 20.0, 1, 2),
    "peep / chirp": (0.05, 0.25, 6.0, 10.5, 1, 1),
    "low / broadband": (0.1, 1.5, 2.0, 5.5, 1, 99),
}
QUESTION = {
    "title": "תמונת רפרטואר: בחירת קריאות",
    "intro": ("מועמדים לתמונה להנהלה, מקובצים לפי סוג גס (לפי משך ותדר, לא סיווג). פס כחול = הזיהוי של הגלאי. "
              "סמן ⭐ את היפה ביותר מכל סוג, ובשדה ההערה כתוב את שם הסוג כפי שאתה שומע אותו."),
    "options": [["pick", "⭐ בחירה לתמונה"], ["maybe", "אפשרי"], ["no", "לא"]],
}


def _segments(notes: str) -> int:
    return len([s for s in notes.split(";") if s]) if notes else 1


def pick(rows: List[Dict]) -> List[Dict]:
    out = []
    for name, (d0, d1, f0, f1, n0, n1) in CATEGORIES.items():
        pool = [r for r in rows if d0 <= float(r["duration_s"]) <= d1 and f0 * 1000 <= float(r["peak_freq_hz"]) <= f1 * 1000
                and n0 <= _segments(r["notes"]) <= n1]
        seen, kept = set(), []
        for r in sorted(pool, key=lambda r: -float(r["snr_db"])):
            if r["recording_id"] not in seen:
                seen.add(r["recording_id"])
                kept.append({**r, "category": name})
            if len(kept) == PER_CAT:
                break
        out += kept
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    length = {p.stem: json.loads(p.read_text())["duration_s"] for p in (a.run / "files").glob("*.json")}
    rows = [r for r in csv.DictReader(open(a.run / "calls.tsv"), delimiter="\t")
            if r["quality"] == "ok" and float(r["confidence"]) >= MIN_CONF and 7 <= int(r["start_time"][11:13]) <= 18
            and float(r["start_s"]) >= PAD_S + EDGE_S and float(r["end_s"]) <= length[r["recording_id"]] - PAD_S - EDGE_S]
    chosen = pick(rows)
    paths = {r: json.loads((a.run / "files" / f"{r}.json").read_text())["path"] for r in {c["recording_id"] for c in chosen}}
    (a.out / "clips").mkdir(parents=True, exist_ok=True)
    items = []
    for c in chosen:
        s, e = float(c["start_s"]), float(c["end_s"])
        t0 = max(0.0, s - PAD_S)
        with sf.SoundFile(paths[c["recording_id"]]) as f:
            f.seek(int(t0 * SR))
            x = f.read(int((e + PAD_S - t0) * SR), dtype="float32", always_2d=True).mean(axis=1)
        sf.write(a.out / "clips" / f"{c['call_id']}.flac", x, SR, subtype="PCM_24")
        items.append({"item_id": c["call_id"], "category": c["category"], "dur_s": c["duration_s"],
                      "peak_khz": round(float(c["peak_freq_hz"]) / 1000, 1), "snr_db": c["snr_db"],
                      "when": c["start_time"][:16], "start_s": round(t0, 4),
                      "marks": json.dumps([[round(s - t0, 3), round(e - t0, 3), "b", "detection"]])})
    with open(a.out / "items.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0]), delimiter="\t")
        w.writeheader()
        w.writerows(items)
    (a.out / "review.json").write_text(json.dumps(QUESTION | {"models": ["", "detection"]}, ensure_ascii=False, indent=1))
    print(f"{len(rows)} eligible calls; {len(items)} candidates", flush=True)


if __name__ == "__main__":
    main()
