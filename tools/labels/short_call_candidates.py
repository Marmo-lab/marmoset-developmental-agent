"""Short-call candidates for hand labelling: training material for a detector that catches what v2a misses.

Why (user, 2026-11-16). The production detector v2a misses 20-37% of short calls (tsik, seep, alarm: 0.1-0.3 s) on
the synthetic set. In our own recordings it still misses about 39% of the short high calls a plain energy rule finds
(``SYNTH_EVAL.md``). The hand labels hold almost none of them, and v2a met them only as other labs' donors and as
unchecked rule candidates. Real examples from our colony, checked by ear with their edges set by hand, are the
direct fix. v2a stays the production model; a model retrained on these is a candidate until it beats v2a on tests no
decision touched.

Candidates [ours]:
* **Rule.** ``tools/explorations/find_short_high.find``: the same energy rule, independent of any learned model.
* **Recordings.** ``N_RECORDINGS`` day recordings (07:00-17:59) of the corpus run, drawn with ``SEED``. Left out:
  every file a hand-label set came from (calibration, held-out, blind test), and the 60 recordings ``find_short_high``
  scanned for v2's training and evaluation. So nothing here overlaps an earlier training or test minute.
* **Selection.** ``N_MISSED`` candidates that no detection of the run overlaps (what v2a misses: the point), and
  ``N_CAUGHT`` that one does (controls: they show how often the rule's candidate is a call at all). Within each pool,
  spread over the candidate's level (``high_db`` terciles), so quiet candidates are not crowded out by loud ones, with
  at most ``PER_RECORDING`` per recording.
* **Halves.** The recordings are split with ``generate.half``. Labels from the ``test`` half are kept out of training
  and measure the retrained model on real short calls (``meta.tsv``; the page does not show it).

Each item is a ``CLIP_S`` clip of the 4-capsule mean centred on the candidate, with two strips: the rule's candidate
and the run's detections. On the page the user answers call / not a call / unsure and, for a call, clicks its start
and end on the spectrogram (``edit_bounds``).

Usage (cluster, tools/cluster/short_call_candidates.sbatch):
    python tools/labels/short_call_candidates.py scan RUN OUT [--task-index I --n-tasks N]
    python tools/labels/short_call_candidates.py select RUN OUT
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
from typing import Dict, List, Set

import numpy as np
import soundfile as sf

from tools.explorations.find_short_high import DAY_HOURS, find, run_calls
from tools.synth.generate import half

SR = 48_000
SEED = 2026_11_16
N_RECORDINGS = 80
N_MISSED = 320
N_CAUGHT = 80
PER_RECORDING = 6
CLIP_S = 2.0
LABEL_SETS = ("data/labels/handlabels.csv", "data/labels/heldout_labels.csv", "data/labels/test_labels.csv")
QUESTION = {
    "title": "קריאות קצרות: תיוג לאימון",
    "intro": ("כל פריט: 2 שניות סביב מועמד לקריאה קצרה וגבוהה. פס אפור = המועמד של כלל האנרגיה, פס כחול = מה שהגלאי "
              "v2a זיהה (לרוב אין). אם זו קריאה: סמן 'קריאה', ואז לחץ על הספקטרוגרמה בתחילת הקריאה ובסופה. "
              "גבולות לפי מה שנראה ונשמע. אם יש בקליפ קריאה נוספת, אין צורך לסמן אותה."),
    "options": [["call", "קריאה"], ["not", "לא קריאה"], ["unsure", "לא בטוח"]],
    "edit_bounds": True,
}


def excluded_paths(work: Path) -> Set[str]:
    """Raw files that any earlier hand-label set or the v2 short-high scan came from."""
    out = {r["source_path"] for p in LABEL_SETS for r in csv.DictReader(open(p))}
    scan = work / "short_high" / "scan"
    ids = {p.stem for p in scan.glob("*.tsv")} if scan.exists() else set()
    return out | {"id:" + i for i in ids}


def pool(run: Path, excluded: Set[str]) -> List[Dict]:
    files = sorted((json.loads(p.read_text()) for p in (run / "files").glob("*.json")), key=lambda f: f["recording_id"])
    days = [f for f in files if dt.datetime.fromisoformat(f["start_time"]).hour in DAY_HOURS and f["duration_s"] >= 600
            and f["path"] not in excluded and "id:" + f["recording_id"] not in excluded]
    return sorted(random.Random(SEED).sample(days, min(N_RECORDINGS, len(days))), key=lambda f: f["recording_id"])


def _ov(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) > max(a0, b0)


def _write(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def scan(run: Path, out: Path, work: Path, task: int, n_tasks: int) -> None:
    (out / "scan").mkdir(parents=True, exist_ok=True)
    for rec in pool(run, excluded_paths(work))[task::n_tasks]:
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
        if rows:
            _write(out / "scan" / f"{rec['recording_id']}.tsv", rows)
        print(f"{rec['recording_id']}: {len(rows)} candidates, {sum(r['in_run_call'] for r in rows)} inside run calls", flush=True)


def _spread(rows: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    """``n`` rows spread evenly over the three ``high_db`` terciles, at most ``PER_RECORDING`` per recording."""
    if not rows:
        return []
    cuts = np.percentile([float(r["high_db"]) for r in rows], [100 / 3, 200 / 3])
    bins: List[List[Dict]] = [[], [], []]
    for r in rows:
        bins[int(np.searchsorted(cuts, float(r["high_db"])))].append(r)
    for b in bins:
        rng.shuffle(b)
    per: Dict[str, int] = defaultdict(int)
    chosen: List[Dict] = []
    while len(chosen) < n and any(bins):
        for b in bins:
            while b and len(chosen) < n:
                r = b.pop()
                if per[r["recording_id"]] < PER_RECORDING:
                    per[r["recording_id"]] += 1
                    chosen.append(r)
                    break
    return chosen


def select(run: Path, out: Path, work: Path) -> None:
    recs = {f["recording_id"]: f for f in pool(run, excluded_paths(work))}
    rows = [r for p in sorted((out / "scan").glob("*.tsv")) for r in csv.DictReader(open(p), delimiter="\t")]
    rng = random.Random(SEED)
    chosen = (_spread([r for r in rows if r["in_run_call"] == "0"], N_MISSED, rng)
              + _spread([r for r in rows if r["in_run_call"] == "1"], N_CAUGHT, rng))
    rng.shuffle(chosen)                                   # missed and caught mixed, so the page does not give it away
    test_half = set(half(sorted(recs), "test"))
    (out / "clips").mkdir(exist_ok=True)
    items, meta = [], []
    for n, r in enumerate(chosen):
        item_id = f"sc_{n:03d}"
        start, dur = float(r["start_s"]), float(r["duration_s"])
        t0 = max(0.0, start + dur / 2 - CLIP_S / 2)
        with sf.SoundFile(recs[r["recording_id"]]["path"]) as f:
            f.seek(int(t0 * SR))
            x = f.read(int(CLIP_S * SR), dtype="float32", always_2d=True).mean(axis=1)
        sf.write(out / "clips" / f"{item_id}.flac", x, SR, subtype="PCM_24")
        clip_s = len(x) / SR
        marks = [[round(start - t0, 3), round(start + dur - t0, 3), "a", "candidate"]]
        marks += [[round(max(0.0, a - t0), 3), round(min(clip_s, b - t0), 3), "b", "v2a"]
                  for a, b in run_calls(run, r["recording_id"]) if _ov(a, b, t0, t0 + clip_s)]
        items.append({"item_id": item_id, "clip_s": round(clip_s, 3), "marks": json.dumps(marks)})
        meta.append({"item_id": item_id, "cand_id": r["cand_id"], "recording_id": r["recording_id"],
                     "file_path": recs[r["recording_id"]]["path"], "clip_start_s": round(t0, 4),
                     "missed_by_v2a": int(r["in_run_call"] == "0"), "high_db": r["high_db"], "peak_hz": r["peak_hz"],
                     "half": "test" if r["recording_id"] in test_half else "train"})
    _write(out / "items.tsv", items)
    _write(out / "meta.tsv", meta)
    (out / "review.json").write_text(json.dumps(QUESTION | {"models": ["candidate", "v2a"]}, ensure_ascii=False, indent=1))
    summary = {"n_recordings": len(recs), "n_scanned": len(list((out / "scan").glob("*.tsv"))), "n_candidates": len(rows),
               "n_missed_by_v2a": sum(r["in_run_call"] == "0" for r in rows), "n_items": len(items),
               "n_items_missed": sum(m["missed_by_v2a"] for m in meta), "n_items_test_half": sum(m["half"] == "test" for m in meta)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["scan", "select"])
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("WORK_DIR", ".")))
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args()
    if a.step == "scan":
        scan(a.run, a.out, a.work, a.task_index, a.n_tasks)
    else:
        select(a.run, a.out, a.work)


if __name__ == "__main__":
    main()
