"""Merge a corpus run's per-recording call catalogues into one chronological call corpus.

The product since 2026-10-22 is individual calls with exact wall-clock times (SESSION_STATE).
``detect_corpus.py --export-calls`` writes ``calls/<recording_id>.csv`` and
``call_audio/<recording_id>.flac`` per recording, in whatever order the array tasks ran.
This step builds, in place and without copying audio:

``calls.tsv``
    One row per call, sorted by wall-clock start (guardrail 5; ties keep recording order).
    ``call_id`` is ``<recording_id>_<index>``. The audio is ``audio_path`` from
    ``audio_offset_s`` for ``audio_duration_s`` seconds; the call itself starts
    ``audio_lead_s`` into that slice and lasts ``duration_s`` (:func:`read_call`).
``calls_summary.json``
    Totals, calls and call-hours per recording date, and the duplicate copies dropped.

Each call gets a ``quality`` flag. ``short_low_conf`` marks a call shorter than
``SHORT_S`` whose detector confidence is under ``LOW_CONF`` [ours, 2026-10-25]. In the user's
listening check of the pilot, 17 of the 20 short or low-confidence calls were "not a call",
against 0 of 20 random calls. On the labelled clips the same rule removes 12 of 470
cross-validated detections and lowers recall from 0.933 to 0.918, so some of these are faint
real calls. They are therefore flagged, not deleted: 3.6% of the pilot's calls, 0.7% of its
call time. Training excludes them by default.

Duplicate copies on the shared drive are kept once, by ``merge_corpus.deduplicate``: same
file name and start time is one recording, and the copy with the most calls wins.

A recording without call audio (a catalogue-only run, or one still running) is an error,
since dropping it silently would leave a hole in the chronology.

Usage:
    python tools/corpus/merge_calls.py OUT
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from tools.corpus.merge_corpus import deduplicate

FIELDS = ["call_id", "recording_id", "start_time", "duration_s", "start_s", "end_s", "snr_db", "peak_freq_hz",
          "confidence", "quality", "notes", "audio_path", "audio_offset_s", "audio_duration_s", "audio_lead_s"]
SHORT_S = 0.18  # [ours] all 17 "not a call" verdicts were 0.11-0.17 s
LOW_CONF = 0.6  # [ours] ... with confidence 0.48-0.55


def quality(row: Dict[str, str]) -> str:
    """``short_low_conf`` or ``ok``; a missing confidence (rule detectors) is never low."""
    conf = float(row["confidence"]) if row.get("confidence") not in (None, "", "nan") else float("nan")
    return "short_low_conf" if float(row["duration_s"]) < SHORT_S and conf < LOW_CONF else "ok"


def merge(out: Path) -> Dict[str, object]:
    """Write ``out/calls.tsv`` and ``out/calls_summary.json``; return the summary."""
    files = [json.loads(p.read_text()) for p in (out / "files").glob("*.json")]
    rids, dropped = deduplicate(files)
    rows: List[Dict[str, str]] = []
    for order, rid in enumerate(rids):
        catalogue = list(csv.DictReader(open(out / "calls" / f"{rid}.csv")))
        if catalogue and not catalogue[0].get("audio_offset_s"):
            raise SystemExit(f"{rid}: no call audio; run detect_corpus.py with --export-calls")
        for k, r in enumerate(catalogue):
            rows.append({"call_id": f"{rid}_{k:05d}", "recording_id": rid, "audio_path": f"call_audio/{rid}.flac",
                         "quality": quality(r), "_order": order, **{f: r[f] for f in FIELDS if f in r}})
    rows.sort(key=lambda r: (r["start_time"], r["_order"]))
    with open(out / "calls.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    per_date: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0])
    for r in rows:
        d = per_date[r["start_time"][:10]]
        d[0] += 1
        d[1] += float(r["duration_s"]) / 3600
    summary = {
        "n_calls": len(rows), "call_hours": round(sum(v[1] for v in per_date.values()), 2),
        "n_short_low_conf": sum(r["quality"] == "short_low_conf" for r in rows),
        "n_recordings": len(rids), "duplicates_dropped": dropped,
        "first": rows[0]["start_time"] if rows else None, "last": rows[-1]["start_time"] if rows else None,
        "per_date": {d: {"calls": v[0], "call_hours": round(v[1], 3)} for d, v in sorted(per_date.items())},
    }
    (out / "calls_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def read_call(run: Path, row: Dict[str, str], with_margin: bool = False) -> np.ndarray:
    """One call's audio from its recording's FLAC (``calls.tsv`` row). Without margin, exactly
    the catalogued call; with it, the stored context on both sides as well."""
    path = run / row["audio_path"]
    sr = sf.info(path).samplerate
    off, n = float(row["audio_offset_s"]), float(row["audio_duration_s"])
    if not with_margin:
        off, n = off + float(row["audio_lead_s"]), float(row["duration_s"])
    with sf.SoundFile(path) as f:
        f.seek(int(round(off * sr)))
        return f.read(int(round(n * sr)), dtype="float32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="a detect_corpus.py run directory")
    s = merge(Path(ap.parse_args().out))
    print(f"{s['n_calls']} calls ({s['n_short_low_conf']} flagged short_low_conf), {s['call_hours']} h, {s['n_recordings']} recordings, "
          f"{len(s['duplicates_dropped'])} duplicate copies dropped; {s['first']} .. {s['last']}")


if __name__ == "__main__":
    main()
