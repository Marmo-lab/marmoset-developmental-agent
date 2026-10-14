"""Build the recordings manifest by probing every raw WAV's header.

Walks ``AnimalRoomRecordings/Rec_*`` and writes the CSV that
``dataset_builder.load_recordings_from_manifest`` expects
(``path,start_time,recording_id``), plus format columns for QC. Each file's
``start_time`` comes from its broadcast-extension chunk, so chronological ordering is
exact rather than inferred from filenames.

Files that do not match the expected colony format (48 kHz, 4-channel, with a bext
timestamp) are written to a separate rejects CSV with the reason, never silently
dropped and never silently processed. The 250 kHz ultrasonic folder lands there.

Header-only: reads a few KB per file, not the 2 GB of audio, so the full corpus takes
minutes rather than hours.

Usage:
    python tools/build_manifest.py <raw-root> -o manifest.csv [-r rejects.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

from marmoset_developmental_agent.io.raw_reader import RawFileInfo, probe

FIELDS = ["path", "start_time", "recording_id", "sample_rate", "channels",
          "subtype", "frames", "duration_s", "folder"]


def _row(i: RawFileInfo) -> dict:
    return {
        "path": i.path,
        "start_time": i.start_time.isoformat() if i.start_time else "",
        "recording_id": i.recording_id,
        "sample_rate": i.sample_rate,
        "channels": i.channels,
        "subtype": i.subtype,
        "frames": i.frames,
        "duration_s": f"{i.duration_s:.3f}",
        "folder": Path(i.path).parent.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="AnimalRoomRecordings directory")
    ap.add_argument("-o", "--out", required=True, help="manifest CSV for accepted files")
    ap.add_argument("-r", "--rejects", default=None, help="CSV for rejected files")
    ap.add_argument("--glob", default="Rec_*/*.WAV")
    ap.add_argument("--limit", type=int, default=None, help="probe at most N files (smoke test)")
    a = ap.parse_args()

    paths = sorted(Path(a.root).glob(a.glob))
    if a.limit:
        paths = paths[: a.limit]
    if not paths:
        raise SystemExit(f"no files matched {a.glob!r} under {a.root}")
    print(f"probing {len(paths)} files ...", file=sys.stderr)

    ok: List[RawFileInfo] = []
    bad: List[RawFileInfo] = []
    for n, p in enumerate(paths, 1):
        info = probe(p)
        (ok if info.ok else bad).append(info)
        if n % 250 == 0:
            print(f"  {n}/{len(paths)}  ok={len(ok)} rejected={len(bad)}", file=sys.stderr)

    ok.sort(key=lambda i: (i.start_time or __import__("datetime").datetime.max, i.path))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(_row(i) for i in ok)

    if a.rejects:
        with open(a.rejects, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS + ["reason"])
            w.writeheader()
            w.writerows({**_row(i), "reason": i.reason} for i in bad)

    total_h = sum(i.duration_s for i in ok) / 3600
    print(f"\naccepted {len(ok)} files, {total_h:.0f} h -> {a.out}", file=sys.stderr)
    if ok:
        print(f"date range: {ok[0].start_time} .. {ok[-1].start_time}", file=sys.stderr)
    if bad:
        print(f"rejected {len(bad)} files -> {a.rejects or '(not written)'}", file=sys.stderr)
        seen: dict = {}
        for i in bad:
            seen[i.reason] = seen.get(i.reason, 0) + 1
        for reason, n in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"    {n:5d}  {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
