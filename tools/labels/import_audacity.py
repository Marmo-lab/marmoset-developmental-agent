"""Import Audacity label exports into one hand-label CSV with absolute sample indices.

Audacity's "Export Labels" writes one line per label, ``start<TAB>end<TAB>text`` in
seconds; if a spectral selection was used, each label is followed by a line starting
with a backslash that carries the frequency range, which is ignored here. Label text
follows LABELING.md: ``c`` (call), ``?`` (unsure), ``n`` (non-call event); an empty
label is treated as ``c``.

``clips.csv`` (from make_clips.py) maps each clip to its source recording and offset,
so every label is also expressed as start/end sample indices at the raw file's native
rate -- the same coordinates as Talia's call_indicies.pkl and our detector output.

A clip with an exported but empty label file is kept as "labelled, no calls" (it shows
up in ``--summary`` with zero calls); a clip with no file at all is "not labelled yet".

Usage:
    python tools/labels/import_audacity.py labeling/clips.csv labeling/labels \
        -o data/labels/handlabels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

VALID = {"c", "?", "n"}
NATIVE_SR = 48_000


def read_audacity(path: Path) -> List[Tuple[float, float, str]]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("\\"):
            continue
        parts = line.split("\t")
        start, end = float(parts[0]), float(parts[1])
        text = (parts[2].strip().lower() if len(parts) > 2 else "") or "c"
        if text not in VALID:
            raise ValueError(f"{path.name}: label {text!r} not one of {sorted(VALID)}")
        if end < start:
            start, end = end, start
        out.append((start, end, text))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips_csv")
    ap.add_argument("labels_dir")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    clips: Dict[str, dict] = {r["clip_id"]: r for r in csv.DictReader(open(a.clips_csv))}
    rows, done, empty = [], [], []
    for cid, c in clips.items():
        f = Path(a.labels_dir) / f"{cid}.txt"
        if not f.exists():
            continue
        labels = read_audacity(f)
        done.append(cid)
        if not labels:
            empty.append(cid)
        off = float(c["offset_s"])
        for s, e, t in labels:
            rows.append({
                "clip_id": cid, "label": t, "clip_start_s": f"{s:.3f}", "clip_end_s": f"{e:.3f}",
                "source_path": c["source_path"],
                "abs_start_sample": int(round((off + s) * NATIVE_SR)),
                "abs_end_sample": int(round((off + e) * NATIVE_SR)),
                "day": c["day"],
            })

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["clip_id", "label", "clip_start_s", "clip_end_s", "source_path",
                                "abs_start_sample", "abs_end_sample", "day"])
        w.writeheader(); w.writerows(rows)
    with open(Path(a.out).with_suffix(".clips_done.txt"), "w") as fh:
        fh.write("\n".join(done) + "\n")

    n = {k: sum(r["label"] == k for r in rows) for k in sorted(VALID)}
    print(f"{len(done)}/{len(clips)} clips labelled ({len(empty)} with no calls); "
          f"labels: {n} -> {a.out}")
    missing = [cid for cid in clips if cid not in done]
    if missing:
        print(f"not labelled yet: {', '.join(missing)}")


if __name__ == "__main__":
    main()
