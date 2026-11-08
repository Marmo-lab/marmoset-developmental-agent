"""Merges the retrained detector makes and production does not, on the hand-labelled clips, for listening.

Why. Retrained with short high calls (v2, docs/SYNTH_EVAL.md), the detector merges more
labelled calls into one detection on the cross-validated hand clips (5% to 13%). The hand
labels never marked short high calls, so a "merge" may join two labelled calls through a real
short call between them (v2 hears sound there and only joins wrongly) or through silence (v2
bridges a true gap). Only listening tells.

A case [ours]: a detection of model B (cross-validated, ``train_call_detector.py cv``) that
overlaps two or more labels of one day clip, where model A's detections do not join those
same labels. Each case is cut from the 4-capsule mean, ``PAD_S`` around the detection (at
most ``MAX_S``), with marks for the labels, B's detection and A's detections. As a hint, the
gaps between the merged labels are tested with the short-high-call energy rule
(``find_short_high.frame_scores`` / ``runs``): ``rule_in_gap`` is 1 when it fires inside one.

Writes a review folder for the library's ``/review`` page (``call_library.py --review``):
``review.json`` (question and answers), ``items.tsv`` (one row per case, marks as JSON in
clip seconds) and ``clips/<item_id>.flac``.

Usage:
    python tools/analysis/merge_review.py A_CV.json B_CV.json OUT [--names base v2] [--work WORK_DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from tools.analysis import find_short_high as fsh
from tools.analysis.train_call_detector import hand_sets

PAD_S = 0.6
MAX_S = 8.0
Span = Tuple[float, float]
QUESTION = {
    "title": "איחודים של הדטקטור החדש",
    "intro": ("כל כרטיס: גלאי v2 איחד שתי קריאות מתויגות (או יותר) לזיהוי אחד, והגלאי בייצור לא. "
              "פס טורקיז = התיוגים הידניים; כחול = הזיהוי של v2; אפור = הזיהויים של הייצור. "
              "השאלה: מה יש ברווח בין הקריאות המתויגות?"),
    "options": [["short_call", "✓ יש קריאה קצרה ברווח (v2 שמע נכון)"], ["silence", "✗ שקט / רעש: איחוד שגוי"],
                ["one_call", "זו בעצם קריאה אחת (התיוג פיצל)"], ["unsure", "? לא בטוח"]],
}


def _ov(a: Span, b: Span) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def load_labels(work: Optional[Path]) -> Tuple[Dict[str, List[Span]], Dict[str, Tuple[str, float, int]]]:
    """Labels per clip (``c`` and ``?``), and per clip its audio file, lead and day flag."""
    labels: Dict[str, List[Span]] = {}
    audio: Dict[str, Tuple[str, float, int]] = {}
    for clips_csv, labels_csv, audio_dir in hand_sets(work):
        for r in csv.DictReader(open(labels_csv)):
            labels.setdefault(r["clip_id"], []).append((float(r["clip_start_s"]), float(r["clip_end_s"])))
        for c in csv.DictReader(open(clips_csv)):
            audio[c["clip_id"]] = (str(Path(audio_dir) / f"{c['clip_id']}.npz"), 0.0, int(c["day"]))
    return labels, audio


def cases(a: Dict[str, List[Span]], b: Dict[str, List[Span]], labels: Dict[str, List[Span]],
          day: Dict[str, int]) -> List[Dict]:
    out = []
    for clip in sorted(b):
        if not day.get(clip, 0):
            continue
        labs = sorted(labels.get(clip, []))
        for det in b[clip]:
            joined = [lab for lab in labs if _ov(det, lab)]
            if len(joined) < 2:
                continue
            if any(all(_ov(d, lab) for lab in joined) for d in a.get(clip, [])):
                continue                                 # A joins them too: not a new merge
            gaps = [(p[1], q[0]) for p, q in zip(joined, joined[1:]) if q[0] > p[1]]
            out.append({"clip_id": clip, "det": det, "labels": joined, "gaps": gaps,
                        "a_dets": [d for d in a.get(clip, []) if _ov(d, (det[0] - PAD_S, det[1] + PAD_S))]})
    return out


def _dets(path: Path) -> Dict[str, List[Span]]:
    d = json.loads(path.read_text())["clips"]
    return {c: [tuple(x) for x in v["detections"]["learned"]["mean"]] for c, v in d.items()}


def build(a_json: Path, b_json: Path, out: Path, names: Tuple[str, str] = ("base", "v2"),
          work: Optional[Path] = None) -> int:
    labels, audio = load_labels(work)
    found = cases(_dets(a_json), _dets(b_json), labels, {k: v[2] for k, v in audio.items()})
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, c in enumerate(found):
        z = np.load(audio[c["clip_id"]][0])
        sr, lead = int(z["sr"]), float(z["lead_s"])
        x = z["capsules"].astype(np.float32).mean(axis=0)
        t0 = max(0.0, c["det"][0] - PAD_S)
        t1 = min(t0 + MAX_S, c["det"][1] + PAD_S)
        seg = x[int((lead + t0) * sr):int((lead + t1) * sr)]
        rule = 0
        for g0, g1 in c["gaps"]:
            i, j = int((lead + g0 - 0.05) * sr), int((lead + g1 + 0.05) * sr)
            if j - i > 2048 and fsh.runs(*fsh.frame_scores(x[i:j])):
                rule = 1
        item = f"{c['clip_id']}_{k:03d}"
        sf.write(out / "clips" / f"{item}.flac", seg, sr, subtype="PCM_24")
        marks = ([[a - t0, b - t0, "label", "תיוג"] for a, b in c["labels"]]
                 + [[c["det"][0] - t0, c["det"][1] - t0, "b", names[1]]]
                 + [[max(0.0, a - t0), min(t1 - t0, b - t0), "a", names[0]] for a, b in c["a_dets"]])
        rows.append({"item_id": item, "clip_id": c["clip_id"], "start_s": round(t0, 3),
                     "n_labels": len(c["labels"]), "max_gap_s": round(max((g1 - g0 for g0, g1 in c["gaps"]), default=0), 3),
                     "rule_in_gap": rule, "marks": json.dumps([[round(m[0], 3), round(m[1], 3), m[2], m[3]] for m in marks],
                                                              ensure_ascii=False)})
    with open(out / "items.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "clip_id", "start_s", "n_labels", "max_gap_s", "rule_in_gap", "marks"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    (out / "review.json").write_text(json.dumps(QUESTION | {"models": list(names)}, ensure_ascii=False, indent=1))
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a_json", type=Path)
    ap.add_argument("b_json", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--names", nargs=2, default=["base", "v2"])
    ap.add_argument("--work", type=Path)
    a = ap.parse_args()
    print(f"{build(a.a_json, a.b_json, a.out, tuple(a.names), a.work)} cases")


if __name__ == "__main__":
    main()
