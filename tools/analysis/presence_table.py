"""Segment-level call presence, machine vs. human, in the format of Hasson et al.'s Table 1 (home pipeline).

Why. Their pipeline is scored by asking, per randomly sampled one-minute segment, "is X present?" for human and
machine, then precision, recall, F1, precision lift (precision / prevalence, i.e. over chance) and Cohen's kappa.
[paper: Hasson lab, "First 1000 Days", pipeline performance metrics]. We ask the same of our detector: "is a
marmoset call present?". Our labelled minutes are 60 s clips, as theirs, but nearly every day minute holds a call,
so the minute is nearly uninformative here; we also cut each minute into 10 s and 1 s segments. [ours]

A segment is human-positive when a ``c`` label overlaps it; segments where only ``?`` labels fall are left out
(the test protocol ignores ``?``). It is machine-positive when a detection overlaps it. Detections: v2a
cross-validated out-of-fold on the 45 calibration + held-out minutes (``analysis/detector_v2/cv_v2a``), and the
frozen v2a on the 15 blind test minutes (``outputs/testset_v2a/test_detections.json``, the single scoring of
2026-11-08; nothing is re-run or tuned here). Intervals: 95% bootstrap over minutes. [ours]

Synthetic rows: the held-out (test) half of the synthetic set (``analysis/synth_eval/v2test``: 1,800 one-minute
recordings, unseen donors planted into unseen night backgrounds at six SNRs, docs/SYNTH_EVAL.md), where the answer
is exact (the planted calls), scored the same way at 10 s and 1 s, overall and per SNR. No noise events are
planted in this set yet (SYNTH_EVAL step 3). [ours]

Usage:
    python tools/analysis/presence_table.py [--out analysis/presence_table]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

Span = Tuple[float, float]
SEG_S = (60.0, 10.0, 1.0)
CLIP_S = 60.0
N_BOOT = 2000


def labels(path: str) -> Dict[str, List[Tuple[float, float, str]]]:
    out: Dict[str, List[Tuple[float, float, str]]] = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["clip_id"], []).append((float(r["clip_start_s"]), float(r["clip_end_s"]), r["label"]))
    return out


def _hit(spans: List[Span], a: float, b: float) -> bool:
    return any(min(e, b) > max(s, a) for s, e in spans)


def segments(labs: List[Tuple[float, float, str]], dets: List[Span], seg: float) -> List[Tuple[int, int]]:
    """(human, machine) presence per segment of one minute; segments with only ``?`` are dropped."""
    c = [(s, e) for s, e, k in labs if k == "c"]
    q = [(s, e) for s, e, k in labs if k == "?"]
    out = []
    for k in range(int(round(CLIP_S / seg))):
        a, b = k * seg, (k + 1) * seg
        h = _hit(c, a, b)
        if not h and _hit(q, a, b):
            continue
        out.append((int(h), int(_hit(dets, a, b))))
    return out


def stats(pairs: np.ndarray) -> Dict[str, float]:
    h, m = pairs[:, 0], pairs[:, 1]
    tp, fp, fn = int((h & m).sum()), int((~h & m).sum()), int((h & ~m).sum())
    n = len(h)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    prev = h.mean()
    po = (h == m).mean()
    pe = h.mean() * m.mean() + (1 - h.mean()) * (1 - m.mean())
    return {"n": n, "prevalence": float(prev), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else float("nan"),
            "lift": p / prev if prev else float("nan"),
            "kappa": float((po - pe) / (1 - pe)) if pe < 1 else float("nan")}


def table(per_clip: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    ids = sorted(per_clip)
    point = stats(np.concatenate([per_clip[c] for c in ids]).astype(bool))
    rng = np.random.default_rng(0)
    boots = [stats(np.concatenate([per_clip[c] for c in rng.choice(ids, len(ids))]).astype(bool)) for _ in range(N_BOOT)]
    ci = {k: [float(np.nanpercentile([b[k] for b in boots], q)) for q in (2.5, 97.5)]
          for k in ("precision", "recall", "f1", "lift", "kappa")}
    return {"minutes": len(ids), **point, "ci95": ci}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("analysis/presence_table"))
    a = ap.parse_args()
    cv = {c: [tuple(x[:2]) for x in v["detections"]["learned"]["mean"]]
          for c, v in json.load(open("analysis/detector_v2/cv_v2a/learned_detections.json"))["clips"].items()}
    test = {c: [tuple(x[:2]) for x in v] for c, v in json.load(open("outputs/testset_v2a/test_detections.json"))["v2a"].items()}
    lab = {**labels("data/labels/handlabels.csv"), **labels("data/labels/heldout_labels.csv"), **labels("data/labels/test_labels.csv")}
    day = {r["clip_id"]: int(r["day"]) for f in ("labeling/clips.csv", "labeling/heldout/clips.csv", "labeling/test/clips.csv")
           for r in csv.DictReader(open(f))}
    sets = {"blind test (15 min, frozen v2a)": test, "cross-validated (45 min, out-of-fold v2a)": cv,
            "cross-validated, day minutes only": {c: d for c, d in cv.items() if day[c]},
            "cross-validated, night minutes only": {c: d for c, d in cv.items() if not day[c]}}
    res: Dict[str, Dict] = {}
    for name, dets in sets.items():
        for seg in SEG_S:
            pc = {c: np.array(segments(lab.get(c, []), dets[c], seg), dtype=int).reshape(-1, 2) for c in dets}
            res[f"{name} | {seg:g} s"] = table(pc)
    syn = Path("analysis/synth_eval/v2test")
    snr = {r["rec_id"]: float(r["snr_db"]) for r in csv.DictReader(open(syn / "recordings.tsv"), delimiter="\t")}
    syn_lab: Dict[str, List[Tuple[float, float, str]]] = {k: [] for k in snr}
    for r in csv.DictReader(open(syn / "events.tsv"), delimiter="\t"):
        syn_lab[r["rec_id"]].append((float(r["onset_s"]), float(r["offset_s"]), "c"))
    syn_det: Dict[str, List[Span]] = {k: [] for k in snr}
    for r in csv.DictReader(open(syn / "detections_v2a" / "all.tsv"), delimiter="\t"):
        syn_det[r["rec_id"]].append((float(r["start_s"]), float(r["end_s"])))
    groups = {"synthetic, all SNRs": sorted(snr)} | {f"synthetic, {db:+g} dB": [k for k in snr if snr[k] == db]
                                                    for db in sorted(set(snr.values()), reverse=True)}
    for name, ids in groups.items():
        for seg in SEG_S[1:]:
            pc = {c: np.array(segments(syn_lab[c], syn_det[c], seg), dtype=int).reshape(-1, 2) for c in ids}
            res[f"{name} | {seg:g} s"] = table(pc)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "presence_table.json").write_text(json.dumps(res, indent=1))
    print(f"{'':<52}{'n':>6}{'prev':>6}{'P':>6}{'R':>6}{'F1':>6}{'lift':>6}{'kappa':>7}")
    for k, r in res.items():
        print(f"{k:<52}{r['n']:>6}{r['prevalence']:>6.2f}{r['precision']:>6.2f}{r['recall']:>6.2f}{r['f1']:>6.2f}"
              f"{r['lift']:>6.2f}{r['kappa']:>7.2f}")


if __name__ == "__main__":
    main()
