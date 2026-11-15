"""SNR of the hand-labelled calls, measured the way the synthetic set sets it.

Why. The synthetic scores (``SYNTH_EVAL.md``) are reported per SNR level, and the corpus ``snr_db`` (peak bin over
its noise floor, ``streaming.py``) is a different measure, so it cannot say which level our colony's calls sit at.
This measures the real calls with the generator's own definition (``tools/synth/generate.py``): SNR = RMS of the
*call alone* over RMS of the background under it, both band-passed with ``generate.bandpass`` (5-20 kHz, 4th-order
Butterworth, zero phase), over the call's duration.

Real audio holds call and background together, so [ours]: P_seg = mean band power over the labelled span, P_bg =
mean band power of the clip's call-free audio within ``LOCAL_S`` of the call (labels widened by ``GUARD_S``; the
whole clip's call-free audio when that is shorter than ``MIN_BG_S``), and SNR = sqrt(max(P_seg / P_bg - 1, 0))
(the call's own power, assuming call and background add in power). A label at or below the background gets SNR 0
(-inf dB). ``snr_naive_db`` keeps P_seg / P_bg without the subtraction. ``snr_floor_db`` uses the background's
*floor* instead: the median power of its ``FRAME_S`` frames. In stationary noise (the synthetic night backgrounds) the
median frame power is about the mean, so the two agree; in day audio, transients beside a call (cage, other noise)
raise the mean but not the median. The synthetic set plants such transients separately (``plant_noise.py``), so the
floor is the like-for-like comparison with its SNR levels. Audio: the raw recording's 4-capsule mean,
what the detector hears and what the synthetic backgrounds were cut from (``backgrounds.py``).

Bias, stated: the labeller marks calls they hear, so quiet calls are under-represented here.

``summary`` maps each call to the nearest synthetic level (in dB, clipped to the range) and weights the
synthetic per-level recall by the real distribution: the synthetic score at our colony's own SNR mix.

Usage:
    python tools/analysis/real_snr.py measure OUT [--labels data/labels/*.csv]   (cluster: reads raw recordings)
    python tools/analysis/real_snr.py summary OUT [--scores analysis/synth_eval/v2test/scores_v2a.json]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf

from tools.synth.generate import SNR_LEVELS, bandpass

SR = 48_000
CLIP_S = 60.0
EDGE_S = 1.0          # extra audio read on each side so the zero-phase filter has no edge inside the clip
GUARD_S = 0.1         # call-free means at least this far from any label
LOCAL_S = 3.0
MIN_BG_S = 0.5
FRAME_S = 0.05
LABEL_SETS = ("data/labels/handlabels.csv", "data/labels/heldout_labels.csv", "data/labels/test_labels.csv")
FIELDS = ["set", "clip_id", "label", "start_s", "end_s", "duration_s", "bg_s", "bg_scope", "snr", "snr_db",
          "snr_naive_db", "snr_floor_db"]


def call_snr(bp: np.ndarray, spans: Sequence[Tuple[float, float]], k: int) -> Dict:
    """SNR of label ``k`` in a band-passed clip ``bp`` whose labels are ``spans`` (seconds)."""
    n = len(bp)
    free = np.ones(n, bool)
    for a, b in spans:
        free[max(0, int((a - GUARD_S) * SR)):int((b + GUARD_S) * SR)] = False
    a, b = spans[k]
    i0, i1 = int(round(a * SR)), int(round(b * SR))
    p_seg = float(np.mean(np.square(bp[i0:i1])))
    near = np.zeros(n, bool)
    near[max(0, int((a - LOCAL_S) * SR)):int((b + LOCAL_S) * SR)] = True
    bg, scope = free & near, "local"
    if bg.sum() < MIN_BG_S * SR:
        bg, scope = free, "clip"
    p_bg = float(np.mean(np.square(bp[bg])))
    snr = float(np.sqrt(max(p_seg / p_bg - 1.0, 0.0)))
    m = int(FRAME_S * SR)
    frames = [np.mean(np.square(bp[i:i + m])) for i in range(0, n - m + 1, m) if bg[i:i + m].all()]
    p_floor = float(np.median(frames)) if frames else p_bg
    floor = float(np.sqrt(max(p_seg / p_floor - 1.0, 0.0)))
    return {"bg_s": round(bg.sum() / SR, 2), "bg_scope": scope, "snr": round(snr, 4),
            "snr_db": round(20 * np.log10(snr), 2) if snr > 0 else float("-inf"),
            "snr_naive_db": round(10 * np.log10(p_seg / p_bg), 2),
            "snr_floor_db": round(20 * np.log10(floor), 2) if floor > 0 else float("-inf")}


def read_clip(path: str, start: int) -> Tuple[np.ndarray, int]:
    """``CLIP_S`` of the 4-capsule mean from ``start`` (samples), with ``EDGE_S`` on each side when available."""
    with sf.SoundFile(path) as f:
        s0 = max(0, start - int(EDGE_S * SR))
        f.seek(s0)
        x = f.read(start - s0 + int((CLIP_S + EDGE_S) * SR), dtype="float64", always_2d=True).mean(axis=1)
    return x, start - s0


def measure(labels: Sequence[Path], out: Path) -> None:
    rows = []
    for lf in labels:
        by_clip = defaultdict(list)
        for r in csv.DictReader(open(lf)):
            if r["label"] in ("c", "?"):
                by_clip[r["clip_id"]].append(r)
        for clip, rs in sorted(by_clip.items()):
            origin = int(np.median([int(r["abs_start_sample"]) - round(float(r["clip_start_s"]) * SR) for r in rs]))
            x, lead = read_clip(rs[0]["source_path"], origin)
            bp = bandpass(x)[lead:lead + int(CLIP_S * SR)]
            spans = [(float(r["clip_start_s"]), float(r["clip_end_s"])) for r in rs]
            for k, r in enumerate(rs):
                a, b = spans[k]
                rows.append({"set": lf.stem, "clip_id": clip, "label": r["label"], "start_s": a, "end_s": b,
                             "duration_s": round(b - a, 3), **call_snr(bp, spans, k)})
            print(f"{lf.stem} {clip}: {len(rs)} labels", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calls.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} labels -> {out / 'calls.tsv'}")


def nearest_level(db: float) -> float:
    levels = np.array([20 * np.log10(s) for s in SNR_LEVELS])
    return SNR_LEVELS[int(np.argmin(np.abs(levels - db)))] if np.isfinite(db) else min(SNR_LEVELS)


def _quant(v: List[float]) -> Dict[str, float]:
    return {f"p{q}": round(float(np.percentile(v, q)), 1) for q in (5, 10, 25, 50, 75, 90)}


def summary(out: Path, scores: Path) -> None:
    rows = list(csv.DictReader(open(out / "calls.tsv"), delimiter="\t"))
    res = {col: _summary(rows, scores, col) for col in ("snr_db", "snr_floor_db")}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def _summary(rows: List[Dict], scores: Path, col: str) -> Dict:
    sc = {float(k): v for k, v in json.loads(scores.read_text())["views"]["front_end"]["by_snr"].items()}
    res: Dict[str, Dict] = {}
    groups = {"all": rows, "c only": [r for r in rows if r["label"] == "c"],
              **{s: [r for r in rows if r["set"] == s] for s in sorted({r["set"] for r in rows})},
              "short (< 0.5 s)": [r for r in rows if float(r["duration_s"]) < 0.5],
              "long (>= 0.5 s)": [r for r in rows if float(r["duration_s"]) >= 0.5]}
    for name, rs in groups.items():
        db = [float(r[col]) for r in rs]
        finite = [d for d in db if np.isfinite(d)]
        share = defaultdict(int)
        for d in db:
            share[nearest_level(d)] += 1
        share = {f"{20 * np.log10(s):+.1f} dB": round(share[s] / len(db), 3) for s in SNR_LEVELS}
        w = {s: sum(nearest_level(d) == s for d in db) / len(db) for s in SNR_LEVELS}
        res[name] = {
            "n": len(rs), "snr_db": _quant(finite), "at_or_below_background": sum(not np.isfinite(d) for d in db),
            "below_+3.5_dB": round(sum(d < 20 * np.log10(1.5) for d in db) / len(db), 3),
            "naive_snr_db_median": round(float(np.median([float(r["snr_naive_db"]) for r in rs])), 1),
            "share_by_nearest_level": share,
            "synthetic_v2a_weighted": {k: round(sum(w[s] * sc[s][k] for s in SNR_LEVELS), 3)
                                       for k in ("recall", "recall_main", "precision_main", "recall_iou")},
        }
    res["synthetic_v2a_by_level"] = {f"{sc[k]['snr_db']:+.1f} dB": {m: round(sc[k][m], 3) for m in ("recall", "recall_main")}
                                     for k in sc}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("measure", "summary"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--labels", type=Path, nargs="*", default=[Path(p) for p in LABEL_SETS])
    ap.add_argument("--scores", type=Path, default=Path("analysis/synth_eval/v2test/scores_v2a.json"))
    a = ap.parse_args()
    measure(a.labels, a.out) if a.step == "measure" else summary(a.out, a.scores)


if __name__ == "__main__":
    main()
