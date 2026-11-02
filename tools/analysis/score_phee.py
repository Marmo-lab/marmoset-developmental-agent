"""Score the call detector against the lab's phee database: a phee-only, out-of-domain check.

The lab's phee database (``Omer-Lab-Shared/data/analysis/phee_database/pheeDB_*.csv``, columns
explained in ``Data base.txt`` there) lists phee calls from the two-monkey and artificial-marmoset
experiments, each with its experiment recording (``expWavFile``), the caller's microphone channel and
``OnsetMS`` / ``OffsetMS``. The newest snapshot, ``pheeDB_15122024_164547.csv``, holds the two-monkey
experiments only (46,100 phees: every phee of the January 2024 snapshot plus 947 more). This
gives a second, much larger reference than our hand labels, with three limits that set
what the score can mean:

* **Phee only.** Other call types (twitters, trills, chirps...) are not listed, so a
  detection that overlaps no phee is not necessarily a false alarm. Precision is therefore
  not reported; the score is phee *recall* plus boundary agreement. Detections that
  overlap no listed phee are counted per minute as "unlisted", an upper bound on false
  alarms, with the long ones (>= ``LONG_S``, phee-like) reported separately.
* **Other recordings.** The experiments are two monkeys in separate rooms, each with its
  own microphone: 96 kHz stereo, one close channel per monkey, not the colony room's
  48 kHz four-capsule recorder the detector was trained on. Detection runs on the caller's
  channel, resampled to 48 kHz by the usual front-end; this tests how the detector
  carries over to other audio.
* **Detected, not hand-marked.** ``Data base.txt``: the onsets and offsets are "as found
  in the detection" (the lab's phee detection, not hand labelling), so boundary errors are
  disagreements between two detectors.

A phee is detected if a detection on its caller's channel overlaps it (as
``score_detectors.score_clip``), and matched at IoU >= 0.5 as the stricter count;
``train_call_detector.match_breakdown`` sorts each phee into one-to-one, split, merged or
missed, and the signed onset / offset errors (detection minus phee) of the one-to-one
matches are reported. The other monkey's phees are also heard on a channel (crosstalk);
detections of those are counted as such, not as unlisted. Intervals are 95%
bootstrap intervals over experiments [ours], since calls within one experiment are not
independent.

Two steps, so the metrics can change without re-running the detector:

    # cluster (compute node): detect on a seeded sample of experiments
    python tools/analysis/score_phee.py detect DB.csv OUT [--exp-type phee2monkeys] [--n-exps 30]
        [--task-index I --n-tasks N]
    # anywhere: score the saved detections
    python tools/analysis/score_phee.py report OUT
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SHARE = "/ems/elsc-labs/omer-d/Omer-Lab-Shared"  # where the database's Z:\Omer-Lab-Shared lives on the cluster
LONG_S = 0.5  # [ours] phee-like length; the report prints the database's phee durations to check it
SHORT_S, LOW_CONF = 0.18, 0.6  # merge_calls.py: the short_low_conf flag
Interval = Tuple[float, float]


def cluster_path(windows_path: str) -> str:
    """``Z:\\Omer-Lab-Shared\\data\\...`` -> the same file on the cluster's share."""
    p = windows_path.replace("\\", "/")
    return SHARE + p[p.index("/Omer-Lab-Shared") + len("/Omer-Lab-Shared"):]


def read_db(path: Path, exp_type: str) -> Dict[str, Dict[str, object]]:
    """Experiments of ``exp_type`` marked legit: ``{expname: {"wav", "phees": [(ch, on_s, off_s, caller)]}}``."""
    exps: Dict[str, Dict[str, object]] = {}
    seen = set()
    for r in csv.DictReader(open(path, newline="")):
        if r["expType"].lower() != exp_type.lower() or r["legitExp"] != "1" or r["pheeID"] in seen:
            continue
        seen.add(r["pheeID"])
        e = exps.setdefault(r["expname"], {"wav": cluster_path(r["expWavFile"]), "phees": []})
        e["phees"].append((int(r["channel"]), float(r["OnsetMS"]) / 1000, float(r["OffsetMS"]) / 1000, r["caller"]))
    return exps


class _Channel:
    """One channel of a WAV as a mono ``AudioSource`` (streaming.FileSource, column ``ch``)."""

    def __init__(self, path: str, ch: int) -> None:
        from marmoset_developmental_agent.preprocessing.streaming import FileSource

        self._src, self._ch = FileSource(path), ch
        self.sample_rate, self.frames, self.channels = self._src.sample_rate, self._src.frames, 1

    def read(self, start: int, n: int) -> np.ndarray:
        return self._src.read(start, n)[:, [self._ch]]


def detect(db: Path, out: Path, exp_type: str, n_exps: int, seed: int, task_index: int, n_tasks: int) -> None:
    """Run the production detector on each channel of a seeded sample of experiments."""
    from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
    from marmoset_developmental_agent.preprocessing.streaming import detect_recording

    cfg = PreprocessingConfig()
    exps = read_db(db, exp_type)
    names = sorted(n for n, e in exps.items() if os.path.exists(e["wav"]))
    sample = sorted(random.Random(seed).sample(names, min(n_exps, len(names))))
    (out / "exps").mkdir(parents=True, exist_ok=True)
    (out / "sample.json").write_text(json.dumps({"db": str(db), "exp_type": exp_type, "seed": seed,
                                                 "available": len(names), "listed": len(exps), "sample": sample,
                                                 "frame_features": cfg.frame_features}, indent=1))
    for name in sample[task_index::n_tasks]:
        dest = out / "exps" / f"{name}.json"
        if dest.exists():
            continue
        t0, e = time.time(), exps[name]
        channels = sorted({p[0] for p in e["phees"]})
        dets: Dict[str, List[Tuple[float, float, float]]] = {}
        dur = 0.0
        for ch in channels:
            src = _Channel(str(e["wav"]), ch - 1)
            dur = src.frames / src.sample_rate
            dets[str(ch)] = [(r.start_s, r.end_s, r.confidence)
                             for r in detect_recording(src, dt.datetime(2000, 1, 1, 12), cfg)]
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps({"exp": name, "wav": e["wav"], "duration_s": dur, "phees": e["phees"],
                                   "detections": dets, "elapsed_s": time.time() - t0}))
        tmp.replace(dest)
        print(f"{name}: {dur / 60:.1f} min, {len(e['phees'])} phees, "
              f"{sum(len(v) for v in dets.values())} detections, {time.time() - t0:.0f} s", flush=True)


def _iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def score_exp(x: Dict[str, object], drop_short_low_conf: bool) -> Dict[str, List[float]]:
    """Per-experiment counts; lists so experiments concatenate before the bootstrap."""
    from tools.analysis.train_call_detector import _ov, match_breakdown

    out: Dict[str, List[float]] = defaultdict(list)
    all_phees = [(a, b) for _, a, b, _ in x["phees"]]
    for ch, dets in x["detections"].items():
        d = [(s, e) for s, e, c in dets if not (drop_short_low_conf and e - s < SHORT_S and c < LOW_CONF)]
        mine = [(a, b) for c, a, b, _ in x["phees"] if str(c) == ch]
        for p in mine:
            out["hit"].append(float(any(_ov(q, p) > 0 for q in d)))
            out["hit_iou"].append(float(any(_iou(q, p) >= 0.5 for q in d)))
            out["phee_s"].append(p[1] - p[0])
        for k, v in match_breakdown(d, mine, []).items():
            out[k] += v
        for p in mine:  # signed errors of one-to-one matches (match_breakdown keeps only magnitudes)
            hit = [q for q in d if _ov(q, p) > 0]
            if len(hit) == 1 and not any(m != p and _ov(hit[0], m) > 0 for m in mine):
                out["onset_signed"].append(hit[0][0] - p[0])
                out["offset_signed"].append(hit[0][1] - p[1])
        for q in d:
            if any(_ov(q, p) > 0 for p in mine):
                continue
            kind = "crosstalk" if any(_ov(q, p) > 0 for p in all_phees) else "unlisted"
            out[kind].append(q[1] - q[0])
    out["minutes"].append(x["duration_s"] / 60 * len(x["detections"]))
    return out


def _stats(per_exp: List[Dict[str, List[float]]]) -> Dict[str, float]:
    tot: Dict[str, List[float]] = defaultdict(list)
    for s in per_exp:
        for k, v in s.items():
            tot[k] += v
    n = max(1, len(tot["hit"]))
    minutes = sum(tot["minutes"]) or 1.0
    unl = np.array(tot["unlisted"])
    return {
        "phees": len(tot["hit"]), "R": float(np.mean(tot["hit"])) if tot["hit"] else float("nan"),
        "R_iou": float(np.mean(tot["hit_iou"])) if tot["hit_iou"] else float("nan"),
        "one_to_one": len(tot["one_to_one"]) / n, "split": len(tot["split"]) / n,
        "merged": len(tot["merged"]) / n, "missed": len(tot["missed"]) / n,
        "onset_med_ms": 1000 * float(np.median(tot["onset_signed"])) if tot["onset_signed"] else float("nan"),
        "offset_med_ms": 1000 * float(np.median(tot["offset_signed"])) if tot["offset_signed"] else float("nan"),
        "onset_abs_med_ms": 1000 * float(np.median(np.abs(tot["onset_signed"]))) if tot["onset_signed"] else float("nan"),
        "offset_abs_med_ms": 1000 * float(np.median(np.abs(tot["offset_signed"]))) if tot["offset_signed"] else float("nan"),
        "edges_50ms": float(np.mean([abs(a) <= 0.05 and abs(b) <= 0.05 for a, b in
                                     zip(tot["onset_signed"], tot["offset_signed"])])) if tot["onset_signed"] else float("nan"),
        "unlisted_per_min": len(unl) / minutes, "unlisted_long_per_min": float((unl >= LONG_S).sum()) / minutes,
        "crosstalk_per_min": len(tot["crosstalk"]) / minutes,
    }


def _bootstrap(per_exp: List[Dict[str, List[float]]], keys: List[str], n: int = 2000, seed: int = 0) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    draws = [_stats([per_exp[i] for i in rng.integers(0, len(per_exp), len(per_exp))]) for _ in range(n)]
    return {k: (float(np.nanpercentile([d[k] for d in draws], 2.5)), float(np.nanpercentile([d[k] for d in draws], 97.5)))
            for k in keys}


def report(out: Path) -> None:
    xs = [json.loads(p.read_text()) for p in sorted((out / "exps").glob("*.json"))]
    if not xs:
        raise SystemExit(f"no detections under {out}/exps; run the detect step first")
    meta = json.loads((out / "sample.json").read_text())
    lines = [f"phee database: {meta['db']}", f"type {meta['exp_type']}: {len(xs)} of {meta['available']} experiments "
             f"with audio (seed {meta['seed']}), {sum(len(x['phees']) for x in xs)} phees, "
             f"{sum(x['duration_s'] for x in xs) / 3600:.1f} h; detector {meta['frame_features']}", ""]
    durs = np.array([b - a for x in xs for _, a, b, _ in x["phees"]])
    lines.append(f"phee durations (s): p5 {np.percentile(durs, 5):.2f}  median {np.median(durs):.2f}  "
                 f"p95 {np.percentile(durs, 95):.2f}")
    result: Dict[str, object] = {"meta": meta, "n_exps": len(xs)}
    keys = ["R", "R_iou", "one_to_one", "edges_50ms", "unlisted_per_min", "unlisted_long_per_min"]
    for drop in (False, True):
        name = "without short_low_conf" if drop else "all detections"
        per = [score_exp(x, drop) for x in xs]
        s, ci = _stats(per), _bootstrap(per, keys)
        result[name] = {"stats": s, "ci95": ci}
        lines += ["", f"=== {name} ===",
                  f"phee recall (any overlap)  {s['R']:.3f}  [{ci['R'][0]:.3f}, {ci['R'][1]:.3f}]",
                  f"phee recall (IoU >= 0.5)   {s['R_iou']:.3f}  [{ci['R_iou'][0]:.3f}, {ci['R_iou'][1]:.3f}]",
                  f"one-to-one {s['one_to_one']:.3f}  split {s['split']:.3f}  merged {s['merged']:.3f}  "
                  f"missed {s['missed']:.3f}",
                  f"edges, detection minus phee (median ms): onset {s['onset_med_ms']:+.0f} (|{s['onset_abs_med_ms']:.0f}|)  "
                  f"offset {s['offset_med_ms']:+.0f} (|{s['offset_abs_med_ms']:.0f}|);  both within 50 ms "
                  f"{s['edges_50ms']:.3f}",
                  f"detections overlapping no listed phee: {s['unlisted_per_min']:.2f}/min "
                  f"(>= {LONG_S} s: {s['unlisted_long_per_min']:.2f}/min); the other monkey's phees: "
                  f"{s['crosstalk_per_min']:.2f}/min"]
    per = [score_exp(x, False) for x in xs]
    flat = [(d, h) for p in per for d, h in zip(p["phee_s"], p["hit"])]
    lines += ["", "recall by phee duration (all detections):"]
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)):
        h = [hit for d, hit in flat if lo <= d < hi]
        if h:
            lines.append(f"  {lo:g}-{hi:g} s: {np.mean(h):.3f}  (n={len(h)})")
    lines += ["", "per experiment (recall, phees):"]
    for x, p in zip(xs, per):
        lines.append(f"  {x['exp']}: {np.mean(p['hit']) if p['hit'] else float('nan'):.2f}  ({len(p['hit'])})")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    (out / "scores.json").write_text(json.dumps(result, indent=1))
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("db")
    d.add_argument("out")
    d.add_argument("--exp-type", default="phee2monkeys")
    d.add_argument("--n-exps", type=int, default=30)
    d.add_argument("--seed", type=int, default=20261101)
    d.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    d.add_argument("--n-tasks", type=int, default=1)
    r = sub.add_parser("report")
    r.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "detect":
        detect(Path(a.db), Path(a.out), a.exp_type, a.n_exps, a.seed, a.task_index, a.n_tasks)
    else:
        report(Path(a.out))


if __name__ == "__main__":
    main()
