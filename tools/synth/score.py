"""Score the production detector on a synthetic set: every planted call is a known answer.

Step 5 of the synthetic evaluation (docs/SYNTH_METHODS.md section 6).

``detect`` (cluster, array): the production front-end (``load_config()``, the learned
detector at its threshold) on each recording of the set, as a mono 48 kHz file. Night
exclusion is off: the recordings carry no wall-clock time. Each detection keeps its
``merge_calls`` quality flag (``short_low_conf``: shorter than 0.18 s and confidence under
0.6, excluded from training; ``long``: 8 s or more, kept). Writes ``SET/detections/<task>.tsv``.

``score`` (anywhere): planted calls against detections, per recording.

* **IoU >= 0.5, one to one**: greedy, best pairs first, the rule of ``DETECTOR_EVAL.md``
  (``tools/analysis/score_detectors.py``, ``score_clip``). Recall_iou and precision_iou.
* **Overlap**: a planted call is found if any detection overlaps it; a detection is right
  if it overlaps any planted call. Tolerant of split or merged calls.
* **Main** [ours, with the user, 2026-11-05]: IoU >= 0.5, except that a phee (any
  ``SPLIT_OK`` type) counts as found when any detection overlaps it, and every detection
  overlapping it counts as right. The detector splits multi-note phees at the note gaps; the
  notes still reach the corpus, which is what the developmental agent needs.
* **False alarms per hour**: detections that overlap no planted call, per hour of audio.
* **Edges**: median |onset| and |offset| error (ms) of the IoU matches.

Two views of the detections: ``detector`` (all of them) and ``front_end`` (without
``short_low_conf``, what reaches the training set). Splits: SNR level; call type x SNR
for recall. Writes ``SET/scores.json`` and ``SET/scores.png``.

Usage (on the cluster, via tools/cluster/synth_score.sbatch):
    python tools/synth/score.py detect SET [--task-index I --n-tasks N]
    python tools/synth/score.py score SET
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

SHORT_S, LOW_CONF, LONG_S = 0.18, 0.6, 8.0  # merge_calls.quality
SPLIT_OK = ("phee",)  # call-type prefixes scored by overlap in the main view
Interval = Tuple[float, float]


def quality(duration_s: float, confidence: float) -> str:
    return "short_low_conf" if duration_s < SHORT_S and confidence < LOW_CONF else "long" if duration_s >= LONG_S else "ok"


def detect(folder: Path, task: int, n_tasks: int) -> None:
    from marmoset_developmental_agent.config.agent_config import load_config
    from marmoset_developmental_agent.preprocessing.streaming import FileSource, detect_recording

    cfg = load_config().preprocessing
    recs = list(csv.DictReader(open(folder / "recordings.tsv"), delimiter="\t"))[task::n_tasks]
    rows = []
    for r in recs:
        for d in detect_recording(FileSource(str(folder / "audio" / f"{r['rec_id']}.flac")), dt.datetime(2000, 1, 1, 12), cfg):
            rows.append({"rec_id": r["rec_id"], "start_s": round(d.start_s, 4), "end_s": round(d.end_s, 4),
                         "confidence": round(d.confidence, 3), "quality": quality(d.duration_s, d.confidence)})
    (folder / "detections").mkdir(exist_ok=True)
    tmp = folder / "detections" / f"{task:03d}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rec_id", "start_s", "end_s", "confidence", "quality"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, folder / "detections" / f"{task:03d}.tsv")
    print(f"task {task}: {len(recs)} recordings, {len(rows)} detections", flush=True)


def _ov(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match_iou(dets: Sequence[Interval], calls: Sequence[Interval]) -> List[Tuple[int, int]]:
    """(detection, call) pairs at IoU >= 0.5, greedy one to one, best pairs first (as ``score_clip``)."""
    pairs = sorted(((_ov(d, c) / (max(d[1], c[1]) - min(d[0], c[0])), i, j)
                    for i, d in enumerate(dets) for j, c in enumerate(calls) if _ov(d, c) > 0), reverse=True)
    used_d, used_c, out = set(), set(), []
    for iou, i, j in pairs:
        if iou < 0.5:
            break
        if i not in used_d and j not in used_c:
            used_d.add(i)
            used_c.add(j)
            out.append((i, j))
    return out


def split_ok(call_type: str) -> bool:
    return call_type.startswith(SPLIT_OK)


def score_recording(dets: Sequence[Interval], calls: Sequence[Interval],
                    splittable: Sequence[bool] = ()) -> Dict[str, object]:
    """Per call: found by overlap, at IoU, in the main view, edge errors; per detection: the same.

    ``splittable[j]`` marks a call scored by overlap in the main view (a phee; see above)."""
    splittable = list(splittable) or [False] * len(calls)
    m = dict((j, i) for i, j in match_iou(dets, calls))
    hit = [any(_ov(c, d) > 0 for d in dets) for c in calls]
    matched = set(m.values())
    return {
        "call_hit": hit,
        "call_iou": [j in m for j in range(len(calls))],
        "call_main": [j in m or (splittable[j] and hit[j]) for j in range(len(calls))],
        "det_main": [i in matched or any(sp and _ov(d, c) > 0 for c, sp in zip(calls, splittable))
                     for i, d in enumerate(dets)],
        "onset_err": [abs(dets[m[j]][0] - calls[j][0]) for j in range(len(calls)) if j in m],
        "offset_err": [abs(dets[m[j]][1] - calls[j][1]) for j in range(len(calls)) if j in m],
        "det_hit": [any(_ov(d, c) > 0 for c in calls) for d in dets],
        "n_iou": len(m),
    }


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def score(folder: Path) -> Dict:
    recs = list(csv.DictReader(open(folder / "recordings.tsv"), delimiter="\t"))
    events: Dict[str, List[Dict]] = defaultdict(list)
    for e in csv.DictReader(open(folder / "events.tsv"), delimiter="\t"):
        events[e["rec_id"]].append(e)
    dets: Dict[str, List[Dict]] = defaultdict(list)
    for p in sorted((folder / "detections").glob("*.tsv")):
        for d in csv.DictReader(open(p), delimiter="\t"):
            dets[d["rec_id"]].append(d)
    seconds = {r["rec_id"]: 60.0 for r in recs}
    out: Dict = {"n_recordings": len(recs), "n_calls": sum(len(v) for v in events.values()), "views": {}}
    for view in ("detector", "front_end"):
        by_snr: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        by_type: Dict[Tuple[str, str], List[bool]] = defaultdict(list)
        by_type_any: Dict[Tuple[str, str], List[bool]] = defaultdict(list)
        by_type_main: Dict[Tuple[str, str], List[bool]] = defaultdict(list)
        for r in recs:
            ds = [(float(d["start_s"]), float(d["end_s"])) for d in dets[r["rec_id"]]
                  if view == "detector" or d["quality"] != "short_low_conf"]
            ev = events[r["rec_id"]]
            s = score_recording(ds, [(float(e["onset_s"]), float(e["offset_s"])) for e in ev],
                                [split_ok(e["call_type"]) for e in ev])
            b = by_snr[r["snr"]]
            for k in ("call_hit", "call_iou", "call_main", "det_main", "onset_err", "offset_err", "det_hit"):
                b[k] += s[k]
            b["n_iou"].append(s["n_iou"])
            b["seconds"].append(seconds[r["rec_id"]])
            for e, hit, iou, main in zip(ev, s["call_hit"], s["call_iou"], s["call_main"]):
                by_type[(e["call_type"], r["snr"])].append(iou)
                by_type_any[(e["call_type"], r["snr"])].append(hit)
                by_type_main[(e["call_type"], r["snr"])].append(main)
        levels = {}
        for snr, b in sorted(by_snr.items(), key=lambda kv: -float(kv[0])):
            fa = sum(not h for h in b["det_hit"])
            levels[snr] = {
                "snr_db": round(20 * np.log10(float(snr)), 2), "n_calls": len(b["call_hit"]), "n_det": len(b["det_hit"]),
                "recall_main": _div(sum(b["call_main"]), len(b["call_main"])),
                "precision_main": _div(sum(b["det_main"]), len(b["det_main"])),
                "recall": _div(sum(b["call_hit"]), len(b["call_hit"])), "precision": _div(sum(b["det_hit"]), len(b["det_hit"])),
                "recall_iou": _div(sum(b["call_iou"]), len(b["call_iou"])), "precision_iou": _div(sum(b["n_iou"]), len(b["det_hit"])),
                "false_alarms_per_h": fa * 3600 / sum(b["seconds"]),
                "onset_err_ms": 1000 * float(np.median(b["onset_err"])) if b["onset_err"] else float("nan"),
                "offset_err_ms": 1000 * float(np.median(b["offset_err"])) if b["offset_err"] else float("nan")}
        types = {}
        for (t, snr), v in sorted(by_type.items()):
            types.setdefault(t, {})[snr] = {"n": len(v), "recall_iou": _div(sum(v), len(v)),
                                            "recall_main": _div(sum(by_type_main[(t, snr)]), len(v)),
                                            "recall": _div(sum(by_type_any[(t, snr)]), len(v))}
        out["views"][view] = {"by_snr": levels, "by_type": types}
    (folder / "scores.json").write_text(json.dumps(out, indent=1))
    plot(out, folder / "scores.png")
    return out


def plot(s: Dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = s["views"]["front_end"]
    snrs = sorted(v["by_snr"], key=float)
    x = [v["by_snr"][k]["snr_db"] for k in snrs]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=120)
    ax = axes[0]
    for t, per in sorted(v["by_type"].items()):
        ax.plot(x, [per.get(k, {}).get("recall", np.nan) for k in snrs], color="#8a8984", lw=0.8, alpha=0.6)
        ax.annotate(t, (x[-1], per.get(snrs[-1], {}).get("recall", np.nan)), fontsize=6, color="#52514e",
                    xytext=(3, 0), textcoords="offset points", va="center")
    ax.plot(x, [v["by_snr"][k]["recall"] for k in snrs], color="#2a78d6", lw=2.4, marker="o", ms=4, label="all types")
    ax.set_title("Recall by call type (overlap)", fontsize=10, loc="left")
    ax = axes[1]
    for key, color, label in (("recall_main", "#1a4f99", "recall (main)"), ("precision_main", "#a8431a", "precision (main)"),
                              ("recall", "#2a78d6", "recall (overlap)"), ("recall_iou", "#2a78d6", "recall (IoU ≥ 0.5)"),
                              ("precision", "#eb6834", "precision (overlap)"), ("precision_iou", "#eb6834", "precision (IoU ≥ 0.5)")):
        ax.plot(x, [v["by_snr"][k][key] for k in snrs], color=color, marker="o", ms=4,
                ls="--" if "iou" in key else "-", lw=3 if "main" in key else 1.4, label=label)
    ax.set_title("All calls (main: IoU >= 0.5, phees by overlap)", fontsize=10, loc="left")
    ax = axes[2]
    ax.plot(x, [v["by_snr"][k]["false_alarms_per_h"] for k in snrs], color="#52514e", lw=2, marker="o", ms=4)
    ax.set_title("False alarms per hour (no planted call)", fontsize=10, loc="left")
    for i, ax in enumerate(axes):
        ax.set_xlabel("SNR (dB, 5-20 kHz, over the call)")
        if i < 2:
            ax.set_ylim(0, 1.02)
        ax.grid(color="#e4e3df", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Learned detector on synthetic recordings (front end; {s['n_recordings']} recordings, "
                 f"{s['n_calls']} planted calls, night background)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["detect", "score"])
    ap.add_argument("set", type=Path)
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args()
    if a.step == "detect":
        detect(a.set, a.task_index, a.n_tasks)
        return
    s = score(a.set)
    for view, v in s["views"].items():
        print(view)
        for snr, r in v["by_snr"].items():
            print(f"  SNR {float(snr):>4g} ({r['snr_db']:+.1f} dB): main R {r['recall_main']:.3f} "
                  f"P {r['precision_main']:.3f} | overlap R {r['recall']:.3f} P {r['precision']:.3f} "
                  f"R_iou {r['recall_iou']:.3f} P_iou {r['precision_iou']:.3f} FA/h {r['false_alarms_per_h']:.0f} "
                  f"edges {r['onset_err_ms']:.0f}/{r['offset_err_ms']:.0f} ms")


if __name__ == "__main__":
    main()
