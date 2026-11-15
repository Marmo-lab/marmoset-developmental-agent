"""Detections at recording-file edges: how many, what they are, and what they do to the corpus.

Why. Picking calls for a figure (2026-11-14), 17 of the 72 strongest candidates ended exactly at
the end of their recording file (all 12 short 6-10 kHz "peeps"). The production detector
(``streaming.detect_recording``) reads each block with ``stream_context_s`` of extra audio on both
sides, so the high-pass filter, the resampler and the STFT never see a hard edge inside a file;
at the file's own start and end there is no audio to read, and they do. Two explanations, with
different cures: an *artifact* (the edge transient looks like a call) or a *cut call* (the
recorder splits files while an animal calls, so a real call is halved across two files).

Steps [ours]:
* ``stats RUN OUT``: on the deduplicated run (``merge_calls``), calls whose padded end is within
  ``TOUCH_S`` of their file's end (or whose start is within ``TOUCH_S`` of its start), against the
  rate expected if calls ignored file edges; their duration, SNR, confidence and hour against all
  calls; continuity between consecutive files; their share of ``calls_train.tsv``.
* ``context RUN OUT`` (array): the decisive test. The production detector runs on the last
  ``WIN_S`` of a file twice: alone (a hard edge, as in production) and with the next ``NEXT_S`` of
  audio appended (the next file, when the recording continues). An edge detection that vanishes
  with the continuation is an artifact; one that carries on across the junction is a cut call.
  Control: the same on ``N_CASES`` random interior cut points, where the "edge" is made by us.
* ``summary OUT``: both into ``summary.json``.

Usage (cluster, tools/cluster/file_edges.sbatch):
    python tools/analysis/file_edges.py stats RUN OUT
    python tools/analysis/file_edges.py context RUN OUT --task I --n-tasks N
    python tools/analysis/file_edges.py summary OUT
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

TOUCH_S = 0.05
GAP_OK_S = 1.0           # consecutive files closer than this are one continuous recording
WIN_S = 70.0             # > stream_block_s + stream_context_s: the edge sits in a block's core
NEXT_S = 10.0
N_CASES = 150
JUNCTION_S = 0.3         # a joined-run detection this close before the junction "is" the edge detection


def _files(run: Path) -> Dict[str, Dict]:
    dup = set(json.loads((run / "calls_summary.json").read_text()).get("duplicates_dropped", []))
    fs = {}
    for p in (run / "files").glob("*.json"):
        f = json.loads(p.read_text())
        if f["recording_id"] not in dup:
            f["t0"] = dt.datetime.fromisoformat(f["start_time"])
            fs[f["recording_id"]] = f
    return fs


def _next_file(fs: Dict[str, Dict]) -> Dict[str, Tuple[str, float]]:
    """Each file's successor in time and the gap (s) between its end and the successor's start."""
    order = sorted(fs.values(), key=lambda f: f["t0"])
    out = {}
    for a, b in zip(order, order[1:]):
        gap = (b["t0"] - a["t0"]).total_seconds() - a["duration_s"]
        out[a["recording_id"]] = (b["recording_id"], gap)
    return out


def stats(run: Path, out: Path) -> Dict:
    fs = _files(run)
    nxt = _next_file(fs)
    calls = [r for r in csv.DictReader(open(run / "calls.tsv"), delimiter="\t") if r["recording_id"] in fs]
    train = {r["call_id"] for r in csv.DictReader(open(run / "calls_train.tsv"), delimiter="\t")}
    end_rows, start_rows = [], []
    tail_hist = np.zeros(300)                 # call ends in the last 30 s, 0.1 s bins (index 0 = at the end)
    for r in calls:
        dur = fs[r["recording_id"]]["duration_s"]
        s, e = float(r["start_s"]), float(r["end_s"])
        to_end = dur - e
        if to_end < 30:
            tail_hist[min(299, int(max(0.0, to_end) / 0.1))] += 1
        if to_end <= TOUCH_S:
            end_rows.append(r)
        if s <= TOUCH_S:
            start_rows.append(r)
    hours = sum(f["duration_s"] for f in fs.values()) / 3600
    per_bin_expected = float(np.mean(tail_hist[10:]))       # 1-30 s from the end: the flat rate

    def describe(rows: List[Dict]) -> Dict:
        if not rows:
            return {"n": 0}
        g = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
        hour = np.array([int(r["start_time"][11:13]) for r in rows])
        return {"n": len(rows), "median_duration_s": round(float(np.median(g("duration_s"))), 3),
                "median_snr_db": round(float(np.median(g("snr_db"))), 1),
                "median_confidence": round(float(np.median(g("confidence"))), 3),
                "median_peak_khz": round(float(np.median(g("peak_freq_hz"))) / 1000, 2),
                "share_night_19_to_7": round(float(np.mean((hour >= 19) | (hour < 7))), 3),
                "share_in_calls_train": round(float(np.mean([r["call_id"] in train for r in rows])), 3),
                "call_hours": round(float(g("duration_s").sum()) / 3600, 2)}

    continuous = [rid for rid, (_, gap) in nxt.items() if abs(gap) < GAP_OK_S]
    end_ids = {r["recording_id"] for r in end_rows}
    start_ids = {r["recording_id"] for r in start_rows}
    res = {
        "recordings": len(fs), "hours": round(hours, 1), "calls": len(calls), "calls_train": len(train),
        "touch_s": TOUCH_S,
        "end_touching": describe(end_rows), "start_touching": describe(start_rows), "all_calls": describe(calls),
        "files_with_end_touching": len(end_ids), "files_with_start_touching": len(start_ids),
        "share_of_files_with_end_touching": round(len(end_ids) / len(fs), 3),
        "last_0.1s_bin_vs_flat_rate": round(float(tail_hist[0] / per_bin_expected), 1) if per_bin_expected else None,
        "tail_hist_first_10_bins": tail_hist[:10].astype(int).tolist(), "tail_flat_rate_per_bin": round(per_bin_expected, 1),
        "continuous_successor": len(continuous),
        "end_touching_with_start_touching_successor": sum(nxt[rid][0] in start_ids for rid in end_ids if rid in nxt
                                                          and abs(nxt[rid][1]) < GAP_OK_S),
        "end_touching_of_continuous_files": sum(rid in end_ids for rid in continuous),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(res, indent=1))
    with open(out / "end_touching.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(end_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(end_rows)
    _plot_tail(tail_hist, per_bin_expected, out / "tail.png")
    return res


def _plot_tail(h: np.ndarray, flat: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(np.arange(len(h))[:60] * 0.1, h[:60], width=0.09, color="#3987e5")
    ax.axhline(flat, color="k", lw=0.8, ls="--", label="rate 1-30 s from the end")
    ax.set_xlabel("call end, seconds before the file's end")
    ax.set_ylabel("calls per 0.1 s bin")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def _cases(run: Path, out: Path) -> List[Dict]:
    """File-end cases (an end-touching call and a continuous successor) and random interior controls."""
    fs = _files(run)
    nxt = _next_file(fs)
    rng = random.Random(0)
    ends = sorted({r["recording_id"] for r in csv.DictReader(open(out / "end_touching.tsv"), delimiter="\t")})
    ends = [rid for rid in ends if rid in nxt and abs(nxt[rid][1]) < GAP_OK_S]
    cases = [{"kind": "file_end", "rid": rid, "cut_s": fs[rid]["duration_s"], "next": nxt[rid][0]}
             for rid in rng.sample(ends, min(N_CASES, len(ends)))]
    day = [f for f in fs.values() if 7 <= f["t0"].hour <= 17 and f["duration_s"] > 600]
    for f in rng.sample(day, min(N_CASES, len(day))):
        cases.append({"kind": "interior", "rid": f["recording_id"], "cut_s": rng.uniform(WIN_S + 5, f["duration_s"] - NEXT_S - 5),
                      "next": None})
    for c in cases:
        c["path"] = fs[c["rid"]]["path"]
        c["next_path"] = fs[c["next"]]["path"] if c["next"] else None
    return cases


def _detect(x: np.ndarray, sr: int, cfg) -> List[Tuple[float, float]]:
    from marmoset_developmental_agent.preprocessing.streaming import ArraySource, detect_recording
    return [(r.start_s, r.end_s) for r in detect_recording(ArraySource(x, sr), dt.datetime(2021, 1, 1, 12), cfg)]


def _read(path: str, start_s: float, dur_s: float) -> Tuple[np.ndarray, int]:
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        a = max(0, int(round(start_s * sr)))
        f.seek(a)
        return f.read(int(round(dur_s * sr)), dtype="float32", always_2d=True), sr


def context(run: Path, out: Path, task: int, n_tasks: int) -> None:
    import dataclasses

    from marmoset_developmental_agent.config.agent_config import load_config
    cfg = dataclasses.replace(load_config().preprocessing, exclude_night=False)
    (out / "context").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, c in list(enumerate(_cases(run, out)))[task::n_tasks]:
        x, sr = _read(c["path"], c["cut_s"] - WIN_S, WIN_S)
        if c["kind"] == "file_end":
            y, _ = _read(c["next_path"], 0.0, NEXT_S)
        else:
            y, _ = _read(c["path"], c["cut_s"], NEXT_S)
        edge = len(x) / sr
        alone = _detect(x, sr, cfg)
        joined = _detect(np.concatenate([x, y]), sr, cfg)
        a_edge: Optional[Tuple[float, float]] = next((d for d in alone if d[1] >= edge - TOUCH_S), None)
        j_near = [d for d in joined if d[1] > edge - JUNCTION_S and d[0] < edge]
        rows.append({"case": k, "kind": c["kind"], "rid": c["rid"], "cut_s": round(c["cut_s"], 3),
                     "alone_edge_detection": int(a_edge is not None),
                     "alone_edge_duration_s": round(a_edge[1] - a_edge[0], 3) if a_edge else "",
                     "joined_detection_at_junction": int(bool(j_near)),
                     "joined_crosses_junction": int(any(d[1] > edge + TOUCH_S for d in j_near)),
                     "joined_max_end_after_edge_s": round(max((d[1] - edge for d in j_near), default=0.0), 3)})
        print(rows[-1], flush=True)
    if rows:
        with open(out / "context" / f"{task:03d}.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)


def summary(out: Path) -> Dict:
    rows = [r for p in sorted((out / "context").glob("*.tsv")) for r in csv.DictReader(open(p), delimiter="\t")]
    by: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        b = by[r["kind"]]
        b["alone"].append(int(r["alone_edge_detection"]))
        if r["alone_edge_detection"] == "1":
            b["vanishes"].append(int(r["joined_detection_at_junction"] == "0"))
            b["crosses"].append(int(r["joined_crosses_junction"] == "1"))
    res = {"stats": json.loads((out / "stats.json").read_text()), "context": {}}
    for kind, b in by.items():
        n = len(b["alone"])
        res["context"][kind] = {
            "cases": n, "edge_detection_when_cut": round(float(np.mean(b["alone"])), 3),
            "of_those_vanish_with_continuation (artifact)": round(float(np.mean(b["vanishes"])), 3) if b["vanishes"] else None,
            "of_those_continue_across_junction (cut call)": round(float(np.mean(b["crosses"])), 3) if b["crosses"] else None}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["stats", "context", "summary"])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    a = ap.parse_args()
    if a.step == "stats":
        print(json.dumps(stats(*a.paths), indent=1))
    elif a.step == "context":
        context(*a.paths, a.task, a.n_tasks)
    else:
        print(json.dumps(summary(a.paths[0]), indent=1))


if __name__ == "__main__":
    main()
