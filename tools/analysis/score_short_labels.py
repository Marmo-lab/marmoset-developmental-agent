"""Detectors on the held-out short-call labels: do they catch the calls, and leave the rejected candidates alone?

Why (user, 2026-11-17). The candidate v3 is v2a's recipe plus the user's short-call labels from one half of the
recordings (``train_call_detector.py --short``). This scores any checkpoint on the other half, which no training saw:
* **calls caught:** the share of the user's calls that a detection overlaps, split by whether the v5 corpus run
  (v2a) had caught the candidate. On the ones it missed, v2a must score near 0 here; that checks the method.
* **rejected candidates fired on:** the share of "not a call" items whose candidate span a detection overlaps.
Both through the production path (``streaming.detect_recording`` on the raw 4-capsule audio of a ``WINDOW_S`` window
around the labelling clip, so the stream's own context applies), only the checkpoint swapped. Two views, as the
synthetic scores: ``detector`` (every detection) and ``front_end`` (without ``short_low_conf``, what reaches the
corpus). 95% intervals: bootstrap over recordings, since items from one recording share its room and animals.

``figure``: the learning curve, calls caught and candidates fired on against the number of labelled items each model
trained on (v2a = 0), into ``figures/``.

Usage (cluster, tools/cluster/score_short_labels.sbatch):
    python tools/analysis/score_short_labels.py score LABELS OUT --model NAME=PATH:N_ITEMS [...]
    python tools/analysis/score_short_labels.py figure OUT/short_label_scores.json FIGURE
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

from tools.analysis.train_call_detector import _last_verdicts
from tools.synth.score import quality

WINDOW_S = 10.0
LEAD_S = 4.0            # window start before the labelling clip's start (the clip is 2 s, so 4 s follow it)
N_BOOT = 2000


def items(labels: Path, half: str = "test") -> List[Dict]:
    """The usable answers of one half: calls with edges (span = the edges) and "not a call" (span = the candidate)."""
    meta = {r["item_id"]: r for r in csv.DictReader(open(labels / "meta.tsv"), delimiter="\t")}
    marks = {r["item_id"]: json.loads(r["marks"]) for r in csv.DictReader(open(labels / "items.tsv"), delimiter="\t")}
    out = []
    for k, v in sorted(_last_verdicts(labels / "review_feedback.jsonl").items()):
        m = meta[k]
        if m["half"] != half or v["verdict"] not in ("call", "not") or (v["verdict"] == "call" and not v.get("bounds")):
            continue
        span = v["bounds"] if v["verdict"] == "call" else marks[k][0][:2]
        out.append({"item_id": k, "kind": v["verdict"], "span": [float(span[0]), float(span[1])], "path": m["file_path"],
                    "clip_start_s": float(m["clip_start_s"]), "recording_id": m["recording_id"],
                    "missed_by_v2a": m["missed_by_v2a"] == "1"})
    return out


def detect_item(it: Dict, cfg) -> List[Tuple[float, float, str]]:
    """Detections (clip seconds, quality) in the item's window, by the production streaming path."""
    from marmoset_developmental_agent.preprocessing.streaming import ArraySource, detect_recording
    with sf.SoundFile(it["path"]) as f:
        sr = f.samplerate
        s = max(0, int(round((it["clip_start_s"] - LEAD_S) * sr)))
        f.seek(s)
        x = f.read(int(WINDOW_S * sr), dtype="float32", always_2d=True)
    shift = it["clip_start_s"] - s / sr
    return [(d.start_s - shift, d.end_s - shift, quality(d.duration_s, d.confidence))
            for d in detect_recording(ArraySource(x, sr), dt.datetime(2000, 1, 1, 12), cfg)]


def _boot(per_rec: Dict[str, List[bool]], rng: np.random.Generator) -> List[float]:
    """Share of True, and its 95% interval over recordings."""
    recs = sorted(per_rec)
    hits = np.array([sum(per_rec[r]) for r in recs], float)
    ns = np.array([len(per_rec[r]) for r in recs], float)
    if ns.sum() == 0:
        return [float("nan")] * 3
    draws = rng.integers(len(recs), size=(N_BOOT, len(recs)))
    b = hits[draws].sum(1) / np.maximum(ns[draws].sum(1), 1)
    return [float(hits.sum() / ns.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def score(labels: Path, out: Path, models: List[str]) -> Dict:
    from marmoset_developmental_agent.config.agent_config import load_config
    its = items(labels)
    res: Dict = {"n_calls": sum(i["kind"] == "call" for i in its), "n_not": sum(i["kind"] == "not" for i in its),
                 "models": {}}
    rows = []
    for spec in models:
        name, rest = spec.split("=", 1)
        path, n_items = rest.rsplit(":", 1)
        cfg = dataclasses.replace(load_config().preprocessing, learned_model_path=path)
        groups: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
        for it in its:
            dets = detect_item(it, cfg)
            a, b = it["span"]
            for view in ("detector", "front_end"):
                hit = any(min(b, e) > max(a, s) for s, e, q in dets if view == "detector" or q != "short_low_conf")
                key = f"{view}/{it['kind']}"
                groups[key][it["recording_id"]].append(hit)
                if it["kind"] == "call":
                    groups[f"{key}/{'missed' if it['missed_by_v2a'] else 'caught'}_by_v2a"][it["recording_id"]].append(hit)
                if view == "front_end":
                    rows.append({"model": name, "item_id": it["item_id"], "kind": it["kind"],
                                 "missed_by_v2a": int(it["missed_by_v2a"]), "hit": int(hit)})
        rng = np.random.default_rng(0)
        res["models"][name] = {"path": path, "n_train_items": int(n_items),
                               **{k: _boot(v, rng) for k, v in sorted(groups.items())}}
        print(name, {k: [round(x, 3) for x in v] for k, v in res["models"][name].items() if isinstance(v, list)}, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "short_label_scores.json").write_text(json.dumps(res, indent=1))
    with open(out / "short_label_hits.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return res


def figure(scores: Path, dest: Path) -> None:
    """The learning curve: front-end view, every model by the number of labelled items it trained on. Models named
    ``*_s<k>`` are reruns with another seed: hollow markers off the line, so the seed spread is visible."""
    import re

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = json.loads(scores.read_text())
    ms = sorted(s["models"].items(), key=lambda kv: (kv[1]["n_train_items"], kv[0]))
    line = [(n, m) for n, m in ms if not re.search(r"_s\d+$", n)]
    seeds = [(n, m) for n, m in ms if re.search(r"_s\d+$", n)]
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=200)
    for dx, (key, color, label) in zip((-2, 0, 2), (
            ("front_end/call/missed_by_v2a", "#eb6834", "calls v2a missed: caught"),
            ("front_end/call", "#2a78d6", "all labelled calls: caught"),
            ("front_end/not", "#52514e", "rejected candidates: fired on (lower is better)"))):
        if not all(key in m for _, m in ms):
            continue
        for group, filled in ((line, True), (seeds, False)):
            if not group:
                continue
            x = np.array([m["n_train_items"] for _, m in group], float) + dx + (0 if filled else 4)
            y = np.array([m[key] for _, m in group])
            ax.errorbar(x, y[:, 0], yerr=[y[:, 0] - y[:, 1], y[:, 2] - y[:, 0]], color=color, marker="o", ms=5,
                        mfc=color if filled else "white", lw=1.8, capsize=3, elinewidth=1, label=label if filled else None,
                        ls=("--" if "not" in key else "-") if filled else "none")
    for name, m in ms:
        ax.annotate(name, (m["n_train_items"], m["front_end/call"][2]), xytext=(0, 4), textcoords="offset points",
                    fontsize=7, color="#6b6a66", ha="left" if name in dict(seeds) else "center")
    ax.set_xlabel("hand-labelled short-call items in training (0 = v2a; hollow = another seed)", fontsize=9)
    ax.set_ylabel("share of held-out items", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors="#6b6a66")
    ax.set_title(f"Learning curve on the held-out half ({s['n_calls']} calls, {s['n_not']} rejected candidates; 95% CI)",
                 fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight")
    print(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["score", "figure"])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--model", action="append", default=[], help="NAME=PATH:N_TRAIN_ITEMS")
    x = ap.parse_args()
    score(x.a, x.b, x.model) if x.step == "score" else figure(x.a, x.b)


if __name__ == "__main__":
    main()
