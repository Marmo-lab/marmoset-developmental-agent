"""Score the frozen detectors once on the blind test set, as fixed in advance by docs/TEST_PROTOCOL.md.

Run it from a checkout of the frozen commit (8d815bb), so the detector, configuration and
scoring code are exactly the frozen ones; this script only drives them:

    git worktree add /tmp/frozen 8d815bb
    cd /tmp/frozen && PYTHONPATH=. python <repo>/tools/analysis/score_testset.py \\
        --root <repo> --out <repo>/outputs/testset_v1

``--model NAME=PATH:SHA_PREFIX`` (repeatable) scores other frozen checkpoints from the current
commit (TEST_PROTOCOL.md, addendum 2026-11-08: v2a, with v1 again for a same-code comparison).
Without it, v1 alone, as in the first scoring.

Detectors (all on the capsule mean, as in every earlier evaluation):

* ``learned``: ``models/call_detector_v1.pt`` (sha256 prefix checked) with
  ``PreprocessingConfig.learned()``, features and post-processing exactly as
  ``train_call_detector.load_clips`` / ``to_calls`` make them;
* ``learned, no short_low_conf``: the same detections without those shorter than 0.18 s
  with confidence (mean call probability over the call) below 0.6 (protocol addendum);
* ``rule (reference)``: ``PreprocessingConfig()``, the harmonic rule detector;
* ``talia``: ``PreprocessingConfig.talia()``, the GmSLM front-end port, for comparison.

Metrics are the protocol's: ``score_detectors.score_clip`` / ``summarise`` (P, R, F1 by
overlap; P, R at IoU >= 0.5; time P and R; chunk P and R), ``train_call_detector.match_breakdown``
(one-to-one, split, merged, missed; onset / offset errors of one-to-one matches), and the
share of one-to-one matches with both edges within 30 and 50 ms. Labels are counted as
``c`` (``?`` ignored; primary) and as ``c+?``. 95% intervals: 2000 bootstrap resamples of
the 15 clips (clip-level, because calls within a clip are not independent) [ours].

Recall is also broken down by label SNR (``score_detectors.label_snr_db``) and by label
duration. Divergence from the protocol: recall by shape cluster is not reported. The
45-clip clusters came from t-SNE + HDBSCAN, which cannot place new calls, and that
analysis was not saved as code, so duration bands stand in for it.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.io.channels import combine
from marmoset_developmental_agent.preprocessing.filter_5khz import highpass_filter, resample_audio
from marmoset_developmental_agent.preprocessing.learned_detector import learned_features, load_call_detector, normalise
from tools.analysis.score_detectors import _detect, _merge, label_snr_db, score_clip, summarise
from tools.analysis.train_call_detector import match_breakdown, to_calls

Interval = Tuple[float, float]
CLIP_S = 60.0
SHA_PREFIX = "da94fdadead78ed3"
SHORT_S, LOW_CONF = 0.18, 0.6
N_BOOT = 2000
SNR_BANDS = ((-99.0, 10.0), (10.0, 20.0), (20.0, 99.0))
DUR_BANDS = ((0.0, 0.18), (0.18, 0.5), (0.5, 99.0))


def learned_detections(npz: Path, cfg: PreprocessingConfig, model: Path) -> List[Tuple[float, float, float]]:
    """``(start, end, confidence)`` in clip time, as ``load_clips`` + ``predict`` + ``to_calls`` do it."""
    z = np.load(npz)
    sr, lead = int(z["sr"]), float(z["lead_s"])
    x = combine(z["capsules"].astype(np.float32).T, "mean", sr)[0]
    x = highpass_filter(x, sr, cfg.highpass_cutoff_hz, cfg.highpass_order)
    feats, t = learned_features(x, sr, cfg)
    clip = {"x": normalise(feats), "t": t - lead, "lead": np.float64(lead)}
    net = load_call_detector(str(model))
    with torch.no_grad():
        prob = torch.sigmoid(net(torch.from_numpy(clip["x"])[None]))[0].numpy()
    out = []
    for s, e in to_calls(prob, clip, cfg):
        inside = (clip["t"] >= s) & (clip["t"] <= e)
        out.append((s, e, float(prob[0][inside].mean()) if inside.any() else 0.0))
    return out


def rule_detections(npz: Path, cfg: PreprocessingConfig) -> List[Interval]:
    """``score_detectors.detect_clip`` for one configuration on the capsule mean."""
    z = np.load(npz)
    sr, lead = int(z["sr"]), float(z["lead_s"])
    sigs = [resample_audio(x, sr, cfg.detection_sample_rate)
            for x in combine(z["capsules"].astype(np.float32).T, "mean", sr)]
    calls = _merge([iv for x in sigs for iv in _detect(x, cfg.detection_sample_rate, cfg)])
    return [(max(0.0, a - lead), min(CLIP_S, b - lead)) for a, b in calls if b - lead > 0 and a - lead < CLIP_S]


PerClip = Dict[str, Tuple[Dict[str, float], Dict[str, List[float]]]]


def per_clip(dets: Dict[str, List[Interval]], labels: Dict[str, List[Tuple[float, float, str]]],
             ids: Sequence[str], mode: str) -> PerClip:
    """``score_clip`` counts and ``match_breakdown`` lists of every clip, computed once."""
    out: PerClip = {}
    for c in ids:
        calls = [(s, e) for s, e, k in labels[c] if k == "c" or (mode == "c+?" and k == "?")]
        unsure = [(s, e) for s, e, k in labels[c] if k == "?"] if mode == "c" else []
        out[c] = (score_clip(dets[c], calls, unsure), match_breakdown(dets[c], calls, unsure))
    return out


def metrics(pc: PerClip, ids: Sequence[str]) -> Dict[str, float]:
    """The protocol's metrics over ``ids`` (a clip may repeat, for the bootstrap)."""
    agg: Dict[str, List[float]] = {k: [] for k in ("one_to_one", "split", "merged", "missed", "onset", "offset")}
    for c in ids:
        for k, v in pc[c][1].items():
            agg[k] += v
    m = summarise([pc[c][0] for c in ids])
    n = sum(len(agg[k]) for k in ("one_to_one", "split", "merged", "missed"))
    for k in ("one_to_one", "split", "merged", "missed"):
        m[k] = len(agg[k]) / n if n else float("nan")
    on, off = np.array(agg["onset"]), np.array(agg["offset"])
    m["onset_ms"] = float(1000 * np.median(on)) if len(on) else float("nan")
    m["offset_ms"] = float(1000 * np.median(off)) if len(off) else float("nan")
    m["edges_30ms"] = float(np.mean((on <= .03) & (off <= .03))) if len(on) else float("nan")
    m["edges_50ms"] = float(np.mean((on <= .05) & (off <= .05))) if len(on) else float("nan")
    return m


def bootstrap(pc: PerClip, ids: Sequence[str], keys: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(0)
    draws: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(N_BOOT):
        m = metrics(pc, list(rng.choice(ids, len(ids))))
        for k in keys:
            draws[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in draws.items()}


def recall_by(dets: Dict[str, List[Interval]], labs: List[Tuple[str, float, float, float]],
              bands: Sequence[Tuple[float, float]]) -> List[Dict[str, float]]:
    """Overlap recall of ``c`` labels per band of ``labs`` = (clip, start, end, value)."""
    out = []
    for lo, hi in bands:
        sel = [(c, s, e) for c, s, e, v in labs if lo <= v < hi]
        hit = sum(any(min(e, b) > max(s, a) for a, b in dets[c]) for c, s, e in sel)
        out.append({"band": [lo, hi], "n": len(sel), "recall": hit / len(sel) if sel else float("nan")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the working repo (labels, test audio)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[], help="NAME=PATH:SHA_PREFIX, a frozen checkpoint")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    models: Dict[str, Tuple[Path, str]] = {}
    for spec in a.model or [f"learned=models/call_detector_v1.pt:{SHA_PREFIX}"]:
        name, rest = spec.split("=", 1)
        path, prefix = rest.rsplit(":", 1)
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if not sha.startswith(prefix):
            raise SystemExit(f"{name}: model sha {sha[:16]} is not the frozen {prefix}")
        models[name] = (Path(path), sha)
    ids = [r["clip_id"] for r in csv.DictReader(open(root / "labeling/test/clips.csv"))]
    labels: Dict[str, List[Tuple[float, float, str]]] = {c: [] for c in ids}
    for r in csv.DictReader(open(root / "data/labels/test_labels.csv")):
        labels[r["clip_id"]].append((float(r["clip_start_s"]), float(r["clip_end_s"]), r["label"]))
    audio = root / "labeling/test/audio48"

    cfg_l = PreprocessingConfig.learned()
    dets: Dict[str, Dict[str, List[Interval]]] = {}
    raws = {}
    for name, (model, _) in models.items():
        raw = {c: learned_detections(audio / f"{c}.npz", cfg_l, model) for c in ids}
        raws[name] = raw
        dets[name] = {c: [(s, e) for s, e, _ in raw[c]] for c in ids}
        dets[f"{name}, no short_low_conf"] = {c: [(s, e) for s, e, p in raw[c] if not (e - s < SHORT_S and p < LOW_CONF)]
                                             for c in ids}
    dets["rule (reference)"] = {c: rule_detections(audio / f"{c}.npz", PreprocessingConfig()) for c in ids}
    dets["talia"] = {c: rule_detections(audio / f"{c}.npz", PreprocessingConfig.talia()) for c in ids}

    keys = ("P", "R", "F1", "P_iou", "R_iou", "P_time", "R_time", "one_to_one", "edges_50ms")
    subsets = {"all 15": ids, "ts_07 only": ["ts_07"], "without ts_07": [c for c in ids if c != "ts_07"]}
    results: Dict[str, object] = {"model_sha256": {n: v[1] for n, v in models.items()}, "config_learned": dataclasses.asdict(cfg_l), "scores": {}}
    for mode in ("c", "c+?"):
        for name, d in dets.items():
            pc = per_clip(d, labels, ids, mode)
            for sub, sids in subsets.items():
                m = metrics(pc, sids)
                if len(sids) > 1:
                    m["ci95"] = bootstrap(pc, sids, keys)
                results["scores"][f"{sub} | {mode} | {name}"] = m

    snr_labs, dur_labs = [], []
    for c in ids:
        calls = [(s, e) for s, e, k in labels[c] if k == "c"]
        for (s, e), v in zip(calls, label_snr_db(str(audio / f"{c}.npz"), calls)):
            snr_labs.append((c, s, e, v))
            dur_labs.append((c, s, e, e - s))
    results["recall_by_snr_db"] = {n: recall_by(d, snr_labs, SNR_BANDS) for n, d in dets.items()}
    results["recall_by_duration_s"] = {n: recall_by(d, dur_labs, DUR_BANDS) for n, d in dets.items()}
    results["counts"] = {"labels": {k: sum(1 for c in ids for *_, t in labels[c] if t == k) for k in ("c", "?")},
                         "detections": {n: sum(len(v) for v in d.values()) for n, d in dets.items()}}

    out.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out / "test_scores.json", "w"), indent=1)
    json.dump({n: {c: [list(x) for x in v] for c, v in d.items()} for n, d in dets.items()} | {"learned_conf": raws},
              open(out / "test_detections.json", "w"), indent=1)
    print(f"models {', '.join(f'{n} {v[1][:16]}' for n, v in models.items())} · labels {results['counts']['labels']} · detections {results['counts']['detections']}")
    hdr = f"{'':<44}" + "".join(f"{k:>11}" for k in keys) + f"{'on_ms':>7}{'off_ms':>7}{'<=30':>6}"
    for mode in ("c", "c+?"):
        print(f"\n=== labels counted: {mode} (± = half the 95% bootstrap interval) ===\n{hdr}")
        for sub in subsets:
            for name in dets:
                m, ci = results["scores"][f"{sub} | {mode} | {name}"], {}
                ci = m.get("ci95", {})
                cells = "".join(f"{m[k]:6.2f}" + (f"±{(ci[k][1] - ci[k][0]) / 2:4.2f}" if k in ci else "     ")
                                for k in keys)
                print(f"{sub + ' · ' + name:<44}{cells}{m['onset_ms']:7.0f}{m['offset_ms']:7.0f}{m['edges_30ms']:6.2f}")
    for what in ("recall_by_snr_db", "recall_by_duration_s"):
        print(f"\n{what}")
        for n, rows in results[what].items():
            print(f"  {n:<28}" + "  ".join(f"{r['band']}: {r['recall']:.2f} (n={r['n']})" for r in rows))


if __name__ == "__main__":
    main()
