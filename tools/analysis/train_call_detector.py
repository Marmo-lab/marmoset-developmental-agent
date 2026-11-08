"""Train and evaluate the learned call detector (preprocessing/learned_detector.py).

Data: the 45 hand-labelled one-minute clips (30 calibration, 15 held-out), as the 4-capsule
48 kHz ``<clip>.npz`` blocks that ``score_detectors.py extract`` wrote. Features are made
with the production function on the production detection signal: capsule mean, high-pass
at the config's cutoff, ``learned_features``. Frame targets: ``c`` = 1, unlabelled = 0,
``?`` and the context outside the labelled minute = ignored.

``cv``  Clip-grouped k-fold cross-validation. Clips are sorted by recording day and dealt
        round-robin into folds, so every fold spans the campaign. Each clip is scored by
        a model that never saw it. Writes the out-of-fold probabilities, then runs the
        full production post-processing (``activity_to_calls``) on them and writes the
        detections as a ``score_detectors``-style JSON next to the rule detectors, so
        ``evaluate`` and ``review_labels.py`` can read them.
``fit`` Train one model on all clips and save the checkpoint the pipeline loads.
``evaluate``  Score the cross-validated detections next to rule detectors from
        ``score_detectors.py --sweep harmonic`` JSONs, on the day clips (night = detections
        on the night clips). Besides P/R, IoU and chunk metrics it reports how each label
        was matched: one-to-one, split into several detections, merged with a neighbour,
        or missed, with the onset/offset error of the one-to-one matches.

Extra sources (``--synth DIR``, ``--real DIR``; user, 2026-11-08). The hand labels hold no
short high call (tsik, tsik-ek, seep), and the detector misses them (docs/SYNTH_EVAL.md):
* ``--synth``: a synthetic set of ``tools/synth/generate.py`` (built with ``--half train``).
  Every planted call is 1 with its edges, the call-free night background 0.
* ``--real``: ``find_short_high.py minutes``: real day minutes whose positives (rule candidates
  the user heard as calls, and the run's calls) are 1; the rest is ignored (it may hold calls
  nobody found), and so are its edges (the rule's bounds are approximate). Crops from these
  minutes are centred on a random positive frame, so each carries supervision.
Each batch draws ``--mix`` (synthetic, real) of its crops from these sources and the rest from
the hand clips, so the 45 labelled minutes keep half the weight. ``cv`` folds the hand clips
only; the extra sources are in every fold's training set.

On the cluster (``tools/cluster/train_detector.sbatch``): ``--work WORK_DIR`` reads the same
clips from the cluster layout (``hand_sets``), and ``cv --fold K`` trains and scores one fold
(an array task, ``fold_K.npz``); ``cv --merge`` joins the folds into the outputs of a full ``cv``.
A fold's model is the same either way (seed = fold index).

Training: random 4 s crops, batch 32, AdamW with a one-cycle schedule (peak 2e-3),
``--steps`` updates, masked BCE. A random linear spectral tilt (sd 2 dB across the band)
is applied to half the crops, for robustness to how different recordings colour the
spectrum. Seeds are fixed; on CPU a run is reproducible.

Usage:
    python tools/analysis/train_call_detector.py cv --out outputs/learned_cv
    python tools/analysis/train_call_detector.py fit --out models/call_detector_v1.pt
    python tools/analysis/train_call_detector.py cv --heads 1 --out outputs/learned_cv_v2 \
        --synth SYNTH_TRAIN_SET --real SHORT_HIGH/minutes          # with the extra sources
    python tools/analysis/train_call_detector.py evaluate outputs/learned_cv/learned_detections.json \
        --rules outputs/calib_harmonic.json outputs/heldout_harmonic.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.io.channels import combine
from marmoset_developmental_agent.preprocessing.filter_5khz import FrameActivity, highpass_filter
from marmoset_developmental_agent.preprocessing.learned_detector import (
    CallNet, learned_features, normalise, save_call_detector)
from marmoset_developmental_agent.preprocessing.segmentation import activity_to_calls

SETS = (("labeling/clips.csv", "data/labels/handlabels.csv", "labeling/clip_audio48"),
        ("labeling/heldout/clips.csv", "data/labels/heldout_labels.csv", "labeling/heldout/audio48"))
CLIP_S = 60.0
CROP = 400  # frames, 4 s at 10 ms
EDGE_SIGMA_FRAMES = 2.0  # onset / offset targets: Gaussian bumps of sd 20 ms on the labelled edges
EDGE_POS_WEIGHT = 9.0  # edge frames are rare; weight loss terms by 1 + 9 * target

Clip = Dict[str, np.ndarray]


def hand_sets(work: Optional[Path] = None) -> Tuple[Tuple[str, str, str], ...]:
    """(clips.csv, labels.csv, audio dir) per labelled set: the repo's ``labeling/`` layout, or the
    cluster's ``$WORK_DIR`` one (the same files, checked by checksum 2026-11-08)."""
    if work is None:
        return SETS
    return ((str(work / "labeling" / "clips.csv"), SETS[0][1], str(work / "clip_audio48")),
            (str(work / "labeling_heldout" / "clips.csv"), SETS[1][1], str(work / "heldout_audio48")))


def load_clips(cfg: PreprocessingConfig, sets: Tuple[Tuple[str, str, str], ...] = SETS) -> Dict[str, Clip]:
    """Production features and frame targets for every labelled clip."""
    out: Dict[str, Clip] = {}
    for clips_csv, labels_csv, audio_dir in sets:
        labels: Dict[str, List[Tuple[str, float, float]]] = {}
        for r in csv.DictReader(open(labels_csv)):
            labels.setdefault(r["clip_id"], []).append((r["label"], float(r["clip_start_s"]), float(r["clip_end_s"])))
        for c in csv.DictReader(open(clips_csv)):
            z = np.load(Path(audio_dir) / f"{c['clip_id']}.npz")
            sr, lead = int(z["sr"]), float(z["lead_s"])
            x = combine(z["capsules"].astype(np.float32).T, "mean", sr)[0]
            x = highpass_filter(x, sr, cfg.highpass_cutoff_hz, cfg.highpass_order)
            feats, t = learned_features(x, sr, cfg)
            t = t - lead
            inside = (t >= 0) & (t <= CLIP_S)
            y = np.where(inside, 0, -1).astype(np.float32)
            on = np.where(inside, 0, -1).astype(np.float32)
            off = on.copy()
            hop = t[1] - t[0]
            for lab, a, b in labels.get(c["clip_id"], []):
                y[(t >= a) & (t <= b)] = 1 if lab == "c" else -1
                for edge, tgt in ((a, on), (b, off)):
                    near = np.abs(t - edge) <= 4 * EDGE_SIGMA_FRAMES * hop
                    if lab == "c":
                        bump = np.exp(-0.5 * ((t[near] - edge) / (EDGE_SIGMA_FRAMES * hop)) ** 2)
                        tgt[near] = np.where(tgt[near] >= 0, np.maximum(tgt[near], bump), tgt[near])
                    else:
                        tgt[near] = -1  # a ? label's edges are neither edges nor non-edges
            out[c["clip_id"]] = {"x": normalise(feats), "y": np.stack([y, on, off]), "t": t,
                                 "day": np.int64(c["day"]), "lead": np.float64(lead)}
    return out


def _features(x: np.ndarray, sr: int, cfg: PreprocessingConfig) -> Tuple[np.ndarray, np.ndarray]:
    x = highpass_filter(x, sr, cfg.highpass_cutoff_hz, cfg.highpass_order)
    feats, t = learned_features(x, sr, cfg)
    return normalise(feats), t


def _edges(t: np.ndarray, spans: List[Tuple[float, float]], tgt: np.ndarray, which: int) -> None:
    hop = t[1] - t[0]
    for span in spans:
        edge = span[which]
        near = np.abs(t - edge) <= 4 * EDGE_SIGMA_FRAMES * hop
        tgt[near] = np.maximum(tgt[near], np.exp(-0.5 * ((t[near] - edge) / (EDGE_SIGMA_FRAMES * hop)) ** 2))


def load_synth(cfg: PreprocessingConfig, folder: Path) -> Dict[str, Clip]:
    """A synthetic set as clips: planted calls 1 (with edges), background 0."""
    import soundfile as sf
    events: Dict[str, List[Tuple[float, float]]] = {}
    for e in csv.DictReader(open(folder / "events.tsv"), delimiter="\t"):
        events.setdefault(e["rec_id"], []).append((float(e["onset_s"]), float(e["offset_s"])))
    out: Dict[str, Clip] = {}
    for r in csv.DictReader(open(folder / "recordings.tsv"), delimiter="\t"):
        x, sr = sf.read(folder / "audio" / f"{r['rec_id']}.flac", dtype="float32")
        feats, t = _features(x, sr, cfg)
        spans = events.get(r["rec_id"], [])
        y, on, off = np.zeros(len(t), np.float32), np.zeros(len(t), np.float32), np.zeros(len(t), np.float32)
        for a, b in spans:
            y[(t >= a) & (t <= b)] = 1
        _edges(t, spans, on, 0)
        _edges(t, spans, off, 1)
        out[f"synth:{r['rec_id']}"] = {"x": feats, "y": np.stack([y, on, off]), "t": t, "day": np.int64(-1),
                                       "lead": np.float64(0)}
    return out


def load_real(cfg: PreprocessingConfig, folder: Path) -> Dict[str, Clip]:
    """Real minutes as clips: positives 1, everything else (and every edge) ignored."""
    import soundfile as sf
    spans: Dict[str, List[Tuple[float, float]]] = {}
    for r in csv.DictReader(open(folder / "positives.tsv"), delimiter="\t"):
        spans.setdefault(r["minute_id"], []).append((float(r["start_s"]), float(r["end_s"])))
    out: Dict[str, Clip] = {}
    for mid, ss in sorted(spans.items()):
        x, sr = sf.read(folder / f"{mid}.flac", dtype="float32")
        feats, t = _features(x, sr, cfg)
        y = np.full(len(t), -1, np.float32)
        for a, b in ss:
            y[(t >= a) & (t <= b)] = 1
        if not (y == 1).any():
            continue
        ign = np.full(len(t), -1, np.float32)
        out[f"real:{mid}"] = {"x": feats, "y": np.stack([y, ign, ign.copy()]), "t": t, "day": np.int64(-1),
                              "lead": np.float64(0), "pos": np.flatnonzero(y == 1)}
    return out


def _crop_start(c: Clip, rng: np.random.Generator) -> int:
    n = c["x"].shape[1]
    if "pos" in c:                                    # centre a crop of a real minute on a positive frame
        return int(np.clip(c["pos"][rng.integers(len(c["pos"]))] - rng.integers(CROP // 4, 3 * CROP // 4), 0, n - CROP))
    return int(rng.integers(0, n - CROP))


def train(clips: Dict[str, Clip], ids: List[str], steps: int, seed: int, n_out: int = 3,
          extra: Tuple[Tuple[List[str], float], ...] = ()) -> CallNet:
    """``extra``: (clip ids, share of each batch) per extra source; the hand ``ids`` fill the rest."""
    rng = np.random.default_rng(seed)
    pools = [(ids, 1.0 - sum(w for e, w in extra if e))] + [(e, w) for e, w in extra if e]
    shares = np.array([w for _, w in pools])
    torch.manual_seed(seed)
    net = CallNet(next(iter(clips.values()))["x"].shape[0], n_out=n_out)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    net.train()
    for _ in range(steps):
        xs, ys = [], []
        for _ in range(32):
            pool = pools[rng.choice(len(pools), p=shares / shares.sum())][0] if len(pools) > 1 else ids
            c = clips[pool[rng.integers(len(pool))]]
            s = _crop_start(c, rng)
            x = c["x"][:, s:s + CROP].copy()
            if rng.random() < 0.5:
                x += (rng.normal(0, 0.1) * np.linspace(-1, 1, x.shape[0]))[:, None].astype(np.float32)
            xs.append(x)
            ys.append(c["y"][:n_out, s:s + CROP])
        x, y = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))  # y (B, n_out, T)
        m = (y >= 0).float()
        w = m.clone()
        w[:, 1:] *= 1 + EDGE_POS_WEIGHT * y[:, 1:].clamp(min=0)
        bce = nn.functional.binary_cross_entropy_with_logits(net(x), y.clamp(min=0), reduction="none")
        loss = sum((bce[:, k] * w[:, k]).sum() / w[:, k].sum().clamp(min=1.0) for k in range(n_out))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


def predict(net: CallNet, clip: Clip) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(clip["x"])[None]))[0].numpy()


def to_calls(prob: np.ndarray, clip: Clip, cfg: PreprocessingConfig) -> List[Tuple[float, float]]:
    """Production post-processing on ``(n_out, T)`` probabilities: threshold, then the shared
    ``activity_to_calls`` (which uses the edge heads when present). Clip time."""
    prob = np.atleast_2d(prob)
    act = FrameActivity(frame_times_s=clip["t"] + clip["lead"], active=prob[0] >= cfg.learned_threshold,
                        hop_s=cfg.learned_hop_length / cfg.detection_sample_rate,
                        onset_prob=prob[1] if len(prob) == 3 else None,
                        offset_prob=prob[2] if len(prob) == 3 else None)
    lead = float(clip["lead"])
    return [(max(0.0, c.start_s - lead), min(CLIP_S, c.end_s - lead))
            for c in activity_to_calls(act, cfg) if c.end_s - lead > 0 and c.start_s - lead < CLIP_S]


def load_extra(cfg: PreprocessingConfig, synth: List[Path], real: List[Path],
               mix: Tuple[float, float]) -> Tuple[Dict[str, Clip], Tuple[Tuple[List[str], float], ...]]:
    clips: Dict[str, Clip] = {}
    groups = []
    for dirs, loader, share in ((synth, load_synth, mix[0]), (real, load_real, mix[1])):
        got: Dict[str, Clip] = {}
        for d in dirs:
            got.update(loader(cfg, d))
        clips.update(got)
        groups.append((sorted(got), share))
        if dirs:
            print(f"{loader.__name__}: {len(got)} clips, {share:.0%} of each batch", flush=True)
    return clips, tuple(groups)


def cv(cfg: PreprocessingConfig, out: Path, folds: int, steps: int, n_out: int = 3,
       synth: List[Path] = (), real: List[Path] = (), mix: Tuple[float, float] = (0.25, 0.25),
       sets: Tuple[Tuple[str, str, str], ...] = SETS, only: Optional[int] = None, merge: bool = False) -> None:
    """All folds, or (``only``) one fold to ``fold_K.npz``, or (``merge``) join the fold files."""
    clips = load_clips(cfg, sets)
    ids = sorted(clips, key=lambda c: (int(clips[c]["day"]), c))
    groups = [ids[i::folds] for i in range(folds)]
    out.mkdir(parents=True, exist_ok=True)
    probs: Dict[str, np.ndarray] = {}
    if merge:
        for k in range(folds):
            probs.update(dict(np.load(out / f"fold_{k}.npz")))
    else:
        more, extra = load_extra(cfg, list(synth), list(real), mix)
        clips.update(more)
        t0 = time.time()
        for k, test in enumerate(groups):
            if only is not None and k != only:
                continue
            net = train(clips, [c for c in ids if c not in test], steps, seed=k, n_out=n_out, extra=extra)
            fold = {c: predict(net, clips[c]) for c in test}
            probs.update(fold)
            if only is not None:
                np.savez(out / f"fold_{k}.npz", **fold)
            print(f"fold {k + 1}/{folds} ({len(test)} clips) done, {time.time() - t0:.0f} s", flush=True)
        if only is not None:
            return
    np.savez(out / "oof_probs.npz", **probs)
    dets = {c: {"learned": {"mean": to_calls(probs[c], clips[c], cfg)}} for c in ids}
    json.dump({"created": dt.datetime.now().isoformat(timespec="seconds"), "folds": folds, "steps": steps,
               "config": dataclasses.asdict(cfg), "clips": {c: {"detections": dets[c]} for c in ids}},
              open(out / "learned_detections.json", "w"), indent=1)
    print(f"wrote {out}/oof_probs.npz and learned_detections.json")


def fit(cfg: PreprocessingConfig, out: Path, steps: int, n_out: int = 3,
        synth: List[Path] = (), real: List[Path] = (), mix: Tuple[float, float] = (0.25, 0.25),
        sets: Tuple[Tuple[str, str, str], ...] = SETS) -> None:
    clips = load_clips(cfg, sets)
    ids = sorted(clips)
    more, extra = load_extra(cfg, list(synth), list(real), mix)
    clips.update(more)
    net = train(clips, ids, steps, seed=0, n_out=n_out, extra=extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_call_detector(net, out)
    print(f"wrote {out}")


RULES = {"rule, 2026-10-20": "harm-:floor0.2322:max4", "rule, harmonic (10-21)": "harm10:floor0.15:max8"}


def _ov(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match_breakdown(dets: List[Tuple[float, float]], calls: List[Tuple[float, float]],
                    unsure: List[Tuple[float, float]]) -> Dict[str, List[float]]:
    """How each label was matched, and the boundary errors of the one-to-one matches."""
    out: Dict[str, List[float]] = {k: [] for k in ("one_to_one", "split", "merged", "missed", "onset", "offset")}
    for lab in calls:
        hit = [d for d in dets if _ov(d, lab) > 0]
        if not hit:
            out["missed"].append(1)
        elif len(hit) > 1:
            out["split"].append(1)
        elif any(m != lab and _ov(hit[0], m) > 0 for m in calls):
            out["merged"].append(1)
        else:
            out["one_to_one"].append(1)
            out["onset"].append(abs(hit[0][0] - lab[0]))
            out["offset"].append(abs(hit[0][1] - lab[1]))
    return out


def evaluate(learned_json: str, rule_jsons: List[str]) -> None:
    from tools.analysis.score_detectors import score_clip, summarise

    rules: Dict[str, Dict] = {}
    for js in rule_jsons:
        rules.update(json.load(open(js))["clips"])
    learned = json.load(open(learned_json))["clips"]
    ids = sorted(learned)
    candidates = {name: {c: rules[c]["detections"][key]["mean"] for c in ids} for name, key in RULES.items()}
    candidates["learned (cross-validated)"] = {c: learned[c]["detections"]["learned"]["mean"] for c in ids}
    for mode in ("c", "c+?"):
        print(f"\n=== day clips, labels counted: {mode} ===")
        print(f"{'detector':<27}{'P':>5}{'R':>6}{'F1':>6}{'Piou':>6}{'Riou':>6}{'P_t':>6}{'R_t':>6}{'chP':>6}{'chR':>6}"
              f"{'night':>6}{'1:1':>6}{'split':>6}{'merge':>6}{'miss':>6}{'on_ms':>7}{'off_ms':>7}{'<=50ms':>7}")
        for name, dets in candidates.items():
            rows, night = [], 0
            agg: Dict[str, List[float]] = {k: [] for k in ("one_to_one", "split", "merged", "missed", "onset", "offset")}
            for c in ids:
                labs = rules[c]["labels"]
                calls = [(s, e) for s, e, k in labs if k == "c" or (mode == "c+?" and k == "?")]
                unsure = [(s, e) for s, e, k in labs if k == "?"] if mode == "c" else []
                d = [tuple(x) for x in dets[c]]
                if rules[c]["day"]:
                    rows.append(score_clip(d, calls, unsure))
                    for k, v in match_breakdown(d, calls, unsure).items():
                        agg[k] += v
                else:
                    night += len(d)
            sm = summarise(rows)
            n = sum(len(agg[k]) for k in ("one_to_one", "split", "merged", "missed"))
            on, off = np.array(agg["onset"]), np.array(agg["offset"])
            print(f"{name:<27}{sm['P']:5.2f}{sm['R']:6.2f}{sm['F1']:6.2f}{sm['P_iou']:6.2f}{sm['R_iou']:6.2f}"
                  f"{sm['P_time']:6.2f}{sm['R_time']:6.2f}{sm['chunk_P']:6.2f}{sm['chunk_R']:6.2f}{night:6d}"
                  + "".join(f"{len(agg[k]) / n:6.2f}" for k in ("one_to_one", "split", "merged", "missed"))
                  + f"{1000 * np.median(on):7.0f}{1000 * np.median(off):7.0f}{np.mean((on <= .05) & (off <= .05)):7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cv")
    c.add_argument("--out", required=True)
    c.add_argument("--folds", type=int, default=9)
    c.add_argument("--steps", type=int, default=1000)
    c.add_argument("--heads", type=int, choices=(1, 3), default=3, help="1 = call only (v1); 3 = call + onset + offset")
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--steps", type=int, default=1000)
    f.add_argument("--heads", type=int, choices=(1, 3), default=3)
    for p_ in (c, f):
        p_.add_argument("--synth", type=Path, action="append", default=[], help="synthetic set (generate.py --half train)")
        p_.add_argument("--real", type=Path, action="append", default=[], help="find_short_high.py minutes folder")
        p_.add_argument("--mix", default="0.25,0.25", help="share of each batch from synthetic, real")
        p_.add_argument("--work", type=Path, help="read the hand clips from the cluster's WORK_DIR layout")
    c.add_argument("--fold", type=int, help="train and score only this fold (writes fold_K.npz)")
    c.add_argument("--merge", action="store_true", help="join fold_K.npz files into the cv outputs")
    e = sub.add_parser("evaluate")
    e.add_argument("learned_json")
    e.add_argument("--rules", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "evaluate":
        evaluate(a.learned_json, a.rules)
        return
    torch.set_num_threads(4)
    cfg = PreprocessingConfig.learned()
    mix = tuple(float(v) for v in a.mix.split(","))
    sets = hand_sets(a.work)
    if a.cmd == "cv":
        cv(cfg, Path(a.out), a.folds, a.steps, a.heads, a.synth, a.real, mix, sets, a.fold, a.merge)
    else:
        fit(cfg, Path(a.out), a.steps, a.heads, a.synth, a.real, mix, sets)


if __name__ == "__main__":
    main()
