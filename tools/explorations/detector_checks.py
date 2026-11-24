"""Three checks of detector v3's settings on its own cross-validated predictions (2026-11-24).

D1  Threshold: P, R, IoU matches, splits / merges and edge errors at frame thresholds 0.2-0.8, on the out-of-fold
    probabilities of the v3 cross-validation (each clip scored by a model that never saw it). The blind test set is
    not touched.
D3  Frames at 0.3 <= p < 0.5: where they lie (inside a ``c`` label, a ``?`` label, or background), and how many of the
    calls missed at 0.5 reach that range at all, i.e. what lowering the threshold could recover.
D4  The cost of 250 Hz bands: for every ``c`` label, its contrast in the 47 Hz STFT bins and in the detector's 250 Hz
    bands (median over the call's frames of the loudest bin / band, in dB above that bin's / band's median over the
    clip). Pooling a narrow tone with its noise neighbours lowers its contrast; if the missed calls are the ones that
    lose most, the band width hides calls. The fine contrast is biased upward (the maximum over five times more noisy
    values), so only missed against caught is compared, never fine against band directly.

Input: ``oof_probs.npz`` from ``train_call_detector.py cv`` (``$WORK_DIR/detector/v3/cv``) and the hand-labelled
clips of ``train_call_detector.SETS``. ``short``: D4 on the held-out half of the short-call labels (``score_short_labels.items``), the calls v3 caught or
missed in ``short_label_hits.tsv`` (front-end view); contrast over the same 10 s window the scoring read.
Usage:
    PYTHONPATH=. python tools/explorations/detector_checks.py OOF_PROBS.npz OUT_DIR
    PYTHONPATH=. python tools/explorations/detector_checks.py short LABELS_DIR HITS.tsv OUT_DIR    (cluster)
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.io.channels import combine
from marmoset_developmental_agent.preprocessing.filter_5khz import highpass_filter
from marmoset_developmental_agent.preprocessing.learned_detector import learned_features
from tools.analysis.score_detectors import score_clip, summarise
from tools.analysis.train_call_detector import CLIP_S, SETS, load_clips, match_breakdown, to_calls

THRESHOLDS = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
Span = Tuple[float, float]


def labels() -> Dict[str, List[Tuple[str, float, float]]]:
    import csv
    out: Dict[str, List[Tuple[str, float, float]]] = {}
    for _, labels_csv, _ in SETS:
        for r in csv.DictReader(open(labels_csv)):
            out.setdefault(r["clip_id"], []).append((r["label"], float(r["clip_start_s"]), float(r["clip_end_s"])))
    return out


def threshold_curve(probs, clips, labs, cfg: PreprocessingConfig) -> List[Dict]:
    rows = []
    for th in THRESHOLDS:
        c2 = dataclasses.replace(cfg, learned_threshold=th)
        per, night = [], 0
        agg: Dict[str, List[float]] = {k: [] for k in ("one_to_one", "split", "merged", "missed", "onset", "offset")}
        for cid, clip in clips.items():
            d = [tuple(x) for x in to_calls(probs[cid], clip, c2)]
            if not int(clip["day"]):
                night += len(d)
                continue
            calls = [(a, b) for k, a, b in labs.get(cid, []) if k == "c"]
            unsure = [(a, b) for k, a, b in labs.get(cid, []) if k == "?"]
            per.append(score_clip(d, calls, unsure))
            for k, v in match_breakdown(d, calls, unsure).items():
                agg[k] += v
        sm = summarise(per)
        n = sum(len(agg[k]) for k in ("one_to_one", "split", "merged", "missed"))
        on, off = np.array(agg["onset"]), np.array(agg["offset"])
        rows.append({"threshold": th, **{k: round(float(v), 3) for k, v in sm.items()}, "night_dets": night,
                     **{k: round(len(agg[k]) / n, 3) for k in ("one_to_one", "split", "merged", "missed")},
                     "onset_ms": round(1000 * float(np.median(on))), "offset_ms": round(1000 * float(np.median(off))),
                     "edges_50ms": round(float(np.mean((on <= .05) & (off <= .05))), 3)})
    return rows


def mid_frames(probs, clips, labs, cfg: PreprocessingConfig) -> Dict:
    """Where the 0.3-0.5 frames of the day clips lie, and what the calls missed at 0.5 reach."""
    where = {"c": 0, "?": 0, "background": 0}
    missed_max: List[float] = []
    for cid, clip in clips.items():
        if not int(clip["day"]):
            continue
        p, t = probs[cid][0], clip["t"]
        inside = (t >= 0) & (t <= CLIP_S)
        lab = np.full(t.shape, "background", object)
        for k, a, b in labs.get(cid, []):
            lab[(t >= a) & (t <= b)] = k
        mid = inside & (p >= 0.3) & (p < 0.5)
        for k in where:
            where[k] += int((mid & (lab == k)).sum())
        dets = to_calls(probs[cid], clip, cfg)
        for k, a, b in labs.get(cid, []):
            if k == "c" and not any(min(b, d[1]) > max(a, d[0]) for d in dets):
                missed_max.append(float(p[(t >= a) & (t <= b)].max()))
    m = np.array(missed_max)
    return {"mid_frames": where, "missed_at_0.5": len(m),
            "missed_max_p": {"<0.1": int((m < .1).sum()), "0.1-0.3": int(((m >= .1) & (m < .3)).sum()),
                             "0.3-0.5": int(((m >= .3) & (m < .5)).sum())}}


def band_cost(probs, clips, labs, cfg: PreprocessingConfig) -> List[Dict]:
    """Per ``c`` label of the day clips: fine (47 Hz) and band (250 Hz) contrast, caught or missed."""
    rows = []
    for clips_csv, _, audio_dir in SETS:
        import csv
        for c in csv.DictReader(open(clips_csv)):
            cid = c["clip_id"]
            if not int(c["day"]):
                continue
            z = np.load(Path(audio_dir) / f"{cid}.npz")
            sr, lead = int(z["sr"]), float(z["lead_s"])
            x = highpass_filter(combine(z["capsules"].astype(np.float32).T, "mean", sr)[0], sr,
                                cfg.highpass_cutoff_hz, cfg.highpass_order)
            band, t = learned_features(x, sr, cfg)          # exactly the detector's input, before clipping
            f, _, s = signal.stft(x, fs=sr, nperseg=cfg.learned_n_fft, noverlap=cfg.learned_n_fft - cfg.learned_hop_length,
                                  boundary=None, padded=False)
            fine = 10 * np.log10(np.abs(s[f >= cfg.highpass_cutoff_hz]) ** 2 + 1e-20)
            fine -= np.median(fine, axis=1, keepdims=True)
            t = t - lead
            dets = to_calls(probs[cid], clips[cid], cfg)
            for k, a, b in labs.get(cid, []):
                if k != "c":
                    continue
                sel = (t >= a) & (t <= b)
                if not sel.any():
                    continue
                cf, cb = float(np.median(fine[:, sel].max(0))), float(np.median(band[:, sel].max(0)))
                rows.append({"clip": cid, "start": a, "end": b, "dur": round(b - a, 3),
                             "caught": any(min(b, d[1]) > max(a, d[0]) for d in dets),
                             "max_p": round(float(probs[cid][0][sel].max()), 3),
                             "fine_db": round(cf, 1), "band_db": round(cb, 1), "loss_db": round(cf - cb, 1)})
    return rows


def contrast(x: np.ndarray, sr: int, span: Span, cfg: PreprocessingConfig) -> Tuple[float, float]:
    """(fine, band) contrast of ``span`` (seconds into ``x``): see D4 in the module docstring."""
    x = highpass_filter(x, sr, cfg.highpass_cutoff_hz, cfg.highpass_order)
    band, t = learned_features(x, sr, cfg)
    f, _, s = signal.stft(x, fs=sr, nperseg=cfg.learned_n_fft, noverlap=cfg.learned_n_fft - cfg.learned_hop_length,
                          boundary=None, padded=False)
    fine = 10 * np.log10(np.abs(s[f >= cfg.highpass_cutoff_hz]) ** 2 + 1e-20)
    fine -= np.median(fine, axis=1, keepdims=True)
    sel = (t >= span[0]) & (t <= span[1])
    return float(np.median(fine[:, sel].max(0))), float(np.median(band[:, sel].max(0)))


def short(labels_dir: str, hits_tsv: str, out_dir: str) -> None:
    import csv
    import soundfile as sf
    from marmoset_developmental_agent.preprocessing.filter_5khz import resample_audio
    from tools.analysis.score_short_labels import LEAD_S, WINDOW_S, items
    cfg = PreprocessingConfig()
    hit = {r["item_id"]: r["hit"] == "1" for r in csv.DictReader(open(hits_tsv), delimiter="\t") if r["model"] == "v3"}
    rows = []
    for it in items(Path(labels_dir)):
        if it["kind"] != "call" or it["item_id"] not in hit:
            continue
        with sf.SoundFile(it["path"]) as fh:
            sr = fh.samplerate
            s0 = max(0, int(round((it["clip_start_s"] - LEAD_S) * sr)))
            fh.seek(s0)
            x = fh.read(int(WINDOW_S * sr), dtype="float32", always_2d=True)
        x = combine(x, "mean", sr)[0]
        if sr != cfg.detection_sample_rate:
            x = resample_audio(x, sr, cfg.detection_sample_rate)
        shift = it["clip_start_s"] - s0 / sr
        a, b = it["span"]
        cf, cb = contrast(x, cfg.detection_sample_rate, (a + shift, b + shift), cfg)
        rows.append({"item": it["item_id"], "dur": round(b - a, 3), "caught": hit[it["item_id"]],
                     "fine_db": round(cf, 1), "band_db": round(cb, 1), "loss_db": round(cf - cb, 1)})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(Path(out_dir) / "band_cost_short.json", "w"), indent=1)
    _print_cost(rows)


def _print_cost(rows: List[Dict]) -> None:
    caught = [r for r in rows if r["caught"]]
    missed = [r for r in rows if not r["caught"]]
    print(f"\nD4 band cost: {len(caught)} caught, {len(missed)} missed")
    for name, g in (("caught", caught), ("missed", missed)):
        a = lambda k, g=g: np.array([r[k] for r in g])  # noqa: E731
        print(f"  {name:7s} fine {np.median(a('fine_db')):5.1f}  band {np.median(a('band_db')):5.1f}  "
              f"loss median {np.median(a('loss_db')):4.1f} (p25-p75 {np.percentile(a('loss_db'), 25):.1f}-"
              f"{np.percentile(a('loss_db'), 75):.1f})  dur median {np.median(a('dur')):.2f} s")


def main(oof: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessingConfig()
    probs = dict(np.load(oof))
    clips = load_clips(cfg)
    labs = labels()
    d1 = threshold_curve(probs, clips, labs, cfg)
    d3 = mid_frames(probs, clips, labs, cfg)
    d4 = band_cost(probs, clips, labs, cfg)
    json.dump({"threshold_curve": d1, "mid_frames": d3, "band_cost": d4}, open(out / "detector_checks.json", "w"),
              indent=1)
    print("D1 threshold curve (day clips, c labels, ? ignored)")
    keys = ["threshold", "P", "R", "F1", "P_iou", "R_iou", "P_time", "R_time", "night_dets", "split", "merged",
            "missed", "onset_ms", "offset_ms", "edges_50ms"]
    print(" ".join(f"{k:>9}" for k in keys))
    for r in d1:
        print(" ".join(f"{r[k]:>9}" for k in keys))
    print("\nD3", json.dumps(d3))
    _print_cost(d4)
    for r in sorted([r for r in d4 if not r["caught"]], key=lambda r: -r["loss_db"]):
        print("   missed", r)


if __name__ == "__main__":
    short(*sys.argv[2:5]) if sys.argv[1] == "short" else main(sys.argv[1], sys.argv[2])
