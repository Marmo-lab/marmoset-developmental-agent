"""Acoustic inventory of the hand-labelled calls: a data-driven prior for the unit count k.

The question (OPEN_DECISIONS D24): is there a "phoneme-like" level in marmoset calls
whose size could guide how many discrete units the encoder should emit? Three views,
from coarse to fine, all on the hand-labelled calls (exact boundaries, no detector):

1. **Calls.** Duration, dominant frequency, frequency range, notes per call.
2. **Notes** -- the closest analogue of a phoneme: a contiguous tonal piece of a call,
   cut where energy drops out for > 20 ms or the dominant frequency jumps > 1.5 kHz.
   Each note is described by (log duration, start / end / mean frequency, range), and a
   Gaussian mixture picks the number of note types by BIC.
3. **Frames** at 20 ms, the encoder's own resolution: the in-call spectral shape
   (24 bands of 500 Hz over 4-16 kHz, level-normalised, 10 PCA dims), clustered by GMM
   for a range of k with BIC, plus the clip's background frames for comparison.

BIC on continuous acoustic data has no "true" k; what matters is where it stops
improving quickly (the knee), and how that compares with the encoder sweep.

Usage:
    python tools/analysis/call_inventory.py labeling/clips.csv data/labels/handlabels.csv \
        --audio labeling/clip_audio48 --json-out outputs/call_inventory.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

BAND = (4_000.0, 16_000.0)
HOP_S = 0.010
ACTIVE_DB = 10.0  # a frame is tonal-active if its peak bin is this far above the bin's floor
NOTE_GAP_FRAMES = 2  # gaps up to 20 ms do not break a note
NOTE_MIN_FRAMES = 3  # notes shorter than 30 ms are dropped
NOTE_JUMP_HZ = 1_500.0
NOTE_KS = list(range(1, 31))
FRAME_KS = [5, 10, 20, 30, 50, 75, 100, 150, 200]


def _spectrogram(npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(freqs, times in clip seconds, power dB relative to each bin's median, lead)."""
    z = np.load(npz)
    sr, lead = int(z["sr"]), float(z["lead_s"])
    x = z["capsules"].astype(np.float32).mean(axis=0)
    hop = int(HOP_S * sr)
    f, t, s = signal.stft(x, fs=sr, nperseg=1024, noverlap=1024 - hop)
    sel = (f >= BAND[0]) & (f <= BAND[1])
    p = 10 * np.log10(np.abs(s[sel]) ** 2 + 1e-20)
    return f[sel], t - lead, p - np.median(p, axis=1, keepdims=True), lead


def _notes(f: np.ndarray, rel: np.ndarray) -> List[Dict[str, float]]:
    """Split one call's frames into tonal notes; ``rel`` is (F, T) inside the call."""
    peak_db, peak_f = rel.max(axis=0), f[rel.argmax(axis=0)]
    active = peak_db > ACTIVE_DB
    notes, cur, gap = [], [], 0
    for i in range(rel.shape[1]):
        if active[i] and cur and abs(peak_f[i] - peak_f[cur[-1]]) > NOTE_JUMP_HZ:
            notes.append(cur)
            cur, gap = [], 0
        if active[i]:
            cur.append(i)
            gap = 0
        elif cur:
            gap += 1
            if gap > NOTE_GAP_FRAMES:
                notes.append(cur)
                cur, gap = [], 0
    if cur:
        notes.append(cur)
    out = []
    for n in notes:
        if len(n) < NOTE_MIN_FRAMES:
            continue
        fr = peak_f[n] / 1000
        out.append({"dur_s": len(n) * HOP_S, "f_start": float(fr[0]), "f_end": float(fr[-1]),
                    "f_mean": float(fr.mean()), "f_range": float(fr.max() - fr.min())})
    return out


def _frame_features(f: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """20 ms frames x 24 band levels (500 Hz bands), each frame's mean level removed."""
    edges = np.arange(BAND[0], BAND[1] + 1, 500.0)
    bands = np.stack([rel[(f >= lo) & (f < hi)].mean(axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
    t2 = bands.shape[1] // 2 * 2
    bands = bands[:, :t2].reshape(len(edges) - 1, -1, 2).mean(axis=2)  # 10 ms -> 20 ms
    return (bands - bands.mean(axis=0, keepdims=True)).T


def _bic_curve(x: np.ndarray, ks: List[int], seed: int = 0) -> Dict[int, float]:
    return {k: float(GaussianMixture(k, covariance_type="full", n_init=2, random_state=seed,
                                     reg_covar=1e-3).fit(x).bic(x)) for k in ks if k < len(x) // 5}


def _knee(curve: Dict[int, float]) -> int:
    """Smallest k that achieves 90% of the BIC drop from the first k to the best k."""
    ks = sorted(curve)
    best = min(curve.values())
    drop = curve[ks[0]] - best
    return next(k for k in ks if curve[ks[0]] - curve[k] >= 0.9 * drop)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips_csv")
    ap.add_argument("labels_csv")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()

    labels: Dict[str, List[Tuple[float, float]]] = {}
    for r in csv.DictReader(open(a.labels_csv)):
        if r["label"] == "c":
            labels.setdefault(r["clip_id"], []).append((float(r["clip_start_s"]), float(r["clip_end_s"])))
    calls, notes, call_frames, bg_frames = [], [], [], []
    for cid, labs in sorted(labels.items()):
        f, t, rel, _ = _spectrogram(Path(a.audio) / f"{cid}.npz")
        inside = np.zeros(len(t), bool)
        for s, e in labs:
            sel = (t >= s) & (t <= e)
            inside |= sel
            r = rel[:, sel]
            ns = _notes(f, r)
            pk = f[r.argmax(axis=0)][r.max(axis=0) > ACTIVE_DB] / 1000
            calls.append({"dur_s": e - s, "n_notes": len(ns),
                          "f_dom": float(np.median(pk)) if pk.size else float("nan"),
                          "f_range": float(pk.max() - pk.min()) if pk.size else 0.0})
            notes += ns
            call_frames.append(_frame_features(f, r))
        clip = (t >= 0) & (t <= 60) & ~inside
        bg_frames.append(_frame_features(f, rel[:, clip]))

    cf = np.concatenate(call_frames)
    bf = np.concatenate(bg_frames)
    note_x = StandardScaler().fit_transform(np.array(
        [[np.log10(n["dur_s"]), n["f_start"], n["f_end"], n["f_mean"], n["f_range"]] for n in notes]))
    pca = PCA(10, random_state=0).fit(np.concatenate([cf, bf]))
    note_curve = _bic_curve(note_x, NOTE_KS)
    frame_curve = _bic_curve(pca.transform(cf), FRAME_KS)
    bg_curve = _bic_curve(pca.transform(bf[np.random.default_rng(0).choice(len(bf), min(len(bf), len(cf)),
                                                                          replace=False)]), FRAME_KS)

    d = np.array([c["dur_s"] for c in calls])
    nn = np.array([c["n_notes"] for c in calls])
    fd = np.array([c["f_dom"] for c in calls])
    res = {
        "n_calls": len(calls), "n_notes": len(notes),
        "n_call_frames_20ms": int(len(cf)), "n_background_frames_20ms": int(len(bf)),
        "call_duration_s_pct10_50_90": np.percentile(d, [10, 50, 90]).round(3).tolist(),
        "dominant_freq_khz_pct10_50_90": np.nanpercentile(fd, [10, 50, 90]).round(2).tolist(),
        "notes_per_call": {str(k): int((nn == k).sum()) for k in range(0, 6)} | {"6+": int((nn >= 6).sum())},
        "note_duration_s_pct10_50_90": np.percentile([n["dur_s"] for n in notes], [10, 50, 90]).round(3).tolist(),
        "frames_per_call_20ms_median": float(np.median(d) / 0.02),
        "note_types_bic": note_curve, "note_types_knee": _knee(note_curve),
        "note_types_bic_min": min(note_curve, key=note_curve.get),
        "call_frame_bic": frame_curve, "call_frame_knee": _knee(frame_curve),
        "background_frame_bic": bg_curve, "background_frame_knee": _knee(bg_curve),
        "call_time_fraction_of_day_clips": float(len(cf) / (len(cf) + len(bf))),
    }
    Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.json_out, "w"), indent=1)
    for k, v in res.items():
        if not isinstance(v, dict) or k == "notes_per_call":
            print(f"{k}: {v}")
    print("note-type BIC:", {k: round(v) for k, v in note_curve.items()})
    print("call-frame BIC:", {k: round(v) for k, v in frame_curve.items()})
    print("background-frame BIC:", {k: round(v) for k, v in bg_curve.items()})


if __name__ == "__main__":
    main()
