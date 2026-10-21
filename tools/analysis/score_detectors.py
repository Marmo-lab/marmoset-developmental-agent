"""Score every detector x channel strategy x threshold against the hand labels.

Two stages, because the raw audio lives on the cluster and the labels and iteration
live locally:

``extract`` (cluster, read-only on the raw data)
    For each clip in ``clips.csv``, read the raw 4-capsule block covering the clip plus
    ``CONTEXT_S`` on each side and save the capsules as ``<clip>.npz`` (float16; native
    48 kHz by default, about 25 MB a clip, so that detection rates above 22050 Hz can be
    tested). Strategies and resampling to each detection rate happen at scoring time,
    exactly as ``preprocess_waveform`` does them. The ``best_snr`` capsule is chosen from excerpts spread across the *whole* source file,
    as in production, so it is computed here and stored.

``score`` (local)
    Run each detector configuration on each strategy's signal, keep the detections that
    fall inside the labelled minute, and compare them with the labels:

    * **event, overlap**: a label is found if any detection overlaps it; a detection is
      correct if it overlaps any label. Tolerant of fragmented boundaries.
    * **event, IoU >= 0.5**: one-to-one greedy matching. Strict about boundaries.
    * **time**: the fraction of labelled call-seconds that are covered by detections
      (recall), and the fraction of detected seconds that lie inside labels (precision).
    * **chunks**: run the production ``chunk_recording`` on the detections. Reports the
      fraction of chunks that contain a labelled call, and the fraction of labelled calls
      whose midpoint lands inside some chunk. This is what decides the training data.

    ``?`` labels are scored both ways: counted as calls (``c+?``), or ignored (``c``),
    in which case detections that touch only a ``?`` count as neither right nor wrong.
    Night clips have no labelled calls, so for them only the false-alarm rate is reported.

Usage:
    python tools/analysis/score_detectors.py extract labeling/clips.csv --out ~/marmoset_work/clip_audio48
    python tools/analysis/score_detectors.py score labeling/clips.csv data/labels/handlabels.csv \
        --audio labeling/clip_audio48 --json-out outputs/detector_scores.json [--sweep rate|tonal]
    python tools/analysis/score_detectors.py score labeling/heldout/clips.csv data/labels/heldout_labels.csv \
        --audio labeling/heldout/audio48 --json-out outputs/heldout_scores_tonal.json --sweep tonal --strategies mean
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.io.channels import STRATEGIES, combine
from marmoset_developmental_agent.preprocessing.filter_5khz import (
    compute_frame_activity, highpass_filter, resample_audio)
from marmoset_developmental_agent.preprocessing.segmentation import CallSegment, activity_to_calls, chunk_recording

CONTEXT_S = 2.0  # audio kept on each side of the labelled minute (filter and STFT edges)
CLIP_S = 60.0
BAND_HZ = (5_000.0, 11_000.0)  # label SNR band: what both detectors see at 22050 Hz

Interval = Tuple[float, float]
Overrides = Dict[str, object]


def _threshold_sweep() -> List[Tuple[str, Overrides]]:
    """Each detector's main threshold. Talia's is an absolute |STFT| level, so its sweep
    doubles as the gain-invariant comparison between strategies whose on-axis gains
    differ (mean 0.5, capsule ~0.8, beam 1.0)."""
    out: List[Tuple[str, Overrides]] = []
    for v in (0.005, 0.01, 0.02, 0.04, 0.08):
        out.append((f"talia@{v:g}", {"frame_features": "talia", "amp_threshold": v}))
    for v in (6.0, 9.0, 12.0, 15.0, 18.0):
        out.append((f"spectral@{v:g}", {"frame_features": "spectral", "bin_energy_threshold_db": v}))
    return out


def _rate_sweep() -> List[Tuple[str, Overrides]]:
    """Detection rate / band ceiling x threshold x call floor for the spectral detector,
    with Talia's configuration and a relaxed variant of it as references."""
    rates: List[Tuple[str, Overrides]] = [
        ("22k", {}),
        ("32k", {"detection_sample_rate": 32_000}),
        ("48k-16k", {"detection_sample_rate": 48_000, "spectral_max_freq_hz": 16_000.0}),
        ("48k-20k", {"detection_sample_rate": 48_000, "spectral_max_freq_hz": 20_000.0}),
    ]
    out: List[Tuple[str, Overrides]] = [
        ("talia@0.02", {"frame_features": "talia"}),
        ("talia@0.02:relaxed", {"frame_features": "talia", "abrupt_up_max_span_s": 0.0,
                                "noise_min_duration_s": 0.05, "min_call_duration_s": 0.1}),
    ]
    for thr in (12.0, 15.0, 18.0):
        for floor in (0.2322, 0.1):
            for rname, kw in rates:
                out.append((f"spectral@{thr:g}:{rname}:floor{floor:g}",
                            {"frame_features": "spectral", "bin_energy_threshold_db": thr,
                             "min_call_duration_s": floor, **kw}))
    return out


def _tonal_sweep() -> List[Tuple[str, Overrides]]:
    """Tonality rejection strength at 48 kHz with a 16 kHz ceiling, for two thresholds and
    two call floors (docs/DETECTOR_EVAL.md finding 9). ``tonal-`` is the rejection off."""
    out: List[Tuple[str, Overrides]] = []
    for thr in (12.0, 15.0):
        for floor in (0.2322, 0.1):
            for mtf in (None, 0.2, 0.3, 0.4, 0.5, 0.6):
                out.append((f"spectral@{thr:g}:floor{floor:g}:tonal{'-' if mtf is None else f'{mtf:g}'}",
                            {"frame_features": "spectral", "bin_energy_threshold_db": thr,
                             "min_call_duration_s": floor, "detection_sample_rate": 48_000,
                             "spectral_max_freq_hz": 16_000.0, "min_tonal_fraction": mtf}))
    return out


def _harmonic_sweep() -> List[Tuple[str, Overrides]]:
    """Harmonic-comb exemption x call floor x call ceiling (DETECTOR_EVAL finding 11),
    on the production detector. ``harm-:floor0.2322:max4`` is the detector held-out-tested
    on 2026-10-18."""
    out: List[Tuple[str, Overrides]] = []
    for floor, ceil in ((0.2322, 4.0), (0.15, 4.0), (0.15, 8.0), (0.1, 8.0)):
        for h in (None, 4.0, 6.0, 8.0, 10.0, 12.0):
            out.append((f"harm{'-' if h is None else f'{h:g}'}:floor{floor:g}:max{ceil:g}",
                        {"frame_features": "spectral", "detection_sample_rate": 48_000,
                         "spectral_max_freq_hz": 16_000.0, "min_tonal_fraction": 0.5,
                         "min_call_duration_s": floor, "max_call_duration_s": ceil,
                         "harmonic_min_contrast_db": h}))
    return out


SWEEPS = {"threshold": _threshold_sweep, "rate": _rate_sweep, "tonal": _tonal_sweep,
          "harmonic": _harmonic_sweep}


# --------------------------------------------------------------------------- #
# extract (cluster)
# --------------------------------------------------------------------------- #
def extract(clips_csv: str, out_dir: str, rate: int) -> None:
    import soundfile as sf

    from marmoset_developmental_agent.io.channels import pick_best_channel
    from marmoset_developmental_agent.io.raw_reader import _sample_excerpts

    os.makedirs(out_dir, exist_ok=True)
    for c in csv.DictReader(open(clips_csv)):
        info = sf.info(c["source_path"])
        sr = info.samplerate
        t0 = max(0.0, float(c["offset_s"]) - CONTEXT_S)
        lead = float(c["offset_s"]) - t0  # clip start within the saved block
        n = int(round((lead + CLIP_S + CONTEXT_S) * sr))
        with sf.SoundFile(c["source_path"]) as f:
            f.seek(int(round(t0 * sr)))
            block = f.read(n, dtype="float32", always_2d=True)  # (n, 4)
        caps = np.stack([resample_audio(block[:, i], sr, rate) for i in range(block.shape[1])])
        best = pick_best_channel(_sample_excerpts(c["source_path"], info), sr)
        np.savez_compressed(Path(out_dir) / f"{c['clip_id']}.npz",
                            capsules=caps.astype(np.float16), sr=rate, lead_s=lead, best_ch=best)
        print(f"{c['clip_id']}: {caps.shape[1] / rate:.1f} s at {rate} Hz, lead {lead:.2f} s, best capsule {best}")


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def _merge(iv: Sequence[Interval]) -> List[Interval]:
    out: List[Interval] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _detect(x: np.ndarray, sr: int, cfg: PreprocessingConfig) -> List[Interval]:
    d = highpass_filter(x, sr, cfg.highpass_cutoff_hz, cfg.highpass_order)
    return [(c.start_s, c.end_s) for c in activity_to_calls(compute_frame_activity(d, sr, cfg), cfg)]


def _configs(sweep: str) -> List[Tuple[str, PreprocessingConfig]]:
    # Overrides are relative to Talia's configuration (22050 Hz, no band ceiling), the
    # default when the sweeps were defined, so names such as "22k" keep their meaning.
    return [(name, dataclasses.replace(PreprocessingConfig.talia(), **kw)) for name, kw in SWEEPS[sweep]()]


def detect_clip(npz_path: str, sweep: str = "threshold",
                strategies: Sequence[str] = STRATEGIES) -> Dict[str, Dict[str, List[Interval]]]:
    """``{config: {strategy: [(start, end), ...]}}`` in clip time, clipped to [0, 60].

    Strategies combine the capsules at the stored rate, with the production selection
    band; each signal is then resampled to the configuration's detection rate, as
    ``preprocess_waveform`` does.
    """
    z = np.load(npz_path)
    caps = z["capsules"].astype(np.float32)
    sr, lead, best = int(z["sr"]), float(z["lead_s"]), int(z["best_ch"])
    signals = {s: combine(caps.T, s, sr, channel=best if s == "best_snr" else None) for s in strategies}
    cache: Dict[Tuple[str, int], List[np.ndarray]] = {}
    out: Dict[str, Dict[str, List[Interval]]] = {}
    for name, cfg in _configs(sweep):
        out[name] = {}
        det_sr = cfg.detection_sample_rate
        for s, sigs in signals.items():
            if (s, det_sr) not in cache:
                cache[(s, det_sr)] = [resample_audio(x, sr, det_sr) for x in sigs]
            calls = _merge([iv for x in cache[(s, det_sr)] for iv in _detect(x, det_sr, cfg)])
            out[name][s] = [(max(0.0, a - lead), min(CLIP_S, b - lead))
                            for a, b in calls if b - lead > 0 and a - lead < CLIP_S]
    return out


def label_snr_db(npz_path: str, labels: Sequence[Interval]) -> List[float]:
    """In-band level of each label over the clip's median frame level, on the mean signal."""
    from scipy import signal as sps

    z = np.load(npz_path)
    lead, sr = float(z["lead_s"]), 22_050
    x = resample_audio(z["capsules"].astype(np.float32).mean(axis=0), int(z["sr"]), sr)
    x = highpass_filter(x, sr, 5_000.0, 5)
    f, t, spec = sps.stft(x, fs=sr, nperseg=2048, noverlap=1536)
    e = (np.abs(spec[(f >= BAND_HZ[0]) & (f <= BAND_HZ[1])]) ** 2).sum(axis=0)
    t = t - lead
    floor = np.median(e[(t >= 0) & (t <= CLIP_S)])
    out = []
    for s, en in labels:
        sel = (t >= s) & (t <= en)
        out.append(float(10 * np.log10(np.mean(e[sel]) / floor)) if sel.any() else float("nan"))
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _ov(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _total(iv: Sequence[Interval]) -> float:
    return sum(e - s for s, e in _merge(iv))


def _intersect(a: Sequence[Interval], b: Sequence[Interval]) -> float:
    return sum(_ov(x, y) for x in _merge(a) for y in _merge(b))


def score_clip(dets: Sequence[Interval], calls: Sequence[Interval], unsure: Sequence[Interval]) -> Dict[str, float]:
    """Raw counts for one clip; ``calls`` are the labels that count, ``unsure`` are ignored."""
    ignored = [d for d in dets if not any(_ov(d, c) > 0 for c in calls) and any(_ov(d, u) > 0 for u in unsure)]
    scored = [d for d in dets if d not in ignored]
    # IoU >= 0.5, greedy one-to-one, best pairs first.
    pairs = sorted(((_ov(d, c) / (max(d[1], c[1]) - min(d[0], c[0])), i, j)
                    for i, d in enumerate(scored) for j, c in enumerate(calls)), reverse=True)
    used_d, used_c = set(), set()
    for iou, i, j in pairs:
        if iou < 0.5:
            break
        if i not in used_d and j not in used_c:
            used_d.add(i)
            used_c.add(j)
    chunks = chunk_recording(np.zeros(int(CLIP_S * 16_000), np.float32),
                             [CallSegment(s, e) for s, e in dets], PreprocessingConfig(), "x", dt.datetime(2021, 1, 1))
    windows = [(c.offset_s, c.offset_s + 10.0) for c in chunks]
    all_labels = list(calls) + list(unsure)
    return {
        "n_det": len(scored), "n_lab": len(calls),
        "det_hit": sum(any(_ov(d, c) > 0 for c in calls) for d in scored),
        "lab_hit": sum(any(_ov(c, d) > 0 for d in dets) for c in calls),
        "iou_tp": len(used_c),
        "lab_s": _total(calls), "det_s": _total(scored),
        "lab_cov_s": _intersect(calls, dets), "det_in_s": _intersect(scored, all_labels),
        "n_chunks": len(windows),
        "chunk_hit": sum(any(_ov(w, c) > 0 for c in all_labels) for w in windows),
        "lab_in_chunk": sum(any(w[0] <= (c[0] + c[1]) / 2 <= w[1] for w in windows) for c in calls),
    }


def summarise(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    p, r = div(s["det_hit"], s["n_det"]), div(s["lab_hit"], s["n_lab"])
    return {
        "P": p, "R": r, "F1": div(2 * p * r, p + r),
        "P_iou": div(s["iou_tp"], s["n_det"]), "R_iou": div(s["iou_tp"], s["n_lab"]),
        "P_time": div(s["det_in_s"], s["det_s"]), "R_time": div(s["lab_cov_s"], s["lab_s"]),
        "chunk_P": div(s["chunk_hit"], s["n_chunks"]), "chunk_R": div(s["lab_in_chunk"], s["n_lab"]),
        "n_det": s["n_det"], "n_lab": s["n_lab"], "n_chunks": s["n_chunks"],
    }


def score(clips_csv: str, labels_csv: str, audio_dir: str, json_out: str, workers: int,
          sweep: str, strategies: Sequence[str]) -> None:
    clips = {c["clip_id"]: c for c in csv.DictReader(open(clips_csv))}
    labels: Dict[str, List[Tuple[float, float, str]]] = {k: [] for k in clips}
    for r in csv.DictReader(open(labels_csv)):
        labels[r["clip_id"]].append((float(r["clip_start_s"]), float(r["clip_end_s"]), r["label"]))
    ids = [k for k in clips if (Path(audio_dir) / f"{k}.npz").exists()]
    if not ids:
        raise SystemExit(f"no <clip>.npz under {audio_dir}; run `extract` on the cluster first")
    paths = [str(Path(audio_dir) / f"{k}.npz") for k in ids]
    with ProcessPoolExecutor(workers) as ex:
        dets = dict(zip(ids, ex.map(partial(detect_clip, sweep=sweep, strategies=strategies), paths)))

    names = [n for n, _ in _configs(sweep)]
    results: Dict[str, Dict] = {"clips": {}, "summary": {}}
    for cid in ids:
        labs = labels[cid]
        results["clips"][cid] = {
            "day": clips[cid]["day"] == "1", "labels": labs,
            "snr_db": label_snr_db(str(Path(audio_dir) / f"{cid}.npz"), [(s, e) for s, e, _ in labs]),
            "detections": dets[cid],
        }
    for mode in ("c+?", "c"):
        for name in names:
            for strat in strategies:
                day_rows, night_det = [], 0
                for cid in ids:
                    labs = labels[cid]
                    calls = [(s, e) for s, e, t in labs if t == "c" or (mode == "c+?" and t == "?")]
                    unsure = [(s, e) for s, e, t in labs if t == "?"] if mode == "c" else []
                    d = dets[cid][name][strat]
                    if results["clips"][cid]["day"]:
                        day_rows.append(score_clip(d, calls, unsure))
                    else:
                        night_det += len(d)
                n_night = sum(not results["clips"][c]["day"] for c in ids)
                summ = summarise(day_rows)
                summ["night_det_per_min"] = night_det / max(1, n_night)
                results["summary"][f"{mode}|{name}|{strat}"] = summ
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(results, f, indent=1)
    report(results)
    print(f"\nwrote {json_out}")


def report(results: Dict) -> None:
    summ = results["summary"]
    for mode in ("c+?", "c"):
        print(f"\n=== day clips, labels counted: {mode} ===")
        print(f"{'config':<34}{'strategy':<16}{'P':>6}{'R':>6}{'F1':>6}{'P_iou':>7}{'R_iou':>7}"
              f"{'P_t':>6}{'R_t':>6}{'chP':>6}{'chR':>6}{'det':>6}{'nightFA/min':>12}")
        for key, s in summ.items():
            m, name, strat = key.split("|")
            if m != mode:
                continue
            print(f"{name:<34}{strat:<16}{s['P']:6.2f}{s['R']:6.2f}{s['F1']:6.2f}{s['P_iou']:7.2f}"
                  f"{s['R_iou']:7.2f}{s['P_time']:6.2f}{s['R_time']:6.2f}{s['chunk_P']:6.2f}"
                  f"{s['chunk_R']:6.2f}{s['n_det']:6d}{s['night_det_per_min']:12.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("clips_csv")
    e.add_argument("--out", required=True)
    e.add_argument("--rate", type=int, default=48_000, help="stored rate (default: native 48 kHz)")
    s = sub.add_parser("score")
    s.add_argument("clips_csv")
    s.add_argument("labels_csv")
    s.add_argument("--audio", required=True)
    s.add_argument("--json-out", required=True)
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.add_argument("--sweep", choices=sorted(SWEEPS), default="threshold")
    s.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    s.add_argument("--report-only", action="store_true", help="re-print tables from --json-out")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.clips_csv, a.out, a.rate)
    elif a.report_only:
        report(json.load(open(a.json_out)))
    else:
        score(a.clips_csv, a.labels_csv, a.audio, a.json_out, a.workers, a.sweep, a.strategies)


if __name__ == "__main__":
    main()
