"""Summarise a corpus run (``detect_corpus.py``): what the training corpus will look like.

Reads ``files/*.json`` and ``calls/*.csv`` (no audio) and reports:

1. **Coverage**: recordings and hours done, out of the manifest; failures.
2. **Per recording date**: audio hours, calls, chunks, and chunk-hours, i.e. how much
   training audio each day of the colony's life yields (``per_date.csv``).
3. **Time of day**: calls per audio-hour for each clock hour, which settles the night
   question (OPEN_DECISIONS; DETECTOR_EVAL finding 7).
4. **Durations**: detected call durations against the hand labels'.
5. **Quality**: SNR and tonal-fraction distributions of the detected calls.
6. **Fixed tones**: a device sound (the door beep, DETECTOR_EVAL finding 10) has an almost
   constant dominant frequency and recurs at exactly that frequency. Among calls whose
   frequency IQR is within ``FLAT_IQR_HZ``, the most frequent STFT bins are listed with their
   counts and clock hours, and ``fixed_tone_examples.csv`` lists examples to listen to.
7. **Labelled minutes** (``--labels CLIPS LABELS``, repeatable): the catalogue restricted to
   each labelled clip, scored as in ``tools/analysis/score_detectors.py``. This checks that
   the production path reproduces the evaluated detector.

Usage:
    python tools/corpus/summarize_corpus.py OUT MANIFEST \
        [--labels labeling/clips.csv data/labels/handlabels.csv] \
        [--labels labeling/heldout/clips.csv data/labels/heldout_labels.csv]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from tools.analysis.score_detectors import CLIP_S, score_clip, summarise

BIN_HZ = 48_000 / 2048  # one STFT bin at the production detection rate
FLAT_IQR_HZ = 1.5 * BIN_HZ
DURATION_BINS = [0.0, 0.3, 0.5, 1.0, 2.0, 4.1]


def _calls(out: Path, rid: str) -> List[Dict[str, float]]:
    rows = list(csv.DictReader(open(out / "calls" / f"{rid}.csv")))
    for r in rows:
        for k in ("start_s", "end_s", "duration_s", "snr_db", "tonal_fraction", "peak_freq_hz", "peak_freq_iqr_hz"):
            r[k] = float(r[k])
    return rows


def _hours_by_clock(manifest: Sequence[Dict[str, str]]) -> np.ndarray:
    """Audio hours recorded in each clock hour, on a one-minute grid."""
    h = np.zeros(24)
    for m in manifest:
        start, dur = dt.datetime.fromisoformat(m["start_time"]), float(m["duration_s"])
        for k in range(int(dur // 60)):
            h[(start + dt.timedelta(minutes=k)).hour] += 1 / 60
        h[(start + dt.timedelta(seconds=dur)).hour] += (dur % 60) / 3600
    return h


def _quantiles(x: Sequence[float], qs: Sequence[float] = (5, 25, 50, 75, 95)) -> str:
    v = np.percentile(np.asarray(x, dtype=float), qs) if len(x) else [float("nan")] * len(qs)
    return "  ".join(f"p{q}={a:.2f}" for q, a in zip(qs, v))


def _histogram(x: Sequence[float]) -> str:
    counts, _ = np.histogram(x, DURATION_BINS)
    n = max(1, len(x))
    return "  ".join(f"{a:g}-{b:g}s {c / n:5.1%}" for a, b, c in zip(DURATION_BINS, DURATION_BINS[1:], counts))


def label_check(files: Dict[str, Dict], out: Path, clips_csv: str, labels_csv: str) -> Dict[str, object]:
    """Score the catalogue inside each labelled minute whose source recording was processed."""
    by_path = {f["path"]: rid for rid, f in files.items()}
    labels: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for r in csv.DictReader(open(labels_csv)):
        labels[r["clip_id"]].append((float(r["clip_start_s"]), float(r["clip_end_s"]), r["label"]))
    day_rows, night_det, n_night, missing = [], 0, 0, []
    for c in csv.DictReader(open(clips_csv)):
        rid = by_path.get(c["source_path"])
        if rid is None:
            missing.append(c["clip_id"])
            continue
        off = float(c["offset_s"])
        dets = [(max(0.0, r["start_s"] - off), min(CLIP_S, r["end_s"] - off)) for r in _calls(out, rid)
                if r["end_s"] > off and r["start_s"] < off + CLIP_S]
        if c["day"] == "1":
            day_rows.append(score_clip(dets, [(s, e) for s, e, _ in labels[c["clip_id"]]], []))
        else:
            night_det += len(dets)
            n_night += 1
    res: Dict[str, object] = {"clips_scored": len(day_rows) + n_night, "clips_missing": missing}
    if day_rows:
        res["day_c+?"] = summarise(day_rows)
    if n_night:
        res["night_det_per_min"] = night_det / n_night
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("manifest")
    ap.add_argument("--labels", nargs=2, action="append", default=[], metavar=("CLIPS_CSV", "LABELS_CSV"))
    ap.add_argument("--n-examples", type=int, default=20, help="examples per fixed-tone frequency")
    a = ap.parse_args()
    out = Path(a.out)
    manifest = list(csv.DictReader(open(a.manifest)))
    files = {json.loads(p.read_text())["recording_id"]: json.loads(p.read_text()) for p in (out / "files").glob("*.json")}
    errors = sorted(p.stem for p in (out / "errors").glob("*.txt")) if (out / "errors").exists() else []
    report: Dict[str, object] = {}

    # 1. coverage
    done_h = sum(f["duration_s"] for f in files.values()) / 3600
    total_h = sum(float(m["duration_s"]) for m in manifest) / 3600
    print(f"== coverage: {len(files)}/{len(manifest)} recordings, {done_h:.0f}/{total_h:.0f} h; "
          f"{len(errors)} failed{': ' + ', '.join(errors[:5]) if errors else ''}")
    report["coverage"] = {"recordings": len(files), "of": len(manifest), "hours": done_h, "errors": errors}

    calls: List[Dict[str, float]] = []
    per_date: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_clock_calls = np.zeros(24)
    for rid, f in files.items():
        rows = _calls(out, rid)
        calls += [{**r, "recording_id": rid, "path": f["path"]} for r in rows]
        d = per_date[f["start_time"][:10]]
        d["audio_h"] += f["duration_s"] / 3600
        d["calls"] += f["n_calls"]
        d["chunks"] += f["n_chunks"]
        d["call_s"] += f["call_s"]
        for r in rows:
            by_clock_calls[dt.datetime.fromisoformat(r["start_time"]).hour] += 1
    print(f"   {len(calls)} calls, {sum(f['n_chunks'] for f in files.values())} chunks "
          f"({sum(f['n_chunks'] for f in files.values()) * 10 / 3600:.0f} h of 10 s chunks)")

    # 2. per date
    with open(out / "per_date.csv", "w") as fh:
        fh.write("date,audio_h,calls,chunks,chunk_h,calls_per_audio_h\n")
        for day in sorted(per_date):
            d = per_date[day]
            fh.write(f"{day},{d['audio_h']:.2f},{d['calls']:.0f},{d['chunks']:.0f},{d['chunks'] * 10 / 3600:.2f},"
                     f"{d['calls'] / max(d['audio_h'], 1e-9):.1f}\n")
    chunk_h = np.array([per_date[k]["chunks"] * 10 / 3600 for k in sorted(per_date)])
    if len(chunk_h):
        print(f"== per date ({len(per_date)} dates, per_date.csv): chunk-hours per date {_quantiles(chunk_h)}")
        report["chunk_h_per_date"] = {"quantiles": _quantiles(chunk_h), "n_dates": len(per_date)}

    # 3. time of day
    processed = [m for m in manifest if m["path"] in {f["path"] for f in files.values()}]
    hours = _hours_by_clock(processed)
    rate = by_clock_calls / np.maximum(hours, 1e-9)
    print("== calls per audio-hour by clock hour")
    print("   " + "  ".join(f"{h:02d}:{rate[h]:.0f}" for h in range(24)))
    report["calls_per_audio_h_by_hour"] = rate.round(1).tolist()

    # 4. durations
    det_d = [c["duration_s"] for c in calls]
    print("== durations")
    print(f"   detected  {_quantiles(det_d)}\n             {_histogram(det_d)}")
    lab_d: List[float] = []
    for _, labels_csv in a.labels:
        lab_d += [float(r["clip_end_s"]) - float(r["clip_start_s"]) for r in csv.DictReader(open(labels_csv))]
    if lab_d:
        print(f"   labels    {_quantiles(lab_d)}\n             {_histogram(lab_d)}")
    report["durations"] = {"detected": _quantiles(det_d), "labels": _quantiles(lab_d) if lab_d else None}

    # 5. quality
    snr = [c["snr_db"] for c in calls if not np.isnan(c["snr_db"])]
    ton = [c["tonal_fraction"] for c in calls if not np.isnan(c["tonal_fraction"])]
    print(f"== quality\n   SNR dB         {_quantiles(snr)}\n   tonal fraction {_quantiles(ton)}")
    report["quality"] = {"snr_db": _quantiles(snr), "tonal_fraction": _quantiles(ton)}

    # 6. fixed tones
    flat = [c for c in calls if c["peak_freq_iqr_hz"] <= FLAT_IQR_HZ]
    bins = Counter(round(c["peak_freq_hz"] / BIN_HZ) for c in flat)
    print(f"== fixed tones: {len(flat)} calls ({len(flat) / max(1, len(calls)):.1%}) have a frequency IQR "
          f"<= {FLAT_IQR_HZ:.0f} Hz; most common bins:")
    tones = []
    with open(out / "fixed_tone_examples.csv", "w") as fh:
        fh.write("freq_hz,recording_id,path,start_s,end_s,start_time\n")
        for b, n in bins.most_common(10):
            members = [c for c in flat if round(c["peak_freq_hz"] / BIN_HZ) == b]
            clock = Counter(dt.datetime.fromisoformat(c["start_time"]).hour for c in members).most_common(3)
            print(f"   {b * BIN_HZ:7.0f} Hz  {n:6d} calls  top hours {clock}")
            tones.append({"freq_hz": b * BIN_HZ, "n": n, "top_hours": clock})
            step = max(1, len(members) // a.n_examples)
            for c in members[::step][: a.n_examples]:
                fh.write(f"{b * BIN_HZ:.0f},{c['recording_id']},{c['path']},{c['start_s']:.3f},{c['end_s']:.3f},"
                         f"{c['start_time']}\n")
    report["fixed_tones"] = {"n_flat": len(flat), "top": tones}

    # 7. labelled minutes
    for clips_csv, labels_csv in a.labels:
        res = label_check(files, out, clips_csv, labels_csv)
        report[f"labels:{labels_csv}"] = res
        s = res.get("day_c+?")
        line = (f"P {s['P']:.2f} R {s['R']:.2f} F1 {s['F1']:.2f} chunk P {s['chunk_P']:.2f} chunk R {s['chunk_R']:.2f}"
                if s else "no day clips")
        print(f"== labelled minutes, {labels_csv}: {res['clips_scored']} clips ({len(res['clips_missing'])} not "
              f"processed): {line}; night det/min {res.get('night_det_per_min', float('nan')):.2f}")

    (out / "summary.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {out / 'summary.json'}, {out / 'per_date.csv'}, {out / 'fixed_tone_examples.csv'}")


if __name__ == "__main__":
    main()
