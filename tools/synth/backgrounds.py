"""Background candidates: call-free night recordings, as the bed for synthetic recordings.

Step 2 of the synthetic evaluation (docs/SYNTH_EVAL.md). The planted calls must be the only
calls in a synthetic recording, so the background must hold none, and it must still sound
like the colony: a bed of noise near the level the detector meets between calls by day.

Pool [ours]. The recordings of a corpus run in which the production detector found no call
(``files/*.json`` with ``n_calls == 0``): nearly all are night hours. One per date, from
the heart of the night (``NIGHT_HOURS``), seeded, so the dates spread. A second, smaller
pool of day recordings is scanned only for the reference level (``kind="day"``).

Scan [ours], per recording, on the 4-capsule mean (``detection_channel_strategy="mean"``,
what the detector hears; the synthetic recordings are made from the same signal):

* ``detections``: the production detector rerun at ``LOW_THRESHOLD`` instead of
  ``learned_threshold`` (0.5): anything faintly call-like shows up here.
* ``level_db``: power in the call band ``BAND_HZ`` per second, in dB (full scale). The day
  reference is each day recording's 10th percentile (the background between calls),
  medianed over the day pool.
* a long-term spectrum (``<id>_ltsa.npy``): 250 Hz bands x 1 s, in dB, for the listening page.

Select [ours]. Night recordings with no detection at the low threshold, ranked by how close
their median level is to the day reference, at most one per date; the top ``--n`` are
proposed. The user listens on the library's ``/backgrounds`` page and approves.

Writes under OUT: ``scan/<recording_id>.json`` and ``scan/<recording_id>_ltsa.npy`` (one
array task each share), ``candidates.json`` and ``png/<recording_id>.png`` (``select``).

Usage (on the cluster, via tools/cluster/backgrounds.sbatch):
    python tools/synth/backgrounds.py scan RUN OUT [--task-index I --n-tasks N]
    python tools/synth/backgrounds.py select RUN OUT [--n 10]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import signal

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig, load_config
from marmoset_developmental_agent.preprocessing.streaming import FileSource, detect_recording

NIGHT_HOURS = (21, 22, 23, 0, 1, 2, 3)
DAY_HOURS = range(9, 17)
N_DAY = 30
MIN_DURATION_S = 600                             # a partial hour is too short to loop from
LOW_THRESHOLD = 0.1
BAND_HZ = (5_000, 20_000)                        # the call band, as the planned SNR definition
LTSA_BAND_HZ = 250                               # the learned detector's band width
BLOCK_S = 60


def pool(run: Path, seed: int = 0) -> List[Dict]:
    """The recordings to scan: one call-free night recording per date, and ``N_DAY`` day ones."""
    files = sorted((json.loads(p.read_text()) for p in (run / "files").glob("*.json")), key=lambda f: f["recording_id"])
    rng = random.Random(seed)
    nights: Dict[str, List[Dict]] = {}
    for f in files:
        t = dt.datetime.fromisoformat(f["start_time"])
        if f["n_calls"] == 0 and t.hour in NIGHT_HOURS and f["duration_s"] >= MIN_DURATION_S:
            nights.setdefault(f["start_time"][:10], []).append(f)
    out = [{**rng.choice(v), "kind": "night"} for _, v in sorted(nights.items())]
    days = [f for f in files if dt.datetime.fromisoformat(f["start_time"]).hour in DAY_HOURS and f["n_calls"] > 0]
    out += [{**f, "kind": "day"} for f in rng.sample(days, min(N_DAY, len(days)))]
    return out


def scan(rec: Dict, cfg: PreprocessingConfig) -> Dict:
    src = FileSource(rec["path"])
    sr = src.sample_rate
    hop = sr // 50                               # 50 frames a second, so seconds are whole frames
    levels, ltsa = [], []
    for s0 in range(0, src.frames, BLOCK_S * sr):
        x = src.read(s0, BLOCK_S * sr + 1024 - hop).mean(axis=1)    # + one window: the block's last second is whole
        if len(x) < sr:
            break
        f, _, z = signal.stft(x, sr, nperseg=1024, noverlap=1024 - hop, boundary=None, padded=False)
        p = np.abs(z) ** 2
        n = p.shape[1] // 50
        per_s = p[:, :n * 50].reshape(len(f), n, 50).mean(axis=2)
        band = (f >= BAND_HZ[0]) & (f < BAND_HZ[1])
        levels += list(10 * np.log10(per_s[band].sum(0) + 1e-20))
        edges = np.arange(0, sr / 2 + 1, LTSA_BAND_HZ)
        ltsa.append(np.stack([per_s[(f >= lo) & (f < hi)].mean(0) for lo, hi in zip(edges[:-1], edges[1:])]))
    low = dataclasses.replace(cfg, learned_threshold=LOW_THRESHOLD, exclude_night=False)
    found = detect_recording(src, dt.datetime.fromisoformat(rec["start_time"]), low)
    return {"recording_id": rec["recording_id"], "path": rec["path"], "start_time": rec["start_time"],
            "duration_s": src.frames / sr, "kind": rec["kind"], "level_db": [round(float(v), 1) for v in levels],
            "detections": [{"start_s": round(r.start_s, 2), "duration_s": round(r.duration_s, 3),
                            "confidence": round(r.confidence, 3), "peak_freq_hz": round(r.peak_freq_hz)} for r in found],
            "ltsa": 10 * np.log10(np.concatenate(ltsa, axis=1) + 1e-20)}


def select(out: Path, n: int) -> Dict:
    scans = [json.loads(p.read_text()) for p in sorted((out / "scan").glob("*.json"))]
    day = [s for s in scans if s["kind"] == "day"]
    ref = float(np.median([np.percentile(s["level_db"], 10) for s in day])) if day else float("nan")
    rows = []
    for s in scans:
        if s["kind"] != "night":
            continue
        lv = np.array(s["level_db"])
        rows.append({k: s[k] for k in ("recording_id", "path", "start_time", "duration_s")} | {
            "n_low": len(s["detections"]), "max_conf": max((d["confidence"] for d in s["detections"]), default=0.0),
            "level_med_db": round(float(np.median(lv)), 1), "level_p5_db": round(float(np.percentile(lv, 5)), 1),
            "level_p95_db": round(float(np.percentile(lv, 95)), 1), "detections": s["detections"],
            "level_db": s["level_db"]})
    rows.sort(key=lambda r: (r["n_low"] > 0, abs(r["level_med_db"] - ref)))
    dates = set()
    for r in rows:
        r["selected"] = r["n_low"] == 0 and r["start_time"][:10] not in dates and len(dates) < n
        if r["selected"]:
            dates.add(r["start_time"][:10])
    (out / "png").mkdir(exist_ok=True)
    for r in rows:
        if r["selected"]:
            _ltsa_png(np.load(out / "scan" / f"{r['recording_id']}_ltsa.npy").astype(np.float32), out / "png" / f"{r['recording_id']}.png")
    result = {"day_ref_db": round(ref, 1), "n_day": len(day), "low_threshold": LOW_THRESHOLD, "band_hz": BAND_HZ,
              "candidates": rows}
    (out / "candidates.json").write_text(json.dumps(result))
    return result


def _ltsa_png(ltsa: np.ndarray, path: Path) -> None:
    """The hour's long-term spectrum, 0-24 kHz by time in minutes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 2.4), dpi=80)
    ax = fig.add_axes([0.035, 0.14, 0.96, 0.82])
    hi = np.percentile(ltsa, 99.5)
    ax.imshow(ltsa, origin="lower", aspect="auto", cmap="magma", vmin=hi - 50, vmax=hi,
              extent=[0, ltsa.shape[1] / 60, 0, ltsa.shape[0] * LTSA_BAND_HZ / 1000])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=7, colors="#555")
    fig.savefig(path, format="png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["scan", "select"])
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    if a.step == "select":
        r = select(a.out, a.n)
        print(f"day reference {r['day_ref_db']} dB from {r['n_day']} day recordings")
        for c in r["candidates"]:
            print(f"{'*' if c['selected'] else ' '} {c['recording_id']} {c['start_time'][:16]} low-thr {c['n_low']:3d} "
                  f"level {c['level_med_db']} dB [{c['level_p5_db']}, {c['level_p95_db']}]")
        return
    cfg = load_config().preprocessing
    (a.out / "scan").mkdir(parents=True, exist_ok=True)
    for rec in pool(a.run)[a.task_index::a.n_tasks]:
        dest = a.out / "scan" / f"{rec['recording_id']}.json"
        if dest.exists():
            continue
        s = scan(rec, cfg)
        np.save(a.out / "scan" / f"{rec['recording_id']}_ltsa.npy", s.pop("ltsa").astype(np.float16))
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(s))
        os.replace(tmp, dest)
        print(f"{rec['recording_id']} {rec['kind']}: {len(s['detections'])} low-threshold detections, "
              f"median level {np.median(s['level_db']):.1f} dB", flush=True)


if __name__ == "__main__":
    main()
