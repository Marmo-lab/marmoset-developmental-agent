"""Listening package for a corpus run: every detection of one fixed tone, and random chunks.

Two things are hard to judge from numbers alone: whether a fixed-frequency "call" found by
``summarize_corpus.py`` is a device sound, and whether the training chunks sound clean. This
writes, under ``--out``, an ``index.html`` with players and spectrograms (open it locally;
nothing is uploaded) and:

``tone/``
    Every catalogue call whose dominant frequency is within 1.5 STFT bins of ``--tone-hz``
    and whose frequency IQR is within ``FLAT_IQR_HZ`` (the fixed-tone rule of
    ``summarize_corpus.py``), duplicate recordings removed (``merge_corpus.deduplicate``).
    ``clips/`` holds each detection with 0.5 s of context, from the raw file (4-capsule
    mean, 32 kHz, peak-normalised). ``<date>.wav`` stitches one date's detections with
    0.4 s of silence between them, high-passed at the detector's 5 kHz so the tone is not
    masked by low-frequency room noise. ``tone.csv`` lists them all.
``chunks/``
    ``--n-chunks`` chunks drawn at random (seeded) from ``manifest.tsv``, i.e. the training
    audio itself (16 kHz, ``ambisonic_beam``), each with a spectrogram marking its calls.
    The raw level is far below full scale, so each chunk is peak-normalised for
    listening: loudness across chunks means nothing.

Usage (cluster; reads the run and raw audio, writes only under --out):
    python tools/corpus/listen_corpus.py $WORK_DIR/corpus/v2_day --out $WORK_DIR/listen_v2
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

from marmoset_developmental_agent.preprocessing.filter_5khz import highpass_filter, resample_audio
from tools.corpus.merge_corpus import deduplicate
from tools.corpus.summarize_corpus import BIN_HZ, FLAT_IQR_HZ

LISTEN_SR = 32_000  # keeps everything below 16 kHz, the detection ceiling
CONTEXT_S = 0.5
GAP_S = 0.4


def _peak_norm(x: np.ndarray, dbfs: float = -3.0) -> np.ndarray:
    return (x * (10 ** (dbfs / 20) / max(float(np.abs(x).max()), 1e-9))).astype(np.float32)


def _spectrogram_png(x: np.ndarray, sr: int, spans: List[Tuple[float, float]], title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    f, t, s = signal.spectrogram(x, sr, nperseg=512, noverlap=384)
    db = 10 * np.log10(s + 1e-14)
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.pcolormesh(t, f / 1000, db, shading="auto", vmin=np.percentile(db, 30), vmax=db.max())
    for a, b in spans:
        ax.axvspan(a, b, ymin=0.96, ymax=1.0, color="orange")
    ax.set(ylabel="kHz", xlabel="s", title=title)
    fig.tight_layout()
    fig.savefig(out, dpi=60)
    plt.close(fig)


def _read_raw(path: str, start_s: float, end_s: float) -> Tuple[np.ndarray, int]:
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        a = max(0, int(start_s * sr))
        f.seek(a)
        x = f.read(int(end_s * sr) - a, dtype="float32", always_2d=True).mean(axis=1)
    return x, sr


def tone_package(run: Path, out: Path, tone_hz: float) -> List[Tuple[str, int]]:
    """Write ``out/tone``; return ``(date, n_detections)`` per date."""
    files = {(f := json.loads(p.read_text()))["recording_id"]: f for p in (run / "files").glob("*.json")}
    keep, _ = deduplicate(list(files.values()))
    hits = []
    for rid in keep:
        for r in csv.DictReader(open(run / "calls" / f"{rid}.csv")):
            if (abs(float(r["peak_freq_hz"]) - tone_hz) <= 1.5 * BIN_HZ
                    and float(r["peak_freq_iqr_hz"]) <= FLAT_IQR_HZ):
                hits.append((r["start_time"], rid, float(r["start_s"]), float(r["end_s"])))
    hits.sort()
    (out / "tone" / "clips").mkdir(parents=True, exist_ok=True)
    by_date: Dict[str, List[np.ndarray]] = defaultdict(list)
    with open(out / "tone" / "tone.csv", "w") as fh:
        fh.write("clip,start_time,recording_id,path,start_s,end_s\n")
        for k, (t, rid, s, e) in enumerate(hits):
            x, sr = _read_raw(files[rid]["path"], s - CONTEXT_S, e + CONTEXT_S)
            name = f"{k:04d}_{t[:19].replace(':', '')}.wav"
            sf.write(out / "tone" / "clips" / name, _peak_norm(resample_audio(x, sr, LISTEN_SR)), LISTEN_SR,
                     subtype="PCM_16")
            hp = resample_audio(highpass_filter(x, sr, 5_000.0), sr, LISTEN_SR)
            by_date[t[:10]] += [_peak_norm(hp), np.zeros(int(GAP_S * LISTEN_SR), np.float32)]
            fh.write(f"{name},{t},{rid},{files[rid]['path']},{s:.3f},{e:.3f}\n")
    for date, parts in by_date.items():
        sf.write(out / "tone" / f"{date}.wav", np.concatenate(parts), LISTEN_SR, subtype="PCM_16")
    return [(d, len(p) // 2) for d, p in sorted(by_date.items())]


def chunk_package(run: Path, out: Path, n: int, seed: int) -> List[Dict[str, str]]:
    """Write ``out/chunks``; return the chosen manifest rows in chronological order."""
    rows = list(csv.DictReader(open(run / "manifest.tsv"), delimiter="\t"))
    chosen = sorted(random.Random(seed).sample(rows, n), key=lambda r: r["start_time"])
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(chosen):
        x, sr = sf.read(run / r["path"], dtype="float32")
        spans = [tuple(map(float, c.split("-"))) for c in r["calls"].split(";") if c]
        stem = f"{k:02d}_{r['start_time'][:19].replace(':', '')}"
        sf.write(out / "chunks" / f"{stem}.wav", _peak_norm(x), sr, subtype="PCM_16")
        _spectrogram_png(x, sr, spans, f"{r['start_time'][:19]}  {r['recording_id']}  {len(spans)} calls",
                         out / "chunks" / f"{stem}.png")
        r["stem"] = stem
    return chosen


def write_index(out: Path, tone_hz: float, dates: List[Tuple[str, int]], chunks: List[Dict[str, str]]) -> None:
    def audio(src: str) -> str:
        return f'<audio controls preload="none" src="{html.escape(src)}"></audio>'

    parts = [f"""<!doctype html><html><head><meta charset="utf-8"><title>Corpus listening</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;max-width:1100px}}td{{padding:4px 8px}}
img{{max-width:100%}}.chunk{{margin:18px 0;border-top:1px solid #ccc;padding-top:8px}}</style></head><body>
<h1>Corpus listening ({dt.date.today()})</h1>
<h2>{tone_hz:.0f} Hz fixed tone: {sum(n for _, n in dates)} detections</h2>
<p>One file per date: every detection of that date, high-passed at 5 kHz, peak-normalised, 0.4 s apart.
Individual raw clips (0.5 s context) are in <code>tone/clips/</code>, listed in <code>tone/tone.csv</code>.</p>
<table>"""]
    parts += [f"<tr><td>{d}</td><td>{n}</td><td>{audio(f'tone/{d}.wav')}</td></tr>" for d, n in dates]
    parts.append(f"""</table><h2>{len(chunks)} random training chunks</h2>
<p>Drawn at random from manifest.tsv, chronological. The 16 kHz training audio itself, peak-normalised for
listening. Orange bars mark the detected calls.</p>""")
    for r in chunks:
        parts.append(f'<div class="chunk"><b>{r["start_time"][:19]}</b> {html.escape(r["recording_id"])} '
                     f'(day {r["day"]})<br>{audio("chunks/" + r["stem"] + ".wav")}<br>'
                     f'<img src="chunks/{r["stem"]}.png"></div>')
    parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="corpus run directory with manifest.tsv (merge_corpus.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tone-hz", type=float, default=10_875.0)
    ap.add_argument("--n-chunks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    dates = tone_package(run, out, a.tone_hz)
    chunks = chunk_package(run, out, a.n_chunks, a.seed)
    write_index(out, a.tone_hz, dates, chunks)
    print(f"{sum(n for _, n in dates)} tone detections on {len(dates)} dates, {len(chunks)} chunks -> {out}/index.html")


if __name__ == "__main__":
    main()
