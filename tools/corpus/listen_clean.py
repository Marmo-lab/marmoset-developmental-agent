"""Listening page: original vs spectrally cleaned calls, sampled across the cleaned recordings.

Reads a run's ``clean/<id>.json`` (recordings cleaned by ``clean_corpus.py``), draws calls
at random (fixed seed), at most ``--per-recording`` from each, and writes a self-contained
folder: each call as original and cleaned WAV (peak-normalised separately, so compare
cleanliness and timbre, not loudness) with spectrograms on one shared 60 dB scale below
the original's peak, plus the verdict buttons of ``tools/labels/listen_feedback.py``.

Usage (where the run's audio is, i.e. the cluster):
    python tools/corpus/listen_clean.py RUN OUT [--n 60] [--per-recording 1] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

from tools.labels.listen_feedback import add_feedback  # noqa: E402

STYLE = ("body{font-family:system-ui,sans-serif;margin:16px;max-width:1250px;direction:rtl}.pair{display:flex;gap:10px;"
         "flex-wrap:wrap;border-top:1px solid #ccc;padding-top:8px;margin-top:14px}.ex{width:300px;font-size:12px}"
         "img{width:300px;height:150px;display:block}audio{width:300px}")


def _slice(path: Path, row: Dict[str, str]) -> Tuple[np.ndarray, int]:
    sr = sf.info(path).samplerate
    with sf.SoundFile(path) as f:
        f.seek(int(round(float(row["audio_offset_s"]) * sr)))
        return f.read(int(round(float(row["audio_duration_s"]) * sr)), dtype="float32"), sr


def _png(x: np.ndarray, sr: int, ref_db: float, lead: float, dur: float, path: Path) -> None:
    f, t, s = signal.stft(x, sr, nperseg=1024, noverlap=784)
    p = 20 * np.log10(np.abs(s) + 1e-12) - ref_db
    fig, ax = plt.subplots(figsize=(4.3, 2.1))
    ax.imshow(p, origin="lower", aspect="auto", cmap="magma", vmin=-60, vmax=0, extent=[0, len(x) / sr, 0, sr / 2000])
    ax.axvline(lead, color="c", lw=0.8)
    ax.axvline(lead + dur, color="c", lw=0.8)
    ax.set_yticks([0, 8, 16, 24])
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def build(run: Path, out: Path, n: int, per_recording: int, seed: int) -> int:
    rng = random.Random(seed)
    rids = sorted(p.stem for p in (run / "clean").glob("*.json"))
    picks: List[Tuple[str, Dict[str, str]]] = []
    for rid in rids:
        rows = list(csv.DictReader(open(run / "calls" / f"{rid}.csv")))
        picks += [(rid, r) for r in rng.sample(rows, min(per_recording, len(rows)))]
    rng.shuffle(picks)
    picks = sorted(picks[:n], key=lambda p: p[1]["start_time"])
    (out / "a").mkdir(parents=True, exist_ok=True)
    html = [f"<!doctype html><html><head><meta charset='utf-8'><title>Cleaned calls</title><style>{STYLE}</style></head>"
            f"<body><h1>ניקוי ספקטרלי על הקורפוס: {len(picks)} קריאות אקראיות מ‑{len(rids)} הקלטות</h1>"
            "<p>לכל קריאה: <b>מקור</b> מול <b>נקי</b>, עם 50 ms של הקשר מכל צד (בין הקווים הכחולים: הקריאה). "
            "כל גרסה מנורמלת לשיא של עצמה בהשמעה; בתמונות, שתיהן באותה סקאלה של 60 dB מתחת לשיא המקור. "
            "סמן ליד ה<b>נקי</b>: ✓ אם נקי ונשמע כמו הקריאה, ✗ אם נפגע או לא קריאה, ↔ לגבולות, והערה על מה שנשאר.</p>"]
    for rid, r in picks:
        orig, sr = _slice(run / "call_audio" / f"{rid}.flac", r)
        clean, _ = _slice(run / "call_audio_clean" / f"{rid}.flac", r)
        ref = 20 * np.log10(np.abs(signal.stft(orig, sr, nperseg=1024, noverlap=784)[2]).max() + 1e-12)
        lead, dur = float(r["audio_lead_s"]), float(r["duration_s"])
        cid = f"{rid}_{float(r['start_s']):.2f}"
        cells = []
        for tag, x, name in (("orig", orig, "מקור"), ("clean", clean, "נקי")):
            sf.write(out / "a" / f"{cid}_{tag}.wav", 0.8 * x / (np.abs(x).max() + 1e-12), sr, subtype="PCM_16")
            _png(x, sr, ref, lead, dur, out / "a" / f"{cid}_{tag}.png")
            cells.append(f"<div class='ex'><b>{name}</b><img src='a/{cid}_{tag}.png'><audio controls preload='none' "
                         f"src='a/{cid}_{tag}.wav'></audio></div>")
        html.append(f"<div class='pair'><div style='width:100%'>{r['start_time'][:19]} · {dur:.2f} s · conf "
                    f"{float(r['confidence']):.2f} · SNR {float(r['snr_db']):.0f} dB</div>{''.join(cells)}</div>")
    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html))
    add_feedback(out / "index.html")
    return len(picks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--per-recording", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    k = build(Path(a.run), Path(a.out), a.n, a.per_recording, a.seed)
    print(f"{k} calls -> {a.out}/index.html")


if __name__ == "__main__":
    main()
