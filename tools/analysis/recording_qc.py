"""Quality check of a newly copied recording folder: did the recorder capture sound, and what.

Why. New colony recordings (from 2026-11, Zoom H3-VR at 96 kHz, ambisonic A-format) arrive
outside the 48 kHz manifest, and the recorder can stop without finalising its last file
(RIFF/data sizes left at 0 while the audio bytes are on disk). Before any pipeline work we
need to know, per file and over time, whether each capsule recorded, at what level, whether
it clipped, and whether there is energy in the call band. [ours]

Reads PCM_24 directly from the ``data`` chunk (``pcm_view``), so a file whose header was never
finalised is read for the bytes it holds rather than reported empty. Per ``WIN_S`` window it
writes: RMS per capsule (dBFS), the fraction of clipped samples, and the mean-capsule level in
``BANDS_HZ`` (dB, from a Welch PSD). Per file it also cuts two ``CLIP_S`` excerpts of the
capsule mean for listening: the window with the most 5-12 kHz energy relative to its file
median (``loud``) and the median window (``typical``).

Usage:
    python tools/analysis/recording_qc.py scan FOLDER OUT --task K --n-tasks N
    python tools/analysis/recording_qc.py summary OUT
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import welch

from marmoset_developmental_agent.io.raw_reader import read_bext_datetime

WIN_S = 10.0
CLIP_S = 15.0
CLIP_LEVEL = 0.999
BANDS_HZ: List[Tuple[float, float]] = [(100, 1_000), (1_000, 5_000), (5_000, 12_000), (12_000, 24_000), (24_000, 46_000)]
CALL_BAND = 2                          # 5-12 kHz: phee/trill/twitter fundamentals [ours, DETECTOR_EVAL.md]
FIELDS = ["file", "t_s", "wall"] + [f"rms{c}" for c in range(4)] + ["clip_frac"] + [f"band{b}" for b in range(len(BANDS_HZ))]


def pcm_view(path: Path) -> Tuple[np.ndarray, int, int]:
    """``(bytes as uint8 (frames, channels*3), sample_rate, channels)`` of a 24-bit PCM WAV.

    The frame count comes from the file size, not the ``data`` chunk size, which an
    interrupted recording leaves at 0.
    """
    with open(path, "rb") as f:
        head = f.read(1 << 17)
    i, sr, ch = 12, 0, 0
    while i < len(head) - 8:
        cid, size = head[i:i + 4], struct.unpack("<I", head[i + 4:i + 8])[0]
        if cid == b"fmt ":
            _, ch, sr, _, _, bits = struct.unpack("<HHIIHH", head[i + 8:i + 24])
            if bits != 24:
                raise ValueError(f"{path}: {bits}-bit, expected 24")
        if cid == b"data":
            off = i + 8
            n = (path.stat().st_size - off) // (3 * ch)
            return np.memmap(path, np.uint8, "r", off, (n, 3 * ch)), sr, ch
        i += 8 + size + (size & 1)
    raise ValueError(f"{path}: no data chunk")


def decode(raw: np.ndarray, ch: int) -> np.ndarray:
    """uint8 ``(n, ch*3)`` little-endian int24 -> float32 ``(n, ch)`` in [-1, 1)."""
    b = raw.reshape(len(raw), ch, 3).astype(np.int32)
    v = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    v = np.where(v >= 1 << 23, v - (1 << 24), v)
    return (v / float(1 << 23)).astype(np.float32)


def _db(p: float) -> float:
    return float(10 * np.log10(max(p, 1e-20)))


def window_stats(x: np.ndarray, sr: int) -> Dict[str, float]:
    m = x.mean(axis=1)
    f, p = welch(m, sr, nperseg=4096)
    out = {f"rms{c}": _db(float(np.mean(x[:, c].astype(np.float64) ** 2))) for c in range(x.shape[1])}
    out["clip_frac"] = float(np.mean(np.abs(x) >= CLIP_LEVEL))
    for k, (lo, hi) in enumerate(BANDS_HZ):
        sel = (f >= lo) & (f < min(hi, sr / 2))
        out[f"band{k}"] = _db(float(p[sel].sum() * (f[1] - f[0]))) if sel.any() else float("nan")
    return out


def scan_file(path: Path, out: Path) -> List[Dict]:
    raw, sr, ch = pcm_view(path)
    start = read_bext_datetime(path)
    win = int(WIN_S * sr)
    rows = []
    for k in range(len(raw) // win):
        r = {"file": path.name, "t_s": k * WIN_S,
             "wall": (start + dt.timedelta(seconds=k * WIN_S)).isoformat() if start else ""}
        r.update(window_stats(decode(np.asarray(raw[k * win:(k + 1) * win]), ch), sr))
        rows.append(r)
    if rows:
        call = np.array([r[f"band{CALL_BAND}"] for r in rows])
        (out / "clips").mkdir(parents=True, exist_ok=True)
        for tag, k in (("loud", int(np.argmax(call - np.median(call)))), ("typical", int(np.argsort(call)[len(call) // 2]))):
            a = int(rows[k]["t_s"] * sr)
            seg = decode(np.asarray(raw[a:a + int(CLIP_S * sr)]), ch).mean(axis=1)
            sf.write(out / "clips" / f"{path.stem}_{tag}_{int(rows[k]['t_s'])}s.flac", seg, sr, subtype="PCM_24")
    return rows


def scan(folder: Path, out: Path, task: int, n_tasks: int) -> None:
    files = sorted(folder.glob("*.WAV"))[task::n_tasks]
    (out / "windows").mkdir(parents=True, exist_ok=True)
    for p in files:
        rows = scan_file(p, out)
        with open(out / "windows" / f"{p.stem}.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"{p.name}: {len(rows)} windows", flush=True)


def summary(out: Path) -> None:
    """All windows into ``windows.tsv`` and a level-over-time figure ``levels.png``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for p in sorted((out / "windows").glob("*.tsv")) for r in csv.DictReader(open(p), delimiter="\t")]
    rows = [r for r in rows if r["wall"]]
    rows.sort(key=lambda r: r["wall"])
    with open(out / "windows.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    t = np.array([dt.datetime.fromisoformat(r["wall"]) for r in rows])
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for c in range(4):
        ax[0].plot(t, [float(r[f"rms{c}"]) for r in rows], lw=0.6, label=f"capsule {c + 1}")
    ax[0].set_ylabel("RMS (dBFS)")
    ax[0].legend(ncol=4, fontsize=8, frameon=False)
    for k, (lo, hi) in enumerate(BANDS_HZ):
        ax[1].plot(t, [float(r[f"band{k}"]) for r in rows], lw=0.6, label=f"{lo / 1000:g}-{hi / 1000:g} kHz")
    ax[1].set_ylabel("band level (dBFS)")
    ax[1].legend(ncol=5, fontsize=8, frameon=False)
    ax[2].plot(t, [100 * float(r["clip_frac"]) for r in rows], lw=0.6, color="k")
    ax[2].set_ylabel("clipped samples (%)")
    ax[2].set_xlabel("recorder clock (bext)")
    for a in ax:
        a.grid(alpha=0.3, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "levels.png", dpi=130)
    span = (t[-1] - t[0]).total_seconds() / 3600 if len(t) else 0
    print(f"{len(rows)} windows = {len(rows) * WIN_S / 3600:.2f} h, {t[0]} -> {t[-1]} ({span:.2f} h span)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("folder", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--task", type=int, default=0)
    s.add_argument("--n-tasks", type=int, default=1)
    m = sub.add_parser("summary")
    m.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a.folder, a.out, a.task, a.n_tasks)
    else:
        summary(a.out)


if __name__ == "__main__":
    main()
