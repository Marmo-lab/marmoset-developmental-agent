"""Repertoire figure: the user's chosen calls, original audio, spectrogram with the detector's call marked below.

Why. A presentation figure (2026-11-14) of what the production detector finds in the colony. The user picked
the calls on the library's ``/review`` page (``figure_candidates.py``); no call kinds are named, because we
detect but do not classify. Audio: the original 4-capsule mean from the raw recording (what the detector
hears), uncleaned, with ``figure_candidates.PAD_S`` on each side.

Layout (with the user): one panel per call, waveform on top, spectrogram, and the detector's call marked below.
Spectrogram [ours]: 1024-point Hann window (21 ms) with a 1 ms hop, power in dB relative to the panel's
loudest bin inside the shown band, over ``DB_RANGE`` dB, ``FMIN``-``FMAX`` kHz (5-15 kHz at Didi's request,
2026-11-14: where the calls' fundamentals are).

Usage: python tools/explorations/repertoire_figure.py CANDIDATES_DIR OUT_STEM [--ids ID ...]
    (default: the items the user marked "pick", in page order)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

FMIN, FMAX = 5.0, 15.0
DB_RANGE = 55.0
TITLE = "Automated detection of spontaneous marmoset vocalizations in the colony"
SUBTITLE = "Step 1 toward a language model for marmoset communication"
# v5_calls, duplicates dropped (merge_calls): 2,602 recordings, 2,658.1 h over 137 days; 1,849,363 calls.
STATS = "2,658 recording hours  ·  1.85 million detected calls  ·  17 animals in the colony"
BAR = "#2a9d8f"
# The waveform is band-passed to the same band the spectrogram shows: the room's rumble and broadband noise
# otherwise fill the trace and hide a quiet call. The spectrogram itself is the original audio, cropped.
HP = signal.butter(4, (FMIN * 1000, FMAX * 1000), btype="bandpass", fs=48_000, output="sos")


def picked(folder: Path) -> List[str]:
    v = {}
    for line in (folder / "review_feedback.jsonl").read_text().splitlines():
        d = json.loads(line)
        v[d["item_id"]] = d.get("verdict")
    order = [r["item_id"] for r in csv.DictReader(open(folder / "items.tsv"), delimiter="\t")]
    return list(dict.fromkeys(i for i in order if v.get(i) == "pick"))     # a call can sit in two bins: once


def draw(folder: Path, ids: List[str], out: Path, cols: int = 3) -> None:
    """One panel per call in a grid: waveform on top, spectrogram, the detection below.

    Panels are the same size, so each has its own time axis (labelled); frequency and colour are the same in
    all. The waveform is scaled to its own peak and band-passed like the spectrogram (``HP``): it shows the
    call's shape in time, not its loudness."""
    items = {r["item_id"]: r for r in csv.DictReader(open(folder / "items.tsv"), delimiter="\t")}
    rows = (len(ids) + cols - 1) // cols
    head = 2.0
    fig = plt.figure(figsize=(5.4 * cols, 4.3 * rows + head))
    outer = fig.add_gridspec(rows, cols, hspace=0.36, wspace=0.14, left=0.075, right=0.99, top=1 - head / (4.3 * rows + head),
                             bottom=0.05)
    for k, i in enumerate(ids):
        x, sr = sf.read(folder / "clips" / f"{i}.flac", dtype="float64")
        a, b = json.loads(items[i]["marks"])[0][:2]
        dur = len(x) / sr
        g = outer[k // cols, k % cols].subgridspec(3, 1, height_ratios=[2.2, 6, 0.55], hspace=0.06)
        wav = fig.add_subplot(g[0])
        t = np.arange(len(x)) / sr
        w = signal.sosfiltfilt(HP, x)
        wav.plot(t, w / np.abs(w).max(), color="#33393b", lw=0.4)
        wav.set_xlim(0, dur)
        wav.set_ylim(-1.05, 1.05)
        wav.axis("off")
        wav.text(0.0, 1.02, "ABCDEFGHIJ"[k], transform=wav.transAxes, fontsize=17, fontweight="bold", va="bottom")
        f, tt, z = signal.stft(x, sr, window="hann", nperseg=1024, noverlap=1024 - sr // 1000)
        p = 20 * np.log10(np.abs(z) + 1e-12)
        keep = (f >= FMIN * 1000) & (f <= FMAX * 1000)
        p = p[keep] - p[keep].max()
        ax = fig.add_subplot(g[1])
        ax.imshow(p, origin="lower", aspect="auto", cmap="magma", vmin=-DB_RANGE, vmax=0,
                  extent=[0, dur, f[keep][0] / 1000, f[keep][-1] / 1000], interpolation="bilinear")
        ax.set_xlim(0, dur)
        ax.tick_params(bottom=False, labelbottom=False, labelsize=12)
        ax.set_yticks([5, 10, 15])
        if k % cols == 0:
            ax.set_ylabel("Frequency (kHz)", fontsize=13)
        for sp in ax.spines.values():
            sp.set_visible(False)
        bar = fig.add_subplot(g[2])
        bar.axvspan(a, b, color=BAR, lw=0)
        bar.set_xlim(0, dur)
        bar.set_ylim(0, 1)
        bar.set_yticks([])
        step = 0.5 if dur > 1.5 else 0.2
        bar.set_xticks(np.arange(0, dur + 1e-9, step))
        bar.tick_params(labelsize=12, length=3)
        for sp in ("top", "right", "left"):
            bar.spines[sp].set_visible(False)
        bar.set_xlabel("Time (s)", fontsize=13, labelpad=1)
        if k % cols == 0:
            bar.text(-0.02, 0.5, "detected", transform=bar.transAxes, ha="right", va="center", fontsize=12, color=BAR, fontweight="bold")
    H = fig.get_figheight()
    fig.text(0.5, 1 - 0.30 / H, TITLE, ha="center", va="top", fontsize=24, fontweight="bold")
    fig.text(0.5, 1 - 0.80 / H, SUBTITLE, ha="center", va="top", fontsize=18, style="italic")
    fig.text(0.5, 1 - 1.17 / H, STATS, ha="center", va="top", fontsize=15, color="#444")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(out.with_suffix(".png"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    draw(a.folder, a.ids or picked(a.folder), a.out)


if __name__ == "__main__":
    main()
