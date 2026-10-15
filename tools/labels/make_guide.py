"""Render the visual labelling guide: what a marmoset call looks like vs. noise.

The panels are chosen *by eye*, not by any detector. A first version took its "call"
examples from Talia's detections and its "noise" examples from the candidates her
"abrupt up" rule rejects -- but on inspection one of those rejected "clicks" (41.0 s)
holds a clear rising call, another (29.0 s) is a tonal harmonic stack, and her detected
boundaries cover only fragments of longer calling. A guide built that way teaches the
labeller the detector's mistakes, which is the opposite of what labels are for.

No boundaries are drawn: the labeller works blind to detector output (LABELING.md).

Rows: clear calls / not calls / tricky cases, each panel titled with what to do.
Sources are 210503_001.WAV (beam-steered, as the clips are) and the night clips of the
labelling set (already beam-steered and normalised).

Usage (on the cluster):
    python tools/labels/make_guide.py <210503_001.wav> <labeling-dir> -o guide.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

from marmoset_developmental_agent.io.channels import ambisonic_beams, band_energy  # noqa: E402

# (row, source, start_s, end_s, title). source "raw" = 210503_001, else a clip file name.
PANELS: List[Tuple[int, str, float, float, str]] = [
    (0, "raw", 59.9, 63.1, "CALLS: bands + sweeps 5-12 kHz\n(several animals overlap: label each group)"),
    (0, "raw", 41.4, 44.5, "CALLS: steady band ~7 kHz with\nupsweeps; faint copy near 12 kHz"),
    (0, "raw", 68.8, 71.3, "CALLS: short rising sweeps,\neach one a label (unless gap < 0.1 s)"),
    (0, "raw", 356.4, 359.3, "CALL: wavy band ~7 kHz (trill)\nlabel the whole wavy stretch"),
    (1, "raw", 19.9, 22.3, "NOT A CALL: clicks = thin vertical\nlines covering all frequencies"),
    (1, "clip_01.wav", 10.0, 13.0, "NOT A CALL: night background\n(nothing to label here)"),
    (1, "clip_05.wav", 30.0, 33.0, "NOT A CALL: night background\n(empty label file is a valid answer)"),
    (1, "raw", 38.3, 40.3, "LEFT: faint call (label it)\nRIGHT: loud broadband burst (not a call)"),
    (2, "raw", 39.9, 42.4, "TRICKY: a real sweep among clicks\n-> label the sweep `c`, ignore the clicks"),
    (2, "raw", 28.0, 30.4, "TRICKY: evenly spaced flat lines\n(~3/6/8 kHz), unusual -> label `?`"),
    (2, "raw", 26.1, 29.6, "TRICKY: dense clicks hiding faint tones\n-> label only bands you can see, else `?`"),
    (2, "raw", 62.5, 65.6, "TRICKY: two animals overlapping\n-> one label covering both"),
]


def load(src: str, start: float, end: float, raw: str, clips: Path) -> Tuple[np.ndarray, int]:
    if src == "raw":
        sr = sf.info(raw).samplerate
        with sf.SoundFile(raw) as f:
            f.seek(int(start * sr))
            block = f.read(int((end - start) * sr), dtype="float32", always_2d=True)
        beams = ambisonic_beams(block.T)
        return beams[int(np.argmax(band_energy(beams, sr)))], sr
    x, sr = sf.read(clips / "clips" / src, start=None, dtype="float32")
    return x[int(start * sr): int(end * sr)], sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw_wav")
    ap.add_argument("labeling_dir")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    col = {0: 0, 1: 0, 2: 0}
    for row, src, s, e, title in PANELS:
        ax = axes[row][col[row]]
        col[row] += 1
        x, sr = load(src, s, e, a.raw_wav, Path(a.labeling_dir))
        fr, t, z = signal.stft(x, fs=sr, nperseg=1024, noverlap=768)
        db = 20 * np.log10(np.abs(z) + 1e-12)
        vmax = np.percentile(db, 99.8)
        ax.pcolormesh(t, fr / 1000, db, shading="auto", cmap="magma", vmin=vmax - 60, vmax=vmax)
        ax.set_ylim(0, 24)
        ax.set_title(title, fontsize=9, color={0: "darkgreen", 1: "darkred", 2: "darkorange"}[row])
        ax.set_xlabel("seconds", fontsize=8); ax.set_ylabel("kHz", fontsize=8)
        ax.tick_params(labelsize=7)
    for row, label in enumerate(["CALLS -> label `c`", "NOT CALLS -> don't label (optional `n`)",
                                 "TRICKY CASES"]):
        axes[row][0].annotate(label, xy=(-0.32, 0.5), xycoords="axes fraction", rotation=90,
                              va="center", ha="center", fontsize=12, fontweight="bold")
    fig.suptitle("What marmoset calls look like in the spectrogram (same view as Audacity, 0-24 kHz)",
                 fontsize=13)
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    fig.savefig(a.out, dpi=105)
    print(f"{len(PANELS)} panels -> {a.out}")


if __name__ == "__main__":
    main()
