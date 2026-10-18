"""Draw a clip's spectrogram with its hand labels on top, for self-review.

Six rows of 10 s each, labels as coloured bars along the top edge (green = c,
yellow = ?, red = n) with thin boundary lines, after the same normalisation the
importer applies. Use it after labelling a clip to spot missed calls or labels that
don't sit on a call.

Usage:
    python tools/labels/review_labels.py labeling            # every labelled clip
    python tools/labels/review_labels.py labeling clip_06    # just one
Writes labeling/review/<clip>.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_audacity import normalise, read_audacity  # noqa: E402

COLOURS = {"c": "lime", "?": "yellow", "n": "red"}


def render(root: Path, cid: str) -> Path:
    x, sr = sf.read(root / "clips" / f"{cid}.wav", dtype="float32")
    labels, _, _ = normalise(read_audacity(root / "labels" / f"{cid}.txt"))
    rows = int(np.ceil(len(x) / sr / 10))
    fig, axes = plt.subplots(rows, 1, figsize=(16, 2.5 * rows))
    for r, ax in enumerate(np.atleast_1d(axes)):
        seg = x[r * 10 * sr:(r + 1) * 10 * sr]
        f, t, z = signal.stft(seg, fs=sr, nperseg=1024, noverlap=768)
        db = 20 * np.log10(np.abs(z) + 1e-12)
        vmax = np.percentile(db, 99.8)
        ax.pcolormesh(t + 10 * r, f / 1000, db, shading="auto", cmap="magma", vmin=vmax - 60, vmax=vmax)
        ax.set_ylim(0, 20); ax.set_xlim(10 * r, 10 * r + 10)
        ax.set_xticks(np.arange(10 * r, 10 * r + 10.1, 1)); ax.tick_params(labelsize=7)
        ax.set_ylabel("kHz", fontsize=7)
        for s, e, lab in labels:
            if e > 10 * r and s < 10 * r + 10:
                ax.axvspan(s, e, ymin=0.93, ymax=1.0, color=COLOURS[lab], alpha=0.9)
                ax.axvline(s, color="cyan", lw=0.6); ax.axvline(e, color="cyan", lw=0.6)
    fig.suptitle(f"{cid}: {len(labels)} labels (green c, yellow ?, red n)", fontsize=12)
    fig.tight_layout()
    out = root / "review" / f"{cid}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=80); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the labeling/ directory")
    ap.add_argument("clips", nargs="*", help="clip ids; default: every labelled clip")
    a = ap.parse_args()
    root = Path(a.root)
    ids = a.clips or sorted(p.stem for p in (root / "labels").glob("clip_*.txt"))
    for cid in ids:
        print(render(root, cid))


if __name__ == "__main__":
    main()
