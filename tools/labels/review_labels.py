"""Draw a clip's spectrogram with its hand labels on top, for self-review.

Six rows of 10 s each, labels as coloured bars along the top edge (green = c,
yellow = ?, red = n) with thin boundary lines, after the same normalisation the
importer applies. Use it after labelling a clip to spot missed calls or labels that
don't sit on a call.

With ``--scores`` (the JSON from tools/analysis/score_detectors.py) and one or two
``--show detector@param/strategy``, the chosen detectors' calls are drawn as bars along
the bottom edge (blue, then white), so misses and false alarms can be inspected.

Usage:
    python tools/labels/review_labels.py labeling            # every labelled clip
    python tools/labels/review_labels.py labeling clip_06    # just one
    python tools/labels/review_labels.py labeling clip_23 --scores outputs/detector_scores.json \
        --show talia@0.02/mean --show spectral@15/mean
Writes labeling/review/<clip>.png (or <clip>__det.png when detections are drawn).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_audacity import normalise, read_audacity  # noqa: E402

COLOURS = {"c": "lime", "?": "yellow", "n": "red"}
DET_COLOURS = ("deepskyblue", "white")


def render(root: Path, cid: str, dets: Optional[Dict[str, List[List[float]]]] = None) -> Path:
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
        for k, iv in enumerate((dets or {}).values()):
            lo = 0.07 * (1 - k)
            for s, e in iv:
                if e > 10 * r and s < 10 * r + 10:
                    ax.axvspan(s, e, ymin=lo, ymax=lo + 0.06, color=DET_COLOURS[k], alpha=0.9)
    legend = "".join(f", {DET_COLOURS[k]} = {n}" for k, n in enumerate(dets or {}))
    fig.suptitle(f"{cid}: {len(labels)} labels (green c, yellow ?, red n){legend}", fontsize=12)
    fig.tight_layout()
    out = root / "review" / (f"{cid}__det.png" if dets else f"{cid}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=80); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the labeling/ directory")
    ap.add_argument("clips", nargs="*", help="clip ids; default: every labelled clip")
    ap.add_argument("--scores", help="detector_scores.json from score_detectors.py")
    ap.add_argument("--show", action="append", default=[],
                    help="detector@param/strategy to draw (at most 2), e.g. spectral@15/mean")
    a = ap.parse_args()
    if a.show and not a.scores:
        ap.error("--show needs --scores")
    if len(a.show) > len(DET_COLOURS):
        ap.error(f"at most {len(DET_COLOURS)} --show")
    scores = json.load(open(a.scores))["clips"] if a.scores else {}
    root = Path(a.root)
    ids = a.clips or sorted(p.stem for p in (root / "labels").glob("clip_*.txt"))
    for cid in ids:
        dets = {s: scores[cid]["detections"][s.split("/")[0]][s.split("/")[1]] for s in a.show} or None
        print(render(root, cid, dets))


if __name__ == "__main__":
    main()
