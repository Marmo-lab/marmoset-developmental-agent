"""Donor calls: single, typed marmoset calls cut from a clip set, for planting into synthetic recordings.

The synthetic evaluation (docs/SYNTH_EVAL.md) plants known calls into call-free
background so the detector can be scored against an exact answer, per call type and SNR.
This script prepares the calls to plant (the "donors") and nothing else: it reads a clip
set, converts it to our rate, finds where the call sits in each clip, and writes the
result to the work directory for listening on the library's ``/donors`` page. The source
is read only (it lives under the lab's shared data).

Source [paper]. Turesson et al. 2016, PLoS ONE 11(9): e0163041, "Machine learning
algorithms for automatic classification of marmoset vocalizations": 11 call types, about
30 clips each, 44.1 kHz mono, one folder per type. The clips are padded with silence
(our scan: a 1.15 s tsik clip holds about 0.15 s of call), so the call bounds are ours.

Rate [ours]. 44.1 kHz to 48 kHz with ``resample_poly(160, 147)``: everything below
22 kHz is kept, which covers our detector's band (1-24 kHz, of which the calls use 2-20).

Bounds [ours]. An STFT (1024 points, 10 ms hop: the detector's frame) summed over
``F_LO_HZ``-``F_HI_HZ``, in dB and smoothed over 3 frames. The floor is the clip's 10th
percentile; a frame is active when it is ``margin`` above it, where ``margin`` is a third
of the clip's peak-over-floor, held between ``MIN_RISE_DB`` and ``MAX_RISE_DB`` (a quiet
clip must still have bounds, a loud one must not swallow its own reverberation). Runs
shorter than ``MIN_RUN`` frames are dropped (clicks); the call runs from the first to the
last active frame, so the gaps between the syllables of a twitter or a trill stay inside.
The bounds are a proposal: the page shows them, and the user's verdicts decide.

Writes ``OUT/audio/<donor_id>.flac`` (the whole clip at 48 kHz, so the page shows the
padding) and ``OUT/donors.tsv`` (donor_id, call_type, source, onset_s, offset_s,
duration_s, rise_db, peak_freq_hz).

Usage (light, but on a compute node: ``sbatch -p elscn.q --wrap "..."``):
    python tools/synth/donor_calls.py SRC OUT
    SRC: the folder holding one sub-folder of .wav clips per call type
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

SR = 48_000
HOP = SR // 100                                  # 10 ms, the detector's frame
F_LO_HZ, F_HI_HZ = 2_000, 20_000                 # call band, as call_map.py
MIN_RISE_DB, MAX_RISE_DB = 3.0, 12.0
MIN_RUN = 2                                      # frames


@dataclass
class Donor:
    donor_id: str
    call_type: str
    source: str
    onset_s: float
    offset_s: float
    duration_s: float
    rise_db: float
    peak_freq_hz: float


def to_48k(x: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 at 48 kHz (44.1 kHz by the exact 160/147 ratio, other rates by their gcd)."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g)
    return x.astype(np.float32)


def call_bounds(x: np.ndarray) -> Tuple[float, float, float, float]:
    """(onset_s, offset_s, rise_db, peak_freq_hz) of the call in a 48 kHz clip; see the module docstring."""
    f, _, z = signal.stft(x, SR, nperseg=1024, noverlap=1024 - HOP, boundary=None, padded=False)
    band = (f >= F_LO_HZ) & (f <= F_HI_HZ)
    power = np.abs(z[band]) ** 2
    e = np.convolve(10 * np.log10(power.sum(0) + 1e-12), np.ones(3) / 3, mode="same")
    floor = np.percentile(e, 10)
    rise = float(e.max() - floor)
    active = e > floor + min(MAX_RISE_DB, max(MIN_RISE_DB, rise / 3))
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= MIN_RUN]
    if not runs:
        runs = [(int(e.argmax()), int(e.argmax()) + 1)]
    a, b = runs[0][0], runs[-1][1]
    half = 512 / SR                              # frame k covers [k*HOP, k*HOP + 1024): report its centre
    onset, offset = a * HOP / SR + half, (b - 1) * HOP / SR + half
    peak = float(f[band][power[:, a:b].sum(1).argmax()])
    return onset, max(offset, onset + HOP / SR), rise, peak


def prepare(src: Path, out: Path) -> List[Donor]:
    (out / "audio").mkdir(parents=True, exist_ok=True)
    donors = []
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        for k, path in enumerate(sorted(folder.glob("*.wav"))):
            x, sr = sf.read(path, dtype="float32")
            x = to_48k(x, sr)
            onset, offset, rise, peak = call_bounds(x)
            d = Donor(f"{folder.name}_{k:03d}", folder.name, f"{folder.name}/{path.name}", round(onset, 3),
                      round(offset, 3), round(offset - onset, 3), round(rise, 1), round(peak))
            sf.write(out / "audio" / f"{d.donor_id}.flac", x, SR)
            donors.append(d)
    with open(out / "donors.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(donors[0])), delimiter="\t")
        w.writeheader()
        w.writerows(asdict(d) for d in donors)
    return donors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    donors = prepare(a.src, a.out)
    for t in sorted({d.call_type for d in donors}):
        ds = [d for d in donors if d.call_type == t]
        print(f"{t:12s} n={len(ds):3d} dur med {np.median([d.duration_s for d in ds]):.2f} s  "
              f"rise med {np.median([d.rise_db for d in ds]):.0f} dB", flush=True)


if __name__ == "__main__":
    main()
