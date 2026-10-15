"""Cut a stratified set of one-minute clips for hand labelling (OPEN_DECISIONS C9).

Clips are drawn from the corpus manifest (tools/build_manifest.py) with a fixed seed:
mostly daytime (where nearly all calls are) plus some night clips (to measure false
positives on near-silent audio), each from a different recording date where possible,
at a random offset that avoids the first and last minute of the source file.

Each clip is written as 48 kHz mono 16-bit WAV, steered with ``ambisonic_beam`` (one
beam chosen for the whole clip, so the audio is continuous) and peak-normalised to
-3 dBFS so quiet recordings are visible in a spectrogram. Normalisation changes level
only, never timing, and labels are times -- so it does not affect anything we measure.

A sidecar ``clips.csv`` maps every clip back to its source file, offset, wall-clock time
and applied gain, so labels can be converted to absolute sample indices in the raw
recordings.

Usage (on the cluster):
    python tools/labels/make_clips.py $WORK_DIR/manifests/recordings.csv \
        -o $WORK_DIR/labeling --day 24 --night 6
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from marmoset_developmental_agent.io.channels import ambisonic_beams, band_energy

CLIP_S = 60.0
MARGIN_S = 60.0


def is_day(t: dt.datetime) -> bool:
    return 6 <= t.hour < 19


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--day", type=int, default=24)
    ap.add_argument("--night", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261015)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest)))
    rng = random.Random(a.seed)
    picked = []
    for want, day in ((a.day, True), (a.night, False)):
        pool = []
        for r in rows:
            t0 = dt.datetime.fromisoformat(r["start_time"])
            dur = float(r["duration_s"])
            if dur < CLIP_S + 2 * MARGIN_S:
                continue
            off = rng.uniform(MARGIN_S, dur - MARGIN_S - CLIP_S)
            t = t0 + dt.timedelta(seconds=off)
            if is_day(t) == day:
                pool.append((r, off, t))
        rng.shuffle(pool)
        used_dates: set = set()
        for r, off, t in pool:  # one clip per date first, then allow repeats
            if t.date() not in used_dates:
                picked.append((r, off, t)); used_dates.add(t.date())
            if sum(1 for p in picked if is_day(p[2]) == day) == want:
                break

    out = Path(a.out); (out / "clips").mkdir(parents=True, exist_ok=True)
    picked.sort(key=lambda p: p[2])
    with open(out / "clips.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "file", "source_path", "offset_s", "duration_s",
                    "wall_time", "day", "beam_index", "gain"])
        for i, (r, off, t) in enumerate(picked, 1):
            sr = int(r["sample_rate"])
            with sf.SoundFile(r["path"]) as fh:
                fh.seek(int(off * sr))
                block = fh.read(int(CLIP_S * sr), dtype="float32", always_2d=True)
            beams = ambisonic_beams(block.T)
            b = int(np.argmax(band_energy(beams, sr)))
            x = beams[b]
            gain = float(10 ** (-3 / 20) / (np.abs(x).max() + 1e-12))
            name = f"clip_{i:02d}.wav"
            sf.write(out / "clips" / name, (x * gain).astype(np.float32), sr, subtype="PCM_16")
            w.writerow([f"clip_{i:02d}", name, r["path"], f"{off:.3f}", CLIP_S,
                        t.isoformat(timespec="seconds"), int(is_day(t)), b, f"{gain:.3f}"])
            print(f"clip_{i:02d}  {t:%Y-%m-%d %H:%M}  {'day' if is_day(t) else 'night'}  "
                  f"{Path(r['path']).parent.name}/{Path(r['path']).name} +{off:.0f}s  gain x{gain:.0f}")
    print(f"\n{len(picked)} clips -> {out}/clips, index {out}/clips.csv")


if __name__ == "__main__":
    main()
