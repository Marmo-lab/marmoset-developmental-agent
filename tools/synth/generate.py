"""Synthetic recordings: donor calls planted into approved background at a set SNR, with an exact answer.

Step 4 of the synthetic evaluation; the full method, with every number and its status, is
docs/SYNTH_METHODS.md section 5, and this docstring only maps it onto the code.

``extract`` writes each approved background as a mono 48 kHz FLAC (the 4-capsule mean, what
the detector hears; ``backgrounds.py``), so generation never rereads the 4-channel hours.
``make`` then builds ``--n`` one-minute recordings per SNR level:

* **Background** [ours]: a 60 s excerpt of one approved background at a uniform random
  offset. A background's usable spans (``spans``; a whole night recording is one span, the
  day background will be many short ones) are joined with ``CROSSFADE_S`` raised-cosine
  crossfades when one span is too short.
* **Calls** [ours]: their number is Poisson with mean ``CALLS_PER_MIN`` (the v4 day mean,
  below). Each call draws a type uniformly over the donor types, then a donor uniformly
  within the type. A draw whose calls cannot fit (with ``GAP_S`` between calls and
  ``EDGE_S`` at the edges) is redrawn. The free time is split at uniform random points
  (sorted uniforms), so every non-overlapping arrangement is equally likely.
* **SNR** [ours, with the user]: one level per recording. Per call, the gain ``g`` makes
  ``RMS(bp(g s)) / RMS(bp(b))`` equal the level, where ``bp`` is a 4th-order Butterworth
  band-pass over ``BAND_HZ`` applied forward and backward (zero phase), ``s`` the faded
  donor and ``b`` the background under it. The call is added: ``y = b + g s``.
* **Fades** [ours]: ``FADE_S`` raised-cosine ramps at both ends of each donor.
* **Donors are planted raw** [ours, with the user]: the clip between its bounds, its own
  recording noise included. The corpus's spectral cleaning (``preprocessing/clean_calls.py``,
  noise from the clip's padding) removed that noise but, by ear (2026-11-05), made the calls
  sound unnatural; the user prefers a little noise. It stays as ``--clean-donors``. Known cost:
  at high SNR a quiet-in-band donor (alarm, twitter) carries a faint broadband box of its own
  noise.

``CALLS_PER_MIN`` [ours]: the mean over the hours 07:00-17:59 of the per-hour means over
138 dates of the v4 run (``tools/analysis/calls_per_hour.py``; ``figures/``), 1,271 calls
per recorded hour. It counts every detection, quality flags included.

Writes under OUT: ``audio/<rec_id>.flac`` (mono 48 kHz, 24-bit), ``recordings.tsv`` (one row
per recording: SNR, background and offset, number of calls, seed) and ``events.tsv`` (one row
per planted call: the answer, plus the gain).

Usage (on the cluster, via tools/cluster/synth.sbatch):
    python tools/synth/generate.py extract BACKGROUNDS          # approved candidates -> BACKGROUNDS/audio/
    python tools/synth/generate.py make DONORS BACKGROUNDS OUT [--n 300] [--snr 10,3.2,2,1.5,1,0.7]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

SR = 48_000
LENGTH_S = 60.0
CALLS_PER_MIN = 1271 / 60
GAP_S = 0.5            # closer calls are a segmentation question; the hand labels measure that
EDGE_S = 0.5
FADE_S = 0.005
CROSSFADE_S = 0.05
BAND_HZ = (5_000, 20_000)
SNR_LEVELS = (10.0, 3.2, 2.0, 1.5, 1.0, 0.7)       # amplitude ratios: +20, +10, +6, +3.5, 0, -3 dB
_BP = signal.butter(4, BAND_HZ, btype="bandpass", fs=SR, output="sos")


@dataclass
class Event:
    rec_id: str
    kind: str
    donor_id: str
    call_type: str
    onset_s: float
    offset_s: float
    snr: float
    gain: float


class Background(Protocol):
    """A mono 48 kHz background read in pieces (an hour is ~700 MB as float32)."""

    frames: int

    def read(self, start: int, n: int) -> np.ndarray: ...


class FileBackground:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.frames = sf.info(path).frames

    def read(self, start: int, n: int) -> np.ndarray:
        return sf.read(self.path, start=start, frames=n, dtype="float64")[0]


class ArrayBackground:
    def __init__(self, x: np.ndarray) -> None:
        self.x, self.frames = x, len(x)

    def read(self, start: int, n: int) -> np.ndarray:
        return self.x[start:start + n].astype(np.float64)


def bandpass(x: np.ndarray) -> np.ndarray:
    return signal.sosfiltfilt(_BP, x)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-30))


def faded(x: np.ndarray) -> np.ndarray:
    n = min(int(FADE_S * SR), len(x) // 2)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    y = x.astype(np.float64).copy()
    y[:n] *= ramp
    y[len(y) - n:] *= ramp[::-1]
    return y


def excerpt(audio: Background, spans: Sequence[Tuple[int, int]], rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    """``LENGTH_S`` of background and where it starts: from one span if any is long enough,
    otherwise random spans joined with crossfades (the start is then that of the first)."""
    n = int(LENGTH_S * SR)
    long = [(a, b) for a, b in spans if b - a >= n]
    if long:
        a, b = long[rng.integers(len(long))]
        start = int(rng.integers(a, b - n + 1))
        return audio.read(start, n), start
    xf = int(CROSSFADE_S * SR)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, xf))
    out = np.zeros(0)
    first = -1
    while len(out) < n:
        a, b = spans[rng.integers(len(spans))]
        piece = audio.read(a, min(b - a, n))
        first = a if first < 0 else first
        if len(out) and len(piece) > xf:
            out[-xf:] = out[-xf:] * ramp[::-1] + piece[:xf] * ramp
            piece = piece[xf:]
        out = np.concatenate([out, piece])
    return out[:n], first


def place(durations: Sequence[float], rng: np.random.Generator) -> List[float] | None:
    """Onsets (s) for calls in the given order, uniform over the non-overlapping arrangements; None if they cannot fit."""
    free = LENGTH_S - 2 * EDGE_S - sum(durations) - GAP_S * (len(durations) - 1)
    if free < 0:
        return None
    cuts = np.sort(rng.uniform(0, free, len(durations)))
    onsets, t = [], EDGE_S
    for k, d in enumerate(durations):
        t_k = t + cuts[k] - (cuts[k - 1] if k else 0)
        onsets.append(float(t_k))
        t = t_k + d + GAP_S
    return onsets


def make_recording(rec_id: str, snr: float, donors: Dict[str, List[Dict]], donor_audio: Dict[str, np.ndarray],
                   background: Background, spans: Sequence[Tuple[int, int]], rng: np.random.Generator
                   ) -> Tuple[np.ndarray, List[Event], int]:
    types = sorted(donors)
    while True:
        picks = [donors[t][rng.integers(len(donors[t]))] for t in rng.choice(types, rng.poisson(CALLS_PER_MIN * LENGTH_S / 60))]
        onsets = place([len(donor_audio[d["donor_id"]]) / SR for d in picks], rng)
        if onsets is not None:
            break
    b, start = excerpt(background, spans, rng)
    b_bp = bandpass(b)
    y = b.copy()
    events = []
    for d, on in zip(picks, onsets):
        s = faded(donor_audio[d["donor_id"]])
        i = int(round(on * SR))
        g = snr * rms(b_bp[i:i + len(s)]) / rms(bandpass(s))
        y[i:i + len(s)] += g * s
        events.append(Event(rec_id, "call", d["donor_id"], d["call_type"], round(i / SR, 4),
                            round((i + len(s)) / SR, 4), snr, round(g, 6)))
    return y, events, start


def load_donors(folder: Path, clean: bool = False) -> Tuple[Dict[str, List[Dict]], Dict[str, np.ndarray]]:
    """Donors by type, and each donor's audio between its bounds; raw unless ``clean`` (see above)."""
    from marmoset_developmental_agent.config.agent_config import CleaningConfig
    from marmoset_developmental_agent.preprocessing.clean_calls import clean_call

    cfg = CleaningConfig()
    by_type: Dict[str, List[Dict]] = {}
    audio = {}
    for r in csv.DictReader(open(folder / "donors.tsv"), delimiter="\t"):
        x, sr = sf.read(folder / "audio" / f"{r['donor_id']}.flac", dtype="float32")
        assert sr == SR, (r["donor_id"], sr)
        call = (float(r["onset_s"]), float(r["offset_s"]))
        if clean:
            x = clean_call(x, SR, call, [call], cfg)[0]
        audio[r["donor_id"]] = x[int(call[0] * SR):int(call[1] * SR)]
        by_type.setdefault(r["call_type"], []).append(r)
    return by_type, audio


def approved(backgrounds: Path) -> List[Dict]:
    """Candidates the user approved on the ``/backgrounds`` page (the last verdict per recording wins)."""
    cands = {c["recording_id"]: c for c in json.loads((backgrounds / "candidates.json").read_text())["candidates"]}
    verdicts: Dict[str, str] = {}
    path = backgrounds / "bg_feedback.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        d = json.loads(line)
        verdicts[d["recording_id"]] = d.get("verdict")
    return [cands[r] for r, v in sorted(verdicts.items()) if v == "approve"]


def extract(backgrounds: Path) -> None:
    (backgrounds / "audio").mkdir(exist_ok=True)
    for c in approved(backgrounds):
        dest = backgrounds / "audio" / f"{c['recording_id']}.flac"
        if dest.exists():
            continue
        with sf.SoundFile(c["path"]) as f:
            assert f.samplerate == SR, (c["path"], f.samplerate)
            with sf.SoundFile(dest.with_suffix(".tmp.flac"), "w", SR, 1, subtype="PCM_24") as out:
                for block in f.blocks(blocksize=60 * SR, dtype="float32", always_2d=True):
                    out.write(block.mean(axis=1))
        dest.with_suffix(".tmp.flac").rename(dest)
        print(f"{c['recording_id']}: extracted", flush=True)


def make(donors_dir: Path, backgrounds: Path, out: Path, n: int, levels: Sequence[float], seed: int,
         clean: bool = False) -> None:
    donors, donor_audio = load_donors(donors_dir, clean)
    bgs = sorted((backgrounds / "audio").glob("*.flac"))
    assert bgs, f"no extracted background in {backgrounds / 'audio'}"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    recs, events = [], []
    for li, snr in enumerate(levels):
        for k in range(n):
            rec_id = f"snr{snr:g}_{k:04d}"
            rng = np.random.default_rng([seed, li, k])
            bg_path = bgs[rng.integers(len(bgs))]
            bg = FileBackground(bg_path)
            y, ev, start = make_recording(rec_id, snr, donors, donor_audio, bg, [(0, bg.frames)], rng)
            peak = float(np.abs(y).max())
            scale = 0.99 / peak if peak > 0.99 else 1.0      # a uniform scale leaves every SNR as it is
            sf.write(out / "audio" / f"{rec_id}.flac", (y * scale).astype(np.float32), SR, subtype="PCM_24")
            recs.append({"rec_id": rec_id, "snr": snr, "snr_db": round(20 * np.log10(snr), 2), "background": bg_path.stem,
                         "background_kind": "night", "background_start_s": round(start / SR, 3), "n_calls": len(ev),
                         "scale": round(scale, 6), "seed": f"{seed},{li},{k}"})
            events += [asdict(e) | {"gain": round(e.gain * scale, 6)} for e in ev]
        print(f"SNR {snr:g}: {n} recordings", flush=True)
    for name, rows in (("recordings.tsv", recs), ("events.tsv", events)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "make"])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--snr", default=",".join(f"{s:g}" for s in SNR_LEVELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean-donors", action="store_true", help="spectrally clean the donors (off by default)")
    a = ap.parse_args()
    if a.step == "extract":
        extract(a.paths[0])
    else:
        make(*a.paths, n=a.n, levels=[float(s) for s in a.snr.split(",")], seed=a.seed, clean=a.clean_donors)


if __name__ == "__main__":
    main()
