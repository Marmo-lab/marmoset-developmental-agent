"""Is the learned detector blind to high frequency, or to the short high calls themselves?

Each donor (``tools/synth/donor_calls.py``, raw) is planted alone in the middle of 20 s of
approved night background at +20 and +6 dB (amplitude ratio 10 and 2, the SNR definition of
``tools/synth/generate.py``), once as it is and once shifted 5 kHz by single-sideband
modulation (the analytic signal times ``exp(2 pi i df t)``, after a 3-19 kHz band-pass so
the donor's low rumble is not shifted into the band). Short high types (tsik, tsik-ek, seep)
are shifted down into the usual call band; every other type is shifted up to about 13 kHz.
The production detector's frame probabilities (``detection_activity``) are read inside the
call: ``max_p`` >= ``learned_threshold`` counts as found.

Reading [ours]: if shifting the short high calls down rescues them, the blind spot is
frequency; if shifting normal calls up loses them while the short high calls stay missed
when shifted down, it is both frequency and the calls' short sweeping shape. Result
(2026-11-05, docs/SYNTH_EVAL.md): the second.

Writes one row per planted call: type, version, snr, max_p, mean_p, peak_hz, dur_s.

Usage (local or on a compute node; reads raw audio only):
    python tools/analysis/synth_shift_probe.py DONORS BACKGROUND_AUDIO_DIR OUT.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

import tools.synth.generate as g
from marmoset_developmental_agent.config.agent_config import load_config
from marmoset_developmental_agent.preprocessing.streaming import detection_activity

SR = g.SR
SHIFT_HZ = 5_000
SHORT_HIGH = ("tsik", "tsik_ek", "seep")
LEVELS = (10.0, 2.0)
WINDOW_S = 20
BG_MINUTES = 40
_PRE = signal.butter(6, (3_000, 19_000), btype="bandpass", fs=SR, output="sos")


def shift(x: np.ndarray, hz: float) -> np.ndarray:
    """Single-sideband frequency shift by ``hz`` (after the 3-19 kHz pre-filter)."""
    a = signal.hilbert(signal.sosfiltfilt(_PRE, x))
    return np.real(a * np.exp(2j * np.pi * hz * np.arange(len(x)) / SR))


def peak_hz(x: np.ndarray) -> float:
    """Frequency of the largest mean STFT power within 5-20 kHz."""
    x = np.pad(x, (0, max(0, 1024 - len(x))))
    f, _, z = signal.stft(x, SR, nperseg=1024, noverlap=1024 - 480)
    p = (np.abs(z) ** 2).mean(axis=1)
    sel = (f >= 5_000) & (f <= 20_000)
    return float(f[sel][np.argmax(p[sel])])


def probe(donors_dir: Path, bg_dir: Path, seed: int = 0) -> List[Tuple]:
    cfg = load_config().preprocessing
    donors, audio = g.load_donors(donors_dir)
    bgs = [sf.read(p, dtype="float32", frames=BG_MINUTES * 60 * SR)[0] for p in sorted(bg_dir.glob("*.flac"))]
    rng = np.random.default_rng(seed)
    w = WINDOW_S * SR
    out = []
    for t, ds in sorted(donors.items()):
        hz = -SHIFT_HZ if t in SHORT_HIGH else SHIFT_HZ
        for dn in ds:
            s0 = audio[dn["donor_id"]].astype(np.float64)
            for vname, s in (("as is", g.faded(s0)), (f"shift {hz / 1000:+g} kHz", g.faded(shift(s0, hz)))):
                for snr in LEVELS:
                    bg = bgs[rng.integers(len(bgs))]
                    st = rng.integers(0, len(bg) - w)
                    b = bg[st:st + w].astype(np.float64)
                    i = w // 2 - len(s) // 2
                    y = b.copy()
                    y[i:i + len(s)] += snr * g.rms(g.bandpass(b)[i:i + len(s)]) / g.rms(g.bandpass(s)) * s
                    act = detection_activity(y.astype(np.float32), SR, cfg)
                    inside = (act.frame_times_s >= i / SR) & (act.frame_times_s <= (i + len(s)) / SR)
                    p = act.call_prob[inside] if inside.any() else np.zeros(1)
                    out.append((t, vname, snr, round(float(p.max()), 4), round(float(p.mean()), 4),
                                peak_hz(s), round(len(s) / SR, 4)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("donors", type=Path)
    ap.add_argument("backgrounds", type=Path, help="folder of mono background FLACs (generate.py extract)")
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    rows = probe(a.donors, a.backgrounds)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["type", "version", "snr", "max_p", "mean_p", "peak_hz", "dur_s"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
