"""Measure where Talia's absolute |STFT| threshold falls in a real recording.

Her detector keeps a time-frequency bin when ``|STFT| > AMP_THRESHOLD`` with
AMP_THRESHOLD = 0.02 (``segment_processor.py``), an *absolute* magnitude on audio that
``librosa.load`` never level-normalises. Her own diagnostic plots for 210503_001 show
``freq_range`` flat at exactly 0 for whole 45 s windows despite visible spectral energy,
and all 23 of her detections for that 62-minute file fall in its first 358 s -- which is
what you would see if the threshold is simply never crossed after that point.

This script tests that directly, and in the same pass answers the channel question:
it reports the bin-survival rate for the 4-capsule mean (what her pipeline computed)
against each individual capsule, so we can see whether the ambisonic mixdown is itself
costing us detections.

Reads in windows rather than whole-file: one 62-minute 4-channel file is ~2.9 GB as
float32 and must never be loaded at once.

Usage:
    python tools/analysis/level_audit.py <file.wav> [--windows 40] [--window-s 15]
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

AMP_THRESHOLD = 0.02       # [talia] segment_processor.py
DETECT_SR = 22_050         # [talia] librosa.load default
HP_CUTOFF, HP_ORDER = 5_000.0, 5
N_FFT, HOP = 2048, 512


def _prepare(x: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample to the detection rate and high-pass, exactly as her pipeline does."""
    if src_sr != DETECT_SR:
        g = np.gcd(src_sr, DETECT_SR)
        x = signal.resample_poly(x, DETECT_SR // g, src_sr // g)
    sos = signal.butter(HP_ORDER, HP_CUTOFF / (DETECT_SR / 2), btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, x.astype(np.float64))


def _mag(x: np.ndarray) -> np.ndarray:
    """Magnitude STFT with *librosa's* scaling, which is what her threshold assumes.

    This matters enormously and is easy to get wrong: ``scipy.signal.stft`` divides by
    the window sum (1024 for a 2048-point Hann) while ``librosa.stft`` does not. Using
    scipy here makes every magnitude 1024x smaller and AMP_THRESHOLD looks unreachable
    -- a false result. Her code calls ``librosa.stft``, so we must too.
    """
    return np.abs(librosa.stft(x.astype(np.float32), n_fft=N_FFT, hop_length=HOP))


def analyse(path: str, n_windows: int, window_s: float) -> Dict:
    info = sf.info(path)
    sr, n_ch, total = info.samplerate, info.channels, info.frames
    win = int(window_s * sr)
    starts = np.linspace(0, max(0, total - win), n_windows).astype(np.int64)

    rows: List[Dict] = []
    all_mag_mean: List[np.ndarray] = []
    for s0 in starts:
        block, _ = sf.read(path, start=int(s0), frames=win, dtype="float32", always_2d=True)
        row: Dict[str, float] = {"t_s": float(s0 / sr), "peak_raw": float(np.abs(block).max())}

        mix = _prepare(block.mean(axis=1), sr)          # her 4-capsule average
        m = _mag(mix)
        all_mag_mean.append(np.percentile(m, [50, 90, 99, 99.9, 100]))
        row["mean_mix_frac_above"] = float((m > AMP_THRESHOLD).mean())
        row["mean_mix_max_mag"] = float(m.max())
        row["mean_mix_frames_any"] = float((m > AMP_THRESHOLD).any(axis=0).mean())

        best = 0.0
        for c in range(n_ch):
            mc = _mag(_prepare(block[:, c], sr))
            frac = float((mc > AMP_THRESHOLD).mean())
            row[f"ch{c}_frac_above"] = frac
            row[f"ch{c}_max_mag"] = float(mc.max())
            best = max(best, frac)
        row["best_channel_frac_above"] = best
        rows.append(row)

    pct = np.mean(np.stack(all_mag_mean), axis=0)
    return {
        "file": path,
        "sample_rate": sr,
        "channels": n_ch,
        "duration_s": total / sr,
        "amp_threshold": AMP_THRESHOLD,
        "mag_percentiles_mean_mix": {
            "p50": float(pct[0]), "p90": float(pct[1]), "p99": float(pct[2]),
            "p99.9": float(pct[3]), "max": float(pct[4]),
        },
        "windows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--window-s", type=float, default=15.0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    r = analyse(a.path, a.windows, a.window_s)
    p = r["mag_percentiles_mean_mix"]
    print(f"{r['file']}\n  {r['sample_rate']} Hz, {r['channels']} ch, {r['duration_s']:.0f} s")
    print(f"\n|STFT| percentiles of the 4-capsule mean, librosa scaling "
          f"(threshold = {AMP_THRESHOLD}):")
    for k, v in p.items():
        print(f"    {k:>6}: {v:.5f}   {'ABOVE' if v > AMP_THRESHOLD else 'below'} threshold")

    print(f"\n{'t (s)':>8} {'peak_raw':>9} {'mean-mix':>9} {'best-ch':>9} {'gain':>6} {'frames':>7}")
    for w in r["windows"]:
        g = w["best_channel_frac_above"] / w["mean_mix_frac_above"] if w["mean_mix_frac_above"] > 0 else float("inf")
        print(f"{w['t_s']:8.0f} {w['peak_raw']:9.4f} {w['mean_mix_frac_above']:9.5f} "
              f"{w['best_channel_frac_above']:9.5f} {g:6.2f}x {w['mean_mix_frames_any']:7.3f}")

    mm = np.array([w["mean_mix_frac_above"] for w in r["windows"]])
    bc = np.array([w["best_channel_frac_above"] for w in r["windows"]])
    print(f"\nwindows where the 4-ch mean yields ZERO surviving bins: {int((mm == 0).sum())}/{len(mm)}")
    print(f"windows where the best single channel yields zero:       {int((bc == 0).sum())}/{len(bc)}")
    if mm.sum() > 0:
        print(f"mean bin-survival, best channel vs 4-ch mean:            {bc.sum() / mm.sum():.2f}x")

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(r, f, indent=1)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
