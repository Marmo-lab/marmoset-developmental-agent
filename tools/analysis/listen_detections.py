"""Stitch every call a detector finds in fresh daytime audio into one WAV, for listening.

Picks ``--n-files`` daytime recordings from the corpus manifest (seeded; recordings that
supplied a hand-labelled clip are excluded, so this is audio the detector was never
scored on), reads ``--minutes`` from a random daytime offset in each, and runs our
spectral detector (the best configuration in docs/DETECTOR_EVAL.md: 15 dB, 48 kHz with a
16 kHz ceiling) on the 4-capsule mean. Talia's detector runs on the same audio, so each
call is flagged by whether hers would have found it.

Writes to ``--out``:
    calls.wav       every detection, in order, with 0.15 s of context on each side and
                    0.6 s of silence between calls. Each call is peak-normalised to -6 dBFS
                    so faint ones are audible: loudness in this file means nothing.
    calls.csv       one row per call: source file, wall-clock time, duration, position
                    in calls.wav, and whether Talia's detector also found it.
    calls.png       spectrogram of calls.wav in 10 s rows; a bar over each call, green if
                    Talia's detector found it too, orange if only ours did.

Usage (cluster; reads raw audio, writes only under --out):
    python tools/analysis/listen_detections.py $WORK_DIR/manifests/recordings.csv \
        --exclude $WORK_DIR/labeling/clips.csv --out $WORK_DIR/listen
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import soundfile as sf

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.preprocessing.filter_5khz import preprocess_waveform
from marmoset_developmental_agent.preprocessing.segmentation import activity_to_calls

OURS = dataclasses.replace(PreprocessingConfig(), frame_features="spectral", bin_energy_threshold_db=15.0,
                           detection_sample_rate=48_000, spectral_max_freq_hz=16_000.0)
TALIA = dataclasses.replace(PreprocessingConfig(), frame_features="talia")
CONTEXT_S, GAP_S, PEAK = 0.15, 0.6, 0.5
DAY_HOURS = (8.0, 18.0)


def _is_day(t: dt.datetime) -> bool:
    h = t.hour + t.minute / 60
    return DAY_HOURS[0] <= h < DAY_HOURS[1]


def pick_segments(manifest: str, exclude: Set[str], n: int, minutes: float,
                  seed: int) -> List[Tuple[str, dt.datetime, float]]:
    """``(path, wall-clock start of segment, offset_s)`` for ``n`` daytime segments."""
    rng = np.random.default_rng(seed)
    seg = minutes * 60
    rows = [r for r in csv.DictReader(open(manifest)) if r["path"] not in exclude]
    rng.shuffle(rows)
    out: List[Tuple[str, dt.datetime, float]] = []
    for r in rows:
        start, dur = dt.datetime.fromisoformat(r["start_time"]), float(r["duration_s"])
        offsets = [o for o in np.arange(0, dur - seg, 60.0)
                   if _is_day(start + dt.timedelta(seconds=float(o)))
                   and _is_day(start + dt.timedelta(seconds=float(o + seg)))]
        if offsets:
            o = float(rng.choice(offsets))
            out.append((r["path"], start + dt.timedelta(seconds=o), o))
        if len(out) == n:
            break
    return out


def _calls(x: np.ndarray, sr: int, cfg: PreprocessingConfig) -> List[Tuple[float, float]]:
    return [(c.start_s, c.end_s) for c in activity_to_calls(preprocess_waveform(x, sr, cfg)[1], cfg)]


def render_png(wav: np.ndarray, sr: int, spans: List[Tuple[float, float, bool]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    rows = max(1, int(np.ceil(len(wav) / sr / 10)))
    fig, axes = plt.subplots(rows, 1, figsize=(16, 2.2 * rows))
    for r, ax in enumerate(np.atleast_1d(axes)):
        seg = wav[r * 10 * sr:(r + 1) * 10 * sr]
        if len(seg) < 2048:
            ax.axis("off")
            continue
        f, t, z = signal.stft(seg, fs=sr, nperseg=1024, noverlap=768)
        db = 20 * np.log10(np.abs(z) + 1e-9)
        vmax = np.percentile(db, 99.8)
        ax.pcolormesh(t + 10 * r, f / 1000, db, shading="auto", cmap="magma", vmin=vmax - 60, vmax=vmax)
        ax.set_ylim(0, 20)
        ax.set_xlim(10 * r, 10 * r + 10)
        ax.tick_params(labelsize=7)
        for s, e, both in spans:
            if e > 10 * r and s < 10 * r + 10:
                ax.axvspan(s, e, ymin=0.93, ymax=1.0, color="lime" if both else "orange", alpha=0.9)
    fig.suptitle("detected calls (bar: green = Talia's detector also found it, orange = only ours)")
    fig.tight_layout()
    fig.savefig(out, dpi=70)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--exclude", help="clips.csv whose source recordings to skip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-files", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20261018)
    a = ap.parse_args()

    exclude = {r["source_path"] for r in csv.DictReader(open(a.exclude))} if a.exclude else set()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pieces: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    spans: List[Tuple[float, float, bool]] = []
    pos = 0.0
    for path, t0, offset in pick_segments(a.manifest, exclude, a.n_files, a.minutes, a.seed):
        sr = sf.info(path).samplerate
        with sf.SoundFile(path) as f:
            f.seek(int(offset * sr))
            x = f.read(int(a.minutes * 60 * sr), dtype="float32", always_2d=True).mean(axis=1)
        ours, hers = _calls(x, sr, OURS), _calls(x, sr, TALIA)
        print(f"{Path(path).name} @ {t0:%Y-%m-%d %H:%M}: ours {len(ours)} calls, Talia's {len(hers)}")
        for s, e in ours:
            i0, i1 = int(max(0.0, s - CONTEXT_S) * sr), int(min(len(x) / sr, e + CONTEXT_S) * sr)
            piece = x[i0:i1] - x[i0:i1].mean()
            piece = piece * (PEAK / (np.abs(piece).max() + 1e-12))
            fade = min(len(piece) // 2, int(0.01 * sr))
            ramp = np.linspace(0, 1, fade)
            piece[:fade] *= ramp
            piece[len(piece) - fade:] *= ramp[::-1]
            both = any(min(e, he) > max(s, hs) for hs, he in hers)
            dur = len(piece) / sr
            rows.append({"source": Path(path).name, "wall_time": (t0 + dt.timedelta(seconds=s)).isoformat(),
                         "duration_s": f"{e - s:.2f}", "wav_start_s": f"{pos:.2f}",
                         "wav_end_s": f"{pos + dur:.2f}", "talia_found": int(both)})
            spans.append((pos + CONTEXT_S, pos + dur - CONTEXT_S, both))
            pieces += [piece, np.zeros(int(GAP_S * sr), np.float32)]
            pos += dur + GAP_S
    if not pieces:
        raise SystemExit("no calls found")
    wav = np.concatenate(pieces).astype(np.float32)
    sf.write(out / "calls.wav", wav, sr, subtype="PCM_16")
    with open(out / "calls.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    render_png(wav, sr, spans, out / "calls.png")
    n_both = sum(int(r["talia_found"]) for r in rows)
    print(f"{len(rows)} calls, {len(wav) / sr / 60:.1f} min of audio; Talia's detector found {n_both} of them "
          f"-> {out}/calls.wav")


if __name__ == "__main__":
    main()
