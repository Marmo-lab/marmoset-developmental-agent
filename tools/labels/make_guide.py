"""Render the visual labelling guide: what a marmoset call looks like vs. noise.

Uses 210503_001.WAV, where our detector port reproduces Talia's 23 detections exactly.
Top rows: some of those detections (her detector's calls, with +-1 s of context).
Bottom row: the candidates her "abrupt up" rule rejects -- short broadband clicks --
found by re-running the port with that rule disabled. Both are real audio from the
colony, drawn at full bandwidth (0-24 kHz) from the beam-steered signal a labeller
will actually be looking at.

Usage (on the cluster):
    python tools/labels/make_guide.py <raw.wav> -o guide.png
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from verify_talia_port import detect  # noqa: E402

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig  # noqa: E402
from marmoset_developmental_agent.io.channels import ambisonic_beams, band_energy  # noqa: E402


def panel(ax, path: str, start: int, end: int, sr: int, title: str, ctx_s: float = 1.0) -> None:
    a = max(0, start - int(ctx_s * sr)); b = end + int(ctx_s * sr)
    with sf.SoundFile(path) as f:
        f.seek(a)
        block = f.read(b - a, dtype="float32", always_2d=True)
    beams = ambisonic_beams(block.T)
    x = beams[int(np.argmax(band_energy(beams, sr)))]
    fr, t, z = signal.stft(x, fs=sr, nperseg=1024, noverlap=768)
    db = 20 * np.log10(np.abs(z) + 1e-12)
    vmax = np.percentile(db, 99.8)
    ax.pcolormesh(t, fr / 1000, db, shading="auto", cmap="magma", vmin=vmax - 60, vmax=vmax)
    ax.axvline((start - a) / sr, color="cyan", lw=1, ls="--")
    ax.axvline((end - a) / sr, color="cyan", lw=1, ls="--")
    ax.set_ylim(0, 24); ax.set_title(title, fontsize=9)
    ax.set_xlabel("s", fontsize=8); ax.set_ylabel("kHz", fontsize=8); ax.tick_params(labelsize=7)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    sr = sf.info(a.wav).samplerate

    cfg = PreprocessingConfig(frame_features="talia")
    calls = detect(a.wav, cfg)
    loose = detect(a.wav, dataclasses.replace(cfg, abrupt_up_freq_range_hz=float("inf")))
    clicks = [c for c in loose if not any(min(c[1], k[1]) > max(c[0], k[0]) for k in calls)]

    by_len = sorted(calls, key=lambda c: c[1] - c[0], reverse=True)
    show = by_len[:4] + by_len[len(by_len) // 2: len(by_len) // 2 + 4]
    n_rows = 3 if clicks else 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.4 * n_rows))
    for ax, (s, e) in zip(axes[:2].ravel(), show):
        panel(ax, a.wav, s, e, sr, f"CALL  t={s/sr:.1f}s  dur {(e-s)/sr:.2f}s")
    if clicks:
        for ax, (s, e) in zip(axes[2], clicks[:4]):
            panel(ax, a.wav, s, e, sr, f"NOT A CALL (click)  t={s/sr:.1f}s")
        for ax in axes[2][len(clicks[:4]):]:
            ax.axis("off")
    fig.suptitle("Marmoset calls (rows 1-2) vs. noise (row 3) -- 210503_001, beam-steered, "
                 "dashed = detected boundaries", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"{len(calls)} calls, {len(clicks)} clicks -> {a.out}")


if __name__ == "__main__":
    main()
