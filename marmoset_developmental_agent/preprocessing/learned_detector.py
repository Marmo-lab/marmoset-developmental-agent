"""Learned call detector: a small frame classifier trained on the hand labels.

Why learned. Both rule detectors (``"talia"``, ``"spectral"``) define a call acoustically:
energy above a floor, gaps under ~90 ms bridged. The user's labels define it by ear:
notes joined across silences, neighbouring calls kept apart, call-like noise ignored. No
threshold setting reconciles the two. Against all 527 labels, the spectral detector of
2026-10-21 split 11% of the labelled calls into pieces and merged 9% with a neighbour, and
21% of its detections touched no label (DETECTOR_EVAL finding 12). A classifier trained on
the labels learns the labeller's segmentation instead. This changes only the
preprocessing: the developmental agent downstream remains unsupervised.

Input [ours]. The detection signal at 48 kHz, high-passed at ``highpass_cutoff_hz`` (1 kHz
for this path, so cage noise keeps its low-frequency signature). STFT with a 1024-point
Hann window (21 ms) and a 480-sample hop (10 ms). Power is pooled into 250 Hz bands from
the cutoff to 24 kHz, which covers the call fundamentals (4.6-9.7 kHz) and their
harmonics (DETECTOR_EVAL finding 11). Each band is expressed in dB above its median over
the block, as the spectral detector does, so the input is invariant to recorder gain and
to the filter's shape.

Model [ours]. Three 1D convolutions over time (bands as channels, kernel 5, the last
dilated, about 0.13 s receptive field), then a bidirectional GRU for context, then one logit
per frame. The loss is masked binary cross-entropy: ``c`` labels are 1, background is 0,
and ``?`` labels and frames outside the labelled minute are ignored. Frames with
``p >= learned_threshold`` are active. Grouping into calls and the duration limits are the
shared ``activity_to_calls`` path.

Training and evaluation: ``tools/analysis/train_call_detector.py``. The shipped checkpoint
was fit on all 45 labelled clips; its quality figures come from clip-grouped
cross-validation, where each clip is scored by a model that never saw it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy import signal
from torch import nn

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]


def band_edges(cfg: PreprocessingConfig) -> np.ndarray:
    """Band edges in Hz: ``learned_band_hz`` wide, from the high-pass cutoff to Nyquist."""
    return np.arange(cfg.highpass_cutoff_hz, cfg.detection_sample_rate / 2 + 1e-6, cfg.learned_band_hz)


def learned_features(audio: np.ndarray, sample_rate: int, cfg: PreprocessingConfig) -> Tuple[np.ndarray, np.ndarray]:
    """``(features (F, T), frame centre times (T,))``: band power in dB above each band's median."""
    if audio.ndim != 1:
        raise ValueError(f"learned features take one signal, got shape {audio.shape}")
    if sample_rate != cfg.detection_sample_rate:
        raise ValueError(f"learned features expect the detection rate {cfg.detection_sample_rate}, got {sample_rate}")
    n_fft, hop = cfg.learned_n_fft, cfg.learned_hop_length
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    f, t, s = signal.stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    p = np.abs(s) ** 2
    edges = band_edges(cfg)
    bands = np.stack([p[(f >= lo) & (f < hi)].mean(axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
    db = 10 * np.log10(bands + 1e-20)
    return (db - np.median(db, axis=1, keepdims=True)).astype(np.float32), t


def normalise(features: np.ndarray) -> np.ndarray:
    """Clip to [-10, 50] dB above the floor and scale to about unit range."""
    return np.clip(features, -10.0, 50.0) / 20.0


class CallNet(nn.Module):
    """Frame classifier: bands as channels, 1D convs over time, BiGRU, one logit per frame."""

    def __init__(self, n_bands: int, hidden: int = 96) -> None:
        super().__init__()

        def block(i: int, o: int, d: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(o), nn.ReLU())

        self.n_bands, self.hidden = n_bands, hidden
        self.cnn = nn.Sequential(block(n_bands, hidden, 1), block(hidden, hidden, 1), block(hidden, hidden, 2))
        self.gru = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, F, T)`` normalised features -> ``(B, T)`` logits."""
        h, _ = self.gru(self.cnn(x).transpose(1, 2))
        return self.out(h).squeeze(-1)


def resolve_model_path(path: str) -> Path:
    """Relative paths are taken from the repository root, so the default works from any cwd."""
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def save_call_detector(net: CallNet, path: Path) -> None:
    torch.save({"n_bands": net.n_bands, "hidden": net.hidden, "state_dict": net.state_dict()}, path)


@lru_cache(maxsize=4)
def load_call_detector(path: str) -> CallNet:
    """Load (once per process) a checkpoint written by :func:`save_call_detector`."""
    ckpt = torch.load(resolve_model_path(path), map_location="cpu", weights_only=True)
    net = CallNet(ckpt["n_bands"], ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    return net.eval()


def call_probability(features: np.ndarray, net: CallNet) -> np.ndarray:
    """Per-frame call probability for one block of features."""
    if features.shape[0] != net.n_bands:
        raise ValueError(f"model expects {net.n_bands} bands, features have {features.shape[0]}")
    with torch.no_grad():
        x = torch.from_numpy(normalise(features))[None]
        return torch.sigmoid(net(x))[0].numpy()
