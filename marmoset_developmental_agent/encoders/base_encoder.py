"""Common interface for audio -> discrete-unit encoders (vocalization-to-unit, V2U).

Both encoders share the wav2vec2/HuBERT convolutional front-end: 7 conv layers,
kernels (10,3,3,3,3,2,2), strides (5,2,2,2,2,2,2), total stride 320. At 16 kHz that
is one frame every 20 ms (50 Hz), so a 10 s chunk (160000 samples) gives 499 frames.

NO-DEDUPLICATION RULE. ``encode`` returns exactly one token per frame. Runs of
identical tokens are never collapsed, because unit duration carries information
in marmoset calls (GmSLM Sec. 5.3, Table 4). :func:`check_no_dedup` enforces this.
Collapsing belongs nowhere in the pipeline. For duration statistics, use
:func:`run_lengths`, which does not modify tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

CONV_LAYERS: Tuple[Tuple[int, int], ...] = ((10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2))


def conv_output_length(num_samples: int, layers: Sequence[Tuple[int, int]] = CONV_LAYERS) -> int:
    """Frames produced by an unpadded conv stack: L <- floor((L - k) / s) + 1 per layer."""
    n = num_samples
    for k, s in layers:
        n = (n - k) // s + 1
        if n <= 0:
            return 0
    return n


class BaseAudioEncoder(nn.Module, ABC):
    """Abstract V2U encoder: waveform (B, L) @ 16 kHz -> LongTensor units (B, T)."""

    sample_rate: int = 16_000
    conv_layers: Tuple[Tuple[int, int], ...] = CONV_LAYERS
    #: True if the encoder learns online during developmental training (DINO-SR).
    is_trainable: bool = False

    @property
    def frame_rate(self) -> float:
        """Frames per second (50.0 for the standard conv stack at 16 kHz)."""
        stride = 1
        for _, s in self.conv_layers:
            stride *= s
        return self.sample_rate / stride

    def num_frames(self, num_samples: int) -> int:
        return conv_output_length(num_samples, self.conv_layers)

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Number of discrete units (50 for GmSLM HuBERT, 512 for DINO-SR)."""

    @abstractmethod
    def extract_features(self, wav: torch.Tensor) -> torch.Tensor:
        """Continuous frame features (B, T, D) from which units are quantised."""

    @abstractmethod
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Discrete units (B, T), one per 20 ms frame, repeats preserved."""

    # ---- optional hooks for trainable encoders --------------------------------
    def fit(self, wavs: Iterable[torch.Tensor]) -> None:
        """Offline fitting (for example k-means). No-op by default."""

    def training_step(self, wav: torch.Tensor) -> Optional[torch.Tensor]:
        """Self-supervised loss for one batch, or None if not trainable."""
        return None

    def after_optimizer_step(self) -> None:
        """Called after each encoder optimizer step (for example an EMA teacher update)."""

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # ---- helpers ---------------------------------------------------------------
    @staticmethod
    def _as_batch(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2:
            raise ValueError(f"expected waveform (B, L), got {tuple(wav.shape)}")
        return wav.float()

    @staticmethod
    def _normalize(wav: torch.Tensor) -> torch.Tensor:
        """Per-utterance zero-mean / unit-variance (fairseq ``normalize: true``)."""
        return torch.nn.functional.layer_norm(wav, wav.shape[-1:])


def check_no_dedup(tokens: torch.Tensor, num_samples: int, encoder: BaseAudioEncoder) -> None:
    """Raise if ``tokens`` has fewer frames than the conv stack produces (collapsed repeats)."""
    expected = encoder.num_frames(num_samples)
    if tokens.shape[-1] != expected:
        raise AssertionError(
            f"Token sequence has {tokens.shape[-1]} frames, expected {expected}: "
            "consecutive duplicates must NOT be removed (GmSLM no-dedup rule)."
        )


def run_lengths(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(values, run lengths) of a 1-D token sequence. Analysis only: never feed to the uLM."""
    if tokens.dim() != 1:
        raise ValueError("run_lengths expects a 1-D sequence")
    return torch.unique_consecutive(tokens, return_counts=True)
