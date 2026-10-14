"""Unit-to-Vocalization (U2V) HiFi-GAN (Kong et al., 2020; Polyak et al., 2021).

GmSLM (Sec. 3.2, A.3) resynthesises marmoset audio from units with Polyak et al.'s
unit HiFi-GAN, *without* speaker embeddings or F0 (COLONYDB has no speaker labels).
It trains for 400k steps on three 24 GB GPUs. The generator:

    units (B, T) -> Embedding(V, 128) -> Conv1d(128 -> C, k=7)
    -> for each (rate r_i, kernel k_i): LeakyReLU -> ConvTranspose1d(C/2^i -> C/2^(i+1), k_i, stride r_i)
                                        -> MRF (mean of ResBlocks with kernels 3/7/11, dilations 1/3/5)
    -> LeakyReLU -> Conv1d(-> 1, k=7) -> tanh

With rates (5, 4, 4, 2, 2) the product is 320, so one 20 ms unit gives 320 samples at 16 kHz.

Training (multi-period/multi-scale discriminators, mel + feature-matching losses) is out of
scope here. This module runs inference from a ``checkpoint_path`` (a dict with a
``generator`` state_dict, as in the speech-resynthesis repo). Without one it falls back to a
mock: ``additive`` renders each unit as a 5-8 kHz tone (an audible, interpretable sketch),
while ``neural`` runs the randomly initialised generator (shape check only).
"""

from __future__ import annotations

import logging
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from marmoset_developmental_agent.config.agent_config import VocoderConfig

logger = logging.getLogger(__name__)
LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    """HiFi-GAN v1 residual block (dilated conv pairs)."""

    def __init__(self, ch: int, kernel: int, dilations: List[int]) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(ch, ch, kernel, dilation=d, padding=d * (kernel - 1) // 2) for d in dilations
        )
        self.convs2 = nn.ModuleList(nn.Conv1d(ch, ch, kernel, padding=(kernel - 1) // 2) for _ in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
        return x


class UnitHiFiGANGenerator(nn.Module):
    def __init__(self, n_units: int, cfg: VocoderConfig) -> None:
        super().__init__()
        if len(cfg.upsample_rates) != len(cfg.upsample_kernel_sizes):
            raise ValueError("upsample_rates and upsample_kernel_sizes must align")
        self.hop = math.prod(cfg.upsample_rates)
        self.unit_emb = nn.Embedding(n_units, cfg.unit_embed_dim)
        C = cfg.upsample_initial_channel
        self.conv_pre = nn.Conv1d(cfg.unit_embed_dim, C, 7, padding=3)
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        ch = C
        for r, k in zip(cfg.upsample_rates, cfg.upsample_kernel_sizes):
            out_ch = max(1, ch // 2)
            self.ups.append(nn.ConvTranspose1d(ch, out_ch, k, r, padding=(k - r) // 2))
            self.mrfs.append(
                nn.ModuleList(ResBlock(out_ch, rk, rd) for rk, rd in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilations))
            )
            ch = out_ch
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, units: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(self.unit_emb(units).transpose(1, 2))
        for up, mrf in zip(self.ups, self.mrfs):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            x = sum(block(x) for block in mrf) / len(mrf)
        return torch.tanh(self.conv_post(F.leaky_relu(x))).squeeze(1)  # (B, T * hop)


class HiFiGANU2V(nn.Module):
    """Unit-to-waveform wrapper: ``synthesize(units) -> waveform`` at 16 kHz."""

    def __init__(self, n_units: int, cfg: VocoderConfig, sample_rate: int = 16_000) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_units = n_units
        self.sample_rate = sample_rate
        self.generator = UnitHiFiGANGenerator(n_units, cfg)
        self.is_mock = cfg.checkpoint_path is None
        if not self.is_mock:
            state = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
            self.generator.load_state_dict(state.get("generator", state))
            logger.info("Loaded U2V HiFi-GAN from %s", cfg.checkpoint_path)
        self.generator.eval()

    @property
    def hop_length(self) -> int:
        return self.generator.hop

    @torch.no_grad()
    def synthesize(self, units: torch.Tensor) -> torch.Tensor:
        """units (T,) or (B, T) -> waveform (B, T * hop_length)."""
        if units.dim() == 1:
            units = units.unsqueeze(0)
        units = units.long().clamp(0, self.n_units - 1)
        if self.is_mock and self.cfg.mock_mode == "additive":
            return self._additive(units)
        return self.generator(units.to(next(self.generator.parameters()).device))

    def _additive(self, units: torch.Tensor) -> torch.Tensor:
        """Phase-continuous tone whose pitch encodes the unit: f(u) = 5 kHz + 3 kHz * u / V."""
        freq = 5_000 + 3_000 * units.float() / max(1, self.n_units - 1)  # (B, T)
        freq = freq.repeat_interleave(self.hop_length, dim=1)
        phase = 2 * math.pi * torch.cumsum(freq, dim=1) / self.sample_rate
        return 0.3 * torch.sin(phase)
