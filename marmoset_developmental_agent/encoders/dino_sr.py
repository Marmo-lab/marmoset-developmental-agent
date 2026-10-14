"""DINO-SR encoder: self-distillation with online clustering (Liu et al., NeurIPS 2023).

Compact re-implementation of ``references/hasson_dev/training/speech_learning/models/dinosr.py``
(itself derived from ``references/dinosr``), used by the Hasson developmental agent.

Architecture
------------
wav (16 kHz) -> 7-layer conv extractor (stride 320, 20 ms) -> LayerNorm -> Linear(conv_dim -> D)
-> [student: masked with learned ``mask_emb``] -> conv positional embedding -> N transformer layers.
The *teacher* is an EMA copy of the transformer part (``ema_transformer_only``). It sees the
same pre-encoder features without masking.

Objective (per training step)
-----------------------------
For each of the top-K teacher layers l (K = ``average_top_k_layers``):
  y_l = InstanceNorm_time(teacher_l(x))                      (B, T, D)
  z_l = argmin_v || y_l - c_{l,v} ||^2                       online codebook, V entries
  L  += CE( head_l(student_N(x_masked))[masked], z_l[masked] )
L is divided by the number of masked frames. Masking is span-based: prob 0.8, span 10.

Codebook update (EMA over the masked targets assigned to each codeword):
  n_{l,v} <- a n_{l,v} + (1 - a) count_v,   s_{l,v} <- a s_{l,v} + (1 - a) sum_{t: z=v} y_t
  c_{l,v}  = s_{l,v} / n_{l,v},             a = codebook_decay (only for used codewords)

Teacher update after each optimizer step: theta_T <- tau theta_T + (1 - tau) theta_S, with tau
annealed linearly from ``ema_decay`` to ``ema_end_decay`` over ``ema_anneal_end_step`` steps.

Discrete units for the uLM come from the codebook of transformer layer ``layer_for_tokens``.
The index is **0-based**, as in the official code (``layer_i = i + encoder_layers - n_codebooks``).
Hasson's ``layer_for_tokens: 9`` with 12 layers and K=8 therefore means the 10th layer,
codebook index 9 - 12 + 8 = 5. (GmSLM's HuBERT "layer 1" is instead the output of the
first layer; see ``marmoset_hubert.py``.) One unit per 20 ms frame, with no deduplication.

Simplifications vs. the reference implementation: a single positional conv (not a
5-deep stack), plain ``nn.TransformerEncoderLayer`` outputs as layer targets (instead
of the pre-residual FFN output), no layerdrop, EMA over every teacher parameter.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from marmoset_developmental_agent.config.agent_config import DinoSRConfig
from marmoset_developmental_agent.encoders.base_encoder import CONV_LAYERS, BaseAudioEncoder


class ConvFeatureExtractor(nn.Module):
    """wav2vec2 conv stack in ``layer_norm`` mode (Conv -> LayerNorm(channels) -> GELU)."""

    def __init__(self, dim: int, layers: Tuple[Tuple[int, int], ...] = CONV_LAYERS) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_ch = 1
        for k, s in layers:
            self.convs.append(nn.Conv1d(in_ch, dim, k, s, bias=False))
            self.norms.append(nn.LayerNorm(dim))
            in_ch = dim

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.unsqueeze(1)  # (B, 1, L)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = F.gelu(norm(x.transpose(1, 2)).transpose(1, 2))
        return x.transpose(1, 2)  # (B, T, C)


class _TransformerStack(nn.Module):
    """Conv positional embedding + post-LN transformer, returning every layer's output."""

    def __init__(self, cfg: DinoSRConfig) -> None:
        super().__init__()
        k = cfg.conv_pos_kernel if cfg.conv_pos_kernel % 2 == 1 else cfg.conv_pos_kernel + 1
        self.pos_conv = nn.Conv1d(cfg.embed_dim, cfg.embed_dim, k, padding=k // 2, groups=cfg.conv_pos_groups)
        self.layer_norm = nn.LayerNorm(cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                cfg.embed_dim, cfg.num_heads, cfg.ffn_dim, cfg.dropout,
                activation="gelu", batch_first=True, norm_first=False,
            )
            for _ in range(cfg.num_layers)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = x + F.gelu(self.pos_conv(x.transpose(1, 2))).transpose(1, 2)
        x = self.dropout(self.layer_norm(x))
        outs: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return x, outs


def compute_span_mask(
    batch: int, frames: int, mask_prob: float, span: int, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """wav2vec2-style span mask: about ``mask_prob * T / span`` spans of length ``span`` (overlaps allowed)."""
    mask = torch.zeros(batch, frames, dtype=torch.bool)
    span = min(span, frames)
    n_spans = max(1, int(mask_prob * frames / span + torch.rand(1, generator=generator).item()))
    for b in range(batch):
        starts = torch.randint(0, frames - span + 1, (n_spans,), generator=generator)
        for s in starts.tolist():
            mask[b, s : s + span] = True
    return mask


class DinoSREncoder(BaseAudioEncoder):
    """Trainable DINO-SR discretiser (``cfg.codebook_size`` units, 20 ms frames)."""

    is_trainable = True

    def __init__(self, cfg: DinoSRConfig, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.cfg = cfg
        K, V, D = cfg.average_top_k_layers, cfg.codebook_size, cfg.embed_dim
        if not 1 <= K <= cfg.num_layers:
            raise ValueError("average_top_k_layers must be in [1, num_layers]")
        self.token_codebook = cfg.layer_for_tokens - cfg.num_layers + K
        if not 0 <= self.token_codebook < K:
            raise ValueError(
                f"layer_for_tokens={cfg.layer_for_tokens} (0-based) has no codebook; valid range is "
                f"[{cfg.num_layers - K}, {cfg.num_layers - 1}]"
            )
        self.feature_extractor = ConvFeatureExtractor(cfg.conv_dim)
        self.feat_norm = nn.LayerNorm(cfg.conv_dim)
        self.post_extract_proj = nn.Linear(cfg.conv_dim, D)
        self.mask_emb = nn.Parameter(torch.empty(D).uniform_())
        self.student = _TransformerStack(cfg)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.heads = nn.ModuleList(nn.Linear(D, V) for _ in range(K))
        # Codebook as running sums / counts (codeword = sum / count), fp32 buffers.
        self.register_buffer("codebook_sums", torch.randn(K, V, D) / V**0.5)
        self.register_buffer("codebook_counts", torch.ones(K, V))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self._gen = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------ #
    def get_vocab_size(self) -> int:
        return self.cfg.codebook_size

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("teacher.")]

    def _pre_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        wav = self._as_batch(wav).to(self.mask_emb.device)
        if self.cfg.normalize_waveform:
            wav = self._normalize(wav)
        return self.post_extract_proj(self.feat_norm(self.feature_extractor(wav)))

    def _codewords(self, i: int) -> torch.Tensor:
        return self.codebook_sums[i] / self.codebook_counts[i].unsqueeze(1)

    @staticmethod
    def _instance_norm(x: torch.Tensor) -> torch.Tensor:
        return F.instance_norm(x.transpose(1, 2).float()).transpose(1, 2)

    def _teacher_targets(self, feats: torch.Tensor) -> List[torch.Tensor]:
        was_training = self.teacher.training
        self.teacher.eval()
        with torch.no_grad():
            _, outs = self.teacher(feats)
        self.teacher.train(was_training)
        K = self.cfg.average_top_k_layers
        return [self._instance_norm(o) for o in outs[-K:]]

    @staticmethod
    def _nearest(x: torch.Tensor, codewords: torch.Tensor) -> torch.Tensor:
        # ||x||^2 + ||c||^2 - 2 x.c  (argmin over codewords)
        d = (x**2).sum(-1, keepdim=True) + (codewords**2).sum(-1) - 2 * x @ codewords.t()
        return d.argmin(-1)

    # ------------------------------------------------------------------ #
    def training_step(self, wav: torch.Tensor) -> torch.Tensor:
        feats = self._pre_encoder(wav)  # (B, T, D)
        B, T, _ = feats.shape
        mask = compute_span_mask(B, T, self.cfg.mask_prob, self.cfg.mask_length, self._gen).to(feats.device)
        x = torch.where(mask.unsqueeze(-1), self.mask_emb.to(feats.dtype), feats)
        student_out, _ = self.student(x)
        targets = self._teacher_targets(feats.detach())
        loss = feats.new_zeros(())
        n_masked = int(mask.sum())
        for i, tgt in enumerate(targets):
            y = tgt[mask]  # (M, D)
            with torch.no_grad():
                ids = self._nearest(y, self._codewords(i))
            logits = self.heads[i](student_out[mask]).float()
            loss = loss + F.cross_entropy(logits, ids, reduction="sum")
            if self.training and self.cfg.codebook_decay < 1:
                self._update_codebook(i, y, ids)
        return loss / max(1, n_masked)

    @torch.no_grad()
    def _update_codebook(self, i: int, y: torch.Tensor, ids: torch.Tensor) -> None:
        V = self.cfg.codebook_size
        onehot = F.one_hot(ids, V).float()  # (M, V)
        count = onehot.sum(0)
        memory = onehot.t() @ y  # (V, D)
        alpha = torch.where(count > 0, torch.full_like(count, self.cfg.codebook_decay), torch.ones_like(count))
        self.codebook_counts[i] = alpha * self.codebook_counts[i] + (1 - alpha) * count
        self.codebook_sums[i] = alpha.unsqueeze(1) * self.codebook_sums[i] + (1 - alpha).unsqueeze(1) * memory

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        """EMA teacher update with linearly annealed decay."""
        step = int(self.num_updates)
        c = self.cfg
        if step >= c.ema_anneal_end_step:
            tau = c.ema_end_decay
        else:
            tau = c.ema_decay + (c.ema_end_decay - c.ema_decay) * step / max(1, c.ema_anneal_end_step)
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.mul_(tau).add_(ps.detach(), alpha=1 - tau)
        self.num_updates += 1

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_features(self, wav: torch.Tensor, layer: Optional[int] = None) -> torch.Tensor:
        """Instance-normalised teacher output of 0-based ``layer`` (default ``layer_for_tokens``), (B, T, D)."""
        layer = self.cfg.layer_for_tokens if layer is None else layer
        feats = self._pre_encoder(wav)
        self.teacher.eval()
        _, outs = self.teacher(feats)
        return self._instance_norm(outs[layer])

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Nearest codeword of the ``layer_for_tokens`` codebook per frame: (B, T), repeats preserved."""
        y = self.extract_features(wav)
        B, T, D = y.shape
        return self._nearest(y.reshape(-1, D), self._codewords(self.token_codebook)).view(B, T)
