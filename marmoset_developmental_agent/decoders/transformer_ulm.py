"""Causal unit language model (uLM) over discrete vocalization units.

GmSLM (A.3) uses fairseq ``transformer_lm_big``: 12 layers, 16 heads, embed 1024,
FFN 4096, dropout 0.1, up to 3072 units per sample, trained on *non-deduplicated*
units. The model factorises

    p(u_1..u_T) = prod_t p(u_t | BOS, u_1..u_{t-1})

and is trained with the next-token NLL. The pre-LN transformer here, with learned
positions and tied input/output embeddings, is equivalent in function.

Hasson's 1kd_learning_model instead uses fairseq's ``TransformerUnitLanguageModel``
(MS-TLM) on *deduplicated* units plus a duration stream. That contradicts the GmSLM
no-dedup finding and is deliberately NOT reproduced here. Duration is carried by
the repeated tokens themselves.

Vocabulary layout: 0 = PAD, 1 = BOS, 2 = EOS, unit ``u`` -> id ``u + 3``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from marmoset_developmental_agent.config.agent_config import DecoderConfig


class UnitLanguageModel(nn.Module):
    PAD, BOS, EOS = 0, 1, 2
    N_SPECIAL = 3

    def __init__(self, n_units: int, cfg: DecoderConfig) -> None:
        super().__init__()
        self.n_units = n_units
        self.cfg = cfg
        self.vocab_size = n_units + self.N_SPECIAL
        self.tok_emb = nn.Embedding(self.vocab_size, cfg.embed_dim, padding_idx=self.PAD)
        self.pos_emb = nn.Embedding(cfg.max_positions + 1, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            cfg.embed_dim, cfg.num_heads, cfg.ffn_dim, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, cfg.num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, self.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        nn.init.normal_(self.tok_emb.weight, std=cfg.embed_dim**-0.5)

    # ------------------------------------------------------------------ #
    def _prepare(
        self, units: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """units (B, T) -> input [BOS, u...], target [u..., EOS], pad mask (True = pad)."""
        if units.dim() == 1:
            units = units.unsqueeze(0)
        B, T = units.shape
        T = min(T, self.cfg.max_positions)
        units = units[:, :T]
        device = self.tok_emb.weight.device
        units = units.to(device)
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
        lengths = lengths.to(device).clamp(max=T)
        pos = torch.arange(T + 1, device=device).unsqueeze(0)
        ids = units + self.N_SPECIAL
        inp = torch.cat([torch.full((B, 1), self.BOS, device=device, dtype=torch.long), ids], 1)
        tgt = torch.cat([ids, torch.full((B, 1), self.PAD, device=device, dtype=torch.long)], 1)
        tgt.scatter_(1, lengths.unsqueeze(1), self.EOS)
        pad = pos > lengths.unsqueeze(1)  # input positions after the last unit
        inp = inp.masked_fill(pad, self.PAD)
        tgt = tgt.masked_fill(pad, self.PAD)
        return inp, tgt, pad

    def forward(
        self, ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, return_hidden: bool = False
    ) -> torch.Tensor:
        T = ids.shape[1]
        pos = torch.arange(T, device=ids.device)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
        h = self.final_norm(self.blocks(x, mask=causal, src_key_padding_mask=pad_mask, is_causal=True))
        return h if return_hidden else self.lm_head(h)

    # ------------------------------------------------------------------ #
    def loss(self, units: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean next-unit NLL (nats/token), including the EOS prediction."""
        inp, tgt, pad = self._prepare(units, lengths)
        logits = self(inp, pad)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), tgt.reshape(-1), ignore_index=self.PAD)

    @torch.no_grad()
    def token_log_probs(
        self, units: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """log p(u_t | u_<t) for every unit (EOS excluded). Returns (logp (B, T), valid mask (B, T))."""
        inp, tgt, pad = self._prepare(units, lengths)
        logp = F.log_softmax(self(inp, pad).float(), -1)
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[:, :-1]
        valid = (tgt[:, :-1] >= self.N_SPECIAL)
        return tok_lp.masked_fill(~valid, 0.0), valid

    @torch.no_grad()
    def sequence_log_prob(
        self, units: torch.Tensor, lengths: Optional[torch.Tensor] = None, reduction: str = "sum"
    ) -> torch.Tensor:
        """Log-likelihood per sequence: summed or length-normalised ("mean")."""
        lp, valid = self.token_log_probs(units, lengths)
        total = lp.sum(-1)
        return total if reduction == "sum" else total / valid.sum(-1).clamp(min=1)

    @torch.no_grad()
    def hidden_states(self, units: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Final-layer state after reading each unit: (B, T, D) (used by probes)."""
        inp, _, pad = self._prepare(units, lengths)
        return self(inp, pad, return_hidden=True)[:, 1:]

    @torch.no_grad()
    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Ancestral sampling continuation of a 1-D unit prompt. Returns units only (no specials).

        GmSLM uses beam-search sampling (beam 5, temperature 1.5) in fairseq. Plain
        temperature sampling is used here for simplicity.
        """
        device = self.tok_emb.weight.device
        ids = torch.cat([torch.tensor([self.BOS], device=device), prompt.to(device) + self.N_SPECIAL]).unsqueeze(0)
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.cfg.max_positions :])[:, -1, self.N_SPECIAL :] / max(temperature, 1e-5)
            if top_k:
                kth = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=generator).to(device)
            ids = torch.cat([ids, nxt + self.N_SPECIAL], 1)
        return ids[0, 1:] - self.N_SPECIAL
