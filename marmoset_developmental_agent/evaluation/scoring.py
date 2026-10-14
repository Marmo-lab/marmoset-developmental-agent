"""Shared helpers: waveform -> units -> uLM log-likelihoods."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from marmoset_developmental_agent.decoders.transformer_ulm import UnitLanguageModel
from marmoset_developmental_agent.encoders.base_encoder import BaseAudioEncoder

LN2 = float(np.log(2.0))


@torch.no_grad()
def encode_waveforms(
    encoder: BaseAudioEncoder, wavs: Sequence[np.ndarray], batch_size: int = 16
) -> List[torch.Tensor]:
    """Tokenise variable-length waveforms. Equal lengths are batched together. No dedup."""
    encoder.eval()
    device = next(encoder.parameters()).device
    out: List[torch.Tensor] = [torch.empty(0)] * len(wavs)
    by_len: Dict[int, List[int]] = defaultdict(list)
    for i, w in enumerate(wavs):
        by_len[len(w)].append(i)
    for idxs in by_len.values():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s : s + batch_size]
            x = torch.from_numpy(np.stack([wavs[i] for i in chunk])).float().to(device)
            toks = encoder.encode(x).cpu()
            for i, t in zip(chunk, toks):
                out[i] = t
    return out


@torch.no_grad()
def token_log_probs(ulm: UnitLanguageModel, units: Sequence[torch.Tensor], batch_size: int = 16) -> List[np.ndarray]:
    """Per-token log p(u_t | u_<t) for each sequence (natural log)."""
    ulm.eval()
    res: List[np.ndarray] = []
    for s in range(0, len(units), batch_size):
        batch = units[s : s + batch_size]
        lengths = torch.tensor([len(u) for u in batch])
        padded = torch.nn.utils.rnn.pad_sequence(list(batch), batch_first=True)
        lp, valid = ulm.token_log_probs(padded, lengths)
        for row, v in zip(lp.cpu().numpy(), valid.cpu().numpy()):
            res.append(row[v])
    return res


def score_waveforms(
    encoder: BaseAudioEncoder, ulm: UnitLanguageModel, wavs: Sequence[np.ndarray],
    reduction: str = "mean", batch_size: int = 16,
) -> np.ndarray:
    """Sequence log-likelihood per waveform (sum or per-token mean)."""
    lps = token_log_probs(ulm, encode_waveforms(encoder, wavs, batch_size), batch_size)
    return np.array([lp.sum() if reduction == "sum" else lp.mean() for lp in lps])


@dataclass
class SurprisalReport:
    n_sequences: int
    n_tokens: int
    mean_nll_nats: float
    mean_surprisal_bits: float
    perplexity: float

    def as_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def surprisal_report(log_probs: Sequence[np.ndarray]) -> SurprisalReport:
    """Token-weighted surprisal: S = -mean log2 p;  PPL = exp(mean NLL)."""
    allv = np.concatenate([np.asarray(x) for x in log_probs]) if log_probs else np.zeros(0)
    nll = float(-allv.mean()) if len(allv) else float("nan")
    return SurprisalReport(len(log_probs), int(len(allv)), nll, nll / LN2, float(np.exp(nll)))


def pairwise_accuracy(real: np.ndarray, fake: np.ndarray) -> float:
    """P(score(real) > score(fake)), ties counting 1/2 (sWUGGY / sBLIMP convention)."""
    if len(real) == 0:
        return float("nan")
    return float(np.mean((real > fake) + 0.5 * (real == fake)))
