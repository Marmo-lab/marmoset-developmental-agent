"""Developmental trajectory tracking (Hasson framework).

Evaluated every ``eval_every_days`` simulated days (weekly by default, matching Hasson's
``dataset_save_cycle_every: 7`` and ``--extract-next-cycle-only``):

1. **Next-week surprisal / PPL**: the current model scores the *unseen* chunks of days
   D+1..D+7 before training on them.
     S = -(1/N) sum_t log2 p(u_t | u_<t)   (bits/unit),   PPL = exp(-(1/N) sum_t ln p)
2. **Out-of-domain PPL**: the same statistics on a fixed held-out set (for example PHEEDB or
   another colony).
3. **Learned lexicon**: a marmoset adaptation of
   ``references/hasson_dev/analysis/audio_postprocessing/{2_smooth_probs,3_add_words_learned}.py``.
   Words (MFA-aligned) become calls (from the segmenter):
     a. token probabilities p_t = exp(log p(u_t | u_<t)) are smoothed along the sequence with
        Hasson's decaying weighted average
            m_t = (p_t w_t + m_{t-1} W_{t-1} f) / (W_{t-1} f + w_t),   W_t = W_{t-1} f + w_t
        with factor f = ``lexicon_smooth_factor`` and w_t = 1 (no dedup, so every frame has duration 1);
     b. within a call, a new segment starts where m_t < confidence, or where
        m_t < patience and m_t - m_{t-1} < stability;
     c. a call instance is *learned* if its largest segment has every m_t >= confidence and
        covers >= ``lexicon_min_coverage`` of the call's frames;
     d. a call *type* enters the lexicon the first evaluation where at least
        ``lexicon_min_learned_fraction`` of its instances are learned.
   Call types come from labels when available (mock data, annotated sets). Otherwise
   the key is "unlabelled", and only aggregate learned fractions are meaningful.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from marmoset_developmental_agent.config.agent_config import EvaluationConfig
from marmoset_developmental_agent.decoders.transformer_ulm import UnitLanguageModel
from marmoset_developmental_agent.encoders.base_encoder import BaseAudioEncoder
from marmoset_developmental_agent.evaluation.scoring import encode_waveforms, surprisal_report, token_log_probs
from marmoset_developmental_agent.preprocessing.dataset_builder import ChronologicalDataset
from marmoset_developmental_agent.preprocessing.segmentation import Chunk

logger = logging.getLogger(__name__)


def smooth_probs(p: np.ndarray, factor: float = 0.5, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Hasson's ``mean_weighted_average`` (causal, exponentially forgetting)."""
    w = np.ones_like(p) if weights is None else weights
    out = np.empty_like(p, dtype=float)
    mwa, mwa_w = 0.0, 0.0
    for i in range(len(p)):
        mwa_w *= factor
        denom = mwa_w + w[i]
        mwa = (p[i] * w[i] + mwa * mwa_w) / (denom if denom else 1.0)
        mwa_w += w[i]
        out[i] = mwa
    return out


def call_is_learned(m: np.ndarray, confidence: float, patience: float, stability: float, min_coverage: float) -> bool:
    """Apply the segment-break rule to smoothed probs ``m`` of one call's frames."""
    if len(m) == 0:
        return False
    diff = np.diff(m, prepend=m[0] + 1.0)  # first frame never breaks on the slope rule
    breaks = (m < confidence) | ((m < patience) & (diff < stability))
    best = run = 0
    for b, v in zip(breaks, m):
        run = 0 if (b or v < confidence) else run + 1
        best = max(best, run)
    return best / len(m) >= min_coverage


@dataclass
class LexiconEntry:
    call_type: str
    first_learned_day: Optional[int] = None
    history: List[Dict[str, float]] = field(default_factory=list)


class LearnedLexicon:
    def __init__(self, cfg: EvaluationConfig, frame_rate: float) -> None:
        self.cfg = cfg
        self.frame_rate = frame_rate
        self.entries: Dict[str, LexiconEntry] = {}

    def update(self, day: int, chunks: Sequence[Chunk], log_probs: Sequence[np.ndarray]) -> Dict[str, float]:
        c = self.cfg
        learned: Dict[str, List[bool]] = defaultdict(list)
        for chunk, lp in zip(chunks, log_probs):
            m = smooth_probs(np.exp(lp), c.lexicon_smooth_factor)
            for call in chunk.calls:
                s = int(call.start_s * self.frame_rate)
                e = min(len(m), int(np.ceil(call.end_s * self.frame_rate)))
                if e - s < 2:
                    continue
                ok = call_is_learned(m[s:e], c.lexicon_confidence, c.lexicon_patience, c.lexicon_stability,
                                     c.lexicon_min_coverage)
                learned[call.call_type or "unlabelled"].append(ok)
        fractions: Dict[str, float] = {}
        for ctype, flags in learned.items():
            frac = float(np.mean(flags))
            fractions[ctype] = frac
            entry = self.entries.setdefault(ctype, LexiconEntry(ctype))
            entry.history.append({"day": day + 1, "learned_fraction": frac, "n_instances": len(flags)})
            if entry.first_learned_day is None and frac >= c.lexicon_min_learned_fraction:
                entry.first_learned_day = day + 1
        return fractions

    @property
    def size(self) -> int:
        return sum(e.first_learned_day is not None for e in self.entries.values())

    def as_dict(self) -> Dict[str, object]:
        return {k: {"first_learned_day": e.first_learned_day, "history": e.history} for k, e in self.entries.items()}


class DevelopmentalTracker:
    """Callback for ``DevelopmentalTrainer.run(on_day_end=...)``."""

    def __init__(
        self,
        encoder: BaseAudioEncoder,
        ulm: UnitLanguageModel,
        dataset: ChronologicalDataset,
        cfg: EvaluationConfig,
        ood_waveforms: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        self.encoder, self.ulm, self.dataset, self.cfg = encoder, ulm, dataset, cfg
        self.ood = list(ood_waveforms) if ood_waveforms is not None else None
        self.lexicon = LearnedLexicon(cfg, encoder.frame_rate)
        self.history: List[Dict[str, object]] = []

    def __call__(self, day: int, _log: object = None) -> None:
        if (day + 1) % self.cfg.eval_every_days:
            return
        rec: Dict[str, object] = {"day": day + 1}
        nxt = self.dataset.days(day + 1, day + 1 + self.cfg.eval_every_days)[: self.cfg.max_eval_chunks]
        if nxt:
            lps = token_log_probs(self.ulm, encode_waveforms(self.encoder, [c.audio() for c in nxt]))
            rec["next_period"] = surprisal_report(lps).as_dict()
            rec["lexicon_learned_fraction"] = self.lexicon.update(day, nxt, lps)
            rec["lexicon_size"] = self.lexicon.size
        if self.ood:
            lps = token_log_probs(self.ulm, encode_waveforms(self.encoder, self.ood))
            rec["out_of_domain"] = surprisal_report(lps).as_dict()
        self.history.append(rec)
        nx = rec.get("next_period", {})
        od = rec.get("out_of_domain", {})
        logger.info(
            "Trajectory day %d | next-period surprisal %s bits, PPL %s | OOD PPL %s | lexicon %s",
            day + 1,
            f"{nx['mean_surprisal_bits']:.3f}" if nx else "n/a",
            f"{nx['perplexity']:.2f}" if nx else "n/a",
            f"{od['perplexity']:.2f}" if od else "n/a",
            rec.get("lexicon_learned_fraction", "n/a"),
        )
