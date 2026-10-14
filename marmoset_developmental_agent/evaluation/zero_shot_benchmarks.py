"""GmSLM zero-shot structural benchmarks (Sec. 3.3, Fig. 4).

Each task pairs a real sequence with a pseudo-marmoset distractor. Accuracy is the
fraction of pairs where the uLM gives the real one the higher likelihood (chance = 50%).

* **Shuffle**: calls within a 10 s segment are randomly reordered. Gap positions and
  durations are kept, so only call order changes.
* **Reversal**: the whole segment is time-reversed.
* **Concat**: first half of segment A + second half of segment B (split at the time
  midpoint, so lengths are equal). The real counterpart is A.
* **Phee eval** (PHEEDB call-response pairs): the call is kept and the response replaced:
  - CallerChange: the response comes from a *different* individual;
  - ReceiverChange: same responder, but the response was addressed to someone else.

Reported GmSLM accuracies: Shuffle 84.8, Concat 79.9, Reversal 90.5, ReceiverChange 65.3,
CallerChange 71.5.

Likelihoods are length-normalised by default (``EvaluationConfig.score_reduction = "mean"``).
Shuffle, Reversal and Concat pairs have equal length, so sum and mean agree there; for
Phee pairs, whose responses differ in length, mean avoids a length bias.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from marmoset_developmental_agent.config.agent_config import EvaluationConfig
from marmoset_developmental_agent.decoders.transformer_ulm import UnitLanguageModel
from marmoset_developmental_agent.encoders.base_encoder import BaseAudioEncoder
from marmoset_developmental_agent.evaluation.scoring import pairwise_accuracy, score_waveforms
from marmoset_developmental_agent.preprocessing.segmentation import CallSegment, Chunk
from marmoset_developmental_agent.utils.synthetic_data import PheeExchange

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Distractor construction
# --------------------------------------------------------------------------- #
def _spans(calls: Sequence[CallSegment], sr: int, n: int) -> List[Tuple[int, int]]:
    spans = sorted((max(0, int(c.start_s * sr)), min(n, int(c.end_s * sr))) for c in calls)
    out: List[Tuple[int, int]] = []
    for s, e in spans:  # drop overlaps
        if e > s and (not out or s >= out[-1][1]):
            out.append((s, e))
    return out


def shuffle_calls(wav: np.ndarray, calls: Sequence[CallSegment], sr: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    """Reorder call snippets, keeping gap snippets in place. None if < 2 calls."""
    spans = _spans(calls, sr, len(wav))
    if len(spans) < 2:
        return None
    snippets = [wav[s:e] for s, e in spans]
    perm = rng.permutation(len(spans))
    while np.all(perm == np.arange(len(spans))):
        perm = rng.permutation(len(spans))
    pieces = [wav[: spans[0][0]]]
    for slot, (s, e) in enumerate(spans):
        pieces.append(snippets[perm[slot]])
        nxt = spans[slot + 1][0] if slot + 1 < len(spans) else len(wav)
        pieces.append(wav[e:nxt])
    out = np.concatenate(pieces)
    assert len(out) == len(wav)
    return out


def reverse_audio(wav: np.ndarray) -> np.ndarray:
    return wav[::-1].copy()


def concat_halves(wav_a: np.ndarray, wav_b: np.ndarray) -> np.ndarray:
    """First half of A + second half of B (same total length as A)."""
    n = len(wav_a)
    mid = n // 2
    tail = wav_b[mid:n]
    if len(tail) < n - mid:
        tail = np.pad(tail, (0, n - mid - len(tail)))
    return np.concatenate([wav_a[:mid], tail])


def _dialogue(call: np.ndarray, gap_s: float, response: np.ndarray, sr: int) -> np.ndarray:
    return np.concatenate([call, np.zeros(int(gap_s * sr), dtype=np.float32), response]).astype(np.float32)


def build_phee_pairs(
    pairs: Sequence[PheeExchange], sr: int, rng: np.random.Generator
) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """Return {task: (real_sequences, distractor_sequences)} for CallerChange / ReceiverChange.

    The response pool holds every vocalisation in PHEEDB, labelled with (emitter, addressee).
    """
    pool = [(p.response_audio, p.responder_id, p.response_receiver_id) for p in pairs]
    pool += [(p.call_audio, p.caller_id, p.responder_id) for p in pairs]
    out: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {"CallerChange": ([], []), "ReceiverChange": ([], [])}
    for p in pairs:
        real = _dialogue(p.call_audio, p.gap_s, p.response_audio, sr)
        cands = {
            # a different individual (neither caller nor true responder) produces the response
            "CallerChange": [a for a, emitter, _ in pool if emitter not in (p.responder_id, p.caller_id)],
            # same responder, response addressed to someone other than the caller
            "ReceiverChange": [a for a, emitter, to in pool if emitter == p.responder_id and to != p.caller_id],
        }
        for task, c in cands.items():
            if c:
                fake = c[int(rng.integers(len(c)))]
                out[task][0].append(real)
                out[task][1].append(_dialogue(p.call_audio, p.gap_s, fake, sr))
    return out


# --------------------------------------------------------------------------- #
# Benchmark runner
# --------------------------------------------------------------------------- #
class ZeroShotBenchmarks:
    def __init__(self, encoder: BaseAudioEncoder, ulm: UnitLanguageModel, cfg: EvaluationConfig, seed: int = 0) -> None:
        self.encoder, self.ulm, self.cfg = encoder, ulm, cfg
        self.seed = seed

    def _acc(self, real: List[np.ndarray], fake: List[np.ndarray]) -> Dict[str, float]:
        if not real:
            return {"accuracy": float("nan"), "n_pairs": 0}
        r = score_waveforms(self.encoder, self.ulm, real, self.cfg.score_reduction)
        f = score_waveforms(self.encoder, self.ulm, fake, self.cfg.score_reduction)
        return {"accuracy": pairwise_accuracy(r, f), "n_pairs": len(real), "mean_delta_logp": float(np.mean(r - f))}

    def run(self, chunks: Sequence[Chunk], phee_pairs: Optional[Sequence[PheeExchange]] = None) -> Dict[str, Dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        chunks = list(chunks)[: self.cfg.max_eval_chunks]
        wavs = [c.audio() for c in chunks]
        sr = chunks[0].sample_rate if chunks else 16_000
        results: Dict[str, Dict[str, float]] = {}

        real, fake = [], []
        for c, w in zip(chunks, wavs):
            s = shuffle_calls(w, c.calls, sr, rng)
            if s is not None:
                real.append(w)
                fake.append(s)
        results["Shuffle"] = self._acc(real, fake)

        results["Reversal"] = self._acc(wavs, [reverse_audio(w) for w in wavs])

        eligible = [i for i, c in enumerate(chunks) if len(c.calls) >= self.cfg.concat_min_calls]
        real, fake = [], []
        if len(eligible) >= 2:
            for i in eligible:
                j = int(rng.choice([k for k in eligible if k != i]))
                real.append(wavs[i])
                fake.append(concat_halves(wavs[i], wavs[j]))
        results["Concat"] = self._acc(real, fake)

        if phee_pairs:
            for task, (r, f) in build_phee_pairs(phee_pairs, sr, rng).items():
                results[task] = self._acc(r, f)
        return results


def format_results(results: Dict[str, Dict[str, float]]) -> str:
    lines = [f"{'Task':<16}{'Acc (%)':>9}{'Pairs':>8}"]
    for task, r in results.items():
        lines.append(f"{task:<16}{100 * r['accuracy']:>9.2f}{int(r['n_pairs']):>8d}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="GmSLM zero-shot benchmarks (Shuffle, Reversal, Concat, Phee CallerChange/ReceiverChange)."
    )
    parser.add_argument("--checkpoint", help="Agent checkpoint saved by run_pipeline.py (day_XXXX.pt).")
    parser.add_argument("--manifest", help="manifest.tsv written by dataset_builder.export_manifest (eval chunks).")
    parser.add_argument("--mock", action="store_true", help="Use synthetic data (and untrained models if no checkpoint).")
    parser.add_argument("--encoder", choices=["dinosr", "hubert"], default="dinosr", help="Encoder for --mock without checkpoint.")
    parser.add_argument("--output", help="Write results JSON here.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from marmoset_developmental_agent.agent.factory import build_encoder, build_ulm, load_checkpoint
    from marmoset_developmental_agent.config.agent_config import AgentConfig
    from marmoset_developmental_agent.preprocessing.dataset_builder import load_manifest
    from marmoset_developmental_agent.utils.synthetic_data import generate_mock_phee_dialogues
    from marmoset_developmental_agent.utils.mock_pipeline import mock_dataset

    if args.checkpoint:
        cfg, encoder, ulm = load_checkpoint(args.checkpoint)
    elif args.mock:
        cfg = AgentConfig.mock()
        encoder = build_encoder(cfg, args.encoder)
        ulm = build_ulm(cfg, encoder.get_vocab_size())
    else:
        parser.error("provide --checkpoint or --mock")

    if args.manifest:
        chunks = load_manifest(args.manifest).chunks
        phee = None
    elif args.mock:
        ds, _ = mock_dataset(cfg)
        chunks, phee = ds.chunks, generate_mock_phee_dialogues(seed=cfg.seed + 1)
    else:
        parser.error("provide --manifest or --mock")

    if hasattr(encoder, "is_fitted") and not encoder.is_fitted:  # untrained mock HuBERT
        import torch

        encoder.fit(torch.from_numpy(c.audio()).unsqueeze(0) for c in chunks)
    results = ZeroShotBenchmarks(encoder, ulm, cfg.evaluation, cfg.seed).run(chunks, phee)
    print(format_results(results))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
