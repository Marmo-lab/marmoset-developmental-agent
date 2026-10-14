"""Extended metrics: inter-call interval (ICI) distortion and pitch-shift sensitivity.

Both follow the zero-shot logic of the GmSLM benchmarks: perturb a real segment along
one controlled acoustic dimension and measure how much the uLM's likelihood drops.

ICI distortion
    Every inter-call gap is time-scaled by a factor s, and the calls themselves are left
    untouched. Gap audio is tiled or truncated, not zero-filled, so no artificial digital
    silence is added. The result is trimmed or padded back to the original length.
    Marmoset turn-taking and bout structure depend on ICI timing, so a model that captured
    it should prefer s = 1.

Pitch shift
    Only the call regions are pitch-shifted by n semitones (duration preserved, via
    ``librosa.effects.pitch_shift``). The background stays unchanged. This tests whether
    units and uLM encode the frequency range of each caller or call type.

Per condition we report
    accuracy        P(logp(original) > logp(perturbed))   (chance 0.5)
    delta_surprisal mean over segments of [S(perturbed) - S(original)]  (bits/unit, > 0 = surprised)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from marmoset_developmental_agent.config.agent_config import EvaluationConfig
from marmoset_developmental_agent.decoders.transformer_ulm import UnitLanguageModel
from marmoset_developmental_agent.encoders.base_encoder import BaseAudioEncoder
from marmoset_developmental_agent.evaluation.scoring import LN2, pairwise_accuracy, score_waveforms
from marmoset_developmental_agent.preprocessing.segmentation import CallSegment, Chunk


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Tile / truncate ``x`` to length ``n`` (empty -> zeros)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.resize(x, n).astype(np.float32)


def rescale_intercall_intervals(
    wav: np.ndarray, calls: Sequence[CallSegment], factor: float, sr: int
) -> Optional[np.ndarray]:
    """Scale every gap *between* calls by ``factor``. None if fewer than two calls."""
    spans = sorted((int(c.start_s * sr), int(c.end_s * sr)) for c in calls)
    if len(spans) < 2:
        return None
    pieces: List[np.ndarray] = [wav[: spans[0][0]]]
    for i, (s, e) in enumerate(spans):
        pieces.append(wav[s:e])
        if i + 1 < len(spans):
            gap = wav[e : spans[i + 1][0]]
            pieces.append(_fit(gap, int(round(len(gap) * factor))))
    tail = wav[spans[-1][1] :]
    pieces.append(tail)
    out = np.concatenate(pieces)
    if len(out) >= len(wav):
        return out[: len(wav)].astype(np.float32)
    # Pad with tiled post-call background so no calls are duplicated.
    filler = tail if len(tail) else wav[spans[0][1] : spans[1][0]]
    return np.concatenate([out, _fit(filler, len(wav) - len(out))]).astype(np.float32)


def pitch_shift_calls(wav: np.ndarray, calls: Sequence[CallSegment], semitones: float, sr: int) -> np.ndarray:
    """Shift only the call regions by ``semitones`` (length preserved)."""
    import librosa

    out = wav.copy()
    for c in calls:
        s, e = int(c.start_s * sr), min(len(wav), int(c.end_s * sr))
        if e - s < 512:
            continue
        out[s:e] = librosa.effects.pitch_shift(wav[s:e], sr=sr, n_steps=semitones)
    return out


def _condition(
    encoder: BaseAudioEncoder, ulm: UnitLanguageModel, orig: List[np.ndarray], pert: List[np.ndarray]
) -> Dict[str, float]:
    if not orig:
        return {"accuracy": float("nan"), "delta_surprisal_bits": float("nan"), "n": 0}
    lo = score_waveforms(encoder, ulm, orig, "mean")
    lp = score_waveforms(encoder, ulm, pert, "mean")
    return {
        "accuracy": pairwise_accuracy(lo, lp),
        "delta_surprisal_bits": float(np.mean(lo - lp) / LN2),
        "n": len(orig),
    }


def ici_distortion_sensitivity(
    encoder: BaseAudioEncoder, ulm: UnitLanguageModel, chunks: Sequence[Chunk], cfg: EvaluationConfig
) -> Dict[str, Dict[str, float]]:
    chunks = list(chunks)[: cfg.max_eval_chunks]
    out: Dict[str, Dict[str, float]] = {}
    for f in cfg.ici_scale_factors:
        orig, pert = [], []
        for c in chunks:
            p = rescale_intercall_intervals(c.audio(), c.calls, f, c.sample_rate)
            if p is not None:
                orig.append(c.audio())
                pert.append(p)
        out[f"x{f:g}"] = _condition(encoder, ulm, orig, pert)
    return out


def pitch_shift_sensitivity(
    encoder: BaseAudioEncoder, ulm: UnitLanguageModel, chunks: Sequence[Chunk], cfg: EvaluationConfig
) -> Dict[str, Dict[str, float]]:
    chunks = [c for c in list(chunks)[: cfg.max_eval_chunks] if c.calls]
    out: Dict[str, Dict[str, float]] = {}
    for st in cfg.pitch_shift_semitones:
        pert = [pitch_shift_calls(c.audio(), c.calls, st, c.sample_rate) for c in chunks]
        out[f"{st:+g}st"] = _condition(encoder, ulm, [c.audio() for c in chunks], pert)
    return out
