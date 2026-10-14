"""Synthetic marmoset colony audio for mock / CI runs.

The generator is *not* a biological model; it produces signals that exercise
every code path with known ground truth:

* Four call types with caricatured spectro-temporal shapes, all with energy in
  the 5-8 kHz marmoset band (GmSLM A.7):
  ``phee`` (0.8-2.0 s tonal whistle), ``trill`` (0.3-0.8 s sinusoidal FM),
  ``twitter`` (bout of 5-9 short upward sweeps), ``tsik`` (0.25-0.35 s down-sweep).
* Caller identity shifts the fundamental; phee calls additionally carry a
  *receiver*-specific FM signature (a toy version of vocal labelling, Oren et al. 2024).
* Call order follows a first-order Markov chain, so the Shuffle / Concat /
  Reversal benchmarks have real structure to detect.
* Low-frequency "human speech" harmonics, mains hum, white noise and 5 ms
  broadband clicks are mixed in so the 5 kHz high-pass and the duration-based
  noise gate have something to remove.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CALL_TYPES: Tuple[str, ...] = ("phee", "trill", "twitter", "tsik")

# Rows: previous call type, columns: next call type (order = CALL_TYPES).
CALL_TRANSITIONS: np.ndarray = np.array(
    [
        [0.60, 0.15, 0.10, 0.15],  # phee -> mostly phee (antiphonal exchanges)
        [0.10, 0.50, 0.30, 0.10],  # trill
        [0.05, 0.40, 0.45, 0.10],  # twitter
        [0.30, 0.10, 0.10, 0.50],  # tsik
    ]
)


@dataclass
class CallAnnotation:
    """Ground-truth call event inside a recording (seconds from recording start)."""

    start_s: float
    end_s: float
    call_type: str
    caller_id: int
    receiver_id: Optional[int] = None


@dataclass
class SyntheticRecording:
    audio: np.ndarray  # float32, mono
    sample_rate: int
    day: int  # calendar recording day (1-indexed)
    start_time: dt.datetime
    recording_id: str
    calls: List[CallAnnotation] = field(default_factory=list)


@dataclass
class PheeExchange:
    """One call-and-response pair (PHEEDB-style, GmSLM Sec. 3.1 / 3.3)."""

    call_audio: np.ndarray
    response_audio: np.ndarray
    gap_s: float
    caller_id: int
    responder_id: int  # who produces the response
    response_receiver_id: int  # whom the response is addressed to (== caller_id for real pairs)


# --------------------------------------------------------------------------- #
# Call synthesis
# --------------------------------------------------------------------------- #
def _tone(freq_hz: np.ndarray, sr: int) -> np.ndarray:
    """Sine with instantaneous frequency ``freq_hz`` (per sample)."""
    phase = 2 * np.pi * np.cumsum(freq_hz) / sr
    return np.sin(phase)


def _envelope(n: int, attack: float = 0.1) -> np.ndarray:
    a = max(1, int(n * attack))
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a)
    env[-a:] = np.linspace(1, 0, a)
    return env


def synth_call(
    call_type: str,
    sr: int,
    rng: np.random.Generator,
    caller_id: int = 0,
    receiver_id: Optional[int] = None,
) -> np.ndarray:
    """Render one call. Caller shifts f0 by 150 Hz steps; receiver sets phee FM rate."""
    f_shift = 150.0 * (caller_id % 6)
    if call_type == "phee":
        dur = rng.uniform(0.8, 2.0)
        t = np.arange(int(dur * sr)) / sr
        fm_rate = 3.0 + 2.0 * (receiver_id if receiver_id is not None else 0)
        freq = 6_500 + f_shift + 150 * np.sin(2 * np.pi * fm_rate * t) + 300 * t / dur
        sig = _tone(freq, sr) + 0.3 * _tone(2 * freq, sr) * (2 * freq < sr / 2)
    elif call_type == "trill":
        dur = rng.uniform(0.3, 0.8)
        t = np.arange(int(dur * sr)) / sr
        freq = 7_000 + f_shift + 400 * np.sin(2 * np.pi * 28 * t)
        sig = _tone(freq, sr)
    elif call_type == "twitter":
        n_syll = rng.integers(5, 10)
        parts: List[np.ndarray] = []
        for _ in range(n_syll):
            sd = rng.uniform(0.03, 0.05)
            ts = np.arange(int(sd * sr)) / sr
            freq = 5_500 + f_shift + (2_000 / sd) * ts
            parts.append(_tone(freq, sr) * _envelope(len(ts), 0.2))
            parts.append(np.zeros(int(rng.uniform(0.04, 0.07) * sr)))
        sig = np.concatenate(parts)
    elif call_type == "tsik":
        dur = rng.uniform(0.25, 0.35)
        t = np.arange(int(dur * sr)) / sr
        freq = 7_400 + f_shift - 2_000 * t / dur
        sig = _tone(freq, sr)
    else:
        raise ValueError(f"Unknown call type {call_type!r}")
    return (sig * _envelope(len(sig)) * rng.uniform(0.3, 0.6)).astype(np.float32)


def _background(n: int, sr: int, rng: np.random.Generator, human_speech: bool) -> np.ndarray:
    t = np.arange(n) / sr
    bg = 0.003 * rng.standard_normal(n)
    bg += 0.02 * np.sin(2 * np.pi * 50 * t)  # mains hum
    if human_speech:
        # Harmonic "voice" (f0 ~ 120-220 Hz) gated in syllable-like bursts.
        f0 = 170 + 50 * np.sin(2 * np.pi * 0.3 * t)
        voice = sum(_tone(k * f0, sr) / k for k in range(1, 12))
        gate = (np.sin(2 * np.pi * 3.5 * t) > 0.2) * (np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6)) > 0)
        bg += 0.05 * voice * gate
    for _ in range(int(n / sr / 4)):  # broadband clicks (door, cage) ~ every 4 s
        i = rng.integers(0, max(1, n - int(0.005 * sr)))
        bg[i : i + int(0.005 * sr)] += 0.3 * rng.standard_normal(int(0.005 * sr))
    return bg.astype(np.float32)


# --------------------------------------------------------------------------- #
# Colony recordings
# --------------------------------------------------------------------------- #
def generate_recording(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    day: int,
    start_time: dt.datetime,
    n_individuals: int = 4,
    human_speech: bool = True,
) -> SyntheticRecording:
    """One continuous colony recording with Markov-ordered calls and background noise."""
    n = int(duration_s * sr)
    audio = _background(n, sr, rng, human_speech)
    calls: List[CallAnnotation] = []
    t = rng.uniform(0.3, 1.0)
    prev = int(rng.integers(len(CALL_TYPES)))
    while True:
        ctype_idx = int(rng.choice(len(CALL_TYPES), p=CALL_TRANSITIONS[prev]))
        ctype = CALL_TYPES[ctype_idx]
        caller = int(rng.integers(n_individuals))
        receiver = int((caller + rng.integers(1, n_individuals)) % n_individuals) if ctype == "phee" else None
        sig = synth_call(ctype, sr, rng, caller, receiver)
        s = int(t * sr)
        if s + len(sig) >= n:
            break
        audio[s : s + len(sig)] += sig
        calls.append(CallAnnotation(t, t + len(sig) / sr, ctype, caller, receiver))
        prev = ctype_idx
        # Inter-call interval: short within bouts, occasionally a long pause.
        t += len(sig) / sr + (rng.uniform(0.3, 1.2) if rng.random() < 0.8 else rng.uniform(2.0, 5.0))
    return SyntheticRecording(
        audio=np.clip(audio, -1, 1).astype(np.float32),
        sample_rate=sr,
        day=day,
        start_time=start_time,
        recording_id=f"day{day:03d}_{start_time:%H%M%S}",
        calls=calls,
    )


def generate_mock_colony(
    n_days: int = 3,
    seconds_per_day: float = 60.0,
    recordings_per_day: int = 2,
    native_sample_rate: int = 48_000,
    n_individuals: int = 4,
    seed: int = 0,
    shuffle_order: bool = True,
) -> List[SyntheticRecording]:
    """Multi-day colony recordings, returned in *shuffled* order by default so that
    the chronological dataset builder has real sorting work to do."""
    rng = np.random.default_rng(seed)
    t0 = dt.datetime(2025, 1, 1, 8, 0, 0)
    recs: List[SyntheticRecording] = []
    for day in range(1, n_days + 1):
        for r in range(recordings_per_day):
            start = t0 + dt.timedelta(days=day - 1, hours=2 * r)
            recs.append(
                generate_recording(
                    seconds_per_day / recordings_per_day, native_sample_rate, rng, day, start, n_individuals
                )
            )
    if shuffle_order:
        rng.shuffle(recs)  # type: ignore[arg-type]
    return recs


def generate_mock_phee_dialogues(
    n_individuals: int = 4,
    n_pairs: int = 24,
    sr: int = 16_000,
    seed: int = 1,
) -> List[PheeExchange]:
    """Labelled phee call-response pairs (real pairs: response addressed to the caller)."""
    rng = np.random.default_rng(seed)
    pairs: List[PheeExchange] = []
    for _ in range(n_pairs):
        caller = int(rng.integers(n_individuals))
        responder = int((caller + rng.integers(1, n_individuals)) % n_individuals)
        call = synth_call("phee", sr, rng, caller, receiver_id=responder)
        resp = synth_call("phee", sr, rng, responder, receiver_id=caller)
        pairs.append(PheeExchange(call, resp, float(rng.uniform(0.5, 2.0)), caller, responder, caller))
    return pairs


def mock_daily_metadata(recordings: Sequence[SyntheticRecording]) -> List[Dict[str, object]]:
    """Flat metadata rows (one per recording), as a real manifest CSV would provide."""
    return [
        {
            "recording_id": r.recording_id,
            "day": r.day,
            "start_time": r.start_time.isoformat(),
            "duration_s": len(r.audio) / r.sample_rate,
            "sample_rate": r.sample_rate,
            "n_calls": len(r.calls),
        }
        for r in recordings
    ]
