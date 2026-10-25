"""Spectral cleaning of calls (preprocessing/clean_calls.py) on a synthetic call with noise and clicks."""

from __future__ import annotations

import numpy as np
from scipy import signal

from marmoset_developmental_agent.config.agent_config import CleaningConfig
from marmoset_developmental_agent.preprocessing.clean_calls import calls_in_window, clean_call

SR = 48_000
CALL = (1.5, 2.1)


def _scene(seed: int = 0):
    """4 s of broadband noise, a 9 -> 7 kHz sweep at CALL, and 3 ms broadband clicks inside
    and outside the call."""
    rng = np.random.default_rng(seed)
    n = 4 * SR
    noise = 0.01 * rng.standard_normal(n)
    sweep = np.zeros(n)
    m = int((CALL[1] - CALL[0]) * SR)
    sweep[int(CALL[0] * SR):int(CALL[0] * SR) + m] = 0.05 * signal.chirp(np.arange(m) / SR, 9_000, m / SR, 7_000)
    clicks = np.zeros(n)
    for t0 in (0.7, 1.8, 3.2):
        k = int(t0 * SR)
        clicks[k:k + 144] = 0.3 * rng.standard_normal(144)
    return noise + sweep + clicks, sweep


def _in_line_share(x: np.ndarray) -> float:
    """Share of the energy inside CALL that lies within 400 Hz of the sweep's frequency."""
    f, t, X = signal.stft(x, SR, nperseg=1024, noverlap=1024 - 240)
    sel = (t >= CALL[0] + 0.02) & (t <= CALL[1] - 0.02)
    inst = 9_000 - 2_000 * (t[sel] - CALL[0]) / (CALL[1] - CALL[0])
    P = np.abs(X[:, sel]) ** 2
    near = np.abs(f[:, None] - inst[None, :]) <= 400
    return float(P[near].sum() / P.sum())


def test_cleaning_keeps_the_call_and_removes_noise_and_clicks():
    x, sweep = _scene()
    y, info = clean_call(x, SR, CALL, [CALL], CleaningConfig())
    assert y.shape == x.shape
    assert _in_line_share(y) > max(0.9, _in_line_share(x) + 0.15)  # energy concentrated on the call line
    inside = slice(int(CALL[0] * SR), int(CALL[1] * SR))
    assert np.corrcoef(y[inside], sweep[inside])[0, 1] > 0.9  # the call itself is preserved
    outside = np.r_[0:int((CALL[0] - 0.05) * SR), int((CALL[1] + 0.05) * SR):len(y)]
    assert np.abs(y[outside]).max() < 1e-3  # silent outside the call, clicks included
    assert info["click_frames_in_call"] > 0 and info["noise_s"] > 2.0


def test_noise_profile_ignores_other_calls_and_window_helper():
    x, _ = _scene(1)
    # a second, louder "call" elsewhere must not enter the noise profile
    x[int(3.0 * SR):int(3.5 * SR)] += 0.2 * np.sin(2 * np.pi * 8_000 * np.arange(int(0.5 * SR)) / SR)
    y, info = clean_call(x, SR, CALL, [CALL, (3.0, 3.5)], CleaningConfig())
    assert info["noise_s"] < 3.0 and _in_line_share(y) > 0.9
    assert calls_in_window([(1.0, 2.0), (5.0, 6.0), (9.5, 10.5)], 4.0, 10.0) == [(1.0, 2.0), (5.5, 6.5)]
