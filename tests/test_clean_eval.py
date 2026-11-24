"""Shadow filtering (tools/analysis/clean_eval.py): the per-cell gain must reproduce the production cleanings exactly,
and split the cleaned output exactly into cleaned call plus leftover noise."""

import numpy as np

from marmoset_developmental_agent.config.agent_config import CleaningConfig
from marmoset_developmental_agent.preprocessing.clean_calls import clean_call
from tools.analysis import clean_eval as ce
from tools.explorations.ridge_listen import ridge_clean

SR = ce.SR
CALL = (1.0, 1.6)


def _pair(seed: int = 0):
    """A harmonic FM call (fundamental 7 kHz +- 500 Hz, two harmonics) inside 3 s of noise with two clicks."""
    rng = np.random.default_rng(seed)
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    on = (t >= CALL[0]) & (t <= CALL[1])
    phase = 2 * np.pi * np.cumsum(7000 + 500 * np.sin(2 * np.pi * 8 * t)) / SR
    s = np.where(on, 0.02 * np.sin(phase) + 0.008 * np.sin(2 * phase), 0.0)
    b = 0.003 * rng.standard_normal(n)
    for c in (0.5, 1.3):                         # clicks, one inside the call, so declick acts
        i = int(c * SR)
        b[i:i + 40] += 0.2 * rng.standard_normal(40)
    return s, b


def test_gain_reproduces_production():
    cfg = CleaningConfig()
    s, b = _pair()
    y = s + b
    f, t, Y = ce.front(y, cfg)
    for name, setting in (("spectral", ("spectral", None)), ("ridge", ("ridge", (9.0, 500.0, 0.020)))):
        H = ce.cell_gain(Y, f, t, CALL, [CALL], cfg, setting)
        got = ce.inverse(H * Y, len(y), cfg)
        want = clean_call(y, SR, CALL, [CALL], cfg)[0] if name == "spectral" else \
            ridge_clean(y, SR, CALL, [CALL], cfg, peak_db=9.0, min_len_s=0.020)
        assert np.max(np.abs(got - want)) < 1e-6 * max(1.0, float(np.abs(want).max())), name


def test_shadow_split_is_exact():
    cfg = CleaningConfig()
    s, b = _pair(1)
    f, t, S = ce.front(s, cfg)
    _, _, B = ce.front(b, cfg)
    _, _, Y = ce.front(s + b, cfg)
    assert np.allclose(S + B, Y, atol=1e-9)
    H = ce.cell_gain(Y, f, t, CALL, [CALL], cfg, ("spectral", None))
    n = len(s)
    assert np.allclose(ce.inverse(H * S, n, cfg) + ce.inverse(H * B, n, cfg), ce.inverse(H * Y, n, cfg), atol=1e-9)


def test_metrics_identity_and_mute():
    cfg = CleaningConfig()
    s, b = _pair(2)
    keep = {"keep": ("ridge", (-200.0, 500.0, 0.0))}      # every local maximum is a ridge: nearly everything kept
    m = ce.metrics(s, b, CALL, cfg, keep)["keep"]
    assert m["call_kept"] > -3.0
    mute = {"mute": ("ridge", (200.0, 500.0, 0.020))}      # no peak clears 200 dB: the mask is empty
    m = ce.metrics(s, b, CALL, cfg, mute)["mute"]
    assert m["call_kept"] < -200 and m["noise_left"] < -200 and abs(m["call_sdr"]) < 1e-6


def test_scale_to_hits_target():
    cfg = CleaningConfig()
    s, b = _pair(3)
    for target in (16.0, 30.0):
        g, snr = ce.scale_to(s, b, CALL, target, cfg)
        assert abs(snr - target) <= 0.25
        assert g > 0
