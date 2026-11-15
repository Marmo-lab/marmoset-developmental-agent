"""real_snr: a call planted at a known SNR (the generator's definition) is measured back at that SNR."""

import numpy as np
import pytest

from tools.analysis.real_snr import SR, call_snr, nearest_level
from tools.synth.generate import bandpass, rms


@pytest.mark.parametrize("snr", [10.0, 2.0, 1.0])
def test_planted_call_measured_back(snr: float) -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(0, 0.01, 20 * SR)
    t = np.arange(int(0.4 * SR)) / SR
    s = np.sin(2 * np.pi * (7000 * t + 4000 * t ** 2))
    i = 10 * SR
    g = snr * rms(bandpass(b)[i:i + len(s)]) / rms(bandpass(s))
    x = b.copy()
    x[i:i + len(s)] += g * s
    r = call_snr(bandpass(x), [(10.0, 10.4), (4.0, 4.5)], 0)
    assert r["bg_scope"] == "local"
    assert r["snr"] == pytest.approx(snr, rel=0.1)


def test_call_below_background_is_zero_and_level_mapping() -> None:
    rng = np.random.default_rng(1)
    bp = bandpass(rng.normal(0, 0.01, 10 * SR))
    bp[5 * SR:int(5.3 * SR)] *= 0.5
    assert call_snr(bp, [(5.0, 5.3)], 0)["snr"] == 0.0
    assert nearest_level(25.0) == 10.0 and nearest_level(3.0) == 1.5 and nearest_level(float("-inf")) == 0.7


def test_floor_ignores_a_transient_beside_the_call() -> None:
    rng = np.random.default_rng(2)
    b = rng.normal(0, 0.01, 20 * SR)
    t = np.arange(int(0.4 * SR)) / SR
    s = np.sin(2 * np.pi * 8000 * t)
    i = 10 * SR
    g = 2.0 * rms(bandpass(b)[i:i + len(s)]) / rms(bandpass(s))
    x = b.copy()
    x[i:i + len(s)] += g * s
    x[int(11 * SR):int(11.2 * SR)] += rng.normal(0, 0.2, int(0.2 * SR))      # a loud bang 0.6 s after the call
    r = call_snr(bandpass(x), [(10.0, 10.4)], 0)
    assert r["snr_db"] < 0.0
    assert 10 ** (r["snr_floor_db"] / 20) == pytest.approx(2.0, rel=0.15)
