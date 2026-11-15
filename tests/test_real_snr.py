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
