"""The learned call detector: features, model contract, and the pipeline path end to end."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import signal

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.preprocessing.filter_5khz import compute_frame_activity, preprocess_waveform
from marmoset_developmental_agent.preprocessing.learned_detector import (
    CallNet, band_edges, learned_features, load_call_detector, save_call_detector)
from marmoset_developmental_agent.preprocessing.segmentation import activity_to_calls

SR = 48_000


def _clip(seconds: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = 1e-3 * rng.standard_normal(int(seconds * SR))
    n = int(0.6 * SR)
    x[SR:SR + n] += 0.05 * signal.chirp(np.arange(n) / SR, 9_000, 0.6, 7_000)
    return x.astype(np.float32)


@pytest.fixture
def cfg(tmp_path: Path) -> PreprocessingConfig:
    """The learned preset pointing at a freshly initialised (untrained) checkpoint."""
    c = PreprocessingConfig.learned()
    torch.manual_seed(0)
    path = tmp_path / "net.pt"
    save_call_detector(CallNet(len(band_edges(c)) - 1).eval(), path)
    return dataclasses.replace(c, learned_model_path=str(path))


def test_features_are_gain_invariant_and_on_a_10_ms_grid(cfg: PreprocessingConfig) -> None:
    """Each band is in dB above its own median, so recorder gain cancels exactly. The STFT
    hop is 480 samples at 48 kHz, i.e. 10 ms per frame."""
    x = _clip()
    a, t = learned_features(x, SR, cfg)
    b, _ = learned_features(10 * x, SR, cfg)
    assert a.shape[0] == len(band_edges(cfg)) - 1 == 92  # 1-24 kHz in 250 Hz bands
    np.testing.assert_allclose(a, b, atol=1e-3)
    assert np.allclose(np.diff(t), 0.01)


def test_model_gives_one_probability_per_frame_and_head(cfg: PreprocessingConfig) -> None:
    feats, t = learned_features(_clip(), SR, cfg)
    net = load_call_detector(cfg.learned_model_path)
    with torch.no_grad():
        assert net(torch.from_numpy(feats)[None]).shape == (1, 1, len(t))
        assert CallNet(feats.shape[0], n_out=3).eval()(torch.from_numpy(feats)[None]).shape == (1, 3, len(t))


def test_edge_heads_split_two_calls_and_move_edges_to_the_peaks(cfg: PreprocessingConfig) -> None:
    """With onset / offset probabilities, one run of active frames that holds an offset peak
    followed by an onset peak becomes two calls, and each edge moves to its head's peak."""
    from marmoset_developmental_agent.preprocessing.filter_5khz import FrameActivity

    n = 300
    t = np.arange(n) * 0.01
    active = np.zeros(n, bool)
    active[50:200] = True  # one run, 1.5 s
    on, off = np.zeros(n), np.zeros(n)
    on[[47, 130]] = 0.9  # true starts: 3 frames before the run, and mid-run
    off[[110, 203]] = 0.9  # an end mid-run, and 3 frames after the run
    act = FrameActivity(frame_times_s=t, active=active, hop_s=0.01, onset_prob=on, offset_prob=off)
    calls = activity_to_calls(act, dataclasses.replace(cfg, boundary_pad_s=0.0))
    assert [(round(c.start_s, 2), round(c.end_s, 2)) for c in calls] == [(0.47, 1.10), (1.30, 2.03)]
    unsplit = activity_to_calls(act, dataclasses.replace(cfg, boundary_pad_s=0.0, learned_split_prob=None))
    assert [(round(c.start_s, 2), round(c.end_s, 2)) for c in unsplit] == [(0.47, 2.03)]


def test_learned_path_runs_through_the_shared_pipeline(cfg: PreprocessingConfig) -> None:
    """compute_frame_activity dispatches to the classifier; activity_to_calls groups its frames
    with the preset's limits and applies no tonal rejection; preprocess_waveform works."""
    act = compute_frame_activity(_clip(), SR, cfg)
    assert act.call_prob is not None and act.call_prob.shape == act.active.shape
    assert ((act.call_prob >= 0) & (act.call_prob <= 1)).all()
    assert act.hop_s == pytest.approx(0.01)
    act.active[:] = False
    act.active[100:160] = True  # 0.6 s of frames the classifier calls active
    act.tonal[:] = False  # would be rejected by the spectral path's tonal rule
    calls = activity_to_calls(act, cfg)
    assert len(calls) == 1 and calls[0].duration_s == pytest.approx(0.59 + 2 * cfg.boundary_pad_s, abs=0.01)
    training, activity = preprocess_waveform(_clip(), SR, cfg)
    assert len(training) == 6 * cfg.target_sample_rate and activity.call_prob is not None


def test_learned_features_refuse_the_wrong_rate_or_shape(cfg: PreprocessingConfig) -> None:
    with pytest.raises(ValueError, match="detection rate"):
        learned_features(_clip(), 22_050, cfg)
    with pytest.raises(ValueError, match="one signal"):
        learned_features(_clip()[None], SR, cfg)


def test_learned_yamls_are_the_learned_preset() -> None:
    from marmoset_developmental_agent.config.agent_config import load_config

    root = Path(__file__).resolve().parents[1] / "config"
    assert load_config(root / "learned.yaml").preprocessing == PreprocessingConfig.learned()
    day = dataclasses.replace(PreprocessingConfig.learned(), exclude_night=True)
    assert load_config(root / "corpus_day_learned.yaml").preprocessing == day


def test_shipped_checkpoint_matches_the_preset() -> None:
    """The committed model loads under the preset's feature settings."""
    cfg = PreprocessingConfig.learned()
    path = Path(__file__).resolve().parents[1] / cfg.learned_model_path
    if not path.exists():
        pytest.skip("no shipped checkpoint yet")
    assert load_call_detector(cfg.learned_model_path).n_bands == len(band_edges(cfg)) - 1
