"""Sanity checks: shapes, no-dedup, replay maths, schedules, preprocessing, benchmarks, smoke run."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import numpy as np
import pytest
import torch
from scipy import signal

from marmoset_developmental_agent.agent.lr_scheduler import SigmoidLRScheduler, sigmoid_gate
from marmoset_developmental_agent.agent.replay_buffer import SleepReplayBuffer
from marmoset_developmental_agent.config.agent_config import (
    AgentConfig,
    ExposureScheduleConfig,
    PreprocessingConfig,
    ReplayConfig,
    SigmoidScheduleConfig,
    load_config,
    save_config,
)
from marmoset_developmental_agent.decoders.transformer_ulm import UnitLanguageModel
from marmoset_developmental_agent.encoders.base_encoder import check_no_dedup, conv_output_length, run_lengths
from marmoset_developmental_agent.encoders.dino_sr import DinoSREncoder
from marmoset_developmental_agent.encoders.marmoset_hubert import MarmosetHuBERTEncoder
from marmoset_developmental_agent.evaluation.developmental_eval import call_is_learned, smooth_probs
from marmoset_developmental_agent.evaluation.new_metrics import rescale_intercall_intervals
from marmoset_developmental_agent.evaluation.zero_shot_benchmarks import concat_halves, reverse_audio, shuffle_calls
from marmoset_developmental_agent.preprocessing.dataset_builder import DailyExposureSchedule
from marmoset_developmental_agent.preprocessing.filter_5khz import highpass_filter, preprocess_waveform
from marmoset_developmental_agent.preprocessing.segmentation import CallSegment, activity_to_calls
from marmoset_developmental_agent.utils.mock_pipeline import mock_dataset
from marmoset_developmental_agent.vocoder.hifigan_u2v import HiFiGANU2V

SR = 16_000
CHUNK = 10 * SR


@pytest.fixture(scope="module")
def cfg() -> AgentConfig:
    return AgentConfig.mock()


@pytest.fixture(scope="module")
def dataset(cfg):
    ds, info = mock_dataset(cfg, n_calendar_days=2, seconds_per_day=40)
    return ds


@pytest.fixture(scope="module")
def hubert(cfg, dataset):
    enc = MarmosetHuBERTEncoder(cfg.encoder.hubert)
    enc.fit(torch.from_numpy(c.audio()).unsqueeze(0) for c in dataset.chunks[:4])
    return enc


# --------------------------------------------------------------------------- #
# Encoders: shapes and the no-dedup rule
# --------------------------------------------------------------------------- #
def test_conv_frame_count():
    assert conv_output_length(CHUNK) == 499  # 10 s @ 16 kHz, 20 ms frames


def test_hubert_shapes_and_repeats_preserved(hubert):
    wav = torch.zeros(2, CHUNK)
    wav[:, : SR // 2] = 0.1 * torch.randn(SR // 2)
    toks = hubert.encode(wav)
    assert toks.shape == (2, 499) and toks.dtype == torch.long
    assert int(toks.max()) < hubert.get_vocab_size() == 50
    check_no_dedup(toks, CHUNK, hubert)
    # Constant (silent) tail -> identical frames -> long runs must survive.
    assert run_lengths(toks[0])[1].max() > 10
    assert hubert.frame_rate == 50.0


def test_check_no_dedup_rejects_collapsed(hubert):
    toks = hubert.encode(torch.zeros(1, CHUNK))
    collapsed = run_lengths(toks[0])[0].unsqueeze(0)
    with pytest.raises(AssertionError):
        check_no_dedup(collapsed, CHUNK, hubert)


def test_dinosr_training_step_and_encode(cfg):
    enc = DinoSREncoder(cfg.encoder.dinosr)
    wav = 0.1 * torch.randn(2, CHUNK)
    loss = enc.training_step(wav)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert enc.heads[0].weight.grad is not None
    assert all(p.grad is None for p in enc.teacher.parameters())
    before = [p.clone() for p in enc.teacher.parameters()]
    torch.optim.SGD(enc.trainable_parameters(), lr=0.1).step()
    enc.after_optimizer_step()
    assert any(not torch.equal(a, b) for a, b in zip(before, enc.teacher.parameters()))
    toks = enc.encode(wav)
    assert toks.shape == (2, 499) and int(toks.max()) < 512
    check_no_dedup(toks, CHUNK, enc)


def test_dinosr_rejects_layer_without_codebook(cfg):
    bad = type(cfg.encoder.dinosr)(**{**cfg.encoder.dinosr.__dict__, "layer_for_tokens": 0})
    with pytest.raises(ValueError):
        DinoSREncoder(bad)


# --------------------------------------------------------------------------- #
# uLM
# --------------------------------------------------------------------------- #
def test_ulm_loss_shapes_and_causality(cfg):
    torch.manual_seed(0)
    lm = UnitLanguageModel(50, cfg.decoder).eval()
    units = torch.randint(0, 50, (3, 40))
    assert torch.isfinite(lm.loss(units))
    lp, valid = lm.token_log_probs(units)
    assert lp.shape == (3, 40) and valid.all()
    changed = units.clone()
    changed[:, 30] = (changed[:, 30] + 1) % 50
    lp2, _ = lm.token_log_probs(changed)
    assert torch.allclose(lp[:, :30], lp2[:, :30], atol=1e-5)  # past unaffected by future
    assert not torch.allclose(lp[:, 31:], lp2[:, 31:])


def test_ulm_respects_lengths(cfg):
    lm = UnitLanguageModel(50, cfg.decoder).eval()
    units = torch.randint(0, 50, (2, 20))
    lp, valid = lm.token_log_probs(units, torch.tensor([20, 12]))
    assert valid[1, :12].all() and not valid[1, 12:].any()
    solo, _ = lm.token_log_probs(units[1:, :12])
    assert torch.allclose(lp[1, :12], solo[0], atol=1e-5)


# --------------------------------------------------------------------------- #
# Replay buffer & schedules
# --------------------------------------------------------------------------- #
def _buffer(kernel: str, size_weighted: bool, lam: float = 0.5) -> SleepReplayBuffer:
    buf = SleepReplayBuffer(ReplayConfig(passes_per_day=25, kernel=kernel, decay_lambda=lam, size_weighted=size_weighted))
    start = 0
    for d, n in enumerate([2, 4, 6, 8]):
        buf.add_day(d, np.arange(start, start + n))
        start += n
    return buf


def test_replay_exponential_probabilities():
    days, p = _buffer("exponential", False, lam=0.5).day_probabilities(3)
    w = np.exp(-0.5 * (3 - days))
    assert np.allclose(p, w / w.sum()) and math.isclose(p.sum(), 1.0)
    assert np.all(np.diff(p) > 0)  # more recent days more likely


def test_replay_inverse_recency_matches_official():
    days, p = _buffer("inverse_recency", True).day_probabilities(3)
    sizes = np.array([2, 4, 6, 8])
    official = 1 / (np.arange(4)[::-1] + 1) * sizes  # past_sampling_dataset.py
    assert np.allclose(p, official / official.sum())


def test_replay_day_schedule_counts_and_sources():
    buf = SleepReplayBuffer(ReplayConfig(passes_per_day=25, kernel="exponential", decay_lambda=0.3), seed=1)
    buf.add_day(0, np.arange(0, 5))
    buf.add_day(1, np.arange(5, 12))
    today = np.arange(12, 16)
    sched = buf.build_day_schedule(2, today)
    assert np.array_equal(sched.wake, today)
    assert len(sched.sleep) == 24 * len(today) and len(sched.order) == 25 * len(today)
    assert sched.sleep.min() >= 0 and sched.sleep.max() < 16  # only days <= D
    counts = np.bincount([0 if i < 5 else 1 if i < 12 else 2 for i in sched.sleep], minlength=3)
    assert counts[2] > counts[0]  # recency bias


def test_replay_exclude_current_day():
    buf = SleepReplayBuffer(ReplayConfig(include_current_day=False))
    buf.add_day(0, np.arange(3))
    sched = buf.build_day_schedule(1, np.arange(3, 6))
    assert sched.sleep.max() < 3


def test_sigmoid_lr_schedule():
    c = SigmoidScheduleConfig(lr_max=1e-3, t_p=100, k=0.1, floor=1e-6)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = SigmoidLRScheduler(opt, c)
    assert math.isclose(sch.step(100), 1e-3 * (0.5 + 1e-6))
    assert opt.param_groups[0]["lr"] == sch.get_last_lr()
    lrs = [sch.lr_at(t) for t in range(0, 300, 10)]
    assert all(b >= a for a, b in zip(lrs, lrs[1:]))
    assert sch.lr_at(0) < 1e-6 and sch.lr_at(300) > 0.99e-3
    assert math.isfinite(sigmoid_gate(-1e9, 0, 1)) and math.isfinite(sigmoid_gate(1e9, 0, 1))


def test_exposure_schedule_day_sizes():
    sched = DailyExposureSchedule(ExposureScheduleConfig([2, 4], days_per_period=2), seconds_per_sample=10.0)
    assert sched.samples_for_day(0) == 720 and sched.samples_for_day(2) == 1440
    assert sched.samples_for_day(99) == 1440  # final value repeats
    assert sched.day_sizes(720 * 2 + 100) == [720, 720, 100]


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_highpass_5khz():
    sr = 48_000
    t = np.arange(sr) / sr
    low = highpass_filter(np.sin(2 * np.pi * 1_000 * t).astype(np.float32), sr)
    high = highpass_filter(np.sin(2 * np.pi * 7_000 * t).astype(np.float32), sr)
    rms = lambda x: float(np.sqrt(np.mean(x[sr // 10 : -sr // 10] ** 2)))  # noqa: E731
    assert 20 * np.log10(rms(low) / (1 / np.sqrt(2))) < -60
    assert abs(20 * np.log10(rms(high) / (1 / np.sqrt(2)))) < 0.5


def test_dataset_chronological_and_chunk_shapes(dataset, cfg):
    starts = [c.start_time for c in dataset.chunks]
    assert starts == sorted(starts)
    assert sum(dataset.day_sizes) == len(dataset)
    for c in dataset.chunks:
        assert c.audio().shape == (CHUNK,) and c.sample_rate == SR
        for call in c.calls:
            assert cfg.preprocessing.min_call_duration_s <= call.duration_s <= cfg.preprocessing.max_call_duration_s
            assert 0 <= call.start_s < call.end_s <= 10.0


# --------------------------------------------------------------------------- #
# Evaluation perturbations
# --------------------------------------------------------------------------- #
def _toy_chunk():
    wav = np.zeros(CHUNK, dtype=np.float32)
    calls = [CallSegment(1.0, 1.5), CallSegment(3.0, 4.0), CallSegment(6.0, 6.3)]
    for i, c in enumerate(calls):
        wav[int(c.start_s * SR) : int(c.end_s * SR)] = i + 1
    return wav, calls


def test_zero_shot_perturbations():
    wav, calls = _toy_chunk()
    s = shuffle_calls(wav, calls, SR, np.random.default_rng(0))
    assert s is not None and len(s) == len(wav) and not np.array_equal(s, wav)
    assert np.allclose(np.sort(s), np.sort(wav))  # same material, new order
    assert np.array_equal(reverse_audio(reverse_audio(wav)), wav)
    other = -np.ones_like(wav)
    c = concat_halves(wav, other)
    assert len(c) == len(wav) and np.array_equal(c[: len(wav) // 2], wav[: len(wav) // 2]) and (c[len(wav) // 2 :] == -1).all()
    assert shuffle_calls(wav, calls[:1], SR, np.random.default_rng(0)) is None


def test_ici_rescale():
    wav, calls = _toy_chunk()
    assert np.array_equal(rescale_intercall_intervals(wav, calls, 1.0, SR), wav)
    stretched = rescale_intercall_intervals(wav, calls, 2.0, SR)
    assert len(stretched) == len(wav)
    first_call2 = np.flatnonzero(stretched == 2)[0] / SR
    assert math.isclose(first_call2, 1.5 + 2 * 1.5, abs_tol=1e-3)  # gap 1.5 s -> 3 s


def test_lexicon_learning_rule():
    assert call_is_learned(np.full(20, 0.6), 0.05, 1.0, -0.25, 0.9)
    dip = np.full(20, 0.6)
    dip[10] = 0.01
    assert not call_is_learned(dip, 0.05, 1.0, -0.25, 0.9)
    m = smooth_probs(np.array([1.0, 0.0, 0.0]), 0.5)
    assert math.isclose(m[0], 1.0) and m[1] < m[0] and m[2] < m[1]


def test_vocoder_output_length(cfg):
    units = torch.randint(0, 50, (2, 25))
    for mode in ("additive", "neural"):
        vcfg = type(cfg.vocoder)(**{**cfg.vocoder.__dict__, "mock_mode": mode})
        voc = HiFiGANU2V(50, vcfg)
        assert voc.synthesize(units).shape == (2, 25 * 320)


def test_config_yaml_roundtrip(tmp_path, cfg):
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    again = load_config(p)
    assert again.to_dict() == cfg.to_dict()
    p.write_text("training:\n  not_a_key: 1\n")
    with pytest.raises(KeyError):
        load_config(p)


# --------------------------------------------------------------------------- #
# End-to-end smoke (HuBERT path is fastest)
# --------------------------------------------------------------------------- #
def test_pipeline_smoke(tmp_path):
    import run_pipeline

    rc = run_pipeline.main(["--mock", "--encoder", "hubert", "--days", "2", "--skip-extended",
                            "--output-dir", str(tmp_path)])
    assert rc == 0
    import json

    res = json.loads((tmp_path / "results.json").read_text())
    assert len(res["training"]) == 2
    assert {"Shuffle", "Reversal", "Concat", "CallerChange", "ReceiverChange"} <= set(res["zero_shot"])
    assert (tmp_path / "checkpoints" / "day_0002.pt").exists()


# --------------------------------------------------------------------------- #
# Audit reconciliation (docs/audit/GMSLM_CODE_AUDIT.md)
# --------------------------------------------------------------------------- #
def test_talia_durations_round_trip_to_her_frame_counts():
    """Our seconds-valued constants must map back to the exact frame counts she used.

    Her code hard-coded frame counts at hop 512 / 22050 Hz; we store seconds so the
    detection rate can change without moving the thresholds. The two must agree at
    her rate, or we have silently redefined a threshold.
    """
    p = PreprocessingConfig.talia()
    assert p.detection_sample_rate == 22_050 and p.hop_length == 512
    assert p.to_frames(p.min_call_duration_s) == 10  # paper rounds to 0.25 s
    assert p.min_run_members == 3  # find_consecutive_series keeps groups with > 2 members
    assert p.to_frames(p.abrupt_up_max_span_s) == 20  # DURATION_FOR_ABBRUPT_UP_CALC
    assert p.to_frames(p.merge_gap_s) == 4  # MAX_DISTANCE_BETWEEN_NEI_IN_CALLS
    assert p.to_frames(p.noise_merge_gap_s) == 20
    assert p.to_frames(p.noise_min_duration_s) == 20  # paper rounds to 0.5 s
    assert p.to_frames(p.noise_max_duration_s) == 80  # paper rounds to 2 s
    assert p.to_frames(p.max_call_duration_s) == 172  # exactly 4 s, as in the paper
    assert round(p.boundary_pad_s * p.detection_sample_rate) == 500


def _tone_burst(
    sr: int, dur_s: float, freq_hz: float, onset_s: float, total_s: float,
    noise: float = 1e-5, amplitude: float = 0.05,
) -> np.ndarray:
    """A pure tone of ``dur_s`` inside ``total_s`` of low-level broadband noise.

    Levels matter here, and not only for realism. The ``"talia"`` detector thresholds
    the *absolute* |STFT| magnitude at 0.02, so a synthetic noise floor that is too hot
    makes every bin "survive", `freq_range` saturate and the three-way classification
    collapse -- the detector then reports nothing, for the opposite reason to the one
    it fails for on real data. The defaults here sit in the same regime as the real
    colony recordings (broadband RMS ~5e-4 across 4 capsules, in-band content far
    lower), so the test exercises the detector rather than an artefact of the fixture.
    The ``"spectral"`` path is scale-invariant and unaffected.
    """
    rng = np.random.default_rng(0)
    wav = (noise * rng.standard_normal(int(total_s * sr))).astype(np.float32)
    n = int(dur_s * sr)
    s = int(onset_s * sr)
    t = np.arange(n) / sr
    wav[s : s + n] += (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    return wav


@pytest.mark.parametrize("features", ["talia", "spectral"])
def test_both_feature_paths_detect_a_tone_burst(features):
    p = PreprocessingConfig.talia() if features == "talia" else PreprocessingConfig()
    wav = _tone_burst(48_000, dur_s=1.0, freq_hz=7_000, onset_s=1.0, total_s=3.0)
    audio_16k, activity = preprocess_waveform(wav, 48_000, p)
    assert audio_16k.dtype == np.float32
    assert len(audio_16k) == pytest.approx(3.0 * p.target_sample_rate, abs=2)
    assert activity.active.any()
    calls = activity_to_calls(activity, p)
    assert calls, f"{features} path found no call in a 1 s 7 kHz burst"
    best = max(calls, key=lambda c: min(c.end_s, 2.0) - max(c.start_s, 1.0))
    assert best.start_s == pytest.approx(1.0, abs=0.15)
    assert best.end_s == pytest.approx(2.0, abs=0.15)
    for c in calls:
        assert p.min_call_duration_s <= c.duration_s <= p.max_call_duration_s


def test_talia_path_populates_its_features_and_spectral_path_does_not():
    wav = _tone_burst(48_000, dur_s=0.8, freq_hz=7_000, onset_s=0.5, total_s=2.0)
    talia = preprocess_waveform(wav, 48_000, PreprocessingConfig.talia())[1]
    assert talia.freq_range_hz is not None and talia.bandwidth_bins is not None
    assert talia.density is None and talia.variance_db2 is None
    assert talia.bandwidth_bins.max() > 0
    spectral = preprocess_waveform(wav, 48_000, PreprocessingConfig())[1]
    assert spectral.density is not None and spectral.variance_db2 is not None
    assert spectral.freq_range_hz is None
    assert talia.tonal is None and spectral.tonal is not None
    assert talia.peak_freq_hz is None
    loud = spectral.peak_db > 20
    assert np.median(spectral.peak_freq_hz[loud]) == pytest.approx(7_000, abs=48_000 / 2048)


def test_spectral_detector_sees_high_calls_only_at_a_higher_rate():
    """A 13 kHz call is above the 11 kHz Nyquist of Talia's 22050 Hz detection rate; at
    48 kHz with a 16 kHz band ceiling it is found, and a ceiling below it hides it again.
    The noise floor is realistic (~31 dB broadband SNR): against the fixture's default near-silent
    floor, the resampler's finite stop-band leaks a detectable alias even at 22050 Hz."""
    wav = _tone_burst(48_000, dur_s=0.8, freq_hz=13_000, onset_s=1.0, total_s=3.0, noise=1e-3, amplitude=0.05)

    def calls(**kw):
        p = PreprocessingConfig(**kw)
        return activity_to_calls(preprocess_waveform(wav, 48_000, p)[1], p)

    assert not calls(detection_sample_rate=22_050, spectral_max_freq_hz=None)
    found = calls()  # the production default: 48 kHz with a 16 kHz ceiling
    assert any(c.start_s < 1.4 and c.end_s > 1.4 for c in found)
    assert not calls(detection_sample_rate=48_000, spectral_max_freq_hz=12_000.0)


def test_tonality_rejection_keeps_an_fm_call_and_drops_band_noise():
    """A 9 -> 7 kHz FM sweep and a 6-10 kHz noise burst of the same length both pass the
    density / variance tests, as cage noise in the call band does on real data. Only the
    sweep is a narrow line, so ``min_tonal_fraction`` removes the burst and keeps the call."""
    sr, n = 48_000, int(0.6 * 48_000)
    rng = np.random.default_rng(0)
    wav = (1e-3 * rng.standard_normal(4 * sr)).astype(np.float32)
    wav[sr:sr + n] += 0.05 * signal.chirp(np.arange(n) / sr, 9_000, 0.6, 7_000).astype(np.float32)
    burst = signal.sosfilt(signal.butter(6, [6_000, 10_000], "bandpass", fs=sr, output="sos"), rng.standard_normal(n))
    wav[int(2.5 * sr):int(2.5 * sr) + n] += (0.003 * burst / burst.std()).astype(np.float32)

    def mids(**kw):
        p = PreprocessingConfig(**kw)
        return sorted(round((c.start_s + c.end_s) / 2) for c in activity_to_calls(preprocess_waveform(wav, sr, p)[1], p))

    assert mids(min_tonal_fraction=None) == [1, 3]
    assert mids(min_tonal_fraction=0.5) == [1]


def test_detection_runs_at_detection_rate_not_native_rate():
    """Frame times must be spaced by hop / detection_sample_rate regardless of input rate."""
    p = PreprocessingConfig.talia()
    wav = _tone_burst(48_000, dur_s=0.5, freq_hz=7_000, onset_s=0.5, total_s=2.0)
    activity = preprocess_waveform(wav, 48_000, p)[1]
    assert activity.hop_s == pytest.approx(p.hop_s())
    assert float(np.diff(activity.frame_times_s).mean()) == pytest.approx(p.hop_s(), rel=1e-6)


def test_training_audio_is_unfiltered_unless_requested():
    """Her pipeline high-passed for detection only (audit S3.1 step 6)."""
    sr = 48_000
    t = np.arange(2 * sr) / sr
    # Low-frequency dominant, so removing it changes the RMS unmistakably.
    wav = (np.sin(2 * np.pi * 500 * t) + 0.05 * np.sin(2 * np.pi * 7_000 * t)).astype(np.float32)
    rms = lambda x: float(np.sqrt(np.mean(x**2)))  # noqa: E731
    kept = preprocess_waveform(wav, sr, PreprocessingConfig(highpass_training_audio=False))[0]
    dropped = preprocess_waveform(wav, sr, PreprocessingConfig(highpass_training_audio=True))[0]
    assert rms(kept) == pytest.approx(1 / np.sqrt(2), rel=0.05)  # 500 Hz still there
    assert rms(dropped) < 0.1 * rms(kept)  # 500 Hz filtered out


def test_unit_sweep_defaults_and_normalised_surprisal():
    """D15: the k sweep defaults to {50, 100, 200}, and normalised surprisal puts every k
    on the same scale (1.0 = chance), since raw bits/unit grow with vocabulary size."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from sweep_units import summarise

    assert AgentConfig().encoder.hubert.n_clusters_sweep == [50, 100, 200]
    for k in (50, 100, 200):
        chance_bits = math.log2(k + 3)
        row = summarise({"n_clusters": k,
                         "trajectory": [{"next_period": {"mean_surprisal_bits": chance_bits}}],
                         "zero_shot": {"Shuffle": {"accuracy": 0.5}}})
        assert row["normalised_surprisal"] == pytest.approx(1.0)
        assert row["zero_shot"]["Shuffle"] == 0.5


# --------------------------------------------------------------------------- #
# Night exclusion
# --------------------------------------------------------------------------- #
def test_night_window_wraps_midnight():
    from marmoset_developmental_agent.preprocessing.segmentation import in_night_window

    p = PreprocessingConfig()  # 19:00 -> 06:00
    at = lambda h, m=0: dt.datetime(2021, 6, 1, h, m)  # noqa: E731
    assert in_night_window(at(23), p) and in_night_window(at(0), p) and in_night_window(at(5, 59), p)
    assert in_night_window(at(19), p)
    assert not in_night_window(at(6), p) and not in_night_window(at(12), p) and not in_night_window(at(18, 59), p)
    q = PreprocessingConfig(night_start_hour=0, night_end_hour=5)  # non-wrapping window
    assert in_night_window(at(3), q) and not in_night_window(at(23), q)


def _burst_recording(start: dt.datetime, seconds: float = 20.0):
    from marmoset_developmental_agent.preprocessing.dataset_builder import Recording

    sr = 48_000
    wav = np.zeros(int(seconds * sr), np.float32)
    for k in range(2, int(seconds) - 2, 4):  # a 0.5 s 7 kHz "call" every 4 s
        wav += _tone_burst(sr, 0.5, 7_000, k, seconds)
    return Recording(audio=wav, sample_rate=sr, start_time=start, recording_id="night_test")


def test_exclude_night_drops_night_and_keeps_day():
    from marmoset_developmental_agent.preprocessing.dataset_builder import preprocess_recording

    on = PreprocessingConfig(frame_features="spectral", exclude_night=True)
    off = dataclasses.replace(on, exclude_night=False)
    night = _burst_recording(dt.datetime(2021, 6, 1, 23, 0))
    day = _burst_recording(dt.datetime(2021, 6, 1, 12, 0))
    assert preprocess_recording(night, off)[0], "fixture should produce chunks with the option off"
    assert preprocess_recording(night, on)[0] == []
    assert len(preprocess_recording(day, on)[0]) == len(preprocess_recording(day, off)[0])


def test_exclude_night_trims_a_recording_that_straddles_dusk():
    from marmoset_developmental_agent.preprocessing.dataset_builder import preprocess_recording
    from marmoset_developmental_agent.preprocessing.segmentation import recording_entirely_at_night

    on = PreprocessingConfig(frame_features="spectral", exclude_night=True)
    rec = _burst_recording(dt.datetime(2021, 6, 1, 18, 59, 50), seconds=20.0)  # dusk at +10 s
    assert not recording_entirely_at_night(rec.start_time, 20.0, on)
    for chunk in preprocess_recording(rec, on)[0]:
        for call in chunk.calls:
            onset = chunk.start_time + dt.timedelta(seconds=call.start_s)
            assert onset.time() < dt.time(19, 0)
