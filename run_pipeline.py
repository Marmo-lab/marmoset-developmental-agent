#!/usr/bin/env python
"""End-to-end Marmoset Developmental Agent pipeline.

    python run_pipeline.py --mock                       # synthetic data, tiny models, CPU, ~1-2 min
    python run_pipeline.py --config config/default.yaml # real data (see config/default.yaml)

Stages:
  1. data        synthetic colony (mock) or raw recordings / chunk manifest (real)
  2. preprocess  5 kHz HPF -> noise gate -> 0.25-4 s calls -> 10 s @ 16 kHz chunks -> F(d) days
  3. encoders    HuBERT (L1, 50 units) + DINO-SR (512 units): tokenise, check no-dedup
  4. training    chronological days: wake + sleep replay, sigmoid-gated decoder LR
  5. vocoder     uLM continuation -> unit HiFi-GAN -> waveform
  6. evaluation  zero-shot (Shuffle/Reversal/Concat/Phee), trajectory, probes, ICI & pitch metrics
  7. summary     results.json + console table
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from marmoset_developmental_agent.agent.factory import build_encoder, build_ulm, resolve_device, save_checkpoint
from marmoset_developmental_agent.agent.trainer import DevelopmentalTrainer
from marmoset_developmental_agent.config.agent_config import AgentConfig, load_config, save_config
from marmoset_developmental_agent.encoders.base_encoder import BaseAudioEncoder, check_no_dedup, run_lengths
from marmoset_developmental_agent.evaluation.developmental_eval import DevelopmentalTracker
from marmoset_developmental_agent.evaluation.downstream_probes import run_probes
from marmoset_developmental_agent.evaluation.new_metrics import ici_distortion_sensitivity, pitch_shift_sensitivity
from marmoset_developmental_agent.evaluation.zero_shot_benchmarks import ZeroShotBenchmarks, format_results
from marmoset_developmental_agent.preprocessing.dataset_builder import (
    ChronologicalDataset,
    build_chronological_dataset,
    export_manifest,
    load_manifest,
    load_recordings_from_manifest,
)
from marmoset_developmental_agent.utils.log import setup_logging
from marmoset_developmental_agent.utils.mock_pipeline import mock_dataset
from marmoset_developmental_agent.utils.synthetic_data import PheeExchange, generate_mock_phee_dialogues
from marmoset_developmental_agent.vocoder.hifigan_u2v import HiFiGANU2V

log = logging.getLogger("pipeline")


def stage(i: int, name: str) -> None:
    log.info("=" * 72)
    log.info("[%d/7] %s", i, name)
    log.info("=" * 72)


def load_data(cfg: AgentConfig, mock: bool) -> tuple[ChronologicalDataset, Dict[str, Any], List[PheeExchange]]:
    if mock:
        stage(1, "Generating synthetic multi-day colony audio (48 kHz, shuffled file order)")
        ds, info = mock_dataset(cfg)
        phee = generate_mock_phee_dialogues(seed=cfg.seed + 1)
        return ds, info, phee
    stage(1, "Loading real data")
    if cfg.data.chunk_manifest:
        return load_manifest(cfg.data.chunk_manifest, cfg.preprocessing.target_sample_rate), {}, []
    if not cfg.data.recordings_manifest:
        raise SystemExit("config.data.recordings_manifest or config.data.chunk_manifest is required without --mock")
    recs = load_recordings_from_manifest(cfg.data.recordings_manifest)
    ds, info = build_chronological_dataset(recs, cfg.preprocessing, cfg.exposure)
    if cfg.data.export_dir:
        log.info("Exported chunk manifest -> %s", export_manifest(ds, cfg.data.export_dir))
    log.warning("PHEEDB loader not implemented yet; Phee eval skipped on real data.")
    return ds, info, []


def token_stats(name: str, enc: BaseAudioEncoder, wav: torch.Tensor) -> Dict[str, Any]:
    toks = enc.encode(wav)
    check_no_dedup(toks, wav.shape[-1], enc)
    runs = torch.cat([run_lengths(t)[1] for t in toks]).float()
    s = {
        "vocab_size": enc.get_vocab_size(),
        "frame_rate_hz": enc.frame_rate,
        "token_shape": list(toks.shape),
        "units_used": int(toks.unique().numel()),
        "mean_run_length_frames": float(runs.mean()),
    }
    log.info(
        "%-8s tokens %s | vocab %d | %.0f Hz | %d units used | mean run %.2f frames (repeats kept)",
        name, tuple(toks.shape), s["vocab_size"], s["frame_rate_hz"], s["units_used"], s["mean_run_length_frames"],
    )
    return s


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="Synthetic data + tiny models (no GPU / datasets needed).")
    ap.add_argument("--config", type=str, help="YAML overriding defaults (mock defaults if --mock).")
    ap.add_argument("--encoder", choices=["dinosr", "hubert"], help="Encoder driving training (overrides config).")
    ap.add_argument("--days", type=int, help="Number of simulated days to train (overrides config).")
    ap.add_argument("--output-dir", type=str, help="Output directory (overrides config).")
    ap.add_argument("--seed", type=int, help="Random seed (overrides config).")
    ap.add_argument("--skip-extended", action="store_true", help="Skip probes and ICI / pitch-shift metrics.")
    args = ap.parse_args(argv)

    setup_logging()
    t_start = time.time()
    cfg = load_config(args.config, mock=args.mock)
    if args.encoder:
        cfg.encoder.active = args.encoder
    if args.days is not None:
        cfg.training.num_days = args.days
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config_resolved.yaml")
    log.info("Mode: %s | device: %s | encoder: %s | output: %s", "MOCK" if args.mock else "REAL", device,
             cfg.encoder.active, out_dir)
    results: Dict[str, Any] = {"mode": "mock" if args.mock else "real", "encoder": cfg.encoder.active}

    # 1-2. Data + preprocessing -------------------------------------------------
    ds, info, phee = load_data(cfg, args.mock)
    stage(2, "Preprocessing summary (5 kHz HPF, noise gate, 0.25-4 s calls, 10 s chunks @ 16 kHz, F(d) days)")
    summary = {**ds.summary(), **info}
    log.info("Dataset: %s", json.dumps(summary, default=str))
    starts = [c.start_time for c in ds.chunks]
    assert all(a <= b for a, b in zip(starts, starts[1:])), "chunks are not chronological"
    results["dataset"] = summary

    n_train_days = cfg.training.num_days or max(1, ds.num_days - cfg.data.eval_holdout_days)
    n_train_days = min(n_train_days, ds.num_days)
    test_chunks = ds.days(n_train_days, ds.num_days) or ds.day(n_train_days - 1)
    train_chunks = ds.days(0, n_train_days)
    log.info("Training days: %d (%d chunks) | held-out evaluation chunks: %d",
             n_train_days, len(train_chunks), len(test_chunks))

    # 3. Encoders -------------------------------------------------------------
    stage(3, "Dual encoders: marmoset HuBERT (layer 1, k-means) vs DINO-SR (online codebook)")
    encoders: Dict[str, BaseAudioEncoder] = {}
    for name in ("hubert", "dinosr"):
        encoders[name] = build_encoder(cfg, name).to(device)
    hub = encoders["hubert"]
    if cfg.encoder.hubert.pretrained_path is None:
        log.warning("HuBERT: no pretrained_path -> random-init weights (mock); units are not meaningful.")
    hub.fit(torch.from_numpy(c.audio()).unsqueeze(0).to(device) for c in train_chunks)  # type: ignore[attr-defined]
    probe_wav = torch.from_numpy(np.stack([c.audio() for c in ds.day(0)[:2]])).to(device)
    results["encoders"] = {n: token_stats(n, e, probe_wav) for n, e in encoders.items()}

    # 4. Developmental training ----------------------------------------------------
    stage(4, f"Developmental training: {n_train_days} simulated days, X={cfg.training.replay.passes_per_day} "
             f"passes/day ({cfg.training.replay.kernel} replay kernel)")
    encoder = encoders[cfg.encoder.active]
    ulm = build_ulm(cfg, encoder.get_vocab_size()).to(device)
    log.info("uLM: %.2fM params, vocab %d (+3 specials)", sum(p.numel() for p in ulm.parameters()) / 1e6, ulm.n_units)
    ckpt_dir = out_dir / "checkpoints"
    trainer = DevelopmentalTrainer(
        encoder, ulm, ds, cfg.training,
        encoder_lr=cfg.encoder.dinosr.lr, encoder_weight_decay=cfg.encoder.dinosr.weight_decay,
        device=device, seed=cfg.seed, checkpoint_dir=ckpt_dir,
        checkpoint_fn=lambda p: save_checkpoint(p, cfg, cfg.encoder.active, encoder, ulm,
                                                {"global_step": trainer.global_step}),
    )
    tracker = DevelopmentalTracker(encoder, ulm, ds, cfg.evaluation,
                                   ood_waveforms=[np.concatenate([p.call_audio, p.response_audio]) for p in phee] or None)
    trainer.run(n_train_days, on_day_end=tracker)
    results["training"] = trainer.history_dicts()
    results["trajectory"] = tracker.history
    results["lexicon"] = tracker.lexicon.as_dict()

    # 5. Vocoder --------------------------------------------------------------
    stage(5, "Unit-to-waveform HiFi-GAN (no speaker / F0 conditioning)")
    vocoder = HiFiGANU2V(encoder.get_vocab_size(), cfg.vocoder).to(device)
    prompt = encoder.encode(torch.from_numpy(test_chunks[0].audio()).unsqueeze(0).to(device))[0, :100]
    g = torch.Generator().manual_seed(cfg.seed)
    cont = ulm.generate(prompt, max_new_tokens=100, temperature=1.0, generator=g)
    wav = vocoder.synthesize(cont)
    log.info("Prompt %d units + 100 generated -> waveform %s (%.2f s @ %d Hz, %s vocoder)",
             len(prompt), tuple(wav.shape), wav.shape[-1] / vocoder.sample_rate, vocoder.sample_rate,
             "mock-" + cfg.vocoder.mock_mode if vocoder.is_mock else "trained")
    try:
        import soundfile as sf

        sf.write(out_dir / "generated_continuation.wav", wav[0].cpu().numpy(), vocoder.sample_rate)
    except Exception as e:  # pragma: no cover - optional
        log.warning("Could not write wav: %s", e)
    results["vocoder"] = {"n_units": int(len(cont)), "n_samples": int(wav.shape[-1]), "mock": vocoder.is_mock}

    # 6. Evaluation -------------------------------------------------------------
    stage(6, "Evaluation on held-out days")
    zs = ZeroShotBenchmarks(encoder, ulm, cfg.evaluation, seed=cfg.seed).run(test_chunks, phee)
    results["zero_shot"] = zs
    if not args.skip_extended:
        probe_chunks = test_chunks + train_chunks  # mock data is small; real runs should use a labelled set
        results["probes"] = run_probes(encoder, ulm, probe_chunks, cfg.evaluation, "ulm", cfg.seed)
        results["ici_distortion"] = ici_distortion_sensitivity(encoder, ulm, test_chunks, cfg.evaluation)
        results["pitch_shift"] = pitch_shift_sensitivity(encoder, ulm, test_chunks, cfg.evaluation)

    # 7. Summary ---------------------------------------------------------------------
    stage(7, "Summary")
    print("\nZero-shot structural benchmarks (chance = 50%; untrained mock models hover near chance)")
    print(format_results(zs))
    if results["trajectory"]:
        print("\nDevelopmental trajectory (next-period surprisal on unseen data)")
        for r in results["trajectory"]:
            nx = r.get("next_period")
            if nx:
                print(f"  day {r['day']:>3}: {nx['mean_surprisal_bits']:.3f} bits/unit, PPL {nx['perplexity']:.2f}, "
                      f"lexicon size {r.get('lexicon_size', 0)}")
    for k in ("probes", "ici_distortion", "pitch_shift"):
        if k in results:
            print(f"\n{k}:")
            for name, v in results[k].items():
                if "f1" in v:
                    print(f"  {name:<16} acc {v['accuracy']:.3f}  macro-F1 {v['f1']:.3f}  (chance {v['chance']:.3f}, n_val {v['n_val']})")
                else:
                    print(f"  {name:<16} acc {v['accuracy']:.3f}  delta-surprisal {v['delta_surprisal_bits']:+.4f} bits  (n={v['n']})")
    results["wall_time_s"] = time.time() - t_start
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Done in %.1fs. Results -> %s", results["wall_time_s"], out_dir / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
