"""Chronological day-by-day developmental training with sleep replay.

For simulated day D = 0, 1, ...:
  1. schedule = replay_buffer.build_day_schedule(D, dataset.day_indices(D))
     -> wake (today's chunks, in order) followed by sleep (recency-weighted replay).
  2. For each batch in wake + sleep:
       a. Encoder step (trainable encoders only, i.e. DINO-SR): self-distillation loss,
          optimizer step, EMA-teacher update.
       b. Tokenise the batch with the current encoder (teacher codebook; no dedup).
       c. Decoder step: next-unit NLL, with lr from the sigmoid gate at t.
  3. Call ``on_day_end`` (developmental evaluation hooks) and checkpoint every
     ``checkpoint_every_days`` days (Hasson saves weekly: ``dataset_save_cycle_every: 7``).

Encoder and decoder use separate AdamW optimizers (beta = (0.9, 0.98), eps = 1e-6, as in
both reference configs). No gradient flows from the uLM into the encoder, because units
are discrete, same as in the official joint model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from marmoset_developmental_agent.agent.lr_scheduler import SigmoidLRScheduler
from marmoset_developmental_agent.agent.replay_buffer import SleepReplayBuffer
from marmoset_developmental_agent.config.agent_config import TrainingConfig
from marmoset_developmental_agent.decoders.transformer_ulm import UnitLanguageModel
from marmoset_developmental_agent.encoders.base_encoder import BaseAudioEncoder, check_no_dedup
from marmoset_developmental_agent.preprocessing.dataset_builder import ChronologicalDataset

logger = logging.getLogger(__name__)


@dataclass
class DayLog:
    day: int
    n_wake: int
    n_sleep: int
    steps: int
    global_step: int
    encoder_loss: Optional[float]
    ulm_loss_wake: float
    ulm_loss_sleep: float
    decoder_lr_start: float
    decoder_lr_end: float
    seconds: float


class DevelopmentalTrainer:
    def __init__(
        self,
        encoder: BaseAudioEncoder,
        ulm: UnitLanguageModel,
        dataset: ChronologicalDataset,
        cfg: TrainingConfig,
        encoder_lr: float = 1e-4,
        encoder_weight_decay: float = 0.01,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_fn: Optional[Callable[[Path], None]] = None,
    ) -> None:
        if encoder.get_vocab_size() != ulm.n_units:
            raise ValueError("encoder vocabulary and uLM n_units differ")
        self.encoder = encoder.to(device)
        self.ulm = ulm.to(device)
        self.dataset = dataset
        self.cfg = cfg
        self.device = device
        self.train_encoder = encoder.is_trainable and cfg.train_encoder
        self.enc_opt = (
            torch.optim.AdamW(encoder.trainable_parameters(), lr=encoder_lr, betas=(0.9, 0.98), eps=1e-6,
                              weight_decay=encoder_weight_decay)
            if self.train_encoder else None
        )
        self.dec_opt = torch.optim.AdamW(ulm.parameters(), lr=cfg.decoder_lr.lr_max, betas=(0.9, 0.98), eps=1e-6,
                                         weight_decay=cfg.decoder_weight_decay)
        self.scheduler = SigmoidLRScheduler(self.dec_opt, cfg.decoder_lr)
        self.replay = SleepReplayBuffer(cfg.replay, seed=seed)
        self.global_step = 0
        self.history: List[DayLog] = []
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_fn = checkpoint_fn
        # Frozen encoders (HuBERT) always emit the same units -> cache per chunk index.
        self._token_cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    def _schedule_t(self, day: int, frac: float) -> float:
        return float(self.global_step) if self.cfg.decoder_lr.unit == "step" else day + frac

    def _tokens(self, idx: np.ndarray, wav: torch.Tensor) -> torch.Tensor:
        if self.train_encoder:
            toks = self.encoder.encode(wav)
        else:
            ids = idx.tolist()
            missing = list(dict.fromkeys(i for i in ids if i not in self._token_cache))
            if missing:
                pos = [ids.index(i) for i in missing]
                new = self.encoder.encode(wav[pos])
                for i, t in zip(missing, new):
                    self._token_cache[i] = t
            toks = torch.stack([self._token_cache[i] for i in ids])
        check_no_dedup(toks, wav.shape[-1], self.encoder)
        return toks

    def train_step(self, idx: np.ndarray, day: int, frac: float) -> Dict[str, float]:
        wav = torch.from_numpy(self.dataset.waveforms(idx)).to(self.device)
        out: Dict[str, float] = {}
        if self.train_encoder:
            assert self.enc_opt is not None
            self.encoder.train()
            enc_loss = self.encoder.training_step(wav)
            assert enc_loss is not None
            self.enc_opt.zero_grad(set_to_none=True)
            enc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.trainable_parameters(), self.cfg.grad_clip)
            self.enc_opt.step()
            self.encoder.after_optimizer_step()
            out["encoder_loss"] = float(enc_loss.detach())
        self.encoder.eval()
        units = self._tokens(idx, wav)

        self.ulm.train()
        out["decoder_lr"] = self.scheduler.step(self._schedule_t(day, frac))
        loss = self.ulm.loss(units)
        self.dec_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ulm.parameters(), self.cfg.grad_clip)
        self.dec_opt.step()
        out["ulm_loss"] = float(loss.detach())
        self.global_step += 1
        return out

    def _batches(self, order: np.ndarray) -> List[np.ndarray]:
        bs = self.cfg.batch_size
        n = (len(order) // bs) * bs if self.cfg.drop_last else len(order)
        return [order[i : i + bs] for i in range(0, n, bs)]

    def train_day(self, day: int) -> DayLog:
        t0 = time.time()
        sched = self.replay.build_day_schedule(day, self.dataset.day_indices(day))
        n_wake = len(sched.wake)
        batches = self._batches(sched.order)
        enc_losses: List[float] = []
        wake_l: List[float] = []
        sleep_l: List[float] = []
        lr0 = self.scheduler.lr_at(self._schedule_t(day, 0.0))
        seen = 0
        for b_i, idx in enumerate(batches):
            stats = self.train_step(idx, day, b_i / max(1, len(batches)))
            if "encoder_loss" in stats:
                enc_losses.append(stats["encoder_loss"])
            (wake_l if seen < n_wake else sleep_l).append(stats["ulm_loss"])
            seen += len(idx)
            if self.global_step % self.cfg.log_every_steps == 0:
                logger.info(
                    "day %d step %d | phase %s | enc %.3f | uLM %.3f | dec lr %.2e",
                    day + 1, self.global_step, "wake" if seen <= n_wake else "sleep",
                    stats.get("encoder_loss", float("nan")), stats["ulm_loss"], stats["decoder_lr"],
                )
        log = DayLog(
            day=day,
            n_wake=n_wake,
            n_sleep=len(sched.sleep),
            steps=len(batches),
            global_step=self.global_step,
            encoder_loss=float(np.mean(enc_losses)) if enc_losses else None,
            ulm_loss_wake=float(np.mean(wake_l)) if wake_l else float("nan"),
            ulm_loss_sleep=float(np.mean(sleep_l)) if sleep_l else float("nan"),
            decoder_lr_start=lr0,
            decoder_lr_end=self.scheduler.get_last_lr(),
            seconds=time.time() - t0,
        )
        self.history.append(log)
        return log

    def run(
        self,
        num_days: Optional[int] = None,
        on_day_end: Optional[Callable[[int, DayLog], None]] = None,
    ) -> List[DayLog]:
        n = self.dataset.num_days if num_days is None else min(num_days, self.dataset.num_days)
        for day in range(n):
            log = self.train_day(day)
            logger.info(
                "Day %d/%d done: wake %d + sleep %d samples, %d steps, enc %s, uLM wake %.3f / sleep %.3f, "
                "dec lr %.2e -> %.2e (%.1fs)",
                day + 1, n, log.n_wake, log.n_sleep, log.steps,
                "n/a" if log.encoder_loss is None else f"{log.encoder_loss:.3f}",
                log.ulm_loss_wake, log.ulm_loss_sleep, log.decoder_lr_start, log.decoder_lr_end, log.seconds,
            )
            if on_day_end is not None:
                on_day_end(day, log)
            if self.checkpoint_dir is not None and self.checkpoint_fn is not None and (
                (day + 1) % self.cfg.checkpoint_every_days == 0 or day == n - 1
            ):
                self.checkpoint_fn(self.checkpoint_dir / f"day_{day + 1:04d}.pt")
        return self.history

    def history_dicts(self) -> List[Dict[str, object]]:
        return [asdict(h) for h in self.history]
