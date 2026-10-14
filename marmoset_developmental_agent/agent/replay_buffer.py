"""Sleep-consolidation replay buffer with recency-decaying day sampling.

A simulated day D has two phases, following Hasson's ``PastSamplingDataset``
(``references/hasson_dev/training/speech_learning/dataset/past_sampling_dataset.py``):

* **wake**: the day's own n_D samples, once, in chronological order;
* **sleep**: (X - 1) * n_D samples replayed from days 1..D (today included), shuffled.

So the day contains X * n_D samples in total, with X = ``passes_per_day``
(Hasson's ``dataset_cycle_repeats``, 25 in the published config).

Sleep samples pick a day d with probability

    P(d | D) = w(D - d) * n_d^s / sum_{d'} w(D - d') * n_{d'}^s

then a sample uniformly (with replacement) inside that day. Kernels:

* ``exponential``      w(Delta) = exp(-lambda * Delta)   (project spec)
* ``inverse_recency``  w(Delta) = 1 / (Delta + 1)        (official 1kd_learning_model)

``s = 1`` when ``size_weighted`` is on. The official code weights by day size, so under
a growing F(d) later days are over-represented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from marmoset_developmental_agent.config.agent_config import ReplayConfig


@dataclass
class DaySchedule:
    day: int
    wake: np.ndarray  # global sample indices, chronological
    sleep: np.ndarray  # global sample indices, shuffled replay

    @property
    def order(self) -> np.ndarray:
        return np.concatenate([self.wake, self.sleep])


class SleepReplayBuffer:
    """Keeps per-day index pools and draws recency-weighted replay."""

    def __init__(self, cfg: ReplayConfig, seed: int = 0) -> None:
        if cfg.passes_per_day < 1:
            raise ValueError("passes_per_day (X) must be >= 1")
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._days: Dict[int, np.ndarray] = {}

    def add_day(self, day: int, indices: np.ndarray) -> None:
        self._days[day] = np.asarray(indices, dtype=np.int64)

    @property
    def days(self) -> Tuple[int, ...]:
        return tuple(sorted(self._days))

    def day_probabilities(self, current_day: int) -> Tuple[np.ndarray, np.ndarray]:
        """(days, P(d | D)) over stored days <= D (< D if include_current_day is False)."""
        days = np.array(
            [d for d in self.days if d < current_day or (self.cfg.include_current_day and d == current_day)]
        )
        if len(days) == 0:
            return days, np.array([])
        delta = (current_day - days).astype(float)
        if self.cfg.kernel == "exponential":
            w = np.exp(-self.cfg.decay_lambda * delta)
        elif self.cfg.kernel == "inverse_recency":
            w = 1.0 / (delta + 1.0)
        else:
            raise ValueError(f"unknown kernel {self.cfg.kernel!r}")
        if self.cfg.size_weighted:
            w = w * np.array([len(self._days[d]) for d in days])
        return days, w / w.sum()

    def sample_replay(self, current_day: int, n: int) -> np.ndarray:
        days, p = self.day_probabilities(current_day)
        if n <= 0 or len(days) == 0:
            return np.zeros(0, dtype=np.int64)
        picked = self.rng.choice(days, size=n, p=p)
        out = np.empty(n, dtype=np.int64)
        for d in np.unique(picked):
            sel = picked == d
            out[sel] = self.rng.choice(self._days[int(d)], size=int(sel.sum()), replace=True)
        self.rng.shuffle(out)
        return out

    def build_day_schedule(self, current_day: int, today_indices: np.ndarray) -> DaySchedule:
        """Register today and return wake + sleep indices (X * n_D in total)."""
        self.add_day(current_day, today_indices)
        n_sleep = (self.cfg.passes_per_day - 1) * len(today_indices)
        return DaySchedule(current_day, np.asarray(today_indices), self.sample_replay(current_day, n_sleep))
