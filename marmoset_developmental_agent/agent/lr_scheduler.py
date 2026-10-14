"""Sigmoid learning-rate gate for the decoder (uLM).

    lr(t) = lr_max * ( 1 / (1 + exp(-k (t - T_p))) + floor )

For t << T_p the decoder barely learns (lr ~ lr_max * floor) while the encoder's units
are still reorganising. Around T_p the gate opens smoothly, and for t >> T_p the rate
approaches lr_max. ``t`` is the global update count (``unit="step"``) or the
fractional simulated day (``unit="day"``).

Relation to the official code: ``references/hasson_dev`` implements the same curve as a
*loss weight* on the uLM NLL, ``criterion.loss_weights.ulm_nll = "sigmoid-X0-K-Y"``
(for example ``sigmoid-279_000-0.0003-0.000001``), evaluated at the update number. Encoder
and uLM share one Adam optimizer there. Adam normalises gradient scale, so a loss weight
only suppresses learning once w*|g| falls near Adam's eps (1e-6). Gating the decoder's own
learning rate, as done here, gives the intended effect directly and independently of
gradient magnitude. The (x0, k, y_shift) triple maps one-to-one to (T_p, k, floor).
"""

from __future__ import annotations

import math
from typing import Any, Dict

from torch.optim import Optimizer

from marmoset_developmental_agent.config.agent_config import SigmoidScheduleConfig


def sigmoid_gate(t: float, t_p: float, k: float, floor: float = 0.0) -> float:
    """Numerically stable 1 / (1 + exp(-k (t - t_p))) + floor."""
    z = k * (t - t_p)
    s = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
    return s + floor


class SigmoidLRScheduler:
    """Sets every param group's LR to ``lr_max * sigmoid_gate(t)`` on :meth:`step`."""

    def __init__(self, optimizer: Optimizer, cfg: SigmoidScheduleConfig) -> None:
        self.optimizer = optimizer
        self.cfg = cfg
        self.last_t: float = 0.0
        self.step(0.0)

    def lr_at(self, t: float) -> float:
        return self.cfg.lr_max * sigmoid_gate(t, self.cfg.t_p, self.cfg.k, self.cfg.floor)

    def step(self, t: float) -> float:
        lr = self.lr_at(t)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.last_t = t
        return lr

    def get_last_lr(self) -> float:
        return self.lr_at(self.last_t)

    def state_dict(self) -> Dict[str, Any]:
        return {"last_t": self.last_t}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step(state["last_t"])
