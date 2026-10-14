"""Downstream classification probes for call type and caller identity (GmSLM Sec. 5.2, A.5).

Recipe from the GmSLM appendix:
* input: temporal **mean and variance** of a call's frame representations, concatenated -> (2D,);
* model: three fully connected layers of decreasing size, each followed by LayerNorm + ReLU,
  then a linear layer to the logits. The brief calls these "3-layer linear probes", but they
  are shallow MLPs;
* 90/10 train/validation split stratified by label; Adam + polynomial LR decay; 20 epochs;
* metrics: macro recall / precision / F1 (GmSLM: call type F1 90.7, caller F1 90.1 on
  InfantMarmosetsVox).

Representation sources: ``"ulm"`` (final uLM hidden states, as in GmSLM Table 3) or
``"encoder"`` (the encoder's continuous features).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn

from marmoset_developmental_agent.config.agent_config import EvaluationConfig
from marmoset_developmental_agent.decoders.transformer_ulm import UnitLanguageModel
from marmoset_developmental_agent.encoders.base_encoder import BaseAudioEncoder
from marmoset_developmental_agent.preprocessing.segmentation import Chunk


def pool_mean_var(x: torch.Tensor) -> torch.Tensor:
    """(T, D) -> (2D,) = [mean_t x, var_t x]."""
    return torch.cat([x.mean(0), x.var(0, unbiased=False)])


class ProbeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], n_classes: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ProbeResult:
    task: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_train: int
    n_val: int
    n_classes: int
    chance: float


@torch.no_grad()
def extract_call_representations(
    encoder: BaseAudioEncoder,
    ulm: UnitLanguageModel,
    chunks: Sequence[Chunk],
    source: Literal["ulm", "encoder"] = "ulm",
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Pooled representation per labelled call -> (X, call_types, caller_ids)."""
    encoder.eval()
    ulm.eval()
    device = next(ulm.parameters()).device
    X: List[np.ndarray] = []
    types: List[str] = []
    callers: List[int] = []
    fr = encoder.frame_rate
    for ch in chunks:
        labelled = [c for c in ch.calls if c.call_type is not None and c.caller_id is not None]
        if not labelled:
            continue
        wav = torch.from_numpy(ch.audio()).unsqueeze(0).to(device)
        if source == "ulm":
            h = ulm.hidden_states(encoder.encode(wav))[0]
        else:
            h = encoder.extract_features(wav)[0]
        for c in labelled:
            s, e = int(c.start_s * fr), min(h.shape[0], int(np.ceil(c.end_s * fr)))
            if e - s < 2:
                continue
            X.append(pool_mean_var(h[s:e].float()).cpu().numpy())
            types.append(str(c.call_type))
            callers.append(int(c.caller_id))  # type: ignore[arg-type]
    dim = 2 * (ulm.cfg.embed_dim if source == "ulm" else 1)
    return (np.stack(X) if X else np.zeros((0, dim))), types, callers


def train_probe(
    X: np.ndarray, labels: Sequence[object], cfg: EvaluationConfig, task: str, seed: int = 0
) -> ProbeResult:
    classes, y = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    if len(classes) < 2:
        raise ValueError(f"probe '{task}' needs >= 2 classes, got {len(classes)}")
    counts = np.bincount(y)
    stratify = y if counts.min() >= 2 and int(len(y) * cfg.probe_val_fraction) >= len(classes) else None
    tr, va = train_test_split(np.arange(len(y)), test_size=cfg.probe_val_fraction, random_state=seed, stratify=stratify)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6  # standardise on train split
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y)
    torch.manual_seed(seed)
    model = ProbeMLP(X.shape[1], cfg.probe_hidden_sizes, len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.probe_lr)
    sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=cfg.probe_epochs, power=1.0)
    bs = 64
    g = torch.Generator().manual_seed(seed)
    for _ in range(cfg.probe_epochs):
        model.train()
        perm = torch.as_tensor(tr)[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(perm), bs):
            b = perm[i : i + bs]
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xt[va]).argmax(-1).numpy()
    p, r, f, _ = precision_recall_fscore_support(y[va], pred, average="macro", zero_division=0)
    return ProbeResult(task, float(accuracy_score(y[va], pred)), float(p), float(r), float(f),
                       len(tr), len(va), len(classes), float(counts.max() / counts.sum()))


def run_probes(
    encoder: BaseAudioEncoder, ulm: UnitLanguageModel, chunks: Sequence[Chunk], cfg: EvaluationConfig,
    source: Literal["ulm", "encoder"] = "ulm", seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    X, types, callers = extract_call_representations(encoder, ulm, chunks, source)
    out: Dict[str, Dict[str, float]] = {}
    for task, labels in (("call_type", types), ("caller_identity", callers)):
        if len(labels) >= 10 and len(set(labels)) >= 2:
            out[task] = asdict(train_probe(X, labels, cfg, task, seed))
    return out
