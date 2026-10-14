"""Construction, checkpointing and reloading of encoder + uLM pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import torch

from marmoset_developmental_agent.config.agent_config import AgentConfig, _from_dict
from marmoset_developmental_agent.decoders.transformer_ulm import UnitLanguageModel
from marmoset_developmental_agent.encoders.base_encoder import BaseAudioEncoder
from marmoset_developmental_agent.encoders.dino_sr import DinoSREncoder
from marmoset_developmental_agent.encoders.marmoset_hubert import MarmosetHuBERTEncoder

EncoderName = Literal["dinosr", "hubert"]


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_encoder(cfg: AgentConfig, name: Optional[EncoderName] = None) -> BaseAudioEncoder:
    name = name or cfg.encoder.active
    if name == "dinosr":
        return DinoSREncoder(cfg.encoder.dinosr, seed=cfg.seed)
    if name == "hubert":
        return MarmosetHuBERTEncoder(cfg.encoder.hubert, seed=cfg.seed)
    raise ValueError(f"unknown encoder {name!r}")


def build_ulm(cfg: AgentConfig, n_units: int) -> UnitLanguageModel:
    torch.manual_seed(cfg.seed)
    return UnitLanguageModel(n_units, cfg.decoder)


def save_checkpoint(
    path: Union[str, Path], cfg: AgentConfig, encoder_name: str, encoder: BaseAudioEncoder,
    ulm: UnitLanguageModel, extra: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.to_dict(),
            "encoder_name": encoder_name,
            "encoder": encoder.state_dict(),
            "ulm": ulm.state_dict(),
            "n_units": ulm.n_units,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: Union[str, Path], device: str = "cpu"
) -> Tuple[AgentConfig, BaseAudioEncoder, UnitLanguageModel]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = _from_dict(AgentConfig, ckpt["config"])
    encoder = build_encoder(cfg, ckpt["encoder_name"])
    encoder.load_state_dict(ckpt["encoder"])
    ulm = build_ulm(cfg, ckpt["n_units"])
    ulm.load_state_dict(ckpt["ulm"])
    return cfg, encoder.to(device).eval(), ulm.to(device).eval()
