"""Phase 2: vocalization-to-unit encoders (one unit per 20 ms frame, never deduplicated)."""

from marmoset_developmental_agent.encoders.base_encoder import BaseAudioEncoder, check_no_dedup, run_lengths
from marmoset_developmental_agent.encoders.dino_sr import DinoSREncoder
from marmoset_developmental_agent.encoders.marmoset_hubert import MarmosetHuBERTEncoder

__all__ = ["BaseAudioEncoder", "DinoSREncoder", "MarmosetHuBERTEncoder", "check_no_dedup", "run_lengths"]
