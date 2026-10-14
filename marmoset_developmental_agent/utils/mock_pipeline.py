"""Glue for building the synthetic chronological dataset used by --mock runs and tests."""

from __future__ import annotations

from typing import Dict, List, Tuple

from marmoset_developmental_agent.config.agent_config import AgentConfig
from marmoset_developmental_agent.preprocessing.dataset_builder import (
    ChronologicalDataset,
    Recording,
    build_chronological_dataset,
)
from marmoset_developmental_agent.preprocessing.segmentation import CallSegment
from marmoset_developmental_agent.utils.synthetic_data import SyntheticRecording, generate_mock_colony


def to_recording(r: SyntheticRecording) -> Recording:
    return Recording(
        audio=r.audio,
        sample_rate=r.sample_rate,
        start_time=r.start_time,
        recording_id=r.recording_id,
        day=r.day,
        annotations=[CallSegment(c.start_s, c.end_s, c.call_type, c.caller_id, c.receiver_id) for c in r.calls],
    )


def mock_recordings(cfg: AgentConfig, n_calendar_days: int = 4, seconds_per_day: float = 70.0) -> List[Recording]:
    return [to_recording(r) for r in generate_mock_colony(n_calendar_days, seconds_per_day, seed=cfg.seed)]


def mock_dataset(
    cfg: AgentConfig, n_calendar_days: int = 4, seconds_per_day: float = 70.0
) -> Tuple[ChronologicalDataset, Dict[str, object]]:
    """Synthetic multi-day colony -> GmSLM front-end -> chronological F(d) day split."""
    return build_chronological_dataset(
        mock_recordings(cfg, n_calendar_days, seconds_per_day), cfg.preprocessing, cfg.exposure
    )
