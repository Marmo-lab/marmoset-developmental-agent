"""Marmoset-adapted HuBERT V2U (GmSLM Sec. 3.2, Appendix A.3).

GmSLM recipe:
  1. Start from hubert-base (12 layers, 768-d, 20 ms frames) pretrained on LibriSpeech.
  2. Train k-means on intermediate-layer features over COLONYDB (MiniBatchKMeans,
     50% of train+valid, mini-batch 10k frames, k-means++ with 20 restarts).
  3. Fine-tune HuBERT on COLONYDB with the new quantiser as its masked-prediction target
     (mask span l=10, p=8%, Adam beta=(0.9, 0.98), peak LR 3.5e-3, 150k steps).
  4. Take units from **layer 1** of the fine-tuned model, quantised with **50 clusters**.
     Earlier layers work better for marmosets (GmSLM Fig. 6), since the calls are more
     acoustic than contextual.

Steps 1-3 happen offline in fairseq. This module covers step 4 (feature extraction
and k-means) and loads a *converted* HF checkpoint via ``pretrained_path``. To convert a
fairseq checkpoint, use ``transformers/models/hubert/convert_hubert_original_pytorch_checkpoint_to_pytorch.py``.
Without a checkpoint it builds a randomly initialised model with the configured
architecture (mock mode). The pipeline then still runs, but the units carry no meaning.

Repeated units are preserved (see ``base_encoder`` no-dedup rule).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from marmoset_developmental_agent.config.agent_config import HuBERTEncoderConfig
from marmoset_developmental_agent.encoders.base_encoder import CONV_LAYERS, BaseAudioEncoder

logger = logging.getLogger(__name__)


class MarmosetHuBERTEncoder(BaseAudioEncoder):
    """Frozen HuBERT layer-``cfg.layer`` features + k-means (``cfg.n_clusters``)."""

    is_trainable = False

    def __init__(self, cfg: HuBERTEncoderConfig, seed: int = 0) -> None:
        super().__init__()
        from transformers import HubertConfig, HubertModel

        self.cfg = cfg
        if cfg.pretrained_path:
            self.hubert = HubertModel.from_pretrained(cfg.pretrained_path)
        else:
            torch.manual_seed(seed)
            hcfg = HubertConfig(
                hidden_size=cfg.hidden_size,
                num_hidden_layers=cfg.num_hidden_layers,
                num_attention_heads=cfg.num_attention_heads,
                intermediate_size=cfg.intermediate_size,
                conv_dim=tuple([cfg.conv_dim] * len(CONV_LAYERS)),
                conv_kernel=tuple(k for k, _ in CONV_LAYERS),
                conv_stride=tuple(s for _, s in CONV_LAYERS),
                num_conv_pos_embeddings=cfg.num_conv_pos_embeddings,
                num_conv_pos_embedding_groups=cfg.num_conv_pos_embedding_groups,
                apply_spec_augment=False,
            )
            self.hubert = HubertModel(hcfg)
        n_layers = self.hubert.config.num_hidden_layers
        if not 0 <= cfg.layer <= n_layers:
            raise ValueError(f"layer {cfg.layer} outside [0, {n_layers}]")
        self.hubert.eval()
        for p in self.hubert.parameters():
            p.requires_grad = False
        # k-means centroids live in a buffer so they are saved in the state_dict.
        self.register_buffer("centroids", torch.zeros(0, self.hubert.config.hidden_size))
        self.seed = seed

    # ------------------------------------------------------------------ #
    def get_vocab_size(self) -> int:
        return self.cfg.n_clusters

    @property
    def is_fitted(self) -> bool:
        return self.centroids.shape[0] == self.cfg.n_clusters

    @torch.no_grad()
    def extract_features(self, wav: torch.Tensor) -> torch.Tensor:
        """Layer-``cfg.layer`` hidden states (B, T, D).

        ``hidden_states[0]`` is the transformer input (after positional conv), and
        ``hidden_states[i]`` is the output of transformer layer i. This matches the
        fairseq/textlesslib convention, where "layer 1" means the first layer's output.
        """
        wav = self._as_batch(wav).to(self.centroids.device)
        if self.cfg.normalize_waveform:
            wav = self._normalize(wav)
        out = self.hubert(wav, output_hidden_states=True)
        return out.hidden_states[self.cfg.layer]

    def fit(self, wavs: Iterable[torch.Tensor]) -> None:
        """Fit MiniBatchKMeans on a random ``kmeans_fit_fraction`` of frames."""
        rng = np.random.default_rng(self.seed)
        feats = []
        for w in wavs:
            f = self.extract_features(w).reshape(-1, self.centroids.shape[1]).cpu().numpy()
            keep = rng.random(len(f)) < self.cfg.kmeans_fit_fraction
            feats.append(f[keep])
        x = np.concatenate(feats)
        if len(x) < self.cfg.n_clusters:
            raise ValueError(f"only {len(x)} frames to fit {self.cfg.n_clusters} clusters")
        km = MiniBatchKMeans(
            n_clusters=self.cfg.n_clusters,
            batch_size=self.cfg.kmeans_batch_size,
            n_init=self.cfg.kmeans_n_init,
            init="k-means++",
            random_state=self.seed,
        ).fit(x)
        self.centroids = torch.as_tensor(km.cluster_centers_, dtype=torch.float32, device=self.centroids.device)
        logger.info("HuBERT k-means fitted: %d clusters on %d frames (inertia %.3g)", self.cfg.n_clusters, len(x), km.inertia_)

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Call fit() (k-means) or load centroids before encode()")
        feats = self.extract_features(wav)  # (B, T, D)
        d = torch.cdist(feats, self.centroids.unsqueeze(0).expand(feats.shape[0], -1, -1))
        return d.argmin(-1)  # (B, T): one unit per frame, NO deduplication

    def load_kmeans(self, centroids: np.ndarray) -> None:
        """Load centroids from an external k-means (for example a textlesslib/joblib model)."""
        if centroids.shape != (self.cfg.n_clusters, self.centroids.shape[1]):
            raise ValueError(f"centroid shape {centroids.shape} mismatches config")
        self.centroids = torch.as_tensor(centroids, dtype=torch.float32, device=self.centroids.device)

    def train(self, mode: bool = True) -> "MarmosetHuBERTEncoder":
        super().train(mode)
        self.hubert.eval()  # always frozen
        return self

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # type: ignore[override]
        if "centroids" in state_dict:
            self.centroids = torch.empty_like(state_dict["centroids"])
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
