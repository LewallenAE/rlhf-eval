#!/usr/bin/env python3
"""
Semantic Similarity Detector.

Flags preference pairs where chosen and rejected responses are too
semantically similar.  High similarity suggests the preference label
is arbitrary — the two responses are essentially identical, so calling
one "better" injects noise into reward-model training.

Uses sentence-transformers (all-MiniLM-L6-v2 by default) with
normalized embeddings so cosine similarity reduces to a dot product.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
import logging
from typing import Any

# ----------------- Third Party Library -----------------
import numpy as np
from tqdm import tqdm

# ----------------- Application Imports -----------------
from .base import BaseDetector

# ----------------- Module-level Configuration -----------------
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSimilarityDetector(BaseDetector):
    """Detect pairs where chosen and rejected are near-duplicates.

    Empirical results on HH-RLHF show a cliff at P99.5 (~0.9999),
    revealing ~800 effectively identical response pairs.
    """

    name = "semantic_similarity"
    description = "Flags pairs where chosen/rejected responses are too similar"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model_name: Sentence-transformers model identifier.
            device: ``"cuda"``, ``"cpu"``, or ``None`` for auto-detect.
        """
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        logger.info(
            "Model loaded on device: %s", self._model.device
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into normalized embeddings."""
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def score(self, chosen_text: str, rejected_text: str) -> float:
        """Return cosine similarity between *chosen_text* and *rejected_text*."""
        embeddings = self._encode([chosen_text, rejected_text])
        similarity: float = float(np.dot(embeddings[0], embeddings[1]))
        return similarity

    def score_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Efficiently score multiple pairs.

        Encodes all chosen and rejected texts in two batched calls,
        then computes row-wise dot products (cosine similarity with
        pre-normalized vectors).
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        chosen_texts = [c for c, _ in pairs]
        rejected_texts = [r for _, r in pairs]

        logger.info("Encoding %d chosen texts", len(chosen_texts))
        chosen_emb = self._encode(chosen_texts, batch_size=batch_size, show_progress=True)

        logger.info("Encoding %d rejected texts", len(rejected_texts))
        rejected_emb = self._encode(rejected_texts, batch_size=batch_size, show_progress=True)

        # Row-wise dot product == cosine similarity for unit vectors
        similarities = np.sum(chosen_emb * rejected_emb, axis=1)
        return similarities.astype(np.float32)

    def get_default_threshold_percentile(self) -> float:
        """P99.5 — empirical cliff for near-duplicate responses."""
        return 99.5

    def get_metadata(self, chosen_text: str, rejected_text: str) -> dict[str, Any]:
        """Return similarity score and model name."""
        return {
            "similarity": self.score(chosen_text, rejected_text),
            "model": self._model_name,
        }
