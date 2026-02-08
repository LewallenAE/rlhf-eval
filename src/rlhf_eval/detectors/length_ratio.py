#!/usr/bin/env python3
"""
Length Ratio Detector.

Flags suspiciously short responses to complex prompts.  A very brief
answer to a detailed question often means the model deflected or
gave a non-answer, making the preference label unreliable.

The detector uses a *prompt* field — unlike most detectors that only
compare chosen vs rejected text, this one also needs the human prompt.
When only chosen/rejected texts are available the score falls back to
the ratio between the shorter and longer response.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
import logging
from typing import Any

# ----------------- Third Party Library -----------------
import numpy as np

# ----------------- Application Imports -----------------
from .base import BaseDetector

# ----------------- Module-level Configuration -----------------
logger = logging.getLogger(__name__)

DEFAULT_MIN_PROMPT_WORDS = 30
DEFAULT_RATIO_THRESHOLD = 0.5


class LengthRatioDetector(BaseDetector):
    """Flag pairs where one response is suspiciously short for the prompt.

    The score is ``min(len_chosen, len_rejected) / max(len_chosen, len_rejected)``.
    A very low ratio (near 0) means one response is drastically shorter.
    The detector flags when the score falls *below* the threshold AND
    the prompt is sufficiently complex.
    """

    name = "length_ratio"
    description = "Flags very short responses to complex prompts"

    def __init__(
        self,
        min_prompt_words: int = DEFAULT_MIN_PROMPT_WORDS,
    ) -> None:
        """
        Args:
            min_prompt_words: Minimum prompt word count to consider the
                              prompt "complex" enough to warrant a long answer.
        """
        self._min_prompt_words = min_prompt_words

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def score(self, chosen_text: str, rejected_text: str) -> float:
        """Return the ratio of the shorter to longer response length.

        Returns 1.0 when both are empty (not flagged).
        """
        chosen_len = self._word_count(chosen_text)
        rejected_len = self._word_count(rejected_text)
        max_len = max(chosen_len, rejected_len)
        if max_len == 0:
            return 1.0
        return min(chosen_len, rejected_len) / max_len

    def score_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Score all pairs (CPU-only, batch_size unused)."""
        return np.array(
            [self.score(c, r) for c, r in pairs],
            dtype=np.float32,
        )

    def is_flagged(self, score: float, threshold: float) -> bool:
        """Flag when the ratio is *below* threshold (shorter = worse)."""
        return score <= threshold

    def get_default_threshold_percentile(self) -> float:
        # 0.5th percentile — flag the most extreme length mismatches
        return 0.5

    def get_metadata(self, chosen_text: str, rejected_text: str) -> dict[str, Any]:
        return {
            "chosen_words": self._word_count(chosen_text),
            "rejected_words": self._word_count(rejected_text),
            "ratio": self.score(chosen_text, rejected_text),
        }
