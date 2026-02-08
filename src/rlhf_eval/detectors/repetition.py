#!/usr/bin/env python3
"""
Repetition Detector.

Flags degenerate text patterns such as repeated words or phrases.
Measures the unique-word ratio (unique words / total words) and
n-gram repetition rate.  Low ratios indicate copy-paste or
degenerate generation artifacts.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
import logging
from collections import Counter
from typing import Any

# ----------------- Third Party Library -----------------
import numpy as np

# ----------------- Application Imports -----------------
from .base import BaseDetector

# ----------------- Module-level Configuration -----------------
logger = logging.getLogger(__name__)

DEFAULT_UNIQUENESS_THRESHOLD = 0.3
DEFAULT_NGRAM_SIZE = 3


class RepetitionDetector(BaseDetector):
    """Flag responses with degenerate repetition.

    The score is the *minimum* unique-word ratio across chosen and
    rejected texts.  Lower values are worse, so ``is_flagged`` is
    overridden to flag when the score falls *below* the threshold.
    """

    name = "repetition"
    description = "Flags degenerate text with high word/n-gram repetition"

    def __init__(
        self,
        ngram_size: int = DEFAULT_NGRAM_SIZE,
    ) -> None:
        """
        Args:
            ngram_size: Size of n-grams for the repetition check.
        """
        self._ngram_size = ngram_size

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def unique_word_ratio(text: str) -> float:
        """Return ``len(unique_words) / len(total_words)``.

        Returns 1.0 for empty text (not flagged).
        """
        words = text.lower().split()
        if not words:
            return 1.0
        return len(set(words)) / len(words)

    @staticmethod
    def ngram_repetition_rate(text: str, n: int = 3) -> float:
        """Return the fraction of n-grams that are duplicates.

        0.0 = no repeated n-grams; 1.0 = all n-grams identical.
        Returns 0.0 for texts shorter than *n* words.
        """
        words = text.lower().split()
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values())
        return repeated / len(ngrams)

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def score(self, chosen_text: str, rejected_text: str) -> float:
        """Return the *minimum* unique-word ratio of both texts.

        A low score means at least one response is heavily repetitive.
        """
        return min(
            self.unique_word_ratio(chosen_text),
            self.unique_word_ratio(rejected_text),
        )

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
        """Flag when score is *below* threshold (lower = more repetitive)."""
        return score <= threshold

    def get_default_threshold_percentile(self) -> float:
        # Use 0.5th percentile since lower scores are worse
        return 0.5

    def get_metadata(self, chosen_text: str, rejected_text: str) -> dict[str, Any]:
        return {
            "chosen_unique_ratio": self.unique_word_ratio(chosen_text),
            "rejected_unique_ratio": self.unique_word_ratio(rejected_text),
            "chosen_ngram_rep": self.ngram_repetition_rate(chosen_text, self._ngram_size),
            "rejected_ngram_rep": self.ngram_repetition_rate(rejected_text, self._ngram_size),
            "ngram_size": self._ngram_size,
        }
