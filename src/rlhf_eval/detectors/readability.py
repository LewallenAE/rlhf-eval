#!/usr/bin/env python3
"""
Readability Mismatch Detector.

Flags preference pairs where the rejected response has notably better
readability than the chosen response.  This may indicate a labeling
error — a more readable answer was marked as worse.

Uses the textstat library (Flesch-Kincaid Grade Level) to quantify
readability.  A *positive* mismatch score means the rejected text is
easier to read (lower grade level ⇒ more accessible).
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


class ReadabilityMismatchDetector(BaseDetector):
    """Flag pairs where the rejected response is more readable than the chosen."""

    name = "readability_mismatch"
    description = "Flags pairs where rejected has notably better readability than chosen"

    def __init__(self, min_words: int = 10) -> None:
        """
        Args:
            min_words: Minimum word count for a meaningful readability
                       score.  Shorter texts get a score of 0.0
                       (not flagged).
        """
        self._min_words = min_words

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flesch_kincaid_grade(text: str) -> float:
        """Return the Flesch-Kincaid grade level for *text*."""
        import textstat

        return float(textstat.flesch_kincaid_grade(text))

    def _readable_enough(self, text: str) -> bool:
        """Return True if the text has enough words for a valid score."""
        return len(text.split()) >= self._min_words

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def score(self, chosen_text: str, rejected_text: str) -> float:
        """Return readability mismatch score.

        Positive values mean the rejected text is *more* readable
        (lower grade level).  The score is ``chosen_grade - rejected_grade``.
        A large positive value suggests the rejected answer was easier
        to read yet was marked as worse.
        """
        if not (self._readable_enough(chosen_text) and self._readable_enough(rejected_text)):
            return 0.0

        chosen_grade = self._flesch_kincaid_grade(chosen_text)
        rejected_grade = self._flesch_kincaid_grade(rejected_text)
        return chosen_grade - rejected_grade

    def score_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Score all pairs.  textstat is CPU-only so *batch_size* is unused."""
        scores = np.array(
            [self.score(c, r) for c, r in pairs],
            dtype=np.float32,
        )
        return scores

    def get_default_threshold_percentile(self) -> float:
        return 99.5

    def get_metadata(self, chosen_text: str, rejected_text: str) -> dict[str, Any]:
        chosen_grade = self._flesch_kincaid_grade(chosen_text)
        rejected_grade = self._flesch_kincaid_grade(rejected_text)
        return {
            "chosen_grade_level": chosen_grade,
            "rejected_grade_level": rejected_grade,
            "grade_difference": chosen_grade - rejected_grade,
        }
