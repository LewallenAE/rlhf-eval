#!/usr/bin/env python3
"""
Refusal Preference Bias Detector.

Flags pairs where the chosen response is a refusal but the rejected
response is substantive/helpful.  This pattern indicates that human
labelers may have over-rewarded refusals, teaching the reward model
that refusing is always better than engaging — a known failure mode
in RLHF training.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
import logging
import re
from typing import Any

# ----------------- Third Party Library -----------------
import numpy as np

# ----------------- Application Imports -----------------
from .base import BaseDetector

# ----------------- Module-level Configuration -----------------
logger = logging.getLogger(__name__)

# Patterns that indicate a refusal / decline to help
REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bI can'?t help\b",
        r"\bI'?m not able to\b",
        r"\bI cannot\b",
        r"\bI'?m unable to\b",
        r"\bI don'?t think I should\b",
        r"\bI'?m sorry,? but I\b",
        r"\bI will not\b",
        r"\bI won'?t\b",
        r"\bI'?m not going to\b",
        r"\bas an AI(?: language model)?,? I\b",
        r"\bI'?m not comfortable\b",
        r"\bthat'?s not something I can\b",
        r"\bI must decline\b",
        r"\bI have to refuse\b",
    ]
]

# Minimum word count for a response to be considered "substantive"
MIN_SUBSTANTIVE_WORDS = 20


class RefusalBiasDetector(BaseDetector):
    """Detect over-refusal bias in preference labeling.

    Score is 1.0 when chosen is a refusal AND rejected is substantive,
    0.0 otherwise.  This is a binary detector — the flagging threshold
    should be set to 0.5 so any score of 1.0 gets flagged.
    """

    name = "refusal_bias"
    description = "Flags pairs where chosen refuses but rejected is helpful"

    def __init__(
        self,
        min_substantive_words: int = MIN_SUBSTANTIVE_WORDS,
    ) -> None:
        """
        Args:
            min_substantive_words: Word count threshold for considering
                                   a response substantive.
        """
        self._min_substantive_words = min_substantive_words

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_refusal(text: str) -> bool:
        """Return True if *text* matches any known refusal pattern."""
        return any(pattern.search(text) for pattern in REFUSAL_PATTERNS)

    def is_substantive(self, text: str) -> bool:
        """Return True if *text* is long enough to be a real answer."""
        return len(text.split()) >= self._min_substantive_words

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def score(self, chosen_text: str, rejected_text: str) -> float:
        """Return 1.0 if chosen is a refusal and rejected is substantive."""
        if self.is_refusal(chosen_text) and self.is_substantive(rejected_text):
            return 1.0
        return 0.0

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

    def get_default_threshold_percentile(self) -> float:
        # Binary detector: flag all positives
        return 99.5

    def get_fixed_threshold(self) -> float | None:
        """Binary detector: use fixed threshold of 0.5 (any score of 1.0 is flagged)."""
        return 0.5

    def get_metadata(self, chosen_text: str, rejected_text: str) -> dict[str, Any]:
        matched = [
            p.pattern for p in REFUSAL_PATTERNS if p.search(chosen_text)
        ]
        return {
            "chosen_is_refusal": self.is_refusal(chosen_text),
            "rejected_is_substantive": self.is_substantive(rejected_text),
            "matched_patterns": matched,
            "rejected_word_count": len(rejected_text.split()),
        }
