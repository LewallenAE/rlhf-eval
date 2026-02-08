#!/usr/bin/env python3
"""
Unsafe Prompt Detector.

Flags prompts that contain toxic or unsafe content where neither
response may be valid for RLHF training.  When the user's request
itself is harmful, both the "chosen" and "rejected" labels become
unreliable — the pair should be reviewed or excluded.

Uses keyword patterns for fast, dependency-light detection.
Can be extended with a pre-trained toxicity classifier.
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

# Categories of unsafe content with associated patterns.
# Each category maps to a list of compiled regexes.
UNSAFE_CATEGORIES: dict[str, list[re.Pattern[str]]] = {
    "violence": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\bhow to (kill|murder|harm|hurt|attack)\b",
            r"\bmak(?:e|ing) (a )?(bomb|explosive|weapon)\b",
            r"\bmass (shooting|casualt)",
            r"\btorture\b",
        ]
    ],
    "illegal_activity": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(how to )?(hack|steal|rob|break into)\b",
            r"\bmak(?:e|ing) (meth|cocaine|drugs)\b",
            r"\bforg(?:e|ing) (documents?|identit)",
            r"\bmoney launder",
        ]
    ],
    "self_harm": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\bhow to (commit suicide|kill myself|end my life)\b",
            r"\bself[- ]harm\b",
            r"\bsuicid(e|al)\b",
        ]
    ],
    "csam": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\bchild (porn|abuse|exploit)",
            r"\bminor.{0,20}(sexual|nude|naked)",
        ]
    ],
    "hate_speech": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(racial|ethnic) (slur|supremac)",
            r"\bgenocide.{0,20}(justif|good|right)",
        ]
    ],
}


class UnsafePromptDetector(BaseDetector):
    """Flag prompts that contain unsafe or toxic content.

    Unlike other detectors that compare chosen vs rejected, this
    detector scores the **prompt** (passed as *chosen_text*) and
    ignores the response text.  In the pipeline, the caller should
    pass the prompt as *chosen_text*.

    Score is the count of distinct unsafe categories matched (0–5).
    Any score >= 1 is flagged.
    """

    name = "unsafe_prompt"
    description = "Flags prompts with toxic or unsafe content"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def detect_categories(text: str) -> list[str]:
        """Return list of unsafe category names matched in *text*."""
        matched: list[str] = []
        for category, patterns in UNSAFE_CATEGORIES.items():
            if any(p.search(text) for p in patterns):
                matched.append(category)
        return matched

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def score(self, chosen_text: str, rejected_text: str) -> float:
        """Return the number of unsafe categories found in the prompt.

        *chosen_text* is treated as the prompt.  *rejected_text* is
        ignored (kept for interface compatibility).
        """
        categories = self.detect_categories(chosen_text)
        return float(len(categories))

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
        """Flag if any unsafe category is matched (score >= 1)."""
        return score >= threshold

    def get_default_threshold_percentile(self) -> float:
        return 99.5

    def get_fixed_threshold(self) -> float | None:
        """Discrete detector: flag any prompt with >= 1 unsafe category."""
        return 1.0

    def get_metadata(self, chosen_text: str, rejected_text: str) -> dict[str, Any]:
        categories = self.detect_categories(chosen_text)
        return {
            "unsafe_categories": categories,
            "category_count": len(categories),
        }
