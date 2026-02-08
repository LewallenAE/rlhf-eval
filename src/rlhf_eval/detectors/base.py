#!/usr/bin/env python3
"""
Abstract base class for all quality detectors.

Every detector scores preference pairs (chosen vs rejected responses)
and flags problematic examples based on configurable thresholds.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
import logging
from abc import ABC, abstractmethod
from typing import Any

# ----------------- Third Party Library -----------------
import numpy as np

# ----------------- Application Imports -----------------


# ----------------- Module-level Configuration -----------------
logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract base class for all quality detectors.

    Subclasses must implement ``score`` and ``score_batch``.  The base
    class provides default threshold logic (flag when score >= threshold)
    and a hook for detector-specific metadata.
    """

    name: str
    description: str

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def score(self, chosen_text: str, rejected_text: str) -> float:
        """Score a single preference pair.

        Args:
            chosen_text: The chosen/preferred response text.
            rejected_text: The rejected response text.

        Returns:
            Float score whose interpretation depends on the detector.
        """
        ...

    @abstractmethod
    def score_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Score multiple pairs efficiently.

        Args:
            pairs: List of (chosen_text, rejected_text) tuples.
            batch_size: Processing batch size for the underlying model.

        Returns:
            1-D array of scores, same length as *pairs*.
        """
        ...

    # ------------------------------------------------------------------
    # Flagging helpers
    # ------------------------------------------------------------------

    def is_flagged(self, score: float, threshold: float) -> bool:
        """Determine if a score should be flagged.

        Default: flag when ``score >= threshold``.
        Override for detectors where *lower* is worse.
        """
        return score >= threshold

    def get_default_threshold_percentile(self) -> float:
        """Return the default percentile used to compute the flagging threshold."""
        return 99.5

    def get_fixed_threshold(self) -> float | None:
        """Return a fixed threshold instead of using percentile-based computation.

        Returns ``None`` by default, meaning the pipeline should compute
        the threshold from a percentile of the score distribution.
        Override in binary/discrete detectors where percentile-based
        thresholds don't work (e.g. scores are only 0.0 or 1.0).
        """
        return None

    # ------------------------------------------------------------------
    # Metadata hook
    # ------------------------------------------------------------------

    def get_metadata(self, chosen_text: str, rejected_text: str) -> dict[str, Any]:
        """Return detector-specific metadata about a pair.

        Override in subclasses to provide extra context (e.g. individual
        readability scores, matched refusal patterns, etc.).
        """
        return {}
