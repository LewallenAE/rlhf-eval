#!/usr/bin/env python3
"""
Quality detectors for RLHF preference pairs.

Each detector scores a (chosen, rejected) pair and flags problematic
examples based on configurable thresholds.

Detectors with heavy dependencies (sentence-transformers, textstat) are
imported lazily so that missing optional packages only break the
detectors that need them, not the entire package.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import BaseDetector

if TYPE_CHECKING:
    from .length_ratio import LengthRatioDetector as LengthRatioDetector
    from .readability import ReadabilityMismatchDetector as ReadabilityMismatchDetector
    from .refusal_bias import RefusalBiasDetector as RefusalBiasDetector
    from .repetition import RepetitionDetector as RepetitionDetector
    from .semantic_similarity import SemanticSimilarityDetector as SemanticSimilarityDetector
    from .unsafe_prompt import UnsafePromptDetector as UnsafePromptDetector

__all__ = [
    "BaseDetector",
    "LengthRatioDetector",
    "ReadabilityMismatchDetector",
    "RefusalBiasDetector",
    "RepetitionDetector",
    "SemanticSimilarityDetector",
    "UnsafePromptDetector",
    "DETECTOR_REGISTRY",
]

logger = logging.getLogger(__name__)


def _build_registry() -> dict[str, type[BaseDetector]]:
    """Build detector registry with lazy imports.

    Each detector is imported individually so a missing optional
    dependency (e.g. sentence-transformers) only skips that one
    detector instead of crashing the whole package.
    """
    registry: dict[str, type[BaseDetector]] = {}

    _imports: list[tuple[str, str, str]] = [
        ("rlhf_eval.detectors.semantic_similarity", "SemanticSimilarityDetector", "semantic_similarity"),
        ("rlhf_eval.detectors.readability", "ReadabilityMismatchDetector", "readability_mismatch"),
        ("rlhf_eval.detectors.repetition", "RepetitionDetector", "repetition"),
        ("rlhf_eval.detectors.length_ratio", "LengthRatioDetector", "length_ratio"),
        ("rlhf_eval.detectors.refusal_bias", "RefusalBiasDetector", "refusal_bias"),
        ("rlhf_eval.detectors.unsafe_prompt", "UnsafePromptDetector", "unsafe_prompt"),
    ]

    import importlib

    for module_path, class_name, detector_name in _imports:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            registry[detector_name] = cls
        except ImportError as exc:
            logger.debug(
                "Skipping detector '%s': %s", detector_name, exc
            )

    return registry


DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = _build_registry()
