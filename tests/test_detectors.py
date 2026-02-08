#!/usr/bin/env python3
"""Tests for CPU-only quality detectors.

Covers: RepetitionDetector, LengthRatioDetector, RefusalBiasDetector, UnsafePromptDetector.

SemanticSimilarityDetector and ReadabilityMismatchDetector require external deps
(sentence-transformers, textstat) and are tested separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlhf_eval.detectors.repetition import RepetitionDetector
from rlhf_eval.detectors.length_ratio import LengthRatioDetector
from rlhf_eval.detectors.refusal_bias import RefusalBiasDetector
from rlhf_eval.detectors.unsafe_prompt import UnsafePromptDetector
from rlhf_eval.detectors.readability import ReadabilityMismatchDetector
from rlhf_eval.detectors.semantic_similarity import SemanticSimilarityDetector


# ===================================================================
# RepetitionDetector
# ===================================================================


class TestRepetitionDetector:
    def setup_method(self) -> None:
        self.detector = RepetitionDetector()

    def test_name(self) -> None:
        assert self.detector.name == "repetition"

    def test_normal_text_high_ratio(self) -> None:
        chosen = "The quick brown fox jumps over the lazy dog"
        rejected = "A different sentence with many unique words here"
        score = self.detector.score(chosen, rejected)
        assert score > 0.5

    def test_degenerate_text_low_ratio(self) -> None:
        chosen = "fuck off fuck off fuck off fuck off fuck off"
        rejected = "A normal response with varied vocabulary"
        score = self.detector.score(chosen, rejected)
        assert score < 0.4

    def test_both_degenerate_returns_min(self) -> None:
        chosen = "cat cat cat cat cat"
        rejected = "dog dog dog dog dog"
        score = self.detector.score(chosen, rejected)
        assert score == pytest.approx(0.2)

    def test_empty_text_returns_one(self) -> None:
        score = self.detector.score("", "some text")
        assert score == pytest.approx(1.0)

    def test_is_flagged_below_threshold(self) -> None:
        assert self.detector.is_flagged(0.2, 0.3) is True

    def test_is_flagged_above_threshold(self) -> None:
        assert self.detector.is_flagged(0.8, 0.3) is False

    def test_score_batch(self) -> None:
        pairs = [
            ("hello world hello world", "good response"),
            ("unique words in this sentence here", "another good response"),
        ]
        scores = self.detector.score_batch(pairs)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2

    def test_score_batch_empty(self) -> None:
        scores = self.detector.score_batch([])
        assert len(scores) == 0

    def test_unique_word_ratio(self) -> None:
        assert RepetitionDetector.unique_word_ratio("a b c d e") == pytest.approx(1.0)
        assert RepetitionDetector.unique_word_ratio("a a a a a") == pytest.approx(0.2)

    def test_ngram_repetition_rate(self) -> None:
        assert RepetitionDetector.ngram_repetition_rate("a b c a b c a b c", n=3) > 0
        assert RepetitionDetector.ngram_repetition_rate("a b c d e f g h i", n=3) == 0.0

    def test_get_metadata(self) -> None:
        meta = self.detector.get_metadata("cat cat cat", "dog cat bird")
        assert "chosen_unique_ratio" in meta
        assert "rejected_unique_ratio" in meta
        assert "chosen_ngram_rep" in meta
        assert "rejected_ngram_rep" in meta

    def test_get_default_threshold_percentile(self) -> None:
        assert self.detector.get_default_threshold_percentile() == 0.5


# ===================================================================
# LengthRatioDetector
# ===================================================================


class TestLengthRatioDetector:
    def setup_method(self) -> None:
        self.detector = LengthRatioDetector()

    def test_name(self) -> None:
        assert self.detector.name == "length_ratio"

    def test_equal_length(self) -> None:
        score = self.detector.score("one two three", "four five six")
        assert score == pytest.approx(1.0)

    def test_very_different_lengths(self) -> None:
        chosen = "yes"
        rejected = "This is a much longer response with many words in it"
        score = self.detector.score(chosen, rejected)
        assert score < 0.2

    def test_both_empty(self) -> None:
        score = self.detector.score("", "")
        assert score == pytest.approx(1.0)

    def test_is_flagged_below_threshold(self) -> None:
        assert self.detector.is_flagged(0.1, 0.5) is True

    def test_is_flagged_above_threshold(self) -> None:
        assert self.detector.is_flagged(0.8, 0.5) is False

    def test_score_batch(self) -> None:
        pairs = [
            ("short", "This is a much longer response"),
            ("equal length here", "also equal length"),
        ]
        scores = self.detector.score_batch(pairs)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
        assert scores[0] < scores[1]

    def test_get_metadata(self) -> None:
        meta = self.detector.get_metadata("one two", "one two three four")
        assert meta["chosen_words"] == 2
        assert meta["rejected_words"] == 4
        assert "ratio" in meta

    def test_get_default_threshold_percentile(self) -> None:
        assert self.detector.get_default_threshold_percentile() == 0.5


# ===================================================================
# RefusalBiasDetector
# ===================================================================


class TestRefusalBiasDetector:
    def setup_method(self) -> None:
        self.detector = RefusalBiasDetector()

    def test_name(self) -> None:
        assert self.detector.name == "refusal_bias"

    def test_refusal_chosen_substantive_rejected(self) -> None:
        chosen = "I can't help with that request."
        rejected = "Here is a detailed explanation of how to do that task. " * 5
        score = self.detector.score(chosen, rejected)
        assert score == pytest.approx(1.0)

    def test_both_substantive(self) -> None:
        chosen = "Here is a good response with enough words to be substantive really."
        rejected = "Here is another good response with different words for variety sake."
        score = self.detector.score(chosen, rejected)
        assert score == pytest.approx(0.0)

    def test_refusal_but_rejected_too_short(self) -> None:
        chosen = "I cannot help with that."
        rejected = "Sure, here you go."
        score = self.detector.score(chosen, rejected)
        assert score == pytest.approx(0.0)

    def test_is_refusal_patterns(self) -> None:
        assert RefusalBiasDetector.is_refusal("I can't help with that")
        assert RefusalBiasDetector.is_refusal("I'm not able to do that")
        assert RefusalBiasDetector.is_refusal("I cannot provide that information")
        assert RefusalBiasDetector.is_refusal("As an AI, I don't think")
        assert RefusalBiasDetector.is_refusal("I will not do that")
        assert not RefusalBiasDetector.is_refusal("Sure, here is the answer")

    def test_score_batch(self) -> None:
        pairs = [
            ("I can't help with that.", "Here is a very detailed answer. " * 5),
            ("Here is a helpful response.", "Another helpful response."),
        ]
        scores = self.detector.score_batch(pairs)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)

    def test_get_metadata(self) -> None:
        meta = self.detector.get_metadata(
            "I can't help with that.",
            "Here is the answer to your question with many words. " * 3,
        )
        assert meta["chosen_is_refusal"] is True
        assert meta["rejected_is_substantive"] is True
        assert len(meta["matched_patterns"]) > 0

    def test_get_default_threshold_percentile(self) -> None:
        assert self.detector.get_default_threshold_percentile() == 99.5


# ===================================================================
# UnsafePromptDetector
# ===================================================================


class TestUnsafePromptDetector:
    def setup_method(self) -> None:
        self.detector = UnsafePromptDetector()

    def test_name(self) -> None:
        assert self.detector.name == "unsafe_prompt"

    def test_safe_prompt(self) -> None:
        score = self.detector.score("What is the capital of France?", "")
        assert score == pytest.approx(0.0)

    def test_violence_detected(self) -> None:
        score = self.detector.score("how to kill someone", "")
        assert score >= 1.0

    def test_illegal_activity_detected(self) -> None:
        score = self.detector.score("how to hack into a system", "")
        assert score >= 1.0

    def test_multiple_categories(self) -> None:
        score = self.detector.score(
            "how to kill someone and how to hack into their computer", ""
        )
        assert score >= 2.0

    def test_is_flagged_any_match(self) -> None:
        assert self.detector.is_flagged(1.0, 1.0) is True
        assert self.detector.is_flagged(0.0, 1.0) is False

    def test_score_batch(self) -> None:
        pairs = [
            ("What is Python?", ""),
            ("how to make a bomb", ""),
        ]
        scores = self.detector.score_batch(pairs)
        assert scores[0] == pytest.approx(0.0)
        assert scores[1] >= 1.0

    def test_detect_categories(self) -> None:
        cats = UnsafePromptDetector.detect_categories("how to kill and how to hack into")
        assert "violence" in cats
        assert "illegal_activity" in cats

    def test_get_metadata(self) -> None:
        meta = self.detector.get_metadata("how to kill someone", "")
        assert "unsafe_categories" in meta
        assert "category_count" in meta
        assert meta["category_count"] >= 1

    def test_get_default_threshold_percentile(self) -> None:
        assert self.detector.get_default_threshold_percentile() == 99.5


# ===================================================================
# ReadabilityMismatchDetector
# ===================================================================


class TestReadabilityMismatchDetector:
    def setup_method(self) -> None:
        self.detector = ReadabilityMismatchDetector()

    def test_name(self) -> None:
        assert self.detector.name == "readability_mismatch"

    def test_similar_readability(self) -> None:
        chosen = "The quick brown fox jumps over the lazy dog in the sunny meadow."
        rejected = "A fast brown fox leaps over a sleepy dog in the warm field."
        score = self.detector.score(chosen, rejected)
        # Similar grade levels → score near 0
        assert abs(score) < 5.0

    def test_different_readability(self) -> None:
        # Complex chosen, simple rejected
        chosen = (
            "The epistemological ramifications of quantum superposition necessitate "
            "a fundamental reconceptualization of deterministic philosophical frameworks "
            "and their ontological presuppositions regarding observable phenomena."
        )
        rejected = "The cat sat on a mat. The dog ran in the sun. It was a fun day."
        score = self.detector.score(chosen, rejected)
        # chosen is harder to read → higher grade level → positive score
        assert score > 0

    def test_short_text_returns_zero(self) -> None:
        score = self.detector.score("Too short", "Also short")
        assert score == pytest.approx(0.0)

    def test_empty_text_returns_zero(self) -> None:
        score = self.detector.score("", "Some longer text with enough words for testing.")
        assert score == pytest.approx(0.0)

    def test_score_batch(self) -> None:
        pairs = [
            ("Short text with enough words for a readability test easily.",
             "Another text with sufficient words for a readability test easily."),
            ("Very simple easy text with lots of short words for a good test.",
             "Extraordinarily sesquipedalian lexicographic constructions obfuscate comprehension immensely throughout."),
        ]
        scores = self.detector.score_batch(pairs)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2

    def test_get_metadata(self) -> None:
        meta = self.detector.get_metadata(
            "This is a simple sentence with enough words for a score.",
            "This is another simple sentence with enough words for a score.",
        )
        assert "chosen_grade_level" in meta
        assert "rejected_grade_level" in meta
        assert "grade_difference" in meta

    def test_get_default_threshold_percentile(self) -> None:
        assert self.detector.get_default_threshold_percentile() == 99.5

    def test_get_fixed_threshold_is_none(self) -> None:
        # Readability uses percentile-based threshold, not fixed
        assert self.detector.get_fixed_threshold() is None


# ===================================================================
# SemanticSimilarityDetector
# ===================================================================


@pytest.fixture(scope="module")
def sem_detector():
    """Load semantic similarity model once for all tests in this module."""
    return SemanticSimilarityDetector(device="cpu")


class TestSemanticSimilarityDetector:
    def test_name(self, sem_detector) -> None:
        assert sem_detector.name == "semantic_similarity"

    def test_identical_texts_high_similarity(self, sem_detector) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        score = sem_detector.score(text, text)
        assert score > 0.99

    def test_similar_texts_high_similarity(self, sem_detector) -> None:
        chosen = "Paris is the capital of France."
        rejected = "The capital city of France is Paris."
        score = sem_detector.score(chosen, rejected)
        assert score > 0.8

    def test_different_texts_lower_similarity(self, sem_detector) -> None:
        chosen = "The weather is sunny and warm today."
        rejected = "Quantum computing uses qubits for parallel computation."
        score = sem_detector.score(chosen, rejected)
        assert score < 0.5

    def test_score_batch(self, sem_detector) -> None:
        pairs = [
            ("Hello world", "Hello world"),
            ("The sky is blue", "Computers process data"),
        ]
        scores = sem_detector.score_batch(pairs)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
        assert scores[0] > scores[1]  # identical > unrelated

    def test_score_batch_empty(self, sem_detector) -> None:
        scores = sem_detector.score_batch([])
        assert len(scores) == 0

    def test_get_metadata(self, sem_detector) -> None:
        meta = sem_detector.get_metadata("Hello world", "Hello world")
        assert "similarity" in meta
        assert "model" in meta
        assert meta["similarity"] > 0.99
        assert meta["model"] == "all-MiniLM-L6-v2"

    def test_get_default_threshold_percentile(self, sem_detector) -> None:
        assert sem_detector.get_default_threshold_percentile() == 99.5

    def test_get_fixed_threshold_is_none(self, sem_detector) -> None:
        assert sem_detector.get_fixed_threshold() is None

    def test_score_returns_float(self, sem_detector) -> None:
        score = sem_detector.score("test", "test")
        assert isinstance(score, float)


# ===================================================================
# BaseDetector interface compliance
# ===================================================================


class TestBaseDetectorCompliance:
    """Verify all detectors implement the full BaseDetector interface."""

    DETECTORS = [
        RepetitionDetector(),
        LengthRatioDetector(),
        RefusalBiasDetector(),
        UnsafePromptDetector(),
        ReadabilityMismatchDetector(),
    ]

    @pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
    def test_has_name(self, detector) -> None:
        assert isinstance(detector.name, str)
        assert len(detector.name) > 0

    @pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
    def test_has_description(self, detector) -> None:
        assert isinstance(detector.description, str)
        assert len(detector.description) > 0

    @pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
    def test_score_returns_float(self, detector) -> None:
        score = detector.score("hello world", "goodbye world")
        assert isinstance(score, float)

    @pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
    def test_score_batch_returns_ndarray(self, detector) -> None:
        pairs = [("hello", "world"), ("foo", "bar")]
        result = detector.score_batch(pairs)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    @pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
    def test_is_flagged_returns_bool(self, detector) -> None:
        result = detector.is_flagged(0.5, 0.5)
        assert isinstance(result, bool)

    @pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
    def test_get_default_threshold_percentile_returns_float(self, detector) -> None:
        result = detector.get_default_threshold_percentile()
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0

    @pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
    def test_get_metadata_returns_dict(self, detector) -> None:
        result = detector.get_metadata("hello", "world")
        assert isinstance(result, dict)
