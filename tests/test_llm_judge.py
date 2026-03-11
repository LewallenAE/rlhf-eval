#!/usr/bin/env python3
"""Tests for LLMJudgeDetector.

These tests use mocking to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


class TestLLMJudgeDetector:
    """Tests for LLMJudgeDetector using mocked OpenAI responses."""

    @pytest.fixture
    def detector(self):
        """Create detector with mocked OpenAI client."""
        with patch("openai.AsyncOpenAI"):
            from rlhf_eval.detectors.llm_judge import LLMJudgeDetector
            return LLMJudgeDetector(api_key="test-key")

    def test_name(self, detector) -> None:
        from rlhf_eval.detectors.llm_judge import LLMJudgeDetector
        assert LLMJudgeDetector.name == "llm_judge"

    def test_description(self, detector) -> None:
        assert len(detector.description) > 0

    def test_is_flagged_below_threshold(self, detector) -> None:
        assert detector.is_flagged(2.9, 3.0) is True

    def test_is_flagged_above_threshold(self, detector) -> None:
        assert detector.is_flagged(3.0, 3.0) is False

    def test_is_flagged_equal_threshold(self, detector) -> None:
        assert detector.is_flagged(3.0, 3.0) is False

    def test_get_fixed_threshold(self, detector) -> None:
        assert detector.get_fixed_threshold() == 3.0

    def test_get_default_threshold_percentile(self, detector) -> None:
        pct = detector.get_default_threshold_percentile()
        assert isinstance(pct, float)
        assert 0.0 <= pct <= 100.0

    def test_score_batch_empty(self, detector) -> None:
        result = detector.score_batch([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_score_batch_returns_ndarray(self, detector) -> None:
        """Test score_batch with mocked async judge."""
        mock_result = {
            "helpfulness_delta": 4,
            "honest_preference": 3,
            "label_confidence": 5,
            "min_score": 3.0,
            "judge_flagged": False,
        }

        with patch.object(detector, "_judge_batch_async", new_callable=AsyncMock) as mock_judge:
            mock_judge.return_value = [mock_result, mock_result]
            pairs = [
                ("What is Python?", "Python is a programming language.", "Python is a snake."),
                ("What is 2+2?", "It is 4.", "It is 5."),
            ]
            scores = detector.score_batch(pairs)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
        assert all(s == pytest.approx(3.0) for s in scores)

    def test_score_batch_flagged_pair(self, detector) -> None:
        """Test that a low-scoring pair is properly reflected in the score."""
        flagged_result = {
            "helpfulness_delta": 2,
            "honest_preference": 2,
            "label_confidence": 1,
            "min_score": 1.0,
            "judge_flagged": True,
        }
        good_result = {
            "helpfulness_delta": 5,
            "honest_preference": 4,
            "label_confidence": 5,
            "min_score": 4.0,
            "judge_flagged": False,
        }

        with patch.object(detector, "_judge_batch_async", new_callable=AsyncMock) as mock_judge:
            mock_judge.return_value = [flagged_result, good_result]
            pairs = [
                ("prompt1", "chosen1", "rejected1"),
                ("prompt2", "chosen2", "rejected2"),
            ]
            scores = detector.score_batch(pairs)

        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(4.0)
        assert detector.is_flagged(scores[0], 3.0) is True
        assert detector.is_flagged(scores[1], 3.0) is False

    def test_get_metadata_after_score_batch(self, detector) -> None:
        """Test that metadata is cached after score_batch."""
        mock_result = {
            "helpfulness_delta": 4,
            "honest_preference": 3,
            "label_confidence": 5,
            "min_score": 3.0,
            "judge_flagged": False,
        }

        with patch.object(detector, "_judge_batch_async", new_callable=AsyncMock) as mock_judge:
            mock_judge.return_value = [mock_result]
            prompt = "What is Python?"
            chosen = "Python is a programming language."
            pairs = [(prompt, chosen, "Python is a snake.")]
            detector.score_batch(pairs)

        meta = detector.get_metadata(prompt, chosen)
        assert "helpfulness_delta" in meta
        assert "honest_preference" in meta
        assert "label_confidence" in meta

    def test_get_metadata_empty_when_not_cached(self, detector) -> None:
        """get_metadata returns empty dict for uncached pairs."""
        meta = detector.get_metadata("unknown prompt", "unknown chosen")
        assert meta == {}

    def test_score_batch_handles_errors_gracefully(self, detector) -> None:
        """score_batch should fall back to default 3.0 on API errors."""
        error_result = {
            "helpfulness_delta": 3,
            "honest_preference": 3,
            "label_confidence": 3,
            "min_score": 3.0,
            "judge_flagged": False,
            "error": "API timeout",
        }

        with patch.object(detector, "_judge_batch_async", new_callable=AsyncMock) as mock_judge:
            mock_judge.return_value = [error_result]
            scores = detector.score_batch([("p", "c", "r")])

        assert scores[0] == pytest.approx(3.0)
        assert detector.is_flagged(scores[0], 3.0) is False


class TestLLMJudgeBaseCompliance:
    """Verify LLMJudgeDetector implements BaseDetector interface."""

    @pytest.fixture
    def detector(self):
        with patch("openai.AsyncOpenAI"):
            from rlhf_eval.detectors.llm_judge import LLMJudgeDetector
            return LLMJudgeDetector(api_key="test-key")

    def test_has_name_attr(self, detector) -> None:
        assert isinstance(detector.name, str)
        assert len(detector.name) > 0

    def test_has_description_attr(self, detector) -> None:
        assert isinstance(detector.description, str)
        assert len(detector.description) > 0

    def test_is_flagged_returns_bool(self, detector) -> None:
        result = detector.is_flagged(2.5, 3.0)
        assert isinstance(result, bool)

    def test_get_fixed_threshold_returns_float(self, detector) -> None:
        result = detector.get_fixed_threshold()
        assert isinstance(result, float)

    def test_score_batch_empty_returns_ndarray(self, detector) -> None:
        result = detector.score_batch([])
        assert isinstance(result, np.ndarray)
