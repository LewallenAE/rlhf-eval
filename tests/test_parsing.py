#!/usr/bin/env python3
"""Comprehensive tests for rlhf_eval.utils.parsing and rlhf_eval.utils.stats."""

from __future__ import annotations

import numpy as np
import pytest

from rlhf_eval.utils.parsing import (
    all_assistant_responses,
    all_human_prompts,
    count_turns,
    last_assistant_response,
    last_human_prompt,
    parse_conversation,
)
from rlhf_eval.utils.stats import (
    compare_distributions,
    compute_percentile,
    compute_stats,
    find_threshold_cliff,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SINGLE_TURN = "Human: Hello\n\nAssistant: Hi there!"

MULTI_TURN = (
    "Human: What is Python?\n\n"
    "Assistant: Python is a programming language.\n\n"
    "Human: Is it easy to learn?\n\n"
    "Assistant: Yes, it has a gentle learning curve."
)

MULTI_TURN_THREE = (
    "Human: First question\n\n"
    "Assistant: First answer\n\n"
    "Human: Second question\n\n"
    "Assistant: Second answer\n\n"
    "Human: Third question\n\n"
    "Assistant: Third answer"
)

HUMAN_ONLY = "Human: Just a human turn here"

ASSISTANT_ONLY = "Assistant: Just an assistant turn here"


# ===================================================================
# parse_conversation
# ===================================================================


class TestParseConversation:
    def test_single_turn(self) -> None:
        result = parse_conversation(SINGLE_TURN)
        assert result == [("human", "Hello"), ("assistant", "Hi there!")]

    def test_multi_turn(self) -> None:
        result = parse_conversation(MULTI_TURN)
        assert len(result) == 4
        assert result[0] == ("human", "What is Python?")
        assert result[1] == ("assistant", "Python is a programming language.")
        assert result[2] == ("human", "Is it easy to learn?")
        assert result[3] == ("assistant", "Yes, it has a gentle learning curve.")

    def test_empty_string(self) -> None:
        assert parse_conversation("") == []

    def test_whitespace_only(self) -> None:
        assert parse_conversation("   \n\n  ") == []

    def test_no_speaker_tags(self) -> None:
        assert parse_conversation("Just some random text") == []

    def test_human_only(self) -> None:
        result = parse_conversation(HUMAN_ONLY)
        assert result == [("human", "Just a human turn here")]

    def test_assistant_only(self) -> None:
        result = parse_conversation(ASSISTANT_ONLY)
        assert result == [("assistant", "Just an assistant turn here")]

    def test_roles_are_lowercase(self) -> None:
        result = parse_conversation(SINGLE_TURN)
        for role, _ in result:
            assert role in ("human", "assistant")

    def test_content_is_stripped(self) -> None:
        text = "Human:   padded content   \n\nAssistant:   also padded   "
        result = parse_conversation(text)
        assert result[0][1] == "padded content"
        assert result[1][1] == "also padded"

    def test_content_with_internal_newlines(self) -> None:
        text = "Human: line one\nline two\nline three\n\nAssistant: response"
        result = parse_conversation(text)
        assert result[0] == ("human", "line one\nline two\nline three")
        assert result[1] == ("assistant", "response")

    def test_preamble_before_first_speaker_is_ignored(self) -> None:
        text = "Some preamble text\n\nHuman: hello\n\nAssistant: hi"
        result = parse_conversation(text)
        assert len(result) == 2
        assert result[0][0] == "human"


# ===================================================================
# last_assistant_response
# ===================================================================


class TestLastAssistantResponse:
    def test_single_turn(self) -> None:
        assert last_assistant_response(SINGLE_TURN) == "Hi there!"

    def test_multi_turn(self) -> None:
        assert (
            last_assistant_response(MULTI_TURN)
            == "Yes, it has a gentle learning curve."
        )

    def test_no_assistant(self) -> None:
        assert last_assistant_response(HUMAN_ONLY) == ""

    def test_empty_string(self) -> None:
        assert last_assistant_response("") == ""

    def test_three_turns(self) -> None:
        assert last_assistant_response(MULTI_TURN_THREE) == "Third answer"


# ===================================================================
# last_human_prompt
# ===================================================================


class TestLastHumanPrompt:
    def test_single_turn(self) -> None:
        assert last_human_prompt(SINGLE_TURN) == "Hello"

    def test_multi_turn(self) -> None:
        assert last_human_prompt(MULTI_TURN) == "Is it easy to learn?"

    def test_no_human(self) -> None:
        assert last_human_prompt(ASSISTANT_ONLY) == ""

    def test_empty_string(self) -> None:
        assert last_human_prompt("") == ""

    def test_three_turns(self) -> None:
        assert last_human_prompt(MULTI_TURN_THREE) == "Third question"


# ===================================================================
# all_assistant_responses
# ===================================================================


class TestAllAssistantResponses:
    def test_multi_turn(self) -> None:
        result = all_assistant_responses(MULTI_TURN)
        assert result == (
            "Python is a programming language.\n"
            "Yes, it has a gentle learning curve."
        )

    def test_no_assistant(self) -> None:
        assert all_assistant_responses(HUMAN_ONLY) == ""

    def test_empty_string(self) -> None:
        assert all_assistant_responses("") == ""

    def test_single_assistant(self) -> None:
        assert all_assistant_responses(SINGLE_TURN) == "Hi there!"


# ===================================================================
# all_human_prompts
# ===================================================================


class TestAllHumanPrompts:
    def test_multi_turn(self) -> None:
        result = all_human_prompts(MULTI_TURN)
        assert result == "What is Python?\nIs it easy to learn?"

    def test_no_human(self) -> None:
        assert all_human_prompts(ASSISTANT_ONLY) == ""

    def test_empty_string(self) -> None:
        assert all_human_prompts("") == ""


# ===================================================================
# count_turns
# ===================================================================


class TestCountTurns:
    def test_single_turn(self) -> None:
        assert count_turns(SINGLE_TURN) == {"human": 1, "assistant": 1}

    def test_multi_turn(self) -> None:
        assert count_turns(MULTI_TURN) == {"human": 2, "assistant": 2}

    def test_three_turns(self) -> None:
        assert count_turns(MULTI_TURN_THREE) == {"human": 3, "assistant": 3}

    def test_empty(self) -> None:
        assert count_turns("") == {"human": 0, "assistant": 0}

    def test_human_only(self) -> None:
        assert count_turns(HUMAN_ONLY) == {"human": 1, "assistant": 0}

    def test_assistant_only(self) -> None:
        assert count_turns(ASSISTANT_ONLY) == {"human": 0, "assistant": 1}


# ===================================================================
# compute_stats
# ===================================================================


class TestComputeStats:
    def test_basic(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_stats(values)
        assert result["mean"] == pytest.approx(3.0)
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(5.0)
        assert result["p50"] == pytest.approx(3.0)

    def test_single_value(self) -> None:
        result = compute_stats(np.array([42.0]))
        assert result["mean"] == pytest.approx(42.0)
        assert result["std"] == pytest.approx(0.0)
        assert result["min"] == pytest.approx(42.0)
        assert result["max"] == pytest.approx(42.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_stats(np.array([]))

    def test_all_keys_present(self) -> None:
        result = compute_stats(np.arange(100, dtype=float))
        for key in ("mean", "std", "min", "max", "p50", "p90", "p95", "p99", "p99_5"):
            assert key in result


# ===================================================================
# compute_percentile
# ===================================================================


class TestComputePercentile:
    def test_median(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_percentile(values, 50) == pytest.approx(3.0)

    def test_extremes(self) -> None:
        values = np.arange(101, dtype=float)
        assert compute_percentile(values, 0) == pytest.approx(0.0)
        assert compute_percentile(values, 100) == pytest.approx(100.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_percentile(np.array([]), 50)

    def test_out_of_range_raises(self) -> None:
        values = np.array([1.0])
        with pytest.raises(ValueError, match="between 0 and 100"):
            compute_percentile(values, 101)
        with pytest.raises(ValueError, match="between 0 and 100"):
            compute_percentile(values, -1)


# ===================================================================
# find_threshold_cliff
# ===================================================================


class TestFindThresholdCliff:
    def test_obvious_cliff(self) -> None:
        # 999 values near 0, then 1 outlier at 100
        values = np.concatenate([np.zeros(999), np.array([100.0])])
        pct, val, jump = find_threshold_cliff(values)
        # The cliff should be at the highest percentile where the outlier appears
        assert jump > 0
        assert val > 0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            find_threshold_cliff(np.array([]))

    def test_too_few_percentiles_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            find_threshold_cliff(np.array([1.0, 2.0]), percentiles=[50.0])

    def test_returns_three_floats(self) -> None:
        values = np.arange(1000, dtype=float)
        result = find_threshold_cliff(values)
        assert len(result) == 3
        pct, val, jump = result
        assert isinstance(pct, float)
        assert isinstance(val, float)
        assert isinstance(jump, float)


# ===================================================================
# compare_distributions
# ===================================================================


class TestCompareDistributions:
    def test_identical_distributions(self) -> None:
        values = np.arange(100, dtype=float)
        result = compare_distributions(values, values)
        assert result["ks_statistic"] == pytest.approx(0.0)
        assert result["mean_diff"] == pytest.approx(0.0)

    def test_different_distributions(self) -> None:
        a = np.zeros(100)
        b = np.ones(100)
        result = compare_distributions(a, b)
        assert result["ks_statistic"] == pytest.approx(1.0)
        assert result["mean_diff"] == pytest.approx(-1.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compare_distributions(np.array([]), np.array([1.0]))
        with pytest.raises(ValueError, match="empty"):
            compare_distributions(np.array([1.0]), np.array([]))

    def test_all_keys_present(self) -> None:
        a = np.arange(50, dtype=float)
        b = np.arange(50, 100, dtype=float)
        result = compare_distributions(a, b)
        for key in ("ks_statistic", "p_value", "mean_diff", "std_diff"):
            assert key in result
