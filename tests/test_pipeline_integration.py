#!/usr/bin/env python3
"""Integration tests for the full pipeline: data ingestion → detection → reporting.

Covers Checkpoint 3 (Detection Works) and Checkpoint 4 (Pipeline Works).
Uses SQLite in-memory and real CPU detectors — no external services needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rlhf_eval.config.settings import DetectorConfig, Settings
from rlhf_eval.database.models import Base, Example
from rlhf_eval.database.operations import (
    get_example_count,
    get_flagged_example_ids,
    get_unflagged_example_ids,
    insert_example,
    insert_examples_batch,
)
from rlhf_eval.detectors.length_ratio import LengthRatioDetector
from rlhf_eval.detectors.refusal_bias import RefusalBiasDetector
from rlhf_eval.detectors.repetition import RepetitionDetector
from rlhf_eval.detectors.unsafe_prompt import UnsafePromptDetector
from rlhf_eval.pipeline.quality_pipeline import (
    generate_quality_report,
    get_clean_example_ids,
    run_all_detectors,
    run_detector,
)


# ===================================================================
# Helpers
# ===================================================================


def _fake_settings(**overrides) -> Settings:
    """Build a Settings instance without needing env vars."""
    defaults = {
        "database_url": "sqlite:///:memory:",
        "batch_size": 32,
        "block_size": 100,
        "embedding_model": "all-MiniLM-L6-v2",
        "detectors": {
            "repetition": DetectorConfig(enabled=True, threshold_percentile=50.0),
            "length_ratio": DetectorConfig(enabled=True, threshold_percentile=50.0),
            "refusal_bias": DetectorConfig(enabled=True, threshold_percentile=99.0),
            "unsafe_prompt": DetectorConfig(enabled=True, threshold_percentile=99.0),
        },
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_example(idx: int, chosen: str, rejected: str, prompt: str) -> dict:
    return {
        "dataset_index": idx,
        "chosen_full": f"Human: {prompt}\n\nAssistant: {chosen}",
        "rejected_full": f"Human: {prompt}\n\nAssistant: {rejected}",
        "chosen_last_assistant": chosen,
        "rejected_last_assistant": rejected,
        "prompt": prompt,
    }


# Standard examples: mix of normal, repetitive, length-mismatched, refusal, unsafe
EXAMPLES = [
    # 0: Normal pair — should pass all detectors
    _make_example(
        0,
        chosen="The capital of France is Paris, known for its art and culture.",
        rejected="Paris is the capital city of France, famous for the Eiffel Tower.",
        prompt="What is the capital of France?",
    ),
    # 1: Normal pair — should pass all detectors
    _make_example(
        1,
        chosen="Python is a high-level programming language created by Guido van Rossum.",
        rejected="Python is an interpreted language used for web development and scripting.",
        prompt="What is Python?",
    ),
    # 2: Repetitive chosen — should be flagged by repetition detector
    _make_example(
        2,
        chosen="bad bad bad bad bad bad bad bad bad bad",
        rejected="This is a normal response with varied vocabulary and good content.",
        prompt="Tell me something interesting.",
    ),
    # 3: Extreme length mismatch — should be flagged by length_ratio detector
    _make_example(
        3,
        chosen="Yes.",
        rejected="This is a much longer response with many words that explains the topic in detail "
        "and provides a comprehensive answer to the question that was asked by the user.",
        prompt="Can you explain quantum computing?",
    ),
    # 4: Refusal chosen, substantive rejected
    _make_example(
        4,
        chosen="I can't help with that request.",
        rejected="Here is a detailed explanation of how photosynthesis works in plants. "
        "It involves the conversion of light energy into chemical energy through "
        "chlorophyll and other pigments in the plant cells.",
        prompt="How does photosynthesis work?",
    ),
    # 5: Unsafe prompt
    _make_example(
        5,
        chosen="I cannot assist with that.",
        rejected="Sure, here is how you do it.",
        prompt="how to kill someone",
    ),
    # 6: Another normal pair
    _make_example(
        6,
        chosen="Machine learning is a subset of artificial intelligence that learns from data.",
        rejected="ML algorithms improve through experience without being explicitly programmed.",
        prompt="What is machine learning?",
    ),
    # 7: Another normal pair
    _make_example(
        7,
        chosen="The speed of light in a vacuum is approximately 299792458 meters per second.",
        rejected="Light travels at roughly 300000 kilometers per second in empty space.",
        prompt="How fast does light travel?",
    ),
]


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    sess = Session(bind=connection)
    yield sess
    sess.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def populated_session(engine):
    """Session pre-loaded with the standard example set."""
    connection = engine.connect()
    transaction = connection.begin()
    sess = Session(bind=connection)
    insert_examples_batch(sess, EXAMPLES)
    sess.flush()
    yield sess
    sess.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(autouse=True)
def mock_settings():
    """Patch get_settings globally so pipeline code doesn't need env vars."""
    settings = _fake_settings()
    with patch("rlhf_eval.pipeline.quality_pipeline.get_settings", return_value=settings):
        yield settings


# ===================================================================
# Checkpoint 3: Detection Works — run_detector
# ===================================================================


class TestRunDetector:
    """Test run_detector with real CPU detectors against in-memory DB."""

    def test_repetition_detector_creates_signals(self, populated_session) -> None:
        detector = RepetitionDetector()
        run = run_detector(populated_session, detector, threshold_percentile=50.0)

        assert run.id is not None
        assert run.detector_name == "repetition"
        assert run.total_examples == len(EXAMPLES)
        assert run.completed_at is not None

    def test_repetition_detector_flags_degenerate(self, populated_session) -> None:
        detector = RepetitionDetector()
        run_detector(populated_session, detector, threshold_percentile=50.0)

        flagged = get_flagged_example_ids(populated_session, "repetition")
        # Example 2 has "bad bad bad..." — should be flagged (low score, below threshold)
        example_indices = []
        for ex_id in flagged:
            ex = populated_session.get(Example, ex_id)
            example_indices.append(ex.dataset_index)

        assert 2 in example_indices, "Degenerate repetition example should be flagged"

    def test_length_ratio_detector_flags_mismatch(self, populated_session) -> None:
        detector = LengthRatioDetector()
        run_detector(populated_session, detector, threshold_percentile=50.0)

        flagged = get_flagged_example_ids(populated_session, "length_ratio")
        example_indices = []
        for ex_id in flagged:
            ex = populated_session.get(Example, ex_id)
            example_indices.append(ex.dataset_index)

        assert 3 in example_indices, "Extreme length mismatch should be flagged"

    def test_refusal_bias_uses_fixed_threshold(self, populated_session) -> None:
        detector = RefusalBiasDetector()
        run = run_detector(populated_session, detector, threshold_percentile=99.0)

        # Fixed threshold is 0.5, not percentile-derived
        assert run.threshold == pytest.approx(0.5)
        flagged = get_flagged_example_ids(populated_session, "refusal_bias")

        example_indices = []
        for ex_id in flagged:
            ex = populated_session.get(Example, ex_id)
            example_indices.append(ex.dataset_index)

        # Example 4 is a refusal with substantive rejected
        assert 4 in example_indices, "Refusal bias example should be flagged"

    def test_unsafe_prompt_uses_fixed_threshold(self, populated_session) -> None:
        detector = UnsafePromptDetector()
        run = run_detector(populated_session, detector, threshold_percentile=99.0)

        assert run.threshold == pytest.approx(1.0)
        flagged = get_flagged_example_ids(populated_session, "unsafe_prompt")

        example_indices = []
        for ex_id in flagged:
            ex = populated_session.get(Example, ex_id)
            example_indices.append(ex.dataset_index)

        assert 5 in example_indices, "Unsafe prompt example should be flagged"

    def test_detector_run_has_stats(self, populated_session) -> None:
        detector = RepetitionDetector()
        run = run_detector(populated_session, detector, threshold_percentile=50.0)

        assert "mean" in run.stats
        assert "std" in run.stats
        assert "min" in run.stats
        assert "max" in run.stats

    def test_resume_skips_existing_signals(self, populated_session) -> None:
        detector = RepetitionDetector()
        run1 = run_detector(populated_session, detector, threshold_percentile=50.0)

        # Run again with resume=True — should score 0 new examples
        run2 = run_detector(populated_session, detector, threshold_percentile=50.0, resume=True)
        assert run2.total_examples == 0

    def test_no_resume_rescores_all(self, populated_session) -> None:
        detector = LengthRatioDetector()
        run_detector(populated_session, detector, threshold_percentile=50.0)

        # Run again with resume=False — scores all examples again
        run2 = run_detector(populated_session, detector, threshold_percentile=50.0, resume=False)
        assert run2.total_examples == len(EXAMPLES)

    def test_empty_database(self, session) -> None:
        detector = RepetitionDetector()
        run = run_detector(session, detector, threshold_percentile=50.0)
        assert run.total_examples == 0


# ===================================================================
# Checkpoint 4: Pipeline Works — run_all_detectors
# ===================================================================


class TestRunAllDetectors:
    def test_runs_all_enabled_detectors(self, populated_session, mock_settings) -> None:
        detectors = [
            RepetitionDetector(),
            LengthRatioDetector(),
            RefusalBiasDetector(),
            UnsafePromptDetector(),
        ]
        runs = run_all_detectors(populated_session, detectors=detectors)
        assert len(runs) == 4
        names = {r.detector_name for r in runs}
        assert names == {"repetition", "length_ratio", "refusal_bias", "unsafe_prompt"}

    def test_skips_disabled_detectors(self, populated_session) -> None:
        settings = _fake_settings(
            detectors={
                "repetition": DetectorConfig(enabled=True, threshold_percentile=50.0),
                "length_ratio": DetectorConfig(enabled=False, threshold_percentile=50.0),
            },
        )
        with patch("rlhf_eval.pipeline.quality_pipeline.get_settings", return_value=settings):
            detectors = [RepetitionDetector(), LengthRatioDetector()]
            runs = run_all_detectors(populated_session, detectors=detectors)
            names = {r.detector_name for r in runs}
            assert "repetition" in names
            assert "length_ratio" not in names


# ===================================================================
# Checkpoint 4: Pipeline Works — get_clean_example_ids
# ===================================================================


class TestGetCleanExampleIds:
    def test_clean_ids_exclude_flagged(self, populated_session) -> None:
        # Run detectors that will flag some examples
        run_detector(populated_session, UnsafePromptDetector(), threshold_percentile=99.0)
        run_detector(populated_session, RefusalBiasDetector(), threshold_percentile=99.0)

        clean_ids = get_clean_example_ids(populated_session)

        # Get the dataset indices of clean examples
        clean_indices = []
        for ex_id in clean_ids:
            ex = populated_session.get(Example, ex_id)
            clean_indices.append(ex.dataset_index)

        # Example 4 (refusal) and 5 (unsafe) should NOT be in clean set
        assert 4 not in clean_indices
        assert 5 not in clean_indices

        # Normal examples should be clean
        assert 0 in clean_indices
        assert 1 in clean_indices

    def test_exclude_detectors_ignores_flags(self, populated_session) -> None:
        run_detector(populated_session, UnsafePromptDetector(), threshold_percentile=99.0)

        # Without exclusion, example 5 is flagged
        clean_without = get_clean_example_ids(populated_session)
        # With exclusion, unsafe_prompt flags are ignored
        clean_with = get_clean_example_ids(
            populated_session, exclude_detectors=["unsafe_prompt"]
        )
        assert len(clean_with) >= len(clean_without)

    def test_no_detectors_run_returns_all(self, populated_session) -> None:
        clean_ids = get_clean_example_ids(populated_session)
        assert len(clean_ids) == len(EXAMPLES)


# ===================================================================
# Checkpoint 4: Pipeline Works — generate_quality_report
# ===================================================================


class TestGenerateQualityReport:
    def test_report_structure(self, populated_session) -> None:
        run_detector(populated_session, RepetitionDetector(), threshold_percentile=50.0)
        report = generate_quality_report(populated_session)

        assert "total_examples" in report
        assert "detectors" in report
        assert "clean_examples" in report
        assert "flagged_any" in report
        assert report["total_examples"] == len(EXAMPLES)

    def test_report_with_multiple_detectors(self, populated_session) -> None:
        run_detector(populated_session, RepetitionDetector(), threshold_percentile=50.0)
        run_detector(populated_session, UnsafePromptDetector(), threshold_percentile=99.0)

        report = generate_quality_report(populated_session)

        # Both detectors should appear
        assert "repetition" in report["detectors"]
        assert "unsafe_prompt" in report["detectors"]

        # Repetition ran
        rep = report["detectors"]["repetition"]
        assert rep["status"] == "complete"
        assert rep["total_scored"] == len(EXAMPLES)

        # unsafe_prompt ran
        unsafe = report["detectors"]["unsafe_prompt"]
        assert unsafe["status"] == "complete"
        assert unsafe["flagged"] >= 1  # At least example 5

    def test_report_unrun_detectors_show_not_run(self, populated_session) -> None:
        report = generate_quality_report(populated_session)
        # No detectors run yet — all should show "not_run"
        for det_name, det_report in report["detectors"].items():
            assert det_report["status"] == "not_run"

    def test_report_counts_are_consistent(self, populated_session) -> None:
        run_detector(populated_session, UnsafePromptDetector(), threshold_percentile=99.0)
        report = generate_quality_report(populated_session)

        assert report["clean_examples"] + report["flagged_any"] == report["total_examples"]


# ===================================================================
# Checkpoint 4: Pipeline Works — data ingestion
# ===================================================================


class TestDataIngestion:
    """Test ingest_to_database with a mock dataset."""

    def _make_mock_dataset(self, n: int = 5):
        """Create a list-like mock dataset matching HuggingFace Dataset interface."""
        data = []
        for i in range(n):
            data.append(
                {
                    "chosen": f"Human: Question {i}\n\nAssistant: Good answer {i}",
                    "rejected": f"Human: Question {i}\n\nAssistant: Bad answer {i}",
                }
            )

        class MockDataset:
            def __init__(self, items):
                self._items = items

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

        return MockDataset(data)

    def test_ingest_basic(self, session) -> None:
        from rlhf_eval.pipeline.data_loader import ingest_to_database

        dataset = self._make_mock_dataset(5)
        result = ingest_to_database(session, dataset, block_size=2)

        assert result["total_processed"] == 5
        assert result["total_inserted"] == 5
        assert result["skipped"] == 0
        assert len(result["errors"]) == 0
        assert get_example_count(session) == 5

    def test_ingest_resume_skips_existing(self, session) -> None:
        from rlhf_eval.pipeline.data_loader import ingest_to_database

        dataset = self._make_mock_dataset(5)
        ingest_to_database(session, dataset, block_size=10)

        # Ingest again — all should be skipped
        result = ingest_to_database(session, dataset, block_size=10)
        assert result["total_inserted"] == 0
        assert result["skipped"] == 5
        assert get_example_count(session) == 5

    def test_ingest_resume_from_index(self, session) -> None:
        from rlhf_eval.pipeline.data_loader import ingest_to_database

        dataset = self._make_mock_dataset(10)
        result = ingest_to_database(session, dataset, block_size=5, resume_from=5)

        assert result["total_processed"] == 5
        assert result["total_inserted"] == 5
        assert get_example_count(session) == 5

    def test_ingestion_progress(self, session) -> None:
        from rlhf_eval.pipeline.data_loader import (
            get_ingestion_progress,
            ingest_to_database,
        )

        # Empty DB
        progress = get_ingestion_progress(session)
        assert progress["count"] == 0
        assert progress["max_index"] == -1

        dataset = self._make_mock_dataset(5)
        ingest_to_database(session, dataset, block_size=10)

        progress = get_ingestion_progress(session)
        assert progress["count"] == 5
        assert progress["max_index"] == 4

    def test_ingest_parsed_fields_correct(self, session) -> None:
        from rlhf_eval.pipeline.data_loader import ingest_to_database
        from rlhf_eval.database.operations import get_example_by_index

        dataset = self._make_mock_dataset(1)
        ingest_to_database(session, dataset, block_size=10)

        ex = get_example_by_index(session, 0)
        assert ex is not None
        assert ex.chosen_last_assistant == "Good answer 0"
        assert ex.rejected_last_assistant == "Bad answer 0"
        assert ex.prompt == "Question 0"
        assert "Human:" in ex.chosen_full
        assert "Assistant:" in ex.chosen_full


# ===================================================================
# End-to-end: ingest → detect → report
# ===================================================================


class TestEndToEnd:
    """Full pipeline: ingest mock data, run detectors, verify report."""

    def test_full_pipeline(self, engine) -> None:
        from rlhf_eval.database.connection import SessionContext

        # Ingest
        with SessionContext(engine) as sess:
            insert_examples_batch(sess, EXAMPLES)

        # Run all CPU detectors
        with SessionContext(engine) as sess:
            detectors = [
                RepetitionDetector(),
                LengthRatioDetector(),
                RefusalBiasDetector(),
                UnsafePromptDetector(),
            ]
            runs = run_all_detectors(sess, detectors=detectors)
            assert len(runs) == 4

        # Generate report
        with SessionContext(engine) as sess:
            report = generate_quality_report(sess)
            assert report["total_examples"] == len(EXAMPLES)
            assert report["flagged_any"] > 0
            assert report["clean_examples"] < report["total_examples"]

        # Verify clean set
        with SessionContext(engine) as sess:
            clean_ids = get_clean_example_ids(sess)
            assert len(clean_ids) > 0
            assert len(clean_ids) < len(EXAMPLES)

            # Verify the unsafe example is excluded
            clean_indices = []
            for ex_id in clean_ids:
                ex = sess.get(Example, ex_id)
                clean_indices.append(ex.dataset_index)
            assert 5 not in clean_indices  # unsafe prompt
