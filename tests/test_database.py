#!/usr/bin/env python3
"""Integration tests for the database layer using SQLite in-memory.

Covers: models (table creation, relationships), connection (engine, session,
SessionContext), and operations (all CRUD functions).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from rlhf_eval.database.models import (
    Base,
    DetectorRun,
    Evaluation,
    Example,
    QualitySignal,
    RewardModel,
)
from rlhf_eval.database.connection import (
    SessionContext,
    get_engine,
    get_session,
    get_session_factory,
)
from rlhf_eval.database.operations import (
    get_all_example_ids,
    get_example_by_id,
    get_example_by_index,
    get_example_count,
    get_examples_paginated,
    get_flagged_example_ids,
    get_latest_detector_run,
    get_reward_model,
    get_signals_by_detector,
    get_unflagged_example_ids,
    insert_detector_run,
    insert_evaluation,
    insert_example,
    insert_examples_batch,
    insert_quality_signal,
    insert_quality_signals_batch,
    insert_reward_model,
    signal_exists,
    update_detector_run,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine):
    """Provide a transactional session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    sess = Session(bind=connection)
    yield sess
    sess.close()
    transaction.rollback()
    connection.close()


def _make_example(idx: int = 0) -> dict:
    """Helper to build example kwargs."""
    return {
        "dataset_index": idx,
        "chosen_full": f"Human: Q{idx}\n\nAssistant: Chosen answer {idx}",
        "rejected_full": f"Human: Q{idx}\n\nAssistant: Rejected answer {idx}",
        "chosen_last_assistant": f"Chosen answer {idx}",
        "rejected_last_assistant": f"Rejected answer {idx}",
        "prompt": f"Q{idx}",
    }


# ===================================================================
# Model / table creation
# ===================================================================


class TestModels:
    def test_tables_created(self, engine) -> None:
        table_names = Base.metadata.tables.keys()
        assert "examples" in table_names
        assert "quality_signals" in table_names
        assert "detector_runs" in table_names
        assert "reward_models" in table_names
        assert "evaluations" in table_names

    def test_example_columns(self) -> None:
        cols = {c.name for c in Example.__table__.columns}
        expected = {
            "id", "dataset_index", "chosen_full", "rejected_full",
            "chosen_last_assistant", "rejected_last_assistant",
            "prompt", "created_at",
        }
        assert expected == cols

    def test_quality_signal_columns(self) -> None:
        cols = {c.name for c in QualitySignal.__table__.columns}
        expected = {
            "id", "example_id", "detector_name", "score",
            "flagged", "metadata", "created_at",
        }
        assert expected == cols

    def test_quality_signal_db_column_name_is_metadata(self) -> None:
        """The Python attr is signal_metadata but the DB column is metadata."""
        col = QualitySignal.__table__.c["metadata"]
        assert col is not None
        assert col.name == "metadata"

    def test_detector_run_columns(self) -> None:
        cols = {c.name for c in DetectorRun.__table__.columns}
        expected = {
            "id", "detector_name", "threshold", "percentile_used",
            "total_examples", "flagged_count", "stats",
            "started_at", "completed_at",
        }
        assert expected == cols

    def test_reward_model_columns(self) -> None:
        cols = {c.name for c in RewardModel.__table__.columns}
        expected = {
            "id", "name", "dataset_filter", "training_config",
            "metrics", "model_path", "created_at",
        }
        assert expected == cols

    def test_evaluation_columns(self) -> None:
        cols = {c.name for c in Evaluation.__table__.columns}
        expected = {"id", "reward_model_id", "eval_type", "results", "created_at"}
        assert expected == cols

    def test_example_has_integer_pk(self) -> None:
        pk_col = Example.__table__.c["id"]
        assert pk_col.primary_key

    def test_quality_signal_fk_to_examples(self) -> None:
        fk = list(QualitySignal.__table__.c["example_id"].foreign_keys)
        assert len(fk) == 1
        assert fk[0].target_fullname == "examples.id"


# ===================================================================
# Connection
# ===================================================================


class TestConnection:
    def test_get_engine_with_url(self) -> None:
        eng = get_engine("sqlite:///:memory:")
        assert eng is not None

    def test_get_engine_no_url_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("RLHF_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(ValueError, match="No database URL"):
            get_engine()

    def test_get_session_factory(self) -> None:
        eng = create_engine("sqlite:///:memory:")
        factory = get_session_factory(eng)
        assert factory is not None
        sess = factory()
        assert isinstance(sess, Session)
        sess.close()

    def test_get_session(self) -> None:
        eng = create_engine("sqlite:///:memory:")
        sess = get_session(eng)
        assert isinstance(sess, Session)
        sess.close()

    def test_session_context_commits(self) -> None:
        eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(eng)
        with SessionContext(eng) as sess:
            ex = Example(**_make_example(0))
            sess.add(ex)
        # After context exits, should be committed
        with SessionContext(eng) as sess:
            count = get_example_count(sess)
            assert count == 1

    def test_session_context_rollback_on_error(self) -> None:
        eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(eng)
        try:
            with SessionContext(eng) as sess:
                ex = Example(**_make_example(0))
                sess.add(ex)
                raise RuntimeError("forced error")
        except RuntimeError:
            pass
        # Should have rolled back
        with SessionContext(eng) as sess:
            count = get_example_count(sess)
            assert count == 0


# ===================================================================
# Example operations
# ===================================================================


class TestExampleOperations:
    def test_insert_example(self, session) -> None:
        ex = insert_example(session, **_make_example(0))
        assert ex.id is not None
        assert ex.dataset_index == 0

    def test_insert_examples_batch(self, session) -> None:
        batch = [_make_example(i) for i in range(5)]
        result = insert_examples_batch(session, batch)
        assert len(result) == 5
        assert all(ex.id is not None for ex in result)

    def test_get_example_by_index(self, session) -> None:
        insert_example(session, **_make_example(42))
        found = get_example_by_index(session, 42)
        assert found is not None
        assert found.dataset_index == 42

    def test_get_example_by_index_not_found(self, session) -> None:
        assert get_example_by_index(session, 999) is None

    def test_get_example_by_id(self, session) -> None:
        ex = insert_example(session, **_make_example(0))
        found = get_example_by_id(session, ex.id)
        assert found is not None
        assert found.id == ex.id

    def test_get_examples_paginated(self, session) -> None:
        insert_examples_batch(session, [_make_example(i) for i in range(10)])
        page1 = get_examples_paginated(session, offset=0, limit=3)
        page2 = get_examples_paginated(session, offset=3, limit=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id

    def test_get_example_count(self, session) -> None:
        assert get_example_count(session) == 0
        insert_examples_batch(session, [_make_example(i) for i in range(5)])
        assert get_example_count(session) == 5

    def test_get_all_example_ids(self, session) -> None:
        insert_examples_batch(session, [_make_example(i) for i in range(3)])
        ids = get_all_example_ids(session)
        assert len(ids) == 3

    def test_dataset_index_unique(self, session) -> None:
        insert_example(session, **_make_example(0))
        with pytest.raises(Exception):
            insert_example(session, **_make_example(0))
            session.flush()


# ===================================================================
# Quality signal operations
# ===================================================================


class TestQualitySignalOperations:
    def test_insert_quality_signal(self, session) -> None:
        ex = insert_example(session, **_make_example(0))
        sig = insert_quality_signal(
            session,
            example_id=ex.id,
            detector_name="test_detector",
            score=0.95,
            flagged=True,
            signal_metadata={"key": "value"},
        )
        assert sig.id is not None
        assert sig.score == pytest.approx(0.95)
        assert sig.flagged is True

    def test_insert_quality_signals_batch(self, session) -> None:
        ex = insert_example(session, **_make_example(0))
        signals = [
            {
                "example_id": ex.id,
                "detector_name": "det_a",
                "score": 0.5,
                "flagged": False,
                "signal_metadata": None,
            },
            {
                "example_id": ex.id,
                "detector_name": "det_b",
                "score": 0.9,
                "flagged": True,
                "signal_metadata": {"info": "flagged"},
            },
        ]
        result = insert_quality_signals_batch(session, signals)
        assert len(result) == 2

    def test_get_signals_by_detector(self, session) -> None:
        ex = insert_example(session, **_make_example(0))
        insert_quality_signal(
            session, example_id=ex.id, detector_name="det_a", score=0.5, flagged=False,
        )
        insert_quality_signal(
            session, example_id=ex.id, detector_name="det_b", score=0.9, flagged=True,
        )
        signals = get_signals_by_detector(session, "det_a")
        assert len(signals) == 1
        assert signals[0].detector_name == "det_a"

    def test_get_flagged_example_ids(self, session) -> None:
        exs = insert_examples_batch(session, [_make_example(i) for i in range(3)])
        insert_quality_signal(
            session, example_id=exs[0].id, detector_name="det", score=0.9, flagged=True,
        )
        insert_quality_signal(
            session, example_id=exs[1].id, detector_name="det", score=0.1, flagged=False,
        )
        insert_quality_signal(
            session, example_id=exs[2].id, detector_name="det", score=0.8, flagged=True,
        )
        flagged = get_flagged_example_ids(session, "det")
        assert len(flagged) == 2
        assert exs[0].id in flagged
        assert exs[2].id in flagged

    def test_get_unflagged_example_ids(self, session) -> None:
        exs = insert_examples_batch(session, [_make_example(i) for i in range(2)])
        insert_quality_signal(
            session, example_id=exs[0].id, detector_name="det", score=0.9, flagged=True,
        )
        insert_quality_signal(
            session, example_id=exs[1].id, detector_name="det", score=0.1, flagged=False,
        )
        unflagged = get_unflagged_example_ids(session, "det")
        assert len(unflagged) == 1
        assert exs[1].id in unflagged

    def test_signal_exists(self, session) -> None:
        ex = insert_example(session, **_make_example(0))
        assert signal_exists(session, ex.id, "det") is False
        insert_quality_signal(
            session, example_id=ex.id, detector_name="det", score=0.5, flagged=False,
        )
        assert signal_exists(session, ex.id, "det") is True
        assert signal_exists(session, ex.id, "other") is False

    def test_example_relationship(self, session) -> None:
        ex = insert_example(session, **_make_example(0))
        insert_quality_signal(
            session, example_id=ex.id, detector_name="det", score=0.5, flagged=False,
        )
        session.flush()
        session.refresh(ex)
        assert len(ex.quality_signals) == 1
        assert ex.quality_signals[0].detector_name == "det"


# ===================================================================
# Detector run operations
# ===================================================================


class TestDetectorRunOperations:
    def test_insert_detector_run(self, session) -> None:
        now = datetime.now(timezone.utc)
        run = insert_detector_run(
            session,
            detector_name="semantic_similarity",
            threshold=0.95,
            percentile_used=99.5,
            total_examples=1000,
            flagged_count=5,
            stats={"mean": 0.48, "std": 0.15},
            started_at=now,
        )
        assert run.id is not None
        assert run.detector_name == "semantic_similarity"
        assert run.completed_at is None

    def test_update_detector_run(self, session) -> None:
        now = datetime.now(timezone.utc)
        run = insert_detector_run(
            session,
            detector_name="test",
            threshold=0.0,
            percentile_used=99.5,
            total_examples=0,
            flagged_count=0,
            stats={},
            started_at=now,
        )
        updated = update_detector_run(
            session,
            run.id,
            threshold=0.85,
            total_examples=500,
            flagged_count=10,
            completed_at=datetime.now(timezone.utc),
        )
        assert updated.threshold == pytest.approx(0.85)
        assert updated.total_examples == 500
        assert updated.completed_at is not None

    def test_update_detector_run_not_found(self, session) -> None:
        with pytest.raises(ValueError, match="not found"):
            update_detector_run(session, 9999, threshold=0.5)

    def test_get_latest_detector_run(self, session) -> None:
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
        insert_detector_run(
            session, detector_name="det", threshold=0.5, percentile_used=99.5,
            total_examples=100, flagged_count=1, stats={}, started_at=t1,
        )
        insert_detector_run(
            session, detector_name="det", threshold=0.9, percentile_used=99.5,
            total_examples=200, flagged_count=5, stats={}, started_at=t2,
        )
        latest = get_latest_detector_run(session, "det")
        assert latest is not None
        assert latest.threshold == pytest.approx(0.9)
        assert latest.total_examples == 200

    def test_get_latest_detector_run_not_found(self, session) -> None:
        assert get_latest_detector_run(session, "nonexistent") is None


# ===================================================================
# Reward model + evaluation operations
# ===================================================================


class TestRewardModelOperations:
    def test_insert_and_get_reward_model(self, session) -> None:
        rm = insert_reward_model(
            session,
            name="test-model",
            dataset_filter="clean",
            training_config={"lr": 1e-5, "epochs": 3},
        )
        assert rm.id is not None

        found = get_reward_model(session, rm.id)
        assert found is not None
        assert found.name == "test-model"
        assert found.dataset_filter == "clean"
        assert found.training_config["lr"] == pytest.approx(1e-5)

    def test_get_reward_model_not_found(self, session) -> None:
        assert get_reward_model(session, 9999) is None

    def test_insert_evaluation(self, session) -> None:
        rm = insert_reward_model(
            session,
            name="model-1",
            dataset_filter="unfiltered",
            training_config={},
        )
        ev = insert_evaluation(
            session,
            reward_model_id=rm.id,
            eval_type="accuracy",
            results={"accuracy": 0.72, "samples": 500},
        )
        assert ev.id is not None
        assert ev.eval_type == "accuracy"
        assert ev.results["accuracy"] == pytest.approx(0.72)
