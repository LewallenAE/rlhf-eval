"""CRUD operations for all database models."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import DetectorRun, Evaluation, Example, QualitySignal, RewardModel


# ── Example operations ──────────────────────────────────────────────


def insert_example(session: Session, **kwargs) -> Example:
    """Insert a single example."""
    example = Example(**kwargs)
    session.add(example)
    session.flush()
    return example


def insert_examples_batch(session: Session, examples: list[dict]) -> list[Example]:
    """Insert multiple examples in a batch."""
    objs = [Example(**data) for data in examples]
    session.add_all(objs)
    session.flush()
    return objs


def get_example_by_index(session: Session, dataset_index: int) -> Example | None:
    """Get an example by its original dataset index."""
    return session.execute(
        select(Example).where(Example.dataset_index == dataset_index)
    ).scalar_one_or_none()


def get_example_by_id(session: Session, id: int) -> Example | None:
    """Get an example by primary key."""
    return session.get(Example, id)


def get_examples_paginated(
    session: Session, offset: int, limit: int
) -> list[Example]:
    """Get examples with pagination."""
    return list(
        session.execute(
            select(Example).offset(offset).limit(limit)
        ).scalars().all()
    )


def get_example_count(session: Session) -> int:
    """Get total number of examples."""
    result = session.execute(select(func.count(Example.id)))
    return result.scalar_one()


def get_all_example_ids(session: Session) -> list[int]:
    """Get all example IDs."""
    return list(
        session.execute(select(Example.id)).scalars().all()
    )


# ── Quality signal operations ───────────────────────────────────────


def insert_quality_signal(session: Session, **kwargs) -> QualitySignal:
    """Insert a single quality signal."""
    signal = QualitySignal(**kwargs)
    session.add(signal)
    session.flush()
    return signal


def insert_quality_signals_batch(
    session: Session, signals: list[dict]
) -> list[QualitySignal]:
    """Insert multiple quality signals in a batch."""
    objs = [QualitySignal(**data) for data in signals]
    session.add_all(objs)
    session.flush()
    return objs


def get_signals_by_detector(
    session: Session, detector_name: str
) -> list[QualitySignal]:
    """Get all signals for a specific detector."""
    return list(
        session.execute(
            select(QualitySignal).where(
                QualitySignal.detector_name == detector_name
            )
        ).scalars().all()
    )


def get_flagged_example_ids(session: Session, detector_name: str) -> list[int]:
    """Get example IDs flagged by a specific detector."""
    return list(
        session.execute(
            select(QualitySignal.example_id).where(
                QualitySignal.detector_name == detector_name,
                QualitySignal.flagged == True,  # noqa: E712
            )
        ).scalars().all()
    )


def get_unflagged_example_ids(session: Session, detector_name: str) -> list[int]:
    """Get example IDs NOT flagged by a specific detector."""
    return list(
        session.execute(
            select(QualitySignal.example_id).where(
                QualitySignal.detector_name == detector_name,
                QualitySignal.flagged == False,  # noqa: E712
            )
        ).scalars().all()
    )


def signal_exists(session: Session, example_id: int, detector_name: str) -> bool:
    """Check if a signal already exists for this example + detector combo."""
    result = session.execute(
        select(func.count(QualitySignal.id)).where(
            QualitySignal.example_id == example_id,
            QualitySignal.detector_name == detector_name,
        )
    )
    return result.scalar_one() > 0


# ── Detector run operations ─────────────────────────────────────────


def insert_detector_run(session: Session, **kwargs) -> DetectorRun:
    """Insert a detector run record."""
    run = DetectorRun(**kwargs)
    session.add(run)
    session.flush()
    return run


def update_detector_run(session: Session, run_id: int, **kwargs) -> DetectorRun:
    """Update a detector run with new values."""
    run = session.get(DetectorRun, run_id)
    if run is None:
        raise ValueError(f"DetectorRun {run_id} not found")
    for key, value in kwargs.items():
        setattr(run, key, value)
    session.flush()
    return run


def get_latest_detector_run(
    session: Session, detector_name: str
) -> DetectorRun | None:
    """Get the most recent run for a detector."""
    return session.execute(
        select(DetectorRun)
        .where(DetectorRun.detector_name == detector_name)
        .order_by(DetectorRun.started_at.desc())
        .limit(1)
    ).scalar_one_or_none()


# ── Reward model operations ─────────────────────────────────────────


def insert_reward_model(session: Session, **kwargs) -> RewardModel:
    """Insert a reward model record."""
    model = RewardModel(**kwargs)
    session.add(model)
    session.flush()
    return model


def get_reward_model(session: Session, id: int) -> RewardModel | None:
    """Get a reward model by ID."""
    return session.get(RewardModel, id)


# ── Evaluation operations ───────────────────────────────────────────


def insert_evaluation(session: Session, **kwargs) -> Evaluation:
    """Insert an evaluation record."""
    evaluation = Evaluation(**kwargs)
    session.add(evaluation)
    session.flush()
    return evaluation
