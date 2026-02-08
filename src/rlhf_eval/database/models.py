"""SQLAlchemy models for the RLHF Evaluation Harness."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Example(Base):
    __tablename__ = "examples"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_index: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    chosen_full: Mapped[str] = mapped_column(Text)
    rejected_full: Mapped[str] = mapped_column(Text)
    chosen_last_assistant: Mapped[str] = mapped_column(Text)
    rejected_last_assistant: Mapped[str] = mapped_column(Text)
    prompt: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    quality_signals: Mapped[list["QualitySignal"]] = relationship(
        back_populates="example"
    )


class QualitySignal(Base):
    __tablename__ = "quality_signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    example_id: Mapped[int] = mapped_column(ForeignKey("examples.id"), index=True)
    detector_name: Mapped[str] = mapped_column(String(100), index=True)
    score: Mapped[float] = mapped_column(Float)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    signal_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    example: Mapped["Example"] = relationship(back_populates="quality_signals")


class DetectorRun(Base):
    __tablename__ = "detector_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    detector_name: Mapped[str] = mapped_column(String(100))
    threshold: Mapped[float] = mapped_column(Float)
    percentile_used: Mapped[float] = mapped_column(Float)
    total_examples: Mapped[int] = mapped_column(Integer)
    flagged_count: Mapped[int] = mapped_column(Integer)
    stats: Mapped[dict] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class RewardModel(Base):
    __tablename__ = "reward_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    dataset_filter: Mapped[str] = mapped_column(String(50))
    training_config: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    model_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    reward_model_id: Mapped[int] = mapped_column(ForeignKey("reward_models.id"))
    eval_type: Mapped[str] = mapped_column(String(100))
    results: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
