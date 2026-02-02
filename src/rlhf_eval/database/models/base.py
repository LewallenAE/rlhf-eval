#!/usr/bin/env python3
"""
This is the Base Model for SQLAlchemy with Mixins
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
from datetime import datetime
from uuid import uuid4

# ----------------- Third Party Library -----------------
from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ----------------- Application Imports -----------------


# ----------------- Module-level Configuration -----------------

class Base(DeclarativeBase):
    """Base class for all models"""
    pass


class TimestampMixin:
    """Adds created_at and updated_at columns."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

class UUIDMixin:
    """Adds UUID primary key."""

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )