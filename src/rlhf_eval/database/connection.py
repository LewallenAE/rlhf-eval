"""Database connection and session management."""

import os
from typing import Self

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def get_engine(database_url: str | None = None) -> Engine:
    """Create database engine. Uses RLHF_DATABASE_URL or DATABASE_URL env var if not provided."""
    url = database_url or os.environ.get("RLHF_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError(
            "No database URL provided. Set RLHF_DATABASE_URL env var or pass directly."
        )
    return create_engine(url, pool_size=5, echo=False)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create session factory."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_session(engine: Engine) -> Session:
    """Get a new session."""
    factory = get_session_factory(engine)
    return factory()


class SessionContext:
    """Context manager for database sessions with automatic commit/rollback."""

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or get_engine()
        self._session: Session | None = None

    def __enter__(self) -> Session:
        self._session = get_session(self._engine)
        return self._session

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        if self._session is None:
            return
        if exc_type is not None:
            self._session.rollback()
        else:
            self._session.commit()
        self._session.close()
