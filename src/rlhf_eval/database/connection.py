#!/usr/bin/env python3
"""
Database manager converting postgresql to postgresql with asyncpg, ensured cleanup, and disposal of the connection pool.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
from collections.abc import AsyncGenerator

# ----------------- Third Party Library -----------------
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ----------------- Application Imports -----------------


# ----------------- Module-level Configuration -----------------

class DatabaseManager:
    """ Manages async database connections."""
    def __init__(self, database_url: str, pool_size: int = 5) -> None:
        # convert postgres:// to postgresql+asyncpg://
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )

        self.engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            echo=False,    # must be true for SQL logging
        )

        self.session_factory = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield a session ensuring cleanup"""
        async with self.session_factory() as session:
            yield session
    
    async def close(self) -> None:
        """Dispose of the connection pool"""
        await self.engine.dispose()